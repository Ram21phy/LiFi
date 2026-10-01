"""
models/vlc_channel.py
=====================
Indoor visible-light line-of-sight (LOS) optical channel.

Two transmitter models are provided and kept strictly separate:

1. **Lambertian VLC LOS** (Barry & Kahn / Komine & Nakagawa)

       H(0) = (m+1) A / (2 pi d^2) * cos^m(phi) * T_s(psi) * g(psi) * cos(psi),
       for 0 <= psi <= Psi_c, and H(0) = 0 otherwise.

   with
       m      = -ln 2 / ln( cos(Phi_1/2) )      Lambertian (mode) order
       A      = physical receiver aperture / detector area           [m^2]
       d      = Tx-Rx separation                                     [m]
       phi    = irradiance (radiation) angle w.r.t. the Tx normal    [rad]
       psi    = incidence angle w.r.t. the Rx normal                 [rad]
       T_s    = transmission of the optical band-pass filter
       g(psi) = n^2 / sin^2(Psi_c)   non-imaging concentrator gain
       Psi_c  = receiver field-of-view half angle                    [rad]

   H(0) is the DC channel gain, i.e. the *fraction of transmitted optical
   power collected by the receiver*; it is therefore directly usable as the
   channel transmittance eta_channel of the quantum link.

2. **Collimated Gaussian beam** (more representative of a real free-space QKD
   transmitter):

       w(d)   = w0 + d * theta_div
       T_geo  = 1 - exp( -2 a^2 / w(d)^2 )

   where a is the receiver aperture radius.  The FOV cut and the concentrator
   gain are *not* applied in this mode (a collimated beam is collected by the
   aperture itself); a cos(psi) projection of the aperture is retained.

Everything here is ANALYTIC and purely *classical optics*: no QKD assumption
whatsoever enters this module.  That separation is deliberate (requirement 24).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def lambertian_order(half_power_semiangle_deg: float) -> float:
    """m = -ln2 / ln(cos(Phi_1/2)).  Diverges as Phi_1/2 -> 0."""
    phi = math.radians(min(max(half_power_semiangle_deg, 1e-3), 89.999))
    c = math.cos(phi)
    if c >= 1.0:
        return 1e9
    return -math.log(2.0) / math.log(c)


def semiangle_from_order(m: float) -> float:
    """Inverse of `lambertian_order`, returns Phi_1/2 in degrees."""
    m = max(m, 1e-6)
    return math.degrees(math.acos(2.0 ** (-1.0 / m)))


def concentrator_gain(psi_rad: float, fov_rad: float, n_index: float,
                      enabled: bool = True) -> float:
    """
    Ideal non-imaging concentrator: g(psi) = n^2 / sin^2(Psi_c) inside the FOV.
    Returns 1.0 when disabled, 0.0 outside the FOV.
    n=refraction index of the concentrator material, Psi_c=FOV half-angle.
    """
    if psi_rad > fov_rad:
        return 0.0
    if not enabled:
        return 1.0
    s = math.sin(fov_rad)
    if s <= 1e-12:
        return n_index ** 2 / 1e-24
    return (n_index ** 2) / (s ** 2)


# --------------------------------------------------------------------------
@dataclass
class ChannelGeometry:
    distance_m: float
    irradiance_angle_deg: float     # phi, at the transmitter
    incidence_angle_deg: float      # psi, at the receiver
    in_fov: bool
    tx_position: Tuple[float, float, float] = (0, 0, 0)
    rx_position: Tuple[float, float, float] = (0, 0, 0)
    horizontal_separation_m: float = 0.0
    vertical_separation_m: float = 0.0


@dataclass
class ChannelResult:
    geometry: ChannelGeometry
    lambertian_order: float
    channel_gain_h0: float           # dimensionless power ratio (LOS DC gain)
    geometric_attenuation: float     # (m+1)A cos^m(phi) cos(psi) / (2 pi d^2)
    concentrator_gain: float
    filter_transmission: float
    optical_efficiency: float
    pointing_transmittance: float
    turbulence_mean_transmittance: float
    extra_loss_factor: float
    eta_channel: float               # total optical channel transmittance (Tx->detector face)
    eta_channel_db: float
    model: str = "Lambertian (VLC LOS)"
    provenance: str = "analytic"

    def as_dict(self) -> Dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "geometry"}
        d.update({
            "distance_m": self.geometry.distance_m,
            "irradiance_angle_deg": self.geometry.irradiance_angle_deg,
            "incidence_angle_deg": self.geometry.incidence_angle_deg,
            "in_fov": self.geometry.in_fov,
        })
        return d


# --------------------------------------------------------------------------
def compute_geometry(room, fov_deg: float) -> ChannelGeometry:
    """Resolve Tx/Rx positions and orientations into (d, phi, psi)."""
    tx = np.array([room.tx_x, room.tx_y, room.tx_z], dtype=float)
    rx = np.array([room.rx_x, room.rx_y, room.rx_z], dtype=float)

    link = rx - tx
    d = float(np.linalg.norm(link))
    if d < 1e-6:
        d = 1e-6
        link = np.array([1e-6, 0.0, 0.0])

    if room.override_distance:
        # keep the direction, rescale to the requested link distance
        link = _unit(link) * float(room.link_distance_m)
        rx = tx + link
        d = float(room.link_distance_m)

    u = _unit(link)

    n_tx = _unit(np.array(room.tx_normal, dtype=float))
    if room.tx_aim_at_rx:
        n_tx = u
    n_rx = _unit(np.array(room.rx_normal, dtype=float))
    if room.rx_aim_at_tx:
        n_rx = -u

    cos_phi = float(np.clip(np.dot(n_tx, u), -1.0, 1.0))
    cos_psi = float(np.clip(np.dot(n_rx, -u), -1.0, 1.0))

    phi = math.degrees(math.acos(max(cos_phi, -1.0)))
    psi = math.degrees(math.acos(max(cos_psi, -1.0)))

    return ChannelGeometry(
        distance_m=d,
        irradiance_angle_deg=phi,
        incidence_angle_deg=psi,
        in_fov=(psi <= fov_deg and cos_phi > 0.0),
        tx_position=tuple(tx),
        rx_position=tuple(rx),
        horizontal_separation_m=float(np.linalg.norm(link[:2])),
        vertical_separation_m=float(abs(link[2])),
    )


# --------------------------------------------------------------------------
def lambertian_gain(distance_m: float,
                    phi_deg: float,
                    psi_deg: float,
                    aperture_area_m2: float,
                    fov_deg: float,
                    lamb_order: float,
                    filter_transmission: float = 1.0,
                    n_index: float = 1.5,
                    use_concentrator: bool = True) -> Tuple[float, float, float]:
    """
    Returns (H0, geometric_attenuation, g_concentrator).

    H0 = geometric_attenuation * T_s * g(psi)
    with geometric_attenuation = (m+1) A cos^m(phi) cos(psi) / (2 pi d^2).
    """
    if distance_m <= 0:
        return 0.0, 0.0, 0.0
    phi = math.radians(phi_deg)
    psi = math.radians(psi_deg)
    fov = math.radians(fov_deg)

    if psi > fov or phi >= math.pi / 2:
        return 0.0, 0.0, 0.0

    cos_phi = max(math.cos(phi), 0.0)
    cos_psi = max(math.cos(psi), 0.0)

    # cos^m with large m handled in log space for stability
    if cos_phi <= 0.0:
        cos_m = 0.0
    else:
        cos_m = math.exp(lamb_order * math.log(cos_phi))

    geo = ((lamb_order + 1.0) * aperture_area_m2 /
           (2.0 * math.pi * distance_m ** 2)) * cos_m * cos_psi
    g = concentrator_gain(psi, fov, n_index, enabled=use_concentrator)
    h0 = geo * filter_transmission * g
    # H(0) is a power *fraction*; clamp for absurd parameter combinations
    return float(min(h0, 1.0)), float(geo), float(g)


def gaussian_beam_gain(distance_m: float,
                       psi_deg: float,
                       aperture_area_m2: float,
                       waist_mm: float,
                       divergence_mrad: float,
                       filter_transmission: float = 1.0) -> Tuple[float, float]:
    """
    Collimated-beam geometric transmittance.

        w(d)  = w0 + d*theta
        T_geo = [1 - exp(-2 a^2 / w(d)^2)] * cos(psi)

    Returns (H0, T_geo_before_filter).
    """
    w0 = waist_mm * 1e-3
    theta = divergence_mrad * 1e-3
    w = w0 + max(distance_m, 0.0) * theta
    a = math.sqrt(max(aperture_area_m2, 0.0) / math.pi)
    if w <= 0:
        t_geo = 1.0
    else:
        t_geo = 1.0 - math.exp(-2.0 * a * a / (w * w))
    t_geo *= max(math.cos(math.radians(psi_deg)), 0.0)
    return float(min(t_geo * filter_transmission, 1.0)), float(t_geo)


# --------------------------------------------------------------------------
def evaluate_channel(params,
                     filter_transmission_at_signal: float,
                     pointing_transmittance: float = 1.0,
                     turbulence_mean_transmittance: float = 1.0,
                     distance_override: Optional[float] = None) -> ChannelResult:
    """
    Full channel evaluation from SimulationParameters.

    `filter_transmission_at_signal` is T_filter evaluated at the *signal*
    wavelength and is supplied by models.background_noise.SpectralFilter so
    that the filter is described in exactly one place.
    """
    ch = params.channel
    room = params.room

    geom = compute_geometry(room, ch.rx_fov_deg)
    if distance_override is not None:
        # rebuild geometry along the same direction at the requested distance
        geom = ChannelGeometry(
            distance_m=float(distance_override),
            irradiance_angle_deg=geom.irradiance_angle_deg,
            incidence_angle_deg=geom.incidence_angle_deg,
            in_fov=geom.in_fov,
            tx_position=geom.tx_position,
            rx_position=geom.rx_position,
            horizontal_separation_m=geom.horizontal_separation_m,
            vertical_separation_m=geom.vertical_separation_m,
        )

    A = ch.rx_aperture_area_cm2 * 1e-4   # cm^2 -> m^2
    m = (ch.lambertian_order_override
         if ch.lambertian_order_override
         else lambertian_order(ch.tx_half_power_semiangle_deg))

    if ch.channel_model.startswith("Collimated"):
        h0, geo = gaussian_beam_gain(
            geom.distance_m, geom.incidence_angle_deg, A,
            ch.beam_waist_mm, ch.beam_divergence_mrad,
            filter_transmission_at_signal)
        g = 1.0
    else:
        h0, geo, g = lambertian_gain(
            geom.distance_m, geom.irradiance_angle_deg, geom.incidence_angle_deg,
            A, ch.rx_fov_deg, m, filter_transmission_at_signal,
            ch.concentrator_index, ch.use_concentrator)

    extra = 10.0 ** (-abs(ch.extra_loss_db) / 10.0)

    eta_channel = (h0 * ch.optical_efficiency * extra
                   * pointing_transmittance * turbulence_mean_transmittance)
    eta_channel = float(np.clip(eta_channel, 0.0, 1.0))

    eta_db = -10.0 * math.log10(eta_channel) if eta_channel > 0 else float("inf")

    return ChannelResult(
        geometry=geom,
        lambertian_order=float(m),
        channel_gain_h0=float(h0),
        geometric_attenuation=float(geo),
        concentrator_gain=float(g),
        filter_transmission=float(filter_transmission_at_signal),
        optical_efficiency=float(ch.optical_efficiency),
        pointing_transmittance=float(pointing_transmittance),
        turbulence_mean_transmittance=float(turbulence_mean_transmittance),
        extra_loss_factor=float(extra),
        eta_channel=eta_channel,
        eta_channel_db=float(eta_db),
        model=ch.channel_model,
    )


def received_optical_power(tx_power_w: float, eta_channel: float) -> float:
    """P_rx = P_tx * H(0) * eta_opt ... (eta_channel already contains them)."""
    return tx_power_w * eta_channel


def received_photon_rate(tx_photon_rate: float, eta_channel: float) -> float:
    return tx_photon_rate * eta_channel
