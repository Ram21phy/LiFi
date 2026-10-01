"""
models/background_noise.py
==========================
Physical model of the indoor background optical noise reaching Bob's detector.

The guiding principle (requirement 5) is that background noise is **never** an
arbitrary QBER number.  It is always built up as a *number of background
photons per detection gate*, starting from radiometry.

-------------------------------------------------------------------------
Core relationship
-------------------------------------------------------------------------
For a spectrally resolved background whose spectral irradiance at the receiver
aperture is p(lambda) [W m^-2 nm^-1]:

    P_bg = integral_lambda  p(lambda) * A * T_f(lambda) * G_coll  dlambda    [W]

where

    A         receiver physical aperture / detector area                 [m^2]
    T_f(l)    optical band-pass filter transmission at wavelength l      [-]
    G_coll    background collection factor of the receiver optics        [-]

For a *narrow* filter of bandwidth dLambda centred on lambda_0 this reduces to
the form requested in the specification,

    P_bg  ~=  p(lambda_0) * A * dLambda * T_filter * G_coll
    N_bg  =   P_bg * eta_opt * eta_det * dt / E_ph(lambda_0)     [counts/gate]

-------------------------------------------------------------------------
Collection factor G_coll
-------------------------------------------------------------------------
Two regimes are implemented and the choice is exposed to the user, because
they behave *very* differently when the field of view is varied:

  (a) "Bare detector" (no concentrator):
          G_coll = Omega_proj = pi sin^2(Psi_c)
      Background collection scales as sin^2(FOV): narrowing the FOV directly
      suppresses diffuse background.

  (b) "Ideal non-imaging concentrator":
          the concentrator gain g = n^2 / sin^2(Psi_c) exactly cancels the
          sin^2(Psi_c) of the accepted solid angle, so
          G_coll = pi n^2,  i.e. INDEPENDENT of FOV.
      This is etendue conservation.  Narrowing the FOV then helps only by
      *geometrically excluding discrete sources* (a ceiling luminaire that
      leaves the field of view), which the discrete-source model below
      handles explicitly.

This is a real and often-overlooked effect, and it is one of the things this
simulator is designed to let a researcher explore.

-------------------------------------------------------------------------
Sources implemented
-------------------------------------------------------------------------
A. LED room illumination  - discrete ceiling luminaires, Lambertian emission,
   phosphor-converted white spectrum (blue pump + broad phosphor band);
   LOS contribution through the VLC channel gain from each luminaire to Bob,
   plus a diffuse multi-reflection term.
B. Fluorescent lighting   - Hg emission lines + phosphor continuum.
C. Sunlight               - 5778 K blackbody spectrum scaled to the requested
   outdoor irradiance, attenuated by window transmission and geometry.
D. User-defined           - direct entry of background optical power, photon
   flux, count rate, or spectral irradiance.
E. Isotropic preset       - a single literature spectral-irradiance number.

Diffuse multi-reflection term
-----------------------------
A simplified "integrating sphere" estimate is used for light that has bounced
off the room surfaces:

    E_diffuse = P_emitted_total * rho_avg / [ S_room * (1 - rho_avg) ]   [W/m^2]

with S_room the total interior surface area and rho_avg the area-weighted mean
reflectivity.  This is labelled SIMPLIFIED BACKGROUND MODEL everywhere; it
does not resolve individual bounce geometry the way a full ray-tracing
impulse-response calculation would.

All results here are ANALYTIC + SIMPLIFIED BACKGROUND MODEL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.photon_model import photon_energy
from models.vlc_channel import lambertian_gain, lambertian_order
from utils import constants as C


# ==========================================================================
#  Wavelength grid and normalised spectra  [units: 1/nm, integrate to 1]
# ==========================================================================
def wavelength_grid(center_nm: Optional[float] = None,
                    bandwidth_nm: Optional[float] = None) -> np.ndarray:
    """
    Wavelength grid for all spectral integrations.

    A single uniform grid cannot resolve a 0.1 nm band-pass filter *and* cover
    300-1100 nm at a sensible cost, so the grid is built in three nested tiers
    when the filter is known:

        coarse : 300 - 1100 nm, 0.5 nm            (out-of-band floor, broad spectra)
        medium : centre +/- max(10, 20 dLambda)   (spectral structure nearby)
        fine   : centre +/- 8 dLambda             (the pass-band itself)

    The fine tier guarantees at least ~75 samples across the FWHM whatever the
    bandwidth, which is what makes the "background scales linearly with filter
    bandwidth" self-test pass to four digits.
    """
    base = np.linspace(C.SPECTRUM_LAMBDA_MIN_NM,
                       C.SPECTRUM_LAMBDA_MAX_NM,
                       C.SPECTRUM_N_POINTS)
    if center_nm is None or bandwidth_nm is None or bandwidth_nm <= 0:
        return base

    lo_g, hi_g = C.SPECTRUM_LAMBDA_MIN_NM, C.SPECTRUM_LAMBDA_MAX_NM
    tiers = [base]
    for half, n in ((max(10.0, 20.0 * bandwidth_nm), 1001),
                    (8.0 * bandwidth_nm, 1201)):
        lo = max(center_nm - half, lo_g)
        hi = min(center_nm + half, hi_g)
        if hi > lo:
            tiers.append(np.linspace(lo, hi, n))
    return np.unique(np.concatenate(tiers))


def _gauss(lam: np.ndarray, peak: float, fwhm: float) -> np.ndarray:
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    return np.exp(-0.5 * ((lam - peak) / sigma) ** 2)


def _normalise(lam: np.ndarray, s: np.ndarray) -> np.ndarray:
    area = np.trapezoid(s, lam) if hasattr(np, "trapezoid") else np.trapz(s, lam)
    if area <= 0:
        return np.zeros_like(s)
    return s / area


def white_led_spectrum(lam: np.ndarray) -> np.ndarray:
    """Normalised spectral power density of a phosphor-converted white LED."""
    p = C.WHITE_LED_SPECTRUM
    s = (p["blue_weight"] * _gauss(lam, p["blue_peak_nm"], p["blue_fwhm_nm"]) /
         (p["blue_fwhm_nm"] + 1e-12)
         + p["phosphor_weight"] * _gauss(lam, p["phosphor_peak_nm"],
                                         p["phosphor_fwhm_nm"]) /
         (p["phosphor_fwhm_nm"] + 1e-12))
    return _normalise(lam, s)


def fluorescent_spectrum(lam: np.ndarray) -> np.ndarray:
    """Normalised tri-phosphor fluorescent lamp spectrum (lines + continuum)."""
    s = np.zeros_like(lam)
    for line, w in zip(C.FLUORESCENT_LINES_NM, C.FLUORESCENT_LINE_WEIGHTS):
        s += w * _gauss(lam, line, C.FLUORESCENT_LINE_FWHM_NM) / C.FLUORESCENT_LINE_FWHM_NM
    cont = C.FLUORESCENT_CONTINUUM
    s += cont["weight"] * _gauss(lam, cont["peak_nm"], cont["fwhm_nm"]) / cont["fwhm_nm"]
    return _normalise(lam, s)


def solar_spectrum(lam: np.ndarray) -> np.ndarray:
    """
    Normalised solar spectrum, approximated by a 5778 K blackbody.

    Planck spectral radiance (per wavelength):
        B(l,T) ~ 1/l^5 * 1/(exp(hc/(l k T)) - 1)
    """
    l_m = lam * C.NM
    x = C.HC / (l_m * C.BOLTZMANN_K * C.SOLAR_BLACKBODY_T_K)
    b = 1.0 / (l_m ** 5) / (np.expm1(np.clip(x, 1e-6, 700.0)))
    return _normalise(lam, b)


def flat_spectrum(lam: np.ndarray) -> np.ndarray:
    return _normalise(lam, np.ones_like(lam))


# ==========================================================================
#  Optical band-pass filter
# ==========================================================================
@dataclass
class SpectralFilter:
    center_nm: float
    bandwidth_nm: float
    peak_transmission: float
    shape: str = "Gaussian"
    out_of_band_rejection_db: float = 60.0

    @staticmethod
    def from_params(fp) -> "SpectralFilter":
        return SpectralFilter(fp.center_nm, fp.bandwidth_nm,
                              fp.peak_transmission, fp.shape,
                              fp.out_of_band_rejection_db)

    def transmission(self, lam):
        """T_f(lambda), vectorised. `bandwidth_nm` is the FWHM."""
        lam = np.asarray(lam, dtype=float)
        bw = max(self.bandwidth_nm, 1e-6)
        floor = self.peak_transmission * 10.0 ** (-abs(self.out_of_band_rejection_db) / 10.0)
        if self.shape == "Top-hat":
            t = np.where(np.abs(lam - self.center_nm) <= bw / 2.0,
                         self.peak_transmission, floor)
        elif self.shape == "Lorentzian":
            hw = bw / 2.0
            t = self.peak_transmission / (1.0 + ((lam - self.center_nm) / hw) ** 2)
            t = np.maximum(t, floor)
        else:  # Gaussian
            t = self.peak_transmission * _gauss(lam, self.center_nm, bw)
            t = np.maximum(t, floor)
        return t

    def transmission_at(self, lam_nm: float) -> float:
        return float(np.atleast_1d(self.transmission(lam_nm))[0])

    def effective_bandwidth_nm(self) -> float:
        """
        Noise-equivalent bandwidth  int T(l) dl / T_peak.
        For a Gaussian of FWHM B this is  B * sqrt(pi/(4 ln2)) ~ 1.064 B.
        """
        lam = wavelength_grid(self.center_nm, self.bandwidth_nm)
        t = self.transmission(lam)
        area = float(np.trapezoid(t, lam)) if hasattr(np, "trapezoid") else float(np.trapz(t, lam))
        return area / max(self.peak_transmission, 1e-15)


# ==========================================================================
#  Receiver background-collection factor
# ==========================================================================
def background_collection_factor(fov_deg: float, n_index: float,
                                 use_concentrator: bool) -> float:
    """
    G_coll [sr, projected], as discussed in the module docstring.

      bare detector      : pi sin^2(Psi_c)
      ideal concentrator : pi n^2            (FOV-independent, etendue-limited)
    """
    psi = math.radians(min(max(fov_deg, 1e-3), 90.0))
    if use_concentrator:
        return math.pi * n_index ** 2
    return math.pi * math.sin(psi) ** 2


# ==========================================================================
#  Luminaire placement helper
# ==========================================================================
def default_luminaire_positions(room, n: int) -> List[Tuple[float, float]]:
    """Symmetric ceiling grid: 1, 2, 4, 6, 9 ... luminaires."""
    if n <= 0:
        return []
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    xs = [(i + 0.5) * room.length_x / cols for i in range(cols)]
    ys = [(j + 0.5) * room.width_y / rows for j in range(rows)]
    pos = [(x, y) for y in ys for x in xs]
    return pos[:n]


def room_surface_area(room) -> float:
    lx, ly, lz = room.length_x, room.width_y, room.height_z
    return 2 * (lx * ly) + 2 * (lx * lz) + 2 * (ly * lz)


def average_reflectivity(room) -> float:
    lx, ly, lz = room.length_x, room.width_y, room.height_z
    a_floor = lx * ly
    a_ceil = lx * ly
    a_wall = 2 * lx * lz + 2 * ly * lz
    tot = a_floor + a_ceil + a_wall
    return (a_floor * room.floor_reflectivity
            + a_ceil * room.ceiling_reflectivity
            + a_wall * room.wall_reflectivity) / tot


# ==========================================================================
#  Individual background contributions
# ==========================================================================
@dataclass
class BackgroundContribution:
    name: str
    optical_power_w: float = 0.0        # in-band power at the detector face
    photon_flux_hz: float = 0.0         # in-band photons/s at the detector face
    count_rate_hz: float = 0.0          # after eta_opt * eta_det
    spectral_irradiance_w_m2_nm: float = 0.0   # at the aperture, at lambda_0
    los_fraction: float = 0.0
    note: str = ""


def _discrete_luminaire_irradiance(room, positions, mount_height, power_w,
                                   semiangle_deg, rx_fov_deg,
                                   aperture_area_m2) -> Tuple[float, int]:
    """
    Total LOS optical power (all wavelengths) collected from discrete ceiling
    luminaires, and the number of luminaires inside the receiver FOV.

    Uses the same Lambertian LOS gain as the quantum channel, with the receiver
    assumed upward-facing (normal +z) for background collection.  The
    concentrator is *not* applied here; it is folded into G_coll separately for
    the diffuse term, and for discrete sources the aperture already defines the
    collection.
    """
    m = lambertian_order(semiangle_deg)
    rx = np.array([room.rx_x, room.rx_y, room.rx_z], dtype=float)
    n_rx = np.array([0.0, 0.0, 1.0])
    total = 0.0
    n_in = 0
    for (lx, ly) in positions:
        src = np.array([lx, ly, mount_height], dtype=float)
        link = rx - src
        d = float(np.linalg.norm(link))
        if d < 1e-3:
            d = 1e-3
        u = link / d
        cos_phi = float(np.clip(np.dot(np.array([0.0, 0.0, -1.0]), u), 0.0, 1.0))
        cos_psi = float(np.clip(np.dot(n_rx, -u), 0.0, 1.0))
        psi_deg = math.degrees(math.acos(min(cos_psi, 1.0)))
        if psi_deg > rx_fov_deg or cos_phi <= 0.0:
            continue
        n_in += 1
        h0 = ((m + 1.0) * aperture_area_m2 / (2.0 * math.pi * d ** 2)) \
            * (cos_phi ** m) * cos_psi
        total += power_w * h0
    return total, n_in


def _diffuse_irradiance(room, total_emitted_w: float) -> float:
    """Integrating-sphere diffuse irradiance on any interior surface [W/m^2]."""
    rho = average_reflectivity(room)
    S = room_surface_area(room)
    if S <= 0 or rho >= 0.999:
        return 0.0
    return total_emitted_w * rho / (S * (1.0 - rho))


# ==========================================================================
#  Master background model
# ==========================================================================
@dataclass
class BackgroundResult:
    contributions: List[BackgroundContribution] = field(default_factory=list)
    total_optical_power_w: float = 0.0
    total_photon_flux_hz: float = 0.0
    total_count_rate_hz: float = 0.0          # R_bg, per detector
    mu_bg_per_gate: float = 0.0               # background counts per gate
    effective_filter_bandwidth_nm: float = 0.0
    filter_transmission_at_signal: float = 0.0
    collection_factor_sr: float = 0.0
    integration_time_s: float = 0.0
    spectrum_lambda_nm: Optional[np.ndarray] = None
    spectrum_irradiance: Optional[np.ndarray] = None   # W m^-2 nm^-1 at aperture
    provenance: str = "analytic + simplified background model"

    def breakdown(self) -> Dict[str, float]:
        return {c.name: c.count_rate_hz for c in self.contributions}


def evaluate_background(params,
                        spectral_filter: Optional[SpectralFilter] = None,
                        integration_time_s: Optional[float] = None,
                        wavelength_nm_override: Optional[float] = None
                        ) -> BackgroundResult:
    """
    Compute the background count rate R_bg [counts/s] per detector and the
    mean number of background counts per detection gate mu_bg.

    `integration_time_s` defaults to the detector gate width (gated mode) or to
    the pulse period (free-running mode).
    """
    room = params.room
    ch = params.channel
    det = params.detector
    bg = params.background
    lam0 = float(wavelength_nm_override or params.source.wavelength_nm)

    sf = spectral_filter or SpectralFilter.from_params(params.filt)
    lam = wavelength_grid(sf.center_nm, sf.bandwidth_nm)
    t_filt = sf.transmission(lam)
    t_sig = sf.transmission_at(lam0)
    bw_eff = sf.effective_bandwidth_nm()

    A = ch.rx_aperture_area_cm2 * 1e-4
    G = background_collection_factor(ch.rx_fov_deg, ch.concentrator_index,
                                     ch.use_concentrator)
    e_ph = photon_energy(lam0)

    if integration_time_s is None:
        if det.operation_mode == "Gated":
            integration_time_s = det.gate_width_ns * 1e-9
        else:
            integration_time_s = 1.0 / max(params.source.pulse_rate_hz, 1.0)

    eta_rx = ch.optical_efficiency * det.efficiency

    contributions: List[BackgroundContribution] = []
    spectral_irr_total = np.zeros_like(lam)   # W m^-2 nm^-1 at the aperture

    def trapz(y, x):
        return float(np.trapezoid(y, x)) if hasattr(np, "trapezoid") else float(np.trapz(y, x))

    # ------------------------------------------------------------------
    # A. LED room illumination
    # ------------------------------------------------------------------
    if bg.led.enabled and bg.led.n_luminaires > 0:
        p_led = bg.led.optical_power_per_luminaire_w
        if bg.led.drive_by_illuminance:
            # lux -> total optical watts, via luminous efficacy of radiation.
            floor_area = room.length_x * room.width_y
            lumens_needed = bg.led.target_illuminance_lux * floor_area
            p_total = lumens_needed / C.WHITE_LED_LER_LM_PER_W_OPT
            p_led = p_total / max(bg.led.n_luminaires, 1)
        positions = (bg.led.positions if bg.led.positions
                     else default_luminaire_positions(room, bg.led.n_luminaires))
        s_led = white_led_spectrum(lam)   # 1/nm

        # LOS term: collected broadband power * spectral shape * filter
        p_los, n_in_fov = _discrete_luminaire_irradiance(
            room, positions, bg.led.mount_height_m, p_led,
            bg.led.semiangle_deg, ch.rx_fov_deg, A)
        # convert collected power into an equivalent spectral irradiance at the
        # aperture so that everything can be integrated on one grid
        irr_los = (p_los / max(A, 1e-12)) * s_led        # W m^-2 nm^-1

        irr_diff = np.zeros_like(lam)
        if bg.led.include_reflections:
            e_diff = _diffuse_irradiance(room, p_led * bg.led.n_luminaires)
            irr_diff = e_diff * s_led

        irr = irr_los + irr_diff
        spectral_irr_total += irr

        p_inband_los = trapz(irr_los * A * t_filt, lam)
        p_inband_diff = trapz(irr_diff * A * t_filt * (G / math.pi), lam)
        p_inband = p_inband_los + p_inband_diff
        flux = p_inband / e_ph
        contributions.append(BackgroundContribution(
            name="LED room illumination",
            optical_power_w=p_inband,
            photon_flux_hz=flux,
            count_rate_hz=flux * eta_rx,
            spectral_irradiance_w_m2_nm=float(np.interp(lam0, lam, irr)),
            los_fraction=(p_inband_los / p_inband if p_inband > 0 else 0.0),
            note=f"{n_in_fov}/{len(positions)} luminaires inside the FOV; "
                 f"{p_led:.2f} W_opt each",
        ))

    # ------------------------------------------------------------------
    # B. Fluorescent lighting
    # ------------------------------------------------------------------
    if bg.fluorescent.enabled and bg.fluorescent.n_luminaires > 0:
        p_f = bg.fluorescent.optical_power_per_luminaire_w
        positions = (bg.fluorescent.positions if bg.fluorescent.positions
                     else default_luminaire_positions(room, bg.fluorescent.n_luminaires))
        s_f = fluorescent_spectrum(lam)
        p_los, n_in_fov = _discrete_luminaire_irradiance(
            room, positions, bg.fluorescent.mount_height_m, p_f,
            bg.fluorescent.semiangle_deg, ch.rx_fov_deg, A)
        irr_los = (p_los / max(A, 1e-12)) * s_f
        irr_diff = np.zeros_like(lam)
        if bg.fluorescent.include_reflections:
            e_diff = _diffuse_irradiance(room, p_f * bg.fluorescent.n_luminaires)
            irr_diff = e_diff * s_f
        irr = irr_los + irr_diff
        spectral_irr_total += irr
        p_inband = (trapz(irr_los * A * t_filt, lam)
                    + trapz(irr_diff * A * t_filt * (G / math.pi), lam))
        flux = p_inband / e_ph
        contributions.append(BackgroundContribution(
            name="Fluorescent lighting",
            optical_power_w=p_inband,
            photon_flux_hz=flux,
            count_rate_hz=flux * eta_rx,
            spectral_irradiance_w_m2_nm=float(np.interp(lam0, lam, irr)),
            note=f"{n_in_fov}/{len(positions)} lamps in FOV; Hg lines at "
                 f"{', '.join(str(x) for x in C.FLUORESCENT_LINES_NM)} nm; "
                 f"{C.FLUORESCENT_MODULATION_HZ:.0f} Hz intensity modulation "
                 f"not resolved (mean rate only)",
        ))

    # ------------------------------------------------------------------
    # C. Sunlight through a window
    # ------------------------------------------------------------------
    if bg.sunlight.enabled:
        s_sun = solar_spectrum(lam)
        sp = bg.sunlight
        # Treat the window as a Lambertian secondary source of total
        # transmitted power  E_out * A_win * T_win, seen from distance d_win.
        p_win = sp.outdoor_irradiance_w_m2 * sp.window_area_m2 * sp.window_transmission
        d = max(sp.window_distance_m, 1e-3)
        # Lambertian (m = 1) emission from the window into the room
        h_win = (2.0 * A) / (2.0 * math.pi * d ** 2)
        if sp.direct_sun_in_fov:
            # worst case: the solar disc itself is inside the FOV -> the
            # aperture sees the full transmitted irradiance
            irr_direct = (sp.outdoor_irradiance_w_m2 * sp.window_transmission) * s_sun
        else:
            irr_direct = (p_win * h_win / max(A, 1e-12)) * s_sun
        e_diff = _diffuse_irradiance(room, p_win)
        irr_diff = e_diff * s_sun
        irr = irr_direct + irr_diff
        spectral_irr_total += irr
        p_inband = (trapz(irr_direct * A * t_filt, lam)
                    + trapz(irr_diff * A * t_filt * (G / math.pi), lam))
        flux = p_inband / e_ph
        contributions.append(BackgroundContribution(
            name="Sunlight (window)",
            optical_power_w=p_inband,
            photon_flux_hz=flux,
            count_rate_hz=flux * eta_rx,
            spectral_irradiance_w_m2_nm=float(np.interp(lam0, lam, irr)),
            note=("direct solar disc inside FOV (worst case)"
                  if sp.direct_sun_in_fov else
                  "window treated as a Lambertian secondary source"),
        ))

    # ------------------------------------------------------------------
    # E. Isotropic literature preset
    # ------------------------------------------------------------------
    if bg.use_isotropic_preset:
        p_n = bg.isotropic_spectral_irradiance_w_m2_nm
        irr = p_n * np.ones_like(lam)
        spectral_irr_total += irr
        # etendue-limited collection for a uniform radiance field
        p_inband = trapz(irr * A * t_filt * (G / math.pi), lam)
        flux = p_inband / e_ph
        contributions.append(BackgroundContribution(
            name=f"Isotropic preset ({bg.isotropic_preset})",
            optical_power_w=p_inband,
            photon_flux_hz=flux,
            count_rate_hz=flux * eta_rx,
            spectral_irradiance_w_m2_nm=p_n,
            note="spectrally flat literature value, collected over the receiver etendue",
        ))

    # ------------------------------------------------------------------
    # D. User-defined
    # ------------------------------------------------------------------
    if bg.user.enabled:
        v = float(bg.user.value)
        mode = bg.user.mode
        if mode.startswith("Background optical power"):
            p_inband = v
            flux = p_inband / e_ph
            rate = flux * eta_rx
        elif mode.startswith("Background photon flux"):
            flux = v
            p_inband = flux * e_ph
            rate = flux * eta_rx
        elif mode.startswith("Spectral irradiance"):
            irr = v * np.ones_like(lam)
            spectral_irr_total += irr
            p_inband = trapz(irr * A * t_filt * (G / math.pi), lam)
            flux = p_inband / e_ph
            rate = flux * eta_rx
        else:  # count rate
            rate = v
            flux = rate / max(eta_rx, 1e-15)
            p_inband = flux * e_ph
        contributions.append(BackgroundContribution(
            name="User-defined background",
            optical_power_w=p_inband,
            photon_flux_hz=flux,
            count_rate_hz=rate,
            note=f"entered as: {mode}",
        ))

    total_p = sum(c.optical_power_w for c in contributions)
    total_flux = sum(c.photon_flux_hz for c in contributions)
    total_rate = sum(c.count_rate_hz for c in contributions)

    return BackgroundResult(
        contributions=contributions,
        total_optical_power_w=total_p,
        total_photon_flux_hz=total_flux,
        total_count_rate_hz=total_rate,
        mu_bg_per_gate=total_rate * integration_time_s,
        effective_filter_bandwidth_nm=bw_eff,
        filter_transmission_at_signal=t_sig,
        collection_factor_sr=G,
        integration_time_s=integration_time_s,
        spectrum_lambda_nm=lam,
        spectrum_irradiance=spectral_irr_total,
    )


# ==========================================================================
#  Named comparison scenarios (requirement 18)
# ==========================================================================
#
# The scenarios are defined by ILLUMINANCE (lux), because that is the quantity
# a lighting engineer actually specifies, and converted to radiometric optical
# watts inside the model through the luminous efficacy of radiation.  The lux
# values follow EN 12464-1 lighting practice.
#
BACKGROUND_SCENARIOS = {
    "1 - No background noise": dict(
        led=False, lux=0.0, fluorescent=False, sunlight=False,
        note="dark laboratory; only detector dark counts remain"),
    "2 - Low indoor illumination": dict(
        led=True, lux=10.0, n_lum=4, fluorescent=False, sunlight=False,
        note="heavily dimmed room / night lighting, ~10 lx"),
    "3 - Moderate indoor illumination": dict(
        led=True, lux=150.0, n_lum=4, fluorescent=False, sunlight=False,
        note="living-room level, ~150 lx (EN 12464-1)"),
    "4 - High indoor illumination": dict(
        led=True, lux=500.0, n_lum=9, fluorescent=False, sunlight=False,
        note="standard office level, ~500 lx (EN 12464-1)"),
    "5 - High illumination + daylight": dict(
        led=True, lux=500.0, n_lum=9, fluorescent=False, sunlight=True,
        note="office lighting plus a sunlit window"),
}


def apply_scenario(params, scenario_name: str):
    """Return a *copy* of params with the named background scenario applied."""
    p = params.copy()
    sc = BACKGROUND_SCENARIOS.get(scenario_name)
    if sc is None:
        return p
    p.background.led.enabled = bool(sc.get("led", False))
    if p.background.led.enabled:
        p.background.led.n_luminaires = int(sc.get("n_lum", 4))
        p.background.led.drive_by_illuminance = True
        p.background.led.target_illuminance_lux = float(sc.get("lux", 0.0))
        p.background.led.positions = []
    p.background.fluorescent.enabled = bool(sc.get("fluorescent", False))
    p.background.sunlight.enabled = bool(sc.get("sunlight", False))
    p.background.use_isotropic_preset = False
    p.background.user.enabled = False
    return p
