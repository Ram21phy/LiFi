"""
qkd/bb84.py
===========
Orchestrator: turns a `SimulationParameters` object into the full analytic
result chain

    indoor room -> VLC channel -> signal photons -> background photons
    -> detector clicks -> BB84 sifting -> QBER -> mutual information
    -> privacy amplification -> secret key rate

The polarisation-encoded BB84 protocol implemented is the standard one:

    Z basis = {|H>, |V>},  X basis = {|D>, |A>}

    1. Alice picks a basis at random (p_Z / 1-p_Z)
    2. Alice picks a bit at random
    3. Alice prepares |H>, |V>, |D> or |A>
    4. Alice sends the weak coherent pulse
    5. Bob picks a measurement basis at random
    6. Bob records a click or a no-click
    7. Bases are announced publicly
    8. Only matched-basis detections are kept  (sifting)
    9. QBER is estimated on a sample
   10. Error correction leaks  f_EC h2(E) Q  bits
   11. Privacy amplification removes  Q_1 h2(e_1) + multiphoton  bits
   12. The remaining bits are the secret key

This module performs the ANALYTIC version of that chain.  The event-by-event
version lives in simulation/monte_carlo.py and uses exactly the same physical
inputs, which is what makes the cross-check in the self-test meaningful.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from models import background_noise as bgm
from models import detector as detm
from models import photon_model as pm
from models import pointing_error as pem
from models import polarization as polm
from models import turbulence as turbm
from models import vlc_channel as vlcm
from qkd import decoy_state as decoy
from qkd import information_metrics as im
from qkd import key_rate as kr
from qkd import qber as qberm
from utils import constants as C


# ==========================================================================
@dataclass
class BB84Result:
    """Everything the GUI needs, in one object."""
    params: Any

    # --- stage results ---------------------------------------------------
    source: pm.PhotonModelResult = None
    channel: vlcm.ChannelResult = None
    background: bgm.BackgroundResult = None
    detector: detm.DetectorResult = None
    polarization: polm.PolarizationResult = None
    pointing: pem.PointingResult = None
    turbulence: turbm.TurbulenceResult = None
    qber: qberm.QBERBudget = None
    info: im.InformationMetrics = None
    key: kr.KeyRateResult = None
    decoy: Optional[decoy.DecoyEstimate] = None

    # --- derived scalars -------------------------------------------------
    eta_system: float = 0.0
    received_photons_per_pulse: float = 0.0
    received_optical_power_w: float = 0.0
    received_photon_rate_hz: float = 0.0
    background_count_rate_hz: float = 0.0
    dark_count_rate_hz: float = 0.0
    total_count_rate_hz: float = 0.0
    signal_count_rate_hz: float = 0.0
    detection_probability: float = 0.0
    sifted_key_rate_bps: float = 0.0
    secret_key_rate_bps: float = 0.0
    secret_key_fraction: float = 0.0
    signal_to_background_ratio: float = 0.0
    signal_to_noise_ratio: float = 0.0

    warnings: list = field(default_factory=list)
    provenance: str = "analytic (asymptotic)"

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        g = self.channel.geometry
        return {
            # channel
            "Link distance [m]": g.distance_m,
            "Irradiance angle phi [deg]": g.irradiance_angle_deg,
            "Incidence angle psi [deg]": g.incidence_angle_deg,
            "Inside receiver FOV": g.in_fov,
            "Lambertian order m": self.channel.lambertian_order,
            "Channel gain H(0)": self.channel.channel_gain_h0,
            "Channel transmittance eta_ch": self.channel.eta_channel,
            "Channel loss [dB]": self.channel.eta_channel_db,
            "System transmittance eta_sys": self.eta_system,
            "Received signal photons / pulse": self.received_photons_per_pulse,
            "Received optical power [W]": self.received_optical_power_w,
            # background
            "Background photons / gate (mu_bg)": self.detector.mu_bg,
            "Background count rate [1/s]": self.background_count_rate_hz,
            "Dark count rate [1/s]": self.dark_count_rate_hz,
            "Vacuum yield Y_0": self.detector.y0_vacuum_yield,
            "Signal-to-background ratio": self.signal_to_background_ratio,
            "Signal-to-noise ratio": self.signal_to_noise_ratio,
            "Effective filter bandwidth [nm]": self.background.effective_filter_bandwidth_nm,
            # qkd
            "Detection probability Q_mu": self.qber.gain_q_mu,
            "Sifting factor q": self.key.q_sift,
            "Sifted key rate [bit/s]": self.sifted_key_rate_bps,
            "QBER": self.qber.qber_total,
            "Mutual information I(A:B) [bit/sifted bit]": self.info.mutual_information_per_bit,
            "Mutual information rate [bit/s]": self.info.mutual_information_rate_bps,
            "Eve information / PA term [bit/sifted bit]": self.info.eve_information_gllp,
            "Single-photon gain Q_1": self.key.q1,
            "Single-photon error e_1": self.key.e1,
            "Secret key fraction [bit/sifted bit]": self.secret_key_fraction,
            "Secret key rate [bit/s]": self.secret_key_rate_bps,
            "Positive key rate": self.key.secure,
        }


# ==========================================================================
def run_analytic(params,
                 distance_override: Optional[float] = None,
                 background_rate_override: Optional[float] = None,
                 self_consistent_afterpulse: bool = True) -> BB84Result:
    """
    Full analytic evaluation.

    `distance_override` and `background_rate_override` exist so that sweeps and
    heat-maps can vary one quantity without rebuilding the whole parameter set.
    """
    warnings: list = []

    # ---- 1. source ------------------------------------------------------
    src = pm.evaluate_source(params.source)
    ideal_single = params.source.source_model.startswith("Ideal")

    # ---- 2. spectral filter --------------------------------------------
    sfilt = bgm.SpectralFilter.from_params(params.filt)
    t_sig = sfilt.transmission_at(params.source.wavelength_nm)
    if t_sig < 0.05:
        warnings.append(
            f"The band-pass filter transmits only {t_sig*100:.2f}% at the signal "
            f"wavelength ({params.source.wavelength_nm:.1f} nm). Check that the "
            f"filter centre ({params.filt.center_nm:.1f} nm) matches the source.")

    # ---- 3. geometry-dependent impairments ------------------------------
    geom0 = vlcm.compute_geometry(params.room, params.channel.rx_fov_deg)
    dist = distance_override if distance_override is not None else geom0.distance_m
    point = pem.evaluate_pointing(params, dist)
    turb = turbm.evaluate_turbulence(params, dist)

    # ---- 4. channel -----------------------------------------------------
    chan = vlcm.evaluate_channel(
        params, filter_transmission_at_signal=t_sig,
        pointing_transmittance=point.mean_transmittance,
        turbulence_mean_transmittance=turb.mean_transmittance,
        distance_override=distance_override)

    if not chan.geometry.in_fov and chan.channel_gain_h0 <= 0:
        warnings.append(
            f"The receiver is outside the field of view "
            f"(incidence angle {chan.geometry.incidence_angle_deg:.1f} deg > "
            f"FOV {params.channel.rx_fov_deg:.1f} deg): channel gain is zero.")

    # ---- 5. background --------------------------------------------------
    bg = bgm.evaluate_background(params, spectral_filter=sfilt)
    r_bg = (background_rate_override if background_rate_override is not None
            else bg.total_count_rate_hz)

    # ---- 6. detector ----------------------------------------------------
    eta_sys = float(np.clip(chan.eta_channel * params.detector.efficiency, 0.0, 1.0))
    p_click_signal = (eta_sys if ideal_single
                      else pm.signal_click_probability_poisson(params.source.mu, eta_sys))

    det = detm.evaluate_detector(params, r_bg, expected_click_probability=p_click_signal)
    if self_consistent_afterpulse:
        # one fixed-point iteration so that afterpulsing sees the true click rate
        q_tmp = 1.0 - (1.0 - det.y0_vacuum_yield) * (1.0 - p_click_signal)
        det = detm.evaluate_detector(params, r_bg, expected_click_probability=q_tmp)

    # ---- 7. polarisation / intrinsic error ------------------------------
    pol = polm.evaluate_polarization(params)
    e_pointing = 0.0
    if params.channel.enable_pointing_error and point.enabled:
        # beam wander does not rotate polarisation; it is modelled as loss.
        e_pointing = 0.0
    e_d = float(min(pol.e_intrinsic + e_pointing, 0.5))

    # ---- 8. QBER --------------------------------------------------------
    if params.protocol.external_qber_enabled:
        budget = qberm.compute_qber(eta_sys, params.source.mu, det.mu_bg,
                                    det.mu_dark, e_d,
                                    det.n_detectors_per_basis,
                                    det.mu_afterpulse,
                                    e_polarization=pol.e_intrinsic,
                                    e_pointing=e_pointing,
                                    ideal_single_photon=ideal_single)
        budget.qber_total = float(np.clip(params.protocol.external_qber, 0.0, 0.5))
        budget.provenance = "user-supplied external QBER (model bypassed)"
        warnings.append("External QBER mode is active: the modelled QBER has "
                        "been overridden by the value you entered.")
    else:
        budget = qberm.compute_qber(eta_sys, params.source.mu, det.mu_bg,
                                    det.mu_dark, e_d,
                                    det.n_detectors_per_basis,
                                    det.mu_afterpulse,
                                    e_polarization=pol.e_intrinsic,
                                    e_pointing=e_pointing,
                                    ideal_single_photon=ideal_single)

    q_mu = budget.gain_q_mu
    e_mu = budget.qber_total

    # ---- 9. single-photon parameters ------------------------------------
    y1_exact, e1_exact = qberm.single_photon_yield_and_error(
        eta_sys, det.mu_bg, det.mu_dark, e_d,
        det.n_detectors_per_basis, det.mu_afterpulse)

    decoy_est = None
    proto = params.protocol.protocol
    if proto.startswith("Decoy"):
        nu1, nu2 = params.source.decoy_nu1, params.source.decoy_nu2
        if nu1 >= params.source.mu:
            warnings.append("Decoy intensity nu1 must be smaller than mu; "
                            "falling back to the infinite-decoy limit.")
            q1, e1 = y1_exact * params.source.mu * math.exp(-params.source.mu), e1_exact
            decoy_est = decoy.infinite_decoy_bounds(y1_exact, e1_exact, params.source.mu)
        else:
            # simulate the decoy measurements with the SAME physical model
            dec_meas = {}
            for name, nu in (("nu1", nu1), ("nu2", nu2)):
                b = qberm.compute_qber(eta_sys, nu, det.mu_bg, det.mu_dark, e_d,
                                       det.n_detectors_per_basis,
                                       det.mu_afterpulse,
                                       e_polarization=pol.e_intrinsic,
                                       e_pointing=e_pointing,
                                       ideal_single_photon=False)
                dec_meas[name] = (b.gain_q_mu, b.qber_total)
            decoy_est = decoy.vacuum_weak_decoy(
                params.source.mu, nu1, nu2, q_mu, e_mu,
                dec_meas["nu1"][0], dec_meas["nu1"][1],
                dec_meas["nu2"][0], dec_meas["nu2"][1])
            q1, e1 = decoy_est.q1_lower, decoy_est.e1_upper
            if not decoy_est.valid and decoy_est.message:
                warnings.append(decoy_est.message)
    elif proto.startswith("Infinite"):
        q1 = y1_exact * params.source.mu * math.exp(-params.source.mu)
        e1 = e1_exact
        decoy_est = decoy.infinite_decoy_bounds(y1_exact, e1_exact, params.source.mu)
    else:  # Standard BB84 (GLLP)
        if ideal_single:
            q1, e1 = q_mu, e_mu
        else:
            q1, e1, p_multi = decoy.gllp_single_photon_bounds(
                params.source.mu, q_mu, e_mu, y0=det.y0_vacuum_yield)
            if q1 <= 0:
                warnings.append(
                    f"GLLP: the multi-photon probability P(n>=2) = {p_multi:.3e} "
                    f"exceeds the total gain Q_mu = {q_mu:.3e}. No single-photon "
                    f"contribution can be certified without decoy states - "
                    f"reduce mu or switch to decoy-state BB84.")

    # ---- 10. sifting and key rate ---------------------------------------
    q_sift = kr.sifting_factor(params.protocol.basis_bias_z,
                               params.protocol.use_efficient_bb84)
    key = kr.asymptotic_key_rate(q_sift, params.source.pulse_rate_hz,
                                 q_mu, e_mu, q1, e1,
                                 params.protocol.error_correction_efficiency,
                                 protocol=proto)

    # dead-time saturation on the *reported* rates
    sat = det.dead_time_saturation_factor
    if sat < 0.95:
        warnings.append(
            f"Detector dead time is saturating the receiver "
            f"(throughput factor {sat:.2f}). Reported sifted and secret rates "
            f"have been de-rated accordingly.")
    key.sifted_key_rate_bps *= sat
    key.secret_key_rate_bps *= sat
    key.detection_rate_bps *= sat

    # ---- 11. information metrics ----------------------------------------
    info = im.evaluate_information(e_mu, key.sifted_key_rate_bps,
                                   q_mu, q1, e1,
                                   params.protocol.error_correction_efficiency)

    # ---- 12. assemble ---------------------------------------------------
    rec_photons = params.source.mu * chan.eta_channel
    res = BB84Result(
        params=params,
        source=src, channel=chan, background=bg, detector=det,
        polarization=pol, pointing=point, turbulence=turb,
        qber=budget, info=info, key=key, decoy=decoy_est,
        eta_system=eta_sys,
        received_photons_per_pulse=rec_photons,
        received_optical_power_w=src.transmitted_optical_power_w * chan.eta_channel,
        received_photon_rate_hz=src.transmitted_photon_rate * chan.eta_channel,
        background_count_rate_hz=r_bg,
        dark_count_rate_hz=params.detector.dark_count_rate_hz,
        signal_count_rate_hz=p_click_signal * params.source.pulse_rate_hz,
        total_count_rate_hz=0.0,   # set immediately below
        detection_probability=q_mu,
        sifted_key_rate_bps=key.sifted_key_rate_bps,
        secret_key_rate_bps=key.secret_key_rate_bps,
        secret_key_fraction=key.secret_fraction_per_sifted_bit,
        signal_to_background_ratio=budget.signal_to_background_ratio,
        signal_to_noise_ratio=budget.signal_to_noise_ratio,
        warnings=warnings,
    )
    res.total_count_rate_hz = q_mu * params.source.pulse_rate_hz * sat

    if turb.enabled and turb.rytov_variance > 1.0:
        res.warnings.append(
            "Rytov variance > 1: the log-normal turbulence model is outside its "
            "validity range. Treat the result as indicative only.")

    return res


# ==========================================================================
def protocol_step_table(params) -> list:
    """Human-readable description of the 12 protocol steps, for the GUI."""
    p = params
    q = kr.sifting_factor(p.protocol.basis_bias_z, p.protocol.use_efficient_bb84)
    return [
        ("1", "Alice selects a basis", f"Z with probability p_Z = {p.protocol.basis_bias_z:.2f}, "
                                       f"X with probability {1-p.protocol.basis_bias_z:.2f}"),
        ("2", "Alice selects a bit", "uniform random, 0 or 1"),
        ("3", "Alice prepares a state", "|H>, |V> (Z basis) or |D>, |A> (X basis)"),
        ("4", "Alice sends the pulse",
         f"weak coherent pulse, mu = {p.source.mu:g}, "
         f"lambda = {p.source.wavelength_nm:g} nm, r_p = {p.source.pulse_rate_hz:.3g} Hz"),
        ("5", "Bob selects a measurement basis", "passive 50/50 choice (or biased)"),
        ("6", "Bob detects / no-click",
         f"gate {p.detector.gate_width_ns:g} ns, eta_det = {p.detector.efficiency:.2f}, "
         f"{p.detector.n_detectors_per_basis} detectors per basis"),
        ("7", "Public basis announcement", "classical authenticated channel"),
        ("8", "Sifting", f"keep matched bases -> q = {q:.3f}"),
        ("9", "QBER estimation", "random sample of the sifted key is disclosed"),
        ("10", "Error correction",
         f"leakage f_EC h2(E) per sifted bit, f_EC = {p.protocol.error_correction_efficiency:g}"),
        ("11", "Privacy amplification",
         "remove Q_1 h2(e_1) + the full multi-photon contribution"),
        ("12", "Secret key", "R = q r_p { Q_1[1-h2(e_1)] - f_EC Q_mu h2(E_mu) }"),
    ]
