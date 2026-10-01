"""
models/detector.py
==================
Simplified single-photon avalanche diode (SPAD) receiver model.

Per detection gate of width dt the model tracks three *independent* Poissonian
click mechanisms:

    mu_bg    = R_bg   * dt        background optical counts   (per detector)
    mu_dark  = R_dark * dt        dark counts                 (per detector)
    mu_ap    = p_ap * P_click_prev  afterpulses (first-order approximation)

    mu_noise = mu_bg + mu_dark + mu_ap

No-click probability of ONE detector from noise alone:   exp(-mu_noise)
Click probability of ONE detector from noise alone:      1 - exp(-mu_noise)

A standard passive BB84 receiver has two detectors per basis, and only the two
detectors of the *measured* basis can produce a sifted event, so the vacuum
yield (the probability that an empty pulse nevertheless yields a detection) is

    Y_0 = 1 - exp( -N_det * mu_noise ),      N_det = 2 by default.

Dead time is applied as a non-paralysable saturation correction on the measured
count rate,

    R_meas = R_true / (1 + R_true * tau_dead),

which is applied to the *reported* rates, and as a duty-cycle reduction of the
achievable sifted rate.

Timing jitter is reported and converted into an effective gate broadening,
    dt_eff = sqrt(dt^2 + (2.355*sigma_jitter)^2),
which increases the collected background if the user selects that option.

LABEL: SIMPLIFIED DETECTOR MODEL.  Afterpulsing memory, twilight pulses,
detector crosstalk and time-dependent recovery are deliberately not modelled;
the class is structured so they can be added without touching the QKD layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DetectorResult:
    efficiency: float
    gate_width_s: float
    effective_gate_width_s: float
    dark_count_rate_hz: float
    background_count_rate_hz: float
    mu_dark: float
    mu_bg: float
    mu_afterpulse: float
    mu_noise_per_detector: float
    n_detectors_per_basis: int
    y0_vacuum_yield: float
    p_noise_click_single_detector: float
    dead_time_s: float
    dead_time_saturation_factor: float
    timing_jitter_s: float
    provenance: str = "analytic + simplified detector model"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def effective_gate_width(gate_width_s: float, jitter_s: float,
                         include_jitter: bool = True) -> float:
    """Quadrature combination of the electrical gate and the timing jitter."""
    if not include_jitter:
        return gate_width_s
    fwhm_jitter = 2.3548 * jitter_s
    return math.sqrt(gate_width_s ** 2 + fwhm_jitter ** 2)


def evaluate_detector(params,
                      background_count_rate_hz: float,
                      expected_click_probability: float = 0.0,
                      include_jitter_in_gate: bool = True) -> DetectorResult:
    """
    Build the per-gate noise budget.

    `expected_click_probability` is used only for the first-order afterpulsing
    estimate; pass the previous iteration's click probability (or the signal
    click probability) for a self-consistent estimate.
    """
    det = params.detector
    dt = det.gate_width_ns * 1e-9
    jitter = det.timing_jitter_ps * 1e-12
    dt_eff = effective_gate_width(dt, jitter, include_jitter_in_gate)

    if det.operation_mode != "Gated":
        dt_eff = 1.0 / max(params.source.pulse_rate_hz, 1.0)

    mu_dark = det.dark_count_rate_hz * dt_eff
    mu_bg = background_count_rate_hz * dt_eff
    mu_ap = det.afterpulse_probability * max(expected_click_probability, 0.0)

    mu_noise = mu_dark + mu_bg + mu_ap
    n_det = max(int(det.n_detectors_per_basis), 1)

    y0 = 1.0 - math.exp(-n_det * mu_noise)
    p_single = 1.0 - math.exp(-mu_noise)

    tau_d = det.dead_time_ns * 1e-9
    # saturation factor evaluated at the *total* incident click rate
    r_true = (background_count_rate_hz + det.dark_count_rate_hz
              + expected_click_probability * params.source.pulse_rate_hz)
    sat = 1.0 / (1.0 + r_true * tau_d) if tau_d > 0 else 1.0

    return DetectorResult(
        efficiency=det.efficiency,
        gate_width_s=dt,
        effective_gate_width_s=dt_eff,
        dark_count_rate_hz=det.dark_count_rate_hz,
        background_count_rate_hz=background_count_rate_hz,
        mu_dark=mu_dark,
        mu_bg=mu_bg,
        mu_afterpulse=mu_ap,
        mu_noise_per_detector=mu_noise,
        n_detectors_per_basis=n_det,
        y0_vacuum_yield=y0,
        p_noise_click_single_detector=p_single,
        dead_time_s=tau_d,
        dead_time_saturation_factor=sat,
        timing_jitter_s=jitter,
    )


# --------------------------------------------------------------------------
def vacuum_yield(mu_bg: float, mu_dark: float, n_detectors: int = 2) -> float:
    """Y_0 = 1 - exp(-N_det (mu_bg + mu_dark))."""
    return 1.0 - math.exp(-n_detectors * (mu_bg + mu_dark))


def signal_click_probability(eta_sys: float, mu: float) -> float:
    """1 - exp(-eta_sys mu) for a Poissonian source."""
    return 1.0 - math.exp(-eta_sys * mu)


def overall_gain(eta_sys: float, mu: float, y0: float) -> float:
    """Q_mu = 1 - (1 - Y_0) exp(-eta_sys mu)."""
    return 1.0 - (1.0 - y0) * math.exp(-eta_sys * mu)


def dead_time_corrected_rate(rate_hz: float, dead_time_s: float) -> float:
    """Non-paralysable dead-time saturation."""
    if dead_time_s <= 0:
        return rate_hz
    return rate_hz / (1.0 + rate_hz * dead_time_s)
