"""
models/pointing_error.py
========================
Optional pointing / beam-wander model (Farid & Hranilovic, JLT 25, 1702 (2007)).

A Gaussian beam of width w(d) at the receiver plane, with a radial pointing
displacement r that is Rayleigh distributed with parameter sigma_s, produces a
fractional collected power

    h_p(r) ~= A_0 exp( -2 r^2 / w_eq^2 )

with

    v      = sqrt(pi) a / (sqrt(2) w),      a = receiver aperture radius
    A_0    = [erf(v)]^2                     (fraction collected on boresight)
    w_eq^2 = w^2 sqrt(pi) erf(v) / (2 v exp(-v^2))

The *mean* transmittance over the Rayleigh-distributed jitter is

    <h_p> = A_0 * gamma^2 / (gamma^2 + 1),     gamma = w_eq / (2 sigma_s)

and the pdf of h_p is

    f(h) = gamma^2 / A_0^(gamma^2) * h^(gamma^2 - 1),    0 <= h <= A_0

which is used when the Monte Carlo samples pointing fades.

Here sigma_s is the *lateral* displacement standard deviation at the receiver,
obtained from the angular jitter sigma_theta as  sigma_s = d * sigma_theta.

When the pointing model is disabled the transmittance is exactly 1 and the
module contributes nothing.  Pointing error is treated as a LOSS mechanism; it
does not by itself flip polarisation, so it does not enter e_d unless the user
enables the optional angular-depolarisation coupling below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.special import erf


@dataclass
class PointingResult:
    enabled: bool
    sigma_lateral_m: float
    beam_radius_m: float
    aperture_radius_m: float
    a0: float
    w_eq_m: float
    gamma: float
    mean_transmittance: float
    induced_error: float = 0.0
    provenance: str = "analytic"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def beam_radius(distance_m: float, waist_mm: float, divergence_mrad: float) -> float:
    return waist_mm * 1e-3 + max(distance_m, 0.0) * divergence_mrad * 1e-3


def evaluate_pointing(params, distance_m: Optional[float] = None) -> PointingResult:
    ch = params.channel
    if distance_m is None:
        from models.vlc_channel import compute_geometry
        distance_m = compute_geometry(params.room, ch.rx_fov_deg).distance_m

    if not ch.enable_pointing_error:
        return PointingResult(False, 0.0, 0.0, 0.0, 1.0, 0.0,
                              float("inf"), 1.0)

    a = math.sqrt(max(ch.rx_aperture_area_cm2 * 1e-4, 1e-12) / math.pi)
    if ch.channel_model.startswith("Collimated"):
        w = beam_radius(distance_m, ch.beam_waist_mm, ch.beam_divergence_mrad)
    else:
        # effective spot radius of a Lambertian source at the receiver plane,
        # taken as d * tan(Phi_1/2)
        w = max(distance_m * math.tan(math.radians(
            max(ch.tx_half_power_semiangle_deg, 0.05))), 1e-6)

    sigma_s = distance_m * ch.pointing_jitter_mrad * 1e-3

    v = math.sqrt(math.pi) * a / (math.sqrt(2.0) * max(w, 1e-12))
    a0 = float(erf(v) ** 2)
    denom = 2.0 * v * math.exp(-v * v)
    w_eq2 = (w ** 2) * math.sqrt(math.pi) * float(erf(v)) / max(denom, 1e-30)
    w_eq = math.sqrt(max(w_eq2, 1e-30))

    if sigma_s <= 0:
        return PointingResult(True, 0.0, w, a, a0, w_eq, float("inf"), a0)

    gamma = w_eq / (2.0 * sigma_s)
    mean_t = a0 * (gamma ** 2) / (gamma ** 2 + 1.0)
    # Normalise out the boresight aperture loss already present in the channel
    # gain so that pointing_error only contributes the *jitter* penalty.
    mean_rel = mean_t / a0 if a0 > 0 else 1.0

    return PointingResult(True, sigma_s, w, a, a0, w_eq, gamma,
                          float(min(max(mean_rel, 0.0), 1.0)))


def sample_pointing_transmittance(rng: np.random.Generator,
                                  res: PointingResult, size: int) -> np.ndarray:
    """
    Draw h_p/A_0 from f(h) = gamma^2 h^(gamma^2-1) on [0,1]
    (inverse-cdf sampling: h = U^(1/gamma^2)).
    """
    if not res.enabled or not np.isfinite(res.gamma) or res.gamma <= 0:
        return np.ones(size)
    u = rng.random(size)
    return np.power(u, 1.0 / (res.gamma ** 2))
