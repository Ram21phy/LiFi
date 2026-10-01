"""
models/turbulence.py
====================
Optional weak-turbulence (scintillation) model.

For a plane/spherical wave over a horizontal path of length L, the Rytov
variance is

    sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6),      k = 2 pi / lambda

In the weak-fluctuation regime the irradiance follows a log-normal
distribution.  The channel fading factor h is written

    h = exp(2 X),    X ~ N(mu_x, sigma_x^2),
    sigma_x^2 = sigma_R^2 / 4,   mu_x = -sigma_x^2

so that E[h] = 1 (the turbulence redistributes power, it does not absorb it)
and Var[h] = exp(sigma_R^2) - 1 = scintillation index.

INDOOR CAVEAT
-------------
Indoor refractive-index structure constants are very small
(Cn^2 ~ 1e-15 ... 1e-13 m^-2/3) and over a few metres the resulting Rytov
variance is typically << 1, i.e. turbulence is a *negligible* effect for a
5 m indoor link.  The model is included because the specification asks for it
and because the same simulator can then be pointed at longer or hotter paths
(e.g. across a corridor with strong HVAC gradients).  The GUI states this.

Because the mean transmittance is unity, turbulence does not change the
*average* channel gain; what it does is make the key rate a non-linear average
over the fading distribution.  `average_over_fading` performs that average with
Gauss-Hermite quadrature, which is the physically correct way to include it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


@dataclass
class TurbulenceResult:
    enabled: bool
    cn2: float
    rytov_variance: float
    scintillation_index: float
    sigma_x: float
    mean_transmittance: float
    regime: str
    provenance: str = "analytic"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def rytov_variance(cn2: float, wavelength_nm: float, distance_m: float) -> float:
    k = 2.0 * math.pi / (wavelength_nm * 1e-9)
    return 1.23 * cn2 * (k ** (7.0 / 6.0)) * (max(distance_m, 1e-6) ** (11.0 / 6.0))


def evaluate_turbulence(params, distance_m: Optional[float] = None) -> TurbulenceResult:
    ch = params.channel
    if distance_m is None:
        from models.vlc_channel import compute_geometry
        distance_m = compute_geometry(params.room, ch.rx_fov_deg).distance_m

    if not ch.enable_turbulence:
        return TurbulenceResult(False, 0.0, 0.0, 0.0, 0.0, 1.0, "disabled")

    sr2 = rytov_variance(ch.turbulence_cn2, params.source.wavelength_nm, distance_m)
    si = math.exp(min(sr2, 50.0)) - 1.0
    sigma_x = math.sqrt(max(sr2, 0.0)) / 2.0
    if sr2 < 0.3:
        regime = "weak (log-normal valid)"
    elif sr2 < 1.0:
        regime = "moderate (log-normal approximate)"
    else:
        regime = "strong (log-normal NOT valid - use gamma-gamma; not implemented)"
    return TurbulenceResult(True, ch.turbulence_cn2, sr2, si, sigma_x, 1.0, regime)


def sample_fading(rng: np.random.Generator, res: TurbulenceResult,
                  size: int) -> np.ndarray:
    if not res.enabled or res.sigma_x <= 0:
        return np.ones(size)
    x = rng.normal(-res.sigma_x ** 2, res.sigma_x, size=size)
    return np.exp(2.0 * x)


def average_over_fading(res: TurbulenceResult,
                        func: Callable[[float], float],
                        n_nodes: int = 21) -> float:
    """
    E[ func(h) ] with h log-normal, using Gauss-Hermite quadrature.

        E[f(h)] = (1/sqrt(pi)) sum_i w_i f( exp(2(sqrt(2) sigma_x t_i + mu_x)) )
    """
    if not res.enabled or res.sigma_x <= 0:
        return float(func(1.0))
    t, w = np.polynomial.hermite.hermgauss(n_nodes)
    mu_x = -res.sigma_x ** 2
    h = np.exp(2.0 * (math.sqrt(2.0) * res.sigma_x * t + mu_x))
    vals = np.array([func(float(hi)) for hi in h])
    return float(np.sum(w * vals) / math.sqrt(math.pi))
