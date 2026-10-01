"""
qkd/decoy_state.py
==================
Vacuum + weak decoy-state estimation, following

    X. Ma, B. Qi, Y. Zhao, H.-K. Lo, "Practical decoy state for quantum key
    distribution", Phys. Rev. A 72, 012326 (2005).

Alice randomly uses three intensities: signal mu, weak decoy nu1, vacuum nu2
(nu2 = 0 by default), with mu > nu1 > nu2 >= 0.

From the measured gains and error rates {Q_mu, E_mu, Q_nu1, E_nu1, Q_nu2,
E_nu2} the following bounds are computed:

Vacuum yield (Eq. 34 of Ma et al.)
    Y_0^L = max{ (nu1 Q_nu2 e^{nu2} - nu2 Q_nu1 e^{nu1}) / (nu1 - nu2),  0 }

Single-photon yield (Eq. 35)
    Y_1^L >= mu / (mu nu1 - mu nu2 - nu1^2 + nu2^2)
             * [ Q_nu1 e^{nu1} - Q_nu2 e^{nu2}
                 - (nu1^2 - nu2^2)/mu^2 * (Q_mu e^{mu} - Y_0^L) ]

Single-photon gain
    Q_1^L = Y_1^L mu e^{-mu}

Single-photon error rate (Eq. 37)
    e_1^U <= (E_nu1 Q_nu1 e^{nu1} - E_nu2 Q_nu2 e^{nu2})
             / ( (nu1 - nu2) Y_1^L )

These are ASYMPTOTIC bounds (infinite number of decoy pulses, no statistical
fluctuations).  Finite-statistics versions live in simulation/finite_size.py
and are labelled separately.

The module also provides `infinite_decoy_bounds`, which returns the *exact*
model values of Y_1 and e_1 -- the limit that an unlimited number of decoy
intensities would attain.  Comparing the two shows the estimation penalty of a
two-decoy implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class DecoyEstimate:
    y0_lower: float
    y1_lower: float
    q1_lower: float
    e1_upper: float
    mu: float
    nu1: float
    nu2: float
    q_mu: float
    e_mu: float
    q_nu1: float
    e_nu1: float
    q_nu2: float
    e_nu2: float
    valid: bool = True
    message: str = ""
    provenance: str = "analytic (asymptotic decoy bounds, Ma et al. 2005)"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def vacuum_weak_decoy(mu: float, nu1: float, nu2: float,
                      q_mu: float, e_mu: float,
                      q_nu1: float, e_nu1: float,
                      q_nu2: float, e_nu2: float) -> DecoyEstimate:
    msg = ""
    valid = True
    if not (mu > nu1 > nu2 >= 0.0):
        return DecoyEstimate(0, 0, 0, 0.5, mu, nu1, nu2, q_mu, e_mu,
                             q_nu1, e_nu1, q_nu2, e_nu2,
                             valid=False,
                             message="Decoy intensities must satisfy mu > nu1 > nu2 >= 0")

    # ---- Y_0 lower bound ------------------------------------------------
    num = nu1 * q_nu2 * math.exp(nu2) - nu2 * q_nu1 * math.exp(nu1)
    y0_l = max(num / (nu1 - nu2), 0.0)

    # ---- Y_1 lower bound ------------------------------------------------
    denom = mu * nu1 - mu * nu2 - nu1 ** 2 + nu2 ** 2
    if abs(denom) < 1e-18:
        return DecoyEstimate(y0_l, 0, 0, 0.5, mu, nu1, nu2, q_mu, e_mu,
                             q_nu1, e_nu1, q_nu2, e_nu2, valid=False,
                             message="Degenerate decoy intensities (denominator ~ 0)")

    bracket = (q_nu1 * math.exp(nu1) - q_nu2 * math.exp(nu2)
               - (nu1 ** 2 - nu2 ** 2) / (mu ** 2) * (q_mu * math.exp(mu) - y0_l))
    y1_l = (mu / denom) * bracket
    if y1_l <= 0:
        y1_l = 0.0
        valid = False
        msg = ("Decoy lower bound on Y_1 is non-positive: the chosen decoy "
               "intensities cannot certify any single-photon contribution.")

    q1_l = y1_l * mu * math.exp(-mu)

    # ---- e_1 upper bound ------------------------------------------------
    if y1_l > 0:
        e_num = e_nu1 * q_nu1 * math.exp(nu1) - e_nu2 * q_nu2 * math.exp(nu2)
        e1_u = e_num / ((nu1 - nu2) * y1_l)
        e1_u = float(min(max(e1_u, 0.0), 0.5))
    else:
        e1_u = 0.5

    return DecoyEstimate(y0_l, max(y1_l, 0.0), max(q1_l, 0.0), e1_u,
                         mu, nu1, nu2, q_mu, e_mu, q_nu1, e_nu1,
                         q_nu2, e_nu2, valid=valid, message=msg)


def infinite_decoy_bounds(y1_exact: float, e1_exact: float, mu: float) -> DecoyEstimate:
    """Exact model values (the infinite-decoy limit)."""
    return DecoyEstimate(
        y0_lower=float("nan"),
        y1_lower=y1_exact,
        q1_lower=y1_exact * mu * math.exp(-mu),
        e1_upper=e1_exact,
        mu=mu, nu1=float("nan"), nu2=float("nan"),
        q_mu=float("nan"), e_mu=float("nan"),
        q_nu1=float("nan"), e_nu1=float("nan"),
        q_nu2=float("nan"), e_nu2=float("nan"),
        message="infinite-decoy limit (exact model values)",
        provenance="analytic (infinite-decoy limit)",
    )


# --------------------------------------------------------------------------
def gllp_single_photon_bounds(mu: float, q_mu: float, e_mu: float,
                              y0: Optional[float] = None):
    """
    GLLP bounds for STANDARD (non-decoy) BB84 with a weak coherent source.

    Without decoy states Alice and Bob cannot distinguish single-photon from
    multi-photon detections, so the pessimistic assumption is made that every
    multi-photon pulse was detected by Eve's PNS attack:

        Q_1^L = max{ Q_mu - P_multi(mu), 0 },  P_multi = 1 - e^-mu - mu e^-mu
        e_1^U = min{ E_mu Q_mu / Q_1^L , 1/2 }

    (The vacuum contribution Y_0 e^{-mu} may optionally be subtracted too, when
    Y_0 is known/bounded, giving a slightly tighter Q_1^L.)

    Returns (Q_1^L, e_1^U, P_multi).
    """
    p_multi = max(0.0, 1.0 - math.exp(-mu) - mu * math.exp(-mu))
    q1 = q_mu - p_multi
    if y0 is not None:
        q1 -= y0 * math.exp(-mu)
    q1 = max(q1, 0.0)
    if q1 <= 0:
        return 0.0, 0.5, p_multi
    e1 = min(e_mu * q_mu / q1, 0.5)
    return float(q1), float(e1), float(p_multi)
