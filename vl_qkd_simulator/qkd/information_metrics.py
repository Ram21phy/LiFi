"""
qkd/information_metrics.py
==========================
Information-theoretic quantities.  Nothing in this module knows anything about
optics: it consumes gains and error rates and produces bits.

Binary entropy
--------------
    h2(x) = -x log2(x) - (1-x) log2(1-x),     h2(0) = h2(1) = 0

Implemented with explicit, numerically stable handling of the endpoints.

Mutual information (binary symmetric channel approximation)
-----------------------------------------------------------
Alice's sifted bit -> Bob's sifted bit is modelled as a BSC with crossover
probability QBER = E.  For a uniform binary input,

    I(A:B) = 1 - h2(E)                                   [bits per sifted bit]
    I_AB   = R_sift * [1 - h2(E)]                        [bits per second]

Eve's information / privacy amplification
-----------------------------------------
Two clearly separated models are offered.

(a) GLLP / decoy asymptotic (the default, and the one used for the key rate).
    Only the single-photon detections can contribute to a secret key.  The
    privacy-amplification cost per *sifted* bit is

        PA = 1 - (Q_1 / Q_mu) [1 - h2(e_1)]

    and the corresponding "information available to Eve" per sifted bit is
    exactly that PA term:  multi-photon detections are conceded to Eve in full
    (PNS attack) and single-photon detections cost h2(e_1).

(b) Individual-attack / intercept-resend reference curve (didactic only):

        I_E^(ind) = h2(E)        (symmetric individual attack, one-way)

    This is NOT used for the key rate; it is plotted alongside to show how much
    more conservative the composable-style GLLP bound is.

The distinction between (a) an asymptotic simplified model and (b) a composably
secure finite-key analysis is made explicit everywhere in the GUI
(requirement 12).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

Number = Union[float, np.ndarray]


# --------------------------------------------------------------------------
def binary_entropy(x: Number) -> Number:
    """
    h2(x) = -x log2 x - (1-x) log2 (1-x), with h2(0)=h2(1)=0.

    Values outside [0,1] are clipped; NaNs propagate as 0 at the endpoints.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        p = float(np.clip(arr, 0.0, 1.0))
        if p <= 0.0 or p >= 1.0 or not math.isfinite(p):
            return 0.0
        return float(-p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p))

    v = np.clip(arr, 0.0, 1.0)
    out = np.zeros_like(v, dtype=float)
    mask = np.isfinite(v) & (v > 0.0) & (v < 1.0)
    vm = v[mask]
    out[mask] = -vm * np.log2(vm) - (1.0 - vm) * np.log2(1.0 - vm)
    return out


def inverse_binary_entropy(y: float, lo: float = 0.0, hi: float = 0.5,
                           tol: float = 1e-12) -> float:
    """Smallest x in [0, 1/2] with h2(x) = y (bisection)."""
    y = float(np.clip(y, 0.0, 1.0))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binary_entropy(mid) < y:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
def mutual_information_per_bit(qber: Number) -> Number:
    """I(A:B) = 1 - h2(QBER)  [bits per sifted bit]."""
    return 1.0 - binary_entropy(qber)


def mutual_information_rate(sifted_rate_hz: Number, qber: Number) -> Number:
    """I_AB = R_sift * [1 - h2(QBER)]  [bits/s]."""
    return np.asarray(sifted_rate_hz) * mutual_information_per_bit(qber)


def eve_information_individual(qber: Number) -> Number:
    """Didactic reference only: I_E = h2(E) for a symmetric individual attack."""
    return binary_entropy(qber)


def privacy_amplification_term(q_mu: float, q1: float, e1: float) -> float:
    """
    PA cost per sifted bit for the GLLP / decoy asymptotic model:

        PA = 1 - (Q_1/Q_mu) [1 - h2(e_1)]

    Returns a value in [0, 1]; 1 means *all* of the sifted key must be
    sacrificed and no secret key can be produced.
    """
    if q_mu <= 0:
        return 1.0
    frac = max(min(q1 / q_mu, 1.0), 0.0)
    pa = 1.0 - frac * (1.0 - float(binary_entropy(e1)))
    return float(min(max(pa, 0.0), 1.0))


def secret_fraction_from_terms(q_mu: float, q1: float, e1: float,
                               e_mu: float, f_ec: float) -> float:
    """
    Secret-key fraction per SIFTED bit:

        r = (Q_1/Q_mu)[1 - h2(e_1)] - f_EC h2(E_mu)
          = [1 - PA] - f_EC h2(E_mu)

    May be negative, which is the signature of "no secure key".
    """
    pa = privacy_amplification_term(q_mu, q1, e1)
    return float((1.0 - pa) - f_ec * float(binary_entropy(e_mu)))


# --------------------------------------------------------------------------
@dataclass
class InformationMetrics:
    qber: float
    mutual_information_per_bit: float
    mutual_information_rate_bps: float
    eve_information_gllp: float           # = PA term, per sifted bit
    eve_information_individual: float     # didactic reference
    secret_fraction_per_sifted_bit: float
    error_correction_leakage_per_bit: float
    provenance: str = "analytic (asymptotic)"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def evaluate_information(qber: float, sifted_rate_hz: float,
                         q_mu: float, q1: float, e1: float,
                         f_ec: float) -> InformationMetrics:
    iab = float(mutual_information_per_bit(qber))
    pa = privacy_amplification_term(q_mu, q1, e1)
    leak = f_ec * float(binary_entropy(qber))
    return InformationMetrics(
        qber=float(qber),
        mutual_information_per_bit=iab,
        mutual_information_rate_bps=float(sifted_rate_hz * iab),
        eve_information_gllp=pa,
        eve_information_individual=float(eve_information_individual(qber)),
        secret_fraction_per_sifted_bit=(1.0 - pa) - leak,
        error_correction_leakage_per_bit=leak,
    )
