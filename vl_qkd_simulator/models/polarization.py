"""
models/polarization.py
======================
Polarisation-encoding error model for BB84.

Three independent mechanisms are combined into the intrinsic *signal* error
probability e_d, i.e. the probability that a photon that genuinely reaches the
correct-basis detector pair lands on the WRONG detector:

1. Reference-frame misalignment by an angle theta between Alice's and Bob's
   polarisation axes.  A photon prepared in |H> and analysed in a frame rotated
   by theta has

        P(wrong) = sin^2(theta)

2. Finite polariser / PBS extinction ratio ER (in dB):

        P(wrong) = 1 / (1 + 10^(ER/10))

3. Channel depolarisation of fraction p_dep (the state becomes maximally mixed
   with probability p_dep, giving a 1/2 error on that fraction):

        P(wrong) = p_dep / 2

The three are combined as independent error channels,

        e_d = 1 - (1 - e_theta)(1 - e_ER)(1 - e_dep)

and clipped to [0, 1/2].  (An error probability above 1/2 is unphysical for a
binary symmetric channel: a receiver would simply flip its bit.)

This module contains NO information-theoretic content; it is pure device
physics.  The user may bypass it entirely with `intrinsic_error_override`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass
class PolarizationResult:
    e_misalignment: float
    e_extinction: float
    e_depolarization: float
    e_intrinsic: float           # e_d
    misalignment_angle_deg: float
    extinction_ratio_db: float
    overridden: bool = False
    provenance: str = "analytic"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def misalignment_error(theta_deg: float) -> float:
    """sin^2(theta)."""
    return float(math.sin(math.radians(theta_deg)) ** 2)


def extinction_error(er_db: float) -> float:
    """1 / (1 + 10^(ER/10)) -- leakage into the orthogonal port."""
    if er_db <= 0:
        return 0.5
    return 1.0 / (1.0 + 10.0 ** (er_db / 10.0))


def depolarization_error(p_dep: float) -> float:
    return 0.5 * max(0.0, min(1.0, p_dep))


def evaluate_polarization(params) -> PolarizationResult:
    pol = params.polarization
    e_th = misalignment_error(pol.misalignment_angle_deg)
    e_er = extinction_error(pol.extinction_ratio_db)
    e_dp = depolarization_error(pol.depolarization)

    if pol.intrinsic_error_override is not None:
        e_d = float(min(max(pol.intrinsic_error_override, 0.0), 0.5))
        return PolarizationResult(e_th, e_er, e_dp, e_d,
                                  pol.misalignment_angle_deg,
                                  pol.extinction_ratio_db, overridden=True)

    e_d = 1.0 - (1.0 - e_th) * (1.0 - e_er) * (1.0 - e_dp)
    e_d = float(min(max(e_d, 0.0), 0.5))
    return PolarizationResult(e_th, e_er, e_dp, e_d,
                              pol.misalignment_angle_deg,
                              pol.extinction_ratio_db)


# --------------------------------------------------------------------------
# BB84 state definitions (used by the Monte Carlo and for display)
# --------------------------------------------------------------------------
BASIS_Z = 0
BASIS_X = 1

STATE_NAMES = {
    (BASIS_Z, 0): "|H>",
    (BASIS_Z, 1): "|V>",
    (BASIS_X, 0): "|D>",
    (BASIS_X, 1): "|A>",
}

BASIS_NAMES = {BASIS_Z: "Z = {|H>, |V>}", BASIS_X: "X = {|D>, |A>}"}

# Jones vectors (for display / didactic purposes)
JONES = {
    "|H>": (1.0, 0.0),
    "|V>": (0.0, 1.0),
    "|D>": (1 / math.sqrt(2), 1 / math.sqrt(2)),
    "|A>": (1 / math.sqrt(2), -1 / math.sqrt(2)),
}


def state_name(basis: int, bit: int) -> str:
    return STATE_NAMES[(basis, bit)]
