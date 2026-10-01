"""
qkd/key_rate.py
===============
Asymptotic BB84 secret-key rate.

Standard BB84 with a weak-coherent source (GLLP)
------------------------------------------------
    R = q r_p { Q_1 [1 - h2(e_1)] - f_EC Q_mu h2(E_mu) }

with Q_1, e_1 obtained from the *pessimistic* GLLP bounds
(qkd.decoy_state.gllp_single_photon_bounds).

Decoy-state BB84
----------------
Identical structure, but Q_1 and e_1 come from the vacuum+weak decoy
estimation (qkd.decoy_state.vacuum_weak_decoy), or from the exact
infinite-decoy model values when the user selects that idealisation.

Basis-sifting factor q
----------------------
    symmetric BB84              q = 1/2
    biased / efficient BB84     q = p_Z^2 + (1 - p_Z)^2

Units
-----
    secret_fraction          bits per SIFTED bit
    secret_key_rate_per_pulse bits per transmitted pulse
    secret_key_rate_bps       bits per second (multiply by r_p)

Everything here is ASYMPTOTIC (infinite key length, no statistical
fluctuations, collective-attack-style one-way post-processing).  It is *not* a
composable finite-key security proof.  The finite-size module applies a
separate, clearly labelled correction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from qkd.information_metrics import binary_entropy, privacy_amplification_term


@dataclass
class KeyRateResult:
    protocol: str
    q_sift: float
    gain_q_mu: float
    qber_e_mu: float
    q1: float
    e1: float
    error_correction_leak_per_pulse: float
    privacy_amplification_per_pulse: float
    secret_fraction_per_sifted_bit: float
    secret_key_per_pulse: float
    secret_key_rate_bps: float
    sifted_key_per_pulse: float
    sifted_key_rate_bps: float
    detection_rate_bps: float
    secure: bool
    message: str = ""
    provenance: str = "analytic (asymptotic)"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def sifting_factor(basis_bias_z: float, efficient: bool) -> float:
    """q = 1/2 for symmetric BB84; p_Z^2 + (1-p_Z)^2 for the biased variant."""
    if not efficient:
        return 0.5
    p = float(min(max(basis_bias_z, 0.0), 1.0))
    return p * p + (1.0 - p) * (1.0 - p)


def asymptotic_key_rate(q_sift: float,
                        pulse_rate_hz: float,
                        q_mu: float,
                        e_mu: float,
                        q1: float,
                        e1: float,
                        f_ec: float,
                        protocol: str = "Standard BB84 (GLLP)") -> KeyRateResult:
    """
    R/pulse = q { Q_1 [1 - h2(e_1)] - f_EC Q_mu h2(E_mu) }
    """
    h_e1 = float(binary_entropy(e1))
    h_emu = float(binary_entropy(e_mu))

    pa_per_pulse = q_sift * q1 * (1.0 - h_e1)      # what survives PA
    ec_per_pulse = q_sift * f_ec * q_mu * h_emu    # what EC leaks

    r_per_pulse = pa_per_pulse - ec_per_pulse
    sifted_per_pulse = q_sift * q_mu

    frac = (r_per_pulse / sifted_per_pulse) if sifted_per_pulse > 0 else 0.0

    secure = r_per_pulse > 0.0
    msg = "" if secure else (
        "No positive asymptotic key rate under the selected model "
        "(error correction + privacy amplification exceed the sifted key).")

    return KeyRateResult(
        protocol=protocol,
        q_sift=float(q_sift),
        gain_q_mu=float(q_mu),
        qber_e_mu=float(e_mu),
        q1=float(q1),
        e1=float(e1),
        error_correction_leak_per_pulse=float(ec_per_pulse),
        privacy_amplification_per_pulse=float(pa_per_pulse),
        secret_fraction_per_sifted_bit=float(frac),
        secret_key_per_pulse=float(r_per_pulse),
        secret_key_rate_bps=float(r_per_pulse * pulse_rate_hz),
        sifted_key_per_pulse=float(sifted_per_pulse),
        sifted_key_rate_bps=float(sifted_per_pulse * pulse_rate_hz),
        detection_rate_bps=float(q_mu * pulse_rate_hz),
        secure=bool(secure),
        message=msg,
    )


def shor_preskill_rate(q_sift: float, pulse_rate_hz: float,
                       q_mu: float, e_mu: float, f_ec: float) -> float:
    """
    Reference curve only: the Shor-Preskill rate for an *ideal single-photon*
    source,  R = q Q [1 - f_EC h2(E) - h2(E)].  Included so the user can see
    how much of the penalty comes from the weak-coherent source.
    """
    h = float(binary_entropy(e_mu))
    return float(q_sift * q_mu * (1.0 - f_ec * h - h) * pulse_rate_hz)
