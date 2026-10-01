"""
simulation/finite_size.py
=========================
SIMPLIFIED finite-key analysis.

This module is deliberately kept separate from qkd/key_rate.py so that the
asymptotic and finite-length results can never be confused (requirement 12/24).

What is implemented
-------------------
A simplified version of the decoy-state finite-key bound of

    C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden,
    "Concise security bounds for practical decoy-state QKD",
    Phys. Rev. A 89, 022307 (2014),

in the form

    l <= s_1 [ 1 - h2(phi_1) ] - lambda_EC
         - 6 log2(19/eps_sec) - log2(2/eps_cor)

where

    N        number of transmitted pulses
    n_Z      number of sifted bits in the key basis
    s_1      lower bound on the number of single-photon detections in n_Z
    phi_1    upper bound on the single-photon PHASE error rate, obtained from
             the bit error rate e_1 plus a statistical-deviation term
    lambda_EC = f_EC * n_Z * h2(E)   bits leaked by error correction
    eps_sec, eps_cor   secrecy and correctness parameters

Statistical deviations use the Hoeffding bound

    delta(n, eps) = sqrt( n ln(1/eps) / 2 )

which is simple, valid, and slightly looser than the Chernoff bounds used in
the original paper.

WHAT THIS IS NOT
----------------
This is *not* a full composable security proof of a physical implementation.
The decoy-state photon-number statistics are treated at the level of the
asymptotic bounds with an added deviation term rather than through the complete
finite-statistics optimisation, and no side-channel, source-flaw or
authentication cost is included.  The GUI labels every number produced here as
"finite-size (simplified)".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from qkd.information_metrics import binary_entropy


@dataclass
class FiniteKeyResult:
    n_pulses: float
    n_sifted: float
    n_single_photon: float
    e1_bit: float
    phi1_phase: float
    ec_leakage_bits: float
    pa_cost_bits: float
    finite_correction_bits: float
    secret_key_length_bits: float
    secret_fraction_per_sifted_bit: float
    secret_key_rate_bps: float
    asymptotic_key_rate_bps: float
    penalty_factor: float
    secure: bool
    message: str = ""
    provenance: str = "finite-size (simplified)"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)


def hoeffding_deviation(n: float, eps: float) -> float:
    """delta = sqrt( n ln(1/eps) / 2 )."""
    if n <= 0 or eps <= 0 or eps >= 1:
        return 0.0
    return math.sqrt(0.5 * n * math.log(1.0 / eps))


def finite_key_length(n_pulses: float,
                      q_mu: float,
                      e_mu: float,
                      q1: float,
                      e1: float,
                      f_ec: float,
                      q_sift: float,
                      eps_sec: float = 1e-10,
                      eps_cor: float = 1e-15,
                      pulse_rate_hz: float = 1.0,
                      asymptotic_rate_bps: float = 0.0,
                      throughput_factor: float = 1.0) -> FiniteKeyResult:
    """
    Simplified finite-key secret length for the given block of `n_pulses`.
    """
    n_sift = q_sift * q_mu * n_pulses
    if n_sift < 1.0:
        return FiniteKeyResult(n_pulses, n_sift, 0, e1, 0.5, 0, 0, 0, 0.0, 0.0,
                               0.0, asymptotic_rate_bps, 0.0, False,
                               "Block too small: fewer than one sifted bit.")

    # single-photon detections inside the sifted block, with a statistical
    # (Hoeffding) penalty on the estimate of Q_1
    s1_mean = q_sift * q1 * n_pulses
    s1 = max(s1_mean - hoeffding_deviation(n_sift, eps_sec / 7.0), 0.0)

    # phase-error rate: bit-error estimate plus a finite-statistics deviation
    if s1 > 0:
        dev = hoeffding_deviation(s1, eps_sec / 7.0) / s1
    else:
        dev = 0.5
    phi1 = min(e1 + dev, 0.5)

    lambda_ec = f_ec * n_sift * float(binary_entropy(e_mu))
    pa = s1 * (1.0 - float(binary_entropy(phi1)))
    corr = 6.0 * math.log2(19.0 / max(eps_sec, 1e-300)) + math.log2(2.0 / max(eps_cor, 1e-300))

    length = pa - lambda_ec - corr
    secure = length > 0

    frac = length / n_sift if n_sift > 0 else 0.0
    # `throughput_factor` carries the detector dead-time de-rating so that the
    # finite-size rate is directly comparable with the asymptotic one.
    rate = frac * q_sift * q_mu * pulse_rate_hz * throughput_factor
    penalty = (rate / asymptotic_rate_bps) if asymptotic_rate_bps > 0 else 0.0

    msg = "" if secure else (
        "No positive key under the simplified finite-key analysis for this "
        "block size. Increase the block size, reduce the QBER, or accept a "
        "larger eps_sec.")

    return FiniteKeyResult(
        n_pulses=float(n_pulses),
        n_sifted=float(n_sift),
        n_single_photon=float(s1),
        e1_bit=float(e1),
        phi1_phase=float(phi1),
        ec_leakage_bits=float(lambda_ec),
        pa_cost_bits=float(pa),
        finite_correction_bits=float(corr),
        secret_key_length_bits=float(max(length, 0.0)),
        secret_fraction_per_sifted_bit=float(max(frac, 0.0)),
        secret_key_rate_bps=float(max(rate, 0.0)),
        asymptotic_key_rate_bps=float(asymptotic_rate_bps),
        penalty_factor=float(penalty),
        secure=bool(secure),
        message=msg,
    )


def evaluate_finite_key(params, analytic_result) -> FiniteKeyResult:
    """Convenience wrapper driven by SimulationParameters + a BB84Result."""
    pr = params.protocol
    return finite_key_length(
        n_pulses=float(pr.block_size_bits),
        q_mu=analytic_result.qber.gain_q_mu,
        e_mu=analytic_result.qber.qber_total,
        q1=analytic_result.key.q1,
        e1=analytic_result.key.e1,
        f_ec=pr.error_correction_efficiency,
        q_sift=analytic_result.key.q_sift,
        eps_sec=pr.epsilon_sec,
        eps_cor=pr.epsilon_cor,
        pulse_rate_hz=params.source.pulse_rate_hz,
        asymptotic_rate_bps=analytic_result.secret_key_rate_bps,
        throughput_factor=analytic_result.detector.dead_time_saturation_factor,
    )
