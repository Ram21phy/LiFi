"""
qkd/qber.py
===========
Analytic QBER model with an explicit, additive error budget.

Construction (per detection gate, matched bases)
------------------------------------------------
Let

    p_s  = 1 - exp(-eta_sys mu)      probability of at least one SIGNAL click
    Y_0  = 1 - exp(-N_det mu_noise)  probability of at least one NOISE click
                                     (background + dark + afterpulse) in the
                                     two detectors of the measured basis
    e_d                              intrinsic (polarisation/pointing) error

Signal and noise clicks are independent, so the three disjoint outcomes are

    signal only   : p_s (1 - Y_0)     -> error with probability e_d
    noise  only   : (1 - p_s) Y_0     -> error with probability e_0 = 1/2
    both          : p_s Y_0           -> one of the two clicks is kept at
                                         random, giving an error probability
                                         (1/2) e_d + (1/2)(1/2) = e_d/2 + 1/4

Hence

    Q_mu    = 1 - (1 - Y_0) exp(-eta_sys mu)          [overall gain]
    E_mu Q_mu = p_s(1-Y_0) e_d
              + (1-p_s) Y_0 * (1/2)
              + p_s Y_0 * (e_d/2 + 1/4)
    QBER    = E_mu = (E_mu Q_mu) / Q_mu

This is exactly the rule the event-by-event Monte Carlo applies, so the
analytic and Monte-Carlo QBER agree to within sampling noise.  (The familiar
textbook expression  E_mu Q_mu = e_0 Y_0 + e_d (1-e^{-eta mu})  is the
first-order expansion of the above and is also available for comparison.)

Error-budget decomposition
--------------------------
The total  E_mu Q_mu  is split into contributions that sum *exactly* to the
total, so that the displayed budget is a genuine decomposition and not a set of
independently computed numbers:

    Q_signal_error   from e_d, split further into polarisation and pointing
    Q_background     noise-click errors, weighted mu_bg / mu_noise
    Q_dark           noise-click errors, weighted mu_dark / mu_noise
    Q_afterpulse     noise-click errors, weighted mu_ap / mu_noise

All results are ANALYTIC.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

E0_RANDOM = 0.5


@dataclass
class QBERBudget:
    qber_total: float
    gain_q_mu: float
    p_signal_click: float
    y0_vacuum_yield: float
    e_intrinsic: float

    # absolute error-probability contributions (sum == qber_total)
    q_signal_error: float = 0.0
    q_polarization: float = 0.0
    q_pointing: float = 0.0
    q_background: float = 0.0
    q_dark: float = 0.0
    q_afterpulse: float = 0.0

    # useful auxiliaries
    err_prob_total: float = 0.0        # E_mu * Q_mu
    signal_to_background_ratio: float = 0.0
    signal_to_noise_ratio: float = 0.0
    textbook_qber: float = 0.0
    provenance: str = "analytic"

    def as_dict(self) -> Dict[str, float]:
        return dict(self.__dict__)

    def as_percentage_table(self):
        rows = [
            ("Signal (polarisation) errors", self.q_polarization),
            ("Signal (pointing) errors", self.q_pointing),
            ("Background-photon errors", self.q_background),
            ("Dark-count errors", self.q_dark),
            ("Afterpulse errors", self.q_afterpulse),
        ]
        return [(n, v, 100.0 * v) for n, v in rows] + [
            ("TOTAL QBER", self.qber_total, 100.0 * self.qber_total)]


# --------------------------------------------------------------------------
def compute_qber(eta_sys: float,
                 mu: float,
                 mu_bg: float,
                 mu_dark: float,
                 e_intrinsic: float,
                 n_detectors: int = 2,
                 mu_afterpulse: float = 0.0,
                 e_polarization: float = None,
                 e_pointing: float = 0.0,
                 ideal_single_photon: bool = False) -> QBERBudget:
    """
    Full analytic QBER with error budget.

    Parameters
    ----------
    eta_sys : total single-photon transmittance Tx -> detection
    mu      : mean photon number per pulse at the transmitter
    mu_bg   : background counts per gate per detector
    mu_dark : dark counts per gate per detector
    e_intrinsic : e_d, probability that a signal click lands on the wrong detector
    n_detectors : detectors per basis contributing noise (2 for passive BB84)
    """
    mu_noise = max(mu_bg + mu_dark + mu_afterpulse, 0.0)
    y0 = 1.0 - math.exp(-n_detectors * mu_noise)

    if ideal_single_photon:
        p_s = eta_sys
    else:
        p_s = 1.0 - math.exp(-eta_sys * mu)

    q_mu = 1.0 - (1.0 - y0) * (1.0 - p_s)
    if q_mu <= 0:
        return QBERBudget(0.0, 0.0, p_s, y0, e_intrinsic)

    ed = float(min(max(e_intrinsic, 0.0), 0.5))

    # disjoint outcome probabilities
    p_sig_only = p_s * (1.0 - y0)
    p_noise_only = (1.0 - p_s) * y0
    p_both = p_s * y0

    err_signal = p_sig_only * ed + p_both * (0.5 * ed)
    err_noise = p_noise_only * E0_RANDOM + p_both * 0.25
    err_total = err_signal + err_noise

    qber = err_total / q_mu

    # --- split the noise error among its physical origins ----------------
    if mu_noise > 0:
        w_bg = mu_bg / mu_noise
        w_dk = mu_dark / mu_noise
        w_ap = mu_afterpulse / mu_noise
    else:
        w_bg = w_dk = w_ap = 0.0

    q_bg = err_noise * w_bg / q_mu
    q_dk = err_noise * w_dk / q_mu
    q_ap = err_noise * w_ap / q_mu
    q_sig = err_signal / q_mu

    # --- split the signal error between polarisation and pointing --------
    if e_polarization is None:
        e_polarization = ed
    denom = max(e_polarization + e_pointing, 1e-30)
    q_pol = q_sig * (e_polarization / denom)
    q_pnt = q_sig * (e_pointing / denom)

    # --- ratios ----------------------------------------------------------
    sbr = (p_s / (n_detectors * mu_bg)) if mu_bg > 0 else float("inf")
    snr = (p_s / (n_detectors * mu_noise)) if mu_noise > 0 else float("inf")

    # --- textbook first-order form for comparison ------------------------
    textbook = ((E0_RANDOM * y0 + ed * p_s) / q_mu) if q_mu > 0 else 0.0

    return QBERBudget(
        qber_total=float(min(max(qber, 0.0), 1.0)),
        gain_q_mu=float(q_mu),
        p_signal_click=float(p_s),
        y0_vacuum_yield=float(y0),
        e_intrinsic=ed,
        q_signal_error=float(q_sig),
        q_polarization=float(q_pol),
        q_pointing=float(q_pnt),
        q_background=float(q_bg),
        q_dark=float(q_dk),
        q_afterpulse=float(q_ap),
        err_prob_total=float(err_total),
        signal_to_background_ratio=float(sbr),
        signal_to_noise_ratio=float(snr),
        textbook_qber=float(min(max(textbook, 0.0), 1.0)),
    )


# --------------------------------------------------------------------------
def single_photon_yield_and_error(eta_sys: float,
                                  mu_bg: float,
                                  mu_dark: float,
                                  e_intrinsic: float,
                                  n_detectors: int = 2,
                                  mu_afterpulse: float = 0.0):
    """
    Exact (model-internal) single-photon yield Y_1 and error e_1, obtained by
    applying the *same* click-combination rule with p_s -> eta_sys.

        Y_1   = 1 - (1 - Y_0)(1 - eta_sys)
        e_1 Y_1 = eta_sys(1-Y_0) e_d + (1-eta_sys) Y_0 /2
                  + eta_sys Y_0 (e_d/2 + 1/4)

    These are the values an *infinite-decoy* protocol would recover exactly.
    """
    mu_noise = max(mu_bg + mu_dark + mu_afterpulse, 0.0)
    y0 = 1.0 - math.exp(-n_detectors * mu_noise)
    ed = float(min(max(e_intrinsic, 0.0), 0.5))
    eta = float(min(max(eta_sys, 0.0), 1.0))

    y1 = 1.0 - (1.0 - y0) * (1.0 - eta)
    if y1 <= 0:
        return 0.0, 0.0
    err = (eta * (1.0 - y0) * ed
           + (1.0 - eta) * y0 * E0_RANDOM
           + eta * y0 * (0.5 * ed + 0.25))
    e1 = err / y1
    return float(y1), float(min(max(e1, 0.0), 0.5))
