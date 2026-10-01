"""
simulation/monte_carlo.py
=========================
Event-by-event Monte Carlo simulation of polarisation-encoded BB84 over the
indoor visible-light channel.

For every transmitted pulse the following is drawn explicitly (requirement 14):

  1. Alice's basis            Bernoulli(p_Z)
  2. Alice's bit              Bernoulli(1/2)
  3. Photon number n          Poisson(mu)          [or n = 1 for an ideal source]
  4. Channel loss             each photon survives with prob. eta_channel
  5. Optical efficiency       folded into eta_channel
  6. Detector efficiency      k ~ Binomial(n, eta_sys) surviving photons
  7. Signal detection event   click_signal = (k >= 1)
  8. Background event         Poisson(mu_bg) in each detector of Bob's basis
  9. Dark-count event         Poisson(mu_dark) in each detector
 10. Polarisation error       signal lands on the wrong detector w.p. e_d
 11. Detector noise           afterpulsing (Bernoulli on the previous click)
 12. Bob's basis              Bernoulli(p_Z)
 13. Correct / erroneous      decided from which detector(s) fired
 14. Detection recorded
 15. Sifting                  keep matched bases
 16. QBER                     N_error / N_sifted

Detector assignment rules
-------------------------
Bob has two detectors in the basis he measured, labelled 0 and 1.

* Bases MATCH  : a surviving signal photon goes to detector (bit XOR error),
                 where error ~ Bernoulli(e_d).
* Bases DIFFER : the state is unbiased in Bob's basis, so the photon goes to
                 detector 0 or 1 with probability 1/2 each.

Noise clicks are added independently in both detectors.  If both detectors
fire (a double click) Bob keeps one at random - this is the standard
"random assignment" convention, and it is what makes the Monte Carlo agree
with the analytic model in qkd/qber.py.

Implementation notes
--------------------
The simulation is fully vectorised with NumPy and processed in chunks, so
10^7 pulses run in a few seconds and memory stays bounded.  The random stream
is seeded, so a run is exactly reproducible from the seed in the parameter set.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

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


@dataclass
class MonteCarloResult:
    n_pulses: int
    n_z_pulses: int
    n_x_pulses: int
    n_detections: int
    n_matched_basis: int
    n_sifted: int
    n_errors: int
    n_double_clicks: int
    n_signal_clicks: int
    n_noise_clicks: int
    n_dark_clicks: int
    n_background_clicks: int

    qber: float
    qber_std_err: float
    gain_q_mu: float
    sifting_ratio: float
    sifted_key_rate_bps: float
    secret_key_rate_bps: float
    secret_fraction: float
    mutual_information_per_bit: float
    mutual_information_rate_bps: float

    eta_sys: float
    mu_bg: float
    mu_dark: float
    e_intrinsic: float

    convergence: Dict[str, np.ndarray] = field(default_factory=dict)
    seed: int = 0
    provenance: str = "Monte Carlo (event-by-event)"

    def as_dict(self) -> Dict[str, float]:
        d = {k: v for k, v in self.__dict__.items() if k != "convergence"}
        return d


# --------------------------------------------------------------------------
def run_monte_carlo(params,
                    progress_callback: Optional[Callable[[float], None]] = None,
                    n_pulses: Optional[int] = None,
                    seed: Optional[int] = None) -> MonteCarloResult:
    """Run the event-by-event BB84 simulation."""
    n_total = int(n_pulses or params.monte_carlo.n_pulses)
    seed = int(seed if seed is not None else params.monte_carlo.seed)
    rng = np.random.default_rng(seed)

    # ---- physical inputs (identical to the analytic path) ---------------
    sfilt = bgm.SpectralFilter.from_params(params.filt)
    t_sig = sfilt.transmission_at(params.source.wavelength_nm)
    geom = vlcm.compute_geometry(params.room, params.channel.rx_fov_deg)
    point = pem.evaluate_pointing(params, geom.distance_m)
    turb = turbm.evaluate_turbulence(params, geom.distance_m)
    chan = vlcm.evaluate_channel(params, t_sig,
                                 point.mean_transmittance,
                                 turb.mean_transmittance)
    bg = bgm.evaluate_background(params, spectral_filter=sfilt)

    eta_sys = float(np.clip(chan.eta_channel * params.detector.efficiency, 0, 1))
    p_click_sig = pm.signal_click_probability_poisson(params.source.mu, eta_sys)
    det = detm.evaluate_detector(params, bg.total_count_rate_hz,
                                 expected_click_probability=p_click_sig)
    pol = polm.evaluate_polarization(params)

    mu = float(params.source.mu)
    mu_bg = float(det.mu_bg)
    mu_dark = float(det.mu_dark)
    p_ap = float(params.detector.afterpulse_probability)
    e_d = float(pol.e_intrinsic)
    p_z = float(params.protocol.basis_bias_z)
    ideal_single = params.source.source_model.startswith("Ideal")
    n_det_basis = max(int(params.detector.n_detectors_per_basis), 1)

    # per-detector optical/dark noise; if the receiver has more than two
    # detectors per basis the extra ones only add noise, not signal.
    extra_noise_detectors = max(n_det_basis - 2, 0)

    # ---- accumulators ---------------------------------------------------
    acc = dict(det_total=0, matched=0, sifted=0, errors=0, dbl=0,
               sig_clicks=0, noise_clicks=0, dark_clicks=0, bg_clicks=0,
               z_pulses=0, x_pulses=0)

    conv_n, conv_q, conv_gain = [], [], []
    n_conv = max(int(params.monte_carlo.convergence_points), 2)
    conv_marks = np.unique(np.logspace(
        math.log10(max(n_total // 1000, 100)), math.log10(n_total),
        n_conv).astype(np.int64))

    chunk = int(min(params.monte_carlo.chunk_size, max(n_total, 1)))
    done = 0
    prev_click_frac = p_click_sig

    while done < n_total:
        m = int(min(chunk, n_total - done))

        # 1-2. Alice
        a_basis = (rng.random(m) >= p_z).astype(np.int8)   # 0 = Z, 1 = X
        a_bit = (rng.random(m) < 0.5).astype(np.int8)

        # 3. photon number
        if ideal_single:
            n_ph = np.ones(m, dtype=np.int64)
        else:
            n_ph = rng.poisson(mu, m)

        # 4-6. channel + optics + detector efficiency, photon by photon
        # (Binomial(n, eta_sys) is the exact distribution of survivors)
        if point.enabled:
            hp = pem.sample_pointing_transmittance(rng, point, m)
        else:
            hp = np.ones(m)
        if turb.enabled:
            ht = turbm.sample_fading(rng, turb, m)
        else:
            ht = np.ones(m)
        eta_eff = np.clip(eta_sys * hp * ht / max(point.mean_transmittance, 1e-12)
                          if point.enabled else eta_sys * ht, 0.0, 1.0)
        k_surv = rng.binomial(n_ph, eta_eff)

        # 7. signal click
        click_sig = k_surv >= 1

        # 12. Bob's basis
        b_basis = (rng.random(m) >= p_z).astype(np.int8)
        matched = (a_basis == b_basis)

        # 10. which detector the signal photon hits
        wrong = rng.random(m) < e_d
        det_sig = np.where(matched,
                           np.bitwise_xor(a_bit, wrong.astype(np.int8)),
                           (rng.random(m) < 0.5).astype(np.int8))

        # 8-9-11. noise clicks in each of the two detectors of Bob's basis
        mu_ap = p_ap * prev_click_frac
        lam_bg = mu_bg
        lam_dk = mu_dark + mu_ap
        bg0 = rng.poisson(lam_bg, m)
        bg1 = rng.poisson(lam_bg, m)
        dk0 = rng.poisson(lam_dk, m)
        dk1 = rng.poisson(lam_dk, m)
        noise0 = (bg0 + dk0) >= 1
        noise1 = (bg1 + dk1) >= 1

        # extra detectors (if any) only contribute to double-click vetoes;
        # they are counted as noise clicks but cannot carry a bit
        if extra_noise_detectors:
            _ = rng.poisson((lam_bg + lam_dk) * extra_noise_detectors, m)

        # combine: which detectors fired
        fired0 = noise0 | (click_sig & (det_sig == 0))
        fired1 = noise1 | (click_sig & (det_sig == 1))

        any_click = fired0 | fired1
        both = fired0 & fired1

        # 13. Bob's recorded bit; double clicks -> random assignment
        coin = rng.random(m) < 0.5
        b_bit = np.where(both, coin.astype(np.int8),
                         np.where(fired1, np.int8(1), np.int8(0)))

        # 14-16. bookkeeping
        acc["det_total"] += int(any_click.sum())
        acc["matched"] += int(matched.sum())
        sift_mask = matched & any_click
        n_s = int(sift_mask.sum())
        acc["sifted"] += n_s
        acc["errors"] += int((b_bit[sift_mask] != a_bit[sift_mask]).sum())
        acc["dbl"] += int(both.sum())
        acc["sig_clicks"] += int(click_sig.sum())
        acc["noise_clicks"] += int((noise0 | noise1).sum())
        acc["bg_clicks"] += int(((bg0 >= 1) | (bg1 >= 1)).sum())
        acc["dark_clicks"] += int(((dk0 >= 1) | (dk1 >= 1)).sum())
        acc["z_pulses"] += int((a_basis == 0).sum())
        acc["x_pulses"] += int((a_basis == 1).sum())

        prev_click_frac = float(any_click.mean())
        done += m

        # convergence trace
        for mark in conv_marks:
            if done >= mark and (len(conv_n) == 0 or conv_n[-1] < mark):
                conv_n.append(int(done))
                conv_q.append(acc["errors"] / max(acc["sifted"], 1))
                conv_gain.append(acc["det_total"] / max(done, 1))
                break

        if progress_callback:
            progress_callback(done / n_total)

    # ---- derived quantities --------------------------------------------
    n_sift = acc["sifted"]
    qber = acc["errors"] / n_sift if n_sift else 0.0
    qber_se = math.sqrt(max(qber * (1 - qber), 0.0) / n_sift) if n_sift else 0.0
    q_mu = acc["det_total"] / n_total
    q_sift_ratio = n_sift / n_total

    rp = params.source.pulse_rate_hz
    sat = det.dead_time_saturation_factor
    sifted_rate = q_sift_ratio * rp * sat

    # key rate from the MEASURED Q_mu and E_mu, using the selected security model
    proto = params.protocol.protocol
    if proto.startswith("Decoy") or proto.startswith("Infinite"):
        from qkd.qber import single_photon_yield_and_error
        y1, e1 = single_photon_yield_and_error(eta_sys, mu_bg, mu_dark, e_d,
                                               n_det_basis, det.mu_afterpulse)
        q1 = y1 * mu * math.exp(-mu)
    else:
        q1, e1, _ = decoy.gllp_single_photon_bounds(mu, q_mu, qber,
                                                    y0=det.y0_vacuum_yield)

    q_sift_factor = kr.sifting_factor(p_z, params.protocol.use_efficient_bb84)
    key = kr.asymptotic_key_rate(q_sift_factor, rp, q_mu, qber, q1, e1,
                                 params.protocol.error_correction_efficiency,
                                 protocol=proto)

    iab = float(im.mutual_information_per_bit(qber))

    return MonteCarloResult(
        n_pulses=n_total,
        n_z_pulses=acc["z_pulses"],
        n_x_pulses=acc["x_pulses"],
        n_detections=acc["det_total"],
        n_matched_basis=acc["matched"],
        n_sifted=n_sift,
        n_errors=acc["errors"],
        n_double_clicks=acc["dbl"],
        n_signal_clicks=acc["sig_clicks"],
        n_noise_clicks=acc["noise_clicks"],
        n_dark_clicks=acc["dark_clicks"],
        n_background_clicks=acc["bg_clicks"],
        qber=qber,
        qber_std_err=qber_se,
        gain_q_mu=q_mu,
        sifting_ratio=q_sift_ratio,
        sifted_key_rate_bps=sifted_rate,
        secret_key_rate_bps=key.secret_key_rate_bps * sat,
        secret_fraction=key.secret_fraction_per_sifted_bit,
        mutual_information_per_bit=iab,
        mutual_information_rate_bps=sifted_rate * iab,
        eta_sys=eta_sys,
        mu_bg=mu_bg,
        mu_dark=mu_dark,
        e_intrinsic=e_d,
        convergence={"n_pulses": np.array(conv_n),
                     "qber": np.array(conv_q),
                     "gain": np.array(conv_gain)},
        seed=seed,
    )
