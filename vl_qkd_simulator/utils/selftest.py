"""
utils/selftest.py
=================
Start-up self-test suite (requirement 25).

Every test either checks a quantity against an independently-computed closed
form, or checks a LIMITING BEHAVIOUR that the physics demands:

  * photon energy at 520 nm must equal hc/lambda to machine precision
  * the Lambertian channel gain must reduce to the textbook expression at
    normal incidence, and must fall as 1/d^2
  * a Poisson draw of 10^6 samples must reproduce mean and variance = mu
  * the background photon number must scale linearly with filter bandwidth in
    the flat-spectrum limit, and must vanish as the bandwidth -> 0
  * the detector click probability must reduce to 1 - exp(-eta mu)
  * BB84 basis selection must be unbiased at p_Z = 0.5
  * as background -> 0, QBER must approach the intrinsic error floor e_d
  * QBER must increase monotonically with background count rate
  * secret key rate must decrease monotonically with background count rate
  * received photons must decrease with distance
  * at high loss / high noise the secret key rate must become <= 0
  * h2(0) = h2(1) = 0, h2(1/2) = 1, and h2 must be symmetric
  * I(A:B) = 0 at QBER = 1/2 and = 1 at QBER = 0
  * the analytic and Monte-Carlo QBER must agree within statistical error

Run standalone with:   python -m utils.selftest
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

from models import background_noise as bgm
from models import detector as detm
from models import photon_model as pm
from models import vlc_channel as vlcm
from qkd import information_metrics as im
from qkd import qber as qberm
from qkd.bb84 import run_analytic
from utils import constants as C
from utils.parameters import SimulationParameters, default_parameters


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str
    category: str = ""
    duration_ms: float = 0.0


def _rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


# ==========================================================================
def test_photon_energy() -> TestResult:
    lam = 520.0
    e = pm.photon_energy(lam)
    expect = C.PLANCK_H * C.SPEED_OF_LIGHT / (lam * 1e-9)
    ok = _rel(e, expect) < 1e-12
    ev = pm.photon_energy_ev(lam)
    return TestResult(
        "1. Photon energy E = hc/lambda", ok,
        f"E(520 nm) = {e:.6e} J = {ev:.4f} eV "
        f"(expected {expect:.6e} J); relative error {_rel(e, expect):.2e}",
        "Photon model")


def test_channel_gain() -> TestResult:
    # normal incidence, m = 1 (Lambertian, 60 deg semi-angle), no concentrator
    A, d = 1e-4, 2.0
    h0, geo, g = vlcm.lambertian_gain(d, 0.0, 0.0, A, 90.0, 1.0,
                                      filter_transmission=1.0,
                                      n_index=1.0, use_concentrator=False)
    expect = (1.0 + 1.0) * A / (2 * math.pi * d ** 2)
    ok1 = _rel(h0, expect) < 1e-12
    # inverse-square law
    h_far, _, _ = vlcm.lambertian_gain(2 * d, 0.0, 0.0, A, 90.0, 1.0, 1.0, 1.0, False)
    ok2 = _rel(h_far, h0 / 4.0) < 1e-12
    # outside the FOV the gain must be exactly zero
    h_out, _, _ = vlcm.lambertian_gain(d, 0.0, 50.0, A, 30.0, 1.0, 1.0, 1.5, True)
    ok3 = (h_out == 0.0)
    # Lambertian order from the half-power semi-angle: m(60 deg) = 1
    ok4 = _rel(vlcm.lambertian_order(60.0), 1.0) < 1e-9
    ok = ok1 and ok2 and ok3 and ok4
    return TestResult(
        "2. Lambertian VLC channel gain H(0)", ok,
        f"H(0) at d=2 m, normal incidence = {h0:.6e} (closed form {expect:.6e}); "
        f"1/d^2 scaling {'OK' if ok2 else 'FAILED'}; "
        f"zero outside FOV {'OK' if ok3 else 'FAILED'}; "
        f"m(60 deg) = {vlcm.lambertian_order(60.0):.6f} (expected 1)",
        "Channel")


def test_photon_number_generation() -> TestResult:
    rng = np.random.default_rng(2024)
    mu, n = 0.1, 1_000_000
    s = pm.sample_photon_numbers(rng, mu, n)
    mean, var = s.mean(), s.var()
    tol = 5.0 / math.sqrt(n) * math.sqrt(mu)      # 5 sigma
    ok1 = abs(mean - mu) < tol
    ok2 = abs(var - mu) < 5 * tol
    # analytic pmf
    ok3 = _rel(pm.poisson_pmf(0, mu), math.exp(-mu)) < 1e-12
    ok4 = _rel(pm.multiphoton_probability(mu),
               1 - math.exp(-mu) - mu * math.exp(-mu)) < 1e-12
    ok = ok1 and ok2 and ok3 and ok4
    return TestResult(
        "3. Poisson photon-number generation", ok,
        f"mu = {mu}: sampled mean {mean:.6f}, variance {var:.6f} over {n:,} draws "
        f"(both must equal mu); P(0) and P(n>=2) match the closed forms",
        "Photon model")


def test_background_photons() -> TestResult:
    p = default_parameters()
    p.background.led.enabled = False
    p.background.use_isotropic_preset = True
    p.background.isotropic_spectral_irradiance_w_m2_nm = 1e-3
    p.filt.shape = "Top-hat"

    rates = []
    for bw in (0.5, 1.0, 2.0, 4.0):
        p.filt.bandwidth_nm = bw
        r = bgm.evaluate_background(p)
        rates.append(r.total_count_rate_hz)
    # flat spectrum + top-hat filter  =>  exactly linear in bandwidth
    ratios = [rates[i + 1] / rates[i] for i in range(len(rates) - 1)]
    ok1 = all(abs(x - 2.0) < 0.02 for x in ratios)

    # bandwidth -> 0.  The out-of-band blocking floor (60 dB over 800 nm) would
    # otherwise dominate a sub-picometre pass-band, which is physically correct
    # but hides the limit, so the floor is removed for this check.
    p.filt.out_of_band_rejection_db = 300.0
    p.filt.bandwidth_nm = 1e-4
    r0 = bgm.evaluate_background(p)
    ok2 = r0.total_count_rate_hz < rates[0] * 1e-3
    p.filt.out_of_band_rejection_db = 60.0

    # zero illumination -> zero background
    p2 = default_parameters()
    p2.background.led.enabled = False
    p2.background.fluorescent.enabled = False
    p2.background.sunlight.enabled = False
    p2.background.user.enabled = False
    p2.background.use_isotropic_preset = False
    r_zero = bgm.evaluate_background(p2)
    ok3 = r_zero.total_count_rate_hz == 0.0

    ok = ok1 and ok2 and ok3
    return TestResult(
        "4. Background photon calculation", ok,
        f"flat spectrum, top-hat filter: R_bg doubles when the bandwidth doubles "
        f"(ratios {', '.join(f'{x:.4f}' for x in ratios)}); "
        f"R_bg -> 0 as the bandwidth -> 0 ({r0.total_count_rate_hz:.3e} vs "
        f"{rates[0]:.3e} counts/s); no sources -> exactly zero background",
        "Background")


def test_detector_click_probability() -> TestResult:
    eta, mu = 0.2, 0.5
    p = detm.signal_click_probability(eta, mu)
    expect = 1 - math.exp(-eta * mu)
    ok1 = _rel(p, expect) < 1e-12
    # sum over Poisson of 1-(1-eta)^n must reproduce the same number
    ns = np.arange(0, 200)
    brute = float(np.sum(pm.poisson_pmf(ns, mu) * (1 - (1 - eta) ** ns)))
    ok2 = _rel(brute, expect) < 1e-10
    # Y_0 and the gain
    y0 = detm.vacuum_yield(1e-3, 1e-4, 2)
    ok3 = _rel(y0, 1 - math.exp(-2 * 1.1e-3)) < 1e-12
    q = detm.overall_gain(eta, mu, y0)
    ok4 = _rel(q, 1 - (1 - y0) * math.exp(-eta * mu)) < 1e-12
    # zero efficiency -> only noise
    ok5 = _rel(detm.overall_gain(0.0, mu, y0), y0) < 1e-12
    ok = ok1 and ok2 and ok3 and ok4 and ok5
    return TestResult(
        "5. Detector click probability", ok,
        f"P_sig(click) = 1-exp(-eta*mu) = {p:.8f}; direct Poisson sum of "
        f"1-(1-eta)^n = {brute:.8f}; Q_mu = 1-(1-Y_0)exp(-eta*mu) verified; "
        f"eta = 0 reduces the gain to Y_0 exactly",
        "Detector")


def test_basis_selection() -> TestResult:
    rng = np.random.default_rng(7)
    n = 400_000
    basis = (rng.random(n) >= 0.5).astype(int)
    bit = (rng.random(n) < 0.5).astype(int)
    fz = basis.mean()
    fb = bit.mean()
    tol = 5.0 * 0.5 / math.sqrt(n)
    ok1 = abs(fz - 0.5) < tol and abs(fb - 0.5) < tol
    # matched-basis fraction must be 1/2 for two independent unbiased choices
    b2 = (rng.random(n) >= 0.5).astype(int)
    match = float((basis == b2).mean())
    ok2 = abs(match - 0.5) < tol
    from qkd.key_rate import sifting_factor
    ok3 = (abs(sifting_factor(0.5, False) - 0.5) < 1e-15
           and abs(sifting_factor(0.9, True) - (0.81 + 0.01)) < 1e-12)
    ok = ok1 and ok2 and ok3
    return TestResult(
        "6. BB84 basis and bit selection", ok,
        f"P(Z) = {fz:.5f}, P(bit=1) = {fb:.5f}, matched-basis fraction = "
        f"{match:.5f} (all must be 1/2 within 5 sigma); q = 1/2 for symmetric "
        f"BB84 and q = p_Z^2+(1-p_Z)^2 for the biased variant",
        "Protocol")


def test_qber_limits() -> TestResult:
    eta, mu, ed = 1e-3, 0.1, 0.01
    # background -> 0 : QBER must approach e_d
    b0 = qberm.compute_qber(eta, mu, 0.0, 0.0, ed)
    ok1 = _rel(b0.qber_total, ed) < 1e-9
    # background -> very large : QBER must approach 1/2
    bhi = qberm.compute_qber(eta, mu, 10.0, 0.0, ed)
    ok2 = abs(bhi.qber_total - 0.5) < 0.02
    # monotonic increase with background
    mus = np.logspace(-9, -1, 40)
    qs = [qberm.compute_qber(eta, mu, float(m), 0.0, ed).qber_total for m in mus]
    ok3 = all(qs[i + 1] >= qs[i] - 1e-12 for i in range(len(qs) - 1))
    # budget must sum to the total
    b = qberm.compute_qber(eta, mu, 1e-5, 1e-6, ed, 2, 1e-7)
    total = (b.q_polarization + b.q_pointing + b.q_background + b.q_dark
             + b.q_afterpulse)
    ok4 = _rel(total, b.qber_total) < 1e-9
    ok = ok1 and ok2 and ok3 and ok4
    return TestResult(
        "7. QBER model and limiting behaviour", ok,
        f"zero background -> QBER = {b0.qber_total:.6f} = e_d = {ed}; "
        f"saturating background -> QBER = {bhi.qber_total:.4f} -> 1/2; "
        f"QBER monotonically non-decreasing in mu_bg over 8 decades; "
        f"error budget sums to the total ({total:.8f} vs {b.qber_total:.8f})",
        "QBER")


def test_binary_entropy() -> TestResult:
    ok1 = im.binary_entropy(0.0) == 0.0 and im.binary_entropy(1.0) == 0.0
    ok2 = _rel(im.binary_entropy(0.5), 1.0) < 1e-15
    ok3 = _rel(im.binary_entropy(0.11), im.binary_entropy(0.89)) < 1e-12
    arr = im.binary_entropy(np.array([0.0, 0.25, 0.5, 1.0]))
    ok4 = (arr[0] == 0.0 and arr[3] == 0.0
           and abs(arr[2] - 1.0) < 1e-15
           and abs(arr[1] - 0.8112781244591328) < 1e-12)
    # inverse: h2 is quadratically flat at x = 1/2, so invert away from it
    ok5 = (abs(im.inverse_binary_entropy(float(im.binary_entropy(0.1))) - 0.1) < 1e-9
           and abs(im.inverse_binary_entropy(1.0) - 0.5) < 1e-6)
    ok = ok1 and ok2 and ok3 and ok4 and ok5
    return TestResult(
        "8. Binary entropy h2(x)", ok,
        f"h2(0) = h2(1) = 0 exactly (no NaN); h2(1/2) = {im.binary_entropy(0.5):.15f}; "
        f"h2(0.25) = {arr[1]:.12f}; symmetry h2(x) = h2(1-x) verified; "
        f"vectorised and scalar paths agree",
        "Information")


def test_mutual_information() -> TestResult:
    ok1 = _rel(im.mutual_information_per_bit(0.0), 1.0) < 1e-15
    ok2 = abs(im.mutual_information_per_bit(0.5)) < 1e-15
    ok3 = _rel(im.mutual_information_per_bit(0.11), 1 - im.binary_entropy(0.11)) < 1e-15
    # monotonic decrease in [0, 1/2]
    es = np.linspace(0, 0.5, 200)
    iv = im.mutual_information_per_bit(es)
    ok4 = bool(np.all(np.diff(iv) <= 1e-12))
    ok5 = _rel(im.mutual_information_rate(1e6, 0.0), 1e6) < 1e-12
    ok = ok1 and ok2 and ok3 and ok4 and ok5
    return TestResult(
        "9. Mutual information I(A:B) = 1 - h2(QBER)", ok,
        f"I(A:B) = 1 bit at QBER = 0 and exactly 0 at QBER = 1/2; "
        f"I(A:B) at the 11% threshold = {im.mutual_information_per_bit(0.11):.6f} bit; "
        f"monotonically decreasing over [0, 1/2]; rate form consistent",
        "Information")


def test_key_rate() -> TestResult:
    from qkd.key_rate import asymptotic_key_rate
    # noiseless, lossless, ideal single photon: r -> 1 bit per sifted bit
    k = asymptotic_key_rate(0.5, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0)
    ok1 = _rel(k.secret_fraction_per_sifted_bit, 1.0) < 1e-12
    # at QBER = 11% with f_EC = 1 the rate must be ~ 0 (the BB84 threshold)
    k2 = asymptotic_key_rate(0.5, 1.0, 1.0, 0.11, 1.0, 0.11, 1.0)
    ok2 = abs(k2.secret_fraction_per_sifted_bit) < 0.02
    # above the threshold the rate must be negative -> flagged insecure
    k3 = asymptotic_key_rate(0.5, 1.0, 1.0, 0.20, 1.0, 0.20, 1.1)
    ok3 = (not k3.secure) and k3.secret_key_per_pulse < 0
    # monotonic decrease with QBER
    rs = [asymptotic_key_rate(0.5, 1.0, 1e-3, e, 1e-3, e, 1.1).secret_key_per_pulse
          for e in np.linspace(0.0, 0.15, 40)]
    ok4 = all(rs[i + 1] <= rs[i] + 1e-15 for i in range(len(rs) - 1))
    ok = ok1 and ok2 and ok3 and ok4
    return TestResult(
        "10. Secret-key-rate calculation", ok,
        f"ideal limit r = {k.secret_fraction_per_sifted_bit:.6f} bit/sifted bit; "
        f"r = {k2.secret_fraction_per_sifted_bit:+.4f} at the 11% BB84 threshold "
        f"(must be ~0); r < 0 and flagged NOT secure at QBER = 20%; "
        f"monotonically decreasing in QBER",
        "Key rate")


def test_end_to_end_limits() -> TestResult:
    """Chain-level limiting behaviour on the default demonstration scenario."""
    base = default_parameters()
    msgs, oks = [], []

    # (a) received photons must fall with distance
    rec = []
    for d in (1.0, 2.0, 4.0, 8.0):
        p = base.copy()
        p.room.override_distance = True
        p.room.link_distance_m = d
        rec.append(run_analytic(p).received_photons_per_pulse)
    ok_a = all(rec[i + 1] < rec[i] for i in range(len(rec) - 1))
    oks.append(ok_a)
    msgs.append(f"received photons/pulse at d = 1,2,4,8 m: "
                f"{', '.join(f'{x:.3e}' for x in rec)} (strictly decreasing)")

    # (b) QBER must rise and key rate must fall with background count rate
    qs, ks = [], []
    for rbg in (0.0, 1e3, 1e5, 1e7, 1e9):
        p = base.copy()
        p.background.led.enabled = False
        p.background.user.enabled = True
        p.background.user.mode = "Background count rate [counts/s]"
        p.background.user.value = rbg
        r = run_analytic(p)
        qs.append(r.qber.qber_total)
        ks.append(r.secret_key_rate_bps)
    ok_b = all(qs[i + 1] >= qs[i] - 1e-12 for i in range(len(qs) - 1))
    ok_c = all(ks[i + 1] <= ks[i] + 1e-9 for i in range(len(ks) - 1))
    ok_d = ks[-1] <= 0.0
    oks += [ok_b, ok_c, ok_d]
    msgs.append(f"QBER at R_bg = 0, 1e3, 1e5, 1e7, 1e9 counts/s: "
                f"{', '.join(f'{100*x:.3f}%' for x in qs)} (non-decreasing)")
    msgs.append(f"secret key rate: {', '.join(f'{x:.3e}' for x in ks)} bit/s "
                f"(non-increasing, and <= 0 at the highest noise)")

    # (e) zero background -> QBER must equal the intrinsic polarisation floor
    p = base.copy()
    p.background.led.enabled = False
    p.background.user.enabled = False
    p.detector.dark_count_rate_hz = 0.0
    p.detector.afterpulse_probability = 0.0
    r = run_analytic(p)
    ok_e = _rel(r.qber.qber_total, r.polarization.e_intrinsic) < 1e-6
    oks.append(ok_e)
    msgs.append(f"with all noise removed, QBER = {100*r.qber.qber_total:.6f}% "
                f"= intrinsic polarisation error e_d = "
                f"{100*r.polarization.e_intrinsic:.6f}%")

    return TestResult("11. End-to-end limiting behaviour", all(oks),
                      "; ".join(msgs), "Integration")


def test_monte_carlo_vs_analytic() -> TestResult:
    """The event-by-event simulation must reproduce the analytic QBER."""
    from simulation.monte_carlo import run_monte_carlo
    p = default_parameters()
    p.source.mu = 0.5
    p.channel.tx_half_power_semiangle_deg = 25.0
    p.background.led.enabled = False
    p.background.user.enabled = True
    p.background.user.mode = "Background count rate [counts/s]"
    p.background.user.value = 5e6
    p.monte_carlo.n_pulses = 300_000
    p.monte_carlo.seed = 4242

    a = run_analytic(p)
    m = run_monte_carlo(p)

    dq = abs(m.qber - a.qber.qber_total)
    tol_q = max(5.0 * m.qber_std_err, 2e-3)
    ok1 = dq < tol_q

    dg = abs(m.gain_q_mu - a.qber.gain_q_mu)
    se_g = math.sqrt(max(a.qber.gain_q_mu * (1 - a.qber.gain_q_mu), 0) / m.n_pulses)
    ok2 = dg < max(5 * se_g, 1e-5)

    ok = ok1 and ok2
    return TestResult(
        "12. Monte Carlo vs analytic cross-check", ok,
        f"QBER: analytic {100*a.qber.qber_total:.4f}% vs Monte Carlo "
        f"{100*m.qber:.4f}% +/- {100*m.qber_std_err:.4f}% "
        f"({m.n_sifted:,} sifted bits from {m.n_pulses:,} pulses); "
        f"gain Q_mu: analytic {a.qber.gain_q_mu:.6e} vs MC {m.gain_q_mu:.6e}",
        "Integration")


# ==========================================================================
ALL_TESTS: List[Callable[[], TestResult]] = [
    test_photon_energy,
    test_channel_gain,
    test_photon_number_generation,
    test_background_photons,
    test_detector_click_probability,
    test_basis_selection,
    test_qber_limits,
    test_binary_entropy,
    test_mutual_information,
    test_key_rate,
    test_end_to_end_limits,
    test_monte_carlo_vs_analytic,
]


def run_all(include_slow: bool = True) -> List[TestResult]:
    out = []
    for fn in ALL_TESTS:
        if not include_slow and fn in (test_monte_carlo_vs_analytic,
                                       test_end_to_end_limits):
            continue
        t0 = time.perf_counter()
        try:
            r = fn()
        except Exception as exc:                      # a crash is a failure
            r = TestResult(fn.__name__, False, f"EXCEPTION: {exc!r}", "Error")
        r.duration_ms = (time.perf_counter() - t0) * 1e3
        out.append(r)
    return out


if __name__ == "__main__":
    results = run_all()
    width = 78
    print("=" * width)
    print("Indoor VL-QKD simulator - start-up self-test")
    print("=" * width)
    n_pass = 0
    for r in results:
        tag = "PASS" if r.passed else "FAIL"
        n_pass += int(r.passed)
        print(f"[{tag}] {r.name}   ({r.duration_ms:.0f} ms)")
        for line in r.detail.split("; "):
            print(f"       - {line}")
    print("-" * width)
    print(f"{n_pass}/{len(results)} tests passed")
    print("=" * width)
    raise SystemExit(0 if n_pass == len(results) else 1)
