"""
tests/test_models.py
====================
pytest wrapper around the start-up self-test suite, plus a few extra
regression checks that are useful in CI but too slow or too dull for the
launch-time suite.

Run with:   python -m pytest tests -q          (or simply: python -m utils.selftest)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import background_noise as bgm          # noqa: E402
from models import vlc_channel as vlcm              # noqa: E402
from qkd import decoy_state as decoy                # noqa: E402
from qkd import qber as qberm                       # noqa: E402
from qkd.bb84 import run_analytic                   # noqa: E402
from qkd.information_metrics import binary_entropy  # noqa: E402
from simulation.finite_size import finite_key_length  # noqa: E402
from utils.parameters import SimulationParameters, default_parameters  # noqa: E402
from utils.selftest import ALL_TESTS                # noqa: E402


@pytest.mark.parametrize("fn", ALL_TESTS, ids=lambda f: f.__name__)
def test_selftest_case(fn):
    r = fn()
    assert r.passed, f"{r.name} failed: {r.detail}"


# --------------------------------------------------------------------------
def test_parameters_roundtrip():
    p = default_parameters()
    p.source.mu = 0.321
    p.background.led.n_luminaires = 7
    p.filt.bandwidth_nm = 0.037
    q = SimulationParameters.from_json(p.to_json())
    assert q.source.mu == pytest.approx(0.321)
    assert q.background.led.n_luminaires == 7
    assert q.filt.bandwidth_nm == pytest.approx(0.037)


def test_copy_is_deep():
    p = default_parameters()
    q = p.copy()
    q.room.length_x = 99.0
    q.background.led.n_luminaires = 31
    assert p.room.length_x != 99.0
    assert p.background.led.n_luminaires != 31


def test_channel_zero_outside_fov():
    p = default_parameters()
    p.channel.channel_model = "Lambertian (VLC LOS)"
    p.room.rx_aim_at_tx = False
    p.room.rx_normal = (0.0, 0.0, 1.0)
    p.room.tx_z = 2.8
    p.room.rx_z = 2.8            # link is horizontal, receiver faces straight up
    p.channel.rx_fov_deg = 10.0
    r = run_analytic(p)
    assert r.channel.channel_gain_h0 == 0.0
    assert r.eta_system == 0.0


def test_gllp_bound_is_pessimistic_vs_decoy():
    """GLLP without decoy can never certify more single photons than decoy."""
    p = default_parameters()
    p.source.mu = 0.5
    p.protocol.protocol = "Standard BB84 (GLLP)"
    a = run_analytic(p)
    p.protocol.protocol = "Infinite-decoy limit"
    b = run_analytic(p)
    assert a.key.q1 <= b.key.q1 + 1e-12
    assert a.secret_key_rate_bps <= b.secret_key_rate_bps + 1e-6


def test_decoy_bounds_are_valid():
    """Decoy lower bounds must not exceed the exact model values."""
    p = default_parameters()
    p.protocol.protocol = "Decoy-state BB84"
    p.source.mu, p.source.decoy_nu1, p.source.decoy_nu2 = 0.5, 0.1, 0.0
    r = run_analytic(p)
    y1, e1 = qberm.single_photon_yield_and_error(
        r.eta_system, r.detector.mu_bg, r.detector.mu_dark,
        r.qber.e_intrinsic, r.detector.n_detectors_per_basis,
        r.detector.mu_afterpulse)
    assert r.decoy is not None
    assert r.decoy.y1_lower <= y1 * (1 + 1e-6)
    assert r.decoy.e1_upper >= e1 * (1 - 1e-6)


def test_finite_key_never_beats_asymptotic():
    p = default_parameters()
    r = run_analytic(p)
    for n in (1e6, 1e8, 1e10, 1e12):
        f = finite_key_length(n, r.qber.gain_q_mu, r.qber.qber_total,
                              r.key.q1, r.key.e1,
                              p.protocol.error_correction_efficiency,
                              r.key.q_sift, 1e-10, 1e-15,
                              p.source.pulse_rate_hz, r.secret_key_rate_bps,
                              r.detector.dead_time_saturation_factor)
        assert f.secret_key_rate_bps <= r.secret_key_rate_bps + 1e-6


def test_finite_key_converges_to_asymptotic():
    p = default_parameters()
    r = run_analytic(p)
    f = finite_key_length(1e16, r.qber.gain_q_mu, r.qber.qber_total,
                          r.key.q1, r.key.e1,
                          p.protocol.error_correction_efficiency,
                          r.key.q_sift, 1e-10, 1e-15,
                          p.source.pulse_rate_hz, r.secret_key_rate_bps,
                          r.detector.dead_time_saturation_factor)
    assert f.penalty_factor > 0.9


def test_background_scales_with_illuminance():
    p = default_parameters()
    p.background.led.drive_by_illuminance = True
    rates = []
    for lux in (10.0, 100.0, 1000.0):
        p.background.led.target_illuminance_lux = lux
        rates.append(bgm.evaluate_background(p).total_count_rate_hz)
    assert rates[1] / rates[0] == pytest.approx(10.0, rel=1e-6)
    assert rates[2] / rates[1] == pytest.approx(10.0, rel=1e-6)


def test_fov_reduces_background_without_concentrator():
    p = default_parameters()
    p.channel.use_concentrator = False
    p.background.led.enabled = False
    p.background.use_isotropic_preset = True
    p.background.isotropic_spectral_irradiance_w_m2_nm = 1e-4
    p.channel.rx_fov_deg = 60.0
    wide = bgm.evaluate_background(p).total_count_rate_hz
    p.channel.rx_fov_deg = 6.0
    narrow = bgm.evaluate_background(p).total_count_rate_hz
    # bare detector: G_coll = pi sin^2(FOV)
    expect = (math.sin(math.radians(6.0)) / math.sin(math.radians(60.0))) ** 2
    assert narrow / wide == pytest.approx(expect, rel=1e-6)


def test_concentrator_makes_diffuse_background_fov_independent():
    """Etendue conservation: an ideal concentrator cancels the FOV dependence."""
    p = default_parameters()
    p.channel.use_concentrator = True
    p.background.led.enabled = False
    p.background.use_isotropic_preset = True
    p.background.isotropic_spectral_irradiance_w_m2_nm = 1e-4
    p.channel.rx_fov_deg = 60.0
    wide = bgm.evaluate_background(p).total_count_rate_hz
    p.channel.rx_fov_deg = 6.0
    narrow = bgm.evaluate_background(p).total_count_rate_hz
    assert narrow == pytest.approx(wide, rel=1e-9)


def test_monte_carlo_reproducible():
    from simulation.monte_carlo import run_monte_carlo
    p = default_parameters()
    p.monte_carlo.n_pulses = 50_000
    p.monte_carlo.seed = 777
    a = run_monte_carlo(p)
    b = run_monte_carlo(p)
    assert a.n_sifted == b.n_sifted
    assert a.n_errors == b.n_errors
    assert a.qber == b.qber


def test_sweep_runs_for_every_variable():
    from simulation.parameter_sweep import SWEEP_VARIABLES, sweep_1d
    p = default_parameters()
    for name, var in SWEEP_VARIABLES.items():
        df = sweep_1d(p, name, var.default_min, var.default_max, 4,
                      var.default_log,
                      metrics=["Secret key rate [bit/s]", "QBER"])
        assert len(df) == 4, name
        assert df["QBER"].notna().all(), name


def test_binary_entropy_vector_and_scalar_agree():
    xs = np.array([0.0, 1e-9, 0.01, 0.11, 0.5, 0.99, 1.0])
    vec = binary_entropy(xs)
    sca = np.array([binary_entropy(float(x)) for x in xs])
    assert np.allclose(vec, sca, atol=1e-15)
