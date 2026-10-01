"""
simulation/parameter_sweep.py
=============================
Generic one- and two-dimensional parameter sweeps over the analytic model.

A sweep is defined by a `SweepVariable`, which knows how to write a value into
a copy of the parameter set.  Everything else - which metrics to extract, how
to build the grid - is generic, so adding a new sweepable quantity is a
one-line change to SWEEP_VARIABLES.

All sweeps evaluate the ANALYTIC model (qkd.bb84.run_analytic).  The Monte
Carlo is far too slow to sweep densely and is used for verification instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qkd.bb84 import run_analytic


# ==========================================================================
@dataclass
class SweepVariable:
    key: str
    label: str
    unit: str
    setter: Callable[[object, float], None]
    default_min: float
    default_max: float
    default_log: bool = False
    integer: bool = False
    note: str = ""


def _set(path: str):
    """Build a setter for a dotted attribute path, e.g. 'source.mu'."""
    parts = path.split(".")

    def setter(p, v):
        obj = p
        for a in parts[:-1]:
            obj = getattr(obj, a)
        setattr(obj, parts[-1], v)
    return setter


def _set_distance(p, v):
    p.room.override_distance = True
    p.room.link_distance_m = float(v)


def _set_background_rate(p, v):
    p.background.led.enabled = False
    p.background.fluorescent.enabled = False
    p.background.sunlight.enabled = False
    p.background.use_isotropic_preset = False
    p.background.user.enabled = True
    p.background.user.mode = "Background count rate [counts/s]"
    p.background.user.value = float(v)


def _set_wavelength(p, v):
    """Wavelength sweep: keep the band-pass filter centred on the signal."""
    p.source.wavelength_nm = float(v)
    p.filt.center_nm = float(v)


def _set_wavelength_fixed_filter(p, v):
    p.source.wavelength_nm = float(v)


SWEEP_VARIABLES: Dict[str, SweepVariable] = {
    "Distance": SweepVariable(
        "Distance", "Link distance", "m", _set_distance, 0.5, 15.0, False,
        note="Tx/Rx direction is preserved; only the separation is changed."),
    "Wavelength (filter follows)": SweepVariable(
        "Wavelength (filter follows)", "Wavelength", "nm", _set_wavelength,
        400.0, 700.0, False,
        note="The band-pass filter is re-centred on the signal at every point."),
    "Wavelength (filter fixed)": SweepVariable(
        "Wavelength (filter fixed)", "Wavelength", "nm",
        _set_wavelength_fixed_filter, 400.0, 700.0, False,
        note="The filter stays where the user put it; the signal walks off it."),
    "Mean photon number mu": SweepVariable(
        "Mean photon number mu", "Mean photon number", "photons/pulse",
        _set("source.mu"), 1e-3, 1.0, True),
    "Detector efficiency": SweepVariable(
        "Detector efficiency", "Detector efficiency", "-",
        _set("detector.efficiency"), 0.01, 0.99, False),
    "Background count rate": SweepVariable(
        "Background count rate", "Background count rate", "counts/s",
        _set_background_rate, 1e0, 1e9, True,
        note="All modelled illumination sources are replaced by this value."),
    "Dark count rate": SweepVariable(
        "Dark count rate", "Dark count rate", "counts/s",
        _set("detector.dark_count_rate_hz"), 1e0, 1e6, True),
    "Filter bandwidth": SweepVariable(
        "Filter bandwidth", "Optical filter bandwidth", "nm",
        _set("filt.bandwidth_nm"), 0.05, 100.0, True),
    "Filter transmission": SweepVariable(
        "Filter transmission", "Filter peak transmission", "-",
        _set("filt.peak_transmission"), 0.05, 1.0, False),
    "Receiver aperture": SweepVariable(
        "Receiver aperture", "Receiver aperture area", "cm^2",
        _set("channel.rx_aperture_area_cm2"), 0.01, 25.0, True),
    "Receiver FOV": SweepVariable(
        "Receiver FOV", "Receiver FOV half-angle", "deg",
        _set("channel.rx_fov_deg"), 1.0, 85.0, False,
        note="Affects both signal collection (concentrator gain) and how many "
             "luminaires are inside the field of view."),
    "Optical efficiency": SweepVariable(
        "Optical efficiency", "Receiver optical efficiency", "-",
        _set("channel.optical_efficiency"), 0.05, 1.0, False),
    "Extra optical loss": SweepVariable(
        "Extra optical loss", "Additional optical loss", "dB",
        _set("channel.extra_loss_db"), 0.0, 30.0, False),
    "Polarisation misalignment": SweepVariable(
        "Polarisation misalignment", "Polarisation misalignment angle", "deg",
        _set("polarization.misalignment_angle_deg"), 0.0, 20.0, False),
    "Pointing jitter": SweepVariable(
        "Pointing jitter", "Pointing jitter (1 sigma)", "mrad",
        _set("channel.pointing_jitter_mrad"), 0.0, 10.0, False,
        note="Requires the pointing-error model to be enabled."),
    "Turbulence Cn^2": SweepVariable(
        "Turbulence Cn^2", "Refractive-index structure constant", "m^-2/3",
        _set("channel.turbulence_cn2"), 1e-17, 1e-12, True,
        note="Requires the turbulence model to be enabled."),
    "Pulse rate": SweepVariable(
        "Pulse rate", "Pulse repetition rate", "Hz",
        _set("source.pulse_rate_hz"), 1e6, 1e10, True),
    "Error-correction efficiency": SweepVariable(
        "Error-correction efficiency", "f_EC", "-",
        _set("protocol.error_correction_efficiency"), 1.0, 2.0, False),
    "LED luminaire power": SweepVariable(
        "LED luminaire power", "Optical power per LED luminaire", "W",
        _set("background.led.optical_power_per_luminaire_w"), 0.0, 30.0, False),
    "Detector gate width": SweepVariable(
        "Detector gate width", "Detection gate width", "ns",
        _set("detector.gate_width_ns"), 0.05, 100.0, True),
    "Transmitter semi-angle": SweepVariable(
        "Transmitter semi-angle", "Tx half-power semi-angle", "deg",
        _set("channel.tx_half_power_semiangle_deg"), 1.0, 70.0, False),
}


# ==========================================================================
METRICS = {
    "Secret key rate [bit/s]": lambda r: r.secret_key_rate_bps,
    "Sifted key rate [bit/s]": lambda r: r.sifted_key_rate_bps,
    "QBER": lambda r: r.qber.qber_total,
    "QBER [%]": lambda r: 100.0 * r.qber.qber_total,
    "Mutual information I(A:B) [bit/sifted bit]":
        lambda r: r.info.mutual_information_per_bit,
    "Mutual information rate [bit/s]": lambda r: r.info.mutual_information_rate_bps,
    "Eve information / PA term [bit/sifted bit]": lambda r: r.info.eve_information_gllp,
    "Secret key fraction [bit/sifted bit]": lambda r: r.secret_key_fraction,
    "Detection probability Q_mu": lambda r: r.qber.gain_q_mu,
    "Channel gain H(0)": lambda r: r.channel.channel_gain_h0,
    "System transmittance eta_sys": lambda r: r.eta_system,
    "Channel loss [dB]": lambda r: r.channel.eta_channel_db,
    "Received photons / pulse": lambda r: r.received_photons_per_pulse,
    "Background count rate [1/s]": lambda r: r.background_count_rate_hz,
    "Background photons / gate": lambda r: r.detector.mu_bg,
    "Dark counts / gate": lambda r: r.detector.mu_dark,
    "Vacuum yield Y_0": lambda r: r.detector.y0_vacuum_yield,
    "Signal-to-background ratio": lambda r: r.signal_to_background_ratio,
    "Signal-to-noise ratio": lambda r: r.signal_to_noise_ratio,
    "Single-photon gain Q_1": lambda r: r.key.q1,
    "Single-photon error e_1": lambda r: r.key.e1,
    "QBER: polarisation": lambda r: r.qber.q_polarization,
    "QBER: background": lambda r: r.qber.q_background,
    "QBER: dark counts": lambda r: r.qber.q_dark,
    "QBER: afterpulsing": lambda r: r.qber.q_afterpulse,
    "Effective filter bandwidth [nm]":
        lambda r: r.background.effective_filter_bandwidth_nm,
    "Filter transmission at signal":
        lambda r: r.background.filter_transmission_at_signal,
}


def make_grid(vmin: float, vmax: float, n: int, log: bool) -> np.ndarray:
    n = max(int(n), 2)
    if log:
        lo = max(vmin, 1e-30)
        hi = max(vmax, lo * (1 + 1e-9))
        return np.logspace(math.log10(lo), math.log10(hi), n)
    return np.linspace(vmin, vmax, n)


def sweep_1d(params, variable: str, vmin: float, vmax: float,
             n_points: int = 60, log: bool = False,
             metrics: Optional[List[str]] = None,
             progress_callback: Optional[Callable[[float], None]] = None
             ) -> pd.DataFrame:
    """Sweep one parameter, return a tidy DataFrame of all requested metrics."""
    var = SWEEP_VARIABLES[variable]
    grid = make_grid(vmin, vmax, n_points, log)
    metrics = metrics or list(METRICS.keys())

    rows = []
    for i, v in enumerate(grid):
        p = params.copy()
        var.setter(p, float(v))
        try:
            r = run_analytic(p)
            row = {var.label + (f" [{var.unit}]" if var.unit else ""): float(v)}
            for mname in metrics:
                row[mname] = float(METRICS[mname](r))
            row["_secure"] = bool(r.key.secure)
        except Exception as exc:                      # keep the sweep alive
            row = {var.label + (f" [{var.unit}]" if var.unit else ""): float(v)}
            for mname in metrics:
                row[mname] = float("nan")
            row["_secure"] = False
            row["_error"] = str(exc)
        rows.append(row)
        if progress_callback:
            progress_callback((i + 1) / len(grid))

    return pd.DataFrame(rows)


def sweep_2d(params, var_x: str, x_min: float, x_max: float, nx: int, log_x: bool,
             var_y: str, y_min: float, y_max: float, ny: int, log_y: bool,
             metric: str = "Secret key rate [bit/s]",
             progress_callback: Optional[Callable[[float], None]] = None):
    """
    Two-dimensional sweep, returning (x_grid, y_grid, Z) suitable for a heatmap.
    """
    vx = SWEEP_VARIABLES[var_x]
    vy = SWEEP_VARIABLES[var_y]
    xs = make_grid(x_min, x_max, nx, log_x)
    ys = make_grid(y_min, y_max, ny, log_y)
    Z = np.full((len(ys), len(xs)), np.nan)

    total = len(xs) * len(ys)
    k = 0
    for j, yv in enumerate(ys):
        for i, xv in enumerate(xs):
            p = params.copy()
            vx.setter(p, float(xv))
            vy.setter(p, float(yv))
            try:
                r = run_analytic(p)
                Z[j, i] = float(METRICS[metric](r))
            except Exception:
                Z[j, i] = np.nan
            k += 1
        if progress_callback:
            progress_callback(k / total)
    return xs, ys, Z


# ==========================================================================
def filter_optimization(params, bw_min: float = 0.05, bw_max: float = 100.0,
                        n_points: int = 80, qber_limit: float = 0.11,
                        progress_callback=None) -> Tuple[pd.DataFrame, dict]:
    """
    Sweep the optical filter bandwidth and locate the bandwidth that maximises
    the secret key rate subject to QBER <= qber_limit.

    NOTE: the result is a NUMERICAL OPTIMUM of this model, not an
    experimentally validated operating point.
    """
    df = sweep_1d(params, "Filter bandwidth", bw_min, bw_max, n_points, True,
                  metrics=["Secret key rate [bit/s]", "QBER",
                           "Background count rate [1/s]",
                           "Sifted key rate [bit/s]",
                           "Filter transmission at signal",
                           "Effective filter bandwidth [nm]",
                           "Mutual information I(A:B) [bit/sifted bit]"],
                  progress_callback=progress_callback)
    col = "Optical filter bandwidth [nm]"
    ok = df[(df["QBER"] <= qber_limit) & (df["Secret key rate [bit/s]"] > 0)]
    if len(ok) == 0:
        return df, {"found": False,
                    "message": f"No bandwidth in [{bw_min:g}, {bw_max:g}] nm gives a "
                               f"positive key rate with QBER <= {qber_limit:.1%}."}
    best = ok.loc[ok["Secret key rate [bit/s]"].idxmax()]
    return df, {
        "found": True,
        "bandwidth_nm": float(best[col]),
        "secret_key_rate_bps": float(best["Secret key rate [bit/s]"]),
        "qber": float(best["QBER"]),
        "background_rate_hz": float(best["Background count rate [1/s]"]),
        "message": ("Numerical optimum of the present model "
                    "(not an experimentally validated optimum)."),
    }
