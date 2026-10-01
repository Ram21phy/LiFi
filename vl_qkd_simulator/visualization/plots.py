"""
visualization/plots.py
======================
All one-dimensional figures.

Every function takes already-computed data (a DataFrame from
simulation.parameter_sweep, or a BB84Result) and returns a plotly Figure.
No physics is computed here - that separation keeps the plots honest.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.theme import (AMBER, BLUE, CATEGORICAL, GREEN, INK_DIM, LIME,
                                 PINK, ROSE, SEMANTIC, SLATE, TEAL, VIOLET,
                                 add_secure_shading, add_threshold, style)

PROV_ANALYTIC = "analytic model - asymptotic BB84"


# ==========================================================================
#  Generic sweep plot
# ==========================================================================
def sweep_plot(df: pd.DataFrame, metric: str, *, title: str = None,
               log_x: bool = False, log_y: bool = False,
               color: str = TEAL, threshold: float = None,
               threshold_label: str = None, height: int = 400,
               shade_secure: bool = False,
               provenance: str = PROV_ANALYTIC) -> go.Figure:
    xcol = df.columns[0]
    y = df[metric].to_numpy(dtype=float)
    if log_y:
        y = np.where(y > 0, y, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[xcol], y=y, mode="lines", name=metric,
        line=dict(color=color, width=2.4),
        hovertemplate=f"{xcol}: %{{x:.4g}}<br>{metric}: %{{y:.4g}}<extra></extra>"))
    style(fig, title or metric, xcol, metric, height, log_x, log_y, provenance)
    if threshold is not None:
        add_threshold(fig, threshold, threshold_label or "")
    if shade_secure and "_secure" in df.columns:
        add_secure_shading(fig, df[xcol].to_numpy(), df["_secure"].to_numpy())
    fig.update_layout(showlegend=False)
    return fig


def multi_sweep_plot(df: pd.DataFrame, metrics: Sequence[str], *,
                     title: str = None, ytitle: str = "",
                     log_x: bool = False, log_y: bool = False,
                     colors: Optional[Sequence[str]] = None,
                     height: int = 420,
                     provenance: str = PROV_ANALYTIC) -> go.Figure:
    xcol = df.columns[0]
    colors = colors or CATEGORICAL
    fig = go.Figure()
    for i, m in enumerate(metrics):
        if m not in df.columns:
            continue
        y = df[m].to_numpy(dtype=float)
        if log_y:
            y = np.where(y > 0, y, np.nan)
        fig.add_trace(go.Scatter(x=df[xcol], y=y, mode="lines", name=m,
                                 line=dict(color=colors[i % len(colors)], width=2.2)))
    style(fig, title, xcol, ytitle, height, log_x, log_y, provenance)
    return fig


# ==========================================================================
#  Comparison across background scenarios (requirement 18)
# ==========================================================================
def scenario_comparison(dfs: Dict[str, pd.DataFrame], metric: str, *,
                        title: str = None, log_x: bool = False,
                        log_y: bool = False, height: int = 460,
                        threshold: float = None, threshold_label: str = None
                        ) -> go.Figure:
    fig = go.Figure()
    palette = [GREEN, TEAL, BLUE, AMBER, ROSE, VIOLET]
    for i, (name, df) in enumerate(dfs.items()):
        xcol = df.columns[0]
        y = df[metric].to_numpy(dtype=float)
        if log_y:
            y = np.where(y > 0, y, np.nan)
        fig.add_trace(go.Scatter(
            x=df[xcol], y=y, mode="lines", name=name,
            line=dict(color=palette[i % len(palette)], width=2.4)))
    xcol = list(dfs.values())[0].columns[0]
    style(fig, title or metric, xcol, metric, height, log_x, log_y,
          "analytic model - background scenarios compared at identical link parameters")
    if threshold is not None:
        add_threshold(fig, threshold, threshold_label or "")
    return fig


# ==========================================================================
#  QBER error budget
# ==========================================================================
def qber_budget_bar(budget, height: int = 300) -> go.Figure:
    rows = [
        ("Polarisation / misalignment", budget.q_polarization, SEMANTIC["polarization"]),
        ("Pointing", budget.q_pointing, SEMANTIC["pointing"]),
        ("Background photons", budget.q_background, SEMANTIC["background"]),
        ("Dark counts", budget.q_dark, SEMANTIC["dark"]),
        ("Afterpulsing", budget.q_afterpulse, SEMANTIC["afterpulse"]),
    ]
    rows = [r for r in rows if r[1] > 0 or r[0].startswith(("Polar", "Back", "Dark"))]
    fig = go.Figure()
    for name, v, c in rows:
        fig.add_trace(go.Bar(
            x=[100 * v], y=[name], orientation="h", name=name,
            marker=dict(color=c),
            text=[f"{100*v:.3f}%"], textposition="outside",
            textfont=dict(color=INK_DIM, size=11),
            hovertemplate=f"{name}: %{{x:.4f}}%<extra></extra>"))
    style(fig, None, "Contribution to QBER [%]", None, height)
    fig.update_layout(showlegend=False, barmode="stack",
                      margin=dict(l=170, r=70, t=12, b=44))
    fig.update_xaxes(range=[0, max(100 * budget.qber_total * 1.35, 0.01)])
    return fig


def qber_budget_donut(budget, height: int = 300) -> go.Figure:
    labels, values, colors = [], [], []
    for name, v, c in (
            ("Polarisation", budget.q_polarization, SEMANTIC["polarization"]),
            ("Pointing", budget.q_pointing, SEMANTIC["pointing"]),
            ("Background", budget.q_background, SEMANTIC["background"]),
            ("Dark counts", budget.q_dark, SEMANTIC["dark"]),
            ("Afterpulsing", budget.q_afterpulse, SEMANTIC["afterpulse"])):
        if v > 0:
            labels.append(name)
            values.append(v)
            colors.append(c)
    if not values:
        values, labels, colors = [1.0], ["no errors"], [SLATE]
    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.62, sort=False,
        marker=dict(colors=colors, line=dict(color="#0E1420", width=2)),
        textinfo="percent", textfont=dict(size=11),
        hovertemplate="%{label}: %{value:.5f} absolute QBER<extra></extra>"))
    fig.add_annotation(text=f"<b>{100*budget.qber_total:.2f}%</b><br>"
                            f"<span style='font-size:10px'>total QBER</span>",
                       showarrow=False, font=dict(size=20))
    style(fig, None, None, None, height)
    fig.update_layout(showlegend=True, margin=dict(l=8, r=8, t=8, b=8))
    return fig


# ==========================================================================
#  Spectral view: background spectrum vs filter
# ==========================================================================
def spectrum_plot(result, height: int = 440) -> go.Figure:
    """Background spectral irradiance, the filter response and the signal line."""
    from models.background_noise import SpectralFilter

    lam = result.background.spectrum_lambda_nm
    irr = result.background.spectrum_irradiance
    p = result.params
    sf = SpectralFilter.from_params(p.filt)
    t = sf.transmission(lam)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if irr is not None and np.nanmax(irr) > 0:
        fig.add_trace(go.Scatter(
            x=lam, y=irr, name="Background spectral irradiance",
            mode="lines", line=dict(color=AMBER, width=2),
            fill="tozeroy", fillcolor="rgba(245,165,36,0.16)",
            hovertemplate="%{x:.2f} nm<br>%{y:.3e} W m⁻² nm⁻¹<extra></extra>"),
            secondary_y=False)
    fig.add_trace(go.Scatter(
        x=lam, y=t, name="Optical filter transmission", mode="lines",
        line=dict(color=TEAL, width=2, dash="solid"),
        hovertemplate="%{x:.3f} nm<br>T = %{y:.3f}<extra></extra>"),
        secondary_y=True)

    fig.add_vline(x=p.source.wavelength_nm,
                  line=dict(color=BLUE, width=1.4, dash="dot"),
                  annotation_text=f"signal {p.source.wavelength_nm:.1f} nm",
                  annotation_position="bottom right",
                  annotation_font=dict(size=11, color=BLUE))

    style(fig, None, "Wavelength [nm]", None, height,
          provenance="simplified background spectral model")
    fig.update_yaxes(title_text="Spectral irradiance [W m⁻² nm⁻¹]",
                     secondary_y=False, type="log", automargin=True)
    # The Gaussian tails of the analytic spectra reach ~1e-31, which would
    # squash the interesting three decades into a flat line; clamp to six
    # decades below the peak instead.
    if irr is not None and np.isfinite(np.nanmax(irr)) and np.nanmax(irr) > 0:
        top = float(np.log10(np.nanmax(irr)))
        fig.update_yaxes(range=[top - 6.0, top + 0.5], secondary_y=False)
    fig.update_yaxes(title_text="Filter transmission", secondary_y=True,
                     range=[0, 1.05], showgrid=False, automargin=True)
    fig.update_xaxes(range=[max(380, p.filt.center_nm - 180),
                            min(1000, p.filt.center_nm + 220)])
    fig.update_layout(margin=dict(l=78, r=68, t=56, b=62))
    return fig


def background_breakdown_bar(result, height: int = 280) -> go.Figure:
    contribs = result.background.contributions
    if not contribs:
        fig = go.Figure()
        fig.add_annotation(text="No background sources enabled",
                           showarrow=False, font=dict(color=INK_DIM, size=14))
        return style(fig, None, None, None, height)
    names = [c.name for c in contribs]
    vals = [max(c.count_rate_hz, 1e-30) for c in contribs]
    colors = [AMBER, VIOLET, LIME, BLUE, PINK][:len(names)]
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colors),
        text=[f"{v:.3e}" for v in vals], textposition="outside",
        textfont=dict(color=INK_DIM, size=11),
        hovertemplate="%{y}<br>%{x:.4e} counts/s<extra></extra>"))
    style(fig, None, "Background count rate [counts/s]", None, height, log_x=True)
    fig.update_layout(showlegend=False, margin=dict(l=190, r=90, t=12, b=46))
    return fig


# ==========================================================================
#  Monte Carlo
# ==========================================================================
def monte_carlo_convergence(mc, analytic_qber: float = None,
                            height: int = 400) -> go.Figure:
    n = mc.convergence.get("n_pulses", np.array([]))
    q = mc.convergence.get("qber", np.array([]))
    fig = go.Figure()
    if len(n):
        # +/- 1 sigma binomial envelope around the running estimate
        se = np.sqrt(np.clip(q * (1 - q), 0, None) /
                     np.maximum(n * max(mc.sifting_ratio, 1e-12), 1))
        fig.add_trace(go.Scatter(
            x=np.concatenate([n, n[::-1]]),
            y=np.concatenate([100 * (q + se), 100 * (q - se)[::-1]]),
            fill="toself", fillcolor="rgba(45,212,191,0.14)",
            line=dict(width=0), hoverinfo="skip",
            name="± 1σ statistical envelope"))
        fig.add_trace(go.Scatter(
            x=n, y=100 * q, mode="lines+markers", name="Monte Carlo QBER",
            line=dict(color=TEAL, width=2.2),
            marker=dict(size=5, color=TEAL),
            hovertemplate="%{x:,.0f} pulses<br>QBER %{y:.4f}%<extra></extra>"))
    if analytic_qber is not None:
        fig.add_hline(y=100 * analytic_qber,
                      line=dict(color=AMBER, width=1.6, dash="dash"),
                      annotation_text=f"analytic {100*analytic_qber:.3f}%",
                      annotation_position="bottom right",
                      annotation_font=dict(size=11, color=AMBER))
    style(fig, None, "Number of transmitted pulses", "QBER [%]", height,
          log_x=True, provenance="Monte Carlo (event-by-event) vs analytic model")
    return fig


def monte_carlo_counts(mc, height: int = 320) -> go.Figure:
    names = ["Signal clicks", "Background clicks", "Dark/afterpulse clicks",
             "Double clicks", "Total detections", "Sifted bits", "Errors"]
    vals = [mc.n_signal_clicks, mc.n_background_clicks, mc.n_dark_clicks,
            mc.n_double_clicks, mc.n_detections, mc.n_sifted, mc.n_errors]
    colors = [TEAL, AMBER, VIOLET, PINK, BLUE, LIME, ROSE]
    fig = go.Figure(go.Bar(
        x=names, y=[max(v, 0.5) for v in vals], marker=dict(color=colors),
        text=[f"{v:,}" for v in vals], textposition="outside",
        textfont=dict(color=INK_DIM, size=11),
        hovertemplate="%{x}: %{text}<extra></extra>"))
    style(fig, None, None, "Event count", height, log_y=True,
          provenance=f"Monte Carlo, {mc.n_pulses:,} pulses, seed {mc.seed}")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickangle=-20)
    return fig


# ==========================================================================
#  Information-flow / Sankey-style summary
# ==========================================================================
def information_waterfall(result, height: int = 380) -> go.Figure:
    """
    Per-sifted-bit budget: 1 bit -> minus EC leakage -> minus PA -> secret.
    """
    ec = result.info.error_correction_leakage_per_bit
    pa = result.info.eve_information_gllp
    secret = result.secret_key_fraction

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Raw sifted bit", "Error correction<br>leakage",
           "Privacy amplification<br>(Eve's information)", "Secret key"],
        y=[1.0, -ec, -pa, None],
        text=[f"1.000", f"−{ec:.3f}", f"−{pa:.3f}", f"{secret:.3f}"],
        textposition="outside",
        connector=dict(line=dict(color="#3A4759", width=1)),
        increasing=dict(marker=dict(color=TEAL)),
        decreasing=dict(marker=dict(color=ROSE)),
        totals=dict(marker=dict(color=GREEN if secret > 0 else ROSE)),
    ))
    style(fig, None, None, "bits per sifted bit", height,
          provenance="analytic, asymptotic GLLP/decoy accounting")
    fig.update_layout(showlegend=False)
    return fig


def mutual_information_vs_qber(f_ec: float = 1.1, height: int = 400) -> go.Figure:
    """I(A:B), Eve's PA cost and the secret fraction against QBER."""
    from qkd.information_metrics import binary_entropy
    e = np.linspace(1e-6, 0.25, 500)
    iab = 1 - binary_entropy(e)
    ie = binary_entropy(e)
    secret = 1 - binary_entropy(e) - f_ec * binary_entropy(e)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=100 * e, y=iab, name="I(A:B) = 1 − h₂(E)",
                             line=dict(color=TEAL, width=2.4)))
    fig.add_trace(go.Scatter(x=100 * e, y=ie, name="Eve (single-photon PA) = h₂(E)",
                             line=dict(color=ROSE, width=2.2, dash="dash")))
    fig.add_trace(go.Scatter(x=100 * e, y=secret,
                             name=f"Secret fraction (ideal source, f_EC = {f_ec:g})",
                             line=dict(color=GREEN, width=2.4)))
    fig.add_hline(y=0, line=dict(color="#3A4759", width=1))
    style(fig, None, "QBER [%]", "bits per sifted bit", height,
          provenance="analytic, asymptotic")
    fig.update_yaxes(range=[-0.35, 1.05])
    return fig
