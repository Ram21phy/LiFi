"""
visualization/heatmaps.py
=========================
Two-dimensional maps, including the headline research figure

        Secret key rate  R(distance, background count rate)

and its QBER and I(A:B) companions.

Rendering choices
-----------------
* Key-rate maps are drawn on a log colour scale, because R spans many decades;
  non-positive values (no secure key) are masked out and drawn as an explicit
  hatched "no key" region rather than being silently clipped to zero.
* A contour line is overlaid at the secure/insecure boundary (R = 0) and, for
  QBER maps, at the 11% asymptotic BB84 threshold, so the operating boundary is
  readable without squinting at colours.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from visualization.theme import (AMBER, GREEN, INK, INK_DIM, ROSE, SEQUENTIAL,
                                 TEAL, style)


def _log_mask(Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (log10 of the positive part, boolean mask of non-positive)."""
    Zp = np.where(Z > 0, Z, np.nan)
    return np.log10(Zp), ~(Z > 0)


def key_rate_heatmap(x, y, Z, *, xlabel: str, ylabel: str,
                     title: str = None, log_x: bool = False, log_y: bool = True,
                     height: int = 560,
                     provenance: str = "analytic model, asymptotic key rate"
                     ) -> go.Figure:
    Zlog, dead = _log_mask(np.asarray(Z, dtype=float))

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x, y=y, z=Zlog, colorscale=SEQUENTIAL, zsmooth="best",
        colorbar=dict(title=dict(text="log₁₀ R<br>[bit/s]", side="right"),
                      tickfont=dict(color=INK_DIM, size=11),
                      outlinewidth=0, thickness=14, len=0.9),
        hovertemplate=(f"{xlabel}: %{{x:.4g}}<br>{ylabel}: %{{y:.4g}}"
                       f"<br>R = 10^%{{z:.2f}} bit/s<extra></extra>")))

    # explicit "no secure key" overlay
    if dead.any():
        fig.add_trace(go.Heatmap(
            x=x, y=y, z=np.where(dead, 1.0, np.nan),
            colorscale=[[0, "rgba(251,113,133,0.55)"], [1, "rgba(251,113,133,0.55)"]],
            showscale=False,
            hovertemplate=(f"{xlabel}: %{{x:.4g}}<br>{ylabel}: %{{y:.4g}}"
                           f"<br><b>no positive key rate</b><extra></extra>")))

    # secure boundary
    Zf = np.asarray(Z, dtype=float)
    if np.nanmin(Zf) < 0 < np.nanmax(Zf):
        fig.add_trace(go.Contour(
            x=x, y=y, z=Zf, contours=dict(start=0, end=0, size=1,
                                          coloring="none",
                                          showlabels=True,
                                          labelfont=dict(size=10, color=GREEN)),
            line=dict(color=GREEN, width=2), showscale=False,
            hoverinfo="skip", name="R = 0"))

    style(fig, title, xlabel, ylabel, height, log_x, log_y, provenance)
    return fig


def metric_heatmap(x, y, Z, *, xlabel: str, ylabel: str, zlabel: str,
                   title: str = None, log_x: bool = False, log_y: bool = True,
                   height: int = 560, contour_at: Optional[float] = None,
                   contour_label: str = "", percent: bool = False,
                   reverse: bool = False,
                   provenance: str = "analytic model") -> go.Figure:
    Zf = np.asarray(Z, dtype=float)
    if percent:
        Zf = 100.0 * Zf
    scale = SEQUENTIAL[::-1] if reverse else SEQUENTIAL
    if reverse:
        scale = [[1 - s[0], s[1]] for s in SEQUENTIAL][::-1]

    fig = go.Figure(go.Heatmap(
        x=x, y=y, z=Zf, colorscale=scale, zsmooth="best",
        colorbar=dict(title=dict(text=zlabel, side="right"),
                      tickfont=dict(color=INK_DIM, size=11),
                      outlinewidth=0, thickness=14, len=0.9),
        hovertemplate=(f"{xlabel}: %{{x:.4g}}<br>{ylabel}: %{{y:.4g}}"
                       f"<br>{zlabel}: %{{z:.4g}}<extra></extra>")))

    if contour_at is not None and np.nanmin(Zf) < contour_at < np.nanmax(Zf):
        fig.add_trace(go.Contour(
            x=x, y=y, z=Zf,
            contours=dict(start=contour_at, end=contour_at, size=1,
                          coloring="none", showlabels=True,
                          labelfont=dict(size=10, color=ROSE)),
            line=dict(color=ROSE, width=2, dash="dash"),
            showscale=False, hoverinfo="skip", name=contour_label))

    style(fig, title, xlabel, ylabel, height, log_x, log_y, provenance)
    return fig


def position_heatmap(xs, ys, Z, room, *, zlabel: str, title: str = None,
                     log_color: bool = True, height: int = 560,
                     tx=None, rx=None) -> go.Figure:
    """Metric mapped over Bob's position on the room floor plane."""
    Zf = np.asarray(Z, dtype=float)
    if log_color:
        Zplot = np.log10(np.where(Zf > 0, Zf, np.nan))
        cbar_title = f"log₁₀ {zlabel}"
    else:
        Zplot = Zf
        cbar_title = zlabel

    fig = go.Figure(go.Heatmap(
        x=xs, y=ys, z=Zplot, colorscale=SEQUENTIAL, zsmooth="best",
        colorbar=dict(title=dict(text=cbar_title, side="right"),
                      tickfont=dict(color=INK_DIM, size=11),
                      outlinewidth=0, thickness=14, len=0.9),
        hovertemplate="x = %{x:.2f} m<br>y = %{y:.2f} m<br>%{z:.3f}<extra></extra>"))

    if tx is not None:
        fig.add_trace(go.Scatter(
            x=[tx[0]], y=[tx[1]], mode="markers+text", name="Alice",
            marker=dict(size=16, color="#60A5FA", symbol="diamond",
                        line=dict(color=INK, width=1.5)),
            text=["Alice"], textposition="bottom center",
            textfont=dict(color="#60A5FA", size=12)))
    if rx is not None:
        fig.add_trace(go.Scatter(
            x=[rx[0]], y=[rx[1]], mode="markers+text", name="Bob (current)",
            marker=dict(size=16, color=GREEN, symbol="square",
                        line=dict(color=INK, width=1.5)),
            text=["Bob"], textposition="top center",
            textfont=dict(color=GREEN, size=12)))

    style(fig, title, "x [m]", "y [m]", height,
          provenance="analytic model; Bob swept over the room at his current height")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
