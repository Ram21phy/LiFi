"""
visualization/theme.py
======================
One place that defines how every figure in the toolkit looks, so the whole
application reads as a single instrument rather than a pile of plots.

Palette design
--------------
The categorical colours are chosen to stay distinguishable in the most common
forms of colour-vision deficiency (no red/green pair carries meaning on its
own) and to hold their contrast on the dark instrument background.  Semantic
colours are fixed by meaning, not by series order:

    signal / key      teal        "this is the thing you want"
    background noise  amber       "this is what is fighting you"
    dark counts       violet
    error / insecure  rose
    secure region     green
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# --------------------------------------------------------------------------
INK = "#E8EDF4"
INK_DIM = "#9AA7B8"
BG = "#0E1420"
PANEL = "#151D2B"
GRID = "rgba(37,48,67,0.2)"
AXIS = "#3A4759"

TEAL = "#2DD4BF"
AMBER = "#F5A524"
VIOLET = "#A78BFA"
ROSE = "#FB7185"
GREEN = "#4ADE80"
BLUE = "#60A5FA"
PINK = "#F472B6"
LIME = "#BEF264"
SLATE = "#94A3B8"

CATEGORICAL = [TEAL, AMBER, VIOLET, BLUE, ROSE, LIME, PINK, SLATE]

SEMANTIC = {
    "signal": TEAL,
    "key": TEAL,
    "secret_key": TEAL,
    "sifted": BLUE,
    "background": AMBER,
    "dark": VIOLET,
    "afterpulse": PINK,
    "polarization": BLUE,
    "pointing": SLATE,
    "error": ROSE,
    "qber": ROSE,
    "secure": GREEN,
    "insecure": ROSE,
    "eve": ROSE,
    "mutual_information": VIOLET,
}

# Sequential scale for heatmaps: dark -> teal -> amber (perceptually ordered,
# and monotone in lightness so it survives greyscale printing).
SEQUENTIAL = [
    [0.00, "#0B1020"],
    [0.18, "#12304A"],
    [0.38, "#0E6E79"],
    [0.58, "#2DD4BF"],
    [0.78, "#C7E86B"],
    [1.00, "#F5A524"],
]

DIVERGING = [
    [0.0, "#FB7185"],
    [0.5, "#151D2B"],
    [1.0, "#2DD4BF"],
]


def register_template() -> str:
    """Register (once) and return the name of the toolkit plotly template."""
    name = "vlqkd"
    if name in pio.templates:
        return name
    t = go.layout.Template()
    t.layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, 'Segoe UI', system-ui, sans-serif",
                  size=13, color=INK),
        title=dict(font=dict(size=16, color=INK), x=0.0, xanchor="left"),
        colorway=CATEGORICAL,
        xaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   ticks="outside", tickcolor=AXIS, tickfont=dict(color=INK_DIM),
                   title=dict(font=dict(color=INK_DIM)), showline=True,
                   mirror=False, automargin=True),
        yaxis=dict(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                   ticks="outside", tickcolor=AXIS, tickfont=dict(color=INK_DIM),
                   title=dict(font=dict(color=INK_DIM)), showline=True,
                   mirror=False, automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=AXIS, borderwidth=0,
                    font=dict(color=INK_DIM, size=12),
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0.0),
        margin=dict(l=64, r=24, t=56, b=56),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=AXIS,
                        font=dict(color=INK, size=12)),
        colorscale=dict(sequential=SEQUENTIAL, diverging=DIVERGING),
    )
    pio.templates[name] = t
    return name


TEMPLATE = register_template()


def style(fig: go.Figure, title: str = None, xtitle: str = None,
          ytitle: str = None, height: int = 420, log_x: bool = False,
          log_y: bool = False, provenance: str = None) -> go.Figure:
    """Apply the toolkit template plus the usual axis/annotation furniture."""
    fig.update_layout(template=TEMPLATE, height=height)
    if title:
        fig.update_layout(title=dict(text=title))
    if xtitle:
        fig.update_xaxes(title_text=xtitle)
    if ytitle:
        fig.update_yaxes(title_text=ytitle)
    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    if provenance:
        fig.add_annotation(
            text=provenance, xref="paper", yref="paper", x=1.0, y=-0.16,
            xanchor="right", yanchor="top", showarrow=False,
            font=dict(size=10, color=INK_DIM))
    return fig


def add_threshold(fig: go.Figure, y: float, label: str, color: str = ROSE,
                  dash: str = "dash"):
    """Horizontal reference line with an in-plot label."""
    fig.add_hline(y=y, line=dict(color=color, width=1.2, dash=dash),
                  annotation_text=label, annotation_position="top left",
                  annotation_font=dict(size=11, color=color))
    return fig


def add_secure_shading(fig: go.Figure, x, secure_mask, y0=None, y1=None):
    """Shade the x-ranges where a positive key rate exists."""
    import numpy as np
    x = np.asarray(x, dtype=float)
    m = np.asarray(secure_mask, dtype=bool)
    if not m.any():
        return fig
    start = None
    for i in range(len(m)):
        if m[i] and start is None:
            start = x[i]
        if (not m[i] or i == len(m) - 1) and start is not None:
            end = x[i] if not m[i] else x[-1]
            fig.add_vrect(x0=start, x1=end, fillcolor=GREEN, opacity=0.07,
                          line_width=0, layer="below")
            start = None
    return fig
