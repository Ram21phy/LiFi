"""
visualization/room.py
=====================
Room visualisation: 2-D top view and 3-D view of the indoor scene, showing

  * the room envelope
  * ceiling luminaires (LED / fluorescent), sized by optical power
  * Alice (transmitter) and Bob (receiver)
  * the optical propagation path and the beam/radiation cone
  * the receiver field of view
  * the link distance

Nothing here computes physics; positions and angles come from the parameter
set and from models.vlc_channel.compute_geometry.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go

from models.background_noise import default_luminaire_positions
from models.vlc_channel import compute_geometry
from visualization.theme import (AMBER, BLUE, GREEN, INK, INK_DIM, PINK, ROSE,
                                 SLATE, TEAL, VIOLET, style)

WALL = "rgba(148,163,184,0.16)"
FLOOR = "rgba(21,29,43,0.9)"


def _luminaires(params) -> List[Tuple[float, float, float, str, float]]:
    """(x, y, z, kind, power_w) for every enabled ceiling source."""
    out = []
    bg = params.background
    if bg.led.enabled and bg.led.n_luminaires > 0:
        p = bg.led.optical_power_per_luminaire_w
        if bg.led.drive_by_illuminance:
            from utils.constants import WHITE_LED_LER_LM_PER_W_OPT
            area = params.room.length_x * params.room.width_y
            p = (bg.led.target_illuminance_lux * area
                 / WHITE_LED_LER_LM_PER_W_OPT / max(bg.led.n_luminaires, 1))
        pos = bg.led.positions or default_luminaire_positions(
            params.room, bg.led.n_luminaires)
        for (x, y) in pos:
            out.append((x, y, bg.led.mount_height_m, "LED", p))
    if bg.fluorescent.enabled and bg.fluorescent.n_luminaires > 0:
        pos = bg.fluorescent.positions or default_luminaire_positions(
            params.room, bg.fluorescent.n_luminaires)
        for (x, y) in pos:
            out.append((x, y, bg.fluorescent.mount_height_m, "Fluorescent",
                        bg.fluorescent.optical_power_per_luminaire_w))
    return out


# ==========================================================================
def top_view(params, height: int = 520) -> go.Figure:
    r = params.room
    geom = compute_geometry(r, params.channel.rx_fov_deg)
    fig = go.Figure()

    # room envelope
    fig.add_shape(type="rect", x0=0, y0=0, x1=r.length_x, y1=r.width_y,
                  line=dict(color=SLATE, width=2), fillcolor=FLOOR, layer="below")

    # luminaires
    lums = _luminaires(params)
    if lums:
        pmax = max(l[4] for l in lums) or 1.0
        for kind, colour in (("LED", AMBER), ("Fluorescent", VIOLET)):
            sel = [l for l in lums if l[3] == kind]
            if not sel:
                continue
            fig.add_trace(go.Scatter(
                x=[l[0] for l in sel], y=[l[1] for l in sel],
                mode="markers", name=f"{kind} luminaire",
                marker=dict(size=[16 + 22 * (l[4] / pmax) for l in sel],
                            color=colour, opacity=0.35,
                            line=dict(color=colour, width=2)),
                hovertemplate=(f"{kind} luminaire<br>(%{{x:.2f}}, %{{y:.2f}}) m"
                               f"<br>%{{customdata:.3f}} W optical<extra></extra>"),
                customdata=[l[4] for l in sel]))

    # receiver FOV wedge (projected on the floor plane)
    ux, uy = (r.tx_x - r.rx_x), (r.tx_y - r.rx_y)
    nrm = math.hypot(ux, uy) or 1.0
    ux, uy = ux / nrm, uy / nrm
    base = math.atan2(uy, ux)
    fov = math.radians(params.channel.rx_fov_deg)
    reach = geom.distance_m * 1.15
    ang = np.linspace(base - fov, base + fov, 60)
    fig.add_trace(go.Scatter(
        x=np.concatenate([[r.rx_x], r.rx_x + reach * np.cos(ang), [r.rx_x]]),
        y=np.concatenate([[r.rx_y], r.rx_y + reach * np.sin(ang), [r.rx_y]]),
        fill="toself", fillcolor="rgba(45,212,191,0.10)",
        line=dict(color=TEAL, width=1, dash="dot"),
        name=f"Receiver FOV ±{params.channel.rx_fov_deg:.1f}°",
        hoverinfo="skip"))

    # link
    fig.add_trace(go.Scatter(
        x=[r.tx_x, r.rx_x], y=[r.tx_y, r.rx_y], mode="lines",
        line=dict(color=TEAL, width=3), name="Quantum channel",
        hovertemplate="quantum link<extra></extra>"))
    fig.add_annotation(x=0.5 * (r.tx_x + r.rx_x), y=0.5 * (r.tx_y + r.rx_y),
                       text=f"<b>d = {geom.distance_m:.3f} m</b>",
                       showarrow=False, yshift=16,
                       font=dict(color=TEAL, size=13),
                       bgcolor="rgba(14,20,32,0.75)")

    # terminals
    fig.add_trace(go.Scatter(
        x=[r.tx_x], y=[r.tx_y], mode="markers+text", name="Alice (Tx)",
        marker=dict(size=20, color=BLUE, symbol="diamond",
                    line=dict(color=INK, width=1.5)),
        text=["Alice"], textposition="bottom center",
        textfont=dict(color=BLUE, size=12),
        hovertemplate="Alice<br>(%{x:.2f}, %{y:.2f}) m<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[r.rx_x], y=[r.rx_y], mode="markers+text", name="Bob (Rx)",
        marker=dict(size=20, color=GREEN, symbol="square",
                    line=dict(color=INK, width=1.5)),
        text=["Bob"], textposition="top center",
        textfont=dict(color=GREEN, size=12),
        hovertemplate="Bob<br>(%{x:.2f}, %{y:.2f}) m<extra></extra>"))

    style(fig, None, "x [m]", "y [m]", height,
          provenance="room geometry (top view); heights shown in the 3-D view")
    # constrain="domain" keeps the 1:1 aspect ratio by shrinking the plotting
    # area rather than by inflating the axis ranges, so the room stays centred
    fig.update_xaxes(range=[-0.35, r.length_x + 0.35], constrain="domain")
    fig.update_yaxes(range=[-0.35, r.width_y + 0.35],
                     scaleanchor="x", scaleratio=1, constrain="domain")
    return fig


# ==========================================================================
def _box_edges(lx, ly, lz):
    pts = [(0, 0, 0), (lx, 0, 0), (lx, ly, 0), (0, ly, 0),
           (0, 0, lz), (lx, 0, lz), (lx, ly, lz), (0, ly, lz)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [pts[a][0], pts[b][0], None]
        ys += [pts[a][1], pts[b][1], None]
        zs += [pts[a][2], pts[b][2], None]
    return xs, ys, zs


def three_d_view(params, height: int = 600) -> go.Figure:
    r = params.room
    geom = compute_geometry(r, params.channel.rx_fov_deg)
    fig = go.Figure()

    xs, ys, zs = _box_edges(r.length_x, r.width_y, r.height_z)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                               line=dict(color=SLATE, width=3),
                               name="Room", hoverinfo="skip"))

    # floor surface
    fig.add_trace(go.Surface(
        x=[[0, r.length_x], [0, r.length_x]],
        y=[[0, 0], [r.width_y, r.width_y]],
        z=[[0, 0], [0, 0]],
        showscale=False, opacity=0.25,
        colorscale=[[0, "#131C2A"], [1, "#131C2A"]], hoverinfo="skip",
        name="Floor"))

    lums = _luminaires(params)
    if lums:
        pmax = max(l[4] for l in lums) or 1.0
        for kind, colour in (("LED", AMBER), ("Fluorescent", VIOLET)):
            sel = [l for l in lums if l[3] == kind]
            if not sel:
                continue
            fig.add_trace(go.Scatter3d(
                x=[l[0] for l in sel], y=[l[1] for l in sel],
                z=[l[2] for l in sel], mode="markers",
                name=f"{kind} luminaire",
                marker=dict(size=[7 + 10 * (l[4] / pmax) for l in sel],
                            color=colour, opacity=0.75,
                            line=dict(color=colour, width=1)),
                customdata=[l[4] for l in sel],
                hovertemplate=(f"{kind}<br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}}) m"
                               f"<br>%{{customdata:.3f}} W optical<extra></extra>")))

    # beam cone from Alice towards Bob
    tx = np.array([r.tx_x, r.tx_y, r.tx_z])
    rx = np.array([r.rx_x, r.rx_y, r.rx_z])
    u = rx - tx
    d = np.linalg.norm(u) or 1.0
    u = u / d
    # an orthonormal frame around the link axis
    tmp = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(tmp, u)) > 0.95:
        tmp = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(u, tmp)
    e1 /= (np.linalg.norm(e1) or 1.0)
    e2 = np.cross(u, e1)

    if params.channel.channel_model.startswith("Collimated"):
        half_angle = params.channel.beam_divergence_mrad * 1e-3
        cone_label = f"Beam (divergence {params.channel.beam_divergence_mrad:g} mrad)"
        radius_end = max(params.channel.beam_waist_mm * 1e-3 + d * half_angle, 1e-3)
        radius_end = max(radius_end, 0.04)      # visual floor so it is visible
    else:
        half_angle = math.radians(params.channel.tx_half_power_semiangle_deg)
        cone_label = (f"Radiation lobe (Φ½ = "
                      f"{params.channel.tx_half_power_semiangle_deg:g}°)")
        radius_end = d * math.tan(half_angle)

    th = np.linspace(0, 2 * np.pi, 48)
    ring = (rx[:, None] + radius_end * (np.cos(th) * e1[:, None]
                                        + np.sin(th) * e2[:, None]))
    for i in range(0, len(th), 4):
        fig.add_trace(go.Scatter3d(
            x=[tx[0], ring[0, i]], y=[tx[1], ring[1, i]], z=[tx[2], ring[2, i]],
            mode="lines", line=dict(color=TEAL, width=1),
            opacity=0.35, showlegend=(i == 0), name=cone_label,
            hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=ring[0], y=ring[1], z=ring[2], mode="lines",
        line=dict(color=TEAL, width=2), opacity=0.5,
        showlegend=False, hoverinfo="skip"))

    # link line
    fig.add_trace(go.Scatter3d(
        x=[tx[0], rx[0]], y=[tx[1], rx[1]], z=[tx[2], rx[2]], mode="lines",
        line=dict(color=TEAL, width=6), name=f"Link, d = {geom.distance_m:.3f} m",
        hovertemplate=f"d = {geom.distance_m:.3f} m<extra></extra>"))

    fig.add_trace(go.Scatter3d(
        x=[tx[0]], y=[tx[1]], z=[tx[2]], mode="markers+text",
        marker=dict(size=9, color=BLUE, symbol="diamond"),
        text=["Alice"], textposition="top center", name="Alice (Tx)",
        textfont=dict(color=BLUE, size=12)))
    fig.add_trace(go.Scatter3d(
        x=[rx[0]], y=[rx[1]], z=[rx[2]], mode="markers+text",
        marker=dict(size=9, color=GREEN, symbol="square"),
        text=["Bob"], textposition="top center", name="Bob (Rx)",
        textfont=dict(color=GREEN, size=12)))

    fig.update_layout(
        height=height, template="vlqkd",
        scene=dict(
            xaxis=dict(title="x [m]", range=[0, r.length_x],
                       backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(37,48,67,0.333)",
                       color=INK_DIM),
            yaxis=dict(title="y [m]", range=[0, r.width_y],
                       backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(37,48,67,0.333)",
                       color=INK_DIM),
            zaxis=dict(title="z [m]", range=[0, r.height_z],
                       backgroundcolor="rgba(0,0,0,0)", gridcolor="rgba(37,48,67,0.333)",
                       color=INK_DIM),
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.7, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=8, b=0),
        legend=dict(orientation="h", y=-0.02, x=0.0, font=dict(size=11)),
    )
    return fig


# ==========================================================================
def receiver_position_map(params, metric: str = "Secret key rate [bit/s]",
                          n: int = 26, height: int = 540):
    """
    Sweep Bob's (x, y) over the room floor plane at his current height and map
    the chosen metric.  This is the "where can I stand?" figure.
    """
    from qkd.bb84 import run_analytic
    from simulation.parameter_sweep import METRICS

    r = params.room
    xs = np.linspace(0.15, r.length_x - 0.15, n)
    ys = np.linspace(0.15, r.width_y - 0.15, n)
    Z = np.full((n, n), np.nan)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            p = params.copy()
            p.room.rx_x, p.room.rx_y = float(x), float(y)
            try:
                Z[j, i] = float(METRICS[metric](run_analytic(p)))
            except Exception:
                Z[j, i] = np.nan
    return xs, ys, Z
