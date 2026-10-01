"""
app.py
======
Indoor Visible-Light Quantum Key Distribution (BB84) Simulator - Streamlit GUI.

Run with:      streamlit run app.py

The GUI is a thin presentation layer. Every number it displays comes from the
`models/`, `qkd/` and `simulation/` packages, and every panel states whether the
value is analytic, Monte-Carlo, asymptotic, finite-size, or produced by one of
the simplified sub-models.
"""

from __future__ import annotations

import io
import json
import math
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from models import background_noise as bgm
from models.vlc_channel import lambertian_order, semiangle_from_order
from qkd.bb84 import protocol_step_table, run_analytic
from simulation import parameter_sweep as sweepmod
from simulation.finite_size import evaluate_finite_key
from simulation.monte_carlo import run_monte_carlo
from utils import constants as C
from utils import export as ex
from utils.parameters import SimulationParameters, default_parameters
from utils.validation import validate
from visualization import heatmaps as hm
from visualization import plots as pl
from visualization import room as roomviz
from visualization.theme import (AMBER, BLUE, GREEN, INK_DIM, LIME, PINK, ROSE,
                                 SLATE, TEAL, VIOLET)

st.set_page_config(
    page_title="Indoor VL-QKD BB84 Simulator",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================================
#  Styling
# ==========================================================================
CSS = """
<style>
:root{
  --ink:#E8EDF4; --dim:#9AA7B8; --bg:#0E1420; --panel:#151D2B;
  --line:#25304a; --teal:#2DD4BF; --amber:#F5A524; --rose:#FB7185;
  --green:#4ADE80; --violet:#A78BFA; --blue:#60A5FA;
}
.stApp{
  background:
    radial-gradient(1200px 620px at 12% -8%, #16243a 0%, rgba(22,36,58,0) 62%),
    radial-gradient(1000px 540px at 88% 0%, #12303a 0%, rgba(18,48,58,0) 58%),
    var(--bg);
  color:var(--ink);
}
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#121A27 0%, #0E1420 100%);
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown p{color:var(--dim);font-size:.85rem}

h1,h2,h3,h4{color:var(--ink);letter-spacing:-.01em}
hr{border-color:var(--line)}

/* ---------- hero ---------- */
.hero{
  border:1px solid var(--line);
  border-radius:18px;
  padding:22px 26px;
  background:
    linear-gradient(135deg, rgba(45,212,191,.10) 0%, rgba(96,165,250,.06) 45%,
                    rgba(245,165,36,.07) 100%),
    var(--panel);
  margin-bottom:6px;
}
.hero h1{margin:0 0 4px 0;font-size:1.65rem;font-weight:650}
.hero .sub{color:var(--dim);font-size:.92rem;margin:0}
.chainrow{display:flex;flex-wrap:wrap;gap:6px;margin-top:14px}
.chain{
  font-size:.72rem;padding:4px 10px;border-radius:999px;
  border:1px solid var(--line);background:#101827;color:var(--dim);
  white-space:nowrap;
}
.chain.on{border-color:rgba(45,212,191,.45);color:var(--teal);
  background:rgba(45,212,191,.08)}

/* ---------- KPI cards ---------- */
.kgrid{display:grid;gap:12px;margin:4px 0 6px 0}
.kgrid.c4{grid-template-columns:repeat(4,minmax(0,1fr))}
.kgrid.c3{grid-template-columns:repeat(3,minmax(0,1fr))}
.kgrid.c2{grid-template-columns:repeat(2,minmax(0,1fr))}
@media (max-width:1100px){.kgrid.c4{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:760px){.kgrid.c4,.kgrid.c3,.kgrid.c2{grid-template-columns:1fr}}
.kpi{
  border:1px solid var(--line);border-radius:14px;padding:14px 16px;
  background:linear-gradient(180deg,#18212F 0%, #141C29 100%);
  position:relative;overflow:hidden;
}
.kpi:before{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;
  background:var(--accent,#2DD4BF);opacity:.9}
.kpi .lab{font-size:.70rem;letter-spacing:.07em;text-transform:uppercase;
  color:var(--dim);margin-bottom:6px;display:block}
.kpi .val{font-size:1.42rem;font-weight:640;line-height:1.15;
  font-variant-numeric:tabular-nums;color:var(--ink)}
.kpi .unit{font-size:.80rem;color:var(--dim);font-weight:400;margin-left:4px}
.kpi .note{font-size:.72rem;color:var(--dim);margin-top:6px;display:block}

/* ---------- status banner ---------- */
.banner{border-radius:14px;padding:14px 18px;margin:8px 0 14px 0;
  border:1px solid;font-size:.92rem;display:flex;gap:12px;align-items:flex-start}
.banner .big{font-weight:650;font-size:1.02rem}
.banner.ok{border-color:rgba(74,222,128,.45);background:rgba(74,222,128,.09);
  color:#BBF7D0}
.banner.bad{border-color:rgba(251,113,133,.45);background:rgba(251,113,133,.09);
  color:#FECDD3}
.banner.warn{border-color:rgba(245,165,36,.45);background:rgba(245,165,36,.09);
  color:#FDE9C6}

/* ---------- panels ---------- */
.panel{border:1px solid var(--line);border-radius:14px;padding:16px 18px;
  background:var(--panel);margin-bottom:12px}
.panel h4{margin:0 0 8px 0;font-size:.95rem}
.prov{display:inline-block;font-size:.68rem;letter-spacing:.05em;
  text-transform:uppercase;border:1px solid var(--line);border-radius:999px;
  padding:2px 9px;color:var(--dim);background:#101827;margin-right:6px}
.eq{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82rem;
  background:#0B1220;border:1px solid var(--line);border-radius:10px;
  padding:10px 12px;color:#CBD5E1;overflow-x:auto;white-space:pre-wrap}

/* ---------- streamlit widgets ---------- */
/* tab strip - selectors cover both the older baseweb and the newer
   react-aria DOM that Streamlit ships, so the styling survives upgrades */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"]{
  gap:2px;border-bottom:1px solid var(--line);flex-wrap:wrap}
.stTabs [data-baseweb="tab"], .stTabs [data-testid="stTab"]{
  padding:6px 14px;color:var(--dim);font-size:.86rem;background:transparent;
  border-radius:10px 10px 0 0}
.stTabs [data-testid="stTab"] p{margin:0;font-size:.86rem}
.stTabs [aria-selected="true"]{color:var(--teal) !important;
  background:rgba(45,212,191,.08) !important;
  border-bottom:2px solid var(--teal) !important}
.stTabs [aria-selected="true"] p{color:var(--teal) !important;font-weight:600}
div[data-testid="stMetricValue"]{font-size:1.25rem}
.stDataFrame{border:1px solid var(--line);border-radius:10px}
div[data-testid="stExpander"]{border:1px solid var(--line);border-radius:12px;
  background:#131B28}
.stButton>button, .stDownloadButton>button{
  border-radius:10px;border:1px solid var(--line);background:#182231;
  color:var(--ink);font-size:.86rem}
.stButton>button:hover, .stDownloadButton>button:hover{
  border-color:var(--teal);color:var(--teal)}
.small{font-size:.80rem;color:var(--dim)}
.footer{border-top:1px solid var(--line);margin-top:26px;padding-top:12px;
  color:var(--dim);font-size:.78rem}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ==========================================================================
#  Small formatting helpers
# ==========================================================================
def fmt(v: float, sig: int = 4, unit: str = "") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    a = abs(v)
    if a == 0:
        s = "0"
    elif a >= 1e5 or a < 1e-3:
        s = f"{v:.{sig-1}e}".replace("e+0", "e+").replace("e-0", "e-")
    elif a >= 100:
        s = f"{v:,.1f}"
    elif a >= 1:
        s = f"{v:.3f}"
    else:
        s = f"{v:.{sig}f}"
    return s + (f"<span class='unit'>{unit}</span>" if unit else "")


def kpi(label: str, value: str, note: str = "", accent: str = TEAL) -> str:
    return (f"<div class='kpi' style='--accent:{accent}'>"
            f"<span class='lab'>{label}</span>"
            f"<div class='val'>{value}</div>"
            + (f"<span class='note'>{note}</span>" if note else "")
            + "</div>")


def kpi_grid(cards: List[str], cols: int = 4):
    st.markdown(f"<div class='kgrid c{cols}'>" + "".join(cards) + "</div>",
                unsafe_allow_html=True)


def panel(title: str, body_md: str = "", prov: str = ""):
    tag = f"<span class='prov'>{prov}</span>" if prov else ""
    st.markdown(f"<div class='panel'><h4>{title} {tag}</h4>"
                f"<div class='small'>{body_md}</div></div>",
                unsafe_allow_html=True)


# ==========================================================================
#  Session state
# ==========================================================================
if "params" not in st.session_state:
    st.session_state.params = default_parameters()
if "selftest" not in st.session_state:
    st.session_state.selftest = None
if "mc" not in st.session_state:
    st.session_state.mc = None
if "sweep_df" not in st.session_state:
    st.session_state.sweep_df = None
if "sweep_meta" not in st.session_state:
    st.session_state.sweep_meta = {}
if "heat" not in st.session_state:
    st.session_state.heat = None
if "filter_opt" not in st.session_state:
    st.session_state.filter_opt = None
if "posmap" not in st.session_state:
    st.session_state.posmap = None


# ==========================================================================
#  Self-test at launch (requirement 25)
# ==========================================================================
@st.cache_data(show_spinner=False)
def _cached_selftest(_version: str = "1.0"):
    from utils.selftest import run_all
    res = run_all(include_slow=True)
    return [{"name": r.name, "passed": r.passed, "detail": r.detail,
             "category": r.category, "ms": r.duration_ms} for r in res]


if st.session_state.selftest is None:
    with st.spinner("Running start-up self-test…"):
        st.session_state.selftest = _cached_selftest()

ST = st.session_state.selftest
N_PASS = sum(1 for t in ST if t["passed"])
N_TEST = len(ST)


# ==========================================================================
#  SIDEBAR - every physical, optical, channel, detector and BB84 parameter
# ==========================================================================
P: SimulationParameters = st.session_state.params


def sidebar() -> SimulationParameters:
    p = st.session_state.params
    sb = st.sidebar

    sb.markdown("### 🔐 VL-QKD Control Panel")
    sb.caption("Every value below is user-adjustable. Defaults are labelled "
               "**demonstration parameters** and are not experimentally validated.")

    # ---------------- presets ----------------
    with sb.expander("⚡ Scenario presets", expanded=False):
        preset = st.selectbox(
            "Background illumination scenario",
            ["(keep current settings)"] + list(bgm.BACKGROUND_SCENARIOS.keys()),
            key="preset_sel")
        if st.button("Apply background scenario", use_container_width=True):
            if preset != "(keep current settings)":
                st.session_state.params = bgm.apply_scenario(p, preset)
                st.rerun()
        if preset in bgm.BACKGROUND_SCENARIOS:
            st.caption(bgm.BACKGROUND_SCENARIOS[preset].get("note", ""))
        st.divider()
        if st.button("↺ Reset everything to demonstration defaults",
                     use_container_width=True):
            st.session_state.params = default_parameters()
            for k in ("mc", "sweep_df", "heat", "filter_opt", "posmap"):
                st.session_state[k] = None
            st.rerun()

        up = st.file_uploader("Load a parameter set (JSON)", type=["json"],
                              key="param_upload")
        if up is not None:
            try:
                st.session_state.params = SimulationParameters.from_json(
                    up.read().decode("utf-8"))
                st.success("Parameter set loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read that file: {exc}")

    # ---------------- room ----------------
    with sb.expander("🏠 Room & geometry", expanded=False):
        r = p.room
        c1, c2, c3 = st.columns(3)
        r.length_x = c1.number_input("Length x [m]", 1.0, 60.0, float(r.length_x), 0.5)
        r.width_y = c2.number_input("Width y [m]", 1.0, 60.0, float(r.width_y), 0.5)
        r.height_z = c3.number_input("Height z [m]", 1.5, 15.0, float(r.height_z), 0.1)

        st.markdown("**Alice (transmitter)**")
        c1, c2, c3 = st.columns(3)
        r.tx_x = c1.number_input("Tx x [m]", 0.0, float(r.length_x), float(min(r.tx_x, r.length_x)), 0.05)
        r.tx_y = c2.number_input("Tx y [m]", 0.0, float(r.width_y), float(min(r.tx_y, r.width_y)), 0.05)
        r.tx_z = c3.number_input("Tx height [m]", 0.0, float(r.height_z), float(min(r.tx_z, r.height_z)), 0.05)

        st.markdown("**Bob (receiver)**")
        c1, c2, c3 = st.columns(3)
        r.rx_x = c1.number_input("Rx x [m]", 0.0, float(r.length_x), float(min(r.rx_x, r.length_x)), 0.05)
        r.rx_y = c2.number_input("Rx y [m]", 0.0, float(r.width_y), float(min(r.rx_y, r.width_y)), 0.05)
        r.rx_z = c3.number_input("Rx height [m]", 0.0, float(r.height_z), float(min(r.rx_z, r.height_z)), 0.05)

        r.tx_aim_at_rx = st.checkbox("Transmitter aimed at Bob (steered link)",
                                     value=bool(r.tx_aim_at_rx))
        r.rx_aim_at_tx = st.checkbox("Receiver aimed at Alice",
                                     value=bool(r.rx_aim_at_tx))
        r.override_distance = st.checkbox(
            "Override link distance (keep direction)", value=bool(r.override_distance),
            help="Uses the Tx→Rx direction but forces the separation to the "
                 "value below. Useful for distance sweeps.")
        if r.override_distance:
            r.link_distance_m = st.number_input(
                "Link distance d [m]", 0.05, 100.0, float(r.link_distance_m), 0.1)

        st.markdown("**Surface reflectivities** (used by the diffuse "
                    "background model)")
        c1, c2, c3 = st.columns(3)
        r.wall_reflectivity = c1.slider("Walls", 0.0, 0.95, float(r.wall_reflectivity), 0.01)
        r.ceiling_reflectivity = c2.slider("Ceiling", 0.0, 0.95, float(r.ceiling_reflectivity), 0.01)
        r.floor_reflectivity = c3.slider("Floor", 0.0, 0.95, float(r.floor_reflectivity), 0.01)

    # ---------------- source ----------------
    with sb.expander("💡 Source & photon statistics", expanded=True):
        s = p.source
        s.source_model = st.radio("Source model", ["WCP (Poisson)", "Ideal single photon"],
                                  index=0 if s.source_model.startswith("WCP") else 1,
                                  horizontal=True,
                                  help="Weak coherent pulses follow "
                                       "P(n)=e^-µ µⁿ/n!. The ideal single-photon "
                                       "option is a reference bound, not a device.")
        s.wavelength_nm = st.slider("Wavelength λ [nm]", 380.0, 1000.0,
                                    float(s.wavelength_nm), 1.0)
        s.mu = st.number_input("Mean photon number µ [photons/pulse]",
                               1e-4, 5.0, float(s.mu), 0.01, format="%.4f")
        s.pulse_rate_hz = st.number_input(
            "Pulse repetition rate r_p [Hz]", 1e3, 1e11, float(s.pulse_rate_hz),
            1e6, format="%.4g")
        s.source_linewidth_nm = st.number_input(
            "Signal linewidth [nm]", 1e-6, 50.0, float(s.source_linewidth_nm),
            0.01, format="%.4g",
            help="Only used for a consistency warning against the filter width.")
        if p.protocol.protocol.startswith("Decoy"):
            c1, c2 = st.columns(2)
            s.decoy_nu1 = c1.number_input("Decoy ν₁", 1e-4, 1.0,
                                          float(s.decoy_nu1), 0.01, format="%.4f")
            s.decoy_nu2 = c2.number_input("Vacuum ν₂", 0.0, 1.0,
                                          float(s.decoy_nu2), 0.01, format="%.4f")

    # ---------------- channel ----------------
    with sb.expander("📡 Optical channel", expanded=True):
        ch = p.channel
        ch.channel_model = st.radio(
            "Channel model",
            ["Collimated Gaussian beam", "Lambertian (VLC LOS)"],
            index=0 if ch.channel_model.startswith("Collimated") else 1,
            help="The Lambertian model is the standard indoor-VLC LOS DC gain. "
                 "The collimated-beam model is closer to a real free-space QKD "
                 "transmitter.")
        if ch.channel_model.startswith("Lambertian"):
            ch.tx_half_power_semiangle_deg = st.slider(
                "Tx half-power semi-angle Φ½ [deg]", 0.5, 80.0,
                float(ch.tx_half_power_semiangle_deg), 0.5)
            st.caption(f"Lambertian order m = "
                       f"{lambertian_order(ch.tx_half_power_semiangle_deg):.2f}")
        else:
            c1, c2 = st.columns(2)
            ch.beam_waist_mm = c1.number_input("Beam waist w₀ [mm]", 0.05, 100.0,
                                               float(ch.beam_waist_mm), 0.1)
            ch.beam_divergence_mrad = c2.number_input(
                "Beam divergence [mrad]", 0.01, 200.0,
                float(ch.beam_divergence_mrad), 0.1)

        ch.rx_aperture_area_cm2 = st.number_input(
            "Receiver aperture area A [cm²]", 0.001, 500.0,
            float(ch.rx_aperture_area_cm2), 0.1, format="%.4g")
        ch.rx_fov_deg = st.slider("Receiver FOV half-angle Ψ_c [deg]", 0.5, 89.0,
                                  float(ch.rx_fov_deg), 0.5)
        ch.use_concentrator = st.checkbox(
            "Non-imaging concentrator g(ψ)=n²/sin²Ψ_c",
            value=bool(ch.use_concentrator),
            help="Boosts the signal by n²/sin²Ψ_c but, by étendue conservation, "
                 "leaves a FOV-filling diffuse background unchanged.")
        if ch.use_concentrator:
            ch.concentrator_index = st.slider("Concentrator index n", 1.0, 2.5,
                                              float(ch.concentrator_index), 0.05)
        ch.optical_efficiency = st.slider("Receiver optical efficiency η_opt",
                                          0.01, 1.0, float(ch.optical_efficiency), 0.01)
        ch.extra_loss_db = st.number_input("Additional optical loss [dB]", 0.0, 60.0,
                                           float(ch.extra_loss_db), 0.5)

        st.markdown("**Optional impairments**")
        ch.enable_pointing_error = st.checkbox("Pointing error / beam wander",
                                               value=bool(ch.enable_pointing_error))
        if ch.enable_pointing_error:
            ch.pointing_jitter_mrad = st.slider("Pointing jitter σ [mrad]", 0.0, 20.0,
                                                float(ch.pointing_jitter_mrad), 0.05)
        ch.enable_turbulence = st.checkbox("Atmospheric turbulence (log-normal)",
                                           value=bool(ch.enable_turbulence))
        if ch.enable_turbulence:
            e = st.slider("log₁₀ Cn² [m⁻²ᐟ³]", -18.0, -11.0,
                          float(np.log10(max(ch.turbulence_cn2, 1e-18))), 0.1)
            ch.turbulence_cn2 = float(10 ** e)
            st.caption("Indoor Cn² is typically 10⁻¹⁵–10⁻¹³ m⁻²ᐟ³, so this is "
                       "usually a negligible effect over a few metres.")

    # ---------------- filter ----------------
    with sb.expander("🔬 Optical band-pass filter", expanded=True):
        f = p.filt
        f.center_nm = st.slider("Centre λ_filter [nm]", 380.0, 1000.0,
                                float(f.center_nm), 0.5)
        f.bandwidth_nm = st.number_input("Bandwidth Δλ (FWHM) [nm]", 0.001, 300.0,
                                         float(f.bandwidth_nm), 0.01, format="%.4g")
        f.peak_transmission = st.slider("Peak transmission T_filter", 0.01, 1.0,
                                        float(f.peak_transmission), 0.01)
        f.shape = st.selectbox("Filter line shape", ["Gaussian", "Top-hat", "Lorentzian"],
                               index=["Gaussian", "Top-hat", "Lorentzian"].index(f.shape))
        f.out_of_band_rejection_db = st.slider("Out-of-band blocking [dB]", 10.0, 120.0,
                                               float(f.out_of_band_rejection_db), 5.0)
        if st.button("Match filter centre to the signal", use_container_width=True):
            p.filt.center_nm = p.source.wavelength_nm
            st.rerun()

    # ---------------- background ----------------
    with sb.expander("🔆 Background illumination", expanded=True):
        bg = p.background
        st.markdown("**A. LED room illumination**")
        bg.led.enabled = st.checkbox("Enable ceiling LEDs", value=bool(bg.led.enabled))
        if bg.led.enabled:
            c1, c2 = st.columns(2)
            bg.led.n_luminaires = c1.number_input("Number of luminaires", 1, 64,
                                                  int(bg.led.n_luminaires), 1)
            bg.led.mount_height_m = c2.number_input(
                "Mount height [m]", 0.5, float(p.room.height_z),
                float(min(bg.led.mount_height_m, p.room.height_z)), 0.05)
            bg.led.drive_by_illuminance = st.checkbox(
                "Specify by illuminance (lux) instead of optical watts",
                value=bool(bg.led.drive_by_illuminance))
            if bg.led.drive_by_illuminance:
                bg.led.target_illuminance_lux = st.number_input(
                    "Target illuminance [lx]", 0.0, 5000.0,
                    float(bg.led.target_illuminance_lux), 10.0,
                    help="Converted to radiometric watts with a luminous "
                         f"efficacy of {C.WHITE_LED_LER_LM_PER_W_OPT:.0f} lm/W_opt.")
                area = p.room.length_x * p.room.width_y
                w = (bg.led.target_illuminance_lux * area
                     / C.WHITE_LED_LER_LM_PER_W_OPT)
                st.caption(f"≈ {w:.3g} W optical in total "
                           f"({w/max(bg.led.n_luminaires,1):.3g} W per luminaire)")
            else:
                bg.led.optical_power_per_luminaire_w = st.number_input(
                    "Optical power per luminaire [W]", 0.0, 200.0,
                    float(bg.led.optical_power_per_luminaire_w), 0.1, format="%.4g")
            bg.led.semiangle_deg = st.slider("Luminaire semi-angle [deg]", 5.0, 89.0,
                                             float(bg.led.semiangle_deg), 1.0)
            bg.led.include_reflections = st.checkbox(
                "Include diffuse wall/ceiling reflections", value=bool(bg.led.include_reflections))

        st.markdown("**B. Fluorescent lighting**")
        bg.fluorescent.enabled = st.checkbox("Enable fluorescent lamps",
                                             value=bool(bg.fluorescent.enabled))
        if bg.fluorescent.enabled:
            c1, c2 = st.columns(2)
            bg.fluorescent.n_luminaires = c1.number_input("Lamps", 1, 32,
                                                          int(bg.fluorescent.n_luminaires), 1)
            bg.fluorescent.optical_power_per_luminaire_w = c2.number_input(
                "W_opt per lamp", 0.0, 200.0,
                float(bg.fluorescent.optical_power_per_luminaire_w), 0.5)
            st.caption("Spectrum = Hg lines at "
                       + ", ".join(f"{x:g}" for x in C.FLUORESCENT_LINES_NM)
                       + " nm plus a phosphor continuum. The "
                       f"{C.FLUORESCENT_MODULATION_HZ:.0f} Hz intensity "
                       "modulation is not time-resolved (mean rate only).")

        st.markdown("**C. Sunlight through a window**")
        bg.sunlight.enabled = st.checkbox("Enable daylight", value=bool(bg.sunlight.enabled))
        if bg.sunlight.enabled:
            c1, c2 = st.columns(2)
            bg.sunlight.outdoor_irradiance_w_m2 = c1.number_input(
                "Outdoor irradiance [W/m²]", 0.0, 1400.0,
                float(bg.sunlight.outdoor_irradiance_w_m2), 25.0)
            bg.sunlight.window_area_m2 = c2.number_input(
                "Window area [m²]", 0.0, 50.0, float(bg.sunlight.window_area_m2), 0.25)
            c1, c2 = st.columns(2)
            bg.sunlight.window_transmission = c1.slider("Window transmission", 0.0, 1.0,
                                                        float(bg.sunlight.window_transmission), 0.01)
            bg.sunlight.window_distance_m = c2.number_input(
                "Window → Rx [m]", 0.1, 60.0, float(bg.sunlight.window_distance_m), 0.1)
            bg.sunlight.direct_sun_in_fov = st.checkbox(
                "Worst case: solar disc inside the FOV",
                value=bool(bg.sunlight.direct_sun_in_fov))

        st.markdown("**D. User-defined background**")
        bg.user.enabled = st.checkbox("Add a user-defined background term",
                                      value=bool(bg.user.enabled))
        if bg.user.enabled:
            bg.user.mode = st.selectbox(
                "Specify as", ["Background count rate [counts/s]",
                               "Background photon flux [photons/s]",
                               "Background optical power [W]",
                               "Spectral irradiance [W/m^2/nm]"],
                index=["Background count rate [counts/s]",
                       "Background photon flux [photons/s]",
                       "Background optical power [W]",
                       "Spectral irradiance [W/m^2/nm]"].index(bg.user.mode))
            bg.user.value = st.number_input("Value", 0.0, 1e20, float(bg.user.value),
                                            format="%.6g")

        st.markdown("**E. Isotropic literature preset**")
        bg.use_isotropic_preset = st.checkbox("Use a flat spectral-irradiance value",
                                              value=bool(bg.use_isotropic_preset))
        if bg.use_isotropic_preset:
            bg.isotropic_preset = st.selectbox(
                "Preset", list(C.SPECTRAL_IRRADIANCE_PRESETS_W_M2_NM.keys()),
                index=list(C.SPECTRAL_IRRADIANCE_PRESETS_W_M2_NM.keys()).index(
                    bg.isotropic_preset)
                if bg.isotropic_preset in C.SPECTRAL_IRRADIANCE_PRESETS_W_M2_NM else 2)
            default_v = C.SPECTRAL_IRRADIANCE_PRESETS_W_M2_NM[bg.isotropic_preset]
            bg.isotropic_spectral_irradiance_w_m2_nm = st.number_input(
                "p_n [W m⁻² nm⁻¹]", 0.0, 1.0, float(default_v), format="%.6g",
                key=f"iso_{bg.isotropic_preset}")

    # ---------------- detector ----------------
    with sb.expander("🎯 Single-photon detector", expanded=True):
        d = p.detector
        d.efficiency = st.slider("Detection efficiency η_det", 0.01, 1.0,
                                 float(d.efficiency), 0.01)
        d.dark_count_rate_hz = st.number_input("Dark count rate [counts/s]", 0.0, 1e8,
                                               float(d.dark_count_rate_hz), 10.0,
                                               format="%.6g")
        c1, c2 = st.columns(2)
        d.gate_width_ns = c1.number_input("Gate width Δt [ns]", 0.001, 1000.0,
                                          float(d.gate_width_ns), 0.05, format="%.4g")
        d.dead_time_ns = c2.number_input("Dead time [ns]", 0.0, 10000.0,
                                         float(d.dead_time_ns), 1.0, format="%.4g")
        c1, c2 = st.columns(2)
        d.afterpulse_probability = c1.number_input("Afterpulse probability", 0.0, 0.5,
                                                   float(d.afterpulse_probability), 0.001,
                                                   format="%.4f")
        d.timing_jitter_ps = c2.number_input("Timing jitter [ps]", 0.0, 10000.0,
                                             float(d.timing_jitter_ps), 10.0)
        d.operation_mode = st.radio("Operation mode", ["Gated", "Free-running"],
                                    index=0 if d.operation_mode == "Gated" else 1,
                                    horizontal=True,
                                    help="Free-running integrates the background "
                                         "over the whole pulse period instead of "
                                         "the gate.")
        d.n_detectors_per_basis = st.number_input("Detectors per basis", 1, 4,
                                                  int(d.n_detectors_per_basis), 1)

    # ---------------- polarisation ----------------
    with sb.expander("🧭 Polarisation encoding", expanded=False):
        pol = p.polarization
        pol.misalignment_angle_deg = st.slider("Frame misalignment θ [deg]", 0.0, 45.0,
                                               float(pol.misalignment_angle_deg), 0.1)
        pol.extinction_ratio_db = st.slider("Polariser extinction ratio [dB]", 5.0, 50.0,
                                            float(pol.extinction_ratio_db), 0.5)
        pol.depolarization = st.slider("Channel depolarisation fraction", 0.0, 0.2,
                                       float(pol.depolarization), 0.001)
        use_ov = st.checkbox("Override with a direct intrinsic error e_d",
                             value=pol.intrinsic_error_override is not None)
        if use_ov:
            pol.intrinsic_error_override = st.slider(
                "e_d", 0.0, 0.5,
                float(pol.intrinsic_error_override or 0.01), 0.001)
        else:
            pol.intrinsic_error_override = None
        st.caption("e_d = 1 − (1−sin²θ)(1−e_ER)(1−p_dep/2)")

    # ---------------- protocol ----------------
    with sb.expander("🛡️ BB84 protocol & security model", expanded=True):
        pr = p.protocol
        pr.protocol = st.selectbox(
            "Protocol / security model",
            ["Decoy-state BB84", "Standard BB84 (GLLP)", "Infinite-decoy limit"],
            index=["Decoy-state BB84", "Standard BB84 (GLLP)",
                   "Infinite-decoy limit"].index(pr.protocol)
            if pr.protocol in ("Decoy-state BB84", "Standard BB84 (GLLP)",
                               "Infinite-decoy limit") else 0)
        pr.use_efficient_bb84 = st.checkbox("Biased / efficient BB84 (q = p_Z²+(1−p_Z)²)",
                                            value=bool(pr.use_efficient_bb84))
        pr.basis_bias_z = st.slider("Z-basis probability p_Z", 0.05, 0.95,
                                    float(pr.basis_bias_z), 0.01)
        pr.error_correction_efficiency = st.slider("Error-correction efficiency f_EC",
                                                   1.0, 2.5,
                                                   float(pr.error_correction_efficiency), 0.01)
        pr.external_qber_enabled = st.checkbox(
            "Use an external (measured) QBER instead of the model",
            value=bool(pr.external_qber_enabled))
        if pr.external_qber_enabled:
            pr.external_qber = st.slider("External QBER", 0.0, 0.5,
                                         float(pr.external_qber), 0.001)
        pr.finite_key = st.checkbox("Enable simplified finite-key analysis",
                                    value=bool(pr.finite_key))
        if pr.finite_key:
            e = st.slider("log₁₀ block size N [pulses]", 4.0, 14.0,
                          float(np.log10(max(pr.block_size_bits, 1e4))), 0.5)
            pr.block_size_bits = float(10 ** e)
            c1, c2 = st.columns(2)
            pr.epsilon_sec = float(10 ** c1.slider("log₁₀ ε_sec", -20.0, -3.0,
                                                   float(np.log10(pr.epsilon_sec)), 1.0))
            pr.epsilon_cor = float(10 ** c2.slider("log₁₀ ε_cor", -20.0, -3.0,
                                                   float(np.log10(pr.epsilon_cor)), 1.0))

    # ---------------- monte carlo ----------------
    with sb.expander("🎲 Monte-Carlo settings", expanded=False):
        m = p.monte_carlo
        choice = st.select_slider(
            "Number of transmitted pulses",
            options=[10_000, 100_000, 1_000_000, 10_000_000, -1],
            value=m.n_pulses if m.n_pulses in (10_000, 100_000, 1_000_000,
                                               10_000_000) else -1,
            format_func=lambda v: "user-defined" if v == -1 else f"{v:,}")
        if choice == -1:
            m.n_pulses = int(st.number_input("Custom pulse count", 1000, 200_000_000,
                                             int(max(m.n_pulses, 1000)), 10_000))
        else:
            m.n_pulses = int(choice)
        m.seed = int(st.number_input("Random seed", 0, 2**31 - 1, int(m.seed), 1))
        m.convergence_points = int(st.slider("Convergence sample points", 5, 60,
                                             int(m.convergence_points), 1))

    sb.markdown("---")
    sb.markdown(f"<span class='small'>Self-test: "
                f"{'✅' if N_PASS == N_TEST else '⚠️'} {N_PASS}/{N_TEST} passed"
                f"</span>", unsafe_allow_html=True)
    sb.caption(C.DISCLAIMER)
    return p


P = sidebar()

# ==========================================================================
#  Evaluate the model
# ==========================================================================
issues = validate(P)
errors = [i for i in issues if i.severity == "error"]

try:
    R = run_analytic(P)
    model_ok = True
    model_error = ""
except Exception as exc:
    R = None
    model_ok = False
    model_error = repr(exc)

g = R.channel.geometry if R is not None else None


# ==========================================================================
#  Header
# ==========================================================================
chain = ["Indoor room", "VLC channel", "Signal photons", "Background photons",
         "Detector clicks", "BB84 sifting", "QBER", "Mutual information",
         "Privacy amplification", "Secret key"]
chips = "".join(f"<span class='chain on'>{c}</span>" for c in chain)
st.markdown(
    f"""<div class='hero'>
      <h1>Indoor Visible-Light Quantum Key Distribution — BB84 Simulator</h1>
      <p class='sub'>Modular, transparent, reproducible modelling of a
      polarisation-encoded BB84 link through an indoor visible-light channel,
      with a physically built-up model of background optical noise from room
      illumination.</p>
      <div class='chainrow'>{chips}</div>
    </div>""", unsafe_allow_html=True)

if errors:
    st.markdown("<div class='banner bad'><div><span class='big'>"
                "Parameter errors must be fixed before the results are meaningful"
                "</span><br>"
                + "<br>".join(f"• <b>{i.field}</b>: {i.message}" for i in errors)
                + "</div></div>", unsafe_allow_html=True)

if not model_ok:
    st.error(f"The model could not be evaluated with the current parameters: "
             f"{model_error}")
    st.stop()


# ==========================================================================
#  TABS
# ==========================================================================
TABS = st.tabs([
    "📊 Overview", "🏠 Room", "🔆 Background & spectrum", "📉 QBER & information",
    "📈 Standard plots", "🌡️ Research heat-maps", "🔬 Illumination comparison",
    "🎚️ Filter optimisation", "🎲 Monte Carlo", "🧪 Parameter sweep",
    "🔒 Finite key", "📤 Export & report", "📚 Model & self-test",
])

# --------------------------------------------------------------------------
#  TAB 1 - OVERVIEW DASHBOARD
# --------------------------------------------------------------------------
with TABS[0]:
    secure = R.key.secure
    if P.protocol.external_qber_enabled:
        cls, big, detail = "warn", "External QBER mode", (
            "The physically modelled QBER has been replaced by the value you "
            "entered. Channel and background results below are still modelled.")
    elif secure:
        cls, big, detail = "ok", "Positive asymptotic key rate", (
            f"R = {R.secret_key_rate_bps:,.1f} bit/s at QBER = "
            f"{100*R.qber.qber_total:.3f}% under the "
            f"<b>{R.key.protocol}</b> model. This is a simulation result for the "
            f"stated model and assumptions — not a claim of experimental "
            f"security.")
    else:
        cls, big, detail = "bad", "No positive asymptotic key rate under the selected model", (
            f"At QBER = {100*R.qber.qber_total:.3f}% the error-correction and "
            f"privacy-amplification costs exceed the sifted key. "
            f"{R.key.message}")
    st.markdown(f"<div class='banner {cls}'><div><span class='big'>{big}</span>"
                f"<br>{detail}</div></div>", unsafe_allow_html=True)

    st.markdown("#### Quantum key distribution")
    kpi_grid([
        kpi("Secret key rate",
            fmt(R.secret_key_rate_bps, unit=" bit/s"),
            f"{R.key.protocol} · asymptotic",
            GREEN if secure else ROSE),
        kpi("QBER", f"{100*R.qber.qber_total:.3f}<span class='unit'>%</span>",
            f"threshold for one-way BB84 ≈ 11%", ROSE),
        kpi("Sifted key rate", fmt(R.sifted_key_rate_bps, unit=" bit/s"),
            f"sifting factor q = {R.key.q_sift:.3f}", BLUE),
        kpi("Secret key fraction",
            f"{R.secret_key_fraction:.4f}<span class='unit'> bit/sifted bit</span>",
            "after EC and PA", TEAL),
    ], 4)
    kpi_grid([
        kpi("Detection probability Q<sub>µ</sub>", fmt(R.qber.gain_q_mu),
            "clicks per transmitted pulse", VIOLET),
        kpi("Mutual information I(A:B)",
            f"{R.info.mutual_information_per_bit:.4f}"
            f"<span class='unit'> bit/sifted bit</span>",
            f"= {fmt(R.info.mutual_information_rate_bps, unit=' bit/s')} rate", VIOLET),
        kpi("Eve information / PA term",
            f"{R.info.eve_information_gllp:.4f}"
            f"<span class='unit'> bit/sifted bit</span>",
            "1 − (Q₁/Q_µ)[1 − h₂(e₁)]", ROSE),
        kpi("Single-photon error e₁", f"{100*R.key.e1:.3f}<span class='unit'>%</span>",
            f"Q₁ = {fmt(R.key.q1)}", AMBER),
    ], 4)

    st.markdown("#### Channel")
    g = R.channel.geometry
    kpi_grid([
        kpi("Link distance", f"{g.distance_m:.3f}<span class='unit'> m</span>",
            f"φ = {g.irradiance_angle_deg:.2f}° , ψ = {g.incidence_angle_deg:.2f}°",
            BLUE),
        kpi("Channel gain H(0)", fmt(R.channel.channel_gain_h0),
            R.channel.model, BLUE),
        kpi("Channel transmittance", fmt(R.channel.eta_channel),
            f"{R.channel.eta_channel_db:.2f} dB loss", BLUE),
        kpi("Received signal photons", fmt(R.received_photons_per_pulse),
            f"per pulse · η_sys = {fmt(R.eta_system)}", TEAL),
    ], 4)

    st.markdown("#### Background & noise")
    kpi_grid([
        kpi("Background count rate", fmt(R.background_count_rate_hz, unit=" counts/s"),
            f"µ_bg = {fmt(R.detector.mu_bg)} per gate", AMBER),
        kpi("Dark count rate", fmt(R.dark_count_rate_hz, unit=" counts/s"),
            f"µ_dark = {fmt(R.detector.mu_dark)} per gate", VIOLET),
        kpi("Signal-to-background ratio", fmt(R.signal_to_background_ratio),
            "P_signal-click / total background per gate", AMBER),
        kpi("Signal-to-noise ratio", fmt(R.signal_to_noise_ratio),
            "background + dark + afterpulse", AMBER),
    ], 4)
    kpi_grid([
        kpi("Total count rate", fmt(R.total_count_rate_hz, unit=" counts/s"),
            "signal + background + dark, after dead time", SLATE),
        kpi("Signal count rate", fmt(R.signal_count_rate_hz, unit=" counts/s"),
            "signal-induced clicks only", TEAL),
        kpi("Vacuum yield Y₀", fmt(R.detector.y0_vacuum_yield),
            "1 − exp(−N_det µ_noise)", VIOLET),
        kpi("Effective filter bandwidth",
            f"{R.background.effective_filter_bandwidth_nm:.4g}<span class='unit'> nm</span>",
            f"T at signal = {R.background.filter_transmission_at_signal:.3f}", LIME),
    ], 4)

    st.markdown("---")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("##### Where each sifted bit goes")
        st.plotly_chart(pl.information_waterfall(R), use_container_width=True,
                        key="ov_waterfall")
    with c2:
        st.markdown("##### QBER error budget")
        st.plotly_chart(pl.qber_budget_donut(R.qber), use_container_width=True,
                        key="ov_donut")

    if R.warnings:
        st.markdown("##### Model warnings")
        for w in R.warnings:
            st.warning(w)

    nonerr = [i for i in issues if i.severity != "error"]
    if nonerr:
        with st.expander(f"Parameter checks ({len(nonerr)} notes)"):
            for i in nonerr:
                (st.warning if i.severity == "warning" else st.info)(
                    f"**{i.field}** — {i.message}")

# --------------------------------------------------------------------------
#  TAB 2 - ROOM
# --------------------------------------------------------------------------
with TABS[1]:
    st.markdown("#### Indoor scene")
    st.caption("Alice, Bob, the ceiling luminaires, the propagation path, the "
               "beam cone and the receiver field of view. Move the terminals "
               "from the sidebar and every number in the toolkit updates.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(roomviz.top_view(P), use_container_width=True, key="room_top")
    with c2:
        st.plotly_chart(roomviz.three_d_view(P), use_container_width=True, key="room_3d")

    kpi_grid([
        kpi("Distance", f"{g.distance_m:.3f}<span class='unit'> m</span>",
            f"horizontal {g.horizontal_separation_m:.2f} m, "
            f"vertical {g.vertical_separation_m:.2f} m", BLUE),
        kpi("Channel gain", fmt(R.channel.channel_gain_h0),
            "H(0), power fraction collected", BLUE),
        kpi("Received optical power", fmt(R.received_optical_power_w, unit=" W"),
            f"{fmt(R.received_photon_rate_hz, unit=' photons/s')}", TEAL),
        kpi("Inside FOV?", "yes" if g.in_fov else "no",
            f"ψ = {g.incidence_angle_deg:.2f}° vs Ψ_c = {P.channel.rx_fov_deg:.1f}°",
            GREEN if g.in_fov else ROSE),
    ], 4)

    st.markdown("---")
    st.markdown("#### Where can Bob stand?")
    st.caption("Sweeps Bob's position across the room at his current height and "
               "maps the selected quantity. This makes the coupled effect of "
               "distance, incidence angle, FOV and background geometry visible "
               "in one picture.")
    c1, c2, c3 = st.columns([2, 1, 1])
    metric = c1.selectbox("Quantity to map", [
        "Secret key rate [bit/s]", "QBER", "Channel gain H(0)",
        "Sifted key rate [bit/s]", "Background count rate [1/s]",
        "Mutual information I(A:B) [bit/sifted bit]",
        "Received photons / pulse"], key="posmetric")
    grid_n = c2.slider("Grid resolution", 10, 44, 24, 2, key="posn")
    if c3.button("Compute position map", use_container_width=True):
        with st.spinner(f"Evaluating {grid_n*grid_n} receiver positions…"):
            xs, ys, Z = roomviz.receiver_position_map(P, metric, grid_n)
            st.session_state.posmap = (xs, ys, Z, metric)
    if st.session_state.posmap:
        xs, ys, Z, mname = st.session_state.posmap
        logc = mname not in ("QBER", "Mutual information I(A:B) [bit/sifted bit]")
        st.plotly_chart(
            hm.position_heatmap(xs, ys, Z, P.room, zlabel=mname,
                                log_color=logc,
                                tx=(P.room.tx_x, P.room.tx_y),
                                rx=(P.room.rx_x, P.room.rx_y)),
            use_container_width=True, key="posmap_fig")

# --------------------------------------------------------------------------
#  TAB 3 - BACKGROUND & SPECTRUM
# --------------------------------------------------------------------------
with TABS[2]:
    st.markdown("#### Background optical noise")
    st.markdown(
        "<div class='panel'><h4>How the background is built "
        "<span class='prov'>analytic</span>"
        "<span class='prov'>simplified background model</span></h4>"
        "<div class='eq'>P_bg = ∫ p(λ) · A · T_f(λ) · G_coll dλ    "
        "≈  p(λ₀) · A · Δλ · T_filter · G_coll\n"
        "N_bg = P_bg · η_opt · η_det · Δt / E_ph(λ₀)      [counts per gate]\n"
        "µ_bg = R_bg · Δt ,   P_bg(0) = e^(−µ_bg) ,   P_bg(click) = 1 − e^(−µ_bg)\n"
        "µ_total_noise = µ_bg + µ_dark (+ µ_afterpulse)</div>"
        "<br><b>p(λ)</b> spectral irradiance at the aperture [W m⁻² nm⁻¹] · "
        "<b>A</b> aperture area [m²] · <b>T_f</b> filter transmission · "
        "<b>G_coll</b> background collection factor [sr] · "
        "<b>Δλ</b> noise-equivalent filter bandwidth [nm] · "
        "<b>Δt</b> detection gate [s] · <b>E_ph</b> photon energy [J]."
        "</div>", unsafe_allow_html=True)

    kpi_grid([
        kpi("Total background count rate",
            fmt(R.background_count_rate_hz, unit=" counts/s"),
            "per detector, after η_opt·η_det", AMBER),
        kpi("Background per gate µ_bg", fmt(R.detector.mu_bg),
            f"Δt = {R.background.integration_time_s*1e9:.4g} ns "
            f"({P.detector.operation_mode})", AMBER),
        kpi("In-band optical power",
            fmt(R.background.total_optical_power_w, unit=" W"),
            f"{fmt(R.background.total_photon_flux_hz, unit=' photons/s')}", AMBER),
        kpi("Collection factor G_coll",
            f"{R.background.collection_factor_sr:.4g}<span class='unit'> sr</span>",
            "π n² (concentrator) or π sin²Ψ_c (bare)", LIME),
    ], 4)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("##### Background spectrum vs. the optical filter")
        st.plotly_chart(pl.spectrum_plot(R), use_container_width=True, key="spec")
        st.caption("This single figure answers *'how narrow should the filter "
                   "be?'* qualitatively: the amber area inside the teal curve is "
                   "what reaches the detector.")
    with c2:
        st.markdown("##### Contribution by source")
        st.plotly_chart(pl.background_breakdown_bar(R), use_container_width=True,
                        key="bgbar")
        st.dataframe(ex.background_to_dataframe(R.background),
                     use_container_width=True, hide_index=True, height=240)

    st.markdown("---")
    st.markdown("#### Filter sweep: how filtering suppresses background")
    c1, c2, c3, c4 = st.columns(4)
    bw_lo = c1.number_input("Δλ min [nm]", 0.001, 100.0, 0.01, format="%.4g", key="fs1")
    bw_hi = c2.number_input("Δλ max [nm]", 0.01, 500.0, 100.0, format="%.4g", key="fs2")
    npts = c3.slider("Points", 10, 160, 70, key="fs3")
    go_fs = c4.button("Run filter sweep", use_container_width=True, key="fs4")
    if go_fs:
        with st.spinner("Sweeping filter bandwidth…"):
            st.session_state["filt_sweep"] = sweepmod.sweep_1d(
                P, "Filter bandwidth", bw_lo, bw_hi, npts, True,
                metrics=["Background count rate [1/s]", "QBER",
                         "Secret key rate [bit/s]", "Sifted key rate [bit/s]",
                         "Signal-to-background ratio",
                         "Filter transmission at signal"])
    if st.session_state.get("filt_sweep") is not None:
        df = st.session_state["filt_sweep"]
        c1, c2 = st.columns(2)
        c1.plotly_chart(pl.sweep_plot(df, "Background count rate [1/s]",
                                      title="Plot: background count rate vs filter bandwidth",
                                      log_x=True, log_y=True, color=AMBER),
                        use_container_width=True, key="fsw1")
        c2.plotly_chart(pl.sweep_plot(df, "Signal-to-background ratio",
                                      title="Signal-to-background ratio vs filter bandwidth",
                                      log_x=True, log_y=True, color=TEAL),
                        use_container_width=True, key="fsw2")
        c1, c2 = st.columns(2)
        c1.plotly_chart(pl.sweep_plot(df, "QBER", title="Plot 9: QBER vs filter bandwidth",
                                      log_x=True, color=ROSE, threshold=0.11,
                                      threshold_label="11% BB84 threshold"),
                        use_container_width=True, key="fsw3")
        c2.plotly_chart(pl.sweep_plot(df, "Secret key rate [bit/s]",
                                      title="Plot 8: secret key rate vs filter bandwidth",
                                      log_x=True, log_y=True, color=GREEN,
                                      shade_secure=True),
                        use_container_width=True, key="fsw4")

# --------------------------------------------------------------------------
#  TAB 4 - QBER & INFORMATION
# --------------------------------------------------------------------------
with TABS[3]:
    st.markdown("#### QBER: where the errors come from")
    st.markdown(
        "<div class='panel'><h4>Model <span class='prov'>analytic</span></h4>"
        "<div class='eq'>"
        "p_s = 1 − exp(−η_sys µ)          probability of ≥1 signal click\n"
        "Y₀  = 1 − exp(−N_det µ_noise)    probability of ≥1 noise click\n"
        "Q_µ = 1 − (1 − Y₀) exp(−η_sys µ)\n\n"
        "E_µ Q_µ =  p_s(1−Y₀)·e_d                signal-only click, wrong detector\n"
        "        + (1−p_s)Y₀·½                    noise-only click, random bit\n"
        "        + p_s·Y₀·(e_d/2 + ¼)             double click, random assignment\n\n"
        "QBER = E_µ</div>"
        "Noise clicks carry a random bit, so ~half of them are errors. "
        "The budget below is a genuine decomposition: the contributions sum "
        "exactly to the total.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.25, 1])
    with c1:
        st.plotly_chart(pl.qber_budget_bar(R.qber), use_container_width=True,
                        key="qb_bar")
        rows = [{"Mechanism": n, "Contribution": f"{v:.6e}", "Percent": f"{pc:.4f} %"}
                for n, v, pc in R.qber.as_percentage_table()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)
    with c2:
        kpi_grid([
            kpi("Total QBER", f"{100*R.qber.qber_total:.4f}<span class='unit'>%</span>",
                f"textbook first-order form: {100*R.qber.textbook_qber:.4f}%", ROSE),
            kpi("Intrinsic error e_d", f"{100*R.qber.e_intrinsic:.4f}<span class='unit'>%</span>",
                f"misalignment {100*R.polarization.e_misalignment:.4f}% · "
                f"extinction {100*R.polarization.e_extinction:.4f}% · "
                f"depol {100*R.polarization.e_depolarization:.4f}%", BLUE),
        ], 1)
        st.plotly_chart(pl.qber_budget_donut(R.qber, height=280),
                        use_container_width=True, key="qb_donut")

    st.markdown("---")
    st.markdown("#### Information-theoretic accounting")
    kpi_grid([
        kpi("I(A:B) per sifted bit", f"{R.info.mutual_information_per_bit:.5f}",
            "1 − h₂(QBER)", VIOLET),
        kpi("I(A:B) rate", fmt(R.info.mutual_information_rate_bps, unit=" bit/s"),
            "R_sift × [1 − h₂(QBER)]", VIOLET),
        kpi("Error-correction leakage",
            f"{R.info.error_correction_leakage_per_bit:.5f}"
            f"<span class='unit'> bit/sifted bit</span>",
            f"f_EC·h₂(E) with f_EC = {P.protocol.error_correction_efficiency:g}",
            AMBER),
        kpi("Privacy amplification (Eve)",
            f"{R.info.eve_information_gllp:.5f}"
            f"<span class='unit'> bit/sifted bit</span>",
            "GLLP / decoy: multi-photons conceded in full", ROSE),
    ], 4)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(pl.mutual_information_vs_qber(
            P.protocol.error_correction_efficiency),
            use_container_width=True, key="mi_vs_qber")
        st.caption("Plot 15 companion: the information balance as a function of "
                   "QBER for an ideal single-photon source. The green curve "
                   "crosses zero at the familiar ~11% one-way BB84 threshold.")
    with c2:
        st.plotly_chart(pl.information_waterfall(R, height=420),
                        use_container_width=True, key="info_wf")
        st.markdown(
            f"<div class='small'>Reference (didactic only): for a symmetric "
            f"individual attack Eve's information would be h₂(E) = "
            f"{R.info.eve_information_individual:.5f} bit/sifted bit. The GLLP "
            f"/ decoy number above is the one actually used for the key rate "
            f"and is more conservative because it concedes every multi-photon "
            f"detection to Eve.</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### BB84 protocol steps as configured")
    st.dataframe(pd.DataFrame(protocol_step_table(P),
                              columns=["#", "Step", "As configured"]),
                 use_container_width=True, hide_index=True, height=460)

    st.markdown("##### Sifting bookkeeping (analytic expectation)")
    rp = P.source.pulse_rate_hz
    pz = P.protocol.basis_bias_z
    sift_tbl = pd.DataFrame([
        {"Quantity": "Transmitted pulses per second", "Value": f"{rp:,.4g}"},
        {"Quantity": "Z-basis pulses per second", "Value": f"{rp*pz:,.4g}"},
        {"Quantity": "X-basis pulses per second", "Value": f"{rp*(1-pz):,.4g}"},
        {"Quantity": "Detections per second (Q_µ·r_p)",
         "Value": f"{R.total_count_rate_hz:,.4g}"},
        {"Quantity": "Basis-sifting factor q", "Value": f"{R.key.q_sift:.5f}"},
        {"Quantity": "Matched-basis detections per second",
         "Value": f"{R.sifted_key_rate_bps:,.4g}"},
        {"Quantity": "Sifted key bits per second", "Value": f"{R.sifted_key_rate_bps:,.4g}"},
        {"Quantity": "Secret key bits per second", "Value": f"{R.secret_key_rate_bps:,.4g}"},
    ])
    st.dataframe(sift_tbl, use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
#  TAB 5 - STANDARD PLOTS (the 15 required figures)
# --------------------------------------------------------------------------
with TABS[4]:
    st.markdown("#### The standard plot set")
    st.caption("Every curve is produced by the analytic model with the current "
               "sidebar parameters; only the swept variable changes along each "
               "x-axis.")

    c1, c2, c3 = st.columns([1, 1, 1])
    n_pts = c1.slider("Points per curve", 20, 200, 70, 5, key="sp_n")
    d_lo, d_hi = c2.slider("Distance range [m]", 0.2, 40.0, (0.3, 12.0), 0.1,
                           key="sp_d")
    b_lo_e, b_hi_e = c3.slider("log₁₀ background count rate [counts/s]",
                               -2.0, 12.0, (0.0, 10.0), 0.5, key="sp_b")
    c1, c2, c3 = st.columns([1, 1, 1])
    mu_lo, mu_hi = c1.slider("log₁₀ µ", -4.0, 0.7, (-3.0, 0.0), 0.1, key="sp_mu")
    lam_lo, lam_hi = c2.slider("Wavelength range [nm]", 380.0, 1000.0,
                               (400.0, 900.0), 5.0, key="sp_l")
    run_all = c3.button("▶ Generate all 15 plots", use_container_width=True,
                        type="primary", key="sp_go")

    if run_all or st.session_state.get("plots15") is not None:
        if run_all:
            prog = st.progress(0.0, "Computing sweeps…")
            sw = {}
            steps = [
                ("dist", "Distance", d_lo, d_hi, False),
                ("bg", "Background count rate", 10 ** b_lo_e, 10 ** b_hi_e, True),
                ("eta", "Detector efficiency", 0.01, 0.99, False),
                ("bw", "Filter bandwidth", 0.01, 100.0, True),
                ("mu", "Mean photon number mu", 10 ** mu_lo, 10 ** mu_hi, True),
                ("lam", "Wavelength (filter follows)", lam_lo, lam_hi, False),
                ("lamfix", "Wavelength (filter fixed)", lam_lo, lam_hi, False),
            ]
            for i, (key, var, lo, hi, lg) in enumerate(steps):
                prog.progress((i) / len(steps), f"Sweeping {var}…")
                sw[key] = sweepmod.sweep_1d(P, var, lo, hi, n_pts, lg)
            prog.progress(1.0, "Done")
            prog.empty()
            st.session_state["plots15"] = sw
        sw = st.session_state["plots15"]

        def two(figs, keys):
            cols = st.columns(2)
            for col, f, k in zip(cols, figs, keys):
                col.plotly_chart(f, use_container_width=True, key=k)

        st.markdown("##### Distance")
        two([pl.sweep_plot(sw["dist"], "Secret key rate [bit/s]",
                           title="Plot 1 — Secret key rate vs distance",
                           log_y=True, color=GREEN, shade_secure=True),
             pl.sweep_plot(sw["dist"], "QBER",
                           title="Plot 2 — QBER vs distance", color=ROSE,
                           threshold=0.11, threshold_label="11% threshold")],
            ["p1", "p2"])
        two([pl.sweep_plot(sw["dist"], "Mutual information I(A:B) [bit/sifted bit]",
                           title="Plot 3 — Mutual information vs distance",
                           color=VIOLET),
             pl.sweep_plot(sw["dist"], "Signal-to-background ratio",
                           title="Plot 14 — Signal-to-background ratio vs distance",
                           log_y=True, color=AMBER)],
            ["p3", "p14"])

        st.markdown("##### Background count rate")
        two([pl.sweep_plot(sw["bg"], "Secret key rate [bit/s]",
                           title="Plot 4 — Secret key rate vs background count rate",
                           log_x=True, log_y=True, color=GREEN, shade_secure=True),
             pl.sweep_plot(sw["bg"], "QBER",
                           title="Plot 5 — QBER vs background count rate",
                           log_x=True, color=ROSE, threshold=0.11,
                           threshold_label="11% threshold")],
            ["p4", "p5"])
        two([pl.sweep_plot(sw["bg"], "Mutual information I(A:B) [bit/sifted bit]",
                           title="Plot 6 — Mutual information vs background count rate",
                           log_x=True, color=VIOLET),
             pl.sweep_plot(sw["eta"], "Secret key rate [bit/s]",
                           title="Plot 7 — Secret key rate vs detector efficiency",
                           log_y=True, color=GREEN, shade_secure=True)],
            ["p6", "p7"])

        st.markdown("##### Optical filter bandwidth")
        two([pl.sweep_plot(sw["bw"], "Secret key rate [bit/s]",
                           title="Plot 8 — Secret key rate vs filter bandwidth",
                           log_x=True, log_y=True, color=GREEN, shade_secure=True),
             pl.sweep_plot(sw["bw"], "QBER",
                           title="Plot 9 — QBER vs filter bandwidth",
                           log_x=True, color=ROSE, threshold=0.11,
                           threshold_label="11% threshold")],
            ["p8", "p9"])

        st.markdown("##### Mean photon number")
        two([pl.sweep_plot(sw["mu"], "Secret key rate [bit/s]",
                           title="Plot 10 — Secret key rate vs mean photon number µ",
                           log_x=True, log_y=True, color=GREEN, shade_secure=True),
             pl.sweep_plot(sw["mu"], "QBER",
                           title="Plot 11 — QBER vs mean photon number µ",
                           log_x=True, color=ROSE)],
            ["p10", "p11"])

        st.markdown("##### Wavelength")
        st.caption("Two variants are shown: the filter re-centred on the signal "
                   "at every wavelength (the fair comparison), and the filter "
                   "held fixed (what happens if the source drifts).")
        two([pl.sweep_plot(sw["lam"], "Secret key rate [bit/s]",
                           title="Plot 12 — Secret key rate vs wavelength (filter follows)",
                           log_y=True, color=GREEN, shade_secure=True),
             pl.sweep_plot(sw["lam"], "Background count rate [1/s]",
                           title="Plot 13 — Background count rate vs wavelength",
                           log_y=True, color=AMBER)],
            ["p12", "p13"])
        two([pl.sweep_plot(sw["lamfix"], "Secret key rate [bit/s]",
                           title="Plot 12b — Secret key rate vs wavelength (filter fixed)",
                           log_y=True, color=LIME),
             pl.multi_sweep_plot(sw["lam"],
                                 ["QBER: background", "QBER: dark counts",
                                  "QBER: polarisation"],
                                 title="QBER contributions vs wavelength",
                                 ytitle="absolute contribution to QBER",
                                 colors=[AMBER, VIOLET, BLUE])],
            ["p12b", "p13b"])

        st.markdown("##### Secret key fraction vs QBER")
        e_grid = np.linspace(0.0001, 0.2, 240)
        from qkd.information_metrics import binary_entropy
        fec = P.protocol.error_correction_efficiency
        frac_ideal = 1 - binary_entropy(e_grid) - fec * binary_entropy(e_grid)
        frac_model = []
        for e in e_grid:
            q1, qmu = R.key.q1, R.key.gain_q_mu
            pa = 1 - (min(q1 / qmu, 1.0) if qmu > 0 else 0) * (1 - binary_entropy(e))
            frac_model.append((1 - pa) - fec * float(binary_entropy(e)))
        import plotly.graph_objects as go
        f15 = go.Figure()
        f15.add_trace(go.Scatter(x=100 * e_grid, y=frac_ideal,
                                 name="Ideal single-photon source",
                                 line=dict(color=TEAL, width=2.4)))
        f15.add_trace(go.Scatter(x=100 * e_grid, y=frac_model,
                                 name=f"Current link (Q₁/Q_µ = {min(R.key.q1/max(R.key.gain_q_mu,1e-30),1):.3f})",
                                 line=dict(color=GREEN, width=2.4)))
        f15.add_hline(y=0, line=dict(color="#3A4759", width=1))
        f15.add_vline(x=100 * R.qber.qber_total,
                      line=dict(color=ROSE, width=1.4, dash="dot"),
                      annotation_text=f"current QBER "
                                      f"{100*R.qber.qber_total:.2f}%",
                      annotation_font=dict(size=11, color=ROSE))
        from visualization.theme import style as _style
        _style(f15, "Plot 15 — Secret key fraction vs QBER", "QBER [%]",
               "bits per sifted bit", 430,
               provenance="analytic, asymptotic")
        f15.update_yaxes(range=[-0.6, 1.05])
        st.plotly_chart(f15, use_container_width=True, key="p15")

        with st.expander("Download the underlying data"):
            for k, df in sw.items():
                st.download_button(f"CSV — {k}", ex.dataframe_to_csv_bytes(df),
                                   file_name=f"vlqkd_sweep_{k}.csv",
                                   mime="text/csv", key=f"dl_{k}")
    else:
        st.info("Press **Generate all 15 plots** to compute the full standard "
                "plot set with the current parameters.")

# --------------------------------------------------------------------------
#  TAB 6 - RESEARCH HEATMAPS
# --------------------------------------------------------------------------
with TABS[5]:
    st.markdown("#### The headline research figure")
    st.caption("Secret key rate, QBER and mutual information as functions of "
               "**link distance** and **background count rate** — the two "
               "quantities an indoor deployment cannot choose freely.")

    c1, c2, c3 = st.columns(3)
    with c1:
        dx_lo, dx_hi = st.slider("Distance range [m]", 0.2, 40.0, (0.5, 12.0), 0.1,
                                 key="hm_d")
        nx = st.slider("Distance points", 8, 80, 38, key="hm_nx")
    with c2:
        by_lo, by_hi = st.slider("log₁₀ background count rate", -2.0, 12.0,
                                 (2.0, 11.0), 0.5, key="hm_b")
        ny = st.slider("Background points", 8, 80, 38, key="hm_ny")
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        go_hm = st.button("▶ Compute heat-maps", use_container_width=True,
                          type="primary", key="hm_go")
        st.caption(f"{nx}×{ny} = {nx*ny} model evaluations per map.")

    if go_hm:
        prog = st.progress(0.0, "Computing…")
        maps = {}
        for i, (name, metric) in enumerate([
                ("R", "Secret key rate [bit/s]"),
                ("E", "QBER"),
                ("I", "Mutual information I(A:B) [bit/sifted bit]")]):
            def cb(f, i=i, name=name):
                prog.progress((i + f) / 3.0, f"{metric}…")
            xs, ys, Z = sweepmod.sweep_2d(
                P, "Distance", dx_lo, dx_hi, nx, False,
                "Background count rate", 10 ** by_lo, 10 ** by_hi, ny, True,
                metric=metric, progress_callback=cb)
            maps[name] = (xs, ys, Z)
        prog.empty()
        st.session_state.heat = maps

    if st.session_state.heat:
        maps = st.session_state.heat
        xs, ys, Z = maps["R"]
        st.plotly_chart(hm.key_rate_heatmap(
            xs, ys, Z, xlabel="Link distance [m]",
            ylabel="Background count rate [counts/s]",
            title="Secret key rate R(distance, background count rate)",
            log_y=True, height=600),
            use_container_width=True, key="hm_R")
        st.caption("Rose region = no positive asymptotic key rate. The green "
                   "contour is the R = 0 boundary — the operating envelope of "
                   "this link under the present model.")

        c1, c2 = st.columns(2)
        xs, ys, Z = maps["E"]
        c1.plotly_chart(hm.metric_heatmap(
            xs, ys, Z, xlabel="Link distance [m]",
            ylabel="Background count rate [counts/s]", zlabel="QBER [%]",
            title="QBER(distance, background count rate)",
            log_y=True, percent=True, contour_at=11.0,
            contour_label="11% threshold", reverse=True, height=520),
            use_container_width=True, key="hm_E")
        xs, ys, Z = maps["I"]
        c2.plotly_chart(hm.metric_heatmap(
            xs, ys, Z, xlabel="Link distance [m]",
            ylabel="Background count rate [counts/s]",
            zlabel="I(A:B) [bit/sifted bit]",
            title="I(A:B)(distance, background count rate)",
            log_y=True, height=520),
            use_container_width=True, key="hm_I")

        with st.expander("Export heat-map data"):
            for name, label in (("R", "secret_key_rate"), ("E", "qber"),
                                ("I", "mutual_information")):
                xs, ys, Z = maps[name]
                df = pd.DataFrame(Z, index=pd.Index(ys, name="background_count_rate_hz"),
                                  columns=pd.Index(xs, name="distance_m"))
                st.download_button(f"CSV — {label}",
                                   df.to_csv().encode("utf-8"),
                                   file_name=f"vlqkd_heatmap_{label}.csv",
                                   mime="text/csv", key=f"hmdl_{name}")
    else:
        st.info("Press **Compute heat-maps** to build the 2-D maps.")

    st.markdown("---")
    st.markdown("#### Custom 2-D map")
    c1, c2, c3 = st.columns(3)
    vx = c1.selectbox("x variable", list(sweepmod.SWEEP_VARIABLES.keys()),
                      index=list(sweepmod.SWEEP_VARIABLES.keys()).index("Distance"),
                      key="cm_x")
    vy = c2.selectbox("y variable", list(sweepmod.SWEEP_VARIABLES.keys()),
                      index=list(sweepmod.SWEEP_VARIABLES.keys()).index("Filter bandwidth"),
                      key="cm_y")
    vm = c3.selectbox("metric", list(sweepmod.METRICS.keys()), index=0, key="cm_m")
    ax, ay = sweepmod.SWEEP_VARIABLES[vx], sweepmod.SWEEP_VARIABLES[vy]
    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.number_input(f"{ax.label} min", value=float(ax.default_min), format="%.6g", key="cmx0")
    x1 = c2.number_input(f"{ax.label} max", value=float(ax.default_max), format="%.6g", key="cmx1")
    y0 = c3.number_input(f"{ay.label} min", value=float(ay.default_min), format="%.6g", key="cmy0")
    y1 = c4.number_input(f"{ay.label} max", value=float(ay.default_max), format="%.6g", key="cmy1")
    c1, c2, c3, c4 = st.columns(4)
    lx = c1.checkbox("log x", value=ax.default_log, key="cmlx")
    ly = c2.checkbox("log y", value=ay.default_log, key="cmly")
    nn = c3.slider("Grid (n×n)", 8, 60, 28, key="cmn")
    if c4.button("Compute custom map", use_container_width=True, key="cmgo"):
        prog = st.progress(0.0)
        xs, ys, Z = sweepmod.sweep_2d(P, vx, x0, x1, nn, lx, vy, y0, y1, nn, ly,
                                      metric=vm,
                                      progress_callback=lambda f: prog.progress(f))
        prog.empty()
        st.session_state["custom_map"] = (xs, ys, Z, vx, vy, vm, lx, ly)
    if st.session_state.get("custom_map"):
        xs, ys, Z, vx_, vy_, vm_, lx_, ly_ = st.session_state["custom_map"]
        if "key rate" in vm_.lower():
            fig = hm.key_rate_heatmap(xs, ys, Z,
                                      xlabel=f"{sweepmod.SWEEP_VARIABLES[vx_].label} "
                                             f"[{sweepmod.SWEEP_VARIABLES[vx_].unit}]",
                                      ylabel=f"{sweepmod.SWEEP_VARIABLES[vy_].label} "
                                             f"[{sweepmod.SWEEP_VARIABLES[vy_].unit}]",
                                      title=vm_, log_x=lx_, log_y=ly_)
        else:
            fig = hm.metric_heatmap(xs, ys, Z,
                                    xlabel=f"{sweepmod.SWEEP_VARIABLES[vx_].label} "
                                           f"[{sweepmod.SWEEP_VARIABLES[vx_].unit}]",
                                    ylabel=f"{sweepmod.SWEEP_VARIABLES[vy_].label} "
                                           f"[{sweepmod.SWEEP_VARIABLES[vy_].unit}]",
                                    zlabel=vm_, title=vm_, log_x=lx_, log_y=ly_)
        st.plotly_chart(fig, use_container_width=True, key="custom_map_fig")

# --------------------------------------------------------------------------
#  TAB 7 - ILLUMINATION COMPARISON
# --------------------------------------------------------------------------
with TABS[6]:
    st.markdown("#### How indoor illumination changes BB84")
    st.caption("The same link, the same detector, the same filter — only the "
               "room lighting changes. Scenarios are defined in **lux** and "
               "converted to radiometric watts inside the model.")

    c1, c2, c3 = st.columns([2, 1, 1])
    sel = c1.multiselect("Scenarios", list(bgm.BACKGROUND_SCENARIOS.keys()),
                         default=list(bgm.BACKGROUND_SCENARIOS.keys()),
                         key="cmp_sel")
    include_user = c1.checkbox("Also include '6 — current sidebar settings'",
                               value=True, key="cmp_user")
    xvar = c2.selectbox("Sweep variable", ["Distance", "Detector efficiency",
                                           "Filter bandwidth",
                                           "Mean photon number mu",
                                           "Receiver FOV"], key="cmp_var")
    npts_c = c2.slider("Points", 15, 150, 55, key="cmp_n")
    go_cmp = c3.button("▶ Run comparison", use_container_width=True,
                       type="primary", key="cmp_go")

    if go_cmp:
        var = sweepmod.SWEEP_VARIABLES[xvar]
        prog = st.progress(0.0)
        dfs = {}
        names = list(sel) + (["6 — current sidebar settings"] if include_user else [])
        for i, name in enumerate(names):
            pp = P.copy() if name.startswith("6") else bgm.apply_scenario(P, name)
            dfs[name] = sweepmod.sweep_1d(
                pp, xvar, var.default_min, var.default_max, npts_c,
                var.default_log,
                metrics=["Secret key rate [bit/s]", "QBER",
                         "Mutual information I(A:B) [bit/sifted bit]",
                         "Sifted key rate [bit/s]",
                         "Background count rate [1/s]",
                         "Secret key fraction [bit/sifted bit]"])
            prog.progress((i + 1) / len(names))
        prog.empty()
        st.session_state["cmp"] = (dfs, xvar, var.default_log)

    # operating-point table always shown
    rows = []
    for name in bgm.BACKGROUND_SCENARIOS:
        pp = bgm.apply_scenario(P, name)
        rr = run_analytic(pp)
        rows.append({
            "Scenario": name,
            "Note": bgm.BACKGROUND_SCENARIOS[name].get("note", ""),
            "R_bg [counts/s]": f"{rr.background_count_rate_hz:.4e}",
            "µ_bg / gate": f"{rr.detector.mu_bg:.4e}",
            "QBER [%]": f"{100*rr.qber.qber_total:.4f}",
            "I(A:B)": f"{rr.info.mutual_information_per_bit:.4f}",
            "Sifted [bit/s]": f"{rr.sifted_key_rate_bps:.4e}",
            "Secret key [bit/s]": f"{rr.secret_key_rate_bps:.4e}",
            "Secure?": "yes" if rr.key.secure else "NO",
        })
    rr = R
    rows.append({"Scenario": "6 — current sidebar settings", "Note": "",
                 "R_bg [counts/s]": f"{rr.background_count_rate_hz:.4e}",
                 "µ_bg / gate": f"{rr.detector.mu_bg:.4e}",
                 "QBER [%]": f"{100*rr.qber.qber_total:.4f}",
                 "I(A:B)": f"{rr.info.mutual_information_per_bit:.4f}",
                 "Sifted [bit/s]": f"{rr.sifted_key_rate_bps:.4e}",
                 "Secret key [bit/s]": f"{rr.secret_key_rate_bps:.4e}",
                 "Secure?": "yes" if rr.key.secure else "NO"})
    st.markdown("##### Operating point in every scenario")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.session_state.get("cmp"):
        dfs, xvar_, log_ = st.session_state["cmp"]
        st.markdown("##### Scenario curves")
        st.plotly_chart(pl.scenario_comparison(
            dfs, "Secret key rate [bit/s]",
            title=f"Secret key rate vs {xvar_} — background scenarios compared",
            log_x=log_, log_y=True, height=480),
            use_container_width=True, key="cmp_R")
        c1, c2 = st.columns(2)
        c1.plotly_chart(pl.scenario_comparison(
            dfs, "QBER", title=f"QBER vs {xvar_}", log_x=log_,
            threshold=0.11, threshold_label="11% threshold"),
            use_container_width=True, key="cmp_E")
        c2.plotly_chart(pl.scenario_comparison(
            dfs, "Mutual information I(A:B) [bit/sifted bit]",
            title=f"Mutual information vs {xvar_}", log_x=log_),
            use_container_width=True, key="cmp_I")
        with st.expander("Download comparison data"):
            for name, df in dfs.items():
                st.download_button(f"CSV — {name}", ex.dataframe_to_csv_bytes(df),
                                   file_name=f"vlqkd_scenario_{name[:2].strip()}.csv",
                                   mime="text/csv", key=f"cmpdl_{name}")

# --------------------------------------------------------------------------
#  TAB 8 - FILTER OPTIMISATION
# --------------------------------------------------------------------------
with TABS[7]:
    st.markdown("#### Optical filter bandwidth optimisation")
    st.markdown(
        "<div class='banner warn'><div><span class='big'>Numerical optimisation "
        "result</span><br>The bandwidth reported below maximises the secret key "
        "rate <i>of this model</i> subject to the QBER constraint. It is not an "
        "experimentally validated optimum, and it does not account for the "
        "spectral truncation of a real pulsed source by an over-narrow filter."
        "</div></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    o_lo = c1.number_input("Δλ min [nm]", 0.001, 50.0, 0.005, format="%.4g", key="opt1")
    o_hi = c2.number_input("Δλ max [nm]", 0.01, 500.0, 100.0, format="%.4g", key="opt2")
    o_n = c3.slider("Points", 20, 300, 110, key="opt3")
    o_q = c4.slider("Max acceptable QBER", 0.01, 0.20, 0.11, 0.005, key="opt4")
    if st.button("▶ Optimise filter bandwidth", type="primary", key="opt_go"):
        prog = st.progress(0.0)
        df, best = sweepmod.filter_optimization(
            P, o_lo, o_hi, o_n, o_q, progress_callback=lambda f: prog.progress(f))
        prog.empty()
        st.session_state.filter_opt = (df, best)

    if st.session_state.filter_opt:
        df, best = st.session_state.filter_opt
        if best["found"]:
            kpi_grid([
                kpi("Optimal filter bandwidth",
                    f"{best['bandwidth_nm']:.4g}<span class='unit'> nm</span>",
                    "numerical optimum of this model", LIME),
                kpi("Secret key rate there",
                    fmt(best["secret_key_rate_bps"], unit=" bit/s"), "", GREEN),
                kpi("QBER there", f"{100*best['qber']:.4f}<span class='unit'>%</span>",
                    f"constraint ≤ {100*o_q:.1f}%", ROSE),
                kpi("Background count rate there",
                    fmt(best["background_rate_hz"], unit=" counts/s"), "", AMBER),
            ], 4)
        else:
            st.error(best["message"])

        col = "Optical filter bandwidth [nm]"
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        f = make_subplots(specs=[[{"secondary_y": True}]])
        f.add_trace(go.Scatter(x=df[col], y=np.where(df["Secret key rate [bit/s]"] > 0,
                                                     df["Secret key rate [bit/s]"], np.nan),
                               name="Secret key rate", line=dict(color=GREEN, width=2.6)),
                    secondary_y=False)
        f.add_trace(go.Scatter(x=df[col], y=df["Background count rate [1/s]"],
                               name="Background count rate",
                               line=dict(color=AMBER, width=2, dash="dot")),
                    secondary_y=False)
        f.add_trace(go.Scatter(x=df[col], y=100 * df["QBER"], name="QBER [%]",
                               line=dict(color=ROSE, width=2.2)), secondary_y=True)
        if best["found"]:
            f.add_vline(x=best["bandwidth_nm"],
                        line=dict(color=LIME, width=2, dash="dash"),
                        annotation_text=f"optimum {best['bandwidth_nm']:.3g} nm",
                        annotation_font=dict(size=11, color=LIME))
        from visualization.theme import style as _style
        _style(f, "Filter optimisation — numerical result, not experimentally validated",
               "Filter bandwidth Δλ [nm]", "bit/s  and  counts/s", 520,
               log_x=True, log_y=True,
               provenance="analytic model, asymptotic key rate")
        f.update_yaxes(title_text="QBER [%]", secondary_y=True, showgrid=False,
                       range=[0, 55])
        st.plotly_chart(f, use_container_width=True, key="optfig")

        st.dataframe(df.drop(columns=[c for c in df.columns if c.startswith("_")]),
                     use_container_width=True, hide_index=True, height=300)
        st.download_button("CSV — filter optimisation",
                           ex.dataframe_to_csv_bytes(df),
                           file_name="vlqkd_filter_optimisation.csv",
                           mime="text/csv", key="optdl")

# --------------------------------------------------------------------------
#  TAB 9 - MONTE CARLO
# --------------------------------------------------------------------------
with TABS[8]:
    st.markdown("#### Event-by-event Monte-Carlo BB84")
    st.caption("Every pulse is simulated individually: Alice's basis and bit, a "
               "Poisson photon number, per-photon survival through the channel "
               "and the detector, background and dark clicks in each detector, "
               "polarisation error, Bob's basis, double-click handling, "
               "sifting, and the resulting QBER.")

    c1, c2, c3 = st.columns([1, 1, 1])
    c1.markdown(f"**Pulses:** {P.monte_carlo.n_pulses:,}  \n"
                f"**Seed:** {P.monte_carlo.seed}")
    c2.markdown(f"**η_sys:** {R.eta_system:.4e}  \n"
                f"**µ_bg / gate:** {R.detector.mu_bg:.4e}")
    if c3.button("▶ Run Monte Carlo", type="primary", use_container_width=True,
                 key="mc_go"):
        prog = st.progress(0.0, "Simulating…")
        t0 = time.perf_counter()
        mc = run_monte_carlo(P, progress_callback=lambda f: prog.progress(
            min(f, 1.0), f"Simulating… {100*f:.0f}%"))
        prog.empty()
        st.session_state.mc = (mc, time.perf_counter() - t0)

    if st.session_state.mc:
        mc, secs = st.session_state.mc
        dq = mc.qber - R.qber.qber_total
        nsig = abs(dq) / mc.qber_std_err if mc.qber_std_err > 0 else 0.0
        kpi_grid([
            kpi("Monte-Carlo QBER",
                f"{100*mc.qber:.4f}<span class='unit'>% ± "
                f"{100*mc.qber_std_err:.4f}%</span>",
                f"analytic: {100*R.qber.qber_total:.4f}% "
                f"({nsig:.2f} σ away)", ROSE),
            kpi("Sifted bits", f"{mc.n_sifted:,}",
                f"from {mc.n_pulses:,} pulses in {secs:.2f} s", BLUE),
            kpi("Gain Q_µ (measured)", fmt(mc.gain_q_mu),
                f"analytic: {fmt(R.qber.gain_q_mu)}", VIOLET),
            kpi("Secret key rate (from MC statistics)",
                fmt(mc.secret_key_rate_bps, unit=" bit/s"),
                f"analytic: {fmt(R.secret_key_rate_bps, unit=' bit/s')}",
                GREEN if mc.secret_key_rate_bps > 0 else ROSE),
        ], 4)

        c1, c2 = st.columns([1.25, 1])
        c1.plotly_chart(pl.monte_carlo_convergence(mc, R.qber.qber_total),
                        use_container_width=True, key="mc_conv")
        c2.plotly_chart(pl.monte_carlo_counts(mc), use_container_width=True,
                        key="mc_counts")

        st.markdown("##### Event bookkeeping")
        tbl = pd.DataFrame([
            {"Quantity": "Transmitted pulses", "Count": f"{mc.n_pulses:,}"},
            {"Quantity": "Z-basis pulses (Alice)", "Count": f"{mc.n_z_pulses:,}"},
            {"Quantity": "X-basis pulses (Alice)", "Count": f"{mc.n_x_pulses:,}"},
            {"Quantity": "Matched-basis pulses", "Count": f"{mc.n_matched_basis:,}"},
            {"Quantity": "Signal-induced clicks", "Count": f"{mc.n_signal_clicks:,}"},
            {"Quantity": "Background-induced clicks", "Count": f"{mc.n_background_clicks:,}"},
            {"Quantity": "Dark/afterpulse clicks", "Count": f"{mc.n_dark_clicks:,}"},
            {"Quantity": "Double clicks (random assignment)", "Count": f"{mc.n_double_clicks:,}"},
            {"Quantity": "Total detections", "Count": f"{mc.n_detections:,}"},
            {"Quantity": "Sifted key bits", "Count": f"{mc.n_sifted:,}"},
            {"Quantity": "Errors in the sifted key", "Count": f"{mc.n_errors:,}"},
        ])
        st.dataframe(tbl, use_container_width=True, hide_index=True, height=420)
        st.download_button("CSV — Monte-Carlo summary",
                           ex.dataframe_to_csv_bytes(
                               pd.DataFrame([mc.as_dict()])),
                           file_name="vlqkd_monte_carlo.csv", mime="text/csv",
                           key="mcdl")
    else:
        st.info("Press **Run Monte Carlo** to simulate pulse by pulse. "
                "10⁷ pulses take a few seconds.")

# --------------------------------------------------------------------------
#  TAB 10 - PARAMETER SWEEP
# --------------------------------------------------------------------------
with TABS[9]:
    st.markdown("#### Generic parameter sweep")
    st.caption("Choose any independent variable, its range, the number of "
               "points and linear/log spacing. Every information metric is "
               "computed automatically.")

    c1, c2 = st.columns([1.2, 1])
    var_name = c1.selectbox("Independent variable",
                            list(sweepmod.SWEEP_VARIABLES.keys()), key="sw_var")
    var = sweepmod.SWEEP_VARIABLES[var_name]
    if var.note:
        c1.caption(var.note)
    metrics_sel = c2.multiselect(
        "Metrics to plot", list(sweepmod.METRICS.keys()),
        default=["Secret key rate [bit/s]", "QBER",
                 "Mutual information I(A:B) [bit/sifted bit]",
                 "Sifted key rate [bit/s]"], key="sw_met")

    c1, c2, c3, c4, c5 = st.columns(5)
    v0 = c1.number_input(f"min [{var.unit}]", value=float(var.default_min),
                         format="%.6g", key="sw_min")
    v1 = c2.number_input(f"max [{var.unit}]", value=float(var.default_max),
                         format="%.6g", key="sw_max")
    nps = c3.number_input("points", 5, 2000, 80, 5, key="sw_n")
    lg = c4.checkbox("log scale", value=var.default_log, key="sw_log")
    go_sw = c5.button("▶ Run sweep", use_container_width=True, type="primary",
                      key="sw_go")

    if go_sw:
        prog = st.progress(0.0)
        df = sweepmod.sweep_1d(P, var_name, v0, v1, int(nps), lg,
                               metrics=list(sweepmod.METRICS.keys()),
                               progress_callback=lambda f: prog.progress(f))
        prog.empty()
        st.session_state.sweep_df = df
        st.session_state.sweep_meta = {"var": var_name, "log": lg}

    if st.session_state.sweep_df is not None:
        df = st.session_state.sweep_df
        meta = st.session_state.sweep_meta
        lgx = meta.get("log", False)
        for m in metrics_sel:
            logy = ("rate" in m.lower() or "gain" in m.lower()
                    or "ratio" in m.lower() or "H(0)" in m or "eta" in m
                    or "photons" in m.lower())
            colour = (GREEN if "Secret key rate" in m else
                      ROSE if "QBER" in m else
                      VIOLET if "Mutual" in m else
                      AMBER if "Background" in m else BLUE)
            st.plotly_chart(
                pl.sweep_plot(df, m, title=m, log_x=lgx, log_y=logy,
                              color=colour,
                              threshold=0.11 if m == "QBER" else None,
                              threshold_label="11% threshold",
                              shade_secure=("Secret key rate" in m)),
                use_container_width=True, key=f"swfig_{m}")
        st.dataframe(df.drop(columns=[c for c in df.columns if c.startswith("_")]),
                     use_container_width=True, hide_index=True, height=340)
        c1, c2 = st.columns(2)
        c1.download_button("CSV — sweep", ex.dataframe_to_csv_bytes(df),
                           file_name="vlqkd_parameter_sweep.csv", mime="text/csv",
                           key="swdl1")
        c2.download_button("Excel — sweep",
                           ex.to_excel_bytes({"sweep": df,
                                              "parameters": ex.parameters_to_dataframe(P)}),
                           file_name="vlqkd_parameter_sweep.xlsx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet", key="swdl2")

# --------------------------------------------------------------------------
#  TAB 11 - FINITE KEY
# --------------------------------------------------------------------------
with TABS[10]:
    st.markdown("#### Finite-key analysis")
    st.markdown(
        "<div class='banner warn'><div><span class='big'>Simplified finite-key "
        "model</span><br>A Hoeffding-bounded simplification of Lim <i>et al.</i>, "
        "Phys. Rev. A <b>89</b>, 022307 (2014). It shows how the asymptotic rate "
        "degrades with block size; it is <b>not</b> a composable security proof "
        "of a physical implementation.</div></div>", unsafe_allow_html=True)

    if not P.protocol.finite_key:
        st.info("Enable **simplified finite-key analysis** in the sidebar "
                "(BB84 protocol & security model) to activate this tab.")
    else:
        fk = evaluate_finite_key(P, R)
        kpi_grid([
            kpi("Finite-size secret key rate",
                fmt(fk.secret_key_rate_bps, unit=" bit/s"),
                f"asymptotic: {fmt(fk.asymptotic_key_rate_bps, unit=' bit/s')}",
                GREEN if fk.secure else ROSE),
            kpi("Secret key length", fmt(fk.secret_key_length_bits, unit=" bit"),
                f"from a block of {fk.n_pulses:.3g} pulses", TEAL),
            kpi("Finite-size penalty", f"{100*fk.penalty_factor:.2f}<span class='unit'>%</span>",
                "of the asymptotic rate", AMBER),
            kpi("Phase error φ₁", f"{100*fk.phi1_phase:.4f}<span class='unit'>%</span>",
                f"bit error e₁ = {100*fk.e1_bit:.4f}%", ROSE),
        ], 4)
        if not fk.secure:
            st.error(fk.message)

        st.dataframe(pd.DataFrame([
            {"Quantity": "Block size N [pulses]", "Value": f"{fk.n_pulses:.6g}"},
            {"Quantity": "Sifted bits n", "Value": f"{fk.n_sifted:.6g}"},
            {"Quantity": "Certified single-photon detections s₁",
             "Value": f"{fk.n_single_photon:.6g}"},
            {"Quantity": "Error-correction leakage λ_EC [bit]",
             "Value": f"{fk.ec_leakage_bits:.6g}"},
            {"Quantity": "Privacy-amplification yield [bit]", "Value": f"{fk.pa_cost_bits:.6g}"},
            {"Quantity": "Finite-size correction [bit]",
             "Value": f"{fk.finite_correction_bits:.6g}"},
            {"Quantity": "ε_sec", "Value": f"{P.protocol.epsilon_sec:.3g}"},
            {"Quantity": "ε_cor", "Value": f"{P.protocol.epsilon_cor:.3g}"},
        ]), use_container_width=True, hide_index=True)

        st.markdown("##### Key rate vs block size")
        if st.button("Compute block-size scan", key="fk_go"):
            from simulation.finite_size import finite_key_length
            Ns = np.logspace(4, 14, 60)
            rates, fracs = [], []
            for N in Ns:
                f = finite_key_length(float(N), R.qber.gain_q_mu,
                                      R.qber.qber_total, R.key.q1, R.key.e1,
                                      P.protocol.error_correction_efficiency,
                                      R.key.q_sift, P.protocol.epsilon_sec,
                                      P.protocol.epsilon_cor,
                                      P.source.pulse_rate_hz,
                                      R.secret_key_rate_bps,
                                      R.detector.dead_time_saturation_factor)
                rates.append(f.secret_key_rate_bps)
                fracs.append(f.secret_fraction_per_sifted_bit)
            st.session_state["fkscan"] = (Ns, np.array(rates), np.array(fracs))
        if st.session_state.get("fkscan"):
            Ns, rates, fracs = st.session_state["fkscan"]
            import plotly.graph_objects as go
            from visualization.theme import style as _style
            f = go.Figure()
            f.add_trace(go.Scatter(x=Ns, y=np.where(rates > 0, rates, np.nan),
                                   name="finite-size rate",
                                   line=dict(color=TEAL, width=2.6)))
            f.add_hline(y=max(R.secret_key_rate_bps, 1e-12),
                        line=dict(color=GREEN, width=1.6, dash="dash"),
                        annotation_text="asymptotic rate",
                        annotation_font=dict(size=11, color=GREEN))
            _style(f, "Simplified finite-key rate vs block size",
                   "Block size N [transmitted pulses]", "Secret key rate [bit/s]",
                   460, log_x=True, log_y=True,
                   provenance="finite-size (simplified)")
            st.plotly_chart(f, use_container_width=True, key="fkfig")

# --------------------------------------------------------------------------
#  TAB 12 - EXPORT
# --------------------------------------------------------------------------
with TABS[11]:
    st.markdown("#### Export results and generate the research report")

    summary_df = ex.summary_to_dataframe(R)
    budget_df = ex.qber_budget_to_dataframe(R.qber)
    bg_df = ex.background_to_dataframe(R.background)
    par_df = ex.parameters_to_dataframe(P)

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("##### Tables")
        st.dataframe(summary_df, use_container_width=True, hide_index=True,
                     height=420)
    with c2:
        st.markdown("##### Quick downloads")
        st.download_button("⬇ CSV — result summary",
                           ex.dataframe_to_csv_bytes(summary_df),
                           file_name="vlqkd_summary.csv", mime="text/csv",
                           use_container_width=True, key="ex1")
        st.download_button("⬇ CSV — QBER budget",
                           ex.dataframe_to_csv_bytes(budget_df),
                           file_name="vlqkd_qber_budget.csv", mime="text/csv",
                           use_container_width=True, key="ex2")
        st.download_button("⬇ CSV — background sources",
                           ex.dataframe_to_csv_bytes(bg_df),
                           file_name="vlqkd_background.csv", mime="text/csv",
                           use_container_width=True, key="ex3")
        st.download_button("⬇ CSV — parameters",
                           ex.dataframe_to_csv_bytes(par_df),
                           file_name="vlqkd_parameters.csv", mime="text/csv",
                           use_container_width=True, key="ex4")

        sheets = {"Summary": summary_df, "QBER budget": budget_df,
                  "Background": bg_df, "Parameters": par_df}
        if st.session_state.sweep_df is not None:
            sheets["Sweep"] = st.session_state.sweep_df
        if st.session_state.filter_opt:
            sheets["Filter optimisation"] = st.session_state.filter_opt[0]
        st.download_button("⬇ Excel — full workbook", ex.to_excel_bytes(sheets),
                           file_name="vlqkd_results.xlsx",
                           mime="application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet",
                           use_container_width=True, key="ex5")
        st.download_button("⬇ JSON — parameters + results",
                           ex.to_json_bytes(P, R),
                           file_name="vlqkd_run.json", mime="application/json",
                           use_container_width=True, key="ex6")

    st.markdown("---")
    st.markdown("##### Figures (PNG / interactive HTML)")
    figlist = [
        ("Room — top view", roomviz.top_view(P)),
        ("Room — 3-D view", roomviz.three_d_view(P)),
        ("Background spectrum and filter", pl.spectrum_plot(R)),
        ("QBER error budget", pl.qber_budget_bar(R.qber)),
        ("Information waterfall", pl.information_waterfall(R)),
        ("Secret key fraction vs QBER",
         pl.mutual_information_vs_qber(P.protocol.error_correction_efficiency)),
    ]
    if st.session_state.heat:
        xs, ys, Z = st.session_state.heat["R"]
        figlist.append(("Secret key rate heat-map",
                        hm.key_rate_heatmap(xs, ys, Z,
                                            xlabel="Link distance [m]",
                                            ylabel="Background count rate [counts/s]",
                                            title="R(distance, background)")))
    if st.session_state.mc:
        figlist.append(("Monte-Carlo convergence",
                        pl.monte_carlo_convergence(st.session_state.mc[0],
                                                   R.qber.qber_total)))

    cols = st.columns(3)
    for i, (name, fig) in enumerate(figlist):
        col = cols[i % 3]
        png = ex.figure_to_png_bytes(fig)
        if png:
            col.download_button(f"⬇ PNG — {name}", png,
                                file_name=f"vlqkd_{name.lower().replace(' ', '_')}.png",
                                mime="image/png", use_container_width=True,
                                key=f"png_{i}")
        else:
            col.download_button(f"⬇ HTML — {name}",
                                ex.figure_to_html_bytes(fig),
                                file_name=f"vlqkd_{name.lower().replace(' ', '_')}.html",
                                mime="text/html", use_container_width=True,
                                key=f"html_{i}")
    if not ex.figure_to_png_bytes(figlist[0][1]):
        st.caption("Static PNG export needs the `kaleido` package "
                   "(`pip install kaleido`); interactive HTML is offered instead.")

    st.markdown("---")
    st.markdown("##### Research report (PDF)")
    c1, c2 = st.columns([2, 1])
    author = c1.text_input("Author / project label (optional)", "",
                           key="rep_author")
    include_figs = c2.checkbox("Embed figures", value=True, key="rep_figs")
    if st.button("📄 Build research report", type="primary", key="rep_go"):
        with st.spinner("Assembling the report…"):
            sweeps = {}
            if st.session_state.sweep_df is not None:
                sweeps[f"Sweep: {st.session_state.sweep_meta.get('var','')}"] = \
                    st.session_state.sweep_df
            if st.session_state.filter_opt:
                sweeps["Filter optimisation"] = st.session_state.filter_opt[0]
            fk = evaluate_finite_key(P, R) if P.protocol.finite_key else None
            pdf = ex.build_pdf_report(
                P, R,
                figures=figlist if include_figs else None,
                sweeps=sweeps or None,
                mc_result=st.session_state.mc[0] if st.session_state.mc else None,
                finite=fk, author=author)
            st.session_state["pdf"] = pdf
    if st.session_state.get("pdf"):
        st.success(f"Report built ({len(st.session_state['pdf'])/1024:.0f} kB).")
        st.download_button("⬇ Download the PDF report", st.session_state["pdf"],
                           file_name="vlqkd_research_report.pdf",
                           mime="application/pdf", type="primary", key="repdl")

# --------------------------------------------------------------------------
#  TAB 13 - MODEL & SELF-TEST
# --------------------------------------------------------------------------
with TABS[12]:
    st.markdown("#### Start-up self-test")
    if N_PASS == N_TEST:
        st.markdown(f"<div class='banner ok'><div><span class='big'>"
                    f"All {N_TEST} self-tests passed</span><br>"
                    f"Each test checks either a closed-form result or a "
                    f"limiting behaviour the physics demands.</div></div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='banner bad'><div><span class='big'>"
                    f"{N_TEST-N_PASS} of {N_TEST} self-tests failed</span><br>"
                    f"Treat the results below with caution.</div></div>",
                    unsafe_allow_html=True)
    if st.button("Re-run self-test", key="st_go"):
        _cached_selftest.clear()
        st.session_state.selftest = _cached_selftest()
        st.rerun()

    for t in ST:
        with st.expander(f"{'✅' if t['passed'] else '❌'} {t['name']} "
                         f"— {t['ms']:.0f} ms"):
            for line in t["detail"].split("; "):
                st.markdown(f"- {line}")

    st.markdown("---")
    st.markdown("#### Model equations")
    st.caption("The physical channel model and the QKD security model are kept "
               "strictly separate. Nothing in the channel/background/detector "
               "block uses a QKD assumption; nothing in the security block uses "
               "an optical detail beyond η_sys, µ_bg and µ_dark.")
    eq_df = pd.DataFrame([{"Quantity": n, "Expression": e} for n, e in ex.EQUATIONS])
    st.dataframe(eq_df, use_container_width=True, hide_index=True, height=620)

    st.markdown("#### Result provenance in this run")
    st.dataframe(pd.DataFrame([
        {"Stage": "Source / photon statistics", "Model class": R.source.provenance},
        {"Stage": "Optical channel", "Model class": R.channel.provenance},
        {"Stage": "Background noise", "Model class": R.background.provenance},
        {"Stage": "Detector", "Model class": R.detector.provenance},
        {"Stage": "Polarisation", "Model class": R.polarization.provenance},
        {"Stage": "Pointing error", "Model class": R.pointing.provenance},
        {"Stage": "Turbulence", "Model class": R.turbulence.provenance},
        {"Stage": "QBER", "Model class": R.qber.provenance},
        {"Stage": "Information metrics", "Model class": R.info.provenance},
        {"Stage": "Key rate", "Model class": R.key.provenance},
    ]), use_container_width=True, hide_index=True)

    st.markdown("#### Assumptions and limitations")
    for a in ex.ASSUMPTIONS:
        st.markdown(f"- {a}")

    st.markdown("#### References")
    st.markdown("""
- O. Elmabrok and M. Razavi, *Wireless quantum key distribution in indoor
  environments*, J. Opt. Soc. Am. B **35**, 197 (2018); arXiv:1605.05092 —
  indoor QKD background-noise model and system parameters.
- J. M. Kahn and J. R. Barry, *Wireless infrared communications*,
  Proc. IEEE **85**, 265 (1997); J. R. Barry *et al.*, IEEE JSAC **11**, 367
  (1993) — Lambertian LOS channel gain and indoor optical channel modelling.
- T. Komine and M. Nakagawa, *Fundamental analysis for visible-light
  communication system using LED lights*, IEEE Trans. Consum. Electron. **50**,
  100 (2004) — indoor VLC link budget and concentrator gain.
- A. J. C. Moreira, R. T. Valadas, A. M. de Oliveira Duarte, *Optical
  interference produced by artificial light*, Wireless Networks **3**, 131
  (1997) — ambient-light spectral irradiance values.
- X. Ma, B. Qi, Y. Zhao, H.-K. Lo, *Practical decoy state for quantum key
  distribution*, Phys. Rev. A **72**, 012326 (2005) — decoy-state bounds.
- D. Gottesman, H.-K. Lo, N. Lütkenhaus, J. Preskill, *Security of quantum key
  distribution with imperfect devices*, QIC **4**, 325 (2004) — GLLP key rate.
- C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security
  bounds for practical decoy-state QKD*, Phys. Rev. A **89**, 022307 (2014) —
  finite-key analysis.
- A. A. Farid and S. Hranilovic, *Outage capacity optimization for free-space
  optical links with pointing errors*, J. Lightwave Technol. **25**, 1702
  (2007) — pointing-error statistics.
""")

# ==========================================================================
st.markdown(
    f"<div class='footer'><b>Indoor VL-QKD BB84 Simulator</b> — "
    f"modular research toolkit. {C.DISCLAIMER}</div>",
    unsafe_allow_html=True)
