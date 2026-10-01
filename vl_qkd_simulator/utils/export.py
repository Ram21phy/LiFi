"""
utils/export.py
===============
Export of results and generation of the research report.

Formats
-------
CSV    - any DataFrame (sweeps, budgets, summaries)
Excel  - multi-sheet workbook: parameters, summary, error budget, sweeps
JSON   - the complete parameter set plus the full result summary
PNG    - static image of any plotly figure (needs `kaleido`)
PDF    - a structured research report containing the parameters, the model
         equations, the assumptions and the numerical results

The PDF report is built with ReportLab.  It deliberately contains a prominent
assumptions-and-limitations section: the point of the report is to make a
simulation result citable *and* falsifiable, not to make it look authoritative.
"""

from __future__ import annotations

import datetime as _dt
import io
import json
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from utils import constants as C


# ==========================================================================
#  Simple tabular exports
# ==========================================================================
def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def summary_to_dataframe(result) -> pd.DataFrame:
    rows = []
    for k, v in result.summary().items():
        if isinstance(v, bool):
            val = "yes" if v else "no"
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            val = f"{v:.6g}"
        else:
            val = str(v)
        rows.append({"Quantity": k, "Value": val})
    return pd.DataFrame(rows)


def qber_budget_to_dataframe(budget) -> pd.DataFrame:
    rows = [{"Error mechanism": n,
             "Absolute contribution to QBER": f"{v:.6e}",
             "Percent": f"{p:.4f}"}
            for n, v, p in budget.as_percentage_table()]
    return pd.DataFrame(rows)


def background_to_dataframe(bg) -> pd.DataFrame:
    rows = []
    for c in bg.contributions:
        rows.append({
            "Source": c.name,
            "In-band optical power [W]": f"{c.optical_power_w:.6e}",
            "Photon flux [1/s]": f"{c.photon_flux_hz:.6e}",
            "Count rate [counts/s]": f"{c.count_rate_hz:.6e}",
            "Spectral irradiance at signal [W/m^2/nm]":
                f"{c.spectral_irradiance_w_m2_nm:.6e}",
            "Note": c.note,
        })
    rows.append({"Source": "TOTAL",
                 "In-band optical power [W]": f"{bg.total_optical_power_w:.6e}",
                 "Photon flux [1/s]": f"{bg.total_photon_flux_hz:.6e}",
                 "Count rate [counts/s]": f"{bg.total_count_rate_hz:.6e}",
                 "Spectral irradiance at signal [W/m^2/nm]": "",
                 "Note": f"mu_bg = {bg.mu_bg_per_gate:.6e} counts per gate"})
    return pd.DataFrame(rows)


def parameters_to_dataframe(params) -> pd.DataFrame:
    rows = []

    def walk(prefix, d):
        for k, v in d.items():
            if isinstance(v, dict):
                walk(f"{prefix}{k}.", v)
            else:
                rows.append({"Parameter": f"{prefix}{k}", "Value": str(v)})
    walk("", params.to_dict())
    return pd.DataFrame(rows)


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for name, df in sheets.items():
            safe = name[:31].replace("/", "-").replace("\\", "-")
            df.to_excel(xw, sheet_name=safe, index=False)
            ws = xw.sheets[safe]
            for i, col in enumerate(df.columns):
                width = max(12, min(60, int(df[col].astype(str).str.len().max() or 12) + 2,
                                    ))
                ws.set_column(i, i, max(width, len(str(col)) + 2))
    return buf.getvalue()


def to_json_bytes(params, result=None, extra: Optional[dict] = None) -> bytes:
    payload = {
        "software": "Indoor VL-QKD BB84 Simulator",
        "generated_utc": _dt.datetime.utcnow().isoformat() + "Z",
        "disclaimer": C.DISCLAIMER,
        "parameters": params.to_dict(),
    }
    if result is not None:
        payload["results"] = {k: (v if not isinstance(v, (np.floating, np.integer))
                                  else float(v))
                              for k, v in result.summary().items()}
        payload["qber_budget"] = {
            "polarization": result.qber.q_polarization,
            "pointing": result.qber.q_pointing,
            "background": result.qber.q_background,
            "dark_counts": result.qber.q_dark,
            "afterpulsing": result.qber.q_afterpulse,
            "total": result.qber.qber_total,
        }
        payload["background_sources"] = [
            {"name": c.name, "count_rate_hz": c.count_rate_hz,
             "optical_power_w": c.optical_power_w, "note": c.note}
            for c in result.background.contributions]
        payload["provenance"] = {
            "channel": result.channel.provenance,
            "background": result.background.provenance,
            "detector": result.detector.provenance,
            "qber": result.qber.provenance,
            "key_rate": result.key.provenance,
        }
        payload["warnings"] = result.warnings
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2, default=str).encode("utf-8")


def figure_to_png_bytes(fig, width: int = 1400, height: int = 800,
                        scale: float = 2.0) -> Optional[bytes]:
    """Static PNG of a plotly figure. Returns None if kaleido is unavailable."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=scale)
    except Exception:
        return None


def figure_to_html_bytes(fig, title: str = "figure") -> bytes:
    return fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")


# ==========================================================================
#  Research report (PDF)
# ==========================================================================
EQUATIONS = [
    ("Photon energy", "E_ph = h c / lambda"),
    ("Photon statistics (WCP)", "P(n) = exp(-mu) mu^n / n!"),
    ("Multi-photon fraction", "P(n>=2) = 1 - e^-mu - mu e^-mu"),
    ("Lambertian order",
     "m = -ln 2 / ln( cos(Phi_1/2) )"),
    ("VLC LOS channel gain",
     "H(0) = (m+1) A / (2 pi d^2) * cos^m(phi) * T_s(psi) * g(psi) * cos(psi),"
     "  0 <= psi <= Psi_c ;  H(0) = 0 otherwise"),
    ("Concentrator gain", "g(psi) = n^2 / sin^2(Psi_c)   for psi <= Psi_c"),
    ("Collimated-beam transmittance",
     "w(d) = w0 + d theta_div ;  T_geo = [1 - exp(-2 a^2 / w(d)^2)] cos(psi)"),
    ("System transmittance",
     "eta_sys = H(0) * eta_opt * eta_point * eta_turb * eta_det"),
    ("Background in-band power",
     "P_bg = Int p(lambda) A T_f(lambda) G_coll dlambda "
     "~= p(l0) A dLambda T_filter G_coll"),
    ("Background collection factor",
     "G_coll = pi sin^2(Psi_c)  (bare detector)   or   pi n^2  (ideal "
     "concentrator; etendue-limited, FOV-independent)"),
    ("Diffuse multi-reflection (integrating-sphere estimate)",
     "E_diffuse = P_emitted rho / [ S_room (1 - rho) ]"),
    ("Background counts per gate", "mu_bg = R_bg * dt"),
    ("Dark counts per gate", "mu_dark = R_dark * dt"),
    ("Vacuum / noise yield",
     "Y_0 = 1 - exp(-N_det (mu_bg + mu_dark + mu_ap))"),
    ("Signal click probability",
     "P_sig(click|n) = 1 - (1-eta_sys)^n ;  Poisson average = 1 - exp(-eta_sys mu)"),
    ("Overall gain", "Q_mu = 1 - (1 - Y_0) exp(-eta_sys mu)"),
    ("Error probability",
     "E_mu Q_mu = p_s(1-Y_0) e_d + (1-p_s) Y_0 /2 + p_s Y_0 (e_d/2 + 1/4),"
     "  p_s = 1 - exp(-eta_sys mu)"),
    ("Intrinsic polarisation error",
     "e_d = 1 - (1 - sin^2 theta)(1 - e_ER)(1 - p_dep/2),   "
     "e_ER = 1/(1 + 10^(ER/10))"),
    ("Single-photon yield / error",
     "Y_1 = 1 - (1-Y_0)(1-eta_sys) ;  "
     "e_1 Y_1 = eta(1-Y_0) e_d + (1-eta) Y_0/2 + eta Y_0 (e_d/2 + 1/4)"),
    ("GLLP bound (no decoy)",
     "Q_1^L = max(Q_mu - P(n>=2) - Y_0 e^-mu, 0) ;  e_1^U = E_mu Q_mu / Q_1^L"),
    ("Vacuum+weak decoy (Ma et al. 2005)",
     "Y_1^L = mu/(mu v1 - mu v2 - v1^2 + v2^2) * [ Q_v1 e^v1 - Q_v2 e^v2 "
     "- (v1^2-v2^2)/mu^2 (Q_mu e^mu - Y_0^L) ]"),
    ("Binary entropy", "h2(x) = -x log2 x - (1-x) log2 (1-x)"),
    ("Mutual information", "I(A:B) = 1 - h2(E_mu)  [bit per sifted bit]"),
    ("Privacy amplification term",
     "PA = 1 - (Q_1/Q_mu)[1 - h2(e_1)]  [bit per sifted bit]"),
    ("Sifting factor",
     "q = 1/2 (symmetric BB84) ;  q = p_Z^2 + (1-p_Z)^2 (biased/efficient)"),
    ("Asymptotic secret key rate",
     "R = q r_p { Q_1 [1 - h2(e_1)] - f_EC Q_mu h2(E_mu) }"),
    ("Simplified finite-key length",
     "l <= s_1 [1 - h2(phi_1)] - lambda_EC - 6 log2(19/eps_sec) "
     "- log2(2/eps_cor)"),
]

ASSUMPTIONS = [
    "Line-of-sight propagation only; the quantum signal is not assumed to "
    "survive diffuse wall reflections.",
    "The background multi-reflection term uses a single-parameter "
    "integrating-sphere estimate, not a ray-traced impulse response.",
    "Detector model is simplified: afterpulsing is first-order, dead time is "
    "non-paralysable, timing jitter only broadens the effective gate, and "
    "detector crosstalk and twilight pulses are not modelled.",
    "Double clicks are assigned a random bit; no squashing model is applied.",
    "The asymptotic key rate assumes infinite key length, perfect basis "
    "announcement, ideal state preparation apart from the modelled "
    "polarisation error, and no side channels.",
    "Decoy-state bounds are the asymptotic bounds of Ma et al. (2005) "
    "evaluated on model-generated gains, not on measured data.",
    "The finite-key result is a simplified Hoeffding-bounded version of the "
    "Lim et al. (2014) analysis and is NOT a composable security proof of a "
    "physical device.",
    "Turbulence uses a log-normal (weak-fluctuation) model; indoor Rytov "
    "variances are typically far below 1, making it negligible.",
    "Luminaire spectra are analytic approximations (Gaussian phosphor bands, "
    "Hg lines, 5778 K blackbody for sunlight), not measured spectra.",
    "No claim of unconditional or experimentally demonstrated security is "
    "made anywhere in this report.",
]


def build_pdf_report(params, result, figures: Optional[List] = None,
                     sweeps: Optional[Dict[str, pd.DataFrame]] = None,
                     mc_result=None, finite=None,
                     author: str = "") -> bytes:
    """Assemble the research report as a PDF."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                                    SimpleDocTemplate, Spacer, Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Indoor VL-QKD BB84 simulation report")

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=6,
                        textColor=colors.HexColor("#0F766E"))
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=12, spaceBefore=10,
                        spaceAfter=4, textColor=colors.HexColor("#134E4A"))
    BODY = ParagraphStyle("BODY", parent=ss["BodyText"], fontSize=9,
                          leading=12.5, alignment=TA_LEFT)
    MONO = ParagraphStyle("MONO", parent=ss["BodyText"], fontName="Courier",
                          fontSize=7.6, leading=10)
    SMALL = ParagraphStyle("SMALL", parent=ss["BodyText"], fontSize=7.6,
                           leading=10, textColor=colors.HexColor("#475569"))

    story = []
    now = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    story.append(Paragraph("Indoor Visible-Light Quantum Key Distribution "
                           "(BB84) &mdash; Simulation Report", H1))
    story.append(Paragraph(
        f"Generated {now}" + (f" &nbsp;|&nbsp; {author}" if author else "")
        + f" &nbsp;|&nbsp; parameter set: <i>{params.label}</i>", SMALL))
    story.append(Spacer(1, 5))
    story.append(Paragraph(f"<b>Disclaimer.</b> {C.DISCLAIMER}", BODY))
    story.append(Spacer(1, 8))

    def table(df: pd.DataFrame, widths=None, fontsize=7.6):
        data = [list(df.columns)] + df.astype(str).values.tolist()
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), fontsize),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CBD5E1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F1F5F9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        return t

    # --- 1. headline results --------------------------------------------
    story.append(Paragraph("1. Headline numerical results", H2))
    s = result.summary()
    head = pd.DataFrame([
        {"Quantity": "Link distance", "Value": f"{s['Link distance [m]']:.4g} m"},
        {"Quantity": "Channel gain H(0)", "Value": f"{s['Channel gain H(0)']:.4e}"},
        {"Quantity": "System transmittance eta_sys",
         "Value": f"{s['System transmittance eta_sys']:.4e}"},
        {"Quantity": "Received signal photons per pulse",
         "Value": f"{s['Received signal photons / pulse']:.4e}"},
        {"Quantity": "Background count rate",
         "Value": f"{s['Background count rate [1/s]']:.4e} counts/s"},
        {"Quantity": "Background counts per gate",
         "Value": f"{s['Background photons / gate (mu_bg)']:.4e}"},
        {"Quantity": "Dark count rate",
         "Value": f"{s['Dark count rate [1/s]']:.4g} counts/s"},
        {"Quantity": "Signal-to-background ratio",
         "Value": f"{s['Signal-to-background ratio']:.4e}"},
        {"Quantity": "Detection probability Q_mu",
         "Value": f"{s['Detection probability Q_mu']:.4e}"},
        {"Quantity": "QBER", "Value": f"{100*s['QBER']:.4f} %"},
        {"Quantity": "Mutual information I(A:B)",
         "Value": f"{s['Mutual information I(A:B) [bit/sifted bit]']:.4f} bit/sifted bit"},
        {"Quantity": "Eve information / PA term",
         "Value": f"{s['Eve information / PA term [bit/sifted bit]']:.4f} bit/sifted bit"},
        {"Quantity": "Sifted key rate",
         "Value": f"{s['Sifted key rate [bit/s]']:.4e} bit/s"},
        {"Quantity": "Secret key fraction",
         "Value": f"{s['Secret key fraction [bit/sifted bit]']:.4f} bit/sifted bit"},
        {"Quantity": "Secret key rate",
         "Value": f"{s['Secret key rate [bit/s]']:.4e} bit/s"},
        {"Quantity": "Positive asymptotic key rate",
         "Value": "YES" if s["Positive key rate"] else
                  "NO - no positive asymptotic key rate under this model"},
    ])
    story.append(table(head, widths=[85 * mm, 75 * mm], fontsize=8))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Result class: {result.provenance}; channel = {result.channel.provenance}; "
        f"background = {result.background.provenance}; "
        f"detector = {result.detector.provenance}; "
        f"key rate = {result.key.provenance}.", SMALL))

    if result.warnings:
        story.append(Spacer(1, 5))
        story.append(Paragraph("<b>Model warnings</b>", BODY))
        for w in result.warnings:
            story.append(Paragraph(f"&bull; {w}", SMALL))

    # --- 2. QBER budget --------------------------------------------------
    story.append(Paragraph("2. QBER error budget", H2))
    story.append(Paragraph(
        "Every contribution below is computed from the physical model; the "
        "individual terms sum exactly to the total QBER.", BODY))
    story.append(qber_budget_to_dataframe(result.qber).pipe(
        lambda df: table(df, widths=[70 * mm, 50 * mm, 40 * mm], fontsize=8)))

    # --- 3. background ---------------------------------------------------
    story.append(Paragraph("3. Background optical noise model", H2))
    story.append(Paragraph(
        f"Effective (noise-equivalent) filter bandwidth "
        f"{result.background.effective_filter_bandwidth_nm:.4g} nm; "
        f"filter transmission at the signal wavelength "
        f"{result.background.filter_transmission_at_signal:.4f}; "
        f"background collection factor G_coll = "
        f"{result.background.collection_factor_sr:.4g} sr; "
        f"noise integration time "
        f"{result.background.integration_time_s*1e9:.4g} ns.", BODY))
    story.append(table(background_to_dataframe(result.background),
                       widths=[36 * mm, 28 * mm, 26 * mm, 28 * mm, 26 * mm, 30 * mm],
                       fontsize=6.4))

    story.append(PageBreak())

    # --- 4. equations ----------------------------------------------------
    story.append(Paragraph("4. Model equations", H2))
    story.append(Paragraph(
        "The physical channel model and the QKD security model are kept "
        "separate: nothing in Section 4.1&ndash;4.4 uses any QKD assumption, "
        "and nothing in 4.5&ndash;4.8 uses any optical detail beyond "
        "eta_sys, mu_bg and mu_dark.", BODY))
    eq_rows = pd.DataFrame([{"Quantity": n, "Expression": e} for n, e in EQUATIONS])
    story.append(table(eq_rows, widths=[52 * mm, 118 * mm], fontsize=6.8))

    # --- 5. parameters ---------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("5. Complete parameter set", H2))
    story.append(Paragraph(
        "The run is fully reproducible from the values below together with the "
        "Monte-Carlo seed. The same set is available as JSON from the export "
        "panel.", BODY))
    pdf_params = parameters_to_dataframe(params)
    half = (len(pdf_params) + 1) // 2
    left, right = pdf_params.iloc[:half].reset_index(drop=True), \
        pdf_params.iloc[half:].reset_index(drop=True)
    right.columns = ["Parameter ", "Value "]
    both = pd.concat([left, right], axis=1).fillna("")
    story.append(table(both, widths=[46 * mm, 38 * mm, 46 * mm, 38 * mm],
                       fontsize=6.0))

    # --- 6. Monte Carlo --------------------------------------------------
    if mc_result is not None:
        story.append(Paragraph("6. Monte-Carlo verification", H2))
        mc = pd.DataFrame([
            {"Quantity": "Transmitted pulses", "Value": f"{mc_result.n_pulses:,}"},
            {"Quantity": "Random seed", "Value": str(mc_result.seed)},
            {"Quantity": "Z-basis pulses", "Value": f"{mc_result.n_z_pulses:,}"},
            {"Quantity": "X-basis pulses", "Value": f"{mc_result.n_x_pulses:,}"},
            {"Quantity": "Matched-basis pulses", "Value": f"{mc_result.n_matched_basis:,}"},
            {"Quantity": "Total detections", "Value": f"{mc_result.n_detections:,}"},
            {"Quantity": "Sifted bits", "Value": f"{mc_result.n_sifted:,}"},
            {"Quantity": "Errors", "Value": f"{mc_result.n_errors:,}"},
            {"Quantity": "Double clicks", "Value": f"{mc_result.n_double_clicks:,}"},
            {"Quantity": "Monte-Carlo QBER",
             "Value": f"{100*mc_result.qber:.4f} % ± {100*mc_result.qber_std_err:.4f} %"},
            {"Quantity": "Analytic QBER", "Value": f"{100*result.qber.qber_total:.4f} %"},
            {"Quantity": "Monte-Carlo gain Q_mu", "Value": f"{mc_result.gain_q_mu:.6e}"},
            {"Quantity": "Analytic gain Q_mu", "Value": f"{result.qber.gain_q_mu:.6e}"},
        ])
        story.append(table(mc, widths=[85 * mm, 75 * mm], fontsize=8))

    # --- 7. finite key ---------------------------------------------------
    if finite is not None:
        story.append(Paragraph("7. Finite-key analysis (simplified)", H2))
        fk = pd.DataFrame([
            {"Quantity": "Block size (transmitted pulses)", "Value": f"{finite.n_pulses:.4g}"},
            {"Quantity": "Sifted bits in the block", "Value": f"{finite.n_sifted:.4g}"},
            {"Quantity": "Certified single-photon detections", "Value": f"{finite.n_single_photon:.4g}"},
            {"Quantity": "Single-photon bit error e_1", "Value": f"{100*finite.e1_bit:.4f} %"},
            {"Quantity": "Single-photon phase error phi_1", "Value": f"{100*finite.phi1_phase:.4f} %"},
            {"Quantity": "Error-correction leakage", "Value": f"{finite.ec_leakage_bits:.4g} bit"},
            {"Quantity": "Finite-size correction", "Value": f"{finite.finite_correction_bits:.4g} bit"},
            {"Quantity": "Secret key length", "Value": f"{finite.secret_key_length_bits:.4g} bit"},
            {"Quantity": "Finite-size secret key rate", "Value": f"{finite.secret_key_rate_bps:.4e} bit/s"},
            {"Quantity": "Asymptotic secret key rate", "Value": f"{finite.asymptotic_key_rate_bps:.4e} bit/s"},
            {"Quantity": "Finite-size penalty factor", "Value": f"{finite.penalty_factor:.4f}"},
        ])
        story.append(table(fk, widths=[85 * mm, 75 * mm], fontsize=8))
        story.append(Paragraph(
            "This is a SIMPLIFIED finite-key bound (Hoeffding deviations on the "
            "asymptotic decoy estimates). It is not a composable security proof.",
            SMALL))

    # --- 8. sweeps -------------------------------------------------------
    if sweeps:
        story.append(PageBreak())
        story.append(Paragraph("8. Parameter sweeps", H2))
        for name, df in sweeps.items():
            if df is None or len(df) == 0:
                continue
            story.append(Paragraph(f"<b>{name}</b> ({len(df)} points)", BODY))
            show = df.head(40).copy()
            for c in show.columns:
                if show[c].dtype.kind in "fc":
                    show[c] = show[c].map(lambda v: f"{v:.4g}")
            show = show[[c for c in show.columns if not c.startswith("_")]]
            story.append(table(show.iloc[:, :6], fontsize=5.8))
            if len(df) > 40:
                story.append(Paragraph(
                    f"(first 40 of {len(df)} rows shown; the full table is in "
                    f"the CSV/Excel export)", SMALL))
            story.append(Spacer(1, 6))

    # --- 9. figures ------------------------------------------------------
    if figures:
        story.append(PageBreak())
        story.append(Paragraph("9. Figures", H2))
        for title, fig in figures:
            png = figure_to_png_bytes(fig, width=1200, height=680, scale=1.6)
            if png is None:
                story.append(Paragraph(
                    f"[{title}: static image export unavailable - install "
                    f"`kaleido` to embed figures]", SMALL))
                continue
            story.append(KeepTogether([
                Paragraph(f"<b>{title}</b>", BODY),
                Image(io.BytesIO(png), width=168 * mm, height=95 * mm),
                Spacer(1, 8)]))

    # --- 10. assumptions -------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("10. Assumptions and limitations", H2))
    for a in ASSUMPTIONS:
        story.append(Paragraph(f"&bull; {a}", BODY))

    story.append(Paragraph("11. References", H2))
    for line in C.__doc__.split("References")[-1].strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("-"):
            story.append(Paragraph(line, SMALL))

    doc.build(story)
    return buf.getvalue()
