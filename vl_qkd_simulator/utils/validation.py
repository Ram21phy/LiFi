"""
utils/validation.py
===================
Parameter validation.  Returns *messages*, never exceptions, so that the GUI
can keep running and show the user exactly what is inconsistent.

Severities:  "error"   - the model cannot be trusted with this input
             "warning" - physically allowed but probably not what you meant
             "info"    - worth knowing
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from models.vlc_channel import compute_geometry, lambertian_order


@dataclass
class Issue:
    severity: str
    field: str
    message: str


def _in_room(room, x, y, z) -> bool:
    return (0.0 <= x <= room.length_x and 0.0 <= y <= room.width_y
            and 0.0 <= z <= room.height_z)


def validate(params) -> List[Issue]:
    out: List[Issue] = []
    r, s, c, f = params.room, params.source, params.channel, params.filt
    d, p, pr = params.detector, params.polarization, params.protocol

    # ---- room / geometry ------------------------------------------------
    for name, v in (("length_x", r.length_x), ("width_y", r.width_y),
                    ("height_z", r.height_z)):
        if v <= 0:
            out.append(Issue("error", name, "Room dimensions must be positive."))
    if not _in_room(r, r.tx_x, r.tx_y, r.tx_z):
        out.append(Issue("warning", "Tx position",
                         "Alice is outside the room volume."))
    if not _in_room(r, r.rx_x, r.rx_y, r.rx_z):
        out.append(Issue("warning", "Rx position",
                         "Bob is outside the room volume."))

    geom = compute_geometry(r, c.rx_fov_deg)
    if geom.distance_m < 0.05:
        out.append(Issue("warning", "distance",
                         "Alice and Bob are essentially co-located "
                         f"(d = {geom.distance_m:.3f} m)."))
    if not geom.in_fov:
        out.append(Issue("warning", "FOV",
                         f"Incidence angle psi = {geom.incidence_angle_deg:.1f} deg "
                         f"exceeds the receiver FOV of {c.rx_fov_deg:.1f} deg: "
                         f"the LOS channel gain is zero."))

    # ---- source ---------------------------------------------------------
    if s.wavelength_nm < 380 or s.wavelength_nm > 780:
        out.append(Issue("info", "wavelength",
                         f"{s.wavelength_nm:g} nm is outside the nominal visible "
                         f"band (380-780 nm). The model still runs, but this is "
                         f"no longer 'visible light'."))
    if s.mu <= 0:
        out.append(Issue("error", "mu", "Mean photon number must be positive."))
    elif s.mu > 1.0:
        out.append(Issue("warning", "mu",
                         f"mu = {s.mu:g} is large for BB84: the multi-photon "
                         f"fraction is {1-math.exp(-s.mu)-s.mu*math.exp(-s.mu):.2%}, "
                         f"which the GLLP bound concedes entirely to Eve."))
    if s.pulse_rate_hz <= 0:
        out.append(Issue("error", "pulse rate", "Pulse rate must be positive."))
    if s.decoy_nu1 >= s.mu and pr.protocol.startswith("Decoy"):
        out.append(Issue("error", "decoy",
                         "Decoy intensity nu1 must be strictly smaller than mu."))
    if s.decoy_nu2 >= s.decoy_nu1 and pr.protocol.startswith("Decoy"):
        out.append(Issue("error", "decoy",
                         "Decoy intensities must satisfy mu > nu1 > nu2 >= 0."))

    # ---- detector gating vs pulse period --------------------------------
    period_ns = 1e9 / max(s.pulse_rate_hz, 1e-30)
    if d.gate_width_ns > period_ns:
        out.append(Issue("warning", "gate width",
                         f"The detection gate ({d.gate_width_ns:g} ns) is longer "
                         f"than the pulse period ({period_ns:.3g} ns). Adjacent "
                         f"pulses overlap; reduce the gate or the rate."))
    if d.dead_time_ns > period_ns:
        out.append(Issue("info", "dead time",
                         f"Dead time ({d.dead_time_ns:g} ns) exceeds the pulse "
                         f"period ({period_ns:.3g} ns): the detector will "
                         f"saturate well below the nominal rate."))
    if not (0 < d.efficiency <= 1):
        out.append(Issue("error", "detector efficiency",
                         "Detector efficiency must lie in (0, 1]."))
    if d.dark_count_rate_hz < 0:
        out.append(Issue("error", "dark counts", "Dark count rate cannot be negative."))

    # ---- filter ---------------------------------------------------------
    if abs(f.center_nm - s.wavelength_nm) > 2.0 * max(f.bandwidth_nm, 1e-6):
        out.append(Issue("warning", "filter",
                         f"The filter centre ({f.center_nm:g} nm) is more than two "
                         f"bandwidths away from the signal ({s.wavelength_nm:g} nm); "
                         f"almost no signal will get through."))
    if f.bandwidth_nm <= 0:
        out.append(Issue("error", "filter bandwidth", "Bandwidth must be positive."))
    if f.bandwidth_nm < 0.05:
        out.append(Issue("info", "filter bandwidth",
                         "Sub-0.05 nm filters are at the edge of what is "
                         "commercially practical, and start to clip a pulsed "
                         "source's own spectrum."))
    # transform-limited check
    if s.source_linewidth_nm > f.bandwidth_nm:
        out.append(Issue("warning", "filter bandwidth",
                         f"The filter ({f.bandwidth_nm:g} nm) is narrower than the "
                         f"stated source linewidth ({s.source_linewidth_nm:g} nm): "
                         f"the model does NOT account for the resulting signal "
                         f"truncation."))
    if not (0 < f.peak_transmission <= 1):
        out.append(Issue("error", "filter transmission",
                         "Peak transmission must lie in (0, 1]."))

    # ---- optics ---------------------------------------------------------
    if not (0 < c.optical_efficiency <= 1):
        out.append(Issue("error", "optical efficiency",
                         "Optical efficiency must lie in (0, 1]."))
    if c.rx_fov_deg <= 0 or c.rx_fov_deg > 90:
        out.append(Issue("error", "FOV", "FOV half-angle must lie in (0, 90] deg."))
    if c.use_concentrator:
        n = c.concentrator_index
        psi = math.radians(c.rx_fov_deg)
        if n > 1.0 / max(math.sin(psi), 1e-12):
            out.append(Issue("info", "concentrator",
                             f"An ideal concentrator with n = {n:g} and "
                             f"FOV = {c.rx_fov_deg:g} deg gives a gain of "
                             f"{n**2/math.sin(psi)**2:.1f}x. This is the "
                             f"thermodynamic limit; real CPCs fall short of it."))
    m = lambertian_order(c.tx_half_power_semiangle_deg)
    if m > 5000:
        out.append(Issue("info", "semi-angle",
                         f"Semi-angle {c.tx_half_power_semiangle_deg:g} deg gives "
                         f"Lambertian order m = {m:.0f}. At this directivity the "
                         f"Lambertian model is being pushed well past its usual "
                         f"regime - consider the collimated-beam model instead."))

    # ---- polarisation ---------------------------------------------------
    if p.misalignment_angle_deg > 20:
        out.append(Issue("warning", "polarisation",
                         f"A {p.misalignment_angle_deg:g} deg misalignment gives an "
                         f"intrinsic error of {math.sin(math.radians(p.misalignment_angle_deg))**2:.1%}, "
                         f"which alone can exceed the BB84 threshold."))

    # ---- protocol -------------------------------------------------------
    if not (1.0 <= pr.error_correction_efficiency <= 3.0):
        out.append(Issue("warning", "f_EC",
                         "Error-correction efficiency below 1 is unphysical "
                         "(Shannon limit); above ~2 is unusually poor."))
    if pr.external_qber_enabled:
        out.append(Issue("info", "external QBER",
                         "External QBER mode bypasses the physical QBER model. "
                         "Use it only for comparison with measured data."))
    if not (0 <= pr.basis_bias_z <= 1):
        out.append(Issue("error", "basis bias", "p_Z must lie in [0, 1]."))
    if pr.use_efficient_bb84 and abs(pr.basis_bias_z - 0.5) < 1e-9:
        out.append(Issue("info", "basis bias",
                         "Efficient BB84 with p_Z = 0.5 gives q = 0.5, i.e. the "
                         "same sifting factor as the standard protocol."))

    # ---- turbulence -----------------------------------------------------
    if c.enable_turbulence:
        out.append(Issue("info", "turbulence",
                         "Indoor Cn^2 is typically 1e-15...1e-13 m^-2/3, giving a "
                         "Rytov variance far below 1 over a few metres. "
                         "Turbulence is normally negligible for this link."))

    return out


def has_errors(issues: List[Issue]) -> bool:
    return any(i.severity == "error" for i in issues)
