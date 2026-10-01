# Indoor Visible-Light QKD (BB84) Simulator

A modular, research-oriented Python toolkit for simulating a polarisation-encoded
**BB84** quantum-key-distribution link over an **indoor visible-light channel**,
with a physically built-up model of the **background optical noise** produced by
room illumination.

The whole chain is modelled and exposed:

```
indoor room → VLC optical channel → signal photons → background photons
→ detector clicks → BB84 measurements → sifted key → QBER
→ mutual information → privacy amplification → secret key rate
```

> **Read this first.** Everything this software produces is a *numerical
> simulation output* from simplified physical and information-theoretic models.
> It is intended for preliminary design studies and teaching. It is **not** an
> experimental demonstration and it does **not** establish unconditional
> security of any physical device. Every result in the GUI is labelled with its
> model class: *analytic*, *Monte Carlo*, *asymptotic*, *finite-size
> (simplified)*, *simplified detector model*, *simplified background model*.

---

## Quick start

```bash
pip install -r requirements.txt
python -m utils.selftest        # 12 physics/limiting-behaviour checks
streamlit run app.py
```

or simply

```bash
./run.sh
```

The app opens at <http://localhost:8501>. Every physical, optical, channel,
detector and protocol parameter is in the left-hand control panel; every panel
on the right updates immediately.

Run the full regression suite with:

```bash
python -m pytest tests -q
```

---

## What the toolkit computes

| Quantity | Symbol | Where |
|---|---|---|
| Photon detection probability | `Q_µ` | Overview, QBER tab |
| Background counts | `R_bg`, `µ_bg` | Background tab |
| Signal-to-noise / signal-to-background ratio | `SNR`, `SBR` | Overview |
| Quantum bit error rate, fully decomposed | `E_µ` | QBER tab |
| Mutual information | `I(A:B) = 1 − h₂(E)` | QBER tab |
| Eve's information / privacy-amplification term | `PA` | QBER tab |
| Sifted key rate | `q·Q_µ·r_p` | Overview |
| Secret key rate | `R` | Overview |
| Secret key fraction | `R / R_sift` | Overview |

---

## Physics

### Source — weak coherent pulses

```
E_ph = h c / λ
P(n) = e^(−µ) µⁿ / n!
P(n ≥ 2) = 1 − e^(−µ) − µ e^(−µ)
Φ_tx = µ · r_p          P_tx = µ · r_p · E_ph
```

An *ideal single-photon* source is also selectable as a reference bound.

### Channel — two models, kept separate

**Lambertian indoor-VLC LOS DC gain** (Kahn & Barry; Komine & Nakagawa):

```
H(0) = (m+1)A/(2πd²) · cosᵐ(φ) · T_s(ψ) · g(ψ) · cos(ψ)     for 0 ≤ ψ ≤ Ψ_c
H(0) = 0                                                     otherwise
m    = −ln2 / ln(cos Φ½)
g(ψ) = n² / sin²(Ψ_c)          ideal non-imaging concentrator
```

`H(0)` is the fraction of transmitted optical power collected, so it is used
directly as the channel transmittance.

**Collimated Gaussian beam** (closer to a real free-space QKD transmitter):

```
w(d)  = w₀ + d·θ_div
T_geo = [1 − exp(−2a²/w(d)²)] · cos(ψ)
```

Optional impairments: pointing error / beam wander (Farid & Hranilovic
statistics, mean transmittance and Monte-Carlo sampling) and weak-turbulence
log-normal fading (Rytov variance `σ_R² = 1.23 Cn² k^(7/6) L^(11/6)`; indoor
values make this negligible, which the GUI says explicitly).

### Background optical noise — the core of the toolkit

Background is **never** an arbitrary QBER number. It is always built from
radiometry:

```
P_bg = ∫ p(λ) · A · T_f(λ) · G_coll dλ   ≈  p(λ₀) · A · Δλ · T_filter · G_coll
N_bg = P_bg · η_opt · η_det · Δt / E_ph(λ₀)        [counts per gate]
µ_bg = R_bg · Δt
P_bg(0) = e^(−µ_bg)      P_bg(click) = 1 − e^(−µ_bg)
µ_total_noise = µ_bg + µ_dark + µ_afterpulse
```

with

| Symbol | Meaning | Unit |
|---|---|---|
| `p(λ)` | spectral irradiance at the receiver aperture | W m⁻² nm⁻¹ |
| `A` | receiver aperture / detector area | m² |
| `T_f(λ)` | optical band-pass filter transmission | – |
| `G_coll` | background collection factor of the receiver optics | sr |
| `Δλ` | noise-equivalent filter bandwidth | nm |
| `Δt` | detection gate width | s |
| `E_ph` | photon energy at the signal wavelength | J |

**The collection factor is where the field of view really matters.** Two
regimes are implemented and selectable:

* **Bare detector:** `G_coll = π sin²Ψ_c` — narrowing the FOV directly
  suppresses a diffuse background.
* **Ideal non-imaging concentrator:** the gain `n²/sin²Ψ_c` exactly cancels the
  accepted solid angle, so `G_coll = π n²`, **independent of the FOV**. This is
  étendue conservation. Narrowing the FOV then helps only by geometrically
  excluding discrete sources — which the discrete-luminaire model handles
  explicitly. This is a real and often-overlooked effect, and exploring it is
  one of the things the toolkit is built for.

Background sources implemented:

* **A — LED room illumination.** Discrete ceiling luminaires with Lambertian
  emission and a phosphor-converted white spectrum (blue pump + broad phosphor
  band). LOS contribution through the same channel gain as the quantum link
  (only luminaires inside the FOV count), plus a diffuse multi-reflection term
  using an integrating-sphere estimate
  `E_diffuse = P_emitted·ρ / [S_room(1−ρ)]`. Power can be specified in optical
  watts or in **lux** (converted via the luminous efficacy of radiation).
* **B — Fluorescent lighting.** Mercury emission lines at 404.7, 435.8, 546.1,
  578.0, 611.6 nm plus a phosphor continuum. The 100/120 Hz intensity
  modulation is *not* time-resolved (mean rate only), and the GUI says so.
* **C — Sunlight through a window.** 5778 K blackbody spectrum scaled to the
  requested outdoor irradiance, attenuated by window transmission and geometry,
  with an optional "solar disc inside the FOV" worst case.
* **D — User-defined.** Direct entry of background optical power, photon flux,
  count rate, or spectral irradiance.
* **E — Isotropic literature preset.** A single flat `p_n` value; the built-in
  presets follow the indoor optical-wireless literature (e.g. 5.8 × 10⁻⁶
  W cm⁻² nm⁻¹ = 5.8 × 10⁻² W m⁻² nm⁻¹ for bright direct sunlight).

### Detector — simplified SPAD model

```
µ_bg  = R_bg·Δt        µ_dark = R_dark·Δt      µ_ap = p_ap·P_click(prev)
Y₀    = 1 − exp(−N_det·µ_noise)                (N_det = 2 for passive BB84)
P_sig(click|n) = 1 − (1−η_sys)ⁿ    →   1 − exp(−η_sys µ)  for Poisson µ
Q_µ   = 1 − (1 − Y₀)·exp(−η_sys µ)
R_meas = R_true / (1 + R_true τ_dead)          non-paralysable dead time
Δt_eff = √(Δt² + (2.355 σ_jitter)²)
```

### QBER — an exact decomposition

Signal and noise clicks are independent, giving three disjoint outcomes:

```
signal only : p_s(1−Y₀)   → error with probability e_d
noise  only : (1−p_s)Y₀   → error with probability ½
both        : p_s Y₀      → random assignment: e_d/2 + ¼

E_µ Q_µ = p_s(1−Y₀)e_d + (1−p_s)Y₀·½ + p_s Y₀(e_d/2 + ¼)
```

This is *exactly* the rule the event-by-event Monte Carlo applies, which is why
the analytic and Monte-Carlo QBER agree to within sampling noise (self-test 12).
The noise term is then split by `µ_bg : µ_dark : µ_ap`, and the signal term by
polarisation vs pointing, so the displayed budget sums exactly to the total.

Intrinsic polarisation error:

```
e_d = 1 − (1 − sin²θ)(1 − e_ER)(1 − p_dep/2),    e_ER = 1/(1 + 10^(ER/10))
```

### Security models — clearly separated

| Mode | `Q₁`, `e₁` from | Class |
|---|---|---|
| **Standard BB84 (GLLP)** | `Q₁ ≥ Q_µ − P(n≥2) − Y₀e^(−µ)`, `e₁ ≤ E_µQ_µ/Q₁` | asymptotic, pessimistic |
| **Decoy-state BB84** | vacuum + weak decoy bounds (Ma *et al.* 2005) evaluated on model-generated `Q_ν`, `E_ν` | asymptotic |
| **Infinite-decoy limit** | exact model `Y₁`, `e₁` | asymptotic idealisation |

```
q = ½ (symmetric)   or   p_Z² + (1−p_Z)² (biased / efficient)
R = q · r_p · { Q₁[1 − h₂(e₁)] − f_EC · Q_µ · h₂(E_µ) }
```

A **simplified finite-key** analysis (Hoeffding-bounded version of Lim *et al.*
2014) is available in its own tab and is labelled *finite-size (simplified)*
everywhere. It is deliberately kept out of `qkd/key_rate.py` so that asymptotic
and finite-length numbers can never be confused.

### Monte Carlo

Fully vectorised, chunked, seeded. For every pulse: Alice's basis and bit,
Poisson photon number, per-photon survival (`Binomial(n, η)`), optional
pointing and turbulence fading samples, signal click, polarisation error, Bob's
basis, independent background and dark clicks in each of Bob's two detectors,
double-click random assignment, sifting, QBER. 10⁷ pulses run in a few seconds.
A convergence trace with a ±1σ binomial envelope is plotted against the
analytic value.

---

## The GUI

| Tab | What it gives you |
|---|---|
| **Overview** | The full dashboard: channel, background, QKD and security blocks, a per-sifted-bit waterfall and the QBER donut |
| **Room** | 2-D top view and 3-D view of the room, plus a "where can Bob stand?" position map |
| **Background & spectrum** | Background spectral irradiance against the filter response, contribution by source, and a filter-bandwidth sweep |
| **QBER & information** | The full error budget, the information accounting, and the 12 BB84 protocol steps as configured |
| **Standard plots** | The 15 standard figures (key rate / QBER / mutual information vs distance, background rate, detector efficiency, filter bandwidth, µ and wavelength; SBR vs distance; secret fraction vs QBER) |
| **Research heat-maps** | The headline `R(distance, background count rate)` map plus QBER and `I(A:B)` companions, and a custom 2-D map builder |
| **Illumination comparison** | Cases 1–5 (no background → office lighting + daylight) plus your current settings, overlaid |
| **Filter optimisation** | Finds the bandwidth maximising `R` subject to a QBER constraint, clearly labelled a numerical optimum |
| **Monte Carlo** | Event-by-event run, convergence plot, event bookkeeping |
| **Parameter sweep** | Any of 21 independent variables, linear/log, all metrics computed automatically |
| **Finite key** | Simplified finite-size key length and a block-size scan |
| **Export & report** | CSV, Excel, JSON, PNG/HTML figures, and a structured PDF research report |
| **Model & self-test** | The 12 start-up tests with their numerical evidence, the equation list, result provenance, assumptions and references |

---

## Project layout

```
app.py                      Streamlit GUI (presentation only)
models/
  vlc_channel.py            Lambertian + Gaussian-beam channel, geometry
  photon_model.py           WCP / Poisson photon statistics
  background_noise.py       spectra, filter, collection factor, all sources
  detector.py               simplified SPAD model
  polarization.py           BB84 states and intrinsic error model
  pointing_error.py         Farid–Hranilovic pointing statistics
  turbulence.py             log-normal weak turbulence
qkd/
  bb84.py                   orchestrator: parameters → full result chain
  decoy_state.py            GLLP bounds + vacuum/weak decoy (Ma et al.)
  qber.py                   analytic QBER with an exact error budget
  information_metrics.py    h₂, I(A:B), Eve/PA terms
  key_rate.py               asymptotic secret key rate
simulation/
  monte_carlo.py            vectorised event-by-event BB84
  parameter_sweep.py        1-D and 2-D sweeps, filter optimisation
  finite_size.py            simplified finite-key analysis
visualization/
  theme.py                  one place that defines how every figure looks
  plots.py                  1-D figures
  room.py                   2-D / 3-D room views
  heatmaps.py               2-D maps
utils/
  constants.py              physical constants + cited literature values
  parameters.py             the single typed parameter container
  validation.py             parameter checks (errors / warnings / info)
  selftest.py               the 12 start-up tests
  export.py                 CSV / Excel / JSON / PNG / PDF report
data/default_parameters.json
tests/test_models.py        pytest regression suite
```

**Design rules enforced throughout**

1. No module invents a default — everything comes from `SimulationParameters`,
   so a run is fully described (and reproducible) by one JSON blob plus a seed.
2. The physical channel model and the QKD security model never touch. The
   security layer sees only `η_sys`, `µ_bg`, `µ_dark`, `e_d`.
3. The GUI computes nothing. It only formats what the packages return.
4. Every displayed number carries a provenance label.

---

## Default demonstration scenario

All values below are **demonstration parameters**, chosen to put the link at a
workable operating point so that the machinery is visible. They are *not*
experimentally validated and every one of them is adjustable.

| Parameter | Value |
|---|---|
| Room | 5 m × 5 m × 3 m |
| Alice → Bob | (0.5, 1.0, 2.8) → (4.5, 4.0, 2.8) m, i.e. **d = 5.000 m** |
| Wavelength | 520 nm |
| Source | weak coherent pulses, µ = 0.5, 50 MHz |
| Channel | collimated beam, w₀ = 2 mm, 2 mrad divergence |
| Receiver | 1 cm² aperture, 5° FOV, no concentrator, η_opt = 0.80 |
| Filter | 0.10 nm FWHM at 520 nm, T = 0.90, 60 dB blocking |
| Detector | η_det = 0.60, 100 counts/s dark, 0.2 ns gate, 45 ns dead time, 1% afterpulsing |
| Illumination | 4 ceiling LEDs at **10 lx** (heavily dimmed room) |
| Polarisation | 2° misalignment, 25 dB extinction, 0.5% depolarisation |
| Protocol | decoy-state BB84, ν₁ = 0.1, ν₂ = 0, f_EC = 1.10 |

which gives, at the time of writing, `QBER ≈ 6.5 %`, sifted rate ≈ 1.5 Mbit/s
and a secret key rate ≈ 8.8 × 10⁴ bit/s.

### A result worth noticing

Switch the illumination preset from *low* (10 lx) to *moderate* (150 lx) and the
key rate collapses. At 520 nm the signal sits in the middle of the phosphor
emission band of a white LED, so room lighting is the dominant noise term by
many orders of magnitude, and a passive spectral filter alone cannot recover
it. The **Background count rate vs wavelength** figure (Plot 13) shows the
mechanism directly: moving the signal out of the phosphor band changes the
background by decades. This is exactly the kind of question the toolkit exists
to answer, and the honest answer the model gives is that indoor VL-QKD under
full office illumination needs ultra-narrow filtering, short gates, a narrow
field of view — or a wavelength the lamps do not emit.

---

## Questions the toolkit is built to answer

* How does indoor ambient visible-light noise affect BB84 QKD?
* How does the background count rate affect QBER?
* How narrow should the optical filter be? (*Filter optimisation* tab)
* At what distance does the secret key rate become negligible? (*Plot 1*, and
  the `R = 0` contour on the headline heat-map)
* How does detector efficiency affect the secure key rate? (*Plot 7*)
* How does wavelength affect signal photons and background photons?
  (*Plots 12, 13*)
* How does the receiver FOV affect both signal collection and background noise?
  (*Parameter sweep* → Receiver FOV, with and without the concentrator)
* How does optical background noise change the information shared between
  Alice and Bob? (*Illumination comparison*)

---

## Self-test

The suite runs at launch and is also available standalone. Each test checks
either a closed-form result or a limiting behaviour the physics demands:

1. Photon energy equals `hc/λ` to machine precision
2. Lambertian gain matches the closed form, falls as `1/d²`, is exactly zero
   outside the FOV, and `m(60°) = 1`
3. Poisson sampling reproduces mean = variance = µ over 10⁶ draws
4. Background scales linearly with filter bandwidth (flat spectrum, top-hat
   filter) and → 0 as the bandwidth → 0; no sources → exactly zero
5. Click probability reduces to `1 − e^(−ηµ)`, verified against a direct
   Poisson sum; `η = 0` reduces the gain to `Y₀`
6. Basis and bit selection are unbiased; matched-basis fraction is ½
7. **As background → 0, QBER → the intrinsic error floor `e_d`**; as background
   grows, QBER → ½; QBER is monotone in `µ_bg`; the budget sums to the total
8. `h₂(0) = h₂(1) = 0`, `h₂(½) = 1`, symmetry, vector/scalar agreement
9. `I(A:B) = 1` at QBER 0 and `0` at QBER ½, monotone in between
10. Key rate → 1 bit/sifted bit in the ideal limit, ≈ 0 at the 11% threshold,
    negative and flagged insecure above it
11. **End-to-end:** received photons fall with distance; QBER rises and key
    rate falls with background; at high noise the key rate is ≤ 0; with all
    noise removed QBER equals `e_d` exactly
12. **Monte Carlo agrees with the analytic model** within statistical error

---

## References

1. O. Elmabrok and M. Razavi, *Wireless quantum key distribution in indoor
   environments*, J. Opt. Soc. Am. B **35**, 197 (2018); arXiv:1605.05092.
2. J. M. Kahn and J. R. Barry, *Wireless infrared communications*,
   Proc. IEEE **85**, 265 (1997); J. R. Barry *et al.*, IEEE JSAC **11**, 367 (1993).
3. T. Komine and M. Nakagawa, *Fundamental analysis for visible-light
   communication system using LED lights*, IEEE Trans. Consum. Electron.
   **50**, 100 (2004).
4. A. J. C. Moreira, R. T. Valadas, A. M. de Oliveira Duarte, *Optical
   interference produced by artificial light*, Wireless Networks **3**, 131 (1997).
5. X. Ma, B. Qi, Y. Zhao, H.-K. Lo, *Practical decoy state for quantum key
   distribution*, Phys. Rev. A **72**, 012326 (2005).
6. D. Gottesman, H.-K. Lo, N. Lütkenhaus, J. Preskill, *Security of quantum key
   distribution with imperfect devices*, QIC **4**, 325 (2004).
7. C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security
   bounds for practical decoy-state quantum key distribution*,
   Phys. Rev. A **89**, 022307 (2014).
8. A. A. Farid and S. Hranilovic, *Outage capacity optimization for free-space
   optical links with pointing errors*, J. Lightwave Technol. **25**, 1702 (2007).

---

## Licence and citation

Use it, modify it, publish preliminary studies with it. If you do, please state
clearly that the numbers are simulation results from a simplified model, cite
the primary references above for the underlying physics, and report the
parameter JSON and Monte-Carlo seed so the run can be reproduced.
