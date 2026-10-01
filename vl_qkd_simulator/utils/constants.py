"""
utils/constants.py
==================
Physical constants and literature reference values used throughout the
indoor Visible-Light BB84 QKD simulator.

Every constant carries units and, where it is a *literature* value rather than
a fundamental constant, an explicit citation.  Nothing in this file is tuned to
make plots look good.

References
----------
[1] O. Elmabrok and M. Razavi, "Wireless quantum key distribution in indoor
    environments", J. Opt. Soc. Am. B 35, 197 (2018); arXiv:1605.05092.
[2] J. R. Barry, J. M. Kahn, et al., "Simulation of multipath impulse response
    for indoor wireless optical channels", IEEE JSAC 11, 367 (1993); and
    J. M. Kahn and J. R. Barry, "Wireless infrared communications",
    Proc. IEEE 85, 265 (1997).
[3] T. Komine and M. Nakagawa, "Fundamental analysis for visible-light
    communication system using LED lights", IEEE Trans. Consum. Electron. 50,
    100 (2004).
[4] A. J. C. Moreira, R. T. Valadas, A. M. de Oliveira Duarte, "Optical
    interference produced by artificial light", Wireless Networks 3, 131 (1997).
[5] X. Ma, B. Qi, Y. Zhao, H.-K. Lo, "Practical decoy state for quantum key
    distribution", Phys. Rev. A 72, 012326 (2005).
[6] D. Gottesman, H.-K. Lo, N. Lutkenhaus, J. Preskill (GLLP), "Security of
    quantum key distribution with imperfect devices", QIC 4, 325 (2004).
[7] C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, "Concise security
    bounds for practical decoy-state quantum key distribution",
    Phys. Rev. A 89, 022307 (2014).
[8] A. A. Farid and S. Hranilovic, "Outage capacity optimization for free-space
    optical links with pointing errors", J. Lightwave Technol. 25, 1702 (2007).
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# Fundamental physical constants (CODATA 2018, SI units)
# --------------------------------------------------------------------------
PLANCK_H = 6.62607015e-34        # J s        - Planck constant (exact)
SPEED_OF_LIGHT = 2.99792458e8    # m/s        - speed of light in vacuum (exact)
BOLTZMANN_K = 1.380649e-23       # J/K        - Boltzmann constant (exact)
ELEMENTARY_CHARGE = 1.602176634e-19  # C       - elementary charge (exact)
STEFAN_BOLTZMANN = 5.670374419e-8    # W m^-2 K^-4

# Convenience
HC = PLANCK_H * SPEED_OF_LIGHT   # J m
NM = 1e-9                        # m per nm
PS = 1e-12                       # s per ps
NS = 1e-9                        # s per ns

# --------------------------------------------------------------------------
# Information-theoretic constants
# --------------------------------------------------------------------------
E0_RANDOM_ERROR = 0.5            # error probability of a *random* (noise) click
QBER_ABORT_THRESHOLD = 0.11      # asymptotic BB84 one-way-EC threshold, h(e)=1/2
QBER_ABORT_THRESHOLD_SIXSTATE = 0.126

# --------------------------------------------------------------------------
# Background-illumination reference spectral irradiances
# Values from the indoor optical-wireless literature [1,4]. They are *defaults*,
# fully user-overridable in the GUI, and are quoted here in SI (W m^-2 nm^-1).
# The often-quoted "5.8e-6 W cm^-2 nm^-1" for bright direct sunlight equals
# 5.8e-2 W m^-2 nm^-1.
# --------------------------------------------------------------------------
SPECTRAL_IRRADIANCE_PRESETS_W_M2_NM = {
    "Dark room (no illumination)":      0.0,
    "Low indoor illumination":          5.8e-5,
    "Moderate indoor illumination":     5.8e-4,   # ~typical office artificial light
    "High indoor illumination":         5.8e-3,
    "Skylight through window":          5.8e-3,
    "Bright direct sunlight":           5.8e-2,   # 5.8e-6 W/cm^2/nm  [4]
}

# Typical total illuminance targets (EN 12464-1 office lighting standard)
ILLUMINANCE_PRESETS_LUX = {
    "Corridor":         100.0,
    "Living room":      150.0,
    "Classroom":        300.0,
    "Office (standard)": 500.0,
    "Drawing office":   750.0,
    "Laboratory":      1000.0,
}

# Luminous efficacy of radiation for a white phosphor LED, lm/W_optical.
# (Radiometric -> photometric conversion; ~260-300 lm/W_opt for 4000-5000 K.)
WHITE_LED_LER_LM_PER_W_OPT = 280.0
FLUORESCENT_LER_LM_PER_W_OPT = 330.0
SOLAR_LER_LM_PER_W_OPT = 93.0          # AM1.5 global, lm per optical watt

# --------------------------------------------------------------------------
# Spectral models (normalised line shapes are built in background_noise.py)
# --------------------------------------------------------------------------
# Phosphor-converted white LED: blue pump + broad yellow phosphor
WHITE_LED_SPECTRUM = {
    "blue_peak_nm": 450.0, "blue_fwhm_nm": 22.0, "blue_weight": 0.32,
    "phosphor_peak_nm": 560.0, "phosphor_fwhm_nm": 115.0, "phosphor_weight": 0.68,
}

# Tri-phosphor fluorescent lamp: Hg lines + broad phosphor bands
FLUORESCENT_LINES_NM = [404.7, 435.8, 546.1, 578.0, 611.6]
FLUORESCENT_LINE_WEIGHTS = [0.04, 0.16, 0.23, 0.10, 0.18]
FLUORESCENT_LINE_FWHM_NM = 3.0
FLUORESCENT_CONTINUUM = {"peak_nm": 545.0, "fwhm_nm": 130.0, "weight": 0.29}
FLUORESCENT_MODULATION_HZ = 100.0   # 2 x mains (50 Hz); 120 Hz for 60 Hz mains

SOLAR_BLACKBODY_T_K = 5778.0        # effective solar photosphere temperature

# --------------------------------------------------------------------------
# Wavelength grid used for all spectral integrations
# --------------------------------------------------------------------------
SPECTRUM_LAMBDA_MIN_NM = 300.0
SPECTRUM_LAMBDA_MAX_NM = 1100.0
SPECTRUM_N_POINTS = 1601            # 0.5 nm resolution

# --------------------------------------------------------------------------
# Refractive index of a typical compound-parabolic concentrator
# --------------------------------------------------------------------------
CONCENTRATOR_INDEX_DEFAULT = 1.5    # [2,3]

# --------------------------------------------------------------------------
# Result provenance labels, used everywhere in the GUI so that no number is
# ever displayed without saying how it was obtained.
# --------------------------------------------------------------------------
class Provenance:
    ANALYTIC = "analytic"
    MONTE_CARLO = "Monte Carlo"
    ASYMPTOTIC = "asymptotic"
    FINITE_SIZE = "finite-size (simplified)"
    SIMPLIFIED_DETECTOR = "simplified detector model"
    SIMPLIFIED_BACKGROUND = "simplified background model"


DISCLAIMER = (
    "All results are numerical simulation outputs from simplified physical and "
    "information-theoretic models. They are intended for preliminary design "
    "studies and are NOT an experimental demonstration or a proof of "
    "unconditional security of any physical device."
)
