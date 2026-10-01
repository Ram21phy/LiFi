"""
models/photon_model.py
======================
Weak-coherent-pulse (WCP) source model with Poisson photon statistics.

Physics
-------
Photon energy                E_ph = h c / lambda                        [J]
Photon-number distribution   P(n) = exp(-mu) mu^n / n!
Mean transmitted photon rate Phi_tx = mu * r_p                     [photons/s]
Mean transmitted optical power P_tx = mu * r_p * E_ph                   [W]

Multiphoton fraction (the quantity that limits GLLP security):
    P_multi(mu) = 1 - e^{-mu} - mu e^{-mu}

An "Ideal single photon" option is also provided, in which every pulse carries
exactly one photon (P(1) = 1).  This is *not* physical for an LED/laser source
but is useful as an upper-bound reference curve.

All results produced by this module are ANALYTIC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.stats import poisson

from utils.constants import HC, NM


# --------------------------------------------------------------------------
def photon_energy(wavelength_nm: float) -> float:
    """E_ph = hc/lambda  [J].  wavelength in nm."""
    if wavelength_nm <= 0:
        raise ValueError("wavelength must be positive")
    return HC / (wavelength_nm * NM)


def photon_energy_ev(wavelength_nm: float) -> float:
    from utils.constants import ELEMENTARY_CHARGE
    return photon_energy(wavelength_nm) / ELEMENTARY_CHARGE


def poisson_pmf(n, mu: float):
    """P(n) = exp(-mu) mu^n / n!  (vectorised, numerically safe)."""
    return poisson.pmf(n, mu)


def multiphoton_probability(mu: float) -> float:
    """P(n >= 2) for a Poissonian source."""
    return float(max(0.0, 1.0 - np.exp(-mu) - mu * np.exp(-mu)))


def single_photon_probability(mu: float) -> float:
    return float(mu * np.exp(-mu))


def vacuum_probability(mu: float) -> float:
    return float(np.exp(-mu))


# --------------------------------------------------------------------------
@dataclass
class PhotonModelResult:
    wavelength_nm: float
    photon_energy_j: float
    photon_energy_ev: float
    mu: float
    pulse_rate_hz: float
    transmitted_photon_rate: float      # photons/s
    transmitted_optical_power_w: float
    p_vacuum: float
    p_single: float
    p_multi: float
    provenance: str = "analytic"

    def as_dict(self) -> Dict[str, float]:
        d = self.__dict__.copy()
        return d


def evaluate_source(source_params) -> PhotonModelResult:
    """Build the transmitter-side photon budget from SourceParams."""
    lam = float(source_params.wavelength_nm)
    e_ph = photon_energy(lam)
    mu = float(source_params.mu)
    rp = float(source_params.pulse_rate_hz)

    if source_params.source_model.startswith("Ideal"):
        p0, p1, pm = 0.0, 1.0, 0.0
        mean_n = 1.0
    else:
        p0 = vacuum_probability(mu)
        p1 = single_photon_probability(mu)
        pm = multiphoton_probability(mu)
        mean_n = mu

    return PhotonModelResult(
        wavelength_nm=lam,
        photon_energy_j=e_ph,
        photon_energy_ev=photon_energy_ev(lam),
        mu=mu,
        pulse_rate_hz=rp,
        transmitted_photon_rate=mean_n * rp,
        transmitted_optical_power_w=mean_n * rp * e_ph,
        p_vacuum=p0,
        p_single=p1,
        p_multi=pm,
    )


# --------------------------------------------------------------------------
def sample_photon_numbers(rng: np.random.Generator, mu: float, size: int,
                          ideal_single: bool = False) -> np.ndarray:
    """Draw photon numbers for `size` pulses."""
    if ideal_single:
        return np.ones(size, dtype=np.int64)
    return rng.poisson(mu, size=size)


def received_mean_photons(mu: float, eta_sys: float) -> float:
    """Mean photon number that survives to the detector: mu * eta_sys."""
    return mu * eta_sys


def click_probability_given_n(n, eta_sys: float):
    """P_sig(click | n) = 1 - (1 - eta_sys)^n."""
    return 1.0 - np.power(1.0 - eta_sys, n)


def signal_click_probability_poisson(mu: float, eta_sys: float) -> float:
    """
    Average of 1-(1-eta)^n over a Poisson(mu):
        sum_n P(n) [1 - (1-eta)^n] = 1 - exp(-eta*mu)
    """
    return float(1.0 - np.exp(-eta_sys * mu))
