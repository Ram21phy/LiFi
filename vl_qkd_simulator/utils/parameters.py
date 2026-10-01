"""
utils/parameters.py
===================
Single, typed container for every user-adjustable quantity in the simulator.

Design rule: *no* module anywhere else in the code base is allowed to invent a
default.  Everything a model needs comes from here, so a run is fully described
(and therefore reproducible) by one JSON blob.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Dict, List, Optional, Tuple

from utils import constants as C


# ==========================================================================
#  Room / geometry
# ==========================================================================
@dataclass
class RoomParams:
    """Indoor room geometry. All lengths in metres."""
    length_x: float = 5.0
    width_y: float = 5.0
    height_z: float = 3.0

    # Alice (transmitter) position; z measured from the floor.
    # The demonstration geometry places Alice and Bob as two wall/ceiling
    # mounted terminals exactly 5.000 m apart: (4, 3, 0) m separation.
    tx_x: float = 0.5
    tx_y: float = 1.0
    tx_z: float = 2.8

    # Bob (receiver) position
    rx_x: float = 4.5
    rx_y: float = 4.0
    rx_z: float = 2.8

    # Orientation of the transmitter optical axis (unit vector, will be
    # normalised).  Default: pointing from Alice towards Bob ("steered link").
    tx_aim_at_rx: bool = True
    tx_normal: Tuple[float, float, float] = (0.0, 0.0, -1.0)

    # Orientation of the receiver optical axis
    rx_aim_at_tx: bool = True
    rx_normal: Tuple[float, float, float] = (0.0, 0.0, 1.0)

    # Surface reflectivities used by the simplified multi-reflection
    # ("integrating sphere") background model
    wall_reflectivity: float = 0.70
    ceiling_reflectivity: float = 0.80
    floor_reflectivity: float = 0.30

    # If the user prefers to drive the model by link distance rather than by
    # explicit coordinates, this override is applied *after* geometry.
    override_distance: bool = False
    link_distance_m: float = 5.0


# ==========================================================================
#  Source / photon statistics
# ==========================================================================
@dataclass
class SourceParams:
    wavelength_nm: float = 520.0          # green; within the VLC window
    mu: float = 0.50                      # mean photon number per signal pulse
    pulse_rate_hz: float = 50e6           # pulse repetition rate r_p
    source_linewidth_nm: float = 0.02     # spectral width of the quantum signal

    # Weak-coherent-pulse assumption (Poisson photon statistics) vs. an ideal
    # single-photon source (n == 1 deterministically).
    source_model: str = "WCP (Poisson)"   # or "Ideal single photon"

    # Decoy intensities (used only in decoy-state mode)
    decoy_nu1: float = 0.10
    decoy_nu2: float = 0.0


# ==========================================================================
#  Optical channel
# ==========================================================================
@dataclass
class ChannelParams:
    # --- model selection -------------------------------------------------
    channel_model: str = "Collimated Gaussian beam"  # or "Lambertian (VLC LOS)"

    # --- Lambertian transmitter -----------------------------------------
    tx_half_power_semiangle_deg: float = 15.0     # Phi_1/2 -> Lambertian order m
    lambertian_order_override: Optional[float] = None

    # --- Gaussian-beam transmitter --------------------------------------
    beam_waist_mm: float = 2.0                    # w0 at the transmitter
    beam_divergence_mrad: float = 2.0             # full-angle-ish half divergence

    # --- receiver optics -------------------------------------------------
    rx_aperture_area_cm2: float = 1.0             # physical detector/optics area A
    rx_fov_deg: float = 5.0                       # half-angle FOV Psi_c
    concentrator_index: float = C.CONCENTRATOR_INDEX_DEFAULT
    use_concentrator: bool = False
    optical_efficiency: float = 0.80              # eta_opt (lenses, BS, PBS, ...)

    # --- extra, user-specified loss --------------------------------------
    extra_loss_db: float = 0.0

    # --- optional impairments -------------------------------------------
    enable_pointing_error: bool = False
    pointing_jitter_mrad: float = 0.5             # sigma of angular jitter
    enable_turbulence: bool = False
    turbulence_cn2: float = 1e-14                 # m^-2/3, indoor: 1e-15..1e-13


# ==========================================================================
#  Spectral filter
# ==========================================================================
@dataclass
class FilterParams:
    center_nm: float = 520.0
    bandwidth_nm: float = 0.10                    # Delta-lambda (FWHM)
    peak_transmission: float = 0.90               # T_filter at the centre
    shape: str = "Gaussian"                       # "Gaussian" | "Top-hat" | "Lorentzian"
    out_of_band_rejection_db: float = 60.0        # blocking floor


# ==========================================================================
#  Background illumination
# ==========================================================================
@dataclass
class LEDBackgroundParams:
    enabled: bool = True
    n_luminaires: int = 4
    optical_power_per_luminaire_w: float = 0.25   # radiometric optical watts
    drive_by_illuminance: bool = True
    target_illuminance_lux: float = 10.0          # dimmed room (demonstration)
    semiangle_deg: float = 60.0                   # ceiling luminaire Lambertian
    mount_height_m: float = 2.9                   # just under the ceiling
    # luminaire (x, y) positions; if empty, a symmetric grid is generated
    positions: List[Tuple[float, float]] = field(default_factory=list)
    include_reflections: bool = True


@dataclass
class FluorescentBackgroundParams:
    enabled: bool = False
    n_luminaires: int = 2
    optical_power_per_luminaire_w: float = 6.0
    semiangle_deg: float = 60.0
    mount_height_m: float = 2.9
    positions: List[Tuple[float, float]] = field(default_factory=list)
    include_reflections: bool = True


@dataclass
class SunlightBackgroundParams:
    enabled: bool = False
    outdoor_irradiance_w_m2: float = 500.0        # global horizontal irradiance
    window_area_m2: float = 2.0
    window_transmission: float = 0.75
    window_distance_m: float = 3.0                # window -> receiver
    direct_sun_in_fov: bool = False               # worst case: sun disc in FOV


@dataclass
class UserBackgroundParams:
    enabled: bool = False
    mode: str = "Background count rate [counts/s]"
    # one of: "Background optical power [W]",
    #         "Background photon flux [photons/s]",
    #         "Background count rate [counts/s]",
    #         "Spectral irradiance [W/m^2/nm]"
    value: float = 1.0e3


@dataclass
class BackgroundParams:
    led: LEDBackgroundParams = field(default_factory=LEDBackgroundParams)
    fluorescent: FluorescentBackgroundParams = field(default_factory=FluorescentBackgroundParams)
    sunlight: SunlightBackgroundParams = field(default_factory=SunlightBackgroundParams)
    user: UserBackgroundParams = field(default_factory=UserBackgroundParams)
    # isotropic diffuse floor, in case the user wants to short-circuit the
    # source-by-source build-up with a single literature number
    use_isotropic_preset: bool = False
    isotropic_preset: str = "Moderate indoor illumination"
    isotropic_spectral_irradiance_w_m2_nm: float = 5.8e-4


# ==========================================================================
#  Detector
# ==========================================================================
@dataclass
class DetectorParams:
    efficiency: float = 0.60                      # eta_det
    dark_count_rate_hz: float = 100.0             # per detector
    dead_time_ns: float = 45.0
    gate_width_ns: float = 0.20                   # Delta-t detection gate
    afterpulse_probability: float = 0.01
    timing_jitter_ps: float = 350.0
    n_detectors_per_basis: int = 2                # standard passive BB84 receiver
    # "Gated" restricts noise integration to the gate; "Free-running" integrates
    # over the full pulse period (worse background).
    operation_mode: str = "Gated"


# ==========================================================================
#  Polarisation / encoding errors
# ==========================================================================
@dataclass
class PolarizationParams:
    misalignment_angle_deg: float = 2.0           # basis frame rotation
    extinction_ratio_db: float = 25.0             # polariser / PBS extinction
    depolarization: float = 0.005                 # channel depolarising fraction
    intrinsic_error_override: Optional[float] = None   # bypass with a direct e_d


# ==========================================================================
#  Protocol / security
# ==========================================================================
@dataclass
class ProtocolParams:
    protocol: str = "Decoy-state BB84"            # or "Standard BB84 (GLLP)"
    basis_bias_z: float = 0.5                     # p_Z; 0.5 = symmetric, q = 1/2
    use_efficient_bb84: bool = False              # q -> pz^2 + (1-pz)^2
    error_correction_efficiency: float = 1.10     # f_EC
    finite_key: bool = False
    block_size_bits: float = 1e7                  # N sifted bits for finite-size
    epsilon_sec: float = 1e-10
    epsilon_cor: float = 1e-15
    external_qber_enabled: bool = False
    external_qber: float = 0.03


# ==========================================================================
#  Monte-Carlo controls
# ==========================================================================
@dataclass
class MonteCarloParams:
    n_pulses: int = 100_000
    seed: int = 12345
    chunk_size: int = 500_000
    convergence_points: int = 25


# ==========================================================================
#  Master container
# ==========================================================================
@dataclass
class SimulationParameters:
    room: RoomParams = field(default_factory=RoomParams)
    source: SourceParams = field(default_factory=SourceParams)
    channel: ChannelParams = field(default_factory=ChannelParams)
    filt: FilterParams = field(default_factory=FilterParams)
    background: BackgroundParams = field(default_factory=BackgroundParams)
    detector: DetectorParams = field(default_factory=DetectorParams)
    polarization: PolarizationParams = field(default_factory=PolarizationParams)
    protocol: ProtocolParams = field(default_factory=ProtocolParams)
    monte_carlo: MonteCarloParams = field(default_factory=MonteCarloParams)

    label: str = ("DEMONSTRATION PARAMETERS - illustrative only, "
                  "not experimentally validated")

    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SimulationParameters":
        def build(cls, sub):
            if sub is None:
                return cls()
            names = {f.name for f in fields(cls)}
            kwargs = {}
            for k, v in sub.items():
                if k not in names:
                    continue
                kwargs[k] = v
            return cls(**kwargs)

        bg_raw = d.get("background", {}) or {}
        background = BackgroundParams(
            led=build(LEDBackgroundParams, bg_raw.get("led")),
            fluorescent=build(FluorescentBackgroundParams, bg_raw.get("fluorescent")),
            sunlight=build(SunlightBackgroundParams, bg_raw.get("sunlight")),
            user=build(UserBackgroundParams, bg_raw.get("user")),
        )
        for k in ("use_isotropic_preset", "isotropic_preset",
                  "isotropic_spectral_irradiance_w_m2_nm"):
            if k in bg_raw:
                setattr(background, k, bg_raw[k])

        return SimulationParameters(
            room=build(RoomParams, d.get("room")),
            source=build(SourceParams, d.get("source")),
            channel=build(ChannelParams, d.get("channel")),
            filt=build(FilterParams, d.get("filt")),
            background=background,
            detector=build(DetectorParams, d.get("detector")),
            polarization=build(PolarizationParams, d.get("polarization")),
            protocol=build(ProtocolParams, d.get("protocol")),
            monte_carlo=build(MonteCarloParams, d.get("monte_carlo")),
            label=d.get("label", "Loaded parameters"),
        )

    @staticmethod
    def from_json(text: str) -> "SimulationParameters":
        return SimulationParameters.from_dict(json.loads(text))

    # ------------------------------------------------------------------
    def copy(self) -> "SimulationParameters":
        return SimulationParameters.from_dict(json.loads(self.to_json()))


def default_parameters() -> SimulationParameters:
    """The labelled demonstration scenario (5 m x 5 m x 3 m room, 520 nm)."""
    return SimulationParameters()
