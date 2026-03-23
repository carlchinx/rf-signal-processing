"""Configuration dataclasses, typed data containers, and project-wide constants.

All pipeline-stage hyperparameters are defined here so they can be loaded from
a single YAML file via ``runner.load_config``.

Extending the pipeline
-----------------------
To add a new stage:
  1. Define a ``@dataclass`` class below (e.g. ``MyStageConfig``).
  2. Add a field for it in ``RunConfig``.
  3. Register it in ``runner._NESTED_DC_MAP`` so YAML deserialisation works.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, TypedDict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# S-parameter index map
# ---------------------------------------------------------------------------

TRACE_TO_INDEX: Final[dict[str, tuple[int, int]]] = {
    "S11": (0, 0), "S12": (0, 1), "S21": (1, 0), "S22": (1, 1)
}
ROW_MAJOR_TRACE_ORDER: Final[list[str]] = ["S11", "S12", "S21", "S22"]


def _validate_trace(trace: str) -> None:
    if trace not in TRACE_TO_INDEX:
        raise ValueError(
            f"Unknown S-parameter trace '{trace}'. Valid: {list(TRACE_TO_INDEX.keys())}"
        )


# ---------------------------------------------------------------------------
# Stage configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class InterpConfig:
    method: Literal["complex_linear", "complex_cubic", "magphase_pchip"] = "magphase_pchip"
    n_points: int | None = None
    grid_mode: Literal["intersection", "custom"] = "intersection"
    custom_start_hz: float | None = None
    custom_stop_hz: float | None = None


@dataclass
class TDConfig:
    target_trace: str = "S21"
    n_fft: int | None = None
    window_name: str = "tukey"
    window_kwargs: dict[str, Any] = field(default_factory=lambda: {"alpha": 0.2})
    dc_handling: Literal["zero_fill", "hold_first"] = "zero_fill"
    hermitian_real_signal: bool = False
    continuous_scale_df: bool = False


@dataclass
class TDAConfig:
    complex_traces: tuple[str, ...] = ("S11", "S21")
    shift_scalar: Literal["S21_db", "S11_db", "group_delay"] = "S21_db"
    shift_window: int = 64
    shift_lag: int | None = None
    shift_lag_min: int = 4
    shift_lag_max: int | None = 128
    shift_stride: int = 1
    max_points: int = 512
    subsample: Literal["uniform", "curvature"] = "curvature"
    voronoi_qhull_options: str = "Qbb Qc Qz Qx"
    voronoi_volume_clip_quantile: float = 0.95
    gng_steps: int = 1200
    gng_max_nodes: int = 48
    gng_lambda: int = 40
    gng_max_age: int = 60
    gng_eps_winner: float = 0.05
    gng_eps_neighbor: float = 0.006
    gng_alpha: float = 0.5
    gng_beta: float = 0.995
    heatmap_bins: int = 12
    ph_enabled: bool = True
    ph_maxdim: int = 2
    ph_thresh_quantile: float = 0.9
    ph_max_points: int = 256


@dataclass
class VFConfig:
    n_poles_real: int = 2
    n_poles_cmplx: int = 4
    fit_constant: bool = True
    fit_proportional: bool = False
    passivity_enforce: bool = True
    passivity_n_samples: int = 600


@dataclass
class AEConfig:
    enabled: bool = True
    window_size: int = 64
    stride: int = 8
    latent_dim: int = 16
    hidden_dim: int = 256
    batch_size: int = 256
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-4
    noise_std: float = 0.01
    num_workers: int = 0
    compile_model: bool = False


@dataclass
class InverseConfig:
    enabled: bool = True
    n_synthetic_per_unit: int = 500
    target_latent_dim: int = 12
    batch_size: int = 256
    epochs: int = 120
    lr: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 256
    pole_real_sigma: float = 0.08
    pole_imag_sigma: float = 0.08
    residue_sigma: float = 0.10
    constant_sigma: float = 0.05
    proportional_sigma: float = 0.02
    passivity_reject_tol: float = 0.03
    compile_model: bool = False


@dataclass
class BayesConfig:
    """Bayesian uncertainty-quantification configuration (SR-004)."""
    enabled: bool = True
    engine: Literal["numpy_gaussian"] = "numpy_gaussian"
    likelihood: Literal["complex_gaussian", "complex_studentt"] = "complex_gaussian"
    n_draws: int = 400
    cred_mass: float = 0.94
    n_ppc_samples: int = 50
    latent: Literal["none", "savgol_smooth"] = "savgol_smooth"
    smooth_window: int = 21
    smooth_polyorder: int = 3
    ph_posterior_n_draws: int = 80


@dataclass
class RunConfig:
    input_files: list[str]
    output_dir: str
    seed: int = 12345
    interpolation: InterpConfig = field(default_factory=InterpConfig)
    time_domain: TDConfig = field(default_factory=TDConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    vector_fit: VFConfig = field(default_factory=VFConfig)
    autoencoder: AEConfig = field(default_factory=AEConfig)
    inverse: InverseConfig = field(default_factory=InverseConfig)
    bayes: BayesConfig = field(default_factory=BayesConfig)


# ---------------------------------------------------------------------------
# Data container dataclasses
# ---------------------------------------------------------------------------

@dataclass
class S2PBundle:
    name: str
    path: Path
    freq_hz: np.ndarray
    s: np.ndarray          # shape (N, 2, 2) complex128
    z0: np.ndarray
    raw_option_line: str
    comments: list[str]
    touchstone_meta: dict[str, str] = field(default_factory=dict)


@dataclass
class VectorFitState:
    name: str
    vf: Any                # VectorFitting instance (skrf); typed Any to avoid import coupling
    poles: np.ndarray
    residues: np.ndarray
    constant_coeff: np.ndarray
    proportional_coeff: np.ndarray
    rms_error: float | None
    passive_before: bool | None
    passive_after: bool | None
    violation_bands_before: np.ndarray | None


@dataclass
class GNGState:
    nodes: np.ndarray
    errors: np.ndarray
    edges: list[tuple[int, int]]


# ---------------------------------------------------------------------------
# TypedDict schemas for structured dict return values
# ---------------------------------------------------------------------------

class VoronoiArtifacts(TypedDict):
    ridge_points: np.ndarray
    point_values: np.ndarray
    point_table: pd.DataFrame


class VoronoiResult(TypedDict):
    summary: dict[str, float]
    artifacts: VoronoiArtifacts


class GNGTransitionArtifacts(TypedDict):
    assignments: pd.DataFrame
    occupancy: pd.DataFrame
    transition_matrix: pd.DataFrame


class TopologyDescriptor(TypedDict):
    cloud: np.ndarray
    source_index: np.ndarray
    axis_labels: tuple[str, str, str]
    voronoi_summary: dict[str, float]
    voronoi_artifacts: VoronoiArtifacts
    gng_state: GNGState
    gng_summary: dict[str, float]
    gng_transition_summary: dict[str, float]
    gng_artifacts: GNGTransitionArtifacts
    feature: np.ndarray


class PHResult(TypedDict):
    dgms: list[np.ndarray]
    cocycles: list[Any]
    dist_matrix: np.ndarray
    idx_perm: np.ndarray
    r_cover: float


class InverseMeta(TypedDict):
    source_unit: list[str]


class InverseModelInfo(TypedDict):
    x_scaler_mean: list[float]
    x_scaler_scale: list[float]
    pca_explained_variance_ratio: list[float]
    latent_dim: int
    n_train: int


class BayesResult(TypedDict):
    """Posterior sampling result from the complex-Gaussian Bayesian module (SR-004)."""
    s_draws: np.ndarray         # (n_draws, Nf, 2, 2) complex128
    sigma_re: np.ndarray        # (2, 2) float64 — Re noise sigma per S-param
    sigma_im: np.ndarray        # (2, 2) float64 — Im noise sigma per S-param
    ppc_samples_db: np.ndarray  # (n_ppc, Nf) float64 — posterior predictive S21 dB
    cred_mass: float
