"""rf_pipeline — modular RF signal-processing pipeline.

Package layout
--------------
config.py       Configuration dataclasses, typed containers, project constants.
io.py           Touchstone / S2P file loading and validation.
interpolation.py  Frequency-grid unification and S-parameter interpolation.
metrics.py      Frequency-domain scalar metrics and window feature matrix.
time_domain.py  IFFT-based time-domain transforms.
topology.py     TDA embeddings: Voronoi, GNG, Persistent Homology.
vector_fit.py   Vector fitting and synthetic inverse-characterisation dataset.
ml.py           Window autoencoder and MLP inverse regressor (PyTorch).
bayes.py        Complex-Gaussian Bayesian posterior sampling and HDI utilities.
plotting.py     All visualisation / save_* functions.
runner.py       Top-level ``analyze()`` orchestration, config loader, CLI entry.

Adding a new stage
------------------
1. Add a ``@dataclass`` config class to ``config.py`` and register it in
   ``runner._NESTED_DC_MAP``.
2. Add the stage field to ``RunConfig`` in ``config.py``.
3. Implement your module (e.g. ``new_stage.py``) importing from ``config`` and
   whichever other sub-modules you need.
4. Call your stage inside ``runner.analyze()`` at the appropriate point.
5. Re-export public symbols below as needed.
"""

from .bayes import (
    build_hdi_scalar_summary,
    compute_ph_distance_posterior_hdi,
    compute_ph_draws_for_unit,
    compute_pointwise_hdi,
    compute_scalar_hdi_string,
    compute_td_credible_bands,
    run_bayesian_inference,
)
from .config import (
    ROW_MAJOR_TRACE_ORDER,
    TRACE_TO_INDEX,
    AEConfig,
    BayesConfig,
    BayesResult,
    GNGState,
    GNGTransitionArtifacts,
    InterpConfig,
    InverseConfig,
    InverseMeta,
    InverseModelInfo,
    PHResult,
    RunConfig,
    S2PBundle,
    TDAConfig,
    TDConfig,
    TopologyDescriptor,
    VectorFitState,
    VFConfig,
    VoronoiArtifacts,
    VoronoiResult,
)
from .interpolation import build_common_grid, interpolate_bundle
from .io import load_s2p
from .metrics import (
    build_window_feature_matrix,
    choose_scalar_series,
    extract_trace,
    frequency_metrics,
    mag_db,
)
from .ml import (
    MLPRegressorTorch,
    WindowAutoencoder,
    WindowDatasetTorch,
    get_torch_device,
    set_seed,
    train_autoencoder,
    train_inverse_model,
)
from .runner import analyze, load_config, main
from .time_domain import time_domain_from_trace
from .topology import (
    build_complex_topology_cloud,
    build_shift_topology_cloud,
    build_topology_descriptor,
    choose_lag_autocorr,
    compare_topology_features,
    compute_ph_diagrams,
    compute_ph_distance_matrix,
    make_tda_feature_vector,
)
from .vector_fit import build_inverse_dataset, fit_vector_model

__all__ = [
    # config
    "AEConfig", "BayesConfig", "BayesResult", "GNGState", "GNGTransitionArtifacts",
    "InverseConfig", "InverseMeta", "InverseModelInfo", "InterpConfig",
    "PHResult", "ROW_MAJOR_TRACE_ORDER", "RunConfig", "S2PBundle",
    "TDAConfig", "TDConfig", "TRACE_TO_INDEX", "TopologyDescriptor",
    "VFConfig", "VectorFitState", "VoronoiArtifacts", "VoronoiResult",
    # io
    "load_s2p",
    # interpolation
    "build_common_grid", "interpolate_bundle",
    # metrics
    "build_window_feature_matrix", "choose_scalar_series",
    "extract_trace", "frequency_metrics", "mag_db",
    # time_domain
    "choose_lag_autocorr", "time_domain_from_trace",
    # topology
    "build_complex_topology_cloud", "build_shift_topology_cloud",
    "build_topology_descriptor", "compare_topology_features",
    "compute_ph_diagrams", "compute_ph_distance_matrix", "make_tda_feature_vector",
    # vector_fit
    "build_inverse_dataset", "fit_vector_model",
    # ml
    "MLPRegressorTorch", "WindowAutoencoder", "WindowDatasetTorch",
    "get_torch_device", "set_seed", "train_autoencoder", "train_inverse_model",
    # bayes
    "build_hdi_scalar_summary", "compute_ph_distance_posterior_hdi",
    "compute_ph_draws_for_unit", "compute_pointwise_hdi",
    "compute_scalar_hdi_string", "compute_td_credible_bands",
    "run_bayesian_inference",
    # runner
    "analyze", "load_config", "main",
]
