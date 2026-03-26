"""Main pipeline orchestration — parse config, run all analysis stages, write outputs.

This module is the only place that knows how all the sub-modules fit together.
Extending the pipeline means adding imports here and inserting new calls into
``analyze()`` without touching the sub-modules themselves.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None  # type: ignore[assignment]

from .bayes import (
    build_hdi_scalar_summary,
    compute_ph_distance_posterior_hdi,
    compute_ph_draws_for_unit,
    compute_pointwise_hdi,
    compute_td_credible_bands,
    run_bayesian_inference,
)
from .config import (
    AEConfig,
    BayesConfig,
    BayesResult,
    InterpConfig,
    InverseConfig,
    PHResult,
    RunConfig,
    S2PBundle,
    TDAConfig,
    TDConfig,
    TopologyDescriptor,
    VectorFitState,
    VFConfig,
)
from .interpolation import build_common_grid, interpolate_bundle
from .io import load_s2p
from .metrics import (
    choose_scalar_series,
    extract_trace,
    frequency_metrics,
    mag_db,
    reciprocity_error,
    robust_group_delay,
)
from .ml import get_torch_device, set_seed, train_autoencoder, train_inverse_model
from .plotting import (
    _compute_voxel_sv,
    save_3d_voxel_heatmap,
    save_barcode_plot,
    save_complex_plane_plots,
    save_cross_unit_anomaly_summary,
    save_cross_unit_complex_overlay,
    save_cross_unit_delta_panel,
    save_cross_unit_h2_panel,
    save_cross_unit_h3_panel,
    save_cross_unit_s21_envelope,
    save_cross_unit_s21_overlay,
    save_cross_unit_vf_pole_scatter,
    save_gd_credible_band_plot,
    save_gng_plot,
    save_heatmap,
    save_persistence_diagram_plot,
    save_ppc_plot,
    save_s21_credible_band_plot,
    save_shift_register_plot,
    save_td_credible_band_plot,
    save_topomap_complex_plane_density,
    save_topomap_shift_register_density,
    save_topomap_unit_vs_freq,
    save_voronoi_plot,
)
from .time_domain import time_domain_from_trace
from .topology import (
    PH_AVAILABLE as _PH_AVAILABLE,
)
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

# ---------------------------------------------------------------------------
# Integrity helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """SHA-256 hash of a file, read in 64 KiB chunks to keep memory flat."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _collect_dep_versions() -> dict[str, str]:
    """Snapshot installed versions of key dependencies for the run manifest."""
    versions: dict[str, str] = {}
    for mod_name in [
        "numpy", "scipy", "skrf", "pandas", "matplotlib",
        "ripser", "persim", "torch", "sklearn",
    ]:
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = "not_installed"
    return versions


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

_NESTED_DC_MAP: dict[str, type] = {
    "InterpConfig": InterpConfig,
    "TDConfig": TDConfig,
    "TDAConfig": TDAConfig,
    "VFConfig": VFConfig,
    "AEConfig": AEConfig,
    "InverseConfig": InverseConfig,
    "BayesConfig": BayesConfig,
}


def nested_dataclass_from_dict(cls: type, payload: dict[str, Any]) -> RunConfig:
    """Recursively build a ``RunConfig`` from a plain dict (YAML-loaded)."""
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        ann = cls.__annotations__.get(key)
        ann_name = ann if isinstance(ann, str) else getattr(ann, "__name__", None)
        if isinstance(value, dict) and ann_name in _NESTED_DC_MAP:
            kwargs[key] = _NESTED_DC_MAP[ann_name](**value)
        else:
            kwargs[key] = value
    return cls(**kwargs)  # type: ignore[return-value]


def load_config(path: str | None) -> RunConfig:
    """Parse a YAML config file and return a validated ``RunConfig``."""
    if path is None:
        raise ValueError("A YAML config path is required for this script.")
    if yaml is None:
        raise RuntimeError("PyYAML is required to load a YAML config. Install it first.")
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh)
    return nested_dataclass_from_dict(RunConfig, payload)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def analyze(config: RunConfig) -> None:  # noqa: C901  (complexity expected — orchestration)
    """Run the full RF analysis pipeline described by *config*."""
    set_seed(config.seed)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("metrics", "time_domain", "tda", "plots", "vector_fit", "ml", "ph", "bayes"):
        (out_dir / sub).mkdir(exist_ok=True)

    _bayes_rng = np.random.default_rng(config.seed)
    bayes_results_by_unit: dict[str, BayesResult] = {}
    ph_posterior_draws_by_unit: dict[str, list[list[np.ndarray] | None]] = {}

    # ---- Load and interpolate to a common frequency grid ----
    bundles = [load_s2p(p) for p in config.input_files]
    common_f = build_common_grid(bundles, config.interpolation)

    bundles_i: list[S2PBundle] = []
    interp_meta_rows: list[dict[str, Any]] = []
    for b in bundles:
        bi, meta = interpolate_bundle(b, common_f, config.interpolation)
        bundles_i.append(bi)
        interp_meta_rows.append({"unit": b.name, **meta})
    pd.DataFrame(interp_meta_rows).to_csv(out_dir / "interpolation_summary.csv", index=False)

    # ---- Per-unit state ----
    metrics_by_unit: dict[str, pd.DataFrame] = {}
    vf_states: list[VectorFitState] = []
    topology_complex: dict[str, TopologyDescriptor] = {}
    topology_shift: dict[str, TopologyDescriptor] = {}
    tda_feature_bank: dict[str, np.ndarray] = {}
    lag_summary: list[dict[str, Any]] = []
    ph_results: dict[str, PHResult | None] = {}
    clouds_complex: dict[str, np.ndarray] = {}
    clouds_shift: dict[str, np.ndarray] = {}
    lag_by_unit: dict[str, int] = {}

    # ---- Ingestion diagnostics ----
    diag_rows: list[dict[str, Any]] = []
    for b in bundles_i:
        df_arr = np.diff(b.freq_hz)
        diag_rows.append({
            "unit": b.name,
            "n_points": int(len(b.freq_hz)),
            "freq_min_hz": float(b.freq_hz.min()),
            "freq_max_hz": float(b.freq_hz.max()),
            "df_mean_hz": float(df_arr.mean()),
            "df_std_hz": float(df_arr.std()),
            "df_max_deviation_hz": float(np.max(np.abs(df_arr - df_arr.mean()))),
            "reciprocity_max": float(reciprocity_error(b.s).max()),
            "touchstone_version": b.touchstone_meta.get("version", "1.1"),
            "freq_unit": b.touchstone_meta.get("freq_unit", "GHz"),
            "parameter": b.touchstone_meta.get("parameter", "S"),
            "data_format": b.touchstone_meta.get("data_format", "MA"),
            "z0_ohm": b.touchstone_meta.get("z0_ohm", "50.0"),
            "two_port_order": b.touchstone_meta.get("two_port_order", "N11_N21_N12_N22"),
            "two_port_order_source": b.touchstone_meta.get("two_port_order_source", "v1.1_default"),
        })
    pd.DataFrame(diag_rows).to_csv(out_dir / "ingestion_diagnostics.csv", index=False)

    # =====================================================================
    # Per-unit analysis loop
    # =====================================================================
    for idx, bundle in enumerate(bundles_i, start=1):
        unit_label = f"Unit {idx}"

        # -- Frequency-domain metrics --
        metrics_df = frequency_metrics(bundle)
        metrics_by_unit[bundle.name] = metrics_df
        metrics_df.to_csv(
            out_dir / "metrics" / f"{unit_label.lower().replace(' ', '_')}_frequency_metrics.csv",
            index=False,
        )

        # -- Time-domain impulse response --
        td_trace = extract_trace(bundle.s, config.time_domain.target_trace)
        td_df = time_domain_from_trace(bundle.freq_hz, td_trace, config.time_domain)
        td_df.to_csv(
            out_dir / "time_domain" / f"{unit_label.lower().replace(' ', '_')}_time_domain.csv",
            index=False,
        )

        # -- Bayesian inference (SR-004, SR-005) --
        if config.bayes.enabled:
            _br = run_bayesian_inference(bundle, config.bayes, _bayes_rng)
            bayes_results_by_unit[bundle.name] = _br
            _s21_obs_db = mag_db(extract_trace(bundle.s, "S21"))
            _s21_draws_db = mag_db(_br["s_draws"][:, :, 1, 0])
            _s21_med, _s21_lo, _s21_hi = compute_pointwise_hdi(_s21_draws_db, config.bayes.cred_mass)
            save_s21_credible_band_plot(
                bundle.freq_hz, _s21_obs_db, _s21_med, _s21_lo, _s21_hi,
                config.bayes.cred_mass, out_dir / "plots", unit_label,
            )
            save_ppc_plot(
                bundle.freq_hz, _s21_obs_db, _br["ppc_samples_db"],
                out_dir / "plots", unit_label,
            )
            _gd_draws = np.stack(
                [robust_group_delay(bundle.freq_hz, _br["s_draws"][d, :, 1, 0])
                 for d in range(len(_br["s_draws"]))], axis=0,
            )
            _gd_med, _gd_lo, _gd_hi = compute_pointwise_hdi(_gd_draws * 1e9, config.bayes.cred_mass)
            save_gd_credible_band_plot(
                bundle.freq_hz, _gd_med, _gd_lo, _gd_hi,
                config.bayes.cred_mass, out_dir / "plots", unit_label,
            )
            _ts, _imp_med, _imp_lo, _imp_hi = compute_td_credible_bands(
                _br["s_draws"], bundle.freq_hz, (1, 0), config.time_domain, config.bayes.cred_mass,
            )
            save_td_credible_band_plot(
                _ts, _imp_med, _imp_lo, _imp_hi,
                config.bayes.cred_mass, out_dir / "plots", unit_label,
            )
            _hdi_scalars = build_hdi_scalar_summary(
                _br["s_draws"], bundle.freq_hz, config.bayes.cred_mass,
            )
            pd.DataFrame([{"unit": unit_label, **_hdi_scalars}]).to_csv(
                out_dir / "bayes" / f"{unit_label.lower().replace(' ', '_')}_hdi_scalar_summary.csv",
                index=False,
            )
            pd.DataFrame({
                "freq_hz": bundle.freq_hz,
                "s21_db_median": _s21_med,
                "s21_db_hdi_low": _s21_lo,
                "s21_db_hdi_high": _s21_hi,
                "gd_ns_median": _gd_med,
                "gd_ns_hdi_low": _gd_lo,
                "gd_ns_hdi_high": _gd_hi,
            }).to_csv(
                out_dir / "bayes" / f"{unit_label.lower().replace(' ', '_')}_credible_bands.csv",
                index=False,
            )
            if _PH_AVAILABLE and config.bayes.ph_posterior_n_draws > 0:
                ph_posterior_draws_by_unit[bundle.name] = compute_ph_draws_for_unit(
                    _br["s_draws"], config.tda, config.bayes.ph_posterior_n_draws,
                    _bayes_rng, config.tda.complex_traces,
                )

        # -- Complex-plane scatter (C1) --
        save_complex_plane_plots(bundle, metrics_df, out_dir / "plots", unit_label)

        # -- Shift-register scatter (S1) --
        scalar = choose_scalar_series(metrics_df, config.tda.shift_scalar)
        lag = config.tda.shift_lag or choose_lag_autocorr(
            scalar, config.tda.shift_lag_min, config.tda.shift_lag_max,
            window=config.tda.shift_window,
        )
        lag_summary.append({
            "unit": unit_label, "shift_scalar": config.tda.shift_scalar, "lag": lag,
        })
        lag_by_unit[bundle.name] = lag
        save_shift_register_plot(scalar, lag, out_dir / "plots", unit_label, config.tda.shift_scalar)

        # -- Complex topology cloud --
        complex_cloud, complex_index, complex_axes = build_complex_topology_cloud(
            metrics_df, config.tda,
        )
        complex_descriptor = build_topology_descriptor(
            complex_cloud, complex_index, bundle.freq_hz, complex_axes, config.tda,
        )
        topology_complex[bundle.name] = complex_descriptor
        _save_topology_outputs(complex_descriptor, out_dir, unit_label, "complex")
        save_voronoi_plot(
            complex_cloud, complex_descriptor["voronoi_artifacts"],
            complex_descriptor["axis_labels"], out_dir / "plots", unit_label, "complex",
        )
        save_gng_plot(
            complex_cloud, complex_descriptor["gng_state"],
            complex_descriptor["axis_labels"], out_dir / "plots", unit_label, "complex",
        )
        save_heatmap(
            complex_descriptor["gng_artifacts"]["transition_matrix"],
            out_dir / "plots" / f"{unit_label.lower().replace(' ', '_')}_complex_gng_transition_matrix.png",
            f"GNG Transition Matrix (complex) - {unit_label}",
        )
        clouds_complex[bundle.name] = complex_cloud

        # -- Shift topology cloud --
        shift_cloud, shift_index, shift_axes = build_shift_topology_cloud(
            metrics_df, lag, config.tda,
        )
        shift_descriptor = build_topology_descriptor(
            shift_cloud, shift_index, bundle.freq_hz, shift_axes, config.tda,
        )
        topology_shift[bundle.name] = shift_descriptor
        _save_topology_outputs(shift_descriptor, out_dir, unit_label, "shift")
        save_voronoi_plot(
            shift_cloud, shift_descriptor["voronoi_artifacts"],
            shift_descriptor["axis_labels"], out_dir / "plots", unit_label, "shift",
        )
        save_gng_plot(
            shift_cloud, shift_descriptor["gng_state"],
            shift_descriptor["axis_labels"], out_dir / "plots", unit_label, "shift",
        )
        save_heatmap(
            shift_descriptor["gng_artifacts"]["transition_matrix"],
            out_dir / "plots" / f"{unit_label.lower().replace(' ', '_')}_shift_gng_transition_matrix.png",
            f"GNG Transition Matrix (shift) - {unit_label}",
        )
        clouds_shift[bundle.name] = shift_cloud

        # -- TDA feature vectors --
        feat_complex = make_tda_feature_vector(metrics_df, complex_descriptor, config.tda)
        feat_shift = make_tda_feature_vector(metrics_df, shift_descriptor, config.tda)
        tda_feature_bank[bundle.name] = np.concatenate([feat_complex, feat_shift])

        # -- Vector fitting --
        vf_state = fit_vector_model(bundle, config.vector_fit)
        vf_states.append(vf_state)
        n_poles_real = int(np.sum(np.isclose(vf_state.poles.imag, 0.0)))
        n_poles_cmplx = int(np.sum(vf_state.poles.imag > 0.0))
        vf_summary: dict[str, Any] = {
            "unit": unit_label,
            "model_order": n_poles_real + 2 * n_poles_cmplx,
            "n_poles_real": n_poles_real,
            "n_poles_complex": n_poles_cmplx,
            "rms_error": vf_state.rms_error,
            "passive_before": vf_state.passive_before,
            "passive_after": vf_state.passive_after,
        }
        pd.DataFrame([vf_summary]).to_csv(
            out_dir / "vector_fit" / f"{unit_label.lower().replace(' ', '_')}_vf_summary.csv",
            index=False,
        )
        np.savez(
            out_dir / "vector_fit" / f"{unit_label.lower().replace(' ', '_')}_vf_params.npz",
            poles=vf_state.poles,
            residues=vf_state.residues,
            constants=vf_state.constant_coeff,
            proportionals=vf_state.proportional_coeff,
        )

        # -- Persistent homology (SR-006) --
        ph_complex = compute_ph_diagrams(complex_cloud, config.tda)
        ph_shift = compute_ph_diagrams(shift_cloud, config.tda)
        ph_results[bundle.name] = ph_complex
        if ph_complex is not None:
            save_persistence_diagram_plot(ph_complex["dgms"], out_dir / "ph", f"{unit_label} complex")
            save_barcode_plot(ph_complex["dgms"], out_dir / "ph", f"{unit_label} complex")
        if ph_shift is not None:
            save_persistence_diagram_plot(ph_shift["dgms"], out_dir / "ph", f"{unit_label} shift")
            save_barcode_plot(ph_shift["dgms"], out_dir / "ph", f"{unit_label} shift")

    pd.DataFrame(lag_summary).to_csv(out_dir / "tda" / "shift_lag_summary.csv", index=False)

    _lag_global = int(np.median(list(lag_by_unit.values()))) if lag_by_unit else 4

    # =====================================================================
    # Cross-unit analysis
    # =====================================================================

    # H1 — unit × frequency 3D surface
    save_topomap_unit_vs_freq(metrics_by_unit, bundles_i, bundles_i[0].freq_hz, out_dir / "plots")

    # Compute globally consistent vmin/vmax before rendering per-unit density plots
    def _sv_range(
        clouds: dict[str, np.ndarray],
        values_fn: Callable[[S2PBundle], np.ndarray],
        metric_name: str,
    ) -> tuple[float, float]:
        all_sv_parts = [
            _compute_voxel_sv(clouds[b.name], values_fn(b), config.tda.heatmap_bins, metric_name)
            for b in bundles_i
        ]
        filled = [v for v in all_sv_parts if len(v)]
        if not filled:
            return 0.0, 1.0
        combined = np.concatenate(filled)
        return float(combined.min()), float(combined.max())

    _cd_vmin, _cd_vmax = _sv_range(
        clouds_complex,
        lambda b: np.ones(len(clouds_complex[b.name]), dtype=np.float64),
        "density",
    )
    _cr_vmin, _cr_vmax = _sv_range(
        clouds_complex,
        lambda b: np.asarray(
            topology_complex[b.name]["voronoi_artifacts"]["point_values"], dtype=np.float64,
        ),
        "regime_intensity",
    )
    _sd_vmin, _sd_vmax = _sv_range(
        clouds_shift,
        lambda b: np.ones(len(clouds_shift[b.name]), dtype=np.float64),
        "density",
    )
    _sr_vmin, _sr_vmax = _sv_range(
        clouds_shift,
        lambda b: np.asarray(
            topology_shift[b.name]["voronoi_artifacts"]["point_values"], dtype=np.float64,
        ),
        "regime_intensity",
    )

    _h2_hists: list[float] = []
    _h3_hists: list[float] = []
    for _b in bundles_i:
        _m = metrics_by_unit[_b.name]
        _s21 = 10.0 ** (_m["s21_db"].values / 20.0)
        if len(_s21) > _lag_global:
            _H2, _, _ = np.histogram2d(_s21[:-_lag_global], _s21[_lag_global:], bins=60)
            _h2_hists.append(float(np.log1p(_H2.max())))
        _H3, _, _ = np.histogram2d(_m["s21_re"].values, _m["s21_im"].values, bins=60)
        _h3_hists.append(float(np.log1p(_H3.max())))
    _h2_vmax = max(_h2_hists) if _h2_hists else 1.0
    _h3_vmax = max(_h3_hists) if _h3_hists else 1.0

    # Render per-unit 3D density plots with shared colour scales
    for _idx, _bundle in enumerate(bundles_i, start=1):
        _ulabel = f"Unit {_idx}"
        save_3d_voxel_heatmap(
            clouds_complex[_bundle.name],
            np.ones(len(clouds_complex[_bundle.name]), dtype=np.float64),
            topology_complex[_bundle.name]["axis_labels"],
            out_dir / "plots", _ulabel, "complex", "density",
            config.tda.heatmap_bins, "cividis",
            vmin=_cd_vmin, vmax=_cd_vmax,
        )
        save_3d_voxel_heatmap(
            clouds_complex[_bundle.name],
            np.asarray(
                topology_complex[_bundle.name]["voronoi_artifacts"]["point_values"], dtype=np.float64,
            ),
            topology_complex[_bundle.name]["axis_labels"],
            out_dir / "plots", _ulabel, "complex", "regime_intensity",
            config.tda.heatmap_bins, "cividis",
            vmin=_cr_vmin, vmax=_cr_vmax,
        )
        save_3d_voxel_heatmap(
            clouds_shift[_bundle.name],
            np.ones(len(clouds_shift[_bundle.name]), dtype=np.float64),
            topology_shift[_bundle.name]["axis_labels"],
            out_dir / "plots", _ulabel, "shift", "density",
            config.tda.heatmap_bins, "cividis",
            vmin=_sd_vmin, vmax=_sd_vmax,
        )
        save_3d_voxel_heatmap(
            clouds_shift[_bundle.name],
            np.asarray(
                topology_shift[_bundle.name]["voronoi_artifacts"]["point_values"], dtype=np.float64,
            ),
            topology_shift[_bundle.name]["axis_labels"],
            out_dir / "plots", _ulabel, "shift", "regime_intensity",
            config.tda.heatmap_bins, "cividis",
            vmin=_sr_vmin, vmax=_sr_vmax,
        )
        save_topomap_shift_register_density(
            metrics_by_unit[_bundle.name], _lag_global,
            out_dir / "plots", _ulabel,
            vmin=0.0, vmax=_h2_vmax,
        )
        save_topomap_complex_plane_density(
            metrics_by_unit[_bundle.name],
            out_dir / "plots", _ulabel,
            vmin=0.0, vmax=_h3_vmax,
        )

    # Cross-unit overlay and panel figures
    save_cross_unit_s21_overlay(metrics_by_unit, bundles_i, out_dir / "plots")
    save_cross_unit_s21_envelope(metrics_by_unit, bundles_i, out_dir / "plots")
    save_cross_unit_delta_panel(metrics_by_unit, bundles_i, out_dir / "plots")
    save_cross_unit_complex_overlay(metrics_by_unit, bundles_i, out_dir / "plots")
    save_cross_unit_h2_panel(metrics_by_unit, bundles_i, _lag_global, out_dir / "plots")
    save_cross_unit_h3_panel(metrics_by_unit, bundles_i, out_dir / "plots")
    save_cross_unit_vf_pole_scatter(vf_states, bundles_i, bundles_i[0].freq_hz, out_dir / "plots")

    # ---- PH inter-unit bottleneck distance matrix ----
    ph_dist_df = compute_ph_distance_matrix(ph_results, dim=1)
    if ph_dist_df is not None:
        ph_dist_df.to_csv(out_dir / "ph" / "ph_bottleneck_distance_dim1.csv")
        save_heatmap(
            ph_dist_df,
            out_dir / "plots" / "ph_bottleneck_distance_dim1.png",
            "PH Bottleneck Distance (H1) — All Units",
        )

    # ---- PH distance posterior HDI ----
    if config.bayes.enabled and ph_posterior_draws_by_unit and _PH_AVAILABLE:
        _unit_names_list = [b.name for b in bundles_i]
        _ph_dist_hdi = compute_ph_distance_posterior_hdi(
            ph_posterior_draws_by_unit, _unit_names_list, config.bayes.cred_mass, dim=1,
        )
        if _ph_dist_hdi is not None:
            _ph_dist_hdi.to_csv(
                out_dir / "ph" / "ph_bottleneck_distance_dim1_hdi.csv", index=False,
            )

    # ---- GNG / Voronoi inter-unit distance matrices ----
    topology_distances: dict[str, Any] = {
        "voronoi_distance_complex": compare_topology_features(
            [np.asarray(list(topology_complex[b.name]["voronoi_summary"].values()), dtype=np.float64)
             for b in bundles_i],
        ),
        "gng_distance_complex": compare_topology_features(
            [np.asarray(list(topology_complex[b.name]["gng_summary"].values()), dtype=np.float64)
             for b in bundles_i],
        ),
        "voronoi_distance_shift": compare_topology_features(
            [np.asarray(list(topology_shift[b.name]["voronoi_summary"].values()), dtype=np.float64)
             for b in bundles_i],
        ),
        "gng_distance_shift": compare_topology_features(
            [np.asarray(list(topology_shift[b.name]["gng_summary"].values()), dtype=np.float64)
             for b in bundles_i],
        ),
    }
    for name, df in topology_distances.items():
        df.to_csv(out_dir / "tda" / f"{name}.csv")
        save_heatmap(df, out_dir / "plots" / f"{name}.png", name.replace("_", " ").title())

    # ---- Aggregate anomaly summary ----
    _all_distance_dfs: dict[str, Any] = {"ph_bottleneck_h1": ph_dist_df, **topology_distances}
    save_cross_unit_anomaly_summary(bundles_i, _all_distance_dfs, out_dir / "plots")

    # ---- Aggregate frequency-domain summary ----
    rows: list[dict[str, Any]] = []
    for idx, bundle in enumerate(bundles_i, start=1):
        m = metrics_by_unit[bundle.name]
        rows.append({
            "unit": f"Unit {idx}",
            "freq_min_hz": float(bundle.freq_hz.min()),
            "freq_max_hz": float(bundle.freq_hz.max()),
            "n_points": int(len(bundle.freq_hz)),
            "s21_db_max": float(m["s21_db"].max()),
            "s21_db_min": float(m["s21_db"].min()),
            "group_delay_mean_s": float(m["group_delay_s21_s"].mean()),
            "group_delay_std_s": float(m["group_delay_s21_s"].std()),
            "passivity_margin_min": float(m["passivity_margin"].min()),
            "reciprocity_max": float(m["reciprocity_abs_s21_minus_s12"].max()),
        })
    pd.DataFrame(rows).to_csv(out_dir / "summary_frequency_metrics.csv", index=False)

    # ---- TDA topology feature table ----
    topology_feature_rows: list[dict[str, Any]] = []
    for idx, bundle in enumerate(bundles_i, start=1):
        combined = np.concatenate([
            topology_complex[bundle.name]["feature"],
            topology_shift[bundle.name]["feature"],
        ])
        topology_feature_rows.append({
            "unit": f"Unit {idx}",
            **{f"topology_feature_{k}": float(v) for k, v in enumerate(combined)},
        })
    pd.DataFrame(topology_feature_rows).to_csv(
        out_dir / "ml" / "topology_inverse_features.csv", index=False,
    )

    # ---- ML extension 1: window autoencoder ----
    ae_descriptors: dict[str, np.ndarray] | None = None
    if config.autoencoder.enabled:
        ae_descriptors = train_autoencoder(metrics_by_unit, config.autoencoder, out_dir / "ml")
        ae_rows: list[dict[str, Any]] = []
        for idx, bundle in enumerate(bundles_i, start=1):
            ae_rows.append({
                "unit": f"Unit {idx}",
                **{f"ae_{k}": float(v) for k, v in enumerate(ae_descriptors[bundle.name])},
            })
        pd.DataFrame(ae_rows).to_csv(out_dir / "ml" / "autoencoder_unit_descriptors.csv", index=False)

    # ---- ML extension 2: inverse characterisation regressor ----
    if config.inverse.enabled:
        x_inv, y_inv, meta = build_inverse_dataset(
            bundles_i, vf_states, config.tda, config.inverse, ae_descriptors,
        )
        np.save(out_dir / "ml" / "inverse_features.npy", x_inv)
        np.save(out_dir / "ml" / "inverse_targets.npy", y_inv)
        with (out_dir / "ml" / "inverse_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        inv_info = train_inverse_model(x_inv, y_inv, config.inverse, out_dir / "ml")
        with (out_dir / "ml" / "inverse_model_info.json").open("w", encoding="utf-8") as fh:
            json.dump(inv_info, fh, indent=2)

    # ---- Run manifest (SHA-256 provenance, versions, config snapshot) ----
    manifest: dict[str, Any] = {
        "run_id": f"run_{config.seed}",
        "seed": config.seed,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": get_torch_device(),
        "dependency_versions": _collect_dep_versions(),
        "ph_available": _PH_AVAILABLE,
        "input_files": {
            f"Unit_{i + 1}": {
                "path": str(Path(p)),
                "sha256": _sha256_file(Path(p)),
            }
            for i, p in enumerate(config.input_files)
        },
        "config": asdict(config),
        "units": [f"Unit {i + 1}" for i in range(len(bundles_i))],
        "ingestion_diagnostics": diag_rows,
        "bayesian_noise_sigma": {
            bundle.name: {
                "sigma_re_S21": float(bayes_results_by_unit[bundle.name]["sigma_re"][1, 0]),
                "sigma_im_S21": float(bayes_results_by_unit[bundle.name]["sigma_im"][1, 0]),
                "cred_mass": bayes_results_by_unit[bundle.name]["cred_mass"],
                "n_draws": int(len(bayes_results_by_unit[bundle.name]["s_draws"])),
            }
            for bundle in bundles_i
            if bundle.name in bayes_results_by_unit
        } if config.bayes.enabled else {},
    }
    _key_outputs = [
        out_dir / "summary_frequency_metrics.csv",
        out_dir / "ingestion_diagnostics.csv",
        out_dir / "interpolation_summary.csv",
        out_dir / "ph" / "ph_bottleneck_distance_dim1.csv",
    ]
    manifest["output_sha256"] = {
        str(_op.relative_to(out_dir)): _sha256_file(_op)
        for _op in _key_outputs if _op.exists()
    }
    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ---------------------------------------------------------------------------
# Private helper — write topology CSV outputs
# ---------------------------------------------------------------------------

def _save_topology_outputs(
    descriptor: TopologyDescriptor,
    out_dir: Path,
    unit_label: str,
    topology_label: str,
) -> None:
    """Write TDA CSV outputs for one topology cloud (complex or shift)."""
    slug = unit_label.lower().replace(" ", "_")
    tda_dir = out_dir / "tda"
    pd.DataFrame([descriptor["voronoi_summary"]]).to_csv(
        tda_dir / f"{slug}_{topology_label}_voronoi_summary.csv", index=False,
    )
    pd.DataFrame([descriptor["gng_summary"]]).to_csv(
        tda_dir / f"{slug}_{topology_label}_gng_summary.csv", index=False,
    )
    pd.DataFrame([descriptor["gng_transition_summary"]]).to_csv(
        tda_dir / f"{slug}_{topology_label}_gng_transition_summary.csv", index=False,
    )
    descriptor["voronoi_artifacts"]["point_table"].to_csv(
        tda_dir / f"{slug}_{topology_label}_voronoi_points.csv", index=False,
    )
    descriptor["gng_artifacts"]["occupancy"].to_csv(
        tda_dir / f"{slug}_{topology_label}_gng_occupancy.csv", index=False,
    )
    descriptor["gng_artifacts"]["assignments"].to_csv(
        tda_dir / f"{slug}_{topology_label}_gng_assignments.csv", index=False,
    )
    descriptor["gng_artifacts"]["transition_matrix"].to_csv(
        tda_dir / f"{slug}_{topology_label}_gng_transition_matrix.csv",
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def make_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="S2P Frequency-Time-Topology + ML pipeline")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    return p


def main() -> None:
    args = make_argparser().parse_args()
    cfg = load_config(args.config)
    analyze(cfg)


if __name__ == "__main__":
    main()
