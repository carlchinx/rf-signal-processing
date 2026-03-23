"""Matplotlib plotting functions for per-unit and cross-unit RF analysis.

All functions write PNG files directly to disk — none return figure objects.
Output paths follow the ``{unit_label}_{descriptive_name}.png`` convention.

Extending
---------
Add new plot functions here following the ``save_*`` naming convention.
Each function should accept typed arrays / DataFrames and a ``Path`` output
directory, close the figure with ``plt.close(fig)`` before returning, and
produce a single file rather than a multi-page PDF (to stay composable).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize as _MplNormalize  # noqa: N813

import numpy as np
import pandas as pd

from .config import GNGState, S2PBundle, VectorFitState, VoronoiArtifacts


# ---------------------------------------------------------------------------
# Bayesian credible-band and PPC plots
# ---------------------------------------------------------------------------

def save_ppc_plot(
    freq_hz: np.ndarray,
    observed_db: np.ndarray,
    ppc_samples_db: np.ndarray,
    out_dir: Path,
    unit_label: str,
) -> None:
    """Posterior predictive check overlay: replicated vs. observed S21 spectra."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i in range(len(ppc_samples_db)):
        ax.plot(
            freq_hz * 1e-9, ppc_samples_db[i],
            color="steelblue", alpha=0.2, linewidth=0.5,
        )
    ax.plot(freq_hz * 1e-9, observed_db, color="tomato", linewidth=1.8, label="Observed")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_title(f"Posterior Predictive Check — {unit_label}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_ppc_s21.png", dpi=200
    )
    plt.close(fig)


def save_s21_credible_band_plot(
    freq_hz: np.ndarray,
    observed_db: np.ndarray,
    median_db: np.ndarray,
    hdi_low_db: np.ndarray,
    hdi_high_db: np.ndarray,
    cred_mass: float,
    out_dir: Path,
    unit_label: str,
) -> None:
    """S21 dB with pointwise HDI credible band."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.fill_between(
        freq_hz * 1e-9, hdi_low_db, hdi_high_db,
        alpha=0.3, color="steelblue",
        label=f"{int(cred_mass * 100)}% HDI band",
    )
    ax.plot(freq_hz * 1e-9, median_db, color="steelblue", linewidth=1.5, label="Posterior median")
    ax.plot(freq_hz * 1e-9, observed_db, color="tomato", linewidth=1.0, linestyle="--", label="Observed")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_title(f"S21 dB Credible Band ({int(cred_mass * 100)}% HDI) — {unit_label}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_s21_credible_band.png", dpi=200
    )
    plt.close(fig)


def save_gd_credible_band_plot(
    freq_hz: np.ndarray,
    gd_median_ns: np.ndarray,
    gd_hdi_low_ns: np.ndarray,
    gd_hdi_high_ns: np.ndarray,
    cred_mass: float,
    out_dir: Path,
    unit_label: str,
) -> None:
    """Group delay with pointwise HDI credible band."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        freq_hz * 1e-9, gd_hdi_low_ns, gd_hdi_high_ns,
        alpha=0.3, color="mediumpurple",
        label=f"{int(cred_mass * 100)}% HDI band",
    )
    ax.plot(freq_hz * 1e-9, gd_median_ns, color="mediumpurple", linewidth=1.5, label="Posterior median")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Group delay (ns)")
    ax.set_title(f"Group Delay Credible Band ({int(cred_mass * 100)}% HDI) — {unit_label}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_gd_credible_band.png", dpi=200
    )
    plt.close(fig)


def save_td_credible_band_plot(
    time_s: np.ndarray,
    impulse_median: np.ndarray,
    hdi_low: np.ndarray,
    hdi_high: np.ndarray,
    cred_mass: float,
    out_dir: Path,
    unit_label: str,
) -> None:
    """Impulse response |h(t)| with pointwise HDI credible band."""
    t_ns = time_s * 1e9
    cut = int(min(len(t_ns), np.searchsorted(t_ns, 20.0) + 1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(
        t_ns[:cut], hdi_low[:cut], hdi_high[:cut],
        alpha=0.3, color="forestgreen",
        label=f"{int(cred_mass * 100)}% HDI band",
    )
    ax.plot(t_ns[:cut], impulse_median[:cut], color="forestgreen", linewidth=1.5, label="Posterior median")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("|h(t)|")
    ax.set_title(f"Impulse Response Credible Band ({int(cred_mass * 100)}% HDI) — {unit_label}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_impulse_credible_band.png", dpi=200
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Persistent homology plots
# ---------------------------------------------------------------------------

def save_persistence_diagram_plot(
    dgms: list[np.ndarray], out_dir: Path, unit_label: str
) -> None:
    n_dims = len(dgms)
    fig, axes = plt.subplots(1, n_dims, figsize=(4.5 * n_dims, 4.5))
    if n_dims == 1:
        axes = [axes]
    cmaps = ["viridis", "plasma", "inferno"]
    for d, (dgm, ax) in enumerate(zip(dgms, axes)):
        finite = dgm[dgm[:, 1] < np.inf] if len(dgm) else dgm
        if len(finite):
            ax.scatter(finite[:, 0], finite[:, 1], s=20, alpha=0.8,
                       c=np.arange(len(finite)), cmap=cmaps[d % 3])
        lim_max = float(finite[:, 1].max()) if len(finite) else 1.0
        lim_max = max(lim_max, 1.0)
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"Dim {d} – {unit_label} ({len(finite)} pairs)")
    fig.suptitle(f"Persistence Diagrams – {unit_label}")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_persistence_diagram.png", dpi=200
    )
    plt.close(fig)


def save_barcode_plot(
    dgms: list[np.ndarray], out_dir: Path, unit_label: str
) -> None:
    fig, axes = plt.subplots(len(dgms), 1, figsize=(8, 3 * len(dgms)), squeeze=False)
    colors = ["steelblue", "coral", "forestgreen"]
    for d, (dgm, ax) in enumerate(zip(dgms, axes[:, 0])):
        finite = dgm[dgm[:, 1] < np.inf] if len(dgm) else dgm
        if len(finite):
            order = np.argsort(-(finite[:, 1] - finite[:, 0]))
            finite = finite[order]
        for i, (b_val, de) in enumerate(finite[:50]):
            ax.plot([b_val, de], [i, i], linewidth=2.0, color=colors[d % 3], alpha=0.8)
        ax.set_xlabel("Filtration value")
        ax.set_ylabel("Interval")
        ax.set_title(f"Barcode dim {d} – {unit_label} (top {min(50, len(finite))} by persistence)")
    fig.suptitle(f"Barcodes – {unit_label}")
    fig.tight_layout()
    fig.savefig(out_dir / f"{unit_label.lower().replace(' ', '_')}_barcode.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# RF topographic heatmaps
# ---------------------------------------------------------------------------

def save_topomap_unit_vs_freq(
    all_metrics: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    freq_hz: np.ndarray,
    out_dir: Path,
) -> None:
    """H1: 3-D surface map of |S21| dB across unit index × frequency."""
    labels = [f"Unit {i + 1}" for i in range(len(bundles_i))]
    mat = np.vstack([all_metrics[b.name]["s21_db"].values for b in bundles_i])
    freq_ghz = freq_hz / 1e9
    unit_idx = np.arange(len(labels), dtype=np.float64)
    F, U = np.meshgrid(freq_ghz, unit_idx)
    fig = plt.figure(figsize=(13, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(F, U, mat, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Unit")
    ax.set_yticks(unit_idx)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_zlabel("|S21| (dB)")
    ax.set_title("|S21| dB – Unit × Frequency 3D Surface (H1)")
    ax.view_init(elev=25, azim=-55)
    fig.colorbar(surf, ax=ax, shrink=0.55, label="|S21| (dB)")
    fig.tight_layout()
    fig.savefig(out_dir / "topomap_unit_vs_freq_S21dB_3d.png", dpi=220)
    plt.close(fig)


def save_topomap_shift_register_density(
    metrics_df: pd.DataFrame,
    lag: int,
    out_dir: Path,
    unit_label: str,
    nbins: int = 60,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """H2: 3-D density surface of shift-register phase space."""
    s21_lin = 10.0 ** (metrics_df["s21_db"].values / 20.0)
    if len(s21_lin) <= lag:
        return
    H, xe, ye = np.histogram2d(s21_lin[:-lag], s21_lin[lag:], bins=nbins)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    _norm = _MplNormalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else None
    H_plot = np.log1p(H)
    _surf_kw: dict[str, Any] = {"cmap": "cividis", "linewidth": 0, "antialiased": True, "alpha": 0.90}
    if _norm is not None:
        _surf_kw["norm"] = _norm
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XC, YC, H_plot, **_surf_kw)
    ax.set_xlabel("|S21(i)|")
    ax.set_ylabel(f"|S21(i+{lag})|")
    ax.set_zlabel("log(1 + Count)")
    ax.set_title(f"Shift-Register Phase-Space 3D Density – {unit_label} (H2)")
    ax.view_init(elev=30, azim=-50)
    fig.colorbar(surf, ax=ax, shrink=0.55, label="log(1 + Count)")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_topomap_shift_density_3d.png", dpi=220
    )
    plt.close(fig)


def save_topomap_complex_plane_density(
    metrics_df: pd.DataFrame,
    out_dir: Path,
    unit_label: str,
    nbins: int = 60,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """H3: 3-D density surface of Re(S21) vs Im(S21) occupancy."""
    H, xe, ye = np.histogram2d(
        metrics_df["s21_re"].values, metrics_df["s21_im"].values, bins=nbins
    )
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    XC, YC = np.meshgrid(xc, yc, indexing="ij")
    _norm = _MplNormalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else None
    H_plot = np.log1p(H)
    _surf_kw: dict[str, Any] = {"cmap": "cividis", "linewidth": 0, "antialiased": True, "alpha": 0.90}
    if _norm is not None:
        _surf_kw["norm"] = _norm
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XC, YC, H_plot, **_surf_kw)
    ax.set_xlabel("Re(S21)")
    ax.set_ylabel("Im(S21)")
    ax.set_zlabel("log(1 + Count)")
    ax.set_title(f"Complex-Plane Occupancy 3D Density – {unit_label} (H3)")
    ax.view_init(elev=35, azim=-45)
    fig.colorbar(surf, ax=ax, shrink=0.55, label="log(1 + Count)")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_topomap_complex_density_3d.png", dpi=220
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-unit RF scatter plots
# ---------------------------------------------------------------------------

def save_complex_plane_plots(
    bundle: S2PBundle,
    metrics_df: pd.DataFrame,
    out_dir: Path,
    unit_label: str,
) -> None:
    """C1: Physical manifold scatter – (Re(S11), Im(S11), |S21| dB) colour-coded by frequency."""
    freq_ghz = metrics_df["freq_hz"].values / 1e9
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        metrics_df["s11_re"].values, metrics_df["s11_im"].values, metrics_df["s21_db"].values,
        c=freq_ghz, cmap="rainbow", s=6, alpha=0.65, edgecolors="none",
    )
    ax.set_xlabel("Re(S11)")
    ax.set_ylabel("Im(S11)")
    ax.set_zlabel("|S21| (dB)")
    ax.set_title(f"C1: Physical Manifold Scatter – {unit_label}")
    ax.view_init(elev=25, azim=-55)
    fig.colorbar(sc, ax=ax, shrink=0.55, label="Freq (GHz)")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_c1_complex_manifold_scatter.png", dpi=200
    )
    plt.close(fig)


def save_shift_register_plot(
    scalar: np.ndarray,
    lag: int,
    out_dir: Path,
    unit_label: str,
    scalar_name: str = "S21_db",
) -> None:
    """S1: Shift-register scatter – (scalar(i), scalar(i+lag), index) colour-coded by index."""
    if len(scalar) <= lag:
        return
    s_curr = scalar[:-lag]
    s_next = scalar[lag:]
    idx_arr = np.arange(len(s_curr), dtype=np.float64)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(s_curr, s_next, idx_arr, c=idx_arr, cmap="cividis", s=6, alpha=0.65, edgecolors="none")
    ax.set_xlabel(f"{scalar_name}(i)")
    ax.set_ylabel(f"{scalar_name}(i+{lag})")
    ax.set_zlabel("Sample index")
    ax.set_title(f"S1: Shift-Register Scatter – {unit_label} (lag={lag})")
    ax.view_init(elev=30, azim=-50)
    fig.colorbar(sc, ax=ax, shrink=0.55, label="Sample index")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_s1_shift_register_scatter.png", dpi=200
    )
    plt.close(fig)


def save_voronoi_plot(
    cloud: np.ndarray,
    voronoi_artifacts: VoronoiArtifacts,
    axis_labels: tuple[str, str, str],
    out_dir: Path,
    unit_label: str,
    topology_label: str,
) -> None:
    """V1: Voronoi cell descriptor scatter coloured by bounded regime intensity."""
    if len(cloud) == 0:
        return
    pv = np.asarray(voronoi_artifacts["point_values"], dtype=np.float64)
    pv_lo, pv_hi = float(pv.min()), float(pv.max())
    if np.isclose(pv_lo, pv_hi):
        pv_hi = pv_lo + 1.0
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        cloud[:, 0], cloud[:, 1], cloud[:, 2],
        c=pv, cmap="cividis", s=12, alpha=0.75, vmin=pv_lo, vmax=pv_hi, edgecolors="none",
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(f"V1: Voronoi Regime Descriptor – {topology_label} – {unit_label}")
    ax.view_init(elev=28, azim=-60)
    fig.colorbar(sc, ax=ax, shrink=0.55, label="Regime intensity [0–1]")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_{topology_label}_v1_voronoi_descriptor.png",
        dpi=200,
    )
    plt.close(fig)


def save_gng_plot(
    cloud: np.ndarray,
    gng_state: GNGState,
    axis_labels: tuple[str, str, str],
    out_dir: Path,
    unit_label: str,
    topology_label: str,
) -> None:
    """G1: GNG state graph – node scatter + edges, colour-coded by normalised node error."""
    if len(gng_state.nodes) == 0:
        return
    nodes = gng_state.nodes
    errors = gng_state.errors
    err_norm = (errors - errors.min()) / (errors.max() - errors.min() + 1e-12)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for left, right in gng_state.edges:
        if left < len(nodes) and right < len(nodes):
            ax.plot(
                [nodes[left, 0], nodes[right, 0]],
                [nodes[left, 1], nodes[right, 1]],
                [nodes[left, 2], nodes[right, 2]],
                color="gray", lw=0.5, alpha=0.4,
            )
    sc = ax.scatter(
        nodes[:, 0], nodes[:, 1], nodes[:, 2],
        c=err_norm, cmap="cividis", s=40, alpha=0.9, edgecolors="k", linewidths=0.3,
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(f"G1: GNG State Graph – {topology_label} – {unit_label}")
    ax.view_init(elev=28, azim=-60)
    fig.colorbar(sc, ax=ax, shrink=0.55, label="Node error (normalised)")
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_{topology_label}_g1_gng_graph.png",
        dpi=200,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Voxel density helpers
# ---------------------------------------------------------------------------

def _compute_voxel_sv(
    cloud: np.ndarray, values: np.ndarray, bins: int, metric_name: str
) -> np.ndarray:
    """Return per-occupied-voxel stat values (used for global colour-range computation)."""
    if len(cloud) == 0:
        return np.array([], dtype=np.float64)
    bins_i = max(4, int(bins))
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    mins = mins - 0.02 * span
    maxs = maxs + 0.02 * span
    x_edges = np.linspace(mins[0], maxs[0], bins_i + 1)
    y_edges = np.linspace(mins[1], maxs[1], bins_i + 1)
    z_edges = np.linspace(mins[2], maxs[2], bins_i + 1)
    x_idx = np.clip(np.digitize(cloud[:, 0], x_edges) - 1, 0, bins_i - 1)
    y_idx = np.clip(np.digitize(cloud[:, 1], y_edges) - 1, 0, bins_i - 1)
    z_idx = np.clip(np.digitize(cloud[:, 2], z_edges) - 1, 0, bins_i - 1)
    accum = np.zeros((bins_i, bins_i, bins_i), dtype=np.float64)
    counts = np.zeros((bins_i, bins_i, bins_i), dtype=np.float64)
    for xi, yi, zi, value in zip(x_idx, y_idx, z_idx, values):
        accum[xi, yi, zi] += float(value)
        counts[xi, yi, zi] += 1.0
    filled = counts > 0.0
    if not np.any(filled):
        return np.array([], dtype=np.float64)
    stat = counts.copy() if metric_name == "density" else np.divide(
        accum, counts, out=np.zeros_like(accum), where=counts > 0.0
    )
    xi_arr, yi_arr, zi_arr = np.where(filled)
    return stat[xi_arr, yi_arr, zi_arr]


def save_3d_voxel_heatmap(
    cloud: np.ndarray,
    values: np.ndarray,
    axis_labels: tuple[str, str, str],
    out_dir: Path,
    unit_label: str,
    label: str,
    metric_name: str,
    bins: int,
    cmap_name: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """3-D spatial density map rendered as a scatter of voxel centroids."""
    if len(cloud) == 0:
        return
    bins = max(4, int(bins))
    mins = cloud.min(axis=0)
    maxs = cloud.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    mins = mins - 0.02 * span
    maxs = maxs + 0.02 * span
    x_edges = np.linspace(mins[0], maxs[0], bins + 1)
    y_edges = np.linspace(mins[1], maxs[1], bins + 1)
    z_edges = np.linspace(mins[2], maxs[2], bins + 1)
    x_idx = np.clip(np.digitize(cloud[:, 0], x_edges) - 1, 0, bins - 1)
    y_idx = np.clip(np.digitize(cloud[:, 1], y_edges) - 1, 0, bins - 1)
    z_idx = np.clip(np.digitize(cloud[:, 2], z_edges) - 1, 0, bins - 1)
    accum = np.zeros((bins, bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins, bins), dtype=np.float64)
    for xi, yi, zi, value in zip(x_idx, y_idx, z_idx, values):
        accum[xi, yi, zi] += float(value)
        counts[xi, yi, zi] += 1.0
    filled = counts > 0.0
    if not np.any(filled):
        return
    stat = counts.copy() if metric_name == "density" else np.divide(
        accum, counts, out=np.zeros_like(accum), where=counts > 0.0
    )
    xi_arr, yi_arr, zi_arr = np.where(filled)
    cx = 0.5 * (x_edges[xi_arr] + x_edges[xi_arr + 1])
    cy = 0.5 * (y_edges[yi_arr] + y_edges[yi_arr + 1])
    cz = 0.5 * (z_edges[zi_arr] + z_edges[zi_arr + 1])
    sv = stat[xi_arr, yi_arr, zi_arr]
    _vmin = vmin if vmin is not None else float(sv.min())
    _vmax = vmax if vmax is not None else float(sv.max())
    if np.isclose(_vmin, _vmax):
        _vmax = _vmin + 1.0
    sv_norm = np.clip((sv - _vmin) / (_vmax - _vmin), 0.0, 1.0)
    marker_sizes = 18.0 + 262.0 * sv_norm
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        cx, cy, cz,
        c=sv, s=marker_sizes, cmap=cmap_name,
        vmin=_vmin, vmax=_vmax, alpha=0.75, edgecolors="none", depthshade=True,
    )
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(f"3D Density Map - {metric_name.title()} - {label} - {unit_label}")
    ax.view_init(elev=28, azim=-60)
    fig.colorbar(sc, ax=ax, shrink=0.7, label=metric_name.title())
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{unit_label.lower().replace(' ', '_')}_{label}_{metric_name}_heatmap_3d.png",
        dpi=220,
    )
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-unit comparison figures
# ---------------------------------------------------------------------------

def save_cross_unit_s21_overlay(
    metrics_by_unit: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    out_dir: Path,
) -> None:
    """Overlay |S21| dB and group delay for every unit on shared axes."""
    n = len(bundles_i)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n, 1)))
    fig, (ax_s21, ax_gd) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for i, bundle in enumerate(bundles_i):
        m = metrics_by_unit[bundle.name]
        freq_ghz = m["freq_hz"].values / 1e9
        ax_s21.plot(freq_ghz, m["s21_db"].values, color=colors[i], lw=1.0, label=f"Unit {i + 1}")
        ax_gd.plot(freq_ghz, m["group_delay_s21_s"].values * 1e9, color=colors[i], lw=1.0)
    ax_s21.set_ylabel("|S21| (dB)")
    ax_s21.set_title("Cross-Unit |S21| dB Overlay — manufacturing defects appear as outlier traces")
    ax_s21.legend(fontsize=8, ncol=max(1, n // 4), loc="best")
    ax_s21.grid(True, alpha=0.25)
    ax_gd.set_xlabel("Frequency (GHz)")
    ax_gd.set_ylabel("Group delay (ns)")
    ax_gd.set_title("Cross-Unit Group Delay Overlay")
    ax_gd.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_s21_gd_overlay.png", dpi=200)
    plt.close(fig)


def save_cross_unit_complex_overlay(
    metrics_by_unit: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    out_dir: Path,
) -> None:
    """Overlay S21 and S11 complex-plane loci for all units."""
    n = len(bundles_i)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n, 1)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for i, bundle in enumerate(bundles_i):
        m = metrics_by_unit[bundle.name]
        ax1.plot(m["s21_re"].values, m["s21_im"].values, color=colors[i], lw=0.8, alpha=0.85, label=f"Unit {i + 1}")
        ax2.plot(m["s11_re"].values, m["s11_im"].values, color=colors[i], lw=0.8, alpha=0.85, label=f"Unit {i + 1}")
    for ax, title, xlabel, ylabel in [
        (ax1, "S21 Complex-Plane Loci — All Units", "Re(S21)", "Im(S21)"),
        (ax2, "S11 Smith-Plane Loci — All Units", "Re(S11)", "Im(S11)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect("equal", "datalim")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_complex_plane_overlay.png", dpi=200)
    plt.close(fig)


def save_cross_unit_h2_panel(
    metrics_by_unit: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    lag: int,
    out_dir: Path,
    nbins: int = 60,
) -> None:
    """Multi-panel 2-D contour grid of shift-register density, one panel per unit."""
    n = len(bundles_i)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    all_s21 = [10.0 ** (metrics_by_unit[b.name]["s21_db"].values / 20.0) for b in bundles_i]
    arrs_x = [s[:-lag] for s in all_s21 if len(s) > lag]
    arrs_y = [s[lag:] for s in all_s21 if len(s) > lag]
    if not arrs_x:
        return
    xlim = (float(np.concatenate(arrs_x).min()), float(np.concatenate(arrs_x).max()))
    ylim = (float(np.concatenate(arrs_y).min()), float(np.concatenate(arrs_y).max()))
    hmax = 0.0
    for b in bundles_i:
        s21 = 10.0 ** (metrics_by_unit[b.name]["s21_db"].values / 20.0)
        if len(s21) > lag:
            H, _, _ = np.histogram2d(s21[:-lag], s21[lag:], bins=nbins, range=[xlim, ylim])
            hmax = max(hmax, float(np.log1p(H.max())))
    vmax = hmax if hmax > 0.0 else 1.0
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    last_im = None
    for i, bundle in enumerate(bundles_i):
        row, col = divmod(i, ncols)
        ax = axes_grid[row][col]
        s21 = 10.0 ** (metrics_by_unit[bundle.name]["s21_db"].values / 20.0)
        if len(s21) > lag:
            H, xe, ye = np.histogram2d(s21[:-lag], s21[lag:], bins=nbins, range=[xlim, ylim])
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            last_im = ax.contourf(xc, yc, np.log1p(H.T), levels=20, cmap="cividis", vmin=0.0, vmax=vmax)
        ax.set_title(f"Unit {i + 1}", fontsize=9)
        ax.set_xlabel("|S21(i)|", fontsize=7)
        ax.set_ylabel(f"|S21(i+{lag})|", fontsize=7)
        ax.tick_params(labelsize=6)
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes_grid[row][col].set_visible(False)
    fig.suptitle(f"Cross-Unit Shift-Register Phase-Space Density (lag={lag}) — log(1+count)", fontsize=11)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_grid.ravel().tolist(), shrink=0.6, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_h2_shift_density_panel.png", dpi=200)
    plt.close(fig)


def save_cross_unit_h3_panel(
    metrics_by_unit: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    out_dir: Path,
    nbins: int = 60,
) -> None:
    """Multi-panel 2-D contour grid of complex-plane occupancy, one panel per unit."""
    n = len(bundles_i)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    re_all = np.concatenate([metrics_by_unit[b.name]["s21_re"].values for b in bundles_i])
    im_all = np.concatenate([metrics_by_unit[b.name]["s21_im"].values for b in bundles_i])
    xlim = (float(re_all.min()), float(re_all.max()))
    ylim = (float(im_all.min()), float(im_all.max()))
    hmax = 0.0
    for b in bundles_i:
        m = metrics_by_unit[b.name]
        H, _, _ = np.histogram2d(m["s21_re"].values, m["s21_im"].values, bins=nbins, range=[xlim, ylim])
        hmax = max(hmax, float(np.log1p(H.max())))
    vmax = hmax if hmax > 0.0 else 1.0
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    last_im = None
    for i, bundle in enumerate(bundles_i):
        row, col = divmod(i, ncols)
        ax = axes_grid[row][col]
        m = metrics_by_unit[bundle.name]
        H, xe, ye = np.histogram2d(m["s21_re"].values, m["s21_im"].values, bins=nbins, range=[xlim, ylim])
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        last_im = ax.contourf(xc, yc, np.log1p(H.T), levels=20, cmap="cividis", vmin=0.0, vmax=vmax)
        ax.set_title(f"Unit {i + 1}", fontsize=9)
        ax.set_xlabel("Re(S21)", fontsize=7)
        ax.set_ylabel("Im(S21)", fontsize=7)
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=6)
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes_grid[row][col].set_visible(False)
    fig.suptitle("Cross-Unit Complex-Plane Occupancy Density — log(1+count)", fontsize=11)
    if last_im is not None:
        fig.colorbar(last_im, ax=axes_grid.ravel().tolist(), shrink=0.6, label="log(1 + count)")
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_h3_complex_density_panel.png", dpi=200)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, path: Path, title: str) -> None:
    """Annotated heatmap for symmetric distance / similarity matrices."""
    n_rows, n_cols = df.shape
    fig, ax = plt.subplots(figsize=(max(5, n_cols + 1), max(4, n_rows + 1)))
    vals = df.values.astype(float)
    im = ax.imshow(vals, cmap="YlOrRd", interpolation="nearest")
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(df.index, fontsize=9)
    vrange = vals.max() - vals.min() if vals.max() != vals.min() else 1.0
    for i in range(n_rows):
        for j in range(n_cols):
            brightness = (vals[i, j] - vals.min()) / vrange
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center",
                    color="white" if brightness > 0.55 else "black",
                    fontsize=8, fontweight="bold")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_cross_unit_s21_envelope(
    metrics_by_unit: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    out_dir: Path,
) -> None:
    """S21 dB overlay with cross-unit mean ± 2σ statistical envelope."""
    n = len(bundles_i)
    if n == 0:
        return
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n, 1)))
    freq_ghz = metrics_by_unit[bundles_i[0].name]["freq_hz"].values / 1e9
    s21_matrix = np.stack([metrics_by_unit[b.name]["s21_db"].values for b in bundles_i], axis=0)
    mean_s21 = s21_matrix.mean(axis=0)
    std_s21 = s21_matrix.std(axis=0, ddof=0)
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.fill_between(freq_ghz, mean_s21 - 2 * std_s21, mean_s21 + 2 * std_s21,
                    alpha=0.18, color="grey", label="Mean ± 2σ")
    ax.plot(freq_ghz, mean_s21, color="black", lw=1.8, ls="--", label="Mean", zorder=5)
    for i, bundle in enumerate(bundles_i):
        ax.plot(freq_ghz, metrics_by_unit[bundle.name]["s21_db"].values,
                color=colors[i], lw=0.9, alpha=0.9, label=f"Unit {i + 1}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("|S21| (dB)")
    ax.set_title("Cross-Unit |S21| Statistical Envelope — units outside shaded band are outliers")
    ax.legend(fontsize=8, ncol=max(1, (n + 2) // 3), loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_s21_envelope.png", dpi=200)
    plt.close(fig)


def save_cross_unit_delta_panel(
    metrics_by_unit: dict[str, pd.DataFrame],
    bundles_i: list[S2PBundle],
    out_dir: Path,
) -> None:
    """Per-unit deviation from the cross-unit mean S21 dB."""
    n = len(bundles_i)
    if n == 0:
        return
    freq_ghz = metrics_by_unit[bundles_i[0].name]["freq_hz"].values / 1e9
    s21_matrix = np.stack([metrics_by_unit[b.name]["s21_db"].values for b in bundles_i], axis=0)
    mean_s21 = s21_matrix.mean(axis=0)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n, 1)))
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                                   squeeze=False, sharex=True, sharey=True)
    for i, bundle in enumerate(bundles_i):
        row, col = divmod(i, ncols)
        ax = axes_grid[row][col]
        delta = metrics_by_unit[bundle.name]["s21_db"].values - mean_s21
        ax.plot(freq_ghz, delta, color=colors[i], lw=1.0)
        ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"Unit {i + 1}  (RMS Δ={float(np.sqrt((delta**2).mean())):.3f} dB)", fontsize=9)
        ax.set_xlabel("Frequency (GHz)", fontsize=7)
        ax.set_ylabel("ΔS21 (dB)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)
    for j in range(n, nrows * ncols):
        row, col = divmod(j, ncols)
        axes_grid[row][col].set_visible(False)
    fig.suptitle("Per-Unit S21 Deviation from Cross-Unit Mean", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_s21_delta_panel.png", dpi=200)
    plt.close(fig)


def save_cross_unit_vf_pole_scatter(
    vf_states: list[VectorFitState],
    bundles_i: list[S2PBundle],
    freq_hz: np.ndarray,
    out_dir: Path,
) -> None:
    """Overlay vector-fit poles from all units on a normalised complex plane."""
    n = len(vf_states)
    if n == 0:
        return
    f_max = float(freq_hz.max())
    norm = 2.0 * math.pi * f_max
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n, 1)))
    markers = ["o", "s", "^", "D", "v", "P", "*", "X", "h", "+"]
    fig, ax = plt.subplots(figsize=(9, 7))
    for i, (state, bundle) in enumerate(zip(vf_states, bundles_i)):
        poles = state.poles / norm
        ax.scatter(poles.real, poles.imag, color=colors[i], marker=markers[i % len(markers)],
                   s=60, alpha=0.85, edgecolors="none", label=f"Unit {i + 1}")
        ax.scatter(poles.real, -poles.imag, color=colors[i], marker=markers[i % len(markers)],
                   s=30, alpha=0.35, edgecolors="none")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.axvline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("Re(pole) / (2π f_max)")
    ax.set_ylabel("Im(pole) / (2π f_max)")
    ax.set_title("Cross-Unit Vector-Fit Pole Scatter — cluster spread indicates manufacturing variation")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_vf_pole_scatter.png", dpi=200)
    plt.close(fig)


def save_cross_unit_anomaly_summary(
    bundles_i: list[S2PBundle],
    distance_dfs: dict[str, pd.DataFrame | None],
    out_dir: Path,
) -> None:
    """Bar chart ranking units by aggregate anomaly score across all distance metrics."""
    n = len(bundles_i)
    if n < 2:
        return
    labels = [f"Unit {i + 1}" for i in range(n)]
    score_cols: list[np.ndarray] = []
    col_names: list[str] = []
    for name, df in distance_dfs.items():
        if df is None or df.shape != (n, n):
            continue
        vals = df.values.astype(float)
        np.fill_diagonal(vals, np.nan)
        row_means = np.nanmean(vals, axis=1)
        span = row_means.max() - row_means.min()
        if span > 0:
            score_cols.append((row_means - row_means.min()) / span)
            col_names.append(name)
    if not score_cols:
        return
    score_matrix = np.stack(score_cols, axis=1)
    aggregate = score_matrix.mean(axis=1)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, max(n, 1)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_grp = axes[0]
    x = np.arange(n)
    width = 0.8 / max(len(col_names), 1)
    for k, cname in enumerate(col_names):
        offset = (k - len(col_names) / 2.0 + 0.5) * width
        ax_grp.bar(x + offset, score_matrix[:, k], width=width * 0.9,
                   label=cname.replace("_", " "), alpha=0.78)
    ax_grp.set_xticks(x)
    ax_grp.set_xticklabels(labels, fontsize=9)
    ax_grp.set_ylabel("Normalised distance score")
    ax_grp.set_title("Per-Metric Anomaly Score by Unit")
    ax_grp.legend(fontsize=7, ncol=2, loc="upper left")
    ax_grp.grid(True, alpha=0.2, axis="y")
    ax_agg = axes[1]
    ax_agg.bar(labels, aggregate, color=[colors[i] for i in range(n)], edgecolor="black", linewidth=0.6)
    ax_agg.set_ylabel("Aggregate anomaly score (mean normalised)")
    ax_agg.set_title("Aggregate Cross-Unit Anomaly Ranking")
    ax_agg.grid(True, alpha=0.2, axis="y")
    for xi, yi in enumerate(aggregate):
        ax_agg.text(xi, yi + 0.01, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(
        "Cross-Unit Anomaly Summary — higher score = more topologically isolated", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cross_unit_anomaly_summary.png", dpi=200)
    plt.close(fig)
