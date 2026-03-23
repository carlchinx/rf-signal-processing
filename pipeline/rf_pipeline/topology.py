"""Topological data analysis: embeddings, Voronoi, GNG, and persistent homology.

Public API
----------
choose_lag_autocorr, embedding_complex_trajectory, embedding_shift_register,
subsample_point_cloud, subsample_point_cloud_with_indices,
build_complex_topology_cloud, build_shift_topology_cloud,
analyze_voronoi_topology, fit_growing_neural_gas, summarize_gng_graph,
build_gng_transition_artifacts, build_topology_descriptor,
make_tda_feature_vector, compare_topology_features,
compute_ph_diagrams, compute_ph_distance_matrix, PH_AVAILABLE

Extending
---------
To add a new embedding strategy, add a function with the same signature as
``embedding_complex_trajectory`` and call it from ``build_complex_topology_cloud``.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError, Voronoi
from scipy.spatial.distance import cdist

from .config import (
    GNGState,
    PHResult,
    S2PBundle,
    TDAConfig,
    TRACE_TO_INDEX,
    TopologyDescriptor,
    VoronoiArtifacts,
)
from .metrics import extract_trace

# ---------------------------------------------------------------------------
# Optional persistent-homology dependencies
# ---------------------------------------------------------------------------
try:
    from ripser import ripser as _ripser_fn
    from persim import bottleneck as _persim_bottleneck, wasserstein as _persim_wasserstein  # noqa: F401
    PH_AVAILABLE: bool = True
except Exception:
    _ripser_fn = None  # type: ignore[assignment]
    _persim_bottleneck = None  # type: ignore[assignment]
    _persim_wasserstein = None  # type: ignore[assignment]
    PH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Lag selection
# ---------------------------------------------------------------------------

def choose_lag_autocorr(
    y: np.ndarray,
    min_lag: int = 4,
    max_lag: int | None = None,
    window: int | None = None,
) -> int:
    max_lag = max_lag or min(len(y) // 4, 128)
    if window is not None and window >= 2:
        max_feasible = max(min_lag, (len(y) - 1) // (window - 1))
        max_lag = min(max_lag, max_feasible)
    y0 = y - np.mean(y)
    denom = float(np.dot(y0, y0)) + 1e-12
    ac = []
    for lag in range(1, max_lag + 1):
        ac.append(float(np.dot(y0[:-lag], y0[lag:]) / denom))
    ac = np.asarray(ac)
    target = math.e ** -1
    idx = np.where(ac < target)[0]
    if len(idx):
        return max(min_lag, int(idx[0] + 1))
    for i in range(1, len(ac) - 1):
        if ac[i] < ac[i - 1] and ac[i] < ac[i + 1]:
            return max(min_lag, i + 1)
    return min_lag


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embedding_complex_trajectory(
    bundle: S2PBundle, traces: Sequence[str]
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for tr in traces:
        z = extract_trace(bundle.s, tr)
        cols.extend([z.real, z.imag])
    x = np.column_stack(cols).astype(np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    return x


def embedding_shift_register(
    series: np.ndarray, window: int, lag: int, stride: int = 1
) -> np.ndarray:
    if window < 2:
        raise ValueError("window must be >=2")
    rows = []
    stop = len(series) - (window - 1) * lag
    for i in range(0, max(0, stop), stride):
        rows.append(series[i : i + window * lag : lag])
    if not rows:
        raise ValueError("Not enough samples for shift-register embedding")
    x = np.vstack(rows).astype(np.float64)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    return x


# ---------------------------------------------------------------------------
# Point-cloud helpers
# ---------------------------------------------------------------------------

def local_polyline_curvature(x: np.ndarray) -> np.ndarray:
    if len(x) < 3:
        return np.ones(len(x), dtype=np.float64)
    curv = np.zeros(len(x), dtype=np.float64)
    for i in range(1, len(x) - 1):
        v1 = x[i] - x[i - 1]
        v2 = x[i + 1] - x[i]
        n1 = np.linalg.norm(v1) + 1e-12
        n2 = np.linalg.norm(v2) + 1e-12
        cosang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        curv[i] = math.acos(cosang)
    curv[0] = curv[1]
    curv[-1] = curv[-2]
    return curv


def subsample_point_cloud_with_indices(
    x: np.ndarray, max_points: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= max_points:
        idx = np.arange(len(x), dtype=int)
        return x, idx
    if mode == "uniform":
        idx = np.linspace(0, len(x) - 1, max_points).astype(int)
        return x[idx], idx
    curv = local_polyline_curvature(x)
    w = curv + 0.05
    w = w / w.sum()
    candidate = np.arange(1, len(x) - 1)
    n_body = max_points - 2
    chosen = np.random.choice(
        candidate, size=n_body, replace=False, p=w[1:-1] / w[1:-1].sum()
    )
    idx = np.sort(np.concatenate([[0], chosen, [len(x) - 1]]))
    return x[idx], idx


def subsample_point_cloud(
    x: np.ndarray, max_points: int, mode: str
) -> np.ndarray:
    x_sub, _ = subsample_point_cloud_with_indices(x, max_points, mode)
    return x_sub


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)


# ---------------------------------------------------------------------------
# Topology cloud builders
# ---------------------------------------------------------------------------

def build_complex_topology_cloud(
    metrics_df: pd.DataFrame, cfg: TDAConfig
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str]]:
    raw = np.column_stack([
        metrics_df["s11_re"].to_numpy(dtype=np.float64),
        metrics_df["s11_im"].to_numpy(dtype=np.float64),
        metrics_df["s21_mag"].to_numpy(dtype=np.float64),
    ])
    raw = _normalize_columns(raw)
    cloud, idx = subsample_point_cloud_with_indices(raw, cfg.max_points, cfg.subsample)
    return cloud, idx, ("z(Re(S11))", "z(Im(S11))", "z(|S21|)")


def build_shift_topology_cloud(
    metrics_df: pd.DataFrame, lag: int, cfg: TDAConfig
) -> tuple[np.ndarray, np.ndarray, tuple[str, str, str]]:
    mag = metrics_df["s21_mag"].to_numpy(dtype=np.float64)
    gd = metrics_df["group_delay_s21_s"].to_numpy(dtype=np.float64)
    safe_lag = min(max(1, lag), max(1, len(mag) - 2))
    idx = np.arange(0, max(1, len(mag) - safe_lag), max(1, cfg.shift_stride), dtype=int)
    lag_idx = np.clip(idx + safe_lag, 0, len(mag) - 1)
    raw = np.column_stack([mag[idx], mag[lag_idx], gd[idx]])
    raw = _normalize_columns(raw)
    cloud, sub_idx = subsample_point_cloud_with_indices(raw, cfg.max_points, cfg.subsample)
    return cloud, idx[sub_idx], ("z(|S21(i)|)", f"z(|S21(i+{safe_lag})|)", "z(Group delay)")


# ---------------------------------------------------------------------------
# Voronoi topology analysis
# ---------------------------------------------------------------------------

def _polygon_area_3d(vertices: np.ndarray) -> float:
    if len(vertices) < 3:
        return 0.0
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:2].T
    x = projected[:, 0]
    y = projected[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def analyze_voronoi_topology(
    cloud: np.ndarray,
    source_index: np.ndarray,
    freq_hz: np.ndarray,
    cfg: TDAConfig,
) -> tuple[dict[str, float], VoronoiArtifacts]:
    if len(cloud) < 5:
        summary = {
            "points_used": float(len(cloud)),
            "finite_region_fraction": 0.0,
            "boundary_cell_fraction": 1.0,
            "mean_region_volume": 0.0,
            "std_region_volume": 0.0,
            "q90_region_volume": 0.0,
            "mean_density_proxy": 0.0,
            "q90_density_proxy": 0.0,
            "mean_neighbor_degree": 0.0,
            "std_neighbor_degree": 0.0,
            "mean_cell_anisotropy": 0.0,
            "q90_cell_anisotropy": 0.0,
            "mean_ridge_area": 0.0,
            "q90_ridge_area": 0.0,
            "mean_nn_distance": 0.0,
            "std_nn_distance": 0.0,
            "mean_abs_volume_gradient": 0.0,
            "q90_abs_volume_gradient": 0.0,
            "regime_boundary_count": 0.0,
        }
        empty = np.zeros(len(cloud), dtype=np.float64)
        point_table = pd.DataFrame({
            "point_id": np.arange(len(cloud), dtype=int),
            "source_index": source_index.astype(int),
            "freq_hz": freq_hz[source_index].astype(np.float64),
            "cell_volume": empty,
            "density_proxy": empty,
            "neighbor_degree": empty,
            "cell_anisotropy": empty,
            "volume_gradient": empty,
            "regime_boundary": np.zeros(len(cloud), dtype=int),
        })
        return summary, {
            "ridge_points": np.empty((0, 2), dtype=int),
            "point_values": empty,
            "point_table": point_table,
        }

    try:
        vor = Voronoi(cloud, qhull_options=cfg.voronoi_qhull_options)
    except QhullError:
        jitter = cloud + 1e-6 * np.random.standard_normal(cloud.shape)
        vor = Voronoi(jitter, qhull_options=f"{cfg.voronoi_qhull_options} QJ")

    degrees = np.zeros(len(cloud), dtype=np.float64)
    adjacency: dict[int, list[int]] = {idx: [] for idx in range(len(cloud))}
    for p0, p1 in vor.ridge_points:
        degrees[p0] += 1.0
        degrees[p1] += 1.0
        adjacency[p0].append(int(p1))
        adjacency[p1].append(int(p0))

    point_volumes = np.full(len(cloud), np.nan, dtype=np.float64)
    point_anisotropy = np.zeros(len(cloud), dtype=np.float64)
    finite_volumes: list[float] = []
    ridge_areas: list[float] = []
    finite_regions = 0
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region or len(region) < 4:
            continue
        vertices = vor.vertices[region]
        try:
            volume = float(ConvexHull(vertices).volume)
        except QhullError:
            continue
        if volume > 0.0 and np.isfinite(volume):
            finite_regions += 1
            finite_volumes.append(volume)
            point_volumes[point_idx] = volume

    for ridge_vertices in vor.ridge_vertices:
        if not ridge_vertices or -1 in ridge_vertices or len(ridge_vertices) < 3:
            continue
        vertices = vor.vertices[ridge_vertices]
        ridge_area = _polygon_area_3d(vertices)
        if ridge_area > 0.0 and np.isfinite(ridge_area):
            ridge_areas.append(ridge_area)

    dmat = cdist(cloud, cloud)
    np.fill_diagonal(dmat, np.inf)
    for point_idx in range(len(cloud)):
        neighbors = sorted(set(adjacency[point_idx]))
        if len(neighbors) < 3:
            neighbors = np.argsort(dmat[point_idx])[: min(4, len(cloud) - 1)].tolist()
        local = cloud[[point_idx, *neighbors]]
        if len(local) < 3:
            continue
        cov = np.cov(local.T)
        eigvals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 1e-12))
        point_anisotropy[point_idx] = float(eigvals[-1] / (eigvals[0] + 1e-12))

    finite_volumes_arr = np.asarray(finite_volumes, dtype=np.float64)
    if finite_volumes_arr.size:
        clip_level = float(
            np.quantile(finite_volumes_arr, cfg.voronoi_volume_clip_quantile)
        )
        finite_volumes_arr = np.clip(finite_volumes_arr, 0.0, clip_level)
    np.fill_diagonal(dmat, np.inf)
    nn = dmat.min(axis=1)
    density = np.where(
        np.isfinite(point_volumes) & (point_volumes > 0.0),
        1.0 / (point_volumes + 1e-12),
        0.0,
    )
    ordered = np.argsort(source_index)
    ordered_volumes = np.where(
        np.isfinite(point_volumes[ordered]), point_volumes[ordered], 0.0
    )
    volume_grad = np.zeros(len(cloud), dtype=np.float64)
    if len(ordered_volumes) > 1:
        grad_ordered = np.abs(
            np.diff(ordered_volumes, prepend=ordered_volumes[0])
        )
        volume_grad[ordered] = grad_ordered
        hotspot_threshold = (
            float(np.quantile(grad_ordered, 0.95))
            if len(grad_ordered) > 3
            else float(np.max(grad_ordered))
        )
        regime_boundary = (grad_ordered >= hotspot_threshold) & (grad_ordered > 0.0)
    else:
        grad_ordered = np.zeros_like(ordered_volumes)
        hotspot_threshold = 0.0
        regime_boundary = np.zeros_like(ordered_volumes, dtype=bool)
    regime_boundary_flags = np.zeros(len(cloud), dtype=bool)
    regime_boundary_flags[ordered] = regime_boundary

    ridge_areas_arr = np.asarray(ridge_areas, dtype=np.float64)
    finite_density = density[density > 0.0]
    finite_anisotropy = point_anisotropy[point_anisotropy > 0.0]
    summary = {
        "points_used": float(len(cloud)),
        "finite_region_fraction": float(finite_regions / max(1, len(cloud))),
        "boundary_cell_fraction": float(np.mean(~np.isfinite(point_volumes))),
        "mean_region_volume": float(np.mean(finite_volumes_arr)) if finite_volumes_arr.size else 0.0,
        "std_region_volume": float(np.std(finite_volumes_arr)) if finite_volumes_arr.size else 0.0,
        "q90_region_volume": float(np.quantile(finite_volumes_arr, 0.90)) if finite_volumes_arr.size else 0.0,
        "mean_density_proxy": float(np.mean(finite_density)) if finite_density.size else 0.0,
        "q90_density_proxy": float(np.quantile(finite_density, 0.90)) if finite_density.size else 0.0,
        "mean_neighbor_degree": float(np.mean(degrees)),
        "std_neighbor_degree": float(np.std(degrees)),
        "mean_cell_anisotropy": float(np.mean(finite_anisotropy)) if finite_anisotropy.size else 0.0,
        "q90_cell_anisotropy": float(np.quantile(finite_anisotropy, 0.90)) if finite_anisotropy.size else 0.0,
        "mean_ridge_area": float(np.mean(ridge_areas_arr)) if ridge_areas_arr.size else 0.0,
        "q90_ridge_area": float(np.quantile(ridge_areas_arr, 0.90)) if ridge_areas_arr.size else 0.0,
        "mean_nn_distance": float(np.mean(nn)),
        "std_nn_distance": float(np.std(nn)),
        "mean_abs_volume_gradient": float(np.mean(np.abs(grad_ordered))) if len(grad_ordered) else 0.0,
        "q90_abs_volume_gradient": float(np.quantile(np.abs(grad_ordered), 0.90)) if len(grad_ordered) else 0.0,
        "regime_boundary_count": float(np.sum(regime_boundary_flags)),
    }
    _pv_raw = np.where(np.isfinite(point_volumes), point_volumes, 0.0)
    _pv_pos = _pv_raw[_pv_raw > 0.0]
    _p98 = float(np.quantile(_pv_pos, 0.98)) if _pv_pos.size > 0 else 1.0
    _pv_clipped = np.clip(_pv_raw, 0.0, _p98)
    _pv_lo, _pv_hi = float(_pv_clipped.min()), float(_pv_clipped.max())
    point_values = (_pv_clipped - _pv_lo) / (_pv_hi - _pv_lo + 1e-12)
    point_table = pd.DataFrame({
        "point_id": np.arange(len(cloud), dtype=int),
        "source_index": source_index.astype(int),
        "freq_hz": freq_hz[source_index].astype(np.float64),
        "cell_volume": _pv_raw,
        "regime_intensity": point_values,
        "density_proxy": density,
        "neighbor_degree": degrees,
        "cell_anisotropy": point_anisotropy,
        "volume_gradient": volume_grad,
        "regime_boundary": regime_boundary_flags.astype(int),
    }).sort_values("source_index").reset_index(drop=True)
    artifacts: VoronoiArtifacts = {
        "ridge_points": np.asarray(vor.ridge_points, dtype=int),
        "point_values": point_values,
        "point_table": point_table,
    }
    return summary, artifacts


# ---------------------------------------------------------------------------
# Growing Neural Gas
# ---------------------------------------------------------------------------

def _gng_neighbors(edges: dict[tuple[int, int], int], node_idx: int) -> list[int]:
    neighbors: list[int] = []
    for left, right in edges:
        if left == node_idx:
            neighbors.append(right)
        elif right == node_idx:
            neighbors.append(left)
    return neighbors


def _compress_gng_graph(
    nodes: np.ndarray,
    errors: np.ndarray,
    edges: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    if not edges:
        return nodes, errors, edges
    keep = sorted({idx for edge in edges for idx in edge})
    if len(keep) == len(nodes):
        return nodes, errors, edges
    remap = {old: new for new, old in enumerate(keep)}
    new_nodes = nodes[keep]
    new_errors = errors[keep]
    new_edges = {
        (remap[left], remap[right]): age
        for (left, right), age in edges.items()
        if left in remap and right in remap
    }
    return new_nodes, new_errors, new_edges


def fit_growing_neural_gas(cloud: np.ndarray, cfg: TDAConfig) -> GNGState:
    if len(cloud) == 0:
        return GNGState(
            nodes=np.zeros((0, 3), dtype=np.float64),
            errors=np.zeros(0, dtype=np.float64),
            edges=[],
        )
    if len(cloud) == 1:
        return GNGState(nodes=cloud.copy(), errors=np.zeros(1, dtype=np.float64), edges=[])

    initial_idx = np.random.choice(len(cloud), size=2, replace=False)
    nodes = cloud[initial_idx].copy()
    errors = np.zeros(2, dtype=np.float64)
    edges: dict[tuple[int, int], int] = {tuple(sorted((0, 1))): 0}

    for step in range(1, cfg.gng_steps + 1):
        sample = cloud[np.random.randint(len(cloud))]
        distances = np.linalg.norm(nodes - sample, axis=1)
        closest = np.argsort(distances)[:2]
        winner, runner_up = int(closest[0]), int(closest[1])
        errors[winner] += float(distances[winner] ** 2)

        for edge in list(edges):
            if winner in edge:
                edges[edge] += 1

        nodes[winner] += cfg.gng_eps_winner * (sample - nodes[winner])
        for neighbor in _gng_neighbors(edges, winner):
            nodes[neighbor] += cfg.gng_eps_neighbor * (sample - nodes[neighbor])

        edges[tuple(sorted((winner, runner_up)))] = 0
        for edge, age in list(edges.items()):
            if age > cfg.gng_max_age:
                del edges[edge]
        if len(nodes) > 2:
            nodes, errors, edges = _compress_gng_graph(nodes, errors, edges)

        if step % cfg.gng_lambda == 0 and len(nodes) < cfg.gng_max_nodes and edges:
            q = int(np.argmax(errors))
            neighbors = _gng_neighbors(edges, q)
            if neighbors:
                f = neighbors[int(np.argmax(errors[neighbors]))]
                new_node = 0.5 * (nodes[q] + nodes[f])
                nodes = np.vstack([nodes, new_node])
                new_error = 0.5 * (errors[q] + errors[f])
                errors = np.append(errors, new_error)
                errors[q] *= cfg.gng_alpha
                errors[f] *= cfg.gng_alpha
                new_idx = len(nodes) - 1
                edges.pop(tuple(sorted((q, f))), None)
                edges[tuple(sorted((q, new_idx)))] = 0
                edges[tuple(sorted((f, new_idx)))] = 0
        errors *= cfg.gng_beta

    edge_list = [tuple(sorted(edge)) for edge in edges]
    return GNGState(nodes=nodes, errors=errors, edges=sorted(set(edge_list)))


def summarize_gng_graph(cloud: np.ndarray, state: GNGState) -> dict[str, float]:
    n_nodes = len(state.nodes)
    n_edges = len(state.edges)
    if n_nodes == 0:
        return {
            "n_nodes": 0.0,
            "n_edges": 0.0,
            "mean_degree": 0.0,
            "max_degree": 0.0,
            "n_components": 0.0,
            "graph_diameter": 0.0,
            "branch_point_count": 0.0,
            "cycle_count": 0.0,
            "largest_component_fraction": 0.0,
            "mean_edge_length": 0.0,
            "std_edge_length": 0.0,
            "quantization_rmse": 0.0,
            "mean_node_error": 0.0,
            "error_concentration": 0.0,
        }

    degrees = np.zeros(n_nodes, dtype=np.float64)
    adjacency: dict[int, list[int]] = {idx: [] for idx in range(n_nodes)}
    edge_lengths: list[float] = []
    for left, right in state.edges:
        degrees[left] += 1.0
        degrees[right] += 1.0
        adjacency[left].append(right)
        adjacency[right].append(left)
        edge_lengths.append(float(np.linalg.norm(state.nodes[left] - state.nodes[right])))

    seen = set()
    component_sizes: list[int] = []
    for start in range(n_nodes):
        if start in seen:
            continue
        queue = [start]
        seen.add(start)
        size = 0
        while queue:
            node_idx = queue.pop()
            size += 1
            for neighbor in adjacency[node_idx]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        component_sizes.append(size)

    distances = (
        cdist(cloud, state.nodes)
        if len(cloud) and n_nodes
        else np.zeros((0, 0), dtype=np.float64)
    )
    quantization = (
        np.sqrt(np.mean(np.min(distances, axis=1) ** 2)) if distances.size else 0.0
    )
    edge_lengths_arr = np.asarray(edge_lengths, dtype=np.float64)
    diameter = 0.0
    for start in range(n_nodes):
        queue = [(start, 0)]
        seen_local = {start}
        while queue:
            node_idx, dist = queue.pop(0)
            diameter = max(diameter, float(dist))
            for neighbor in adjacency[node_idx]:
                if neighbor not in seen_local:
                    seen_local.add(neighbor)
                    queue.append((neighbor, dist + 1))
    cycle_count = max(0.0, float(n_edges - n_nodes + len(component_sizes)))
    error_concentration = (
        float(np.max(state.errors) / (np.sum(state.errors) + 1e-12))
        if len(state.errors)
        else 0.0
    )
    return {
        "n_nodes": float(n_nodes),
        "n_edges": float(n_edges),
        "mean_degree": float(np.mean(degrees)),
        "max_degree": float(np.max(degrees)) if len(degrees) else 0.0,
        "n_components": float(len(component_sizes)),
        "graph_diameter": diameter,
        "branch_point_count": float(np.sum(degrees >= 3.0)),
        "cycle_count": cycle_count,
        "largest_component_fraction": float(max(component_sizes) / max(1, n_nodes)) if component_sizes else 0.0,
        "mean_edge_length": float(np.mean(edge_lengths_arr)) if edge_lengths_arr.size else 0.0,
        "std_edge_length": float(np.std(edge_lengths_arr)) if edge_lengths_arr.size else 0.0,
        "quantization_rmse": float(quantization),
        "mean_node_error": float(np.mean(state.errors)) if len(state.errors) else 0.0,
        "error_concentration": error_concentration,
    }


def build_gng_transition_artifacts(
    cloud: np.ndarray,
    source_index: np.ndarray,
    freq_hz: np.ndarray,
    state: GNGState,
) -> tuple[dict[str, float], dict]:
    from .config import GNGTransitionArtifacts  # local import avoids rexporting
    n_nodes = len(state.nodes)
    if n_nodes == 0 or len(cloud) == 0:
        empty_assign = pd.DataFrame(columns=["source_index", "freq_hz", "node_id"])
        empty_occ = pd.DataFrame(columns=["node_id", "occupancy_count", "occupancy_fraction", "mean_freq_hz"])
        empty_trans = pd.DataFrame()
        summary = {
            "occupancy_entropy": 0.0,
            "transition_entropy": 0.0,
            "max_node_occupancy_fraction": 0.0,
            "nonself_transition_fraction": 0.0,
        }
        return summary, {
            "assignments": empty_assign,
            "occupancy": empty_occ,
            "transition_matrix": empty_trans,
        }

    distances = cdist(cloud, state.nodes)
    assignments = np.argmin(distances, axis=1).astype(int)
    ordered = np.argsort(source_index)
    ordered_assignments = assignments[ordered]
    ordered_index = source_index[ordered]
    counts = np.bincount(assignments, minlength=n_nodes).astype(np.float64)
    fractions = counts / max(1.0, float(len(assignments)))
    occupancy_entropy = float(
        -np.sum(fractions[fractions > 0.0] * np.log(fractions[fractions > 0.0] + 1e-12))
    )
    transition = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for left, right in zip(ordered_assignments[:-1], ordered_assignments[1:]):
        transition[left, right] += 1.0
    trans_total = float(np.sum(transition))
    trans_prob = transition / max(trans_total, 1.0)
    transition_entropy = float(
        -np.sum(trans_prob[trans_prob > 0.0] * np.log(trans_prob[trans_prob > 0.0] + 1e-12))
    )
    nonself_fraction = float(
        (trans_total - np.trace(transition)) / max(trans_total, 1.0)
    )
    occupancy_df = pd.DataFrame({
        "node_id": np.arange(n_nodes, dtype=int),
        "occupancy_count": counts.astype(int),
        "occupancy_fraction": fractions,
        "mean_freq_hz": [
            float(np.mean(freq_hz[source_index[assignments == node_id]]))
            if np.any(assignments == node_id)
            else np.nan
            for node_id in range(n_nodes)
        ],
    })
    assignment_df = pd.DataFrame({
        "source_index": ordered_index.astype(int),
        "freq_hz": freq_hz[ordered_index].astype(np.float64),
        "node_id": ordered_assignments.astype(int),
    })
    transition_df = pd.DataFrame(
        transition,
        index=[f"node_{idx}" for idx in range(n_nodes)],
        columns=[f"node_{idx}" for idx in range(n_nodes)],
    )
    summary = {
        "occupancy_entropy": occupancy_entropy,
        "transition_entropy": transition_entropy,
        "max_node_occupancy_fraction": float(np.max(fractions)) if len(fractions) else 0.0,
        "nonself_transition_fraction": nonself_fraction,
    }
    return summary, {
        "assignments": assignment_df,
        "occupancy": occupancy_df,
        "transition_matrix": transition_df,
    }


# ---------------------------------------------------------------------------
# Descriptor builders
# ---------------------------------------------------------------------------

def build_topology_descriptor(
    cloud: np.ndarray,
    source_index: np.ndarray,
    freq_hz: np.ndarray,
    axis_labels: tuple[str, str, str],
    cfg: TDAConfig,
) -> TopologyDescriptor:
    voronoi_summary, voronoi_artifacts = analyze_voronoi_topology(
        cloud, source_index, freq_hz, cfg
    )
    gng_state = fit_growing_neural_gas(cloud, cfg)
    gng_summary = summarize_gng_graph(cloud, gng_state)
    gng_transition_summary, gng_artifacts = build_gng_transition_artifacts(
        cloud, source_index, freq_hz, gng_state
    )
    feature = np.asarray(
        [*voronoi_summary.values(), *gng_summary.values(), *gng_transition_summary.values()],
        dtype=np.float64,
    )
    return {
        "cloud": cloud,
        "source_index": source_index,
        "axis_labels": axis_labels,
        "voronoi_summary": voronoi_summary,
        "voronoi_artifacts": voronoi_artifacts,
        "gng_state": gng_state,
        "gng_summary": gng_summary,
        "gng_transition_summary": gng_transition_summary,
        "gng_artifacts": gng_artifacts,
        "feature": feature,
    }


def make_tda_feature_vector(
    metrics_df: pd.DataFrame,
    descriptor: TopologyDescriptor,
    cfg: TDAConfig,
) -> np.ndarray:
    feat: list[float] = []
    s21_db = metrics_df["s21_db"].to_numpy()
    gd = metrics_df["group_delay_s21_s"].to_numpy()
    recip = metrics_df["reciprocity_abs_s21_minus_s12"].to_numpy()
    feat.extend([
        float(np.max(s21_db)),
        float(np.min(s21_db)),
        float(np.mean(s21_db)),
        float(np.std(s21_db)),
        float(np.mean(gd)),
        float(np.std(gd)),
        float(np.mean(recip)),
        float(np.max(recip)),
    ])
    feat.extend(descriptor["feature"].tolist())
    return np.asarray(feat, dtype=np.float64)


def compare_topology_features(features: Sequence[np.ndarray]) -> pd.DataFrame:
    names = [f"Unit {idx + 1}" for idx in range(len(features))]
    matrix = np.zeros((len(features), len(features)), dtype=np.float64)
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            distance = float(np.linalg.norm(features[i] - features[j]))
            matrix[i, j] = matrix[j, i] = distance
    return pd.DataFrame(matrix, index=names, columns=names)


# ---------------------------------------------------------------------------
# Persistent homology (SR-006)
# ---------------------------------------------------------------------------

def compute_ph_diagrams(cloud: np.ndarray, cfg: TDAConfig) -> PHResult | None:
    """Compute Vietoris-Rips persistence diagrams via ripser."""
    if not PH_AVAILABLE or not cfg.ph_enabled:
        return None
    cloud_sub = subsample_point_cloud(cloud, cfg.ph_max_points, cfg.subsample)
    cloud_norm = _normalize_columns(cloud_sub)
    dists = cdist(cloud_norm, cloud_norm)
    thresh = float(np.quantile(dists, cfg.ph_thresh_quantile))
    return _ripser_fn(cloud_norm, maxdim=cfg.ph_maxdim, thresh=thresh)


def compute_ph_distance_matrix(
    ph_results: dict[str, PHResult | None], dim: int = 1
) -> pd.DataFrame | None:
    if not PH_AVAILABLE:
        return None
    keys = [k for k, v in ph_results.items() if v is not None]
    if not keys:
        return None
    n = len(keys)
    mat = np.zeros((n, n), dtype=np.float64)
    for i, ka in enumerate(keys):
        for j, kb in enumerate(keys):
            if i == j:
                continue
            da = (
                ph_results[ka]["dgms"][dim]
                if dim < len(ph_results[ka]["dgms"])
                else np.empty((0, 2))
            )
            db = (
                ph_results[kb]["dgms"][dim]
                if dim < len(ph_results[kb]["dgms"])
                else np.empty((0, 2))
            )
            try:
                mat[i, j] = _persim_bottleneck(da, db)
            except Exception:
                mat[i, j] = np.nan
    return pd.DataFrame(mat, index=keys, columns=keys)
