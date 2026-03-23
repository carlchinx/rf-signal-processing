"""Vector fitting and synthetic inverse-characterization dataset construction.

The inverse-dataset pipeline perturbs vector-fit poles/residues to generate
synthetic S-parameter variants, filters passivity violations, then extracts
topology + AE features as training inputs.

Extending
---------
To change the synthetic perturbation strategy, replace ``perturb_vf_state``
with a new function having the same signature and update ``build_inverse_dataset``
to call it.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:
    import skrf as rf
except Exception as exc:
    raise RuntimeError("scikit-rf is required.") from exc

try:
    from skrf.vectorFitting import VectorFitting as _VF
except ImportError:
    _VF = rf.VectorFitting

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
except Exception:
    StandardScaler = None  # type: ignore[assignment,misc]
    PCA = None  # type: ignore[assignment]

from .config import (
    InverseConfig,
    InverseMeta,
    ROW_MAJOR_TRACE_ORDER,
    S2PBundle,
    TDAConfig,
    VFConfig,
    VectorFitState,
)
from .metrics import (
    choose_scalar_series,
    frequency_metrics,
    passivity_metrics,
)
from .topology import (
    build_complex_topology_cloud,
    build_shift_topology_cloud,
    build_topology_descriptor,
    choose_lag_autocorr as _cla,
    make_tda_feature_vector,
)


# ---------------------------------------------------------------------------
# Vector model fitting
# ---------------------------------------------------------------------------

def fit_vector_model(bundle: S2PBundle, cfg: VFConfig) -> VectorFitState:
    net = rf.Network()
    net.frequency = rf.Frequency.from_f(bundle.freq_hz, unit="Hz")
    net.s = bundle.s.copy()
    net.z0 = bundle.z0.copy()

    vf = _VF(net)
    vf.vector_fit(
        n_poles_real=cfg.n_poles_real,
        n_poles_cmplx=cfg.n_poles_cmplx,
        parameter_type="s",
        fit_constant=cfg.fit_constant,
        fit_proportional=cfg.fit_proportional,
        enforce_dc=True,
    )
    passive_before = None
    violation_bands = None
    try:
        passive_before = bool(vf.is_passive())
        try:
            violation_bands = vf.passivity_test()
        except Exception as e:
            warnings.warn(f"Passivity test failed for {bundle.name}: {e}")
            violation_bands = None
        if cfg.passivity_enforce and not passive_before:
            vf.passivity_enforce(
                n_samples=cfg.passivity_n_samples,
                f_max=float(bundle.freq_hz.max()),
                parameter_type="s",
                preserve_dc=True,
            )
    except Exception as e:
        warnings.warn(f"Passivity check/enforce failed for {bundle.name}: {e}")
        passive_before = None
    passive_after = None
    try:
        passive_after = bool(vf.is_passive())
    except Exception as e:
        warnings.warn(f"Post-enforce passivity check failed for {bundle.name}: {e}")
        passive_after = None

    rms_error = None
    try:
        rms_error = float(vf.get_rms_error())
    except Exception as e:
        warnings.warn(f"RMS error retrieval failed for {bundle.name}: {e}")
        rms_error = None

    return VectorFitState(
        name=bundle.name,
        vf=vf,
        poles=np.asarray(vf.poles, dtype=np.complex128),
        residues=np.asarray(vf.residues, dtype=np.complex128),
        constant_coeff=np.asarray(vf.constant_coeff, dtype=np.complex128),
        proportional_coeff=np.asarray(vf.proportional_coeff, dtype=np.complex128),
        rms_error=rms_error,
        passive_before=passive_before,
        passive_after=passive_after,
        violation_bands_before=(
            np.asarray(violation_bands) if violation_bands is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# VF response synthesis
# ---------------------------------------------------------------------------

def _vf_response_1d(
    freq_hz: np.ndarray,
    poles: np.ndarray,
    residues_1d: np.ndarray,
    d: complex,
    e: complex,
) -> np.ndarray:
    s = 1j * 2.0 * np.pi * freq_hz
    h = np.full_like(s, fill_value=d, dtype=np.complex128) + s * e
    for pole, residue in zip(poles, residues_1d):
        if abs(pole.imag) < 1e-15:
            h += residue / (s - pole)
        else:
            h += residue / (s - pole) + np.conj(residue) / (s - np.conj(pole))
    return h


def vf_to_s(
    freq_hz: np.ndarray,
    poles: np.ndarray,
    residues: np.ndarray,
    constants: np.ndarray,
    proportionals: np.ndarray,
) -> np.ndarray:
    s_mat = np.empty((len(freq_hz), 2, 2), dtype=np.complex128)
    traces = [
        _vf_response_1d(freq_hz, poles, residues[idx], constants[idx], proportionals[idx])
        for idx in range(4)
    ]
    s_mat[:, 0, 0] = traces[0]
    s_mat[:, 0, 1] = traces[1]
    s_mat[:, 1, 0] = traces[2]
    s_mat[:, 1, 1] = traces[3]
    return s_mat


# ---------------------------------------------------------------------------
# Perturbation helpers
# ---------------------------------------------------------------------------

def sample_stable_poles(
    poles: np.ndarray, real_sigma: float, imag_sigma: float
) -> np.ndarray:
    out = poles.copy()
    for i, p in enumerate(out):
        re = -abs(p.real) * np.exp(real_sigma * np.random.randn())
        im = 0.0 if abs(p.imag) < 1e-15 else max(1e-6, abs(p.imag) * np.exp(imag_sigma * np.random.randn()))
        out[i] = re + 1j * im
    return out


def perturb_vf_state(
    state: VectorFitState, cfg: InverseConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poles = sample_stable_poles(state.poles, cfg.pole_real_sigma, cfg.pole_imag_sigma)
    residues = state.residues * (
        1.0
        + cfg.residue_sigma
        * (np.random.randn(*state.residues.shape) + 1j * np.random.randn(*state.residues.shape))
    )
    constants = state.constant_coeff * (
        1.0
        + cfg.constant_sigma
        * (np.random.randn(*state.constant_coeff.shape) + 1j * np.random.randn(*state.constant_coeff.shape))
    )
    proportionals = state.proportional_coeff * (
        1.0
        + cfg.proportional_sigma
        * (
            np.random.randn(*state.proportional_coeff.shape)
            + 1j * np.random.randn(*state.proportional_coeff.shape)
        )
    )
    return poles, residues, constants, proportionals


def flatten_vf_params(
    poles: np.ndarray,
    residues: np.ndarray,
    constants: np.ndarray,
    proportionals: np.ndarray,
) -> np.ndarray:
    vec = []
    vec.extend(np.real(poles).tolist())
    vec.extend(np.imag(poles).tolist())
    for tr_name in ["S11", "S21"]:
        idx = ROW_MAJOR_TRACE_ORDER.index(tr_name)
        vec.extend(np.real(residues[idx]).tolist())
        vec.extend(np.imag(residues[idx]).tolist())
        vec.append(float(
            np.real(constants[idx]).item()
            if np.ndim(constants[idx]) == 0
            else np.real(constants[idx])
        ))
        vec.append(float(
            np.imag(constants[idx]).item()
            if np.ndim(constants[idx]) == 0
            else np.imag(constants[idx])
        ))
        vec.append(float(
            np.real(proportionals[idx]).item()
            if np.ndim(proportionals[idx]) == 0
            else np.real(proportionals[idx])
        ))
        vec.append(float(
            np.imag(proportionals[idx]).item()
            if np.ndim(proportionals[idx]) == 0
            else np.imag(proportionals[idx])
        ))
    return np.asarray(vec, dtype=np.float64)


def s_bundle_from_array(name: str, freq_hz: np.ndarray, s: np.ndarray) -> S2PBundle:
    z0 = np.full((len(freq_hz), 2), 50.0 + 0.0j, dtype=np.complex128)
    return S2PBundle(
        name=name,
        path=Path(name),
        freq_hz=freq_hz,
        s=s,
        z0=z0,
        raw_option_line="# Hz S RI R 50",
        comments=[],
    )


# ---------------------------------------------------------------------------
# Inverse dataset construction
# ---------------------------------------------------------------------------

def build_inverse_dataset(
    bundles: Sequence[S2PBundle],
    vf_states: Sequence[VectorFitState],
    tda_cfg: TDAConfig,
    inv_cfg: InverseConfig,
    ae_descriptor_lookup: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, InverseMeta]:
    if StandardScaler is None or PCA is None:
        raise RuntimeError(
            "scikit-learn is required for inverse dataset construction."
        )

    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    meta: InverseMeta = {"source_unit": []}

    for bundle, state in zip(bundles, vf_states):
        for _ in range(inv_cfg.n_synthetic_per_unit):
            poles, residues, constants, proportionals = perturb_vf_state(state, inv_cfg)
            s_syn = vf_to_s(bundle.freq_hz, poles, residues, constants, proportionals)
            sigma_max, _ = passivity_metrics(s_syn)
            if float(np.max(sigma_max)) > 1.0 + inv_cfg.passivity_reject_tol:
                continue
            syn_bundle = s_bundle_from_array(f"{bundle.name}_syn", bundle.freq_hz, s_syn)
            fmet = frequency_metrics(syn_bundle)
            complex_cloud, complex_index, complex_axes = build_complex_topology_cloud(fmet, tda_cfg)
            complex_descriptor = build_topology_descriptor(
                complex_cloud, complex_index, syn_bundle.freq_hz, complex_axes, tda_cfg
            )
            feat = make_tda_feature_vector(fmet, complex_descriptor, tda_cfg)

            scalar = choose_scalar_series(fmet, tda_cfg.shift_scalar)
            lag = tda_cfg.shift_lag or _cla(
                scalar, tda_cfg.shift_lag_min, tda_cfg.shift_lag_max,
                window=tda_cfg.shift_window,
            )
            shift_cloud, shift_index, shift_axes = build_shift_topology_cloud(fmet, lag, tda_cfg)
            shift_descriptor = build_topology_descriptor(
                shift_cloud, shift_index, syn_bundle.freq_hz, shift_axes, tda_cfg
            )
            feat_shift = make_tda_feature_vector(fmet, shift_descriptor, tda_cfg)
            feat = np.concatenate([feat, feat_shift])
            if ae_descriptor_lookup is not None and bundle.name in ae_descriptor_lookup:
                feat = np.concatenate([feat, ae_descriptor_lookup[bundle.name]])

            target = flatten_vf_params(poles, residues, constants, proportionals)
            x_rows.append(feat)
            y_rows.append(target)
            meta["source_unit"].append(bundle.name)

    if len(x_rows) == 0:
        raise RuntimeError(
            "All synthetic samples were rejected by the passivity filter. "
            "Increase n_synthetic_per_unit or relax passivity_reject_tol."
        )
    x = np.vstack(x_rows).astype(np.float32)
    y = np.vstack(y_rows).astype(np.float32)
    return x, y, meta
