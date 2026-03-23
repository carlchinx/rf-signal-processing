"""Frequency-grid unification and S-parameter interpolation.

Extending
---------
To add a new interpolation method, add a key→function entry to the
``interp_map`` dict inside ``interpolate_bundle`` and list it in the
``InterpConfig.method`` Literal type in ``config.py``.
"""
from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from .config import InterpConfig, S2PBundle


def build_common_grid(bundles: Sequence[S2PBundle], cfg: InterpConfig) -> np.ndarray:
    if cfg.grid_mode == "custom":
        if cfg.custom_start_hz is None or cfg.custom_stop_hz is None or cfg.n_points is None:
            raise ValueError("custom grid requires start/stop/n_points")
        return np.linspace(cfg.custom_start_hz, cfg.custom_stop_hz, cfg.n_points, dtype=np.float64)

    start = max(b.freq_hz.min() for b in bundles)
    stop = min(b.freq_hz.max() for b in bundles)
    if not start < stop:
        raise ValueError("Frequency intersection is empty.")
    n_points = cfg.n_points or min(len(b.freq_hz) for b in bundles)
    return np.linspace(start, stop, n_points, dtype=np.float64)


def _interp_complex_linear(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y.real) + 1j * np.interp(x_new, x, y.imag)


def _interp_complex_cubic(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    re = CubicSpline(x, y.real, extrapolate=False)(x_new)
    im = CubicSpline(x, y.imag, extrapolate=False)(x_new)
    result = re + 1j * im
    if np.any(np.isnan(result)):
        warnings.warn(
            "NaN values produced during cubic spline interpolation; "
            "check frequency grid overlap."
        )
        result = np.nan_to_num(result, nan=0.0)
    return result


def _interp_magphase_pchip(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    mag = np.abs(y)
    phase = np.unwrap(np.angle(y))
    mag_i = PchipInterpolator(x, mag, extrapolate=False)(x_new)
    phase_i = PchipInterpolator(x, phase, extrapolate=False)(x_new)
    result = mag_i * np.exp(1j * phase_i)
    if np.any(np.isnan(result)):
        warnings.warn(
            "NaN values produced during PCHIP interpolation; "
            "check frequency grid overlap."
        )
        result = np.nan_to_num(result, nan=0.0)
    return result


def interpolate_bundle(
    bundle: S2PBundle, common_f: np.ndarray, cfg: InterpConfig
) -> tuple[S2PBundle, dict[str, float]]:
    interp_map = {
        "complex_linear": _interp_complex_linear,
        "complex_cubic": _interp_complex_cubic,
        "magphase_pchip": _interp_magphase_pchip,
    }
    interp_fn = interp_map[cfg.method]

    s_new = np.empty((len(common_f), 2, 2), dtype=np.complex128)
    for r in range(2):
        for c in range(2):
            s_new[:, r, c] = interp_fn(bundle.freq_hz, bundle.s[:, r, c], common_f)

    # Error bound: compare original magnitude to back-interpolated magnitude of S21.
    s21_back = interp_fn(common_f, s_new[:, 1, 0], bundle.freq_hz)
    interp_error = float(np.max(np.abs(np.abs(bundle.s[:, 1, 0]) - np.abs(s21_back))))

    z0 = np.repeat(
        np.atleast_2d(bundle.z0[0]).astype(np.complex128), len(common_f), axis=0
    )
    out = S2PBundle(
        name=bundle.name,
        path=bundle.path,
        freq_hz=common_f,
        s=s_new,
        z0=z0,
        raw_option_line=bundle.raw_option_line,
        comments=bundle.comments,
        touchstone_meta=bundle.touchstone_meta,
    )
    return out, {"interp_error_mag_s21": interp_error}
