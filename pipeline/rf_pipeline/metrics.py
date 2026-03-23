"""Frequency-domain RF metrics for 2-port S-parameters.

Public API
----------
mag_db, robust_group_delay, passivity_metrics, reciprocity_error,
extract_trace, frequency_metrics, choose_scalar_series,
build_window_feature_matrix

Extending
---------
Add new feature columns to the DataFrame returned by ``frequency_metrics``
and update ``build_window_feature_matrix`` if the new columns should feed
the window autoencoder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from .config import TRACE_TO_INDEX, S2PBundle, _validate_trace


def mag_db(x: np.ndarray, floor_db: float = -300.0) -> np.ndarray:
    out = 20.0 * np.log10(np.maximum(np.abs(x), 10.0 ** (floor_db / 20.0)))
    return out.astype(np.float64)


def robust_group_delay(
    freq_hz: np.ndarray,
    s_trace: np.ndarray,
    sg_window: int = 15,
    polyorder: int = 3,
) -> np.ndarray:
    phase = np.unwrap(np.angle(s_trace))
    omega = 2.0 * np.pi * freq_hz
    d_omega = float(np.mean(np.diff(omega)))
    win = min(sg_window, len(phase) - (1 - len(phase) % 2))
    if win < 5:
        win = 5 if len(phase) >= 5 else max(3, len(phase) // 2 * 2 + 1)
    if win % 2 == 0:
        win += 1
    if win <= polyorder:
        polyorder = max(1, win - 2)
    dphi_domega = savgol_filter(
        phase,
        window_length=win,
        polyorder=polyorder,
        deriv=1,
        delta=d_omega,
        mode="interp",
    )
    return -dphi_domega


def passivity_metrics(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return max singular value and min eigenvalue of I − S S^H per frequency."""
    sigma_max = np.empty(s.shape[0], dtype=np.float64)
    q_min_eig = np.empty(s.shape[0], dtype=np.float64)
    eye = np.eye(2, dtype=np.complex128)
    for i in range(s.shape[0]):
        sigma_max[i] = float(np.max(np.linalg.svd(s[i], compute_uv=False)))
        q = eye - s[i] @ s[i].conj().T
        q_min_eig[i] = float(np.min(np.real(np.linalg.eigvals(q))))
    return sigma_max, q_min_eig


def reciprocity_error(s: np.ndarray) -> np.ndarray:
    return np.abs(s[:, 1, 0] - s[:, 0, 1]).astype(np.float64)


def extract_trace(s: np.ndarray, trace: str) -> np.ndarray:
    _validate_trace(trace)
    r, c = TRACE_TO_INDEX[trace]
    return s[:, r, c]


def frequency_metrics(bundle: S2PBundle) -> pd.DataFrame:
    sigma_max, q_min_eig = passivity_metrics(bundle.s)
    recip = reciprocity_error(bundle.s)
    s11 = extract_trace(bundle.s, "S11")
    s21 = extract_trace(bundle.s, "S21")
    s12 = extract_trace(bundle.s, "S12")
    s22 = extract_trace(bundle.s, "S22")
    gd21 = robust_group_delay(bundle.freq_hz, s21)
    return pd.DataFrame(
        {
            "freq_hz": bundle.freq_hz,
            "s11_re": s11.real,
            "s11_im": s11.imag,
            "s11_mag": np.abs(s11),
            "s11_db": mag_db(s11),
            "s21_re": s21.real,
            "s21_im": s21.imag,
            "s21_mag": np.abs(s21),
            "s21_db": mag_db(s21),
            "s21_phase_rad": np.angle(s21),
            "s21_phase_unwrap_rad": np.unwrap(np.angle(s21)),
            "s21_phase_deg": np.rad2deg(np.angle(s21)),
            "s12_mag": np.abs(s12),
            "s22_mag": np.abs(s22),
            "group_delay_s21_s": gd21,
            "passivity_sigma_max": sigma_max,
            "passivity_margin": 1.0 - sigma_max,
            "passivity_q_min_eig": q_min_eig,
            "reciprocity_abs_s21_minus_s12": recip,
        }
    )


def choose_scalar_series(metrics_df: pd.DataFrame, name: str) -> np.ndarray:
    """Select a 1-D scalar series from a frequency-metrics DataFrame."""
    if name == "S11_db":
        return metrics_df["s11_db"].to_numpy(dtype=np.float64)
    if name == "group_delay":
        return metrics_df["group_delay_s21_s"].to_numpy(dtype=np.float64)
    return metrics_df["s21_db"].to_numpy(dtype=np.float64)


def build_window_feature_matrix(metrics_df: pd.DataFrame) -> np.ndarray:
    """Stack and z-score 6 RF channels into a (N_freq, 6) float32 matrix."""
    cols = [
        metrics_df["s11_db"].to_numpy(),
        metrics_df["s21_db"].to_numpy(),
        metrics_df["s21_phase_unwrap_rad"].to_numpy(),
        metrics_df["group_delay_s21_s"].to_numpy(),
        metrics_df["reciprocity_abs_s21_minus_s12"].to_numpy(),
        metrics_df["passivity_margin"].to_numpy(),
    ]
    x = np.column_stack(cols).astype(np.float32)
    x = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
    return x
