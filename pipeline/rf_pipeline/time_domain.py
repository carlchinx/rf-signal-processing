"""Time-domain transforms: IFFT impulse/step responses.

Extending
---------
To add a new window function, pass any name understood by
``scipy.signal.get_window`` to ``TDConfig.window_name``.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import get_window

from .config import TDConfig


def apply_frequency_window(
    x: np.ndarray, window_name: str, window_kwargs: dict[str, Any]
) -> np.ndarray:
    n = len(x)
    try:
        w = get_window((window_name, *window_kwargs.values()), n, fftbins=True)
    except Exception:
        try:
            w = get_window(window_name, n, fftbins=True)
        except Exception:
            w = np.ones(n, dtype=np.float64)
    return x * w


def insert_dc(
    freq_hz: np.ndarray, x: np.ndarray, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    if np.isclose(freq_hz[0], 0.0):
        return freq_hz, x
    x0 = x[0] if mode == "hold_first" else (0.0 + 0.0j)
    return np.concatenate([[0.0], freq_hz]), np.concatenate([[x0], x])


def time_domain_from_trace(
    freq_hz: np.ndarray, x: np.ndarray, cfg: TDConfig
) -> pd.DataFrame:
    f, trace = insert_dc(freq_hz, x, cfg.dc_handling)
    trace = apply_frequency_window(trace, cfg.window_name, cfg.window_kwargs)
    df = float(np.mean(np.diff(f)))

    if cfg.hermitian_real_signal:
        spec = np.concatenate([trace, np.conj(trace[-2:0:-1])])
    else:
        spec = trace

    n_fft = cfg.n_fft or (1 << int(math.ceil(math.log2(len(spec)))))
    spec = np.pad(spec, (0, max(0, n_fft - len(spec))))
    impulse = np.fft.ifft(spec, n=n_fft)
    if cfg.continuous_scale_df:
        impulse = impulse * df
    dt = 1.0 / (n_fft * df)
    time_s = np.arange(n_fft, dtype=np.float64) * dt
    step = np.cumsum(impulse) * dt
    return pd.DataFrame(
        {
            "time_s": time_s,
            "impulse_re": impulse.real,
            "impulse_im": impulse.imag,
            "impulse_mag": np.abs(impulse),
            "step_re": step.real,
            "step_im": step.imag,
            "step_mag": np.abs(step),
        }
    )
