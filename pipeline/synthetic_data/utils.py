"""Shared S2P I/O and RF feature extraction utilities for the synthetic_data scripts.

Both ``generate_synthetic.py`` and ``compare_models.py`` previously contained
identical copies of ``parse_s2p`` and the 15-feature ``extract_rf_features``
function.  This module provides a single canonical implementation so fixes and
additions propagate automatically to both scripts.

Usage::

    from utils import parse_s2p, write_s2p, extract_rf_features

API
---
parse_s2p(path)
    Read a Touchstone 1.1 S2P file (Hz, RI format).
    Returns ``(freq_hz, s_complex)`` as NumPy arrays.

write_s2p(path, freq, s, comment="")
    Write a Touchstone 1.1 S2P file in Hz RI R 50 notation.

extract_rf_features(freq, s)
    Compute 15 scalar RF quality features from complex S-parameter arrays.
    Returns a plain ``dict`` ready for a ``pandas.DataFrame`` row.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def parse_s2p(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a Touchstone 1.1 S2P file in Hz, RI format.

    Parameters
    ----------
    path:
        Path to the ``.s2p`` file.

    Returns
    -------
    freq:
        1-D float64 array of frequencies in Hz, shape ``(N,)``.
    s:
        2-D complex128 array of S-parameters, shape ``(N, 4)`` with columns
        ``[S11, S21, S12, S22]`` (row-major Touchstone 1.1 order).
    """
    freq: list[float] = []
    rows: list[list[complex]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("!") or line.startswith("#"):
                continue
            vals = list(map(float, line.split()))
            if len(vals) < 9:
                continue
            freq.append(vals[0])
            rows.append([
                complex(vals[1], vals[2]),   # S11
                complex(vals[3], vals[4]),   # S21
                complex(vals[5], vals[6]),   # S12
                complex(vals[7], vals[8]),   # S22
            ])
    return np.array(freq, dtype=np.float64), np.array(rows, dtype=complex)


def write_s2p(
    path: Path,
    freq: np.ndarray,
    s: np.ndarray,
    comment: str = "",
) -> None:
    """Write a Touchstone 1.1 S2P file in Hz, RI, R 50 format.

    Parameters
    ----------
    path:
        Destination file path.  Parent directories must exist.
    freq:
        Frequency array in Hz, shape ``(N,)``.
    s:
        Complex S-parameter array, shape ``(N, 4)`` — ``[S11, S21, S12, S22]``.
    comment:
        Optional single-line comment written after the header (no ``!`` prefix
        needed — it will be added automatically).
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("! Synthetic 2-port S-parameter data\n")
        if comment:
            fh.write(f"! {comment}\n")
        fh.write("# Hz S RI R 50\n")
        for i, fi in enumerate(freq):
            parts: list[str] = []
            for c in s[i]:
                parts.append(f"{c.real:.8e}")
                parts.append(f"{c.imag:.8e}")
            fh.write(f"{fi:.6e}  " + "  ".join(parts) + "\n")


def extract_rf_features(freq: np.ndarray, s: np.ndarray) -> dict:
    """Compute 15 scalar RF quality features from complex S-parameter arrays.

    Parameters
    ----------
    freq:
        Frequency array in Hz, shape ``(N,)``.
    s:
        Complex S-parameter array, shape ``(N, 4)`` — ``[S11, S21, S12, S22]``.

    Returns
    -------
    dict
        Keys and descriptions:

        * ``s21_max_db`` — peak |S21| in dB
        * ``s21_min_db`` — minimum |S21| in dB
        * ``s21_pb_rms_db`` — RMS |S21| within the 3 dB passband
        * ``pb_ripple_db`` — peak-to-peak S21 ripple inside passband
        * ``bw_3db_mhz`` — 3 dB bandwidth in MHz
        * ``f_center_ghz`` — geometric centre of passband in GHz
        * ``f_3db_low_ghz`` — lower 3 dB edge in GHz
        * ``f_3db_high_ghz`` — upper 3 dB edge in GHz
        * ``gd_mean_ns`` — mean group delay inside passband in ns
        * ``gd_std_ps`` — group delay standard deviation in ps
        * ``gd_peak_ns`` — peak absolute group delay in ns
        * ``s11_pb_mean_db`` — mean |S11| inside passband in dB
        * ``s22_pb_mean_db`` — mean |S22| inside passband in dB
        * ``reciprocity_max`` — max |S21 − S12| (reciprocity error)
        * ``passivity_margin`` — 1 − max(|S11|² + |S21|²), negative = non-passive
    """
    eps = 1e-30

    s21_db = 20.0 * np.log10(np.abs(s[:, 1]) + eps)
    s11_db = 20.0 * np.log10(np.abs(s[:, 0]) + eps)
    s22_db = 20.0 * np.log10(np.abs(s[:, 3]) + eps)

    s21_max_db = float(np.max(s21_db))
    s21_min_db = float(np.min(s21_db))

    # 3 dB passband mask
    pb_mask = s21_db >= s21_max_db - 3.0
    f_pb = freq[pb_mask]

    bw_3db_mhz     = float((f_pb[-1] - f_pb[0]) / 1e6)  if f_pb.size > 1 else 0.0
    f_center_ghz   = float(np.mean(f_pb) / 1e9)          if f_pb.size > 0 else float(np.mean(freq) / 1e9)
    f_3db_low_ghz  = float(f_pb[0]  / 1e9)               if f_pb.size > 0 else float(freq[0]  / 1e9)
    f_3db_high_ghz = float(f_pb[-1] / 1e9)               if f_pb.size > 0 else float(freq[-1] / 1e9)

    pb_ripple_db = (
        float(np.max(s21_db[pb_mask]) - np.min(s21_db[pb_mask]))
        if pb_mask.any()
        else 0.0
    )

    # Group delay (unwrapped phase derivative)
    phase21 = np.unwrap(np.angle(s[:, 1]))
    gd_s = -np.diff(phase21) / (2.0 * np.pi * np.diff(freq))
    gd_pb = gd_s[pb_mask[:-1]] if pb_mask[:-1].any() else gd_s
    gd_mean_ns = float(np.mean(gd_pb) * 1e9)
    gd_std_ps  = float(np.std(gd_pb)  * 1e12)
    gd_peak_ns = float(np.max(np.abs(gd_pb)) * 1e9)

    # Reflection inside passband
    s11_pb_mean_db = float(np.mean(s11_db[pb_mask])) if pb_mask.any() else float(np.mean(s11_db))
    s22_pb_mean_db = float(np.mean(s22_db[pb_mask])) if pb_mask.any() else float(np.mean(s22_db))

    # Reciprocity |S21 – S12|
    recip_max = float(np.max(np.abs(s[:, 1] - s[:, 2])))

    # Passivity proxy: max(|S11|² + |S21|²) ≤ 1 for lossless passive
    sigma_proxy = np.abs(s[:, 0]) ** 2 + np.abs(s[:, 1]) ** 2
    passivity_margin = float(1.0 - np.max(sigma_proxy))

    # RMS S21 over passband
    s21_pb_rms_db = (
        float(np.sqrt(np.mean(s21_db[pb_mask] ** 2)))
        if pb_mask.any()
        else s21_max_db
    )

    return {
        "s21_max_db"      : s21_max_db,
        "s21_min_db"      : s21_min_db,
        "s21_pb_rms_db"   : s21_pb_rms_db,
        "pb_ripple_db"    : pb_ripple_db,
        "bw_3db_mhz"      : bw_3db_mhz,
        "f_center_ghz"    : f_center_ghz,
        "f_3db_low_ghz"   : f_3db_low_ghz,
        "f_3db_high_ghz"  : f_3db_high_ghz,
        "gd_mean_ns"      : gd_mean_ns,
        "gd_std_ps"       : gd_std_ps,
        "gd_peak_ns"      : gd_peak_ns,
        "s11_pb_mean_db"  : s11_pb_mean_db,
        "s22_pb_mean_db"  : s22_pb_mean_db,
        "reciprocity_max" : recip_max,
        "passivity_margin": passivity_margin,
    }
