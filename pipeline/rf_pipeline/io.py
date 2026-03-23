"""Touchstone / S2P file loading and ingestion validation.

Supports Touchstone v1.1 (Hz RI) and detects v2.x keyword blocks for
two-port ordering.  All provenance fields are surfaced in the returned
``S2PBundle`` so the downstream ingestion-diagnostics CSV is fully traceable.

Extending
---------
To support a new file format, add a loader function with the same signature
as ``load_s2p`` and call it from ``runner.analyze`` based on file extension.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import skrf as rf
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "scikit-rf is required. Install it before running this script."
    ) from exc

from .config import S2PBundle


def _read_option_line_and_comments(path: Path) -> tuple[str, list[str]]:
    option = ""
    comments: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("!"):
                comments.append(stripped)
                continue
            if stripped.startswith("#"):
                option = stripped
                break
    if not option:
        raise ValueError(f"No Touchstone option line found in {path}")
    return option, comments


def _parse_option_line(option_line: str) -> dict[str, str]:
    """Parse Touchstone v1.1 option line into explicit field values (SR-002).

    Extracts frequency unit, parameter type, data format (RI/MA/DB), and
    reference impedance from the ``#`` option line.  All fields are logged
    to the ingestion-diagnostics CSV so provenance is fully traceable.
    """
    parts = option_line.lstrip("#").split()
    freq_unit = "GHz"
    parameter = "S"
    data_format = "MA"
    z0_ohm = "50.0"
    i = 0
    while i < len(parts):
        upper = parts[i].upper()
        if upper in ("HZ", "KHZ", "MHZ", "GHZ"):
            freq_unit = upper
        elif upper in ("S", "Y", "Z", "H", "G"):
            parameter = upper
        elif upper in ("RI", "MA", "DB"):
            data_format = upper
        elif upper == "R" and i + 1 < len(parts):
            z0_ohm = parts[i + 1]
            i += 1
        i += 1
    return {
        "version": "1.1",
        "freq_unit": freq_unit,
        "parameter": parameter,
        "data_format": data_format,
        "z0_ohm": z0_ohm,
        "two_port_order": "N11_N21_N12_N22",
        "two_port_order_source": "v1.1_default",
    }


def _detect_v2_keywords(path: Path, meta: dict[str, str]) -> dict[str, str]:
    """Scan for Touchstone v2.x keyword blocks and update *meta* in-place (SR-003).

    Looks for ``[Version]`` and ``[Two-Port Data Order]`` keywords.  When
    ``[Two-Port Data Order]`` is present it overrides the v1.1 default ordering
    and the provenance source is recorded so that all audit logs show exactly
    which ordering rule was applied.
    """
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            low = stripped.lower()
            if low.startswith("[version]"):
                parts = stripped.split()
                if len(parts) >= 2:
                    meta["version"] = parts[-1]
            elif low.startswith("[two-port data order]"):
                parts = stripped.split()
                if len(parts) >= 2:
                    meta["two_port_order"] = parts[-1]
                    meta["two_port_order_source"] = "v2_keyword"
    return meta


def load_s2p(path: str | Path) -> S2PBundle:
    """Load a 2-port Touchstone file and return an ``S2PBundle``.

    Parameters
    ----------
    path:
        Path to a ``.s2p`` file (Touchstone v1.1 or v2.x).

    Returns
    -------
    S2PBundle
        Validated bundle with freq_hz (Hz), s (N,2,2) complex128, and full
        provenance metadata.

    Raises
    ------
    ValueError
        If the file does not contain 2-port data or has a non-monotonic
        frequency axis.
    RuntimeError
        If scikit-rf is not installed.
    """
    p = Path(path)
    option_line, comments = _read_option_line_and_comments(p)
    touchstone_meta = _parse_option_line(option_line)
    _detect_v2_keywords(p, touchstone_meta)
    net = rf.Network(str(p))
    if net.nports != 2:
        raise ValueError(f"Expected 2-port data, got {net.nports} in {p}")
    if not np.all(np.diff(net.f) > 0):
        raise ValueError(f"Frequency axis is not strictly increasing: {p}")
    return S2PBundle(
        name=p.stem,
        path=p,
        freq_hz=np.asarray(net.f, dtype=np.float64),
        s=np.asarray(net.s, dtype=np.complex128),
        z0=np.asarray(net.z0),
        raw_option_line=option_line,
        comments=comments,
        touchstone_meta=touchstone_meta,
    )
