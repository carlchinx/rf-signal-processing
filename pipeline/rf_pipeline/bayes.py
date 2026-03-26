"""Bayesian uncertainty quantification for S-parameter measurements (SR-004, SR-005).

Models each S-parameter channel as a complex-Gaussian posterior:
    Re[S(f)] ~ N(smooth(Re[S_obs(f)]), σ_re²)
    Im[S(f)] ~ N(smooth(Im[S_obs(f)]), σ_im²)

Noise sigma is estimated from Savitzky-Golay residuals (diagnostics-first).
Posterior samples propagate uncertainty through group delay, time-domain IFFT,
and (when ripser/persim are available) persistent-homology diagrams.

Extending
---------
To add a Student-t likelihood, replace the Gaussian sampling block in
``run_bayesian_inference`` while keeping the ``BayesResult`` TypedDict contract.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.distance import cdist

try:
    from persim import bottleneck as _persim_bottleneck
    from ripser import ripser as _ripser_fn
    _PH_AVAILABLE: bool = True
except Exception:
    _ripser_fn = None  # type: ignore[assignment]
    _persim_bottleneck = None  # type: ignore[assignment]
    _PH_AVAILABLE = False

from .config import TRACE_TO_INDEX, BayesConfig, BayesResult, TDAConfig, TDConfig
from .metrics import mag_db, robust_group_delay
from .time_domain import time_domain_from_trace
from .topology import _normalize_columns, subsample_point_cloud

# ---------------------------------------------------------------------------
# HDI utilities
# ---------------------------------------------------------------------------

def _hdi_min_width(samples: np.ndarray, cred_mass: float) -> tuple[float, float]:
    """Minimum-width Bayesian credible interval (HDI).

    Equivalent to ``arviz.hdi(samples, hdi_prob=cred_mass)`` for 1-D arrays:
    find the shortest interval that contains at least ``cred_mass`` fraction
    of the sorted samples.
    """
    x = np.sort(np.asarray(samples, dtype=np.float64).ravel())
    n = len(x)
    n_in = int(math.floor(cred_mass * n))
    if n_in >= n:
        return float(x[0]), float(x[-1])
    widths = x[n_in:] - x[: n - n_in]
    idx = int(np.argmin(widths))
    return float(x[idx]), float(x[idx + n_in])


def _estimate_noise_sigma(
    trace: np.ndarray, window: int = 21, polyorder: int = 3
) -> tuple[float, float]:
    """Estimate Re/Im noise sigma from Savitzky-Golay residuals."""
    n = len(trace)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win < 5:
        win = 5 if n >= 5 else max(3, n // 2 * 2 + 1)
    if win % 2 == 0:
        win += 1
    po = min(polyorder, max(1, win - 2))
    re_res = trace.real - savgol_filter(trace.real, win, po)
    im_res = trace.imag - savgol_filter(trace.imag, win, po)
    return float(np.std(re_res)) + 1e-15, float(np.std(im_res)) + 1e-15


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def run_bayesian_inference(
    bundle: S2PBundle,  # noqa: F821 — imported in runner, not needed here
    cfg: BayesConfig,
    rng: np.random.Generator,
) -> BayesResult:
    """Sample from the complex-Gaussian posterior for all four S-parameters.

    Likelihood per channel per frequency point::

        Re[S_obs(f_k)] ~ N(mu_re(f_k), sigma_re²)
        Im[S_obs(f_k)] ~ N(mu_im(f_k), sigma_im²)

    When ``cfg.latent == 'savgol_smooth'`` the prior mean is a Savitzky-Golay
    smooth of the observation (equivalent to a smooth spline prior).
    """
    from .config import S2PBundle as _S2PBundle  # local to avoid circular at module level
    assert isinstance(bundle, _S2PBundle)

    Nf = len(bundle.freq_hz)
    n_draws = cfg.n_draws
    s_draws = np.empty((n_draws, Nf, 2, 2), dtype=np.complex128)
    sigma_re = np.zeros((2, 2), dtype=np.float64)
    sigma_im = np.zeros((2, 2), dtype=np.float64)

    for r in range(2):
        for c in range(2):
            trace = bundle.s[:, r, c]
            sr, si = _estimate_noise_sigma(trace, cfg.smooth_window, cfg.smooth_polyorder)
            sigma_re[r, c] = sr
            sigma_im[r, c] = si

            if cfg.latent == "savgol_smooth":
                win = min(cfg.smooth_window, Nf if Nf % 2 == 1 else Nf - 1)
                if win < 5:
                    win = 5 if Nf >= 5 else max(3, Nf // 2 * 2 + 1)
                if win % 2 == 0:
                    win += 1
                po = min(cfg.smooth_polyorder, max(1, win - 2))
                mu_re: np.ndarray = savgol_filter(trace.real, win, po)
                mu_im: np.ndarray = savgol_filter(trace.imag, win, po)
            else:
                mu_re = trace.real.copy()
                mu_im = trace.imag.copy()

            s_draws[:, :, r, c] = (
                mu_re[np.newaxis, :] + rng.normal(0.0, sr, (n_draws, Nf))
                + 1j * (mu_im[np.newaxis, :] + rng.normal(0.0, si, (n_draws, Nf)))
            )

    # Posterior predictive check (PPC) — one extra noise replicate per draw.
    s21_draws = s_draws[:, :, 1, 0]
    sr21, si21 = sigma_re[1, 0], sigma_im[1, 0]
    ppc_n = min(cfg.n_ppc_samples, n_draws)
    ppc_idx = rng.choice(n_draws, size=ppc_n, replace=False)
    ppc_re = s21_draws[ppc_idx].real + rng.normal(0.0, sr21, (ppc_n, Nf))
    ppc_im = s21_draws[ppc_idx].imag + rng.normal(0.0, si21, (ppc_n, Nf))
    ppc_samples_db = mag_db(ppc_re + 1j * ppc_im)

    return BayesResult(
        s_draws=s_draws,
        sigma_re=sigma_re,
        sigma_im=sigma_im,
        ppc_samples_db=ppc_samples_db,
        cred_mass=cfg.cred_mass,
    )


# ---------------------------------------------------------------------------
# HDI summaries
# ---------------------------------------------------------------------------

def compute_pointwise_hdi(
    draws: np.ndarray, cred_mass: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pointwise posterior median, HDI-low, HDI-high from a ``(n_draws, Nf)`` array."""
    median = np.median(draws, axis=0)
    Nf = draws.shape[1]
    low = np.empty(Nf, dtype=np.float64)
    high = np.empty(Nf, dtype=np.float64)
    for k in range(Nf):
        low[k], high[k] = _hdi_min_width(draws[:, k], cred_mass)
    return median, low, high


def compute_scalar_hdi_string(samples_1d: np.ndarray, cred_mass: float) -> str:
    """Format a 1-D posterior as ``'median [HDI_low, HDI_high]'`` (ArviZ convention)."""
    med = float(np.median(samples_1d))
    lo, hi = _hdi_min_width(samples_1d, cred_mass)
    return f"{med:.6g} [{lo:.6g}, {hi:.6g}]"


def build_hdi_scalar_summary(
    s_draws: np.ndarray, freq_hz: np.ndarray, cred_mass: float
) -> dict[str, str]:
    """HDI-formatted scalar summaries for |S21|, group delay, and passivity."""
    s21 = s_draws[:, :, 1, 0]
    s21_db = mag_db(s21)
    gd_draws = np.stack(
        [robust_group_delay(freq_hz, s21[d]) for d in range(len(s21))], axis=0
    )
    sigma_max_draws = np.stack(
        [
            np.array([
                float(np.max(np.linalg.svd(
                    np.array([[s_draws[d, k, 0, 0], s_draws[d, k, 0, 1]],
                               [s_draws[d, k, 1, 0], s_draws[d, k, 1, 1]]]),
                    compute_uv=False,
                )))
                for k in range(s_draws.shape[1])
            ])
            for d in range(len(s_draws))
        ],
        axis=0,
    )
    return {
        "s21_db_max": compute_scalar_hdi_string(s21_db.max(axis=1), cred_mass),
        "s21_db_min": compute_scalar_hdi_string(s21_db.min(axis=1), cred_mass),
        "s21_db_mean": compute_scalar_hdi_string(s21_db.mean(axis=1), cred_mass),
        "group_delay_mean_ns": compute_scalar_hdi_string(
            gd_draws.mean(axis=1) * 1e9, cred_mass
        ),
        "group_delay_std_ps": compute_scalar_hdi_string(
            gd_draws.std(axis=1) * 1e12, cred_mass
        ),
        "passivity_sigma_max_mean": compute_scalar_hdi_string(
            sigma_max_draws.mean(axis=1), cred_mass
        ),
    }


# ---------------------------------------------------------------------------
# Time-domain credible bands
# ---------------------------------------------------------------------------

def compute_td_credible_bands(
    s_draws: np.ndarray,
    freq_hz: np.ndarray,
    trace_rc: tuple[int, int],
    cfg: TDConfig,
    cred_mass: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate posterior draws through the full IFFT pipeline (SR-005).

    Returns ``(time_s, impulse_median, hdi_low, hdi_high)``.
    """
    r, c = trace_rc
    impulse_draws: list[np.ndarray] = []
    time_s_ref: np.ndarray | None = None
    for d in range(len(s_draws)):
        trace = s_draws[d, :, r, c]
        td_df = time_domain_from_trace(freq_hz, trace, cfg)
        if time_s_ref is None:
            time_s_ref = td_df["time_s"].to_numpy()
        impulse_draws.append(td_df["impulse_mag"].to_numpy())
    draws_arr = np.stack(impulse_draws, axis=0)
    median, low, high = compute_pointwise_hdi(draws_arr, cred_mass)
    if time_s_ref is None:
        time_s_ref = np.arange(draws_arr.shape[1], dtype=np.float64)
    return time_s_ref, median, low, high


# ---------------------------------------------------------------------------
# PH posterior draws
# ---------------------------------------------------------------------------

def compute_ph_draws_for_unit(
    s_draws: np.ndarray,
    cfg: TDAConfig,
    n_ph_draws: int,
    rng: np.random.Generator,
    traces: tuple[str, ...],
) -> list[list[np.ndarray] | None]:
    """Compute PH diagrams on a random subset of posterior draws (SR-004 topology)."""
    if not _PH_AVAILABLE:
        return []
    n_total = len(s_draws)
    chosen = rng.choice(n_total, size=min(n_ph_draws, n_total), replace=False)
    results: list[list[np.ndarray] | None] = []
    for d in chosen:
        s_sample = s_draws[d]
        cols: list[np.ndarray] = []
        for tr in traces:
            r_, c_ = TRACE_TO_INDEX[tr]
            z = s_sample[:, r_, c_]
            cols.extend([z.real, z.imag])
        cloud = np.column_stack(cols).astype(np.float64)
        cloud = (cloud - cloud.mean(0)) / (cloud.std(0) + 1e-12)
        cloud_sub = subsample_point_cloud(cloud, cfg.ph_max_points, cfg.subsample)
        cloud_norm = _normalize_columns(cloud_sub)
        dists = cdist(cloud_norm, cloud_norm)
        thresh = float(np.quantile(dists, cfg.ph_thresh_quantile))
        try:
            ph = _ripser_fn(cloud_norm, maxdim=cfg.ph_maxdim, thresh=thresh)
            results.append(ph["dgms"])
        except Exception:
            results.append(None)
    return results


def compute_ph_distance_posterior_hdi(
    ph_draws_by_unit: dict[str, list[list[np.ndarray] | None]],
    unit_names: list[str],
    cred_mass: float,
    dim: int = 1,
) -> pd.DataFrame | None:
    """HDI over PH bottleneck distances from posterior topology draws.

    Reports ``median [HDI_low, HDI_high]`` per unit pair.
    """
    if not _PH_AVAILABLE or not ph_draws_by_unit:
        return None
    n_draws = max((len(v) for v in ph_draws_by_unit.values()), default=0)
    if n_draws == 0:
        return None
    pair_samples: dict[tuple[str, str], list[float]] = {
        (u1, u2): [] for u1 in unit_names for u2 in unit_names
    }
    for d in range(n_draws):
        dgms_d: dict[str, list[np.ndarray] | None] = {}
        for uname in unit_names:
            draws_list = ph_draws_by_unit.get(uname, [])
            dgms_d[uname] = draws_list[d] if d < len(draws_list) else None
        for u1 in unit_names:
            for u2 in unit_names:
                if u1 == u2:
                    pair_samples[(u1, u2)].append(0.0)
                    continue
                g1 = dgms_d.get(u1)
                g2 = dgms_d.get(u2)
                if g1 is None or g2 is None:
                    continue
                if (
                    dim < len(g1) and dim < len(g2)
                    and len(g1[dim]) and len(g2[dim])
                ):
                    try:
                        pair_samples[(u1, u2)].append(
                            float(_persim_bottleneck(g1[dim], g2[dim]))
                        )
                    except Exception:
                        pass
    rows = []
    for u1 in unit_names:
        for u2 in unit_names:
            s = np.asarray(pair_samples.get((u1, u2), [0.0]), dtype=np.float64)
            if len(s) == 0:
                s = np.zeros(1)
            lo, hi = _hdi_min_width(s, cred_mass)
            rows.append({
                "unit_a": u1, "unit_b": u2,
                "median": float(np.median(s)),
                "hdi_low": lo, "hdi_high": hi,
                "hdi_string": compute_scalar_hdi_string(s, cred_mass),
            })
    return pd.DataFrame(rows)
