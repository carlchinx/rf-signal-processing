#!/usr/bin/env python3
"""
Synthetic S2P data generator for classical ML training.
Source: Unit 1 – Unit 4 real bandpass filter measurements (1–4 GHz).

Output (all written to this script's directory):
  s2p/synth_NNNN.s2p    – 2 000 synthetic Touchstone 1.1 files
  synthetic_features.csv – ML-ready feature matrix (153 features per row)
  real_features.csv      – same features for the 4 real units
  combined_features.csv  – real + synthetic merged, with sample_type column

Feature layers (all three propagated per sample)
──────────────────────────────────────────────
  RF scalar (15)   – derived directly from blended S-parameters
  TDA topology (74) – convex blend of real unit topology_inverse_features.csv
  AE latent (64)   – convex blend of real unit autoencoder_unit_descriptors.csv
  Total            : 153 features

Generation model
────────────────
Each synthetic sample is produced in three stages:

  1. Dirichlet blend (α = [2, 2, 2, 2]) of the four real S-parameter arrays,
     giving a smooth convex interpolation between the observed unit population.

  2. Physical perturbations applied in the polar / frequency domain:
       • Global S21/S12 gain shift  ΔG  ~ N(0, 0.80² dB)
       • Group-delay (phase) shift  Δτ  ~ N(0, 80 ps²)  → e^{j2πfΔτ}
       • Passband frequency stretch δs  ~ N(1, 0.003²)   → resonator shift
       • S11/S22 amplitude jitter   δr  ~ N(1, 0.02²)

  3. Additive complex Gaussian noise at the measurement floor (σ = 5×10⁻⁴).

Perturbation widths are set to 1.5× the observed inter-unit spread so the
synthetic population slightly exceeds but remains physically plausible.

Cluster label (for binary classification):
  cluster = 0  →  Units 1/2 insertion-loss style  (s21_max < −2.0 dB)
  cluster = 1  →  Units 3/4 insertion-loss style  (s21_max ≥ −2.0 dB)

Usage:
  python generate_synthetic.py            # from repo root or this directory
  python generate_synthetic.py --n 1000   # custom sample count
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids CP1252 UnicodeEncodeError for arrows/symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
DEFAULT_N_SAMPLES      = 2000
SEED                   = 2026
CLUSTER_THRESHOLD_DB   = -2.0     # s21_max boundary separating the two clusters

# Perturbation parameters (≈1.5× observed inter-unit spread)
SIGMA_GAIN_DB          = 0.80     # global S21/S12 gain shift std (dB)
SIGMA_TAU_PS           = 80.0     # group-delay shift std (ps)
SIGMA_STRETCH          = 0.003    # frequency-axis stretch std (fractional)
SIGMA_S11_AMP          = 0.02     # S11/S22 amplitude jitter std (linear)
SIGMA_NOISE            = 5e-4     # additive complex Gaussian noise floor
DIRICHLET_ALPHA        = 2.0      # Dirichlet concentration (blend weights)

# File locations
SCRIPT_DIR   = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
S2P_FILES    = [
    PIPELINE_DIR / "data" / "unit_1.s2p",
    PIPELINE_DIR / "data" / "unit_2.s2p",
    PIPELINE_DIR / "data" / "unit_3.s2p",
    PIPELINE_DIR / "data" / "unit_4.s2p",
]
UNIT_LABELS  = ["Unit 1", "Unit 2", "Unit 3", "Unit 4"]
RESULTS_DIR     = SCRIPT_DIR / "results"
S2P_OUT_DIR     = RESULTS_DIR / "s2p"
ANALYSIS_ML_DIR = PIPELINE_DIR / "outputs" / "s2p_tda_rtx4070" / "ml"  # read-only

log = logging.getLogger(__name__)

# ── Shared S2P utilities (from utils.py) ─────────────────────────────────────
from utils import parse_s2p, write_s2p  # noqa: E402
from utils import extract_rf_features as extract_features  # noqa: E402


# ── Main ──────────────────────────────────────────────────────────────────────
def main(n_samples: int = DEFAULT_N_SAMPLES) -> None:
    rng = np.random.default_rng(SEED)
    RESULTS_DIR.mkdir(exist_ok=True)   # ensure results/ exists before any writes
    S2P_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load real units ───────────────────────────────────────────────────────
    log.info("Loading real units ...")
    units_s, freqs_all = [], []
    for path in S2P_FILES:
        if not path.exists():
            log.error("Source file not found: %s", path)
            sys.exit(1)
        freq, s = parse_s2p(path)
        freqs_all.append(freq)
        units_s.append(s)

    freq_ref  = freqs_all[0]
    N_freq    = len(freq_ref)
    for i, f in enumerate(freqs_all[1:], 1):
        if len(f) != N_freq:
            sys.exit(f"ERROR: Unit {i+1} has {len(f)} points; expected {N_freq}")

    units_arr = np.stack(units_s, axis=0)   # (4, N_freq, 4_params)

    # Pre-compute polar representation for magnitude-preserving blending
    # Blending in RI domain causes destructive interference → artificially low |S21|
    units_mag   = np.abs(units_arr)                          # (4, N_freq, 4)
    units_phase = np.unwrap(np.angle(units_arr), axis=1)     # (4, N_freq, 4)

    log.info("  %d units loaded  |  %d frequency points  (%.3f\u2013%.3f GHz)",
             len(S2P_FILES), N_freq, freq_ref[0] / 1e9, freq_ref[-1] / 1e9)

    # ── Load pipeline feature matrices (TDA topology + AE latent) ────────────
    log.info("Loading pipeline feature matrices ...")
    tda_df = pd.read_csv(ANALYSIS_ML_DIR / "topology_inverse_features.csv")
    ae_df  = pd.read_csv(ANALYSIS_ML_DIR / "autoencoder_unit_descriptors.csv")

    unit_order = [f"Unit {i + 1}" for i in range(4)]
    tda_df = tda_df.set_index("unit").loc[unit_order].reset_index()
    ae_df  = ae_df.set_index("unit").loc[unit_order].reset_index()

    tda_feat_cols = [c for c in tda_df.columns if c.startswith("topology_feature_")]
    ae_feat_cols  = [c for c in ae_df.columns  if c.startswith("ae_")]

    tda_real = tda_df[tda_feat_cols].values.astype(float)   # (4, 74)
    ae_real  = ae_df[ae_feat_cols].values.astype(float)     # (4, 64)

    # Noise scale: 10 % of inter-unit std per feature (models blend approximation error)
    tda_noise_sigma = tda_real.std(axis=0) * 0.10
    ae_noise_sigma  = ae_real.std(axis=0)  * 0.10

    log.info("  TDA features : %d  |  AE descriptors : %d",
             tda_real.shape[1], ae_real.shape[1])
    log.info("  Real unit features will be sourced from analysis pipeline by compare_models.py")

    # ── Synthetic generation ──────────────────────────────────────────────────
    log.info("Generating %d synthetic samples ...", n_samples)
    synth_rows = []

    for i in range(n_samples):
        # 1. Dirichlet convex blend in polar domain (preserves filter shape)
        # Blending magnitudes + wrapped phases avoids complex destructive interference
        weights = rng.dirichlet([DIRICHLET_ALPHA] * 4)
        mag_blend   = np.einsum("u,unp->np", weights, units_mag)    # (N_freq, 4)
        phase_blend = np.einsum("u,unp->np", weights, units_phase)  # (N_freq, 4)
        s_blend = mag_blend * np.exp(1j * phase_blend)

        # Propagate TDA topology and AE latent features using the same blend weights.
        # Convex combination of real-unit pipeline features approximates what the
        # full TDA/AE pipeline would produce for the blended S-parameter signal.
        # Small Gaussian perturbation (10% of inter-unit std) models approximation error.
        tda_synth = weights @ tda_real + rng.normal(0.0, tda_noise_sigma)  # (74,)
        ae_synth  = weights @ ae_real  + rng.normal(0.0, ae_noise_sigma)   # (64,)

        # 2a. Global S21/S12 gain shift (dB → linear)
        delta_gain_db  = rng.normal(0.0, SIGMA_GAIN_DB)
        delta_gain_lin = 10 ** (delta_gain_db / 20.0)

        # 2b. Group-delay shift → linear phase slope on S21/S12
        delta_tau    = rng.normal(0.0, SIGMA_TAU_PS) * 1e-12
        phase_shift  = np.exp(1j * 2 * np.pi * freq_ref * delta_tau)

        # 2c. Frequency-axis stretch (shifts effective resonant frequency)
        delta_stretch  = rng.normal(1.0, SIGMA_STRETCH)
        freq_stretched = freq_ref * delta_stretch

        # 2d. S11/S22 amplitude jitter
        s11_scale = float(np.clip(rng.normal(1.0, SIGMA_S11_AMP), 0.80, 1.20))

        # Apply perturbations
        s_synth = s_blend.copy()
        s_synth[:, 1] *= delta_gain_lin * phase_shift   # S21
        s_synth[:, 2] *= delta_gain_lin * phase_shift   # S12
        s_synth[:, 0] *= s11_scale                       # S11
        s_synth[:, 3] *= s11_scale                       # S22

        # Resample onto the original frequency grid after stretch
        for p in range(4):
            s_synth[:, p] = (
                np.interp(freq_ref, freq_stretched, s_synth[:, p].real)
                + 1j * np.interp(freq_ref, freq_stretched, s_synth[:, p].imag)
            )

        # 3. Additive measurement noise floor
        s_synth += (
            rng.normal(0.0, SIGMA_NOISE, (N_freq, 4))
            + 1j * rng.normal(0.0, SIGMA_NOISE, (N_freq, 4))
        )

        # Write S2P file
        sample_id = f"synth_{i + 1:04d}"
        write_s2p(S2P_OUT_DIR / f"{sample_id}.s2p", freq_ref, s_synth,
                  comment=sample_id)

        # Extract features
        feats = extract_features(freq_ref, s_synth)
        feats["sample_id"]        = sample_id
        feats["sample_type"]      = "synthetic"
        feats["cluster"]          = int(feats["s21_max_db"] >= CLUSTER_THRESHOLD_DB)
        feats["blend_w1"]         = round(float(weights[0]), 4)
        feats["blend_w2"]         = round(float(weights[1]), 4)
        feats["blend_w3"]         = round(float(weights[2]), 4)
        feats["blend_w4"]         = round(float(weights[3]), 4)
        feats["dominant_unit"]    = int(np.argmax(weights) + 1)
        feats["gen_gain_db"]      = round(delta_gain_db, 4)
        feats["gen_tau_ps"]       = round(delta_tau * 1e12, 4)
        feats["gen_stretch_ppm"]  = round((delta_stretch - 1.0) * 1e6, 2)
        for j, col in enumerate(tda_feat_cols):
            feats[col] = round(float(tda_synth[j]), 8)
        for j, col in enumerate(ae_feat_cols):
            feats[col] = round(float(ae_synth[j]), 8)
        synth_rows.append(feats)

        if (i + 1) % 100 == 0:
            log.info("  %d/%d", i + 1, n_samples)

    synth_df = pd.DataFrame(synth_rows)
    synth_df.to_csv(RESULTS_DIR / "synthetic_features.csv", index=False)
    log.info("  \u2192 results/synthetic_features.csv  (%d rows)", len(synth_df))
    log.info("     (real unit rows are NOT written here; load from analysis pipeline)")

    # ── Summary ───────────────────────────────────────────────────────────────
    rf_summary_cols = [
        "s21_max_db", "s21_min_db", "bw_3db_mhz", "f_center_ghz",
        "pb_ripple_db", "gd_mean_ns", "gd_std_ps", "gd_peak_ns",
        "s11_pb_mean_db", "s22_pb_mean_db", "reciprocity_max", "passivity_margin",
    ]
    n_tda = len(tda_feat_cols)
    n_ae  = len(ae_feat_cols)
    log.info("\n── Feature counts ──────────────────────────────────────────────────")
    log.info("  RF scalar : %d  |  TDA topology : %d  |  AE latent : %d",
             len(rf_summary_cols) + 3, n_tda, n_ae)
    log.info("  Total feature columns per sample : %d",
             len(rf_summary_cols) + 3 + n_tda + n_ae)
    log.info("\n── Synthetic RF feature statistics ─────────────────────────────────")
    log.info("\n%s", synth_df[rf_summary_cols].describe().round(4).to_string())

    counts = synth_df["cluster"].value_counts().sort_index()
    log.info("\nCluster split  (threshold %.1f dB on s21_max):", CLUSTER_THRESHOLD_DB)
    labels_map = {0: "Units 1/2 style \u2014 higher insertion loss",
                  1: "Units 3/4 style \u2014 lower insertion loss"}
    for cl, cnt in counts.items():
        log.info("  Cluster %d  (%s) : %d samples",
                 int(cl), labels_map.get(int(cl), ""), int(cnt))

    log.info("\nS2P files  \u2192 %s/  (%d files)", S2P_OUT_DIR, n_samples)
    log.info("Features   \u2192 %s/", RESULTS_DIR)

    _generate_plots(synth_df, S2P_FILES, UNIT_LABELS, RESULTS_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_plots(synth_df: pd.DataFrame, s2p_files, unit_labels,
                    results_dir: Path) -> None:
    """Save synthetic-data exploration plots to results/plots/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    plot_dir = results_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    RF_PLOT_COLS   = [
        "s21_max_db", "s21_min_db", "pb_ripple_db", "bw_3db_mhz",
        "f_center_ghz", "gd_mean_ns", "s11_pb_mean_db", "passivity_margin",
    ]
    CLU_COLORS  = {0: "#4C72B0", 1: "#DD8452"}
    UNIT_COLORS = ["#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Load real-unit RF features for overlay
    real_rows = []
    for path, label in zip(s2p_files, unit_labels):
        if Path(path).exists():
            f, s   = parse_s2p(path)
            feats  = extract_features(f, s)
            feats["unit"]    = label
            feats["cluster"] = int(feats["s21_max_db"] >= CLUSTER_THRESHOLD_DB)
            real_rows.append(feats)
    real_df = pd.DataFrame(real_rows) if real_rows else pd.DataFrame()

    log.info("Plots")

    # ── 1. RF feature distributions ───────────────────────────────────────────
    cols_avail = [c for c in RF_PLOT_COLS if c in synth_df.columns]
    ncols_h, nrows_h = 4, (len(cols_avail) + 3) // 4
    fig, axes = plt.subplots(nrows_h, ncols_h,
                              figsize=(ncols_h * 3.2, nrows_h * 2.8))
    axes = axes.flatten()
    for i, col in enumerate(cols_avail):
        ax = axes[i]
        for cl, color in CLU_COLORS.items():
            vals = synth_df.loc[synth_df["cluster"] == cl, col].dropna()
            ax.hist(vals, bins=40, color=color, alpha=0.55,
                    label=f"Cluster {cl}", density=True)
        if not real_df.empty and col in real_df.columns:
            for j, (_, ru) in enumerate(real_df.iterrows()):
                ax.axvline(ru[col], color=UNIT_COLORS[j % 4], lw=1.5, ls="--",
                           label=ru["unit"] if i == 0 else None)
        ax.set_title(col.replace("_", " "), fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3, linewidth=0.5)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        if i == 0:
            ax.legend(fontsize=6, ncol=2)
    for ax in axes[len(cols_avail):]:
        ax.set_visible(False)
    fig.suptitle(
        "Synthetic Data \u2014 RF Feature Distributions\n"
        "(dashed lines = real unit values; shading = synthetic clusters)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "synth_rf_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 plots/synth_rf_distributions.png")

    # ── 2. PCA of RF features ──────────────────────────────────────────────────
    pca_cols = [c for c in RF_PLOT_COLS if c in synth_df.columns]
    if len(pca_cols) >= 2:
        X_synth = synth_df[pca_cols].values.astype(float)
        X_real  = real_df[pca_cols].values.astype(float) if not real_df.empty else None
        X_all   = X_synth if X_real is None else np.vstack([X_synth, X_real])
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_all)
        pca      = PCA(n_components=2, random_state=SEED)
        X_pca    = pca.fit_transform(X_scaled)
        X_s_pca  = X_pca[:len(X_synth)]
        X_r_pca  = X_pca[len(X_synth):] if X_real is not None else None
        var_exp  = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(7.0, 5.5))
        clusters = synth_df["cluster"].values
        for cl, color in CLU_COLORS.items():
            mask = clusters == cl
            ax.scatter(X_s_pca[mask, 0], X_s_pca[mask, 1],
                       c=color, s=4, alpha=0.22,
                       label=f"Synthetic cluster {cl}", rasterized=True)
        if X_r_pca is not None:
            for j, (_, ru) in enumerate(real_df.iterrows()):
                ax.scatter(X_r_pca[j, 0], X_r_pca[j, 1],
                           c=UNIT_COLORS[j % 4], s=160, marker="*",
                           edgecolors="black", linewidths=0.8, zorder=5,
                           label=ru["unit"])
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}% var.)", fontsize=9)
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}% var.)", fontsize=9)
        ax.set_title(
            "PCA of RF Features \u2014 Synthetic Population\n"
            "(\u2605 = real units; shading = synthetic clusters)",
            fontsize=10, fontweight="bold",
        )
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(alpha=0.3, linewidth=0.7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        fig.savefig(plot_dir / "synth_feature_pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  \u2192 plots/synth_feature_pca.png")

    # ── 3. Blend-weight distribution (dominant unit composition) ──────────────
    if "blend_w1" in synth_df.columns:
        w_cols = ["blend_w1", "blend_w2", "blend_w3", "blend_w4"]
        w_cols = [c for c in w_cols if c in synth_df.columns]
        dom_labels = [f"Unit {i+1}" for i in range(len(w_cols))]
        fig, axes2 = plt.subplots(1, len(w_cols),
                                   figsize=(len(w_cols) * 2.8, 3.5), sharey=False)
        if len(w_cols) == 1:
            axes2 = [axes2]
        for ax, wc, lbl, color in zip(axes2, w_cols, dom_labels, UNIT_COLORS):
            ax.hist(synth_df[wc].values, bins=40, color=color, alpha=0.75)
            ax.set_title(lbl, fontsize=9, fontweight="bold")
            ax.set_xlabel("Blend weight", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.3, linewidth=0.5)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)
        axes2[0].set_ylabel("Count", fontsize=8)
        fig.suptitle(
            "Dirichlet Blend Weight Distributions per Unit\n"
            "(Dirichlet \u03b1=2; higher weight \u21d2 sample closer to that unit)",
            fontsize=9, fontweight="bold",
        )
        plt.tight_layout()
        fig.savefig(plot_dir / "synth_blend_weights.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("  \u2192 plots/synth_blend_weights.png")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Generate synthetic S2P data for classical ML training."
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_SAMPLES,
        help=f"Number of synthetic samples to generate (default: {DEFAULT_N_SAMPLES})"
    )
    args = parser.parse_args()
    main(n_samples=args.n)
