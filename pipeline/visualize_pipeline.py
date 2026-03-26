#!/usr/bin/env python3
"""
Post-doctoral visualization suite for the RF S2P analysis pipeline.
Generates 13 publication-quality figures explaining all intermediate outputs.

Usage:
    cd pipeline
    python visualize_pipeline.py
"""
import ast
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs" / "s2p_tda_rtx4070"
SYNTH   = ROOT / "synthetic_data" / "results"
FIG_OUT = OUTPUTS / "plots" / "report"

UNITS   = [1, 2, 3, 4]
SERIALS = ["unit_1", "unit_2", "unit_3", "unit_4"]
_SER        = {u: s for u, s in zip(UNITS, SERIALS)}        # file-path lookup only
SERIAL_MAP  = {s: f"U{u}" for s, u in zip(SERIALS, UNITS)}  # TDA matrix relabelling only

# Wong (2011) colorblind-safe palette
UNIT_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7"}
UNIT_LABEL  = {u: f"Unit {u}" for u in UNITS}               # NO serials in any figure
LAYER_COLORS = {"rf": "#1f77b4", "tda": "#ff7f0e", "ae": "#2ca02c", "all": "#d62728"}
LAYER_LABELS = {"rf": "RF scalar", "tda": "TDA topo.", "ae": "AE latent", "all": "All fused"}

# 16:9 canonical figure sizes
W9  = (16,  9.000)
W10 = (18, 10.125)
W11 = (20, 11.250)

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       120,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "axes.grid":        True,
    "grid.alpha":       0.28,
    "grid.linewidth":   0.5,
    "mathtext.fontset": "stix",
    "lines.linewidth":  1.4,
})
SAVEKW = dict(dpi=300, bbox_inches="tight", facecolor="white")


# ── Utilities ─────────────────────────────────────────────────────────────────
def save(fig, name: str) -> Path:
    path = FIG_OUT / name
    fig.savefig(path, **SAVEKW)
    plt.close(fig)
    print(f"  [OK] {path.name}")
    return path


def parse_hdi(s: str):
    """Parse 'val [low, high]' string → (median, low, high)."""
    m = re.search(r"(-?[\d.eE+\-]+)\s*\[(-?[\d.eE+\-]+),\s*(-?[\d.eE+\-]+)\]", str(s))
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    try:
        v = float(s)
        return v, v, v
    except Exception:
        return np.nan, np.nan, np.nan


def fghz(df) -> np.ndarray:
    return df.freq_hz.values * 1e-9


def pb_mask(df, thresh=-3.0) -> np.ndarray:
    return df.s21_db.values > thresh


def despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def math_box(ax, text, loc="upper right", fs=7.5):
    props = dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                 alpha=0.75, edgecolor="gray", linewidth=0.6)
    locs = {
        "upper right": (0.97, 0.97, "right", "top"),
        "upper left":  (0.03, 0.97, "left",  "top"),
        "lower right": (0.97, 0.03, "right", "bottom"),
        "lower left":  (0.03, 0.03, "left",  "bottom"),
    }
    x, y, ha, va = locs.get(loc, (0.97, 0.97, "right", "top"))
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fs,
            ha=ha, va=va, bbox=props, multialignment="left")


# ── Data loading ──────────────────────────────────────────────────────────────
def load_all() -> dict:
    print("Loading pipeline outputs …")
    m   = {u: pd.read_csv(OUTPUTS / "metrics"     / f"unit_{u}_frequency_metrics.csv") for u in UNITS}
    bcb = {u: pd.read_csv(OUTPUTS / "bayes"       / f"unit_{u}_credible_bands.csv")    for u in UNITS}
    bhi = {u: pd.read_csv(OUTPUTS / "bayes"       / f"unit_{u}_hdi_scalar_summary.csv") for u in UNITS}
    vfs = {u: pd.read_csv(OUTPUTS / "vector_fit"  / f"unit_{u}_vf_summary.csv")        for u in UNITS}
    vfp = {}
    for u in UNITS:
        try:
            vfp[u] = np.load(OUTPUTS / "vector_fit" / f"unit_{u}_vf_params.npz",
                              allow_pickle=True)
        except Exception:
            vfp[u] = None
    gng = {u: pd.read_csv(OUTPUTS / "tda" / f"unit_{u}_complex_gng_summary.csv")        for u in UNITS}
    gtm = {u: pd.read_csv(OUTPUTS / "tda" / f"unit_{u}_complex_gng_transition_matrix.csv",
                           index_col=0) for u in UNITS}
    vpt = {u: pd.read_csv(OUTPUTS / "tda" / f"unit_{u}_complex_voronoi_points.csv")     for u in UNITS}
    td  = {u: pd.read_csv(OUTPUTS / "time_domain" / f"unit_{u}_time_domain.csv")        for u in UNITS}
    aed = pd.read_csv(OUTPUTS / "ml" / "autoencoder_unit_descriptors.csv")
    aeh = pd.read_csv(OUTPUTS / "ml" / "autoencoder_training_history.csv")
    ivh = pd.read_csv(OUTPUTS / "ml" / "inverse_training_history.csv")
    phd = pd.read_csv(OUTPUTS / "ph" / "ph_bottleneck_distance_dim1.csv", index_col=0)
    vcx = pd.read_csv(OUTPUTS / "tda" / "voronoi_distance_complex.csv",   index_col=0)
    vsh = pd.read_csv(OUTPUTS / "tda" / "voronoi_distance_shift.csv",     index_col=0)
    gcx = pd.read_csv(OUTPUTS / "tda" / "gng_distance_complex.csv",       index_col=0)
    gsh = pd.read_csv(OUTPUTS / "tda" / "gng_distance_shift.csv",         index_col=0)
    sf  = pd.read_csv(SYNTH / "synthetic_features.csv")
    clf = pd.read_csv(SYNTH / "results_classification.csv")
    reg = pd.read_csv(SYNTH / "results_regression.csv")
    lrc = pd.read_csv(SYNTH / "results_lro_classification.csv")
    lrr = pd.read_csv(SYNTH / "results_lro_regression.csv")
    print("  [OK] all data loaded")
    return dict(m=m, bcb=bcb, bhi=bhi, vfs=vfs, vfp=vfp, gng=gng, gtm=gtm, vpt=vpt,
                td=td, aed=aed, aeh=aeh, ivh=ivh, phd=phd, vcx=vcx, vsh=vsh,
                gcx=gcx, gsh=gsh, sf=sf, clf=clf, reg=reg, lrc=lrc, lrr=lrr)


# ── VF reconstruction ─────────────────────────────────────────────────────────
def vf_reconstruct_s21(npz, freq_hz: np.ndarray):
    """
    Rational model: H(s) = D + E·s + Σ_k [r_k/(s-p_k) + conj(r_k)/(s-conj(p_k))]
    where s = j·2π·f.  S21 -> residue row index 1.
    """
    if npz is None:
        return None
    keys = list(npz.keys())

    def get_key(*candidates):
        for c in candidates:
            if c in keys:
                return npz[c]
        return None

    poles    = get_key("poles")
    residues = get_key("residues")
    const    = get_key("constant_coefficients", "constants", "d")
    prop     = get_key("proportional_coefficients", "proportionals", "e")

    if poles is None or residues is None:
        return None

    omega = 2.0 * np.pi * freq_hz
    s     = 1j * omega

    d_s21 = float(np.real(const[1])) if (const is not None and len(const) > 1) else 0.0
    e_s21 = float(np.real(prop[1]))  if (prop  is not None and len(prop)  > 1) else 0.0
    res21 = residues[1] if residues.ndim > 1 else residues

    H = np.full(len(s), d_s21 + e_s21 * s, dtype=complex)
    for pole, res in zip(poles, res21):
        H += res / (s - pole)
        if abs(pole.imag) > 1e3:
            H += np.conj(res) / (s - np.conj(pole))
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Fig 01 · S-Parameter Spectral Survey
# ─────────────────────────────────────────────────────────────────────────────
def fig01(m):
    print("fig01 …")
    fig = plt.figure(figsize=W11)
    gs  = gridspec.GridSpec(4, 4, hspace=0.46, wspace=0.30, figure=fig)

    specs = [
        ("S11", lambda df: 20*np.log10(np.clip(df.s11_mag.values, 1e-10, None)),
         r"$|S_{11}|$ (dB)", [-40, 2],  -10),
        ("S21", lambda df: df.s21_db.values,
         r"$|S_{21}|$ (dB)", [-58, 6],  -3),
        ("S12", lambda df: 20*np.log10(np.clip(df.s12_mag.values, 1e-10, None)),
         r"$|S_{12}|$ (dB)", [-58, 6],  -3),
        ("S22", lambda df: 20*np.log10(np.clip(df.s22_mag.values, 1e-10, None)),
         r"$|S_{22}|$ (dB)", [-40, 2], -10),
    ]

    for row, (name, extractor, ylabel, ylim, ref_db) in enumerate(specs):
        for col, u in enumerate(UNITS):
            ax = fig.add_subplot(gs[row, col])
            despine(ax)
            df = m[u]
            f  = fghz(df)
            y  = extractor(df)
            ax.plot(f, y, color=UNIT_COLORS[u], lw=1.3)
            ax.axhline(ref_db, color="gray", lw=0.8, ls="--", alpha=0.6)
            ax.set_xlim(1.0, 4.0)
            ax.set_ylim(ylim)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
            if row == 0:
                ax.set_title(UNIT_LABEL[u], fontsize=8.5, fontweight="bold")
            if col == 0:
                ax.set_ylabel(ylabel)
            if row < 3:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("Frequency (GHz)")
            # Passband shading on S21/S12 rows
            if row in (1, 2):
                pb = pb_mask(df)
                if pb.any():
                    fp = f[pb]
                    ax.axvspan(fp.min(), fp.max(), color="gold", alpha=0.09, zorder=0)
            # Annotate S21 peak
            if row == 1:
                idx = int(np.argmax(y))
                ax.annotate(
                    f"$|S_{{21}}|_{{\\rm max}}$\n$={y[idx]:.2f}$ dB",
                    xy=(f[idx], y[idx]),
                    xytext=(f[idx] + 0.25, y[idx] - 9),
                    fontsize=6, color=UNIT_COLORS[u],
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=UNIT_COLORS[u]),
                    ha="left",
                )
            ax.annotate(f"{ref_db} dB", xy=(3.95, ref_db),
                        fontsize=6, color="gray", ha="right", va="bottom")

    fig.suptitle(
        r"S-Parameter Spectral Survey — All 4 Units, 1–4 GHz  "
        r"($S_{11},\,S_{21},\,S_{12},\,S_{22}$)",
        fontsize=12, fontweight="bold", y=1.002,
    )
    save(fig, "fig01_sparam_spectral_survey.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 02 · Passband S21 with Bayesian 95 % HDI + group delay twin axis
# ─────────────────────────────────────────────────────────────────────────────
def fig02(m, bcb):
    print("fig02 …")
    fig, axes = plt.subplots(4, 1, figsize=W10, sharex=True)
    fig.subplots_adjust(hspace=0.35)

    for i, u in enumerate(UNITS):
        ax  = axes[i]
        despine(ax)
        df  = m[u];  bay = bcb[u]
        f   = fghz(df)

        ax.fill_between(f, bay.s21_db_hdi_low, bay.s21_db_hdi_high,
                        color=UNIT_COLORS[u], alpha=0.22, label="95 % HDI")
        ax.plot(f, bay.s21_db_median,  color=UNIT_COLORS[u], lw=1.8, label="Posterior median")
        ax.plot(f, df.s21_db,          color="black", lw=0.9, ls="--", alpha=0.65, label="Observed")
        ax.axhline(-3,  color="green", lw=0.8, ls=":", alpha=0.7)
        ax.axhline(-40, color="red",   lw=0.7, ls=":", alpha=0.5)

        pb = pb_mask(df)
        if pb.any():
            fp = f[pb]
            ax.axvspan(fp.min(), fp.max(), color="gold", alpha=0.09, zorder=0)
            bw = (fp.max() - fp.min()) * 1e3
            fc = fp.mean()
            math_box(ax,
                     rf"$B_{{-3\,\rm dB}}\approx{bw:.0f}\ \rm MHz$"
                     + "\n" + rf"$f_c\approx{fc:.3f}\ \rm GHz$",
                     loc="upper left")

        ax.set_ylim(-62, 8)
        ax.set_ylabel(r"$|S_{21}|$ (dB)")
        ax.set_title(UNIT_LABEL[u], fontsize=9, fontweight="bold")

        # Twin axis: group delay (passband + transition region only)
        ax2 = ax.twinx()
        mask_gd  = df.s21_db.values > -20
        gd_ns    = df.group_delay_s21_s.values * 1e9
        gd_med   = bay.gd_ns_median.values
        gd_lo    = np.clip(bay.gd_ns_hdi_low.values,  0, 6)
        gd_hi    = np.clip(bay.gd_ns_hdi_high.values, 0, 6)
        ax2.fill_between(f[mask_gd], gd_lo[mask_gd], gd_hi[mask_gd],
                         color="sienna", alpha=0.13)
        ax2.plot(f[mask_gd], gd_ns[mask_gd],              color="sienna", lw=1.1, ls="-.")
        ax2.plot(f[mask_gd], np.clip(gd_med[mask_gd], 0, 6), color="sienna",
                 lw=0.8, ls=":", alpha=0.7)
        ax2.set_ylabel(r"$\tau_g$ (ns)", color="sienna", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="sienna", labelsize=7)
        ax2.set_ylim(0, 3.8)
        ax2.spines["top"].set_visible(False)

        if i == 0:
            l1, lb1 = ax.get_legend_handles_labels()
            ax.legend(l1, lb1, loc="lower center", ncol=3, fontsize=7.5)

    axes[-1].set_xlabel("Frequency (GHz)")
    axes[-1].set_xlim(1.0, 4.0)
    fig.suptitle(
        r"$|S_{21}|$ with Bayesian 95 % HDI Credible Band "
        r"and Group Delay $\tau_g = -\frac{d\phi}{d\omega}$",
        fontsize=11, fontweight="bold", y=1.002,
    )
    save(fig, "fig02_s21_bayesian_hdi_groupdelay.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 03 · Group Delay Dispersion Analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig03(m, bcb):
    print("fig03 …")
    from scipy.signal import savgol_filter as sgf

    fig = plt.figure(figsize=W9)
    gs  = gridspec.GridSpec(2, 2, hspace=0.46, wspace=0.38, figure=fig)

    # (0,0) Full-span GD overlay
    ax00 = fig.add_subplot(gs[0, 0]);  despine(ax00)
    for u in UNITS:
        df = m[u]
        ax00.plot(fghz(df), df.group_delay_s21_s.values * 1e9,
                  color=UNIT_COLORS[u], lw=1.2, label=f"Unit {u}", alpha=0.88)
    ax00.set_xlim(1.0, 4.0);  ax00.set_ylim(-0.2, 2.8)
    ax00.set_xlabel("Frequency (GHz)");  ax00.set_ylabel(r"$\tau_g$ (ns)")
    ax00.set_title("Group Delay — Full Span (1–4 GHz)")
    ax00.legend(ncol=2, fontsize=8)
    math_box(ax00, r"$\tau_g(f)=-\frac{1}{2\pi}\frac{d\phi}{df}$")

    # (0,1) Passband GD with Bayesian HDI
    ax01 = fig.add_subplot(gs[0, 1]);  despine(ax01)
    for u in UNITS:
        df = m[u];  bay = bcb[u];  f = fghz(df)
        pb = pb_mask(df, -3.0)
        if not pb.any():
            continue
        fp  = f[pb]
        gd  = df.group_delay_s21_s.values[pb] * 1e9
        ax01.fill_between(fp,
                          np.clip(bay.gd_ns_hdi_low.values[pb],  0, 5),
                          np.clip(bay.gd_ns_hdi_high.values[pb], 0, 5),
                          color=UNIT_COLORS[u], alpha=0.22)
        ax01.plot(fp, bay.gd_ns_median.values[pb], color=UNIT_COLORS[u], lw=1.4,
                  label=f"Unit {u}  Δτ={gd.max()-gd.min():.3f} ns")
    ax01.set_xlabel("Frequency (GHz)");  ax01.set_ylabel(r"$\tau_g$ (ns)")
    ax01.set_title("Passband GD with 95 % HDI")
    ax01.legend(fontsize=7.5)

    # (1,0) GDD = -d²φ/dω²
    ax10 = fig.add_subplot(gs[1, 0]);  despine(ax10)
    for u in UNITS:
        df    = m[u]
        omega = 2 * np.pi * df.freq_hz.values
        phi   = df.s21_phase_unwrap_rad.values
        gdd   = -np.gradient(np.gradient(phi, omega), omega) * 1e18  # -> fs^2
        gdd_s = sgf(gdd, window_length=51, polyorder=3)
        ax10.plot(fghz(df), np.clip(gdd_s, -3e4, 3e4),
                  color=UNIT_COLORS[u], lw=1.2, label=f"Unit {u}", alpha=0.88)
    ax10.axhline(0, color="black", lw=0.7, ls="--")
    ax10.set_xlim(1.0, 4.0);  ax10.set_xlabel("Frequency (GHz)")
    ax10.set_ylabel(r"GDD (fs$^2$)")
    ax10.set_title(r"Group Delay Dispersion $\beta_2=-\frac{d^2\phi}{d\omega^2}$")
    ax10.legend(ncol=2, fontsize=8)
    math_box(ax10, r"$\beta_2 = -\frac{d\tau_g}{d\omega}$" + "\n(fs² ≡ ps/GHz)", loc="upper right")

    # (1,1) Mean |ΔGD| inter-unit heatmap
    ax11 = fig.add_subplot(gs[1, 1]);  despine(ax11)
    gd_arr = {u: m[u].group_delay_s21_s.values * 1e9 for u in UNITS}
    delta  = np.zeros((4, 4))
    for i, ui in enumerate(UNITS):
        for j, uj in enumerate(UNITS):
            delta[i, j] = np.mean(np.abs(gd_arr[ui] - gd_arr[uj]))
    im = ax11.imshow(delta, cmap="YlOrRd", vmin=0, aspect="equal")
    tks = [f"U{u}" for u in UNITS]
    ax11.set_xticks(range(4));  ax11.set_xticklabels(tks)
    ax11.set_yticks(range(4));  ax11.set_yticklabels(tks)
    for ii in range(4):
        for jj in range(4):
            c = "white" if delta[ii, jj] > 0.55 * delta.max() else "black"
            ax11.text(jj, ii, f"{delta[ii,jj]:.3f}", ha="center", va="center",
                      fontsize=9, color=c)
    plt.colorbar(im, ax=ax11, shrink=0.85, label=r"$\langle|\Delta\tau_g|\rangle$ (ns)")
    ax11.set_title(r"Mean GD Deviation $\langle|\Delta\tau_g|\rangle$ (ns)")

    fig.suptitle("Group Delay Dispersion Analysis", fontsize=12, fontweight="bold", y=1.01)
    save(fig, "fig03_group_delay_dispersion.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 04 · Passivity & Reciprocity Audit
# ─────────────────────────────────────────────────────────────────────────────
def fig04(m):
    print("fig04 …")
    fig, axes = plt.subplots(3, 1, figsize=W10, sharex=True)
    fig.subplots_adjust(hspace=0.42)

    ax0 = axes[0];  despine(ax0)
    for u in UNITS:
        df = m[u]
        ax0.plot(fghz(df), df.passivity_sigma_max, color=UNIT_COLORS[u],
                 lw=1.1, label=f"Unit {u}", alpha=0.88)
    ax0.axhline(1.0, color="red", lw=1.2, ls="--", alpha=0.85,
                label=r"$\sigma_{\max}=1$ (boundary)")
    ax0.set_ylim(0.975, 1.010)
    ax0.set_ylabel(r"$\sigma_{\max}(\mathbf{S})$")
    ax0.set_title(r"Passivity: $\sigma_{\max}(\mathbf{S}(j\omega))\leq 1\;\forall\omega$")
    ax0.legend(ncol=3, fontsize=8)
    math_box(ax0,
             r"$\mathbf{I}-\mathbf{S}^H\mathbf{S}\succeq 0$" + "\n"
             r"$\Leftrightarrow\;\sigma_{\max}(\mathbf{S})\leq 1$",
             loc="lower right")

    ax1 = axes[1];  despine(ax1)
    for u in UNITS:
        df = m[u];  pm = df.passivity_margin.values
        ax1.plot(fghz(df), pm, color=UNIT_COLORS[u], lw=1.1, label=f"Unit {u}", alpha=0.88)
        ax1.fill_between(fghz(df), 0, pm, color=UNIT_COLORS[u], alpha=0.10)
    ax1.axhline(0, color="red", lw=0.9, ls="--", alpha=0.7)
    ax1.set_ylim(-0.002, 0.030)
    ax1.set_ylabel(r"$\epsilon_P = 1-\sigma_{\max}(\mathbf{S})$")
    ax1.set_title(r"Passivity Margin $\epsilon_P$")
    ax1.legend(ncol=3, fontsize=8)

    ax2 = axes[2];  despine(ax2)
    for u in UNITS:
        df  = m[u]
        rec = np.clip(df.reciprocity_abs_s21_minus_s12.values, 1e-7, None)
        ax2.semilogy(fghz(df), rec, color=UNIT_COLORS[u], lw=1.1, label=f"Unit {u}", alpha=0.88)
    ax2.axhline(1e-3, color="orange", lw=0.8, ls=":", alpha=0.8, label=r"$10^{-3}$")
    ax2.axhline(1e-4, color="green",  lw=0.8, ls=":", alpha=0.8, label=r"$10^{-4}$")
    ax2.set_ylabel(r"$|S_{21}-S_{12}|$")
    ax2.set_title(r"Reciprocity Residual $|S_{21}(j\omega)-S_{12}(j\omega)|$ (log scale)")
    ax2.legend(ncol=4, fontsize=8)
    math_box(ax2, r"Reciprocity: $S_{21}=S_{12}$" + "\n(lossless passive 2-port)",
             loc="upper right")

    axes[-1].set_xlabel("Frequency (GHz)")
    axes[-1].set_xlim(1.0, 4.0)
    fig.suptitle(r"Passivity \& Reciprocity Diagnostics — IEEE Audit",
                 fontsize=12, fontweight="bold", y=1.01)
    save(fig, "fig04_passivity_reciprocity_audit.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 05 · Vector Fitting Pole-Zero Maps
# ─────────────────────────────────────────────────────────────────────────────
def fig05(vfp, vfs):
    print("fig05 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.42, figure=fig)

    for idx, u in enumerate(UNITS):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col]);  despine(ax)
        ax.set_facecolor("#f8f8ff")
        ax.axvline(0, color="red", lw=1.3, ls="--", alpha=0.85, label="Stability boundary")
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
        ax.axvspan(-200, 0, color="lightgreen", alpha=0.05, zorder=0)

        row_s = vfs[u].iloc[0]
        title = (f"Unit {u}  |  Order: {row_s.get('model_order','?')}  |  "
                 f"RMS: {row_s.get('rms_error', float('nan')):.4f}  |  "
                 f"Passive after enforcement: {row_s.get('passive_after','?')}")

        npz = vfp[u]
        if npz is None:
            ax.text(0.5, 0.5, "VF params unavailable",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=8);  continue

        keys   = list(npz.keys())
        poles  = next((npz[k] for k in ["poles"] if k in keys), None)
        resids = next((npz[k] for k in ["residues"] if k in keys), None)

        if poles is None:
            ax.text(0.5, 0.5, f"Keys: {keys}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=7)
            ax.set_title(title, fontsize=8);  continue

        p_re = poles.real / (2 * np.pi * 1e9)
        p_im = poles.imag / (2 * np.pi * 1e9)
        is_r = np.abs(p_im) < 0.01
        is_c = ~is_r

        if is_r.any():
            ax.scatter(p_re[is_r], p_im[is_r], marker="s", s=90,
                       color=UNIT_COLORS[u], edgecolors="black", lw=0.8, zorder=6,
                       label=f"Real poles (n={is_r.sum()})")
        if is_c.any():
            ax.scatter(p_re[is_c], p_im[is_c], marker="x", s=110,
                       color=UNIT_COLORS[u], linewidths=2.2, zorder=6,
                       label=f"Complex poles (n={is_c.sum()})")
            if resids is not None and resids.ndim > 1:
                res21    = resids[1]
                res_mag  = np.abs(res21[is_c])
                if res_mag.max() > 0:
                    rn = res_mag / res_mag.max()
                    for pr, pi, r in zip(p_re[is_c], p_im[is_c], rn):
                        if r > 0.05:
                            ax.annotate("", xy=(0, 0), xytext=(pr, pi),
                                        arrowprops=dict(arrowstyle="-|>", color="gray",
                                                        lw=0.5 + r * 1.2, alpha=0.45))

        x_lo = max(p_re.min() * 1.3, -200)
        ax.set_xlim(x_lo, 5);  ax.set_ylim(-45, 45)
        ax.set_xlabel(r"$\sigma/(2\pi)$ (GHz) — damping rate", fontsize=8)
        ax.set_ylabel(r"$j\omega/(2\pi)$ (GHz) — natural freq.", fontsize=8)
        ax.set_title(title, fontsize=8)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=7)

        passive = str(row_s.get("passive_after", "")).strip().lower()
        sym  = "[Passive]" if passive in ("true", "1") else "[Non-passive]"
        col_ = "green"     if passive in ("true", "1") else "red"
        ax.text(0.03, 0.92, sym, transform=ax.transAxes, fontsize=8,
                color=col_, fontweight="bold",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor=col_))

    fig.suptitle(
        r"Vector Fitting Pole-Zero Maps — $s$-Plane  "
        r"$H(s)=\sum_k r_k/(s-p_k)+d$, arrows $\propto|r_k|$",
        fontsize=11, fontweight="bold", y=1.01)
    save(fig, "fig05_vf_pole_zero_map.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 06 · VF Rational Model Quality — Fitted vs Measured
# ─────────────────────────────────────────────────────────────────────────────
def fig06(m, vfp, vfs):
    print("fig06 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(2, 4, hspace=0.12, wspace=0.30,
                            height_ratios=[3, 1], figure=fig)

    for col, u in enumerate(UNITS):
        at = fig.add_subplot(gs[0, col])
        ab = fig.add_subplot(gs[1, col])
        for ax in (at, ab):
            despine(ax)
        df   = m[u];  f = fghz(df)
        rms  = float(vfs[u].iloc[0].get("rms_error", np.nan))
        pass_ = str(vfs[u].iloc[0].get("passive_after", "")).strip().lower()

        at.plot(f, df.s21_db, color="black", lw=0.9, ls="-", alpha=0.85, label="Measured")
        H_db = None
        npz  = vfp[u]
        if npz is not None:
            H = vf_reconstruct_s21(npz, df.freq_hz.values)
            if H is not None:
                H_db = 20 * np.log10(np.abs(H).clip(1e-10))
                order = vfs[u].iloc[0].get("model_order", "?")
                at.plot(f, H_db, color=UNIT_COLORS[u], lw=1.5, ls="--",
                        label=f"VF order {order}")

        pb = pb_mask(df)
        if pb.any():
            fp = f[pb]
            at.axvspan(fp.min(), fp.max(), color="gold", alpha=0.08, zorder=0)
            ab.axvspan(fp.min(), fp.max(), color="gold", alpha=0.08, zorder=0)

        at.text(0.03, 0.06, rf"RMS$={rms:.4f}$", transform=at.transAxes,
                fontsize=7.5, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        at.set_ylim(-60, 8);  at.set_xlim(1.0, 4.0)
        at.tick_params(labelbottom=False)
        at.set_title(f"Unit {u}", fontsize=9, fontweight="bold")
        if col == 0:
            at.set_ylabel(r"$|S_{21}|$ (dB)")
            at.legend(loc="lower right", fontsize=7.5)

        # Residual bar chart
        if H_db is not None:
            res = H_db - df.s21_db.values
            step = f[1] - f[0]
            ab.bar(f, res, width=step * 0.9, color=UNIT_COLORS[u], alpha=0.7)
        ab.axhline(0,  color="black", lw=0.7)
        ab.axhline( 2, color="gray",  lw=0.5, ls=":")
        ab.axhline(-2, color="gray",  lw=0.5, ls=":")
        ab.set_ylim(-6, 6);  ab.set_xlim(1.0, 4.0)
        ab.set_xlabel("Frequency (GHz)", fontsize=8)
        if col == 0:
            ab.set_ylabel(r"$\Delta|S_{21}|$ (dB)", fontsize=8)

        dot_c = "green" if pass_ in ("true", "1") else "red"
        at.plot([], [], "o", color=dot_c, ms=8, label=f"Passive: {pass_}")

    fig.suptitle(
        r"VF Rational Model $|S_{21}|$: Fitted vs Measured (residual panel below)",
        fontsize=11, fontweight="bold", y=1.01)
    save(fig, "fig06_vf_model_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 07 · TDA Distance Matrix Atlas
# ─────────────────────────────────────────────────────────────────────────────
def fig07(vcx, vsh, gcx, gsh):
    print("fig07 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(2, 2, hspace=0.55, wspace=0.45, figure=fig)

    panels = [
        (gs[0, 0], vcx, "Blues",   "Voronoi Distance — Complex S-plane",
         r"$d_{\mathcal{V}}(i,j)=\|\bar{\mathbf{c}}_i-\bar{\mathbf{c}}_j\|_F$"),
        (gs[0, 1], vsh, "Oranges", "Voronoi Distance — Shift-register Embedding",
         r"$d_{\mathcal{V}}^{\rm shift}(i,j)$"),
        (gs[1, 0], gcx, "Greens",  "GNG Graph Distance — Complex Manifold",
         r"$d_{\rm GNG}(i,j)=d_{\rm graph}(G_i,G_j)$"),
        (gs[1, 1], gsh, "Purples", "GNG Graph Distance — Shift Embedding",
         r"$d_{\rm GNG}^{\rm shift}(i,j)$"),
    ]

    def relabel(df):
        ren = {c: SERIAL_MAP.get(c, c) for c in list(df.index) + list(df.columns)}
        return df.rename(index=ren, columns=ren)

    for ss, dist_df, cmap, title, eqlabel in panels:
        ax  = fig.add_subplot(ss);  despine(ax)
        df2 = relabel(dist_df)
        mat = df2.values.astype(float)
        vmin = mat[mat > 0].min() if (mat > 0).any() else 1e-6
        vmax = mat.max()
        use_log = vmax / max(vmin, 1e-10) > 100
        norm = mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if use_log else \
               mcolors.Normalize(vmin=0, vmax=vmax)
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="equal")
        tks = list(df2.index)
        ax.set_xticks(range(4));  ax.set_xticklabels(tks, fontsize=8)
        ax.set_yticks(range(4));  ax.set_yticklabels(tks, fontsize=8)
        for ii in range(4):
            for jj in range(4):
                v   = mat[ii, jj]
                txt = f"{v:.2e}" if v > 999 else f"{v:.3f}"
                fc  = "white" if (im.norm(v + 1e-30) if not use_log else
                                  (np.log10(v+1e-30)-np.log10(max(vmin,1e-30))) /
                                  max(np.log10(vmax)-np.log10(max(vmin,1e-30)), 1e-6)) > 0.55 \
                      else "black"
                ax.text(jj, ii, txt, ha="center", va="center", fontsize=7.5, color=fc)
        plt.colorbar(im, ax=ax, shrink=0.82)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(eqlabel, fontsize=7.5)

    fig.suptitle("TDA Distance Matrix Atlas — Topological Dissimilarity Between Units",
                 fontsize=12, fontweight="bold", y=1.01)
    save(fig, "fig07_tda_distance_atlas.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 08 · Persistent Homology Analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig08(phd, vpt, gng, gtm):
    print("fig08 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(2, 2, hspace=0.52, wspace=0.38, figure=fig)

    # (0,0) H1 Bottleneck Distance Matrix
    ax00 = fig.add_subplot(gs[0, 0]);  despine(ax00)
    df_ph = phd.copy()
    df_ph.index   = [SERIAL_MAP.get(c, c) for c in df_ph.index]
    df_ph.columns = [SERIAL_MAP.get(c, c) for c in df_ph.columns]
    mat_ph = df_ph.values.astype(float)
    im00   = ax00.imshow(mat_ph, cmap="RdYlGn_r", vmin=0, vmax=mat_ph.max(), aspect="equal")
    tks_ph = list(df_ph.index)
    ax00.set_xticks(range(4));  ax00.set_xticklabels(tks_ph)
    ax00.set_yticks(range(4));  ax00.set_yticklabels(tks_ph)
    for ii in range(4):
        for jj in range(4):
            v  = mat_ph[ii, jj]
            fc = "white" if v > 0.55 * mat_ph.max() else "black"
            ax00.text(jj, ii, f"{v:.3f}", ha="center", va="center", fontsize=9, color=fc)
    plt.colorbar(im00, ax=ax00, shrink=0.83, label=r"$d_B^{(1)}$")
    ax00.set_title(r"$\mathrm{H}_1$ Bottleneck Distance Matrix", fontsize=9, fontweight="bold")
    math_box(ax00, r"$d_B=\inf_\gamma\sup_{x}\|x-\gamma(x)\|_\infty$", loc="lower right")

    # (0,1) Voronoi Regime Intensity scatter
    ax01 = fig.add_subplot(gs[0, 1]);  despine(ax01)
    for u in UNITS:
        vp  = vpt[u]
        vol = vp.cell_volume.clip(upper=vp.cell_volume.quantile(0.95))
        nv  = (vol - vol.min()) / (vol.max() - vol.min() + 1e-10)
        ms  = 4 + nv * 28
        ax01.scatter(vp.freq_hz * 1e-9, vp.regime_intensity,
                     s=ms, color=UNIT_COLORS[u], alpha=0.42, label=f"Unit {u}")
        if "regime_boundary" in vp.columns:
            rb = vp[vp.regime_boundary == 1]
            ax01.scatter(rb.freq_hz * 1e-9, rb.regime_intensity,
                         s=50, marker="*", color=UNIT_COLORS[u],
                         edgecolors="black", lw=0.5, zorder=5)
    ax01.set_xlabel("Frequency (GHz)");  ax01.set_ylabel(r"Regime Intensity $\rho_r$")
    ax01.set_title("Voronoi Regime Intensity\n(* = regime boundary; size prop. cell volume)")
    ax01.legend(ncol=2, fontsize=7.5)

    # (1,0) GNG Topology Statistics bar chart
    ax10 = fig.add_subplot(gs[1, 0]);  despine(ax10)
    sample_gng  = gng[1]
    cands       = ["n_nodes", "n_edges", "graph_diameter", "quantization_rmse"]
    ktp         = [k for k in cands if k in sample_gng.columns]
    if ktp:
        xp = np.arange(len(ktp));  w = 0.18
        for i, u in enumerate(UNITS):
            row_g = gng[u].iloc[0]
            vals  = [float(row_g.get(k, 0)) for k in ktp]
            ax10.bar(xp + i * w, vals, width=w * 0.88, color=UNIT_COLORS[u],
                     label=f"Unit {u}", alpha=0.85)
        ax10.set_xticks(xp + 1.5 * w)
        nice = [k.replace("_", " ").replace("quantization rmse", "Quant. RMSE") for k in ktp]
        ax10.set_xticklabels(nice, fontsize=8)
        ax10.set_ylabel("Value");  ax10.legend(ncol=2, fontsize=7.5)
    ax10.set_title("GNG Topology Statistics (Complex Manifold)")

    # (1,1) Markov Transition Matrix — Unit 1
    ax11 = fig.add_subplot(gs[1, 1]);  despine(ax11)
    P_raw  = gtm[1].values.astype(float)
    rs     = P_raw.sum(axis=1, keepdims=True)
    P_norm = np.where(rs > 0, P_raw / rs, 0.0)
    im11   = ax11.imshow(P_norm, cmap="hot_r", vmin=0, vmax=1.0, aspect="auto")
    plt.colorbar(im11, ax=ax11, shrink=0.83, label=r"$P_{ij}$")
    ax11.set_xlabel(r"To state $j$", fontsize=8)
    ax11.set_ylabel(r"From state $i$", fontsize=8)
    ax11.set_title(r"Markov Transition $P_{ij}$ — Unit 1 (Complex GNG)")
    math_box(ax11, r"$\mathbf{P}=D^{-1}\mathbf{C},\;D_{ii}=\sum_j C_{ij}$")

    fig.suptitle("Persistent Homology and Topological Analysis",
                 fontsize=12, fontweight="bold", y=1.01)
    save(fig, "fig08_persistent_homology.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 09 · Autoencoder Latent Space
# ─────────────────────────────────────────────────────────────────────────────
def fig09(aed, aeh, ivh):
    print("fig09 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(1, 3, wspace=0.42, figure=fig)

    # (0) Training convergence
    ax0 = fig.add_subplot(gs[0, 0]);  despine(ax0)
    ax0.semilogy(aeh.epoch, aeh.loss, color="steelblue", lw=1.8, label="Autoencoder")
    ax0.set_xlabel("Epoch");  ax0.set_ylabel("MSE Loss (log)", color="steelblue")
    ax0.tick_params(axis="y", labelcolor="steelblue")
    ax0b = ax0.twinx()
    ax0b.semilogy(ivh.epoch, ivh.loss, color="darkorange", lw=1.8, ls="--",
                  label="Inverse model")
    ax0b.set_ylabel("Inverse model loss (log)", color="darkorange")
    ax0b.tick_params(axis="y", labelcolor="darkorange")
    ax0b.spines["top"].set_visible(False)
    ax0.annotate(f"Final: {aeh.loss.iloc[-1]:.5f}",
                 xy=(aeh.epoch.iloc[-1], aeh.loss.iloc[-1]),
                 xytext=(aeh.epoch.max() * 0.6, aeh.loss.iloc[-1] * 3),
                 fontsize=7, color="steelblue",
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="steelblue"))
    ax0b.annotate(f"Final: {ivh.loss.iloc[-1]:.5f}",
                  xy=(ivh.epoch.iloc[-1], ivh.loss.iloc[-1]),
                  xytext=(ivh.epoch.max() * 0.5, ivh.loss.iloc[-1] * 4),
                  fontsize=7, color="darkorange",
                  arrowprops=dict(arrowstyle="->", lw=0.7, color="darkorange"))
    l1, lb1 = ax0.get_legend_handles_labels()
    l2, lb2 = ax0b.get_legend_handles_labels()
    ax0.legend(l1 + l2, lb1 + lb2, loc="upper right", fontsize=7.5)
    ax0.set_title("Training Convergence\n(AE + Inverse Model)")

    # (1) PCA biplot of 64-dim AE descriptors
    ax1 = fig.add_subplot(gs[0, 1]);  despine(ax1)
    ae_cols = [c for c in aed.columns if c.startswith("ae_")]
    unit_col = aed.columns[0]
    X     = aed[ae_cols].values
    pca2  = PCA(n_components=2);  Z = pca2.fit_transform(X)
    ev    = pca2.explained_variance_ratio_
    for i, u in enumerate(UNITS):
        ax1.scatter(Z[i, 0], Z[i, 1], s=250, color=UNIT_COLORS[u],
                    edgecolors="black", lw=1.3, zorder=5)
        ax1.annotate(f"U{u}", xy=(Z[i, 0], Z[i, 1]),
                     xytext=(Z[i, 0] + 0.07 * ((Z[:, 0].max()-Z[:, 0].min()) or 1),
                             Z[i, 1] + 0.04 * ((Z[:, 1].max()-Z[:, 1].min()) or 1)),
                     fontsize=9, fontweight="bold", color=UNIT_COLORS[u])
    loadings = pca2.components_.T
    scale    = 0.75 * max(np.abs(Z).max(), 0.1) / max(np.abs(loadings).max(), 1e-8)
    top_dims = (set(np.argsort(np.abs(loadings[:, 0]))[-5:]) |
                set(np.argsort(np.abs(loadings[:, 1]))[-5:]))
    for d in sorted(top_dims):
        ax1.annotate("", xy=(loadings[d, 0]*scale, loadings[d, 1]*scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=0.8, alpha=0.6))
        ax1.text(loadings[d, 0]*scale*1.08, loadings[d, 1]*scale*1.08,
                 f"$z_{{{d}}}$", fontsize=6.5, color="gray", ha="center")
    ax1.set_xlabel(f"PC1 ({ev[0]*100:.1f} %)")
    ax1.set_ylabel(f"PC2 ({ev[1]*100:.1f} %)")
    ax1.set_title(f"PCA Biplot — 64-dim AE Descriptors\n(cumul. var. {sum(ev)*100:.1f} %)")
    math_box(ax1, r"$\mathbf{z}_{\rm AE}\in\mathbb{R}^{64}$" + "\n(mean-pooled window latents)",
             loc="lower right")

    # (2) Descriptor heatmap
    ax2 = fig.add_subplot(gs[0, 2]);  despine(ax2)
    Z_mat = aed.set_index(unit_col)[ae_cols].values
    vabs  = np.abs(Z_mat).max()
    im    = ax2.imshow(Z_mat, cmap="RdBu_r", aspect="auto",
                       norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-vabs, vmax=vabs))
    ax2.set_yticks(range(4));  ax2.set_yticklabels([f"U{u}" for u in UNITS])
    ax2.set_xlabel(r"Latent dim. ($z_0\ldots z_{63}$)")
    ax2.set_title(r"AE Descriptor Matrix $\mathbf{Z}_{\rm AE}\in\mathbb{R}^{4\times 64}$")
    for k in range(1, 8):
        ax2.axvline(8*k - 0.5, color="gray", lw=0.5, alpha=0.5)
    plt.colorbar(im, ax=ax2, orientation="horizontal", pad=0.18, label="Activation")

    fig.suptitle("Autoencoder Latent Space — Convergence, PCA Biplot, Descriptor Heatmap",
                 fontsize=11, fontweight="bold", y=1.03)
    save(fig, "fig09_autoencoder_latent_space.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 · ML Benchmark Heatmaps
# ─────────────────────────────────────────────────────────────────────────────
def fig10(clf, reg):
    print("fig10 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(1, 3, wspace=0.52, figure=fig)

    MODEL_CLF = ["LogReg","LinearSVM","RBF-SVM","RandomForest",
                 "GradBoost","k-NN (k=7)","GaussProcC"]
    MODEL_REG = ["Ridge","LinearSVR","RBF-SVR","RandomForest",
                 "GradBoost","k-NN (k=7)","GaussProcR"]
    LAYERS    = ["rf","tda","ae","all"]
    LLABELS   = ["RF scalar","TDA topo.","AE latent","All fused"]

    def pivot(df, models, metric, task=None):
        d = df[df["task"] == task].copy() if task and "task" in df.columns else df.copy()
        g = d.groupby(["model","feature_layer"])[metric].mean().unstack()
        for mm in models:
            if mm not in g.index:
                g.loc[mm] = np.nan
        for ll in LAYERS:
            if ll not in g.columns:
                g[ll] = np.nan
        return g.reindex(index=models, columns=LAYERS).values

    def draw_heatmap(ax, mat, models, cmap, norm, title, cb_label):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(4));  ax.set_xticklabels(LLABELS, rotation=22, ha="right", fontsize=8)
        ax.set_yticks(range(len(models)));  ax.set_yticklabels(models, fontsize=8)
        for ii in range(len(models)):
            for jj in range(4):
                v = mat[ii, jj]
                if not np.isnan(v):
                    fc = "white" if (v > 0.85 or v < 0.0) else "black"
                    ax.text(jj, ii, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color=fc)
        plt.colorbar(im, ax=ax, shrink=0.85, label=cb_label)
        ax.set_title(title, fontsize=10, fontweight="bold")

    draw_heatmap(fig.add_subplot(gs[0, 0]),
                 pivot(clf, MODEL_CLF, "f1_macro", "binary_cluster"),
                 MODEL_CLF, "YlGn",
                 mcolors.Normalize(vmin=0.3, vmax=1.0),
                 "Task 1: Binary Cluster\n$F_1$-Macro (5-fold CV)",
                 "$F_1$-macro")

    draw_heatmap(fig.add_subplot(gs[0, 1]),
                 pivot(clf, MODEL_CLF, "f1_macro", "dominant_unit_4class"),
                 MODEL_CLF, "YlOrRd",
                 mcolors.Normalize(vmin=0.4, vmax=1.0),
                 "Task 2: 4-Class Dominant Unit\n$F_1$-Macro (5-fold CV)",
                 "$F_1$-macro")

    draw_heatmap(fig.add_subplot(gs[0, 2]),
                 pivot(reg, MODEL_REG, "r2"),
                 MODEL_REG, "RdYlGn",
                 mcolors.TwoSlopeNorm(vcenter=0, vmin=-0.2, vmax=1.0),
                 "Task 3: $|S_{21}|_{\\rm max}$ Regression\n$R^2$ (5-fold CV)",
                 "$R^2$")

    fig.suptitle(
        "ML Benchmark — 5-fold CV  ·  7 Estimators $\\times$ 4 Feature Layers",
        fontsize=12, fontweight="bold", y=1.03)
    save(fig, "fig10_ml_benchmark_heatmaps.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 · Leave-Real-Out Generalization
# ─────────────────────────────────────────────────────────────────────────────
def fig11(lrc, lrr):
    print("fig11 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(1, 2, wspace=0.45, figure=fig)

    MODEL_ORDER = ["LogReg","LinearSVM","RBF-SVM","RandomForest",
                   "GradBoost","k-NN (k=7)","GaussProcC"]
    LAYERS = ["rf","tda","ae","all"]
    LC_C   = {"rf":"#1f77b4","tda":"#ff7f0e","ae":"#2ca02c","all":"#d62728"}
    LC_L   = {"rf":"RF scalar","tda":"TDA topo.","ae":"AE latent","all":"All fused"}

    # Panel 0 — binary accuracy bars
    ax0   = fig.add_subplot(gs[0, 0]);  despine(ax0)
    blrc  = lrc[lrc.task == "binary_cluster"] if "task" in lrc.columns else lrc
    xp    = np.arange(len(MODEL_ORDER));  w = 0.20
    for i, layer in enumerate(LAYERS):
        sub  = blrc[blrc.feature_layer == layer].set_index("model")
        vals = [float(sub.loc[mdl, "accuracy"]) if mdl in sub.index else 0.0
                for mdl in MODEL_ORDER]
        ax0.bar(xp + i*w, vals, width=w*0.9, color=LC_C[layer],
                label=LC_L[layer], alpha=0.85)
    ax0.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax0.set_ylim(0, 1.20);  ax0.set_yticks([0, .25, .50, .75, 1.0])
    ax0.set_xticks(xp + 1.5*w)
    ax0.set_xticklabels(MODEL_ORDER, rotation=28, ha="right", fontsize=7.5)
    ax0.set_ylabel("LRO Accuracy")
    ax0.set_title("LRO: Binary Cluster Classification\n(train: 2 000 synthetic -> test: 4 real)")
    ax0.legend(ncol=2, fontsize=8)
    math_box(ax0, r"$\mathrm{Acc}=\frac{1}{N}\sum_i\mathbf{1}[\hat{y}_i=y_i]$")

    # Panel 1 — regression scatter (RF layer)
    ax1  = fig.add_subplot(gs[0, 1]);  despine(ax1)
    lrr_rf = lrr[lrr.feature_layer == "rf"].copy()
    cmap_fn = plt.cm.get_cmap("tab10")
    plotted = 0;  all_true = None
    for i_mdl, (_, row) in enumerate(lrr_rf.iterrows()):
        try:
            preds = np.array(ast.literal_eval(str(row["preds"])), dtype=float)
            trues = np.array(ast.literal_eval(str(row["true"])),  dtype=float)
        except Exception:
            continue
        if all_true is None:
            all_true = trues
            lo, hi   = trues.min() - 0.5, trues.max() + 0.5
            ax1.plot([lo, hi], [lo, hi], "k--", lw=0.9, alpha=0.5, label="Perfect")
        rmse = float(row.get("rmse", np.nan))
        r2   = float(row.get("r2",   np.nan))
        ax1.scatter(trues, preds, s=70, color=cmap_fn(i_mdl % 10), zorder=5, alpha=0.88,
                    label=f"{row.get('model','')} R²={r2:.3f} RMSE={rmse:.3f}")
        plotted += 1
    if plotted:
        ax1.set_xlim(lo, hi);  ax1.set_ylim(lo, hi)
        ax1.set_xlabel(r"True $|S_{21}|_{\max}$ (dB)")
        ax1.set_ylabel(r"Predicted $|S_{21}|_{\max}$ (dB)")
        ax1.legend(fontsize=6.2, ncol=1, loc="upper left")
        math_box(ax1,
                 r"$R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}$",
                 loc="lower right")
    else:
        ax1.text(0.5, 0.5, "LRO regression data unavailable",
                 ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title(r"LRO Regression: $|S_{21}|_{\max}$ (RF feature layer)")

    fig.suptitle("Leave-Real-Out Generalization: Synthetic-to-Real Transfer",
                 fontsize=12, fontweight="bold", y=1.03)
    save(fig, "fig11_leave_real_out.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 · Synthetic Data Characterization
# ─────────────────────────────────────────────────────────────────────────────
def fig12(sf, m, bhi):
    print("fig12 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(2, 2, hspace=0.48, wspace=0.40, figure=fig)

    RF_COLS = [c for c in ["s21_max_db","bw_3db_mhz","f_center_ghz","gd_mean_ns","pb_ripple_db"]
               if c in sf.columns]

    real_rf = {}
    for u in UNITS:
        df = m[u];  pb = df[df.s21_db > -3]
        real_rf[u] = {
            "s21_max_db":   df.s21_db.max(),
            "bw_3db_mhz":   (pb.freq_hz.max()-pb.freq_hz.min())/1e6 if len(pb)>1 else np.nan,
            "f_center_ghz": pb.freq_hz.mean()/1e9                    if len(pb)>1 else np.nan,
            "gd_mean_ns":   df.group_delay_s21_s.mean()*1e9,
            "pb_ripple_db": (pb.s21_db.max()-pb.s21_db.min())        if len(pb)>1 else np.nan,
        }

    # (0,0) PCA biplot — real vs synthetic
    ax00 = fig.add_subplot(gs[0, 0]);  despine(ax00)
    X_s  = sf[RF_COLS].dropna().values
    scaler = StandardScaler().fit(X_s)
    pca_s  = PCA(n_components=2)
    Z_s    = pca_s.fit_transform(scaler.transform(X_s))
    ev_s   = pca_s.explained_variance_ratio_

    for c_val, c_color, c_label in [(0,"#2196F3","Cluster 0 (U1/2 style)"),
                                     (1,"#FF5722","Cluster 1 (U3/4 style)")]:
        mask = sf["cluster"].values == c_val
        ax00.scatter(Z_s[mask, 0], Z_s[mask, 1], s=5, alpha=0.18, color=c_color, label=c_label)

    X_r    = np.array([[real_rf[u].get(c, np.nan) for c in RF_COLS] for u in UNITS])
    Z_r    = pca_s.transform(scaler.transform(np.nan_to_num(X_r)))
    for i, u in enumerate(UNITS):
        ax00.scatter(Z_r[i,0], Z_r[i,1], s=200, marker="*", color=UNIT_COLORS[u],
                     edgecolors="black", lw=1.5, zorder=10, label=f"Real U{u}")
        ax00.annotate(f"U{u}", xy=(Z_r[i,0], Z_r[i,1]),
                      xytext=(Z_r[i,0]+0.12, Z_r[i,1]+0.06),
                      fontsize=8.5, fontweight="bold", color=UNIT_COLORS[u])

    ld = pca_s.components_.T
    sc = 0.7*max(np.abs(Z_r).max(), np.abs(Z_s).max()) / max(np.abs(ld).max(), 1e-8)
    for k, c in enumerate(RF_COLS):
        ax00.annotate("", xy=(ld[k,0]*sc, ld[k,1]*sc), xytext=(0,0),
                      arrowprops=dict(arrowstyle="->", color="darkred", lw=1.1, alpha=0.75))
        nice = c.replace("_"," ").replace("db","(dB)").replace("mhz","(MHz)").replace("ghz","(GHz)")
        ax00.text(ld[k,0]*sc*1.12, ld[k,1]*sc*1.12, nice, fontsize=6.8, color="darkred", ha="center")

    ax00.set_xlabel(f"PC1 ({ev_s[0]*100:.1f} %)");  ax00.set_ylabel(f"PC2 ({ev_s[1]*100:.1f} %)")
    ax00.set_title("PCA Biplot: Real (*) vs. Synthetic RF Scalars")
    ax00.legend(fontsize=6.5, ncol=2, loc="lower right")

    # (0,1) Dirichlet blend weight PCA
    ax01 = fig.add_subplot(gs[0, 1]);  despine(ax01)
    blend_cols = [c for c in ["blend_w1","blend_w2","blend_w3","blend_w4"] if c in sf.columns]
    if blend_cols:
        W    = sf[blend_cols].values
        pca_w = PCA(n_components=2);  Z_w = pca_w.fit_transform(W)
        ev_w  = pca_w.explained_variance_ratio_
        for c_val, c_color, c_label in [(0,"#2196F3","Cluster 0"),(1,"#FF5722","Cluster 1")]:
            mask = sf["cluster"].values == c_val
            ax01.scatter(Z_w[mask,0], Z_w[mask,1], s=5, alpha=0.22, color=c_color, label=c_label)
        ax01.set_xlabel(f"Blend PC1 ({ev_w[0]*100:.1f} %)")
        ax01.set_ylabel(f"Blend PC2 ({ev_w[1]*100:.1f} %)")
        ax01.legend(fontsize=8)
        math_box(ax01, r"$\mathbf{w}\sim\mathrm{Dir}(\alpha)$, $\alpha=[2,2,2,2]$"
                 + "\n" + r"$\mathbf{S}_{\rm synth}=\sum_k w_k\mathbf{S}_k$")
    ax01.set_title(r"Dirichlet Blend Weight PCA")

    # (1,0) BW KDE
    ax10 = fig.add_subplot(gs[1, 0]);  despine(ax10)
    if "bw_3db_mhz" in sf.columns:
        bw = sf["bw_3db_mhz"].dropna().values
        kd = gaussian_kde(bw)
        xv = np.linspace(bw.min()-15, bw.max()+15, 500)
        ax10.fill_between(xv, kd(xv), alpha=0.30, color="steelblue")
        ax10.plot(xv, kd(xv), color="steelblue", lw=1.8, label="Synthetic KDE")
        for u in UNITS:
            v = real_rf[u].get("bw_3db_mhz", np.nan)
            if not np.isnan(v):
                ax10.axvline(v, color=UNIT_COLORS[u], lw=1.8, ls="--", label=f"U{u}")
        ax10.set_xlabel(r"$B_{-3\mathrm{dB}}$ (MHz)");  ax10.set_ylabel("KDE density")
        ax10.set_title(r"$B_{-3\mathrm{dB}}$: Synthetic vs. Real Units")
        ax10.legend(fontsize=7.5, ncol=3)

    # (1,1) s21_max KDE + Bayesian HDI
    ax11 = fig.add_subplot(gs[1, 1]);  despine(ax11)
    if "s21_max_db" in sf.columns:
        sd = sf["s21_max_db"].dropna().values
        kd2 = gaussian_kde(sd)
        xv2 = np.linspace(sd.min()-0.5, sd.max()+0.5, 500)
        yv2 = kd2(xv2)
        ax11.fill_between(xv2, yv2, alpha=0.28, color="darkorange")
        ax11.plot(xv2, yv2, color="darkorange", lw=1.8, label=r"Synthetic KDE")
        for u in UNITS:
            v = real_rf[u].get("s21_max_db", np.nan)
            ax11.axvline(v, color=UNIT_COLORS[u], lw=1.8, ls="--", label=f"U{u}")
            try:
                row_h = bhi[u].iloc[0]
                col_h = next((c for c in ["s21_db_max","s21_max_db"] if c in row_h.index), None)
                if col_h:
                    _, lo_h, hi_h = parse_hdi(row_h[col_h])
                    ax11.fill_betweenx([0, 0.10*yv2.max()], lo_h, hi_h,
                                       color=UNIT_COLORS[u], alpha=0.35)
            except Exception:
                pass
        ax11.set_xlabel(r"$|S_{21}|_{\max}$ (dB)");  ax11.set_ylabel("KDE density")
        ax11.set_title(r"$|S_{21}|_{\max}$: KDE with Bayesian 95 % HDI bands")
        ax11.legend(fontsize=7.5, ncol=3)

    fig.suptitle("Synthetic Data Characterization — Dirichlet Blend, RF Distributions, PCA",
                 fontsize=12, fontweight="bold", y=1.01)
    save(fig, "fig12_synthetic_characterization.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 13 · Bayesian Scalar HDI Comparison (forest plot)
# ─────────────────────────────────────────────────────────────────────────────
def fig13(bhi):
    print("fig13 …")
    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(3, 2, hspace=0.65, wspace=0.50, figure=fig)

    # Build HDI dict for every unit
    hdi_d = {}
    for u in UNITS:
        row_h   = bhi[u].iloc[0]
        hdi_d[u] = {col: parse_hdi(row_h[col]) for col in row_h.index}

    available = set.intersection(*[set(hdi_d[u].keys()) for u in UNITS])

    panels = [
        ("s21_db_max",               r"Peak Insertion Loss $|S_{21}|_{\max}$",
         r"$|S_{21}|_{\max}$ (dB)"),
        ("s21_db_min",               r"Stopband Attenuation $|S_{21}|_{\min}$",
         r"$|S_{21}|_{\min}$ (dB)"),
        ("s21_db_mean",              r"Mean Insertion Loss $\langle|S_{21}|\rangle$",
         r"$\langle|S_{21}|\rangle$ (dB)"),
        ("group_delay_mean_ns",      r"Mean Group Delay $\langle\tau_g\rangle$",
         r"$\langle\tau_g\rangle$ (ns)"),
        ("group_delay_std_ps",       r"GD Dispersion $\sigma_{\tau_g}$",
         r"$\sigma_{\tau_g}$ (ps)"),
        ("passivity_sigma_max_mean", r"Mean Passivity Bound $\langle\sigma_{\max}\rangle$",
         r"$\langle\sigma_{\max}(\mathbf{S})\rangle$"),
    ]

    for idx, (key, title, xlabel) in enumerate(panels):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, col]);  despine(ax)

        if key not in available:
            ax.text(0.5, 0.5, f"'{key}' absent from HDI summary",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=9);  continue

        meds, los, his = [], [], []
        for i, u in enumerate(UNITS):
            med, lo, hi = hdi_d[u][key]
            meds.append(med);  los.append(lo);  his.append(hi)
            ax.plot([lo, hi], [i, i], color=UNIT_COLORS[u], lw=4.5,
                    solid_capstyle="round", alpha=0.85)
            ax.scatter(med, i, s=110, color=UNIT_COLORS[u],
                       edgecolors="black", lw=1.2, zorder=6)
            fmt = ".2f" if abs(med) < 100 else ".1f"
            span = abs(hi - lo) + 1e-12
            ax.text(hi + 0.02*span, i, f" {med:{fmt}}", va="center", ha="left", fontsize=8)

        grand = np.nanmean(meds)
        ax.axvline(grand, color="gray", lw=0.9, ls=":", alpha=0.7,
                   label=f"Cross-unit mean: {grand:.4g}")
        if "passivity" in key:
            ax.axvline(1.0, color="red", lw=1.0, ls="--", alpha=0.8,
                       label=r"$\sigma_{\max}=1$")

        ax.set_yticks(range(4))
        ax.set_yticklabels([f"Unit {u}" for u in UNITS], fontsize=8.5)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        math_box(ax, "95 % HDI", loc="upper right")

    fig.suptitle(
        "Bayesian HDI Scalar Summary — Inter-Unit Credible Interval Comparison\n"
        "(horizontal bars = 95 % HDI; dot = posterior median)",
        fontsize=11, fontweight="bold", y=1.02)
    fig.text(0.5, -0.005,
             "Intervals from complex-Gaussian posterior draws on S-parameter spectral noise.",
             ha="center", fontsize=7.5, style="italic", color="gray")
    save(fig, "fig13_bayesian_scalar_hdi_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 14 · 3-D S21 Spectral Waterfall
# ─────────────────────────────────────────────────────────────────────────────
def fig14(m):
    """
    Waterfall display: S21(f) for each unit rendered as a filled polygon at a
    distinct Y-depth position in 3-D.  A translucent gold plane marks the
    -3 dB passband boundary.  Implemented with mpl_toolkits PolyCollection
    projected onto a 3-D axes so that the spectral envelopes are visually
    separable without colour overlap.
    """
    print("fig14 …")
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=W11)
    ax  = fig.add_subplot(111, projection="3d")

    y_pos   = {u: i for i, u in enumerate(UNITS)}
    f_ref   = fghz(m[1])

    verts   = []
    colors  = []
    for u in UNITS:
        df  = m[u]
        f   = fghz(df)
        z   = df.s21_db.values.copy()
        # closed polygon: left wall → trace → right wall → floor
        xs  = np.concatenate([[f[0]], f, [f[-1]]])
        zs  = np.concatenate([[-80],  z, [-80]])
        y   = y_pos[u]
        poly = list(zip(xs, [y]*len(xs), zs))
        verts.append(poly)
        colors.append(UNIT_COLORS[u])

    pc = Poly3DCollection(verts, alpha=0.55, linewidths=0.7)
    pc.set_facecolor(colors)
    pc.set_edgecolor(colors)
    ax.add_collection3d(pc)

    # -3 dB reference plane
    flo, fhi = f_ref.min(), f_ref.max()
    ylo, yhi = -0.5, len(UNITS) - 0.5
    XX, YY = np.meshgrid([flo, fhi], [ylo, yhi])
    ZZ = np.full_like(XX, -3.0)
    ax.plot_surface(XX, YY, ZZ, color="gold", alpha=0.20, zorder=0)
    ax.plot_wireframe(XX, YY, ZZ, color="goldenrod", linewidth=0.6, alpha=0.55, zorder=1)

    ax.set_xlabel(r"Frequency $f$ (GHz)", labelpad=8)
    ax.set_ylabel("Unit", labelpad=8)
    ax.set_zlabel(r"$|S_{21}|$ (dB)", labelpad=8)
    ax.set_yticks(list(y_pos.values()))
    ax.set_yticklabels([UNIT_LABEL[u] for u in UNITS], fontsize=8)
    ax.set_zlim(-80, 5)
    ax.view_init(elev=24, azim=-48)

    ax.text2D(0.5, 0.98,
              r"3-D S$_{21}$ Spectral Waterfall — $|S_{21}(f)|$ per Unit"
              "\n(gold plane: $-3$ dB passband boundary)",
              ha="center", va="top", transform=ax.transAxes,
              fontsize=10, fontweight="bold")

    ax.text2D(0.02, 0.92,
              r"$H(j\omega)\triangleq S_{21}$" + "\n"
              r"Passband: $|S_{21}| \geq -3\,\mathrm{dB}$",
              transform=ax.transAxes, fontsize=7.5,
              va="top", ha="left",
              bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                        alpha=0.75, edgecolor="gray", linewidth=0.6))

    fig.tight_layout(pad=0.4)
    save(fig, "fig14_3d_s21_waterfall.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 15 · 3-D TDA Phase-Space Embedding
# ─────────────────────────────────────────────────────────────────────────────
def fig15(m):
    """
    Projects each unit's S-parameter trace into the three-dimensional
    embedding space (Re S11, Im S11, |S21|) used by the Voronoi TDA stage.
    Points are coloured by normalised frequency so that the spectral
    trajectory through the scattering-parameter manifold is visible.
    The 2x2 layout allows per-unit detail without clutter.
    """
    print("fig15 …")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=W11)
    for i, u in enumerate(UNITS):
        ax  = fig.add_subplot(2, 2, i+1, projection="3d")
        df  = m[u]
        x   = df.s11_re.values
        y   = df.s11_im.values
        z   = df.s21_mag.values           # linear magnitude
        f   = df.freq_hz.values
        fnorm = (f - f.min()) / (f.max() - f.min() + 1e-20)

        sc = ax.scatter(x, y, z, c=fnorm, cmap="plasma",
                        s=4, alpha=0.75, linewidths=0)

        # passband trajectory overlay
        pb = z >= 10**(-3/20)             # linear -3 dB threshold
        if pb.any():
            ax.plot(x[pb], y[pb], z[pb], color=UNIT_COLORS[u],
                    lw=1.5, alpha=0.85, label="Passband")

        ax.set_xlabel(r"$\mathrm{Re}\,S_{11}$", fontsize=7, labelpad=4)
        ax.set_ylabel(r"$\mathrm{Im}\,S_{11}$", fontsize=7, labelpad=4)
        ax.set_zlabel(r"$|S_{21}|$",            fontsize=7, labelpad=4)
        ax.set_title(UNIT_LABEL[u], fontsize=9, fontweight="bold")
        ax.view_init(elev=22, azim=-55)
        ax.tick_params(labelsize=6)

        if i == 0:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.08)
            cbar.set_label(r"$f / f_{\max}$", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        r"3-D TDA Phase-Space Embedding: $(\mathrm{Re}\,S_{11},\,\mathrm{Im}\,S_{11},\,|S_{21}|)$"
        "\nTrajectory through scattering-parameter manifold coloured by normalised frequency",
        fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.8)
    save(fig, "fig15_3d_tda_embedding.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 16 · 3-D Vector-Fit Transfer-Function Surface
# ─────────────────────────────────────────────────────────────────────────────
def fig16(vfp, m):
    """
    Evaluates the rational pole-residue VF model H(s) = D + E·s + Sigma r_k/(s-p_k)
    on a 2-D grid in the left half of the complex plane (s = sigma + j*omega,
    sigma <= 0) and renders |H(s)| as a surface.  The jw-axis (sigma=0) slice
    corresponds to the measured S21 frequency response.

    This exposes the poles, resonant peaks, and passivity region of the fitted
    rational model — a classical tool in network theory and system identification.
    """
    print("fig16 …")
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=W11)

    for i, u in enumerate(UNITS):
        ax = fig.add_subplot(2, 2, i+1, projection="3d")

        npz = vfp.get(u)
        df  = m[u]
        f_hz = df.freq_hz.values
        omega_max = 2.0 * np.pi * f_hz.max()

        # Evaluate on coarse grid — fine enough for surface, fast enough to render
        n_sig, n_om = 40, 60
        sigma_vec = np.linspace(-0.35 * omega_max, 0.0, n_sig)  # stable LHP
        omega_vec = np.linspace(-omega_max, omega_max, n_om)
        SIG, OMG  = np.meshgrid(sigma_vec, omega_vec)
        S_grid    = SIG + 1j * OMG

        if npz is not None:
            H_grid = np.zeros_like(S_grid, dtype=complex)
            H_grid = vf_reconstruct_s21(npz, OMG.ravel() / (2*np.pi))
            if H_grid is None:
                H_grid = np.zeros_like(S_grid, dtype=complex)
            else:
                # Build full surface: for sigma != 0 use analytic continuation
                poles    = npz["poles"]
                residues = npz["residues"]
                const    = npz.get("constants",    npz.get("constant_coefficients",    None))
                prop     = npz.get("proportionals", npz.get("proportional_coefficients", None))
                d_val = float(np.real(const[1])) if (const is not None and len(const) > 1) else 0.0
                e_val = float(np.real(prop[1]))  if (prop  is not None and len(prop)  > 1) else 0.0
                res21 = residues[1] if residues.ndim > 1 else residues
                s_flat = S_grid.ravel()
                H_flat = np.full(len(s_flat), d_val + e_val * s_flat, dtype=complex)
                for pole, res in zip(poles, res21):
                    H_flat += res / (s_flat - pole)
                    if abs(pole.imag) > 1e3:
                        H_flat += np.conj(res) / (s_flat - np.conj(pole))
                H_grid = H_flat.reshape(S_grid.shape)
        else:
            H_grid = np.zeros_like(S_grid, dtype=complex)

        Z_surf = np.abs(H_grid)
        Z_surf = np.clip(Z_surf, 0, 5)            # cap for visual clarity

        # Normalised axes for labelling
        X_plot = SIG / omega_max                   # sigma / omega_max (dimensionless)
        Y_plot = OMG / (2*np.pi * 1e9)            # omega/(2pi) in GHz

        surf = ax.plot_surface(X_plot, Y_plot, Z_surf,
                               cmap="inferno", alpha=0.82,
                               linewidth=0, antialiased=True)

        # jw-axis edge (sigma=0 slice)
        j_idx   = np.argmin(np.abs(sigma_vec))
        n_om    = omega_vec.shape[0]
        ax.plot(np.zeros(n_om), Y_plot[:, j_idx], Z_surf[:, j_idx],
                color=UNIT_COLORS[u], lw=2.0, label=r"$j\omega$-axis")

        ax.set_xlabel(r"$\sigma/\omega_{\max}$", fontsize=7, labelpad=4)
        ax.set_ylabel(r"$\omega / 2\pi$ (GHz)",  fontsize=7, labelpad=4)
        ax.set_zlabel(r"$|H(\sigma+j\omega)|$",  fontsize=7, labelpad=4)
        ax.set_title(UNIT_LABEL[u], fontsize=9, fontweight="bold")
        ax.view_init(elev=28, azim=-50)
        ax.tick_params(labelsize=6)

        if i == 0:
            cbar = fig.colorbar(surf, ax=ax, fraction=0.032, pad=0.10, shrink=0.7)
            cbar.set_label(r"$|H(s)|$", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        r"3-D Vector-Fit Transfer-Function Surface: $|H(\sigma+j\omega)|$, $\sigma \leq 0$"
        "\n"
        r"Rational model $H(s)=D+Es+\sum_k r_k/(s-p_k)$; "
        r"$j\omega$-axis slice = measured $S_{21}$",
        fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8)
    save(fig, "fig16_3d_vf_transfer_surface.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 17 · Time-Domain Impulse and Step Responses
# ─────────────────────────────────────────────────────────────────────────────
def fig17(td, m):
    """
    Computes the time-domain impulse h(t) and step s(t) responses from the
    complex frequency-domain S21 via the inverse discrete Fourier transform
    with Tukey windowing (alpha=0.15) to suppress Gibbs artefacts:

        H[n] = S21[n] * w[n]       (Tukey window)
        h(t) = IDFT{H[n]}          (analytic impulse)
        s(t) = cumsum(h(t)) * dt   (causal step integral)

    Key metrics:
      - Rise time  t_r = t(90%) - t(10%) of |s(t)|
      - Delay time t_d = t(|h(t)| = |h|_max)
      - Ringing Q estimated from envelope decay
    """
    print("fig17 …")
    from scipy.signal.windows import tukey

    fig = plt.figure(figsize=W11)
    gs  = gridspec.GridSpec(2, 4, hspace=0.50, wspace=0.38, figure=fig)

    for i, u in enumerate(UNITS):
        ax_imp = fig.add_subplot(gs[0, i]);  despine(ax_imp)
        ax_stp = fig.add_subplot(gs[1, i]);  despine(ax_stp)

        # ── Build IFFT from frequency metrics (full Nyquist band) ──────────
        df_m   = m[u]
        freq   = df_m.freq_hz.values
        s21_re = df_m.s21_re.values
        s21_im = df_m.s21_im.values
        N      = len(freq)
        dt     = 1.0 / (2.0 * freq.max())          # Nyquist sampling interval
        t_ns   = np.arange(N) * dt * 1e9            # time in ns

        # Tukey window (alpha=0.15) to taper spectral leakage
        win = tukey(N, alpha=0.15)
        H   = (s21_re + 1j * s21_im) * win
        h   = np.real(np.fft.ifft(H)) * 2 * freq.max()   # scale to physical units
        s   = np.cumsum(h) * dt                           # step = integral of impulse

        # Normalise for comparison
        h_norm = h / (np.abs(h).max() + 1e-30)
        s_norm = s / (np.abs(s).max() + 1e-30)

        col = UNIT_COLORS[u]

        # Peak delay
        t_d = t_ns[np.argmax(np.abs(h_norm))]

        # ── Impulse panel ──────────────────────────────────────────────────
        ax_imp.plot(t_ns, h_norm, color=col, lw=1.3)
        ax_imp.axvline(t_d, color=col, lw=0.9, ls="--", alpha=0.65)
        ax_imp.axhline(0, color="gray", lw=0.5, ls=":")
        ax_imp.set_xlim(t_ns[0], min(t_ns[-1], t_d*6 + 0.5))
        ax_imp.set_xlabel(r"$t$ (ns)", fontsize=8)
        if i == 0:
            ax_imp.set_ylabel(r"$|h(t)|$ (norm.)", fontsize=8)
        ax_imp.set_title(UNIT_LABEL[u], fontsize=9, fontweight="bold")
        math_box(ax_imp, fr"$t_d={t_d:.2f}$ ns", loc="upper right", fs=7)

        # ── Step panel ─────────────────────────────────────────────────────
        s_pos = np.clip(s_norm, 0, None)
        ax_stp.plot(t_ns, s_pos, color=col, lw=1.3)
        # 10 % / 90 % rise time
        try:
            i10 = np.where(s_pos >= 0.10)[0][0]
            i90 = np.where(s_pos >= 0.90)[0][0]
            t_r = t_ns[i90] - t_ns[i10]
            ax_stp.axvspan(t_ns[i10], t_ns[i90], color=col, alpha=0.14)
            ax_stp.axhline(0.10, color="gray", lw=0.5, ls=":", alpha=0.6)
            ax_stp.axhline(0.90, color="gray", lw=0.5, ls=":", alpha=0.6)
            math_box(ax_stp, fr"$t_r={t_r:.2f}$ ns", loc="lower right", fs=7)
        except IndexError:
            pass
        ax_stp.set_xlim(t_ns[0], min(t_ns[-1], t_d*6 + 0.5))
        ax_stp.set_ylim(-0.05, 1.15)
        ax_stp.set_xlabel(r"$t$ (ns)", fontsize=8)
        if i == 0:
            ax_stp.set_ylabel(r"$s(t)$ (norm.)", fontsize=8)

    fig.suptitle(
        r"Time-Domain Analysis: IFFT Impulse $h(t)$ and Step $s(t)$ Responses"
        "\n"
        r"Tukey-windowed IDFT of $S_{21}(f)$; "
        r"$t_d$ = peak delay, $t_r$ = 10%–90% rise time",
        fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.8)
    save(fig, "fig17_time_domain_impulse_step.png")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 18 · Smith Chart — S11 Reflection Coefficient
# ─────────────────────────────────────────────────────────────────────────────
def fig18(m):
    """
    Plots S11(f) = (Z - Z0)/(Z + Z0) on the normalised Smith chart — the
    bilinear Mobius transform mapping the right half of the Z-plane to the
    unit disk in the Gamma-plane.

    Constant-resistance circles: |Gamma - r/(r+1)| = 1/(r+1)  for r in {0,0.2,0.5,1,2,5}
    Constant-reactance arcs:    Im(Gamma)^2 + (Re(Gamma)-1)^2 = (1/x)^2  clipped to |Gamma|<=1

    Colour-coding by normalised frequency allows the frequency trajectory
    through the chart to be read directly.
    """
    print("fig18 …")

    fig = plt.figure(figsize=W10)
    gs  = gridspec.GridSpec(2, 2, hspace=0.20, wspace=0.20, figure=fig)

    # ── Smith chart grid (precomputed, drawn on every subplot) ─────────────
    def draw_smith_grid(ax, lw_grid=0.35, c_grid="#aaaaaa"):
        ax.set_aspect("equal")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.axis("off")
        # Outer circle |Gamma|=1
        theta = np.linspace(0, 2*np.pi, 512)
        ax.plot(np.cos(theta), np.sin(theta), color="black", lw=0.9)
        # Constant-resistance circles: centre (r/(r+1), 0), radius 1/(r+1)
        for r in [0, 0.2, 0.5, 1.0, 2.0, 5.0]:
            cx, rad = r/(r+1), 1.0/(r+1)
            phi = np.linspace(0, 2*np.pi, 360)
            xc  = cx + rad * np.cos(phi)
            yc  =       rad * np.sin(phi)
            mask = xc**2 + yc**2 <= 1.001
            ax.plot(xc[mask], yc[mask], color=c_grid, lw=lw_grid)
        # Constant-reactance arcs: centre (1, 1/x), radius |1/x|
        for x in [0.2, 0.5, 1.0, 2.0, 5.0]:
            for sign in [1, -1]:
                cx, cy, rad = 1.0, sign/x, abs(1.0/x)
                phi = np.linspace(0, 2*np.pi, 720)
                xc  = cx + rad * np.cos(phi)
                yc  = cy + rad * np.sin(phi)
                mask = xc**2 + yc**2 <= 1.001
                ax.plot(xc[mask], yc[mask], color=c_grid, lw=lw_grid, ls="--")
        # Real axis
        ax.axhline(0, color=c_grid, lw=lw_grid)
        # Reference labels
        for r, lbl in [(0, "0"), (1, "1"), (5, "5")]:
            cx = r/(r+1)
            ax.text(cx + 1/(r+1), 0.02, lbl, fontsize=5.5, ha="center",
                    color="gray", va="bottom")
        ax.text(0.02, 0.97, r"$Z_0=50\,\Omega$", transform=ax.transAxes,
                fontsize=6.5, color="gray", va="top")

    for i, u in enumerate(UNITS):
        ax  = fig.add_subplot(gs[i // 2, i % 2])
        draw_smith_grid(ax)

        df  = m[u]
        s11 = df.s11_re.values + 1j * df.s11_im.values
        f   = df.freq_hz.values
        fnorm = (f - f.min()) / (f.max() - f.min() + 1e-20)

        sc = ax.scatter(s11.real, s11.imag, c=fnorm, cmap="plasma",
                        s=5, alpha=0.8, linewidths=0, zorder=5)
        # Arrow at mid-frequency to show trajectory direction
        mid = len(f) // 2
        ax.annotate("", xy=(s11.real[mid+1], s11.imag[mid+1]),
                    xytext=(s11.real[mid], s11.imag[mid]),
                    arrowprops=dict(arrowstyle="-|>", color=UNIT_COLORS[u],
                                   lw=1.2, mutation_scale=8))
        ax.set_title(UNIT_LABEL[u], fontsize=9, fontweight="bold", pad=3)

        # RL annotations in passband
        pb = pb_mask(df)
        if pb.any():
            rl_pb = 20*np.log10(np.clip(np.abs(s11[pb]), 1e-10, None))
            math_box(ax,
                     fr"$|S_{{11}}|_\mathrm{{pb}}={np.median(rl_pb):.1f}$ dB",
                     loc="lower left", fs=6.5)

        if i == 0:
            cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03, orientation="vertical")
            cbar.set_label(r"$f/f_{\max}$", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        r"Smith Chart — $S_{11}(\Gamma)$ Reflection Coefficient Trajectory"
        "\n"
        r"Constant-$r$ circles and constant-$x$ arcs of the "
        r"Möbius map $\Gamma = (Z-Z_0)/(Z+Z_0)$; "
        r"colour = normalised frequency",
        fontsize=10, fontweight="bold", y=1.01)
    fig.tight_layout(pad=0.6)
    save(fig, "fig18_smith_chart_s11.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    print(f"Output -> {FIG_OUT}\n")
    d = load_all()
    print("\nGenerating 16 figures …\n")
    fig01(d["m"])
    fig02(d["m"], d["bcb"])
    fig03(d["m"], d["bcb"])
    fig04(d["m"])
    fig05(d["vfp"], d["vfs"])
    fig06(d["m"],   d["vfp"], d["vfs"])
    fig07(d["vcx"], d["vsh"], d["gcx"], d["gsh"])
    fig08(d["phd"], d["vpt"], d["gng"], d["gtm"])
    fig09(d["aed"], d["aeh"], d["ivh"])
    fig10(d["clf"], d["reg"])
    fig11(d["lrc"], d["lrr"])
    fig12(d["sf"],  d["m"],   d["bhi"])
    fig13(d["bhi"])
    fig14(d["m"])
    fig15(d["m"])
    fig16(d["vfp"], d["m"])
    fig17(d["td"],  d["m"])
    fig18(d["m"])
    print(f"\nDone. All 18 figures in:\n  {FIG_OUT}")


if __name__ == "__main__":
    main()
