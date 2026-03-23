#!/usr/bin/env python3
"""
ML model comparison for synthetic RF bandpass filter data.

Evaluates classical ML models across three feature layers derived from the
s2p_tda_rtx4070 analysis pipeline, directly connecting the model comparison
to the topological and spectral findings:

  RF layer   (15 columns) – scalar features extracted from S-parameters
  TDA layer  (74 columns) – topology features from GNG/Voronoi/PH pipeline
  AE layer   (64 columns) – autoencoder latent descriptors
  All        (153 columns) – full feature set

Three tasks, grounded in the prior analysis:
  Task 1  Binary classification    cluster (0 = Unit-1/2 style, 1 = Unit-3/4 style)
  Task 2  4-class classification   dominant_unit (1–4); synthetic samples only
  Task 3  Regression               s21_max_db (insertion-loss peak prediction)

Evaluation:
  • Stratified 5-fold CV (Tasks 1 & 2) / KFold (Task 3)
  • Leave-real-out extra pass: train on 2 000 synthetic, test on 4 real units
  • Results saved to results_classification.csv and results_regression.csv
  • Summary table printed with interpretation vs. analysis findings

Usage:
  python compare_models.py            # standard run
  python compare_models.py --no-gp    # skip slow Gaussian-Process models
  python compare_models.py --n 500    # sub-sample synthetic data for speed
"""

import argparse
import sys
import logging
import warnings
from pathlib import Path

# Ensure UTF-8 output on Windows (avoids CP1252 UnicodeEncodeError for symbols)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, brier_score_loss, f1_score, mean_absolute_error,
    mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR      = Path(__file__).resolve().parent
PIPELINE_DIR    = SCRIPT_DIR.parent
ANALYSIS_ML_DIR = PIPELINE_DIR / "outputs" / "s2p_tda_rtx4070" / "ml"
S2P_FILES       = [
    PIPELINE_DIR / "data" / "040401AA.s2p",
    PIPELINE_DIR / "data" / "040402AA.s2p",
    PIPELINE_DIR / "data" / "040403AA.s2p",
    PIPELINE_DIR / "data" / "040404AA.s2p",
]
UNIT_LABELS     = ["Unit 1", "Unit 2", "Unit 3", "Unit 4"]
# All comparison results go here — keeps synthetic_data/ root clean
OUT_DIR         = SCRIPT_DIR / "results"
PLOT_DIR        = OUT_DIR / "plots"
SEED            = 2026

log = logging.getLogger(__name__)

# ── Feature layer column definitions ─────────────────────────────────────────
RF_FEATURE_COLS = [
    "s21_max_db", "s21_min_db", "s21_pb_rms_db",
    "pb_ripple_db", "bw_3db_mhz", "f_center_ghz",
    "f_3db_low_ghz", "f_3db_high_ghz",
    "gd_mean_ns", "gd_std_ps", "gd_peak_ns",
    "s11_pb_mean_db", "s22_pb_mean_db",
    "reciprocity_max", "passivity_margin",
]
TDA_COL_PREFIX = "topology_feature_"
AE_COL_PREFIX  = "ae_"

CLUSTER_THRESHOLD_DB = -2.0

# ── Analysis findings for cross-referencing ───────────────────────────────────
ANALYSIS_FINDINGS = """
Known findings from the s2p_tda_rtx4070 pipeline:
  • Units 3 & 4 form a tight topological cluster (GNG complex distance ≈ 1.0).
  • Unit 1 is most isolated (highest VF RMS 1.65, only non-passive unit).
  • Unit 2 is the only passivity-compliant unit.
  • AE latent descriptors are nearly indistinguishable across all 4 units.
  • PH H1 bottleneck: Units 1/2 nearest (0.055); Unit 4 most isolated (> 0.21).
  • Binary cluster boundary (s21_max ≥ −2.0 dB) separates Unit 1/2 from 3/4.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Real-unit feature helpers — delegated to utils.py to avoid duplication
# ─────────────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(SCRIPT_DIR))  # ensure utils.py (same directory) is importable
from utils import parse_s2p as _parse_s2p          # noqa: E402
from utils import extract_rf_features as _extract_rf_features  # noqa: E402


def load_real_unit_features() -> pd.DataFrame:
    """
    Build the full 153-feature real-unit DataFrame directly from analysis pipeline
    outputs.  Nothing from the synthetic_data directory is used.

    RF features  : extracted live from the 4 real S2P files
    TDA features : loaded from outputs/s2p_tda_rtx4070/ml/topology_inverse_features.csv
    AE features  : loaded from outputs/s2p_tda_rtx4070/ml/autoencoder_unit_descriptors.csv
    """
    for p in [ANALYSIS_ML_DIR, *S2P_FILES]:
        if not Path(p).exists():
            sys.exit(f"ERROR: required analysis file not found: {p}")

    tda_df = pd.read_csv(ANALYSIS_ML_DIR / "topology_inverse_features.csv")
    ae_df  = pd.read_csv(ANALYSIS_ML_DIR / "autoencoder_unit_descriptors.csv")
    tda_df = tda_df.set_index("unit").loc[UNIT_LABELS].reset_index()
    ae_df  = ae_df.set_index("unit").loc[UNIT_LABELS].reset_index()
    tda_feat_cols = [c for c in tda_df.columns if c.startswith("topology_feature_")]
    ae_feat_cols  = [c for c in ae_df.columns  if c.startswith("ae_")]

    rows = []
    for idx, (path, label) in enumerate(zip(S2P_FILES, UNIT_LABELS)):
        freq, s = _parse_s2p(path)
        feats = _extract_rf_features(freq, s)
        feats["sample_id"]     = label.replace(" ", "_")
        feats["sample_type"]   = "real"
        feats["cluster"]       = int(feats["s21_max_db"] >= CLUSTER_THRESHOLD_DB)
        feats["dominant_unit"] = idx + 1
        for col in tda_feat_cols:
            feats[col] = float(tda_df.loc[idx, col])
        for col in ae_feat_cols:
            feats[col] = float(ae_df.loc[idx, col])
        rows.append(feats)
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

def build_classifiers(include_gp: bool) -> dict:
    models = {
        "LogReg"       : LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
        "LinearSVM"    : LinearSVC(max_iter=2000, C=1.0, random_state=SEED),
        "RBF-SVM"      : SVC(kernel="rbf", probability=True, random_state=SEED),
        "RandomForest" : RandomForestClassifier(n_estimators=200, random_state=SEED,
                                                 n_jobs=-1),
        "GradBoost"    : GradientBoostingClassifier(n_estimators=200, random_state=SEED),
        "k-NN (k=7)"   : KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
    }
    if include_gp:
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        models["GaussProcC"] = GaussianProcessClassifier(kernel=kernel,
                                                          random_state=SEED)
    return models


def build_regressors(include_gp: bool) -> dict:
    models = {
        "Ridge"        : Ridge(alpha=1.0),
        "LinearSVR"    : LinearSVR(max_iter=5000, C=1.0, random_state=SEED),
        "RBF-SVR"      : SVR(kernel="rbf"),
        "RandomForest" : RandomForestRegressor(n_estimators=200, random_state=SEED,
                                                n_jobs=-1),
        "GradBoost"    : GradientBoostingRegressor(n_estimators=200, random_state=SEED),
        "k-NN (k=7)"   : KNeighborsRegressor(n_neighbors=7, n_jobs=-1),
    }
    if include_gp:
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        models["GaussProcR"] = GaussianProcessRegressor(kernel=kernel,
                                                          random_state=SEED,
                                                          normalize_y=True)
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validation helpers
# ─────────────────────────────────────────────────────────────────────────────

def cv_classify(name: str, clf, X: np.ndarray, y: np.ndarray,
                n_splits: int = 5) -> list[dict]:
    """
    Stratified k-fold CV for classification.
    Returns one dict per fold with accuracy, f1_macro, auc_roc, brier_score.
    LinearSVC does not support predict_proba so AUC/Brier are skipped for it.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    has_proba = hasattr(clf, "predict_proba")
    n_classes = len(np.unique(y))
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        row = {
            "model"    : name,
            "fold"     : fold + 1,
            "accuracy" : float(accuracy_score(y[te], preds)),
            "f1_macro" : float(f1_score(y[te], preds, average="macro")),
            "auc_roc"  : None,
            "brier"    : None,
        }
        if has_proba:
            proba = pipe.predict_proba(X[te])
            if n_classes == 2:
                row["auc_roc"] = float(roc_auc_score(y[te], proba[:, 1]))
                row["brier"]   = float(brier_score_loss(
                    y[te], proba[:, 1], pos_label=int(y.max())))
            else:
                row["auc_roc"] = float(roc_auc_score(
                    y[te], proba, multi_class="ovr", average="macro"))
        rows.append(row)
    return rows


def cv_regress(name: str, reg, X: np.ndarray, y: np.ndarray,
               n_splits: int = 5) -> list[dict]:
    """KFold CV for regression. Returns one dict per fold with RMSE, MAE, R²."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X)):
        pipe = Pipeline([("scaler", StandardScaler()), ("reg", clone(reg))])
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        rows.append({
            "model": name,
            "fold" : fold + 1,
            "rmse" : float(np.sqrt(mean_squared_error(y[te], preds))),
            "mae"  : float(mean_absolute_error(y[te], preds)),
            "r2"   : float(r2_score(y[te], preds)),
        })
    return rows


def leave_real_out_classify(name: str, clf, X_train: np.ndarray, y_train: np.ndarray,
                             X_real: np.ndarray, y_real: np.ndarray) -> dict:
    """Train on all synthetic; test on 4 real units. Returns summary dict."""
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clone(clf))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_real)
    has_proba = hasattr(clf, "predict_proba")
    n_classes = len(np.unique(y_train))
    result = {
        "model"    : name,
        "accuracy" : float(accuracy_score(y_real, preds)),
        "f1_macro" : float(f1_score(y_real, preds, average="macro", zero_division=0.0)),
    }
    if has_proba:
        proba = pipe.predict_proba(X_real)
        if n_classes == 2:
            result["auc_roc"] = float(roc_auc_score(y_real, proba[:, 1]))
        else:
            result["auc_roc"] = float(roc_auc_score(
                y_real, proba, multi_class="ovr", average="macro"))
    return result


def leave_real_out_regress(name: str, reg, X_train: np.ndarray, y_train: np.ndarray,
                            X_real: np.ndarray, y_real: np.ndarray) -> dict:
    """Train on synthetic; test on 4 real units. Returns summary dict."""
    pipe = Pipeline([("scaler", StandardScaler()), ("reg", clone(reg))])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_real)
    return {
        "model": name,
        "rmse" : float(np.sqrt(mean_squared_error(y_real, preds))),
        "mae"  : float(mean_absolute_error(y_real, preds)),
        "r2"   : float(r2_score(y_real, preds) if len(y_real) > 1 else float("nan")),
        "preds": preds.tolist(),
        "true" : y_real.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def mean_cv_table(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    """Average fold metrics, grouped by (feature_layer, model)."""
    return (
        df.groupby(["feature_layer", "model"])[metrics]
        .mean()
        .round(4)
        .reset_index()
    )


def print_banner(text: str) -> None:
    width = 72
    log.info("\n" + "─" * width)
    log.info("  %s", text)
    log.info("─" * width)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(include_gp: bool = False, n_subsample: int | None = None) -> None:
    # ── Load data — synthetic from generator; real from analysis pipeline ─────
    OUT_DIR.mkdir(exist_ok=True)
    synth_path = OUT_DIR / "synthetic_features.csv"   # generator writes here
    if not synth_path.exists():
        sys.exit(f"ERROR: {synth_path} not found.  Run generate_synthetic.py first.")

    synth_data = pd.read_csv(synth_path)
    log.info(f"Loaded {len(synth_data)} synthetic samples from synthetic_features.csv")

    log.info("Loading real unit features from analysis pipeline outputs ...")
    real_data = load_real_unit_features()
    log.info(f"  {len(real_data)} real units  [{ANALYSIS_ML_DIR.relative_to(PIPELINE_DIR)}]")

    if n_subsample is not None and n_subsample < len(synth_data):
        synth_data = synth_data.sample(n=n_subsample, random_state=SEED).reset_index(drop=True)
        log.info(f"  Sub-sampled synthetic data to {n_subsample} rows")

    # Feature layer columns derived from synthetic schema (real data shares same columns)
    tda_cols = [c for c in synth_data.columns if c.startswith(TDA_COL_PREFIX)]
    ae_cols  = [c for c in synth_data.columns if c.startswith(AE_COL_PREFIX)]
    rf_cols  = [c for c in RF_FEATURE_COLS if c in synth_data.columns]
    all_cols = rf_cols + tda_cols + ae_cols

    log.info(f"\n  RF features  : {len(rf_cols)}")
    log.info(f"  TDA features : {len(tda_cols)}")
    log.info(f"  AE features  : {len(ae_cols)}")
    log.info(f"  Combined     : {len(all_cols)}")
    log.info(f"\n  Training set : {len(synth_data)} synthetic samples (CV runs on synthetic only)")
    log.info(f"  LRO test set : {len(real_data)} real units (sourced from analysis pipeline)")
    log.info(ANALYSIS_FINDINGS)

    # Classification layer map — full rf_cols
    layer_map = {
        "rf"  : rf_cols,
        "tda" : tda_cols,
        "ae"  : ae_cols,
        "all" : all_cols,
    }
    # Regression layer map — s21_max_db excluded (it is the regression target;
    # including it would trivially leak the label and collapse GBM residuals to 0)
    rf_reg_cols = [c for c in rf_cols if c != "s21_max_db"]
    layer_map_reg = {
        "rf"  : rf_reg_cols,
        "tda" : tda_cols,
        "ae"  : ae_cols,
        "all" : rf_reg_cols + tda_cols + ae_cols,
    }

    classifiers = build_classifiers(include_gp)
    regressors  = build_regressors(include_gp)

    clf_rows_binary  = []   # Task 1 — binary cluster
    clf_rows_4class  = []   # Task 2 — dominant unit (synthetic only)
    reg_rows         = []   # Task 3 — s21_max_db regression

    lro_binary_rows  = []   # leave-real-out binary
    lro_4class_rows  = []   # leave-real-out 4-class
    lro_reg_rows     = []   # leave-real-out regression

    # ── Task 1 & 3: binary classification + regression on combined data ───────
    print_banner("TASK 1 — Binary cluster classification  (cluster 0 vs 1)")
    log.info("  (cluster 0 = Unit-1/2 insertion-loss style, "
          "cluster 1 = Unit-3/4 style)\n")

    for layer_name, cols in layer_map.items():
        # CV trains and evaluates on synthetic data only
        X = synth_data[cols].values.astype(float)
        y = synth_data["cluster"].values.astype(int)

        X_real = real_data[cols].values.astype(float)
        y_real = real_data["cluster"].values.astype(int)

        for mname, clf in classifiers.items():
            log.info(f"  [{layer_name:3s}] {mname:<14s}")
            fold_rows = cv_classify(mname, clf, X, y)
            for r in fold_rows:
                r["feature_layer"] = layer_name
                r["task"]          = "binary_cluster"
            clf_rows_binary.extend(fold_rows)

            lro = leave_real_out_classify(mname, clf, X, y, X_real, y_real)
            lro["feature_layer"] = layer_name
            lro["task"]          = "binary_cluster"
            lro_binary_rows.append(lro)

            avg_acc = np.mean([r["accuracy"] for r in fold_rows])
            avg_f1  = np.mean([r["f1_macro"] for r in fold_rows])
            lro_acc = lro["accuracy"]
            log.info(f"CV acc={avg_acc:.3f}  f1={avg_f1:.3f}  |  LRO acc={lro_acc:.2f}")

    print_banner("TASK 2 — 4-class dominant-unit classification  (synthetic only)")
    log.info("  (dominant = argmax blend weight; 4 units as named in analysis)\n")

    for layer_name, cols in layer_map.items():
        X = synth_data[cols].values.astype(float)
        y = synth_data["dominant_unit"].values.astype(int)

        for mname, clf in classifiers.items():
            log.info(f"  [{layer_name:3s}] {mname:<14s}")
            fold_rows = cv_classify(mname, clf, X, y, n_splits=5)
            for r in fold_rows:
                r["feature_layer"] = layer_name
                r["task"]          = "dominant_unit_4class"
            clf_rows_4class.extend(fold_rows)

            # leave-real-out using all 4 real units as test (dominant_unit = 1–4 map)
            X_real  = real_data[cols].values.astype(float)
            y_real  = real_data["dominant_unit"].values.astype(int)
            lro = leave_real_out_classify(mname, clf, X, y, X_real, y_real)
            lro["feature_layer"] = layer_name
            lro["task"]          = "dominant_unit_4class"
            lro_4class_rows.append(lro)

            avg_acc = np.mean([r["accuracy"] for r in fold_rows])
            avg_f1  = np.mean([r["f1_macro"] for r in fold_rows])
            lro_acc = lro["accuracy"]
            log.info(f"CV acc={avg_acc:.3f}  f1={avg_f1:.3f}  |  LRO acc={lro_acc:.2f}")

    print_banner("TASK 3 — Regression: s21_max_db (insertion loss peak)")
    log.info("  (s21_max_db excluded from feature sets to prevent target leakage)\n")

    for layer_name, cols in layer_map_reg.items():
        X = synth_data[cols].values.astype(float)
        y = synth_data["s21_max_db"].values.astype(float)

        X_real = real_data[cols].values.astype(float)
        y_real = real_data["s21_max_db"].values.astype(float)

        for mname, reg in regressors.items():
            log.info(f"  [{layer_name:3s}] {mname:<14s}")
            try:
                fold_rows = cv_regress(mname, reg, X, y)
            except Exception as exc:
                log.error("Model %s failed: %s", mname, exc)

                continue
            for r in fold_rows:
                r["feature_layer"] = layer_name
                r["task"]          = "regression_s21_max"
            reg_rows.extend(fold_rows)

            lro = leave_real_out_regress(mname, reg, X, y, X_real, y_real)
            lro["feature_layer"] = layer_name
            lro_reg_rows.append(lro)

            avg_rmse = np.mean([r["rmse"] for r in fold_rows])
            avg_r2   = np.mean([r["r2"]   for r in fold_rows])
            lro_rmse = lro["rmse"]
            log.info(f"CV rmse={avg_rmse:.4f}  r²={avg_r2:.3f}  |  LRO rmse={lro_rmse:.4f}")

    # ── Persist results ───────────────────────────────────────────────────────
    clf_df  = pd.DataFrame(clf_rows_binary + clf_rows_4class)
    reg_df  = pd.DataFrame(reg_rows)
    lro_clf = pd.DataFrame(lro_binary_rows + lro_4class_rows)
    lro_reg = pd.DataFrame(lro_reg_rows)

    clf_df.to_csv(OUT_DIR / "results_classification.csv", index=False)
    reg_df.to_csv(OUT_DIR / "results_regression.csv", index=False)
    lro_clf.to_csv(OUT_DIR / "results_lro_classification.csv", index=False)
    lro_reg.to_csv(OUT_DIR / "results_lro_regression.csv", index=False)

    # ── Summary tables ────────────────────────────────────────────────────────
    print_banner("SUMMARY — Task 1: Binary Cluster  (5-fold CV mean)")
    binary_mean = mean_cv_table(
        clf_df[clf_df["task"] == "binary_cluster"],
        ["accuracy", "f1_macro", "auc_roc", "brier"]
    )
    log.info(binary_mean.sort_values(["feature_layer", "f1_macro"], ascending=[True, False])
                      .to_string(index=False))

    print_banner("SUMMARY — Task 2: Dominant Unit 4-class  (5-fold CV mean)")
    fourclass_mean = mean_cv_table(
        clf_df[clf_df["task"] == "dominant_unit_4class"],
        ["accuracy", "f1_macro"]
    )
    log.info(fourclass_mean.sort_values(["feature_layer", "f1_macro"], ascending=[True, False])
                         .to_string(index=False))

    print_banner("SUMMARY — Task 3: s21_max_db Regression  (5-fold CV mean)")
    reg_mean = mean_cv_table(reg_df, ["rmse", "mae", "r2"])
    log.info(reg_mean.sort_values(["feature_layer", "r2"], ascending=[True, False])
                   .to_string(index=False))

    print_banner("SUMMARY — Leave-Real-Out: Binary Cluster")
    log.info("  (Train: 2 000 synthetic  |  Test: Unit 1 – Unit 4  |  Expected: 4/4 correct)")
    lro_b_bin = lro_clf[lro_clf["task"] == "binary_cluster"].copy()
    log.info(lro_b_bin.sort_values(["feature_layer", "accuracy"], ascending=[True, False])
                    .to_string(index=False))

    print_banner("SUMMARY — Leave-Real-Out: s21_max_db Regression")
    log.info("  RF layer uses 14 correlated features (s21_max_db excluded as target)")
    lro_r = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ("preds", "true")} for r in lro_reg_rows])
    log.info(lro_r.sort_values(["feature_layer", "rmse"]).to_string(index=False))

    # ── Interpretation ────────────────────────────────────────────────────────
    print_banner("INTERPRETATION  (vs. pipeline analysis findings)")
    _interpret(binary_mean, fourclass_mean, reg_mean,
               pd.DataFrame(lro_binary_rows),
               pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ("preds", "true")} for r in lro_reg_rows]))

    print_banner("PLOTS")
    _generate_plots(binary_mean, fourclass_mean, reg_mean,
                    lro_clf, lro_reg_rows, PLOT_DIR)

    log.info(f"\nResults written to {OUT_DIR}/")


def _interpret(binary: pd.DataFrame, fourclass: pd.DataFrame,
               reg: pd.DataFrame,
               lro_b: pd.DataFrame, lro_r: pd.DataFrame) -> None:
    """Print narrative interpretation cross-referencing analysis findings."""

    def best_model(df: pd.DataFrame, metric: str, layer: str) -> str:
        sub = df[df["feature_layer"] == layer]
        if sub.empty:
            return "N/A"
        idx = sub[metric].idxmax() if metric in ("f1_macro", "accuracy", "r2", "auc_roc") \
              else sub[metric].idxmin()
        return sub.loc[idx, "model"]

    def layer_rank(df: pd.DataFrame, metric: str) -> list[str]:
        agg = df.groupby("feature_layer")[metric].mean()
        ascending = metric in ("rmse", "mae")
        return agg.sort_values(ascending=ascending).index.tolist()

    log.info("")
    # Task 1 — binary cluster discrimination
    b_rank = layer_rank(binary, "f1_macro")
    top_m  = best_model(binary, "f1_macro", b_rank[0])
    log.info(f"[Task 1 — Binary Cluster]")
    log.info(f"  Feature layer ranking (F1): {' > '.join(b_rank)}")
    log.info(f"  Best: {top_m} on '{b_rank[0]}' layer")
    if b_rank[0] == "rf":
        log.info("  ✓ Expected: s21_max_db is the exact cluster boundary — RF layer trivially solves this.")
    elif b_rank[0] in ("tda", "all"):
        log.info("  ✓ TDA layer also discriminates: encodes Unit-3/4 topological similarity.")
    ae_f1 = binary[binary["feature_layer"] == "ae"]["f1_macro"].mean()
    rf_f1 = binary[binary["feature_layer"] == "rf"]["f1_macro"].mean()
    if ae_f1 < rf_f1 - 0.05:
        log.info(f"  ✓ AE layer underperforms RF ({ae_f1:.3f} vs {rf_f1:.3f}): "
              "consistent with near-identical AE latents across all units.")
    # LRO
    lro_best = lro_b.loc[lro_b["accuracy"].idxmax()]
    log.info(f"  Leave-real-out best: {lro_best['model']} [{lro_best['feature_layer']}]"
          f"  acc={lro_best['accuracy']:.2f}")
    if lro_best["accuracy"] == 1.0:
        log.info("  ✓ Perfect generalisation to real units — synthetic distribution covers real population.")

    # Task 2 — 4-class unit identification
    fc_rank = layer_rank(fourclass, "f1_macro")
    top_m2  = best_model(fourclass, "f1_macro", fc_rank[0])
    log.info(f"\n[Task 2 — 4-class Dominant Unit]")
    log.info(f"  Feature layer ranking (F1): {' > '.join(fc_rank)}")
    log.info(f"  Best: {top_m2} on '{fc_rank[0]}' layer")
    tda_f1 = fourclass[fourclass["feature_layer"] == "tda"]["f1_macro"].mean()
    rf_f1_4c = fourclass[fourclass["feature_layer"] == "rf"]["f1_macro"].mean()
    if tda_f1 > rf_f1_4c:
        log.info("  ✓ TDA > RF for unit identification: topology encodes fine-grained inter-unit "
              "variation (GNG/PH distances confirm 4-way structure).")
    upper = 1.0 / 4 + 0.05
    if fourclass["f1_macro"].max() < upper + 0.30:
        log.info(f"  ℹ  Inherently hard: dominant unit is a soft argmax label; "
              "Dirichlet α=2 spreads probability mass → boundary ambiguity.")

    # Task 3 — regression
    r_rank = layer_rank(reg, "r2")
    top_m3 = best_model(reg, "r2", r_rank[0])
    log.info(f"\n[Task 3 — s21_max_db Regression]")
    log.info(f"  Feature layer ranking (R²): {' > '.join(r_rank)}")
    log.info(f"  Best: {top_m3} on '{r_rank[0]}' layer")
    rf_r2  = reg[reg["feature_layer"] == "rf"]["r2"].mean()
    tda_r2 = reg[reg["feature_layer"] == "tda"]["r2"].mean()
    if rf_r2 > tda_r2:
        log.info("  ✓ RF > TDA for s21_max regression: insertion loss is a direct S-parameter "
              "scalar, trivially recoverable from RF features without topology.")
    # LRO
    lro_r_best = lro_r.loc[lro_r["rmse"].idxmin()]
    log.info(f"  Leave-real-out best: {lro_r_best['model']} [{lro_r_best['feature_layer']}]"
          f"  rmse={lro_r_best['rmse']:.4f} dB")

    log.info("")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_plots(
    binary_mean: pd.DataFrame,
    fourclass_mean: pd.DataFrame,
    reg_mean: pd.DataFrame,
    lro_clf: pd.DataFrame,
    lro_reg_rows: list[dict],
    plot_dir: Path,
) -> None:
    """Generate and save all model-comparison plots to plot_dir."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    LAYERS       = ["rf", "tda", "ae", "all"]
    LAYER_LABELS = ["RF", "TDA", "AE", "All"]
    LAYER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # ── helpers ──────────────────────────────────────────────────────────────
    def _clf_heatmap(df: pd.DataFrame, metric: str, title: str,
                     fname: str, cmap: str = "YlGn") -> None:
        pivot = df.pivot_table(index="model", columns="feature_layer", values=metric)
        cols  = [c for c in LAYERS if c in pivot.columns]
        idx   = sorted(pivot.index.tolist())
        pivot = pivot.loc[idx, cols]
        fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(idx) * 0.52 + 1.2)))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([LAYER_LABELS[LAYERS.index(c)] for c in cols], fontsize=9)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(idx, fontsize=9)
        for i in range(len(idx)):
            for j in range(len(cols)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if val > 0.82 else "black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Feature Layer", fontsize=9)
        ax.set_ylabel("Model", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  \u2192 {fname}")

    def _lro_bar(df: pd.DataFrame, metric: str, ylabel: str,
                 title: str, fname: str, lower_is_better: bool = False) -> None:
        models          = sorted(df["model"].unique())
        layers_present  = [l for l in LAYERS if l in df["feature_layer"].unique()]
        x     = np.arange(len(layers_present))
        width = 0.78 / len(models)
        fig, ax = plt.subplots(figsize=(max(6.5, len(layers_present) * 1.6), 4.8))
        cmap_m = plt.cm.tab10(np.linspace(0, 0.9, len(models)))
        for i, m in enumerate(models):
            heights = [
                df.loc[(df["feature_layer"] == l) & (df["model"] == m), metric]
                  .values[0] if len(df.loc[(df["feature_layer"] == l) & (df["model"] == m), metric]) > 0
                  else np.nan
                for l in layers_present
            ]
            ax.bar(x + i * width - (len(models) - 1) * width / 2,
                   heights, width=width * 0.88, label=m, color=cmap_m[i], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS[LAYERS.index(l)] for l in layers_present],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8, ncol=min(3, len(models)),
                  loc="upper right" if lower_is_better else "lower right")
        ax.tick_params(axis="y", labelsize=8)
        if not lower_is_better:
            ax.set_ylim(0, 1.10)
            ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7, zorder=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  \u2192 {fname}")

    # ── 1 & 2: classification heatmaps ───────────────────────────────────────
    _clf_heatmap(binary_mean, "f1_macro",
                 "Task 1 \u2014 Binary Cluster: CV F1 macro (5-fold mean)",
                 "task1_cv_f1_heatmap.png")
    _clf_heatmap(fourclass_mean, "f1_macro",
                 "Task 2 \u2014 4-class Unit ID: CV F1 macro (5-fold mean)",
                 "task2_cv_f1_heatmap.png")

    # ── 3: regression R\u00b2 heatmap ──────────────────────────────────────────────
    r2_piv  = reg_mean.pivot_table(index="model", columns="feature_layer", values="r2")
    cols    = [c for c in LAYERS if c in r2_piv.columns]
    idx     = sorted(r2_piv.index.tolist())
    r2_piv  = r2_piv.loc[idx, cols]
    vmin_r2 = min(-0.15, float(np.nanmin(r2_piv.values)) - 0.05)
    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(idx) * 0.52 + 1.2)))
    im = ax.imshow(r2_piv.values, aspect="auto", cmap="RdYlGn",
                   vmin=vmin_r2, vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LAYER_LABELS[LAYERS.index(c)] for c in cols], fontsize=9)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx, fontsize=9)
    for i in range(len(idx)):
        for j in range(len(cols)):
            val = r2_piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > 0.72 or val < -0.05 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Task 3 \u2014 s21_max Regression: CV R\u00b2 (5-fold mean)",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Feature Layer", fontsize=9)
    ax.set_ylabel("Model", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_dir / "task3_cv_r2_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 task3_cv_r2_heatmap.png")

    # ── 4: LRO accuracy bar charts ────────────────────────────────────────────
    lro_bin = lro_clf[lro_clf["task"] == "binary_cluster"].copy()
    lro_4c  = lro_clf[lro_clf["task"] == "dominant_unit_4class"].copy()
    _lro_bar(lro_bin, "accuracy", "LRO Accuracy",
             "LRO \u2014 Binary Cluster (train: synthetic, test: 4 real units)",
             "lro_binary_accuracy_bar.png")
    _lro_bar(lro_4c, "accuracy", "LRO Accuracy",
             "LRO \u2014 4-class Unit ID (train: synthetic, test: 4 real units)",
             "lro_4class_accuracy_bar.png")

    # ── 5: LRO regression RMSE bar chart ────────────────────────────────────
    lro_r = pd.DataFrame([{k: v for k, v in row.items()
                            if k not in ("preds", "true")} for row in lro_reg_rows])
    _lro_bar(lro_r, "rmse", "LRO RMSE (dB)",
             "LRO \u2014 s21_max Regression (train: synthetic, test: 4 real units)",
             "lro_regression_rmse_bar.png", lower_is_better=True)

    # ── 6: LRO regression scatter — predicted vs actual ──────────────────────
    lro_r_full = pd.DataFrame(lro_reg_rows)
    layers_p   = [l for l in LAYERS if l in lro_r_full["feature_layer"].unique()]
    ncols_s    = 2
    nrows_s    = (len(layers_p) + 1) // ncols_s
    fig, axes  = plt.subplots(nrows_s, ncols_s,
                               figsize=(9.0, nrows_s * 4.2), squeeze=False)
    cmap_m = plt.cm.tab10(np.linspace(0, 0.9, lro_r_full["model"].nunique()))
    model_order = sorted(lro_r_full["model"].unique())
    model_color = {m: cmap_m[i] for i, m in enumerate(model_order)}
    # best model per layer (lowest LRO RMSE)
    best_model_per_layer = (
        lro_r.groupby("feature_layer")["rmse"]
              .idxmin()
              .apply(lambda idx_val: lro_r.loc[idx_val, "model"])
    )
    for plot_idx, layer in enumerate(layers_p):
        ax  = axes[plot_idx // ncols_s][plot_idx % ncols_s]
        sub = lro_r_full[lro_r_full["feature_layer"] == layer]
        best_m = best_model_per_layer.get(layer, "")
        all_true, all_pred = [], []
        for _, row in sub.iterrows():
            t = row["true"]  if isinstance(row["true"],  list) else [row["true"]]
            p = row["preds"] if isinstance(row["preds"], list) else [row["preds"]]
            m = row["model"]
            all_true.extend(t)
            all_pred.extend(p)
            ax.scatter(t, p,
                       color=model_color[m], s=30, alpha=0.85, zorder=3,
                       marker="*" if m == best_m else "o",
                       label=f"{m}{'  \u2605best' if m == best_m else ''}")
        mn = min(min(all_true), min(all_pred))
        mx = max(max(all_true), max(all_pred))
        pad = (mx - mn) * 0.10
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                "k--", lw=1.2, alpha=0.45, zorder=2)
        ax.set_xlabel("True s21_max (dB)", fontsize=8)
        ax.set_ylabel("Predicted s21_max (dB)", fontsize=8)
        layer_lbl = LAYER_LABELS[LAYERS.index(layer)] if layer in LAYERS else layer
        ax.set_title(f"LRO Regression \u2014 {layer_lbl} Layer",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linewidth=0.7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    for idx in range(len(layers_p), nrows_s * ncols_s):
        axes[idx // ncols_s][idx % ncols_s].set_visible(False)
    fig.suptitle(
        "LRO Regression: Predicted vs Actual s21_max (dB)\n"
        "(\u2605 = best-RMSE model per layer; dashed = ideal)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "lro_regression_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 lro_regression_scatter.png")

    # ── 7: feature-layer summary — best CV metric across all three tasks ─────
    best_per_layer = {
        "Task 1\nBinary F1"  : binary_mean.groupby("feature_layer")["f1_macro"].max(),
        "Task 2\n4-class F1" : fourclass_mean.groupby("feature_layer")["f1_macro"].max(),
        "Task 3\nR\u00b2"           : reg_mean.groupby("feature_layer")["r2"].max(),
    }
    task_cols = list(best_per_layer.keys())
    x     = np.arange(len(task_cols))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for i, (layer, color, lbl) in enumerate(
            zip(LAYERS, LAYER_COLORS, LAYER_LABELS)):
        vals = [float(best_per_layer[t].get(layer, np.nan)) for t in task_cols]
        ax.bar(x + i * width - (len(LAYERS) - 1) * width / 2,
               vals, width=width * 0.90,
               label=lbl, color=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(task_cols, fontsize=9)
    ax.set_ylabel("Best CV metric  (F1 macro / R\u00b2)", fontsize=9)
    ax.set_ylim(-0.15, 1.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(title="Feature Layer", fontsize=9, title_fontsize=9,
              ncol=4, loc="upper right")
    ax.set_title("Best CV Performance per Feature Layer \u2014 All Tasks",
                 fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.35, linewidth=0.7, zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_dir / "feature_layer_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 feature_layer_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_plots(
    binary_mean: pd.DataFrame,
    fourclass_mean: pd.DataFrame,
    reg_mean: pd.DataFrame,
    lro_clf: pd.DataFrame,
    lro_reg_rows: list[dict],
    plot_dir: Path,
) -> None:
    """Generate and save all model-comparison plots to plot_dir."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    LAYERS       = ["rf", "tda", "ae", "all"]
    LAYER_LABELS = ["RF", "TDA", "AE", "All"]
    LAYER_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # ── helpers ────────────────────────────────────────────────────────────────
    def _clf_heatmap(df: pd.DataFrame, metric: str, title: str,
                     fname: str, cmap: str = "YlGn") -> None:
        pivot = df.pivot_table(index="model", columns="feature_layer", values=metric)
        cols  = [c for c in LAYERS if c in pivot.columns]
        idx   = sorted(pivot.index.tolist())
        pivot = pivot.loc[idx, cols]
        fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(idx) * 0.52 + 1.2)))
        im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([LAYER_LABELS[LAYERS.index(c)] for c in cols], fontsize=9)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels(idx, fontsize=9)
        for i in range(len(idx)):
            for j in range(len(cols)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if val > 0.82 else "black")
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
        cbar.ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Feature Layer", fontsize=9)
        ax.set_ylabel("Model", fontsize=9)
        for sp in ax.spines.values():
            sp.set_visible(False)
        plt.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  \u2192 {fname}")

    def _lro_bar(df: pd.DataFrame, metric: str, ylabel: str,
                 title: str, fname: str, lower_is_better: bool = False) -> None:
        models         = sorted(df["model"].unique())
        layers_present = [l for l in LAYERS if l in df["feature_layer"].unique()]
        x     = np.arange(len(layers_present))
        width = 0.78 / len(models)
        fig, ax = plt.subplots(figsize=(max(6.5, len(layers_present) * 1.6), 4.8))
        cmap_m = plt.cm.tab10(np.linspace(0, 0.9, len(models)))
        for i, m in enumerate(models):
            heights = [
                df.loc[(df["feature_layer"] == l) & (df["model"] == m), metric]
                  .values[0]
                if len(df.loc[(df["feature_layer"] == l) & (df["model"] == m), metric]) > 0
                else np.nan
                for l in layers_present
            ]
            ax.bar(x + i * width - (len(models) - 1) * width / 2,
                   heights, width=width * 0.88, label=m, color=cmap_m[i], zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS[LAYERS.index(l)] for l in layers_present],
                           fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.legend(fontsize=8, ncol=min(3, len(models)),
                  loc="upper right" if lower_is_better else "lower right")
        ax.tick_params(axis="y", labelsize=8)
        if not lower_is_better:
            ax.set_ylim(0, 1.10)
            ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
        ax.grid(axis="y", alpha=0.3, linewidth=0.7, zorder=0)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        plt.tight_layout()
        fig.savefig(plot_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  \u2192 {fname}")

    # ── 1 & 2: classification heatmaps ────────────────────────────────────────
    _clf_heatmap(binary_mean, "f1_macro",
                 "Task 1 \u2014 Binary Cluster: CV F1 macro (5-fold mean)",
                 "task1_cv_f1_heatmap.png")
    _clf_heatmap(fourclass_mean, "f1_macro",
                 "Task 2 \u2014 4-class Unit ID: CV F1 macro (5-fold mean)",
                 "task2_cv_f1_heatmap.png")

    # ── 3: regression R² heatmap ───────────────────────────────────────────────
    r2_piv  = reg_mean.pivot_table(index="model", columns="feature_layer", values="r2")
    cols    = [c for c in LAYERS if c in r2_piv.columns]
    idx     = sorted(r2_piv.index.tolist())
    r2_piv  = r2_piv.loc[idx, cols]
    vmin_r2 = min(-0.15, float(np.nanmin(r2_piv.values)) - 0.05)
    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(idx) * 0.52 + 1.2)))
    im = ax.imshow(r2_piv.values, aspect="auto", cmap="RdYlGn",
                   vmin=vmin_r2, vmax=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([LAYER_LABELS[LAYERS.index(c)] for c in cols], fontsize=9)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels(idx, fontsize=9)
    for i in range(len(idx)):
        for j in range(len(cols)):
            val = r2_piv.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, fontweight="bold",
                        color="white" if val > 0.72 or val < -0.05 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("Task 3 \u2014 s21_max Regression: CV R\u00b2 (5-fold mean)",
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Feature Layer", fontsize=9)
    ax.set_ylabel("Model", fontsize=9)
    for sp in ax.spines.values():
        sp.set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_dir / "task3_cv_r2_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 task3_cv_r2_heatmap.png")

    # ── 4: LRO accuracy bar charts ─────────────────────────────────────────────
    lro_bin = lro_clf[lro_clf["task"] == "binary_cluster"].copy()
    lro_4c  = lro_clf[lro_clf["task"] == "dominant_unit_4class"].copy()
    _lro_bar(lro_bin, "accuracy", "LRO Accuracy",
             "LRO \u2014 Binary Cluster (train: synthetic, test: 4 real units)",
             "lro_binary_accuracy_bar.png")
    _lro_bar(lro_4c, "accuracy", "LRO Accuracy",
             "LRO \u2014 4-class Unit ID (train: synthetic, test: 4 real units)",
             "lro_4class_accuracy_bar.png")

    # ── 5: LRO regression RMSE bar chart ──────────────────────────────────────
    lro_r = pd.DataFrame([{k: v for k, v in row.items()
                            if k not in ("preds", "true")} for row in lro_reg_rows])
    _lro_bar(lro_r, "rmse", "LRO RMSE (dB)",
             "LRO \u2014 s21_max Regression (train: synthetic, test: 4 real units)",
             "lro_regression_rmse_bar.png", lower_is_better=True)

    # ── 6: LRO regression scatter — predicted vs actual ────────────────────────
    lro_r_full = pd.DataFrame(lro_reg_rows)
    layers_p   = [l for l in LAYERS if l in lro_r_full["feature_layer"].unique()]
    ncols_s    = 2
    nrows_s    = (len(layers_p) + 1) // ncols_s
    fig, axes  = plt.subplots(nrows_s, ncols_s,
                               figsize=(9.0, nrows_s * 4.2), squeeze=False)
    cmap_m      = plt.cm.tab10(np.linspace(0, 0.9, lro_r_full["model"].nunique()))
    model_order = sorted(lro_r_full["model"].unique())
    model_color = {m: cmap_m[i] for i, m in enumerate(model_order)}
    best_model_per_layer = (
        lro_r.groupby("feature_layer")["rmse"]
              .idxmin()
              .apply(lambda idx_val: lro_r.loc[idx_val, "model"])
    )
    for plot_idx, layer in enumerate(layers_p):
        ax     = axes[plot_idx // ncols_s][plot_idx % ncols_s]
        sub    = lro_r_full[lro_r_full["feature_layer"] == layer]
        best_m = best_model_per_layer.get(layer, "")
        all_true, all_pred = [], []
        for _, row in sub.iterrows():
            t = row["true"]  if isinstance(row["true"],  list) else [row["true"]]
            p = row["preds"] if isinstance(row["preds"], list) else [row["preds"]]
            m = row["model"]
            all_true.extend(t)
            all_pred.extend(p)
            ax.scatter(t, p,
                       color=model_color[m], s=30, alpha=0.85, zorder=3,
                       marker="*" if m == best_m else "o",
                       label=f"{m}{'  \u2605best' if m == best_m else ''}")
        mn = min(min(all_true), min(all_pred))
        mx = max(max(all_true), max(all_pred))
        pad = (mx - mn) * 0.10
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                "k--", lw=1.2, alpha=0.45, zorder=2)
        ax.set_xlabel("True s21_max (dB)", fontsize=8)
        ax.set_ylabel("Predicted s21_max (dB)", fontsize=8)
        layer_lbl = LAYER_LABELS[LAYERS.index(layer)] if layer in LAYERS else layer
        ax.set_title(f"LRO Regression \u2014 {layer_lbl} Layer",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=7, ncol=2, loc="upper left")
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3, linewidth=0.7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
    for idx in range(len(layers_p), nrows_s * ncols_s):
        axes[idx // ncols_s][idx % ncols_s].set_visible(False)
    fig.suptitle(
        "LRO Regression: Predicted vs Actual s21_max (dB)\n"
        "(\u2605 = best-RMSE model per layer; dashed = ideal)",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(plot_dir / "lro_regression_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 lro_regression_scatter.png")

    # ── 7: feature-layer summary — best CV metric across all three tasks ───────
    best_per_layer = {
        "Task 1\nBinary F1"  : binary_mean.groupby("feature_layer")["f1_macro"].max(),
        "Task 2\n4-class F1" : fourclass_mean.groupby("feature_layer")["f1_macro"].max(),
        "Task 3\nR\u00b2"           : reg_mean.groupby("feature_layer")["r2"].max(),
    }
    task_cols = list(best_per_layer.keys())
    x     = np.arange(len(task_cols))
    width = 0.18
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for i, (layer, color, lbl) in enumerate(
            zip(LAYERS, LAYER_COLORS, LAYER_LABELS)):
        vals = [float(best_per_layer[t].get(layer, np.nan)) for t in task_cols]
        ax.bar(x + i * width - (len(LAYERS) - 1) * width / 2,
               vals, width=width * 0.90,
               label=lbl, color=color, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(task_cols, fontsize=9)
    ax.set_ylabel("Best CV metric  (F1 macro / R\u00b2)", fontsize=9)
    ax.set_ylim(-0.15, 1.12)
    ax.axhline(0, color="black", lw=0.8)
    ax.legend(title="Feature Layer", fontsize=9, title_fontsize=9,
              ncol=4, loc="upper right")
    ax.set_title("Best CV Performance per Feature Layer \u2014 All Tasks",
                 fontsize=10, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.35, linewidth=0.7, zorder=0)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    fig.savefig(plot_dir / "feature_layer_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  \u2192 feature_layer_summary.png")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(message)s",
        stream=sys.stdout,
    )
    parser = argparse.ArgumentParser(
        description="Compare classical ML models on RF synthetic feature data."
    )
    parser.add_argument(
        "--no-gp", dest="no_gp", action="store_true",
        help="Skip slow Gaussian Process models (recommended for n > 1 000)."
    )
    parser.add_argument(
        "--n", type=int, default=None, metavar="N",
        help="Sub-sample synthetic data to N rows for faster experimentation."
    )
    args = parser.parse_args()
    main(include_gp=not args.no_gp, n_subsample=args.n)
