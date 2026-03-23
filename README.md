# RF Signal Processing Pipeline

End-to-end analysis pipeline for 2-port Touchstone S2P measurements. Combines
frequency-domain diagnostics, topological data analysis (TDA), Bayesian
inference, and machine learning for RF bandpass filter characterisation under
limited-data constraints.

---

**Author:** Dr. Charles C. Phiri, CITP, Senior IEEE Member, Fellow (ICTAM)  
**Affiliation:** Independent Researcher / ICTAM Fellow  
**Date:** March 2026  

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).  
You are free to share and adapt this material for any purpose, provided appropriate credit is given, a link to the licence is included, and any changes are indicated.

> **Citation:** If you use this repository, pipeline code, or derived results in your work, please cite as:
>
> Phiri, C. C. (2026). *RF Signal Processing Pipeline: Bandpass Filter Characterisation via TDA, Bayesian Inference, and Machine Learning.*
> GitHub Repository. https://github.com/carlchinx/rf-signal-processing

---

## Architecture

```
rf-signal-processing/
├── pipeline/                       Real-measurement analysis
│   ├── s2p_tda_rtx4070.py          Main pipeline (~3 500 lines)
│   ├── config.yaml                 All hyperparameters
│   ├── requirements.txt            Python dependencies
│   ├── data/                       Four raw S2P measurements
│   ├── outputs/                    All real-unit analysis outputs
│   │   └── s2p_tda_rtx4070/
│   │       ├── metrics/            Per-unit frequency-domain CSVs
│   │       ├── tda/                Voronoi, GNG, PH topology CSVs
│   │       ├── ml/                 Autoencoder + inverse-model artefacts
│   │       ├── bayes/              Credible-band and HDI summaries
│   │       ├── time_domain/        IFFT impulse/step CSVs
│   │       ├── vector_fit/         Rational-model parameters
│   │       └── plots/              Cross-unit comparison figures
│   └── synthetic_data/             Synthetic-data generation and ML benchmarking
│       ├── generate_synthetic.py   Dirichlet-blend S2P generator + feature extractor
│       ├── compare_models.py       Classical ML benchmark across feature layers
│       └── results/                All synthetic-data outputs
│           ├── s2p/                2 000 synthetic Touchstone files
│           ├── synthetic_features.csv
│           ├── results_classification.csv
│           ├── results_regression.csv
│           └── plots/              11 benchmark and EDA figures
└── docs/                           Research reports and supporting material
```

---

## Pipeline Stages

### 1 · Real-Unit Analysis (`s2p_tda_rtx4070.py`)

| Stage | What it does |
|---|---|
| Ingestion | Strict Touchstone 1.1 parsing via scikit-rf; hash-tagged for reproducibility |
| Interpolation | Unifies four files onto a common frequency grid (PCHIP / mag-phase) |
| Frequency metrics | Insertion loss, return loss, passivity, reciprocity diagnostics |
| Time domain | IFFT impulse and step responses with Tukey windowing |
| Vector fitting | Rational pole-residue models; passivity enforcement |
| TDA – Voronoi | 3D Voronoi regime analysis on `(Re S₁₁, Im S₁₁, |S₂₁|)` and shift-register clouds |
| TDA – GNG | Growing Neural Gas state-transition graphs; Bottleneck / Wasserstein PH distances |
| Bayesian | Per-unit credible bands and HDI scalar summaries |
| Autoencoder | Self-supervised window autoencoder on local RF windows |
| Inverse model | Physics-augmented inverse regressor mapping RF+topology → latent system parameters |

### 2 · Synthetic Generation (`generate_synthetic.py`)

Produces 2 000 synthetic S2P files via Dirichlet blending of the four real
units, with physically motivated perturbations (gain shift, phase delay,
frequency stretch, amplitude jitter). Extracts a 153-feature vector per
sample (RF scalar · TDA topology · AE latent).

### 3 · ML Benchmarking (`compare_models.py`)

Evaluates six classical estimators across all feature-layer combinations on
three tasks:

| Task | Type | Target |
|---|---|---|
| 1 | Binary classification | `cluster` (Unit-1/2 vs Unit-3/4 style) |
| 2 | 4-class classification | `dominant_unit` (1–4) |
| 3 | Regression | `s21_max_db` (insertion-loss peak) |

Evaluation: stratified 5-fold CV + leave-real-out generalisation test.

---

## Quick Start

### Prerequisites

- Python 3.10 or later
- CUDA-capable GPU recommended (tested on RTX 4070 laptop)

### Install

```bash
# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

# Install dependencies
pip install -r pipeline/requirements.txt

# Install PyTorch — pick the build that matches your CUDA stack:
# https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Run the main pipeline

```bash
cd pipeline
python s2p_tda_rtx4070.py --config config.yaml
```

Outputs are written to `pipeline/outputs/s2p_tda_rtx4070/`.

### Run the synthetic pipeline

```bash
cd pipeline/synthetic_data

# Step 1: generate synthetic data (reads real-unit artefacts from ../outputs/)
python generate_synthetic.py

# Step 2: benchmark classical ML models
python compare_models.py          # full run
python compare_models.py --no-gp  # skip Gaussian Process (faster)
```

Outputs are written to `pipeline/synthetic_data/results/`.

---

## Configuration

All pipeline hyperparameters are declared in [`pipeline/config.yaml`](pipeline/config.yaml).
Key sections:

| Section | Controls |
|---|---|
| `interpolation` | Frequency-grid unification method and resolution |
| `tda` | Voronoi and GNG topology parameters |
| `vector_fit` | Rational model pole/residue counts |
| `autoencoder` | Window size, latent dimension, training schedule |
| `inverse` | Synthetic augmentation scale and inverse-model training |

---

## Cross-Script Data Flow

```
s2p_tda_rtx4070.py
    └─ writes ──► ml/topology_inverse_features.csv
    └─ writes ──► ml/autoencoder_unit_descriptors.csv
                       │
                       ▼
            generate_synthetic.py
                └─ writes ──► results/synthetic_features.csv
                                    │
                                    ▼
                         compare_models.py
                             └─ writes ──► results/results_*.csv
                             └─ writes ──► results/plots/
```

---

## Notes on Limited-Data ML

The pipeline is designed for the four-file regime, not big data.

- The **autoencoder** learns from many local windows extracted from each trace.
- The **inverse model** augments via physics-based vector-fit perturbations —
  not fantasy labels.
- The **synthetic generator** uses Dirichlet convex blending so all samples
  remain within the observed unit population.
- Leave-real-out evaluation is included to assess generalisation honestly.

---

## Dependencies

| Package | Purpose |
|---|---|
| `scikit-rf` | Touchstone parsing, vector fitting |
| `scipy` | Interpolation, signal processing, spatial structures |
| `numpy`, `pandas` | Numerics and tabular outputs |
| `matplotlib` | All visualisations (headless Agg backend) |
| `scikit-learn` | Classical ML models and cross-validation |
| `PyTorch` | GPU-accelerated autoencoder and inverse regressor |
| `ripser`, `persim` | Persistent homology (optional, auto-detected) |
| `PyYAML` | Configuration loading |

See [`pipeline/requirements.txt`](pipeline/requirements.txt) for pinned versions.

---

## Docs

Extended research and design rationale live in [`docs/`](docs/):

- [`deep-research-report.md`](docs/deep-research-report.md) — comprehensive analysis report
- [`rf_visualization_full_spec.md`](docs/rf_visualization_full_spec.md) — visualisation specification
- [`From Archive to Insight.md`](docs/From%20Archive%20to%20Insight.md) — narrative overview

---

## Reproducibility

All scripts root their paths to `Path(__file__).resolve().parent` — no
hardcoded absolute paths. Set the `seed` key in `config.yaml` (default `12345`)
and `SEED = 2026` in `generate_synthetic.py` to reproduce exact results.
