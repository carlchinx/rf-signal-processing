# RF Signal Processing Pipeline — `pipeline/`

End-to-end real-measurement analysis pipeline for four 2-port Touchstone S2P
files. See the [repository root README](../README.md) for the full project
overview, architecture diagram, and quick-start guide.

---

## Pipeline overview

`s2p_tda_rtx4070.py` runs these stages in sequence:

| Stage | Output location |
|---|---|
| S2P ingestion and interpolation | `ingestion_diagnostics.csv`, `interpolation_summary.csv` |
| Frequency-domain metrics | `metrics/unit_N_frequency_metrics.csv` |
| Time-domain (IFFT) | `time_domain/unit_N_time_domain.csv` |
| Vector fitting | `vector_fit/unit_N_vf_params.npz`, `unit_N_vf_summary.csv` |
| Voronoi topology | `tda/unit_N_{complex,shift}_voronoi_*.csv` |
| Growing Neural Gas | `tda/unit_N_{complex,shift}_gng_*.csv` |
| GNG distance matrices | `tda/gng_distance_{complex,shift}.csv` |
| Persistent homology | `ph/ph_bottleneck_distance_dim1.csv` |
| Bayesian inference | `bayes/unit_N_{credible_bands,hdi_scalar_summary}.csv` |
| Self-supervised autoencoder | `ml/window_autoencoder.pt`, `autoencoder_training_history.csv` |
| Inverse regressor | `ml/inverse_regressor.pt`, `inverse_training_history.csv` |
| Summary metrics | `summary_frequency_metrics.csv`, `run_manifest.json` |
| Cross-unit plots | `plots/*.png` |

---

## Install

```bash
python -m venv ../.venv
..\.venv\Scripts\activate       # Windows
source ../.venv/bin/activate    # Linux / macOS

pip install -r requirements.txt

# Install PyTorch for your CUDA stack — example for CUDA 12.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# Selector: https://pytorch.org/get-started/locally/
```

---

## Run

```bash
python s2p_tda_rtx4070.py --config config.yaml
```

S2P input files are read from `data/`. All outputs are written under
`outputs/s2p_tda_rtx4070/`.

---

## Synthetic data sub-pipeline

After the main pipeline completes, run the two scripts in `synthetic_data/`:

```bash
cd synthetic_data

# Generate 2 000 synthetic S2P files and extract features
python generate_synthetic.py

# Benchmark classical ML models across feature layers
python compare_models.py          # full run (includes Gaussian Process)
python compare_models.py --no-gp  # skip GP models for speed
```

Outputs are written to `synthetic_data/results/`.

---

## Configuration quick reference

Edit `config.yaml` before running. Key knobs:

```yaml
seed: 12345                    # global RNG seed — controls full reproducibility
interpolation:
  n_points: 1530               # unified frequency-grid resolution
tda:
  gng_max_nodes: 48            # topology graph complexity
autoencoder:
  latent_dim: 16               # RF window embedding dimension
inverse:
  n_synthetic_per_unit: 500    # augmentation volume for inverse model
```

---

## Notes on limited-data ML

- The **autoencoder** learns from hundreds of overlapping local windows per
  trace, not from four label-free files directly.
- The **inverse model** trains on physics-augmented synthetic data generated
  by perturbing vector-fitted rational models — never on fantasy labels.
- Evaluate any model with leave-real-out testing before drawing conclusions.
