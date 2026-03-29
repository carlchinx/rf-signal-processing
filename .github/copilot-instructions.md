# RF Signal Processing — Epistemic Foundation

This codebase embodies a post-doctoral philosophy of algorithmic inquiry:
**measurement is partial, models are lenses, and understanding emerges from the tension between perspectives.**

## Core Axioms

1. **Uncertainty is primary, not secondary.** Touchstone files encode no noise model.
   Every numerical output inherits epistemic debt. The Bayesian posterior (complex-Gaussian
   with Savitzky–Golay smoothness prior) is our admission of ignorance — never treat
   point estimates as ground truth.

2. **Orthogonality of perspectives.** Three feature layers capture nearly independent
   information about the same physical object:
   - **Scalar** (15 dims): compressed electrical summaries — interpretable, lossy
   - **Topological** (74 dims): structural invariants — regime transitions, cycles, connectivity
   - **Latent** (64 dims): self-supervised nonlinear compression — captures what neither above can name
   
   A method that works on only one layer has explained a projection, not the object.

3. **Topology before parametrics.** Persistent homology is robust to noise and
   invariant to small deformations. Begin with what *persists* across filtration scales;
   parametric models (vector fit, regression) refine after topological structure is established.

4. **Physics constrains generation.** Synthetic data must satisfy: Re(poles) < 0 (stability),
   σ_max(S) ≤ 1 (passivity), causality (analytic in right half-plane). Statistical convenience
   never overrides physical law.

5. **Composition over inheritance.** The pipeline is a DAG of pure-ish functions
   (config → dataframes). Each stage is independently testable. State flows through
   typed containers (S2PBundle, RunConfig, BayesResult), never through globals.

## When Writing Code

- **Frequency is always Hz internally.** Display in GHz only at the plotting boundary.
- **S-parameters are (Nf, 2, 2) complex128.** Posterior samples add a leading axis: (n_draws, Nf, 2, 2).
- **Traces are strings**: `"S11"`, `"S21"`, `"S12"`, `"S22"`, resolved via `TRACE_TO_INDEX`.
- **Graceful degradation**: optional dependencies (ripser, torch) trigger skip-with-warning, never crash.
- **No inheritance** except `nn.Module`. Compose functions; dataclasses carry state.
- **Config is declarative YAML → nested dataclasses.** New hyperparameters go in `config.py` with defaults.

## When Reasoning About Results

Ask three questions before interpreting any output:
1. **What prior is encoded?** (SG window, Dirichlet α, VF pole count — every choice is a claim about reality)
2. **What information was discarded?** (Passband definition at −3 dB, curvature subsampling to 512 pts, HDI at 94%)
3. **Does the conclusion survive a change of lens?** (If TDA says outlier but scalars say normal, investigate — don't average)

## Module Map

```
runner.py          — orchestrator (the only module that knows the full DAG)
io.py              — Touchstone parsing (v1.1/v2.1)
interpolation.py   — PCHIP mag-phase grid unification
metrics.py         — 15 RF scalar metrics
time_domain.py     — IFFT impulse/step response
bayes.py           — complex-Gaussian posterior, HDI, PPC
topology.py        — embeddings, Voronoi, GNG, persistent homology (~700 lines)
vector_fit.py      — rational pole-residue models, synthetic augmentation
ml.py              — autoencoder, inverse regressor (PyTorch)
plotting.py        — 31 plot types
config.py          — 15 dataclasses, ~350 lines
```
