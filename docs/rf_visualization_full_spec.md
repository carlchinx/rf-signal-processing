# Full Visualization and Reporting Specification for the RF S2P Bayesian Frequency–Time–Topology Pipeline

## 1. Purpose

This specification defines the required visualization, topographic analysis, uncertainty reporting, and output styling for a four-unit S2P analysis pipeline. It extends the existing engineering specification by turning frequency-domain, time-domain, complex-plane, shift-register, Voronoi, Growing Neural Gas, and Bayesian outputs into a coherent reporting system.

The baseline engineering specification already requires:

- four S2P inputs;
- canonical internal units;
- frequency-domain metrics including magnitude, dB magnitude, phase, unwrapped phase, group delay, and passivity/reciprocity diagnostics;
- time-domain impulse and step responses using an explicitly defined IFFT convention;
- point-cloud construction from complex-trajectory, magnitude–phase, and sliding-window / Takens-style embeddings;
- Bayesian posterior draws, credible intervals, and posterior predictive checks;
- reproducible figures and machine-readable artifacts.
This document defines how those requirements shall be realized visually and what must be corrected in the current outputs.

## 2. Scope

This specification covers:

1. figure semantics;
2. axis definitions and unit handling;
3. color system;
4. 3D density and topographic surface rendering;
5. Bayesian uncertainty overlays and credible surfaces;
6. cross-unit comparability rules;
7. content-specific plot classes;
8. acceptance criteria;
9. output file naming and packaging.

It does not replace the existing computational specification. It constrains the visualization and reporting layer that sits on top of it.

## 3. Binding principles

### 3.1 Scientific traceability

Every figure shall be derivable from a declared data product and configuration snapshot. The existing pipeline already requires reproducible artifacts, machine-readable outputs, and a run manifest with seeds, versions, and hashes.

### 3.2 Canonical units

Internal canonical units are:

- frequency: Hz;
- time: seconds;
- S-parameters: dimensionless complex ratios;
- phase: radians internally, degrees only for I/O display;
- magnitude in dB: float64 display quantity.

Any plot shall either use these physical units directly or explicitly label the transformed coordinates.

### 3.3 Explicit uncertainty

The current engineering specification requires Bayesian credible intervals for derived metrics, posterior predictive checks, and Monte Carlo propagation of uncertainty into topology. Every figure with posterior support shall include at least one uncertainty artifact, such as a credible band, HDI shell, or uncertainty-thickness map.

No final visualization may suppress uncertainty when posterior draws exist. Uncertainty is a first-class output, not an optional add-on.

## 4. Current-output gap analysis

The provided outputs reveal the following defects.

### 4.1 Mislabelled transformed axes

Several plots labelled with physical quantities such as `|S21(i)|` or `|S21(i+lag)|` include negative values on one or both axes. A magnitude axis cannot be negative. This indicates one of two failures:

1. the plotted coordinates are standardized or otherwise transformed but labelled as physical magnitudes; or
2. the magnitude field itself is corrupted before plotting.

This is a hard scientific error. It shall be corrected before further reporting.

### 4.2 Scatter plots misrepresented as density maps

Plots titled “3D Density Map” currently use sparse marker clouds whose marker size and color encode a density-like quantity, but the vertical axis is often a physical metric such as `|S21|` or `Group delay`. Those are not density surfaces. They are weighted scatter plots.

The specification below separates three plot types that are currently conflated:

- physical manifold scatter;
- density surface;
- regime-intensity surface.

### 4.3 Unbounded regime-intensity scales

The current regime-intensity colorbars span extreme values, including values of order `1e8`, which destroys interpretability and makes cross-unit comparison impossible. This indicates that the regime-intensity field is either multiplicative without normalization or dominated by a singular factor.

### 4.4 Occupancy spikes flatten the rest of the surface

The complex-plane occupancy surfaces show a single giant spike and a nearly flat remainder. That means the current count surface is using a linear scale on a distribution with a severe concentration imbalance.

### 4.5 Inconsistent lag across units without explicit comparability policy

The shift-register topographic outputs appear to use different lags for different units. Variable lag can be scientifically valid, but only if it is documented and the comparison is clearly framed as unit-specific embedding. Direct cross-unit geometric comparison is invalid if the lags differ and the viewer is not told that.

### 4.6 Wasted canvas and weak viewpoint control

The `Unit × Frequency × |S21| dB` surface uses only a small fraction of the frame, with excessive unused whitespace. The view angle also compresses the y-axis categories, reducing readability.

### 4.7 Inconsistent color semantics

The current outputs mix sequential colormaps for different semantics without a declared color grammar. Density, regime intensity, counts, and physical magnitude need different normalization rules even if some share a sequential palette family.

## 5. Visualization data contracts

Every figure shall declare one of the following contracts.

### 5.1 Contract A: Physical manifold scatter

A physical manifold scatter plot represents samples in a physical or transformed embedding:

- x-axis: coordinate 1;
- y-axis: coordinate 2;
- z-axis: coordinate 3;
- color: auxiliary scalar;
- marker size: optional uncertainty or occupancy.

Examples:

- `(Re(S11), Im(S11), |S21|)`;
- `(|S21(i)|, |S21(i+lag)|, group_delay)`;
- `(Re(S21), Im(S21), frequency_GHz)`.

This plot shall not be labelled as a density map.

### 5.2 Contract B: Density surface

A density surface represents estimated occupancy or probability density over a chosen 2D support:

- x-axis: embedding coordinate 1;
- y-axis: embedding coordinate 2;
- z-axis: estimated density, count, or posterior mean occupancy;
- color: same scalar as z unless a different explicitly stated field is used.

Examples:

- `(Re(S21), Im(S21)) → count surface`;
- `(|S21(i)|, |S21(i+lag)|) → KDE density surface`.

### 5.3 Contract C: Regime-intensity surface

A regime-intensity surface represents a bounded state-saliency field over a chosen support:

- x-axis: embedding coordinate 1;
- y-axis: embedding coordinate 2;
- z-axis: regime intensity in `[0,1]` or a documented bounded score;
- color: same field or its uncertainty thickness.

It shall not use arbitrary unbounded scales.

### 5.4 Contract D: Bayesian credible surface

A credible surface represents posterior uncertainty of a scalar field:

- median surface: `z50(x,y)`;
- lower credible surface: `zL(x,y)`;
- upper credible surface: `zU(x,y)`;
- uncertainty thickness: `Δz(x,y) = zU - zL`.

Every surface with posterior support shall be able to emit this contract.

## 6. Mathematical definitions

### 6.1 Frequency-domain field

For unit `u` and frequency index `k`:

- complex S-parameters: `Sij_u(f_k)`;
- magnitude field: `M21_u(f_k) = |S21_u(f_k)|`;
- dB field: `D21_u(f_k) = 20 log10 |S21_u(f_k)|`;
- phase field: `Φ21_u(f_k) = unwrap(arg(S21_u(f_k)))`;
- group delay field:
  `τg_u(f_k) = -(1 / 2π) dΦ21_u / df`.

These are already mandated in the baseline spec.

### 6.2 Embeddings

#### Complex trajectory embedding

A default complex trajectory embedding is

`x_k = [Re(S11(f_k)), Im(S11(f_k)), Re(S21(f_k)), Im(S21(f_k))]`.

The original engineering spec defines this class of embeddings and requires explicit normalization metadata.

#### Magnitude–phase embedding

`x_k = [|S21(f_k)|_dB, angle(S21(f_k))]`.

#### Sliding-window / shift-register embedding

For scalar signal `y_k` and lag `ℓ`, order `m`:

`x_k = [y_k, y_{k+ℓ}, …, y_{k+(m-1)ℓ}]`.

The baseline engineering spec explicitly includes this embedding because it reveals cyclic structure not visible in raw one-dimensional series.

### 6.3 Density estimation

For posterior draw `m`, embedding points `ξ_i^(m) ∈ R^2`, define KDE density

`ρ^(m)(u,v) = (1/Nh_u h_v) Σ_i K((u-u_i^(m))/h_u) K((v-v_i^(m))/h_v)`.

Discrete occupancy may alternatively be used:

`C_ab^(m) = Σ_i 1{ ξ_i^(m) ∈ B_ab }`.

Density surfaces shall use either KDE or adaptive-binned occupancy, but the chosen estimator must be declared.

### 6.4 Regime intensity

The current regime-intensity output shall be replaced with a bounded field. Define per-sample regime score

`r_i = sigmoid( α κ_i + β |dτg/df|_i + γ Δrec_i + δ η_i )`,

where:

- `κ_i` = local trajectory curvature or shift-register turning score;
- `Δrec_i = |S21_i - S12_i|` is reciprocity residual;
- `η_i` = posterior uncertainty or passivity-margin anomaly;
- `α, β, γ, δ` are explicit weights in configuration.

Then define the regime-intensity surface over support `(u,v)` as either

`R(u,v) = E_m[ Σ_i r_i^(m) K_h((u,v)-ξ_i^(m)) ]`

or its occupancy-weighted average.

`R(u,v)` shall be normalized to `[0,1]` or to a documented percentile-clipped range.

### 6.5 Bayesian credible surfaces

For any scalar surface `Z^(m)(u,v)` computed per posterior draw:

- median surface: `Z50(u,v) = median_m Z^(m)(u,v)`;
- lower credible surface: `ZL(u,v) = HDI_low_m Z^(m)(u,v)`;
- upper credible surface: `ZU(u,v) = HDI_high_m Z^(m)(u,v)`.

The baseline spec already requires HDI-based scalar and functional reporting, including plot captions with interval type, credibility mass, number of posterior draws, and pointwise vs simultaneous semantics.

## 7. Normative plot families

### 7.1 Frequency-domain family

#### F1. Unit × Frequency × `|S21| dB` surface

Purpose: compare passband shape across units.

Axes:

- x = frequency in GHz;
- y = unit index 1–4;
- z = posterior median `|S21| dB`.

Rendering:

- full-frame surface or four aligned ribbons;
- optional translucent lower and upper HDI curtains;
- shared sequential colormap;
- no excess whitespace.

Requirements:

- fixed frequency range across all units;
- fixed z-range across all units;
- y labels `Unit 1 … Unit 4`;
- camera angle fixed for every rerun.

#### F2. Frequency credible-band small multiples

One panel per unit for:

- `S11_dB`;
- `S21_dB`;
- `phase(S21)`;
- `group_delay(S21)`.

These are already part of the baseline minimum figure set and shall remain.

### 7.2 Complex-plane family

#### C1. Physical manifold scatter

Axes:

- x = `Re(S11)` or `Re(S21)`;
- y = `Im(S11)` or `Im(S21)`;
- z = `|S21|`, `frequency_GHz`, or `group_delay_ns`.

Color:

- frequency progression for trajectory understanding, or
- regime score if frequency is shown elsewhere.

Requirements:

- if standardized coordinates are used, axes shall be labelled `z(Re(S11))`, `z(Im(S11))`, etc.;
- if physical values are used, ranges shall be physically plausible and consistent with labels.

#### C2. Complex-plane occupancy surface

Axes:

- x = `Re(S21)`;
- y = `Im(S21)`;
- z = `log1p(count)` or posterior mean density.

Requirements:

- this is the correct replacement for the current “Complex-Plane Occupancy 3D Density” spike plots;
- use adaptive bins or KDE;
- use log scale on z to prevent one spike from flattening the map;
- overlay the median complex trajectory as a thin line.

#### C3. Complex-plane regime-intensity surface

Axes:

- x = `Re(S11)`;
- y = `Im(S11)`;
- z = regime intensity `R(u,v)`.

Requirements:

- `R` must be bounded;
- HDI shell optional but strongly recommended.

### 7.3 Shift-register family

#### S1. Shift-register phase-space scatter

Axes:

- x = `|S21(i)|` or transformed equivalent;
- y = `|S21(i+ℓ)|`;
- z = `group_delay`, `frequency`, or local regime score.

Requirements:

- direct cross-unit comparison requires a common lag `ℓ`;
- if unit-specific lag is used, the lag must appear in filename, title, metadata, and comparison caption.

#### S2. Shift-register density surface

Axes:

- x = `|S21(i)|`;
- y = `|S21(i+ℓ)|`;
- z = `log1p(count)` or posterior mean KDE density.

Requirements:

- magnitude axes shall be nonnegative if labelled as magnitudes;
- otherwise transformed axes must be explicitly labelled.

#### S3. Shift-register regime-intensity surface

Axes:

- x = `|S21(i)|`;
- y = `|S21(i+ℓ)|`;
- z = bounded regime intensity.

Requirements:

- this replaces the current unbounded `1e8` regime-intensity visuals.

### 7.4 Voronoi and GNG family

#### V1. Voronoi descriptor surface

Axes:

- x = embedding coordinate 1;
- y = embedding coordinate 2;
- z = Voronoi cell volume, inverse volume, or anisotropy.

Requirements:

- use bounded regions only unless a boundary completion policy is explicitly declared;
- uncertainty shown as HDI shell if computed from posterior draws.

#### G1. GNG state graph plot

Required panels:

- 3D node cloud in embedding space;
- edge network;
- node occupancy color;
- node reconstruction error size;
- frequency-ordered transition matrix as companion heatmap or surface.

### 7.5 Bayesian uncertainty family

#### U1. Credible surface overlay

For any density or regime surface with posterior support:

- render median surface opaque;
- render lower and upper HDI surfaces semi-transparent;
- render uncertainty-thickness map as companion panel.

#### U2. Uncertainty topomap

For any support `(u,v)` define

`U(u,v) = ZU(u,v) - ZL(u,v)`.

Render `U` as a separate 3D surface or 2D contour to identify unreliable regions.

## 8. Visual design system

### 8.1 Background and typography

- background: white or near-white for publication mode;
- grid lines: light neutral gray;
- titles: sentence case, short, no internal code names such as `H1`, `H2`, `H3` unless the report defines them once;
- font family: sans serif with consistent size ladder.

### 8.2 Color grammar

#### Sequential fields

Use one sequential family for:

- density;
- occupancy;
- `|S21| dB` surfaces;
- regime intensity if represented as a positive scalar.

Recommended default: a perceptually ordered sequential palette such as `cividis` or `viridis`.

#### Signed fields

Use a diverging palette for:

- centered group-delay anomaly;
- signed phase residual;
- standardized coordinates if visualized directly.

#### Cyclic fields

Use a cyclic palette only for phase when the wrap is intentional.

### 8.3 Color normalization rules

- density / occupancy: `log1p` normalization by default;
- regime intensity: clipped to `[0,1]` or to posterior `2nd–98th` percentile range;
- cross-unit plots: shared normalization across all units;
- single-unit exploratory plots: local normalization allowed, but only when marked as exploratory.

### 8.4 Marker, surface, and line priorities

- trajectories: thin solid line;
- posterior samples: low-alpha points;
- median surfaces: medium opacity;
- HDI shells: low opacity;
- outliers: distinct marker shape, not only color.

### 8.5 View-angle policy

Each 3D plot class shall have a fixed default view:

- unit×frequency surfaces: low oblique view with unit axis readable;
- complex-plane occupancy: top-oblique view emphasizing z relief;
- shift-register density: slightly higher elevation to separate near-origin spike from distal regimes.

The azimuth and elevation must be stored in configuration.

## 9. Cross-unit comparability rules

1. same axis ranges for the same plot family;
2. same colormap and normalization for the same plot family;
3. same camera angle for the same plot family;
4. same lag for direct shift-register comparison, unless explicitly comparing adaptive-lag embeddings;
5. same smoothing bandwidth policy across units.

The baseline engineering spec already requires that all threshold and resolution choices be explicit configuration parameters, not hidden constants. fileciteturn2file6

## 10. Bayesian uncertainty requirements

The following are mandatory when posterior draws exist.

### 10.1 Frequency-domain figures

- median curve;
- pointwise HDI band;
- optional simultaneous band.

### 10.2 Time-domain figures

- median impulse or step response;
- pointwise HDI band;
- explicit note of IFFT convention and DC handling, which the baseline spec already requires.

### 10.3 Density and regime surfaces

- posterior median surface;
- lower and upper HDI surfaces;
- uncertainty-thickness companion plot.

### 10.4 Topological summaries

- posterior distributions of topological summary metrics;
- posterior probabilities for between-unit differences above practical thresholds, consistent with the baseline Bayesian comparison policy.

## 11. Data integrity and labeling rules

### 11.1 Label semantics

A label shall match the data exactly.

Examples:

- if plotted x-values are z-scored `Re(S11)`, label must be `z(Re(S11))`, not `Re(S11)`;
- if plotted y-values are standardized `|S21(i+ℓ)|`, label must say `z(|S21(i+ℓ)|)`;
- if plotted values are physical magnitudes, they must be nonnegative.

### 11.2 Unit labeling

All public visuals shall use `Unit 1` through `Unit 4`. The previous file names remain internal provenance only.

### 11.3 Metadata in captions

Each figure caption shall include:

- embedding type;
- posterior draw count if applicable;
- credibility mass;
- interval type;
- lag and window order for shift-register embeddings;
- whether axes are physical or normalized.

The baseline spec already requires credible-band metadata in legends/captions.

## 12. Output file naming

### 12.1 Required naming pattern

`{unit}_{family}_{metric}_{support}_{statistic}_{version}.{ext}`

Examples:

- `unit_1_freq_S21dB_frequency_median_v1.png`
- `unit_1_complex_density_res21_im21_median_v1.png`
- `unit_1_complex_density_res21_im21_hdi95_v1.png`
- `unit_1_shift_density_mag_maglag12_median_v1.png`
- `unit_1_shift_regime_mag_maglag12_median_v1.png`
- `all_units_freq_S21dB_surface_median_v1.png`

### 12.2 Companion machine-readable outputs

Every figure shall have a paired data file:

- CSV, NetCDF, or Parquet for plotted values;
- JSON metadata sidecar with:
  - config hash;
  - lag;
  - normalization;
  - smoothing bandwidth;
  - camera angle;
  - color normalization range.

This is consistent with the baseline requirement for machine-readable artifacts plus figures and manifests.

## 13. Acceptance criteria

A visualization build passes only if all criteria below hold.

### 13.1 Scientific correctness

- No magnitude-labelled axis contains negative values.
- Every transformed axis is explicitly labelled as transformed.
- Every density map has density or count on the z-axis.
- Every regime-intensity map uses bounded or percentile-clipped values.

### 13.2 Cross-unit comparability

- Units 1–4 share axis ranges, color scales, and view angles within each plot family.
- Variable-lag shift-register plots are marked as unit-specific and not used in direct numeric comparison panels.

### 13.3 Uncertainty visibility

- If posterior draws exist, at least one uncertainty artifact accompanies each major figure family.
- Credibility mass and interval type are printed in the figure caption or legend. This is already a normative requirement in the baseline spec.

### 13.4 Rendering quality

- No plot wastes more than 25% of the canvas in blank margins.
- Font sizes are legible at 1600×900 and in print at 300 dpi.
- Colorbars always have units or semantic labels.

## 14. Prioritized remediation plan

### Phase 1 — mandatory corrections

1. Fix all axis-label/data mismatches.
2. Replace weighted scatter plots titled as density maps with actual density or count surfaces.
3. Replace unbounded regime-intensity scales with normalized bounded fields.
4. Apply `Unit 1`–`Unit 4` consistently.
5. Re-render `Unit × Frequency × |S21| dB` to occupy the full frame.

### Phase 2 — uncertainty-aware visuals

1. Add posterior median + HDI surfaces for all density/regime maps.
2. Add uncertainty-thickness surfaces.
3. Add posterior probability surfaces for occupancy above threshold.

### Phase 3 — topology and state structure visuals

1. Add Voronoi descriptor surfaces.
2. Add GNG state-graph visuals.
3. Add transition-probability surfaces for shift-register regimes.

### Phase 4 — publication mode

1. unify color grammar;
2. enforce style templates;
3. export PNG, SVG, and PDF;
4. emit caption-ready metadata blocks.

## 15. Minimum revised figure set

The revised minimum figure set shall be:

1. `all_units_freq_S21dB_surface_median`
2. `all_units_freq_S21dB_surface_hdi`
3. `unit_k_complex_manifold_scatter`
4. `unit_k_complex_density_surface`
5. `unit_k_complex_regime_surface`
6. `unit_k_shift_phase_space_scatter`
7. `unit_k_shift_density_surface`
8. `unit_k_shift_regime_surface`
9. `unit_k_density_uncertainty_surface`
10. `unit_k_regime_uncertainty_surface`
11. `unit_k_voronoi_volume_surface`
12. `unit_k_gng_state_graph`
13. `all_units_topology_distance_surface`

## 16. Conformance to the baseline engineering specification

This visualization specification is intentionally consistent with the baseline uploaded engineering specification in the following ways:

- It preserves the four-file S2P scope and canonical units.
- It preserves the core embedding classes: complex trajectory, magnitude–phase, and sliding-window / Takens.
- It extends Bayesian credible-interval reporting from curves to 3D density and topographic surfaces.
- It preserves reproducible artifacts, figure outputs, and manifest linkage.
- It strengthens the report builder role already defined in the architecture.

## 17. Final directive

The current plotting layer shall be treated as exploratory only. It shall not be used for final scientific reporting until:

- density semantics are corrected;
- axis labels match the plotted data exactly;
- regime intensity is normalized;
- uncertainty surfaces are added; and
- cross-unit comparability is enforced.

