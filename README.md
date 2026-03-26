# CosmoCluster

A force-based clustering research framework designed around a **cosmology-inspired force schedule**:

1. **Expansion-dominant dynamics** for global separation.
2. **Gravity-dominant dynamics** for local basin formation and hierarchy.

These regimes are connected through a gradual transition, not a discrete phase switch.

The objective is to build a practical, publication-grade clustering system that can be benchmarked against common workflows (e.g., scikit-learn + HDBSCAN), integrated into production pipelines, and evaluated continuously as the codebase evolves.

---

## Why CosmoCluster?

Traditional clustering methods each encode a structural bias (spherical, density-separated, graph-connected, etc.). CosmoCluster explores an alternative view:

- **Clusters as basins of attraction in an interaction field**
- **Clusters as equilibria under repulsion + cohesion**

This framing is especially promising for:

- filamentary or manifold-like structure
- hierarchical clusters
- partially overlapping groups
- embedding-native applications (text/image/scientific vectors)

---

## Core Design: Continuous Expansion → Gravity Transition

CosmoCluster is intentionally built around a hybrid dynamic inspired by cosmology, with a **continuous schedule** rather than two hard-switched stages.

### Early Regime — Expansion-Dominant Dynamics

Goal: create robust global separation and reduce early collapse.

Typical behavior:
- initialize points in latent/normalized space
- apply global repulsive forces
- preserve local geometry through graph constraints (kNN or mutual kNN)
- optionally anneal step size and damping for stability

Outputs:
- separated point configuration
- updated neighborhood structure
- diagnostics for geometric distortion

### Late Regime — Gravity-Dominant Dynamics

Goal: progressively form locally coherent basins and multi-scale cluster structure as attraction increases.

Typical behavior:
- assign mass from local density/centrality
- compute attraction potential on graph neighborhoods
- move points/nodes by local force flow or mode-seeking updates
- converge to attractor basins

Outputs:
- basin assignments
- cluster labels at one or more scales
- optional hierarchy/merge tree

### Transition Schedule (Cosmology-Aligned)

Instead of a discrete handoff, use an annealed force mixture over iteration time \(t\):

- repulsion weight \(w_r(t)\): high at start, decays smoothly
- gravity weight \(w_g(t)\): low at start, rises smoothly
- damping/temperature schedules that stabilize trajectories

Conceptually:

- early iterations prioritize separation and manifold unfolding
- middle iterations balance separation with local condensation
- late iterations prioritize attractor basin settling

This produces smoother dynamics, fewer brittle boundary effects, and a more faithful cosmological analogy than abrupt phase switching.

---

## Algorithm Components (Modular)

The framework should support plug-and-play choices for:

- **Representation**: raw vectors, normalized embeddings, cosine/L2 spaces
- **Graph Construction**: kNN, mutual kNN, adaptive radius
- **Mass / Density**: constant, kNN-density, centrality, learned weights
- **Force Laws**: attraction and repulsion exponents, cutoff radius, damping
- **Dynamics**: continuous updates vs graph-flow updates
- **Cluster Extraction**: basin endpoints, connected components, persistence/stability
- **Hierarchy**: coarse-to-fine merging with stability metrics
- **Field Scheduling**: default cosmology transition, pulsed schedules, or fully custom per-step weighting callbacks

### Dynamic Field Scheduling (New)

`HybridParams` now supports dynamic field control:

- `field_mode="cosmology"` (default): the original smooth expansion→gravity transition.
- `field_mode="pulsed"`: applies one or more square-wave pulses (`FieldPulse`) over normalized phase progress.
- `field_schedule=...`: a custom callable for full per-step control.

This makes it possible to emulate **pulsed field gel electrophoresis–style forcing** where attractive and repulsive components are periodically emphasized to encourage band/cluster focusing.

Example pulsed setup:

```python
from cosmocluster import CosmoClusterMinimal, FieldPulse, HybridParams

params = HybridParams(
    field_mode="pulsed",
    field_pulses=(
        # burst repulsion in early expansion windows
        FieldPulse(start=0.00, end=0.55, period=0.12, duty_cycle=0.35, repulsion_gain=0.75),
        # burst gravity in late settling windows
        FieldPulse(start=0.45, end=1.00, period=0.15, duty_cycle=0.50, gravity_gain=0.90),
    ),
)

model = CosmoClusterMinimal(params).fit(X)
```

Custom schedule example:

```python
from cosmocluster import CosmoClusterMinimal, FieldWeights, HybridParams

def schedule(phase: str, step: int, total_steps: int) -> FieldWeights:
    p = step / max(1, total_steps - 1)
    if phase == "expansion":
        # alternate between unfold and tighten
        on = int(p * 10) % 2 == 0
        return FieldWeights(repulsion=1.2 if on else 0.4, cohesion=0.3 if on else 1.0, gravity=0.0)
    return FieldWeights(repulsion=0.0, cohesion=0.0, gravity=0.4 + 1.1 * p)

params = HybridParams(field_schedule=schedule)
model = CosmoClusterMinimal(params).fit(X)
```

### Mechanistic Explainability First (Physics-Inspired)

CosmoCluster should prioritize **mechanistic explainability** grounded in the model’s force dynamics, not just traditional post-hoc feature attribution.

Recommended mechanistic explainability artifacts per run:

- per-point force decomposition over time: repulsive, attractive, damping components
- trajectory diagnostics: path length, turning behavior, and attractor approach rate
- basin formation provenance: when and why a point entered a basin under evolving force balance
- local energy/potential landscape summaries around emerging attractors
- cluster-level “field cards” describing basin depth, capture radius, stability, and merge history

---

## Early Evaluation-First Development

Evaluation should be implemented **before large-scale algorithm branching**.

### Minimum evaluation loop (required in early iterations)

For every experiment run:

- run CosmoCluster and baselines (at least HDBSCAN + one scikit-learn baseline)
- compute quantitative metrics
- store run configuration + metrics + artifacts
- generate comparable visual summaries

Recommended metrics:

- internal: silhouette, Davies–Bouldin, Calinski–Harabasz
- external (when labels exist): ARI, NMI, macro-F1
- structure-focused: cluster persistence/stability across seeds and hyperparameters
- explainability-focused: force-balance consistency, basin-capture stability, trajectory reproducibility
- runtime + memory (especially for laptop-scale constraints)

---

## Planned Task Ladder (Frequent Evaluation)

Build a sequence of tasks that increases realism while preserving repeatable benchmarking.

### Stage A: Synthetic Data (foundation)

Use controlled datasets to validate dynamics and failure modes:

- moons, circles, blobs (vary noise)
- anisotropic manifolds
- filament + bridge datasets
- hierarchical nested clusters

Purpose:
- verify expansion prevents collapse
- verify gravity recovers local structure
- stress-test hyperparameter sensitivity

Implemented starter suite: `synthetic_suite.py::create_hdbscan_challenge_suite()` with 10 challenge structures (bridges, filaments, gapped rings, spirals, hub-spoke, ladder manifold, bottlenecks, shell-core, hierarchical subclusters).

### Stage B: Topic Modeling / Text Embeddings

Target workflow:

- text embeddings from sentence-transformers or similar
- optional BERTopic-compatible setup for comparison
- compare cluster quality and topic coherence behavior

Evaluate:
- ARI/NMI when labels exist
- topic coherence proxies
- qualitative separability in 2D projections

### Stage C: Image Embeddings

Target workflow:

- CLIP / ViT / CNN-derived embeddings
- evaluate both semantic purity and hierarchical grouping

Evaluate:
- supervised metrics on labeled subsets
- retrieval consistency within discovered clusters
- visual audit using 2D embedding plots + class overlays

---


## Performance Testing Suite

Run the benchmark harness against common clustering algorithms (including HDBSCAN when installed):

```bash
python performance_suite.py --n-samples 1500 --repeats 3 --outdir artifacts/performance
```

Outputs include:
- `benchmark_results.csv`: per-run metrics (ARI, NMI, silhouette, Davies-Bouldin, CH, runtime).
- `benchmark_summary.csv`: aggregate ranking by algorithm.
- `REPORT.md`: markdown report with tables and embedded visualizations.
- `quality_bar.png`, `runtime_boxplot.png`, `ari_heatmap.png`: visual summaries.
- `geometry_diagnostics.html` + `clusters_html/*.html`: interactive geometry and colorized cluster scatter plots per synthetic case.
- These artifacts are generated locally at runtime and are ignored by git.

The suite always runs `CosmoClusterMinimal`, K-Means, DBSCAN, and Agglomerative; it adds HDBSCAN automatically if the `hdbscan` package is available.

## Production Integration Strategy

CosmoCluster should feel natural in existing Python ML stacks.

### API shape (recommended)

- scikit-learn-like estimator interface:
  - `fit(X)`
  - `fit_predict(X)`
  - `predict(X_new)` (optional initially)
  - `get_params()` / `set_params()`
- pipeline-compatible behavior where possible
- deterministic seed controls for reproducibility

### Baseline interoperability

- direct comparison hooks for:
  - `sklearn.cluster` methods
  - `hdbscan.HDBSCAN`
- shared preprocessing wrappers
- shared metric computation utilities

### Packaging and ops

- `pyproject.toml`-based packaging
- clear optional dependency groups (`text`, `vision`, `viz`, `dev`)
- CI checks for basic fit/evaluate runs on synthetic datasets

---

## Experiment Harness (MacBook Air Friendly)

The harness should support broad sweeps while remaining lightweight and reproducible on Apple Silicon laptops.

### Required capabilities

- declarative experiment specs (YAML/JSON/Python config)
- grid/random sweeps over force-law and graph parameters
- repeated seeds with aggregate reporting
- baseline + CosmoCluster side-by-side execution
- artifact tracking (metrics, plots, configs, logs)

### Hardware-aware execution

- optimized for macOS + Apple Silicon (M-series)
- leverage accelerated libraries where available (NumPy/PyTorch/Metal stack when relevant)
- configurable batch sizes / neighbor graph caching
- runtime guardrails for thermal/power constraints

### Tracking and reporting

Recommended stack options:

- **MLflow** or **Weights & Biases** for run tracking
- **Pandas + Plotly** as the default analysis and visualization stack
- optional secondary plotting tools only when a specific figure requires them
- auto-generated experiment reports:
  - metric tables
  - Pareto front (quality vs runtime)
  - sensitivity heatmaps
  - embedding visualizations across phases
  - mechanistic dashboards (force decomposition timelines, trajectory atlases, basin evolution maps)

---

## Publication-Oriented Workflow

If publication is a target, design the repository around reproducible science from day one.

### Reproducibility checklist

- fixed seeds and logged software/hardware metadata
- versioned datasets and split definitions
- locked experiment manifests
- one-command regeneration of key figures/tables

### Suggested paper narrative arc

1. Motivation: limits of current clustering assumptions.
2. Method: expansion→gravity dual dynamics.
3. Benchmarks: synthetic + text + image embeddings.
4. Ablations: force exponents, graph type, mass function, schedules.
5. Practicality: runtime/resource profile on consumer hardware.
6. Discussion: failure cases + future work.

### Target outputs to maintain continuously

- canonical benchmark leaderboard file
- figure gallery for major experiments
- ablation summaries updated each iteration
- mechanistic explainability appendix figures for each benchmark family

---

## Proposed Repository Structure

```text
cosmocluster/
  core/
    expansion.py
    gravity.py
    hybrid.py
    graph.py
    extraction.py
  baselines/
    sklearn_wrappers.py
    hdbscan_wrappers.py
  eval/
    metrics.py
    stability.py
    reports.py
  tasks/
    synthetic/
    topic_modeling/
    image_embeddings/
  experiments/
    configs/
    runners/
    tracking/
    analysis/
  viz/
    phase_plots.py
    benchmark_plots.py
  docs/
    method.md
    benchmarks.md
```

---

## Immediate Next Steps

1. Implement minimal hybrid prototype: expansion phase + gravity phase + basin extraction.
2. Add evaluation harness on synthetic benchmarks with fixed seeds.
3. Integrate HDBSCAN + one sklearn baseline into common runner.
4. Add experiment tracking and auto-generated visual report.
5. Expand to BERTopic-style text workflow, then image embeddings.
6. Freeze a reproducible benchmark suite and begin publication drafting.

---

## Vision

CosmoCluster aims to become both:

- a practical clustering engine for embedding-first ML pipelines, and
- a research platform for discovering multi-scale structure through physically inspired dynamics.

The standard for progress is not just better clusters, but **repeatable evidence**: robust experiments, frequent evaluation, strong visual communication, and clear comparisons against established baselines.

---

## Minimal Hybrid Prototype (Initial Implementation)

A minimal reference implementation now exists in `cosmocluster.py` with three explicit stages:

1. **Expansion phase**: global repulsion + local graph cohesion.
2. **Gravity phase**: mass-weighted attraction on the kNN graph.
3. **Basin extraction**: graph ascent to local mass attractors.

Quick start:

```python
import numpy as np
from cosmocluster import CosmoClusterMinimal, HybridParams

# X: (n_samples, n_features)
X = np.random.randn(300, 8)

model = CosmoClusterMinimal(
    HybridParams(k=12, expansion_steps=50, gravity_steps=50, random_state=42)
)
labels = model.fit_predict(X)

print(labels.shape)
print(np.unique(labels).size, "clusters")
```

This is intentionally simple and CPU-friendly so we can iterate quickly on
force laws, schedules, diagnostics, and extraction strategies.


## Quick Synthetic Parameter Experiment (Cosmology vs Pulsed vs Legacy)

Use `synthetic_experiment.py` to run a **small, fast experiment** that:

1. searches a compact candidate set for `field_mode="cosmology"`
2. searches a compact candidate set for `field_mode="pulsed"`
3. compares the best of both against legacy baselines (`hdbscan` when installed, `kmeans`, `dbscan`, `agglomerative`)

Example:

```bash
python synthetic_experiment.py --n-samples 1200 --noise 0.03 --max-cases 6 --outdir artifacts/small_experiment
```

Key outputs:

- `search_cosmology.csv`
- `search_pulsed.csv`
- `comparison.csv`
- `comparison_summary.csv`
- `recommended_params.json`
- `REPORT.md`

This gives a practical starting point for “reasonable” cosmological and pulsed field parameters before running the full benchmark suite.
