# causasv — Causal Feature Attribution via Asymmetric Shapley Values

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![PyPI](https://img.shields.io/pypi/v/causasv.svg)](https://pypi.org/project/causasv/)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
<br>
[![MSRV](https://img.shields.io/badge/MSRV-1.85%2B-orange.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)

**English** | [日本語](README_ja.md) | [中文](README_zh.md)

`causasv` computes **Asymmetric Shapley Values (ASV)** for causal feature attribution over user-supplied DAGs. It is a Rust-first engine with Python bindings, designed for XAI workflows where feature importance should respect known causal structure.

## When to use causasv

**Use causasv when:**
- you have a known causal DAG among your features
- standard SHAP may assign credit through descendants or mediators
- you need exact or uncertainty-aware approximate ASV
- you want a fast Rust core with Python bindings and CI bounds on estimates

**Do not use causasv when:**
- you do not have a causal DAG (→ use SHAP or Captum)
- you need generic model explainability without causal structure
- you need deep learning layer or neuron attribution
- you need causal effect estimation or discovery itself (→ use DoWhy)

## What is ASV?

Asymmetric Shapley Values (ASV) generalize Shapley values by averaging only over **topologically valid orderings** of features, rather than all permutations. Given a causal DAG G and a value function v:

```
φ_i = (1 / |Π(G)|) Σ_{π ∈ Π(G)} [v(pre(i,π) ∪ {i}) − v(pre(i,π))]
```

where Π(G) is the set of all linear extensions (topological orderings) of G, and pre(i,π) is the set of features appearing before feature i in ordering π.

## How ASV differs from SHAP

Standard SHAP averages over all n! feature permutations, ignoring causal structure. ASV restricts the average to permutations consistent with the causal DAG — causes always appear before their effects. This produces attributions that respect the direction of causality.

## Why causal DAGs matter

When features have causal relationships, SHAP can assign attribution to a variable for effects that are actually mediated by its descendants. ASV prevents this by constraining which orderings are considered valid.

## Installation

```bash
pip install causasv
```

Wheels for Linux (x86_64 manylinux), macOS (universal2), and Windows (x86_64) are published on [PyPI](https://pypi.org/project/causasv/). For Rust, add to `Cargo.toml`:

```toml
[dependencies]
causasv = "0.8"
```

## Rust example

```rust
use causasv::{AsvExplainer, Dag, SamplingConfig};

fn main() -> Result<(), causasv::CausasvError> {
    let mut dag = Dag::new();
    let education = dag.add_node("education");
    let income = dag.add_node("income");
    let risk = dag.add_node("risk_score");
    dag.add_edge(education, income)?;
    dag.add_edge(income, risk)?;
    dag.validate()?;

    let explainer = AsvExplainer::new(dag);

    // Approximate ASV via importance-weighted topological order sampling.
    let values = explainer.approximate(
        |coalition| {
            // User-supplied value function: score given a coalition of features.
            Ok(coalition.len() as f64)
        },
        SamplingConfig::new(10_000).with_seed(42),
    )?;

    for (node, value) in &values.values {
        println!("Node {:?}: ASV = {:.4}", node, value);
    }
    Ok(())
}
```

## Python example

For most users, `explain_quality()` is the recommended entry point — it selects exact computation when feasible and falls back to uncertainty-aware approximate sampling with confidence intervals:

```python
from causasv import CausalDAG, ASVExplainer, explain_quality

dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])
explainer = ASVExplainer(dag)

info = explain_quality(
    explainer,
    value_fn=lambda feature_names: my_model_score(feature_names),
    ci=0.95,   # include 95% confidence intervals
    seed=42,
)
print(info["values"])           # dict[str, float] — ASV per feature
print(info["ci_low"])           # dict[str, float] — 95% CI lower bounds
print(info["ci_high"])          # dict[str, float] — 95% CI upper bounds
print(info["selected_method"])  # e.g. "exact_dag_sparse", "uniform_sparse_adaptive", or "uniform_sparse_adaptive_batch"
print(info["stderr"])           # dict[str, float] — per-feature standard error
```

The Python `value_fn` receives a sorted list of feature names present in the coalition and must return a float.

**Lower-level API** — for explicit method control, use `explain_with_diagnostics()`:

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=10_000,
    seed=42,
)
print(info["values"])     # dict[str, float]
print(info["ess"])        # float — ESS ≈ n_samples means reliable; ESS ≪ n_samples means high variance
print(info["ess_ratio"])  # float — ESS / n_samples ∈ (0, 1]; close to 1 is good
print(info["n_samples"])  # int
print(info["seed"])       # int | None
print(info["is_exact"])   # bool
print(info["method"])     # str — the method name passed in (e.g. "approx")
```

Use `explain_adaptive()` for automatic convergence detection and per-feature confidence intervals:

```python
info = explainer.explain_adaptive(
    value_fn=lambda feature_names: my_model_score(feature_names),
    min_samples=1_000,
    max_samples=100_000,
    batch_size=1_000,
    seed=42,
    ci=0.95,          # optional: add ci_low / ci_high to result
)
print(info["values"])     # dict[str, float]
print(info["stderr"])     # dict[str, float] — IS standard error per feature
print(info["ci_low"])     # dict[str, float] — lower bound of 95% confidence interval
print(info["ci_high"])    # dict[str, float] — upper bound of 95% confidence interval
print(info["converged"])  # bool — True if rel_tol was reached before max_samples
print(info["ess_ratio"])  # float — ESS / n_samples; close to 1 is good
```

For large models where calling the value function once per coalition is slow, pass `value_fn_batch` to `explain_quality()`. This routes to uniform sparse adaptive batch sampling (ESS = n_samples, no IS variance, CI always returned) for n ≤ 63:

```python
# value_fn_batch receives list[list[str]] and must return list[float]
info = explain_quality(
    explainer,
    value_fn_batch=lambda coalitions: [my_model_score(c) for c in coalitions],
    ci=0.95,
    seed=42,
)
print(info["values"])           # dict[str, float]
print(info["ci_low"])           # dict[str, float]
print(info["selected_method"])  # "uniform_sparse_adaptive_batch" (or "approx_adaptive_batch" for n>63)
```

The batched path reduces Python GIL round-trips from O(n × batch_size) to O(unique_masks_per_batch). For explicit IS-adaptive batched evaluation, use `explainer.explain_adaptive_batch()` directly.

For deterministic parallel approximation, pass `parallel=True` with a `seed`:

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=100_000,
    seed=42,
    parallel=True,
    num_threads=4,   # None = rayon default
)
print(info["deterministic"])  # True when seed + parallel
```

Use `explain_stability()` to verify that approximate rankings are consistent across seeds:

```python
from causasv import explain_stability

result = explain_stability(
    explainer,
    value_fn=lambda feature_names: my_model_score(feature_names),
    seeds=[1, 2, 3, 4, 5],
    method="approx",
    n_samples=10_000,
)
print(result["rank_stability"])   # mean pairwise Kendall tau; 1.0 = perfectly stable
print(result["std_values"])       # dict[str, float] — small means stable estimates
print(result["mean_values"])      # dict[str, float] — mean ASV across seeds
```

Use `explain_safe()` to apply the diagnostics checklist above automatically instead of
checking `ess_ratio`/`rank_stability`/CI bounds by hand:

```python
from causasv import explain_safe

info = explain_safe(
    explainer,
    value_fn=lambda feature_names: my_model_score(feature_names),
    ci=0.95,
    seed=42,
)
print(info["warnings"])           # list[str] — e.g. low ess_ratio or low rank_stability
print(info["rank_stability"])     # float | None — None when the result is already exact
print(info["unstable_features"])  # list[str] — features whose CI still straddles 0
```

Use `ASVEnsembleExplainer` to measure sensitivity across multiple candidate DAGs:

```python
from causasv import CausalDAG, ASVEnsembleExplainer

dag1 = CausalDAG.from_edges([("A", "B"), ("B", "C")])
dag2 = CausalDAG.from_edges([("A", "B"), ("A", "C")])
ensemble = ASVEnsembleExplainer([dag1, dag2])
result = ensemble.explain_with_sensitivity(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="auto",
)
print(result["mean_values"])     # dict[str, float] — mean ASV across DAGs
print(result["std_values"])      # dict[str, float] — std across DAGs; 0 = DAG-invariant
print(result["rank_stability"])  # float — mean pairwise Kendall tau across DAG pairs
print(result["per_dag_values"])  # list[dict[str, float]] — one dict per DAG
```

Inspect and export the DAG:

```python
dag.nodes()                     # ["education", "income", "risk_score"]
dag.edges()                     # [("education", "income"), ("income", "risk_score")]
dag.to_dot()                    # 'digraph {\n  education -> income;\n  ...\n}'
dag.to_json()                   # '{"nodes":[...],"edges":[...]}'
dag.ancestors("risk_score")     # ["education", "income"]
dag.descendants("education")    # ["income", "risk_score"]
dag.topological_layers()        # [["education"], ["income"], ["risk_score"]]

# Restore a DAG from JSON
dag2 = CausalDAG.from_json(dag.to_json())

# Convert to networkx (networkx must be installed separately)
import networkx as nx
G = nx.DiGraph(dag.edges())
```

Use `TabularExplainer` for a higher-level API with sklearn-compatible models (requires numpy):

```python
from causasv import CausalDAG, TabularExplainer

dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])

explainer = TabularExplainer.from_model(
    model=my_classifier,      # any sklearn-compatible model
    dag=dag,
    background=X_train,       # reference dataset; absent features filled per `baseline`
    feature_names=["education", "income", "risk_score"],
    baseline="mean",          # "mean" | "median" | "sample" | "background_expectation" | callable
)
values = explainer.explain_instance(X_test[0], method="auto")
# values: dict[str, float] mapping feature name → ASV value
```

`baseline` controls how absent features are filled in: `"mean"`/`"median"` use a single
summary row; `"sample"` uses one real (seeded, reproducible) background row instead of a
synthetic average; `"background_expectation"` averages the model's prediction over every
background row (true marginal expectation — more accurate for correlated features, at the
cost of `len(background)` model calls per coalition); or pass a callable
`(background: np.ndarray) -> np.ndarray` for a custom baseline row.

Or build the value function directly with `make_tabular_value_fn` for full control:

```python
from causasv import make_tabular_value_fn

value_fn = make_tabular_value_fn(model=my_classifier, x=X_test[0],
                                  background=X_train, feature_names=[...],
                                  baseline="mean")
values = ASVExplainer(dag).explain(value_fn, method="auto")
```

## Exact vs Approximate (advanced — manual method control)

| Method | When to use | API |
|--------|-------------|-----|
| `exact` | Small DAGs (n ≤ ~8); enumerates all linear extensions | `explainer.exact(value_fn)` |
| `exact_tree` | Rooted directed trees; order-ideal DP | `explainer.exact_tree(value_fn)` |
| `exact_dag` | General DAGs, n ≤ 20; dense order-ideal DP | `explainer.exact_dag(value_fn)` |
| `exact_dag_sparse` | Sparse DAGs, n ≤ 63; BFS over valid order ideals only | `explainer.exact_dag_sparse(value_fn)` |
| `uniform_sparse` | Sparse DAGs, n ≤ 63; zero-variance uniform sampling (ESS = n_samples) | `explainer.approximate_uniform_sparse(value_fn, cfg)` |
| `approx` | Any DAG; IS-weighted sampling | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

`auto` dispatch: n ≤ 8 → `exact`; rooted tree → `exact_tree`; n ≤ 20 → `exact_dag_sparse` if edge_count ≤ 2n else `exact_dag`; 20 < n ≤ 28 → `exact_dag_sparse`; 28 < n ≤ 63 → `exact_dag_sparse` if order ideals ≤ 250k (sparse preflight), else `approx`; n > 63 → `approx`.

`exact_dag_sparse` visits only valid order ideals (sets where every node's parents are also present). For sparse DAGs (chains, trees, few branching points), this can be orders of magnitude fewer states than 2^n. Returns `n_order_ideals`, `state_ratio`, and `memory_mb` diagnostics.

`approximate_uniform_sparse` samples each linear extension with equal probability 1/L(G) using a lazily memoized `dp_ind` table (HashMap), so ESS = n_samples exactly — no IS weight variance. Use `explain_adaptive(method="uniform_sparse")` for adaptive stopping with per-feature stderr and CI.

The IS approximate estimator uses self-normalized importance sampling to correct for the bias introduced by the frontier sampler, so the efficiency axiom (Σφ_i = v(V) − v(∅)) holds exactly even for approximate results. ESS = (Σw)² / Σw²: ESS ≈ n_samples means reliable; ESS ≪ n_samples means high variance.

**Approximation diagnostics checklist** — before trusting an approximate result:
1. `info["ess_ratio"]` ≥ 0.1 (or 1.0 if using `uniform_sparse`)
2. Run `explain_stability()` with multiple seeds; `rank_stability` ≥ 0.9
3. Use `explain_adaptive()` if you need per-feature stderr and CI bounds

See [docs/correctness.md](docs/correctness.md) for axiom proofs, ESS interpretation, and the full checklist.

**Choosing between `auto` and `auto_quality`:**
- Use `auto` for exploratory work where CI is not required — it dispatches to exact methods when feasible and IS-weighted approximation otherwise.
- Use `auto_quality` (or `explain_quality()` in Python) when you need confidence intervals or a guaranteed ESS = n_samples on approximate paths. Every code path returns `stderr`; the approximate fallback is uniform sparse adaptive (not IS-weighted), so ESS is always equal to n_samples.

See [docs/benchmark_corpus.md](docs/benchmark_corpus.md) for measured runtime and method selection across 8 canonical DAGs.
See [docs/comparison_shap.md](docs/comparison_shap.md) for a quantitative runtime and attribution comparison against SHAP KernelExplainer.

## Status

Experimental — v0.8.6. Public API may change before v1.0.

## Algorithm status

| Method | Implementation | Notes |
|--------|---------------|-------|
| `exact` | Enumerates all linear extensions | Reference oracle; practical for n ≤ ~8 |
| `exact_tree` | Rooted tree validation + order-ideal DP | Efficient for trees; hook-length formula |
| `exact_dag` | Order-ideal DP over 2^n states | General DAGs, n ≤ 20; O(2^n × n) |
| `exact_dag_sparse` | BFS over valid order ideals + lazy dp_ind | Sparse DAGs, n ≤ 63; memory-bounded |
| `uniform_sparse` | Lazy dp_ind HashMap uniform sampler | Sparse DAGs n ≤ 63; ESS = n_samples exactly |
| `approx` | Self-normalized IS over topological orderings | Any DAG; corrects frontier-sampler bias |

The brute-force `exact` implementation is used as the reference oracle in tests for all other methods.

The `exact_tree` DP enumerates valid pre-sets via order ideals and weights each by the hook-length formula, avoiding explicit enumeration of all linear extensions. Caterpillar trees of depth 30 see orders-of-magnitude speedups over brute-force.

The `exact_dag` DP computes two tables over all 2^n bitmasks: `dp_fwd[S]` (orderings of valid order ideals S) and `dp_ind[T]` (linear extensions of any induced subgraph G[T]). The ASV for each node i accumulates `dp_fwd[S] × dp_ind[V\(S∪{i})] × (v(S∪{i}) − v(S))` over all valid transitions. This is the order-ideal DP generalized from trees to arbitrary DAGs.

## Feature matrix

| Feature | Rust | Python | Status |
|---------|:----:|:------:|--------|
| Exact ASV (brute-force) | ✓ | ✓ | Stable |
| Rooted-tree exact DP | ✓ | ✓ | Experimental |
| General DAG exact DP (n ≤ 20) | ✓ | ✓ | Experimental |
| Sparse exact DAG DP (n ≤ 63) | ✓ | ✓ | Experimental |
| Uniform sparse sampling (ESS = n_samples) | ✓ | ✓ | Experimental |
| Adaptive uniform sparse + CI | ✓ | ✓ | Experimental |
| Approximate ASV with ESS | ✓ | ✓ | Experimental |
| Adaptive approximation + CI | ✓ | ✓ | Experimental |
| Seeded deterministic parallel approx | ✓ | ✓ | Experimental |
| Batched coalition evaluation | ✓ | ✓ | Experimental |
| sklearn / NumPy helper (TabularExplainer) | — | ✓ | Experimental |
| DAG ensemble / sensitivity ASV | — | ✓ | Experimental |
| DAG structural inspection | — | ✓ | Experimental |
| Graph export (DOT / JSON / networkx) | — | ✓ | Experimental |

## Paper correspondence

*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*

| Algorithm component | causasv |
|---------------------|---------|
| ASV definition | ✓ `exact` (brute-force oracle) |
| Rooted tree exact algorithm | ✓ `exact_tree` (order-ideal DP + hook-length formula) |
| General DAG exact DP | ✓ `exact_dag` (order-ideal DP, n ≤ 20) |
| Importance-sampling approximation for general DAGs | ✓ `approx` |
| Sparse exact DAG DP | ✓ `exact_dag_sparse` (BFS over order ideals, n ≤ 28) |
| Causal discovery | — out of scope |

- `exact_tree` implements the order-ideal enumeration + hook-length weighting for rooted directed trees.
- `exact_dag` implements the two-table order-ideal DP for general DAGs (n ≤ 20).
- `approx` implements importance-weighted topological ordering sampling for any DAG.
- `exact` is a brute-force baseline oracle used as a correctness reference in tests.

## Performance

Selected results on Apple M-series (arm64, release build), `v(S) = |S|`. See [docs/benchmarks.md](docs/benchmarks.md) for full tables.

| DAG | n | Method | Time |
|-----|---|--------|------|
| Chain | 7 | `exact` (brute-force) | 2.7 µs |
| Balanced tree | 15 | `exact_tree` (DP) | 2.8 ms |
| Caterpillar | 10 | `exact_tree` (DP) | 170 µs |
| Chain | 10 | `exact_dag` (dense DP) | **23 µs** |
| Chain | 16 | `exact_dag` (dense DP, 65k states) | 3.0 ms |
| Chain | 16 | `exact_dag_sparse` (17 order ideals, via `auto`) | **11 µs** (~280×) |
| Chain | 24 | `exact_dag_sparse` | 15 µs |
| Two parallel chains | 20 | `exact_dag` (dense, 1M states) | **55 ms** |
| Two parallel chains | 20 | `exact_dag_sparse` (121 states) | **91 µs** (~600×) |
| Diamond | 10 | `approx` seeded (10k samples) | **16 ms** |
| Diamond | 10 | `approximate_adaptive_batched` (10k max) | 2.4 ms |
| Chain | 20 | `approx` serial seeded (10k) | **19 ms** |
| Chain | 20 | `approx` parallel 4t seeded (10k) | 7.4 ms |
| Balanced tree | 31 | `approx` seeded (10k samples) | 83 ms |

Run `cargo bench` to reproduce. HTML reports saved to `target/criterion/`.

## Current limitations

- Brute-force exact ASV is exponential in the number of linear extensions; only practical for n ≤ ~8 nodes.
- `exact_tree` requires a rooted directed tree (single root, all other nodes have in-degree 1). For general DAGs with n ≤ 20, use `exact_dag`. For sparse DAGs with n ≤ 28, use `exact_dag_sparse`. For larger DAGs, use `approx`.
- Python bindings provide `nodes()`, `edges()`, `to_dot()`, and `make_tabular_value_fn`; graph-level DOT export works but Rust-side export is not yet implemented.
- No built-in causal discovery, model training, or automatic graph construction.

## Compared to other tools

`causasv` is not a SHAP replacement or a general-purpose explainability framework.
It solves one narrow problem:

> Computing Asymmetric Shapley Values over a user-supplied causal DAG.

| Tool | Focus | DAG-aware ordering | Exact ASV | CI / stderr |
|------|-------|--------------------|-----------|-------------|
| [SHAP](https://github.com/shap/shap) | Generic feature attribution | No | No | limited |
| [DoWhy](https://github.com/py-why/dowhy) | Causal effect estimation | Graph-based causal workflow | No | method-dependent |
| [Captum](https://captum.ai/) | PyTorch model interpretability | No | No | No |
| [shapiq](https://github.com/mmschlk/shapiq) | Shapley interactions (any order) | No | partial (different target) | benchmarked |
| [shapr](https://github.com/NorskRegnesentral/shapr) | Conditional / causal Shapley (R + Python) | Yes — broader scope | No | R-first |
| [shapflex](https://pypi.org/project/shapflex/) | ASV with causal knowledge (Python alpha) | Yes — similar concept | No | No |
| **causasv** | DAG-aware ASV attribution | **Yes** | **Yes (exact or uniform sparse)** | **Yes (stderr + CI)** |

The main differences from `shapr` and `shapflex`: `causasv` is a Rust-first engine
that requires the user to supply an explicit causal DAG and a value function.
It does not perform causal discovery and does not depend on the data distribution.
`explain_quality()` / `auto_quality()` provide exact results when the DAG is sparse
enough, with uncertainty quantification always available on approximate paths.

## Scope

- **Does** compute ASV (causal feature attribution) given a DAG and a value function
- **Does not** do causal discovery, model training, or feature selection
- **Not** a SHAP replacement — ASV and SHAP answer different questions

## Building Python bindings

```bash
cd py
python -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --features python
python -m pytest tests/
```

## Citation

> Fryer, D., Strümke, I., & Nguyen, H. (2021). *Shapley values for feature selection: The good, the bad, and the axioms.* IEEE Access.

For the asymmetric formulation and efficient tree computation, see the paper that inspired this library:

> Beyond Shapley: Efficient Computation of Asymmetric Shapley Values

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
