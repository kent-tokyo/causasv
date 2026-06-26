# causasv — Causal Feature Attribution via Asymmetric Shapley Values

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kent-tokyo/causasv/actions/workflows/codeql.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/codeql.yml)
[![Security](https://github.com/kent-tokyo/causasv/actions/workflows/security.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/security.yml)
<br>
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![Downloads](https://img.shields.io/crates/d/causasv.svg)](https://crates.io/crates/causasv)
[![GitHub release](https://img.shields.io/github/v/release/kent-tokyo/causasv)](https://github.com/kent-tokyo/causasv/releases)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)
<br>
[![MSRV](https://img.shields.io/badge/MSRV-1.85%2B-orange.svg)](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)

**English** | [日本語](README_ja.md)

`causasv` computes **Asymmetric Shapley Values (ASV)** for causal feature attribution over user-supplied DAGs. It is a Rust-first engine with Python bindings, designed for XAI workflows where feature importance should respect known causal structure.

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

```python
from causasv import CausalDAG, ASVExplainer

# From a list of edges
dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])

# Or from a networkx DiGraph
# import networkx as nx; G = nx.DiGraph(); G.add_edge(...)
# dag = CausalDAG.from_networkx(G)

explainer = ASVExplainer(dag)

values = explainer.explain(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="auto",   # exact for n≤8, exact_tree for rooted trees, approx otherwise
    n_samples=10_000,
    seed=42,
)
# values: dict[str, float] mapping feature name → ASV value
```

The Python `value_fn` receives a sorted list of feature names present in the coalition and must return a float.

For approximate methods, use `explain_with_diagnostics()` to inspect the Effective Sample Size (ESS):

```python
info = explainer.explain_with_diagnostics(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=10_000,
)
print(info["values"])    # dict[str, float]
print(info["ess"])       # float — ESS ≈ n_samples means reliable; ESS ≪ n_samples means high variance
print(info["is_exact"])  # False
```

## Exact vs Approximate

| Method | When to use | API |
|--------|-------------|-----|
| `exact` | Small DAGs (n ≤ ~8); enumerates all linear extensions | `explainer.exact(value_fn)` |
| `exact_tree` | Rooted directed trees; order-ideal DP | `explainer.exact_tree(value_fn)` |
| `exact_dag` | General DAGs, n ≤ 20; order-ideal DP | `explainer.exact_dag(value_fn)` |
| `approx` | Any DAG (n > 20); importance-weighted sampling | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

`auto` dispatch: n ≤ 8 → `exact`; rooted tree → `exact_tree`; n ≤ 20 → `exact_dag`; else → `approx`.

The approximate estimator uses self-normalized importance sampling to correct for the bias introduced by the frontier sampler, so the efficiency axiom (Σφ_i = v(V) − v(∅)) holds exactly even for approximate results.

The result includes `effective_sample_size` (ESS = (Σw)² / Σw²): ESS ≈ n_samples means IS weights are uniform and the estimate is reliable; ESS ≪ n_samples indicates high weight variance.

## Status

Experimental — v0.5.0. Public API may change before v1.0.

## Algorithm status

| Method | Implementation | Notes |
|--------|---------------|-------|
| `exact` | Enumerates all linear extensions | Reference oracle; practical for n ≤ ~8 |
| `exact_tree` | Rooted tree validation + order-ideal DP | Efficient for trees; hook-length formula |
| `exact_dag` | Order-ideal DP over 2^n states | General DAGs, n ≤ 20; O(2^n × n) |
| `approx` | Self-normalized IS over topological orderings | Any DAG; corrects frontier-sampler bias |

The brute-force `exact` implementation is used as the reference oracle in tests for all other methods.

The `exact_tree` DP enumerates valid pre-sets via order ideals and weights each by the hook-length formula, avoiding explicit enumeration of all linear extensions. Caterpillar trees of depth 30 see orders-of-magnitude speedups over brute-force.

The `exact_dag` DP computes two tables over all 2^n bitmasks: `dp_fwd[S]` (orderings of valid order ideals S) and `dp_ind[T]` (linear extensions of any induced subgraph G[T]). The ASV for each node i accumulates `dp_fwd[S] × dp_ind[V\(S∪{i})] × (v(S∪{i}) − v(S))` over all valid transitions. This is the order-ideal DP generalized from trees to arbitrary DAGs.

## Feature matrix

| Feature | Rust | Python | Status |
|---------|:----:|:------:|--------|
| Exact ASV (brute-force) | ✅ | ✅ | Stable |
| Rooted-tree exact DP | ✅ | ✅ | Experimental |
| General DAG exact DP (n ≤ 20) | ✅ | ✅ | Experimental |
| Approximate ASV with ESS | ✅ | ✅ | Experimental |
| Adaptive approximation | 🚧 | 🚧 | Planned (v0.6) |
| sklearn / NumPy helper | ❌ | 🚧 | Planned (v0.7) |
| Graph export (DOT / networkx) | 🚧 | 🚧 | Planned (v0.7) |

## Paper correspondence

*Beyond Shapley: Efficient Computation of Asymmetric Shapley Values*

| Algorithm component | causasv |
|---------------------|---------|
| ASV definition | ✅ `exact` (brute-force oracle) |
| Rooted tree exact algorithm | ✅ `exact_tree` (order-ideal DP + hook-length formula) |
| General DAG exact DP | ✅ `exact_dag` (order-ideal DP, n ≤ 20) |
| Importance-sampling approximation for general DAGs | ✅ `approx` |
| General DAG optimized DP | 🚧 planned |
| Causal discovery | ❌ out of scope |

- `exact_tree` implements the order-ideal enumeration + hook-length weighting for rooted directed trees.
- `exact_dag` implements the two-table order-ideal DP for general DAGs (n ≤ 20).
- `approx` implements importance-weighted topological ordering sampling for any DAG.
- `exact` is a brute-force baseline oracle used as a correctness reference in tests.

## Performance

Benchmarks on Apple M-series (arm64, release build). `v(S) = |S|` (additive value function).

| Benchmark | n | L(T) | Method | Time |
|-----------|---|-------|--------|------|
| Balanced binary tree | 7 | 80 | `exact` (enumerate) | ~70 µs |
| Balanced binary tree | 7 | 80 | `exact_tree` (DP) | ~145 µs |
| Balanced binary tree | 15 | ~22 M | `exact` | — (infeasible) |
| Balanced binary tree | 15 | ~22 M | `exact_tree` (DP) | ~7.8 ms |
| Caterpillar tree | 10 | 945 | `exact_tree` (DP) | ~347 µs |
| Approximate (chain) | 10 | — | `approx` (1k samples) | ~2.9 ms |

> Exact enumeration would require visiting ~22 million valid causal orderings for n=15;
> `exact_tree` computes the same ASV in milliseconds via order-ideal DP.

Note: for n ≤ ~8, `exact` is often faster than `exact_tree` due to lower allocator overhead.
`exact_tree` becomes the only feasible exact method for larger trees.
Run with `cargo bench` to reproduce.

## Current limitations

- Brute-force exact ASV is exponential in the number of linear extensions; only practical for n ≤ ~8 nodes.
- `exact_tree` requires a rooted directed tree (single root, all other nodes have in-degree 1). For general DAGs, use `exact` (small n) or `approx`.
- Python bindings are minimal; NumPy integration and richer ergonomics are planned.
- No built-in causal discovery, model training, or automatic graph construction.

## Compared to other tools

`causasv` is not a SHAP replacement or a general-purpose explainability framework.
It solves one narrow problem:

> Computing Asymmetric Shapley Values over a user-supplied causal DAG.

| Tool | Focus | ASV / causal DAG |
|------|-------|-----------------|
| [SHAP](https://github.com/shap/shap) | General-purpose Shapley / SHAP | No — standard Shapley only |
| [Captum](https://captum.ai/) | PyTorch model interpretability | No |
| [shapr](https://github.com/NorskRegnesentral/shapr) | Conditional / causal Shapley (R + Python) | Yes — broader scope, R-first |
| [shapflex](https://pypi.org/project/shapflex/) | ASV with causal knowledge (Python alpha) | Yes — similar concept |
| **causasv** | ASV over user-supplied causal DAGs | **Core focus** |

The main differences from `shapr` and `shapflex`: `causasv` is a Rust-first engine
that requires the user to supply an explicit causal DAG and a value function.
It does not perform causal discovery and does not depend on the data distribution.

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
