# causasv

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)

Fast causal Asymmetric Shapley Values for Rust and Python.

`causasv` is a Rust-first engine for computing Asymmetric Shapley Values over user-supplied causal DAGs. It is designed for explainable AI workflows where feature attribution should respect known causal structure.

This crate does not learn causal graphs. It assumes that the user provides a valid directed acyclic graph and a value function.

## What it is NOT

- Not a causal discovery tool — provide your own DAG
- Not a generic SHAP replacement — computes ASV, not SHAP
- Not a model trainer or feature selector
- Not a deep learning explainability framework

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

dag = CausalDAG()
dag.add_edge("education", "income")
dag.add_edge("income", "risk_score")

explainer = ASVExplainer(dag)

values = explainer.explain(
    value_fn=lambda feature_names: my_model_score(feature_names),
    method="approx",
    n_samples=10_000,
    seed=42,
)
# values: dict[str, float] mapping feature name → ASV value
```

The Python `value_fn` receives a sorted list of feature names present in the coalition and must return a float.

## Exact vs Approximate

| Method | When to use | API |
|--------|-------------|-----|
| `exact` | Small DAGs (n ≤ ~8); enumerates all linear extensions | `explainer.exact(value_fn)` |
| `exact_tree` | Rooted directed trees; validates tree structure | `explainer.exact_tree(value_fn)` |
| `approx` | Any DAG size; importance-weighted sampling | `explainer.approximate(value_fn, SamplingConfig::new(n))` |

The approximate estimator uses self-normalized importance sampling to correct for the bias introduced by the frontier sampler, so the efficiency axiom (Σφ_i = v(V) − v(∅)) holds exactly even for approximate results.

## Status

Experimental — v0.1.0. Public API may change before v1.0.

## Algorithm status

| Method | Implementation | Notes |
|--------|---------------|-------|
| `exact` | Enumerates all linear extensions | Reference oracle; practical for n ≤ ~8 |
| `exact_tree` | Validates rooted tree, then enumerates | Efficient tree DP planned for v0.2.0 |
| `approx` | Self-normalized IS over topological orderings | Corrects frontier-sampler bias |

The brute-force `exact` implementation is used as the reference oracle in tests for all other methods.

## Current limitations

- Brute-force exact ASV is exponential in the number of linear extensions; only practical for n ≤ ~8 nodes.
- `exact_tree` validates and computes using linear extension enumeration; efficient tree DP is planned for v0.2.0.
- Python bindings are minimal in v0.1.0; NumPy integration and richer ergonomics are planned.
- No built-in causal discovery, model training, or automatic graph construction.

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
