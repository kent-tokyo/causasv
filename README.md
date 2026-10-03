# causasv

[![CI](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml/badge.svg)](https://github.com/kent-tokyo/causasv/actions/workflows/ci.yml)
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-blue.svg)](https://github.com/kent-tokyo/causasv/security/code-scanning)
[![Crates.io](https://img.shields.io/crates/v/causasv.svg)](https://crates.io/crates/causasv)
[![PyPI](https://img.shields.io/pypi/v/causasv.svg)](https://pypi.org/project/causasv/)
[![Docs.rs](https://docs.rs/causasv/badge.svg)](https://docs.rs/causasv)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

**English** | [日本語](README_ja.md) | [中文](README_zh.md)

Fast causal Asymmetric Shapley Values for Rust and Python.

`causasv` computes Asymmetric Shapley Values (ASV) over a causal DAG and value
function supplied by the caller. ASV averages marginal contributions over
topological orderings, so a cause is never introduced after its descendants.

`causasv` does not learn causal graphs, train models, or infer causality from
data. Use it when the graph is already available and its assumptions can be
defended outside this library.

## Install

Python wheels are published for Linux x86_64, macOS universal2, and Windows
x86_64:

```bash
pip install causasv
```

Rust:

```toml
[dependencies]
causasv = "0.8"
```

The Rust crate requires Rust 1.85 or newer. Python requires 3.9 or newer.

## ASV in one formula

For a DAG `G`, let `Π(G)` be its topological orderings and `pre(i, π)` the
features before `i` in ordering `π`:

```text
φᵢ = 1 / |Π(G)| · Σπ∈Π(G) [v(pre(i,π) ∪ {i}) - v(pre(i,π))]
```

Standard Shapley values average over every permutation. ASV uses only
permutations allowed by the supplied DAG. They agree for additive value
functions and can differ when feature interactions depend on order. See the
[SHAP comparison](docs/comparison_shap.md) for a worked example.

## Rust quick start

```rust
use causasv::{AsvExplainer, Dag, SamplingConfig};

fn main() -> Result<(), causasv::CausasvError> {
    let mut dag = Dag::new();
    let education = dag.add_node("education");
    let income = dag.add_node("income");
    let risk = dag.add_node("risk_score");
    dag.add_edge(education, income)?;
    dag.add_edge(income, risk)?;

    let result = AsvExplainer::new(dag).auto(
        |coalition| Ok(coalition.len() as f64),
        SamplingConfig::new(10_000).with_seed(42),
    )?;

    println!("{:?}", result.values);
    Ok(())
}
```

`auto()` uses an exact method when feasible and falls back to seeded
importance sampling. Use `auto_quality()` when approximate paths also need
standard errors and convergence metadata.

## Python quick start

`explain_quality()` is the recommended Python entry point. It tries exact
methods first and otherwise returns an adaptive estimate with uncertainty
diagnostics.

```python
from causasv import ASVExplainer, CausalDAG, explain_quality

dag = CausalDAG.from_edges([
    ("education", "income"),
    ("income", "risk_score"),
])

result = explain_quality(
    ASVExplainer(dag),
    value_fn=lambda features: my_model_score(features),
    seed=42,
    ci=0.95,
)

print(result["values"])
print(result["selected_method"])
print(result["stderr"])
print(result["ci_low"], result["ci_high"])
```

The callback receives a sorted `list[str]` containing the features in the
coalition. For an expensive model, pass `value_fn_batch` instead of
`value_fn` to evaluate many coalitions per Python call.

## Choosing a method

Prefer `auto()` or `auto_quality()` unless a test or benchmark requires a
specific algorithm.

| Method | Use | Bound |
| --- | --- | --- |
| `exact` | brute-force reference oracle | practical around `n <= 8` |
| `exact_tree` | exact rooted-tree DP | shape-budgeted; bitmask limit `n <= 64` |
| `exact_dag` | exact dense order-ideal DP | `n <= 20` |
| `exact_dag_sparse` | exact sparse order-ideal DP | default `n <= 28`; configurable up to 63 |
| `uniform_sparse` | equal-probability topological sampling | `n <= 63` |
| `approx` | importance-sampling estimate | no node-count limit |

Exact feasibility depends on graph structure, state count, and memory—not only
on node count. The canonical limits and dispatch rules are in
[Correctness](docs/correctness.md#exact-method-bounds).

For approximate results:

- keep a seed when reproducibility matters
- inspect `stderr`, confidence intervals, convergence, and fallback metadata
- inspect `ess_ratio` for importance-sampling paths
- rerun across seeds when feature ranking is consequential

`explain_safe()` automates the main checks. `explain_stability()` reports
cross-seed rank stability.

## Included capabilities

- DAG validation, topological sorting, and order enumeration
- brute-force, rooted-tree, dense-DAG, and sparse-DAG exact ASV
- fixed-sample, adaptive, uniform sparse, parallel, and batched approximation
- deterministic seeds, ESS, standard errors, confidence intervals, and
  fallback diagnostics
- Rust and Python APIs
- CPDAG representation and consistent DAG extension
- d-convex and strong d-convex graph reduction
- optional Python helpers for tabular models and DAG sensitivity

Graph reduction operates on a caller-supplied graph. It does not perform
causal discovery or effect estimation. Its assumptions and clean-room
implementation record are documented in
[Strong d-convex hulls](docs/strong_d_convex_hulls.md).

## Documentation

- [Correctness and method limits](docs/correctness.md)
- [Criterion benchmark results](docs/benchmarks.md)
- [Canonical Python benchmark corpus](docs/benchmark_corpus.md)
- [ASV and SHAP comparison](docs/comparison_shap.md)
- [Strong d-convex hull assumptions](docs/strong_d_convex_hulls.md)
- [quietset instability integration](docs/integrations/quietset_label_instability.md)
- [Rust API](https://docs.rs/causasv)

Measurements are evidence for the named version, hardware, graph, and value
function only. They are not general performance or attribution-quality claims.

## Current limits

- The graph and value function must be supplied by the caller.
- `exact_tree` can reject a modest-sized but highly branching tree when its
  estimated combinatorial cost exceeds `ExactTreeConfig`.
- Dense and sparse exact methods use bitmask representations and therefore
  have fixed node limits. Importance-sampling methods use a large-coalition
  backend above 64 nodes.
- CPDAG validation checks structure and consistent extendability; it does not
  prove that every directed edge is compelled in a genuine equivalence class.
- The graph-reduction guarantee depends on distributional and graph
  assumptions that the library cannot verify.

## Status

Experimental — v0.8.8. Public APIs may change before v1.0.

The brute-force implementation remains the correctness oracle for optimized
methods on small graphs. See [CHANGELOG.md](CHANGELOG.md) for released and
unreleased changes.

## Build the Python extension

```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytest
cd py
maturin develop --features python
python -m pytest
```

## Citation

Project citation metadata is in [CITATION.cff](CITATION.cff). The ASV design is
inspired by *Beyond Shapley: Efficient Computation of Asymmetric Shapley
Values*.

The graph-reduction module independently implements mathematical definitions
from Deng, Sun, Li, and Liu, “Estimate Collapsibility of Causal Effects in
Completed Partial DAGs via Strong d-Convex Hulls,” arXiv:2606.08941 (2026).
See the [citation and scope record](docs/strong_d_convex_hulls.md).

## License

Licensed under either [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at
your option.
