# Roadmap

## Done (≤ v0.8.1)

- `exact` — brute-force enumeration of all linear extensions; oracle for n ≤ ~8
- `exact_tree` — order-ideal DP for rooted directed trees; hook-length weighting
- `exact_dag` — dense order-ideal DP for general DAGs, n ≤ 20; O(2^n × n)
- `exact_dag_sparse` — BFS over valid order ideals only, n ≤ 28; memory-bounded
- `auto` dispatch — exact → exact_tree → exact_dag → exact_dag_sparse → approx
- `auto` fallback diagnostics — `fallback_from`, `fallback_reason`, `selected_method`
- Approximate ASV via self-normalized importance sampling over topological orderings
- Kahan summation and log-weight normalization across all IS paths
- Adaptive approximation with convergence criteria, per-feature stderr, and CI
- Batched coalition evaluation (`value_fn_batch`) — reduces Python GIL acquisitions
- Seeded deterministic parallel sampling (splitmix64 per-worker seeds)
- ESS diagnostics, overflow guard (`u64::checked_add`), property-based tests (10 axioms)
- Python: `ASVExplainer`, `CausalDAG`, `TabularExplainer`, `ASVEnsembleExplainer`
- Python: `explain_stability()`, `explain_adaptive()`, `explain_adaptive_batch()`
- Python: `make_tabular_value_fn()`
- DAG: `ancestors()`, `descendants()`, `topological_layers()`, `to_json()`, `from_json()`
- Graph export: DOT, JSON, networkx
- `docs/benchmarks.md`, `CHANGELOG.md`, `examples/`

## Now (v0.8.x)

Stabilizing. No new algorithms. Focus: correctness evidence, diagnostics, docs.

- [x] docs/benchmarks.md with benchmark tables
- [x] CHANGELOG.md
- [ ] docs/correctness.md — axiom test table + approximation guarantees
- [ ] CI version-sync to include README.md status line
- [ ] examples/stability_diagnostics.py

## Next (v0.9.0, tentative)

Only if there is clear user demand:

- Optional stderr for fixed-sample `approx` (currently only `approx_adaptive` returns it)
- Parallel log-weight normalization (requires two-pass global max — needs careful design)
- `Dag::edges()` as a stable first-class public API

## Non-goals (permanently out of scope)

Causal discovery, model training, GPU acceleration, deep learning-specific
explainability, automatic graph construction, Web UI.
