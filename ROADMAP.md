# Roadmap

## v0.5.0 (current)

- General DAG exact DP (`exact_dag`) — order-ideal DP over 2^n states, n ≤ 20
- `auto` dispatch: exact → exact_tree → exact_dag → approx
- Python bindings via PyO3/maturin
- `explain_with_diagnostics()` returning ESS, n_samples, is_exact

## v0.6.0 (planned)

**Adaptive approximation**

- `approx_adaptive` method: run sampling in batches, stop when ESS ratio converges
- `AdaptiveSamplingConfig`: min_samples, max_samples, batch_size, rel_tol, ess_ratio_min, seed
- Python: `explain_with_diagnostics(..., method="approx_adaptive")` returns converged, stderr per node

## v0.7.0 (planned)

**Python ergonomics**

- `make_tabular_value_fn(model, x, background, feature_names)` — wrap a sklearn-compatible predict call as a value function
- Graph export: `to_dot()` (Graphviz DOT string), `to_networkx()` (returns a `networkx.DiGraph`)

## v0.8.0 (planned)

**Benchmarking and examples**

- Criterion HTML benchmark reports published in CI (GitHub Pages)
- SHAP vs ASV comparison example with synthetic causal DAG
- General DAG optimized DP — sub-2^n algorithm for structured graphs

---

Non-goals (permanently out of scope): causal discovery, model training, automatic graph construction, GPU acceleration, deep learning-specific explainability.
