# Changelog

All notable changes to causasv are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Changed
- `src/numerics.rs`: extracted `kahan_add` from `approx.rs` into a dedicated
  numerics module; no behavior change

### Added
- `examples/benchmark_batched_value_fn.py`: demonstrates wall-clock speedup of
  `value_fn_batch` over normal `value_fn` across batch sizes 64 / 256 / 1024

---

## [0.7.0] — 2026-06

### Added
- `exact_dag_sparse`: BFS-based sparse order-ideal DP for general DAGs up to n=28;
  returns `n_order_ideals`, `state_ratio`, and `memory_mb` diagnostics
- `explain_adaptive` / `explain_adaptive_batch`: adaptive IS sampling with
  per-feature confidence intervals (`ci_low`, `ci_high`, `stderr`)
- `explain_stability`: multi-seed stability diagnostics (mean, std, rank stability)
- `ASVEnsembleExplainer`: sensitivity analysis across multiple candidate DAGs
- Seeded deterministic parallel approximation via per-worker splitmix64 seeds
- `value_fn_batch` parameter on `explain()` and `explain_with_diagnostics()`;
  reduces Python GIL acquisitions from O(n_samples × n) to O(n_samples / batch_size)
- `TabularExplainer` and `make_tabular_value_fn` for sklearn-compatible models
- `dag.inspect()`, `dag.topological_layers()`, `dag.ancestors()`, `dag.descendants()`
- `auto()` fallback diagnostics: `fallback_from`, `fallback_reason`, `selected_method`
  are set when `exact_dag_sparse` falls back to `approximate`
- Property-based tests covering 10 ASV axioms via proptest
  (efficiency, dummy, additivity, relabeling invariance, exact/sparse consistency)
- `docs/benchmarks.md` with full benchmark tables

### Changed
- `approx.rs`: Kahan compensated summation applied consistently across all IS paths
  (seeded parallel workers, adaptive_batched); per-batch log-weight normalization
  in adaptive paths prevents overflow on extreme frontier distributions
- `AsvExplainer` precomputes `parents_mask` once in `new()`, eliminating repeated
  O(n²) computation per `exact_dag` / `exact_dag_sparse` call
- `parents_raw` visibility narrowed to `pub(crate)`

### Fixed
- `py.detach()` retained as the correct GIL-release API for PyO3 0.29
  (`py.allow_threads()` does not exist in this version)

---

## [0.6.0] — earlier

- Adaptive IS sampling (`approximate_adaptive`)
- `exact_dag` dense order-ideal DP for general DAGs up to n=20
- Python bindings via PyO3 / maturin

## [0.5.0] — earlier

- Exact brute-force ASV (`exact`)
- Rooted-tree exact DP (`exact_tree`)
- Basic approximate IS sampling (`approximate`)
- Python bindings: `CausalDAG`, `ASVExplainer`
