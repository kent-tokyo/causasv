# Changelog

All notable changes to causasv are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [0.8.3] — 2026-06

### Performance
- `approx.rs`: eliminate 2 of 3 per-sample Vec allocations in the seeded sequential `approximate_asv` path by reusing `SamplerScratch` (was the only approx path not using `sample_one_into`); now consistent with all parallel / batched / adaptive paths
- `approx.rs`: replace per-batch `HashSet<u64>` coalition dedup in `approximate_asv_batched` and `approximate_asv_adaptive_batched` with `Vec<u64> + sort_unstable + dedup`; cheaper for u64 keys without hash allocation overhead
- `asv.rs` / `graph.rs`: `auto()` dispatch for 8 < n ≤ 20 now uses `exact_dag_sparse` first when `edge_count ≤ 2n` (sparse heuristic), falling back to `exact_dag` on failure; chains, trees, and other sparse DAGs in this range route to the sparse path which visits far fewer order ideals

### Internals
- `graph.rs`: add `pub(crate) edge_count()` helper used by the auto dispatch heuristic
- `sampler.rs`: mark `sample_one` as `#[cfg(test)]`; only test code uses it now

### Cargo
- `Cargo.toml`: add `rust-version = "1.85"` (edition 2024 MSRV); makes the minimum Rust version explicit and produces a clear error for older toolchains

### Benchmarks
- `benches/asv_bench.rs`: add `approx_chain_20_10k_seeded`, `approx_balanced_tree_31_10k_seeded`, `adaptive_batch_diamond_10`, `exact_dag_vs_sparse_chain_16` to measure the above improvements

---

## [0.8.2] — 2026-06

### Performance
- `dag_dp.rs`: replace `HashMap<u64,f64>` value cache with `Vec<f64>[mask]` (NaN sentinel, n≤20 → ≤8MB); hoist `v(S)` lookup outside candidate-node loop; convert `dp_fwd`/`dp_ind`/ASV accumulation inner loops from O(n) full scan to bit iteration over relevant set bits — **−85 to −90% on exact_dag paths**
- `sampler.rs`: incremental frontier in `sample_one` (build once, maintain via `swap_remove` + child push — O(n+edges)/sample vs O(n²)); add `SamplerScratch` + `sample_one_into` for per-worker scratch reuse in all parallel/batched/adaptive paths (2 of 3 per-sample Vec allocs eliminated) — **−70 to −80% on approx paths**
- `cache.rs`: `mask_to_coalition` now uses `trailing_zeros` loop over set bits instead of 0..64 scan

### Tests
- `tests/approx_accuracy_tests.rs`: golden corpus — 9 tests comparing approx vs exact on chain, fork, collider, diamond, two-parallel-chains, balanced-tree with additive and weighted value functions

### Benchmarks
- Added `approx_diamond_10_10k_seeded` and `approx_balanced_tree_15_10k_seeded` criterion groups

---

## [0.8.1] — 2026-06

### Changed
- `src/approx.rs`: seeded serial `approximate_asv` path now collects all samples
  before processing, applying per-batch log-weight normalization (`max_log_w` subtraction)
  consistent with the batched, adaptive, and adaptive-batched paths

### Docs
- README badges reduced from 11 to 7 (removed CodeQL, Security, Downloads, GitHub release)
- README_ja and README_zh synced to v0.8.0 feature set (were at v0.6.0)

---

## [0.8.0] — 2026-06

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
