# Changelog

All notable changes to causasv are documented here.
Versions follow [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Fixed
- `approximate`/`approximate_adaptive`/`approximate_batched`/`approximate_adaptive_batched`
  (and, transitively, `auto()`/`auto_quality()`'s fallback into them, and the Python
  `explain()`/`explain_quality()`/`explain_quality_batch()` paths that call them) rejected
  any DAG with n > 64 nodes with `InvalidConfig`, contradicting the public docs' "works for
  any DAG size" claim. Root cause: a sample's growing prefix coalition was represented as a
  single `u64` bitmask, which cannot address a `NodeId >= 64`. The frontier sampler itself
  never touched a bitmask, so the fix is scoped to the coalition representation:
  `src/coalition.rs` adds `LargeCoalition`, a growable word-vector bitset (`ceil(n/64)` `u64`
  words), and `src/approx_large.rs` mirrors the four approximate estimators over it — with a
  bounded `LargeCoalitionCache` (persistent, entry-capped) on the non-batched paths and a
  round-scoped cache (not persisted across sampling batches) on the batched paths, instead of
  an unbounded `HashMap<u64, f64>`. The existing n ≤ 64 fast path is unchanged; each `_large`
  function is dispatched to only when n > 64. `exact`/`exact_dag`/`exact_dag_sparse`/
  `uniform_sparse`/`uniform_sparse_adaptive` are unaffected and remain n ≤ 63/64 (they still
  pack a coalition into a `u64` for their dense/sparse DP or zero-variance sampling, which is
  a structural limit, not a preflight choice).
- `auto()`/`auto_quality()`'s rooted-tree fallback previously always reported
  `fallback_reason: "tree shape exceeded exact_tree's feasibility budget"` when falling back
  to `approximate`/`approximate_adaptive` — accurate for a bushy tree that blew
  `ExactTreeConfig`'s cost budget, but wrong for a tree with n > 64 whose real failure cause
  is `exact_tree`'s own bitmask limit (previously unreachable, since `approximate()` itself
  used to error out first for n > 64, so `auto()` never got as far as attaching this message
  to a successful result). Now distinguishes the two causes based on which `exact_tree` error
  variant actually fired.

### Added
- Backend-parity tests (`src/approx_large.rs`, internal, plus a `proptest` generalization):
  on the same n ≤ 64 DAG/seed/sampling order, the n ≤ 64 (`u64`) and n > 64
  (`LargeCoalition`) backends produce bitwise-identical results for the serial-seeded path —
  the primary correctness oracle for n > 64, since no independent exact method exists there.
- `tests/large_dag_approx_tests.rs`: 65-node additive-chain and non-tree (two-disjoint-chains)
  accuracy checks across all four approximate entry points plus `auto`/`auto_quality`;
  128/256-node smoke tests; seeded-parallel determinism; `value_fn_batch` validation (short
  return, long return, `Err` propagation, and NaN pass-through are all explicit, not a silent
  `zip`-truncation) for n > 64.
- Criterion benchmarks for the n=64→65 coalition-representation boundary (chain, seeded
  parallel, adaptive, batched, adaptive-batched) plus non-tree/collider (diamond-chain) and
  fork-chain (caterpillar) shapes at n ≈ 65/128 — see `docs/benchmarks.md`.

### Docs
- `docs/correctness.md`: new "Large-DAG approximate paths (n > 64)" section explaining the
  coalition representation, why bounded (not unbounded) caching is used, and how correctness
  is verified without an independent exact oracle.
- README / README_ja / README_zh: replaced the "n > 64 rooted trees have no working method"
  limitation with an accurate description (`approx`/`approx_adaptive`/`approx_batched`/
  `approx_adaptive_batch` have no node-count limit; `exact_dag_sparse`/`uniform_sparse`
  remain n ≤ 63 structurally); added measured n=64/65/128/256 timings.
- `ROADMAP.md`: full rewrite — it predated the CPDAG/strong-d-convex-hull work in 0.8.7/0.8.8
  and still listed this PR's fix as a hypothetical future item.

## [0.8.8] — 2026-07-27

### Added
- `Cpdag` type (`src/cpdag.rs`): completed partial DAG representation with directed (compelled) and undirected edges, mirroring `Dag`'s insertion-order `NodeId` conventions. `validate_pdag`/`validate_cpdag` (structural checks; the latter reuses `consistent_extension` as its extendability oracle rather than implementing chain-component chordality separately — documented boundary: does not verify directed edges are the compelled edges of a genuine Markov equivalence class). `consistent_extension`: independent Rust implementation of the Dor-Tarsi (1992)/Chickering (2002) PDAG-to-DAG algorithm, deterministic candidate selection, `Err(NotExtendable)` when no valid orientation exists (e.g. a chordless undirected cycle). `Cpdag::induced_subgraph`. Python bindings (`CausalCPDAG`), JSON round-trip.
- `Dag::d_convex_hull`/`Dag::strong_d_convex_hull` (`src/d_convex.rs`) and `Cpdag::strong_d_convex_hull`: graph-reduction algorithms computing the minimal node set that, under the assumptions established in the referenced paper, preserves a causal-effect estimate after marginalizing out everything else. Independent implementation of the CVM/ICHA/ISCHA algorithms and the CPDAG-invariance theorem from Deng, Sun, Li & Liu, "Estimate Collapsibility of Causal Effects in Completed Partial DAGs via Strong d-Convex Hulls," arXiv:2606.08941 (2026, CC BY 4.0) — see `docs/strong_d_convex_hulls.md` for the full citation, clean-room notes, and scope/assumptions (in particular: proven only for non-adjacent target-variable pairs). `Cpdag::strong_d_convex_hull` computes via one consistent DAG extension, sound per the paper's Theorem 5 (hull vertex set is invariant across a CPDAG's whole Markov equivalence class). `Dag::induced_subgraph` (the `Dag`-level counterpart to `Cpdag::induced_subgraph`). Python bindings on both `CausalDAG` and `CausalCPDAG`. No IDA-based causal-effect estimation on the reduced graph yet — this release only performs the graph reduction step.
- `CITATION.cff` gains a second `references` entry for the strong-d-convex-hull paper (authors, DOI, URL, CC BY 4.0 license), alongside the existing ASV-paper reference.

### Tests
- `tests/cpdag_tests.rs` / `tests/cpdag_property_tests.rs`: construction, all edge-conflict error variants, extension correctness (chordal triangle extends, chordless 4-cycle fails), a proptest property test deriving a CPDAG from a random DAG's compelled/non-compelled edge split.
- `tests/d_convex_tests.rs` / `tests/d_convex_property_tests.rs`: hand-built fixtures on classical causal structures (chain, fork, collider, multi-parent collider) plus an independent brute-force d-separation oracle (Pearl 1988 path enumeration and collider blocking — a different algorithm from the moralization-based implementation) that cross-checks every random DAG proptest generates. The oracle caught a real over-inclusion bug during development: the paper's Algorithm 3 pseudocode literally absorbs a violating node's entire parent set, but its own Theorem 2 proof text says only the non-adjacent parents must be included — fixed to absorb only the specific violating parent pairs.
- `py/tests/test_cpdag.py` / `py/tests/test_d_convex.py`: mirror the Rust fixtures through the Python API, plus Rust/Python result-parity and interop checks (e.g. a hull's reduced graph feeding straight into `ASVExplainer`).
- `benches/d_convex_bench.rs`: Criterion benchmarks across chain, fork-chain, layered-sparse, and collider-rich DAG shapes.

## [0.8.7] — 2026-07-27

### Changed
- crates.io `categories` gains `artificial-intelligence` and `mathematics` (alongside existing `science`, `algorithms`) for discoverability.
- PyPI `classifiers` gains `Topic :: Scientific/Engineering :: Mathematics` alongside the existing `Artificial Intelligence` topic.
- Crate-level doc comment (`src/lib.rs`, shown on docs.rs) now mentions causal feature attribution, XAI, and SHAP-style feature importance, matching the README's framing.

### CI
- Linux aarch64 wheel cross-build removed from the release matrix; it fails under QEMU inside the manylinux container regardless of `docker/setup-qemu-action`. `release.yml` gained a `workflow_dispatch` trigger for manual reruns.

## [0.8.6] — 2026-07-27

### Added
- `AsvExplainer::approximate_uniform_sparse_adaptive_batched()`: batched quality path — collects an entire convergence batch of topological orderings, deduplicates the prefix coalition masks they need, and calls `value_fn_batch` once per batch. Reduces Python GIL round-trips from O(n × batch_size) to O(unique_masks_per_batch). ESS = n_samples exactly (no IS variance). Requires n ≤ 63; falls back to IS-adaptive batch for n > 63.
- `ASVExplainer.explain_quality_batch()` Python method: same dict contract as `explain_quality` (values, stderr, ci, ci_low, ci_high, selected_method, fallback_from, fallback_reason); routes n ≤ 63 to uniform sparse batch, n > 63 to IS-adaptive batch.
- `explain_safe()` helper (`py/causasv/helpers.py`): wraps `explain_quality()`/`explain_stability()` and returns ESS-ratio and rank-stability warnings plus a list of `unstable_features` (CI spans zero), so callers don't have to manually judge whether a result is trustworthy.
- `TabularExplainer`/`make_tabular_value_fn`: `baseline` gains `"median"`, `"sample"`, `"background_expectation"` modes and accepts a custom callable, in addition to the existing `"mean"` default. `background_expectation` is vectorized via an optional `predict_fn_batch` (one call instead of one per background row) and `TabularExplainer.explain_instance_quality_batch()` stacks background rows across an entire sampling batch for the same reduction.
- `ExactTreeConfig` / `exact_tree_with_config()` / `TreeExactCostEstimate` (issue #36): an O(n), allocation-free preflight that estimates `exact_tree`'s real cost (the product of every ancestor level's side-sibling order-ideal count, not just node count) before enumerating, and returns a structured `ExactTreeBudgetExceeded` error instead of hanging on a bushy-but-modest-n rooted tree. `auto()`/`auto_quality()` now run this preflight for rooted trees and fall back through `exact_dag_sparse` then `approximate`/`approximate_adaptive` when a shape is rejected.
- `causasv.instability` package (opt-in, not imported by base `causasv`; matches the `plot.py` lazy-dependency pattern): a loosely-coupled adapter turning quietset label-instability reports into ASV attribution over a user-supplied DAG. quietset itself is untouched — only its JSONL output is read by field name.
  - `bundle.py` (Phase 1): joins quietset's `Observation`/`StabilityReport` JSONL into a tabular `(sample_id, config_id)` dataset via a bundle manifest, keeping `cell_features` (the config axis being compared) disjoint from `replicate_axes` (the axis quietset varies internally to measure instability). Rejects target leakage, raw seed/shuffle_seed as features (no causal/ordinal meaning), mixed-file config drift, and silent 0-fill for missing values.
  - `modeling.py` (Phase 2): `fit_instability_model` — logistic/ridge regression as the reproducible default, `HistGradientBoosting` opt-in via `model="hgb"`. Grouped CV (`GroupKFold`/`StratifiedGroupKFold`, default group = `sample_id`) so a physical sample never crosses train/test even across config cells. `min_cv_metric` has no built-in default; when set and unmet, a warning is recorded rather than blocking the run.
  - `value.py` (Phase 3): `load_attribution_dag`, `make_global_value_fn` (retrain-per-coalition, memoized by coalition), `make_local_value_fn` (delegates to `make_tabular_value_fn`) — DAG-aware ASV value functions over raw (pre-encoding) feature names.
  - `sensitivity.py` (Phase 4): `explain_with_dag_sensitivity` — runs `explain_safe` per DAG and, across multiple DAGs, aggregates mean/std ASV, Kendall-tau rank stability, sign stability, and rank spread. Requires all DAGs to share the same node set (checked up front, unlike `ASVEnsembleExplainer`).
  - `report.py` (Phase 5): `build_attribution_report`/`summarize_attribution`/`dump_attribution_json` — `causasv-instability-attribution-v1` output schema, bucketing each feature into exactly one of robustly_attributed/uncertain/dag_sensitive/insufficient_evidence.
  - `examples/quietset_label_instability.py` (Phase 6): runnable CLI wiring Phases 1–5 (`--bundle` or `--observations`/`--scored`, `--dag` repeatable, `--mode global/local`). Never subprocesses quietset. See `docs/integrations/quietset_label_instability.md` for the bundle manifest schema and non-causal-claim framing.
- Type stubs: `py/causasv/py.typed` (PEP 561 marker) + `py/causasv/causasv.pyi` covering `CausalDAG`/`ASVExplainer`'s public methods, so mypy/pyright resolve real types instead of `Any` at the FFI boundary.
- `docs/comparison_shap.md`: quantitative runtime and attribution comparison against SHAP `KernelExplainer`, with `examples/compare_causasv_shap.py`.

### Changed
- `causasv.explain_quality(value_fn_batch=…)` now routes through `explain_quality_batch()` instead of `explain_adaptive_batch()`. The batch path now returns ESS = n_samples and uniform sparse CI bounds for n ≤ 63, instead of IS-weighted estimates.
- `auto()`/`auto_quality()`'s `n ≤ 20` dispatch branch now uses the same order-ideal BFS preflight as the other branches (budgeted at half the dense state count) instead of a stale `edge_count ≤ 2n` heuristic, which misjudged disconnected graphs (m=0 but up to 2ⁿ order ideals). Removes the now-unused `Dag::edge_count()`.
- `auto()`'s `20 < n ≤ 28` branch now preflights against a memory-based `sparse_state_budget()` before attempting `exact_dag_sparse`, falling back to `approximate` with `fallback_reason` set when infeasible, instead of relying on failure after the fact.
- IS/uniform sampling loops (serial, parallel, batched, adaptive, uniform — 10 call sites) now carry the previous step's cached mask forward instead of re-querying it, cutting cache lookups from 2n to n+1 per sample. Measured −17% to −36% (p<0.05) across approx benchmarks with the in-repo `v(S) = |S|`; see `docs/benchmarks.md` for the caveat that this is an upper bound when the caller's `value_fn` dominates cost.
- `normal_quantile` (Beasley-Springer-Moro normal quantile approximation) moved from `python.rs` into `numerics.rs`, gated behind the `python` feature, so it can be unit-tested directly (previously PyO3-only, untestable under `extension-module`).
- `py/causasv/instability.py` (~1400 lines) split into a package (`py/causasv/instability/{bundle,modeling,value,sensitivity,report}.py`) by the phase it was built in. No behavior change — all existing import paths continue to resolve identically.

### Fixed
- `CausalDAG.from_json` (P0): the hand-rolled parser matched the literal substrings `"from":"`/`"to":"`/`"nodes":[` with no space, so any standard `json.dumps()` output (which uses `", "`/`": "` spacing) silently produced an **empty graph with no error** instead of failing loudly. Replaced with structural parsing via `serde_json` (now a direct dependency), which is whitespace/formatting-insensitive by construction. `to_json` was also rewritten via `serde_json::json!` for correct string escaping.
- `exact_tree` feasibility guard against combinatorial explosion (issue #36): `exact_tree`'s cost depends on tree *shape*, not node count — a modest-n (≈61) but wide/deep rooted tree could force `auto()`/`auto_quality()` to hang and eventually OOM. See `ExactTreeConfig` under Added. The default budget (50,000 single-node terms / 200,000 total) is calibrated against measured wall-clock time, not term counts alone: a balanced binary tree of height 4 (n=31) has "only" ~3.6M estimated terms but takes ~20-25s to actually run, since per-combination cost isn't O(1). Known follow-up, out of scope here: `approximate()` has its own undocumented n ≤ 64 bitmask cap, so a rooted tree with n > 64 currently has no working exact or approximate method.
- `cargo-audit`/`pip-audit` findings: `crossbeam-epoch` bumped (transitively, via `cargo update -p crossbeam-epoch --precise 0.9.20`) fixing RUSTSEC-2026-0204; CI's `setuptools`/`pip` upgraded before running `pip-audit`, fixing PYSEC-2026-3447; `ruff` pinned to 0.15.20 in both CI and `py/pyproject.toml`'s dev extra after an unpinned install picked up a new lint rule and took down every Python CI job at once.
- `_validate_cv_folds` (instability `modeling.py`/`value.py`): `cv_folds <= total groups` wasn't sufficient for `StratifiedGroupKFold`, which also needs ≥1 group per class per fold — a small/imbalanced binary target could pass the first check and still crash inside sklearn. Now checked once up front and shared by `fit_instability_model` and `make_global_value_fn`.

### Dependencies
- `rand` 0.10.1 → 0.10.2, `thiserror` 2.0.18 → 2.0.19 (patch bumps).
- `actions/upload-artifact` 4 → 7, `actions/download-artifact` 4 → 8, `actions/setup-python` 6 → 7 (GitHub Actions).

### Tests
- `tests/golden_tests.rs`: algebraic identity tests — `v(S) = Σwᵢ → ASV_i = wᵢ` verified at < 1e-10 tolerance across all three exact code paths (exact, exact_dag, exact_dag_sparse) on chain / fork / collider topologies.
- `tests/approx_batch_tests.rs`: batch accuracy vs exact_dag (diamond, weighted v), ESS = n_samples invariant, convergence flag, additive identity.
- `tests/uniform_sampler_tests.rs`: non-additive `v(S) = |S|²` test for `approximate_uniform_sparse_adaptive` (verifies correctness beyond additive identity).
- `tests/property_tests.rs`: `prop_approx_matches_exact_dag_per_node` — random DAGs × non-additive `v(S) = |S|²`, per-node error vs `exact_dag` (errors can't cancel in the sum, unlike the existing efficiency-axiom check).
- `tests/approx_accuracy_tests.rs`: `test_adaptive_ci_coverage_additive` — 30-seed empirical 95% CI coverage check (≥0.75 coverage rate) on the Rust side, mirroring `py/tests/test_ci_coverage.py`.
- `tests/exact_tree_feasibility_tests.rs` (issue #36): default-budget and custom-budget rejection/acceptance cases, `auto()`/`auto_quality()` fallback-chain behavior when `exact_tree` is infeasible.
- `py/tests/test_ci_coverage.py`: empirical 95% CI coverage check on 25-node dense DAG (forces uniform_sparse_adaptive path); split into quick (10 seeds, CI) and `@pytest.mark.slow` (30 seeds, skipped by default).
- `py/tests/test_diagnostics.py`: key-presence contract test — both exact and approximate paths must return all required dict keys.
- `py/tests/test_instability.py`: full 15-item spec coverage for the `causasv.instability` package (byte-identical JSON, grouped-CV non-leakage, leakage-guard rejection, malformed/cyclic DAG rejection, CI/ESS reporting, seed and cross-DAG rank stability, end-to-end CLI subprocess test), plus a contract test asserting the Phase-1→6 package split preserves the full prior flat-module symbol surface.

### CI
- `.github/workflows/ci.yml`: add `python -m py_compile` syntax sweep over all `.py` files before maturin build; add `examples/benchmark_corpus.py` run step alongside `quality_workflow.py`; add `ruff check` step; Rust job now runs across ubuntu/macos/windows instead of ubuntu-only; Python job now tests 3.9–3.13 instead of 3.11-only.
- `py/pyproject.toml`: register `slow` pytest marker; `addopts = "-m 'not slow'"` excludes slow tests from default CI run; add dynamic metadata fields and classifiers.
- Add `cargo-llvm-cov`/`pytest-cov` coverage measurement (report-only, no threshold gate).
- Add `cargo-deny check licenses` as a blocking CI gate, with an allow-list in `deny.toml` built from the dependency tree's actual licenses (MIT, Apache-2.0, Apache-2.0 WITH LLVM-exception, Unicode-3.0).
- `release.yml`: Linux release wheels now build both `x86_64` and `aarch64` targets.

### Docs
- `docs/comparison_shap.md`: add caveat noting the runtime comparison reflects DAG-known sparse conditions (chain DAG, exact sparse DP) where causasv's advantage is largest.
- Add module-level and per-item rustdoc to `CausasvError`, `Dag`, and the `topo` module.
- `CONTRIBUTING.md`: development setup and pre-PR checklist (fmt, clippy, ruff, tests).
- README (all 3 languages): `exact_tree`/`ExactTreeConfig` feasibility-guard documentation (issue #36), `auto` dispatch text corrections, `explain_safe()`/`TabularExplainer` baseline docs, `docs/integrations/quietset_label_instability.md`.

---

## [0.8.5] — 2026-06

### Added
- `AsvExplainer::auto_quality()`: quality-first dispatch — exact when feasible, then `approximate_uniform_sparse_adaptive` (ESS = n_samples, zero IS variance, stderr + CI always returned); IS adaptive only for n > 63
- `ASVExplainer.explain_quality()` Python method: returns `values`, `stderr`, `ci_low`, `ci_high`, `selected_method`, `converged`, `fallback_reason` in one call
- `causasv.explain_quality()` top-level Python helper: routes `value_fn` to `explain_quality()`, `value_fn_batch` to `explain_adaptive_batch()`; CI computed without scipy
- `causasv.plot` module: `plot.bar(values)` and `plot.waterfall(values, base_value=...)` with matplotlib as optional dependency

### Fixed
- `helpers.py`: CI quantile bug — `_normal_quantile(ci)` → `_normal_quantile((1.0 + ci) / 2.0)` in the batched path; ci=0.95 now correctly gives z≈1.96 instead of z≈1.64
- `auto_quality()` n≤63 branch: add `estimate_sparse_feasible(250k)` preflight before `exact_dag_sparse_with_config()`; dense DAGs no longer run a 2GiB BFS before falling back

### Docs
- README × 3: "When to use causasv" guidance section (Use / Do not use)
- README × 3: top Python example replaced with `explain_quality(…, ci=0.95)` pattern; `explainer.explain()` demoted to "Lower-level API"

### CI
- `.github/workflows/ci.yml`: Python job now runs `py/tests/smoke_test.py` — exercises `explain_quality`, CI bound ordering, and `plot` import; catches public-API regressions before merge

---

## [0.8.4] — 2026-06

### Added
- `approximate_uniform_sparse()`: uniform topological ordering sampler for sparse DAGs with n > 20; uses lazily memoized `dp_ind` (HashMap) instead of the 2^n precomputed slice — ESS = n_samples exactly (no IS weight variance), memory-bounded at 2 GiB
- `approximate_uniform_sparse_adaptive()`: adaptive variant with automatic convergence stopping (rel_tol), per-node stderr, and CI support; uniform weights mean no ESS gate — simpler convergence than IS adaptive
- Python `method="uniform_sparse"` in `explain()` and `explain_with_diagnostics()` 
- Python `explain_adaptive(method="uniform_sparse")` for adaptive uniform sparse with CI

### Performance
- `approx.rs`: seeded single-threaded `approximate_asv` path converted to one-pass streaming with incremental `global_max_log_w` rescaling (same as batched paths); eliminates `samples: Vec<SampledOrdering>` and all `ordering.clone()` calls — last per-sample allocation removed
- `auto()` dispatch extended to 28 < n ≤ 63: new `estimate_sparse_feasible()` BFS preflight (no dp_ind, no value_fn) counts order ideals up to 250k budget; sparse chains/trees with n > 28 now get exact results instead of falling to approx

### Internals
- `dag_dp_sparse.rs`: add `estimate_sparse_feasible(dag, parents_mask, budget) -> bool` — fast preflight BFS used by auto() for the n > 28 branch
- `dag_dp_sparse.rs`: add `dp_ind_lazy_pub()` wrapper to expose `dp_ind_lazy` to `sampler.rs` for sparse uniform sampling
- `sampler.rs`: add `sample_uniform_sparse_into()` — uniform topological sampler using lazy dp_ind HashMap

### Tests
- `tests/uniform_sampler_tests.rs`: 5 new tests — ESS == n_samples (chain/diamond/fork/two-parallel-chains), exact agreement with `exact()` on small DAGs (≤ 5% error), adaptive convergence check, memory limit behavior

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
