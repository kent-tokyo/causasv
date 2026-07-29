# Roadmap

## Done (through this PR, on top of v0.8.8)

Core ASV computation:
- `exact` — brute-force enumeration of all linear extensions; oracle for n ≤ ~8
- `exact_tree` — order-ideal DP for rooted directed trees; hook-length weighting;
  `ExactTreeConfig`/`exact_tree_with_config` feasibility preflight against
  shape-driven (not just node-count-driven) combinatorial cost
- `exact_dag` — dense order-ideal DP for general DAGs, n ≤ 20; O(2^n × n)
- `exact_dag_sparse` — BFS over valid order ideals only, n ≤ 63; memory-bounded
- `auto` / `auto_quality` dispatch — exact → exact_tree → exact_dag →
  exact_dag_sparse → approx, with `fallback_from`/`fallback_reason`/`selected_method`
  diagnostics on every fallback

Approximate ASV:
- Self-normalized importance sampling over topological orderings — **no node-count
  limit**, including `auto`/`auto_quality`'s fallback into it. Coalitions use a
  plain `u64` bitmask for n ≤ 64 and a growable word-vector bitset
  (`src/coalition.rs`) for n > 64, with a bounded coalition→value cache on the
  large-n path (see [docs/correctness.md](docs/correctness.md#large-dag-approximate-paths-n--64)).
  `exact_dag_sparse`/`uniform_sparse`/`uniform_sparse_adaptive` remain n ≤ 63 —
  that ceiling is structural (they still pack a coalition into a `u64`), not a
  preflight choice, and is unaffected by the above.
- Kahan summation and log-weight normalization across all IS paths
- Adaptive approximation with convergence criteria, per-feature stderr, and CI
- Batched coalition evaluation (`value_fn_batch`) — reduces Python GIL acquisitions
- Seeded deterministic parallel sampling (splitmix64 per-worker seeds)
- Uniform (zero-IS-variance) sampling for sparse DAGs, n ≤ 63, with adaptive
  variant and batched adaptive variant
- ESS diagnostics, overflow guard (`u64::checked_add`), property-based tests

CPDAG / graph reduction:
- `Cpdag` — PDAG representation, `validate_pdag`/`validate_cpdag`,
  `consistent_extension` (Dor-Tarsi/Chickering PDAG-to-DAG)
- `Dag::d_convex_hull` / `Dag::strong_d_convex_hull` and
  `Cpdag::strong_d_convex_hull` (CVM/ICHA/ISCHA per Deng/Sun/Li/Liu,
  arXiv:2606.08941) — minimal-set graph reduction preserving a causal-effect
  estimate after marginalizing out every other variable
- `induced_subgraph` on both `Dag` and `Cpdag`

Python:
- `ASVExplainer`, `CausalDAG`, `CausalCPDAG`, `TabularExplainer`, `ASVEnsembleExplainer`
- `explain_quality()`, `explain_safe()`, `explain_stability()`,
  `explain_adaptive()`, `explain_adaptive_batch()`, `explain_quality_batch()`
- `make_tabular_value_fn()`
- DAG: `ancestors()`, `descendants()`, `topological_layers()`, `to_json()`,
  `from_json()`, `to_dot()`, `from_networkx()`, `inspect()`
- Graph export: DOT, JSON, networkx

Docs and process:
- `docs/benchmarks.md`, `docs/correctness.md`, `docs/comparison_shap.md`,
  `docs/benchmark_corpus.md`, `docs/strong_d_convex_hulls.md`, `CHANGELOG.md`, `examples/`

## Now (stabilizing)

No new algorithms planned in the immediate term. Focus: correctness evidence,
diagnostics, docs, and closing gaps between what the public API promises and
what it actually does (the n > 64 approximate-path gap fixed by this PR was
one such gap; the CI version-sync check against `README.md`'s status line is
another example of this ongoing work).

## Next candidates

Only if there is clear user demand, roughly in priority order:

1. **Strict CPDAG validation / canonicalization** — `Cpdag::validate_cpdag`'s
   current scope is narrower than "these are exactly the compelled edges of a
   genuine Markov equivalence class"; tightening that is a separate, scoped
   piece of work.
2. **Reduced-subgraph IDA** as an optional, separately scoped module — the
   paper's IDA-style effect estimation on top of the strong d-convex hull
   reduction is a real scope jump (statistical estimation, likely a new
   `numpy`-class dependency), deliberately not started yet.
3. **Linux aarch64 wheel** reconsideration — the cross-build under QEMU was
   disabled (broken); revisit if there's real demand for it.
4. Optional stderr for fixed-sample `approx` (currently only `approx_adaptive`
   returns it)
5. Parallel log-weight normalization for the seeded-parallel `approx` path
   (currently per-worker weights aren't rescaled against each other — see
   `src/approx.rs`'s seeded-parallel branch comment; would need a two-pass
   global max and careful validity argument)
6. `Dag::edges()` as a stable first-class public API

## Non-goals (permanently out of scope)

Causal discovery, model training, GPU acceleration, deep learning-specific
explainability, automatic graph construction, Web UI.
