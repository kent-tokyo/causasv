# causasv — Correctness Evidence

This document explains how causasv verifies ASV correctness, and how to
interpret and trust approximate results.

---

## ASV axioms and property-based tests

The five core ASV axioms are verified by proptest property-based tests
in `tests/property_tests.rs` (10 tests, random DAG generation via `arb_dag`).

| Axiom | What it says | Test |
|-------|-------------|------|
| **Efficiency** | Σ φᵢ = v(V) − v(∅) | `prop_efficiency_exact`, `prop_efficiency_exact_dag`, `prop_efficiency_exact_tree`, `prop_efficiency_approx` |
| **Dummy** | If v(S ∪ {i}) = v(S) for all S, then φᵢ = 0 | `prop_dummy_zero_value_function` |
| **Additivity** | φᵢ(v + w) = φᵢ(v) + φᵢ(w) | `prop_additivity` |
| **Relabeling invariance** | Permuting node labels permutes values | `prop_relabeling_invariance` |
| **Consistency across methods** | exact ≈ exact_dag ≈ exact_dag_sparse ≈ exact_tree | `prop_exact_matches_exact_dag`, `prop_exact_tree_matches_exact`, `prop_exact_dag_sparse_matches_exact_dag` |

Self-normalized IS (`approx`) preserves the **efficiency** axiom exactly because
numerator and denominator share the same weight sum: Σ(wᵢ Δᵢ) / Σwᵢ.

---

## Why the frontier sampler + self-normalized IS is correct

The frontier sampler draws topological orderings non-uniformly: at each step it
picks uniformly among currently-available nodes (in-degree-zero in the remaining
subgraph). This introduces a sampling bias — some orderings are more likely than
others.

Self-normalized importance sampling (SNIS) corrects for this. Each ordering π is
drawn with probability q(π). Its IS weight is wᵢ = 1/q(π). The SNIS estimator:

```
φᵢ ≈ Σ_π [wᵢ (v(pre(i,π) ∪ {i}) − v(pre(i,π)))] / Σ_π wᵢ
```

converges to the true uniform-over-orderings average as n_samples → ∞, regardless
of q (as long as q(π) > 0 for all valid orderings, which the frontier sampler
guarantees). The efficiency axiom holds exactly for any finite sample, not just
in expectation.

**Log-weight normalization:** All IS paths subtract `max(log_q)` before `exp()`
to prevent float overflow on extreme frontier distributions. Since SNIS is invariant
to a common scale factor on all weights, this does not change ASV values.

---

## Interpreting approximate results

### Effective Sample Size (ESS)

ESS = (Σw)² / Σw² estimates how many independent samples the weighted sample
corresponds to. ESS ≈ n_samples means weights are nearly uniform (reliable).
ESS ≪ n_samples means a few orderings dominate (high variance).

**Rule of thumb:** ESS/n_samples ≥ 0.1 before trusting rankings.

### Standard error and confidence intervals

`explain_adaptive()` with `ci=0.95` returns per-feature `stderr`, `ci_low`,
`ci_high` using a normal approximation:

```
ci_low  = φᵢ − z₀.₉₇₅ × stderr
ci_high = φᵢ + z₀.₉₇₅ × stderr
```

For fixed-sample `approx`, stderr is not computed (use `explain_adaptive()` if
you need it).

### Approximation diagnostics checklist

Before trusting an approximate result:

1. **Check ESS ratio.** `info["ess_ratio"]` should be ≥ 0.1.
2. **Run seed stability.** Call `explain_stability(explainer, value_fn, seeds=[...])` and
   check `rank_stability ≥ 0.9` (Kendall tau). See `examples/stability_diagnostics.py`.
3. **Increase n_samples** until rankings stop changing. `explain_adaptive()` does this
   automatically (stops when `max_rel_change < rel_tol`).
4. **Use CI for borderline features.** If two features have overlapping `ci_low`/`ci_high`,
   their relative order is not statistically reliable.

---

## Exact method bounds

| Method | n limit | States visited | Notes |
|--------|---------|----------------|-------|
| `exact` | ~8 | All L(G) orderings | Exponential in L(G); use only for small graphs |
| `exact_tree` | shape-dependent | up to `ExactTreeConfig` budget | Only for rooted directed trees — see below |
| `exact_dag` | 20 | 2ⁿ bitmasks | O(2ⁿ × n) time; ~16 MB for n=20 |
| `exact_dag_sparse` | 28 | ≤ 2ⁿ order ideals (BFS) | Much faster for sparse DAGs; memory-bounded (default 2 GiB) |

`auto()` selects the method automatically and reports what it chose in
`info["selected_method"]`. If `exact_dag_sparse` hits the memory or overflow
limit, it falls back to `approx` and sets `info["fallback_from"]`.

## Large-DAG approximate paths (n > 64)

`approx` / `approx_adaptive` / `approx_batched` / `approx_adaptive_batch` have
no node-count limit. The frontier sampler that drives all of them
(`sample_one_into`) never represents a coalition as a bitmask at all — it only
needs `Vec`-sized scratch space — so the only place a coalition needs a
concrete representation is where a sample's growing prefix is turned into the
`Vec<NodeId>` handed to a value function, and where that prefix is used as a
cache key.

- For n ≤ 64, that representation is a single `u64` (`1u64 << node.0`).
- For n > 64, it is a growable word-vector bitset (`LargeCoalition` in
  `src/coalition.rs`): `ceil(n / 64)` `u64` words, word `w` holding the bits
  for nodes `[64w, 64w+64)`. Ascending `NodeId` order falls out of the
  word/bit layout directly (word index, then bit index), so there is no
  `HashSet`/`HashMap` iteration order to leak into the coalitions a value
  function sees.

Both representations are fed by the *same* sampler and the *same*
self-normalized IS math (log-weight rescaling, Kahan summation, ESS/stderr
formulas) — only the coalition type and its cache differ. Because of that,
correctness for n > 64 is verified two ways:

1. **Backend parity** (`src/approx_large.rs`, internal `#[cfg(test)]` tests,
   plus a `proptest` generalization): on the *same* n ≤ 64 DAG, same seed,
   same sampling order, the n ≤ 64 (`u64`) and n > 64 (`LargeCoalition`)
   backends must agree — bitwise, for the serial-seeded, adaptive-serial,
   batched, and adaptive-batched paths alike. This holds even though the
   batched paths' caching differs between backends (see below) because
   caching only changes how many times a value is *computed*, never what it
   computes to, and the underlying value function is deterministic. This is
   the primary correctness oracle: there is no independent *exact* method for
   n > 64 to compare against (`exact_dag_sparse` and `uniform_sparse` both
   still require n ≤ 63 — see below).
2. **Closed-form additive check** (`tests/large_dag_approx_tests.rs`): for
   `v(S) = |S|`, the true ASV is exactly 1.0 per node on *any* DAG, so a
   65/128/256-node chain gives a direct accuracy check without needing an
   oracle at all.

**Bounded, not unbounded, caching.** The n ≤ 64 path's `HashMap<u64, f64>`
cache is implicitly bounded (at most `2^n` distinct keys). A `Box<[u64]>`-keyed
cache has no such ceiling, and for large, sparsely-branching DAGs, distinct IS
samples rarely revisit the same prefix past the first couple of steps — an
unbounded cache would grow roughly `n` entries per sample at a vanishing hit
rate. `approximate`/`approximate_adaptive` use a `LargeCoalitionCache` capped
at a fixed entry count *per cache instance* (lookups keep working past the
cap; new inserts are just skipped, so correctness never depends on the cap).
"Per instance" matters for the parallel paths: seeded-parallel builds one
cache per worker thread, so aggregate memory is (roughly) the cap × thread
count; unseeded-parallel builds one cache per Rayon fold split, which is not
guaranteed to equal the thread count.

The batched paths (`approximate_batched`/`approximate_adaptive_batched`) share
one `LargeCoalitionCache` across every sampling round of a call, admission-capped
the same way as the non-batched paths (lookups always work; new inserts are
declined once the cap is hit; nothing is ever evicted). Each round still does
its own dedup first (`sort_unstable` + `dedup` on that round's sampled
prefixes) so a round's repeated coalitions never query the cache twice for the
same key — only the *first* time a coalition is seen across the whole run
reaches `value_fn_batch`. On a DAG shape with high structural repetition
across rounds (a chain has only one valid ordering, so every round revisits
the *same* n+1 coalitions), this collapses `value_fn_batch` traffic from once
per round to once *total*: a dedicated test
(`persistent_cache_collapses_batched_calls_on_chain`, `src/approx_large.rs`)
drives a 65-node chain through 10 rounds and asserts exactly one call reaches
the batch value function.

That call-count reduction is a real win for an expensive value function (a
Python model callback measured in milliseconds), but it does **not** show up
in this crate's own Criterion benchmarks, because those use a cheap synthetic
callback — there, the dominant per-round cost is the coalition-key
bookkeeping itself (snapshotting, sorting, and deduping `batch_size × (n+1)`
`Box<[u64]>` keys every round), which runs regardless of whether the value
underneath is already cached. A controlled same-process comparison confirms
both ends of this: with a 50µs/call synthetic cost the persistent cache saves
under 2% of wall time (bookkeeping dominates), but with a 5ms/call cost — closer
to a real model-inference callback — it saves roughly (rounds − 1) × 5ms, i.e.
most of the added cost from repeated invocation. In other words: this change
helps exactly the case it was designed for (expensive value functions), and is
neutral on cheap ones. See docs/benchmarks.md for the Criterion numbers this
implies for `cargo bench`'s synthetic callback specifically.

See [docs/benchmarks.md](docs/benchmarks.md) for the measured n=64→65
boundary cost (a smooth increase, not a cliff — driven by hashing/comparing a
`&[u64]` slice on every cache lookup instead of a bare `u64`, since the
coalition buffer itself is reused across samples rather than reallocated) and
the cache-bound stress test in `src/approx_large.rs`.

**`exact_tree`'s cost is shape-dependent, not just a function of node count.**
A node's cost is the product of every ancestor level's side-sibling order-ideal
count, so a tree with several wide/deep branches can reach billions of
combinations at a modest `n` (a 61-node tree from a real report — see
[issue #36](https://github.com/kent-tokyo/causasv/issues/36) — hit ~8×10¹⁰).
Before calling `enumerate_order_ideals`, `exact_tree_with_config` runs an O(n),
allocation-free cost estimate (`estimate_tree_exact_cost`) and rejects with
`ExactTreeBudgetExceeded` if either the largest single node's cost or the total
summed over the tree exceeds the configured `ExactTreeConfig` budget (default:
50,000 / 200,000 — calibrated against measured wall-clock time, since
per-combination cost does not stay O(1) as combinations grow: a 31-node
balanced binary tree has "only" ~3.6M estimated total terms but takes ~20-25s
to actually run). `auto()`/`auto_quality()` fall back through
`exact_dag_sparse` and then `approximate`/`approximate_adaptive` when
`exact_tree` rejects a shape — never `approximate_uniform_sparse_adaptive` for
this fallback specifically, since its own internal memo has no comparable
budget yet and could grow unbounded on the same "dangerous" shape.
