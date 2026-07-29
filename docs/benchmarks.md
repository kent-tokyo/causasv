# causasv — Benchmark Results

Machine: Apple M-series (arm64), release build (`cargo bench`).
Value function: `v(S) = |S|` (additive) unless noted.
Reproduce: `cargo bench` from the repository root.

---

## Exact methods

### exact (brute-force) vs exact_tree (order-ideal DP)

| DAG | n | Method | Time |
|-----|---|--------|------|
| Chain | 7 | `exact` (brute-force) | 2.7 µs |
| Balanced binary tree | 7 | `exact` (brute-force) | 39.9 µs |
| Balanced binary tree | 7 | `exact_tree` (DP) | 51.6 µs |
| Balanced binary tree | 15 | `exact_tree` (DP) | 2.79 ms |
| Caterpillar | 10 | `exact_tree` (DP) | 169 µs |

For n ≤ ~8, `exact` is faster than `exact_tree` due to lower allocation overhead.
For rooted trees at n ≥ 10, brute-force `exact` is no longer practical, making
`exact_tree` the only exact method worth considering — but `exact_tree`'s own
feasibility is shape-dependent, not just a function of n: its cost is the
product of every ancestor level's side-sibling order-ideal count, so a rooted
tree can be rejected by `ExactTreeConfig`'s default budget (50,000 / 200,000)
even at a small or moderate n if it branches widely and deeply enough. A
complete binary tree of height 4 (n=31, no unusual branching) is already
rejected — its deepest leaf reaches 176,020 combinations and takes ~20-25s to
actually run. See [correctness.md](correctness.md#exact-method-bounds) and
[issue #36](https://github.com/kent-tokyo/causasv/issues/36).

### exact_dag (dense order-ideal DP, O(2^n × n))

| DAG | n | Time |
|-----|---|------|
| Chain | 10 | 27.7 µs |
| Two parallel chains | 12 | 163 µs |
| Diamond (src + 8 mid + snk) | 10 | 127 µs |
| Chain | 16 | 5.28 ms |

### exact_dag_sparse (BFS over valid order ideals)

| DAG | n | Order ideals visited | Time |
|-----|---|---------------------|------|
| Chain | 24 | 25 of 2^24 = 16 M | 14.9 µs |

A chain has exactly n+1 valid order ideals — sparse DP is maximally efficient here.

### exact_dag vs exact_dag_sparse — direct comparison

| DAG | n | Method | States visited | Time |
|-----|---|--------|----------------|------|
| Two parallel chains | 20 | `exact_dag` (dense) | 2^20 = 1,048,576 | 87.9 ms |
| Two parallel chains | 20 | `exact_dag_sparse` | (10+1)² = 121 | 91 µs |

**~1000× speedup** for a DAG with highly constrained order ideals.
Two parallel chains of k each have exactly (k+1)² valid order ideals.

---

## Approximate methods

### Standard approximate (IS-weighted sampling)

| DAG | n | Samples | Time |
|-----|---|---------|------|
| Chain | 10 | 1k | 916 µs |
| Balanced tree | 15 | 1k | 1.94 ms |

### Batched vs normal (chain n=10, 1k samples)

| Method | batch_size | Time | vs normal |
|--------|-----------|------|-----------|
| `approximate` | — | 881 µs | baseline |
| `approximate_batched` | 256 | 815 µs | −7% |

The pure-Rust gain is modest (coalition deduplication). In Python, the real benefit
is GIL reacquisition: `batch_size=256` reduces Python call overhead from O(n_samples × n)
to O(n_samples / batch_size).

### Seeded deterministic parallel (chain n=20, 10k samples)

| Mode | Threads | Time | Speedup vs serial |
|------|---------|------|-------------------|
| Serial seeded | 1 | 18.2 ms | 1.0× |
| Parallel seeded | 2 | 12.0 ms | 1.5× |
| Parallel seeded | 4 | 7.4 ms | 2.5× |

Same seed + same num_threads → bitwise-identical results across runs.

### Fixed-sample vs adaptive approximation (chain n=10, max 10k samples)

| Method | Max samples | Time | Notes |
|--------|------------|------|-------|
| `approximate` | 10,000 (fixed) | 7.4 ms | always runs all samples |
| `approximate_adaptive` | 10,000 (max) | 1.5 ms | stops early on convergence |

**~5× speedup** when the estimates converge before the sample budget is exhausted.
`approximate_adaptive` stops when `max_rel_change < rel_tol` AND `ess_ratio >= ess_ratio_min`
(defaults: rel_tol=0.01, ess_ratio_min=0.10). For an additive value function on a chain,
convergence is fast and the adaptive method uses far fewer samples.

### Large-DAG approximate paths (n > 64) — coalition-representation boundary

`approximate`/`approximate_adaptive`/`approximate_batched`/`approximate_adaptive_batched`
switch from a `u64` bitmask coalition to a growable word-vector bitset
(`LargeCoalition`, `src/coalition.rs`) for n > 64 — see
[correctness.md](correctness.md#large-dag-approximate-paths-n--64). These
numbers check that switch doesn't introduce a discontinuous cost jump. All
numbers below are from a single, internally-consistent measurement pass
(every row below from the same `cargo bench` invocation, back to back) —
**`cargo bench` on this machine is noisy enough between separate sessions
that absolute numbers and even boundary percentages should not be trusted to
more than roughly a factor of 2.** Concretely: an earlier measurement pass of
this exact serial-path code (unchanged since) recorded a 64→65 step of +37%;
this pass recorded +75% for the same code path. Treat "cost grows smoothly,
no cliff at n=65" as the reliable qualitative claim; treat specific
percentages as one machine's one snapshot, not a guarantee.

#### Chain, serial seeded (2k samples)

| n | Backend | Time |
|---|---------|------|
| 64 | `u64` | 2.48 ms |
| 65 | `LargeCoalition` | 4.33 ms |
| 128 | `LargeCoalition` | 8.55 ms |
| 256 | `LargeCoalition` | 22.0 ms |

64→65 is a **+75% step** in this pass (measured at +37% in an earlier
session — see the noise note above), then cost grows roughly linearly with n
from there — no cliff either time. The coalition buffer itself is reused
across samples (not reallocated — see `LargeCoalition::clear()`), so this
step is not allocation overhead; it's hashing/comparing a `&[u64]` slice on
every cache lookup instead of a bare `u64`, which is unavoidable once
addressing more than 64 nodes needs more than one machine word.

#### Chain, seeded parallel (4 threads, 10k samples)

| n | Backend | Time |
|---|---------|------|
| 64 | `u64` | 6.51 ms |
| 65 | `LargeCoalition` | 11.5 ms |

+77% here, in the same range as this pass's serial step (+75%): each of the
4 workers builds its own `LargeCoalitionCache`, so the same per-lookup
overhead is paid 4× over, once per worker (see the "per cache instance, not
per aggregate" note in correctness.md). Still a smooth, explainable step,
not a cliff.

#### Chain, adaptive (single-threaded, max 2k samples)

| n | Time |
|---|------|
| 64 | 2.86 ms |
| 65 | 3.72 ms |
| 128 | 7.73 ms |

64→65 here is +30% — smaller than the serial path's +75% in this same pass,
despite near-identical per-sample mechanics (both single-threaded,
rescale+Kahan). This gap is itself an illustration of the noise this section
warns about: the absolute deltas involved are low-single-digit milliseconds,
well within this machine's run-to-run variance.

#### Chain, batched (batch_size=256, 2k samples)

| n | Time |
|---|------|
| 64 | 3.36 ms |
| 65 | 7.75 ms |
| 128 | 16.3 ms |

+131% here — this is the one boundary percentage that *has* reproduced
consistently across measurement sessions (+133% previously, +131% now),
unlike the smaller non-batched deltas above. The n > 64 batched path now
shares one bounded `LargeCoalitionCache` across every round of a call (same
admission-capped design as the non-batched paths), so on a chain (single
valid ordering, same n+1 coalitions revisited every round) `value_fn_batch`
is called once total, not once per round — see
`persistent_cache_collapses_batched_calls_on_chain` in `src/approx_large.rs`
and the mechanism writeup in correctness.md. That said, the boundary jump
measured here does **not** shrink versus a round-scoped cache, because this
benchmark's callback is trivial: the per-round cost is dominated by
coalition-key bookkeeping (snapshotting, sorting, and deduping
`batch_size × (n+1)` keys), which runs every round regardless of whether the
value underneath is already cached — and that bookkeeping cost is what
reproduces so consistently here, not callback cost. The call-count reduction
is a real win specifically for expensive value functions (a real Python
model, costing milliseconds per call) — confirmed with a controlled
same-process comparison in correctness.md — just not one this
trivial-callback benchmark can show.

#### Chain, adaptive batched (batch_size=256, max 2k samples)

| n | Time |
|---|------|
| 64 | 3.68 ms |
| 65 | 7.80 ms |
| 128 | 17.9 ms |

+112% — same driver as the non-adaptive batched path above.

#### Non-chain shapes (genuine IS variance, not a single deterministic ordering)

| Shape | n | Backend | Time (2k samples, serial seeded) |
|-------|---|---------|-----------------------------------|
| Diamond-chain (layered, collider sinks) | 64 | `u64` | 2.53 ms |
| Diamond-chain (layered, collider sinks) | 127 | `LargeCoalition` | 6.79 ms |
| Caterpillar (fork-chain: chain + 1 leaf/node) | 66 | `LargeCoalition` | 3.50 ms |
| Caterpillar (fork-chain: chain + 1 leaf/node) | 128 | `LargeCoalition` | 7.12 ms |

Caterpillar costs noticeably more per node than the pure chain at a
comparable n (3.50 ms/66 vs 4.33 ms/65 — note the chain figure here is *n=65*,
one node larger): every main-chain node has a real 2-way sampling choice
(continue the chain or take the leaf), so distinct samples visit more
distinct prefixes and the coalition cache is warm less often — a real,
expected difference in *workload*, not a representation-cost artifact.
Unlike the chain, this shape doesn't maximize repetition across rounds the
same way (above), since distinct samples share less structure with each
other than a chain's samples do.

#### Cache boundedness (not a Criterion benchmark — see `src/approx_large.rs`)

`LargeCoalitionCache::len()` is asserted `<= max_entries` after 5,000 samples
on a fully disconnected (antichain) 80-node DAG — the worst case for cache
growth, since no two samples share any prefix past the empty coalition. This
confirms memory does not scale with total sample count, only with the
configured cap (`DEFAULT_LARGE_CACHE_MAX_ENTRIES`, currently 50,000 per cache
instance — see correctness.md for why this is per-instance, not aggregate,
under the parallel paths).

#### Batched-path cache: shared across rounds, still bounded

The batched large paths share one `LargeCoalitionCache` across every round of
a call (admission-capped, same as the non-batched paths — see correctness.md),
so memory is bounded by the cache's entry cap plus `n × batch_size` for the
round in flight, not by the run's total sample count. An internal test
(`batched_chunking_does_not_change_additive_result`, `src/approx_large.rs`)
forces one round's unique-coalition count past
`MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH` (4,096) on a 70-node antichain and
confirms the result is unaffected by the resulting chunked `value_fn_batch`
calls; a second test (`persistent_cache_collapses_batched_calls_on_chain`)
confirms the cache is actually reused across rounds, not just bounded within
one.

Reproduce the timing numbers above with:
```
cargo bench --bench asv_bench -- 'approx_boundary_chain|approx_diamond_chain_large|approx_caterpillar_large'
```

### Prefix-mask cache lookup dedup (2026-07-03)

Every sampling loop looked up both `without` (the coalition before adding the current
node) and `with_node` (after) on each step of a sampled linear extension — but `without`
at step *t* is always the same mask as `with_node` at step *t-1*, already computed and
cached one step earlier. Carrying that value forward in a local variable instead of
re-querying the cache cuts lookups from 2n to n+1 per sample, across all IS/uniform
sampling variants (serial, parallel, batched, adaptive).

| Benchmark | Change (median) |
|-----------|-----------------|
| `approx_chain_10_1k` | −30.8% |
| `approx_balanced_tree_15_1k` | −28.0% |
| `approx_vs_batched_chain_10_1k/normal` | −30.8% |
| `approx_vs_batched_chain_10_1k/batched_b256` | −22.3% |
| `approx_chain_20_10k_parallel/serial_seeded` | −31.5% |
| `approx_chain_20_10k_parallel/parallel_2t` | −26.4% |
| `approx_chain_20_10k_parallel/parallel_4t` | −25.9% |
| `approx_diamond_10_10k_seeded` | −28.9% |
| `approx_balanced_tree_15_10k_seeded` | −31.1% |
| `approx_chain_20_10k_seeded` | −34.6% |
| `approx_balanced_tree_31_10k_seeded` | −17.2% |
| `approx_vs_adaptive_chain_10/fixed_10k` | −35.9% |
| `approx_vs_adaptive_chain_10/adaptive_max10k` | −27.9% |
| `approx_vs_uniform_diamond_10_1k/frontier_IS_1k` | −26.7% |
| `approx_vs_uniform_diamond_10_1k/uniform_1k` | −27.8% |

All changes are statistically significant (p < 0.05, Criterion `--baseline` comparison,
100 samples each). **Caveat**: these use the cheap in-repo value function `v(S) = |S|`,
so the measured win is essentially pure HashMap-lookup overhead — it's the upper bound.
When the caller's `value_fn` is expensive (e.g. a Python model callback), that cost
dominates and this fix contributes proportionally less to wall-clock time.

---

## Notes

- All timings are wall-clock median from Criterion (100 samples, 3 s warmup).
- `exact_dag` and `exact_dag_sparse` both benefit from the `parents_mask` cache in `AsvExplainer::new()` (precomputed once, shared across calls).
- `approx_chain_10_1k` and `approx_balanced_tree_15_1k` improved ~10% vs the pre-PR-1 baseline, attributable to Kahan summation consistency across code paths.
- Criterion HTML reports are saved to `target/criterion/`.
