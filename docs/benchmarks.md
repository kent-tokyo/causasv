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
`exact_tree` becomes the only feasible exact method at n ≥ 10.

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

---

## Notes

- All timings are wall-clock median from Criterion (100 samples, 3 s warmup).
- `exact_dag` and `exact_dag_sparse` both benefit from the `parents_mask` cache in `AsvExplainer::new()` (precomputed once, shared across calls).
- `approx_chain_10_1k` and `approx_balanced_tree_15_1k` improved ~10% vs the pre-PR-1 baseline, attributable to Kahan summation consistency across code paths.
- Criterion HTML reports are saved to `target/criterion/`.
