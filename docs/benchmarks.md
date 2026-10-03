# Criterion benchmark snapshots

These results were collected on Apple M-series arm64 hardware with release
builds during v0.8.x development. Unless a table says otherwise, the value
function is the cheap additive function `v(S) = |S|`.

The tables answer narrow implementation questions. They do not establish
performance on another machine or model, and runtime does not establish
attribution quality. Rerun the relevant benchmark before making a release
claim.

```bash
cargo bench
```

Criterion reports are written to `target/criterion/`.

## Exact methods

### Brute force and rooted-tree DP

| DAG | n | Method | Time |
| --- | ---: | --- | ---: |
| Chain | 7 | `exact` | 2.7 us |
| Balanced binary tree | 7 | `exact` | 39.9 us |
| Balanced binary tree | 7 | `exact_tree` | 51.6 us |
| Balanced binary tree | 15 | `exact_tree` | 2.79 ms |
| Caterpillar | 10 | `exact_tree` | 169 us |

For very small graphs, brute force can be cheaper than the tree DP. Tree-DP
feasibility is shape-dependent: `ExactTreeConfig` rejects a tree before
enumeration when its estimated cartesian-product work exceeds the configured
budget. See [correctness.md](correctness.md#exact-method-bounds).

### Dense and sparse order-ideal DP

| DAG | n | Method | States | Time |
| --- | ---: | --- | ---: | ---: |
| Chain | 10 | `exact_dag` | `2^10` masks | 27.7 us |
| Chain | 16 | `exact_dag` | `2^16` masks | 5.28 ms |
| Chain | 24 | `exact_dag_sparse` | 25 ideals | 14.9 us |
| Two parallel chains | 20 | `exact_dag` | 1,048,576 masks | 87.9 ms |
| Two parallel chains | 20 | `exact_dag_sparse` | 121 ideals | 91 us |

Sparse DP is advantageous when the DAG admits far fewer order ideals than
`2^n`. The comparison is not a general speedup claim: weakly constrained DAGs
can approach the dense state count.

## Approximate methods

| Case | Samples | Time | Note |
| --- | ---: | ---: | --- |
| Chain, n=10, serial | 1,000 | 0.9 ms | fixed-sample IS |
| Balanced tree, n=15, serial | 1,000 | 1.9 ms | fixed-sample IS |
| Chain, n=20, serial seeded | 10,000 | 18.2 ms | one thread |
| Chain, n=20, seeded parallel | 10,000 | 7.4 ms | four threads |
| Chain, n=10, adaptive | up to 10,000 | 1.5 ms | stopped early on this additive case |

Seeded parallel results are deterministic for the same seed and worker count.
The current fixed-sample parallel path still needs a global log-weight
normalization audit for extreme frontier distributions; use serial or adaptive
sampling for that case. See [correctness.md](correctness.md#approximate-estimator).

Batched evaluation mainly reduces calls across the Python boundary. A cheap
Rust callback does not represent the benefit for a real model, so pure-Rust
batched timings should not be advertised as model-inference speedups.

## Coalition representation boundary

Approximate methods use one `u64` through 64 nodes and a growable word-vector
above 64 nodes. This snapshot checks the boundary with 2,000 seeded samples.

| Chain size | Backend | Serial time |
| ---: | --- | ---: |
| 64 | `u64` | 2.48 ms |
| 65 | large coalition | 4.33 ms |
| 128 | large coalition | 8.55 ms |
| 256 | large coalition | 22.0 ms |

The 64-to-65 step measured between roughly 37% and 75% in separate unchanged
runs on the same machine. Treat the qualitative result—no order-of-magnitude
cliff—as the supported observation. Do not quote one percentage as stable.

Other measured large-DAG shapes:

| Shape | n | Samples | Time |
| --- | ---: | ---: | ---: |
| Diamond chain | 64 | 2,000 | 2.53 ms |
| Diamond chain | 127 | 2,000 | 6.79 ms |
| Caterpillar | 66 | 2,000 | 3.50 ms |
| Caterpillar | 128 | 2,000 | 7.12 ms |

Different shapes produce different cache reuse and sampling variance, so the
rows are not interchangeable per-node cost estimates.

Reproduce the boundary group with:

```bash
cargo bench --bench asv_bench -- \
  'approx_boundary_chain|approx_diamond_chain_large|approx_caterpillar_large'
```

## Memory and callback checks

Tests complement wall-clock benchmarks:

- an 80-node antichain verifies that each large-coalition cache stays within
  its admission cap
- batched tests verify that results are unchanged when a batch is chunked
- a 65-node chain verifies that the batched cache is reused across rounds
- batch callbacks returning the wrong number of values produce a structured
  error on both small and large backends

These checks establish bounds and behavior, not whole-process RSS. Parallel
execution may own several caches, so aggregate memory must be measured
separately when it matters.

## Python benchmark corpus

The end-to-end Python corpus records graph shape, selected method, exactness,
ESS, and runtime for eight canonical DAGs:

```bash
python examples/benchmark_corpus.py
```

See [benchmark_corpus.md](benchmark_corpus.md) for the pinned environment and
recorded output.
