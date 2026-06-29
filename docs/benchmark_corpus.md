# causasv Benchmark Corpus

A structured comparison of causasv methods across 8 canonical DAG shapes.

Reproduce: `python examples/benchmark_corpus.py` (requires causasv installed via maturin).

**Setup:** Apple M-series (arm64) · Rust 1.96.0 stable · Python 3.13.6 · causasv v0.8.5 (`8ddf97f`) · release build · `v(S) = |S|` · n_samples = 5 000 · seed = 42

---

## DAG corpus definitions

| Name | n | edges | Structure |
|------|---|-------|-----------|
| `chain_5` | 5 | 4 | n0 → n1 → n2 → n3 → n4 |
| `fork_4` | 4 | 3 | root → {a, b, c} |
| `diamond_6` | 6 | 8 | src → {m0,m1,m2,m3} → snk |
| `collider_5` | 5 | 4 | {a,b,c} → z → out |
| `two_chains_10` | 10 | 8 | a0→…→a4  ‖  b0→…→b4 |
| `balanced_tree_15` | 15 | 14 | Balanced binary tree, depth 3 |
| `chain_24` | 24 | 23 | n0 → … → n23 (sparse, n > 20) |
| `dense_8` | 7 | 12 | Chain backbone + cross-edges |

---

## Method characteristics

| Method | Exact? | ESS | CI | Scales to |
|--------|--------|-----|-----|-----------|
| `exact` | ✓ | — | — | n ≤ ~8 |
| `exact_dag` | ✓ | — | — | n ≤ 20 |
| `exact_dag_sparse` | ✓ | — | — | n ≤ 63 (memory-bounded) |
| `uniform_sparse` | ✗ | = n_samples | via `explain_quality` | n ≤ 63 (sparse DAGs) |
| `auto` | DAG-dependent | IS-weighted | ✗ | any |
| `auto_quality` | DAG-dependent | = n_samples (approx path) | ✓ | any |

---

## Benchmark results (Apple M-series, arm64)

```
DAG                      n method               runtime_ms  exact  ess_ratio  selected
chain_5                  5 exact                       0.5    yes     —        exact
chain_5                  5 exact_dag                   0.1    yes     —        exact_dag
chain_5                  5 exact_dag_sparse            0.3    yes     —        exact_dag_sparse
chain_5                  5 uniform_sparse             48.6    no    1.000      uniform_sparse
chain_5                  5 auto                        0.1    yes     —        exact
chain_5                  5 auto_quality                0.1    yes     —        exact

fork_4                   4 exact                       0.1    yes     —        exact
fork_4                   4 exact_dag                   0.0    yes     —        exact_dag
fork_4                   4 auto                        0.2    yes     —        exact
fork_4                   4 auto_quality                0.1    yes     —        exact

diamond_6                6 exact                       0.3    yes     —        exact
diamond_6                6 exact_dag                   0.1    yes     —        exact_dag
diamond_6                6 auto                        0.3    yes     —        exact
diamond_6                6 auto_quality                0.3    yes     —        exact

two_chains_10           10 exact_dag                   1.4    yes     —        exact_dag
two_chains_10           10 exact_dag_sparse            0.5    yes     —        exact_dag_sparse
two_chains_10           10 auto                        1.4    yes     —        exact_dag_sparse
two_chains_10           10 auto_quality                0.4    yes     —        exact_dag_sparse

balanced_tree_15        15 exact_dag                  14.0    yes     —        exact_dag
balanced_tree_15        15 exact_dag_sparse           12.8    yes     —        exact_dag_sparse
balanced_tree_15        15 auto                       37.4    yes     —        exact_tree
balanced_tree_15        15 auto_quality               43.4    yes     —        exact_tree

chain_24                24 exact_dag_sparse            1.1    yes     —        exact_dag_sparse
chain_24                24 uniform_sparse            271.8    no    1.000      uniform_sparse
chain_24                24 auto                        0.8    yes     —        exact_tree
chain_24                24 auto_quality                0.8    yes     —        exact_tree

dense_8                  7 exact                       0.1    yes     —        exact
dense_8                  7 exact_dag                   0.1    yes     —        exact_dag
dense_8                  7 auto                        0.1    yes     —        exact
dense_8                  7 auto_quality                0.1    yes     —        exact
```

---

## Key observations

- **auto / auto_quality correctly routes small DAGs to exact** — no approximation overhead for n ≤ 8.
- **chain_24 is a rooted tree** — auto dispatches to `exact_tree`, which is exact and very fast. The sparse DP is available as a fallback for non-tree sparse DAGs at this size.
- **uniform_sparse ESS = n_samples always** — no IS weight variance regardless of DAG structure, but has higher per-sample overhead than IS-weighted frontier sampling.
- **two_chains_10**: sparse DP visits only (5+1)² = 36 valid order ideals vs 2¹⁰ = 1024 for dense DP — `auto` correctly uses `exact_dag_sparse`.

---

## auto vs auto_quality

| Aspect | `auto` | `auto_quality` |
|--------|--------|----------------|
| Exact path | ✓ same | ✓ same |
| Approximate fallback | IS-weighted (`approx`) | Uniform sparse adaptive |
| ESS on approx path | variable (IS weights) | = n_samples always |
| CI / stderr | ✗ | ✓ always |
| Use when | speed matters, CI not required | CI or ESS guarantee needed |

`auto_quality` is recommended for production workflows; `auto` is faster for exploratory use when uncertainty quantification is not needed.
