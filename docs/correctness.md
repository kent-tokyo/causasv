# Correctness and method limits

This document records the correctness oracles, estimator assumptions, method
limits, and diagnostics used by `causasv`.

## Reference oracles

`AsvExplainer::exact` enumerates every topological ordering and is the primary
oracle on small DAGs. Optimized methods are checked against it on their shared
domain.

Property tests in `tests/property_tests.rs` cover:

| Property | Expected result |
| --- | --- |
| Efficiency | `sum(phi) = v(V) - v(empty)` |
| Dummy | a feature with zero marginal contribution receives zero |
| Additivity | `phi(v + w) = phi(v) + phi(w)` |
| Relabeling | relabeling nodes relabels the result |
| Method agreement | exact methods agree on supported small DAGs |

Additional tests cover valid topological orderings, deterministic seeds,
adaptive convergence, batch callback validation, CPDAG extension, and graph
reduction. For large DAGs, closed-form additive value functions provide an
oracle when no exact implementation can run.

## Approximate estimator

The frontier sampler chooses uniformly from the nodes currently available in
a topological sort. This does not sample complete topological orderings
uniformly. The `approx` family therefore uses self-normalized importance
sampling (SNIS):

```text
phi_i ~= sum_pi w(pi) * delta_i(pi) / sum_pi w(pi)
w(pi) = 1 / q(pi)
```

Every valid ordering has positive sampling probability, so the estimator
converges to the uniform-over-orderings ASV average. Efficiency holds for each
finite sample because each ordering's marginal contributions telescope to
`v(V) - v(empty)` and share the same normalized weight.

Serial, adaptive, and batched paths rescale log weights against a running
maximum to avoid overflow. The seeded-parallel and unseeded-parallel
fixed-sample paths still accumulate direct exponentiated weights; use a serial
or adaptive path for extreme frontier distributions until the parallel
normalization audit is complete.

Uniform sparse sampling uses a memoized count of valid suffix orderings. Its
samples have equal weight, so `ESS = n_samples`, but its state table remains
limited to 63 nodes and can be expensive on weakly constrained DAGs.

## Interpreting approximate results

- **ESS:** `(sum w)^2 / sum(w^2)`. A low ESS means a small number of orderings
  dominate the result. `ESS / n_samples >= 0.1` is a screening rule, not a
  proof of accuracy.
- **Standard error and confidence interval:** adaptive methods report
  per-feature standard errors and optional normal-approximation intervals.
- **Convergence:** `converged=true` means the configured relative-change and
  ESS conditions were met before `max_samples`; it is not an external accuracy
  guarantee.
- **Rank stability:** rerun important analyses across seeds. The Python
  `explain_stability()` helper reports mean pairwise Kendall tau.

Use `explain_quality()` for an exact-first Python workflow and
`explain_safe()` when ESS, rank stability, and intervals should be checked
together. Overlapping confidence intervals do not establish a reliable
feature order.

## Exact method bounds

| Method | Default or hard bound | Work | Notes |
| --- | --- | --- | --- |
| `exact` | practical around `n <= 8`; hard bitmask bound 64 | all topological orderings | reference oracle |
| `exact_tree` | hard bitmask bound 64 plus shape budget | tree order ideals | rooted directed trees only |
| `exact_dag` | hard bound 20 | all `2^n` masks | dense DP |
| `exact_dag_sparse` | default `max_nodes=28`; configurable to 63 | valid order ideals | default 2 GiB memory guard |
| `uniform_sparse` family | hard bound 63 | memoized valid states | approximate, equal weights |
| `approx` family | no node-count limit | sampled orderings | importance sampling |

The default `ExactDagConfig` limit and the structural 63-node bitmask limit
are different. `exact_dag_sparse()` uses the default limit of 28.
`exact_dag_sparse_with_config()` and automatic dispatch may raise `max_nodes`
to 63 when a preflight finds at most 250,000 order ideals.

`exact_tree` feasibility depends on tree shape. Before enumeration it estimates
the largest per-node cartesian product and total work. The default
`ExactTreeConfig` budgets are 50,000 and 200,000 terms. Exceeding either
returns `ExactTreeBudgetExceeded`; automatic methods then try a sparse exact
path or an approximate fallback. This prevents a moderately sized but highly
branching tree from triggering an impractical allocation.

## Automatic dispatch

`auto()` prefers exact methods and uses fixed-sample importance sampling as the
final fallback. `auto_quality()` uses the same exact paths, then an adaptive
path that reports standard errors and convergence metadata.

In outline:

1. `n <= 8`: brute-force `exact`.
2. Rooted tree: `exact_tree` if the shape budget permits it.
3. `n <= 20`: sparse or dense exact DP, selected by state preflight.
4. `20 < n <= 63`: sparse exact DP when the state count is manageable;
   otherwise uniform sparse adaptive or importance-sampling adaptive.
5. `n > 63`: importance sampling (`approx` or `approx_adaptive`).

Results expose `method_used`, `fallback_from`, and `fallback_reason`. Callers
should record these fields instead of inferring the method from graph size.

## Large-DAG approximate paths

Approximate paths use one `u64` coalition through 64 nodes and a growable
word-vector coalition above 64 nodes. The sampler and SNIS formulas are shared;
only the coalition representation and cache differ.

The large backend is checked by:

1. running both backends on the same small DAG, seed, and sampled orderings and
   requiring identical results where the paths overlap; and
2. checking 65-, 128-, and 256-node additive DAGs against the closed-form ASV
   value of `1.0` per node.

Large-coalition caches are admission-capped. After the cap is reached, lookups
continue but new values are not retained, so the cap affects repeated work and
memory rather than correctness. Parallel fixed-sample execution creates a
cache per worker or Rayon fold; aggregate memory can therefore exceed a single
cache's cap.

Batched large-DAG paths share their bounded cache across rounds and still
deduplicate each in-flight batch. This reduces calls to expensive Python value
functions without changing the estimator. See [benchmarks.md](benchmarks.md)
for the measured representation boundary and reproduction commands.
