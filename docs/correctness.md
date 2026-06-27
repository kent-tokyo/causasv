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
| `exact_tree` | unlimited | n+1 order ideals | Only for rooted directed trees |
| `exact_dag` | 20 | 2ⁿ bitmasks | O(2ⁿ × n) time; ~16 MB for n=20 |
| `exact_dag_sparse` | 28 | ≤ 2ⁿ order ideals (BFS) | Much faster for sparse DAGs; memory-bounded (default 2 GiB) |

`auto()` selects the method automatically and reports what it chose in
`info["selected_method"]`. If `exact_dag_sparse` hits the memory or overflow
limit, it falls back to `approx` and sets `info["fallback_from"]`.
