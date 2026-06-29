"""Empirical test: 95% CI from explain_quality must cover the true ASV in ≥80% of runs.

Uses a 25-node DAG whose order-ideal count far exceeds the sparse-preflight budget
(250k), forcing the uniform_sparse_adaptive path where CI bounds are computed from
stderr.  For additive v(S)=|S|, the true ASV is 1.0 for every feature.
"""
from causasv import CausalDAG, ASVExplainer, explain_quality


def _make_large_collider():
    """n0→n1..n24 and n1..n23→n24; 8M+ order ideals → forces approximate path."""
    edges = [(f"n0", f"n{i}") for i in range(1, 25)]
    edges += [(f"n{i}", "n24") for i in range(1, 24)]
    return CausalDAG.from_edges(edges)


def test_ci_coverage_95():
    dag = _make_large_collider()
    explainer = ASVExplainer(dag)

    # Additive value function: true ASV_i = 1.0 for all features.
    value_fn = lambda xs: float(len(xs))

    covered = 0
    n_seeds = 30
    for seed in range(n_seeds):
        r = explain_quality(
            explainer,
            value_fn=value_fn,
            ci=0.95,
            seed=seed,
            min_samples=500,
            max_samples=5_000,
        )
        if r.get("is_exact"):
            covered += 1  # exact → trivially correct
            continue
        lo, hi = r.get("ci_low", {}), r.get("ci_high", {})
        if lo and all(lo.get(k, -1.0) <= 1.0 <= hi.get(k, 3.0) for k in r["values"]):
            covered += 1

    coverage = covered / n_seeds
    assert coverage >= 0.80, (
        f"95% CI coverage {coverage:.1%} < 80% threshold ({covered}/{n_seeds} covered)"
    )


if __name__ == "__main__":
    test_ci_coverage_95()
    print("CI coverage test passed")
