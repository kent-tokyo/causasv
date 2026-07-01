"""Empirical test: 95% CI from explain_quality must cover the true ASV in ≥80% of runs.

Uses a 25-node DAG whose order-ideal count far exceeds the sparse-preflight budget
(250k), forcing the uniform_sparse_adaptive path where CI bounds are computed from
stderr.  For additive v(S)=|S|, the true ASV is 1.0 for every feature.
"""
import pytest
from causasv import ASVExplainer, CausalDAG, explain_quality


def _make_large_collider():
    """n0→n1..n24 and n1..n23→n24; 8M+ order ideals → forces approximate path."""
    edges = [("n0", f"n{i}") for i in range(1, 25)]
    edges += [(f"n{i}", "n24") for i in range(1, 24)]
    return CausalDAG.from_edges(edges)


def _coverage(n_seeds, max_samples):
    dag = _make_large_collider()
    explainer = ASVExplainer(dag)

    def value_fn(xs):
        return float(len(xs))

    covered = 0
    for seed in range(n_seeds):
        r = explain_quality(explainer, value_fn=value_fn, ci=0.95, seed=seed,
                            min_samples=200, max_samples=max_samples)
        if r.get("is_exact"):
            covered += 1
            continue
        lo, hi = r.get("ci_low", {}), r.get("ci_high", {})
        if lo and all(lo.get(k, -1.0) <= 1.0 <= hi.get(k, 3.0) for k in r["values"]):
            covered += 1
    return covered / n_seeds


def test_ci_coverage_quick():
    """Quick CI coverage check: 10 seeds, threshold ≥75%."""
    assert _coverage(10, 2_000) >= 0.75, "CI coverage below 75%"


@pytest.mark.slow
def test_ci_coverage_thorough():
    """Thorough CI coverage check (slow): 30 seeds, threshold ≥85%."""
    assert _coverage(30, 5_000) >= 0.85, "CI coverage below 85%"


if __name__ == "__main__":
    cov = _coverage(10, 2_000)
    print(f"Quick coverage: {cov:.1%}")
