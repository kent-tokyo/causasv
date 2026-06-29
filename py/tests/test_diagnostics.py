"""Contract test: explain_quality must always return the expected dict keys."""
from causasv import CausalDAG, ASVExplainer, explain_quality

REQUIRED_KEYS = {"values", "stderr", "ess", "ess_ratio", "n_samples", "seed",
                 "is_exact", "selected_method", "converged", "fallback_from", "fallback_reason"}
CI_KEYS = {"ci", "ci_low", "ci_high"}

value_fn = lambda xs: float(len(xs))


def _chain_explainer(n):
    edges = [(f"n{i}", f"n{i+1}") for i in range(n - 1)]
    return ASVExplainer(CausalDAG.from_edges(edges))


def test_exact_path_keys():
    """Chain-3: auto_quality uses exact → is_exact=True, all required keys present."""
    r = explain_quality(_chain_explainer(3), value_fn=value_fn, ci=0.95, seed=0)
    assert REQUIRED_KEYS <= r.keys(), f"missing keys: {REQUIRED_KEYS - r.keys()}"
    assert CI_KEYS <= r.keys(), f"missing CI keys: {CI_KEYS - r.keys()}"
    assert r["is_exact"] is True
    for k in r["values"]:
        assert r["ci_low"][k] <= r["ci_high"][k], "CI bounds inverted"


def test_approx_path_keys():
    """25-node dense DAG: auto_quality falls to approx → is_exact=False, ess_ratio present."""
    edges = [(f"n0", f"n{i}") for i in range(1, 25)]
    edges += [(f"n{i}", "n24") for i in range(1, 24)]
    dag = CausalDAG.from_edges(edges)
    r = explain_quality(ASVExplainer(dag), value_fn=value_fn, ci=0.95, seed=0,
                        min_samples=200, max_samples=2_000)
    assert REQUIRED_KEYS <= r.keys(), f"missing keys: {REQUIRED_KEYS - r.keys()}"
    assert CI_KEYS <= r.keys(), f"missing CI keys: {CI_KEYS - r.keys()}"
    assert r["is_exact"] is False
    assert r["ess_ratio"] is not None
    assert r["selected_method"] is not None
