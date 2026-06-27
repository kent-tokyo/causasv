"""Tests for ASVEnsembleExplainer."""

import pytest
from causasv import ASVEnsembleExplainer, ASVExplainer, CausalDAG


def additive_fn(names):
    return float(len(names))


@pytest.fixture
def chain_dag():
    return CausalDAG.from_edges([("a", "b"), ("b", "c")])


@pytest.fixture
def diamond_dag():
    return CausalDAG.from_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])


def test_ensemble_single_dag_matches_explainer(chain_dag):
    """Ensemble of one DAG must return the same values as a direct ASVExplainer."""
    single = ASVExplainer(chain_dag).explain(additive_fn, method="exact")
    result = ASVEnsembleExplainer([chain_dag]).explain_with_sensitivity(
        additive_fn, method="exact"
    )
    for k, v in single.items():
        assert abs(result["mean_values"][k] - v) < 1e-9


def test_ensemble_single_dag_std_is_zero(chain_dag):
    """Single DAG → std = 0 for all features."""
    result = ASVEnsembleExplainer([chain_dag]).explain_with_sensitivity(
        additive_fn, method="exact"
    )
    for v in result["std_values"].values():
        assert abs(v) < 1e-12


def test_ensemble_rank_stability_identical_dags():
    """Identical DAGs with a non-symmetric value fn → rank_stability = 1.0."""
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])

    def nonlinear(names):
        s = set(names)
        return ("a" in s) * 3 + ("b" in s) * 2 + ("c" in s) * 6 * ("a" in s and "c" in s)

    result = ASVEnsembleExplainer([dag, dag]).explain_with_sensitivity(nonlinear, method="exact")
    assert abs(result["rank_stability"] - 1.0) < 1e-9


def test_ensemble_result_keys(chain_dag, diamond_dag):
    result = ASVEnsembleExplainer([chain_dag]).explain_with_sensitivity(additive_fn, method="exact")
    assert set(result.keys()) == {"mean_values", "std_values", "rank_stability", "per_dag_values"}


def test_ensemble_per_dag_values_count():
    dag1 = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    dag2 = CausalDAG.from_edges([("a", "b"), ("a", "c")])
    dag3 = CausalDAG.from_edges([("a", "c"), ("b", "c")])
    result = ASVEnsembleExplainer([dag1, dag2, dag3]).explain_with_sensitivity(
        additive_fn, method="exact"
    )
    assert len(result["per_dag_values"]) == 3


def test_ensemble_rank_stability_range():
    """rank_stability must be in [-1, 1]."""
    dag1 = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    dag2 = CausalDAG.from_edges([("a", "b"), ("a", "c")])
    result = ASVEnsembleExplainer([dag1, dag2]).explain_with_sensitivity(
        additive_fn, method="exact"
    )
    assert -1.0 <= result["rank_stability"] <= 1.0


def test_ensemble_mean_equals_average():
    """mean_values must equal the arithmetic mean of per_dag_values."""
    dag1 = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    dag2 = CausalDAG.from_edges([("a", "b"), ("a", "c")])
    result = ASVEnsembleExplainer([dag1, dag2]).explain_with_sensitivity(
        additive_fn, method="exact"
    )
    for f in result["mean_values"]:
        expected = sum(d.get(f, 0.0) for d in result["per_dag_values"]) / len(
            result["per_dag_values"]
        )
        assert abs(result["mean_values"][f] - expected) < 1e-9


def test_ensemble_empty():
    """Empty ensemble returns empty dicts and rank_stability=1.0."""
    result = ASVEnsembleExplainer([]).explain_with_sensitivity(additive_fn)
    assert result["mean_values"] == {}
    assert result["std_values"] == {}
    assert result["rank_stability"] == 1.0
    assert result["per_dag_values"] == []
