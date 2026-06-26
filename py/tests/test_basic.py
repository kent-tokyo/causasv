import pytest
from causasv import ASVExplainer, CausalDAG


def test_add_edge_creates_nodes_automatically():
    dag = CausalDAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    # Should not raise
    dag.validate()


def test_exact_additive_chain():
    # Single ordering [a, b, c]: each marginal = 1.
    dag = CausalDAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda names: float(len(names)), method="exact")
    assert abs(values["a"] - 1.0) < 1e-9
    assert abs(values["b"] - 1.0) < 1e-9
    assert abs(values["c"] - 1.0) < 1e-9


def test_approx_reproducible():
    dag = CausalDAG()
    dag.add_edge("x", "y")
    explainer = ASVExplainer(dag)
    v1 = explainer.explain(lambda n: float(len(n)), method="approx", n_samples=500, seed=42)
    v2 = explainer.explain(lambda n: float(len(n)), method="approx", n_samples=500, seed=42)
    assert v1 == v2


def test_approx_default_method():
    # Default method is "approx"; should work without specifying.
    dag = CausalDAG()
    dag.add_edge("a", "b")
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda n: float(len(n)), n_samples=100, seed=1)
    assert set(values.keys()) == {"a", "b"}


def test_efficiency_axiom():
    # Σ φ_i = v(all) - v(empty) for any method.
    dag = CausalDAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    all_names = {"a", "b", "c"}
    explainer = ASVExplainer(dag)

    def v(names):
        s = set(names)
        if not s:
            return 0.0
        if s == all_names:
            return 9.0
        return float(len(s)) ** 2

    values = explainer.explain(v, method="exact")
    assert abs(sum(values.values()) - (9.0 - 0.0)) < 1e-9


def test_invalid_cycle_raises():
    dag = CausalDAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "a")  # creates a cycle
    with pytest.raises(ValueError, match="cycle"):
        ASVExplainer(dag)


def test_self_loop_raises():
    dag = CausalDAG()
    with pytest.raises(ValueError, match="self-loop"):
        dag.add_edge("a", "a")


def test_unknown_method_raises():
    dag = CausalDAG()
    dag.add_edge("a", "b")
    explainer = ASVExplainer(dag)
    with pytest.raises(ValueError, match="unknown method"):
        explainer.explain(lambda n: 1.0, method="invalid")


def test_exact_tree_method():
    dag = CausalDAG()
    dag.add_edge("root", "left")
    dag.add_edge("root", "right")
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda n: float(len(n)), method="exact_tree")
    # All should be finite
    for v in values.values():
        assert v == v  # not NaN


def test_result_keys_are_node_names():
    dag = CausalDAG()
    dag.add_edge("education", "income")
    dag.add_edge("income", "risk_score")
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda n: float(len(n)), method="exact")
    assert set(values.keys()) == {"education", "income", "risk_score"}


def test_from_edges():
    dag = CausalDAG.from_edges([
        ("education", "income"),
        ("income", "risk_score"),
    ])
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda n: float(len(n)), method="exact")
    assert set(values.keys()) == {"education", "income", "risk_score"}


def test_from_edges_same_result_as_add_edge():
    dag1 = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    dag2 = CausalDAG()
    dag2.add_edge("a", "b")
    dag2.add_edge("b", "c")
    v1 = ASVExplainer(dag1).explain(lambda n: float(len(n)), method="exact")
    v2 = ASVExplainer(dag2).explain(lambda n: float(len(n)), method="exact")
    assert v1 == v2


def test_from_edges_cycle_raises():
    with pytest.raises(ValueError, match="cycle"):
        dag = CausalDAG.from_edges([("a", "b"), ("b", "a")])
        ASVExplainer(dag)


def test_from_networkx():
    nx = pytest.importorskip("networkx")
    G = nx.DiGraph()
    G.add_edge("a", "b")
    G.add_edge("b", "c")
    dag = CausalDAG.from_networkx(G)
    values = ASVExplainer(dag).explain(lambda n: float(len(n)), method="exact")
    assert set(values.keys()) == {"a", "b", "c"}
    # Additive v(S)=|S|: only one ordering [a,b,c], all marginals = 1
    assert abs(values["a"] - 1.0) < 1e-9
    assert abs(values["b"] - 1.0) < 1e-9
    assert abs(values["c"] - 1.0) < 1e-9
