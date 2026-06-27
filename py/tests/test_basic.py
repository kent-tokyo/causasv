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


def test_auto_method_default():
    # Default method is now "auto"; chain n=3 → exact path
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda n: float(len(n)))  # no method= means "auto"
    assert abs(values["a"] - 1.0) < 1e-9
    assert abs(values["b"] - 1.0) < 1e-9
    assert abs(values["c"] - 1.0) < 1e-9


def test_auto_matches_explicit_exact():
    dag1 = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    dag2 = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    v_auto = ASVExplainer(dag1).explain(lambda n: float(len(n)), method="auto")
    v_exact = ASVExplainer(dag2).explain(lambda n: float(len(n)), method="exact")
    assert v_auto == v_exact


def test_exact_dag_method():
    # Diamond DAG: a->b, a->c, b->d, c->d — general DAG, not a tree.
    dag = CausalDAG.from_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda n: float(len(n)), method="exact_dag")
    assert set(values.keys()) == {"a", "b", "c", "d"}
    # Efficiency axiom
    assert abs(sum(values.values()) - 4.0) < 1e-9


def test_explain_with_diagnostics_keys():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_with_diagnostics(lambda n: float(len(n)), method="exact")
    expected_keys = {"values", "ess", "ess_ratio", "n_samples", "seed", "is_exact", "method"}
    assert expected_keys == set(info.keys())


def test_explain_with_diagnostics_exact():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_with_diagnostics(lambda n: float(len(n)), method="exact")
    assert info["is_exact"] is True
    assert info["ess"] is None
    assert info["ess_ratio"] is None
    assert info["method"] == "exact"
    assert info["n_samples"] == 1  # one topological ordering for a chain
    assert set(info["values"].keys()) == {"a", "b", "c"}


def test_explain_with_diagnostics_approx():
    dag = CausalDAG.from_edges([("x", "y")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_with_diagnostics(
        lambda n: float(len(n)), method="approx", n_samples=500, seed=7
    )
    assert info["is_exact"] is False
    assert isinstance(info["ess"], float) and info["ess"] > 0
    assert isinstance(info["ess_ratio"], float) and 0 < info["ess_ratio"] <= 1.0
    assert info["n_samples"] == 500
    assert info["seed"] == 7
    assert info["method"] == "approx"


def test_dag_nodes():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert dag.nodes() == ["a", "b", "c"]


def test_dag_edges():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert dag.edges() == [("a", "b"), ("b", "c")]


def test_dag_to_dot():
    dag = CausalDAG.from_edges([("a", "b")])
    dot = dag.to_dot()
    assert dot.startswith("digraph {")
    assert "a -> b" in dot
    assert dot.strip().endswith("}")


def test_explain_adaptive_keys():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_adaptive(
        lambda n: float(len(n)), min_samples=100, max_samples=500, batch_size=100, seed=0
    )
    expected = {"values", "ess", "ess_ratio", "n_samples", "seed", "is_exact",
                "method", "converged", "stderr"}
    assert expected == set(info.keys())
    assert info["method"] == "approx_adaptive"
    assert info["is_exact"] is False
    assert isinstance(info["converged"], bool)
    assert isinstance(info["stderr"], dict)


def test_explain_adaptive_ci_keys():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_adaptive(
        lambda n: float(len(n)), min_samples=500, max_samples=5_000, seed=1, ci=0.95
    )
    assert "ci_low" in info and "ci_high" in info and "ci" in info
    assert info["ci"] == 0.95
    for k in info["values"]:
        assert info["ci_low"][k] <= info["values"][k] <= info["ci_high"][k], (
            f"CI violation on {k}: [{info['ci_low'][k]:.4f}, {info['ci_high'][k]:.4f}]"
            f" does not contain {info['values'][k]:.4f}"
        )


def test_explain_adaptive_ci_width():
    """Wider CI level → wider interval."""
    dag = CausalDAG.from_edges([("x", "y")])
    explainer = ASVExplainer(dag)
    fn = lambda n: float(len(n))
    info_90 = explainer.explain_adaptive(fn, min_samples=1_000, max_samples=5_000, seed=2, ci=0.90)
    info_99 = explainer.explain_adaptive(fn, min_samples=1_000, max_samples=5_000, seed=2, ci=0.99)
    for k in info_90["values"]:
        width_90 = info_90["ci_high"][k] - info_90["ci_low"][k]
        width_99 = info_99["ci_high"][k] - info_99["ci_low"][k]
        assert width_99 >= width_90, f"99% CI should be wider than 90% CI on {k}"


def test_explain_adaptive_ci_no_ci():
    """Without ci=, ci_low/ci_high/ci are absent from the result."""
    dag = CausalDAG.from_edges([("a", "b")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_adaptive(lambda n: float(len(n)), min_samples=100, max_samples=500, seed=0)
    assert "ci_low" not in info
    assert "ci_high" not in info
    assert "ci" not in info


def test_explain_adaptive_ci_invalid():
    dag = CausalDAG.from_edges([("a", "b")])
    explainer = ASVExplainer(dag)
    with pytest.raises(Exception):
        explainer.explain_adaptive(lambda n: float(len(n)), ci=1.5)


def test_explain_adaptive_converges():
    # chain n=3 with additive v(S)=|S| should converge quickly
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_adaptive(
        lambda n: float(len(n)),
        min_samples=500, max_samples=10_000, batch_size=500, rel_tol=0.01, seed=42
    )
    assert info["converged"] is True
    assert info["n_samples"] < 10_000


def test_explain_adaptive_matches_exact():
    dag = CausalDAG.from_edges([("a", "b")])
    explainer = ASVExplainer(dag)
    exact = explainer.explain(lambda n: float(len(n)), method="exact")
    adaptive = explainer.explain_adaptive(
        lambda n: float(len(n)), min_samples=2_000, max_samples=20_000, seed=1
    )
    for node in exact:
        assert abs(adaptive["values"][node] - exact[node]) < 0.05


def test_make_tabular_value_fn():
    np = pytest.importorskip("numpy")
    from causasv import make_tabular_value_fn

    # Fake model: sum of present feature values.
    class SumModel:
        def predict(self, X):
            return X.sum(axis=1)

    background = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    x = np.array([10.0, 20.0, 30.0])
    feature_names = ["f0", "f1", "f2"]

    value_fn = make_tabular_value_fn(SumModel(), x, background, feature_names)

    # Empty coalition → baseline sum
    baseline_sum = background.mean(axis=0).sum()
    assert abs(value_fn([]) - baseline_sum) < 1e-9

    # Full coalition → x.sum()
    assert abs(value_fn(["f0", "f1", "f2"]) - x.sum()) < 1e-9
