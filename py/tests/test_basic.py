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
    v1 = explainer.explain(
        lambda n: float(len(n)), method="approx", n_samples=500, seed=42
    )
    v2 = explainer.explain(
        lambda n: float(len(n)), method="approx", n_samples=500, seed=42
    )
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
    dag = CausalDAG.from_edges(
        [
            ("education", "income"),
            ("income", "risk_score"),
        ]
    )
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


def test_auto_selected_method_diagnostics():
    # chain n=3: auto should dispatch to exact (n <= 8)
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    info = ASVExplainer(dag).explain_with_diagnostics(
        lambda n: float(len(n)), method="auto"
    )
    assert info["method"] == "auto"
    assert info["selected_method"] == "exact"
    assert info["is_exact"] is True
    assert info["fallback_from"] is None


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
    # Keep in sync with explain_with_diagnostics() in src/python.rs
    expected_keys = {
        "values",
        "ess",
        "ess_ratio",
        "n_samples",
        "seed",
        "is_exact",
        "method",
        "parallel",
        "num_threads",
        "deterministic",
        "n_order_ideals",
        "state_ratio",
        "memory_mb",
        "fallback_from",
        "fallback_reason",
        "selected_method",
    }
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


def test_dag_to_json():
    import json

    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    data = json.loads(dag.to_json())
    assert set(data["nodes"]) == {"a", "b", "c"}
    assert {"from": "a", "to": "b"} in data["edges"]
    assert {"from": "b", "to": "c"} in data["edges"]


def test_dag_from_json_roundtrip():
    dag = CausalDAG.from_edges([("x", "y"), ("y", "z")])
    restored = CausalDAG.from_json(dag.to_json())
    assert restored.nodes() == dag.nodes()
    assert restored.edges() == dag.edges()


def test_dag_ancestors():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert sorted(dag.ancestors("c")) == ["a", "b"]
    assert dag.ancestors("b") == ["a"]
    assert dag.ancestors("a") == []


def test_dag_descendants():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert sorted(dag.descendants("a")) == ["b", "c"]
    assert dag.descendants("b") == ["c"]
    assert dag.descendants("c") == []


def test_dag_topological_layers():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert dag.topological_layers() == [["a"], ["b"], ["c"]]

    dag2 = CausalDAG.from_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    layers2 = dag2.topological_layers()
    assert layers2[0] == ["a"]
    assert sorted(layers2[1]) == ["b", "c"]
    assert layers2[2] == ["d"]


def test_dag_inspect_keys():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    info = dag.inspect()
    expected = {
        "n_nodes", "n_edges", "is_dag", "is_rooted_tree",
        "n_roots", "n_leaves", "max_depth", "recommended_method", "estimated_dense_states",
    }
    assert expected == set(info.keys())


def test_dag_inspect_chain():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    info = dag.inspect()
    assert info["n_nodes"] == 3
    assert info["n_edges"] == 2
    assert info["is_dag"] is True
    assert info["is_rooted_tree"] is True
    assert info["n_roots"] == 1
    assert info["n_leaves"] == 1
    assert info["max_depth"] == 2
    assert info["recommended_method"] == "exact"
    assert info["estimated_dense_states"] == 8  # 2^3


def test_dag_inspect_diamond():
    # a->b, a->c, b->d, c->d
    dag = CausalDAG.from_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    info = dag.inspect()
    assert info["n_nodes"] == 4
    assert info["n_edges"] == 4
    assert info["is_rooted_tree"] is False
    assert info["n_roots"] == 1
    assert info["n_leaves"] == 1
    assert info["max_depth"] == 2


def test_dag_inspect_recommended_method():
    # Large chain (n=25) is a rooted tree → exact_tree
    edges = [(f"n{i}", f"n{i+1}") for i in range(24)]
    dag = CausalDAG.from_edges(edges)
    info = dag.inspect()
    assert info["n_nodes"] == 25
    assert info["is_rooted_tree"] is True
    assert info["recommended_method"] == "exact_tree"
    assert info["estimated_dense_states"] == 2 ** 25


def test_dag_inspect_dense_states_none_for_large():
    # n=64 → estimated_dense_states should be None (overflow guard)
    edges = [(f"n{i}", f"n{i+1}") for i in range(63)]
    dag = CausalDAG.from_edges(edges)
    info = dag.inspect()
    assert info["estimated_dense_states"] is None


def test_explain_adaptive_keys():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_adaptive(
        lambda n: float(len(n)),
        min_samples=100,
        max_samples=500,
        batch_size=100,
        seed=0,
    )
    expected = {
        "values",
        "ess",
        "ess_ratio",
        "n_samples",
        "seed",
        "is_exact",
        "method",
        "converged",
        "stderr",
    }
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

    def fn(n):
        return float(len(n))

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

    def fn(n):
        return float(len(n))

    info = explainer.explain_adaptive(fn, min_samples=100, max_samples=500, seed=0)
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
        min_samples=500,
        max_samples=10_000,
        batch_size=500,
        rel_tol=0.01,
        seed=42,
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


def test_tabular_explainer():
    np = pytest.importorskip("numpy")
    from causasv import TabularExplainer

    class SumModel:
        def predict(self, X):
            return X.sum(axis=1)

    background = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([10.0, 20.0])
    feature_names = ["f0", "f1"]
    dag = CausalDAG.from_edges([("f0", "f1")])

    explainer = TabularExplainer.from_model(SumModel(), dag, background, feature_names)
    values = explainer.explain_instance(x, method="exact")
    assert set(values.keys()) == {"f0", "f1"}
    # efficiency axiom
    baseline_sum = float(background.mean(axis=0).sum())
    assert abs(sum(values.values()) - (x.sum() - baseline_sum)) < 1e-6


def test_tabular_explainer_forwards_baseline():
    np = pytest.importorskip("numpy")
    from causasv import TabularExplainer

    class SumModel:
        def predict(self, X):
            return X.sum(axis=1)

    background = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([10.0, 20.0])
    feature_names = ["f0", "f1"]
    dag = CausalDAG.from_edges([("f0", "f1")])

    explainer = TabularExplainer.from_model(
        SumModel(), dag, background, feature_names, baseline="median"
    )
    values = explainer.explain_instance(x, method="exact")
    median_sum = float(np.median(background, axis=0).sum())
    assert abs(sum(values.values()) - (x.sum() - median_sum)) < 1e-6


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


def _tabular_test_fixture():
    np = pytest.importorskip("numpy")

    class SumModel:
        def predict(self, X):
            return X.sum(axis=1)

    background = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    x = np.array([10.0, 20.0, 30.0])
    feature_names = ["f0", "f1", "f2"]
    return np, SumModel(), background, x, feature_names


def test_make_tabular_value_fn_median_baseline():
    np, model, background, x, feature_names = _tabular_test_fixture()
    from causasv import make_tabular_value_fn

    value_fn = make_tabular_value_fn(model, x, background, feature_names, baseline="median")
    median_sum = np.median(background, axis=0).sum()
    assert abs(value_fn([]) - median_sum) < 1e-9
    assert abs(value_fn(["f0", "f1", "f2"]) - x.sum()) < 1e-9


def test_make_tabular_value_fn_sample_baseline():
    _, model, background, x, feature_names = _tabular_test_fixture()
    from causasv import make_tabular_value_fn

    value_fn = make_tabular_value_fn(model, x, background, feature_names, baseline="sample")
    # Empty coalition must equal one of the two background rows' sums exactly
    # (a real row, not a synthetic average); seeded so this is deterministic.
    row_sums = {float(row.sum()) for row in background}
    assert value_fn([]) in row_sums
    assert abs(value_fn(["f0", "f1", "f2"]) - x.sum()) < 1e-9


def test_make_tabular_value_fn_background_expectation():
    _, model, background, x, feature_names = _tabular_test_fixture()
    from causasv import make_tabular_value_fn

    value_fn = make_tabular_value_fn(
        model, x, background, feature_names, baseline="background_expectation"
    )
    # Empty coalition: mean of predict() over the untouched background rows.
    assert abs(value_fn([]) - background.sum(axis=1).mean()) < 1e-9
    # Full coalition: every background row is fully overwritten by x.
    assert abs(value_fn(["f0", "f1", "f2"]) - x.sum()) < 1e-9
    # Partial coalition: f0 comes from x in every row, f1/f2 stay from background.
    row0 = x[0] + background[0, 1] + background[0, 2]
    row1 = x[0] + background[1, 1] + background[1, 2]
    expected = (row0 + row1) / 2
    assert abs(value_fn(["f0"]) - expected) < 1e-9


def test_make_tabular_value_fn_custom_imputer_baseline():
    np, model, background, x, feature_names = _tabular_test_fixture()
    from causasv import make_tabular_value_fn

    value_fn = make_tabular_value_fn(
        model, x, background, feature_names, baseline=lambda bg: bg.max(axis=0)
    )
    assert abs(value_fn([]) - background.max(axis=0).sum()) < 1e-9


def test_make_tabular_value_fn_unknown_baseline_raises():
    _, model, background, x, feature_names = _tabular_test_fixture()
    from causasv import make_tabular_value_fn

    with pytest.raises(ValueError):
        make_tabular_value_fn(model, x, background, feature_names, baseline="bogus")


# ---------------------------------------------------------------------------
# value_fn_batch tests
# ---------------------------------------------------------------------------


def test_value_fn_batch_same_result_as_single():
    """Batched and single-call paths must agree exactly for the same seed."""
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)

    def value_fn(names):
        return float(len(names))

    def value_fn_batch(coalitions):
        return [float(len(c)) for c in coalitions]

    single = explainer.explain(value_fn=value_fn, method="approx", n_samples=1000, seed=7)
    batched = explainer.explain(
        value_fn_batch=value_fn_batch, method="approx", n_samples=1000, seed=7, batch_size=1
    )
    for k in single:
        assert abs(single[k] - batched[k]) < 1e-9, f"mismatch on {k}: {single[k]} vs {batched[k]}"


def test_value_fn_batch_diagnostics():
    dag = CausalDAG.from_edges([("x", "y")])
    explainer = ASVExplainer(dag)

    def batch_fn(coalitions):
        return [float(len(c)) for c in coalitions]

    info = explainer.explain_with_diagnostics(
        value_fn_batch=batch_fn, method="approx", n_samples=500, seed=3, batch_size=50
    )
    assert "values" in info
    assert "ess" in info
    assert set(info["values"].keys()) == {"x", "y"}


def test_value_fn_batch_efficiency():
    """Batch size > 1 should reduce call count (smoke test: just confirm it runs)."""
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c"), ("a", "c")])
    explainer = ASVExplainer(dag)
    call_count = {"n": 0}

    def batch_fn(coalitions):
        call_count["n"] += 1
        return [float(len(c)) for c in coalitions]

    explainer.explain(
        value_fn_batch=batch_fn, method="approx", n_samples=1000, seed=99, batch_size=100
    )
    # With batch_size=100 and 1000 samples, we expect ~10 batch calls (not 1000+)
    assert call_count["n"] <= 20, f"too many batch calls: {call_count['n']}"


def test_value_fn_batch_error_on_missing():
    dag = CausalDAG.from_edges([("a", "b")])
    explainer = ASVExplainer(dag)

    with pytest.raises(Exception, match="value_fn"):
        explainer.explain(method="approx", n_samples=100)


# ---------------------------------------------------------------------------
# Deterministic parallel sampling tests
# ---------------------------------------------------------------------------


def test_seeded_parallel_reproducible():
    """Same seed + parallel=True must give identical values across runs."""
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c"), ("a", "c")])
    explainer = ASVExplainer(dag)

    def fn(names):
        return float(len(names))

    v1 = explainer.explain(fn, method="approx", n_samples=2000, seed=42, parallel=True)
    v2 = explainer.explain(fn, method="approx", n_samples=2000, seed=42, parallel=True)
    assert v1 == v2


def test_seeded_parallel_close_to_serial():
    """Seeded parallel and serial should agree within statistical tolerance."""
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)

    def fn(names):
        return float(len(names))

    serial = explainer.explain(fn, method="approx", n_samples=5000, seed=7, parallel=False)
    par = explainer.explain(fn, method="approx", n_samples=5000, seed=7, parallel=True)
    for k in serial:
        assert abs(serial[k] - par[k]) < 0.05, (
            f"serial vs parallel diverged on {k}: {serial[k]:.4f} vs {par[k]:.4f}"
        )


def test_seeded_parallel_diagnostics():
    dag = CausalDAG.from_edges([("x", "y")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_with_diagnostics(
        lambda n: float(len(n)), method="approx", n_samples=500, seed=3, parallel=True
    )
    assert info["parallel"] is True
    assert info["deterministic"] is True
    assert info["seed"] == 3


def test_unseeded_parallel_no_determinism_flag():
    dag = CausalDAG.from_edges([("x", "y")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_with_diagnostics(
        lambda n: float(len(n)), method="approx", n_samples=200
    )
    assert info["parallel"] is False
    assert info["deterministic"] is False


# ---------------------------------------------------------------------------
# exact_dag_sparse tests
# ---------------------------------------------------------------------------


def test_exact_dag_sparse_matches_exact_dag():
    """Sparse DP must produce the same values as dense DP."""
    dag = CausalDAG.from_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    explainer = ASVExplainer(dag)

    def fn(names):
        return float(len(names))

    dense = explainer.explain(fn, method="exact_dag")
    sparse = explainer.explain(fn, method="exact_dag_sparse")
    for k in dense:
        assert abs(dense[k] - sparse[k]) < 1e-9, f"mismatch on {k}"


def test_exact_dag_sparse_diagnostics():
    dag = CausalDAG.from_edges([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    explainer = ASVExplainer(dag)
    info = explainer.explain_with_diagnostics(
        lambda n: float(len(n)), method="exact_dag_sparse"
    )
    assert info["is_exact"] is True
    assert isinstance(info["n_order_ideals"], int) and info["n_order_ideals"] > 0
    assert isinstance(info["state_ratio"], float) and 0 < info["state_ratio"] <= 1.0
    assert isinstance(info["memory_mb"], float) and info["memory_mb"] > 0


def test_exact_dag_sparse_chain_large():
    """Chain DAG n=25 has only 26 order ideals — sparse should handle it easily."""
    edges = [(f"n{i}", f"n{i+1}") for i in range(24)]
    dag = CausalDAG.from_edges(edges)
    explainer = ASVExplainer(dag)

    def fn(names):
        return float(len(names))

    result = explainer.explain(fn, method="exact_dag_sparse")
    # Chain has one ordering; all values should be 1.0
    for v in result.values():
        assert abs(v - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# explain_stability tests
# ---------------------------------------------------------------------------


def test_explain_stability_keys():
    from causasv import explain_stability

    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)
    result = explain_stability(
        explainer, lambda n: float(len(n)), seeds=[1, 2, 3], method="approx", n_samples=500
    )
    assert set(result.keys()) == {"mean_values", "std_values", "rank_stability", "per_seed_values"}
    assert set(result["mean_values"].keys()) == {"a", "b", "c"}
    assert set(result["per_seed_values"].keys()) == {1, 2, 3}


def test_explain_stability_exact_has_zero_std():
    """Exact method is deterministic — std must be 0 regardless of seeds."""
    from causasv import explain_stability

    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    explainer = ASVExplainer(dag)

    def distinct_fn(names):
        s = set(names)
        return ("a" in s) * 3 + ("b" in s) * 2 + ("c" in s) * 7 * ("a" in s and "c" in s)

    result = explain_stability(explainer, distinct_fn, seeds=[1, 2, 3, 4], method="exact")
    for f, s in result["std_values"].items():
        assert abs(s) < 1e-12, f"expected std=0 for exact method on {f}, got {s}"
    assert abs(result["rank_stability"] - 1.0) < 1e-9


def test_explain_stability_rank_stability_range():
    from causasv import explain_stability

    dag = CausalDAG.from_edges([("x", "y")])
    explainer = ASVExplainer(dag)
    result = explain_stability(
        explainer, lambda n: float(len(n)), seeds=[10, 20, 30], method="approx", n_samples=200
    )
    assert -1.0 <= result["rank_stability"] <= 1.0


def test_explain_stability_mean_equals_average():
    from causasv import explain_stability

    dag = CausalDAG.from_edges([("a", "b")])
    explainer = ASVExplainer(dag)
    seeds = [5, 6, 7]
    result = explain_stability(
        explainer, lambda n: float(len(n)), seeds=seeds, method="approx", n_samples=300
    )
    for f in result["mean_values"]:
        expected = sum(result["per_seed_values"][s][f] for s in seeds) / len(seeds)
        assert abs(result["mean_values"][f] - expected) < 1e-12


def test_explain_stability_empty_seeds_raises():
    from causasv import explain_stability

    dag = CausalDAG.from_edges([("a", "b")])
    explainer = ASVExplainer(dag)
    with pytest.raises(ValueError, match="seeds"):
        explain_stability(explainer, lambda n: float(len(n)), seeds=[])
