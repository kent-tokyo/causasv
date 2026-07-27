import json

import pytest

from causasv import CausalCPDAG, CausalDAG


def test_add_directed_edge_creates_nodes():
    g = CausalCPDAG()
    g.add_directed_edge("a", "b")
    assert g.nodes() == ["a", "b"]
    assert g.directed_edges() == [("a", "b")]


def test_add_undirected_edge_creates_nodes():
    g = CausalCPDAG()
    g.add_undirected_edge("a", "b")
    assert g.undirected_edges() == [("a", "b")]


def test_from_edges():
    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    assert g.nodes() == ["a", "b", "c"]
    assert g.directed_edges() == [("a", "b")]
    assert g.undirected_edges() == [("b", "c")]


def test_from_edges_defaults_empty():
    g = CausalCPDAG.from_edges()
    assert g.nodes() == []


def test_self_loop_raises():
    g = CausalCPDAG()
    with pytest.raises(ValueError, match="self-loop"):
        g.add_directed_edge("a", "a")


def test_conflicting_edge_raises():
    g = CausalCPDAG()
    g.add_directed_edge("a", "b")
    with pytest.raises(ValueError, match="conflicting"):
        g.add_undirected_edge("a", "b")


def test_duplicate_edge_raises():
    g = CausalCPDAG()
    g.add_directed_edge("a", "b")
    with pytest.raises(ValueError, match="duplicate"):
        g.add_directed_edge("a", "b")


def test_validate_pdag_directed_cycle_raises():
    g = CausalCPDAG.from_edges(directed=[("a", "b"), ("b", "c"), ("c", "a")])
    with pytest.raises(ValueError, match="cycle"):
        g.validate_pdag()


def test_validate_cpdag_chordless_4cycle_raises():
    g = CausalCPDAG.from_edges(
        undirected=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    )
    with pytest.raises(ValueError, match="no consistent DAG extension"):
        g.validate_cpdag()


def test_validate_cpdag_triangle_ok():
    g = CausalCPDAG.from_edges(undirected=[("a", "b"), ("b", "c"), ("a", "c")])
    g.validate_cpdag()  # should not raise


def test_consistent_extension_returns_causal_dag():
    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    dag = g.consistent_extension()
    assert isinstance(dag, CausalDAG)
    dag.validate()
    assert ("a", "b") in dag.edges()


def test_consistent_extension_chordless_4cycle_raises():
    g = CausalCPDAG.from_edges(
        undirected=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")]
    )
    with pytest.raises(ValueError, match="no consistent DAG extension"):
        g.consistent_extension()


def test_extension_interops_with_asv_explainer():
    """The CausalDAG returned by consistent_extension() must behave like any
    other CausalDAG -- proves the two types interop cleanly."""
    from causasv import ASVExplainer

    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    dag = g.consistent_extension()
    explainer = ASVExplainer(dag)
    values = explainer.explain(lambda names: float(len(names)), method="exact")
    assert set(values.keys()) == {"a", "b", "c"}


def test_induced_subgraph_preserves_edge_kind():
    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    sub = g.induced_subgraph(["a", "b", "c"])
    assert sub.directed_edges() == [("a", "b")]
    assert sub.undirected_edges() == [("b", "c")]


def test_induced_subgraph_drops_edges_touching_removed_nodes():
    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    sub = g.induced_subgraph(["a", "b"])
    assert sub.nodes() == ["a", "b"]
    assert sub.undirected_edges() == []


def test_induced_subgraph_unknown_node_raises():
    g = CausalCPDAG.from_edges(directed=[("a", "b")])
    with pytest.raises(ValueError, match="unknown node"):
        g.induced_subgraph(["a", "missing"])


def test_to_json_round_trip():
    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    restored = CausalCPDAG.from_json(g.to_json())
    assert restored.nodes() == g.nodes()
    assert restored.directed_edges() == g.directed_edges()
    assert restored.undirected_edges() == g.undirected_edges()


def test_to_json_shape():
    g = CausalCPDAG.from_edges(directed=[("a", "b")], undirected=[("b", "c")])
    data = json.loads(g.to_json())
    assert data["nodes"] == ["a", "b", "c"]
    assert {"from": "a", "to": "b"} in data["directed_edges"]
    assert {"a": "b", "b": "c"} in data["undirected_edges"]


def test_from_json_rejects_malformed_json():
    with pytest.raises(ValueError, match="malformed JSON"):
        CausalCPDAG.from_json("not json at all")


def test_from_json_accepts_standard_json_dumps_spacing():
    payload = {
        "nodes": ["A", "B"],
        "directed_edges": [{"from": "A", "to": "B"}],
        "undirected_edges": [],
    }
    compact = json.dumps(payload, separators=(",", ":"))
    default_spaced = json.dumps(payload)
    from_compact = CausalCPDAG.from_json(compact)
    from_spaced = CausalCPDAG.from_json(default_spaced)
    assert from_compact.nodes() == from_spaced.nodes() == ["A", "B"]
    assert (
        from_compact.directed_edges()
        == from_spaced.directed_edges()
        == [("A", "B")]
    )


def test_rust_python_result_parity():
    """Build the same CPDAG via the Python API and compare consistent_extension
    output against what CausalDAG.from_edges would produce directly -- proves
    the Rust core and Python binding agree."""
    g = CausalCPDAG.from_edges(directed=[("a", "b"), ("b", "c")])
    extended = g.consistent_extension()
    direct = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert extended.nodes() == direct.nodes()
    assert extended.edges() == direct.edges()
