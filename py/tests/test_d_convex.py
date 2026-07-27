import pytest

from causasv import ASVExplainer, CausalCPDAG, CausalDAG


def test_chain_d_convex_hull_absorbs_mediator():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert dag.d_convex_hull(["a", "c"]) == ["a", "b", "c"]


def test_chain_strong_d_convex_hull_matches_d_convex_hull():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert dag.strong_d_convex_hull(["a", "c"]) == ["a", "b", "c"]


def test_fork_d_convex_hull_absorbs_confounder():
    dag = CausalDAG.from_edges([("b", "a"), ("b", "c")])
    assert dag.d_convex_hull(["a", "c"]) == ["a", "b", "c"]


def test_collider_d_convex_hull_stays_minimal():
    dag = CausalDAG.from_edges([("a", "b"), ("c", "b")])
    assert dag.d_convex_hull(["a", "c"]) == ["a", "c"]


def test_multi_parent_collider_strong_hull_absorbs_all_parents():
    dag = CausalDAG.from_edges([("a", "d"), ("b", "d"), ("e", "d")])
    assert dag.d_convex_hull(["d"]) == ["d"]
    assert dag.strong_d_convex_hull(["d"]) == ["a", "b", "d", "e"]


def test_required_is_subset_of_hull():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    hull = dag.d_convex_hull(["a", "c"])
    assert "a" in hull
    assert "c" in hull


def test_hull_idempotent():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    hull = dag.d_convex_hull(["a", "c"])
    assert dag.d_convex_hull(hull) == hull


def test_empty_required_is_trivial():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    assert dag.d_convex_hull([]) == []
    assert dag.strong_d_convex_hull([]) == []


def test_unknown_node_raises():
    dag = CausalDAG.from_edges([("a", "b")])
    with pytest.raises(ValueError, match="unknown node"):
        dag.d_convex_hull(["missing"])
    with pytest.raises(ValueError, match="unknown node"):
        dag.strong_d_convex_hull(["missing"])


def test_induced_subgraph_reduces_to_hull():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    hull = dag.d_convex_hull(["a", "c"])
    reduced = dag.induced_subgraph(hull)
    assert reduced.nodes() == ["a", "b", "c"]
    assert ("a", "b") in reduced.edges()


def test_induced_subgraph_drops_unrelated_nodes():
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    reduced = dag.induced_subgraph(["a"])
    assert reduced.nodes() == ["a"]


def test_hull_then_explain_interop():
    """The reduced DAG from induced_subgraph must behave like any other
    CausalDAG -- proves the hull/reduce/explain pipeline interops cleanly."""
    dag = CausalDAG.from_edges([("a", "b"), ("b", "c")])
    hull = dag.strong_d_convex_hull(["a", "c"])
    reduced = dag.induced_subgraph(hull)
    explainer = ASVExplainer(reduced)
    values = explainer.explain(lambda names: float(len(names)), method="exact")
    assert set(values.keys()) == {"a", "b", "c"}


def test_cpdag_strong_d_convex_hull():
    cpdag = CausalCPDAG.from_edges(directed=[("a", "b"), ("b", "c")])
    assert cpdag.strong_d_convex_hull(["a", "c"]) == ["a", "b", "c"]


def test_cpdag_strong_d_convex_hull_unknown_node_raises():
    cpdag = CausalCPDAG.from_edges(directed=[("a", "b")])
    with pytest.raises(ValueError, match="unknown node"):
        cpdag.strong_d_convex_hull(["missing"])
