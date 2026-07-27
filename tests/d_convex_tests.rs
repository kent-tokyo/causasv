use causasv::{CausasvError, Dag, NodeId};

fn hull_names(dag: &Dag, hull: &indexmap::IndexSet<NodeId>) -> Vec<String> {
    let mut names: Vec<String> = hull
        .iter()
        .map(|&id| dag.node_name(id).unwrap().to_string())
        .collect();
    names.sort_unstable();
    names
}

// ── chain: mediator must be absorbed ────────────────────────────────────────
// a -> b -> c. {a,c} are not adjacent; a,c stay marginally dependent through
// the unblocked chain at b (only conditioning on b would separate them, and
// b is exactly what's being marginalized out) -- so b is a genuine mf-pair
// witness and must be absorbed.

fn chain() -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    (dag, a, b, c)
}

#[test]
fn test_chain_d_convex_hull_absorbs_mediator() {
    let (dag, a, _b, c) = chain();
    let hull = dag.d_convex_hull(&[a, c]).unwrap();
    assert_eq!(hull_names(&dag, &hull), vec!["a", "b", "c"]);
}

#[test]
fn test_chain_strong_d_convex_hull_matches_d_convex_hull() {
    let (dag, a, _b, c) = chain();
    let strong = dag.strong_d_convex_hull(&[a, c]).unwrap();
    assert_eq!(hull_names(&dag, &strong), vec!["a", "b", "c"]);
}

// ── fork: common cause must be absorbed ─────────────────────────────────────
// a <- b -> c. Same reasoning as the chain: marginalizing the confounder b
// changes the marginal dependence between a and c, so b must be absorbed.

fn fork() -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(b, a).unwrap();
    dag.add_edge(b, c).unwrap();
    (dag, a, b, c)
}

#[test]
fn test_fork_d_convex_hull_absorbs_confounder() {
    let (dag, a, _b, c) = fork();
    let hull = dag.d_convex_hull(&[a, c]).unwrap();
    assert_eq!(hull_names(&dag, &hull), vec!["a", "b", "c"]);
}

// ── collider: already d-convex, nothing to absorb ───────────────────────────
// a -> b <- c. a and c are marginally independent (the unconditioned
// collider at b blocks the only path), so {a,c} needs no witness absorbed --
// unlike the chain/fork cases above, this must NOT grow.

fn collider() -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(c, b).unwrap();
    (dag, a, b, c)
}

#[test]
fn test_collider_d_convex_hull_stays_minimal() {
    let (dag, a, _b, c) = collider();
    let hull = dag.d_convex_hull(&[a, c]).unwrap();
    assert_eq!(hull_names(&dag, &hull), vec!["a", "c"]);
}

// ── multi-parent collider: strong d-convexity forces absorbing ALL parents ──
// a -> d, b -> d, e -> d, with a, b, e pairwise non-adjacent. {d} is already
// d-convex on its own (a single node has no inducing-path pair to violate),
// but it is NOT strongly d-convex: d has 3 pairwise non-adjacent parents,
// none of which are in the target set, so d is not linearly ordered w.r.t.
// {d}. Every parent must be absorbed.

fn multi_parent_collider() -> (Dag, NodeId, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let e = dag.add_node("e");
    let d = dag.add_node("d");
    dag.add_edge(a, d).unwrap();
    dag.add_edge(b, d).unwrap();
    dag.add_edge(e, d).unwrap();
    (dag, a, b, e, d)
}

#[test]
fn test_multi_parent_collider_d_convex_hull_is_trivial() {
    let (dag, _a, _b, _e, d) = multi_parent_collider();
    let hull = dag.d_convex_hull(&[d]).unwrap();
    assert_eq!(hull_names(&dag, &hull), vec!["d"]);
}

#[test]
fn test_multi_parent_collider_strong_hull_absorbs_all_parents() {
    let (dag, _a, _b, _e, d) = multi_parent_collider();
    let strong = dag.strong_d_convex_hull(&[d]).unwrap();
    assert_eq!(hull_names(&dag, &strong), vec!["a", "b", "d", "e"]);
}

// ── general properties ───────────────────────────────────────────────────────

#[test]
fn test_required_is_subset_of_hull() {
    let (dag, a, _b, c) = chain();
    let hull = dag.d_convex_hull(&[a, c]).unwrap();
    assert!(hull.contains(&a));
    assert!(hull.contains(&c));
}

#[test]
fn test_d_convex_hull_idempotent() {
    let (dag, a, _b, c) = chain();
    let hull = dag.d_convex_hull(&[a, c]).unwrap();
    let hull_of_hull = dag
        .d_convex_hull(&hull.into_iter().collect::<Vec<_>>())
        .unwrap();
    assert_eq!(
        hull_names(&dag, &hull_of_hull),
        hull_names(&dag, &dag.d_convex_hull(&[a, c]).unwrap())
    );
}

#[test]
fn test_strong_d_convex_hull_superset_of_d_convex_hull() {
    let (dag, _a, _b, _e, d) = multi_parent_collider();
    let plain = dag.d_convex_hull(&[d]).unwrap();
    let strong = dag.strong_d_convex_hull(&[d]).unwrap();
    assert!(plain.iter().all(|n| strong.contains(n)));
}

#[test]
fn test_empty_required_is_trivial() {
    let (dag, ..) = chain();
    let hull = dag.d_convex_hull(&[]).unwrap();
    assert!(hull.is_empty());
    let strong = dag.strong_d_convex_hull(&[]).unwrap();
    assert!(strong.is_empty());
}

#[test]
fn test_invalid_node_id_rejected() {
    let (dag, ..) = chain();
    let bad = NodeId(99);
    assert!(matches!(
        dag.d_convex_hull(&[bad]),
        Err(CausasvError::InvalidNodeId(_))
    ));
    assert!(matches!(
        dag.strong_d_convex_hull(&[bad]),
        Err(CausasvError::InvalidNodeId(_))
    ));
}

// ── Dag::induced_subgraph ────────────────────────────────────────────────────

#[test]
fn test_induced_subgraph_reduces_to_hull() {
    let (dag, a, b, c) = chain();
    let hull = dag.d_convex_hull(&[a, c]).unwrap();
    let reduced = dag
        .induced_subgraph(&hull.into_iter().collect::<Vec<_>>())
        .unwrap();
    assert_eq!(reduced.node_count(), 3);
    assert!(
        reduced
            .children(reduced.node_id("a").unwrap())
            .unwrap()
            .contains(&reduced.node_id("b").unwrap())
    );
    let _ = b;
}

#[test]
fn test_induced_subgraph_drops_edges_touching_removed_nodes() {
    let (dag, a, _b, _c) = chain();
    let reduced = dag.induced_subgraph(&[a]).unwrap();
    assert_eq!(reduced.node_count(), 1);
}

#[test]
fn test_induced_subgraph_invalid_node_id() {
    let (dag, ..) = chain();
    assert!(matches!(
        dag.induced_subgraph(&[NodeId(99)]),
        Err(CausasvError::InvalidNodeId(_))
    ));
}

// ── Cpdag::strong_d_convex_hull (Theorem 5 delegation) ──────────────────────

#[test]
fn test_cpdag_strong_d_convex_hull_via_directed_cpdag() {
    use causasv::Cpdag;
    let mut cpdag = Cpdag::new();
    let a = cpdag.add_node("a");
    let b = cpdag.add_node("b");
    let c = cpdag.add_node("c");
    cpdag.add_directed_edge(a, b).unwrap();
    cpdag.add_directed_edge(b, c).unwrap();
    let hull = cpdag.strong_d_convex_hull(&[a, c]).unwrap();
    let mut names: Vec<&str> = hull
        .iter()
        .map(|&id| cpdag.node_name(id).unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(names, vec!["a", "b", "c"]);
}
