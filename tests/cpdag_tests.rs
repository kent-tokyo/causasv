use causasv::{CausasvError, Cpdag, NodeId};

fn chain() -> (Cpdag, NodeId, NodeId, NodeId) {
    // a -> b -> c, fully directed (already a DAG).
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_directed_edge(a, b).unwrap();
    g.add_directed_edge(b, c).unwrap();
    (g, a, b, c)
}

#[test]
fn test_add_nodes() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    assert_eq!(g.node_count(), 2);
    assert_eq!(g.node_name(a), Some("a"));
    assert_eq!(g.node_id("b"), Some(b));
    assert_eq!(g.node_id("missing"), None);
}

#[test]
fn test_add_node_idempotent() {
    let mut g = Cpdag::new();
    let a1 = g.add_node("a");
    let a2 = g.add_node("a");
    assert_eq!(a1, a2);
    assert_eq!(g.node_count(), 1);
}

#[test]
fn test_directed_edge_self_loop() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    assert!(matches!(
        g.add_directed_edge(a, a),
        Err(CausasvError::SelfLoop(_))
    ));
}

#[test]
fn test_undirected_edge_self_loop() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    assert!(matches!(
        g.add_undirected_edge(a, a),
        Err(CausasvError::SelfLoop(_))
    ));
}

#[test]
fn test_duplicate_directed_edge() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_directed_edge(a, b).unwrap();
    assert!(matches!(
        g.add_directed_edge(a, b),
        Err(CausasvError::DuplicateEdge(_, _))
    ));
}

#[test]
fn test_duplicate_undirected_edge() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_undirected_edge(a, b).unwrap();
    assert!(matches!(
        g.add_undirected_edge(b, a),
        Err(CausasvError::DuplicateEdge(_, _))
    ));
}

#[test]
fn test_conflicting_edge_reverse_directed() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_directed_edge(a, b).unwrap();
    assert!(matches!(
        g.add_directed_edge(b, a),
        Err(CausasvError::ConflictingEdge(_, _))
    ));
}

#[test]
fn test_conflicting_edge_directed_vs_undirected() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_directed_edge(a, b).unwrap();
    assert!(matches!(
        g.add_undirected_edge(a, b),
        Err(CausasvError::ConflictingEdge(_, _))
    ));

    let mut g2 = Cpdag::new();
    let a2 = g2.add_node("a");
    let b2 = g2.add_node("b");
    g2.add_undirected_edge(a2, b2).unwrap();
    assert!(matches!(
        g2.add_directed_edge(a2, b2),
        Err(CausasvError::ConflictingEdge(_, _))
    ));
}

#[test]
fn test_directed_cycle_rejected_by_validate_pdag() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_directed_edge(a, b).unwrap();
    g.add_directed_edge(b, c).unwrap();
    g.add_directed_edge(c, a).unwrap();
    assert!(matches!(
        g.validate_pdag(),
        Err(CausasvError::CycleDetected)
    ));
}

#[test]
fn test_empty_graph_validate() {
    let g = Cpdag::new();
    assert!(matches!(g.validate_pdag(), Err(CausasvError::EmptyGraph)));
}

#[test]
fn test_disconnected_pdag_valid() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    let d = g.add_node("d");
    g.add_directed_edge(a, b).unwrap();
    g.add_undirected_edge(c, d).unwrap();
    g.validate_pdag().unwrap();
    assert_eq!(g.node_count(), 4);
}

#[test]
fn test_all_nodes_insertion_order() {
    let (g, a, b, c) = chain();
    let nodes: Vec<NodeId> = g.all_nodes().collect();
    assert_eq!(nodes, vec![a, b, c]);
}

#[test]
fn test_directed_edges_iteration() {
    let (g, a, b, c) = chain();
    let edges: Vec<(NodeId, NodeId)> = g.directed_edges().collect();
    assert_eq!(edges, vec![(a, b), (b, c)]);
}

#[test]
fn test_undirected_edges_iteration_canonical_order() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_undirected_edge(b, a).unwrap(); // added as (b, a) but a < b
    g.add_undirected_edge(c, b).unwrap();
    let edges: Vec<(NodeId, NodeId)> = g.undirected_edges().collect();
    assert_eq!(edges, vec![(a, b), (b, c)]);
}

// ── consistent_extension ────────────────────────────────────────────────────

#[test]
fn test_extension_of_fully_directed_graph_is_itself() {
    let (g, a, b, c) = chain();
    let dag = g.consistent_extension().unwrap();
    dag.validate().unwrap();
    assert!(dag.children(a).unwrap().contains(&b));
    assert!(dag.children(b).unwrap().contains(&c));
}

#[test]
fn test_extension_orients_single_undirected_edge() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    g.add_undirected_edge(a, b).unwrap();
    let dag = g.consistent_extension().unwrap();
    dag.validate().unwrap();
    // Either orientation is a valid DAG; just confirm it's fully directed & acyclic.
    assert_eq!(dag.node_count(), 2);
}

#[test]
fn test_extension_triangle_k3_is_chordal_and_extendable() {
    // Fully-undirected triangle: a-b, b-c, a-c. Chordal (trivially, K3 has no
    // chordless cycle >= 4), so a consistent extension must exist.
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_undirected_edge(a, b).unwrap();
    g.add_undirected_edge(b, c).unwrap();
    g.add_undirected_edge(a, c).unwrap();
    let dag = g.consistent_extension().unwrap();
    dag.validate().unwrap();
    assert_eq!(dag.node_count(), 3);
    g.validate_cpdag().unwrap();
}

#[test]
fn test_extension_chordless_4cycle_not_extendable() {
    // a-b-c-d-a, no diagonal: chordless 4-cycle. No CPDAG can have this as a
    // chain component (Andersson-Madigan-Perlman chordality theorem), so
    // Dor-Tarsi must fail to find a valid extension.
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    let d = g.add_node("d");
    g.add_undirected_edge(a, b).unwrap();
    g.add_undirected_edge(b, c).unwrap();
    g.add_undirected_edge(c, d).unwrap();
    g.add_undirected_edge(d, a).unwrap();
    assert!(matches!(
        g.consistent_extension(),
        Err(CausasvError::NotExtendable)
    ));
    assert!(matches!(
        g.validate_cpdag(),
        Err(CausasvError::NotExtendable)
    ));
}

#[test]
fn test_extension_preserves_directed_edges() {
    // a -> b (compelled), b - c (undirected). Extension must keep a -> b.
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_directed_edge(a, b).unwrap();
    g.add_undirected_edge(b, c).unwrap();
    let dag = g.consistent_extension().unwrap();
    assert!(dag.children(a).unwrap().contains(&b));
}

#[test]
fn test_extension_preserves_skeleton_size() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_directed_edge(a, b).unwrap();
    g.add_undirected_edge(b, c).unwrap();
    let dag = g.consistent_extension().unwrap();
    let n_edges: usize = dag
        .all_nodes()
        .map(|id| dag.children(id).unwrap().len())
        .sum();
    assert_eq!(n_edges, 2); // one directed + one (now oriented) undirected
}

// ── induced_subgraph ────────────────────────────────────────────────────────

#[test]
fn test_induced_subgraph_preserves_edge_kind() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_directed_edge(a, b).unwrap();
    g.add_undirected_edge(b, c).unwrap();

    let sub = g.induced_subgraph(&[a, b, c]).unwrap();
    let a2 = sub.node_id("a").unwrap();
    let b2 = sub.node_id("b").unwrap();
    let c2 = sub.node_id("c").unwrap();
    assert_eq!(sub.directed_edges().collect::<Vec<_>>(), vec![(a2, b2)]);
    assert_eq!(sub.undirected_edges().collect::<Vec<_>>(), vec![(b2, c2)]);
}

#[test]
fn test_induced_subgraph_drops_edges_touching_removed_nodes() {
    let mut g = Cpdag::new();
    let a = g.add_node("a");
    let b = g.add_node("b");
    let c = g.add_node("c");
    g.add_directed_edge(a, b).unwrap();
    g.add_undirected_edge(b, c).unwrap();

    let sub = g.induced_subgraph(&[a, b]).unwrap();
    assert_eq!(sub.node_count(), 2);
    assert_eq!(sub.undirected_edges().count(), 0);
}

#[test]
fn test_induced_subgraph_invalid_node_id() {
    let (g, _a, _b, _c) = chain();
    let bad = NodeId(99);
    assert!(matches!(
        g.induced_subgraph(&[bad]),
        Err(CausasvError::InvalidNodeId(_))
    ));
}

#[test]
fn test_induced_subgraph_preserves_original_order_not_keep_order() {
    let (g, a, b, c) = chain();
    // Pass keep in reverse order; result should still be a, b, c.
    let sub = g.induced_subgraph(&[c, b, a]).unwrap();
    let names: Vec<&str> = sub
        .all_nodes()
        .map(|id| sub.node_name(id).unwrap())
        .collect();
    assert_eq!(names, vec!["a", "b", "c"]);
}
