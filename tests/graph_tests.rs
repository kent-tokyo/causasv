use causasv::{CausasvError, Dag, NodeId};

fn three_chain() -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    (dag, a, b, c)
}

#[test]
fn test_add_nodes() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    assert_eq!(dag.node_count(), 2);
    assert_eq!(dag.node_name(a), Some("a"));
    assert_eq!(dag.node_name(b), Some("b"));
    assert_eq!(dag.node_id("a"), Some(a));
    assert_eq!(dag.node_id("missing"), None);
}

#[test]
fn test_add_node_idempotent() {
    let mut dag = Dag::new();
    let a1 = dag.add_node("a");
    let a2 = dag.add_node("a");
    assert_eq!(a1, a2);
    assert_eq!(dag.node_count(), 1);
}

#[test]
fn test_add_edge_valid() {
    let (dag, a, b, c) = three_chain();
    dag.validate().unwrap();
    assert!(dag.children(a).unwrap().contains(&b));
    assert!(dag.parents(b).unwrap().contains(&a));
    assert!(dag.children(b).unwrap().contains(&c));
}

#[test]
fn test_self_loop() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    assert!(matches!(dag.add_edge(a, a), Err(CausasvError::SelfLoop(_))));
}

#[test]
fn test_duplicate_edge() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    dag.add_edge(a, b).unwrap();
    assert!(matches!(
        dag.add_edge(a, b),
        Err(CausasvError::DuplicateEdge(_, _))
    ));
}

#[test]
fn test_cycle_detected() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    dag.add_edge(c, a).unwrap(); // cycle: a→b→c→a
    assert!(matches!(dag.validate(), Err(CausasvError::CycleDetected)));
}

#[test]
fn test_disconnected_dag() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    let d = dag.add_node("d");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(c, d).unwrap();
    dag.validate().unwrap(); // two independent chains, valid DAG
    assert_eq!(dag.node_count(), 4);
}

#[test]
fn test_empty_graph_validate() {
    let dag = Dag::new();
    assert!(matches!(dag.validate(), Err(CausasvError::EmptyGraph)));
}

#[test]
fn test_invalid_node_id() {
    let mut dag = Dag::new();
    dag.add_node("a");
    let bad = NodeId(99);
    assert!(matches!(
        dag.children(bad),
        Err(CausasvError::InvalidNodeId(_))
    ));
    assert!(matches!(
        dag.parents(bad),
        Err(CausasvError::InvalidNodeId(_))
    ));
}

#[test]
fn test_all_nodes() {
    let (dag, a, b, c) = three_chain();
    let nodes: Vec<NodeId> = dag.all_nodes().collect();
    assert_eq!(nodes, vec![a, b, c]);
}
