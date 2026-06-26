use causasv::{enumerate_topos, topo_sort, Dag, NodeId};

fn is_valid_topo(dag: &Dag, ordering: &[NodeId]) -> bool {
    if ordering.len() != dag.node_count() {
        return false;
    }
    let mut pos = vec![0usize; dag.node_count()];
    for (i, &node) in ordering.iter().enumerate() {
        pos[node.0 as usize] = i;
    }
    for u in dag.all_nodes() {
        for &v in dag.children(u).unwrap() {
            if pos[u.0 as usize] >= pos[v.0 as usize] {
                return false;
            }
        }
    }
    true
}

#[test]
fn test_topo_sort_chain() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    let d = dag.add_node("d");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    dag.add_edge(c, d).unwrap();
    let order = topo_sort(&dag).unwrap();
    assert_eq!(order, vec![a, b, c, d]);
}

#[test]
fn test_topo_sort_diamond() {
    // a→b, a→c, b→d, c→d
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    let d = dag.add_node("d");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(a, c).unwrap();
    dag.add_edge(b, d).unwrap();
    dag.add_edge(c, d).unwrap();
    let order = topo_sort(&dag).unwrap();
    assert!(is_valid_topo(&dag, &order));
    assert_eq!(order[0], a); // a must be first
    assert_eq!(order[3], d); // d must be last
}

#[test]
fn test_enumerate_topos_chain_has_one_ordering() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    let orderings = enumerate_topos(&dag).unwrap();
    assert_eq!(orderings.len(), 1);
    assert_eq!(orderings[0], vec![a, b, c]);
}

#[test]
fn test_enumerate_topos_two_independent_nodes() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let orderings = enumerate_topos(&dag).unwrap();
    assert_eq!(orderings.len(), 2);
    // deterministic: ascending NodeId order → [a,b] before [b,a]
    assert_eq!(orderings[0], vec![a, b]);
    assert_eq!(orderings[1], vec![b, a]);
}

#[test]
fn test_enumerate_topos_all_valid() {
    // r→{a,b}, a→c: 3 linear extensions
    let mut dag = Dag::new();
    let r = dag.add_node("r");
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(r, a).unwrap();
    dag.add_edge(r, b).unwrap();
    dag.add_edge(a, c).unwrap();
    let orderings = enumerate_topos(&dag).unwrap();
    assert_eq!(orderings.len(), 3);
    for ord in &orderings {
        assert!(is_valid_topo(&dag, ord), "invalid ordering: {:?}", ord);
    }
    // r must be first in all orderings
    for ord in &orderings {
        assert_eq!(ord[0], r);
    }
}

#[test]
fn test_enumerate_topos_deterministic() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    dag.add_edge(a, b).unwrap();
    let ord1 = enumerate_topos(&dag).unwrap();
    let ord2 = enumerate_topos(&dag).unwrap();
    assert_eq!(ord1, ord2);
}

#[test]
fn test_topo_sort_empty() {
    let dag = Dag::new();
    let order = topo_sort(&dag).unwrap();
    assert!(order.is_empty());
}
