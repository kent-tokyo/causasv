use causasv::{AsvExplainer, CausasvError, Dag};

const EPS: f64 = 1e-9;

fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() < eps
}

#[test]
fn test_not_rooted_tree_two_roots() {
    // Two independent nodes: two roots → NotRootedTree
    let mut dag = Dag::new();
    dag.add_node("a");
    dag.add_node("b");
    let explainer = AsvExplainer::new(dag);
    assert!(matches!(
        explainer.exact_tree(|s| Ok(s.len() as f64)),
        Err(CausasvError::NotRootedTree)
    ));
}

#[test]
fn test_not_rooted_tree_diamond() {
    // a→c, b→c, a→b: node c has in-degree 2 → NotRootedTree
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(a, c).unwrap();
    dag.add_edge(b, c).unwrap();
    let explainer = AsvExplainer::new(dag);
    assert!(matches!(
        explainer.exact_tree(|s| Ok(s.len() as f64)),
        Err(CausasvError::NotRootedTree)
    ));
}

#[test]
fn test_chain_matches_bruteforce() {
    let mut dag = Dag::new();
    let nodes: Vec<_> = (0..4).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..3 {
        dag.add_edge(nodes[i], nodes[i + 1]).unwrap();
    }
    let dag2 = {
        let mut d = Dag::new();
        let ns: Vec<_> = (0..4).map(|i| d.add_node(&format!("n{i}"))).collect();
        for i in 0..3 {
            d.add_edge(ns[i], ns[i + 1]).unwrap();
        }
        d
    };
    let exact = AsvExplainer::new(dag)
        .exact(|s| Ok(s.len() as f64))
        .unwrap();
    let tree = AsvExplainer::new(dag2)
        .exact_tree(|s| Ok(s.len() as f64))
        .unwrap();
    for node in exact.values.keys() {
        assert!(
            approx_eq(exact.values[node], tree.values[node], EPS),
            "mismatch at {:?}: exact={} tree={}",
            node,
            exact.values[node],
            tree.values[node]
        );
    }
}

#[test]
fn test_rooted_tree_matches_bruteforce() {
    // r→{a,b}, a→c with v(S) = |S|^2
    let make_dag = || {
        let mut dag = Dag::new();
        let r = dag.add_node("r");
        let a = dag.add_node("a");
        let b = dag.add_node("b");
        let c = dag.add_node("c");
        dag.add_edge(r, a).unwrap();
        dag.add_edge(r, b).unwrap();
        dag.add_edge(a, c).unwrap();
        dag
    };
    let exact = AsvExplainer::new(make_dag())
        .exact(|s| Ok((s.len() as f64).powi(2)))
        .unwrap();
    let tree = AsvExplainer::new(make_dag())
        .exact_tree(|s| Ok((s.len() as f64).powi(2)))
        .unwrap();
    for node in exact.values.keys() {
        assert!(
            approx_eq(exact.values[node], tree.values[node], EPS),
            "mismatch at {:?}: exact={} tree={}",
            node,
            exact.values[node],
            tree.values[node]
        );
    }
}

#[test]
fn test_balanced_tree_matches_bruteforce() {
    // 3-level balanced binary tree: root→{l,r}, l→{ll,lr}, r→{rl,rr}  (7 nodes)
    let make_dag = || {
        let mut dag = Dag::new();
        let root = dag.add_node("root");
        let l = dag.add_node("l");
        let r = dag.add_node("r");
        let ll = dag.add_node("ll");
        let lr = dag.add_node("lr");
        let rl = dag.add_node("rl");
        let rr = dag.add_node("rr");
        dag.add_edge(root, l).unwrap();
        dag.add_edge(root, r).unwrap();
        dag.add_edge(l, ll).unwrap();
        dag.add_edge(l, lr).unwrap();
        dag.add_edge(r, rl).unwrap();
        dag.add_edge(r, rr).unwrap();
        dag
    };
    let exact = AsvExplainer::new(make_dag())
        .exact(|s| Ok((s.len() as f64).powi(2)))
        .unwrap();
    let tree = AsvExplainer::new(make_dag())
        .exact_tree(|s| Ok((s.len() as f64).powi(2)))
        .unwrap();
    for node in exact.values.keys() {
        assert!(
            approx_eq(exact.values[node], tree.values[node], EPS),
            "mismatch at {:?}: exact={} tree={}",
            node,
            exact.values[node],
            tree.values[node]
        );
    }
}

#[test]
fn test_exact_tree_is_marked_exact() {
    let mut dag = Dag::new();
    let r = dag.add_node("r");
    let a = dag.add_node("a");
    dag.add_edge(r, a).unwrap();
    let result = AsvExplainer::new(dag)
        .exact_tree(|s| Ok(s.len() as f64))
        .unwrap();
    assert!(result.is_exact);
    assert!(result.seed.is_none());
}
