use causasv::{AsvExplainer, CausasvError, Dag};

const EPS: f64 = 1e-9;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPS
}

fn additive(coalition: &[causasv::NodeId]) -> Result<f64, CausasvError> {
    Ok(coalition.len() as f64)
}

/// Diamond DAG: a→b, a→c, b→d, c→d  (n=4)
fn diamond() -> Dag {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    let d = dag.add_node("d");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(a, c).unwrap();
    dag.add_edge(b, d).unwrap();
    dag.add_edge(c, d).unwrap();
    dag
}

/// Fork DAG: root→a, root→b, root→c  (n=4, this IS a rooted tree)
fn chain5() -> Dag {
    let mut dag = Dag::new();
    let nodes: Vec<_> = (0..5).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..4 {
        dag.add_edge(nodes[i], nodes[i + 1]).unwrap();
    }
    dag
}

#[test]
fn test_diamond_dag_exact_dag_matches_exact() {
    let dag = diamond();
    let exact = AsvExplainer::new(dag.clone()).exact(additive).unwrap();
    let dp = AsvExplainer::new(dag).exact_dag(additive).unwrap();
    for node in exact.values.keys() {
        assert!(
            approx_eq(exact.values[node], dp.values[node]),
            "diamond mismatch at {node:?}: exact={} dp={}",
            exact.values[node],
            dp.values[node]
        );
    }
}

#[test]
fn test_chain_exact_dag_matches_exact() {
    let dag = chain5();
    let exact = AsvExplainer::new(dag.clone()).exact(additive).unwrap();
    let dp = AsvExplainer::new(dag).exact_dag(additive).unwrap();
    for node in exact.values.keys() {
        assert!(
            approx_eq(exact.values[node], dp.values[node]),
            "chain mismatch at {node:?}: exact={} dp={}",
            exact.values[node],
            dp.values[node]
        );
    }
}

#[test]
fn test_exact_dag_is_marked_exact() {
    let result = AsvExplainer::new(diamond()).exact_dag(additive).unwrap();
    assert!(result.is_exact);
    assert!(result.effective_sample_size.is_none());
}

#[test]
fn test_exact_dag_n21_returns_error() {
    let mut dag = Dag::new();
    for i in 0..21 {
        dag.add_node(&format!("n{i}"));
    }
    let err = AsvExplainer::new(dag).exact_dag(additive).unwrap_err();
    assert!(matches!(err, CausasvError::InvalidConfig(_)));
}

#[test]
fn test_auto_uses_exact_dag_for_medium_dag() {
    // n=9 chain: n>8, not a tree (wait, chain IS a tree... use diamond n=9)
    let mut dag = Dag::new();
    // n=9 diamond-like: a chain of 9 nodes
    let nodes: Vec<_> = (0..9).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..8 {
        dag.add_edge(nodes[i], nodes[i + 1]).unwrap();
    }
    // chain of 9 is a rooted tree, so auto goes exact_tree not exact_dag
    // use a 2-root DAG instead: two independent chains of length 5
    let mut dag2 = Dag::new();
    let a: Vec<_> = (0..5).map(|i| dag2.add_node(&format!("a{i}"))).collect();
    let b: Vec<_> = (0..5).map(|i| dag2.add_node(&format!("b{i}"))).collect();
    for i in 0..4 {
        dag2.add_edge(a[i], a[i + 1]).unwrap();
        dag2.add_edge(b[i], b[i + 1]).unwrap();
    }
    // n=10, 2 roots → not a rooted tree → auto should use exact_dag
    let result = AsvExplainer::new(dag2)
        .auto(additive, causasv::SamplingConfig::new(1000).with_seed(0))
        .unwrap();
    assert!(
        result.is_exact,
        "auto should pick exact_dag for n=10 non-tree DAG"
    );
}
