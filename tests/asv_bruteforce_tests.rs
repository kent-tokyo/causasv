use causasv::{AsvExplainer, Dag, NodeId};

fn chain_3() -> (Dag, NodeId, NodeId, NodeId) {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    (dag, a, b, c)
}

const EPS: f64 = 1e-10;

#[test]
fn test_single_node() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .exact(|s| Ok(if s.contains(&a) { 5.0 } else { 0.0 }))
        .unwrap();
    assert!((result.values[&a] - 5.0).abs() < EPS);
    assert!(result.is_exact);
    assert_eq!(result.n_samples, 1);
}

#[test]
fn test_chain_3_additive() {
    // Only one topological ordering [a,b,c] → all marginals equal 1.
    let (dag, a, b, c) = chain_3();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact(|s| Ok(s.len() as f64)).unwrap();
    assert!((result.values[&a] - 1.0).abs() < EPS);
    assert!((result.values[&b] - 1.0).abs() < EPS);
    assert!((result.values[&c] - 1.0).abs() < EPS);
}

#[test]
fn test_chain_3_nonadditive() {
    // v(S) = |S|^2. Single ordering [a,b,c]:
    // φ_a = 1-0=1, φ_b = 4-1=3, φ_c = 9-4=5
    let (dag, a, b, c) = chain_3();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact(|s| Ok((s.len() as f64).powi(2))).unwrap();
    assert!((result.values[&a] - 1.0).abs() < EPS);
    assert!((result.values[&b] - 3.0).abs() < EPS);
    assert!((result.values[&c] - 5.0).abs() < EPS);
}

#[test]
fn test_fork_dag() {
    // a→b, a→c. Two orderings: [a,b,c] and [a,c,b].
    // v(S) = |S|^2: φ_a=1, φ_b=(3+5)/2=4, φ_c=(5+3)/2=4
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(a, c).unwrap();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact(|s| Ok((s.len() as f64).powi(2))).unwrap();
    assert!((result.values[&a] - 1.0).abs() < EPS);
    assert!((result.values[&b] - 4.0).abs() < EPS);
    assert!((result.values[&c] - 4.0).abs() < EPS);
    assert_eq!(result.n_samples, 2);
}

#[test]
fn test_collider_dag() {
    // a→c, b→c. Two orderings: [a,b,c] and [b,a,c].
    // v(S) = |S|^2: φ_a=(1+3)/2=2, φ_b=(3+1)/2=2, φ_c=5
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, c).unwrap();
    dag.add_edge(b, c).unwrap();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact(|s| Ok((s.len() as f64).powi(2))).unwrap();
    assert!((result.values[&a] - 2.0).abs() < EPS);
    assert!((result.values[&b] - 2.0).abs() < EPS);
    assert!((result.values[&c] - 5.0).abs() < EPS);
}

#[test]
fn test_efficiency_axiom() {
    // Σφ_i = v(V) - v(∅) for any DAG and value function.
    let (dag, _, _, _) = chain_3();
    let v_empty = 0.0;
    let v_full = 10.0;
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .exact(|s| {
            Ok(if s.is_empty() {
                v_empty
            } else if s.len() == 3 {
                v_full
            } else {
                s.len() as f64 * 2.5
            })
        })
        .unwrap();
    let sum: f64 = result.values.values().sum();
    assert!((sum - (v_full - v_empty)).abs() < EPS);
}

#[test]
fn test_exact_is_deterministic() {
    let (dag, a, b, c) = chain_3();
    let explainer = AsvExplainer::new(dag);
    let r1 = explainer.exact(|s| Ok(s.len() as f64)).unwrap();
    let (dag2, _, _, _) = chain_3();
    let explainer2 = AsvExplainer::new(dag2);
    let r2 = explainer2.exact(|s| Ok(s.len() as f64)).unwrap();
    assert_eq!(r1.values[&a], r2.values[&a]);
    assert_eq!(r1.values[&b], r2.values[&b]);
    assert_eq!(r1.values[&c], r2.values[&c]);
}

#[test]
fn test_rooted_tree_r_ab_ac() {
    // r→{a,b}, a→c: 3 orderings, v(S) = |S|^2.
    // ord [r,a,b,c]: φ_r=1, φ_a=3, φ_b=5, φ_c=7
    // ord [r,a,c,b]: φ_r=1, φ_a=3, φ_c=5, φ_b=7
    // ord [r,b,a,c]: φ_r=1, φ_b=3, φ_a=5, φ_c=7
    // Averages: φ_r=1, φ_a=(3+3+5)/3=11/3, φ_b=(5+7+3)/3=5, φ_c=(7+5+7)/3=19/3
    let mut dag = Dag::new();
    let r = dag.add_node("r");
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(r, a).unwrap();
    dag.add_edge(r, b).unwrap();
    dag.add_edge(a, c).unwrap();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact(|s| Ok((s.len() as f64).powi(2))).unwrap();
    assert!((result.values[&r] - 1.0).abs() < EPS);
    assert!((result.values[&a] - 11.0 / 3.0).abs() < EPS);
    assert!((result.values[&b] - 5.0).abs() < EPS);
    assert!((result.values[&c] - 19.0 / 3.0).abs() < EPS);
}
