/// Golden tests: for additive v(S) = Σ w_i, ASV_i = w_i on any DAG.
///
/// Proof: the marginal contribution of node i in any topological ordering is
/// v(S ∪ {i}) − v(S) = w_i (constant, independent of S). Averaging constants
/// over all orderings leaves w_i unchanged, so ASV_i = w_i exactly.
///
/// These tests verify this identity through every exact code path.
use causasv::{AsvExplainer, AsvResult, Dag, ExactDagConfig, NodeId};

const EPS: f64 = 1e-10;

fn check_weights(result: AsvResult, weights: &[f64]) {
    for (i, &w) in weights.iter().enumerate() {
        let got = result.values[&NodeId(i as u32)];
        assert!(
            (got - w).abs() < EPS,
            "node {i}: expected {w}, got {got:.15}"
        );
    }
}

fn chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    dag
}

fn fork(n_leaves: usize) -> Dag {
    let mut dag = Dag::new();
    let root = dag.add_node("r");
    for i in 0..n_leaves {
        let l = dag.add_node(&format!("l{i}"));
        dag.add_edge(root, l).unwrap();
    }
    dag
}

fn collider(n_sources: usize) -> Dag {
    let mut dag = Dag::new();
    let sources: Vec<_> = (0..n_sources)
        .map(|i| dag.add_node(&format!("s{i}")))
        .collect();
    let sink = dag.add_node("sink");
    for &s in &sources {
        dag.add_edge(s, sink).unwrap();
    }
    dag
}

// ── exact() — brute-force topological enumeration ────────────────────────────

#[test]
fn test_additive_weights_exact_chain() {
    let ws = [1.0_f64, 2.0, 3.0, 4.0];
    let e = AsvExplainer::new(chain(4));
    check_weights(
        e.exact(move |s| Ok(s.iter().map(|n| ws[n.0 as usize]).sum()))
            .unwrap(),
        &ws,
    );
}

#[test]
fn test_additive_weights_exact_fork() {
    let ws = [1.0_f64, 2.0, 3.0, 4.0]; // root=NodeId(0), l0..l2=NodeId(1..3)
    let e = AsvExplainer::new(fork(3));
    check_weights(
        e.exact(move |s| Ok(s.iter().map(|n| ws[n.0 as usize]).sum()))
            .unwrap(),
        &ws,
    );
}

#[test]
fn test_additive_weights_exact_collider() {
    let ws = [1.0_f64, 2.0, 3.0, 4.0]; // s0..s2=NodeId(0..2), sink=NodeId(3)
    let e = AsvExplainer::new(collider(3));
    check_weights(
        e.exact(move |s| Ok(s.iter().map(|n| ws[n.0 as usize]).sum()))
            .unwrap(),
        &ws,
    );
}

// ── exact_dag() — dense order-ideal DP (2^n Vec cache) ───────────────────────

#[test]
fn test_additive_weights_exact_dag_chain() {
    let ws: Vec<f64> = (1..=12).map(|i| i as f64).collect();
    let dag = chain(12);
    let e = AsvExplainer::new(dag);
    check_weights(
        e.exact_dag(move |s| Ok(s.iter().map(|n| ws[n.0 as usize]).sum()))
            .unwrap(),
        &(1..=12).map(|i| i as f64).collect::<Vec<_>>(),
    );
}

// ── exact_dag_sparse_with_config() — BFS sparse DP ───────────────────────────

#[test]
fn test_additive_weights_exact_dag_sparse_chain() {
    let ws: Vec<f64> = (1..=15).map(|i| i as f64).collect();
    let dag = chain(15);
    let e = AsvExplainer::new(dag);
    check_weights(
        e.exact_dag_sparse_with_config(
            move |s| Ok(s.iter().map(|n| ws[n.0 as usize]).sum()),
            &ExactDagConfig::default(),
        )
        .unwrap(),
        &(1..=15).map(|i| i as f64).collect::<Vec<_>>(),
    );
}
