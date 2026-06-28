/// Tests for approximate_uniform: uniform topological order sampling with ESS = n_samples.
use causasv::{AsvExplainer, Dag, NodeId, SamplingConfig};

fn additive(s: &[NodeId]) -> Result<f64, causasv::CausasvError> {
    Ok(s.len() as f64)
}

fn make_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    dag
}

fn make_antichain(n: usize) -> Dag {
    let mut dag = Dag::new();
    for i in 0..n {
        dag.add_node(&format!("n{i}"));
    }
    dag
}

fn make_diamond() -> Dag {
    let mut dag = Dag::new();
    let src = dag.add_node("src");
    let m0 = dag.add_node("m0");
    let m1 = dag.add_node("m1");
    let snk = dag.add_node("snk");
    dag.add_edge(src, m0).unwrap();
    dag.add_edge(src, m1).unwrap();
    dag.add_edge(m0, snk).unwrap();
    dag.add_edge(m1, snk).unwrap();
    dag
}

/// ESS must equal n_samples exactly for uniform sampling.
#[test]
fn test_ess_equals_n_samples() {
    for dag in [make_chain(5), make_diamond(), make_antichain(6)] {
        let explainer = AsvExplainer::new(dag);
        let result = explainer
            .approximate_uniform(additive, SamplingConfig::new(500).with_seed(42))
            .unwrap();
        let ess = result.effective_sample_size.unwrap();
        assert_eq!(
            ess, 500.0,
            "ESS should equal n_samples exactly for uniform sampling, got {ess}"
        );
    }
}

/// Efficiency axiom: Σφ_i = v(V) - v(∅) = n - 0 = n.
#[test]
fn test_efficiency_axiom() {
    let dag = make_diamond();
    let n = dag.node_count() as f64;
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate_uniform(additive, SamplingConfig::new(1_000).with_seed(1))
        .unwrap();
    let total: f64 = result.values.values().sum();
    assert!(
        (total - n).abs() < 1e-9,
        "efficiency axiom: expected {n}, got {total}"
    );
}

/// Same seed → identical result (deterministic).
#[test]
fn test_seeded_deterministic() {
    let dag = make_diamond();
    let explainer = AsvExplainer::new(dag);
    let cfg = SamplingConfig::new(200).with_seed(7);
    let r1 = explainer.approximate_uniform(additive, cfg).unwrap();
    let cfg = SamplingConfig::new(200).with_seed(7);
    let r2 = explainer.approximate_uniform(additive, cfg).unwrap();
    for (&node, &v1) in &r1.values {
        let v2 = r2.values[&node];
        assert_eq!(v1, v2, "seeded results differ for {node:?}");
    }
}

/// For additive v(S)=|S|, exact ASV = 1.0 per node on any DAG.
/// approximate_uniform should converge to this with small error.
#[test]
fn test_matches_exact_additive_chain() {
    let dag = make_chain(6);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate_uniform(additive, SamplingConfig::new(2_000).with_seed(42))
        .unwrap();
    for &v in result.values.values() {
        assert!(
            (v - 1.0).abs() < 0.05,
            "expected phi ≈ 1.0 for additive v, got {v:.4}"
        );
    }
}

/// approximate_uniform vs exact_dag on diamond with non-trivial value function.
#[test]
fn test_matches_exact_dag_diamond() {
    let weighted = |s: &[NodeId]| Ok(s.iter().map(|n| (n.0 + 1) as f64).sum::<f64>());
    let dag = make_diamond();
    let explainer = AsvExplainer::new(dag);
    let exact = explainer.exact_dag(weighted).unwrap();
    let approx = explainer
        .approximate_uniform(weighted, SamplingConfig::new(5_000).with_seed(99))
        .unwrap();
    for (&node, &phi_e) in &exact.values {
        let phi_a = approx.values[&node];
        assert!(
            (phi_a - phi_e).abs() < 0.15,
            "node {node:?}: exact={phi_e:.4}, uniform={phi_a:.4}"
        );
    }
}

/// Antichain: every permutation is valid → uniform sampling = exact uniform average.
/// For additive v, phi_i = 1.0 for all i.
#[test]
fn test_antichain_uniform() {
    let dag = make_antichain(5);
    let explainer = AsvExplainer::new(dag);
    // Antichain: frontier IS gives very low ESS because weights vary wildly.
    // Uniform sampling should handle it cleanly.
    let result = explainer
        .approximate_uniform(additive, SamplingConfig::new(500).with_seed(3))
        .unwrap();
    for &v in result.values.values() {
        assert!(
            (v - 1.0).abs() < 0.10,
            "antichain: expected phi ≈ 1.0, got {v:.4}"
        );
    }
}

/// n > 20 must return an error.
#[test]
fn test_rejects_large_dag() {
    let mut dag = Dag::new();
    for i in 0..21 {
        dag.add_node(&format!("n{i}"));
    }
    let explainer = AsvExplainer::new(dag);
    let result = explainer.approximate_uniform(additive, SamplingConfig::new(10));
    assert!(result.is_err(), "should error for n=21");
}
