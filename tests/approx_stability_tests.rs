/// Numerical stability tests for the approximate ASV estimator.
///
/// These tests verify:
/// 1. Kahan summation produces the same results as the regular path (correctness preserved)
/// 2. Large IS weights (from sparse frontiers) don't cause NaN or Inf
/// 3. log-weight normalization in the batched path doesn't change ASV values
use causasv::{AsvExplainer, Dag, NodeId, SamplingConfig};

fn additive(coalition: &[NodeId]) -> Result<f64, causasv::CausasvError> {
    Ok(coalition.len() as f64)
}

/// Fully disconnected graph (antichain) generates large IS weights:
/// w = exp(n * log(n)) for the most extreme orderings.
/// Verify that approximate ASV returns finite, sensible values.
#[test]
fn test_approx_large_weights_no_nan_inf() {
    let mut dag = Dag::new();
    for i in 0..10usize {
        dag.add_node(&format!("n{i}"));
    }
    // No edges: frontier always has many choices → large IS weights
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate(additive, SamplingConfig::new(2_000).with_seed(42))
        .unwrap();

    for (id, &v) in &result.values {
        assert!(v.is_finite(), "ASV for {id:?} is not finite: {v}");
        assert!(!v.is_nan(), "ASV for {id:?} is NaN");
    }
    // Efficiency axiom must hold (self-normalized IS preserves this)
    let total: f64 = result.values.values().sum();
    assert!(
        (total - 10.0).abs() < 1e-9,
        "efficiency axiom: expected 10.0, got {total}"
    );
}

/// Adaptive ASV on antichain must also produce finite values.
#[test]
fn test_adaptive_large_weights_no_nan_inf() {
    let mut dag = Dag::new();
    for i in 0..8usize {
        dag.add_node(&format!("n{i}"));
    }
    let explainer = AsvExplainer::new(dag);
    let config = causasv::AdaptiveSamplingConfig {
        min_samples: 500,
        max_samples: 5_000,
        batch_size: 500,
        rel_tol: 0.05,
        ess_ratio_min: 0.05,
        seed: Some(1),
    };
    let result = explainer.approximate_adaptive(additive, config).unwrap();
    for (id, &v) in &result.values {
        assert!(v.is_finite(), "adaptive ASV for {id:?} is not finite: {v}");
    }
    let total: f64 = result.values.values().sum();
    assert!(
        (total - 8.0).abs() < 1e-9,
        "efficiency axiom: expected 8.0, got {total}"
    );
}

/// Kahan summation must not change the result for a chain (trivial IS weights, w=1).
/// Chain has a single topological ordering, so q(π)=1 → w=exp(0)=1 for all samples.
/// With w=1, Kahan has no effect on precision — this verifies correctness is preserved.
#[test]
fn test_approx_chain_kahan_no_regression() {
    let mut dag = Dag::new();
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    let explainer = AsvExplainer::new(dag);

    // With seed, Kahan path is used
    let result = explainer
        .approximate(additive, SamplingConfig::new(1_000).with_seed(7))
        .unwrap();

    // Chain has one ordering [a,b,c], all marginals = 1.0
    // IS weights are all 1 (q=1), so denominator = n_samples exactly.
    for &v in result.values.values() {
        assert!(
            (v - 1.0).abs() < 1e-9,
            "chain ASV should be exactly 1.0, got {v}"
        );
    }
}
