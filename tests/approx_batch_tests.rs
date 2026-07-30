/// Tests for approximate_uniform_sparse_adaptive_batched.
use causasv::{AdaptiveSamplingConfig, AsvExplainer, CausasvError, Dag, NodeId, SamplingConfig};

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

fn make_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    dag
}

fn additive_batch(coalitions: &[Vec<NodeId>]) -> Result<Vec<f64>, causasv::CausasvError> {
    Ok(coalitions.iter().map(|c| c.len() as f64).collect())
}

fn weighted(s: &[NodeId]) -> Result<f64, causasv::CausasvError> {
    Ok(s.iter().map(|n| (n.0 + 1) as f64).sum())
}

fn weighted_batch(coalitions: &[Vec<NodeId>]) -> Result<Vec<f64>, causasv::CausasvError> {
    Ok(coalitions
        .iter()
        .map(|c| c.iter().map(|n| (n.0 + 1) as f64).sum())
        .collect())
}

/// Batched result must be close to exact_dag on diamond (non-additive v).
#[test]
fn test_batch_matches_exact_dag_diamond() {
    let dag = make_diamond();
    let explainer = AsvExplainer::new(dag);
    let exact = explainer.exact_dag(weighted).unwrap();
    let batched = explainer
        .approximate_uniform_sparse_adaptive_batched(
            weighted_batch,
            AdaptiveSamplingConfig::new()
                .with_max_samples(20_000)
                .with_seed(42),
        )
        .unwrap();
    for (&node, &phi_e) in &exact.values {
        let phi_b = batched.values[&node];
        assert!(
            (phi_b - phi_e).abs() < 0.15,
            "node {node:?}: exact={phi_e:.4}, batch={phi_b:.4}"
        );
    }
}

/// ESS must equal n_samples exactly (uniform weights, no IS correction).
#[test]
fn test_batch_ess_equals_n_samples() {
    let explainer = AsvExplainer::new(make_chain(8));
    let result = explainer
        .approximate_uniform_sparse_adaptive_batched(
            additive_batch,
            AdaptiveSamplingConfig::new()
                .with_max_samples(1_000)
                .with_seed(1),
        )
        .unwrap();
    assert_eq!(
        result.effective_sample_size.unwrap(),
        result.n_samples as f64,
        "ESS must equal n_samples for uniform sampling"
    );
}

/// Convergence flag is set; stderr is non-negative.
#[test]
fn test_batch_converges_and_has_stderr() {
    let explainer = AsvExplainer::new(make_chain(8));
    let result = explainer
        .approximate_uniform_sparse_adaptive_batched(
            additive_batch,
            AdaptiveSamplingConfig::new()
                .with_max_samples(50_000)
                .with_seed(7),
        )
        .unwrap();
    assert_eq!(
        result.converged,
        Some(true),
        "should converge on additive chain-8"
    );
    for (&node, &se) in result.stderr.as_ref().unwrap() {
        assert!(se >= 0.0, "stderr must be ≥ 0 for {node:?}, got {se}");
    }
}

/// Additive v(S)=|S| → ASV_i = 1.0 for all i (via non-batched exact reference).
#[test]
fn test_batch_additive_chain_accuracy() {
    let explainer = AsvExplainer::new(make_chain(6));
    let result = explainer
        .approximate_uniform_sparse_adaptive_batched(
            additive_batch,
            AdaptiveSamplingConfig::new()
                .with_max_samples(10_000)
                .with_seed(0),
        )
        .unwrap();
    for &v in result.values.values() {
        assert!(
            (v - 1.0).abs() < 0.05,
            "expected phi ≈ 1.0 for additive v on chain, got {v:.4}"
        );
    }
}

// ── value_fn_batch length-mismatch validation (n ≤ 64 small path) ──────────
//
// `approximate_batched`/`approximate_adaptive_batched`/
// `approximate_uniform_sparse_adaptive_batched` all zip a `value_fn_batch`
// result against the coalitions that were asked for; a callback returning the
// wrong number of values must be an explicit `CausasvError::ValueFunctionError`
// (via the shared `validate_batch_result_len` helper), not a silently dropped
// tail or an `unwrap()` panic on a coalition that never got cached. See the
// n > 64 equivalents in `tests/large_dag_approx_tests.rs`.

fn short_batch(coalitions: &[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError> {
    Ok(vec![0.0; coalitions.len().saturating_sub(1)])
}

fn long_batch(coalitions: &[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError> {
    let mut v = vec![0.0; coalitions.len()];
    v.push(0.0);
    Ok(v)
}

#[test]
fn small_path_batched_short_return_is_explicit_error() {
    let explainer = AsvExplainer::new(make_chain(40));
    let err = explainer
        .approximate_batched(
            short_batch,
            SamplingConfig::new(200).with_seed(1).with_batch_size(32),
        )
        .unwrap_err();
    assert!(matches!(err, CausasvError::ValueFunctionError(_)));
}

#[test]
fn small_path_batched_long_return_is_explicit_error() {
    let explainer = AsvExplainer::new(make_chain(40));
    let err = explainer
        .approximate_batched(
            long_batch,
            SamplingConfig::new(200).with_seed(1).with_batch_size(32),
        )
        .unwrap_err();
    assert!(matches!(err, CausasvError::ValueFunctionError(_)));
}

#[test]
fn small_path_adaptive_batched_short_return_is_explicit_error() {
    let explainer = AsvExplainer::new(make_chain(40));
    let config = AdaptiveSamplingConfig {
        min_samples: 50,
        max_samples: 200,
        batch_size: 32,
        seed: Some(1),
        ..AdaptiveSamplingConfig::default()
    };
    let err = explainer
        .approximate_adaptive_batched(short_batch, config)
        .unwrap_err();
    assert!(matches!(err, CausasvError::ValueFunctionError(_)));
}

#[test]
fn small_path_uniform_sparse_adaptive_batched_short_return_is_explicit_error() {
    let explainer = AsvExplainer::new(make_chain(8));
    let err = explainer
        .approximate_uniform_sparse_adaptive_batched(
            short_batch,
            AdaptiveSamplingConfig::new()
                .with_max_samples(1_000)
                .with_seed(1),
        )
        .unwrap_err();
    assert!(matches!(err, CausasvError::ValueFunctionError(_)));
}
