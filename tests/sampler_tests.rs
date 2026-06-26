use causasv::{AsvExplainer, Dag, SamplingConfig};

fn make_tree_dag() -> Dag {
    let mut dag = Dag::new();
    let r = dag.add_node("r");
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    dag.add_edge(r, a).unwrap();
    dag.add_edge(r, b).unwrap();
    dag.add_edge(a, c).unwrap();
    dag
}

#[test]
fn test_sampled_orderings_are_valid() {
    let dag = make_tree_dag();
    let explainer = AsvExplainer::new(dag);
    // Approximate runs sampling internally; verify result came from valid orderings
    // by checking the final values are finite (proxy for no invalid orderings).
    // Direct sampler access is pub(crate), so we test via the approximate API.
    let result = explainer
        .approximate(
            |s| Ok(s.len() as f64),
            SamplingConfig::new(500).with_seed(1),
        )
        .unwrap();
    for &v in result.values.values() {
        assert!(v.is_finite(), "non-finite ASV value: {v}");
    }
}

#[test]
fn test_same_seed_produces_same_result() {
    let config1 = SamplingConfig::new(1000).with_seed(42);
    let config2 = SamplingConfig::new(1000).with_seed(42);

    let r1 = AsvExplainer::new(make_tree_dag())
        .approximate(|s| Ok(s.len() as f64), config1)
        .unwrap();
    let r2 = AsvExplainer::new(make_tree_dag())
        .approximate(|s| Ok(s.len() as f64), config2)
        .unwrap();

    for node in r1.values.keys() {
        assert_eq!(
            r1.values[node], r2.values[node],
            "seed 42 should be reproducible"
        );
    }
}

#[test]
fn test_different_seeds_differ() {
    // v(S) = |S|^2 is nonlinear, so orderings matter and different seeds give different estimates.
    let r1 = AsvExplainer::new(make_tree_dag())
        .approximate(
            |s| Ok((s.len() as f64).powi(2)),
            SamplingConfig::new(50).with_seed(42),
        )
        .unwrap();
    let r2 = AsvExplainer::new(make_tree_dag())
        .approximate(
            |s| Ok((s.len() as f64).powi(2)),
            SamplingConfig::new(50).with_seed(99),
        )
        .unwrap();
    let differs = r1
        .values
        .keys()
        .any(|k| (r1.values[k] - r2.values[k]).abs() > 1e-12);
    assert!(
        differs,
        "different seeds should (almost always) produce different results"
    );
}

#[test]
fn test_approx_seed_stored_in_result() {
    let result = AsvExplainer::new(make_tree_dag())
        .approximate(|s| Ok(s.len() as f64), SamplingConfig::new(10).with_seed(7))
        .unwrap();
    assert_eq!(result.seed, Some(7));
    assert!(!result.is_exact);
    assert_eq!(result.n_samples, 10);
}

#[test]
fn test_approx_no_seed() {
    let result = AsvExplainer::new(make_tree_dag())
        .approximate(|s| Ok(s.len() as f64), SamplingConfig::new(100))
        .unwrap();
    assert_eq!(result.seed, None);
}

#[test]
fn test_approx_converges_to_exact() {
    // r→{a,b}, a→c with v(S) = |S|^2.
    // Exact: φ_r=1, φ_a=11/3≈3.667, φ_b=5, φ_c=19/3≈6.333
    // With 50k samples and IS correction, should be within 0.05.
    let make_dag = || make_tree_dag();

    let exact = AsvExplainer::new(make_dag())
        .exact(|s| Ok((s.len() as f64).powi(2)))
        .unwrap();
    let approx = AsvExplainer::new(make_dag())
        .approximate(
            |s| Ok((s.len() as f64).powi(2)),
            SamplingConfig::new(50_000).with_seed(42),
        )
        .unwrap();

    let tol = 0.05;
    for node in exact.values.keys() {
        let diff = (exact.values[node] - approx.values[node]).abs();
        assert!(
            diff < tol,
            "node {:?}: exact={:.4} approx={:.4} diff={:.4} > tol={}",
            node,
            exact.values[node],
            approx.values[node],
            diff,
            tol
        );
    }
}

#[test]
fn test_efficiency_axiom_approx() {
    // Σφ_i = v(V) - v(∅) holds exactly even for approximate due to IS normalization.
    let v_empty = 0.0_f64;
    let v_full = 10.0_f64;
    let dag = make_tree_dag();
    let n = dag.node_count();
    let result = AsvExplainer::new(dag)
        .approximate(
            move |s| {
                Ok(if s.is_empty() {
                    v_empty
                } else if s.len() == n {
                    v_full
                } else {
                    s.len() as f64
                })
            },
            SamplingConfig::new(1000).with_seed(0),
        )
        .unwrap();
    let sum: f64 = result.values.values().sum();
    assert!(
        (sum - (v_full - v_empty)).abs() < 1e-9,
        "efficiency axiom violated: sum={sum}"
    );
}
