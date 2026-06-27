use causasv::{AsvExplainer, Dag, NodeId};

fn additive_value(coalition: &[NodeId]) -> Result<f64, causasv::CausasvError> {
    Ok(coalition.len() as f64)
}

fn diamond_dag() -> Dag {
    // a → b, a → c, b → d, c → d
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

fn chain_dag(n: usize) -> Dag {
    let mut dag = Dag::new();
    let mut prev = dag.add_node("n0");
    for i in 1..n {
        let cur = dag.add_node(&format!("n{i}"));
        dag.add_edge(prev, cur).unwrap();
        prev = cur;
    }
    dag
}

/// sparse DP must match dense DP on diamond (n=4, well within n≤20)
#[test]
fn test_sparse_matches_dense_diamond() {
    let dag = diamond_dag();
    let explainer = AsvExplainer::new(dag);
    let dense = explainer.exact_dag(additive_value).unwrap();
    let sparse = explainer.exact_dag_sparse(additive_value).unwrap();
    for (id, &dv) in &dense.values {
        let sv = sparse.values[id];
        assert!(
            (dv - sv).abs() < 1e-9,
            "mismatch on {id:?}: dense={dv}, sparse={sv}"
        );
    }
    assert!(sparse.is_exact);
    // sparse visits only valid ideals; dense iterates all 2^n masks — counts differ
    assert!(sparse.n_order_ideals.unwrap() <= dense.n_samples);
}

/// sparse must satisfy efficiency axiom Σφ_i = v(V) - v(∅)
#[test]
fn test_sparse_efficiency_axiom() {
    let dag = diamond_dag();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact_dag_sparse(additive_value).unwrap();
    let sum: f64 = result.values.values().sum();
    assert!(
        (sum - 4.0).abs() < 1e-9,
        "efficiency axiom: expected 4.0, got {sum}"
    );
}

/// chain of n=22 — sparse DP should handle it (only 23 order ideals: ∅, {0}, {0,1}, ..., all)
#[test]
fn test_sparse_chain_n22() {
    let dag = chain_dag(22);
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact_dag_sparse(additive_value).unwrap();
    assert!(result.is_exact);
    // Chain has only one topological ordering → exactly one order-ideal per prefix
    assert_eq!(result.n_order_ideals, Some(23)); // ∅ + n=22 prefix ideals
    // All values should equal 1.0 (single ordering, each marginal = 1)
    for &v in result.values.values() {
        assert!((v - 1.0).abs() < 1e-9, "chain value should be 1.0, got {v}");
    }
}

/// state_ratio for chain n=22 should be tiny (23 / 2^22 ≈ 0.0000055)
#[test]
fn test_sparse_chain_state_ratio() {
    let dag = chain_dag(22);
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact_dag_sparse(additive_value).unwrap();
    let ratio = result.state_ratio.unwrap();
    assert!(
        ratio < 0.001,
        "chain state_ratio should be tiny, got {ratio}"
    );
}

/// memory_mb should be reported and positive
#[test]
fn test_sparse_memory_mb_reported() {
    let dag = diamond_dag();
    let explainer = AsvExplainer::new(dag);
    let result = explainer.exact_dag_sparse(additive_value).unwrap();
    let mb = result.memory_mb.unwrap();
    assert!(mb > 0.0, "memory_mb should be positive");
}

/// Antichain on n=21 nodes has 21! ≈ 5.1×10^19 > u64::MAX (1.8×10^19).
/// exact_dag_sparse with max_nodes=28 should hit Overflow, not silently wrap.
/// Marked ignore because BFS visits 2^21 ≈ 2M states (~32 s in debug builds).
#[test]
#[ignore = "slow: 2^21 BFS states; run with --ignored to verify overflow guard"]
fn test_sparse_overflow_on_large_antichain() {
    use causasv::ExactDagConfig;

    let mut dag = Dag::new();
    for i in 0..21usize {
        dag.add_node(&format!("n{i}"));
    }
    // No edges → antichain; L(G) = 21! >> u64::MAX
    let explainer = AsvExplainer::new(dag);
    let config = ExactDagConfig {
        max_nodes: 28,
        memory_limit_bytes: 4 * 1024 * 1024 * 1024, // 4 GiB — high enough not to OOM
    };
    let result = explainer.exact_dag_sparse_with_config(additive_value, &config);
    match result {
        Err(causasv::CausasvError::Overflow(_)) => {} // expected
        Err(e) => panic!("expected Overflow, got {e:?}"),
        Ok(_) => panic!("expected Overflow error but got Ok — overflow guard is missing"),
    }
}
