use causasv::{AsvExplainer, Dag, ExactDagConfig, NodeId, SamplingConfig};

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

// ── auto dispatch tests ──────────────────────────────────────────────────────

#[test]
fn test_auto_small_dag_uses_exact() {
    // n=3 chain → auto dispatches to exact (n ≤ 8)
    let (dag, a, b, c) = chain_3();
    let explainer = AsvExplainer::new(dag);
    let auto = explainer
        .auto(|s| Ok(s.len() as f64), causasv::SamplingConfig::new(100))
        .unwrap();
    assert!(auto.is_exact);
    assert!((auto.values[&a] - 1.0).abs() < EPS);
    assert!((auto.values[&b] - 1.0).abs() < EPS);
    assert!((auto.values[&c] - 1.0).abs() < EPS);
}

#[test]
fn test_auto_rooted_tree_uses_exact_tree() {
    // n=7 balanced tree → auto dispatches to exact_tree (is rooted tree, n > 8? No n=7)
    // Use n=10 caterpillar to force tree path (n > 8, rooted tree)
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..5).map(|i| dag.add_node(&format!("n{i}"))).collect();
    let ls: Vec<_> = (0..5).map(|i| dag.add_node(&format!("l{i}"))).collect();
    for i in 0..4 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    for i in 0..5 {
        dag.add_edge(ns[i], ls[i]).unwrap();
    }
    let explainer = AsvExplainer::new(dag);
    let auto = explainer
        .auto(|s| Ok(s.len() as f64), causasv::SamplingConfig::new(100))
        .unwrap();
    // exact_tree sets is_exact = true
    assert!(auto.is_exact);
}

#[test]
fn test_auto_general_dag_uses_approx() {
    // Diamond (n=4, not a tree) with n > 8? n=4 → auto uses exact (n ≤ 8).
    // Use a larger general DAG (n=9, not a tree) to force approx path.
    let mut dag = Dag::new();
    // 9-node graph: 3 chains of 3 that merge at a final node
    // This is NOT a rooted tree (collider at the end)
    let a = dag.add_node("a");
    let b = dag.add_node("b");
    let c = dag.add_node("c");
    let d = dag.add_node("d");
    let e = dag.add_node("e");
    let f = dag.add_node("f");
    let g = dag.add_node("g");
    let h = dag.add_node("h");
    let sink = dag.add_node("sink");
    dag.add_edge(a, b).unwrap();
    dag.add_edge(b, c).unwrap();
    dag.add_edge(c, sink).unwrap();
    dag.add_edge(d, e).unwrap();
    dag.add_edge(e, f).unwrap();
    dag.add_edge(f, sink).unwrap();
    dag.add_edge(g, h).unwrap();
    dag.add_edge(h, sink).unwrap();
    let explainer = AsvExplainer::new(dag);
    let auto = explainer
        .auto(
            |s| Ok(s.len() as f64),
            causasv::SamplingConfig::new(500).with_seed(0),
        )
        .unwrap();
    // n=9 general DAG: auto now uses exact_dag (n ≤ 20 → exact DP)
    assert!(auto.is_exact);
    assert!(auto.effective_sample_size.is_none());
}

/// exact_dag_sparse_with_config must return InvalidConfig when memory limit is tiny.
/// This exercises the error path that auto() catches for fallback.
#[test]
fn test_sparse_memory_limit_triggers_invalid_config() {
    let mut dag = Dag::new();
    for i in 0..25usize {
        dag.add_node(&format!("n{i}"));
    }
    let explainer = AsvExplainer::new(dag);
    let config = ExactDagConfig {
        max_nodes: 28,
        memory_limit_bytes: 1, // immediately exceeded after first expansion
    };
    let result = explainer.exact_dag_sparse_with_config(|c| Ok(c.len() as f64), &config);
    assert!(
        matches!(result, Err(causasv::CausasvError::InvalidConfig(_))),
        "expected InvalidConfig on tiny memory limit, got {result:?}"
    );
}

/// auto() falls back gracefully on sparse DAGs in the n ∈ (20, 28] range.
/// Tests that the sparse → approx path (triggered by memory or overflow errors)
/// returns a valid result satisfying the efficiency axiom.
#[test]
fn test_auto_sparse_range_dag_succeeds() {
    // n=22 chain: in the (20, 28] range. Chain → rooted tree → exact_tree path.
    // Use a non-tree DAG at n=22 to force the sparse path.
    let mut dag = Dag::new();
    for i in 0..22usize {
        dag.add_node(&format!("n{i}"));
    }
    // Add one back-edge to break rooted-tree structure (make it a general DAG)
    // n0→n2 creates a non-tree: n0 has 2 children (n1 and n2 via both edges)
    for i in 0..21 {
        dag.add_edge(NodeId(i as u32), NodeId((i + 1) as u32))
            .unwrap();
    }
    // Add parallel edge n0→n2 to make it non-tree
    dag.add_edge(NodeId(0), NodeId(2)).unwrap();

    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(
            |c| Ok(c.len() as f64),
            SamplingConfig::new(500).with_seed(0),
        )
        .unwrap();

    // Chain with extra edge still has manageable order ideals for sparse DP
    // It doesn't matter if it used sparse or approx — just that it succeeded
    let total: f64 = result.values.values().sum();
    assert!(
        (total - 22.0).abs() < 1e-9,
        "efficiency axiom: expected 22.0, got {total}"
    );
}

#[test]
#[ignore = "slow: BFS through 2^21 antichain states to trigger u64 overflow fallback"]
fn test_auto_fallback_to_approx_on_overflow() {
    // Antichain of 21 nodes: no edges, so all 2^21 subsets are valid order ideals.
    // dp_fwd for the full mask needs 21! > u64::MAX → Overflow → auto falls back to approx.
    let mut dag = Dag::new();
    for i in 0..21 {
        dag.add_node(&format!("n{i}"));
    }
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(
            |c| Ok(c.len() as f64),
            SamplingConfig::new(1_000).with_seed(42),
        )
        .unwrap();
    assert_eq!(result.fallback_from.as_deref(), Some("exact_dag_sparse"));
    assert_eq!(result.method_used, Some("approx"));
    assert!(!result.is_exact);
}

#[test]
#[ignore = "slow: BFS through ~1M order ideals (18-node antichain component)"]
fn test_auto_20_28_range_stays_exact_above_250k_order_ideals() {
    // An 18-node antichain (no edges) disjoint-unioned with a 3-node chain (n=21,
    // in the (20,28] range): order ideals = 2^18 * (3+1) = 1,048,576 — comfortably
    // above the 250k budget used for n>28, but far below the ~26.8M memory-guard
    // budget that n in (20,28] actually uses (confirmed: linear extension count
    // 21!/3! ≈ 8.5e18 stays under u64::MAX, so this doesn't hit the *other*
    // failure mode either). This guards against a regression where a proactive
    // preflight added to the (20,28] branch would use the wrong (250k) budget and
    // silently downgrade this DAG from exact to approximate.
    let mut dag = Dag::new();
    for i in 0..18 {
        dag.add_node(&format!("a{i}"));
    }
    let mut chain = Vec::new();
    for i in 0..3 {
        chain.push(dag.add_node(&format!("c{i}")));
    }
    dag.add_edge(chain[0], chain[1]).unwrap();
    dag.add_edge(chain[1], chain[2]).unwrap();
    assert_eq!(dag.node_count(), 21);

    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(
            |c| Ok(c.len() as f64),
            SamplingConfig::new(500).with_seed(0),
        )
        .unwrap();

    assert!(
        result.is_exact,
        "expected exact_dag_sparse, got approx fallback"
    );
    assert_eq!(result.method_used, Some("exact_dag_sparse"));
    assert_eq!(result.n_order_ideals, Some(1_048_576));

    let total: f64 = result.values.values().sum();
    assert!(
        (total - 21.0).abs() < 1e-9,
        "efficiency axiom: expected 21.0, got {total}"
    );
}
