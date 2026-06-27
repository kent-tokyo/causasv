/// Property-based tests for ASV axioms on randomly generated DAGs.
///
/// These tests generate random valid DAGs and verify mathematical properties
/// (efficiency, linearity, dummy axiom) that must hold for any correct ASV
/// implementation. They complement the example-based tests in other test files.
use causasv::{AsvExplainer, Dag, NodeId, SamplingConfig};
use proptest::prelude::*;

/// Generate a random valid DAG with 2..=max_n nodes.
///
/// Edges (i → j) only where i < j (topological order by construction).
/// Each candidate edge is independently included with 50% probability.
fn arb_dag(max_n: usize) -> impl Strategy<Value = Dag> {
    (2usize..=max_n).prop_flat_map(|n| {
        let n_pairs = n * (n - 1) / 2;
        prop::collection::vec(prop::bool::ANY, n_pairs).prop_map(move |include| {
            let mut dag = Dag::new();
            for i in 0..n {
                dag.add_node(&format!("n{i}"));
            }
            let mut k = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if include[k] {
                        let _ = dag.add_edge(NodeId(i as u32), NodeId(j as u32));
                    }
                    k += 1;
                }
            }
            dag
        })
    })
}

/// Generate a random rooted tree with 2..=max_n nodes.
/// Each node (except node 0 = root) gets exactly one parent chosen uniformly
/// from earlier-indexed nodes, guaranteeing a valid rooted tree structure.
fn arb_rooted_tree(max_n: usize) -> impl Strategy<Value = Dag> {
    (2usize..=max_n).prop_flat_map(|n| {
        // For each node i ∈ [1, n-1], pick a parent uniformly from [0, i-1]
        prop::collection::vec(any::<u32>(), n - 1).prop_map(move |raws| {
            let mut dag = Dag::new();
            for i in 0..n {
                dag.add_node(&format!("n{i}"));
            }
            for i in 1..n {
                let parent = (raws[i - 1] as usize) % i;
                let _ = dag.add_edge(NodeId(parent as u32), NodeId(i as u32));
            }
            dag
        })
    })
}

// ── Efficiency axiom ──────────────────────────────────────────────────────────

proptest! {
    /// Σ φ_i = v(V) − v(∅) for all DAGs, all methods.
    /// Here v(S) = |S|, so Σ φ_i = n (additive, v(∅)=0, v(V)=n).
    #[test]
    fn prop_efficiency_exact(dag in arb_dag(7)) {
        let n = dag.node_count();
        let explainer = AsvExplainer::new(dag);
        let result = explainer.exact(|c| Ok(c.len() as f64)).unwrap();
        let total: f64 = result.values.values().sum();
        prop_assert!((total - n as f64).abs() < 1e-9,
            "efficiency axiom violated: expected {n}, got {total}");
    }

    #[test]
    fn prop_efficiency_exact_dag(dag in arb_dag(6)) {
        let n = dag.node_count();
        let explainer = AsvExplainer::new(dag);
        let result = explainer.exact_dag(|c| Ok(c.len() as f64)).unwrap();
        let total: f64 = result.values.values().sum();
        prop_assert!((total - n as f64).abs() < 1e-9,
            "efficiency axiom violated: expected {n}, got {total}");
    }

    #[test]
    fn prop_efficiency_exact_tree(dag in arb_rooted_tree(8)) {
        let n = dag.node_count();
        let explainer = AsvExplainer::new(dag);
        let result = explainer.exact_tree(|c| Ok(c.len() as f64)).unwrap();
        let total: f64 = result.values.values().sum();
        prop_assert!((total - n as f64).abs() < 1e-9,
            "efficiency axiom violated: expected {n}, got {total}");
    }
}

// ── exact == exact_dag consistency ───────────────────────────────────────────

proptest! {
    /// For n ≤ 7 DAGs, brute-force exact and order-ideal DP must agree.
    #[test]
    fn prop_exact_matches_exact_dag(dag in arb_dag(7)) {
        let explainer = AsvExplainer::new(dag);
        let exact = explainer.exact(|c| Ok(c.len() as f64)).unwrap();
        let dp = explainer.exact_dag(|c| Ok(c.len() as f64)).unwrap();
        for (id, &ev) in &exact.values {
            let dv = dp.values[id];
            prop_assert!((ev - dv).abs() < 1e-9,
                "exact vs exact_dag mismatch on {id:?}: exact={ev}, dp={dv}");
        }
    }
}

// ── exact_tree == exact consistency ──────────────────────────────────────────

proptest! {
    /// For rooted trees, exact_tree and brute-force exact must agree.
    #[test]
    fn prop_exact_tree_matches_exact(dag in arb_rooted_tree(7)) {
        let explainer = AsvExplainer::new(dag);
        let exact = explainer.exact(|c| Ok(c.len() as f64)).unwrap();
        let tree = explainer.exact_tree(|c| Ok(c.len() as f64)).unwrap();
        for (id, &ev) in &exact.values {
            let tv = tree.values[id];
            prop_assert!((ev - tv).abs() < 1e-9,
                "exact vs exact_tree mismatch on {id:?}: exact={ev}, tree={tv}");
        }
    }
}

// ── Dummy axiom ───────────────────────────────────────────────────────────────

proptest! {
    /// A node that never appears in a non-empty coalition's marginal contribution
    /// should have ASV = 0.
    ///
    /// Here we use v(S) = 0 for all S, so every node is dummy → ASV = 0.
    #[test]
    fn prop_dummy_zero_value_function(dag in arb_dag(7)) {
        let explainer = AsvExplainer::new(dag);
        let result = explainer.exact(|_c| Ok(0.0)).unwrap();
        for (id, &v) in &result.values {
            prop_assert!(v.abs() < 1e-12,
                "dummy axiom violated: node {id:?} has ASV {v} ≠ 0 under zero value function");
        }
    }
}

// ── Approximate efficiency ────────────────────────────────────────────────────

proptest! {
    /// Approximate method must satisfy efficiency Σφ_i ≈ v(V) − v(∅) via
    /// self-normalized IS. The efficiency axiom holds exactly for approx too
    /// because self-normalization preserves the sum-to-total constraint.
    #[test]
    fn prop_efficiency_approx(dag in arb_dag(8)) {
        let n = dag.node_count();
        let explainer = AsvExplainer::new(dag);
        let result = explainer
            .approximate(|c| Ok(c.len() as f64), SamplingConfig::new(5_000).with_seed(0))
            .unwrap();
        let total: f64 = result.values.values().sum();
        // Self-normalized IS preserves the efficiency axiom exactly
        prop_assert!((total - n as f64).abs() < 1e-9,
            "efficiency axiom violated for approx: expected {n}, got {total}");
    }
}
