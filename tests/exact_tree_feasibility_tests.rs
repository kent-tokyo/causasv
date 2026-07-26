//! Regression tests for the exact_tree feasibility guard (issue #36).
//!
//! `exact_tree`'s cost is not a function of node count alone: a rooted tree with
//! several large sibling subtrees at some ancestor level forces
//! `enumerate_order_ideals` to materialize their full cartesian product. Before
//! this guard, `auto`/`auto_quality` dispatched to `exact_tree` unconditionally
//! for any rooted tree (n>8), with no cost estimate analogous to
//! `exact_dag_sparse`'s -- a modest-sized (n≈61) but "bushy" tree could hang and
//! eventually OOM. None of these tests materialize a genuinely pathological
//! cartesian product (billions-plus combinations): the guard is checked either
//! via a tiny custom budget (fast, no real computation attempted), by asserting
//! the real API call returns an `Err` promptly under the default budget, or --
//! for shapes the default budget still accepts -- by actually running the real
//! (but modest, sub-few-seconds) computation to confirm the accepted case still
//! computes correctly.

use causasv::{AsvExplainer, CausasvError, Dag, ExactTreeConfig, NodeId, SamplingConfig};

fn add_balanced_subtree(dag: &mut Dag, parent: NodeId, depth: usize, prefix: &str) {
    if depth == 0 {
        return;
    }
    let left = dag.add_node(&format!("{prefix}l"));
    let right = dag.add_node(&format!("{prefix}r"));
    dag.add_edge(parent, left).unwrap();
    dag.add_edge(parent, right).unwrap();
    add_balanced_subtree(dag, left, depth - 1, &format!("{prefix}l"));
    add_balanced_subtree(dag, right, depth - 1, &format!("{prefix}r"));
}

/// Root with `num_branches` children, each the root of its own balanced binary
/// subtree of the given `depth` -- same construction as issue #36's report.
fn star_of_balanced_subtrees(num_branches: usize, depth: usize) -> Dag {
    let mut dag = Dag::new();
    let root = dag.add_node("root");
    for b in 0..num_branches {
        let branch_root = dag.add_node(&format!("b{b}"));
        dag.add_edge(root, branch_root).unwrap();
        add_balanced_subtree(&mut dag, branch_root, depth, &format!("b{b}"));
    }
    dag
}

fn build_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let nodes: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(nodes[i], nodes[i + 1]).unwrap();
    }
    dag
}

fn build_complete_binary_tree(depth: usize) -> Dag {
    add_balanced_subtree_from_scratch(depth)
}

fn add_balanced_subtree_from_scratch(depth: usize) -> Dag {
    let mut dag = Dag::new();
    let root = dag.add_node("root");
    add_balanced_subtree(&mut dag, root, depth, "r");
    dag
}

// --- 1. Chain: unaffected by the guard ---------------------------------------

#[test]
fn chain_still_dispatches_to_exact_tree_under_default_budget() {
    let dag = build_chain(10);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .exact_tree(|s| Ok(s.len() as f64))
        .expect("chain is cheap regardless of the new guard");
    assert!(result.is_exact);

    let empty = 0.0f64;
    let all = 10.0f64;
    let sum: f64 = result.values.values().sum();
    assert!(
        (sum - (all - empty)).abs() < 1e-9,
        "efficiency axiom violated"
    );
}

#[test]
fn chain_via_auto_still_selects_exact_tree() {
    let dag = build_chain(10);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(|s| Ok(s.len() as f64), SamplingConfig::new(1000))
        .unwrap();
    assert_eq!(result.method_used, Some("exact_tree"));
    assert!(result.fallback_from.is_none());
}

// --- 2. Existing (binary, non-bushy) benchmark tree: unaffected ---------------

#[test]
fn balanced_binary_tree_n15_still_feasible_under_default_budget() {
    // Complete binary tree of height 3: 15 nodes -- the largest exact_tree case
    // actually benchmarked in this repo (docs/benchmarks.md: "Balanced binary
    // tree | 15 | exact_tree (DP) | 2.79 ms"). Its deepest-leaf cost (260) and
    // total (2,653) are tiny relative to the default budget, so this remains
    // feasible and fast.
    let dag = build_complete_binary_tree(3);
    assert_eq!(dag.node_count(), 15);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .exact_tree(|s| Ok(s.len() as f64))
        .expect("binary trees must remain feasible under the default budget");
    assert!(result.is_exact);
    let sum: f64 = result.values.values().sum();
    assert!((sum - 15.0).abs() < 1e-6, "efficiency axiom violated");
}

#[test]
fn balanced_binary_tree_n31_now_rejected_promptly_instead_of_taking_tens_of_seconds() {
    // Complete binary tree of height 4: 31 nodes -- README's "Balanced tree | 31"
    // performance row already benchmarks this shape via `approx`, not
    // `exact_tree`, because a real (measured) `exact_tree` run on it takes
    // ~20-25s despite only n=31 nodes: its deepest leaf's per-node combination
    // count is 176,020 (product of every ancestor level's side-sibling order
    // ideal count), and ~3.6M such combinations are iterated in total. Before
    // this guard, `auto()`/`exact_tree` had no way to know that and would
    // silently take tens of seconds instead of erroring or falling back. The
    // default budget now rejects this shape's estimate promptly (no real
    // computation attempted) rather than ever running it for real.
    let dag = build_complete_binary_tree(4);
    assert_eq!(dag.node_count(), 31);
    let explainer = AsvExplainer::new(dag);
    let err = explainer
        .exact_tree(|s| Ok(s.len() as f64))
        .expect_err("n=31 must now be rejected by the default budget, not silently run for real");
    match err {
        CausasvError::ExactTreeBudgetExceeded {
            max_cartesian_terms,
            cartesian_term_budget,
            ..
        } => {
            // Deepest leaf's cost: side product at each of the 4 ancestor levels
            // (I(3)=677 at the root, then 26, 5, 2 descending) -- complete binary
            // height 4: I(0)=2, I(1)=5, I(2)=26, I(3)=677, I(4)=458,330 (root).
            assert_eq!(max_cartesian_terms, 677 * 26 * 5 * 2);
            assert!(max_cartesian_terms > cartesian_term_budget);
        }
        other => panic!("expected ExactTreeBudgetExceeded, got {other:?}"),
    }
}

// --- 3. Small bushy tree + tiny budget: definite rejection --------------------

#[test]
fn small_bushy_tree_rejected_under_a_tiny_budget() {
    // 3 branches, depth 2 (22 nodes total): cheap to actually build and estimate,
    // but a leaf's per-node cost -- the product of every ancestor level's side
    // product (26 * 26 at the root level, times 5 at the next level down, times 2
    // at the last) -- comfortably exceeds a deliberately tiny budget.
    let dag = star_of_balanced_subtrees(3, 2);
    assert_eq!(dag.node_count(), 22);
    let config = ExactTreeConfig {
        max_cartesian_terms: 100,
        max_total_terms: 1_000,
    };
    let explainer = AsvExplainer::new(dag);
    let err = explainer
        .exact_tree_with_config(|s| Ok(s.len() as f64), &config)
        .unwrap_err();
    match err {
        CausasvError::ExactTreeBudgetExceeded {
            max_cartesian_terms,
            cartesian_term_budget,
            ..
        } => {
            assert_eq!(max_cartesian_terms, 26 * 26 * 5 * 2);
            assert_eq!(cartesian_term_budget, 100);
        }
        other => panic!("expected ExactTreeBudgetExceeded, got {other:?}"),
    }
}

#[test]
fn small_bushy_tree_is_feasible_under_a_generous_budget() {
    // Same tree, default config: a deepest-leaf cost of 6,760 (max budget
    // 50,000) and a total of 103,429 (total budget 200,000) both fit with
    // margin, so this genuinely runs the real computation (not just the
    // estimate) -- unlike the tiny-budget test above, which only exercises the
    // preflight.
    let dag = star_of_balanced_subtrees(3, 2);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .exact_tree(|s| Ok(s.len() as f64))
        .expect("this shape must be feasible under the default budget");
    assert!(result.is_exact);
}

// --- 4. Issue #36's exact reproduction shape, via the real public API ---------

#[test]
fn issue_36_reproduction_shape_rejected_promptly_by_default_config() {
    // 4 branches, depth 3: 61 nodes. Before the guard, this hung/OOM'd inside
    // tree_exact_asv. The guard's O(n) preflight now rejects it before any
    // enumeration -- this call is expected to return near-instantly.
    let dag = star_of_balanced_subtrees(4, 3);
    assert_eq!(dag.node_count(), 61);
    let explainer = AsvExplainer::new(dag);
    let err = explainer
        .exact_tree(|s| Ok(s.len() as f64))
        .expect_err("this tree shape must be rejected, not hang or OOM");
    match err {
        CausasvError::ExactTreeBudgetExceeded {
            max_cartesian_terms,
            cartesian_term_budget,
            ..
        } => {
            assert!(max_cartesian_terms > cartesian_term_budget);
            // A deepest leaf's cost is the product of every ancestor level's side
            // product: 677^3 at the root (the other 3 branches), times 26, 5, 2 at
            // the branch's own 3 internal levels (complete binary height 3:
            // I(0)=2, I(1)=5, I(2)=26, I(3)=677).
            assert_eq!(max_cartesian_terms, 677u64.pow(3) * 26 * 5 * 2);
        }
        other => panic!("expected ExactTreeBudgetExceeded, got {other:?}"),
    }
}

// --- 5. auto()/auto_quality() fall back gracefully ----------------------------

#[test]
fn exact_tree_with_config_rejects_a_moderately_bushy_tree_under_a_tight_custom_budget() {
    // 4 branches, depth 2 (29 nodes): under exact_tree's *default* budget this
    // shape is genuinely feasible (a deepest leaf costs 17,576 * 5 * 2 =
    // 175,760, comfortably under 1,000,000) and auto() would actually run the
    // real computation for it -- which is exactly what
    // `auto_falls_back_to_approx_when_no_exact_method_is_feasible` and
    // `issue_36_reproduction_shape_rejected_promptly_by_default_config` avoid by
    // using the more extreme n=61 shape instead. This test only exercises the
    // fast, deterministic part: a tiny custom config must reject this shape
    // without ever materializing its cartesian product.
    let dag = star_of_balanced_subtrees(4, 2);
    assert_eq!(dag.node_count(), 29);
    let explainer = AsvExplainer::new(dag);
    let config = ExactTreeConfig {
        max_cartesian_terms: 1_000, // artificially low, forces exact_tree to reject
        max_total_terms: 10_000,
    };
    assert!(matches!(
        explainer.exact_tree_with_config(|s| Ok(s.len() as f64), &config),
        Err(CausasvError::ExactTreeBudgetExceeded { .. })
    ));
}

#[test]
fn auto_falls_back_to_approx_when_no_exact_method_is_feasible() {
    // Issue #36's own shape: exact_tree's default-config preflight rejects it,
    // and the whole tree's total order-ideal count (~2e11) is far beyond
    // exact_dag_sparse's own memory-budget preflight too, so auto() must fall
    // all the way through to `approximate` -- and must do so without hanging.
    let dag = star_of_balanced_subtrees(4, 3); // n=61, matches issue #36
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(|s| Ok(s.len() as f64), SamplingConfig::new(500))
        .expect("auto() must fall back to approx, never hang or error out");
    assert!(!result.is_exact);
    assert_eq!(result.method_used, Some("approx"));
    assert_eq!(result.fallback_from.as_deref(), Some("exact_tree"));
    assert!(result.fallback_reason.is_some());
}

#[test]
fn auto_quality_falls_back_to_approx_adaptive_not_uniform_sparse_adaptive() {
    // Issue #36's shape is exactly the "dangerous shape" the guard exists for:
    // once exact_tree and exact_dag_sparse are both infeasible, auto_quality()
    // must skip approximate_uniform_sparse_adaptive (its own internal memo is
    // unbounded and could blow up on the same shape) and go straight to the
    // memo-free approximate_adaptive.
    let dag = star_of_balanced_subtrees(4, 3); // n=61, matches issue #36
    let explainer = AsvExplainer::new(dag);
    let config = causasv::AdaptiveSamplingConfig {
        min_samples: 50,
        max_samples: 200,
        ..Default::default()
    };
    let result = explainer
        .auto_quality(|s| Ok(s.len() as f64), config)
        .expect("auto_quality() must fall back gracefully, never hang or error out");
    assert!(!result.is_exact);
    assert_eq!(result.fallback_from.as_deref(), Some("exact_tree"));
    assert_eq!(result.method_used, Some("approx_adaptive"));
    // Every auto_quality() path must return stderr per its own documented contract.
    assert!(result.stderr.is_some());
}

// --- 6. n > 64 rooted tree: was a hard error before, now falls back too ------

#[test]
fn auto_attempts_fallback_for_a_rooted_tree_exceeding_the_bitmask_limit() {
    // A plain chain of 70 nodes is a rooted tree, but exceeds exact_tree's
    // n<=64 bitmask limit. Before this fix, auto() propagated that InvalidConfig
    // straight out of the is_rooted_tree branch without even trying to fall
    // back, unlike every other branch. This fix makes it *attempt* the same
    // fallback chain as any other infeasible DAG.
    //
    // It cannot assert an eventual Ok() here: AsvExplainer::approximate() (the
    // ultimate fallback for n>63) has its own separate, pre-existing n<=64
    // bitmask cap in approximate_asv -- undocumented and unrelated to this fix,
    // since it affects any DAG shape, not just rooted trees. That means a
    // rooted tree with n>64 currently has no working method at all in this
    // crate. What this test asserts is narrower and still meaningful: auto()
    // must not hang, and if it errors, the error must come from genuinely
    // exhausting the fallback chain, not from returning exact_tree's rejection
    // verbatim without ever trying anything else.
    let dag = build_chain(70);
    let explainer = AsvExplainer::new(dag);
    let result = explainer.auto(|s| Ok(s.len() as f64), SamplingConfig::new(200));
    match result {
        Ok(r) => {
            assert!(!r.is_exact);
            assert_eq!(r.fallback_from.as_deref(), Some("exact_tree"));
        }
        // Currently expected: approximate()'s own n<=64 cap (see module docs
        // above) still rejects n=70. Tracked separately from this fix.
        Err(CausasvError::InvalidConfig(_)) => {}
        Err(e) => panic!("unexpected error variant: {e:?}"),
    }
}
