/// Golden corpus: approximate ASV accuracy vs exact on small, known DAGs.
///
/// For additive v(S) = |S|, the exact ASV is 1.0 per node on any DAG (marginal contribution
/// is always 1 regardless of prefix). Tests verify approximate matches this across DAG shapes
/// with different IS-weight distributions (chain vs wide tree vs diamond).
use causasv::{AdaptiveSamplingConfig, AsvExplainer, Dag, NodeId, SamplingConfig};

fn additive(s: &[NodeId]) -> Result<f64, causasv::CausasvError> {
    Ok(s.len() as f64)
}

fn weighted(s: &[NodeId]) -> Result<f64, causasv::CausasvError> {
    Ok(s.iter().map(|n| (n.0 + 1) as f64).sum())
}

fn make_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    dag
}

fn make_fork(k: usize) -> Dag {
    // root → k children
    let mut dag = Dag::new();
    let root = dag.add_node("root");
    for i in 0..k {
        let child = dag.add_node(&format!("c{i}"));
        dag.add_edge(root, child).unwrap();
    }
    dag
}

fn make_collider(k: usize) -> Dag {
    // k sources → one sink
    let mut dag = Dag::new();
    let sink = dag.add_node("sink");
    for i in 0..k {
        let src = dag.add_node(&format!("s{i}"));
        dag.add_edge(src, sink).unwrap();
    }
    dag
}

fn make_diamond() -> Dag {
    // src → m0, m1 → snk
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

fn make_two_parallel_chains(half: usize) -> Dag {
    let mut dag = Dag::new();
    let a: Vec<_> = (0..half).map(|i| dag.add_node(&format!("a{i}"))).collect();
    let b: Vec<_> = (0..half).map(|i| dag.add_node(&format!("b{i}"))).collect();
    for i in 0..half - 1 {
        dag.add_edge(a[i], a[i + 1]).unwrap();
        dag.add_edge(b[i], b[i + 1]).unwrap();
    }
    dag
}

fn make_balanced_tree(depth: usize) -> Dag {
    let mut dag = Dag::new();
    let root = dag.add_node("root");
    let mut level = vec![root];
    for d in 0..depth {
        let mut next = Vec::new();
        for &p in &level {
            for s in ["l", "r"] {
                let c = dag.add_node(&format!("d{d}{s}{}", p.0));
                dag.add_edge(p, c).unwrap();
                next.push(c);
            }
        }
        level = next;
    }
    dag
}

/// For additive v(S)=|S|, phi_i = 1.0 exactly on any DAG.
/// Checks that approx is within `tol` and ESS ratio is above 0.02.
fn check_additive_approx(dag: Dag, n_samples: usize, tol: f64) {
    let n = dag.node_count();
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate(additive, SamplingConfig::new(n_samples).with_seed(42))
        .unwrap();

    let mut max_err = 0.0f64;
    for &v in result.values.values() {
        max_err = max_err.max((v - 1.0).abs());
    }
    assert!(
        max_err < tol,
        "additive approx: max |phi - 1.0| = {max_err:.4} ≥ tol {tol} (n={n})"
    );

    if let Some(ess) = result.effective_sample_size {
        let ratio = ess / n_samples as f64;
        assert!(ratio > 0.02, "ESS ratio {ratio:.3} too low (n={n})");
    }
}

/// Compares approximate vs exact_dag for a given value function.
fn check_approx_vs_exact<F>(dag: Dag, value_fn: F, n_samples: usize, tol: f64)
where
    F: Fn(&[NodeId]) -> Result<f64, causasv::CausasvError> + Copy + Send + Sync,
{
    let explainer = AsvExplainer::new(dag);
    let exact = explainer.exact_dag(value_fn).unwrap();
    let approx = explainer
        .approximate(value_fn, SamplingConfig::new(n_samples).with_seed(42))
        .unwrap();

    let mut max_err = 0.0f64;
    for (&node, &phi_e) in &exact.values {
        let phi_a = *approx
            .values
            .get(&node)
            .expect("node missing from approx result");
        max_err = max_err.max((phi_a - phi_e).abs());
    }
    assert!(
        max_err < tol,
        "approx vs exact: max error = {max_err:.4} ≥ tol {tol}"
    );
}

// ── additive known-answer tests (phi_i = 1.0) ────────────────────────────────

#[test]
fn test_additive_chain5() {
    // Chain: single ordering, ESS = n_samples → very tight
    check_additive_approx(make_chain(5), 2_000, 0.05);
}

#[test]
fn test_additive_fork5() {
    check_additive_approx(make_fork(4), 2_000, 0.10);
}

#[test]
fn test_additive_collider5() {
    check_additive_approx(make_collider(4), 2_000, 0.10);
}

#[test]
fn test_additive_diamond() {
    check_additive_approx(make_diamond(), 2_000, 0.10);
}

#[test]
fn test_additive_two_parallel_chains6() {
    check_additive_approx(make_two_parallel_chains(3), 2_000, 0.10);
}

#[test]
fn test_additive_balanced_tree7() {
    check_additive_approx(make_balanced_tree(2), 2_000, 0.10);
}

// ── exact vs approx comparison (non-trivial value function) ──────────────────

#[test]
fn test_weighted_approx_vs_exact_chain5() {
    // weighted v: phi_i = (i+1), varies by node — tests that rank order is preserved
    check_approx_vs_exact(make_chain(5), weighted, 4_000, 0.15);
}

#[test]
fn test_weighted_approx_vs_exact_diamond() {
    check_approx_vs_exact(make_diamond(), weighted, 4_000, 0.15);
}

#[test]
fn test_weighted_approx_vs_exact_balanced_tree7() {
    check_approx_vs_exact(make_balanced_tree(2), weighted, 4_000, 0.20);
}

// ── confidence interval coverage (mirrors py/tests/test_ci_coverage.py) ──────

/// z-score for a 95% two-sided normal CI (avoids a stats crate dependency for one constant).
const Z_95: f64 = 1.959963984540054;

/// For additive v(S)=|S|, the true ASV is exactly 1.0 per node on any DAG. Runs
/// approximate_adaptive across many seeds and checks that a 95% CI built from its
/// stderr (value ± Z_95 * stderr) covers 1.0 at close to the nominal rate. The
/// 0.75 floor (not 0.95) mirrors the Python-side quick coverage test's threshold:
/// generous enough to avoid flakiness while still catching a badly miscalibrated
/// stderr (e.g. an off-by-constant-factor bug would fail this easily).
#[test]
fn test_adaptive_ci_coverage_additive() {
    let n_seeds = 30;
    let mut covered = 0;
    for seed in 0..n_seeds {
        let explainer = AsvExplainer::new(make_collider(10));
        let config = AdaptiveSamplingConfig {
            min_samples: 200,
            max_samples: 2_000,
            seed: Some(seed as u64),
            ..AdaptiveSamplingConfig::default()
        };
        let result = explainer.approximate_adaptive(additive, config).unwrap();
        let stderr = result
            .stderr
            .as_ref()
            .expect("adaptive result must have stderr");
        let all_covered = result.values.iter().all(|(node, &v)| {
            let se = stderr[node];
            (v - Z_95 * se) <= 1.0 && 1.0 <= (v + Z_95 * se)
        });
        if all_covered {
            covered += 1;
        }
    }
    let rate = covered as f64 / n_seeds as f64;
    assert!(
        rate >= 0.75,
        "CI coverage {rate:.2} below 0.75 threshold ({covered}/{n_seeds} seeds fully covered)"
    );
}
