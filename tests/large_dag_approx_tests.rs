/// Accuracy/correctness tests for the n > 64 large-DAG approximate ASV paths
/// (`src/coalition.rs` + `src/approx_large.rs`), fixing the gap where
/// `approximate`/`approximate_adaptive`/`approximate_batched`/
/// `approximate_adaptive_batched` and their `auto`/`auto_quality` callers
/// rejected any DAG with more than 64 nodes despite the public docs promising
/// "works for any DAG size".
///
/// Backend parity against the existing n ≤ 64 (`u64`) path lives inside
/// `src/approx_large.rs` as an internal `#[cfg(test)]` module — it needs
/// `pub(crate)` access to force the large backend onto a small DAG, which an
/// external `tests/*.rs` file (this one) cannot do.
use causasv::{AdaptiveSamplingConfig, AsvExplainer, CausasvError, Dag, NodeId, SamplingConfig};
use std::time::Instant;

fn additive(s: &[NodeId]) -> Result<f64, CausasvError> {
    Ok(s.len() as f64)
}

fn additive_batch(coalitions: &[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError> {
    Ok(coalitions.iter().map(|c| c.len() as f64).collect())
}

fn make_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    dag
}

/// Two disjoint chains -> two roots -> NOT a rooted tree, so `auto()`/
/// `auto_quality()` route through the plain "n > 63" branch instead of the
/// `exact_tree` fallback path exercised by `make_chain` (a chain has a single
/// root, so it *is* a rooted tree).
fn make_two_disjoint_chains(half_a: usize, half_b: usize) -> Dag {
    let mut dag = Dag::new();
    let a: Vec<_> = (0..half_a)
        .map(|i| dag.add_node(&format!("a{i}")))
        .collect();
    let b: Vec<_> = (0..half_b)
        .map(|i| dag.add_node(&format!("b{i}")))
        .collect();
    for i in 0..half_a - 1 {
        dag.add_edge(a[i], a[i + 1]).unwrap();
    }
    for i in 0..half_b - 1 {
        dag.add_edge(b[i], b[i + 1]).unwrap();
    }
    dag
}

// ── Phase 7A: 65-node additive chain (a chain is a rooted tree) ─────────────
//
// A chain has exactly one valid topological ordering, so the frontier
// sampler is fully deterministic (q(the one ordering) = 1, w = 1 always) —
// there is zero IS variance, so tolerances below are tight (this is a
// stronger check than a noise-tolerant one: any coalition/cache bug would
// show up as a value that is simply wrong, not masked by sampling noise).

#[test]
fn chain65_additive_approximate() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate(additive, SamplingConfig::new(500).with_seed(42))
        .unwrap();
    for (&node, &v) in &result.values {
        assert!(
            (v - 1.0).abs() < 1e-9,
            "node {node:?}: expected 1.0, got {v}"
        );
    }
}

#[test]
fn chain65_additive_adaptive() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let config = AdaptiveSamplingConfig {
        min_samples: 200,
        max_samples: 2_000,
        seed: Some(42),
        ..AdaptiveSamplingConfig::default()
    };
    let result = explainer.approximate_adaptive(additive, config).unwrap();
    for (&node, &v) in &result.values {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "node {node:?}: expected 1.0, got {v}"
        );
    }
    for &se in result.stderr.as_ref().unwrap().values() {
        assert!(
            se.is_finite() && se >= 0.0,
            "stderr must be finite and ≥ 0, got {se}"
        );
    }
    assert!(result.effective_sample_size.unwrap().is_finite());
}

#[test]
fn chain65_additive_batched() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate_batched(
            additive_batch,
            SamplingConfig::new(500).with_seed(42).with_batch_size(64),
        )
        .unwrap();
    for (&node, &v) in &result.values {
        assert!(
            (v - 1.0).abs() < 1e-9,
            "node {node:?}: expected 1.0, got {v}"
        );
    }
}

#[test]
fn chain65_additive_adaptive_batched() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let config = AdaptiveSamplingConfig {
        min_samples: 200,
        max_samples: 2_000,
        batch_size: 64,
        seed: Some(42),
        ..AdaptiveSamplingConfig::default()
    };
    let result = explainer
        .approximate_adaptive_batched(additive_batch, config)
        .unwrap();
    for (&node, &v) in &result.values {
        assert!(
            (v - 1.0).abs() < 1e-6,
            "node {node:?}: expected 1.0, got {v}"
        );
    }
}

/// `auto()` on a 65-node chain enters the `is_rooted_tree` branch, fails
/// `exact_tree` (n > 64), skips the `n ≤ 63` sparse tier, and lands on
/// `approximate` — this is the exact_tree-fallback path fixed alongside the
/// tree fallback_reason accuracy fix (a plain chain's real failure cause is
/// the bitmask limit, not a "shape exceeded budget" issue).
#[test]
fn chain65_auto_falls_back_through_exact_tree_with_accurate_reason() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(additive, SamplingConfig::new(500).with_seed(42))
        .unwrap();
    assert_eq!(result.method_used, Some("approx"));
    assert_eq!(result.fallback_from.as_deref(), Some("exact_tree"));
    let reason = result.fallback_reason.as_deref().unwrap();
    assert!(
        reason.contains("bitmask"),
        "expected the bitmask-limit reason for a plain chain, got: {reason}"
    );
    assert!(
        !reason.contains("shape"),
        "a plain chain's real cause is the bitmask limit, not a shape/budget issue: {reason}"
    );
    for (&node, &v) in &result.values {
        assert!(
            (v - 1.0).abs() < 1e-9,
            "node {node:?}: expected 1.0, got {v}"
        );
    }
}

#[test]
fn chain65_auto_quality_falls_back_with_accurate_reason() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let config = AdaptiveSamplingConfig {
        min_samples: 200,
        max_samples: 2_000,
        seed: Some(42),
        ..AdaptiveSamplingConfig::default()
    };
    let result = explainer.auto_quality(additive, config).unwrap();
    assert_eq!(result.method_used, Some("approx_adaptive"));
    assert_eq!(result.fallback_from.as_deref(), Some("exact_tree"));
    let reason = result.fallback_reason.as_deref().unwrap();
    assert!(reason.contains("bitmask"), "got: {reason}");
    for &se in result.stderr.as_ref().unwrap().values() {
        assert!(se.is_finite());
    }
}

// ── Phase 7B: 65-node non-tree DAG (two disjoint chains) ────────────────────
//
// Two independent chains have two roots, so `is_rooted_tree` is false and
// `auto()`/`auto_quality()` exercise the plain final "n > 63" branch instead
// of the exact_tree fallback path above.

#[test]
fn two_chains_65_auto_hits_plain_large_branch_no_fallback() {
    let dag = make_two_disjoint_chains(33, 32); // n = 65
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .auto(additive, SamplingConfig::new(2_000).with_seed(7))
        .unwrap();
    assert_eq!(result.method_used, Some("approx"));
    assert!(
        result.fallback_from.is_none(),
        "the plain n>63 branch never went through a fallback"
    );
    for &v in result.values.values() {
        assert!(v.is_finite());
    }
}

#[test]
fn two_chains_65_auto_quality_hits_plain_large_branch() {
    let dag = make_two_disjoint_chains(33, 32);
    let explainer = AsvExplainer::new(dag);
    let config = AdaptiveSamplingConfig {
        min_samples: 200,
        max_samples: 2_000,
        seed: Some(7),
        ..AdaptiveSamplingConfig::default()
    };
    let result = explainer.auto_quality(additive, config).unwrap();
    assert_eq!(result.method_used, Some("approx_adaptive"));
    assert!(result.fallback_from.is_none());
}

#[test]
fn two_chains_65_finite_efficiency_deterministic_and_ess_bounded() {
    let dag = make_two_disjoint_chains(33, 32);
    let n = dag.node_count();
    let explainer = AsvExplainer::new(dag);
    let n_samples = 4_000;

    let r1 = explainer
        .approximate(additive, SamplingConfig::new(n_samples).with_seed(11))
        .unwrap();
    let r2 = explainer
        .approximate(additive, SamplingConfig::new(n_samples).with_seed(11))
        .unwrap();
    let mut total = 0.0f64;
    for (&node, &v1) in &r1.values {
        assert!(v1.is_finite(), "node {node:?} not finite: {v1}");
        assert_eq!(
            v1.to_bits(),
            r2.values[&node].to_bits(),
            "same seed (serial) must be bitwise deterministic"
        );
        total += v1;
    }
    assert!(
        (total - n as f64).abs() < 1e-6,
        "efficiency axiom: expected {n}, got {total}"
    );
    let ess = r1.effective_sample_size.unwrap();
    assert!(
        ess > 0.0 && ess <= n_samples as f64 + 1e-9,
        "ESS {ess} out of (0, n_samples] bounds"
    );

    let adaptive_cfg = AdaptiveSamplingConfig {
        min_samples: 500,
        max_samples: 5_000,
        seed: Some(11),
        ..AdaptiveSamplingConfig::default()
    };
    let ar = explainer
        .approximate_adaptive(additive, adaptive_cfg)
        .unwrap();
    for &se in ar.stderr.as_ref().unwrap().values() {
        assert!(se.is_finite() && se >= 0.0);
    }
    let a_ess = ar.effective_sample_size.unwrap();
    assert!(a_ess > 0.0 && a_ess <= ar.n_samples as f64 + 1e-9);
}

// ── Phase 7C: 128/256-node smoke test ────────────────────────────────────────
//
// No panic, no shift overflow, no InvalidConfig, bounded time/memory. Kept
// light enough (small n_samples) to run in default CI, not `#[ignore]`d.

#[test]
fn smoke_128_and_256_node_chain_no_panic_bounded_time() {
    for &n in &[128usize, 256usize] {
        let dag = make_chain(n);
        let explainer = AsvExplainer::new(dag);

        let start = Instant::now();
        let result = explainer
            .approximate(additive, SamplingConfig::new(500).with_seed(1))
            .unwrap();
        assert!(
            start.elapsed().as_secs() < 30,
            "n={n} approximate took {:?}, too slow for a CI smoke test",
            start.elapsed()
        );
        for &v in result.values.values() {
            assert!(v.is_finite(), "n={n}: non-finite value {v}");
        }

        let adaptive = explainer
            .approximate_adaptive(
                additive,
                AdaptiveSamplingConfig {
                    min_samples: 100,
                    max_samples: 500,
                    seed: Some(1),
                    ..AdaptiveSamplingConfig::default()
                },
            )
            .unwrap();
        for &v in adaptive.values.values() {
            assert!(v.is_finite(), "n={n}: adaptive non-finite value {v}");
        }
        for &se in adaptive.stderr.as_ref().unwrap().values() {
            assert!(se.is_finite(), "n={n}: adaptive non-finite stderr {se}");
        }
        assert!(adaptive.effective_sample_size.unwrap().is_finite());

        let batched = explainer
            .approximate_batched(
                additive_batch,
                SamplingConfig::new(500).with_seed(1).with_batch_size(64),
            )
            .unwrap();
        for &v in batched.values.values() {
            assert!(v.is_finite(), "n={n}: batched non-finite value {v}");
        }
    }
}

// ── Phase 7E: parallel determinism ───────────────────────────────────────────

#[test]
fn two_chains_65_seeded_parallel_deterministic() {
    let dag = make_two_disjoint_chains(33, 32);
    let explainer = AsvExplainer::new(dag);
    let cfg = || {
        SamplingConfig::new(4_000)
            .with_seed(99)
            .with_parallel(true)
            .with_num_threads(4)
    };
    let r1 = explainer.approximate(additive, cfg()).unwrap();
    let r2 = explainer.approximate(additive, cfg()).unwrap();
    for (&node, &v1) in &r1.values {
        assert_eq!(
            v1.to_bits(),
            r2.values[&node].to_bits(),
            "same seed + thread count must be bitwise deterministic"
        );
    }
}

#[test]
fn two_chains_65_unseeded_parallel_finite() {
    let dag = make_two_disjoint_chains(33, 32);
    let explainer = AsvExplainer::new(dag);
    let result = explainer
        .approximate(additive, SamplingConfig::new(2_000).with_parallel(true))
        .unwrap();
    for &v in result.values.values() {
        assert!(v.is_finite());
    }
}

// ── Phase 7F: batched value_fn_batch validation on n > 64 ───────────────────
//
// No silent zip-truncation: a length mismatch is an explicit
// `CausasvError::ValueFunctionError`, not a panic or a quietly-dropped tail.
// (The pre-existing small-path batched functions in `approx.rs` still zip
// silently and can panic on a short return — that is a separate, pre-existing
// issue out of scope for this fix; see the PR description.)

#[test]
fn batched_value_fn_short_return_is_explicit_error() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let short = |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
        Ok(vec![0.0; coalitions.len().saturating_sub(1)])
    };
    let err = explainer
        .approximate_batched(
            short,
            SamplingConfig::new(200).with_seed(1).with_batch_size(32),
        )
        .unwrap_err();
    match err {
        CausasvError::ValueFunctionError(msg) => {
            assert!(msg.contains("returned"), "unexpected message: {msg}")
        }
        other => panic!("expected ValueFunctionError, got {other:?}"),
    }
}

#[test]
fn batched_value_fn_long_return_is_explicit_error() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let long = |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
        let mut v = vec![0.0; coalitions.len()];
        v.push(0.0);
        Ok(v)
    };
    let err = explainer
        .approximate_batched(
            long,
            SamplingConfig::new(200).with_seed(1).with_batch_size(32),
        )
        .unwrap_err();
    assert!(matches!(err, CausasvError::ValueFunctionError(_)));
}

#[test]
fn adaptive_batched_value_fn_short_return_is_explicit_error() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let short = |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
        Ok(vec![0.0; coalitions.len().saturating_sub(1)])
    };
    let config = AdaptiveSamplingConfig {
        min_samples: 50,
        max_samples: 200,
        batch_size: 32,
        seed: Some(1),
        ..AdaptiveSamplingConfig::default()
    };
    let err = explainer
        .approximate_adaptive_batched(short, config)
        .unwrap_err();
    assert!(matches!(err, CausasvError::ValueFunctionError(_)));
}

#[test]
fn batched_value_fn_err_propagates() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let erroring = |_: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
        Err(CausasvError::ValueFunctionError("boom".to_string()))
    };
    let err = explainer
        .approximate_batched(
            erroring,
            SamplingConfig::new(100).with_seed(1).with_batch_size(16),
        )
        .unwrap_err();
    match err {
        CausasvError::ValueFunctionError(msg) => assert_eq!(msg, "boom"),
        other => panic!("expected ValueFunctionError(\"boom\"), got {other:?}"),
    }
}

/// A `value_fn_batch` returning NaN must propagate without panicking — no
/// silent replacement, no crash. `v(∅) = NaN` (triggered only on the empty
/// coalition) deterministically makes exactly the first-inserted node's delta
/// `finite - NaN = NaN` on every sample of a chain (single topological
/// order), while every other node's delta is a difference between two
/// non-empty coalitions and stays finite — a precise, non-tautological check
/// that NaN reaches exactly where it's supposed to and nowhere else.
#[test]
fn batched_value_fn_nan_propagates_without_panicking() {
    let dag = make_chain(65);
    let explainer = AsvExplainer::new(dag);
    let nan_on_empty = |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
        Ok(coalitions
            .iter()
            .map(|c| {
                if c.is_empty() {
                    f64::NAN
                } else {
                    c.len() as f64
                }
            })
            .collect())
    };
    let result = explainer
        .approximate_batched(
            nan_on_empty,
            SamplingConfig::new(300).with_seed(1).with_batch_size(32),
        )
        .unwrap();
    assert!(
        result.values[&NodeId(0)].is_nan(),
        "expected NodeId(0) (first in the chain's only ordering) to be NaN"
    );
    for i in 1..65u32 {
        let v = result.values[&NodeId(i)];
        assert!(v.is_finite(), "NodeId({i}) unexpectedly non-finite: {v}");
    }
}
