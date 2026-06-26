use causasv::{AsvExplainer, Dag, SamplingConfig};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn make_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let nodes: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(nodes[i], nodes[i + 1]).unwrap();
    }
    dag
}

fn make_balanced_tree(depth: usize) -> Dag {
    let mut dag = Dag::new();
    let root = dag.add_node("root");
    let mut current_level = vec![root];
    for d in 0..depth {
        let mut next_level = Vec::new();
        for &parent in &current_level {
            let l = dag.add_node(&format!("d{d}l{}", parent.0));
            let r = dag.add_node(&format!("d{d}r{}", parent.0));
            dag.add_edge(parent, l).unwrap();
            dag.add_edge(parent, r).unwrap();
            next_level.push(l);
            next_level.push(r);
        }
        current_level = next_level;
    }
    dag
}

fn make_caterpillar(chain_len: usize) -> Dag {
    // chain_len main nodes + one leaf per main node
    let mut dag = Dag::new();
    let ns: Vec<_> = (0..chain_len)
        .map(|i| dag.add_node(&format!("n{i}")))
        .collect();
    let ls: Vec<_> = (0..chain_len)
        .map(|i| dag.add_node(&format!("l{i}")))
        .collect();
    for i in 0..chain_len - 1 {
        dag.add_edge(ns[i], ns[i + 1]).unwrap();
    }
    for i in 0..chain_len {
        dag.add_edge(ns[i], ls[i]).unwrap();
    }
    dag
}

// ── exact (brute-force) ──────────────────────────────────────────────────────

fn bench_exact_chain_7(c: &mut Criterion) {
    let dag = make_chain(7);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_chain_7", |b| {
        b.iter(|| explainer.exact(|s| Ok(black_box(s.len() as f64))).unwrap());
    });
}

fn bench_exact_balanced_tree_7(c: &mut Criterion) {
    // n=7 balanced binary tree: L(T) = 80 linear extensions
    let dag = make_balanced_tree(2);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_balanced_tree_7_bruteforce", |b| {
        b.iter(|| explainer.exact(|s| Ok(black_box(s.len() as f64))).unwrap());
    });
}

// ── exact_tree (order-ideal DP) ──────────────────────────────────────────────

fn bench_tree_dp_balanced_7(c: &mut Criterion) {
    // same n=7 tree via order-ideal DP
    let dag = make_balanced_tree(2);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_tree_balanced_7_dp", |b| {
        b.iter(|| {
            explainer
                .exact_tree(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

fn bench_tree_dp_balanced_15(c: &mut Criterion) {
    // n=15 balanced binary tree: L(T) ≈ 22M — brute-force is slow, DP is fast
    let dag = make_balanced_tree(3);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_tree_balanced_15_dp", |b| {
        b.iter(|| {
            explainer
                .exact_tree(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

fn bench_tree_dp_caterpillar_10(c: &mut Criterion) {
    // n=10 caterpillar: L(T) = 945, DP has ~6x fewer pre-sets
    let dag = make_caterpillar(5);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_tree_caterpillar_10_dp", |b| {
        b.iter(|| {
            explainer
                .exact_tree(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

// ── approximate ─────────────────────────────────────────────────────────────

fn bench_approx_chain_10(c: &mut Criterion) {
    let dag = make_chain(10);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("approx_chain_10_1k", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(1_000).with_seed(42),
                )
                .unwrap()
        });
    });
}

fn bench_approx_tree(c: &mut Criterion) {
    let dag = make_balanced_tree(3); // 15 nodes
    let explainer = AsvExplainer::new(dag);
    c.bench_function("approx_balanced_tree_15_1k", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(1_000).with_seed(42),
                )
                .unwrap()
        });
    });
}

criterion_group!(
    benches,
    bench_exact_chain_7,
    bench_exact_balanced_tree_7,
    bench_tree_dp_balanced_7,
    bench_tree_dp_balanced_15,
    bench_tree_dp_caterpillar_10,
    bench_approx_chain_10,
    bench_approx_tree,
);
criterion_main!(benches);
