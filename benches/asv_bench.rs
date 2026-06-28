use causasv::{AdaptiveSamplingConfig, AsvExplainer, Dag, SamplingConfig};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

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

// ── exact_dag (order-ideal DP for general DAGs) ──────────────────────────────

fn make_two_parallel_chains(half: usize) -> Dag {
    // Two independent chains of length `half` sharing nothing — a general DAG.
    let mut dag = Dag::new();
    let a: Vec<_> = (0..half).map(|i| dag.add_node(&format!("a{i}"))).collect();
    let b: Vec<_> = (0..half).map(|i| dag.add_node(&format!("b{i}"))).collect();
    for i in 0..half - 1 {
        dag.add_edge(a[i], a[i + 1]).unwrap();
        dag.add_edge(b[i], b[i + 1]).unwrap();
    }
    dag
}

fn make_diamond_dag(width: usize) -> Dag {
    // source → width middle nodes → sink
    let mut dag = Dag::new();
    let src = dag.add_node("src");
    let mids: Vec<_> = (0..width).map(|i| dag.add_node(&format!("m{i}"))).collect();
    let snk = dag.add_node("snk");
    for &m in &mids {
        dag.add_edge(src, m).unwrap();
        dag.add_edge(m, snk).unwrap();
    }
    dag
}

fn bench_exact_dag_chain_10(c: &mut Criterion) {
    let dag = make_chain(10);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_dag_chain_10", |b| {
        b.iter(|| {
            explainer
                .exact_dag(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

fn bench_exact_dag_two_chains_12(c: &mut Criterion) {
    let dag = make_two_parallel_chains(6); // n = 12
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_dag_two_parallel_chains_12", |b| {
        b.iter(|| {
            explainer
                .exact_dag(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

fn bench_exact_dag_diamond_10(c: &mut Criterion) {
    let dag = make_diamond_dag(8); // n = 10 (src + 8 + snk)
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_dag_diamond_10", |b| {
        b.iter(|| {
            explainer
                .exact_dag(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

fn bench_exact_dag_chain_16(c: &mut Criterion) {
    let dag = make_chain(16);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_dag_chain_16", |b| {
        b.iter(|| {
            explainer
                .exact_dag(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

// ── exact_dag_sparse ─────────────────────────────────────────────────────────

fn bench_exact_dag_sparse_chain_24(c: &mut Criterion) {
    // Chain n=24: beyond exact_dag limit (n>20). Sparse visits only 25 order ideals
    // vs 2^24=16M for dense. Shows the sparsity benefit on a maximally-sparse DAG.
    let dag = make_chain(24);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_dag_sparse_chain_24", |b| {
        b.iter(|| {
            explainer
                .exact_dag_sparse(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
}

fn bench_exact_dag_vs_sparse_two_chains_20(c: &mut Criterion) {
    // Two parallel chains n=20: dense visits 2^20=1M states, sparse visits (10+1)^2=121.
    let dag = make_two_parallel_chains(10); // n = 20
    let explainer = AsvExplainer::new(dag);
    let mut group = c.benchmark_group("exact_dag_vs_sparse_two_chains_20");
    group.bench_function("dense", |b| {
        b.iter(|| {
            explainer
                .exact_dag(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
    group.bench_function("sparse", |b| {
        b.iter(|| {
            explainer
                .exact_dag_sparse(|s| Ok(black_box(s.len() as f64)))
                .unwrap()
        });
    });
    group.finish();
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

// ── approx: normal vs batched ────────────────────────────────────────────────

fn bench_approx_vs_batched_chain_10(c: &mut Criterion) {
    // Batched deduplicates coalitions and evaluates each mask at most once.
    // In pure Rust the gain is moderate; Python users see much larger speedup
    // because batch_size controls how often the GIL is reacquired.
    let dag = make_chain(10);
    let explainer = AsvExplainer::new(dag);
    let mut group = c.benchmark_group("approx_vs_batched_chain_10_1k");
    group.bench_function("normal", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(1_000).with_seed(42),
                )
                .unwrap()
        });
    });
    group.bench_function("batched_b256", |b| {
        b.iter(|| {
            explainer
                .approximate_batched(
                    |coalitions| {
                        Ok(coalitions
                            .iter()
                            .map(|c| black_box(c.len() as f64))
                            .collect())
                    },
                    SamplingConfig::new(1_000)
                        .with_seed(42)
                        .with_batch_size(256),
                )
                .unwrap()
        });
    });
    group.finish();
}

// ── approx: seeded parallel ──────────────────────────────────────────────────

fn bench_approx_parallel_chain_20(c: &mut Criterion) {
    // Seeded parallel: deterministic per-worker seeds via splitmix64.
    // same seed + same num_threads → bitwise-identical results.
    let dag = make_chain(20);
    let explainer = AsvExplainer::new(dag);
    let mut group = c.benchmark_group("approx_chain_20_10k_parallel");
    group.bench_function("serial_seeded", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(10_000).with_seed(42),
                )
                .unwrap()
        });
    });
    group.bench_function("parallel_2t", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(10_000)
                        .with_seed(42)
                        .with_parallel(true)
                        .with_num_threads(2),
                )
                .unwrap()
        });
    });
    group.bench_function("parallel_4t", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(10_000)
                        .with_seed(42)
                        .with_parallel(true)
                        .with_num_threads(4),
                )
                .unwrap()
        });
    });
    group.finish();
}

// ── approx: shape variants at 10k seeded ─────────────────────────────────────

fn bench_approx_diamond_10_10k_seeded(c: &mut Criterion) {
    let dag = make_diamond_dag(8); // n=10: src + 8 middles + snk
    let explainer = AsvExplainer::new(dag);
    c.bench_function("approx_diamond_10_10k_seeded", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(10_000).with_seed(42),
                )
                .unwrap()
        });
    });
}

fn bench_approx_tree_15_10k_seeded(c: &mut Criterion) {
    let dag = make_balanced_tree(3); // n=15
    let explainer = AsvExplainer::new(dag);
    c.bench_function("approx_balanced_tree_15_10k_seeded", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(10_000).with_seed(42),
                )
                .unwrap()
        });
    });
}

// ── approx vs approx_adaptive ────────────────────────────────────────────────

fn bench_approx_vs_adaptive_chain_10(c: &mut Criterion) {
    // Fixed-sample approx vs adaptive convergence: how much does adaptive save?
    // adaptive stops early when rel_change < rel_tol AND ess_ratio >= ess_ratio_min.
    let dag = make_chain(10);
    let explainer = AsvExplainer::new(dag);
    let mut group = c.benchmark_group("approx_vs_adaptive_chain_10");
    group.bench_function("fixed_10k", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(10_000).with_seed(42),
                )
                .unwrap()
        });
    });
    group.bench_function("adaptive_max10k", |b| {
        b.iter(|| {
            explainer
                .approximate_adaptive(
                    |s| Ok(black_box(s.len() as f64)),
                    AdaptiveSamplingConfig::new()
                        .with_max_samples(10_000)
                        .with_seed(42),
                )
                .unwrap()
        });
    });
    group.finish();
}

// ── approximate_uniform vs approximate (frontier IS) ─────────────────────────

fn bench_approx_vs_uniform_diamond_10(c: &mut Criterion) {
    // Diamond n=10: frontier IS has low ESS due to branching; uniform sampling has ESS=n_samples.
    let dag = make_diamond_dag(8); // n=10
    let explainer = AsvExplainer::new(dag);
    let mut group = c.benchmark_group("approx_vs_uniform_diamond_10_1k");
    group.bench_function("frontier_IS_1k", |b| {
        b.iter(|| {
            explainer
                .approximate(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(1_000).with_seed(42),
                )
                .unwrap()
        });
    });
    group.bench_function("uniform_1k", |b| {
        b.iter(|| {
            explainer
                .approximate_uniform(
                    |s| Ok(black_box(s.len() as f64)),
                    SamplingConfig::new(1_000).with_seed(42),
                )
                .unwrap()
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_exact_chain_7,
    bench_exact_balanced_tree_7,
    bench_tree_dp_balanced_7,
    bench_tree_dp_balanced_15,
    bench_tree_dp_caterpillar_10,
    bench_exact_dag_chain_10,
    bench_exact_dag_two_chains_12,
    bench_exact_dag_diamond_10,
    bench_exact_dag_chain_16,
    bench_exact_dag_sparse_chain_24,
    bench_exact_dag_vs_sparse_two_chains_20,
    bench_approx_chain_10,
    bench_approx_tree,
    bench_approx_vs_batched_chain_10,
    bench_approx_parallel_chain_20,
    bench_approx_diamond_10_10k_seeded,
    bench_approx_tree_15_10k_seeded,
    bench_approx_vs_adaptive_chain_10,
    bench_approx_vs_uniform_diamond_10,
);
criterion_main!(benches);
