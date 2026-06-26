use causasv::{AsvExplainer, Dag, SamplingConfig};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

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

fn bench_exact_chain(c: &mut Criterion) {
    let dag = make_chain(7);
    let explainer = AsvExplainer::new(dag);
    c.bench_function("exact_chain_7", |b| {
        b.iter(|| explainer.exact(|s| Ok(black_box(s.len() as f64))).unwrap());
    });
}

fn bench_approx_chain(c: &mut Criterion) {
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
    c.bench_function("approx_balanced_tree_depth3_1k", |b| {
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
    bench_exact_chain,
    bench_approx_chain,
    bench_approx_tree
);
criterion_main!(benches);
