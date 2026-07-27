use causasv::{Cpdag, Dag, NodeId};
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

/// A chain of forks: n0 is a shared confounder for a long downstream chain,
/// plus a fresh confounder every 5 nodes -- exercises the Markov-boundary /
/// ancestor-set machinery with many non-adjacent pairs to check.
fn make_fork_chain(n: usize) -> Dag {
    let mut dag = Dag::new();
    let nodes: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
    for i in 0..n - 1 {
        dag.add_edge(nodes[i], nodes[i + 1]).unwrap();
        if i % 5 == 0 && i + 5 < n {
            dag.add_edge(nodes[i], nodes[i + 5]).unwrap();
        }
    }
    dag
}

/// A layered sparse DAG: each layer's nodes each get one edge to a node in
/// the next layer (round-robin), giving multiple non-adjacent same-layer
/// candidates per Markov boundary check.
fn make_layered_sparse(n_layers: usize, layer_width: usize) -> Dag {
    let mut dag = Dag::new();
    let mut layers: Vec<Vec<NodeId>> = Vec::new();
    for l in 0..n_layers {
        let layer: Vec<NodeId> = (0..layer_width)
            .map(|i| dag.add_node(&format!("l{l}n{i}")))
            .collect();
        if let Some(prev) = layers.last() {
            for (i, &node) in layer.iter().enumerate() {
                dag.add_edge(prev[i % prev.len()], node).unwrap();
            }
        }
        layers.push(layer);
    }
    dag
}

/// A collider-rich DAG: every node in the "sink" layer has several
/// pairwise-non-adjacent parents from the "source" layer -- worst case for
/// the moralization step (many marrying edges).
fn make_collider_rich(n_sources: usize, n_sinks: usize) -> Dag {
    let mut dag = Dag::new();
    let sources: Vec<_> = (0..n_sources)
        .map(|i| dag.add_node(&format!("src{i}")))
        .collect();
    for j in 0..n_sinks {
        let sink = dag.add_node(&format!("sink{j}"));
        for &src in sources.iter().take(4) {
            dag.add_edge(src, sink).unwrap();
        }
    }
    dag
}

fn cpdag_from_dag(dag: &Dag) -> Cpdag {
    let adjacent = |a: NodeId, b: NodeId| {
        dag.children(a).unwrap().contains(&b) || dag.children(b).unwrap().contains(&a)
    };
    let is_compelled = |p: NodeId, c: NodeId| {
        dag.parents(c)
            .unwrap()
            .iter()
            .any(|&p2| p2 != p && !adjacent(p2, p))
    };
    let mut cpdag = Cpdag::new();
    for id in dag.all_nodes() {
        cpdag.add_node(dag.node_name(id).unwrap());
    }
    for id in dag.all_nodes() {
        for &c in dag.children(id).unwrap() {
            if is_compelled(id, c) {
                cpdag.add_directed_edge(id, c).unwrap();
            } else {
                cpdag.add_undirected_edge(id, c).unwrap();
            }
        }
    }
    cpdag
}

// ── d_convex_hull ────────────────────────────────────────────────────────────

fn bench_d_convex_hull_chain_50(c: &mut Criterion) {
    let dag = make_chain(50);
    let required = [NodeId(0), NodeId(49)];
    c.bench_function("d_convex_hull_chain_50", |b| {
        b.iter(|| dag.d_convex_hull(black_box(&required)).unwrap());
    });
}

fn bench_d_convex_hull_fork_chain_50(c: &mut Criterion) {
    let dag = make_fork_chain(50);
    let required = [NodeId(0), NodeId(49)];
    c.bench_function("d_convex_hull_fork_chain_50", |b| {
        b.iter(|| dag.d_convex_hull(black_box(&required)).unwrap());
    });
}

fn bench_d_convex_hull_layered_sparse(c: &mut Criterion) {
    let dag = make_layered_sparse(10, 8); // 80 nodes
    let first = NodeId(0);
    let last = NodeId((dag.node_count() - 1) as u32);
    let required = [first, last];
    c.bench_function("d_convex_hull_layered_sparse_80", |b| {
        b.iter(|| dag.d_convex_hull(black_box(&required)).unwrap());
    });
}

fn bench_d_convex_hull_collider_rich(c: &mut Criterion) {
    let dag = make_collider_rich(10, 20); // 30 nodes, dense moralization
    let required = [NodeId(10), NodeId(29)]; // two sinks
    c.bench_function("d_convex_hull_collider_rich_30", |b| {
        b.iter(|| dag.d_convex_hull(black_box(&required)).unwrap());
    });
}

// ── strong_d_convex_hull ─────────────────────────────────────────────────────

fn bench_strong_d_convex_hull_chain_50(c: &mut Criterion) {
    let dag = make_chain(50);
    let required = [NodeId(0), NodeId(49)];
    c.bench_function("strong_d_convex_hull_chain_50", |b| {
        b.iter(|| dag.strong_d_convex_hull(black_box(&required)).unwrap());
    });
}

fn bench_strong_d_convex_hull_layered_sparse(c: &mut Criterion) {
    let dag = make_layered_sparse(10, 8);
    let first = NodeId(0);
    let last = NodeId((dag.node_count() - 1) as u32);
    let required = [first, last];
    c.bench_function("strong_d_convex_hull_layered_sparse_80", |b| {
        b.iter(|| dag.strong_d_convex_hull(black_box(&required)).unwrap());
    });
}

fn bench_strong_d_convex_hull_collider_rich(c: &mut Criterion) {
    let dag = make_collider_rich(10, 20);
    let required = [NodeId(10), NodeId(29)];
    c.bench_function("strong_d_convex_hull_collider_rich_30", |b| {
        b.iter(|| dag.strong_d_convex_hull(black_box(&required)).unwrap());
    });
}

// ── Cpdag::strong_d_convex_hull (includes consistent_extension) ────────────

fn bench_cpdag_strong_d_convex_hull_layered_sparse(c: &mut Criterion) {
    let dag = make_layered_sparse(10, 8);
    let cpdag = cpdag_from_dag(&dag);
    let first = NodeId(0);
    let last = NodeId((dag.node_count() - 1) as u32);
    let required = [first, last];
    c.bench_function("cpdag_strong_d_convex_hull_layered_sparse_80", |b| {
        b.iter(|| cpdag.strong_d_convex_hull(black_box(&required)).unwrap());
    });
}

criterion_group!(
    benches,
    bench_d_convex_hull_chain_50,
    bench_d_convex_hull_fork_chain_50,
    bench_d_convex_hull_layered_sparse,
    bench_d_convex_hull_collider_rich,
    bench_strong_d_convex_hull_chain_50,
    bench_strong_d_convex_hull_layered_sparse,
    bench_strong_d_convex_hull_collider_rich,
    bench_cpdag_strong_d_convex_hull_layered_sparse,
);
criterion_main!(benches);
