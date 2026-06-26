use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::graph::{Dag, NodeId};

/// Configuration for approximate ASV sampling.
pub struct SamplingConfig {
    pub n_samples: usize,
    pub seed: Option<u64>,
}

impl SamplingConfig {
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            seed: None,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

pub(crate) struct SampledOrdering {
    pub ordering: Vec<NodeId>,
    /// log q(π) under the frontier sampler.
    /// Used by the IS estimator in approx.rs — do not remove.
    pub log_q: f64,
}

pub(crate) fn sample_one(dag: &Dag, rng: &mut ChaCha8Rng) -> SampledOrdering {
    let n = dag.node_count();
    let mut in_deg = dag.in_degrees();
    let mut ordering = Vec::with_capacity(n);
    let mut log_q = 0.0f64;

    for _ in 0..n {
        // Frontier: nodes with in_deg == 0 (placed nodes have in_deg = MAX)
        let frontier: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
        log_q -= (frontier.len() as f64).ln();
        let idx = rng.gen_range(0..frontier.len());
        let node = NodeId(frontier[idx] as u32);
        ordering.push(node);
        in_deg[node.0 as usize] = usize::MAX;
        for &child in dag.children_raw(node) {
            in_deg[child.0 as usize] -= 1;
        }
    }

    SampledOrdering { ordering, log_q }
}

pub(crate) fn make_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Dag;

    fn chain_2() -> Dag {
        let mut dag = Dag::new();
        let a = dag.add_node("a");
        let b = dag.add_node("b");
        dag.add_edge(a, b).unwrap();
        dag
    }

    /// Chain a→b: each step has exactly one frontier node → log_q = -ln(1) - ln(1) = 0.
    #[test]
    fn test_chain_log_q_is_zero() {
        let dag = chain_2();
        let s = sample_one(&dag, &mut make_rng(Some(0)));
        assert!((s.log_q - 0.0).abs() < 1e-12);
    }

    /// Two independent nodes: first pick from {a,b} (prob 0.5), second forced → log_q = -ln(2).
    #[test]
    fn test_two_independent_log_q() {
        let mut dag = Dag::new();
        dag.add_node("a");
        dag.add_node("b");
        let s = sample_one(&dag, &mut make_rng(Some(0)));
        let expected = -(2.0f64.ln());
        assert!((s.log_q - expected).abs() < 1e-12);
    }

    /// For a chain, every sample must have the parent before its child.
    #[test]
    fn test_chain_ordering_always_valid() {
        let dag = chain_2();
        let mut rng = make_rng(Some(1));
        for _ in 0..200 {
            let s = sample_one(&dag, &mut rng);
            assert_eq!(s.ordering.len(), 2);
            let pos: Vec<usize> = s.ordering.iter().map(|id| id.0 as usize).collect();
            assert_eq!(pos[0], 0, "a (NodeId 0) must come first in chain");
        }
    }

    /// Every sample must contain all n nodes exactly once.
    #[test]
    fn test_no_missing_no_duplicate_nodes() {
        let mut dag = Dag::new();
        for i in 0..5usize {
            dag.add_node(&format!("n{i}"));
        }
        let mut rng = make_rng(Some(2));
        for _ in 0..100 {
            let s = sample_one(&dag, &mut rng);
            let mut sorted = s.ordering.clone();
            sorted.sort_unstable();
            let expected: Vec<NodeId> = (0..5).map(|i| NodeId(i as u32)).collect();
            assert_eq!(sorted, expected);
        }
    }

    /// Same seed must produce identical ordering and log_q.
    #[test]
    fn test_same_seed_same_result() {
        let dag = chain_2();
        let s1 = sample_one(&dag, &mut make_rng(Some(42)));
        let s2 = sample_one(&dag, &mut make_rng(Some(42)));
        assert_eq!(s1.ordering, s2.ordering);
        assert_eq!(s1.log_q, s2.log_q);
    }
}
