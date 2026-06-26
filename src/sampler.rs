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
