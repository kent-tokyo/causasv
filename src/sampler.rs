use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::graph::{Dag, NodeId};

/// Configuration for approximate ASV sampling.
pub struct SamplingConfig {
    pub n_samples: usize,
    pub seed: Option<u64>,
    /// When set, enables batched value-function evaluation: collect this many samples,
    /// deduplicate coalitions, call `value_fn_batch` once, then process IS weights.
    /// Reduces Python GIL acquisition overhead for large models.
    pub batch_size: Option<usize>,
    /// When true, use Rayon parallel sampling. Seeded + parallel → per-worker seeds via
    /// splitmix64 bijection so results are deterministic regardless of thread count.
    pub parallel: bool,
    /// Number of worker threads for seeded parallel sampling. None = rayon default.
    pub num_threads: Option<usize>,
}

impl SamplingConfig {
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            seed: None,
            batch_size: None,
            parallel: false,
            num_threads: None,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    pub fn with_num_threads(mut self, num_threads: usize) -> Self {
        self.num_threads = Some(num_threads);
        self
    }
}

/// splitmix64 bijection: maps (global_seed, worker_index) → a deterministic per-worker seed.
///
/// Each worker gets a distinct seed that is a deterministic function of the global seed and
/// its index, so `seed=42, parallel=true` always produces the same aggregate result.
pub(crate) fn worker_seed(global: u64, k: usize) -> u64 {
    let mut x = global
        .wrapping_add(k as u64)
        .wrapping_mul(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Configuration for adaptive approximate ASV sampling.
///
/// Sampling runs in batches and stops when the per-node ASV estimates change by
/// less than `rel_tol` relative to their current values *and* the ESS ratio
/// exceeds `ess_ratio_min`. Falls back to `max_samples` if convergence is never
/// reached.
pub struct AdaptiveSamplingConfig {
    /// Minimum samples before convergence is checked. Default: 1 000.
    pub min_samples: usize,
    /// Hard upper bound on total samples. Default: 100 000.
    pub max_samples: usize,
    /// Samples per batch. Default: 1 000.
    pub batch_size: usize,
    /// Relative tolerance on per-node value change between batches. Default: 0.01.
    pub rel_tol: f64,
    /// Minimum required ESS / n_samples ratio. Default: 0.10.
    pub ess_ratio_min: f64,
    pub seed: Option<u64>,
}

impl Default for AdaptiveSamplingConfig {
    fn default() -> Self {
        Self {
            min_samples: 1_000,
            max_samples: 100_000,
            batch_size: 1_000,
            rel_tol: 0.01,
            ess_ratio_min: 0.10,
            seed: None,
        }
    }
}

impl AdaptiveSamplingConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn with_rel_tol(mut self, tol: f64) -> Self {
        self.rel_tol = tol;
        self
    }

    pub fn with_max_samples(mut self, max: usize) -> Self {
        self.max_samples = max;
        self
    }
}

pub(crate) struct SampledOrdering {
    pub ordering: Vec<NodeId>,
    /// log q(π) under the frontier sampler.
    /// Used by the IS estimator in approx.rs — do not remove.
    pub log_q: f64,
}

/// Reusable scratch for sampling, allocated once per worker.
/// Eliminates per-sample Vec allocation and reduces frontier management from O(n²) to O(n+edges).
pub(crate) struct SamplerScratch {
    pub in_deg: Vec<usize>,
    pub frontier: Vec<usize>, // node indices (usize)
    pub ordering: Vec<NodeId>,
}

impl SamplerScratch {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            in_deg: vec![0; n],
            frontier: Vec::with_capacity(n),
            ordering: Vec::with_capacity(n),
        }
    }
}

/// Sample one topological ordering, writing it into `scratch.ordering`.
/// Returns log q(π). `base_in_deg` must be `dag.in_degrees()` computed once per DAG.
///
/// Frontier is maintained incrementally via swap_remove + child push: O(n + edges) per sample
/// instead of the naive O(n²) full-scan-per-step approach.
pub(crate) fn sample_one_into(
    dag: &Dag,
    rng: &mut StdRng,
    scratch: &mut SamplerScratch,
    base_in_deg: &[usize],
) -> f64 {
    let n = dag.node_count();
    scratch.in_deg.copy_from_slice(base_in_deg);
    scratch.frontier.clear();
    for (i, &d) in scratch.in_deg.iter().enumerate() {
        if d == 0 {
            scratch.frontier.push(i);
        }
    }
    scratch.ordering.clear();
    let mut log_q = 0.0f64;
    for _ in 0..n {
        log_q -= (scratch.frontier.len() as f64).ln();
        let idx = rng.random_range(0..scratch.frontier.len());
        let node_idx = scratch.frontier.swap_remove(idx);
        let node = NodeId(node_idx as u32);
        scratch.ordering.push(node);
        for &child in dag.children_raw(node) {
            let c = child.0 as usize;
            scratch.in_deg[c] -= 1;
            if scratch.in_deg[c] == 0 {
                scratch.frontier.push(c);
            }
        }
    }
    log_q
}

pub(crate) fn sample_one(dag: &Dag, rng: &mut StdRng) -> SampledOrdering {
    let n = dag.node_count();
    let mut in_deg = dag.in_degrees();
    let mut ordering = Vec::with_capacity(n);
    // Build frontier once; update incrementally thereafter: O(n+edges) per sample vs O(n²).
    let mut frontier: Vec<usize> = (0..n).filter(|&i| in_deg[i] == 0).collect();
    let mut log_q = 0.0f64;
    for _ in 0..n {
        log_q -= (frontier.len() as f64).ln();
        let idx = rng.random_range(0..frontier.len());
        let node_idx = frontier.swap_remove(idx);
        let node = NodeId(node_idx as u32);
        ordering.push(node);
        for &child in dag.children_raw(node) {
            let c = child.0 as usize;
            in_deg[c] -= 1;
            if in_deg[c] == 0 {
                frontier.push(c);
            }
        }
    }
    SampledOrdering { ordering, log_q }
}

/// Sample one topological ordering uniformly at random using the precomputed dp_ind table.
///
/// At each step, the next node i is chosen with probability proportional to
/// dp_ind[remaining \ {i}], where remaining = V \ placed. This produces each linear
/// extension with probability 1/L(G), so no IS correction is needed (ESS = n_samples).
///
/// `dp_ind` must be the table returned by `compute_dp_ind(n, parents_mask)`.
/// Writes the result into `ordering` (cleared on entry).
pub(crate) fn sample_uniform_into(
    rng: &mut StdRng,
    dp_ind: &[u64],
    parents_mask: &[u64],
    ordering: &mut Vec<NodeId>,
) {
    let n = parents_mask.len();
    ordering.clear();
    let full = (1u64 << n) - 1;
    let mut placed: u64 = 0;

    for _ in 0..n {
        let remaining = full ^ placed;
        let total = dp_ind[remaining as usize];
        let r: u64 = if total > 1 {
            rng.random_range(0..total)
        } else {
            0
        };
        let mut cum: u64 = 0;
        let mut bits = remaining;
        while bits != 0 {
            let bit = bits & bits.wrapping_neg();
            let i = bit.trailing_zeros() as usize;
            bits ^= bit;
            if parents_mask[i] & placed == parents_mask[i] {
                // i is a source in G[remaining]: all its parents have been placed
                cum += dp_ind[(remaining ^ bit) as usize];
                if r < cum {
                    ordering.push(NodeId(i as u32));
                    placed |= bit;
                    break;
                }
            }
        }
    }
}

pub(crate) fn make_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_rng(&mut rand::rng()),
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
