use std::collections::{BTreeMap, HashMap};

use crate::approx::{
    approximate_asv, approximate_asv_adaptive, approximate_asv_adaptive_batched,
    approximate_asv_batched,
};
use crate::cache::value_cached;
use crate::dag_dp::dag_exact_asv;
use crate::dag_dp_sparse::{ExactDagConfig, dag_exact_asv_sparse};
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::sampler::{AdaptiveSamplingConfig, SamplingConfig};
use crate::topo::enumerate_topos;
use crate::tree::tree_exact_asv;

/// Result of an ASV computation.
#[derive(Debug)]
pub struct AsvResult {
    /// Per-node ASV values in ascending NodeId order.
    pub values: BTreeMap<NodeId, f64>,
    /// Number of topological orderings (exact) or samples used (approximate).
    pub n_samples: usize,
    /// RNG seed used; None for exact computation or unseeded approximate.
    pub seed: Option<u64>,
    /// True if exact (brute-force or tree-exact), false if approximate.
    pub is_exact: bool,
    /// Effective Sample Size for approximate methods: ESS = (Σw)² / Σw².
    /// ESS ≈ n_samples → uniform IS weights (reliable). ESS ≪ n_samples → high variance.
    /// None for exact methods.
    pub effective_sample_size: Option<f64>,
    /// True if adaptive sampling converged before max_samples. None for non-adaptive methods.
    pub converged: Option<bool>,
    /// Per-node IS standard error estimates. None for exact and non-adaptive approx.
    pub stderr: Option<BTreeMap<NodeId, f64>>,
    /// Number of order ideals visited by exact_dag_sparse. None for other methods.
    pub n_order_ideals: Option<usize>,
    /// Fraction of 2^n states visited: n_order_ideals / 2^n. None for other methods.
    pub state_ratio: Option<f64>,
    /// Estimated memory used by sparse DP in MB. None for other methods.
    pub memory_mb: Option<f64>,
    /// Method that was attempted before falling back (e.g. "exact_dag_sparse"). None normally.
    pub fallback_from: Option<String>,
    /// Reason for the fallback (error message). None normally.
    pub fallback_reason: Option<String>,
}

/// Entry point for ASV computation over a causal DAG.
pub struct AsvExplainer {
    dag: Dag,
    // Bitmask of all DAG parents per node. Safe for n ≤ 63 (1u64 << p.0 is always in-range).
    // Empty for n > 63; exact_dag / exact_dag_sparse both error before using it.
    parents_mask: Vec<u64>,
}

impl AsvExplainer {
    pub fn new(dag: Dag) -> Self {
        let n = dag.node_count();
        let parents_mask = if n <= 63 {
            (0..n)
                .map(|i| {
                    dag.parents_raw(NodeId(i as u32))
                        .iter()
                        .fold(0u64, |m, &p| m | (1u64 << p.0))
                })
                .collect()
        } else {
            vec![]
        };
        Self { dag, parents_mask }
    }

    /// Brute-force exact ASV: enumerates all topological orderings.
    /// Correct for any DAG; only practical for n ≤ ~8.
    pub fn exact<F>(&self, value_fn: F) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.dag.validate()?;
        let n = self.dag.node_count();
        if n > 64 {
            return Err(CausasvError::InvalidConfig(format!(
                "bitmask coalitions require n ≤ 64, got {n}"
            )));
        }
        let orderings = enumerate_topos(&self.dag)?;
        let n_orderings = orderings.len();
        let mut phi = vec![0.0f64; n];
        let mut cache = HashMap::<u64, f64>::new();

        for ordering in &orderings {
            let mut prefix_mask: u64 = 0;
            for &node in ordering {
                let without = prefix_mask;
                let with_node = prefix_mask | (1u64 << node.0);
                phi[node.0 as usize] += value_cached(&mut cache, &value_fn, with_node)?
                    - value_cached(&mut cache, &value_fn, without)?;
                prefix_mask = with_node;
            }
        }

        let scale = 1.0 / n_orderings as f64;
        let values = (0..n).map(|i| (NodeId(i as u32), phi[i] * scale)).collect();

        Ok(AsvResult {
            values,
            n_samples: n_orderings,
            seed: None,
            is_exact: true,
            effective_sample_size: None,
            converged: None,
            stderr: None,
            n_order_ideals: None,
            state_ratio: None,
            memory_mb: None,
            fallback_from: None,
            fallback_reason: None,
        })
    }

    /// Approximate ASV via IS-weighted topological order sampling.
    /// Works for any DAG size; use a large n_samples for accuracy.
    pub fn approximate<F>(
        &self,
        value_fn: F,
        config: SamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError> + Send + Sync,
    {
        self.dag.validate()?;
        approximate_asv(&self.dag, value_fn, config)
    }

    /// Exact ASV for rooted directed trees. Returns Err(NotRootedTree) if the graph is not one.
    /// Validates tree structure before computing; otherwise identical to `exact`.
    pub fn exact_tree<F>(&self, value_fn: F) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.dag.validate()?;
        tree_exact_asv(&self.dag, value_fn)
    }

    /// Exact ASV for any DAG via order-ideal DP. Practical for n ≤ 20.
    ///
    /// Computes `dp[mask]` = number of linear extensions with prefix `mask`,
    /// then accumulates weighted marginal contributions. O(2^n × n) time.
    pub fn exact_dag<F>(&self, value_fn: F) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.dag.validate()?;
        dag_exact_asv(&self.dag, value_fn, &self.parents_mask)
    }

    /// Exact ASV for general DAGs using sparse order-ideal DP.
    ///
    /// BFS-enumerates only valid order ideals — sets S where all parents of every
    /// i ∈ S are also in S. For sparse DAGs the number of valid order ideals can be
    /// far less than 2^n, enabling exact computation for n > 20.
    ///
    /// Uses the default `ExactDagConfig` (max_nodes=28, memory_limit_bytes=2GiB).
    /// Returns `Err` if n > max_nodes or if the memory limit is exceeded mid-BFS.
    pub fn exact_dag_sparse<F>(&self, value_fn: F) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.exact_dag_sparse_with_config(value_fn, &ExactDagConfig::default())
    }

    /// Like `exact_dag_sparse` but with a custom `ExactDagConfig`.
    pub fn exact_dag_sparse_with_config<F>(
        &self,
        value_fn: F,
        config: &ExactDagConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.dag.validate()?;
        let (result, _, _, _) =
            dag_exact_asv_sparse(&self.dag, value_fn, config, &self.parents_mask)?;
        Ok(result)
    }

    /// Automatic method selection based on graph size and structure.
    ///
    /// Dispatch rules:
    /// - n ≤ 8: `exact` — brute-force, lowest overhead for small n
    /// - n > 8, rooted directed tree: `exact_tree` — order-ideal DP
    /// - 8 < n ≤ 20: `exact_dag` — dense order-ideal DP (O(2^n × n))
    /// - 20 < n ≤ 28: `exact_dag_sparse` — sparse BFS over order ideals;
    ///   falls back to `approximate` on memory-limit or overflow errors
    /// - n > 28: `approximate` — IS-weighted sampling
    ///
    /// `config` is used only when the approximate path is taken.
    pub fn auto<F>(&self, value_fn: F, config: SamplingConfig) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError> + Send + Sync,
    {
        self.dag.validate()?;
        let n = self.dag.node_count();
        if n <= 8 {
            self.exact(value_fn)
        } else if crate::tree::find_rooted_tree_root(&self.dag).is_ok() {
            self.exact_tree(value_fn)
        } else if n <= 20 {
            self.exact_dag(value_fn)
        } else if n <= 28 {
            // Use a closure to borrow value_fn without consuming it, so we can
            // fall back to approximate if sparse DP hits the memory or overflow limit.
            match self.exact_dag_sparse_with_config(|c| value_fn(c), &ExactDagConfig::default()) {
                Ok(r) => Ok(r),
                Err(CausasvError::InvalidConfig(ref msg))
                | Err(CausasvError::Overflow(ref msg)) => {
                    let mut r = self.approximate(value_fn, config)?;
                    r.fallback_from = Some("exact_dag_sparse".to_string());
                    r.fallback_reason = Some(msg.clone());
                    Ok(r)
                }
                Err(e) => Err(e),
            }
        } else {
            self.approximate(value_fn, config)
        }
    }

    /// Adaptive approximate ASV: runs sampling in batches and stops when estimates
    /// converge (relative change < `config.rel_tol` and ESS ratio ≥ `config.ess_ratio_min`),
    /// or when `config.max_samples` is reached.
    ///
    /// Always single-threaded for deterministic convergence behavior.
    /// Returns per-node standard error estimates alongside ASV values.
    pub fn approximate_adaptive<F>(
        &self,
        value_fn: F,
        config: AdaptiveSamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.dag.validate()?;
        approximate_asv_adaptive(&self.dag, value_fn, config)
    }

    /// Batched approximate ASV: like `approximate`, but evaluates coalitions in batches
    /// via `value_fn_batch(coalitions) -> values` to reduce per-call overhead (e.g. Python GIL).
    ///
    /// Set `config.batch_size` to control how many samples to collect per batch call.
    /// With `batch_size=1`, results are identical to `approximate` for the same seed.
    pub fn approximate_batched<F>(
        &self,
        value_fn_batch: F,
        config: SamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError>,
    {
        self.dag.validate()?;
        approximate_asv_batched(&self.dag, value_fn_batch, config)
    }

    /// Adaptive batched approximate ASV: like `approximate_adaptive`, but evaluates
    /// coalitions in batches via `value_fn_batch`.
    ///
    /// Each sampling batch (of `config.batch_size` samples) becomes one `value_fn_batch` call.
    pub fn approximate_adaptive_batched<F>(
        &self,
        value_fn_batch: F,
        config: AdaptiveSamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError>,
    {
        self.dag.validate()?;
        approximate_asv_adaptive_batched(&self.dag, value_fn_batch, config)
    }
}
