use std::collections::{BTreeMap, HashMap};

use crate::approx::{
    approximate_asv, approximate_asv_adaptive, approximate_asv_adaptive_batched,
    approximate_asv_batched, approximate_asv_uniform, approximate_asv_uniform_sparse,
    approximate_asv_uniform_sparse_adaptive, approximate_asv_uniform_sparse_adaptive_batched,
};
use crate::cache::value_cached;
use crate::dag_dp::{compute_dp_ind, dag_exact_asv};
use crate::dag_dp_sparse::{ExactDagConfig, dag_exact_asv_sparse, estimate_sparse_feasible};
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
    /// Actual method dispatched by `auto()`; None when method was caller-specified.
    pub method_used: Option<&'static str>,
}

/// Entry point for ASV computation over a causal DAG.
pub struct AsvExplainer {
    dag: Dag,
    // Bitmask of all DAG parents per node. Safe for n ≤ 63 (1u64 << p.0 is always in-range).
    // Empty for n > 63; exact_dag / exact_dag_sparse both error before using it.
    parents_mask: Vec<u64>,
    // Cached once at construction; DAG is owned and immutable after new().
    is_rooted_tree: bool,
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
        let is_rooted_tree = crate::tree::find_rooted_tree_root(&dag).is_ok();
        Self {
            dag,
            parents_mask,
            is_rooted_tree,
        }
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
            method_used: None,
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

    /// Approximate ASV via uniform topological order sampling.
    ///
    /// Unlike [`approximate`](Self::approximate) which uses IS-weighted frontier sampling,
    /// this method samples each linear extension with probability exactly 1/L(G) by
    /// consulting the `dp_ind` table at every step. Every sample contributes equally —
    /// there is no importance-weight variance, so ESS = n_samples exactly.
    ///
    /// A 2^n `dp_ind` table is precomputed in O(2^n × n) before sampling begins.
    /// For n_samples much smaller than 2^n this pays off handsomely; for n_samples ≈ 2^n
    /// use [`exact_dag`](Self::exact_dag) instead.
    ///
    /// **Limit**: n ≤ 20 (same as `exact_dag`). For larger DAGs use `approximate()`.
    pub fn approximate_uniform<F>(
        &self,
        value_fn: F,
        config: SamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        let n = self.dag.node_count();
        if n > 20 {
            return Err(CausasvError::InvalidConfig(format!(
                "approximate_uniform requires n ≤ 20 (2^n dp_ind table), got {n}; \
                 use approximate() for larger DAGs"
            )));
        }
        if config.n_samples == 0 {
            return Err(CausasvError::InvalidConfig(
                "n_samples must be > 0".to_string(),
            ));
        }
        self.dag.validate()?;
        let dp_ind = compute_dp_ind(n, &self.parents_mask)?;
        approximate_asv_uniform(value_fn, config, &dp_ind, &self.parents_mask)
    }

    /// Uniform topological ordering sampler for sparse DAGs with n > 20.
    ///
    /// Uses lazily memoized dp_ind (HashMap) instead of a precomputed 2^n table,
    /// enabling uniform sampling for n up to 63 on DAGs with manageable state spaces.
    /// Every linear extension is sampled with equal probability — ESS = n_samples exactly,
    /// with no IS weight variance. Returns `Err(Overflow)` if the dp_ind cache exceeds
    /// 2 GiB (same memory limit as `exact_dag_sparse`).
    pub fn approximate_uniform_sparse<F>(
        &self,
        value_fn: F,
        config: SamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        let n = self.dag.node_count();
        if n > 63 {
            return Err(CausasvError::InvalidConfig(format!(
                "approximate_uniform_sparse requires n ≤ 63 (u64 bitmask), got {n}; \
                 use approximate() for larger DAGs"
            )));
        }
        if config.n_samples == 0 {
            return Err(CausasvError::InvalidConfig(
                "n_samples must be > 0".to_string(),
            ));
        }
        self.dag.validate()?;
        approximate_asv_uniform_sparse(
            &self.dag,
            value_fn,
            config,
            &self.parents_mask,
            2 * 1024 * 1024 * 1024, // 2 GiB — same as ExactDagConfig default
        )
    }

    /// Adaptive uniform topological ordering sampler for sparse DAGs.
    ///
    /// Like `approximate_uniform_sparse` but with adaptive stopping: runs in batches until
    /// per-node estimates stabilize (`rel_tol`) or `max_samples` is reached. Returns per-node
    /// stderr and convergence flag. ESS = n_samples exactly (no IS weight variance).
    pub fn approximate_uniform_sparse_adaptive<F>(
        &self,
        value_fn: F,
        config: AdaptiveSamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        let n = self.dag.node_count();
        if n > 63 {
            return Err(CausasvError::InvalidConfig(format!(
                "approximate_uniform_sparse_adaptive requires n ≤ 63, got {n}"
            )));
        }
        self.dag.validate()?;
        approximate_asv_uniform_sparse_adaptive(
            &self.dag,
            value_fn,
            config,
            &self.parents_mask,
            2 * 1024 * 1024 * 1024,
        )
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
    /// - 8 < n ≤ 20: `exact_dag_sparse` if edge_count ≤ 2n (sparse heuristic), else `exact_dag`
    /// - 20 < n ≤ 28: `exact_dag_sparse` — sparse BFS over order ideals;
    ///   falls back to `approximate` on memory-limit or overflow errors
    /// - 28 < n ≤ 63: `exact_dag_sparse` if order ideal count ≤ 250k (sparse preflight),
    ///   else `approximate`
    /// - n > 63: `approximate` — u64 bitmask limit
    ///
    /// `config` is used only when the approximate path is taken.
    pub fn auto<F>(&self, value_fn: F, config: SamplingConfig) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError> + Send + Sync,
    {
        self.dag.validate()?;
        let n = self.dag.node_count();
        if n <= 8 {
            let mut r = self.exact(value_fn)?;
            r.method_used = Some("exact");
            Ok(r)
        } else if self.is_rooted_tree {
            let mut r = self.exact_tree(value_fn)?;
            r.method_used = Some("exact_tree");
            Ok(r)
        } else if n <= 20 {
            let m = self.dag.edge_count();
            if m <= 2 * n {
                // Sparse DAG: try sparse DP first; fall back to dense on the rare Overflow edge case.
                // Use a closure to borrow value_fn without consuming it (same pattern as n<=28 branch).
                match self.exact_dag_sparse_with_config(|c| value_fn(c), &ExactDagConfig::default())
                {
                    Ok(mut r) => {
                        r.method_used = Some("exact_dag_sparse");
                        Ok(r)
                    }
                    Err(CausasvError::InvalidConfig(_)) | Err(CausasvError::Overflow(_)) => {
                        let mut r = self.exact_dag(value_fn)?;
                        r.method_used = Some("exact_dag");
                        Ok(r)
                    }
                    Err(e) => Err(e),
                }
            } else {
                let mut r = self.exact_dag(value_fn)?;
                r.method_used = Some("exact_dag");
                Ok(r)
            }
        } else if n <= 28 {
            // Use a closure to borrow value_fn without consuming it, so we can
            // fall back to approximate if sparse DP hits the memory or overflow limit.
            match self.exact_dag_sparse_with_config(|c| value_fn(c), &ExactDagConfig::default()) {
                Ok(mut r) => {
                    r.method_used = Some("exact_dag_sparse");
                    Ok(r)
                }
                Err(CausasvError::InvalidConfig(ref msg))
                | Err(CausasvError::Overflow(ref msg)) => {
                    let mut r = self.approximate(value_fn, config)?;
                    r.fallback_from = Some("exact_dag_sparse".to_string());
                    r.fallback_reason = Some(msg.clone());
                    r.method_used = Some("approx");
                    Ok(r)
                }
                Err(e) => Err(e),
            }
        } else if n <= 63 {
            // For n > 28 and n ≤ 63: run a cheap BFS preflight to count order ideals.
            // If the state count is manageable, use exact_dag_sparse with an elevated
            // max_nodes limit; otherwise fall back to approximate.
            if estimate_sparse_feasible(&self.dag, &self.parents_mask, 250_000) {
                let sparse_cfg = ExactDagConfig {
                    max_nodes: n,
                    ..ExactDagConfig::default()
                };
                match self.exact_dag_sparse_with_config(|c| value_fn(c), &sparse_cfg) {
                    Ok(mut r) => {
                        r.method_used = Some("exact_dag_sparse");
                        Ok(r)
                    }
                    Err(CausasvError::InvalidConfig(ref msg))
                    | Err(CausasvError::Overflow(ref msg)) => {
                        let mut r = self.approximate(value_fn, config)?;
                        r.fallback_from = Some("exact_dag_sparse".to_string());
                        r.fallback_reason = Some(msg.clone());
                        r.method_used = Some("approx");
                        Ok(r)
                    }
                    Err(e) => Err(e),
                }
            } else {
                let mut r = self.approximate(value_fn, config)?;
                r.method_used = Some("approx");
                Ok(r)
            }
        } else {
            let mut r = self.approximate(value_fn, config)?;
            r.method_used = Some("approx");
            Ok(r)
        }
    }

    /// Quality-first automatic dispatch: like `auto()` but the IS approximate fallback is
    /// replaced by `approximate_uniform_sparse_adaptive` (zero IS variance, ESS = n_samples,
    /// stderr + CI always available). For n > 63 falls back to `approximate_adaptive`.
    ///
    /// Every code path returns `stderr` and `converged` — use `explain_quality()` in Python
    /// for the friendliest interface.
    ///
    /// Dispatch rules:
    /// - n ≤ 8: `exact`
    /// - rooted tree: `exact_tree`
    /// - 8 < n ≤ 20: `exact_dag_sparse` (sparse-first) or `exact_dag`
    /// - 20 < n ≤ 63: sparse preflight (≤ 250k order ideals) → `exact_dag_sparse`;
    ///   preflight fails or BFS overflows → `approximate_uniform_sparse_adaptive`
    /// - n > 63: `approximate_adaptive`
    pub fn auto_quality<F>(
        &self,
        value_fn: F,
        config: AdaptiveSamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
    {
        self.dag.validate()?;
        let n = self.dag.node_count();
        if n <= 8 {
            let mut r = self.exact(value_fn)?;
            r.method_used = Some("exact");
            Ok(r)
        } else if self.is_rooted_tree {
            let mut r = self.exact_tree(value_fn)?;
            r.method_used = Some("exact_tree");
            Ok(r)
        } else if n <= 20 {
            let m = self.dag.edge_count();
            if m <= 2 * n {
                match self.exact_dag_sparse_with_config(|c| value_fn(c), &ExactDagConfig::default())
                {
                    Ok(mut r) => {
                        r.method_used = Some("exact_dag_sparse");
                        Ok(r)
                    }
                    Err(CausasvError::InvalidConfig(_)) | Err(CausasvError::Overflow(_)) => {
                        let mut r = self.exact_dag(value_fn)?;
                        r.method_used = Some("exact_dag");
                        Ok(r)
                    }
                    Err(e) => Err(e),
                }
            } else {
                let mut r = self.exact_dag(value_fn)?;
                r.method_used = Some("exact_dag");
                Ok(r)
            }
        } else if n <= 63 {
            // Cheap preflight: count order ideals up to 250k before committing to full BFS.
            if estimate_sparse_feasible(&self.dag, &self.parents_mask, 250_000) {
                let sparse_cfg = ExactDagConfig {
                    max_nodes: n.min(63),
                    ..ExactDagConfig::default()
                };
                match self.exact_dag_sparse_with_config(|c| value_fn(c), &sparse_cfg) {
                    Ok(mut r) => {
                        r.method_used = Some("exact_dag_sparse");
                        Ok(r)
                    }
                    Err(CausasvError::InvalidConfig(ref msg))
                    | Err(CausasvError::Overflow(ref msg)) => {
                        let mut r = self.approximate_uniform_sparse_adaptive(value_fn, config)?;
                        r.fallback_from = Some("exact_dag_sparse".to_string());
                        r.fallback_reason = Some(msg.clone());
                        r.method_used = Some("uniform_sparse_adaptive");
                        Ok(r)
                    }
                    Err(e) => Err(e),
                }
            } else {
                // Preflight exceeded 250k budget — skip exact, go to uniform sparse adaptive.
                let mut r = self.approximate_uniform_sparse_adaptive(value_fn, config)?;
                r.method_used = Some("uniform_sparse_adaptive");
                Ok(r)
            }
        } else {
            // n > 63: uniform_sparse_adaptive requires n ≤ 63; use IS adaptive.
            let mut r = self.approximate_adaptive(value_fn, config)?;
            r.method_used = Some("approx_adaptive");
            Ok(r)
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

    /// Batched uniform sparse adaptive ASV.
    ///
    /// Like `approximate_uniform_sparse_adaptive` but evaluates all unique coalition
    /// masks from an entire batch of sampled orderings with a single `value_fn_batch`
    /// call, reducing Python GIL round-trips from O(n × batch_size) to
    /// O(unique_masks_per_batch). ESS = n_samples exactly (no IS variance).
    ///
    /// Requires n ≤ 63. For larger DAGs, use `approximate_adaptive_batched`.
    pub fn approximate_uniform_sparse_adaptive_batched<F>(
        &self,
        value_fn_batch: F,
        config: AdaptiveSamplingConfig,
    ) -> Result<AsvResult, CausasvError>
    where
        F: Fn(&[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError>,
    {
        let n = self.dag.node_count();
        if n > 63 {
            return Err(CausasvError::InvalidConfig(format!(
                "approximate_uniform_sparse_adaptive_batched requires n ≤ 63, got {n}"
            )));
        }
        self.dag.validate()?;
        approximate_asv_uniform_sparse_adaptive_batched(
            &self.dag,
            value_fn_batch,
            config,
            &self.parents_mask,
            2 * 1024 * 1024 * 1024,
        )
    }
}
