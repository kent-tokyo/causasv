use std::collections::{BTreeMap, HashMap};

use crate::approx::approximate_asv;
use crate::cache::value_cached;
use crate::dag_dp::dag_exact_asv;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::sampler::SamplingConfig;
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
}

/// Entry point for ASV computation over a causal DAG.
pub struct AsvExplainer {
    dag: Dag,
}

impl AsvExplainer {
    pub fn new(dag: Dag) -> Self {
        Self { dag }
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
        dag_exact_asv(&self.dag, value_fn)
    }

    /// Automatic method selection based on graph size and structure.
    ///
    /// Dispatch rules:
    /// - n ≤ 8: `exact` — brute-force, lowest overhead for small n
    /// - n > 8, rooted directed tree: `exact_tree` — order-ideal DP
    /// - 8 < n ≤ 20: `exact_dag` — order-ideal DP for general DAGs
    /// - otherwise: `approximate` — IS-weighted sampling
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
        } else {
            self.approximate(value_fn, config)
        }
    }
}
