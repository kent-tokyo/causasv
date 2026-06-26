use std::collections::BTreeMap;

use crate::approx::approximate_asv;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::sampler::SamplingConfig;
use crate::topo::enumerate_topos;
use crate::tree::tree_exact_asv;

/// Result of an ASV computation.
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
        let orderings = enumerate_topos(&self.dag)?;
        let n_orderings = orderings.len();
        let n = self.dag.node_count();
        let mut phi = vec![0.0f64; n];
        let mut coalition = Vec::with_capacity(n);

        for ordering in &orderings {
            for (k, &node) in ordering.iter().enumerate() {
                coalition.clear();
                coalition.extend_from_slice(&ordering[..k]);
                coalition.sort_unstable();
                let v_without = value_fn(&coalition)?;

                coalition.push(node);
                coalition.sort_unstable();
                let v_with = value_fn(&coalition)?;

                phi[node.0 as usize] += v_with - v_without;
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
        F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
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
}
