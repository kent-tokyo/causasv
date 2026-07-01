use std::collections::{HashMap, VecDeque};

use crate::asv::AsvResult;
use crate::cache::value_cached;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};

/// Configuration for sparse order-ideal exact DAG DP.
pub struct ExactDagConfig {
    /// Maximum node count allowed. Default: 28.
    pub max_nodes: usize,
    /// Abort if estimated memory (dp_fwd + dp_ind HashMap entries × ~80 bytes) exceeds this.
    /// Default: 2 GiB.
    pub memory_limit_bytes: usize,
}

impl Default for ExactDagConfig {
    fn default() -> Self {
        Self {
            max_nodes: 28,
            memory_limit_bytes: 2 * 1024 * 1024 * 1024,
        }
    }
}

/// Exact ASV for general DAGs using sparse order-ideal DP.
///
/// Unlike `dag_exact_asv` which iterates all 2^n bitmasks, this BFS-enumerates
/// only valid order ideals — sets S where every i ∈ S has all parents(i) ⊆ S.
/// For sparse DAGs (chains, trees, few branching points), |order_ideals| ≪ 2^n,
/// enabling exact computation for n > 20 within practical time and memory.
///
/// Returns (AsvResult, n_order_ideals, state_ratio, memory_mb).
pub(crate) fn dag_exact_asv_sparse<F>(
    dag: &Dag,
    value_fn: F,
    config: &ExactDagConfig,
    parents_mask: &[u64],
) -> Result<(AsvResult, usize, f64, f64), CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    let n = dag.node_count();
    if n > config.max_nodes {
        return Err(CausasvError::InvalidConfig(format!(
            "exact_dag_sparse: n={n} exceeds max_nodes={}; use approximate method for large DAGs",
            config.max_nodes
        )));
    }
    if n > 63 {
        return Err(CausasvError::InvalidConfig(format!(
            "bitmask representation requires n ≤ 63 for sparse DP, got {n}"
        )));
    }

    let full_mask: u64 = (1u64 << n) - 1;

    // BFS over valid order ideals.
    // dp_fwd[S] = # of linear extensions with S as the set of already-placed nodes.
    let mut dp_fwd: HashMap<u64, u64> = HashMap::new();
    dp_fwd.insert(0, 1);
    let mut queue: VecDeque<u64> = VecDeque::new();
    queue.push_back(0);

    while let Some(s) = queue.pop_front() {
        let fwd = *dp_fwd.get(&s).unwrap();
        for (i, &pmask) in parents_mask.iter().enumerate() {
            if s & (1u64 << i) != 0 {
                continue; // i already in ideal
            }
            if pmask & s == pmask {
                // all parents of i are in S → S ∪ {i} is a valid order ideal
                let s_next = s | (1u64 << i);
                let entry = dp_fwd.entry(s_next).or_insert(0);
                let is_new = *entry == 0;
                *entry = entry.checked_add(fwd).ok_or_else(|| {
                    CausasvError::Overflow(format!(
                        "dp_fwd[{s_next:#b}] overflowed u64 — use approx for this DAG"
                    ))
                })?;
                if is_new {
                    queue.push_back(s_next);
                    // Memory guard: ~80 bytes per entry (two HashMaps combined)
                    if dp_fwd.len() * 80 > config.memory_limit_bytes {
                        return Err(CausasvError::InvalidConfig(format!(
                            "exact_dag_sparse: exceeded memory limit ({} bytes) after {} order ideals; \
                             try approximate method",
                            config.memory_limit_bytes,
                            dp_fwd.len()
                        )));
                    }
                }
            }
        }
    }

    let n_order_ideals = dp_fwd.len();
    let state_ratio = n_order_ideals as f64 / (1u64 << n) as f64;

    // dp_ind[T] = # of linear extensions of G[T] for arbitrary subset T,
    // computed lazily with memoization. Needed for suffix weight computation.
    let mut dp_ind: HashMap<u64, u64> = HashMap::new();

    let total = dp_ind_lazy(full_mask, parents_mask, n, &mut dp_ind)? as f64;

    // Accumulate ASV weights
    let mut phi = vec![0.0f64; n];
    let mut value_cache: HashMap<u64, f64> = HashMap::new();

    for (&s, &fwd_count) in &dp_fwd {
        if fwd_count == 0 {
            continue;
        }
        let w_prefix = fwd_count as f64;
        for (i, &pmask) in parents_mask.iter().enumerate() {
            if s & (1u64 << i) != 0 {
                continue; // i already in S
            }
            if pmask & s != pmask {
                continue; // not all parents of i in S
            }
            // suffix = V \ (S ∪ {i})
            let suffix_mask = full_mask ^ (s | (1u64 << i));
            let w_suffix = dp_ind_lazy(suffix_mask, parents_mask, n, &mut dp_ind)? as f64;
            let s_with_i = s | (1u64 << i);
            phi[i] += w_prefix
                * w_suffix
                * (value_cached(&mut value_cache, &value_fn, s_with_i)?
                    - value_cached(&mut value_cache, &value_fn, s)?);
        }
    }

    let values = (0..n).map(|i| (NodeId(i as u32), phi[i] / total)).collect();

    // Estimate memory: each HashMap entry is ~80 bytes (8 bytes key + 8 bytes value + overhead)
    let memory_mb = ((dp_fwd.len() + dp_ind.len()) * 80) as f64 / (1024.0 * 1024.0);

    let result = AsvResult {
        values,
        n_samples: n_order_ideals,
        seed: None,
        is_exact: true,
        effective_sample_size: None,
        converged: None,
        stderr: None,
        n_order_ideals: Some(n_order_ideals),
        state_ratio: Some(state_ratio),
        memory_mb: Some(memory_mb),
        fallback_from: None,
        fallback_reason: None,
        method_used: None,
    };

    Ok((result, n_order_ideals, state_ratio, memory_mb))
}

/// Lazy memoized computation of dp_ind[mask] = # of linear extensions of G[mask].
///
/// Node i is a source in G[mask] iff none of its parents (in the full graph) are in mask.
/// Recurrence: dp_ind[mask] = Σ dp_ind[mask \ {i}] over sources i in G[mask].
///
/// Returns `Overflow` if the count exceeds u64::MAX.
fn dp_ind_lazy(
    mask: u64,
    parents_mask: &[u64],
    n: usize,
    cache: &mut HashMap<u64, u64>,
) -> Result<u64, CausasvError> {
    if mask == 0 {
        return Ok(1);
    }
    if let Some(&v) = cache.get(&mask) {
        return Ok(v);
    }
    let mut result = 0u64;
    for (i, &pmask) in parents_mask.iter().enumerate().take(n) {
        if mask & (1u64 << i) == 0 {
            continue; // i not in mask
        }
        if pmask & mask == 0 {
            // i has no parents in mask → source in G[mask]
            let sub = dp_ind_lazy(mask ^ (1u64 << i), parents_mask, n, cache)?;
            result = result.checked_add(sub).ok_or_else(|| {
                CausasvError::Overflow(format!(
                    "dp_ind[{mask:#b}] overflowed u64 — use approx for this DAG"
                ))
            })?;
        }
    }
    cache.insert(mask, result);
    Ok(result)
}

/// Public wrapper around `dp_ind_lazy` for use by the sparse uniform sampler in sampler.rs.
pub(crate) fn dp_ind_lazy_pub(
    mask: u64,
    parents_mask: &[u64],
    n: usize,
    cache: &mut HashMap<u64, u64>,
) -> Result<u64, crate::error::CausasvError> {
    dp_ind_lazy(mask, parents_mask, n, cache)
}

/// Order-ideal count above which `dag_exact_asv_sparse`'s memory guard would reject
/// the DAG (that guard rejects once `dp_fwd.len()` times 80 bytes exceeds
/// `config.memory_limit_bytes`). Lets a preflight check make the same accept/reject
/// decision as that guard, proactively instead of via a caught error.
pub(crate) fn sparse_state_budget(config: &ExactDagConfig) -> usize {
    config.memory_limit_bytes / 80
}

/// BFS-count valid order ideals up to `state_budget`; returns `true` if the full DAG
/// has at most that many, `false` if the budget is exceeded.
///
/// Does not compute dp_ind or call value_fn — this is a cheap preflight to decide
/// whether exact_dag_sparse is worth running for n > 28.
pub(crate) fn estimate_sparse_feasible(
    dag: &Dag,
    parents_mask: &[u64],
    state_budget: usize,
) -> bool {
    use std::collections::HashSet;
    let n = dag.node_count();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: std::collections::VecDeque<u64> = std::collections::VecDeque::new();
    visited.insert(0);
    queue.push_back(0);
    while let Some(s) = queue.pop_front() {
        for (i, &pmask) in parents_mask.iter().enumerate().take(n) {
            let bit = 1u64 << i;
            if s & bit == 0 && pmask & s == pmask {
                let ns = s | bit;
                if visited.insert(ns) {
                    if visited.len() > state_budget {
                        return false;
                    }
                    queue.push_back(ns);
                }
            }
        }
    }
    true
}
