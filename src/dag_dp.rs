use std::collections::HashMap;

use crate::asv::AsvResult;
use crate::cache::value_cached;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};

/// Exact ASV for general DAGs using order-ideal DP.
///
/// Two DPs are computed over all 2^n bitmasks:
///
///   dp_fwd[S]  = # of linear extensions of G[S] where S is a valid order ideal
///                (nodes in S can be placed first without violating DAG constraints).
///   dp_ind[T]  = # of linear extensions of the induced subgraph G[T], for any T.
///
/// ASV formula:
///   phi[i] += dp_fwd[S] × dp_ind[V\(S∪{i})] × (v(S∪{i}) − v(S))
///   phi[i] /= dp_ind[V]   (= L(G), the total number of linear extensions)
///
/// Time: O(2^n × n). Space: O(2^n).
/// Practical for n ≤ 20 (~1M states, ~16MB for two dp arrays).
pub(crate) fn dag_exact_asv<F>(
    dag: &Dag,
    value_fn: F,
    parents_mask: &[u64],
) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    let n = dag.node_count();
    if n > 20 {
        return Err(CausasvError::InvalidConfig(format!(
            "exact_dag requires n ≤ 20 (2^n DP states), got {n}"
        )));
    }

    let total_masks = 1usize << n;
    let full_mask = total_masks - 1;

    // dp_fwd[mask]: # of orderings of `mask` consistent with the DAG.
    // Only valid order ideals (downward-closed sets) get nonzero values.
    let mut dp_fwd = vec![0u64; total_masks];
    dp_fwd[0] = 1;
    for mask in 0..total_masks {
        if dp_fwd[mask] == 0 {
            continue;
        }
        let mask64 = mask as u64;
        for (i, &pmask) in parents_mask.iter().enumerate() {
            if mask64 & (1 << i) != 0 {
                continue; // already placed
            }
            if pmask & mask64 == pmask {
                // all parents of i are in mask → i can be placed next
                let next = mask | (1 << i);
                dp_fwd[next] = dp_fwd[next].checked_add(dp_fwd[mask]).ok_or_else(|| {
                    CausasvError::Overflow(format!(
                        "dp_fwd[{mask:#b}] overflowed u64 — use approx for this DAG"
                    ))
                })?;
            }
        }
    }

    // dp_ind[mask]: # of linear extensions of the induced subgraph G[mask], for ANY mask.
    // Recurrence: i is a source in G[mask] iff parents_mask[i] & mask == 0.
    let mut dp_ind = vec![0u64; total_masks];
    dp_ind[0] = 1;
    for mask in 1..total_masks {
        let mask64 = mask as u64;
        for (i, &pmask) in parents_mask.iter().enumerate() {
            if mask64 & (1u64 << i) == 0 {
                continue; // i not in mask
            }
            if pmask & mask64 == 0 {
                // i has no parent in mask → source in G[mask]
                let prev = dp_ind[mask ^ (1usize << i)];
                dp_ind[mask] = dp_ind[mask].checked_add(prev).ok_or_else(|| {
                    CausasvError::Overflow(format!(
                        "dp_ind[{mask:#b}] overflowed u64 — use approx for this DAG"
                    ))
                })?;
            }
        }
    }

    let total = dp_ind[full_mask] as f64; // = L(G)

    // Accumulate ASV weights
    let mut phi = vec![0.0f64; n];
    let mut cache = HashMap::<u64, f64>::new();
    for (mask, &fwd_count) in dp_fwd.iter().enumerate() {
        if fwd_count == 0 {
            continue; // S is not a valid order ideal
        }
        let mask64 = mask as u64;
        let w_prefix = fwd_count as f64;
        for i in 0..n {
            if mask64 & (1u64 << i) != 0 {
                continue; // i already in S
            }
            if parents_mask[i] & mask64 != parents_mask[i] {
                continue; // not all parents of i in S
            }
            let with_i_mask = mask | (1usize << i);
            let suffix_mask = full_mask ^ with_i_mask; // V \ (S ∪ {i})
            let w_suffix = dp_ind[suffix_mask] as f64;
            let mask_with_i = mask64 | (1u64 << i);
            phi[i] += w_prefix
                * w_suffix
                * (value_cached(&mut cache, &value_fn, mask_with_i)?
                    - value_cached(&mut cache, &value_fn, mask64)?);
        }
    }

    let values = (0..n).map(|i| (NodeId(i as u32), phi[i] / total)).collect();

    Ok(AsvResult {
        values,
        n_samples: total_masks,
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
