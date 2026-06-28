use crate::asv::AsvResult;
use crate::cache::mask_to_coalition;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};

/// Compute dp_ind[mask] = # of linear extensions of the induced subgraph G[mask].
/// Recurrence: dp_ind[mask] = Σ_{i: source of G[mask]} dp_ind[mask \ {i}].
/// O(2^n × n). Shared by dag_exact_asv and sample_uniform_into.
pub(crate) fn compute_dp_ind(n: usize, parents_mask: &[u64]) -> Result<Vec<u64>, CausasvError> {
    let total_masks = 1usize << n;
    let mut dp_ind = vec![0u64; total_masks];
    dp_ind[0] = 1;
    for mask in 1..total_masks {
        let mask64 = mask as u64;
        let mut bits = mask64;
        while bits != 0 {
            let bit = bits & bits.wrapping_neg();
            let i = bit.trailing_zeros() as usize;
            bits ^= bit;
            if parents_mask[i] & mask64 == 0 {
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
    Ok(dp_ind)
}

// ponytail: Vec<f64>[mask] beats HashMap here — n≤20 is enforced above, so 2^n ≤ 1M entries (8MB).
// NaN sentinel means uncached; value_fn is assumed to return finite values (same assumption as HashMap path).
fn value_cached_dense<F>(cache: &mut [f64], value_fn: &F, mask: u64) -> Result<f64, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    let v = cache[mask as usize];
    if !v.is_nan() {
        return Ok(v);
    }
    let coalition = mask_to_coalition(mask);
    let v = value_fn(&coalition)?;
    cache[mask as usize] = v;
    Ok(v)
}

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

    let full64 = full_mask as u64;

    // dp_fwd[mask]: # of orderings of `mask` consistent with the DAG.
    // Only valid order ideals (downward-closed sets) get nonzero values.
    // Inner loop iterates only over nodes NOT in mask (bit iteration over complement).
    let mut dp_fwd = vec![0u64; total_masks];
    dp_fwd[0] = 1;
    for mask in 0..total_masks {
        if dp_fwd[mask] == 0 {
            continue;
        }
        let mask64 = mask as u64;
        let mut not_placed = (!mask64) & full64;
        while not_placed != 0 {
            let bit = not_placed & not_placed.wrapping_neg();
            let i = bit.trailing_zeros() as usize;
            not_placed ^= bit;
            if parents_mask[i] & mask64 == parents_mask[i] {
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
    // Inner loop iterates only over nodes IN mask (bit iteration over mask itself).
    let dp_ind = compute_dp_ind(n, parents_mask)?;

    let total = dp_ind[full_mask] as f64; // = L(G)

    // Accumulate ASV weights.
    // v(S) is read once per valid order ideal S (hoisted out of the candidate-node loop).
    // Candidate nodes enumerated via bit iteration over complement of mask.
    let mut phi = vec![0.0f64; n];
    let mut cache = vec![f64::NAN; total_masks];
    for (mask, &fwd_count) in dp_fwd.iter().enumerate() {
        if fwd_count == 0 {
            continue; // S is not a valid order ideal
        }
        let mask64 = mask as u64;
        let w_prefix = fwd_count as f64;
        let v_s = value_cached_dense(&mut cache, &value_fn, mask64)?;
        let mut not_placed = (!mask64) & full64;
        while not_placed != 0 {
            let bit = not_placed & not_placed.wrapping_neg();
            let i = bit.trailing_zeros() as usize;
            not_placed ^= bit;
            if parents_mask[i] & mask64 != parents_mask[i] {
                continue; // not all parents of i in S
            }
            let with_i_mask = mask | (1usize << i);
            let suffix_mask = full_mask ^ with_i_mask; // V \ (S ∪ {i})
            let w_suffix = dp_ind[suffix_mask] as f64;
            let mask_with_i = mask64 | (1u64 << i);
            phi[i] += w_prefix
                * w_suffix
                * (value_cached_dense(&mut cache, &value_fn, mask_with_i)? - v_s);
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
        method_used: None,
    })
}
