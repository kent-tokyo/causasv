use std::collections::HashMap;

use rayon::prelude::*;

use crate::asv::AsvResult;
use crate::cache::value_cached;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::sampler::{make_rng, sample_one, SamplingConfig};

/// Self-normalized importance sampling estimator for ASV.
///
/// The frontier sampler assigns unequal probabilities to orderings, so naive averaging is biased.
/// IS correction: weight each sample by 1/q(π), then self-normalize. This converges to the
/// uniform-over-orderings average without requiring uniform sampling.
pub(crate) fn approximate_asv<F>(
    dag: &Dag,
    value_fn: F,
    config: SamplingConfig,
) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError> + Send + Sync,
{
    if config.n_samples == 0 {
        return Err(CausasvError::InvalidConfig(
            "n_samples must be > 0".to_string(),
        ));
    }
    let n = dag.node_count();
    if n > 64 {
        return Err(CausasvError::InvalidConfig(format!(
            "bitmask coalitions require n ≤ 64, got {n}"
        )));
    }
    let seed = config.seed;

    let (numerator, denominator, sum_w_sq) = if seed.is_some() {
        // Seeded: single-threaded for exact reproducibility
        let mut rng = make_rng(seed);
        let mut numerator = vec![0.0f64; n];
        let mut denominator = 0.0f64;
        let mut sum_w_sq = 0.0f64;
        let mut cache = HashMap::<u64, f64>::new();
        for _ in 0..config.n_samples {
            let sample = sample_one(dag, &mut rng);
            let w = (-sample.log_q).exp(); // IS weight = 1/q(π)
            denominator += w;
            sum_w_sq += w * w;
            let mut prefix_mask: u64 = 0;
            for &node in &sample.ordering {
                let without = prefix_mask;
                let with_node = prefix_mask | (1u64 << node.0);
                numerator[node.0 as usize] += w
                    * (value_cached(&mut cache, &value_fn, with_node)?
                        - value_cached(&mut cache, &value_fn, without)?);
                prefix_mask = with_node;
            }
        }
        (numerator, denominator, sum_w_sq)
    } else {
        // Unseeded: Rayon parallel, per-thread RNG + cache
        (0..config.n_samples)
            .into_par_iter()
            .map_init(
                || (HashMap::<u64, f64>::new(), make_rng(None)),
                |(cache, rng), _| -> Result<(Vec<f64>, f64, f64), CausasvError> {
                    let sample = sample_one(dag, rng);
                    let w = (-sample.log_q).exp();
                    let mut local_num = vec![0.0f64; n];
                    let mut prefix_mask: u64 = 0;
                    for &node in &sample.ordering {
                        let without = prefix_mask;
                        let with_node = prefix_mask | (1u64 << node.0);
                        local_num[node.0 as usize] += w
                            * (value_cached(cache, &value_fn, with_node)?
                                - value_cached(cache, &value_fn, without)?);
                        prefix_mask = with_node;
                    }
                    Ok((local_num, w, w * w))
                },
            )
            .try_fold(
                || (vec![0.0f64; n], 0.0f64, 0.0f64),
                |mut acc, item| {
                    let (local_num, w, wsq) = item?;
                    for i in 0..n {
                        acc.0[i] += local_num[i];
                    }
                    acc.1 += w;
                    acc.2 += wsq;
                    Ok(acc)
                },
            )
            .try_reduce(
                || (vec![0.0f64; n], 0.0f64, 0.0f64),
                |mut a, b| {
                    for i in 0..n {
                        a.0[i] += b.0[i];
                    }
                    Ok((a.0, a.1 + b.1, a.2 + b.2))
                },
            )?
    };

    let values = (0..n)
        .map(|i| (NodeId(i as u32), numerator[i] / denominator))
        .collect();

    // ESS = (Σw)² / Σw²: estimates effective number of samples given IS weight variance.
    let ess = denominator * denominator / sum_w_sq;

    Ok(AsvResult {
        values,
        n_samples: config.n_samples,
        seed,
        is_exact: false,
        effective_sample_size: Some(ess),
    })
}
