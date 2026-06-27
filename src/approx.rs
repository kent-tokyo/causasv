use std::collections::{BTreeMap, HashMap};

use rayon::prelude::*;

use crate::asv::AsvResult;
use crate::cache::value_cached;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::sampler::{AdaptiveSamplingConfig, SamplingConfig, make_rng, sample_one};

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
                    for (a, b) in acc.0.iter_mut().zip(&local_num) {
                        *a += b;
                    }
                    acc.1 += w;
                    acc.2 += wsq;
                    Ok(acc)
                },
            )
            .try_reduce(
                || (vec![0.0f64; n], 0.0f64, 0.0f64),
                |mut a, b| {
                    for (x, y) in a.0.iter_mut().zip(&b.0) {
                        *x += y;
                    }
                    Ok((a.0, a.1 + b.1, a.2 + b.2))
                },
            )?
    };

    let values = (0..n)
        .map(|i| (NodeId(i as u32), numerator[i] / denominator))
        .collect();

    // ESS = (Σw)² / Σw²
    let ess = denominator * denominator / sum_w_sq;

    Ok(AsvResult {
        values,
        n_samples: config.n_samples,
        seed,
        is_exact: false,
        effective_sample_size: Some(ess),
        converged: None,
        stderr: None,
        n_order_ideals: None,
        state_ratio: None,
        memory_mb: None,
    })
}

/// Adaptive IS estimator: runs batches until convergence or max_samples.
///
/// Tracks per-node Σ(w·Δ) and Σ(w·Δ)² to compute IS standard error estimates.
/// Convergence: max relative change < rel_tol AND ess_ratio ≥ ess_ratio_min,
/// after at least min_samples have been drawn.
///
/// ponytail: single-threaded; per-batch parallel if throughput matters
pub(crate) fn approximate_asv_adaptive<F>(
    dag: &Dag,
    value_fn: F,
    config: AdaptiveSamplingConfig,
) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    if config.batch_size == 0 {
        return Err(CausasvError::InvalidConfig(
            "batch_size must be > 0".to_string(),
        ));
    }
    if config.min_samples > config.max_samples {
        return Err(CausasvError::InvalidConfig(
            "min_samples must be ≤ max_samples".to_string(),
        ));
    }
    let n = dag.node_count();
    if n > 64 {
        return Err(CausasvError::InvalidConfig(format!(
            "bitmask coalitions require n ≤ 64, got {n}"
        )));
    }

    let mut rng = make_rng(config.seed);
    let mut cache = HashMap::<u64, f64>::new();

    let mut numerator = vec![0.0f64; n]; // Σ w·Δ_i
    let mut num_sq = vec![0.0f64; n]; // Σ (w·Δ_i)²
    let mut denominator = 0.0f64; // Σ w
    let mut sum_w_sq = 0.0f64; // Σ w²
    let mut total_samples = 0usize;
    let mut prev_values = vec![f64::NAN; n];
    let mut converged = false;

    while total_samples < config.max_samples {
        let batch = config.batch_size.min(config.max_samples - total_samples);

        for _ in 0..batch {
            let sample = sample_one(dag, &mut rng);
            let w = (-sample.log_q).exp();
            denominator += w;
            sum_w_sq += w * w;
            let mut prefix_mask: u64 = 0;
            for &node in &sample.ordering {
                let without = prefix_mask;
                let with_node = prefix_mask | (1u64 << node.0);
                let delta = value_cached(&mut cache, &value_fn, with_node)?
                    - value_cached(&mut cache, &value_fn, without)?;
                let wd = w * delta;
                numerator[node.0 as usize] += wd;
                num_sq[node.0 as usize] += wd * wd;
                prefix_mask = with_node;
            }
        }
        total_samples += batch;

        if total_samples < config.min_samples {
            continue;
        }

        let ess = denominator * denominator / sum_w_sq;
        let ess_ratio = ess / total_samples as f64;
        if ess_ratio < config.ess_ratio_min {
            continue;
        }

        // Check relative change in per-node estimates
        let max_rel_change = (0..n)
            .map(|i| {
                let cur = numerator[i] / denominator;
                let prev = prev_values[i];
                if prev.is_nan() {
                    f64::INFINITY
                } else {
                    (cur - prev).abs() / (prev.abs() + 1e-10)
                }
            })
            .fold(0.0f64, f64::max);

        // Update prev for next batch
        for i in 0..n {
            prev_values[i] = numerator[i] / denominator;
        }

        if max_rel_change < config.rel_tol {
            converged = true;
            break;
        }
    }

    let ess = denominator * denominator / sum_w_sq;

    // IS stderr: sqrt( (Σ(wΔ)²/n - (ΣwΔ)²/n²) / denom² )
    let stderr: BTreeMap<NodeId, f64> = (0..n)
        .map(|i| {
            let n_f = total_samples as f64;
            let mean_num_sq = num_sq[i] / n_f;
            let mean_num = numerator[i] / n_f;
            let var_num = (mean_num_sq - mean_num * mean_num).max(0.0);
            let se = (var_num / n_f).sqrt() / (denominator / n_f);
            (NodeId(i as u32), se)
        })
        .collect();

    let values = (0..n)
        .map(|i| (NodeId(i as u32), numerator[i] / denominator))
        .collect();

    Ok(AsvResult {
        values,
        n_samples: total_samples,
        seed: config.seed,
        is_exact: false,
        effective_sample_size: Some(ess),
        converged: Some(converged),
        stderr: Some(stderr),
        n_order_ideals: None,
        state_ratio: None,
        memory_mb: None,
    })
}
