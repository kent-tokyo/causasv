use std::collections::{BTreeMap, HashMap};

use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::asv::AsvResult;
use crate::cache::value_cached;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::numerics::kahan_add;
use crate::sampler::{
    AdaptiveSamplingConfig, SampledOrdering, SamplerScratch, SamplingConfig, make_rng,
    sample_one_into, sample_uniform_into, worker_seed,
};

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
    let parallel = config.parallel || seed.is_none();

    let base_in_deg = dag.in_degrees();

    let (numerator, denominator, sum_w_sq) = if !parallel {
        // Seeded single-threaded: collect all samples first, then apply per-batch
        // log-weight normalization before exp() for consistency with the batched paths.
        // Self-normalized IS is invariant to a common scale factor, so subtracting
        // max(log_w) does not change ASV values but prevents potential overflow.
        let mut rng = make_rng(seed);
        let mut numerator = vec![0.0f64; n];
        let mut num_comp = vec![0.0f64; n]; // Kahan compensation for numerator[i]
        let mut denominator = 0.0f64;
        let mut denom_comp = 0.0f64;
        let mut sum_w_sq = 0.0f64;
        let mut wsq_comp = 0.0f64;
        let mut cache = HashMap::<u64, f64>::new();
        let mut scratch = SamplerScratch::new(n);
        let samples: Vec<SampledOrdering> = (0..config.n_samples)
            .map(|_| {
                let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
                SampledOrdering {
                    ordering: scratch.ordering.clone(),
                    log_q,
                }
            })
            .collect();
        let max_log_w = samples
            .iter()
            .map(|s| -s.log_q)
            .fold(f64::NEG_INFINITY, f64::max);
        for sample in &samples {
            let w = ((-sample.log_q) - max_log_w).exp(); // log-normalized IS weight
            kahan_add(&mut denominator, &mut denom_comp, w);
            kahan_add(&mut sum_w_sq, &mut wsq_comp, w * w);
            let mut prefix_mask: u64 = 0;
            for &node in &sample.ordering {
                let without = prefix_mask;
                let with_node = prefix_mask | (1u64 << node.0);
                let delta = value_cached(&mut cache, &value_fn, with_node)?
                    - value_cached(&mut cache, &value_fn, without)?;
                kahan_add(
                    &mut numerator[node.0 as usize],
                    &mut num_comp[node.0 as usize],
                    w * delta,
                );
                prefix_mask = with_node;
            }
        }
        (numerator, denominator, sum_w_sq)
    } else if let Some(global_seed) = seed {
        // Seeded parallel: deterministic per-worker seeds via splitmix64.
        // Divide samples across workers so each worker draws a contiguous slice;
        // worker k's seed = worker_seed(global_seed, k) — a bijection, so no two
        // workers share an RNG state.
        let num_threads = config
            .num_threads
            .unwrap_or_else(rayon::current_num_threads);
        let workers: Vec<(usize, u64)> = (0..num_threads)
            .map(|k| {
                let start = (config.n_samples * k) / num_threads;
                let end = (config.n_samples * (k + 1)) / num_threads;
                (end - start, worker_seed(global_seed, k))
            })
            .filter(|(count, _)| *count > 0)
            .collect();

        workers
            .into_par_iter()
            .map(
                |(count, wseed)| -> Result<(Vec<f64>, f64, f64), CausasvError> {
                    let mut rng = make_rng(Some(wseed));
                    let mut cache = HashMap::<u64, f64>::new();
                    let mut scratch = SamplerScratch::new(n);
                    let mut local_num = vec![0.0f64; n];
                    let mut num_c = vec![0.0f64; n];
                    let mut denom = 0.0f64;
                    let mut denom_c = 0.0f64;
                    let mut wsq = 0.0f64;
                    let mut wsq_c = 0.0f64;
                    for _ in 0..count {
                        let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
                        let w = (-log_q).exp();
                        kahan_add(&mut denom, &mut denom_c, w);
                        kahan_add(&mut wsq, &mut wsq_c, w * w);
                        let mut prefix_mask: u64 = 0;
                        for &node in &scratch.ordering {
                            let without = prefix_mask;
                            let with_node = prefix_mask | (1u64 << node.0);
                            let delta = value_cached(&mut cache, &value_fn, with_node)?
                                - value_cached(&mut cache, &value_fn, without)?;
                            kahan_add(
                                &mut local_num[node.0 as usize],
                                &mut num_c[node.0 as usize],
                                w * delta,
                            );
                            prefix_mask = with_node;
                        }
                    }
                    Ok((local_num, denom, wsq))
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
    } else {
        // Unseeded parallel: per-thread state (cache, rng, scratch) created once per thread via
        // try_fold's identity; accumulation is direct into per-thread numerator so no per-sample
        // allocation is needed.
        type UState = (
            HashMap<u64, f64>,
            StdRng,
            SamplerScratch,
            Vec<f64>,
            f64,
            f64,
        );
        let mk_state = || -> UState {
            (
                HashMap::new(),
                make_rng(None),
                SamplerScratch::new(n),
                vec![0.0f64; n],
                0.0f64,
                0.0f64,
            )
        };
        let (_, _, _, acc_num, acc_denom, acc_wsq) = (0..config.n_samples)
            .into_par_iter()
            .try_fold(mk_state, |mut state, _| -> Result<UState, CausasvError> {
                let (cache, rng, scratch, acc_num, acc_denom, acc_wsq) = &mut state;
                let log_q = sample_one_into(dag, rng, scratch, &base_in_deg);
                let w = (-log_q).exp();
                *acc_denom += w;
                *acc_wsq += w * w;
                let mut prefix_mask: u64 = 0;
                for &node in &scratch.ordering {
                    let without = prefix_mask;
                    let with_node = prefix_mask | (1u64 << node.0);
                    acc_num[node.0 as usize] += w
                        * (value_cached(cache, &value_fn, with_node)?
                            - value_cached(cache, &value_fn, without)?);
                    prefix_mask = with_node;
                }
                Ok(state)
            })
            .try_reduce(mk_state, |mut a, b| {
                for (x, y) in a.3.iter_mut().zip(&b.3) {
                    *x += y;
                }
                a.4 += b.4;
                a.5 += b.5;
                Ok(a)
            })?;
        (acc_num, acc_denom, acc_wsq)
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
        fallback_from: None,
        fallback_reason: None,
        method_used: None,
    })
}

/// Batched IS estimator: collects `batch_size` samples per round, deduplicates coalition
/// bitmasks, evaluates them all at once via `value_fn_batch`, then processes IS weights.
///
/// This reduces Python GIL acquisition overhead from O(n_samples × n) calls to
/// O(n_samples / batch_size) calls — a significant speedup for large sklearn/PyTorch models.
///
/// Always single-threaded for reproducibility when seeded.
pub(crate) fn approximate_asv_batched<F>(
    dag: &Dag,
    value_fn_batch: F,
    config: SamplingConfig,
) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError>,
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
    let batch_size = config.batch_size.unwrap_or(256).max(1);
    let seed = config.seed;
    let mut rng = make_rng(seed);
    let mut cache = HashMap::<u64, f64>::new();
    let mut numerator = vec![0.0f64; n];
    let mut denominator = 0.0f64;
    let mut sum_w_sq = 0.0f64;
    let mut remaining = config.n_samples;
    let base_in_deg = dag.in_degrees();
    let mut scratch = SamplerScratch::new(n);
    // Running maximum of log-weights across ALL batches.
    // When the max rises, accumulated sums are rescaled so all samples share the same
    // effective normalization constant — necessary for a valid cross-batch IS estimator.
    let mut global_max_log_w = f64::NEG_INFINITY;

    while remaining > 0 {
        let batch = remaining.min(batch_size);

        // Generate batch samples; scratch reused across iterations to eliminate per-sample alloc.
        let samples: Vec<SampledOrdering> = (0..batch)
            .map(|_| {
                let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
                SampledOrdering {
                    ordering: scratch.ordering.clone(),
                    log_q,
                }
            })
            .collect();

        // Collect unique coalition masks not yet in cache
        let mut uncached: Vec<u64> = Vec::new();
        for s in &samples {
            let mut mask: u64 = 0;
            uncached.push(mask);
            for &node in &s.ordering {
                mask |= 1u64 << node.0;
                uncached.push(mask);
            }
        }
        uncached.sort_unstable();
        uncached.dedup();
        uncached.retain(|m| !cache.contains_key(m));

        // One batch call for all uncached coalitions
        if !uncached.is_empty() {
            let coalitions: Vec<Vec<NodeId>> = uncached
                .iter()
                .map(|&mask| {
                    (0..n)
                        .filter(|&i| mask & (1u64 << i) != 0)
                        .map(|i| NodeId(i as u32))
                        .collect()
                })
                .collect();
            let values = value_fn_batch(&coalitions)?;
            for (&mask, val) in uncached.iter().zip(values.iter()) {
                cache.insert(mask, *val);
            }
        }

        // Update global max and rescale accumulated sums if a new maximum is found.
        let batch_max = samples
            .iter()
            .map(|s| -s.log_q)
            .fold(f64::NEG_INFINITY, f64::max);
        if batch_max > global_max_log_w {
            let scale = (global_max_log_w - batch_max).exp(); // ∈ (0,1]; = 0 on first batch
            for x in numerator.iter_mut() {
                *x *= scale;
            }
            denominator *= scale;
            sum_w_sq *= scale * scale;
            global_max_log_w = batch_max;
        }
        for s in &samples {
            let w = ((-s.log_q) - global_max_log_w).exp();
            denominator += w;
            sum_w_sq += w * w;
            let mut prefix_mask: u64 = 0;
            for &node in &s.ordering {
                let without = *cache.get(&prefix_mask).unwrap();
                let with_node = prefix_mask | (1u64 << node.0);
                let with = *cache.get(&with_node).unwrap();
                numerator[node.0 as usize] += w * (with - without);
                prefix_mask = with_node;
            }
        }

        remaining -= batch;
    }

    let values = (0..n)
        .map(|i| (NodeId(i as u32), numerator[i] / denominator))
        .collect();
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
        fallback_from: None,
        fallback_reason: None,
        method_used: None,
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
    let mut num_comp = vec![0.0f64; n]; // Kahan compensation for numerator
    let mut num_sq = vec![0.0f64; n]; // Σ (w·Δ_i)²
    let mut num_sq_comp = vec![0.0f64; n]; // Kahan compensation for num_sq
    let mut denominator = 0.0f64; // Σ w
    let mut denom_comp = 0.0f64;
    let mut sum_w_sq = 0.0f64; // Σ w²
    let mut wsq_comp = 0.0f64;
    let mut total_samples = 0usize;
    let mut prev_values = vec![f64::NAN; n];
    let mut converged = false;
    let base_in_deg = dag.in_degrees();
    let mut scratch = SamplerScratch::new(n);
    let mut global_max_log_w = f64::NEG_INFINITY;

    while total_samples < config.max_samples {
        let batch = config.batch_size.min(config.max_samples - total_samples);

        let batch_samples: Vec<SampledOrdering> = (0..batch)
            .map(|_| {
                let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
                SampledOrdering {
                    ordering: scratch.ordering.clone(),
                    log_q,
                }
            })
            .collect();

        // Rescale accumulated Kahan sums when a new log-weight maximum is found.
        let batch_max = batch_samples
            .iter()
            .map(|s| -s.log_q)
            .fold(f64::NEG_INFINITY, f64::max);
        if batch_max > global_max_log_w {
            let scale = (global_max_log_w - batch_max).exp();
            let scale_sq = scale * scale;
            for i in 0..n {
                numerator[i] *= scale;
                num_comp[i] *= scale;
                num_sq[i] *= scale_sq;
                num_sq_comp[i] *= scale_sq;
            }
            denominator *= scale;
            denom_comp *= scale;
            sum_w_sq *= scale_sq;
            wsq_comp *= scale_sq;
            global_max_log_w = batch_max;
        }
        for sample in &batch_samples {
            let w = ((-sample.log_q) - global_max_log_w).exp();
            kahan_add(&mut denominator, &mut denom_comp, w);
            kahan_add(&mut sum_w_sq, &mut wsq_comp, w * w);
            let mut prefix_mask: u64 = 0;
            for &node in &sample.ordering {
                let without = prefix_mask;
                let with_node = prefix_mask | (1u64 << node.0);
                let delta = value_cached(&mut cache, &value_fn, with_node)?
                    - value_cached(&mut cache, &value_fn, without)?;
                let wd = w * delta;
                kahan_add(
                    &mut numerator[node.0 as usize],
                    &mut num_comp[node.0 as usize],
                    wd,
                );
                kahan_add(
                    &mut num_sq[node.0 as usize],
                    &mut num_sq_comp[node.0 as usize],
                    wd * wd,
                );
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
        fallback_from: None,
        fallback_reason: None,
        method_used: None,
    })
}

/// Adaptive batched IS estimator: same convergence logic as `approximate_asv_adaptive`
/// but evaluates coalitions in batches via `value_fn_batch` to reduce Python GIL overhead.
///
/// Each sampling batch (of `config.batch_size` samples) becomes one `value_fn_batch` call.
pub(crate) fn approximate_asv_adaptive_batched<F>(
    dag: &Dag,
    value_fn_batch: F,
    config: AdaptiveSamplingConfig,
) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError>,
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
    let mut numerator = vec![0.0f64; n];
    let mut num_comp = vec![0.0f64; n];
    let mut num_sq = vec![0.0f64; n];
    let mut num_sq_comp = vec![0.0f64; n];
    let mut denominator = 0.0f64;
    let mut denom_comp = 0.0f64;
    let mut sum_w_sq = 0.0f64;
    let mut wsq_comp = 0.0f64;
    let mut total_samples = 0usize;
    let mut prev_values = vec![f64::NAN; n];
    let mut converged = false;
    let base_in_deg = dag.in_degrees();
    let mut scratch = SamplerScratch::new(n);
    let mut global_max_log_w = f64::NEG_INFINITY;

    while total_samples < config.max_samples {
        let batch = config.batch_size.min(config.max_samples - total_samples);

        let samples: Vec<SampledOrdering> = (0..batch)
            .map(|_| {
                let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
                SampledOrdering {
                    ordering: scratch.ordering.clone(),
                    log_q,
                }
            })
            .collect();

        // Collect unique uncached coalition masks for this batch
        let mut uncached: Vec<u64> = Vec::new();
        for s in &samples {
            let mut mask: u64 = 0;
            uncached.push(mask);
            for &node in &s.ordering {
                mask |= 1u64 << node.0;
                uncached.push(mask);
            }
        }
        uncached.sort_unstable();
        uncached.dedup();
        uncached.retain(|m| !cache.contains_key(m));

        if !uncached.is_empty() {
            let coalitions: Vec<Vec<NodeId>> = uncached
                .iter()
                .map(|&mask| {
                    (0..n)
                        .filter(|&i| mask & (1u64 << i) != 0)
                        .map(|i| NodeId(i as u32))
                        .collect()
                })
                .collect();
            let values = value_fn_batch(&coalitions)?;
            for (&mask, val) in uncached.iter().zip(values.iter()) {
                cache.insert(mask, *val);
            }
        }

        let batch_max = samples
            .iter()
            .map(|s| -s.log_q)
            .fold(f64::NEG_INFINITY, f64::max);
        if batch_max > global_max_log_w {
            let scale = (global_max_log_w - batch_max).exp();
            let scale_sq = scale * scale;
            for i in 0..n {
                numerator[i] *= scale;
                num_comp[i] *= scale;
                num_sq[i] *= scale_sq;
                num_sq_comp[i] *= scale_sq;
            }
            denominator *= scale;
            denom_comp *= scale;
            sum_w_sq *= scale_sq;
            wsq_comp *= scale_sq;
            global_max_log_w = batch_max;
        }
        for s in &samples {
            let w = ((-s.log_q) - global_max_log_w).exp();
            kahan_add(&mut denominator, &mut denom_comp, w);
            kahan_add(&mut sum_w_sq, &mut wsq_comp, w * w);
            let mut prefix_mask: u64 = 0;
            for &node in &s.ordering {
                let without = *cache.get(&prefix_mask).unwrap();
                let with_node = prefix_mask | (1u64 << node.0);
                let with = *cache.get(&with_node).unwrap();
                let delta = with - without;
                let wd = w * delta;
                kahan_add(
                    &mut numerator[node.0 as usize],
                    &mut num_comp[node.0 as usize],
                    wd,
                );
                kahan_add(
                    &mut num_sq[node.0 as usize],
                    &mut num_sq_comp[node.0 as usize],
                    wd * wd,
                );
                prefix_mask = with_node;
            }
        }

        total_samples += batch;

        if total_samples < config.min_samples {
            continue;
        }

        let ess = denominator * denominator / sum_w_sq;
        if ess / (total_samples as f64) < config.ess_ratio_min {
            continue;
        }

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

        for i in 0..n {
            prev_values[i] = numerator[i] / denominator;
        }

        if max_rel_change < config.rel_tol {
            converged = true;
            break;
        }
    }

    let ess = denominator * denominator / sum_w_sq;
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
        fallback_from: None,
        fallback_reason: None,
        method_used: None,
    })
}

/// Approximate ASV via uniform topological order sampling.
///
/// Each linear extension is sampled with probability 1/L(G) by selecting the next node
/// proportionally to dp_ind[remaining \ {i}]. No IS correction — every sample has equal
/// weight, so ESS = n_samples exactly and there is no importance-weight variance.
///
/// Requires dp_ind precomputed by `compute_dp_ind` (O(2^n × n)). Practical for n ≤ 20.
pub(crate) fn approximate_asv_uniform<F>(
    value_fn: F,
    config: SamplingConfig,
    dp_ind: &[u64],
    parents_mask: &[u64],
) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    let n = parents_mask.len();
    let mut rng = make_rng(config.seed);
    let mut cache = HashMap::<u64, f64>::new();
    let mut numerator = vec![0.0f64; n];
    let mut num_comp = vec![0.0f64; n];
    let mut ordering = Vec::with_capacity(n);

    for _ in 0..config.n_samples {
        sample_uniform_into(&mut rng, dp_ind, parents_mask, &mut ordering);
        let mut prefix_mask: u64 = 0;
        for &node in &ordering {
            let without = prefix_mask;
            let with_node = prefix_mask | (1u64 << node.0);
            let delta = value_cached(&mut cache, &value_fn, with_node)?
                - value_cached(&mut cache, &value_fn, without)?;
            kahan_add(
                &mut numerator[node.0 as usize],
                &mut num_comp[node.0 as usize],
                delta,
            );
            prefix_mask = with_node;
        }
    }

    let n_f = config.n_samples as f64;
    let values = (0..n)
        .map(|i| (NodeId(i as u32), numerator[i] / n_f))
        .collect();

    Ok(AsvResult {
        values,
        n_samples: config.n_samples,
        seed: config.seed,
        is_exact: false,
        effective_sample_size: Some(n_f), // uniform sampling: ESS = n_samples exactly
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
