//! Approximate ASV for DAGs with more than 64 nodes.
//!
//! `crate::sampler::sample_one_into` — the frontier sampler shared by every
//! approximate path — never touches a bitmask; it only needs `Vec`-sized
//! scratch space, so it already works for any `n`. The sole reason
//! `approximate_asv`/`approximate_asv_adaptive`/`approximate_asv_batched`/
//! `approximate_asv_adaptive_batched` in `approx.rs` rejected `n > 64` is that
//! each represents a sample's growing prefix coalition as a `u64` mask. This
//! module provides the same four estimators with that one piece swapped for
//! `LargeCoalition` (`crate::coalition`) and a bounded `LargeCoalitionCache` in
//! place of the unbounded `HashMap<u64, f64>`.
//!
//! Every other piece — self-normalized IS, incremental log-weight rescaling,
//! Kahan summation, the ESS/stderr formulas, per-worker seeding — is copied
//! from `approx.rs`'s small-n path rather than shared through a generic
//! abstraction over both, and each of the three `approximate_asv` sub-paths
//! (seeded serial / seeded parallel / unseeded parallel) intentionally keeps
//! its *own* distinct numerics (see each function's doc comment) rather than
//! being normalized to a common shape. Mirroring instead of merging keeps the
//! n ≤ 64 hot path completely unchanged and keeps each path's behavior
//! independently auditable against its small-n counterpart (see the
//! backend-parity tests at the bottom of this file).

use std::collections::{BTreeMap, HashMap};

use rand::rngs::StdRng;
use rayon::prelude::*;

use crate::asv::AsvResult;
use crate::coalition::{LargeCoalition, words_to_sorted_nodes};
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::numerics::kahan_add;
use crate::sampler::{
    AdaptiveSamplingConfig, SampledOrdering, SamplerScratch, SamplingConfig, make_rng,
    sample_one_into, worker_seed,
};

/// Cap on cached coalition→value entries per `LargeCoalitionCache` instance.
///
/// The n ≤ 64 path's `HashMap<u64, f64>` cache is implicitly bounded (at most
/// `2^n` possible keys, usually far fewer in practice). A `Box<[u64]>`-keyed
/// cache over a coalition space with no such small ceiling has no natural
/// bound: for large, sparsely-branching DAGs, distinct IS samples rarely
/// revisit the same prefix past the first couple of steps, so letting the
/// cache grow forever would add ~n entries per sample at a vanishing hit
/// rate — memory cost would dominate any lookup savings. This constant is a
/// per-cache-instance cap, not an aggregate one: seeded-parallel builds one
/// `LargeCoalitionCache` per worker thread, and unseeded-parallel builds one
/// per Rayon fold split (not guaranteed to equal the thread count), so
/// aggregate memory across a parallel run is roughly this cap × however many
/// instances exist, not this cap alone. The batched paths
/// (`approximate_asv_batched_large`/`approximate_asv_adaptive_batched_large`)
/// are single-threaded and build exactly one instance for the whole call,
/// shared across every sampling round. Chosen so a handful of instances stay
/// in the tens-of-MB range in aggregate (entries × instances × ~100
/// bytes/entry); see docs/benchmarks.md for measured hit rates by shape.
const DEFAULT_LARGE_CACHE_MAX_ENTRIES: usize = 50_000;

/// Cap on distinct coalitions resolved by one `value_fn_batch` call in the
/// batched large-DAG paths. A sampling batch of `batch_size` orderings over
/// `n` nodes can name up to `batch_size * (n + 1)` prefix coalitions before
/// dedup; once the deduplicated count exceeds this cap, the batched paths
/// split it across multiple `value_fn_batch` calls (chunking changes only how
/// many calls are made, never the estimate) instead of handing an unbounded
/// list to the caller in one shot.
const MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH: usize = 4_096;

/// Bounded coalition-value cache for `n > 64` DAGs.
///
/// Once `max_entries` is reached, lookups keep working (existing entries stay
/// valid) but new values are simply recomputed by calling `value_fn` again
/// rather than being memoized — correctness never depends on whether a given
/// insert actually lands, only performance does.
pub(crate) struct LargeCoalitionCache {
    values: HashMap<Box<[u64]>, f64>,
    max_entries: usize,
}

impl LargeCoalitionCache {
    pub(crate) fn new(max_entries: usize) -> Self {
        Self {
            values: HashMap::new(),
            max_entries,
        }
    }

    fn get(&self, words: &[u64]) -> Option<f64> {
        self.values.get(words).copied()
    }

    fn insert(&mut self, key: Box<[u64]>, value: f64) {
        if self.values.len() < self.max_entries {
            self.values.insert(key, value);
        }
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }
}

fn value_cached_large<F>(
    cache: &mut LargeCoalitionCache,
    value_fn: &F,
    coalition: &LargeCoalition,
) -> Result<f64, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    if let Some(v) = cache.get(coalition.words()) {
        return Ok(v);
    }
    let nodes = coalition.to_sorted_nodes();
    let v = value_fn(&nodes)?;
    cache.insert(coalition.snapshot_key(), v);
    Ok(v)
}

/// Resolve every unique coalition sampled this round against `cache`,
/// calling `value_fn_batch` only for coalitions the cache hasn't seen yet
/// (chunked so no single call receives more than
/// `MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH` coalitions). `cache` is owned by
/// the caller and shared across every round of the run — on shapes with
/// structural repetition across rounds (e.g. a chain, whose samples all
/// share the same n+1 prefixes every round), this collapses `value_fn_batch`
/// traffic from one call per round to one call total, since every round
/// after the first is satisfied entirely from the cache. The per-round dedup
/// (`sort_unstable` + `dedup`) still runs first so a round with repeated
/// coalitions across its own samples never queries the cache twice for the
/// same key.
fn resolve_batch_unique_coalitions<F>(
    n: usize,
    samples: &[SampledOrdering],
    value_fn_batch: &F,
    cache: &mut LargeCoalitionCache,
) -> Result<HashMap<Box<[u64]>, f64>, CausasvError>
where
    F: Fn(&[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError>,
{
    let mut unique_keys: Vec<Box<[u64]>> = Vec::new();
    let mut coalition = LargeCoalition::empty(n);
    for s in samples {
        coalition.clear();
        unique_keys.push(coalition.snapshot_key());
        for &node in &s.ordering {
            coalition.insert(node);
            unique_keys.push(coalition.snapshot_key());
        }
    }
    unique_keys.sort_unstable();
    unique_keys.dedup();

    let mut value_cache: HashMap<Box<[u64]>, f64> = HashMap::with_capacity(unique_keys.len());
    let mut misses: Vec<Box<[u64]>> = Vec::new();
    for key in unique_keys {
        match cache.get(&key) {
            Some(v) => {
                value_cache.insert(key, v);
            }
            None => misses.push(key),
        }
    }

    for chunk in misses.chunks(MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH) {
        let coalitions: Vec<Vec<NodeId>> = chunk.iter().map(|k| words_to_sorted_nodes(k)).collect();
        let values = value_fn_batch(&coalitions)?;
        if values.len() != chunk.len() {
            return Err(CausasvError::ValueFunctionError(format!(
                "value_fn_batch returned {} values for {} coalitions",
                values.len(),
                chunk.len()
            )));
        }
        for (key, val) in chunk.iter().zip(values) {
            value_cache.insert(key.clone(), val);
            cache.insert(key.clone(), val);
        }
    }
    Ok(value_cache)
}

/// Self-normalized importance sampling estimator for ASV, for `n > 64` DAGs.
/// See `crate::approx::approximate_asv` for the estimator's correctness
/// argument — this is the same estimator over `LargeCoalition` instead of
/// `u64` prefix masks.
pub(crate) fn approximate_asv_large<F>(
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
    let seed = config.seed;
    let parallel = config.parallel || seed.is_none();
    let base_in_deg = dag.in_degrees();

    let (numerator, denominator, sum_w_sq) = if !parallel {
        // Seeded single-threaded: incremental log-weight rescaling + Kahan,
        // identical numerics to approx.rs's serial-seeded branch.
        let mut rng = make_rng(seed);
        let mut numerator = vec![0.0f64; n];
        let mut num_comp = vec![0.0f64; n];
        let mut denominator = 0.0f64;
        let mut denom_comp = 0.0f64;
        let mut sum_w_sq = 0.0f64;
        let mut wsq_comp = 0.0f64;
        let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);
        let mut scratch = SamplerScratch::new(n);
        let mut coalition = LargeCoalition::empty(n);
        let mut global_max_log_w = f64::NEG_INFINITY;
        for _ in 0..config.n_samples {
            let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
            let log_w = -log_q;
            if log_w > global_max_log_w {
                let scale = (global_max_log_w - log_w).exp();
                let scale_sq = scale * scale;
                for i in 0..n {
                    numerator[i] *= scale;
                    num_comp[i] *= scale;
                }
                denominator *= scale;
                denom_comp *= scale;
                sum_w_sq *= scale_sq;
                wsq_comp *= scale_sq;
                global_max_log_w = log_w;
            }
            let w = (log_w - global_max_log_w).exp();
            kahan_add(&mut denominator, &mut denom_comp, w);
            kahan_add(&mut sum_w_sq, &mut wsq_comp, w * w);
            coalition.clear();
            let mut prev_value = value_cached_large(&mut cache, &value_fn, &coalition)?;
            for &node in &scratch.ordering {
                coalition.insert(node);
                let with_value = value_cached_large(&mut cache, &value_fn, &coalition)?;
                let delta = with_value - prev_value;
                kahan_add(
                    &mut numerator[node.0 as usize],
                    &mut num_comp[node.0 as usize],
                    w * delta,
                );
                prev_value = with_value;
            }
        }
        (numerator, denominator, sum_w_sq)
    } else if let Some(global_seed) = seed {
        // Seeded parallel: Kahan, but no cross-worker log-weight rescaling —
        // identical numerics to approx.rs's seeded-parallel branch (each
        // worker's raw `w = exp(-log_q)` is summed directly; no worker ever
        // sees another's log-weights, so there is nothing to rescale against).
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
                    let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);
                    let mut scratch = SamplerScratch::new(n);
                    let mut coalition = LargeCoalition::empty(n);
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
                        coalition.clear();
                        let mut prev_value = value_cached_large(&mut cache, &value_fn, &coalition)?;
                        for &node in &scratch.ordering {
                            coalition.insert(node);
                            let with_value = value_cached_large(&mut cache, &value_fn, &coalition)?;
                            let delta = with_value - prev_value;
                            kahan_add(
                                &mut local_num[node.0 as usize],
                                &mut num_c[node.0 as usize],
                                w * delta,
                            );
                            prev_value = with_value;
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
        // Unseeded parallel: no Kahan, no rescaling — identical numerics to
        // approx.rs's unseeded-parallel branch.
        type UState = (
            LargeCoalitionCache,
            StdRng,
            SamplerScratch,
            LargeCoalition,
            Vec<f64>,
            f64,
            f64,
        );
        let mk_state = || -> UState {
            (
                LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES),
                make_rng(None),
                SamplerScratch::new(n),
                LargeCoalition::empty(n),
                vec![0.0f64; n],
                0.0f64,
                0.0f64,
            )
        };
        let (_, _, _, _, acc_num, acc_denom, acc_wsq) = (0..config.n_samples)
            .into_par_iter()
            .try_fold(mk_state, |mut state, _| -> Result<UState, CausasvError> {
                let (cache, rng, scratch, coalition, acc_num, acc_denom, acc_wsq) = &mut state;
                let log_q = sample_one_into(dag, rng, scratch, &base_in_deg);
                let w = (-log_q).exp();
                *acc_denom += w;
                *acc_wsq += w * w;
                coalition.clear();
                let mut prev_value = value_cached_large(cache, &value_fn, coalition)?;
                for &node in &scratch.ordering {
                    coalition.insert(node);
                    let with_value = value_cached_large(cache, &value_fn, coalition)?;
                    acc_num[node.0 as usize] += w * (with_value - prev_value);
                    prev_value = with_value;
                }
                Ok(state)
            })
            .try_reduce(mk_state, |mut a, b| {
                for (x, y) in a.4.iter_mut().zip(&b.4) {
                    *x += y;
                }
                a.5 += b.5;
                a.6 += b.6;
                Ok(a)
            })?;
        (acc_num, acc_denom, acc_wsq)
    };

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

/// Adaptive IS estimator for `n > 64` DAGs. See
/// `crate::approx::approximate_asv_adaptive` — same convergence logic and
/// numerics (log-weight rescaling + Kahan, single-threaded), over
/// `LargeCoalition` instead of `u64` prefix masks.
pub(crate) fn approximate_asv_adaptive_large<F>(
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

    let mut rng = make_rng(config.seed);
    let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);

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
    let mut coalition = LargeCoalition::empty(n);
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
            coalition.clear();
            let mut prev_value = value_cached_large(&mut cache, &value_fn, &coalition)?;
            for &node in &sample.ordering {
                coalition.insert(node);
                let with_value = value_cached_large(&mut cache, &value_fn, &coalition)?;
                let delta = with_value - prev_value;
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
                prev_value = with_value;
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

/// Batched IS estimator for `n > 64` DAGs. See
/// `crate::approx::approximate_asv_batched` for the small-n counterpart
/// (rescaling, no Kahan, single-threaded). The coalition→value cache
/// (`LargeCoalitionCache`) is created once and shared across every sampling
/// round — see `resolve_batch_unique_coalitions`.
pub(crate) fn approximate_asv_batched_large<F>(
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
    let batch_size = config.batch_size.unwrap_or(256).max(1);
    let seed = config.seed;
    let mut rng = make_rng(seed);
    let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);
    let mut numerator = vec![0.0f64; n];
    let mut denominator = 0.0f64;
    let mut sum_w_sq = 0.0f64;
    let mut remaining = config.n_samples;
    let base_in_deg = dag.in_degrees();
    let mut scratch = SamplerScratch::new(n);
    let mut coalition = LargeCoalition::empty(n);
    let mut global_max_log_w = f64::NEG_INFINITY;

    while remaining > 0 {
        let batch = remaining.min(batch_size);

        let samples: Vec<SampledOrdering> = (0..batch)
            .map(|_| {
                let log_q = sample_one_into(dag, &mut rng, &mut scratch, &base_in_deg);
                SampledOrdering {
                    ordering: scratch.ordering.clone(),
                    log_q,
                }
            })
            .collect();

        let value_cache =
            resolve_batch_unique_coalitions(n, &samples, &value_fn_batch, &mut cache)?;

        let batch_max = samples
            .iter()
            .map(|s| -s.log_q)
            .fold(f64::NEG_INFINITY, f64::max);
        if batch_max > global_max_log_w {
            let scale = (global_max_log_w - batch_max).exp();
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
            coalition.clear();
            let mut prev_value = *value_cache.get(coalition.words()).unwrap();
            for &node in &s.ordering {
                coalition.insert(node);
                let with_value = *value_cache.get(coalition.words()).unwrap();
                numerator[node.0 as usize] += w * (with_value - prev_value);
                prev_value = with_value;
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

/// Adaptive batched IS estimator for `n > 64` DAGs. See
/// `crate::approx::approximate_asv_adaptive_batched` for the small-n
/// counterpart (rescaling + Kahan, single-threaded). As in
/// `approximate_asv_batched_large`, the coalition cache is created once and
/// shared across the whole run.
pub(crate) fn approximate_asv_adaptive_batched_large<F>(
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

    let mut rng = make_rng(config.seed);
    let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);
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
    let mut coalition = LargeCoalition::empty(n);
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

        let value_cache =
            resolve_batch_unique_coalitions(n, &samples, &value_fn_batch, &mut cache)?;

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
            coalition.clear();
            let mut prev_value = *value_cache.get(coalition.words()).unwrap();
            for &node in &s.ordering {
                coalition.insert(node);
                let with_value = *value_cache.get(coalition.words()).unwrap();
                let delta = with_value - prev_value;
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
                prev_value = with_value;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::approx::{
        approximate_asv, approximate_asv_adaptive, approximate_asv_adaptive_batched,
        approximate_asv_batched,
    };
    use proptest::prelude::*;

    fn make_chain(n: usize) -> Dag {
        let mut dag = Dag::new();
        let ns: Vec<_> = (0..n).map(|i| dag.add_node(&format!("n{i}"))).collect();
        for i in 0..n - 1 {
            dag.add_edge(ns[i], ns[i + 1]).unwrap();
        }
        dag
    }

    fn make_diamond() -> Dag {
        let mut dag = Dag::new();
        let src = dag.add_node("src");
        let m0 = dag.add_node("m0");
        let m1 = dag.add_node("m1");
        let snk = dag.add_node("snk");
        dag.add_edge(src, m0).unwrap();
        dag.add_edge(src, m1).unwrap();
        dag.add_edge(m0, snk).unwrap();
        dag.add_edge(m1, snk).unwrap();
        dag
    }

    fn weighted(s: &[NodeId]) -> Result<f64, CausasvError> {
        Ok(s.iter().map(|n| (n.0 + 1) as f64).sum())
    }

    fn weighted_batch(coalitions: &[Vec<NodeId>]) -> Result<Vec<f64>, CausasvError> {
        Ok(coalitions
            .iter()
            .map(|c| c.iter().map(|n| (n.0 + 1) as f64).sum())
            .collect())
    }

    /// Backend parity (Phase 7D): on the SAME n ≤ 64 DAG, same seed, same
    /// sampling order, the small (u64) and large (word-vec) backends must
    /// agree bitwise for the serial-seeded path — both call `sample_one_into`
    /// identically, and neither backend evicts a cache entry once written
    /// (the large cache only *declines new inserts* past its cap, it never
    /// evicts), so the sequence of values fed into each Kahan accumulator is
    /// provably identical. This can only run as an internal (`#[cfg(test)]`)
    /// test, not `tests/*.rs`, since it needs `pub(crate)` access to force the
    /// large backend onto a small DAG — production dispatch never does this.
    #[test]
    fn backend_parity_serial_seeded_diamond() {
        let dag = make_diamond();
        let small = approximate_asv(&dag, weighted, SamplingConfig::new(500).with_seed(7)).unwrap();
        let large =
            approximate_asv_large(&dag, weighted, SamplingConfig::new(500).with_seed(7)).unwrap();
        for (&node, &v_small) in &small.values {
            let v_large = large.values[&node];
            assert_eq!(
                v_small.to_bits(),
                v_large.to_bits(),
                "node {node:?}: small={v_small}, large={v_large} (expected bitwise parity)"
            );
        }
        assert_eq!(
            small.effective_sample_size.unwrap().to_bits(),
            large.effective_sample_size.unwrap().to_bits()
        );
    }

    #[test]
    fn backend_parity_adaptive_serial_diamond() {
        let dag = make_diamond();
        let config = AdaptiveSamplingConfig {
            min_samples: 200,
            max_samples: 2_000,
            seed: Some(3),
            ..AdaptiveSamplingConfig::default()
        };
        let small = approximate_asv_adaptive(&dag, weighted, config).unwrap();
        let large = approximate_asv_adaptive_large(&dag, weighted, config).unwrap();
        assert_eq!(small.n_samples, large.n_samples);
        for (&node, &v_small) in &small.values {
            assert_eq!(v_small.to_bits(), large.values[&node].to_bits());
        }
    }

    /// Backend parity for the batched paths. This matters more than it might
    /// look: `approximate_asv_batched`/`approximate_asv_adaptive_batched` are
    /// the only large-DAG functions with genuinely new machinery beyond a
    /// coalition-representation swap (snapshot → sort → dedup → chunk →
    /// `words_to_sorted_nodes` round-trip → key lookup, all absent from the
    /// serial paths above) — a bug in the key round-trip or the dedup would
    /// not be caught by `backend_parity_serial_seeded_diamond` alone. Both
    /// backends draw the identical sample sequence for a given seed/n_samples/
    /// batch_size and call a deterministic value function, so despite the
    /// large path's round-scoped (not persistent) cache, bitwise parity still
    /// holds: caching only affects how many times a value is *computed*, never
    /// what it computes to.
    #[test]
    fn backend_parity_batched_diamond() {
        let dag = make_diamond();
        let cfg = || SamplingConfig::new(500).with_seed(7).with_batch_size(64);
        let small = approximate_asv_batched(&dag, weighted_batch, cfg()).unwrap();
        let large = approximate_asv_batched_large(&dag, weighted_batch, cfg()).unwrap();
        for (&node, &v_small) in &small.values {
            let v_large = large.values[&node];
            assert_eq!(
                v_small.to_bits(),
                v_large.to_bits(),
                "node {node:?}: small={v_small}, large={v_large} (expected bitwise parity)"
            );
        }
    }

    #[test]
    fn backend_parity_adaptive_batched_diamond() {
        let dag = make_diamond();
        let config = AdaptiveSamplingConfig {
            min_samples: 200,
            max_samples: 2_000,
            batch_size: 64,
            seed: Some(3),
            ..AdaptiveSamplingConfig::default()
        };
        let small = approximate_asv_adaptive_batched(&dag, weighted_batch, config).unwrap();
        let large = approximate_asv_adaptive_batched_large(&dag, weighted_batch, config).unwrap();
        assert_eq!(small.n_samples, large.n_samples);
        for (&node, &v_small) in &small.values {
            assert_eq!(v_small.to_bits(), large.values[&node].to_bits());
        }
    }

    /// Phase 5 requirement: "chunk分割は推定結果へ影響しない" (chunking must not
    /// affect the estimate). None of the other batched tests exercise the
    /// chunking branch at all — a chain's unique-per-round count is only n+1,
    /// far under `MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH` (4096), so
    /// `unique_keys.chunks(...)` always yields exactly one chunk there. A
    /// 70-node antichain (no structural sharing between samples at all) with
    /// enough samples in one round produces well over 4096 distinct prefixes,
    /// forcing multiple `value_fn_batch` calls — this asserts that actually
    /// happens (not just that the cap constant exists) and that the additive
    /// closed-form answer (ASV_i = 1.0 exactly, independent of DAG shape) still
    /// comes out right despite chunking.
    #[test]
    fn batched_chunking_does_not_change_additive_result() {
        let mut dag = Dag::new();
        for i in 0..70usize {
            dag.add_node(&format!("n{i}"));
        }
        let call_stats = std::cell::RefCell::new((0usize, 0usize)); // (num_calls, max_chunk_len)
        let counting_batch = |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
            let mut stats = call_stats.borrow_mut();
            stats.0 += 1;
            stats.1 = stats.1.max(coalitions.len());
            assert!(
                coalitions.len() <= MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH,
                "chunk exceeded cap: {} > {}",
                coalitions.len(),
                MAX_UNIQUE_COALITIONS_PER_LARGE_BATCH
            );
            Ok(coalitions.iter().map(|c| c.len() as f64).collect())
        };
        let result = approximate_asv_batched_large(
            &dag,
            counting_batch,
            SamplingConfig::new(300).with_seed(11).with_batch_size(300),
        )
        .unwrap();
        let (num_calls, _max_chunk_len) = *call_stats.borrow();
        assert!(
            num_calls >= 2,
            "expected >=4096 unique coalitions to force >=2 value_fn_batch calls, got {num_calls} call(s) \
             — the antichain(70)/300-sample setup should produce far more than 4096 unique prefixes"
        );
        for (&node, &v) in &result.values {
            assert!(
                (v - 1.0).abs() < 1e-6,
                "node {node:?}: expected 1.0 for additive v(S)=|S|, got {v}"
            );
        }
    }

    /// The batched cache is shared across rounds (not rebuilt per round), so
    /// on a shape with full structural repetition across rounds — a chain has
    /// exactly one topological order, so every round's n+1 prefixes are
    /// identical to every other round's — only the first round should ever
    /// reach `value_fn_batch`; every later round is satisfied entirely from
    /// the cache. This exercises `resolve_batch_unique_coalitions` directly
    /// (rather than through the public batched entry point) so the round loop
    /// and the shared cache are both visible to the assertion.
    #[test]
    fn persistent_cache_collapses_batched_calls_on_chain() {
        const ROUNDS: usize = 10;
        const BATCH: usize = 20;
        let n = 65;
        let dag = make_chain(n);
        let base_in_deg = dag.in_degrees();
        let mut rng = make_rng(Some(9));
        let mut scratch = SamplerScratch::new(n);
        let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);
        let call_stats = std::cell::RefCell::new((0usize, 0usize)); // (num_calls, coalitions_evaluated)
        let counting_batch = |coalitions: &[Vec<NodeId>]| -> Result<Vec<f64>, CausasvError> {
            let mut stats = call_stats.borrow_mut();
            stats.0 += 1;
            stats.1 += coalitions.len();
            Ok(coalitions.iter().map(|c| c.len() as f64).collect())
        };

        for _ in 0..ROUNDS {
            let samples: Vec<SampledOrdering> = (0..BATCH)
                .map(|_| {
                    let log_q = sample_one_into(&dag, &mut rng, &mut scratch, &base_in_deg);
                    SampledOrdering {
                        ordering: scratch.ordering.clone(),
                        log_q,
                    }
                })
                .collect();
            resolve_batch_unique_coalitions(n, &samples, &counting_batch, &mut cache).unwrap();
        }

        let (num_calls, coalitions_evaluated) = *call_stats.borrow();
        assert_eq!(
            num_calls, 1,
            "chain(65) has a single topological order, so rounds 2..={ROUNDS} should be \
             100% cache hits and never reach value_fn_batch; got {num_calls} call(s)"
        );
        assert_eq!(coalitions_evaluated, n + 1);
        assert_eq!(cache.len(), n + 1);
    }

    proptest! {
        /// Same property as the two tests above, generalized over random chain
        /// lengths and seeds (repo convention: property tests are the
        /// correctness backbone for new graph-algorithm code here).
        #[test]
        fn prop_backend_parity_serial_seeded_chain(n in 2usize..20, seed in any::<u64>(), n_samples in 50usize..300) {
            let dag = make_chain(n);
            let small = approximate_asv(&dag, weighted, SamplingConfig::new(n_samples).with_seed(seed)).unwrap();
            let large = approximate_asv_large(&dag, weighted, SamplingConfig::new(n_samples).with_seed(seed)).unwrap();
            for (&node, &v_small) in &small.values {
                let v_large = large.values[&node];
                prop_assert_eq!(v_small.to_bits(), v_large.to_bits(),
                    "node {:?}: small={}, large={}", node, v_small, v_large);
            }
        }
    }

    /// Phase 2 acceptance criterion: cache growth must not track sample count
    /// unboundedly. A fully disconnected (antichain) large DAG maximizes
    /// distinct prefixes per sample (no shared structure at all), so it's the
    /// worst case for cache-size growth.
    #[test]
    fn large_cache_stays_bounded_on_antichain() {
        let mut dag = Dag::new();
        for i in 0..80usize {
            dag.add_node(&format!("n{i}"));
        }
        let n = dag.node_count();
        let base_in_deg = dag.in_degrees();
        let mut rng = make_rng(Some(1));
        let mut scratch = SamplerScratch::new(n);
        let mut cache = LargeCoalitionCache::new(DEFAULT_LARGE_CACHE_MAX_ENTRIES);
        for _ in 0..5_000 {
            let _ = sample_one_into(&dag, &mut rng, &mut scratch, &base_in_deg);
            let mut coalition = LargeCoalition::empty(n);
            let _ = value_cached_large(&mut cache, &weighted, &coalition).unwrap();
            for &node in &scratch.ordering {
                coalition.insert(node);
                let _ = value_cached_large(&mut cache, &weighted, &coalition).unwrap();
            }
        }
        assert!(
            cache.len() <= DEFAULT_LARGE_CACHE_MAX_ENTRIES,
            "cache grew to {} entries, exceeding cap {}",
            cache.len(),
            DEFAULT_LARGE_CACHE_MAX_ENTRIES
        );
    }
}
