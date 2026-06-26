use crate::asv::AsvResult;
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
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    if config.n_samples == 0 {
        return Err(CausasvError::InvalidConfig(
            "n_samples must be > 0".to_string(),
        ));
    }
    let n = dag.node_count();
    let seed = config.seed;
    let mut rng = make_rng(seed);
    let mut numerator = vec![0.0f64; n];
    let mut denominator = 0.0f64;
    let mut coalition = Vec::with_capacity(n);

    for _ in 0..config.n_samples {
        let sample = sample_one(dag, &mut rng);
        let w = (-sample.log_q).exp(); // IS weight = 1/q(π)
        denominator += w;

        for (k, &node) in sample.ordering.iter().enumerate() {
            coalition.clear();
            coalition.extend_from_slice(&sample.ordering[..k]);
            coalition.sort_unstable();
            let v_without = value_fn(&coalition)?;

            coalition.push(node);
            coalition.sort_unstable();
            let v_with = value_fn(&coalition)?;

            numerator[node.0 as usize] += w * (v_with - v_without);
        }
    }

    let values = (0..n)
        .map(|i| (NodeId(i as u32), numerator[i] / denominator))
        .collect();

    Ok(AsvResult {
        values,
        n_samples: config.n_samples,
        seed,
        is_exact: false,
    })
}
