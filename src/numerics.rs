/// Kahan compensated addition: accumulates `x` into `*sum` with `*comp`
/// tracking the running error. Reduces floating-point rounding from O(n·ε)
/// to O(ε²) per step.
///
/// Used in approximate ASV accumulation of numerator, denominator, sum_w_sq,
/// and stderr terms to maintain precision over large sample counts.
#[inline]
pub(crate) fn kahan_add(sum: &mut f64, comp: &mut f64, x: f64) {
    let y = x - *comp;
    let t = *sum + y;
    *comp = (t - *sum) - y;
    *sum = t;
}

/// Approximate inverse normal CDF (Beasley-Springer-Moro), accurate to ~0.01
/// for p in (0.9, 0.999). Used to turn a confidence level into a z-score for
/// `value ± z * stderr` bounds (e.g. `normal_quantile(0.975)` for a 95% CI).
///
/// Only consumed by the `python` feature's CI-bound computation (`src/python.rs`).
#[cfg(feature = "python")]
pub(crate) fn normal_quantile(p: f64) -> f64 {
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c = [2.515_517, 0.802_853, 0.010_328];
    let d = [1.432_788, 0.189_269, 0.001_308];
    t - (c[0] + c[1] * t + c[2] * t * t) / (1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t)
}

#[cfg(all(test, feature = "python"))]
mod tests {
    use super::*;

    #[test]
    fn normal_quantile_matches_known_z_scores() {
        // Reference z-scores for common two-sided confidence levels.
        let cases = [(0.90, 1.644854), (0.95, 1.959964), (0.99, 2.575829)];
        for (ci_level, expected_z) in cases {
            let z = normal_quantile((1.0 + ci_level) / 2.0);
            assert!(
                (z - expected_z).abs() < 0.01,
                "normal_quantile for {ci_level} CI: expected z≈{expected_z}, got {z}"
            );
        }
    }
}
