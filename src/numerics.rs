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
