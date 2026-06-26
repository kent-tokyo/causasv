use crate::error::CausasvError;
use crate::graph::NodeId;

/// Type alias for a value function over node coalitions.
/// The library guarantees the coalition slice is sorted in ascending NodeId order before calling,
/// so callers may use it as a stable cache key.
pub type ValueFn<'a> = dyn Fn(&[NodeId]) -> Result<f64, CausasvError> + 'a;
