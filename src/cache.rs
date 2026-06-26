use std::collections::HashMap;

use crate::error::CausasvError;
use crate::graph::NodeId;

pub(crate) fn mask_to_coalition(mask: u64) -> Vec<NodeId> {
    (0..64u32)
        .filter(|&i| mask & (1u64 << i) != 0)
        .map(NodeId)
        .collect()
}

pub(crate) fn vec_to_mask(nodes: &[NodeId]) -> u64 {
    nodes.iter().fold(0u64, |m, &n| m | (1u64 << n.0))
}

pub(crate) fn value_cached<F>(
    cache: &mut HashMap<u64, f64>,
    value_fn: &F,
    mask: u64,
) -> Result<f64, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    if let Some(&v) = cache.get(&mask) {
        return Ok(v);
    }
    let coalition = mask_to_coalition(mask);
    let v = value_fn(&coalition)?;
    cache.insert(mask, v);
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let nodes = vec![NodeId(0), NodeId(2), NodeId(5)];
        assert_eq!(mask_to_coalition(vec_to_mask(&nodes)), nodes);
    }

    #[test]
    fn empty_mask() {
        assert!(mask_to_coalition(0).is_empty());
    }
}
