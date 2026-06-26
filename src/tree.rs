use crate::asv::AsvResult;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use crate::topo::enumerate_topos;

/// Returns the unique root (in-degree 0 node) if this DAG is a rooted directed tree.
/// A rooted directed tree has exactly one root and all other nodes have in-degree 1.
pub(crate) fn find_rooted_tree_root(dag: &Dag) -> Result<NodeId, CausasvError> {
    let in_deg = dag.in_degrees();
    let roots: Vec<NodeId> = in_deg
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| NodeId(i as u32))
        .collect();

    if roots.len() != 1 {
        return Err(CausasvError::NotRootedTree);
    }

    // All non-root nodes must have exactly in-degree 1.
    let all_unit = in_deg
        .iter()
        .enumerate()
        .filter(|(i, _)| NodeId(*i as u32) != roots[0])
        .all(|(_, &d)| d == 1);

    if !all_unit {
        return Err(CausasvError::NotRootedTree);
    }

    Ok(roots[0])
}

/// Exact ASV for a rooted directed tree. Returns Err(NotRootedTree) if the graph is not one.
///
/// ponytail: uses enumerate_topos (same as brute-force); tree-specific DP in v0.2.0.
/// The value of this method is the tree-structure validation upfront.
pub(crate) fn tree_exact_asv<F>(dag: &Dag, value_fn: F) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    find_rooted_tree_root(dag)?;

    let orderings = enumerate_topos(dag)?;
    let n_orderings = orderings.len();
    let n = dag.node_count();
    let mut phi = vec![0.0f64; n];
    let mut coalition = Vec::with_capacity(n);

    for ordering in &orderings {
        for (k, &node) in ordering.iter().enumerate() {
            coalition.clear();
            coalition.extend_from_slice(&ordering[..k]);
            coalition.sort_unstable();
            let v_without = value_fn(&coalition)?;

            coalition.push(node);
            coalition.sort_unstable();
            let v_with = value_fn(&coalition)?;

            phi[node.0 as usize] += v_with - v_without;
        }
    }

    let scale = 1.0 / n_orderings as f64;
    let values = (0..n).map(|i| (NodeId(i as u32), phi[i] * scale)).collect();

    Ok(AsvResult {
        values,
        n_samples: n_orderings,
        seed: None,
        is_exact: true,
    })
}
