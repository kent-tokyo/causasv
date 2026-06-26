use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};

/// Returns a valid topological ordering (smallest NodeId first among ties).
pub fn topo_sort(dag: &Dag) -> Result<Vec<NodeId>, CausasvError> {
    let n = dag.node_count();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut in_deg = dag.in_degrees();
    let mut heap: BinaryHeap<Reverse<NodeId>> = in_deg
        .iter()
        .enumerate()
        .filter(|(_, &d)| d == 0)
        .map(|(i, _)| Reverse(NodeId(i as u32)))
        .collect();
    let mut result = Vec::with_capacity(n);
    while let Some(Reverse(node)) = heap.pop() {
        result.push(node);
        for &child in dag.children_raw(node) {
            in_deg[child.0 as usize] -= 1;
            if in_deg[child.0 as usize] == 0 {
                heap.push(Reverse(child));
            }
        }
    }
    if result.len() != n {
        return Err(CausasvError::CycleDetected);
    }
    Ok(result)
}

/// Enumerates ALL valid topological orderings. Output is deterministic (frontier sorted
/// by NodeId at each step). Only practical for small graphs (n ≤ ~8).
pub fn enumerate_topos(dag: &Dag) -> Result<Vec<Vec<NodeId>>, CausasvError> {
    dag.validate()?;
    let n = dag.node_count();
    let mut in_deg = dag.in_degrees();
    let mut current = Vec::with_capacity(n);
    let mut result = Vec::new();
    backtrack(dag, &mut in_deg, &mut current, &mut result, n);
    Ok(result)
}

fn backtrack(
    dag: &Dag,
    in_deg: &mut Vec<usize>,
    current: &mut Vec<NodeId>,
    result: &mut Vec<Vec<NodeId>>,
    remaining: usize,
) {
    if remaining == 0 {
        result.push(current.clone());
        return;
    }
    // Frontier: nodes with in_deg == 0 (placed nodes have in_deg = MAX)
    // Iterate in ascending NodeId order for deterministic output.
    let frontier: Vec<NodeId> = (0..in_deg.len())
        .filter(|&i| in_deg[i] == 0)
        .map(|i| NodeId(i as u32))
        .collect();

    for node in frontier {
        in_deg[node.0 as usize] = usize::MAX; // mark as placed
        for &child in dag.children_raw(node) {
            in_deg[child.0 as usize] -= 1;
        }
        current.push(node);

        backtrack(dag, in_deg, current, result, remaining - 1);

        current.pop();
        for &child in dag.children_raw(node) {
            in_deg[child.0 as usize] += 1;
        }
        in_deg[node.0 as usize] = 0; // restore (it was 0 when selected)
    }
}
