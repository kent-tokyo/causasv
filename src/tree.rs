use std::collections::BTreeSet;

use crate::asv::AsvResult;
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};

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

/// Subtree size for each node: sizes[v] = number of nodes in the subtree rooted at v.
pub(crate) fn subtree_sizes(dag: &Dag, root: NodeId) -> Vec<usize> {
    let mut sizes = vec![0usize; dag.node_count()];
    dfs_sizes(dag, root, &mut sizes);
    sizes
}

fn dfs_sizes(dag: &Dag, node: NodeId, sizes: &mut Vec<usize>) {
    sizes[node.0 as usize] = 1;
    for &child in dag.children_raw(node) {
        dfs_sizes(dag, child, sizes);
        sizes[node.0 as usize] += sizes[child.0 as usize];
    }
}

/// Path from `root` to parent of `target`, not including `target` itself.
fn ancestors_of(dag: &Dag, target: NodeId, root: NodeId) -> Vec<NodeId> {
    if target == root {
        return vec![];
    }
    let mut path = Vec::new();
    find_path(dag, root, target, &mut path);
    path.pop(); // remove target itself
    path
}

fn find_path(dag: &Dag, current: NodeId, target: NodeId, path: &mut Vec<NodeId>) -> bool {
    path.push(current);
    if current == target {
        return true;
    }
    for &child in dag.children_raw(current) {
        if find_path(dag, child, target, path) {
            return true;
        }
    }
    path.pop();
    false
}

/// Enumerate all order ideals (downward-closed subsets) of a forest given by its root nodes.
/// Each ideal is returned as a sorted Vec<NodeId>. ∅ is always included.
pub(crate) fn enumerate_order_ideals(dag: &Dag, roots: &[NodeId]) -> Vec<Vec<NodeId>> {
    if roots.is_empty() {
        return vec![vec![]];
    }
    let per_tree: Vec<Vec<Vec<NodeId>>> = roots.iter().map(|&r| tree_ideals(dag, r)).collect();
    cartesian_product_vecs(&per_tree)
        .into_iter()
        .map(|mut v| {
            v.sort_unstable();
            v
        })
        .collect()
}

fn tree_ideals(dag: &Dag, root: NodeId) -> Vec<Vec<NodeId>> {
    let children = dag.children_raw(root);
    let children_ideals: Vec<Vec<Vec<NodeId>>> =
        children.iter().map(|&c| tree_ideals(dag, c)).collect();

    let mut result = vec![vec![]]; // ∅ is always an order ideal
    for combo in cartesian_product_vecs(&children_ideals) {
        let mut ideal = Vec::with_capacity(1 + combo.len());
        ideal.push(root);
        ideal.extend_from_slice(&combo);
        ideal.sort_unstable();
        result.push(ideal);
    }
    result
}

/// Cartesian product of groups of Vecs: one Vec chosen from each group, concatenated.
fn cartesian_product_vecs(groups: &[Vec<Vec<NodeId>>]) -> Vec<Vec<NodeId>> {
    if groups.is_empty() {
        return vec![vec![]];
    }
    let mut result = vec![vec![]];
    for group in groups {
        let mut new_result = Vec::new();
        for prev in &result {
            for item in group {
                let mut combined = prev.clone();
                combined.extend_from_slice(item);
                new_result.push(combined);
            }
        }
        result = new_result;
    }
    result
}

/// log L(T[S]) where S is an order ideal of T rooted at `root`.
///
/// Uses L(T[S]) = |S|! / Π_{v ∈ S} s_{T[S]}(v), computed via DFS.
fn log_lin_ext_of_s(dag: &Dag, s: &[NodeId], root: NodeId, log_fact: &[f64]) -> f64 {
    if s.is_empty() {
        return 0.0; // L(∅) = 1
    }
    let s_set: BTreeSet<NodeId> = s.iter().copied().collect();
    let mut sub_sizes = vec![0usize; dag.node_count()];
    dfs_sizes_in_s(dag, root, &s_set, &mut sub_sizes);
    let sum_log: f64 = s
        .iter()
        .map(|&v| (sub_sizes[v.0 as usize] as f64).ln())
        .sum();
    log_fact[s.len()] - sum_log
}

fn dfs_sizes_in_s(dag: &Dag, node: NodeId, s_set: &BTreeSet<NodeId>, sizes: &mut Vec<usize>) {
    if !s_set.contains(&node) {
        return;
    }
    sizes[node.0 as usize] = 1;
    for &child in dag.children_raw(node) {
        if s_set.contains(&child) {
            dfs_sizes_in_s(dag, child, s_set, sizes);
            sizes[node.0 as usize] += sizes[child.0 as usize];
        }
    }
}

/// Exact ASV for rooted directed trees using the order-ideal DP.
///
/// For each node i, enumerates all valid pre-sets S (= anc(i) ∪ order ideals of side subtrees)
/// and weights each by count(S) / L(T) using the hook-length formula:
///
///   log weight(S) = log L(T[S]) + log(N!) + Σ_{v ∈ S∪{i}} log s(v) - log(n!)
///
/// where N = n - |S| - 1.  This is more efficient than brute-force linear extension
/// enumeration for trees with large L(T).
pub(crate) fn tree_exact_asv<F>(dag: &Dag, value_fn: F) -> Result<AsvResult, CausasvError>
where
    F: Fn(&[NodeId]) -> Result<f64, CausasvError>,
{
    let root = find_rooted_tree_root(dag)?;
    let sizes = subtree_sizes(dag, root);

    let n = dag.node_count();
    let mut log_fact = vec![0.0f64; n + 1];
    for k in 1..=n {
        log_fact[k] = log_fact[k - 1] + (k as f64).ln();
    }
    let log_s: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();

    let mut phi = vec![0.0f64; n];

    for i in dag.all_nodes() {
        let anc = ancestors_of(dag, i, root);
        let d = anc.len();

        // For each ancestor a_k, collect side children (not on path to i).
        let mut ideals_per_level: Vec<Vec<Vec<NodeId>>> = Vec::with_capacity(d);
        for k in 0..d {
            let a_k = anc[k];
            let on_path = if k + 1 < d { anc[k + 1] } else { i };
            let side: Vec<NodeId> = dag
                .children_raw(a_k)
                .iter()
                .copied()
                .filter(|&c| c != on_path)
                .collect();
            ideals_per_level.push(enumerate_order_ideals(dag, &side));
        }

        // Iterate over all combinations (one ideal per level).
        for combo in cartesian_product_vecs(&ideals_per_level) {
            // S = ancestors ∪ combo (from disjoint subtrees → no duplicates)
            let mut s_vec: Vec<NodeId> = anc.iter().copied().chain(combo).collect();
            s_vec.sort_unstable();

            // S ∪ {i}
            let mut s_plus_i = s_vec.clone();
            let pos = s_plus_i.partition_point(|&v| v < i);
            s_plus_i.insert(pos, i);

            let log_ls = log_lin_ext_of_s(dag, &s_vec, root, &log_fact);
            let n_comp = n - s_vec.len() - 1;
            let sum_log_si: f64 = s_plus_i.iter().map(|&v| log_s[v.0 as usize]).sum();
            let w = (log_ls + log_fact[n_comp] + sum_log_si - log_fact[n]).exp();

            phi[i.0 as usize] += w * (value_fn(&s_plus_i)? - value_fn(&s_vec)?);
        }
    }

    let values = dag.all_nodes().map(|v| (v, phi[v.0 as usize])).collect();
    Ok(AsvResult {
        values,
        n_samples: 0,
        seed: None,
        is_exact: true,
        effective_sample_size: None,
    })
}
