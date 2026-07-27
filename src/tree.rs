use std::collections::{BTreeSet, HashMap};

use crate::asv::AsvResult;
use crate::cache::{value_cached, vec_to_mask};
use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};

/// Returns the unique root (in-degree 0 node) if this DAG is a rooted directed tree.
/// A rooted directed tree has exactly one root and all other nodes have in-degree 1.
pub(crate) fn find_rooted_tree_root(dag: &Dag) -> Result<NodeId, CausasvError> {
    let in_deg = dag.in_degrees();
    let roots: Vec<NodeId> = in_deg
        .iter()
        .enumerate()
        .filter(|&(_, d)| *d == 0)
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

/// Order-ideal count for the subtree rooted at each node: `counts[v] = 1 +
/// Π counts[child]` (the `1` is the ideal that excludes `v`; the product ranges
/// over every combination of the children's own ideals in the ideal that
/// includes `v`) — the same recurrence `tree_ideals` computes, but as a single
/// `u64` per node instead of the actual `Vec<Vec<NodeId>>` enumeration, and
/// saturating on overflow instead of allocating an impossible amount of memory.
/// This is what makes `estimate_tree_exact_cost` cheap: O(n) multiplications,
/// no materialization, regardless of how the real ideal count would blow up.
fn subtree_ideal_counts(dag: &Dag, root: NodeId) -> Vec<u64> {
    let mut counts = vec![0u64; dag.node_count()];
    dfs_ideal_counts(dag, root, &mut counts);
    counts
}

fn dfs_ideal_counts(dag: &Dag, node: NodeId, counts: &mut Vec<u64>) {
    let mut product: u64 = 1;
    for &child in dag.children_raw(node) {
        dfs_ideal_counts(dag, child, counts);
        product = product.saturating_mul(counts[child.0 as usize]);
    }
    counts[node.0 as usize] = 1u64.saturating_add(product);
}

/// Result of [`estimate_tree_exact_cost`]: a cheap analytical prediction of
/// `tree_exact_asv`'s cost for this specific tree, computed without ever
/// building the cartesian products it warns about.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TreeExactCostEstimate {
    /// Whether both budgets below are satisfied.
    pub(crate) feasible: bool,
    /// The single largest side-sibling order-ideal product that
    /// `enumerate_order_ideals` would have to materialize at once, across every
    /// (ancestor, on-path-child) pair in the tree. This is the "peak" cost —
    /// exceeding it means at least one call would try to build an
    /// astronomically large `Vec` in one shot.
    pub(crate) max_cartesian_terms: u64,
    /// The configured budget `max_cartesian_terms` was compared against.
    pub(crate) cartesian_term_budget: u64,
    /// Total work across the whole `tree_exact_asv` run: the sum, over every
    /// node, of that node's own per-node combination count (the product of
    /// every ancestor level's side-sibling order-ideal count on the path from
    /// the root). This is the "sum" cost — even if no single node is
    /// individually catastrophic, many moderately expensive nodes can still
    /// add up to an infeasible total.
    pub(crate) estimated_total_terms: u64,
    /// The configured budget `estimated_total_terms` was compared against.
    pub(crate) total_term_budget: u64,
}

/// Analytically estimate `tree_exact_asv`'s cost for `dag` (rooted at `root`)
/// against the two budgets, without enumerating a single order ideal.
///
/// `tree_exact_asv`'s cost is not a function of node count alone: for each
/// node `i`, `cartesian_product_vecs` combines *every* ancestor level's
/// side-sibling order ideals at once, so `i`'s cost is the product (not sum) of
/// one side-sibling product per ancestor level on the root-to-`i` path. This
/// mirrors that exact structure via a top-down DFS threading a running product
/// (computed in O(n) total using prefix/suffix products per node's children —
/// see `accumulate_cost`), so pathological cases are caught by (saturating)
/// arithmetic overflow, never by actually running out of memory.
pub(crate) fn estimate_tree_exact_cost(
    dag: &Dag,
    root: NodeId,
    cartesian_term_budget: u64,
    total_term_budget: u64,
) -> TreeExactCostEstimate {
    let ideal_counts = subtree_ideal_counts(dag, root);
    let mut max_cartesian_terms = 0u64;
    let mut estimated_total_terms = 0u64;
    // Node i's own combination count is the PRODUCT of every ancestor level's
    // side-product along the root->i path (tree_exact_asv's inner
    // cartesian_product_vecs(&ideals_per_level) combines *all* of i's ancestor
    // levels at once, not one level in isolation) -- so this threads a running
    // product down the tree, multiplying in one more side-product per edge
    // descended, rather than summing per-edge costs independently.
    accumulate_cost(
        dag,
        root,
        &ideal_counts,
        1,
        &mut max_cartesian_terms,
        &mut estimated_total_terms,
    );
    TreeExactCostEstimate {
        feasible: max_cartesian_terms <= cartesian_term_budget
            && estimated_total_terms <= total_term_budget,
        max_cartesian_terms,
        cartesian_term_budget,
        estimated_total_terms,
        total_term_budget,
    }
}

/// `running_product` = the product of every ancestor level's side-product from
/// `root` down to (but not including) `node`'s own children -- i.e. exactly
/// `tree_exact_asv`'s per-node combination count for `node` itself (1 for the
/// root, whose ancestor chain is empty). Descending to a child `c` multiplies
/// in `side_product(node, c)` = the product of `ideal_counts` over `node`'s
/// *other* children (via prefix/suffix products, O(children) per node rather
/// than O(children²)), matching the "side" set `tree_exact_asv` builds at that
/// level. `max_cartesian_terms` tracks the single largest such per-node
/// product (the biggest single `cartesian_product_vecs` materialization);
/// `estimated_total_terms` sums it across every node (the total combinations
/// iterated over the whole computation).
fn accumulate_cost(
    dag: &Dag,
    node: NodeId,
    ideal_counts: &[u64],
    running_product: u64,
    max_cartesian_terms: &mut u64,
    estimated_total_terms: &mut u64,
) {
    *max_cartesian_terms = (*max_cartesian_terms).max(running_product);
    *estimated_total_terms = (*estimated_total_terms).saturating_add(running_product);

    let children = dag.children_raw(node);
    let m = children.len();
    if m > 0 {
        let child_counts: Vec<u64> = children
            .iter()
            .map(|&c| ideal_counts[c.0 as usize])
            .collect();
        let mut prefix = vec![1u64; m + 1];
        for i in 0..m {
            prefix[i + 1] = prefix[i].saturating_mul(child_counts[i]);
        }
        let mut suffix = vec![1u64; m + 1];
        for i in (0..m).rev() {
            suffix[i] = suffix[i + 1].saturating_mul(child_counts[i]);
        }
        for (j, &child) in children.iter().enumerate() {
            let side_product = prefix[j].saturating_mul(suffix[j + 1]);
            let child_running_product = running_product.saturating_mul(side_product);
            accumulate_cost(
                dag,
                child,
                ideal_counts,
                child_running_product,
                max_cartesian_terms,
                estimated_total_terms,
            );
        }
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
    if n > 64 {
        return Err(CausasvError::InvalidConfig(format!(
            "bitmask coalitions require n ≤ 64, got {n}"
        )));
    }
    let mut log_fact = vec![0.0f64; n + 1];
    for k in 1..=n {
        log_fact[k] = log_fact[k - 1] + (k as f64).ln();
    }
    let log_s: Vec<f64> = sizes.iter().map(|&s| (s as f64).ln()).collect();

    let mut phi = vec![0.0f64; n];
    let mut cache = HashMap::<u64, f64>::new();

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

            let s_mask = vec_to_mask(&s_vec);
            let si_mask = s_mask | (1u64 << i.0);
            phi[i.0 as usize] += w
                * (value_cached(&mut cache, &value_fn, si_mask)?
                    - value_cached(&mut cache, &value_fn, s_mask)?);
        }
    }

    let values = dag.all_nodes().map(|v| (v, phi[v.0 as usize])).collect();
    Ok(AsvResult {
        values,
        n_samples: 0,
        seed: None,
        is_exact: true,
        effective_sample_size: None,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn add_balanced_subtree(dag: &mut Dag, parent: NodeId, depth: usize, prefix: &str) {
        if depth == 0 {
            return;
        }
        let left = dag.add_node(&format!("{prefix}l"));
        let right = dag.add_node(&format!("{prefix}r"));
        dag.add_edge(parent, left).unwrap();
        dag.add_edge(parent, right).unwrap();
        add_balanced_subtree(dag, left, depth - 1, &format!("{prefix}l"));
        add_balanced_subtree(dag, right, depth - 1, &format!("{prefix}r"));
    }

    /// Root with `num_branches` children, each the root of its own balanced binary
    /// subtree of the given `depth` — same shape as issue #36's reproduction case.
    fn star_of_balanced_subtrees(num_branches: usize, depth: usize) -> Dag {
        let mut dag = Dag::new();
        let root = dag.add_node("root");
        for b in 0..num_branches {
            let branch_root = dag.add_node(&format!("b{b}"));
            dag.add_edge(root, branch_root).unwrap();
            add_balanced_subtree(&mut dag, branch_root, depth, &format!("b{b}"));
        }
        dag
    }

    #[test]
    fn subtree_ideal_count_matches_known_small_values() {
        // Single node: only the empty ideal + itself = 2.
        let mut dag = Dag::new();
        let leaf = dag.add_node("leaf");
        assert_eq!(subtree_ideal_counts(&dag, leaf)[leaf.0 as usize], 2);

        // Complete binary tree of height 2 (7 nodes): I(0)=2, I(1)=1+2*2=5, I(2)=1+5*5=26.
        let dag = star_of_balanced_subtrees(1, 2);
        let root = find_rooted_tree_root(&dag).unwrap();
        let counts = subtree_ideal_counts(&dag, root);
        // root -> single branch (depth 2) is the whole tree here (num_branches=1),
        // so root's own count equals 1 + count(branch_root) = 1 + 26 = 27.
        assert_eq!(counts[root.0 as usize], 27);
    }

    #[test]
    fn estimate_tree_exact_cost_cheap_case_is_feasible() {
        let dag = star_of_balanced_subtrees(1, 3); // single branch, no side siblings at all
        let root = find_rooted_tree_root(&dag).unwrap();
        let estimate = estimate_tree_exact_cost(&dag, root, 1_000_000, 10_000_000);
        assert!(estimate.feasible);
    }

    /// Locks in the exact numbers behind issue #36: a rooted tree with only
    /// n=61 nodes whose deepest leaves carry a per-node combination count (the
    /// product of every ancestor level's side-product, matching
    /// `tree_exact_asv`'s `cartesian_product_vecs` over all ancestor levels at
    /// once) that reaches ~8.1e10 -- and this is caught via the O(n) analytical
    /// estimate only, never by actually enumerating a single order ideal.
    #[test]
    fn estimate_tree_exact_cost_flags_issue_36_reproduction_shape() {
        let dag = star_of_balanced_subtrees(4, 3);
        assert_eq!(dag.node_count(), 61);
        let root = find_rooted_tree_root(&dag).unwrap();

        // Matches ExactTreeConfig::default() in asv.rs -- duplicated as literals
        // here rather than referencing that module, since tree.rs is a lower-level
        // module asv.rs depends on, not the reverse.
        let estimate = estimate_tree_exact_cost(&dag, root, 1_000_000, 10_000_000);

        // Each branch root is itself the root of a complete binary tree of height 3:
        // I(0)=2, I(1)=5, I(2)=26, I(3)=677 (matches subtree_ideal_counts). A leaf at
        // the bottom of one branch accumulates one side-product per ancestor level:
        // 677^3 (the other 3 branches, at the overall root) * 26 (sibling subtree at
        // branch depth 1) * 5 (at depth 2) * 2 (at depth 3, the leaf's own sibling).
        let i0 = 2u64;
        let i1 = 1 + i0 * i0;
        let i2 = 1 + i1 * i1;
        let i3 = 1 + i2 * i2;
        assert_eq!(i3, 677);
        let expected_deepest_leaf_cost = i3.pow(3) * i2 * i1 * i0;
        assert_eq!(estimate.max_cartesian_terms, expected_deepest_leaf_cost);
        assert!(!estimate.feasible);
        assert!(estimate.max_cartesian_terms > estimate.cartesian_term_budget);
    }

    #[test]
    fn estimate_tree_exact_cost_never_hangs_on_a_much_larger_bushy_tree() {
        // 6 branches, depth 4 (branch ideal count I(4) = 1+677^2 = 458_330):
        // side product for the root level alone is ~458330^5, astronomically
        // larger than issue #36's case. The estimate must still return instantly
        // (O(n) arithmetic, saturating on overflow) rather than hang or panic.
        let dag = star_of_balanced_subtrees(6, 4);
        let root = find_rooted_tree_root(&dag).unwrap();
        let estimate = estimate_tree_exact_cost(&dag, root, 1_000_000, 10_000_000);
        assert!(!estimate.feasible);
        assert!(estimate.max_cartesian_terms >= 1_000_000);
    }
}
