/// Property-based tests for `Dag::d_convex_hull`/`Dag::strong_d_convex_hull`,
/// cross-checked against an independent brute-force oracle.
///
/// The oracle implements d-separation from scratch via classical path
/// enumeration and collider-blocking (Pearl 1988) -- a completely different
/// code path from `src/d_convex.rs`'s moralization-based shortest-path
/// approach, and from anything in the arXiv:2606.08941 paper. It directly
/// checks Definition 5's two conditions (mf-pair absence, linear ordering)
/// by exhaustive subset enumeration, then finds the minimal strongly
/// d-convex superset by brute force. This is the test suite's actual
/// correctness backbone -- the hand-built fixtures in `d_convex_tests.rs`
/// only cover a few shapes; this covers every small DAG proptest generates.
use causasv::{Dag, NodeId};
use proptest::prelude::*;
use std::collections::BTreeSet;

// ── independent brute-force d-separation oracle ─────────────────────────────

fn adjacent(dag: &Dag, a: NodeId, b: NodeId) -> bool {
    dag.children(a).unwrap().contains(&b) || dag.children(b).unwrap().contains(&a)
}

fn undirected_skeleton(dag: &Dag) -> Vec<Vec<NodeId>> {
    let n = dag.node_count();
    let mut adj = vec![Vec::new(); n];
    for a in dag.all_nodes() {
        for &b in dag.children(a).unwrap() {
            adj[a.0 as usize].push(b);
            adj[b.0 as usize].push(a);
        }
    }
    adj
}

/// Every simple path between `x` and `y` in the undirected skeleton, found
/// via exhaustive DFS. Tractable only for small graphs (n <= ~8).
fn enumerate_simple_paths(skeleton: &[Vec<NodeId>], x: NodeId, y: NodeId) -> Vec<Vec<NodeId>> {
    let mut paths = Vec::new();
    let mut visited = BTreeSet::new();
    let mut current = vec![x];
    visited.insert(x);
    dfs_paths(skeleton, x, y, &mut visited, &mut current, &mut paths);
    paths
}

fn dfs_paths(
    skeleton: &[Vec<NodeId>],
    current_node: NodeId,
    target: NodeId,
    visited: &mut BTreeSet<NodeId>,
    current: &mut Vec<NodeId>,
    paths: &mut Vec<Vec<NodeId>>,
) {
    if current_node == target {
        paths.push(current.clone());
        return;
    }
    for &next in &skeleton[current_node.0 as usize] {
        if visited.insert(next) {
            current.push(next);
            dfs_paths(skeleton, next, target, visited, current, paths);
            current.pop();
            visited.remove(&next);
        }
    }
}

fn is_collider_on_path(dag: &Dag, path: &[NodeId], i: usize) -> bool {
    let prev = path[i - 1];
    let cur = path[i];
    let next = path[i + 1];
    dag.children(prev).unwrap().contains(&cur) && dag.children(next).unwrap().contains(&cur)
}

fn descendants_incl_self(dag: &Dag, v: NodeId) -> BTreeSet<NodeId> {
    let mut result = BTreeSet::new();
    result.insert(v);
    let mut stack = vec![v];
    while let Some(u) = stack.pop() {
        for &c in dag.children(u).unwrap() {
            if result.insert(c) {
                stack.push(c);
            }
        }
    }
    result
}

/// A path is blocked by `s` if it contains a non-collider that's in `s`, or
/// an unblocked collider (neither it nor any descendant is in `s`).
fn path_blocked_by(dag: &Dag, path: &[NodeId], s: &BTreeSet<NodeId>) -> bool {
    for i in 1..path.len() - 1 {
        if is_collider_on_path(dag, path, i) {
            if descendants_incl_self(dag, path[i]).is_disjoint(s) {
                return true;
            }
        } else if s.contains(&path[i]) {
            return true;
        }
    }
    false
}

fn d_connected(
    dag: &Dag,
    skeleton: &[Vec<NodeId>],
    x: NodeId,
    y: NodeId,
    s: &BTreeSet<NodeId>,
) -> bool {
    enumerate_simple_paths(skeleton, x, y)
        .iter()
        .any(|p| !path_blocked_by(dag, p, s))
}

/// {x,y} is an mf-pair of `r` if, for every subset S of r\{x,y}, x and y
/// remain d-connected given S (Definition 3).
fn oracle_is_mf_pair(
    dag: &Dag,
    skeleton: &[Vec<NodeId>],
    r: &BTreeSet<NodeId>,
    x: NodeId,
    y: NodeId,
) -> bool {
    let candidates: Vec<NodeId> = r.iter().copied().filter(|&v| v != x && v != y).collect();
    let k = candidates.len();
    for mask in 0..(1u32 << k) {
        let s: BTreeSet<NodeId> = (0..k)
            .filter(|&i| mask & (1 << i) != 0)
            .map(|i| candidates[i])
            .collect();
        if !d_connected(dag, skeleton, x, y, &s) {
            return false;
        }
    }
    true
}

fn oracle_is_d_convex(dag: &Dag, skeleton: &[Vec<NodeId>], r: &BTreeSet<NodeId>) -> bool {
    let nodes: Vec<NodeId> = r.iter().copied().collect();
    for i in 0..nodes.len() {
        for &y in &nodes[i + 1..] {
            let x = nodes[i];
            if adjacent(dag, x, y) {
                continue;
            }
            if oracle_is_mf_pair(dag, skeleton, r, x, y) {
                return false;
            }
        }
    }
    true
}

fn children_or_self(dag: &Dag, seeds: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
    let mut result = seeds.clone();
    for &v in seeds {
        result.extend(dag.children(v).unwrap().iter().copied());
    }
    result
}

fn ancestors_or_self(dag: &Dag, seeds: &BTreeSet<NodeId>) -> BTreeSet<NodeId> {
    let mut result = seeds.clone();
    let mut stack: Vec<NodeId> = seeds.iter().copied().collect();
    while let Some(v) = stack.pop() {
        for &p in dag.parents(v).unwrap() {
            if result.insert(p) {
                stack.push(p);
            }
        }
    }
    result
}

fn oracle_linearly_ordered(dag: &Dag, v: NodeId, r: &BTreeSet<NodeId>) -> bool {
    let parents = dag.parents(v).unwrap();
    for i in 0..parents.len() {
        for &p2 in &parents[i + 1..] {
            let p1 = parents[i];
            if r.contains(&p1) && r.contains(&p2) {
                continue;
            }
            if !adjacent(dag, p1, p2) {
                return false;
            }
        }
    }
    true
}

fn oracle_is_strongly_d_convex(dag: &Dag, skeleton: &[Vec<NodeId>], r: &BTreeSet<NodeId>) -> bool {
    if !oracle_is_d_convex(dag, skeleton, r) {
        return false;
    }
    let m: BTreeSet<NodeId> = dag.all_nodes().filter(|n| !r.contains(n)).collect();
    let ch_m = children_or_self(dag, &m);
    let an_r = ancestors_or_self(dag, r);
    ch_m.intersection(&an_r)
        .all(|&w| oracle_linearly_ordered(dag, w, r))
}

/// Brute-force minimal strongly d-convex superset of `required`: enumerate
/// every superset (as a bitmask over the non-required nodes) and return the
/// smallest one that satisfies both of Definition 5's conditions.
fn oracle_minimal_strong_d_convex_superset(
    dag: &Dag,
    required: &BTreeSet<NodeId>,
) -> BTreeSet<NodeId> {
    let skeleton = undirected_skeleton(dag);
    let others: Vec<NodeId> = dag.all_nodes().filter(|n| !required.contains(n)).collect();
    let k = others.len();
    let mut best: Option<BTreeSet<NodeId>> = None;
    for mask in 0..(1u32 << k) {
        let h: BTreeSet<NodeId> = required
            .iter()
            .copied()
            .chain((0..k).filter(|&i| mask & (1 << i) != 0).map(|i| others[i]))
            .collect();
        if let Some(b) = &best {
            if h.len() >= b.len() {
                continue;
            }
        }
        if oracle_is_strongly_d_convex(dag, &skeleton, &h) {
            best = Some(h);
        }
    }
    best.expect("the full node set is always strongly d-convex (M = empty set)")
}

// ── generators ───────────────────────────────────────────────────────────────

/// Random small DAG, same construction as `tests/property_tests.rs::arb_dag`.
fn arb_dag(max_n: usize) -> impl Strategy<Value = Dag> {
    (2usize..=max_n).prop_flat_map(|n| {
        let n_pairs = n * (n - 1) / 2;
        prop::collection::vec(prop::bool::ANY, n_pairs).prop_map(move |include| {
            let mut dag = Dag::new();
            for i in 0..n {
                dag.add_node(&format!("n{i}"));
            }
            let mut k = 0;
            for i in 0..n {
                for j in (i + 1)..n {
                    if include[k] {
                        let _ = dag.add_edge(NodeId(i as u32), NodeId(j as u32));
                    }
                    k += 1;
                }
            }
            dag
        })
    })
}

/// A DAG paired with a small non-empty required node subset.
fn arb_dag_with_required(max_n: usize) -> impl Strategy<Value = (Dag, BTreeSet<NodeId>)> {
    arb_dag(max_n).prop_flat_map(|dag| {
        let n = dag.node_count();
        prop::collection::vec(prop::bool::ANY, n).prop_map(move |mask| {
            let mut required: BTreeSet<NodeId> = (0..n)
                .filter(|&i| mask[i])
                .map(|i| NodeId(i as u32))
                .collect();
            if required.is_empty() {
                required.insert(NodeId(0));
            }
            (dag.clone(), required)
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// `Dag::d_convex_hull` matches the brute-force minimal d-convex
    /// superset (checked via the oracle's d-convexity condition alone,
    /// i.e. Definition 5 condition (i) only).
    #[test]
    fn prop_d_convex_hull_matches_oracle((dag, required) in arb_dag_with_required(6)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let actual: BTreeSet<NodeId> = dag.d_convex_hull(&required_vec).unwrap().into_iter().collect();

        // Independently find the minimal d-convex (not strong) superset.
        let skeleton = undirected_skeleton(&dag);
        let others: Vec<NodeId> = dag.all_nodes().filter(|n| !required.contains(n)).collect();
        let k = others.len();
        let mut expected: Option<BTreeSet<NodeId>> = None;
        for mask in 0..(1u32 << k) {
            let h: BTreeSet<NodeId> = required.iter().copied()
                .chain((0..k).filter(|&i| mask & (1 << i) != 0).map(|i| others[i]))
                .collect();
            if let Some(b) = &expected { if h.len() >= b.len() { continue; } }
            if oracle_is_d_convex(&dag, &skeleton, &h) {
                expected = Some(h);
            }
        }
        let expected = expected.expect("full node set is always d-convex (M empty)");

        prop_assert_eq!(actual, expected);
    }

    /// `Dag::strong_d_convex_hull` matches the brute-force minimal strongly
    /// d-convex superset.
    #[test]
    fn prop_strong_d_convex_hull_matches_oracle((dag, required) in arb_dag_with_required(6)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let actual: BTreeSet<NodeId> = dag.strong_d_convex_hull(&required_vec).unwrap().into_iter().collect();
        let expected = oracle_minimal_strong_d_convex_superset(&dag, &required);
        prop_assert_eq!(actual, expected);
    }

    /// `required` is always a subset of both hulls.
    #[test]
    fn prop_required_subset_of_hulls((dag, required) in arb_dag_with_required(7)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let plain = dag.d_convex_hull(&required_vec).unwrap();
        let strong = dag.strong_d_convex_hull(&required_vec).unwrap();
        for &r in &required {
            prop_assert!(plain.contains(&r));
            prop_assert!(strong.contains(&r));
        }
    }

    /// The strong hull is always a superset of the plain d-convex hull.
    #[test]
    fn prop_strong_hull_superset_of_plain_hull((dag, required) in arb_dag_with_required(7)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let plain = dag.d_convex_hull(&required_vec).unwrap();
        let strong = dag.strong_d_convex_hull(&required_vec).unwrap();
        prop_assert!(plain.iter().all(|n| strong.contains(n)));
    }

    /// Idempotence: taking the hull of an already-(strong-)d-convex set
    /// returns that same set.
    #[test]
    fn prop_hull_idempotent((dag, required) in arb_dag_with_required(7)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let hull = dag.d_convex_hull(&required_vec).unwrap();
        let hull_vec: Vec<NodeId> = hull.iter().copied().collect();
        let hull_of_hull = dag.d_convex_hull(&hull_vec).unwrap();
        prop_assert_eq!(hull, hull_of_hull);

        let strong = dag.strong_d_convex_hull(&required_vec).unwrap();
        let strong_vec: Vec<NodeId> = strong.iter().copied().collect();
        let strong_of_strong = dag.strong_d_convex_hull(&strong_vec).unwrap();
        prop_assert_eq!(strong, strong_of_strong);
    }
}

// ── Cpdag::strong_d_convex_hull invariance (Theorem 5) ──────────────────────

/// A random DAG together with the CPDAG derived from its own
/// compelled/non-compelled edge split (same construction as
/// `tests/cpdag_property_tests.rs`), plus a small required set. `dag` and
/// `cpdag.consistent_extension()` are two *independently obtained* members
/// of the same Markov equivalence class: `dag`'s non-compelled edges were
/// oriented by proptest's random generation, while the CPDAG's extension is
/// re-oriented from scratch by Dor-Tarsi (PR 1) -- these very often
/// disagree on at least one non-compelled edge, giving a genuine
/// two-different-extensions comparison rather than a self-comparison.
fn arb_dag_and_its_cpdag(
    max_n: usize,
) -> impl Strategy<Value = (Dag, causasv::Cpdag, BTreeSet<NodeId>)> {
    use causasv::Cpdag;
    (2usize..=max_n).prop_flat_map(|n| {
        let n_pairs = n * (n - 1) / 2;
        (
            prop::collection::vec(prop::bool::ANY, n_pairs),
            prop::collection::vec(prop::bool::ANY, n),
        )
            .prop_map(move |(include, req_mask)| {
                let mut dag = Dag::new();
                for i in 0..n {
                    dag.add_node(&format!("n{i}"));
                }
                let mut pairs = Vec::new();
                let mut k = 0;
                for i in 0..n {
                    for j in (i + 1)..n {
                        if include[k] {
                            let _ = dag.add_edge(NodeId(i as u32), NodeId(j as u32));
                            pairs.push((i, j));
                        }
                        k += 1;
                    }
                }
                // Every edge that is NOT part of an unshielded collider (a
                // "free" edge) becomes undirected in the CPDAG; v-structure
                // edges stay directed (compelled).
                let is_compelled = |p: usize, c: usize| -> bool {
                    dag.parents(NodeId(c as u32))
                        .unwrap()
                        .iter()
                        .any(|&p2| p2.0 as usize != p && !adjacent(&dag, p2, NodeId(p as u32)))
                };
                let mut cpdag = Cpdag::new();
                for i in 0..n {
                    cpdag.add_node(&format!("n{i}"));
                }
                for &(i, j) in &pairs {
                    if is_compelled(i, j) {
                        let _ = cpdag.add_directed_edge(NodeId(i as u32), NodeId(j as u32));
                    } else {
                        let _ = cpdag.add_undirected_edge(NodeId(i as u32), NodeId(j as u32));
                    }
                }
                let mut required: BTreeSet<NodeId> = (0..n)
                    .filter(|&i| req_mask[i])
                    .map(|i| NodeId(i as u32))
                    .collect();
                if required.is_empty() {
                    required.insert(NodeId(0));
                }
                (dag, cpdag, required)
            })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Theorem 5: the strong d-convex hull's vertex set is identical across
    /// every DAG in a CPDAG's Markov equivalence class. Compares the
    /// original randomly-oriented `dag` against the CPDAG's own
    /// deterministically re-oriented `consistent_extension()` -- two
    /// independently obtained equivalence-class members.
    #[test]
    fn prop_strong_hull_invariant_across_equivalent_dags((dag, cpdag, required) in arb_dag_and_its_cpdag(6)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let via_original = dag.strong_d_convex_hull(&required_vec).unwrap();
        let extension = cpdag.consistent_extension().unwrap();
        let via_extension = extension.strong_d_convex_hull(&required_vec).unwrap();
        prop_assert_eq!(via_original, via_extension);
    }

    /// `Cpdag::strong_d_convex_hull` is exactly the delegation it claims to
    /// be: same result as computing on its own consistent extension
    /// directly.
    #[test]
    fn prop_cpdag_hull_matches_own_extension((_dag, cpdag, required) in arb_dag_and_its_cpdag(6)) {
        let required_vec: Vec<NodeId> = required.iter().copied().collect();
        let via_cpdag = cpdag.strong_d_convex_hull(&required_vec).unwrap();
        let via_extension = cpdag.consistent_extension().unwrap().strong_d_convex_hull(&required_vec).unwrap();
        prop_assert_eq!(via_cpdag, via_extension);
    }
}
