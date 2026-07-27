/// Property-based tests for `Cpdag::consistent_extension`, using the same
/// `proptest` style as `tests/property_tests.rs`.
use causasv::{Cpdag, Dag, NodeId};
use proptest::prelude::*;

/// Generate a random valid CPDAG that is *guaranteed extendable* by
/// construction: start from a random DAG (same generator shape as
/// `arb_dag` in `tests/property_tests.rs`), then convert every edge that is
/// not part of a v-structure (i.e. whose two endpoints' other in-edges don't
/// create an unshielded collider) to undirected. This mirrors how a real
/// CPDAG is derived from a DAG's Markov equivalence class: compelled edges
/// (v-structure edges) stay directed, everything else may be undirected.
fn arb_extendable_cpdag(max_n: usize) -> impl Strategy<Value = (Cpdag, usize)> {
    (2usize..=max_n).prop_flat_map(|n| {
        let n_pairs = n * (n - 1) / 2;
        (
            prop::collection::vec(prop::bool::ANY, n_pairs),
            prop::collection::vec(prop::bool::ANY, n_pairs),
        )
            .prop_map(move |(include, undirect)| {
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
                // A directed edge p -> c is a compelled v-structure edge if c
                // has another parent p2 that is NOT adjacent to p (unshielded
                // collider p -> c <- p2). Keep those directed; everything
                // else is eligible to become undirected.
                let is_compelled = |p: usize, c: usize| -> bool {
                    dag.parents(NodeId(c as u32))
                        .unwrap()
                        .iter()
                        .any(|&p2| p2.0 as usize != p && !dag_adjacent(&dag, p2.0 as usize, p))
                };
                let mut cpdag = Cpdag::new();
                for i in 0..n {
                    cpdag.add_node(&format!("n{i}"));
                }
                for (idx, &(i, j)) in pairs.iter().enumerate() {
                    let compelled = is_compelled(i, j);
                    if compelled || !undirect[idx] {
                        let _ = cpdag.add_directed_edge(NodeId(i as u32), NodeId(j as u32));
                    } else {
                        let _ = cpdag.add_undirected_edge(NodeId(i as u32), NodeId(j as u32));
                    }
                }
                (cpdag, n)
            })
    })
}

fn dag_adjacent(dag: &Dag, a: usize, b: usize) -> bool {
    dag.children(NodeId(a as u32))
        .unwrap()
        .contains(&NodeId(b as u32))
        || dag
            .children(NodeId(b as u32))
            .unwrap()
            .contains(&NodeId(a as u32))
}

proptest! {
    /// A CPDAG built this way must always have a consistent extension: it's
    /// derived directly from a real DAG's compelled/non-compelled edge split.
    #[test]
    fn prop_derived_cpdag_always_extendable((cpdag, n) in arb_extendable_cpdag(7)) {
        let dag = cpdag.consistent_extension();
        prop_assert!(dag.is_ok(), "expected extendable CPDAG, got {:?}", dag.err());
        let dag = dag.unwrap();
        prop_assert_eq!(dag.node_count(), n);
        prop_assert!(dag.validate().is_ok());
    }

    /// The extension's skeleton (edge count, ignoring orientation) matches
    /// the CPDAG's original skeleton size.
    #[test]
    fn prop_extension_preserves_skeleton_size((cpdag, _n) in arb_extendable_cpdag(7)) {
        let directed_count = cpdag.directed_edges().count();
        let undirected_count = cpdag.undirected_edges().count();
        let dag = cpdag.consistent_extension().unwrap();
        let dag_edge_count: usize = dag
            .all_nodes()
            .map(|id| dag.children(id).unwrap().len())
            .sum();
        prop_assert_eq!(dag_edge_count, directed_count + undirected_count);
    }

    /// Every originally-directed edge keeps its exact orientation in the
    /// extension (only undirected edges get newly oriented).
    #[test]
    fn prop_extension_preserves_directed_orientations((cpdag, _n) in arb_extendable_cpdag(7)) {
        let dag = cpdag.consistent_extension().unwrap();
        for (from, to) in cpdag.directed_edges() {
            prop_assert!(dag.children(from).unwrap().contains(&to));
        }
    }

    /// `validate_cpdag` agrees with `consistent_extension`'s success/failure.
    #[test]
    fn prop_validate_cpdag_matches_extension_result((cpdag, _n) in arb_extendable_cpdag(7)) {
        prop_assert_eq!(cpdag.validate_cpdag().is_ok(), cpdag.consistent_extension().is_ok());
    }
}
