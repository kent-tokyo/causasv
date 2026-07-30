//! Error type shared by every fallible operation in this crate: graph
//! construction, validation, and ASV computation (exact and approximate).

use crate::graph::NodeId;

/// All ways a causasv operation can fail.
///
/// Returned by `Dag::add_edge`/`Dag::validate` (structural errors), the
/// `exact*`/`approximate*` family on `AsvExplainer` (value-function and
/// configuration errors), and propagated across the PyO3 boundary as
/// `PyValueError` (see `src/python.rs`) rather than panicking.
#[derive(Debug, thiserror::Error)]
pub enum CausasvError {
    /// The graph has a cycle; ASV requires a valid DAG. Raised by `Dag::validate`.
    #[error("cycle detected in graph")]
    CycleDetected,
    /// A `NodeId` doesn't exist in this DAG (e.g. from a stale or foreign id).
    #[error("invalid node id: {0:?}")]
    InvalidNodeId(NodeId),
    /// `Dag::add_edge` was called with the same node as both endpoints.
    #[error("self-loop on node {0:?}")]
    SelfLoop(NodeId),
    /// `Dag::add_edge` was called twice for the same ordered pair of nodes.
    #[error("duplicate edge {0:?} -> {1:?}")]
    DuplicateEdge(NodeId, NodeId),
    /// The DAG has no nodes; ASV is undefined for an empty graph.
    #[error("graph is empty")]
    EmptyGraph,
    /// `exact_tree` was called on a DAG that isn't a single-root rooted tree
    /// (use `exact_dag`/`exact_dag_sparse` for general DAGs instead).
    #[error("graph is not a rooted directed tree")]
    NotRootedTree,
    /// The user-supplied value function returned an error for some coalition.
    #[error("value function error: {0}")]
    ValueFunctionError(String),
    /// A method's configuration is unusable for this DAG (e.g. `n` exceeds
    /// `ExactDagConfig::max_nodes`, or the sparse DP's memory guard was hit).
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// The number of linear extensions overflowed `u64` during exact
    /// computation; callers should fall back to an approximate method.
    #[error("linear extension count overflowed u64: {0}")]
    Overflow(String),
    /// `exact_tree`'s feasibility preflight rejected this tree's *shape* (not just
    /// its node count): some ancestor level's side-sibling order-ideal product, or
    /// the total work summed across the whole tree, exceeds the configured
    /// `ExactTreeConfig` budget. Raised before any cartesian product is built —
    /// unlike a node-count limit, this catches wide/deep ("bushy") trees that are
    /// small in `n` but combinatorially explosive to enumerate. Callers should
    /// fall back to `exact_dag_sparse` or an approximate method.
    #[error(
        "exact_tree: tree shape exceeds the feasibility budget (largest single side-sibling \
         product {max_cartesian_terms} vs budget {cartesian_term_budget}; estimated total work \
         {estimated_total_terms} vs budget {total_term_budget}); use exact_dag_sparse or approx"
    )]
    ExactTreeBudgetExceeded {
        max_cartesian_terms: u64,
        cartesian_term_budget: u64,
        estimated_total_terms: u64,
        total_term_budget: u64,
    },
    /// A `Cpdag::add_directed_edge`/`add_undirected_edge` call conflicts with an
    /// edge that already exists between the same pair of nodes (opposite
    /// direction, or directed vs. undirected).
    #[error(
        "conflicting edge between {0:?} and {1:?}: an edge with a different orientation already exists"
    )]
    ConflictingEdge(NodeId, NodeId),
    /// `Cpdag::consistent_extension` (Dor-Tarsi PDAG-to-DAG extension) found no
    /// valid fully-directed acyclic extension — the PDAG's undirected chain
    /// components aren't consistently orientable (e.g. a chordless undirected
    /// cycle). `Cpdag::validate_cpdag` uses this as its extendability oracle.
    #[error(
        "PDAG has no consistent DAG extension (undirected edges can't be oriented without creating a cycle or a new v-structure)"
    )]
    NotExtendable,
    /// `Dag::d_convex_hull`/`Dag::strong_d_convex_hull` didn't reach a fixed
    /// point within `node_count() + 1` iterations. The paper's own
    /// termination proof bounds both loops at `|V|` iterations for any valid
    /// DAG, so this indicates an implementation bug, not a legitimate
    /// large-graph timeout — please file an issue with a reproduction.
    #[error(
        "d-convex hull computation did not reach a fixed point within the expected iteration bound; this indicates an implementation bug"
    )]
    HullFixedPointNotReached,
}

/// Every batched approximate path (small `u64` backend and large
/// `LargeCoalition` backend alike) hands a caller-supplied `value_fn_batch`
/// a list of coalitions and expects exactly one value back per coalition,
/// then pairs them up positionally (`Iterator::zip`). `zip` silently stops
/// at the shorter side, so a callback that returns too few values would
/// otherwise leave the extra coalitions uncached (and, on the small path,
/// panic later at an `unwrap()` lookup) rather than surfacing the
/// callback's bug; one returning too many silently drops the tail. Call
/// this before zipping so a length mismatch is a `CausasvError`, not a
/// panic or a quietly wrong estimate.
pub(crate) fn validate_batch_result_len(
    expected: usize,
    actual: usize,
) -> Result<(), CausasvError> {
    if expected != actual {
        return Err(CausasvError::ValueFunctionError(format!(
            "value_fn_batch returned {actual} values for {expected} coalitions"
        )));
    }
    Ok(())
}
