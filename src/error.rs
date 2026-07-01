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
}
