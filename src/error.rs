use crate::graph::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum CausasvError {
    #[error("cycle detected in graph")]
    CycleDetected,
    #[error("invalid node id: {0:?}")]
    InvalidNodeId(NodeId),
    #[error("self-loop on node {0:?}")]
    SelfLoop(NodeId),
    #[error("duplicate edge {0:?} -> {1:?}")]
    DuplicateEdge(NodeId, NodeId),
    #[error("graph is empty")]
    EmptyGraph,
    #[error("graph is not a rooted directed tree")]
    NotRootedTree,
    #[error("value function error: {0}")]
    ValueFunctionError(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("linear extension count overflowed u64: {0}")]
    Overflow(String),
}
