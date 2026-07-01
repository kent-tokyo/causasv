//! The causal DAG data structure (`Dag`) and its node handle (`NodeId`).
//! Every other module (topological ordering, exact DP, sampling) operates on
//! the adjacency lists exposed here.

use crate::error::CausasvError;
use indexmap::IndexMap;

/// Stable handle to a node in a `Dag`, assigned in insertion order starting at 0.
/// Valid only for the `Dag` that created it — passing a `NodeId` from a
/// different (or stale) `Dag` returns `CausasvError::InvalidNodeId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}

/// Directed acyclic graph with stable node indexing.
#[derive(Clone, Debug)]
pub struct Dag {
    // ponytail: IndexMap only here for name→id lookup with insertion-order iteration
    name_to_id: IndexMap<String, NodeId>,
    names: Vec<String>,
    children: Vec<Vec<NodeId>>,
    parents: Vec<Vec<NodeId>>,
}

impl Dag {
    /// Create an empty DAG. Nodes and edges are added via `add_node`/`add_edge`.
    pub fn new() -> Self {
        Self {
            name_to_id: IndexMap::new(),
            names: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
        }
    }

    /// Add a node with the given name. Returns existing NodeId if name already exists.
    pub fn add_node(&mut self, name: &str) -> NodeId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = NodeId(self.names.len() as u32);
        self.name_to_id.insert(name.to_string(), id);
        self.names.push(name.to_string());
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
        id
    }

    /// Add a directed edge from `from` to `to`.
    pub fn add_edge(&mut self, from: NodeId, to: NodeId) -> Result<(), CausasvError> {
        if from == to {
            return Err(CausasvError::SelfLoop(from));
        }
        self.check_id(from)?;
        self.check_id(to)?;
        if self.children[from.0 as usize].contains(&to) {
            return Err(CausasvError::DuplicateEdge(from, to));
        }
        self.children[from.0 as usize].push(to);
        self.parents[to.0 as usize].push(from);
        Ok(())
    }

    /// Check for empty graph and cycles. Call before ASV computation.
    pub fn validate(&self) -> Result<(), CausasvError> {
        if self.names.is_empty() {
            return Err(CausasvError::EmptyGraph);
        }
        let mut in_deg = self.in_degrees();
        let mut queue: std::collections::VecDeque<NodeId> = in_deg
            .iter()
            .enumerate()
            .filter(|&(_, d)| *d == 0)
            .map(|(i, _)| NodeId(i as u32))
            .collect();
        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            for &child in &self.children[node.0 as usize] {
                in_deg[child.0 as usize] -= 1;
                if in_deg[child.0 as usize] == 0 {
                    queue.push_back(child);
                }
            }
        }
        if visited != self.names.len() {
            return Err(CausasvError::CycleDetected);
        }
        Ok(())
    }

    /// Number of nodes in the DAG.
    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    /// Total number of edges, used by `auto`'s sparse-vs-dense heuristic (`m ≤ 2n`).
    pub(crate) fn edge_count(&self) -> usize {
        self.children.iter().map(|c| c.len()).sum()
    }

    /// Look up a node's name by id, or `None` if the id is out of range.
    pub fn node_name(&self, id: NodeId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    /// Look up a node's id by name, or `None` if no such node exists.
    pub fn node_id(&self, name: &str) -> Option<NodeId> {
        self.name_to_id.get(name).copied()
    }

    /// Direct children of `id` (nodes `id` has an edge to), or
    /// `CausasvError::InvalidNodeId` if `id` doesn't exist in this DAG.
    pub fn children(&self, id: NodeId) -> Result<&[NodeId], CausasvError> {
        self.check_id(id)?;
        Ok(&self.children[id.0 as usize])
    }

    /// Direct parents of `id` (nodes with an edge to `id`), or
    /// `CausasvError::InvalidNodeId` if `id` doesn't exist in this DAG.
    pub fn parents(&self, id: NodeId) -> Result<&[NodeId], CausasvError> {
        self.check_id(id)?;
        Ok(&self.parents[id.0 as usize])
    }

    /// Iterate over every node id in insertion order.
    pub fn all_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.names.len()).map(|i| NodeId(i as u32))
    }

    pub(crate) fn in_degrees(&self) -> Vec<usize> {
        self.parents.iter().map(|p| p.len()).collect()
    }

    pub(crate) fn children_raw(&self, id: NodeId) -> &[NodeId] {
        &self.children[id.0 as usize]
    }

    pub(crate) fn parents_raw(&self, id: NodeId) -> &[NodeId] {
        &self.parents[id.0 as usize]
    }

    fn check_id(&self, id: NodeId) -> Result<(), CausasvError> {
        if id.0 as usize >= self.names.len() {
            Err(CausasvError::InvalidNodeId(id))
        } else {
            Ok(())
        }
    }
}

impl Default for Dag {
    fn default() -> Self {
        Self::new()
    }
}
