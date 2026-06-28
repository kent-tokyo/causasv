use crate::error::CausasvError;
use indexmap::IndexMap;

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

    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    pub(crate) fn edge_count(&self) -> usize {
        self.children.iter().map(|c| c.len()).sum()
    }

    pub fn node_name(&self, id: NodeId) -> Option<&str> {
        self.names.get(id.0 as usize).map(String::as_str)
    }

    pub fn node_id(&self, name: &str) -> Option<NodeId> {
        self.name_to_id.get(name).copied()
    }

    pub fn children(&self, id: NodeId) -> Result<&[NodeId], CausasvError> {
        self.check_id(id)?;
        Ok(&self.children[id.0 as usize])
    }

    pub fn parents(&self, id: NodeId) -> Result<&[NodeId], CausasvError> {
        self.check_id(id)?;
        Ok(&self.parents[id.0 as usize])
    }

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
