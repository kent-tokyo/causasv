//! Completed partial DAG (CPDAG) representation and PDAG→DAG extension.
//!
//! A [`Cpdag`] mirrors [`Dag`]'s node-indexing conventions (insertion-order
//! `NodeId`s, `Vec`-of-`Vec` adjacency, no hash-order-dependent iteration) but
//! stores two edge kinds: directed edges (compelled orientations) and
//! undirected edges (orientation not determined by the data). This crate does
//! not perform causal discovery — a `Cpdag` must be supplied by the caller
//! (e.g. the output of an external constraint-based structure-learning
//! method), consistent with the non-goal stated in `AGENTS.md`.
//!
//! [`Cpdag::consistent_extension`] implements the classical PDAG-to-DAG
//! extension algorithm (Dor & Tarsi, *"A simple algorithm to construct a
//! consistent extension,"* 1992; formalized as Algorithm 2 in Chickering,
//! *"Learning Equivalence Classes of Bayesian-Network Structures,"* JMLR
//! 2002). This is an independent Rust implementation of that well-established,
//! 30-year-old algorithm, written directly from its published description —
//! no third-party source code was copied — and is unrelated to (and predates)
//! any paper-specific licensing concerns that apply to later CPDAG-reduction
//! work built on top of this type.

use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use indexmap::IndexMap;

/// Partially directed graph: some edges are compelled (directed), others are
/// unresolved (undirected). Node indexing mirrors [`Dag`]: stable `NodeId`s
/// assigned in insertion order.
#[derive(Clone, Debug)]
pub struct Cpdag {
    name_to_id: IndexMap<String, NodeId>,
    names: Vec<String>,
    directed_children: Vec<Vec<NodeId>>,
    directed_parents: Vec<Vec<NodeId>>,
    undirected: Vec<Vec<NodeId>>,
}

impl Cpdag {
    /// Create an empty PDAG. Nodes and edges are added via `add_node`/
    /// `add_directed_edge`/`add_undirected_edge`.
    pub fn new() -> Self {
        Self {
            name_to_id: IndexMap::new(),
            names: Vec::new(),
            directed_children: Vec::new(),
            directed_parents: Vec::new(),
            undirected: Vec::new(),
        }
    }

    /// Add a node with the given name. Returns the existing `NodeId` if the
    /// name already exists.
    pub fn add_node(&mut self, name: &str) -> NodeId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = NodeId(self.names.len() as u32);
        self.name_to_id.insert(name.to_string(), id);
        self.names.push(name.to_string());
        self.directed_children.push(Vec::new());
        self.directed_parents.push(Vec::new());
        self.undirected.push(Vec::new());
        id
    }

    /// Add a compelled (directed) edge from `from` to `to`.
    ///
    /// Errors: `SelfLoop` if `from == to`; `InvalidNodeId` for an unknown id;
    /// `DuplicateEdge` if this exact directed edge already exists;
    /// `ConflictingEdge` if the same pair already has an edge with a
    /// different orientation (reverse-directed, or undirected).
    pub fn add_directed_edge(&mut self, from: NodeId, to: NodeId) -> Result<(), CausasvError> {
        if from == to {
            return Err(CausasvError::SelfLoop(from));
        }
        self.check_id(from)?;
        self.check_id(to)?;
        if self.directed_children[from.idx()].contains(&to) {
            return Err(CausasvError::DuplicateEdge(from, to));
        }
        if self.directed_children[to.idx()].contains(&from)
            || self.undirected[from.idx()].contains(&to)
        {
            return Err(CausasvError::ConflictingEdge(from, to));
        }
        self.directed_children[from.idx()].push(to);
        self.directed_parents[to.idx()].push(from);
        Ok(())
    }

    /// Add an unresolved (undirected) edge between `a` and `b`.
    ///
    /// Errors: `SelfLoop` if `a == b`; `InvalidNodeId` for an unknown id;
    /// `DuplicateEdge` if this undirected edge already exists;
    /// `ConflictingEdge` if a directed edge already exists between the pair.
    pub fn add_undirected_edge(&mut self, a: NodeId, b: NodeId) -> Result<(), CausasvError> {
        if a == b {
            return Err(CausasvError::SelfLoop(a));
        }
        self.check_id(a)?;
        self.check_id(b)?;
        if self.undirected[a.idx()].contains(&b) {
            return Err(CausasvError::DuplicateEdge(a, b));
        }
        if self.directed_children[a.idx()].contains(&b)
            || self.directed_children[b.idx()].contains(&a)
        {
            return Err(CausasvError::ConflictingEdge(a, b));
        }
        self.undirected[a.idx()].push(b);
        self.undirected[b.idx()].push(a);
        Ok(())
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.names.len()
    }

    /// Look up a node's name by id, or `None` if the id is out of range.
    pub fn node_name(&self, id: NodeId) -> Option<&str> {
        self.names.get(id.idx()).map(String::as_str)
    }

    /// Look up a node's id by name, or `None` if no such node exists.
    pub fn node_id(&self, name: &str) -> Option<NodeId> {
        self.name_to_id.get(name).copied()
    }

    /// Iterate over every node id in insertion order.
    pub fn all_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        (0..self.names.len()).map(|i| NodeId(i as u32))
    }

    /// Iterate over directed edges as `(from, to)` pairs, in deterministic
    /// (insertion) order.
    pub fn directed_edges(&self) -> impl Iterator<Item = (NodeId, NodeId)> + '_ {
        self.all_nodes().flat_map(move |from| {
            self.directed_children[from.idx()]
                .iter()
                .map(move |&to| (from, to))
        })
    }

    /// Iterate over undirected edges as `(a, b)` pairs with `a < b`, in
    /// deterministic order (one entry per pair, not two).
    pub fn undirected_edges(&self) -> impl Iterator<Item = (NodeId, NodeId)> + '_ {
        self.all_nodes().flat_map(move |a| {
            self.undirected[a.idx()]
                .iter()
                .filter(move |&&b| b > a)
                .map(move |&b| (a, b))
        })
    }

    /// Cheap structural validation: non-empty, and the directed-edge-only
    /// subgraph has no cycle. Does **not** verify that this is a genuine
    /// CPDAG (see [`Cpdag::validate_cpdag`] for that stronger, still-partial,
    /// guarantee).
    pub fn validate_pdag(&self) -> Result<(), CausasvError> {
        if self.names.is_empty() {
            return Err(CausasvError::EmptyGraph);
        }
        let mut in_deg: Vec<usize> = self.directed_parents.iter().map(Vec::len).collect();
        let mut queue: std::collections::VecDeque<NodeId> = in_deg
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(i, _)| NodeId(i as u32))
            .collect();
        let mut visited = 0usize;
        while let Some(node) = queue.pop_front() {
            visited += 1;
            for &child in &self.directed_children[node.idx()] {
                in_deg[child.idx()] -= 1;
                if in_deg[child.idx()] == 0 {
                    queue.push_back(child);
                }
            }
        }
        if visited != self.names.len() {
            return Err(CausasvError::CycleDetected);
        }
        Ok(())
    }

    /// `validate_pdag` plus: a consistent DAG extension exists (see
    /// [`Cpdag::consistent_extension`]).
    ///
    /// **What this does not guarantee**: that the directed edges are exactly
    /// the *compelled* edges of some genuine Markov equivalence class. A PDAG
    /// whose undirected chain components are extendable but whose directed
    /// edges don't correspond to any real equivalence class (e.g. one that
    /// wasn't produced by a correct discovery/orientation procedure) can
    /// still pass this check. Confirming that would require re-running
    /// Meek's orientation rules forward, which is out of scope for this type
    /// — it represents a caller-supplied graph, not one this crate infers.
    pub fn validate_cpdag(&self) -> Result<(), CausasvError> {
        self.validate_pdag()?;
        self.consistent_extension().map(|_| ())
    }

    /// Construct a consistent DAG extension via the Dor-Tarsi / Chickering
    /// PDAG-to-DAG algorithm: repeatedly find a node `v` that (a) has no
    /// directed edge leaving it to another unprocessed node, and (b) whose
    /// unprocessed neighbors (directed-parents and undirected-neighbors)
    /// form a clique together with `v` — i.e. orienting every remaining
    /// undirected edge at `v` as pointing into `v` introduces no new
    /// v-structure. Orient those edges, mark `v` processed, and repeat.
    ///
    /// Returns `Err(NotExtendable)` if no such node exists at some step
    /// (e.g. a chordless undirected cycle) — that failure is itself the
    /// extendability check `validate_cpdag` relies on.
    ///
    /// Node/edge selection is deterministic: candidates are scanned in
    /// increasing `NodeId` order, so the same input always produces the
    /// same extension regardless of insertion order elsewhere.
    pub fn consistent_extension(&self) -> Result<Dag, CausasvError> {
        let n = self.node_count();
        if n == 0 {
            return Err(CausasvError::EmptyGraph);
        }
        let mut active = vec![true; n];
        let mut remaining_undirected: Vec<Vec<NodeId>> = self.undirected.clone();
        let mut oriented: Vec<(NodeId, NodeId)> = self.directed_edges().collect();

        for _ in 0..n {
            let v = self.find_extension_candidate(&active, &remaining_undirected)?;
            for &w in remaining_undirected[v.idx()].iter() {
                if active[w.idx()] {
                    oriented.push((w, v));
                }
            }
            let neighbors = std::mem::take(&mut remaining_undirected[v.idx()]);
            for w in neighbors {
                remaining_undirected[w.idx()].retain(|&x| x != v);
            }
            active[v.idx()] = false;
        }

        let mut dag = Dag::new();
        for name in &self.names {
            dag.add_node(name);
        }
        for (from, to) in oriented {
            dag.add_edge(from, to)?;
        }
        dag.validate()?;
        Ok(dag)
    }

    fn find_extension_candidate(
        &self,
        active: &[bool],
        remaining_undirected: &[Vec<NodeId>],
    ) -> Result<NodeId, CausasvError> {
        'candidates: for i in 0..active.len() {
            if !active[i] {
                continue;
            }
            let v = NodeId(i as u32);
            // (a) v is a sink: no directed edge to a still-active node.
            if self.directed_children[v.idx()]
                .iter()
                .any(|&w| active[w.idx()])
            {
                continue;
            }
            // (b) for every undirected edge {v, w} about to be oriented into
            // v, w must be adjacent to every OTHER neighbor of v (directed
            // parent or undirected) — otherwise orienting w -> v would
            // create a new v-structure. Pairs of already-directed parents
            // are not checked against each other: any v-structure among
            // them is pre-existing/compelled, not newly introduced here.
            let directed_neighbors: Vec<NodeId> = self.directed_parents[v.idx()]
                .iter()
                .copied()
                .filter(|&w| active[w.idx()])
                .collect();
            let undirected_neighbors = &remaining_undirected[v.idx()];
            for &w in undirected_neighbors {
                let other_neighbors = directed_neighbors
                    .iter()
                    .copied()
                    .chain(undirected_neighbors.iter().copied())
                    .filter(|&y| y != w);
                for y in other_neighbors {
                    if !self.adjacent(w, y) {
                        continue 'candidates;
                    }
                }
            }
            return Ok(v);
        }
        Err(CausasvError::NotExtendable)
    }

    fn adjacent(&self, a: NodeId, b: NodeId) -> bool {
        self.directed_children[a.idx()].contains(&b)
            || self.directed_children[b.idx()].contains(&a)
            || self.undirected[a.idx()].contains(&b)
    }

    /// Return the sub-PDAG induced by `keep`: nodes not in `keep` are
    /// dropped, along with every edge touching a dropped node. Edge kind
    /// (directed/undirected) is preserved for retained pairs. Node order in
    /// the result follows this graph's original insertion order, not
    /// `keep`'s order.
    ///
    /// Errors with `InvalidNodeId` if `keep` contains an id not in this
    /// graph.
    pub fn induced_subgraph(&self, keep: &[NodeId]) -> Result<Cpdag, CausasvError> {
        let mut keep_mask = vec![false; self.node_count()];
        for &id in keep {
            self.check_id(id)?;
            keep_mask[id.idx()] = true;
        }
        let mut sub = Cpdag::new();
        for id in self.all_nodes() {
            if keep_mask[id.idx()] {
                sub.add_node(&self.names[id.idx()]);
            }
        }
        for (from, to) in self.directed_edges() {
            if keep_mask[from.idx()] && keep_mask[to.idx()] {
                let new_from = sub.node_id(&self.names[from.idx()]).unwrap();
                let new_to = sub.node_id(&self.names[to.idx()]).unwrap();
                sub.add_directed_edge(new_from, new_to)?;
            }
        }
        for (a, b) in self.undirected_edges() {
            if keep_mask[a.idx()] && keep_mask[b.idx()] {
                let new_a = sub.node_id(&self.names[a.idx()]).unwrap();
                let new_b = sub.node_id(&self.names[b.idx()]).unwrap();
                sub.add_undirected_edge(new_a, new_b)?;
            }
        }
        Ok(sub)
    }

    /// The strong d-convex hull of `required`, computed by picking one
    /// consistent DAG extension (via [`Cpdag::consistent_extension`]) and
    /// computing its strong d-convex hull there.
    ///
    /// This is sound for any CPDAG because the paper's Theorem 5 proves the
    /// strong d-convex hull's *vertex set* is identical across every DAG in
    /// a CPDAG's Markov equivalence class — so a single extension suffices;
    /// there's no need to enumerate all of them. (That invariance holds for
    /// target pairs the paper's Theorem 5 covers — see
    /// `docs/strong_d_convex_hulls.md` for the exact precondition.)
    pub fn strong_d_convex_hull(
        &self,
        required: &[NodeId],
    ) -> Result<indexmap::IndexSet<NodeId>, CausasvError> {
        self.consistent_extension()?.strong_d_convex_hull(required)
    }

    fn check_id(&self, id: NodeId) -> Result<(), CausasvError> {
        if id.idx() >= self.names.len() {
            Err(CausasvError::InvalidNodeId(id))
        } else {
            Ok(())
        }
    }
}

impl Default for Cpdag {
    fn default() -> Self {
        Self::new()
    }
}

trait NodeIdExt {
    fn idx(self) -> usize;
}

impl NodeIdExt for NodeId {
    fn idx(self) -> usize {
        self.0 as usize
    }
}
