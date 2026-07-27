//! d-convex hull and strong d-convex hull computation over DAGs.
//!
//! Independent implementation of the algorithms in:
//!
//! > Yuxin Deng, Yi Sun, Zhiming Li, Huaxiong Liu, *"Estimate Collapsibility
//! > of Causal Effects in Completed Partial DAGs via Strong d-Convex
//! > Hulls,"* arXiv:2606.08941, June 2026. DOI: 10.48550/arXiv.2606.08941.
//! > Licensed CC BY 4.0.
//!
//! This module implements the paper's mathematical definitions (a strongly
//! d-convex set has no inducing path for its target set, and every child of
//! the marginalized-out complement that's an ancestor of the target set is
//! "linearly ordered" — its parents are pairwise adjacent except when both
//! lie in the target set) and its CVM/ICHA/ISCHA algorithms (collect
//! shortest-inducing-path vertices via a moralized ancestor-restricted
//! subgraph; iterate to a d-convex fixed point; then iterate again absorbing
//! parents of any linear-ordering violation). Control flow, naming, and
//! tests here are original — this is not a translation of, and no source
//! was consulted from, the paper's own reference implementation
//! (`github.com/Jamyang-D/strongly-convex`, which carries no software
//! license). This module is not affiliated with or endorsed by the paper's
//! authors.
//!
//! **Scope**: this crate does not perform causal discovery — the DAG (or,
//! via [`crate::Cpdag::strong_d_convex_hull`], the CPDAG) is caller-supplied.
//! See `docs/strong_d_convex_hulls.md` for the paper's stated assumptions
//! (Gaussian/multinomial distributions, positivity, faithfulness) and scope
//! limits (proven only for non-adjacent target pairs; this is a preprint,
//! definitions may change in a future revision).

use crate::error::CausasvError;
use crate::graph::{Dag, NodeId};
use indexmap::IndexSet;
use std::collections::{HashMap, VecDeque};

impl Dag {
    /// The d-convex hull of `required`: the minimal superset of `required`
    /// containing every vertex on every shortest inducing path between
    /// non-adjacent pairs in `required`'s Markov boundary. Guarantees no
    /// inducing path exists for the result — the weaker of the two
    /// properties `strong_d_convex_hull` guarantees.
    ///
    /// Errors with `InvalidNodeId` if `required` contains an id not in this
    /// DAG.
    pub fn d_convex_hull(&self, required: &[NodeId]) -> Result<IndexSet<NodeId>, CausasvError> {
        for &id in required {
            self.check_id(id)?;
        }
        let h: IndexSet<NodeId> = required.iter().copied().collect();
        self.d_convex_closure(h)
    }

    /// The strong d-convex hull of `required`: the minimal superset that is
    /// both d-convex and has every child of the marginalized-out complement
    /// (that's an ancestor of the result) linearly ordered with respect to
    /// the result. This is the set that preserves a causal-effect estimate
    /// after marginalizing out every other variable (paper's Theorem 1).
    ///
    /// Errors with `InvalidNodeId` if `required` contains an id not in this
    /// DAG.
    pub fn strong_d_convex_hull(
        &self,
        required: &[NodeId],
    ) -> Result<IndexSet<NodeId>, CausasvError> {
        for &id in required {
            self.check_id(id)?;
        }
        let mut h: IndexSet<NodeId> = required.iter().copied().collect();
        let max_iterations = self.node_count() + 1;
        for _ in 0..max_iterations {
            h = self.d_convex_closure(h)?;
            let absorb = self.linear_ordering_violation_parents(&h);
            if absorb.is_empty() {
                return Ok(h);
            }
            h.extend(absorb);
        }
        Err(CausasvError::HullFixedPointNotReached)
    }

    /// Grow `h` by repeatedly absorbing shortest-inducing-path vertices
    /// until a fixed point is reached (no new vertex gets added). Note this
    /// checks *growth*, not raw non-emptiness of the per-round vertex set —
    /// a round can legitimately return a non-empty set of vertices that are
    /// already all in `h` (e.g. when two required vertices become directly
    /// adjacent after moralization, the "shortest path" between them is
    /// just the two endpoints themselves), and only a growth check
    /// terminates correctly in that case.
    fn d_convex_closure(&self, mut h: IndexSet<NodeId>) -> Result<IndexSet<NodeId>, CausasvError> {
        let max_iterations = self.node_count() + 1;
        for _ in 0..max_iterations {
            let absorbed = self.shortest_inducing_path_vertices(&h);
            let before = h.len();
            h.extend(absorbed);
            if h.len() == before {
                return Ok(h);
            }
        }
        Err(CausasvError::HullFixedPointNotReached)
    }

    /// One round of vertex collection: for every non-adjacent pair in
    /// `required`'s Markov boundary, find the shortest path between them in
    /// the moralized subgraph induced by their common ancestors (restricted
    /// to the pair plus the marginalized-out complement), and collect every
    /// vertex on that path.
    fn shortest_inducing_path_vertices(&self, required: &IndexSet<NodeId>) -> IndexSet<NodeId> {
        let marginalized: IndexSet<NodeId> =
            self.all_nodes().filter(|n| !required.contains(n)).collect();
        let boundary = self.markov_boundary_of_set(&marginalized);
        let candidates: Vec<NodeId> = required
            .iter()
            .copied()
            .filter(|n| boundary.contains(n))
            .collect();

        let mut collected = IndexSet::new();
        for i in 0..candidates.len() {
            for &b in &candidates[i + 1..] {
                let a = candidates[i];
                if self.adjacent(a, b) {
                    continue;
                }
                let pair: IndexSet<NodeId> = [a, b].into_iter().collect();
                let ancestral = self.ancestors_or_self(&pair);
                let moralized = self.moralize(&ancestral);
                let allowed: IndexSet<NodeId> = pair
                    .iter()
                    .copied()
                    .chain(marginalized.iter().copied())
                    .filter(|n| ancestral.contains(n))
                    .collect();
                if let Some(path) = shortest_path_restricted(&moralized, a, b, &allowed) {
                    collected.extend(path);
                }
            }
        }
        collected
    }

    /// Parents to absorb to fix linear-ordering violations (Definition 4):
    /// for every child (or self) of the marginalized-out complement that's
    /// an ancestor (or self) of `h`, find every pair of its parents that is
    /// neither adjacent nor both already in `h`, and collect *just that
    /// pair's two members* — not the vertex's entire parent set. A parent
    /// that's already adjacent to every other parent never causes a
    /// violation and must not be forced in; over-absorbing whole parent
    /// sets breaks minimality (a strict superset can satisfy Definition 5
    /// while a smaller set already did).
    fn linear_ordering_violation_parents(&self, h: &IndexSet<NodeId>) -> IndexSet<NodeId> {
        let marginalized: IndexSet<NodeId> = self.all_nodes().filter(|n| !h.contains(n)).collect();
        let children_of_marginalized = self.children_or_self(&marginalized);
        let ancestors_of_h = self.ancestors_or_self(h);

        let mut absorb = IndexSet::new();
        for &w in children_of_marginalized.intersection(&ancestors_of_h) {
            let parents = self.parents_raw(w);
            for i in 0..parents.len() {
                for &p2 in &parents[i + 1..] {
                    let p1 = parents[i];
                    if h.contains(&p1) && h.contains(&p2) {
                        continue;
                    }
                    if !self.adjacent(p1, p2) {
                        absorb.insert(p1);
                        absorb.insert(p2);
                    }
                }
            }
        }
        absorb
    }

    /// `An_G(seeds)`: ancestors of every vertex in `seeds`, plus `seeds`
    /// itself.
    fn ancestors_or_self(&self, seeds: &IndexSet<NodeId>) -> IndexSet<NodeId> {
        let mut result: IndexSet<NodeId> = seeds.clone();
        let mut queue: VecDeque<NodeId> = seeds.iter().copied().collect();
        while let Some(v) = queue.pop_front() {
            for &p in self.parents_raw(v) {
                if result.insert(p) {
                    queue.push_back(p);
                }
            }
        }
        result
    }

    /// `Ch_G(seeds)`: the *direct* children of every vertex in `seeds`, plus
    /// `seeds` itself (not the full descendant closure).
    fn children_or_self(&self, seeds: &IndexSet<NodeId>) -> IndexSet<NodeId> {
        let mut result: IndexSet<NodeId> = seeds.clone();
        for &v in seeds {
            for &c in self.children_raw(v) {
                result.insert(c);
            }
        }
        result
    }

    /// `mb_G(m)`: the Markov boundary of a vertex set — the parents,
    /// children, and co-parents (parents of children) of `m`, excluding `m`
    /// itself. Generalizes the single-vertex Markov blanket
    /// `mb_G(x) = fa_G(ch_G(x) ∪ {x}) ∖ {x}` pointwise over a set.
    fn markov_boundary_of_set(&self, m: &IndexSet<NodeId>) -> IndexSet<NodeId> {
        let closure = self.children_or_self(m);
        let mut result = IndexSet::new();
        for &v in &closure {
            result.insert(v);
            for &p in self.parents_raw(v) {
                result.insert(p);
            }
        }
        result.retain(|n| !m.contains(n));
        result
    }

    /// Adjacency ignoring direction: `a`/`b` are adjacent if either
    /// `a -> b` or `b -> a` exists.
    fn adjacent(&self, a: NodeId, b: NodeId) -> bool {
        self.children_raw(a).contains(&b) || self.children_raw(b).contains(&a)
    }

    /// Moralize the subgraph induced by `nodes`: keep every edge between two
    /// members of `nodes` (undirected), and additionally connect every pair
    /// of a common child's parents that both lie in `nodes` (a v-structure
    /// "marriage" edge — harmless no-op if that pair is already connected).
    /// Returns a symmetric adjacency list sized to `node_count()`; entries
    /// for vertices outside `nodes` are unused.
    fn moralize(&self, nodes: &IndexSet<NodeId>) -> Vec<Vec<NodeId>> {
        let mut adj: Vec<Vec<NodeId>> = vec![Vec::new(); self.node_count()];
        for &w in nodes {
            for &c in self.children_raw(w) {
                if nodes.contains(&c) {
                    add_undirected(&mut adj, w, c);
                }
            }
        }
        for &w in nodes {
            let parents: Vec<NodeId> = self
                .parents_raw(w)
                .iter()
                .copied()
                .filter(|p| nodes.contains(p))
                .collect();
            for i in 0..parents.len() {
                for &p2 in &parents[i + 1..] {
                    add_undirected(&mut adj, parents[i], p2);
                }
            }
        }
        adj
    }
}

fn add_undirected(adj: &mut [Vec<NodeId>], a: NodeId, b: NodeId) {
    if a != b && !adj[a.0 as usize].contains(&b) {
        adj[a.0 as usize].push(b);
        adj[b.0 as usize].push(a);
    }
}

/// Shortest path between `from` and `to` in an undirected adjacency list,
/// visiting only vertices in `allowed`. Deterministic: BFS visits each
/// node's neighbors in the adjacency list's stored order, which is itself
/// insertion-order deterministic.
fn shortest_path_restricted(
    adj: &[Vec<NodeId>],
    from: NodeId,
    to: NodeId,
    allowed: &IndexSet<NodeId>,
) -> Option<Vec<NodeId>> {
    if from == to {
        return Some(vec![from]);
    }
    let mut visited: IndexSet<NodeId> = IndexSet::new();
    visited.insert(from);
    let mut prev: HashMap<NodeId, NodeId> = HashMap::new();
    let mut queue: VecDeque<NodeId> = VecDeque::new();
    queue.push_back(from);
    while let Some(u) = queue.pop_front() {
        for &v in &adj[u.0 as usize] {
            if !allowed.contains(&v) || !visited.insert(v) {
                continue;
            }
            prev.insert(v, u);
            if v == to {
                let mut path = vec![to];
                let mut cur = to;
                while cur != from {
                    cur = prev[&cur];
                    path.push(cur);
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(v);
        }
    }
    None
}
