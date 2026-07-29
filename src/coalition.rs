//! Growable bitset coalition representation for DAGs with more than 64 nodes.
//!
//! Everywhere else in this crate, a coalition is packed into a single `u64`
//! (`1u64 << node.0`), which cannot address a `NodeId >= 64`. That is the root
//! cause of the n > 64 `InvalidConfig` this module exists to remove from the
//! approximate ASV paths. `LargeCoalition` is the natural generalization of the
//! same scheme: `ceil(n / 64)` words, word `w` holding the bits for nodes
//! `[64w, 64w+64)`. The n ≤ 64 fast path is left untouched and keeps using raw
//! `u64` directly — this type is only constructed on the n > 64 side.
//!
//! An earlier draft of this module used a `CoalitionState { Small, Large }`
//! enum so the two representations shared one type. It was dropped: the n ≤ 64
//! hot path in `approx.rs` has no reason to route through an enum it doesn't
//! need, and a `Small` variant nothing ever constructs is dead code under
//! `-D warnings`. A concrete large-only type matches every real call site.

use crate::graph::NodeId;

/// Number of `u64` words needed to represent a coalition over `n` nodes.
pub(crate) fn words_for_n(n: usize) -> usize {
    n.div_ceil(64)
}

/// Convert a raw word-vector coalition key back into ascending `NodeId`s.
///
/// Free function (not only a `LargeCoalition` method) because the batched
/// large-DAG paths hold coalitions as bare `Box<[u64]>` cache keys after
/// deduplication, not as live `LargeCoalition`s, once it's time to build the
/// `Vec<Vec<NodeId>>` a `value_fn_batch` call expects.
pub(crate) fn words_to_sorted_nodes(words: &[u64]) -> Vec<NodeId> {
    let mut out = Vec::with_capacity(words.iter().map(|w| w.count_ones() as usize).sum());
    for (w, &word) in words.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            out.push(NodeId((w * 64 + b) as u32));
            bits &= bits - 1; // clear lowest set bit
        }
    }
    out
}

/// Mutable, growable bitset coalition for n > 64 DAGs.
///
/// The IS sampler only ever *inserts* nodes, one at a time, in the order a
/// sampled topological ordering names them — a sample's coalition is a
/// strictly growing prefix, never queried for arbitrary membership. `contains`
/// therefore exists for the `insert` invariant check and for tests, not as a
/// hot-path operation.
#[derive(Clone, Debug)]
pub(crate) struct LargeCoalition {
    words: Vec<u64>,
    n: usize,
}

impl LargeCoalition {
    pub(crate) fn empty(n_nodes: usize) -> Self {
        Self {
            words: vec![0u64; words_for_n(n_nodes)],
            n: n_nodes,
        }
    }

    pub(crate) fn insert(&mut self, node: NodeId) {
        let i = node.0 as usize;
        debug_assert!(
            i < self.n,
            "NodeId {i} out of range for coalition sized {}",
            self.n
        );
        self.words[i / 64] |= 1u64 << (i % 64);
        debug_assert!(self.contains(node), "insert did not set bit for {node:?}");
    }

    pub(crate) fn contains(&self, node: NodeId) -> bool {
        let i = node.0 as usize;
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Ascending `NodeId` order falls out of the word/bit layout by
    /// construction (word index, then bit index, both increase with node id)
    /// — no sort is needed, so there is no hash-iteration order to leak into
    /// the coalitions handed to a user's value function.
    pub(crate) fn to_sorted_nodes(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.count());
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                out.push(NodeId((w * 64 + b) as u32));
                bits &= bits - 1; // clear lowest set bit
            }
        }
        out
    }

    /// Immutable, hashable snapshot for use as a cache key. Allocates once —
    /// callers should only take a snapshot on a confirmed cache miss (see
    /// `LargeCoalitionCache`), not on every lookup (`words()` borrows for that).
    pub(crate) fn snapshot_key(&self) -> Box<[u64]> {
        self.words.clone().into_boxed_slice()
    }

    /// Borrowed word slice, for cache lookups that shouldn't allocate.
    /// `Box<[u64]>: Borrow<[u64]>`, so this can key a `HashMap<Box<[u64]>, _>`
    /// directly without building an owned key first.
    pub(crate) fn words(&self) -> &[u64] {
        &self.words
    }

    pub(crate) fn count(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_has_no_nodes() {
        let c = LargeCoalition::empty(130);
        assert_eq!(c.count(), 0);
        assert!(c.to_sorted_nodes().is_empty());
        assert!(!c.contains(NodeId(0)));
        assert!(!c.contains(NodeId(129)));
    }

    #[test]
    fn insert_spans_multiple_words() {
        // n=130 -> 3 words; node 129 lives in the third word.
        let mut c = LargeCoalition::empty(130);
        c.insert(NodeId(0));
        c.insert(NodeId(64));
        c.insert(NodeId(129));
        assert_eq!(c.count(), 3);
        assert!(c.contains(NodeId(0)));
        assert!(c.contains(NodeId(64)));
        assert!(c.contains(NodeId(129)));
        assert!(!c.contains(NodeId(63)));
        assert_eq!(
            c.to_sorted_nodes(),
            vec![NodeId(0), NodeId(64), NodeId(129)]
        );
    }

    #[test]
    fn to_sorted_nodes_is_ascending_regardless_of_insert_order() {
        let mut c = LargeCoalition::empty(200);
        for &i in &[150u32, 3, 70, 199, 0, 65] {
            c.insert(NodeId(i));
        }
        let nodes = c.to_sorted_nodes();
        let mut sorted = nodes.clone();
        sorted.sort_unstable();
        assert_eq!(nodes, sorted, "to_sorted_nodes must already be ascending");
    }

    #[test]
    fn snapshot_key_matches_words_and_is_usable_as_hashmap_key() {
        use std::collections::HashMap;
        let mut c = LargeCoalition::empty(100);
        c.insert(NodeId(5));
        c.insert(NodeId(80));
        let key = c.snapshot_key();
        assert_eq!(&*key, c.words());

        let mut map: HashMap<Box<[u64]>, f64> = HashMap::new();
        map.insert(key, 42.0);
        // Borrow<[u64]> lookup: no need to build an owned key to query.
        assert_eq!(map.get(c.words()), Some(&42.0));
    }

    #[test]
    fn words_for_n_rounds_up() {
        assert_eq!(words_for_n(1), 1);
        assert_eq!(words_for_n(64), 1);
        assert_eq!(words_for_n(65), 2);
        assert_eq!(words_for_n(128), 2);
        assert_eq!(words_for_n(129), 3);
        assert_eq!(words_for_n(256), 4);
    }
}
