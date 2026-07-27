# Strong d-convex hulls — graph reduction

`Dag::d_convex_hull`, `Dag::strong_d_convex_hull`, and
`Cpdag::strong_d_convex_hull` (`src/d_convex.rs`) compute the minimal node
set that preserves a causal-effect estimate after marginalizing out every
other variable. This document is the primary citation and scope record for
that feature.

## Citation

> Yuxin Deng, Yi Sun, Zhiming Li, and Huaxiong Liu, *"Estimate Collapsibility
> of Causal Effects in Completed Partial DAGs via Strong d-Convex Hulls,"*
> arXiv:2606.08941, 2026. DOI: [10.48550/arXiv.2606.08941](https://doi.org/10.48550/arXiv.2606.08941).

The paper is licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

BibTeX:

```bibtex
@misc{deng2026estimate,
  title         = {Estimate Collapsibility of Causal Effects in Completed Partial DAGs via Strong d-Convex Hulls},
  author        = {Yuxin Deng and Yi Sun and Zhiming Li and Huaxiong Liu},
  year          = {2026},
  eprint        = {2606.08941},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ML},
  doi           = {10.48550/arXiv.2606.08941}
}
```

## Independent implementation

`src/d_convex.rs` is an independent implementation of this paper's
mathematical definitions and algorithms (Definitions 2–8, Theorem 5, and the
CVM/ICHA/ISCHA algorithms in its Section 3), written in causasv's own code
style and control flow. No source code was consulted from, or is derived
from, the paper's own reference implementation
(`github.com/Jamyang-D/strongly-convex`, which as of this writing carries no
software license — hence the clean-room approach). This module is **not
affiliated with or endorsed by the paper's authors**.

`causasv`'s own code remains licensed MIT OR Apache-2.0 regardless of the
paper's CC BY 4.0 license — only the paper's *mathematical content* was
used, not its text or figures verbatim (any adaptation of a definition into
this crate's documentation is a paraphrase, not a reproduction).

Test correctness is checked against an independently written brute-force
oracle (`tests/d_convex_property_tests.rs`, classical Pearl-1988 path
enumeration and collider-blocking d-separation, a different algorithm from
`src/d_convex.rs`'s moralization-based approach) — never against the
authors' reference implementation.

## What this computes

Given a DAG (or CPDAG) `G` and a target variable set `R`:

- `d_convex_hull(R)` — the minimal superset of `R` with no *inducing path*
  between any non-adjacent pair in the result (paper's Definition 5,
  condition (i); Algorithms 1–2, CVM/ICHA).
- `strong_d_convex_hull(R)` — the minimal superset that is additionally
  *linearly ordered*: every relevant child of the marginalized-out
  complement has pairwise-adjacent parents, except pairs already both in the
  result (Definition 5, condition (ii); Algorithm 3, ISCHA). This is the set
  that provably preserves a causal-effect estimate after marginalization
  (paper's Theorem 1).
- `Cpdag::strong_d_convex_hull(R)` — computed by picking one consistent DAG
  extension (`Cpdag::consistent_extension`, PR 1's Dor-Tarsi/Chickering
  algorithm) and computing the hull there. Sound because the paper's
  Theorem 5 proves the hull's *vertex set* is identical across every DAG in
  a CPDAG's Markov equivalence class — no need to enumerate extensions.

## Scope and assumptions

This module does **not** perform causal discovery. The DAG or CPDAG must be
supplied by the caller (e.g. the output of an external constraint-based
structure-learning method) — consistent with the non-goal stated in
`AGENTS.md`.

The paper's collapsibility guarantee (which is what makes the reduced graph
usable in place of the full one) is proven under conditions this crate
cannot verify from the graph alone — using `strong_d_convex_hull` responsibly
means the caller's data/estimation setup should actually satisfy them:

- **Distribution family**: every distribution in the model is either
  multinomial or Gaussian (paper's Assumption 1).
- **Positivity**: every joint distribution is positive (Assumption 2).
- **Faithfulness**: at least one distribution in the model is faithful to
  the graph.
- **Non-adjacent target pairs**: the CPDAG invariance result (Theorem 5) and
  the causal-estimate-collapsibility result (Theorem 3) are both stated for
  a target set `R` containing at least one pair `{X, Y}` that is
  **non-adjacent** in the graph. A target set without such a pair (e.g. a
  single node, or a set of only mutually adjacent nodes) is not covered by
  those specific theorems — `d_convex_hull`/`strong_d_convex_hull` still
  compute a well-defined minimal set per Definition 5/6 either way, but the
  causal-estimation and cross-extension-invariance guarantees the paper
  proves are stated for the non-adjacent-pair case.
- **No latent variables**: the paper's results assume a fully observed
  causal Bayesian network.

This is based on **preprint v1** (submitted June 2026). Definitions or
algorithms may change in a future revision of the paper; this
implementation was written against the version cited above.

Graph reduction (`d_convex_hull`/`strong_d_convex_hull`/`induced_subgraph`)
is implemented. IDA-based causal-effect estimation on the reduced graph
(the paper's Algorithm 4, "Subgraph IDA for CPDAGs") is **not** implemented
— this crate currently only performs the graph reduction step.

## Testing

- `tests/d_convex_tests.rs` — hand-built fixtures on classical causal
  structures (chain, fork, collider, multi-parent collider), each with an
  expected result derived independently from Definition 5, not copied from
  any paper example or figure.
- `tests/d_convex_property_tests.rs` — the correctness backbone: an
  independent brute-force d-separation oracle checked against
  `Dag::d_convex_hull`/`Dag::strong_d_convex_hull` on every small random DAG
  `proptest` generates, plus idempotence, subset, and CPDAG-invariance
  properties.
- `benches/d_convex_bench.rs` — Criterion benchmarks across chain,
  fork-chain, layered-sparse, and collider-rich DAG shapes.
