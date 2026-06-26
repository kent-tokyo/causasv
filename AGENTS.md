# AGENTS.md

## Project

`causasv` is a Rust-first library for computing **Asymmetric Shapley Values (ASV)** over user-supplied causal directed acyclic graphs.

The project is inspired by the paper:

> “Beyond Shapley: Efficient Computation of Asymmetric Shapley Values”

The goal is not to build a generic SHAP clone. The goal is to build a small, correct, fast, and well-tested ASV engine for causal DAG-based feature attribution.

## Core Positioning

`causasv` should be positioned as:

> Fast causal Asymmetric Shapley Values for Rust and Python.

This crate computes attribution values when the user already has a causal graph. It does **not** learn causal graphs, perform causal discovery, or claim to infer causality from data.

Always keep this distinction clear in code, documentation, and examples.

## Non-goals

Do not implement these in the initial version:

* Causal discovery
* Full SHAP replacement
* Deep learning-specific explainability
* Model training
* Automatic feature engineering
* GUI or web dashboard
* Heavy dataframe framework
* Distributed computation
* GPU acceleration
* XGBoost / LightGBM / sklearn integrations before the core engine is correct

These may be considered later, but not before the core ASV algorithms are correct and tested.

## Initial Milestone: v0.1.0

The first release should focus on correctness, clean APIs, and reproducible tests.

Required features:

* Rust crate named `causasv`
* DAG representation
* Node and edge management
* Cycle detection
* Topological sorting
* Rooted directed tree detection
* Brute-force exact ASV for small DAGs
* Exact ASV path for rooted directed trees
* Approximate ASV for general DAGs
* Topological ordering sampler
* Deterministic RNG seed support
* Basic Python bindings using `pyo3`
* README with clear examples
* Unit tests and property-style tests where practical
* Benchmark scaffolding

## Implementation Priorities

Follow this order:

1. Implement the graph core.
2. Implement validation utilities.
3. Implement exhaustive topological order enumeration for small DAGs.
4. Implement brute-force exact ASV.
5. Implement rooted directed tree exact ASV.
6. Implement random topological ordering sampler.
7. Implement approximate ASV for arbitrary DAGs.
8. Add Rust examples.
9. Add Python bindings.
10. Add benchmarks.
11. Improve README and docs.

Do not start with Python bindings or model integrations. The Rust core must be correct first.

## Repository Structure

Preferred structure:

```text
causasv/
  Cargo.toml
  README.md
  AGENTS.md
  LICENSE-MIT
  LICENSE-APACHE
  src/
    lib.rs
    graph.rs
    error.rs
    topo.rs
    asv.rs
    sampler.rs
    value_function.rs
    tree.rs
    approx.rs
  tests/
    graph_tests.rs
    topo_tests.rs
    asv_bruteforce_tests.rs
    tree_exact_tests.rs
    sampler_tests.rs
  benches/
    asv_bench.rs
  examples/
    basic.rs
    rooted_tree.rs
    approximate_dag.rs
  py/
    pyproject.toml
    README.md
    causasv/
      __init__.py
```

If this structure becomes inconvenient, adjust it, but keep modules small and focused.

## Rust API Design

The public API should be simple and stable.

Preferred public types:

```rust
Dag
NodeId
AsvExplainer
AsvResult
SamplingConfig
ValueFunction
CausasvError
```

Example target API:

```rust
use causasv::{Dag, AsvExplainer, SamplingConfig};

let mut dag = Dag::new();

let education = dag.add_node("education");
let income = dag.add_node("income");
let risk = dag.add_node("risk_score");

dag.add_edge(education, income)?;
dag.add_edge(income, risk)?;

dag.validate()?;

let explainer = AsvExplainer::new(dag);

let values = explainer.approximate(
    |coalition| {
        // User-supplied value function.
        // The coalition tells which features are present.
        Ok(model_value)
    },
    SamplingConfig::new(10_000).with_seed(42),
)?;
```

The actual API may differ, but keep it ergonomic.

## Python API Design

Python bindings should be minimal in v0.1.0.

Target usage:

```python
from causasv import CausalDAG, ASVExplainer

dag = CausalDAG()
dag.add_edge("education", "income")
dag.add_edge("income", "risk_score")

explainer = ASVExplainer(dag)

values = explainer.explain(
    value_fn=my_value_function,
    method="approx",
    n_samples=10_000,
    seed=42,
)
```

Python API should wrap the Rust engine. Do not reimplement algorithms in Python.

## Graph Rules

The graph must always be a directed acyclic graph before ASV computation.

Required validations:

* No self-loops
* No duplicate edges
* No cycles
* Stable node indexing
* Clear error messages
* Deterministic traversal where possible

If a function requires a DAG, validate it or clearly document that the caller must validate first.

## ASV Computation Principles

Correctness is more important than speed.

For small graphs, use brute-force enumeration as the reference implementation.

The brute-force exact implementation should be used in tests to verify optimized algorithms.

Approximation algorithms must support deterministic seeds.

Approximate results should expose metadata, such as:

* number of samples
* seed used
* standard error or variance estimate if available
* whether exact or approximate computation was used

## Topological Ordering Sampler

The topological ordering sampler is critical.

Requirements:

* Must only produce valid topological orderings.
* Must be testable independently.
* Must support deterministic seeded randomness.
* Should expose a simple API.
* Should be benchmarked separately.
* Should not silently produce biased or invalid orders.

If uniform sampling is implemented, document the assumptions and algorithm clearly.

Start with a simple valid sampler. Optimize later.

## Testing Requirements

Every meaningful behavior needs tests.

Required test categories:

### Graph Tests

* Add nodes
* Add edges
* Reject self-loops
* Reject duplicate edges
* Detect cycles
* Topological sort returns valid order
* Disconnected DAGs work

### ASV Tests

* Single-node graph
* Chain graph
* Fork graph
* Collider graph
* Rooted tree graph
* Small DAG brute-force exact values
* Approximation converges toward brute-force values on small graphs
* Seeded approximation is reproducible

### Sampler Tests

* Every sampled ordering is valid
* Same seed produces same sequence
* Different seeds can produce different sequences
* No missing nodes
* No duplicate nodes in an ordering

### Error Tests

* Invalid graph gives clear errors
* Invalid node IDs are rejected
* Empty graph behavior is defined

## Benchmarks

Use Criterion for Rust benchmarks.

Benchmark cases:

* Chain DAG
* Fork DAG
* Balanced rooted tree
* Random DAG
* Dense small DAG
* Sparse larger DAG

Benchmark separately:

* Topological sort
* Topological ordering sampling
* Brute-force ASV
* Rooted tree exact ASV
* Approximate ASV

Do not make performance claims in README unless backed by reproducible benchmarks.

## Documentation Rules

README should include:

* What `causasv` is
* What it is not
* Difference from SHAP
* What ASV means
* Why causal DAGs matter
* Rust example
* Python example
* Exact vs approximate methods
* Current limitations
* Citation section
* License section

Avoid overclaiming.

Use careful language:

Good:

> `causasv` computes ASV over user-supplied causal DAGs.

Bad:

> `causasv` discovers causal explanations automatically.

Good:

> Approximate ASV for general DAGs using sampled topological orderings.

Bad:

> Fast and exact ASV for all DAGs.

## README Opening Draft

Use or adapt this:

```markdown
# causasv

Fast causal Asymmetric Shapley Values for Rust and Python.

`causasv` is a Rust-first engine for computing Asymmetric Shapley Values over user-supplied causal DAGs. It is designed for explainable AI workflows where feature attribution should respect known causal structure.

This crate does not learn causal graphs. It assumes that the user provides a valid directed acyclic graph and a value function.
```

## Code Style

Follow idiomatic Rust.

Requirements:

* Use `thiserror` for errors if helpful.
* Avoid unnecessary dependencies.
* Keep public API small.
* Prefer clear types over clever generics.
* Avoid panics in library code.
* Return `Result` for fallible operations.
* Document all public types and public functions.
* Use deterministic ordering in maps where output stability matters.
* Keep unsafe code out of the project unless absolutely necessary.

## Dependency Policy

Keep dependencies minimal.

Acceptable initial dependencies:

* `thiserror`
* `rand`
* `rand_chacha`
* `indexmap`
* `pyo3` behind optional feature
* `criterion` for benches
* `proptest` or `quickcheck` for tests, if useful

Avoid heavy ML dependencies in the Rust core.

Python integration examples can come later.

## Feature Flags

Suggested feature flags:

```toml
[features]
default = []
python = ["pyo3"]
serde = ["dep:serde"]
```

Do not require Python dependencies for normal Rust usage.

## Error Handling

Errors should be explicit.

Examples:

```rust
CausasvError::CycleDetected
CausasvError::InvalidNodeId
CausasvError::SelfLoop
CausasvError::DuplicateEdge
CausasvError::EmptyGraph
CausasvError::NotRootedTree
CausasvError::ValueFunctionError
```

Do not use vague error messages like `"invalid input"` when a specific reason is known.

## Naming

Use the project name:

* Crate: `causasv`
* Python package: `causasv`
* Repository: `causasv`

Avoid hyphenated names.

Use ASV consistently for Asymmetric Shapley Values.

## CI / GitHub Actions

CodeQL uses the repository's **default CodeQL setup** (enabled via GitHub UI: Settings → Security → Code scanning → Default).

**Do NOT add a custom `.github/workflows/codeql.yml`** — it will conflict with the default setup's SARIF upload and fail with "Code Scanning could not process the submitted SARIF file". This mistake has happened twice; see commits `3bd7838` and `904beac`.

The CodeQL badge in README uses a static `img.shields.io` badge (not a workflow badge):
```markdown
[![CodeQL](https://img.shields.io/badge/CodeQL-enabled-blue.svg)](https://github.com/kent-tokyo/causasv/security/code-scanning)
```

## Quality Bar

Before considering a task complete:

Run:

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test
```

If benchmarks are touched:

```bash
cargo bench
```

If Python bindings are touched:

```bash
cd py
python -m pytest
```

Do not claim success unless the relevant commands pass.

## Commit Style

Use small, focused commits.

Good commit examples:

```text
Add DAG cycle detection
Implement brute-force ASV baseline
Add seeded topological sampler
Add rooted tree validation
Expose minimal Python API
```

Bad commit examples:

```text
misc
update
fix stuff
big changes
```

## Development Philosophy

The project should be boring internally and impressive externally.

Prefer:

* simple algorithms first
* reference implementations
* clear tests
* stable APIs
* honest documentation

Avoid:

* premature optimization
* broad integrations too early
* vague causal claims
* untested mathematical shortcuts
* README hype without benchmarks

## Important Rule for AI Agents

When modifying this repository:

1. Read this file first.
2. Preserve the project scope.
3. Do not silently expand the project into a generic SHAP library.
4. Do not add causal discovery.
5. Keep the Rust core independent from Python.
6. Add tests for every algorithmic change.
7. Prefer correctness over speed.
8. Explain tradeoffs in comments when implementing nontrivial algorithms.
9. Keep public APIs small.
10. Never fake benchmark results or paper reproduction results.

## Future Roadmap

After v0.1.0, possible future versions may include:

### v0.2.0

* Better approximate ASV estimators
* Variance estimates
* More graph generators for benchmarks
* Improved Python ergonomics
* NumPy integration
* Serialization with serde

### v0.3.0

* sklearn examples
* XGBoost examples
* LightGBM examples
* Documentation site
* Paper reproduction benchmarks

### v0.4.0+

* More advanced topological order samplers
* Parallel sampling
* Rayon support
* Incremental graph updates
* Richer attribution reports

Do not implement roadmap items before the core v0.1.0 is stable.

## Final Reminder

`causasv` should be a focused library:

> Given a causal DAG and a value function, compute Asymmetric Shapley Values correctly and efficiently.

Everything else is secondary.
