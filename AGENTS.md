# AGENTS.md

## Project contract

`causasv` is a Rust-first library for computing Asymmetric Shapley Values
(ASV) over caller-supplied causal DAGs.

Position it as:

> Fast causal Asymmetric Shapley Values for Rust and Python.

The caller supplies both the graph and the value function. `causasv` does not
infer that the graph is causally correct.

## Scope

In scope:

- DAG and CPDAG representation and validation
- exact and approximate ASV
- topological-order enumeration and sampling
- deterministic seeds and approximation diagnostics
- graph reduction that directly supports attribution
- a small Rust API and a thin Python wrapper
- tests, benchmarks, and reproducible examples

Out of scope:

- causal discovery or automatic graph construction
- model training or automatic feature engineering
- generic SHAP replacement behavior
- deep-learning-specific attribution
- causal-effect estimation as a separate product
- GUI, web service, distributed execution, or GPU acceleration
- broad model-framework integrations in the Rust core

Do not expand the product because an algorithm lacks a Rust implementation.
Responsibility fit comes first.

## Architecture

Keep modules focused:

- `graph.rs`, `cpdag.rs`: graph types and validation
- `topo.rs`, `sampler.rs`: order enumeration and sampling
- `asv.rs`: public computation entry points and dispatch
- `dag_dp.rs`, `dag_dp_sparse.rs`, `tree.rs`: exact methods
- `approx.rs`, `approx_large.rs`, `coalition.rs`: approximate methods
- `d_convex.rs`: graph reduction
- `python.rs`: PyO3 bindings only
- `py/causasv/`: Python helpers built on the Rust engine

Keep the Rust core independent from Python. Do not reimplement ASV algorithms
in Python.

## API rules

The main public Rust types are:

```text
Dag
Cpdag
NodeId
AsvExplainer
AsvResult
SamplingConfig
AdaptiveSamplingConfig
ExactTreeConfig
ExactDagConfig
CausasvError
```

Requirements:

- keep the public API small and documented
- avoid panics in library code
- return `Result` for fallible operations
- use explicit error variants
- preserve stable node indexing and deterministic traversal where practical
- keep `unsafe` code forbidden
- avoid unnecessary dependencies

The graph must be acyclic before ASV computation. Reject self-loops, duplicate
edges, invalid IDs, and cycles with specific errors.

## Correctness rules

Correctness takes priority over speed.

- Treat brute-force `exact` as the reference oracle on small DAGs.
- Check optimized exact methods against that oracle on their shared domain.
- Test approximation against exact or closed-form results where possible.
- Seeded methods must be reproducible.
- Approximate results must report the applicable sample count, seed, ESS,
  convergence, standard error, and fallback metadata.
- A sampler must never emit an invalid or incomplete topological ordering.
- Add regression tests for every algorithmic fix.
- Explain nontrivial mathematical tradeoffs in comments.

Keep runtime, memory, value-function calls, ESS, attribution error, and model
quality as separate measurements.

## Method boundaries

- `exact`: reference enumeration; practical only for small DAGs.
- `exact_tree`: rooted directed trees; feasibility also depends on tree shape.
- `exact_dag`: dense order-ideal DP, limited to `n <= 20`.
- `exact_dag_sparse`: sparse order-ideal DP; default `max_nodes` is 28 and a
  custom configuration may raise it to 63 within the bitmask limit.
- uniform sparse methods: `n <= 63`.
- importance-sampling approximate methods: no node-count limit; graphs above
  64 nodes use the large-coalition backend.

Document defaults separately from configurable hard limits. Keep
`docs/correctness.md` as the canonical explanation.

## Python boundary

Python should wrap the Rust engine and add ergonomic helpers only. Keep
`explain_quality()` as the recommended high-level entry point. Optional NumPy
or sklearn helpers must not become Rust-core dependencies.

When Python bindings change, update the type stub and Python tests together.

## Dependencies and features

Keep dependencies minimal. PyO3 remains optional:

```toml
[features]
default = []
python = ["dep:pyo3"]
```

Normal Rust builds must not require Python.

## Documentation

Public documentation must state that `causasv` computes ASV over a supplied
graph and value function. Never claim causal discovery, automatic causal
explanations, or exact computation for every DAG.

README files should stay concise and synchronized across English, Japanese,
and Chinese. Put detailed correctness arguments and measurements under
`docs/`.

Do not publish performance claims without a reproducible benchmark containing
the version or commit, hardware, graph shape, value function, and command.

## Third-party methods and licenses

Before implementing a method:

1. Confirm that it belongs to the ASV responsibility above.
2. Confirm that any reused code, fixtures, and data have explicit compatible
   terms.
3. Define a mathematical contract, correctness oracle, failure cases, and
   test or benchmark plan.

A public repository without a license is not reusable source. Do not copy,
translate, port, vendor, or derive implementation details from it. A clean
implementation based on a paper still requires scope fit and review of any
patent or other legal concerns.

## CI and releases

CodeQL uses GitHub's default setup. Do not add
`.github/workflows/codeql.yml`; it conflicts with the default SARIF upload.
The README uses the static CodeQL badge documented in this repository.

Keep these version strings synchronized:

- `Cargo.toml` — `package.version`
- `py/pyproject.toml` — `project.version`
- `CITATION.cff` — `version`
- the human-readable version in all three README files

CI enforces the first three. Update all four locations in a release commit.

Before completing a change, run:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo test
```

If benchmarks change, run the relevant Criterion target. If Python bindings
change, also run:

```bash
cd py
python -m pytest
```

Do not claim success for commands that were not run or did not pass.

## Change discipline

- Preserve unrelated user changes.
- Use small, focused commits.
- Keep released behavior in `CHANGELOG.md`; keep future work out of it.
- Never fabricate benchmark results, paper reproduction results, or strength
  claims.

The final product test is simple:

> Given a causal DAG and a value function, compute ASV correctly and
> efficiently.
