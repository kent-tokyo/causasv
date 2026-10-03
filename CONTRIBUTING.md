# Contributing

## Development setup

```bash
# Rust
cargo build --all-features
cargo test --all-features

# Python bindings (from repo root)
python -m venv .venv && source .venv/bin/activate
pip install "maturin>=1.13" pytest ruff
cd py && maturin develop --features python
pytest tests/
```

## Before opening a PR

- `cargo fmt --check` and `cargo clippy --all-targets --all-features -- -D warnings` must pass.
- `ruff check py/ examples/` must pass (config in `py/pyproject.toml`).
- `cargo test`, `cargo test --all-features`, and `pytest py/tests/` must pass.
- If your change touches the ASV algorithm (`src/sampler.rs`, `src/approx.rs`, `src/dag_dp*.rs`,
  `src/asv.rs`), make sure it doesn't break the correctness axioms checked in
  `tests/property_tests.rs` — see [docs/correctness.md](docs/correctness.md) for what each
  axiom means and why it must hold.
- When changing documentation, keep the three README translations and method
  limits consistent with `docs/correctness.md`.
- Keep `Cargo.toml`, `py/pyproject.toml`, `CITATION.cff`, and the version shown
  in all three README files synchronized when bumping a release. CI enforces
  the first three.

## Branch naming

| Prefix | Purpose |
|--------|---------|
| `feat/*` | New feature |
| `fix/*` | Bug fix |
| `docs/*` | Documentation only |
| `release/*` | Version bump + CHANGELOG + tag |
