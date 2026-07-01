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
- `cargo test` (Rust) and `pytest py/tests/` must pass.
- If your change touches the ASV algorithm (`src/sampler.rs`, `src/approx.rs`, `src/dag_dp*.rs`,
  `src/asv.rs`), make sure it doesn't break the correctness axioms checked in
  `tests/property_tests.rs` — see [docs/correctness.md](docs/correctness.md) for what each
  axiom means and why it must hold.
- Keep `Cargo.toml` / `py/pyproject.toml` / `CITATION.cff` / `README.md` versions in sync when
  bumping the version (enforced by CI's `version-sync` job).

## Branch naming

| Prefix | Purpose |
|--------|---------|
| `feat/*` | New feature |
| `fix/*` | Bug fix |
| `docs/*` | Documentation only |
| `release/*` | Version bump + CHANGELOG + tag |
