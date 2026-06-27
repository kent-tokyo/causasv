"""
Benchmark: normal value_fn vs value_fn_batch for approximate ASV.

Demonstrates GIL-acquisition overhead reduction in Python.
With a normal value_fn, Rust calls back into Python once per coalition
evaluation (O(n_samples * n) calls). With value_fn_batch, it calls back
once per batch (O(n_samples / batch_size) calls).

Usage:
    python examples/benchmark_batched_value_fn.py

Requirements:
    maturin develop --features python
"""

import time

from causasv import ASVExplainer, CausalDAG


def make_chain_dag(n: int) -> CausalDAG:
    return CausalDAG.from_edges([(f"x{i}", f"x{i + 1}") for i in range(n - 1)])


def simulated_model(coalition: list[str]) -> float:
    """CPU-bound pseudo-model: mimics a feature-scoring computation."""
    total = 0.0
    for i, f in enumerate(coalition):
        total += (hash(f) & 0xFF) / 255.0 * (i + 1)
    return total / max(len(coalition), 1)


def simulated_model_batch(coalitions: list[list[str]]) -> list[float]:
    return [simulated_model(c) for c in coalitions]


dag = make_chain_dag(10)
explainer = ASVExplainer(dag)
N_SAMPLES = 2_000
SEED = 42

print(f"DAG: chain n=10, n_samples={N_SAMPLES}, seed={SEED}")
print("-" * 45)

# Normal value_fn: one Python call per coalition
t0 = time.perf_counter()
explainer.explain(
    value_fn=simulated_model, method="approx", n_samples=N_SAMPLES, seed=SEED
)
t_normal = time.perf_counter() - t0
print(f"{'normal value_fn':>20}: {t_normal * 1000:7.1f} ms  (baseline)")

# value_fn_batch with increasing batch sizes
for batch_size in [64, 256, 1024]:
    t0 = time.perf_counter()
    explainer.explain(
        value_fn_batch=simulated_model_batch,
        method="approx",
        n_samples=N_SAMPLES,
        batch_size=batch_size,
        seed=SEED,
    )
    t_batch = time.perf_counter() - t0
    speedup = t_normal / t_batch
    print(
        f"{'value_fn_batch b=' + str(batch_size):>20}: {t_batch * 1000:7.1f} ms"
        f"  ({speedup:.2f}x)"
    )

print()
print("Note: speedup grows with model inference cost. For sklearn or PyTorch")
print("models, batch_size=256-1024 typically saves 5-20x vs normal value_fn.")
