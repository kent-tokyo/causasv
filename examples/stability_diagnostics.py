"""
Demonstrates explain_stability() for verifying that approximate ASV rankings
are consistent across random seeds before trusting the attribution order.

Usage:
    maturin develop --features python
    python examples/stability_diagnostics.py
"""

from causasv import ASVExplainer, CausalDAG, explain_stability

# Simple causal DAG: X1 → X2 → Y, X1 → Y
dag = CausalDAG.from_edges([("X1", "X2"), ("X2", "Y"), ("X1", "Y")])
explainer = ASVExplainer(dag)

# Replace with a real model score in practice.
# This additive value function makes all seeds agree, so rank_stability ≈ 1.0.
def value_fn(coalition: list[str]) -> float:
    weights = {"X1": 3.0, "X2": 1.5, "Y": 0.5}
    return sum(weights.get(f, 0.0) for f in coalition)


print("Running explain_stability() with 10 seeds...")
result = explain_stability(
    explainer,
    value_fn,
    seeds=list(range(1, 11)),
    method="approx",
    n_samples=5_000,
)

rank_stability = result["rank_stability"]
print(f"\nRank stability (mean pairwise Kendall tau): {rank_stability:.3f}")
print(f"  1.0 = perfectly stable rankings across seeds")
print(f"  0.0 = rankings are random (increase n_samples or check value_fn)")
print()

print("Per-feature mean ± std ASV:")
for feat in sorted(result["mean_values"]):
    mean = result["mean_values"][feat]
    std = result["std_values"][feat]
    print(f"  {feat:6s}: {mean:+.4f} ± {std:.4f}")

print()
if rank_stability >= 0.9:
    print("✓ Rankings are stable (rank_stability ≥ 0.9). Safe to trust attribution order.")
elif rank_stability >= 0.7:
    print("⚠ Rankings are moderately stable. Consider increasing n_samples.")
else:
    print("✗ Rankings are unstable. Increase n_samples or use explain_adaptive().")
