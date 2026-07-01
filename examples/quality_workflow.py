"""End-to-end example: explain_quality with CI and optional plot.

Demonstrates the recommended causasv workflow for Python users:
  1. Build a causal DAG from domain knowledge
  2. Define a value function (e.g. wrap a model)
  3. Call explain_quality() — exact when feasible, CI-aware approx otherwise
  4. Inspect values, confidence intervals, and the method that was selected

Run:
    cd py
    pip install maturin && maturin develop --features python
    python ../examples/quality_workflow.py

Optional: pip install matplotlib  (for bar/waterfall charts)

No external ML dependencies required — uses a synthetic additive value function.
"""

from causasv import ASVExplainer, CausalDAG, explain_quality, plot

# ── 1. Build the causal DAG ──────────────────────────────────────────────────
# Chain: education → income → risk_score
# ASV will only average over orderings consistent with this causal order.
dag = CausalDAG.from_edges([
    ("education", "income"),
    ("income",    "risk_score"),
])

# ── 2. Define the value function ─────────────────────────────────────────────
# Simulated additive model: each feature contributes a fixed weight.
# Replace with your own model: v(features) = model.predict(impute_absent(features))
WEIGHTS = {"education": 1.0, "income": 2.5, "risk_score": 0.8}


def value_fn(feature_names: list) -> float:
    """Value of a coalition: sum of weights for present features."""
    return sum(WEIGHTS.get(f, 0.0) for f in feature_names)


# ── 3. explain_quality: exact-first, CI-aware fallback ───────────────────────
explainer = ASVExplainer(dag)
info = explain_quality(
    explainer,
    value_fn=value_fn,
    ci=0.95,   # 95% confidence intervals; None to skip
    seed=42,
    max_samples=20_000,
)

# ── 4. Inspect results ────────────────────────────────────────────────────────
print("=" * 55)
print(f"  Method : {info['selected_method']}")
print(f"  Exact  : {info['is_exact']}")
print(f"  n used : {info['n_samples']}")
if info.get("converged") is not None:
    print(f"  Converged: {info['converged']}")
print()
print(f"{'Feature':<15}  {'ASV':>8}  {'95% CI':>22}  {'stderr':>8}")
print("-" * 55)
for feat in sorted(info["values"], key=lambda k: -abs(info["values"][k])):
    v   = info["values"][feat]
    lo  = info.get("ci_low",  {}).get(feat, float("nan"))
    hi  = info.get("ci_high", {}).get(feat, float("nan"))
    se  = info.get("stderr",  {}).get(feat, float("nan"))
    print(f"{feat:<15}  {v:>8.4f}  [{lo:>8.4f}, {hi:>8.4f}]  {se:>8.4f}")
print("=" * 55)

# ── 5. Optional plots ─────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt

    ax = plot.bar(info["values"], title="ASV — education → income → risk_score")
    plt.savefig("/tmp/asv_bar.png", dpi=100, bbox_inches="tight")
    print("\nBar chart saved to /tmp/asv_bar.png")

    ax2 = plot.waterfall(info["values"], base_value=value_fn([]),
                         title="ASV Waterfall — from v(∅) to v(V)")
    plt.savefig("/tmp/asv_waterfall.png", dpi=100, bbox_inches="tight")
    print("Waterfall chart saved to /tmp/asv_waterfall.png")

except ImportError:
    print("\n(install matplotlib to generate charts: pip install matplotlib)")
