"""
dag_sensitivity.py — ASV sensitivity across multiple candidate DAGs.

Shows how to use ASVEnsembleExplainer when the causal graph is uncertain.
Three expert DAGs over features (education, income, risk_score) are provided;
the ensemble returns mean ASV, std, and rank stability.
"""

from causasv import ASVEnsembleExplainer, CausalDAG

# Three plausible causal graphs over the same features.
# (In practice these come from domain experts or causal discovery output.)
dag_conservative = CausalDAG.from_edges(
    [("education", "income"), ("income", "risk_score")]
)
dag_direct = CausalDAG.from_edges(
    [
        ("education", "income"),
        ("education", "risk_score"),
        ("income", "risk_score"),
    ]
)
dag_parallel = CausalDAG.from_edges(
    [("education", "income"), ("education", "risk_score")]
)


def value_fn(feature_names):
    """Non-linear value function: education and income together synergize."""
    s = set(feature_names)
    edu = 1.0 if "education" in s else 0.0
    inc = 1.0 if "income" in s else 0.0
    risk = 1.0 if "risk_score" in s else 0.0
    return edu + inc + risk + 2.0 * edu * inc  # superadditive term


ensemble = ASVEnsembleExplainer([dag_conservative, dag_direct, dag_parallel])
result = ensemble.explain_with_sensitivity(value_fn, method="exact")

print("Mean ASV across DAGs:")
for f, v in sorted(result["mean_values"].items()):
    std = result["std_values"][f]
    print(f"  {f:20s}: {v:.4f}  ± {std:.4f}")

print(f"\nRank stability (mean pairwise Kendall τ): {result['rank_stability']:.3f}")
print("(1.0 = all DAGs agree on ranking, 0 = random agreement)")

print("\nPer-DAG values:")
dag_names = ["conservative", "direct", "parallel"]
for name, values in zip(dag_names, result["per_dag_values"]):
    formatted = ", ".join(f"{f}={v:.3f}" for f, v in sorted(values.items()))
    print(f"  {name}: {formatted}")
