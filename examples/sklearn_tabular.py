"""ASV for a tabular sklearn model over a causal DAG.

Demonstrates how to wrap a sklearn model's predict call as a causasv value
function. The pattern shown here is a sketch of what the planned
`make_tabular_value_fn` helper (v0.7) will do.

Run: python examples/sklearn_tabular.py
Requires: causasv + scikit-learn + numpy
"""

try:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.datasets import make_classification
except ImportError:
    print("sklearn and numpy are required: pip install scikit-learn numpy")
    raise SystemExit(1)

from causasv import CausalDAG, ASVExplainer

# --- Synthetic dataset ---
X, y = make_classification(n_samples=500, n_features=3, n_informative=3,
                            n_redundant=0, random_state=0)
feature_names = ["education", "income", "risk_score"]

model = GradientBoostingClassifier(n_estimators=50, random_state=0)
model.fit(X, y)

# One instance to explain
x_instance = X[0]
# Background: column means (used as "absent feature" baseline)
background_means = X.mean(axis=0)

# --- DAG: education -> income -> risk_score ---
dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])


def make_value_fn(model, x, background, names):
    """Marginal-mean value function: replace absent features with background mean."""
    name_to_idx = {n: i for i, n in enumerate(names)}

    def value_fn(coalition: list[str]) -> float:
        row = background.copy()
        for name in coalition:
            row[name_to_idx[name]] = x[name_to_idx[name]]
        # ponytail: predict_proba for class 1; change index for regression
        return float(model.predict_proba(row.reshape(1, -1))[0, 1])

    return value_fn


value_fn = make_value_fn(model, x_instance, background_means, feature_names)

explainer = ASVExplainer(dag)
info = explainer.explain_with_diagnostics(
    value_fn=value_fn,
    method="exact",
    n_samples=0,
)

print("=== ASV attributions (education -> income -> risk_score) ===")
for name, val in sorted(info["values"].items()):
    print(f"  {name:<15} {val:+.4f}")
print(f"\nv(all) - v(empty) = {value_fn(feature_names) - value_fn([]):.4f}")
print(f"Sum of ASV        = {sum(info['values'].values()):.4f}  (efficiency axiom)")
print(f"Exact: {info['is_exact']}")
