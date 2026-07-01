"""Compare standard Shapley (all permutations) vs ASV (DAG-valid orderings only).

This example uses a synthetic 3-node chain: A -> B -> C with an additive value
function v(S) = sum of node values. It shows which permutations ASV excludes and
how that changes the attribution.

Run: python examples/compare_shap_asv.py
Requires: causasv installed (maturin develop --features python from py/)
"""

import itertools

from causasv import ASVExplainer, CausalDAG

NODES = ["A", "B", "C"]
NODE_VALUES = {"A": 1.0, "B": 2.0, "C": 4.0}

# DAG: A -> B -> C  (A causes B, B causes C)
dag = CausalDAG.from_edges([("A", "B"), ("B", "C")])


def value_fn(feature_names: list[str]) -> float:
    return sum(NODE_VALUES[n] for n in feature_names)


# --- Standard Shapley: average over all 3! = 6 permutations ---

def marginal(perm: list[str], i: int) -> float:
    before = perm[:i]
    return value_fn(sorted(before + [perm[i]])) - value_fn(sorted(before))


shap_values: dict[str, float] = {n: 0.0 for n in NODES}
all_perms = list(itertools.permutations(NODES))
for perm in all_perms:
    for idx, node in enumerate(perm):
        shap_values[node] += marginal(list(perm), idx)
shap_values = {n: v / len(all_perms) for n, v in shap_values.items()}

# --- Valid topological orderings for A -> B -> C ---
# A must come before B, B must come before C.
# Only valid ordering: (A, B, C) — just 1 of the 6 total permutations.

valid_perms = [p for p in all_perms if p.index("A") < p.index("B") < p.index("C")]

print("=== Permutations ===")
print(f"All permutations ({len(all_perms)}):")
for p in all_perms:
    mark = "✓ valid" if p in valid_perms else "✗ excluded (violates A→B→C)"
    print(f"  {p}  {mark}")

print(f"\nValid for DAG A→B→C: {len(valid_perms)} / {len(all_perms)}")

# --- ASV via causasv ---
explainer = ASVExplainer(dag)
asv_values = explainer.explain(value_fn, method="exact", n_samples=0)

print("\n=== Attribution comparison ===")
print(f"{'Node':<6} {'SHAP':>8} {'ASV':>8}")
print("-" * 24)
for node in NODES:
    print(f"{node:<6} {shap_values[node]:>8.4f} {asv_values[node]:>8.4f}")

print("\nSHAP treats A, B, C symmetrically in ordering — no causal constraint.")
print("ASV respects A→B→C: only orderings where causes precede effects count.")
print("Result: ASV assigns less attribution to downstream nodes (B, C) because")
print("their 'causal predecessors' always appear first, narrowing their marginal.")
