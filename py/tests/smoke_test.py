"""Smoke test for top-level public API: explain_quality, plot import."""
from causasv import ASVExplainer, CausalDAG, explain_quality, plot

dag = CausalDAG.from_edges([("A", "B"), ("B", "C")])
explainer = ASVExplainer(dag)

r = explain_quality(
    explainer,
    value_fn=lambda xs: float(len(xs)),
    ci=0.95,
    seed=42,
    max_samples=5_000,
)

assert "values" in r, f"missing 'values': {list(r.keys())}"
assert "ci_low" in r, f"missing 'ci_low': {list(r.keys())}"
assert "ci_high" in r, f"missing 'ci_high': {list(r.keys())}"
assert "selected_method" in r, f"missing 'selected_method': {list(r.keys())}"
assert "stderr" in r, f"missing 'stderr': {list(r.keys())}"

for feat in r["values"]:
    lo, hi = r["ci_low"][feat], r["ci_high"][feat]
    assert lo <= hi, f"CI bounds inverted for {feat}: {lo:.4f} > {hi:.4f}"

# plot must import; bar() raises ImportError if matplotlib is absent
try:
    plot.bar(r["values"])
except ImportError as e:
    assert "matplotlib" in str(e), f"Unexpected ImportError: {e}"

print("smoke test OK:", r["selected_method"], r["values"])
