# causasv vs SHAP — Comparison

Comparison of causasv (Asymmetric Shapley Values) with SHAP KernelExplainer
across runtime, coalition call count, and attribution accuracy.

Reproduce: `python examples/compare_causasv_shap.py` (requires `pip install shap numpy`).

**Setup:** Apple M-series (arm64) · causasv v0.8.5 · shap 0.52.0 · Python 3.13.6

---

## Runtime comparison — chain DAG

Value function: `v(S) = Σ(i+1 for i in S)` (additive).
causasv uses `auto_quality` (exact when feasible); SHAP uses `KernelExplainer(nsamples=256)`.

| DAG | n | causasv (exact) | SHAP (approx) | Speedup |
|-----|---|-----------------|---------------|---------|
| chain_4 | 4 | **0.46 ms** | 13.96 ms | ~30× |
| chain_6 | 6 | **0.14 ms** | 2.40 ms | ~17× |
| chain_8 | 8 | **0.12 ms** | 4.44 ms | ~37× |
| chain_10 | 10 | **0.35 ms** | 11.31 ms | ~32× |
| chain_15 | 15 | **0.44 ms** | 9.67 ms | ~22× |
| chain_20 | 20 | **0.71 ms** | 6.40 ms | ~9× |

causasv returns exact results in all rows; SHAP returns approximate values.

> **Note:** this comparison uses a chain DAG where causasv applies exact sparse DP (visiting only n+1 order ideals) while SHAP's KernelExplainer samples the full 2ⁿ coalition space. causasv's advantage is largest in DAG-known, sparse settings; SHAP is the right tool when no causal structure is available.

---

## Coalition call count — chain DAG

For a chain with n nodes, causasv's sparse order-ideal DP visits only n+1 valid
order ideals. Exact SHAP would require evaluating 2^n − 1 unique coalitions.

| n | SHAP unique coalitions | causasv order ideals (chain) | Ratio |
|---|------------------------|------------------------------|-------|
| 4 | 15 | 5 | 3× |
| 6 | 63 | 7 | 9× |
| 8 | 255 | 9 | 28× |
| 10 | 1,023 | 11 | 93× |
| 15 | 32,767 | 16 | 2,048× |
| 20 | 1,048,575 | 21 | 49,932× |
| 24 | 16,777,215 | 25 | 671,089× |
| 28 | 268,435,455 | 29 | 9,256,395× |

For n=20, exact SHAP requires >1M coalition evaluations; causasv exact_dag_sparse needs 21.

---

## Attribution comparison — synergistic value function

This experiment shows **when ASV and SHAP diverge**, and why the difference matters.

**Setup:** DAG = A → B → C. B is only valuable when A is present (causal dependency).

```
v(∅)=0    v(A)=1    v(B)=0    v(C)=0.5
v(A,B)=5  v(A,C)=2  v(B,C)=0.5  v(A,B,C)=7
```

**Valid orderings under DAG:** A→B→C only (the only topologically valid order).
**SHAP:** averages over all 3! = 6 orderings equally.

| Feature | ASV (DAG-aware) | SHAP (all perms) | Δ = ASV − SHAP |
|---------|-----------------|------------------|----------------|
| A | **1.0000** | 3.5833 | −2.5833 |
| B | **4.0000** | 2.3333 | +1.6667 |
| C | **2.0000** | 1.0833 | +0.9167 |

Both methods satisfy the efficiency axiom: Σφ_i = v(A,B,C) − v(∅) = 7.

**Interpretation:**

Under the causal DAG, the only valid ordering is A → B → C. In this ordering:
- A contributes v(A) − v(∅) = **1.0** (its standalone value)
- B contributes v(A,B) − v(A) = **4.0** (the full A→B synergy is credited to B)
- C contributes v(A,B,C) − v(A,B) = **2.0**

SHAP averages over all 6 orderings, including invalid ones like B→A→C. In the
ordering B→A→C, A appears after B and receives credit for the large joint value
v(A,B) − v(B) = 5.0, while B receives v(B) − v(∅) = 0.0 for appearing first
without its causal prerequisite.

This leads SHAP to:
- **Overestimate A** (3.58 vs 1.0): A appears to "unlock" B in invalid orderings,
  receiving credit for the synergy that actually belongs to B's causal position.
- **Underestimate B** (2.33 vs 4.0): B is evaluated in orderings where A is absent,
  giving B zero contribution, which dilutes its average Shapley value.

This is the **credit-through-ancestors problem** that ASV is designed to prevent.
When a causal DAG is known, restricting to valid topological orderings produces
attributions that respect the causal structure.

---

## When SHAP and ASV agree

For **additive value functions** (v(S) = Σ f(i) for i in S), the Shapley value
and ASV are identical regardless of DAG structure. In additive cases, marginal
contributions are constant and ordering does not matter.

ASV diverges from SHAP when the value function has **interaction effects** between
features — specifically when cause-effect relationships create asymmetric marginal
contributions that change with ordering.

---

## When to use each

| Scenario | Recommended |
|----------|------------|
| No causal knowledge, any model | SHAP |
| Known causal DAG, need exact values | causasv (`auto` or `auto_quality`) |
| Known causal DAG, need CI bounds | causasv (`explain_quality`) |
| Large sparse DAG (n > 20), exact | causasv `exact_dag_sparse` |
| Need to study SHAP vs ASV difference | Both, compare outputs |
