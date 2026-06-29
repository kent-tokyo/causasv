"""causasv benchmark corpus: DAG corpus × method comparison.

Measures runtime, exactness, ESS ratio, and max stderr across 8 canonical DAGs
using all major causasv computation methods. No external dependencies.

Run:
    cd py
    maturin develop --features python   # build Rust bindings
    python3 ../examples/benchmark_corpus.py

Output: one row per (DAG, method) with timing and quality metrics.
"""

import time

from causasv import CausalDAG, ASVExplainer, explain_quality

# ── DAG corpus ─────────────────────────────────────────────────────────────────

CORPUS = {
    "chain_5":   CausalDAG.from_edges([(f"n{i}", f"n{i+1}") for i in range(4)]),
    "fork_4":    CausalDAG.from_edges([("root","a"),("root","b"),("root","c")]),
    "diamond_6": CausalDAG.from_edges([
        ("src","m0"),("src","m1"),("src","m2"),("src","m3"),
        ("m0","snk"),("m1","snk"),("m2","snk"),("m3","snk"),
    ]),
    "collider_5":CausalDAG.from_edges([("a","z"),("b","z"),("c","z"),("z","out")]),
    "two_chains_10": CausalDAG.from_edges(
        [(f"a{i}", f"a{i+1}") for i in range(4)] + [(f"b{i}", f"b{i+1}") for i in range(4)]
    ),
    "balanced_tree_15": CausalDAG.from_edges([
        ("r","dl0"),("r","dr0"),
        ("dl0","dll0"),("dl0","dlr0"),("dr0","drl0"),("dr0","drr0"),
        ("dll0","llll"),("dll0","lllr"),("dlr0","llrl"),("dlr0","llrr"),
        ("drl0","lrll"),("drl0","lrlr"),("drr0","lrrl"),("drr0","lrrr"),
    ]),
    "chain_24": CausalDAG.from_edges([(f"n{i}", f"n{i+1}") for i in range(23)]),
    "dense_8":  CausalDAG.from_edges([
        ("a","b"),("a","c"),("a","d"),("b","c"),("b","d"),("c","d"),
        ("d","e"),("e","f"),("f","g"),("e","g"),("d","g"),("c","g"),
    ]),
}

# ── Value function ─────────────────────────────────────────────────────────────

def value_fn(features: list) -> float:
    return float(len(features))

# ── Benchmark helpers ──────────────────────────────────────────────────────────

N_SAMPLES = 5_000
SEED      = 42


def run(explainer, method, n):
    """Run one (explainer, method) pair and return metrics dict."""
    kw_approx = {"n_samples": N_SAMPLES, "seed": SEED}
    try:
        t0 = time.perf_counter()
        if method == "exact":
            if n > 9:
                return None   # skip: too slow for brute-force
            r = explainer.explain_with_diagnostics(value_fn, method="exact")
        elif method == "exact_dag":
            if n > 20:
                return None
            r = explainer.explain_with_diagnostics(value_fn, method="exact_dag")
        elif method == "exact_dag_sparse":
            r = explainer.explain_with_diagnostics(value_fn, method="exact_dag_sparse")
        elif method == "uniform_sparse":
            r = explainer.explain_with_diagnostics(
                value_fn, method="uniform_sparse", **kw_approx
            )
        elif method == "auto":
            r = explainer.explain_with_diagnostics(value_fn, method="auto", **kw_approx)
        elif method == "auto_quality":
            r = explainer.explain_quality(
                value_fn, max_samples=N_SAMPLES, seed=SEED
            )
        else:
            return None
        elapsed_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        return {"error": str(e)[:50]}

    ess       = r.get("ess")
    n_samp    = r.get("n_samples", 0)
    ess_ratio = (ess / n_samp) if (ess and n_samp) else None
    stderr    = r.get("stderr", {}) or {}
    max_se    = max(stderr.values()) if stderr else None
    sel       = r.get("selected_method") or r.get("method") or method
    return {
        "runtime_ms": elapsed_ms,
        "is_exact":   r.get("is_exact", False),
        "n_samples":  n_samp,
        "ess_ratio":  ess_ratio,
        "max_stderr": max_se,
        "selected":   sel,
    }


# ── Run and print ──────────────────────────────────────────────────────────────

METHODS = [
    "exact", "exact_dag", "exact_dag_sparse",
    "uniform_sparse", "auto", "auto_quality",
]

HDR = (f"{'DAG':<22} {'n':>3} {'method':<20} {'ms':>8} "
       f"{'exact':>5} {'ess_ratio':>9} {'max_se':>7} {'selected'}")
print(HDR)
print("-" * len(HDR))

for dag_name, dag in CORPUS.items():
    n = len(dag.nodes())
    explainer = ASVExplainer(dag)
    printed = False
    for method in METHODS:
        row = run(explainer, method, n)
        if row is None:
            continue
        if "error" in row:
            print(f"{dag_name:<22} {n:>3} {method:<20} {'ERR':>8}  {row['error']}")
            printed = True
            continue
        rt    = f"{row['runtime_ms']:>7.1f}"
        exact = "yes" if row["is_exact"] else "no"
        ess   = f"{row['ess_ratio']:>9.3f}" if row["ess_ratio"] is not None else f"{'—':>9}"
        se    = f"{row['max_stderr']:>7.4f}" if row["max_stderr"] is not None else f"{'—':>7}"
        sel   = row.get("selected") or "—"
        print(f"{dag_name:<22} {n:>3} {method:<20} {rt} {exact:>5} {ess} {se} {sel}")
        printed = True
    if printed:
        print()

print("-" * len(HDR))
print(f"v(S) = |S| (additive)  ·  n_samples={N_SAMPLES}  ·  seed={SEED}")
