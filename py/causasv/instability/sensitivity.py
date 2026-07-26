"""Phase 4: uncertainty + DAG-ensemble sensitivity.

Single-DAG diagnostics are just causasv.helpers.explain_safe -- it already
does exact-first / CI-aware-approx-fallback plus ESS-ratio and seed
rank-stability warnings, so there's nothing to add here. The one piece that
doesn't exist yet is cross-DAG sensitivity: ASVEnsembleExplainer.explain_with_
sensitivity calls bare explain() (no stderr/CI/ESS), which loses exactly the
diagnostics this schema requires, so it's not used here -- explain_safe is run
once per DAG instead, and the mean/std/rank-stability logic is re-derived
(small enough that importing helpers.py's underscore-prefixed private
functions across module boundaries isn't worth it).
"""

from __future__ import annotations

from typing import Any, Callable

from causasv import ASVExplainer
from causasv.helpers import explain_safe


def _kendall_tau(a: dict[str, float], b: dict[str, float], features: list[str]) -> float:
    n = len(features)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = (a[features[i]] - a[features[j]]) * (b[features[i]] - b[features[j]])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total > 0 else 1.0


def _mean_kendall_tau(per_dag_values: list[dict[str, float]], features: list[str]) -> float:
    taus = [
        _kendall_tau(per_dag_values[i], per_dag_values[j], features)
        for i in range(len(per_dag_values))
        for j in range(i + 1, len(per_dag_values))
    ]
    return sum(taus) / len(taus) if taus else 1.0


def _feature_ranks(values: dict[str, float]) -> dict[str, int]:
    """Rank 1 = highest ASV value; ties broken by feature name for determinism."""
    ordered = sorted(values, key=lambda f: (-values[f], f))
    return {f: i + 1 for i, f in enumerate(ordered)}


def _sign(v: float) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def explain_with_dag_sensitivity(
    dags: list,
    value_fn: Callable[[list[str]], float],
    *,
    seed: int | None = None,
    ci: float = 0.95,
    ess_ratio_min: float = 0.10,
    rank_stability_min: float = 0.90,
    stability_seeds: int = 5,
    rank_spread_sensitive: int = 2,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run ``explain_safe`` per DAG and report cross-DAG sensitivity.

    Works for a single DAG too (``dag_rank_stability`` is then trivially 1.0 and
    no feature is DAG-sensitive) so callers don't need a separate code path for
    the one-DAG case.

    All DAGs must share the same node set -- comparing "the ASV of
    evaluator_family" across DAGs only means something if every DAG actually has
    an ``evaluator_family`` node. A mismatch is rejected with the specific
    per-DAG difference (``ASVEnsembleExplainer``'s own Kendall-tau aggregation
    assumes this and raises a bare ``KeyError`` instead of explaining why).

    ``rank_spread_sensitive``: a feature is flagged ``dag_sensitive`` if its
    rank (1 = highest ASV) varies by at least this many positions across DAGs,
    or if its sign isn't stable. This is a judgment call, not a value from the
    spec -- ``rank_spread`` is returned raw so callers can apply their own cutoff.
    """
    if not dags:
        raise ValueError("dags must be non-empty")

    node_sets = [frozenset(dag.nodes()) for dag in dags]
    first = node_sets[0]
    for i, node_set in enumerate(node_sets[1:], start=1):
        if node_set != first:
            raise ValueError(
                "all DAGs must share the same node set for cross-DAG comparison -- "
                f"dag[0] has {sorted(first)}, dag[{i}] has {sorted(node_set)} "
                f"(symmetric difference: {sorted(first ^ node_set)})"
            )

    per_dag_results = [
        explain_safe(
            ASVExplainer(dag),
            value_fn,
            seed=seed,
            ci=ci,
            ess_ratio_min=ess_ratio_min,
            rank_stability_min=rank_stability_min,
            stability_seeds=stability_seeds,
            **kwargs,
        )
        for dag in dags
    ]

    features = sorted(first)
    k = len(dags)
    per_dag_values = [r["values"] for r in per_dag_results]
    mean_values = {f: sum(v[f] for v in per_dag_values) / k for f in features}
    std_values = {
        f: (sum((v[f] - mean_values[f]) ** 2 for v in per_dag_values) / k) ** 0.5
        for f in features
    }
    dag_rank_stability = _mean_kendall_tau(per_dag_values, features)

    per_dag_ranks = [_feature_ranks(v) for v in per_dag_values]
    rank_spread = {
        f: max(r[f] for r in per_dag_ranks) - min(r[f] for r in per_dag_ranks) for f in features
    }
    sign_stable = {f: len({_sign(v[f]) for v in per_dag_values}) <= 1 for f in features}
    dag_sensitive = {
        f: (not sign_stable[f]) or rank_spread[f] >= rank_spread_sensitive for f in features
    }

    return {
        "per_dag_results": per_dag_results,
        "mean_values": mean_values,
        "std_values": std_values,
        "dag_rank_stability": dag_rank_stability,
        "rank_spread": rank_spread,
        "sign_stable": sign_stable,
        "dag_sensitive": dag_sensitive,
    }
