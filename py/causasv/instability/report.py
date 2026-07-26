"""Phase 5: output schema.

Pure data assembly -- no new ASV computation happens here. Everything in the
report comes from Phase 1-4 outputs; this just shapes them into
causasv-instability-attribution-v1 and sorts/serializes deterministically
(json.dumps(report, sort_keys=True) is byte-identical across repeated runs
given the same seed and input, since every upstream stage is seeded).
"""

from __future__ import annotations

import json
from typing import Any

from .bundle import InstabilityDataset
from .modeling import InstabilityModel
from .sensitivity import _feature_ranks

SCHEMA_VERSION = "causasv-instability-attribution-v1"


def build_attribution_report(
    *,
    dataset: InstabilityDataset,
    model_result: InstabilityModel,
    sensitivity_result: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """Assemble the final report from Phase 1-4 outputs.

    ``asv``/``rank`` always come from ``sensitivity_result["mean_values"]`` --
    for a single DAG this is identical to that DAG's own values (mean of one),
    so there's no special-casing needed between the single- and multi-DAG paths.

    ``stderr``/``ci_low``/``ci_high`` reflect the *first* supplied DAG's
    sampling/estimation uncertainty only -- they do not fold in cross-DAG
    structural disagreement, which is reported separately via
    ``dag_sensitive``/``asv_diagnostics.dag_rank_stability``. Conflating the two
    into one interval would hide exactly the DAG-assumption sensitivity this
    workflow exists to surface.

    ``model.meets_min_cv_metric`` is ``None`` when no threshold was set (no gate
    was requested, not "passed"); the report is otherwise silent about model
    quality having been "good enough" absent an explicit threshold to compare to.
    """
    if mode not in ("global", "local"):
        raise ValueError(f"mode must be 'global' or 'local', got {mode!r}")

    per_dag_results = sensitivity_result["per_dag_results"]
    primary = per_dag_results[0]
    values = sensitivity_result["mean_values"]
    ranks = _feature_ranks(values)
    is_multi_dag = len(per_dag_results) > 1

    features = [
        {
            "name": name,
            "asv": values[name],
            "stderr": primary.get("stderr", {}).get(name),
            "ci_low": primary.get("ci_low", {}).get(name),
            "ci_high": primary.get("ci_high", {}).get(name),
            "rank": ranks[name],
            "sign_stable": sensitivity_result["sign_stable"][name],
            "dag_sensitive": sensitivity_result["dag_sensitive"][name],
        }
        for name in sorted(values)
    ]
    features.sort(key=lambda entry: entry["rank"])

    warnings: list[str] = []
    warning_sources = [model_result.warnings, dataset.warnings]
    warning_sources.extend(r.get("warnings", []) for r in per_dag_results)
    for source in warning_sources:
        for w in source:
            if w not in warnings:
                warnings.append(w)

    min_cv_metric = model_result.metadata.get("min_cv_metric")
    meets_min_cv_metric = (
        None if min_cv_metric is None else model_result.cv_metric_mean >= min_cv_metric
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "target": dataset.metadata["target"],
        "mode": mode,
        "features": features,
        "model": {
            "type": model_result.model_type,
            "group_column": model_result.group_column,
            "cv_metric_name": model_result.cv_metric_name,
            "cv_metric": model_result.cv_metric_mean,
            "min_cv_metric": min_cv_metric,
            "meets_min_cv_metric": meets_min_cv_metric,
        },
        "asv_diagnostics": {
            "method": primary.get("selected_method"),
            "is_exact": primary.get("is_exact"),
            "ess_ratio": primary.get("ess_ratio"),
            "seed_rank_stability": primary.get("rank_stability"),
            "dag_rank_stability": (
                sensitivity_result["dag_rank_stability"] if is_multi_dag else None
            ),
        },
        "warnings": warnings,
    }


def summarize_attribution(report: dict[str, Any]) -> dict[str, list[str]]:
    """Split feature names into display buckets: robustly attributed / uncertain
    (CI straddles zero) / DAG-sensitive / insufficient-evidence (model quality
    below its own requested floor). Each feature lands in exactly one bucket,
    in that priority order -- a model that failed its own quality bar makes
    every feature's attribution suspect, so insufficient-evidence overrides
    DAG-sensitivity and CI width rather than competing with them.

    None of these buckets are a causal claim -- "robustly attributed" means
    only "this DAG and model didn't flag it", not "confirmed to matter".
    """
    insufficient_evidence = report["model"]["meets_min_cv_metric"] is False
    buckets: dict[str, list[str]] = {
        "robustly_attributed": [],
        "uncertain": [],
        "dag_sensitive": [],
        "insufficient_evidence": [],
    }
    for f in report["features"]:
        if insufficient_evidence:
            buckets["insufficient_evidence"].append(f["name"])
        elif f["dag_sensitive"]:
            buckets["dag_sensitive"].append(f["name"])
        elif f["ci_low"] is not None and f["ci_high"] is not None and (
            f["ci_low"] <= 0.0 <= f["ci_high"]
        ):
            buckets["uncertain"].append(f["name"])
        else:
            buckets["robustly_attributed"].append(f["name"])
    return buckets


def dump_attribution_json(report: dict[str, Any]) -> str:
    """Serialize a report deterministically: sorted keys, compact separators --
    the same report always produces byte-identical JSON, independent of dict
    insertion order upstream."""
    return json.dumps(report, sort_keys=True, separators=(",", ":"))
