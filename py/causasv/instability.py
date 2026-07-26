"""Adapter: quietset label-instability data -> causasv ASV attribution.

Scope and honesty guardrails (see docs/integrations/quietset_label_instability.md):

- This module reads quietset's ``Observation``/``StabilityReport`` JSONL *by field
  name only*. It never imports quietset and never re-implements quietset's scoring
  (agreement, entropy, EM, etc.) -- if a value isn't already in the scored/observations
  JSONL, this module does not compute it.
- The output is an Asymmetric Shapley Value **attribution under a user-supplied DAG
  and a fitted prediction model** -- not a causal effect estimate. "evaluator_family
  has the highest ASV" is not "changing evaluator_family fixes instability by X points".
  Treat ASV rankings as hypotheses for a follow-up paired/controlled re-evaluation,
  not as a conclusion to act on directly.

This module is intentionally NOT imported by ``causasv/__init__.py`` (same pattern as
``causasv/plot.py``): importing plain ``causasv`` never pulls in numpy/scikit-learn.
Use ``from causasv import instability`` or ``from causasv.instability import ...``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# causasv's own core (compiled extension + helpers.py) is always available wherever
# causasv.instability can be imported at all, and pulls in neither numpy nor
# scikit-learn itself -- so these are plain top-level imports, unlike numpy/sklearn
# below, which are optional and imported lazily at their point of use.
from causasv import ASVExplainer, CausalDAG
from causasv.helpers import explain_safe, make_tabular_value_fn

# ---------------------------------------------------------------------------
# quietset schema knowledge (reference only -- quietset itself is never imported
# or modified; this list is manually transcribed from
# quietset/crates/quietset/src/schema.rs::StabilityReport and re-checked by
# test_instability.py against a fixture so drift is caught, not silently ignored).
# ---------------------------------------------------------------------------

#: Every StabilityReport field except ``sample_id`` (join key) and ``n_observations``
#: (an evidence *count*, not a statistic computed from the label/score distribution
#: itself -- the one field in the report deliberately not treated as leakage).
#: Any of these requested as a feature would let the model see a value derived from
#: the same label/score distribution the target itself is computed from.
STABILITY_REPORT_FIELDS = frozenset(
    {
        "majority_label",
        "label_agreement",
        "label_agreement_lcb",
        "label_margin",
        "label_entropy",
        "label_distribution",
        "weighted_majority_label",
        "weighted_label_confidence",
        "weighted_label_distribution",
        "majority_weighted_conflict",
        "latent_truth_label",
        "latent_truth_confidence",
        "latent_truth_label_distribution",
        "majority_latent_conflict",
        "evaluator_effective_n",
        "correlated_evaluator_warning",
        "latent_truth_converged",
        "latent_truth_iterations",
        "latent_truth_convergence_delta",
        "latent_truth_demotion_reason",
        "score_mean",
        "score_std",
        "score_range",
        "score_mad",
        "score_iqr",
        "score_sign_agreement",
        "budget_sensitivity",
        "budget_slope",
        "seed_sensitivity",
        "model_agreement",
        "evaluator_agreement",
        "gradient_sign_agreement",
        "update_cosine_mean",
        "update_direction_agreement",
        "teacher_residual_stability",
        "shuffle_seed_sensitivity",
        "loss_recipe_agreement",
        "dead_unit_rate",
        "saturated_unit_rate",
        "trajectory_effect_mean",
        "trajectory_effect_sign_agreement",
        "confidence",
        "adjusted_stability_score",
        "disagreement_score",
        "stability_score",
        "decision",
        "components",
    }
)

#: Raw seed numbers carry no causal/ordinal meaning (seed 42 isn't "bigger" than
#: seed 7). Blocked from cell_features/sample_features by default; see
#: ``allow_raw_seed_feature``.
SEED_LIKE_COLUMNS = frozenset({"seed", "shuffle_seed"})

_MIXED_FILE_ERROR = (
    "aggregate scored report has lost config-level variation; "
    "provide per-cell scored files or config-scoped sample IDs"
)

_BUNDLE_SCHEMA_VERSION = "causasv-instability-bundle-v1"


# ---------------------------------------------------------------------------
# Target definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSpec:
    """One instability target: which scored fields it needs and how to compute it.

    ``min_observations`` is a hard floor, not a suggestion -- e.g. label entropy or
    MAD/IQR are degenerate (undefined or zero by construction) from a single
    observation, so accepting n=1 there would silently manufacture a fake zero
    rather than reflect "not enough evidence".
    """

    name: str
    scored_fields: tuple[str, ...]
    compute: Callable[[dict], float]
    is_binary: bool
    min_observations: int


def _passthrough(field_name: str) -> Callable[[dict], float]:
    def _f(r: dict) -> float:
        return float(r[field_name])

    return _f


def _label_disagreement(r: dict) -> float:
    return 1.0 - float(r["label_agreement"])


def _score_sign_disagreement(r: dict) -> float:
    return 1.0 - float(r["score_sign_agreement"])


def _lcb_risk(r: dict) -> float:
    return 1.0 - float(r["label_agreement_lcb"])


def _review_or_drop(r: dict) -> float:
    return 1.0 if r["decision"] in ("review", "drop") else 0.0


TARGET_DEFINITIONS: dict[str, TargetSpec] = {
    "label_entropy": TargetSpec(
        "label_entropy", ("label_entropy",), _passthrough("label_entropy"), False, 2
    ),
    "label_disagreement": TargetSpec(
        "label_disagreement", ("label_agreement",), _label_disagreement, False, 2
    ),
    "score_mad": TargetSpec("score_mad", ("score_mad",), _passthrough("score_mad"), False, 2),
    "score_iqr": TargetSpec("score_iqr", ("score_iqr",), _passthrough("score_iqr"), False, 2),
    "score_sign_disagreement": TargetSpec(
        "score_sign_disagreement",
        ("score_sign_agreement",),
        _score_sign_disagreement,
        False,
        2,
    ),
    "review_or_drop": TargetSpec("review_or_drop", ("decision",), _review_or_drop, True, 1),
    "lcb_risk": TargetSpec("lcb_risk", ("label_agreement_lcb",), _lcb_risk, False, 2),
}


# ---------------------------------------------------------------------------
# Bundle manifest: cell_features (config axis) vs replicate_axes (measurement axis)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """One (scored, observations) pair plus the config it was produced under.

    ``features`` are cell_features: config values that are constant *within* this
    cell by construction (that's the point -- comparing cells is how attribution
    happens). ``replicate_axes`` are the columns quietset varied *within* the cell
    to produce label/score disagreement in the first place; they are validation-only
    here and are never turned into model features (see SEED_LIKE_COLUMNS).
    """

    config_id: str
    scored_path: str
    observations_path: str
    features: dict[str, Any]
    replicate_axes: tuple[str, ...]


def load_bundle_manifest(path: str) -> list[Cell]:
    """Load a ``causasv-instability-bundle-v1`` manifest into a list of Cells.

    ``scored``/``observations`` paths in the manifest are resolved relative to the
    manifest file's own directory.
    """
    manifest_path = Path(path)
    raw = json.loads(manifest_path.read_text())
    if raw.get("schema_version") != _BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bundle schema_version: {raw.get('schema_version')!r} "
            f"(expected {_BUNDLE_SCHEMA_VERSION!r})"
        )
    cells_raw = raw.get("cells")
    if not cells_raw:
        raise ValueError("bundle manifest has no cells")

    base_dir = manifest_path.parent
    seen_config_ids: set[str] = set()
    cells = []
    for i, c in enumerate(cells_raw):
        config_id = c.get("config_id")
        if not config_id:
            raise ValueError(f"cells[{i}] is missing config_id")
        if config_id in seen_config_ids:
            raise ValueError(f"duplicate config_id in manifest: {config_id!r}")
        seen_config_ids.add(config_id)
        for required in ("scored", "observations"):
            if required not in c:
                raise ValueError(f"cell {config_id!r} is missing {required!r}")
        cells.append(
            Cell(
                config_id=config_id,
                scored_path=str(base_dir / c["scored"]),
                observations_path=str(base_dir / c["observations"]),
                features=dict(c.get("features", {})),
                replicate_axes=tuple(c.get("replicate_axes", ())),
            )
        )
    return cells


def wrap_single_cell(
    observations_path: str,
    scored_path: str,
    *,
    cell_features: tuple[str, ...] | list[str] = (),
    config_id: str = "default",
) -> list[Cell]:
    """Wrap one plain (observations, scored) pair as a 1-cell bundle.

    A single aggregate ``scored.jsonl`` has no config-level granularity by
    construction (quietset already collapsed all conditions into one
    StabilityReport per sample_id) -- so requesting cell_features here is a
    contradiction the caller must resolve upstream (per-cell scored files),
    not something this adapter can guess at.
    """
    if cell_features:
        raise ValueError(_MIXED_FILE_ERROR)
    return [Cell(config_id, str(scored_path), str(observations_path), {}, ())]


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


@dataclass
class InstabilityDataset:
    """Tabular (sample_id, config_id) rows ready for Phase 2 model fitting."""

    feature_names: list[str]
    encoded_feature_names: list[str]
    X: Any  # np.ndarray, shape (n_rows, n_encoded_features)
    y: Any  # np.ndarray, shape (n_rows,)
    sample_ids: list[str]
    config_ids: list[str]
    groups: Any  # np.ndarray[int], shape (n_rows,)
    is_binary: bool
    categorical_columns: dict[str, list[str]]
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _validate_column_roles(
    cell_features: list[str],
    sample_features: list[str],
    allow_raw_seed_feature: bool,
) -> None:
    cf, sf = set(cell_features), set(sample_features)
    overlap = cf & sf
    if overlap:
        raise ValueError(
            f"columns declared as both cell_features and sample_features: {sorted(overlap)}"
        )
    if "sample_id" in cf or "sample_id" in sf:
        raise ValueError("sample_id is a row identifier, not a feature")
    if not allow_raw_seed_feature:
        bad = (cf | sf) & SEED_LIKE_COLUMNS
        if bad:
            raise ValueError(
                f"{sorted(bad)} look like raw seed columns. Raw seed/shuffle_seed values "
                "have no numeric or causal meaning and default to being replicate axes, "
                "not model features -- to study seed sensitivity, use quietset's own "
                "seed_sensitivity/shuffle_seed_sensitivity as the *target* instead. "
                "Pass allow_raw_seed_feature=True to opt into treating them as categorical "
                "features anyway (results won't generalize to unseen seeds)."
            )
    leaking = (cf | sf) & STABILITY_REPORT_FIELDS
    if leaking:
        raise ValueError(
            f"feature(s) {sorted(leaking)} are StabilityReport-derived fields computed from "
            "the same label/score distribution the target is computed from -- this would "
            "leak the target into the features. Choose upstream (config/sample) columns instead."
        )


def build_instability_dataset(
    cells: list[Cell],
    *,
    target: str,
    cell_features: list[str] = (),
    sample_features: list[str] = (),
    group_by: str = "sample_id",
    min_observations: int | None = None,
    missing_numeric: str = "error",
    drop_constant_features: bool = False,
    allow_raw_seed_feature: bool = False,
) -> InstabilityDataset:
    """Join cells into one (sample_id, config_id) analysis table.

    ``missing_numeric``: ``"error"`` (default) hard-fails naming the feature and
    affected row count; ``"drop_rows"`` drops those rows and records the count.
    Categorical missingness always becomes its own explicit ``<missing>`` category
    (never a silent 0) -- there is no numeric-imputation mode here on purpose: an
    adapter should not manufacture values the harness didn't report.
    """
    if target not in TARGET_DEFINITIONS:
        raise ValueError(f"unknown target {target!r}; choose from {sorted(TARGET_DEFINITIONS)}")
    if missing_numeric not in ("error", "drop_rows"):
        raise ValueError(f"unknown missing_numeric policy: {missing_numeric!r}")
    spec = TARGET_DEFINITIONS[target]

    cell_features = list(dict.fromkeys(cell_features))
    sample_features = list(dict.fromkeys(sample_features))
    _validate_column_roles(cell_features, sample_features, allow_raw_seed_feature)

    floor = spec.min_observations
    if min_observations is not None:
        floor = max(floor, min_observations)

    warnings: list[str] = []
    metadata: dict[str, Any] = {
        "target": target,
        "target_definition": {
            "scored_fields": list(spec.scored_fields),
            "is_binary": spec.is_binary,
            "min_observations": floor,
        },
        "feature_list": {"cell_features": cell_features, "sample_features": sample_features},
        "cells": [],
        "input_record_count": 0,
        "dropped_record_count": {},
        "missing_field_count": {},
        "duplicate_sample_handling": "last-write-wins on repeated sample_id lines within one "
        "scored.jsonl; count reported below",
        "duplicate_scored_lines": 0,
        "categorical_encoding": {},
        "constant_features": {"checked": [], "dropped": []},
    }

    def _bump_dropped(k: str, n: int = 1) -> None:
        metadata["dropped_record_count"][k] = metadata["dropped_record_count"].get(k, 0) + n

    def _bump_missing(k: str, n: int = 1) -> None:
        metadata["missing_field_count"][k] = metadata["missing_field_count"].get(k, 0) + n

    rows: dict[tuple[str, str], dict[str, Any]] = {}

    for cell in cells:
        for name in cell.features:
            if name in cell.replicate_axes:
                raise ValueError(
                    f"cell {cell.config_id!r}: {name!r} declared as both a cell feature "
                    "and a replicate axis"
                )
        for name in sample_features:
            if name in cell.replicate_axes:
                raise ValueError(
                    f"cell {cell.config_id!r}: {name!r} declared as both a sample feature "
                    "and a replicate axis"
                )
            if name in cell.features:
                raise ValueError(
                    f"cell {cell.config_id!r}: {name!r} is requested as a sample_feature but "
                    "this cell also declares it as a cell feature -- pass it as a "
                    "cell_feature consistently, or rename one of them"
                )

        scored_records = _read_jsonl(cell.scored_path)
        observation_records = _read_jsonl(cell.observations_path)
        metadata["input_record_count"] += len(scored_records) + len(observation_records)

        scored_by_sample: dict[str, dict] = {}
        for rec in scored_records:
            sid = rec.get("sample_id")
            if not sid:
                continue
            if sid in scored_by_sample:
                metadata["duplicate_scored_lines"] += 1
            scored_by_sample[sid] = rec

        obs_by_sample: dict[str, list[dict]] = {}
        for rec in observation_records:
            sid = rec.get("sample_id")
            if not sid:
                continue
            obs_by_sample.setdefault(sid, []).append(rec)

        # Mixed-file detection: a declared (constant, per-cell) config value must not
        # actually vary inside this cell's own observations.jsonl -- if it does, the
        # file mixes multiple real conditions under one config_id and we must not
        # guess which rows belong to which condition (that would re-implement
        # quietset's own grouping logic on incomplete information).
        for feat_name, declared_value in cell.features.items():
            observed_values = {
                rec[feat_name]
                for obs_list in obs_by_sample.values()
                for rec in obs_list
                if feat_name in rec
            }
            if observed_values and observed_values != {declared_value}:
                raise ValueError(
                    f"cell {cell.config_id!r}: observations.jsonl carries value(s) "
                    f"{sorted(observed_values, key=str)!r} for declared cell feature "
                    f"{feat_name!r}, conflicting with manifest value {declared_value!r}. "
                    + _MIXED_FILE_ERROR
                )

        kept_for_cell = 0
        for sid, scored_rec in scored_by_sample.items():
            obs_list = obs_by_sample.get(sid)
            if not obs_list:
                _bump_dropped("missing_from_observations")
                continue
            if len(obs_list) < floor:
                _bump_dropped("insufficient_observations")
                continue
            missing_target_fields = [f for f in spec.scored_fields if scored_rec.get(f) is None]
            if missing_target_fields:
                _bump_dropped("missing_target_fields")
                for f in missing_target_fields:
                    _bump_missing(f)
                continue

            row_features: dict[str, Any] = dict(cell.features)
            for feat_name in cell_features:
                if feat_name not in row_features:
                    row_features[feat_name] = None
                    _bump_missing(feat_name)
            consistency_error = None
            for feat_name in sample_features:
                values = {rec[feat_name] for rec in obs_list if feat_name in rec}
                if not values:
                    row_features[feat_name] = None
                    _bump_missing(feat_name)
                elif len(values) > 1:
                    consistency_error = (feat_name, values)
                    break
                else:
                    row_features[feat_name] = next(iter(values))
            if consistency_error is not None:
                feat_name, values = consistency_error
                raise ValueError(
                    f"sample {sid!r} in cell {cell.config_id!r}: sample feature {feat_name!r} "
                    f"is not constant across its observations ({sorted(values, key=str)!r}) -- "
                    "sample_features must describe intrinsic, per-sample properties"
                )

            rows[(sid, cell.config_id)] = {
                "target": spec.compute(scored_rec),
                "features": row_features,
            }
            kept_for_cell += 1

        metadata["cells"].append(
            {
                "config_id": cell.config_id,
                "scored_record_count": len(scored_records),
                "observation_record_count": len(observation_records),
                "kept_sample_count": kept_for_cell,
            }
        )

    if not rows:
        raise ValueError(
            "no rows survived adapter validation -- check target field availability, "
            "min_observations, and sample_id overlap between scored/observations files"
        )

    ordered_keys = sorted(rows.keys())
    all_feature_names = cell_features + sample_features

    # Resolve numeric-missing policy before any feature is judged "constant" --
    # dropping rows can change that judgement, so order matters.
    numeric_cols = _infer_numeric_columns(
        [rows[k]["features"] for k in ordered_keys], all_feature_names
    )
    for feat_name in numeric_cols:
        missing_keys = [k for k in ordered_keys if rows[k]["features"].get(feat_name) is None]
        if not missing_keys:
            continue
        if missing_numeric == "error":
            raise ValueError(
                f"numeric feature {feat_name!r} is missing for {len(missing_keys)} row(s) "
                f"(e.g. {missing_keys[0]!r}); pass missing_numeric='drop_rows' to exclude "
                "them explicitly, or backfill the source data -- this adapter never "
                "silently imputes 0"
            )
        for k in missing_keys:
            del rows[k]
        _bump_dropped("missing_numeric_feature", len(missing_keys))
        warnings.append(
            f"dropped {len(missing_keys)} row(s) missing numeric feature {feat_name!r}"
        )
        ordered_keys = [k for k in ordered_keys if k not in missing_keys]

    if not rows:
        raise ValueError("no rows remain after applying the missing_numeric policy")

    constant = []
    for feat_name in all_feature_names:
        values = {rows[k]["features"].get(feat_name) for k in ordered_keys}
        metadata["constant_features"]["checked"].append(feat_name)
        if len(values) <= 1:
            constant.append(feat_name)
    if constant:
        if not drop_constant_features:
            raise ValueError(
                f"feature(s) {constant!r} are constant across the entire combined dataset "
                "(after joining all cells) and cannot be attributed anything -- being "
                "constant within one cell is expected, but this checks the combined table. "
                "Pass drop_constant_features=True to exclude them explicitly."
            )
        warnings.append(f"dropped constant feature(s): {constant}")
        metadata["constant_features"]["dropped"] = constant
        all_feature_names = [f for f in all_feature_names if f not in constant]

    if not all_feature_names:
        raise ValueError("no features remain after constant-feature filtering")

    sample_ids = [k[0] for k in ordered_keys]
    config_ids = [k[1] for k in ordered_keys]
    feature_rows = [rows[k]["features"] for k in ordered_keys]
    y_raw = [rows[k]["target"] for k in ordered_keys]

    X, encoded_names, categorical_columns = _encode_features(feature_rows, all_feature_names)
    groups = _resolve_groups(group_by, sample_ids, config_ids, feature_rows)

    import numpy as np

    metadata["output_sample_count"] = len(ordered_keys)
    metadata["categorical_encoding"] = dict(categorical_columns)
    metadata["group_by"] = group_by

    return InstabilityDataset(
        feature_names=all_feature_names,
        encoded_feature_names=encoded_names,
        X=X,
        y=np.asarray(y_raw, dtype=float),
        sample_ids=sample_ids,
        config_ids=config_ids,
        groups=groups,
        is_binary=spec.is_binary,
        categorical_columns=categorical_columns,
        metadata=metadata,
        warnings=warnings,
    )


def _infer_numeric_columns(
    feature_rows: list[dict[str, Any]], feature_names: list[str]
) -> list[str]:
    """A feature is numeric iff every non-null value across all rows is int/float
    (bool excluded -- Python bools are ints but should read as categorical)."""
    numeric = []
    for name in feature_names:
        non_null = [r.get(name) for r in feature_rows if r.get(name) is not None]
        if non_null and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null
        ):
            numeric.append(name)
    return numeric


def _encode_features(
    feature_rows: list[dict[str, Any]], feature_names: list[str]
) -> tuple[Any, list[str], dict[str, list[str]]]:
    """One-hot encode categoricals (sorted by name, not first-seen) and pass numerics
    through. Category order is sorted so the encoding -- and therefore the output --
    doesn't depend on input row order (required for byte-identical JSON regardless
    of file ordering).
    """
    import numpy as np

    numeric_cols = set(_infer_numeric_columns(feature_rows, feature_names))
    categorical_columns: dict[str, list[str]] = {}
    for name in feature_names:
        if name in numeric_cols:
            continue
        non_null = {str(r.get(name)) for r in feature_rows if r.get(name) is not None}
        categorical_columns[name] = sorted(non_null)

    columns: list[list[float]] = []
    encoded_names: list[str] = []
    for name in feature_names:
        if name in numeric_cols:
            encoded_names.append(name)
            columns.append([float(r.get(name)) for r in feature_rows])
            continue
        cats = categorical_columns[name]
        has_missing = any(r.get(name) is None for r in feature_rows)
        for cat in cats:
            encoded_names.append(f"{name}={cat}")
            columns.append([1.0 if str(r.get(name)) == cat else 0.0 for r in feature_rows])
        if has_missing:
            encoded_names.append(f"{name}=<missing>")
            columns.append([1.0 if r.get(name) is None else 0.0 for r in feature_rows])

    X = np.array(columns, dtype=float).T
    return X, encoded_names, categorical_columns


def _resolve_groups(
    group_by: str,
    sample_ids: list[str],
    config_ids: list[str],
    feature_rows: list[dict[str, Any]],
) -> Any:
    import numpy as np

    if group_by == "sample_id":
        keys = sample_ids
    elif group_by == "config_id":
        keys = config_ids
    else:
        keys = []
        for row in feature_rows:
            if group_by not in row:
                raise ValueError(f"group_by column {group_by!r} not found among declared features")
            v = row[group_by]
            if v is None:
                raise ValueError(
                    f"group_by column {group_by!r} has missing values; cannot form CV groups"
                )
            keys.append(v)
    unique = sorted({str(k) for k in keys})
    index = {k: i for i, k in enumerate(unique)}
    return np.array([index[str(k)] for k in keys], dtype=int)


# ---------------------------------------------------------------------------
# Phase 2: prediction model + grouped CV
#
# A simple, reproducible baseline (logistic/ridge) is the default -- "hgb" is
# opt-in, not the default, so a first run is always the cheap, inspectable
# model per AGENTS.md's "simple algorithms first". CV is grouped by
# dataset.groups (default sample_id) so the same physical sample never
# straddles train/test, matching quietset's own leakage concern for
# source_root_id/opening_family (see calibrate --group-by in quietset's README).
# ---------------------------------------------------------------------------

MODEL_TYPES = ("logistic", "ridge", "hgb")


@dataclass
class InstabilityModel:
    """A fitted estimator (on all rows) plus its grouped-CV evaluation."""

    estimator: Any
    dataset: InstabilityDataset
    model_type: str
    is_binary: bool
    group_column: str
    cv_metric_name: str
    fold_metrics: list[float]
    cv_metric_mean: float
    cv_metric_std: float
    n_folds: int
    n_groups: int
    seed: int
    metadata: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def _validate_cv_folds(dataset: InstabilityDataset, cv_folds: int) -> None:
    """Shared by fit_instability_model and make_global_value_fn -- both build a
    grouped splitter and would otherwise hit the same sklearn error deep inside
    a coalition retrain instead of a clear message up front."""
    if cv_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {cv_folds}")
    n_groups = len(set(dataset.groups.tolist()))
    if cv_folds > n_groups:
        raise ValueError(
            f"cv_folds={cv_folds} exceeds available group count {n_groups} for "
            f"group_by={dataset.metadata.get('group_by')!r}; reduce cv_folds or "
            "supply more groups"
        )
    if dataset.is_binary:
        # cv_folds <= total groups isn't enough for StratifiedGroupKFold: it also
        # needs at least one group per class in every fold, so the real ceiling is
        # the minority class's own group count, not the overall group count.
        class_groups: dict[float, set[int]] = {}
        for g, yv in zip(dataset.groups.tolist(), dataset.y.tolist()):
            class_groups.setdefault(yv, set()).add(g)
        min_class_groups = min(len(gs) for gs in class_groups.values())
        if cv_folds > min_class_groups:
            raise ValueError(
                f"cv_folds={cv_folds} exceeds the minority class's group count "
                f"({min_class_groups}) for group_by={dataset.metadata.get('group_by')!r} -- "
                "StratifiedGroupKFold needs at least one group per class in every fold; "
                "reduce cv_folds or supply more groups for the minority class"
            )


def _make_estimator(model_type: str, is_binary: bool, seed: int) -> Any:
    if model_type == "logistic":
        if not is_binary:
            raise ValueError(
                "model='logistic' requires a binary target; use 'ridge' or 'hgb' "
                "for continuous targets"
            )
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, random_state=seed)
    if model_type == "ridge":
        if is_binary:
            raise ValueError(
                "model='ridge' requires a continuous target; use 'logistic' or 'hgb' "
                "for binary targets"
            )
        from sklearn.linear_model import Ridge

        return Ridge(random_state=seed)
    if model_type == "hgb":
        if is_binary:
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(random_state=seed)
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed)
    raise ValueError(f"unknown model type {model_type!r}; choose from {MODEL_TYPES}")


def _safe_auc(y_true: Any, y_score: Any) -> float:
    """NaN (not an exception) when a fold's test split has only one class --
    AUC is undefined there, and a fold that can't be scored shouldn't crash the
    whole CV run. ``np.nanmean``/``np.nanstd`` skip these when aggregating."""
    from sklearn.metrics import roc_auc_score

    if len({float(v) for v in y_true}) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def fit_instability_model(
    dataset: InstabilityDataset,
    *,
    model: str = "logistic",
    cv_folds: int = 5,
    seed: int = 0,
    min_cv_metric: float | None = None,
) -> InstabilityModel:
    """Fit ``model`` on ``dataset`` with grouped CV, then refit once on all rows.

    ``min_cv_metric`` has no built-in default -- there is no universally-correct
    performance floor, so silence here means "no gate", not "good enough".
    When set and the mean CV metric falls short, a warning is recorded (the
    model is still returned; downstream attribution is expected to surface
    this as "insufficient evidence" rather than refuse to run).
    """
    if model not in MODEL_TYPES:
        raise ValueError(f"unknown model {model!r}; choose from {MODEL_TYPES}")

    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    X, y, groups = dataset.X, dataset.y, dataset.groups
    if dataset.is_binary and len({float(v) for v in y}) < 2:
        raise ValueError(
            "binary target has only one class across the entire dataset -- "
            "nothing to classify or attribute"
        )
    _validate_cv_folds(dataset, cv_folds)
    n_groups = len(set(groups.tolist()))

    if dataset.is_binary:
        splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = GroupKFold(n_splits=cv_folds)
        split_iter = splitter.split(X, y, groups)

    fold_metrics = []
    fold_details = []
    skip_reasons: list[str] = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        if dataset.is_binary and len({float(v) for v in y[train_idx]}) < 2:
            # A training fold with a single class can't fit a classifier at all --
            # this is a small/imbalanced-data reality, not a bug, so the fold is
            # skipped (NaN) rather than crashing the whole CV run.
            fold_metrics.append(float("nan"))
            fold_details.append(
                {
                    "fold": fold_idx,
                    "train_record_count": int(len(train_idx)),
                    "test_record_count": int(len(test_idx)),
                    "metric": None,
                    "skipped_reason": "single-class training fold",
                }
            )
            skip_reasons.append(f"fold {fold_idx}: single-class training fold")
            continue
        fold_estimator = _make_estimator(model, dataset.is_binary, seed)
        fold_estimator.fit(X[train_idx], y[train_idx])
        if dataset.is_binary:
            proba = fold_estimator.predict_proba(X[test_idx])[:, 1]
            metric = _safe_auc(y[test_idx], proba)
        else:
            pred = fold_estimator.predict(X[test_idx])
            metric = -float(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))
        fold_metrics.append(metric)
        fold_details.append(
            {
                "fold": fold_idx,
                "train_record_count": int(len(train_idx)),
                "test_record_count": int(len(test_idx)),
                "metric": metric,
            }
        )

    if all(m != m for m in fold_metrics):  # every fold NaN
        raise ValueError(
            "every CV fold was unscorable (single-class train or test split) -- "
            "the target is too imbalanced or too small for the requested cv_folds/group_by; "
            "reduce cv_folds or check target_summary"
        )

    cv_metric_mean = float(np.nanmean(fold_metrics))
    cv_metric_std = float(np.nanstd(fold_metrics))

    warnings: list[str] = list(skip_reasons)
    n_unscored = sum(1 for m in fold_metrics if m != m) - len(skip_reasons)  # NaN != NaN
    if n_unscored > 0:
        warnings.append(
            f"{n_unscored}/{len(fold_metrics)} fold(s) had only one class in the test "
            "split and could not be scored (AUC undefined); excluded from the mean"
        )
    if min_cv_metric is not None and cv_metric_mean < min_cv_metric:
        warnings.append(
            f"cv_metric_mean {cv_metric_mean:.4f} is below min_cv_metric {min_cv_metric} "
            "-- treat downstream ASV attribution as insufficient-evidence, not a reliable "
            "ranking of factors"
        )

    final_estimator = _make_estimator(model, dataset.is_binary, seed)
    final_estimator.fit(X, y)

    if dataset.is_binary:
        target_summary = {"prevalence": float(np.mean(y))}
    else:
        target_summary = {
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "min": float(np.min(y)),
            "max": float(np.max(y)),
        }

    metric_name = "roc_auc" if dataset.is_binary else "neg_rmse"
    group_column = dataset.metadata.get("group_by", "sample_id")

    metadata = {
        "model_type": model,
        "cv_metric_name": metric_name,
        "cv_folds": cv_folds,
        "n_groups": n_groups,
        "seed": seed,
        "record_count": int(len(y)),
        "target_summary": target_summary,
        "fold_details": fold_details,
        "feature_encoding": {
            "encoded_feature_names": dataset.encoded_feature_names,
            "categorical_columns": dataset.categorical_columns,
        },
        "min_cv_metric": min_cv_metric,
        "group_column": group_column,
    }

    return InstabilityModel(
        estimator=final_estimator,
        dataset=dataset,
        model_type=model,
        is_binary=dataset.is_binary,
        group_column=group_column,
        cv_metric_name=metric_name,
        fold_metrics=fold_metrics,
        cv_metric_mean=cv_metric_mean,
        cv_metric_std=cv_metric_std,
        n_folds=cv_folds,
        n_groups=n_groups,
        seed=seed,
        metadata=metadata,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Phase 3: DAG-aware attribution
#
# ASV nodes are the *raw* (pre-encoding) feature names, e.g. "evaluator_family" --
# not its one-hot columns "evaluator_family=family_x". A DAG describing causal
# ordering among semantic factors doesn't decompose per dummy category, and letting
# it would blow up DAG size with high-cardinality categoricals. Both value function
# builders below therefore accept coalitions of raw feature names and expand each
# into its encoded column group internally.
# ---------------------------------------------------------------------------

#: Sink node name used in the example DAGs (docs/integrations/...): an explanatory
#: "this is what we're predicting" label, not a feature the model can condition on.
#: Stripped from the DAG (see load_attribution_dag) rather than given an ASV value,
#: per the instruction that it needs no ASV of its own.
DEFAULT_SINK_NODE = "instability_prediction"


def _feature_column_groups(dataset: InstabilityDataset) -> dict[str, list[str]]:
    """Map each raw feature name to the encoded column name(s) it expands into."""
    groups: dict[str, list[str]] = {}
    for name in dataset.feature_names:
        if name in dataset.categorical_columns:
            prefix = f"{name}="
            groups[name] = [c for c in dataset.encoded_feature_names if c.startswith(prefix)]
        else:
            groups[name] = [name]
    return groups


def _dag_json(nodes: list[str], edges: list[tuple[str, str]]) -> str:
    """Build a CausalDAG.from_json-compatible string.

    MUST use compact separators: CausalDAG.from_json's parser is a hand-rolled
    substring search for the literal ``"from":"`` / ``"to":"`` / ``"nodes":[``
    patterns (see src/python.rs) -- ``json.dumps``'s default ``", "``/``": "``
    spacing makes every one of those searches miss, silently producing an EMPTY
    graph (no error). Verified empirically before relying on it here.
    """
    return json.dumps(
        {"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]},
        separators=(",", ":"),
    )


def load_attribution_dag(
    path: str, feature_names: list[str], *, sink_node: str = DEFAULT_SINK_NODE
):
    """Load a DAG in ``CausalDAG.to_json()``/``from_json()`` format for attribution.

    ``sink_node`` (default ``"instability_prediction"``), if present, is stripped
    along with its incoming edges -- it's the model's output, not an attributable
    input, so it never needs an ASV of its own. It's rejected outright if it has
    outgoing edges (that would make it an intermediate node, not a sink).

    After stripping, the remaining DAG node set must equal ``feature_names``
    exactly; a mismatch (missing or unknown-to-the-dataset nodes) is rejected
    with the specific names, never silently reconciled.
    """
    raw = json.loads(Path(path).read_text())
    nodes = list(raw.get("nodes", []))
    if not nodes:
        raise ValueError(f"DAG at {path!r} has no nodes")
    edges = [(e["from"], e["to"]) for e in raw.get("edges", [])]

    if sink_node in nodes:
        out_edges = [e for e in edges if e[0] == sink_node]
        if out_edges:
            raise ValueError(
                f"sink node {sink_node!r} has outgoing edge(s) {out_edges!r} -- it must be "
                "a pure sink (no descendants) to be treated as an explanatory target"
            )
        nodes = [n for n in nodes if n != sink_node]
        edges = [e for e in edges if e[1] != sink_node]

    node_set, feature_set = set(nodes), set(feature_names)
    missing_from_dag = feature_set - node_set
    extra_in_dag = node_set - feature_set
    if missing_from_dag or extra_in_dag:
        raise ValueError(
            "DAG node set does not match the dataset's feature set -- "
            f"missing from DAG: {sorted(missing_from_dag)}; "
            f"nodes in DAG that aren't a dataset feature: {sorted(extra_in_dag)}"
        )

    dag = CausalDAG.from_json(_dag_json(nodes, edges))
    dag.validate()
    return dag


# --- Global mode: retrain-per-coalition held-out predictive quality ---------

_BINARY_METRICS: dict[str, Callable[[Any, Any], float]] = {}
_CONTINUOUS_METRICS: dict[str, Callable[[Any, Any], float]] = {}


def _neg_log_loss(y_true: Any, proba: Any) -> float:
    from sklearn.metrics import log_loss

    if len({float(v) for v in y_true}) < 2:
        return float("nan")
    return -float(log_loss(y_true, proba, labels=[0.0, 1.0]))


def _neg_rmse(y_true: Any, pred: Any) -> float:
    import numpy as np

    return -float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(y_true)) ** 2)))


_BINARY_METRICS["neg_log_loss"] = _neg_log_loss
_BINARY_METRICS["auc"] = _safe_auc
_CONTINUOUS_METRICS["neg_rmse"] = _neg_rmse


def _empty_coalition_predict(y_train: Any, n_test: int, is_binary: bool) -> Any:
    """v(empty coalition): the "no information" model -- predict the training
    target's marginal (prevalence for binary, mean for continuous) for every
    held-out row. The conventional Shapley baseline: what you'd get by knowing
    nothing about any feature."""
    import numpy as np

    return np.full(n_test, float(np.mean(y_train)))


def make_global_value_fn(
    dataset: InstabilityDataset,
    *,
    model: str = "logistic",
    cv_folds: int = 5,
    seed: int = 0,
    metric: str | None = None,
) -> Callable[[list[str]], float]:
    """Build a value function: coalition of raw feature names -> held-out
    predictive quality using only those features, via the same grouped CV as
    ``fit_instability_model``.

    Retraining is the expensive part, so results are memoized by
    ``tuple(sorted(coalition))`` on the returned closure. Call this ONCE per
    (dataset, model, cv_folds, seed) and reuse the same value_fn across
    multiple seeds/DAGs (e.g. in a rank-stability sweep) -- constructing a new
    one per call throws the cache away and repeats every retrain.
    """
    import numpy as np
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

    _validate_cv_folds(dataset, cv_folds)

    is_binary = dataset.is_binary
    metrics = _BINARY_METRICS if is_binary else _CONTINUOUS_METRICS
    metric_name = metric or ("neg_log_loss" if is_binary else "neg_rmse")
    if metric_name not in metrics:
        raise ValueError(
            f"unknown metric {metric_name!r} for this target; choose from {list(metrics)}"
        )
    score_fn = metrics[metric_name]

    groups_map = _feature_column_groups(dataset)
    col_index = {name: i for i, name in enumerate(dataset.encoded_feature_names)}
    feature_set = set(dataset.feature_names)

    cache: dict[tuple[str, ...], float] = {}

    def value_fn(coalition: list[str]) -> float:
        key = tuple(sorted(coalition))
        if key in cache:
            return cache[key]

        unknown = set(coalition) - feature_set
        if unknown:
            raise ValueError(f"unknown feature(s) in coalition: {sorted(unknown)}")
        col_idx = sorted({col_index[c] for name in coalition for c in groups_map[name]})

        if is_binary:
            splitter = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(dataset.X, dataset.y, dataset.groups)
        else:
            splitter = GroupKFold(n_splits=cv_folds)
            split_iter = splitter.split(dataset.X, dataset.y, dataset.groups)

        fold_scores = []
        for train_idx, test_idx in split_iter:
            y_train, y_test = dataset.y[train_idx], dataset.y[test_idx]
            if not col_idx:
                pred = _empty_coalition_predict(y_train, len(test_idx), is_binary)
            else:
                if is_binary and len({float(v) for v in y_train}) < 2:
                    continue  # unscorable fold (see fit_instability_model), skip it
                est = _make_estimator(model, is_binary, seed)
                x_train = dataset.X[np.ix_(train_idx, col_idx)]
                x_test = dataset.X[np.ix_(test_idx, col_idx)]
                est.fit(x_train, y_train)
                pred = est.predict_proba(x_test)[:, 1] if is_binary else est.predict(x_test)
            score = score_fn(y_test, pred)
            if score == score:  # skip NaN (unscorable fold)
                fold_scores.append(score)

        value = float(np.mean(fold_scores)) if fold_scores else float("nan")
        cache[key] = value
        return value

    return value_fn


# --- Local mode: single-instance prediction, absent features baseline-filled ---


def make_local_value_fn(
    model_result: InstabilityModel,
    sample_index: int,
    *,
    baseline: str = "mean",
) -> Callable[[list[str]], float]:
    """Build a value function: coalition of raw feature names -> the fitted
    model's prediction for one row, with absent features replaced per
    ``baseline``.

    Delegates entirely to ``causasv.helpers.make_tabular_value_fn`` -- the only
    new logic here is expanding each raw feature name into the encoded column(s)
    it maps to (see module docstring), so one-hot categoricals are included or
    excluded as a whole logical feature, not one dummy column at a time.
    """
    dataset = model_result.dataset
    groups_map = _feature_column_groups(dataset)
    feature_set = set(dataset.feature_names)
    x_row = dataset.X[sample_index]

    inner = make_tabular_value_fn(
        model_result.estimator,
        x_row,
        dataset.X,
        dataset.encoded_feature_names,
        baseline=baseline,
    )

    def value_fn(coalition: list[str]) -> float:
        unknown = set(coalition) - feature_set
        if unknown:
            raise ValueError(f"unknown feature(s) in coalition: {sorted(unknown)}")
        expanded = [c for name in coalition for c in groups_map[name]]
        return inner(expanded)

    return value_fn


# ---------------------------------------------------------------------------
# Phase 4: uncertainty + DAG-ensemble sensitivity
#
# Single-DAG diagnostics are just causasv.helpers.explain_safe -- it already
# does exact-first / CI-aware-approx-fallback plus ESS-ratio and seed
# rank-stability warnings, so there's nothing to add here. The one piece that
# doesn't exist yet is cross-DAG sensitivity: ASVEnsembleExplainer.explain_with_
# sensitivity calls bare explain() (no stderr/CI/ESS), which loses exactly the
# diagnostics this schema requires, so it's not used here -- explain_safe is run
# once per DAG instead, and the mean/std/rank-stability logic is re-derived
# (small enough that importing helpers.py's underscore-prefixed private
# functions across module boundaries isn't worth it).
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 5: output schema
#
# Pure data assembly -- no new ASV computation happens here. Everything in the
# report comes from Phase 1-4 outputs; this just shapes them into
# causasv-instability-attribution-v1 and sorts/serializes deterministically
# (json.dumps(report, sort_keys=True) is byte-identical across repeated runs
# given the same seed and input, since every upstream stage is seeded).
# ---------------------------------------------------------------------------

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
