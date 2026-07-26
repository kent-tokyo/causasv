"""Phase 1: bundle manifest adapter -- JSONL read, feature classification, validation.

quietset itself is never imported or modified; this only reads its
Observation/StabilityReport JSONL by field name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

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
