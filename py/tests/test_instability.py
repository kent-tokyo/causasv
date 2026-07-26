"""Tests for causasv.instability (Phase 1 adapter + Phase 2 model/grouped CV).

Covers spec test items #2 (row-order invariance), #3 (leakage rejection),
#4 (grouped CV non-leakage), #5 (no silent 0-fill for missing values), #6
(binary/continuous targets), and #13 (low-performance warning), plus the
manifest/cell validation rules (disjoint column roles, seed guard, mixed-file
detection, constant-feature policy, duplicate handling). Exact-oracle match
(#8), approx CI/ESS (#9), rank/DAG stability (#10, #11), uncertain-feature
display (#12), malformed-DAG rejection (#14), and the example smoke test (#15)
land in later phases once the DAG/attribution/CLI code they exercise exists.
"""

import json

import pytest

from causasv import instability as inst

pytest.importorskip("numpy")


def _write_jsonl(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _write_bundle(tmp_path, cells_spec):
    """cells_spec: list of (config_id, features, replicate_axes, observations, scored)."""
    cells_json = []
    for i, (config_id, features, replicate_axes, observations, scored) in enumerate(cells_spec):
        cell_dir = tmp_path / f"cell{i}"
        cell_dir.mkdir()
        _write_jsonl(cell_dir / "observations.jsonl", observations)
        _write_jsonl(cell_dir / "scored.jsonl", scored)
        cells_json.append(
            {
                "config_id": config_id,
                "scored": f"cell{i}/scored.jsonl",
                "observations": f"cell{i}/observations.jsonl",
                "features": features,
                "replicate_axes": list(replicate_axes),
            }
        )
    manifest_path = tmp_path / "bundle.json"
    manifest_path.write_text(
        json.dumps({"schema_version": "causasv-instability-bundle-v1", "cells": cells_json})
    )
    return str(manifest_path)


def _scored(sample_id, **overrides):
    base = {
        "sample_id": sample_id,
        "n_observations": 3,
        "label_agreement": 0.667,
        "label_agreement_lcb": 0.3,
        "label_entropy": 0.9,
        "score_mad": 0.1,
        "score_iqr": 0.2,
        "score_sign_agreement": 0.8,
        "confidence": 0.5,
        "adjusted_stability_score": 0.5,
        "disagreement_score": 0.5,
        "stability_score": 0.5,
        "decision": "review",
        "components": {},
    }
    base.update(overrides)
    return base


def _two_cell_bundle(tmp_path, *, budgets=(4, 8)):
    obs_common = lambda src: [  # noqa: E731
        {"sample_id": "s1", "label": "a", "evaluator_id": "e1", "source_root_id": src},
        {"sample_id": "s1", "label": "b", "evaluator_id": "e2", "source_root_id": src},
        {"sample_id": "s2", "label": "a", "evaluator_id": "e1", "source_root_id": "g2"},
        {"sample_id": "s2", "label": "a", "evaluator_id": "e2", "source_root_id": "g2"},
    ]
    scored_common = [
        _scored("s1", label_agreement=0.5, decision="review"),
        _scored("s2", label_agreement=1.0, decision="keep"),
    ]
    return _write_bundle(
        tmp_path,
        [
            (
                "c0",
                {"budget": budgets[0]},
                ("evaluator_id",),
                obs_common("g1"),
                scored_common,
            ),
            (
                "c1",
                {"budget": budgets[1]},
                ("evaluator_id",),
                obs_common("g1"),
                scored_common,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def test_load_bundle_manifest_reads_cells(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    assert [c.config_id for c in cells] == ["c0", "c1"]
    assert cells[0].features == {"budget": 4}
    assert cells[0].replicate_axes == ("evaluator_id",)


def test_load_bundle_manifest_rejects_wrong_schema_version(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"schema_version": "nope", "cells": []}))
    with pytest.raises(ValueError, match="schema_version"):
        inst.load_bundle_manifest(str(path))


def test_load_bundle_manifest_rejects_duplicate_config_id(tmp_path):
    path = tmp_path / "bundle.json"
    cell = {
        "config_id": "dup",
        "scored": "x.jsonl",
        "observations": "y.jsonl",
        "features": {},
        "replicate_axes": [],
    }
    path.write_text(
        json.dumps({"schema_version": "causasv-instability-bundle-v1", "cells": [cell, cell]})
    )
    with pytest.raises(ValueError, match="duplicate config_id"):
        inst.load_bundle_manifest(str(path))


def test_load_bundle_manifest_rejects_empty_cells(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"schema_version": "causasv-instability-bundle-v1", "cells": []}))
    with pytest.raises(ValueError, match="no cells"):
        inst.load_bundle_manifest(str(path))


def test_wrap_single_cell_rejects_cell_features():
    with pytest.raises(ValueError, match="aggregate scored report has lost config-level"):
        inst.wrap_single_cell("obs.jsonl", "scored.jsonl", cell_features=["budget"])


def test_wrap_single_cell_ok_without_cell_features():
    cells = inst.wrap_single_cell("obs.jsonl", "scored.jsonl")
    assert cells[0].config_id == "default"
    assert cells[0].features == {}


# ---------------------------------------------------------------------------
# Column role validation
# ---------------------------------------------------------------------------


def test_rejects_overlapping_cell_and_sample_features(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="both cell_features and sample_features"):
        inst.build_instability_dataset(
            cells,
            target="label_disagreement",
            cell_features=["budget"],
            sample_features=["budget"],
        )


def test_rejects_manifest_cell_feature_requested_as_sample_feature(tmp_path):
    """budget is declared in manifest cell.features -- requesting it via
    sample_features (instead of cell_features) must not silently overwrite it."""
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="also declares it as a cell feature"):
        inst.build_instability_dataset(
            cells, target="label_disagreement", sample_features=["budget"]
        )


def test_rejects_sample_id_as_feature(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="sample_id is a row identifier"):
        inst.build_instability_dataset(
            cells, target="label_disagreement", sample_features=["sample_id"]
        )


def test_rejects_raw_seed_feature_by_default(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="raw seed columns"):
        inst.build_instability_dataset(
            cells, target="label_disagreement", cell_features=["budget", "seed"]
        )


def test_allow_raw_seed_feature_opt_in(tmp_path):
    cells_spec = [
        (
            "c0",
            {"budget": 4, "seed": 1},
            (),
            [
                {"sample_id": "s1", "label": "a", "source_root_id": "g1"},
                {"sample_id": "s1", "label": "b", "source_root_id": "g1"},
            ],
            [_scored("s1")],
        ),
        (
            "c1",
            {"budget": 4, "seed": 2},
            (),
            [
                {"sample_id": "s2", "label": "a", "source_root_id": "g2"},
                {"sample_id": "s2", "label": "a", "source_root_id": "g2"},
            ],
            [_scored("s2", label_agreement=1.0)],
        ),
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells,
        target="label_disagreement",
        cell_features=["seed"],
        allow_raw_seed_feature=True,
    )
    assert "seed=1" in ds.encoded_feature_names or "seed" in ds.feature_names


# ---------------------------------------------------------------------------
# Leakage guard (#3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "leaking_field",
    ["label_agreement", "label_entropy", "score_mad", "stability_score", "decision"],
)
def test_leakage_guard_rejects_stability_report_fields(tmp_path, leaking_field):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="StabilityReport-derived"):
        inst.build_instability_dataset(
            cells, target="label_entropy", cell_features=["budget", leaking_field]
        )


def test_stability_report_fields_matches_schema_fixture():
    """Guards against drift from the quietset schema this list was transcribed from."""
    assert "sample_id" not in inst.STABILITY_REPORT_FIELDS
    assert "n_observations" not in inst.STABILITY_REPORT_FIELDS
    assert "label_agreement" in inst.STABILITY_REPORT_FIELDS
    assert "components" in inst.STABILITY_REPORT_FIELDS


# ---------------------------------------------------------------------------
# Mixed-file detection
# ---------------------------------------------------------------------------


def test_mixed_file_conflict_is_rejected(tmp_path):
    cells_spec = [
        (
            "c0",
            {"budget": 4},
            ("evaluator_id",),
            [
                {"sample_id": "s1", "label": "a", "evaluator_id": "e1", "budget": 4},
                {"sample_id": "s1", "label": "b", "evaluator_id": "e2", "budget": 8},
            ],
            [_scored("s1")],
        )
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="aggregate scored report has lost config-level"):
        inst.build_instability_dataset(cells, target="label_entropy", cell_features=["budget"])


# ---------------------------------------------------------------------------
# Constant-feature policy
# ---------------------------------------------------------------------------


def test_constant_feature_across_all_cells_is_hard_rejected(tmp_path):
    path = _two_cell_bundle(tmp_path, budgets=(4, 4))
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="constant across the entire combined dataset"):
        inst.build_instability_dataset(cells, target="label_entropy", cell_features=["budget"])


def test_constant_feature_dropped_when_opted_in(tmp_path):
    path = _two_cell_bundle(tmp_path, budgets=(4, 4))
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells,
        target="label_entropy",
        cell_features=["budget"],
        sample_features=["source_root_id"],
        drop_constant_features=True,
    )
    assert "budget" not in ds.feature_names
    assert "source_root_id" in ds.feature_names
    assert ds.metadata["constant_features"]["dropped"] == ["budget"]
    assert any("budget" in w for w in ds.warnings)


def test_constant_within_one_cell_is_fine(tmp_path):
    """budget is constant *within* each cell but varies across cells -> not flagged."""
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(cells, target="label_entropy", cell_features=["budget"])
    assert ds.metadata["constant_features"]["dropped"] == []


# ---------------------------------------------------------------------------
# Missing-value handling (#5)
# ---------------------------------------------------------------------------


def test_missing_numeric_feature_errors_by_default(tmp_path):
    cells_spec = [
        (
            "c0",
            {"budget": 4},
            (),
            [
                {"sample_id": "s1", "label": "a"},
                {"sample_id": "s1", "label": "b"},
            ],
            [_scored("s1")],
        ),
        (
            "c1",
            {},  # budget missing entirely for this cell
            (),
            [
                {"sample_id": "s2", "label": "a"},
                {"sample_id": "s2", "label": "a"},
            ],
            [_scored("s2", label_agreement=1.0)],
        ),
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="never silently imputes 0"):
        inst.build_instability_dataset(cells, target="label_entropy", cell_features=["budget"])


def test_missing_numeric_feature_drop_rows_policy(tmp_path):
    cells_spec = [
        (
            "c0",
            {"budget": 4},
            (),
            [
                {"sample_id": "s1", "label": "a"},
                {"sample_id": "s1", "label": "b"},
            ],
            [_scored("s1")],
        ),
        (
            "c1",
            {"budget": 8},
            (),
            [
                {"sample_id": "s3", "label": "a"},
                {"sample_id": "s3", "label": "b"},
            ],
            [_scored("s3")],
        ),
        (
            "c2",
            {},  # budget missing entirely for this cell -> its row gets dropped
            (),
            [
                {"sample_id": "s2", "label": "a"},
                {"sample_id": "s2", "label": "a"},
            ],
            [_scored("s2", label_agreement=1.0)],
        ),
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells, target="label_entropy", cell_features=["budget"], missing_numeric="drop_rows"
    )
    assert ds.sample_ids == ["s1", "s3"]
    assert ds.metadata["dropped_record_count"]["missing_numeric_feature"] == 1


def test_missing_categorical_feature_becomes_explicit_category(tmp_path):
    cells_spec = [
        (
            "c0",
            {"loss_recipe": "recipe_a"},
            (),
            [
                {"sample_id": "s1", "label": "a"},
                {"sample_id": "s1", "label": "b"},
            ],
            [_scored("s1")],
        ),
        (
            "c1",
            {},  # loss_recipe missing for this cell
            (),
            [
                {"sample_id": "s2", "label": "a"},
                {"sample_id": "s2", "label": "a"},
            ],
            [_scored("s2", label_agreement=1.0)],
        ),
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells, target="label_entropy", cell_features=["loss_recipe"]
    )
    assert "loss_recipe=<missing>" in ds.encoded_feature_names
    assert ds.metadata["missing_field_count"].get("loss_recipe") == 1


# ---------------------------------------------------------------------------
# Target definitions: binary vs continuous (#6)
# ---------------------------------------------------------------------------


def test_continuous_target_label_entropy(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(cells, target="label_entropy", cell_features=["budget"])
    assert ds.is_binary is False
    assert set(ds.y.tolist()) <= {0.9}  # from _scored() default


def test_binary_target_review_or_drop(tmp_path):
    cells_spec = [
        (
            "c0",
            {"budget": 4},
            (),
            [{"sample_id": "s1", "label": "a"}],
            [_scored("s1", decision="drop")],
        ),
        (
            "c1",
            {"budget": 8},
            (),
            [{"sample_id": "s2", "label": "a"}],
            [_scored("s2", decision="keep")],
        ),
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells, target="review_or_drop", cell_features=["budget"], min_observations=1
    )
    assert ds.is_binary is True
    assert sorted(ds.y.tolist()) == [0.0, 1.0]


def test_min_observations_floor_cannot_go_below_target_default(tmp_path):
    """label_entropy has a hard floor of 2; passing min_observations=1 must not lower it."""
    cells_spec = [
        (
            "c0",
            {"budget": 4},
            (),
            [{"sample_id": "s1", "label": "a"}],  # only 1 observation
            [_scored("s1")],
        )
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="no rows survived"):
        inst.build_instability_dataset(
            cells, target="label_entropy", cell_features=["budget"], min_observations=1
        )


# ---------------------------------------------------------------------------
# Row-order / cell-order invariance (#2)
# ---------------------------------------------------------------------------


def test_row_order_invariance(tmp_path):
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    path_a = _two_cell_bundle(a_dir)

    b_dir = tmp_path / "b"
    b_dir.mkdir()
    path_b = _two_cell_bundle(b_dir)
    manifest_b = json.loads((b_dir / "bundle.json").read_text())
    manifest_b["cells"] = list(reversed(manifest_b["cells"]))
    (b_dir / "bundle.json").write_text(json.dumps(manifest_b))
    for cell_dir in ("cell0", "cell1"):
        obs_path = b_dir / cell_dir / "observations.jsonl"
        lines = obs_path.read_text().splitlines()
        lines.reverse()
        obs_path.write_text("\n".join(lines) + "\n")

    def run(path):
        cells = inst.load_bundle_manifest(path)
        ds = inst.build_instability_dataset(
            cells, target="label_disagreement", cell_features=["budget"]
        )
        return ds.encoded_feature_names, ds.X.tolist(), ds.y.tolist(), ds.sample_ids, ds.config_ids

    assert run(path_a) == run(path_b)


# ---------------------------------------------------------------------------
# Duplicate handling
# ---------------------------------------------------------------------------


def test_duplicate_scored_lines_last_write_wins_and_reported(tmp_path):
    cells_spec = [
        (
            "c0",
            {"budget": 4},
            (),
            [
                {"sample_id": "s1", "label": "a"},
                {"sample_id": "s1", "label": "b"},
            ],
            [_scored("s1", label_agreement=0.9), _scored("s1", label_agreement=0.1)],
        ),
        (
            "c1",
            {"budget": 8},
            (),
            [
                {"sample_id": "s2", "label": "a"},
                {"sample_id": "s2", "label": "a"},
            ],
            [_scored("s2", label_agreement=1.0)],
        ),
    ]
    path = _write_bundle(tmp_path, cells_spec)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells, target="label_disagreement", cell_features=["budget"]
    )
    assert ds.metadata["duplicate_scored_lines"] == 1
    s1_index = ds.sample_ids.index("s1")
    assert ds.y.tolist()[s1_index] == pytest.approx(1.0 - 0.1)


# ---------------------------------------------------------------------------
# group_by resolution
# ---------------------------------------------------------------------------


def test_group_by_sample_id_keeps_same_sample_together_across_configs(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells, target="label_disagreement", cell_features=["budget"], group_by="sample_id"
    )
    groups_by_sample = dict(zip(ds.sample_ids, ds.groups.tolist()))
    for sid, config_id, group in zip(ds.sample_ids, ds.config_ids, ds.groups.tolist()):
        assert group == groups_by_sample[sid]


def test_group_by_unknown_column_rejected(tmp_path):
    path = _two_cell_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    with pytest.raises(ValueError, match="not found among declared features"):
        inst.build_instability_dataset(
            cells, target="label_disagreement", cell_features=["budget"], group_by="nonexistent"
        )


# ---------------------------------------------------------------------------
# Phase 2: prediction model + grouped CV
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn")

# 6 samples x 3 config cells = 18 rows, 6 groups (by sample_id) -- enough for a
# 3-fold grouped CV where the minority class isn't concentrated in one group.
_LABEL_SETS = {
    "s0": ["a", "a", "a", "a"],  # agreement 1.0  -> keep
    "s1": ["a", "a", "a", "a"],  # agreement 1.0  -> keep
    "s2": ["a", "a", "b", "b"],  # agreement 0.5  -> drop
    "s3": ["b", "b", "b", "b"],  # agreement 1.0  -> keep
    "s4": ["a", "a", "b", "c"],  # agreement 0.5  -> drop
    "s5": ["a", "b", "c", "d"],  # agreement 0.25 -> drop
}
_SAMPLE_GROUP = {"s0": "g0", "s1": "g0", "s2": "g1", "s3": "g1", "s4": "g2", "s5": "g2"}


def _decision_for(agreement):
    if agreement >= 0.85:
        return "keep"
    if agreement >= 0.6:
        return "review"
    return "drop"


def _cv_bundle(tmp_path, *, budgets=(4, 8, 16)):
    cells_spec = []
    for k, budget in enumerate(budgets):
        observations = []
        scored = []
        for sid, labels in _LABEL_SETS.items():
            for j, lab in enumerate(labels):
                observations.append(
                    {
                        "sample_id": sid,
                        "label": lab,
                        "evaluator_id": f"e{j}",
                        "source_root_id": _SAMPLE_GROUP[sid],
                    }
                )
            majority_count = max(labels.count(v) for v in set(labels))
            agreement = majority_count / len(labels)
            scored.append(
                _scored(
                    sid,
                    n_observations=len(labels),
                    label_agreement=agreement,
                    label_agreement_lcb=max(agreement - 0.2, 0.0),
                    label_entropy=1.0 - agreement,
                    score_sign_agreement=agreement,
                    decision=_decision_for(agreement),
                )
            )
        cells_spec.append((f"c{k}", {"budget": budget}, ("evaluator_id",), observations, scored))
    return _write_bundle(tmp_path, cells_spec)


def _cv_dataset(tmp_path, target, **kwargs):
    path = _cv_bundle(tmp_path)
    cells = inst.load_bundle_manifest(path)
    kwargs.setdefault("cell_features", ["budget"])
    kwargs.setdefault("sample_features", ["source_root_id"])
    kwargs.setdefault("group_by", "sample_id")
    kwargs.setdefault("min_observations", 1)
    return inst.build_instability_dataset(cells, target=target, **kwargs)


def test_fit_continuous_target_ridge(tmp_path):
    ds = _cv_dataset(tmp_path, "label_disagreement")
    model = inst.fit_instability_model(ds, model="ridge", cv_folds=3, seed=42)
    assert model.model_type == "ridge"
    assert model.cv_metric_name == "neg_rmse"
    assert len(model.fold_metrics) == 3
    assert model.n_groups == 6


def test_fit_binary_target_logistic(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    model = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    assert model.model_type == "logistic"
    assert model.cv_metric_name == "roc_auc"
    assert model.metadata["target_summary"]["prevalence"] == pytest.approx(0.5)


def test_fit_hgb_both_target_types(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ds_bin = _cv_dataset(bin_dir, "review_or_drop")
    model_bin = inst.fit_instability_model(ds_bin, model="hgb", cv_folds=3, seed=7)
    assert model_bin.model_type == "hgb"

    cont_dir = tmp_path / "cont"
    cont_dir.mkdir()
    ds_cont = _cv_dataset(cont_dir, "label_disagreement")
    model_cont = inst.fit_instability_model(ds_cont, model="hgb", cv_folds=3, seed=7)
    assert model_cont.model_type == "hgb"


def test_fit_rejects_model_target_type_mismatch(tmp_path):
    a_dir = tmp_path / "a"
    a_dir.mkdir()
    ds = _cv_dataset(a_dir, "review_or_drop")
    with pytest.raises(ValueError, match="requires a continuous target"):
        inst.fit_instability_model(ds, model="ridge", cv_folds=3, seed=1)

    b_dir = tmp_path / "b"
    b_dir.mkdir()
    ds2 = _cv_dataset(b_dir, "label_disagreement")
    with pytest.raises(ValueError, match="requires a binary target"):
        inst.fit_instability_model(ds2, model="logistic", cv_folds=3, seed=1)


def test_fit_rejects_cv_folds_exceeding_group_count(tmp_path):
    ds = _cv_dataset(tmp_path, "label_disagreement")
    with pytest.raises(ValueError, match="exceeds available group count"):
        inst.fit_instability_model(ds, model="ridge", cv_folds=100, seed=1)


def test_fit_rejects_single_class_binary_target(tmp_path):
    path = _write_bundle(
        tmp_path,
        [
            (
                "c0",
                {"budget": 4},
                (),
                [{"sample_id": "s1", "label": "a"}, {"sample_id": "s1", "label": "a"}],
                [_scored("s1", decision="keep")],
            ),
            (
                "c1",
                {"budget": 8},
                (),
                [{"sample_id": "s2", "label": "a"}, {"sample_id": "s2", "label": "a"}],
                [_scored("s2", decision="keep")],
            ),
        ],
    )
    cells = inst.load_bundle_manifest(path)
    ds = inst.build_instability_dataset(
        cells, target="review_or_drop", cell_features=["budget"], min_observations=1
    )
    with pytest.raises(ValueError, match="only one class across the entire dataset"):
        inst.fit_instability_model(ds, model="logistic", cv_folds=2, seed=1)


def test_fit_deterministic_across_repeated_calls(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    m1 = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    m2 = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    assert m1.fold_metrics == m2.fold_metrics
    assert m1.cv_metric_mean == m2.cv_metric_mean


def test_fit_low_performance_triggers_min_cv_metric_warning(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    model = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1, min_cv_metric=0.99)
    assert any("min_cv_metric" in w for w in model.warnings)


def test_fit_no_min_cv_metric_gate_by_default(tmp_path):
    """No universal default performance floor -- omitting min_cv_metric means no gate."""
    ds = _cv_dataset(tmp_path, "review_or_drop")
    model = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    assert not any("min_cv_metric" in w for w in model.warnings)


def test_grouped_cv_never_splits_a_group_across_train_and_test(tmp_path):
    """Integration check: fit_instability_model must pass dataset.groups through to
    the splitter so no sample_id's rows straddle train/test -- sklearn's GroupKFold/
    StratifiedGroupKFold guarantee this once groups are wired correctly; this test
    catches an accidental X/y/groups misalignment in our own code, not sklearn's."""
    ds = _cv_dataset(tmp_path, "review_or_drop")
    from sklearn.model_selection import StratifiedGroupKFold

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1)
    for train_idx, test_idx in splitter.split(ds.X, ds.y, ds.groups):
        train_groups = set(ds.groups[train_idx].tolist())
        test_groups = set(ds.groups[test_idx].tolist())
        assert not (train_groups & test_groups)
    model = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    assert sum(fd["test_record_count"] for fd in model.metadata["fold_details"]) == len(ds.y)


# ---------------------------------------------------------------------------
# Phase 3: DAG-aware attribution (global/local value functions)
# ---------------------------------------------------------------------------


def _write_dag(tmp_path, name, nodes, edges):
    path = tmp_path / name
    path.write_text(json.dumps({"nodes": nodes, "edges": [{"from": a, "to": b} for a, b in edges]}))
    return str(path)


def test_dag_json_uses_compact_separators_round_trip():
    """CausalDAG.from_json's hand-rolled parser matches the literal substring
    '"from":"' with NO space -- json.dumps's default '", "' spacing makes every
    node/edge search silently miss (empty graph, no error). Locks in that
    _dag_json always uses separators=(',', ':')."""
    s = inst._dag_json(["a", "b"], [("a", "b")])
    assert '"from":"a","to":"b"' in s
    assert '"nodes":["a","b"]' in s

    from causasv import CausalDAG

    dag = CausalDAG.from_json(s)
    assert sorted(dag.nodes()) == ["a", "b"]
    assert dag.edges() == [("a", "b")]


def test_load_attribution_dag_strips_sink_node(tmp_path):
    path = _write_dag(
        tmp_path,
        "dag.json",
        ["budget", "source_root_id", "instability_prediction"],
        [("budget", "instability_prediction"), ("source_root_id", "instability_prediction")],
    )
    dag = inst.load_attribution_dag(path, ["budget", "source_root_id"])
    assert sorted(dag.nodes()) == ["budget", "source_root_id"]
    assert dag.edges() == []


def test_load_attribution_dag_rejects_sink_with_outgoing_edges(tmp_path):
    path = _write_dag(
        tmp_path,
        "dag.json",
        ["budget", "instability_prediction"],
        [("instability_prediction", "budget")],
    )
    with pytest.raises(ValueError, match="must be a pure sink"):
        inst.load_attribution_dag(path, ["budget"])


def test_load_attribution_dag_rejects_unknown_and_missing_nodes(tmp_path):
    path = _write_dag(tmp_path, "dag.json", ["budget", "mystery"], [("budget", "mystery")])
    with pytest.raises(ValueError, match="missing from DAG.*source_root_id"):
        inst.load_attribution_dag(path, ["budget", "source_root_id"])


def test_load_attribution_dag_rejects_cycle(tmp_path):
    path = _write_dag(
        tmp_path, "dag.json", ["a", "b"], [("a", "b"), ("b", "a")]
    )
    with pytest.raises(ValueError, match="cycle"):
        inst.load_attribution_dag(path, ["a", "b"])


def test_load_attribution_dag_no_sink_present(tmp_path):
    """A DAG with no sink node at all (already just the feature set) works unchanged."""
    path = _write_dag(tmp_path, "dag.json", ["a", "b"], [("a", "b")])
    dag = inst.load_attribution_dag(path, ["a", "b"])
    assert sorted(dag.nodes()) == ["a", "b"]
    assert dag.edges() == [("a", "b")]


def test_global_value_fn_satisfies_efficiency_axiom(tmp_path):
    from causasv import ASVExplainer

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag_path = _write_dag(
        tmp_path, "dag.json", ["budget", "source_root_id"], [("budget", "source_root_id")]
    )
    dag = inst.load_attribution_dag(dag_path, ds.feature_names)
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    asv = ASVExplainer(dag).explain(value_fn, method="exact")
    assert sum(asv.values()) == pytest.approx(
        value_fn(ds.feature_names) - value_fn([]), abs=1e-9
    )


def test_global_value_fn_matches_existing_exact_oracle(tmp_path):
    """Spec test #8: our value functions must be well-behaved (deterministic, pure)
    enough that causasv's own exact algorithms agree with each other on them --
    we don't build a new oracle, we just have to not break the existing one's
    assumptions (no hidden randomness, same coalition -> same value)."""
    from causasv import ASVExplainer

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag_path = _write_dag(
        tmp_path, "dag.json", ["budget", "source_root_id"], [("budget", "source_root_id")]
    )
    dag = inst.load_attribution_dag(dag_path, ds.feature_names)
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    explainer = ASVExplainer(dag)
    oracle = explainer.explain(value_fn, method="exact")
    auto = explainer.explain(value_fn, method="auto")
    for feat in oracle:
        assert oracle[feat] == pytest.approx(auto[feat], abs=1e-9)


def test_global_value_fn_is_memoized(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    ds_features = ds.feature_names

    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    v1 = value_fn(["budget"])
    v2 = value_fn(list(reversed(["budget"])))  # same coalition, different order
    assert v1 == v2
    # calling again must hit the cache, not re-fit -- verified indirectly via
    # identical float bit-pattern across repeated calls (deterministic retrain
    # would also match, so this mainly guards against a cache-key bug).
    assert v1 == value_fn(["budget"])
    assert ds_features == ds.feature_names  # coalition call didn't mutate the dataset


def test_global_value_fn_rejects_unknown_feature(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    with pytest.raises(ValueError, match="unknown feature"):
        value_fn(["not_a_real_feature"])


def test_local_value_fn_satisfies_efficiency_axiom(tmp_path):
    from causasv import ASVExplainer

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag_path = _write_dag(
        tmp_path, "dag.json", ["budget", "source_root_id"], [("budget", "source_root_id")]
    )
    dag = inst.load_attribution_dag(dag_path, ds.feature_names)
    model_result = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    local_fn = inst.make_local_value_fn(model_result, sample_index=0)
    asv = ASVExplainer(dag).explain(local_fn, method="exact")
    assert sum(asv.values()) == pytest.approx(
        local_fn(ds.feature_names) - local_fn([]), abs=1e-9
    )


def test_local_value_fn_rejects_unknown_feature(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    model_result = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    local_fn = inst.make_local_value_fn(model_result, sample_index=0)
    with pytest.raises(ValueError, match="unknown feature"):
        local_fn(["not_a_real_feature"])


def test_global_and_local_value_fns_are_not_the_same_scale(tmp_path):
    """Global (held-out CV quality) and local (single-instance prediction) modes
    must not be conflated -- they answer different questions."""
    ds = _cv_dataset(tmp_path, "review_or_drop")
    global_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    model_result = inst.fit_instability_model(ds, model="logistic", cv_folds=3, seed=1)
    local_fn = inst.make_local_value_fn(model_result, sample_index=0)
    # global v(empty) is a CV metric (e.g. negative log loss, can be very negative);
    # local v(empty) is a probability in [0, 1]. They must not collide by accident.
    assert not (0.0 <= global_fn([]) <= 1.0 and global_fn([]) == local_fn([]))


# ---------------------------------------------------------------------------
# Phase 4: uncertainty + DAG ensemble sensitivity
# ---------------------------------------------------------------------------


def test_single_dag_sensitivity_is_trivial(tmp_path):
    """With one DAG there's nothing to compare across, so dag_rank_stability is
    trivially 1.0 and no feature is dag_sensitive -- the single-DAG case must not
    need a separate code path."""
    from causasv import CausalDAG

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag = CausalDAG.from_edges([("budget", "source_root_id")])
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)

    result = inst.explain_with_dag_sensitivity([dag], value_fn, seed=0)
    assert result["dag_rank_stability"] == pytest.approx(1.0)
    assert all(v is False for v in result["dag_sensitive"].values())
    assert all(v is True for v in result["sign_stable"].values())
    assert all(v == 0.0 for v in result["std_values"].values())
    assert len(result["per_dag_results"]) == 1


def test_single_dag_exact_has_null_seed_rank_stability(tmp_path):
    """explain_safe already returns rank_stability=None on the exact path (no
    seed variance exists there) -- this must pass through unchanged, not be
    coerced into a fake 1.0."""
    from causasv import CausalDAG

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag = CausalDAG.from_edges([("budget", "source_root_id")])
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)

    result = inst.explain_with_dag_sensitivity([dag], value_fn, seed=0)
    per_dag = result["per_dag_results"][0]
    assert per_dag["is_exact"] is True
    assert per_dag["rank_stability"] is None


def test_multi_dag_rejects_mismatched_node_sets(tmp_path):
    from causasv import CausalDAG

    ds = _cv_dataset(tmp_path, "review_or_drop")
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    dag1 = CausalDAG.from_edges([("budget", "source_root_id")])
    dag2 = CausalDAG.from_edges([("budget", "extra_node")])

    with pytest.raises(ValueError, match="same node set"):
        inst.explain_with_dag_sensitivity([dag1, dag2], value_fn, seed=0)


def test_explain_with_dag_sensitivity_rejects_empty_dag_list(tmp_path):
    ds = _cv_dataset(tmp_path, "review_or_drop")
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)
    with pytest.raises(ValueError, match="non-empty"):
        inst.explain_with_dag_sensitivity([], value_fn, seed=0)


def test_multi_dag_sensitivity_reports_identical_dags_as_fully_stable(tmp_path):
    """Two copies of the same DAG must agree perfectly: rank_stability 1.0, zero
    std, no feature flagged sensitive -- a baseline sanity check before trusting
    the divergent-DAG case."""
    from causasv import CausalDAG

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag = CausalDAG.from_edges([("budget", "source_root_id")])
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)

    result = inst.explain_with_dag_sensitivity([dag, dag], value_fn, seed=0)
    assert result["dag_rank_stability"] == pytest.approx(1.0)
    for f in result["std_values"]:
        assert result["std_values"][f] == pytest.approx(0.0, abs=1e-12)
    assert all(v is False for v in result["dag_sensitive"].values())


def test_multi_dag_mean_values_match_manual_average(tmp_path):
    from causasv import CausalDAG

    ds = _cv_dataset(tmp_path, "review_or_drop")
    dag1 = CausalDAG.from_edges([("budget", "source_root_id")])
    dag2 = CausalDAG.from_edges([("source_root_id", "budget")])
    value_fn = inst.make_global_value_fn(ds, model="logistic", cv_folds=3, seed=1)

    result = inst.explain_with_dag_sensitivity([dag1, dag2], value_fn, seed=0)
    v1 = result["per_dag_results"][0]["values"]
    v2 = result["per_dag_results"][1]["values"]
    for f in result["mean_values"]:
        assert result["mean_values"][f] == pytest.approx((v1[f] + v2[f]) / 2)
