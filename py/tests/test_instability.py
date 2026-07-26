"""Tests for causasv.instability (Phase 1: bundle manifest adapter).

Covers spec test items #2 (row-order invariance), #3 (leakage rejection),
#5 (no silent 0-fill for missing values), #6 (binary/continuous targets), and
the manifest/cell validation rules (disjoint column roles, seed guard, mixed-file
detection, constant-feature policy, duplicate handling). Grouped-CV non-leakage
(#4), exact-oracle match (#8), approx CI/ESS (#9), rank/DAG stability (#10, #11),
uncertain-feature display (#12), low-performance warning (#13), malformed-DAG
rejection (#14), and the example smoke test (#15) land in later phases once the
model/DAG/CLI code they exercise exists.
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
