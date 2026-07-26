"""Phase 2: prediction model + grouped CV.

A simple, reproducible baseline (logistic/ridge) is the default -- "hgb" is
opt-in, not the default, so a first run is always the cheap, inspectable
model per AGENTS.md's "simple algorithms first". CV is grouped by
dataset.groups (default sample_id) so the same physical sample never
straddles train/test, matching quietset's own leakage concern for
source_root_id/opening_family (see calibrate --group-by in quietset's README).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bundle import InstabilityDataset

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
