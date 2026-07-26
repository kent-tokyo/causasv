"""Phase 3: DAG-aware attribution -- DAG loading + global/local value functions.

ASV nodes are the *raw* (pre-encoding) feature names, e.g. "evaluator_family" --
not its one-hot columns "evaluator_family=family_x". A DAG describing causal
ordering among semantic factors doesn't decompose per dummy category, and letting
it would blow up DAG size with high-cardinality categoricals. Both value function
builders below therefore accept coalitions of raw feature names and expand each
into its encoded column group internally.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

# causasv's own core (compiled extension + helpers.py) is always available wherever
# causasv.instability can be imported at all, and pulls in neither numpy nor
# scikit-learn itself -- so these are plain top-level imports, unlike numpy/sklearn
# below, which are optional and imported lazily at their point of use.
from causasv import CausalDAG
from causasv.helpers import make_tabular_value_fn

from .bundle import InstabilityDataset
from .modeling import InstabilityModel, _make_estimator, _safe_auc, _validate_cv_folds

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
