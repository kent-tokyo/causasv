"""Pure-Python helpers for common causasv patterns."""


def make_tabular_value_fn(model, x, background, feature_names, *, predict_fn=None):
    """Wrap a sklearn-compatible model as a causasv value function.

    Absent features are replaced by their column mean from ``background``.

    Args:
        model: Any object with ``predict_proba(X)`` or ``predict(X)``.
        x: The instance to explain — array-like, shape (n_features,).
        background: Reference dataset — array-like, shape (n_samples, n_features).
            Column means serve as the "absent feature" baseline.
        feature_names: Ordered list of feature names matching the columns of x/background.
        predict_fn: Optional callable ``(row: np.ndarray shape (1, n)) -> float``.
            Defaults to ``model.predict_proba(row)[0, 1]`` for classifiers or
            ``model.predict(row)[0]`` for regressors.

    Returns:
        A value function ``(coalition: list[str]) -> float`` suitable for
        ``ASVExplainer.explain()`` and ``explain_with_diagnostics()``.

    Example::

        from causasv import CausalDAG, ASVExplainer
        from causasv.helpers import make_tabular_value_fn

        dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])
        value_fn = make_tabular_value_fn(model, X_test[0], X_train, feature_names)
        values = ASVExplainer(dag).explain(value_fn, method="auto")
    """
    import numpy as np  # lazy: only required when this function is called

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    baseline = np.asarray(background, dtype=float).mean(axis=0)
    x = np.asarray(x, dtype=float)

    if predict_fn is None:
        if hasattr(model, "predict_proba"):
            predict_fn = lambda r: float(model.predict_proba(r)[0, 1])
        else:
            predict_fn = lambda r: float(model.predict(r)[0])

    def value_fn(coalition: list) -> float:
        row = baseline.copy()
        for name in coalition:
            row[name_to_idx[name]] = x[name_to_idx[name]]
        return predict_fn(row.reshape(1, -1))

    return value_fn
