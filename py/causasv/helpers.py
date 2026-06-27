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


class TabularExplainer:
    """High-level explainer for tabular models with a causal DAG.

    Wraps ``ASVExplainer`` with a sklearn-compatible model and a background
    dataset so users can call ``explain_instance()`` without constructing a
    value function by hand.

    Example::

        from causasv import TabularExplainer, CausalDAG

        dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])
        explainer = TabularExplainer.from_model(
            model=clf,
            dag=dag,
            background=X_train,
            feature_names=["education", "income", "risk_score"],
        )
        values = explainer.explain_instance(X_test[0], method="auto")
    """

    def __init__(self, explainer, model, background, feature_names, *, predict_fn=None):
        self._explainer = explainer
        self._model = model
        self._background = background
        self._feature_names = list(feature_names)
        self._predict_fn = predict_fn

    @classmethod
    def from_model(cls, model, dag, background, feature_names, *, predict_fn=None):
        """Construct from a sklearn-compatible model and a causal DAG.

        Args:
            model: Any object with ``predict_proba(X)`` or ``predict(X)``.
            dag: A ``CausalDAG`` instance.
            background: Reference dataset used for absent-feature baseline
                (column means). Array-like, shape (n_samples, n_features).
            feature_names: Ordered list of feature names.
            predict_fn: Optional callable ``(row: ndarray shape (1, n)) -> float``.
        """
        from .causasv import ASVExplainer

        return cls(
            ASVExplainer(dag), model, background, feature_names, predict_fn=predict_fn
        )

    def explain_instance(self, x, method="auto", **kwargs):
        """Compute ASV for a single instance ``x``.

        Args:
            x: The instance to explain — array-like, shape (n_features,).
            method: Passed to ``ASVExplainer.explain()`` (default ``"auto"``).
            **kwargs: Additional keyword arguments forwarded to ``explain()``.

        Returns:
            ``dict[str, float]`` mapping feature name to its ASV value.
        """
        value_fn = make_tabular_value_fn(
            self._model, x, self._background, self._feature_names,
            predict_fn=self._predict_fn,
        )
        return self._explainer.explain(value_fn, method=method, **kwargs)

    def explain_instance_with_diagnostics(self, x, method="auto", **kwargs):
        """Like ``explain_instance`` but returns the full diagnostics dict."""
        value_fn = make_tabular_value_fn(
            self._model, x, self._background, self._feature_names,
            predict_fn=self._predict_fn,
        )
        return self._explainer.explain_with_diagnostics(value_fn, method=method, **kwargs)
