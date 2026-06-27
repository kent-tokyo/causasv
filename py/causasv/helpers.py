"""Pure-Python helpers for common causasv patterns."""

from causasv import ASVExplainer


class ASVEnsembleExplainer:
    """Compute ASV over multiple user-supplied DAGs and summarize sensitivity.

    Runs one ASVExplainer per DAG and returns per-DAG values alongside
    mean, standard deviation, and rank stability (mean pairwise Kendall tau).

    Does NOT perform causal discovery — all DAGs must be supplied by the caller.

    Args:
        dags: List of CausalDAG objects. Must share the same node names.

    Example::

        from causasv import CausalDAG
        from causasv.helpers import ASVEnsembleExplainer

        dag1 = CausalDAG.from_edges([("A", "B"), ("B", "C")])
        dag2 = CausalDAG.from_edges([("A", "B"), ("A", "C")])
        ensemble = ASVEnsembleExplainer([dag1, dag2])
        result = ensemble.explain_with_sensitivity(value_fn, method="exact")
        # result["mean_values"], result["std_values"], result["rank_stability"]
    """

    def __init__(self, dags):
        self._explainers = [ASVExplainer(dag) for dag in dags]

    def explain_with_sensitivity(self, value_fn, **kwargs):
        """Compute ASV across all DAGs and return aggregated sensitivity metrics.

        Args:
            value_fn: Coalition value function ``(list[str]) -> float``.
            **kwargs: Forwarded to ``ASVExplainer.explain()`` (method, n_samples, seed, …).

        Returns:
            dict with keys:
            - ``"mean_values"``: dict[str, float] — mean ASV per feature
            - ``"std_values"``: dict[str, float] — std ASV per feature
            - ``"rank_stability"``: float — mean Kendall tau across all DAG pairs (1 = full agreement)
            - ``"per_dag_values"``: list[dict[str, float]] — one dict per DAG
        """
        per_dag = [e.explain(value_fn, **kwargs) for e in self._explainers]
        if not per_dag:
            return {
                "mean_values": {},
                "std_values": {},
                "rank_stability": 1.0,
                "per_dag_values": [],
            }
        features = sorted(per_dag[0].keys())
        k = len(per_dag)
        mean = {f: sum(d[f] for d in per_dag) / k for f in features}
        std = {
            f: (sum((d[f] - mean[f]) ** 2 for d in per_dag) / k) ** 0.5 for f in features
        }
        rank_stability = _mean_kendall_tau(per_dag, features)
        return {
            "mean_values": mean,
            "std_values": std,
            "rank_stability": rank_stability,
            "per_dag_values": per_dag,
        }


def _kendall_tau(a, b, features):
    """Compute Kendall tau between two dicts of feature values. O(n^2)."""
    n = len(features)
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = a[features[i]], a[features[j]]
            yi, yj = b[features[i]], b[features[j]]
            s = (xi - xj) * (yi - yj)
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = n * (n - 1) / 2
    return (concordant - discordant) / total if total > 0 else 1.0


def _mean_kendall_tau(per_dag, features):
    """Mean pairwise Kendall tau across all DAG pairs. Returns 1.0 for < 2 DAGs."""
    taus = [
        _kendall_tau(per_dag[i], per_dag[j], features)
        for i in range(len(per_dag))
        for j in range(i + 1, len(per_dag))
    ]
    return sum(taus) / len(taus) if taus else 1.0


def explain_stability(explainer, value_fn, seeds, **kwargs):
    """Run explain() with multiple seeds and return seed-stability metrics.

    Useful for verifying that approximate ASV rankings are consistent across
    different random seeds before trusting the attribution order.

    Args:
        explainer: ASVExplainer instance.
        value_fn: coalition value function ``(list[str]) -> float``.
        seeds: list of int seeds (at least 2 for meaningful stability).
        **kwargs: forwarded to ``explainer.explain()`` (method, n_samples, etc.).

    Returns:
        dict with keys:
        - ``"mean_values"``: dict[str, float] — mean ASV per feature
        - ``"std_values"``: dict[str, float] — std ASV per feature (0 = perfect stability)
        - ``"rank_stability"``: float — mean pairwise Kendall tau (1 = full rank agreement)
        - ``"per_seed_values"``: dict[int, dict[str, float]] — per-seed results

    Example::

        from causasv import CausalDAG, ASVExplainer
        from causasv.helpers import explain_stability

        dag = CausalDAG.from_edges([("education", "income"), ("income", "risk_score")])
        explainer = ASVExplainer(dag)
        result = explain_stability(
            explainer, value_fn, seeds=[1, 2, 3, 4, 5],
            method="approx", n_samples=10_000,
        )
        print(result["rank_stability"])   # 1.0 = perfectly stable rankings
        print(result["std_values"])       # small = stable estimates
    """
    if not seeds:
        raise ValueError("seeds must be non-empty")
    per_seed = {s: explainer.explain(value_fn, seed=s, **kwargs) for s in seeds}
    features = sorted(per_seed[seeds[0]].keys())
    k = len(seeds)
    mean = {f: sum(per_seed[s][f] for s in seeds) / k for f in features}
    std = {
        f: (sum((per_seed[s][f] - mean[f]) ** 2 for s in seeds) / k) ** 0.5
        for f in features
    }
    return {
        "mean_values": mean,
        "std_values": std,
        "rank_stability": _mean_kendall_tau(list(per_seed.values()), features),
        "per_seed_values": dict(zip(seeds, per_seed.values())),
    }


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
            self._model,
            x,
            self._background,
            self._feature_names,
            predict_fn=self._predict_fn,
        )
        return self._explainer.explain(value_fn, method=method, **kwargs)

    def explain_instance_with_diagnostics(self, x, method="auto", **kwargs):
        """Like ``explain_instance`` but returns the full diagnostics dict."""
        value_fn = make_tabular_value_fn(
            self._model,
            x,
            self._background,
            self._feature_names,
            predict_fn=self._predict_fn,
        )
        return self._explainer.explain_with_diagnostics(
            value_fn, method=method, **kwargs
        )
