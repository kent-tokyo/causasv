"""Pure-Python helpers for common causasv patterns."""

from causasv import ASVExplainer


def explain_quality(
    explainer,
    value_fn=None,
    *,
    value_fn_batch=None,
    seed=None,
    ci=0.95,
    max_samples=100_000,
    min_samples=1_000,
    batch_size=1_000,
    rel_tol=0.01,
):
    """One-stop ASV computation with uncertainty estimates.

    Tries exact methods first, falls back to uniform sparse adaptive sampling
    (ESS = n_samples; no IS weight variance). Always returns stderr and ci bounds.

    Args:
        explainer: ASVExplainer instance.
        value_fn: ``(list[str]) -> float``. Mutually exclusive with value_fn_batch.
        value_fn_batch: ``(list[list[str]]) -> list[float]``. Preferred for large models.
        seed: RNG seed for reproducibility.
        ci: Confidence level for CI bounds, e.g. 0.95 for 95% CI. Set None to skip.
        max_samples: Maximum number of samples for approximate paths.
        min_samples: Minimum samples before convergence is checked.
        batch_size: Samples per convergence-check batch.
        rel_tol: Relative change threshold for convergence.

    Returns:
        dict with keys: values, stderr, ess, ess_ratio, is_exact, selected_method,
        converged, n_samples, seed, fallback_from, fallback_reason,
        and optionally ci, ci_low, ci_high.

    Example::

        from causasv import CausalDAG, ASVExplainer
        from causasv.helpers import explain_quality

        dag = CausalDAG.from_edges([("A", "B"), ("B", "C")])
        result = explain_quality(
            ASVExplainer(dag),
            value_fn=lambda features: model.score(features),
            ci=0.95,
        )
        print(result["values"])     # dict[str, float]
        print(result["ci_low"])     # dict[str, float]
        print(result["ci_high"])    # dict[str, float]
        print(result["selected_method"])  # e.g. "exact_dag_sparse"
    """
    if value_fn is not None and value_fn_batch is not None:
        raise ValueError("Pass either value_fn or value_fn_batch, not both.")
    if value_fn is None and value_fn_batch is None:
        raise ValueError("One of value_fn or value_fn_batch must be provided.")

    if value_fn_batch is not None:
        # Batched path: uniform sparse adaptive (ESS = n_samples, no IS variance).
        kwargs = dict(
            min_samples=min_samples,
            max_samples=max_samples,
            batch_size=batch_size,
            rel_tol=rel_tol,
            seed=seed,
        )
        if ci is not None:
            kwargs["ci"] = ci
        return explainer.explain_quality_batch(value_fn_batch, **kwargs)

    # Single value_fn path: use auto_quality (exact-first, uniform sparse adaptive fallback).
    kwargs = dict(
        min_samples=min_samples,
        max_samples=max_samples,
        batch_size=batch_size,
        rel_tol=rel_tol,
        seed=seed,
    )
    if ci is not None:
        kwargs["ci"] = ci
    return explainer.explain_quality(value_fn, **kwargs)


def explain_safe(
    explainer,
    value_fn=None,
    *,
    value_fn_batch=None,
    seed=None,
    ci=0.95,
    ess_ratio_min=0.1,
    rank_stability_min=0.9,
    stability_seeds=5,
    **kwargs,
):
    """``explain_quality()`` plus automatic trust diagnostics.

    Runs the same exact-first / quality-guaranteed-approximate pipeline as
    ``explain_quality()``, then applies the checklist from the README
    ("Approximation diagnostics checklist") automatically instead of leaving
    it to the caller: flags low ESS ratio, low rank stability across seeds,
    and features whose confidence interval still straddles zero.

    Args:
        explainer: ASVExplainer instance.
        value_fn: ``(list[str]) -> float``. Mutually exclusive with value_fn_batch.
        value_fn_batch: ``(list[list[str]]) -> list[float]``.
        seed: RNG seed for reproducibility (also seeds the stability check).
        ci: Confidence level for CI bounds, e.g. 0.95 for 95% CI.
        ess_ratio_min: Warn if ess_ratio falls below this (default matches
            the README checklist: 0.1).
        rank_stability_min: Warn if rank_stability falls below this (default
            matches the README checklist: 0.9).
        stability_seeds: Number of seeds used for the rank-stability check.
        **kwargs: Forwarded to ``explain_quality()`` / ``explain_stability()``.

    Returns:
        Everything ``explain_quality()`` returns, plus:
        - ``"warnings"``: list[str] — human-readable diagnostic warnings, empty if none.
        - ``"rank_stability"``: float | None — None when the result is exact,
          or when only value_fn_batch was given (explain_stability is value_fn-only).
        - ``"unstable_features"``: list[str] — features whose CI includes 0
          (sign is not distinguishable from "no effect").

    Note:
        Rank-stability checking requires ``value_fn`` (not ``value_fn_batch``),
        since ``explain_stability()`` re-runs the explainer per seed with a
        single-coalition value function.
    """
    result = explain_quality(
        explainer,
        value_fn=value_fn,
        value_fn_batch=value_fn_batch,
        seed=seed,
        ci=ci,
        **kwargs,
    )

    warnings = []
    rank_stability = None
    if not result["is_exact"]:
        ess_ratio = result.get("ess_ratio")
        if ess_ratio is not None and ess_ratio < ess_ratio_min:
            warnings.append(
                f"ess_ratio {ess_ratio:.3f} is below {ess_ratio_min} — "
                "estimate may have high variance; consider more samples."
            )
        if value_fn is not None:
            # method="approx" (not explain()'s default "auto"): we already know the
            # exact path fails for this DAG (is_exact is False above), so re-attempting
            # it once per stability seed would just redundantly re-pay that cost.
            # explain_quality()'s adaptive-style kwargs (min_samples/max_samples/etc.)
            # aren't accepted by explain(), so they aren't forwarded here either.
            base_seed = seed or 0
            stability = explain_stability(
                explainer,
                value_fn,
                seeds=list(range(base_seed, base_seed + stability_seeds)),
                method="approx",
            )
            rank_stability = stability["rank_stability"]
            if rank_stability < rank_stability_min:
                warnings.append(
                    f"rank_stability {rank_stability:.3f} is below {rank_stability_min} — "
                    "feature rankings vary across seeds; consider more samples."
                )

    unstable_features = []
    if "ci_low" in result:
        unstable_features = [
            f
            for f in result["values"]
            if result["ci_low"][f] <= 0.0 <= result["ci_high"][f]
        ]

    return {
        **result,
        "warnings": warnings,
        "rank_stability": rank_stability,
        "unstable_features": unstable_features,
    }


def _normal_quantile(p):
    """Approximate Φ⁻¹(p) without scipy — accurate to ~0.01 for p in (0.9, 0.999)."""
    import math
    # Beasley-Springer-Moro approximation
    a = [2.515517, 0.802853, 0.010328]
    b = [1.432788, 0.189269, 0.001308]
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    num = a[0] + a[1] * t + a[2] * t * t
    den = 1.0 + b[0] * t + b[1] * t * t + b[2] * t * t * t
    return t - num / den


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
            - ``"rank_stability"``: float — mean Kendall tau across all DAG pairs
              (1 = full agreement)
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


def make_tabular_value_fn(model, x, background, feature_names, *, predict_fn=None, baseline="mean"):
    """Wrap a sklearn-compatible model as a causasv value function.

    Absent features are replaced according to ``baseline``.

    Args:
        model: Any object with ``predict_proba(X)`` or ``predict(X)``.
        x: The instance to explain — array-like, shape (n_features,).
        background: Reference dataset — array-like, shape (n_samples, n_features).
            Used to fill in absent features per ``baseline``.
        feature_names: Ordered list of feature names matching the columns of x/background.
        predict_fn: Optional callable ``(row: np.ndarray shape (1, n)) -> float``.
            Defaults to ``model.predict_proba(row)[0, 1]`` for classifiers or
            ``model.predict(row)[0]`` for regressors.
        baseline: How to fill in absent features. One of:

            - ``"mean"`` (default): column means of ``background``.
            - ``"median"``: column medians of ``background``.
            - ``"sample"``: a single row from ``background``, chosen once
              (seeded, reproducible) rather than a synthetic averaged row —
              useful when the mean row is unrealistic for correlated features.
            - ``"background_expectation"``: true marginal expectation —
              substitutes present features into *every* row of ``background``
              and averages the model's prediction over all of them, instead of
              a single summary row. More accurate for correlated features, at
              the cost of ``len(background)`` model calls per coalition
              instead of 1.
            - a callable ``(background: np.ndarray) -> np.ndarray`` returning
              a custom baseline row, e.g. a trimmed mean.

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
    background_arr = np.asarray(background, dtype=float)
    x = np.asarray(x, dtype=float)

    if predict_fn is None:
        if hasattr(model, "predict_proba"):

            def predict_fn(r):
                return float(model.predict_proba(r)[0, 1])
        else:

            def predict_fn(r):
                return float(model.predict(r)[0])

    if baseline == "background_expectation":

        def value_fn(coalition: list) -> float:
            idxs = [name_to_idx[name] for name in coalition]
            rows = background_arr.copy()
            if idxs:
                rows[:, idxs] = x[idxs]
            return float(np.mean([predict_fn(row.reshape(1, -1)) for row in rows]))

        return value_fn

    if callable(baseline):
        baseline_row = baseline(background_arr)
    elif baseline == "mean":
        baseline_row = background_arr.mean(axis=0)
    elif baseline == "median":
        baseline_row = np.median(background_arr, axis=0)
    elif baseline == "sample":
        baseline_row = background_arr[np.random.default_rng(0).integers(len(background_arr))]
    else:
        raise ValueError(f"Unknown baseline: {baseline!r}")

    def value_fn(coalition: list) -> float:
        row = baseline_row.copy()
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

    def __init__(
        self, explainer, model, background, feature_names, *, predict_fn=None, baseline="mean"
    ):
        self._explainer = explainer
        self._model = model
        self._background = background
        self._feature_names = list(feature_names)
        self._predict_fn = predict_fn
        self._baseline = baseline

    @classmethod
    def from_model(cls, model, dag, background, feature_names, *, predict_fn=None, baseline="mean"):
        """Construct from a sklearn-compatible model and a causal DAG.

        Args:
            model: Any object with ``predict_proba(X)`` or ``predict(X)``.
            dag: A ``CausalDAG`` instance.
            background: Reference dataset used for absent-feature baseline.
                Array-like, shape (n_samples, n_features).
            feature_names: Ordered list of feature names.
            predict_fn: Optional callable ``(row: ndarray shape (1, n)) -> float``.
            baseline: How to fill in absent features — see
                ``make_tabular_value_fn`` for the available modes
                (``"mean"``, ``"median"``, ``"sample"``,
                ``"background_expectation"``, or a custom callable).
        """
        from .causasv import ASVExplainer

        return cls(
            ASVExplainer(dag),
            model,
            background,
            feature_names,
            predict_fn=predict_fn,
            baseline=baseline,
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
            baseline=self._baseline,
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
            baseline=self._baseline,
        )
        return self._explainer.explain_with_diagnostics(
            value_fn, method=method, **kwargs
        )
