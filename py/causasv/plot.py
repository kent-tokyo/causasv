"""Minimal ASV visualisation helpers.

Requires matplotlib (not installed by default)::

    pip install matplotlib

Usage::

    from causasv import plot

    values = {"education": 0.4, "income": 0.8, "risk_score": -0.2}
    plot.bar(values)
    plot.waterfall(values, base_value=0.5)
"""


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError(
            "causasv.plot requires matplotlib — install it with: pip install matplotlib"
        ) from exc


def bar(values, *, title="ASV Feature Attribution", ax=None, figsize=(8, 5)):
    """Horizontal bar chart of ASV values, sorted by absolute magnitude.

    Positive contributions are shown in blue, negative in red.

    Args:
        values: dict[str, float] — feature name → ASV value.
        title: Chart title.
        ax: Existing matplotlib Axes. If None, a new figure is created.
        figsize: Figure size when ax is None.

    Returns:
        matplotlib.axes.Axes
    """
    plt = _require_matplotlib()
    features = sorted(values.keys(), key=lambda k: abs(values[k]))
    vals = [values[f] for f in features]
    colors = ["#d62728" if v < 0 else "#1f77b4" for v in vals]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.barh(features, vals, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("ASV")
    plt.tight_layout()
    return ax


def waterfall(values, *, base_value=0.0, title="ASV Waterfall", ax=None, figsize=(9, 5)):
    """Waterfall chart showing cumulative ASV contributions.

    Starts at base_value (e.g. v(∅)), stacks each feature's contribution, and ends
    at base_value + Σφ_i.

    Args:
        values: dict[str, float] — feature name → ASV value.
        base_value: Starting value (e.g. model output for empty coalition).
        title: Chart title.
        ax: Existing matplotlib Axes. If None, a new figure is created.
        figsize: Figure size when ax is None.

    Returns:
        matplotlib.axes.Axes
    """
    plt = _require_matplotlib()
    # Sort by absolute magnitude descending so the biggest movers come first.
    features = sorted(values.keys(), key=lambda k: abs(values[k]), reverse=True)
    vals = [values[f] for f in features]

    labels = ["base"] + features + ["total"]
    bar_vals = [0.0] + vals + [0.0]
    bottoms = [base_value]
    running = base_value
    for v in vals:
        bottoms.append(running)
        running += v
    bottoms.append(0.0)  # total bar starts from 0

    colors = []
    for i, v in enumerate(bar_vals):
        if i == 0 or i == len(bar_vals) - 1:
            colors.append("#aaaaaa")  # base and total: grey
        elif v >= 0:
            colors.append("#1f77b4")  # positive: blue
        else:
            colors.append("#d62728")  # negative: red

    # Total bar: full height from 0
    heights = list(bar_vals)
    heights[-1] = running  # total = base + Σφ

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    ax.bar(labels, heights, bottom=bottoms, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(
        base_value,
        color="black",
        linewidth=0.6,
        linestyle="--",
        label=f"base = {base_value:.3f}",
    )
    ax.set_title(title)
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    plt.tight_layout()
    return ax
