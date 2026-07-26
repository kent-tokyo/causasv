# quietset label instability → causasv ASV attribution

`causasv.instability` (`py/causasv/instability.py`) attributes **why** quietset
flagged certain samples as label-unstable to upstream factors the user cares
about (evaluator family, search budget, model checkpoint, loss recipe, ...),
under a user-supplied causal DAG.

## What this is, and what it is not

quietset measures **which** samples are unstable: `label_agreement`,
`label_entropy`, `score_mad`, `decision` (Keep/Review/Drop), etc., computed
from repeated observations of the same sample.

causasv's ASV attribution answers a different, narrower question: **given a
DAG you supplied and a model fit on your data, how much of the model's
ability to predict that instability is attributable to each upstream
factor?**

This is **not**:

- **Causal discovery.** The DAG is yours; this workflow never infers or
  suggests graph structure.
- **An intervention-effect estimate.** "`evaluator_family` has the highest
  ASV" does **not** mean "changing `evaluator_family` will reduce instability
  by that many points." It means: under this DAG and this fitted model,
  `evaluator_family` explains the most of the model's predictive signal for
  the instability target. A factor with near-zero true effect can still rank
  highest if it happens to correlate with the target in this particular
  sample — ASV does not distinguish correlation from a causal mechanism; only
  the DAG assumption does that, and the DAG assumption is yours to defend, not
  something this workflow verifies.
- A verdict. Use the ranking as a **hypothesis for a follow-up controlled
  re-evaluation** (see "Turning ASV results into experiments" below) — never
  as the sole justification for changing quietset's weights or thresholds.
- A modification to quietset. This adapter only reads quietset's
  `Observation`/`StabilityReport` JSONL by field name; quietset's source is
  never imported, and its scoring logic is never re-implemented here.

## Why not just join the two JSONL files naively?

`StabilityReport` is computed **from** a sample's repeated observations
(varying evaluator, budget, seed, ...). If you collapse those observations
into one row per sample (e.g. via mean/mode), the columns you wanted to
attribute to often become **constant** — there's nothing left to attribute.
If you instead hold every condition fixed to get per-condition rows, each
sample is left with a single observation, and quietset can no longer compute
instability at all.

This adapter resolves that by requiring you to separate:

- **cell_features** — the config axis you want to compare (e.g. `budget`,
  `loss_recipe`, `evaluator_family`). Constant *within* one run of quietset,
  varying *across* runs.
- **replicate_axes** — the axis quietset varied *within* one run to actually
  produce the instability signal (e.g. `evaluator_id`, `seed`,
  `shuffle_seed`). Never turned into a model feature.
- **sample_features** — intrinsic properties of the sample itself, read from
  `observations.jsonl` (e.g. `source_root_id`, `difficulty_proxy`). Expected
  constant for a given `sample_id` across all its observations.

These three sets are disjoint; overlapping any two is a hard error.

## Input: the bundle manifest

Run quietset once per condition you want to compare, then describe the
result as a `causasv-instability-bundle-v1` manifest:

```json
{
  "schema_version": "causasv-instability-bundle-v1",
  "cells": [
    {
      "config_id": "budget4_recipeA_familyX",
      "scored": "runs/cell01/scored.jsonl",
      "observations": "runs/cell01/observations.jsonl",
      "features": {"budget": 4, "loss_recipe": "recipe_a", "evaluator_family": "family_x"},
      "replicate_axes": ["evaluator_id", "seed", "shuffle_seed"]
    }
  ]
}
```

`scored`/`observations` paths are resolved relative to the manifest file.
Each `(sample_id, config_id)` pair becomes one analysis row: the target comes
from that cell's `scored.jsonl`; `cell_features` are stamped from the
manifest; `sample_features` are read (and checked for internal consistency)
from that cell's `observations.jsonl`.

**Single aggregate file (no config comparison):** if you only have one plain
`quietset score` run, pass `--observations`/`--scored` directly instead of a
bundle. `cell_features` must be empty in that mode — a single aggregate
`StabilityReport` has no config-level granularity to compare, and asking for
it produces the explicit error:

```
aggregate scored report has lost config-level variation; provide per-cell
scored files or config-scoped sample IDs
```

### Validation the adapter performs

- `config_id` uniqueness, and `cell_features`/`sample_features`/
  `replicate_axes` mutual disjointness (including the manifest's *actual*
  per-cell `features` keys, not just the caller's declared lists).
- **Mixed-file detection**: if `observations.jsonl` itself carries a field
  matching a declared cell feature and that field *varies* within the cell,
  the run is rejected rather than silently trusting the manifest's stamped
  value — that would mean the file mixes multiple real conditions under one
  `config_id`, and there's no safe way to guess which rows belong to which
  condition without re-implementing quietset's own scoring.
- Minimum observation counts per target (entropy/agreement/dispersion targets
  need ≥2 observations — a single observation makes them undefined by
  construction, not just noisy).
- Global constant-feature check, evaluated on the **combined** table across
  all cells (a feature being constant within one cell is expected and fine).
- `seed`/`shuffle_seed` are blocked from `cell_features`/`sample_features` by
  default — raw seed numbers carry no causal or ordinal meaning. Study seed
  sensitivity via quietset's own `seed_sensitivity`/`shuffle_seed_sensitivity`
  as the **target**, not as a feature. `--allow-raw-seed-feature` opts into
  treating them as categorical anyway, with a warning that results won't
  generalize to unseen seeds.
- **Target leakage guard**: every `StabilityReport` field (except
  `sample_id`/`n_observations`) is blocked from `cell_features`/
  `sample_features` — they're all derived from the same label/score
  distribution the target itself comes from.
- No silent 0-fill for missing values: numeric gaps either hard-fail (default)
  or drop the affected rows (`--missing-numeric drop_rows`), and categorical
  gaps become an explicit `<missing>` category — never a manufactured zero.

## Targets

| `--target` | Definition | Type | Min. observations |
|---|---|---|---|
| `label_entropy` | quietset's `label_entropy` | continuous | 2 |
| `label_disagreement` | `1 - label_agreement` | continuous | 2 |
| `score_mad` | quietset's `score_mad` | continuous | 2 |
| `score_iqr` | quietset's `score_iqr` | continuous | 2 |
| `score_sign_disagreement` | `1 - score_sign_agreement` | continuous | 2 |
| `review_or_drop` | `decision in {review, drop}` | binary | 1 |
| `lcb_risk` | `1 - label_agreement_lcb` | continuous | 2 |

## Model and grouped cross-validation

`fit_instability_model` fits `logistic` (binary) / `ridge` (continuous) by
default — a simple, reproducible baseline, not the most accurate model
available. `--model hgb` opts into `HistGradientBoosting{Classifier,Regressor}`.

Cross-validation is **grouped** (`GroupKFold`/`StratifiedGroupKFold`) by
`--group-by` (default `sample_id`) so the same physical sample never crosses
train/test — including when it appears under multiple config cells. Pass a
`sample_features` column (e.g. `source_root_id`) to group by quietset's own
correlated-content unit instead, mirroring `quietset calibrate --group-by`.

`--min-cv-metric` has **no built-in default** — there is no universally
correct performance floor. Omit it and no gate is applied; set it and a model
that misses it produces a report where every feature lands in the
`insufficient_evidence` summary bucket rather than a misleadingly confident
ranking.

## Attribution modes

- **`global`** — ASV over held-out predictive quality (negative log loss /
  AUC for binary targets, negative RMSE for continuous), recomputing the
  model on each coalition's features via the same grouped CV. Answers: *which
  upstream factors carry the model's overall predictive signal for this
  instability target?*
- **`local`** — ASV over a single sample's prediction, with absent features
  replaced per a baseline (reuses `causasv.helpers.make_tabular_value_fn`
  directly). Answers: *for this one sample, which factors drove its predicted
  instability?*

**Global and local values are on different scales and must never be compared
directly** — the output schema keeps `mode` explicit for this reason.

## DAG format and the `instability_prediction` sink

`--dag` files use causasv's own `CausalDAG.to_json()`/`from_json()` format
(`{"nodes": [...], "edges": [{"from": ..., "to": ...}]}`), so you can build one
programmatically with the Python API and pass it straight through. The DAG's
node set must exactly equal the dataset's feature set (`cell_features` ∪
`sample_features`) — every declared feature needs a node, and every node needs
to be a real feature.

The one exception: a designated sink node (default `instability_prediction`)
representing the model's output, not an attributable input. If present with
no outgoing edges, it and its incoming edges are stripped before building the
DAG — it needs no ASV value of its own. If it has outgoing edges, that's
rejected as a contradiction (a sink can't also be an intermediate node).

Example DAG (`instability_dag.json`), matching a bundle whose
`--cell-features` is `evaluator_family,budget,loss_recipe` and
`--sample-features` is `difficulty_proxy` (the DAG's node set — sink
excluded — must equal this exact feature set, or `load_attribution_dag`
rejects it naming the difference):

```json
{
  "nodes": [
    "difficulty_proxy", "evaluator_family", "loss_recipe", "budget",
    "instability_prediction"
  ],
  "edges": [
    {"from": "difficulty_proxy", "to": "instability_prediction"},
    {"from": "evaluator_family", "to": "instability_prediction"},
    {"from": "loss_recipe", "to": "instability_prediction"},
    {"from": "budget", "to": "instability_prediction"}
  ]
}
```

Multiple candidate DAGs are supported (repeat `--dag`); results are then
checked for cross-DAG sensitivity (see below).

## Uncertainty and DAG sensitivity

Every ASV comes with the full diagnostics `causasv.helpers.explain_safe`
already provides: `stderr`, `ci_low`/`ci_high`, `selected_method`, `is_exact`,
`ess`/`ess_ratio`, and seed-based `rank_stability` (kept as `null` when
`is_exact` is true — there is no seed variance on the exact path).

With multiple `--dag` flags, `dag_rank_stability` (mean pairwise Kendall τ
across DAGs) and per-feature `dag_sensitive`/`sign_stable` flags are also
reported. All supplied DAGs must share the same node set — comparing "the
ASV of `evaluator_family`" across DAGs only means something if every DAG
actually has that node.

## Output schema (`causasv-instability-attribution-v1`)

```json
{
  "schema_version": "causasv-instability-attribution-v1",
  "target": "label_entropy",
  "mode": "global",
  "features": [
    {
      "name": "evaluator_family",
      "asv": 0.31,
      "stderr": 0.04,
      "ci_low": 0.23,
      "ci_high": 0.39,
      "rank": 1,
      "sign_stable": true,
      "dag_sensitive": false
    }
  ],
  "model": {
    "type": "ridge",
    "group_column": "source_root_id",
    "cv_metric_name": "neg_rmse",
    "cv_metric": 0.42,
    "min_cv_metric": null,
    "meets_min_cv_metric": null
  },
  "asv_diagnostics": {
    "method": "exact_dag_sparse",
    "is_exact": true,
    "ess_ratio": 1.0,
    "seed_rank_stability": null,
    "dag_rank_stability": null
  },
  "warnings": []
}
```

`dag_rank_stability` and `seed_rank_stability` are `null`, not a fabricated
`1.0`, whenever the corresponding comparison wasn't actually performed
(single DAG; exact path). `meets_min_cv_metric` is `null` when no
`--min-cv-metric` was requested — silence means "no gate", never "passed".

`summarize_attribution()` splits features into four **display** buckets, in
priority order — none of them are a causal claim:

1. **`insufficient_evidence`** — the model missed its own requested quality
   floor; every feature's attribution is suspect regardless of its own CI or
   DAG-sensitivity.
2. **`dag_sensitive`** — sign or rank changed enough across candidate DAGs
   that the DAG assumption, not the data, is likely driving the result.
3. **`uncertain`** — CI straddles zero; not distinguishable from no effect.
4. **`robustly_attributed`** — none of the above fired. This means "this DAG
   and model didn't flag it" — not "confirmed to matter".

## Turning ASV results into experiments

A high-ranked factor is a hypothesis, not a conclusion. Examples of the
follow-up this workflow is meant to motivate:

- `budget` ranks highest → a paired re-evaluation that changes only `budget`,
  holding everything else fixed.
- `evaluator_family` ranks highest → a re-evaluation restricted to one family
  at a time.
- `shuffle_seed` shows up via its target-side signal
  (`shuffle_seed_sensitivity`) → an order-only ablation with `seed` fixed.
- `loss_recipe` ranks highest → a matched-size recipe ablation.

Do **not** change quietset's scoring weights or accept/reject thresholds on
the basis of an ASV ranking alone.

## Example

```bash
python examples/quietset_label_instability.py \
  --bundle data/instability_bundle.json \
  --target label_entropy \
  --cell-features evaluator_family,budget,loss_recipe \
  --sample-features difficulty_proxy \
  --dag data/instability_dag.json \
  --mode global \
  --model ridge \
  --seed 42 \
  --output results/instability_attribution.json
```

Multiple DAGs: repeat `--dag data/dag_a.json --dag data/dag_b.json`.

Run `python examples/quietset_label_instability.py --help` for the full flag
list (grouped CV, missing-value policy, constant-feature policy, local-mode
sample selection, etc.).

## Non-goals (this workflow does not implement)

- Causal discovery, or any automatic DAG construction.
- DoWhy-style intervention-effect estimation.
- A dependency from quietset core to causasv, or vice versa.
- A hardcoded, quietset-specific default DAG.
- Automatic drop/keep decisions driven by ASV rank.
- Causal conclusions drawn from ASV results alone.
