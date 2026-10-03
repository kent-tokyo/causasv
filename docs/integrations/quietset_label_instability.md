# Attribute quietset label instability with ASV

`causasv.instability` attributes a model's ability to predict quietset label
instability to upstream factors under a caller-supplied DAG.

The result is a diagnostic ranking, not an intervention effect or verdict.
For example, a high ASV for `evaluator_family` means that the fitted model used
that feature strongly under the supplied DAG. It does not show that changing
the evaluator will reduce instability by the ASV amount.

This adapter reads quietset JSONL fields. It neither imports quietset nor
reimplements its scoring logic.

## Data layout

One quietset `StabilityReport` summarizes repeated observations of a sample.
To compare run conditions without losing the variation that produced the
instability score, keep three disjoint groups:

- `cell_features`: configuration varied between quietset runs, such as
  `budget`, `loss_recipe`, or `evaluator_family`
- `replicate_axes`: repeated-observation axes inside a run, such as
  `evaluator_id` or `seed`; these are not model features
- `sample_features`: intrinsic sample fields expected to stay constant for one
  `sample_id`, such as `source_root_id`

Describe multiple conditions with a `causasv-instability-bundle-v1` manifest:

```json
{
  "schema_version": "causasv-instability-bundle-v1",
  "cells": [
    {
      "config_id": "budget4_recipeA_familyX",
      "scored": "runs/cell01/scored.jsonl",
      "observations": "runs/cell01/observations.jsonl",
      "features": {
        "budget": 4,
        "loss_recipe": "recipe_a",
        "evaluator_family": "family_x"
      },
      "replicate_axes": ["evaluator_id", "seed", "shuffle_seed"]
    }
  ]
}
```

Paths are resolved relative to the manifest. Each `(sample_id, config_id)`
pair becomes one analysis row.

A single aggregate `scored.jsonl` can be read directly, but it cannot support
`cell_features`: aggregation has already removed the configuration-level
variation needed for that comparison.

## Validation

The adapter rejects:

- duplicate `config_id` values
- overlap among cell, replicate, and sample feature groups
- a file that mixes several real conditions under one manifest cell
- too few repeated observations for the selected target
- features constant across the combined dataset
- leakage from derived `StabilityReport` fields into model features
- numeric missing values unless `drop_rows` is requested
- a DAG whose feature nodes do not match the analysis table

Raw `seed` and `shuffle_seed` are blocked as features by default because their
numeric values have no causal or ordinal meaning. Use quietset's seed-sensitivity
metric as the target, or opt in explicitly and treat raw seeds as categories.

Categorical missing values become an explicit `<missing>` category. Numeric
missing values are never silently replaced by zero.

## Targets

| Target | Definition | Minimum observations |
| --- | --- | ---: |
| `label_entropy` | quietset `label_entropy` | 2 |
| `label_disagreement` | `1 - label_agreement` | 2 |
| `score_mad` | quietset `score_mad` | 2 |
| `score_iqr` | quietset `score_iqr` | 2 |
| `score_sign_disagreement` | `1 - score_sign_agreement` | 2 |
| `review_or_drop` | decision is Review or Drop | 1 |
| `lcb_risk` | `1 - label_agreement_lcb` | 2 |

## Model and attribution modes

The default model is logistic regression for a binary target and ridge
regression for a continuous target. `--model hgb` selects scikit-learn's
histogram gradient boosting.

Cross-validation is grouped by `sample_id` by default, so the same physical
sample does not appear in both train and test folds. Use a content-group field
such as `source_root_id` when that is the correct independence unit.

`--min-cv-metric` has no default. If a requested quality floor is missed, all
features are reported as `insufficient_evidence` rather than ranked as though
the model were adequate.

Two attribution modes are available:

- `global`: refits the model for each coalition and attributes held-out model
  quality
- `local`: attributes one sample's prediction using a chosen baseline for
  absent features

Global and local ASV values use different value functions and scales. Do not
compare them directly.

## DAG contract

The DAG uses `CausalDAG.to_json()` / `from_json()` format. Its input nodes must
equal `cell_features + sample_features`.

An optional `instability_prediction` node may appear as an output-only sink.
The adapter removes it before attribution. It is rejected if it has outgoing
edges.

```json
{
  "nodes": [
    "difficulty_proxy",
    "evaluator_family",
    "loss_recipe",
    "budget",
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

Repeat `--dag` to compare several candidate DAGs. They must share the same node
set.

## Reading the report

The `causasv-instability-attribution-v1` report includes:

- ASV, standard error, confidence interval, and rank per feature
- fitted model type, grouped-CV metric, and optional quality gate
- selected ASV method, exactness, ESS, and seed stability
- cross-DAG rank and sign sensitivity when several DAGs were supplied
- warnings and a display bucket for each feature

The display buckets are applied in this order:

1. `insufficient_evidence`: the predictive model missed its requested gate
2. `dag_sensitive`: sign or rank depends materially on the candidate DAG
3. `uncertain`: the confidence interval includes zero
4. `robustly_attributed`: none of the checks above fired

`robustly_attributed` means only that this model, data, and DAG passed the
configured diagnostics. It is not a causal conclusion.

Use a high-ranked feature to design a controlled follow-up. Change one factor,
hold the others fixed, and rerun the evaluation. Do not change quietset weights
or keep/review/drop thresholds from an ASV ranking alone.

## Command-line example

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

Run the script with `--help` for grouped-CV, missing-value, local-sample, and
multi-DAG options.

## Non-goals

This workflow does not implement causal discovery, automatic DAG generation,
intervention-effect estimation, automatic quietset decisions, or a dependency
between the two projects' cores.
