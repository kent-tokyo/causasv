"""Adapter: quietset label-instability data -> causasv ASV attribution.

Scope and honesty guardrails (see docs/integrations/quietset_label_instability.md):

- This module reads quietset's ``Observation``/``StabilityReport`` JSONL *by field
  name only*. It never imports quietset and never re-implements quietset's scoring
  (agreement, entropy, EM, etc.) -- if a value isn't already in the scored/observations
  JSONL, this module does not compute it.
- The output is an Asymmetric Shapley Value **attribution under a user-supplied DAG
  and a fitted prediction model** -- not a causal effect estimate. "evaluator_family
  has the highest ASV" is not "changing evaluator_family fixes instability by X points".
  Treat ASV rankings as hypotheses for a follow-up paired/controlled re-evaluation,
  not as a conclusion to act on directly.

This package is intentionally NOT imported by ``causasv/__init__.py`` (same pattern as
``causasv/plot.py``): importing plain ``causasv`` never pulls in numpy/scikit-learn.
Use ``from causasv import instability`` or ``from causasv.instability import ...``.

Internally split by responsibility (bundle/modeling/value/sensitivity/report) --
this file only re-exports the combined public surface so
``from causasv.instability import ...`` keeps working exactly as it did when this
was a single flat module.
"""

from .bundle import (
    SEED_LIKE_COLUMNS,
    STABILITY_REPORT_FIELDS,
    TARGET_DEFINITIONS,
    Cell,
    InstabilityDataset,
    TargetSpec,
    build_instability_dataset,
    load_bundle_manifest,
    wrap_single_cell,
)
from .modeling import (
    MODEL_TYPES,
    InstabilityModel,
    fit_instability_model,
)
from .report import (
    SCHEMA_VERSION,
    build_attribution_report,
    dump_attribution_json,
    summarize_attribution,
)
from .sensitivity import explain_with_dag_sensitivity
from .value import (
    DEFAULT_SINK_NODE,
    load_attribution_dag,
    make_global_value_fn,
    make_local_value_fn,
)
from .value import _dag_json as _dag_json  # re-exported for test_instability.py only

__all__ = [
    "STABILITY_REPORT_FIELDS",
    "SEED_LIKE_COLUMNS",
    "TARGET_DEFINITIONS",
    "TargetSpec",
    "Cell",
    "load_bundle_manifest",
    "wrap_single_cell",
    "InstabilityDataset",
    "build_instability_dataset",
    "MODEL_TYPES",
    "InstabilityModel",
    "fit_instability_model",
    "DEFAULT_SINK_NODE",
    "load_attribution_dag",
    "make_global_value_fn",
    "make_local_value_fn",
    "explain_with_dag_sensitivity",
    "SCHEMA_VERSION",
    "build_attribution_report",
    "summarize_attribution",
    "dump_attribution_json",
]
