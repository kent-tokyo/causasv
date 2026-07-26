"""quietset_label_instability.py -- attribute quietset label instability to
upstream factors (evaluator family, budget, model, loss recipe, ...) via
causasv's Asymmetric Shapley Values over a user-supplied DAG.

IMPORTANT -- what this is NOT:
    This is NOT causal discovery and NOT an intervention-effect estimate.
    The output is an ASV attribution *under the DAG and prediction model you
    supplied*. "evaluator_family has the highest ASV" does not mean "changing
    evaluator_family will fix instability by that many points" -- it means
    "given this causal ordering and this model, evaluator_family explains the
    most of the model's ability to predict instability". Treat high-ASV
    factors as hypotheses for a follow-up controlled re-evaluation (e.g. a
    paired run that only changes budget), not as a conclusion to act on
    directly. See docs/integrations/quietset_label_instability.md.

quietset itself is never imported or run as a subprocess -- this script only
reads the Observation/StabilityReport JSONL files quietset already produced.

Usage (bundle manifest -- the recommended path when comparing multiple configs):
    python examples/quietset_label_instability.py \\
        --bundle data/instability_bundle.json \\
        --target review_or_drop \\
        --cell-features evaluator_family,budget,loss_recipe \\
        --sample-features source_root_id \\
        --dag data/instability_dag.json \\
        --mode global \\
        --model logistic \\
        --seed 42 \\
        --output results/instability_attribution.json

Multiple candidate DAGs (DAG-sensitivity reporting):
    ... --dag data/dag_a.json --dag data/dag_b.json

Single aggregate scored/observations pair (no per-config comparison possible --
cell_features must be empty in this mode; see wrap_single_cell's docstring):
    python examples/quietset_label_instability.py \\
        --observations data/observations.jsonl --scored data/scored.jsonl \\
        --sample-features source_root_id --target label_entropy --dag data/dag.json \\
        --output results/instability_attribution.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from causasv import instability as inst


def _csv_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Attribute quietset label-instability targets to upstream factors "
            "via causasv ASV over a user-supplied DAG. Not causal discovery; "
            "not an intervention-effect estimate -- see the module docstring."
        )
    )
    inputs = p.add_argument_group("input (choose one)")
    inputs.add_argument("--bundle", help="causasv-instability-bundle-v1 manifest JSON")
    inputs.add_argument("--observations", help="single quietset Observation JSONL")
    inputs.add_argument("--scored", help="single quietset StabilityReport JSONL")

    p.add_argument(
        "--target",
        required=True,
        choices=sorted(inst.TARGET_DEFINITIONS),
        help="instability target",
    )
    p.add_argument(
        "--cell-features",
        default="",
        help="comma-separated config-axis columns (from the bundle manifest's per-cell "
        "'features'); must be empty with --observations/--scored (see module docstring)",
    )
    p.add_argument(
        "--sample-features",
        default="",
        help="comma-separated sample-intrinsic columns read from observations.jsonl "
        "(e.g. source_root_id, difficulty_proxy)",
    )
    p.add_argument(
        "--group-by", default="sample_id", help="grouped-CV group column (default: sample_id)"
    )
    p.add_argument("--min-observations", type=int, default=None)
    p.add_argument("--missing-numeric", choices=["error", "drop_rows"], default="error")
    p.add_argument("--drop-constant-features", action="store_true")
    p.add_argument("--allow-raw-seed-feature", action="store_true")

    p.add_argument(
        "--dag",
        action="append",
        dest="dags",
        required=True,
        help="DAG JSON (repeatable for DAG-sensitivity)",
    )
    p.add_argument("--sink-node", default=inst.DEFAULT_SINK_NODE)

    p.add_argument("--mode", choices=["global", "local"], default="global")
    p.add_argument("--sample-id", help="row to explain in --mode local (first matching row)")
    p.add_argument("--sample-index", type=int, help="row index to explain in --mode local")
    p.add_argument("--baseline", default="mean", help="absent-feature baseline for --mode local")

    p.add_argument(
        "--model",
        choices=list(inst.MODEL_TYPES),
        default=None,
        help="default: logistic for binary targets, ridge for continuous",
    )
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--min-cv-metric", type=float, default=None)
    p.add_argument("--metric", default=None, help="global-mode scoring metric override")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ci", type=float, default=0.95)
    p.add_argument("--ess-ratio-min", type=float, default=0.10)
    p.add_argument("--rank-stability-min", type=float, default=0.90)
    p.add_argument("--stability-seeds", type=int, default=5)

    p.add_argument("--output", required=True, help="output JSON path")
    return p


def load_cells(args: argparse.Namespace) -> list:
    if args.bundle and (args.observations or args.scored):
        raise SystemExit("pass either --bundle or --observations/--scored, not both")
    cell_features = _csv_list(args.cell_features)
    if args.bundle:
        return inst.load_bundle_manifest(args.bundle)
    if not (args.observations and args.scored):
        raise SystemExit("pass --bundle, or both --observations and --scored")
    return inst.wrap_single_cell(args.observations, args.scored, cell_features=cell_features)


def resolve_sample_index(ds, args: argparse.Namespace) -> int:
    if args.sample_index is not None:
        return args.sample_index
    if args.sample_id is not None:
        try:
            return ds.sample_ids.index(args.sample_id)
        except ValueError:
            raise SystemExit(f"sample_id {args.sample_id!r} not found in the dataset") from None
    raise SystemExit("--mode local requires --sample-id or --sample-index")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    cells = load_cells(args)
    dataset = inst.build_instability_dataset(
        cells,
        target=args.target,
        cell_features=_csv_list(args.cell_features),
        sample_features=_csv_list(args.sample_features),
        group_by=args.group_by,
        min_observations=args.min_observations,
        missing_numeric=args.missing_numeric,
        drop_constant_features=args.drop_constant_features,
        allow_raw_seed_feature=args.allow_raw_seed_feature,
    )

    model_name = args.model or ("logistic" if dataset.is_binary else "ridge")
    model_result = inst.fit_instability_model(
        dataset,
        model=model_name,
        cv_folds=args.cv_folds,
        seed=args.seed,
        min_cv_metric=args.min_cv_metric,
    )

    if args.mode == "global":
        value_fn = inst.make_global_value_fn(
            dataset, model=model_name, cv_folds=args.cv_folds, seed=args.seed, metric=args.metric
        )
    else:
        sample_index = resolve_sample_index(dataset, args)
        value_fn = inst.make_local_value_fn(model_result, sample_index, baseline=args.baseline)

    dags = [
        inst.load_attribution_dag(path, dataset.feature_names, sink_node=args.sink_node)
        for path in args.dags
    ]
    sensitivity = inst.explain_with_dag_sensitivity(
        dags,
        value_fn,
        seed=args.seed,
        ci=args.ci,
        ess_ratio_min=args.ess_ratio_min,
        rank_stability_min=args.rank_stability_min,
        stability_seeds=args.stability_seeds,
    )

    report = inst.build_attribution_report(
        dataset=dataset, model_result=model_result, sensitivity_result=sensitivity, mode=args.mode
    )
    summary = inst.summarize_attribution(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(inst.dump_attribution_json(report))

    print(f"=== {report['target']} ({report['mode']} mode, model={report['model']['type']}) ===")
    print(f"CV metric ({report['model']['cv_metric_name']}): {report['model']['cv_metric']:.4f}")
    if report["model"]["min_cv_metric"] is not None:
        verdict = "OK" if report["model"]["meets_min_cv_metric"] else "BELOW THRESHOLD"
        print(f"  vs. min_cv_metric {report['model']['min_cv_metric']}: {verdict}")
    print()
    for bucket, names in summary.items():
        if names:
            print(f"{bucket}: {', '.join(names)}")
    if report["warnings"]:
        print("\nwarnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
    print(f"\nWritten to {output_path}")
    print(
        "\nReminder: these are ASV attributions under the supplied DAG and model, "
        "not causal effect estimates. Use the top-ranked factor as a hypothesis "
        "for a follow-up controlled re-evaluation, not as a standalone conclusion."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
