"""RAIT Evaluation Pipeline — demonstration entry point.

Usage:
    python src/main.py

Runs all adapters, the coverage reporter, all three LLM metrics, and the
red-team adversarial pipeline, then prints a structured summary to stdout.
Set the ANTHROPIC_API_KEY environment variable before running.
"""
from __future__ import annotations

import json
from pathlib import Path

from adapters.supplier_a import SupplierAAdapter
from adapters.supplier_b import SupplierBAdapter
from adapters.supplier_c import SupplierCAdapter
from adversarial.pipeline import RedTeamPipeline
from adversarial.semantic_search import SemanticSearch
from coverage.reporter import CoverageReporter
from metrics.adversarial_resistance import AdversarialResistanceMetric
from metrics.attribution_traceability import AttributionTraceabilityMetric
from metrics.fairness_parity import FairnessParityMetric
from schema.canonical import CanonicalInteraction

_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data"
_RED_TEAM_DATASET = _DATA_ROOT / "red_team" / "red_team_dataset.json"

_DIVIDER = "-" * 60

# Data loading


def _load_suppliers() -> tuple[
    list[CanonicalInteraction],
    list[CanonicalInteraction],
    list[CanonicalInteraction],
]:
    with open(_DATA_ROOT / "supplier_a" / "synthetic.json") as f:
        supplier_a_records = SupplierAAdapter().ingest(json.load(f))

    supplier_b_records = SupplierBAdapter().ingest(
        str(_DATA_ROOT / "supplier_b" / "synthetic.csv")
    )

    with open(_DATA_ROOT / "supplier_c" / "synthetic.json") as f:
        supplier_c_records = SupplierCAdapter().ingest(json.load(f))

    return supplier_a_records, supplier_b_records, supplier_c_records


# Coverage report


def _print_coverage(records_by_supplier: dict[str, list]) -> None:
    reporter = CoverageReporter()
    print("Coverage Report:")
    print(_DIVIDER)
    for supplier_id, records in records_by_supplier.items():
        adv = reporter.check_adversarial_coverage(records)
        fair = reporter.check_fairness_coverage(records)
        trans = reporter.check_transparency_coverage(records)
        print(f"  {supplier_id}:")
        print(
            f"    adversarial   {adv.status.value:<14s}"
            f"  {adv.scoreable_count}/{adv.total_count} scoreable"
        )
        print(
            f"    fairness      {fair.status.value:<14s}"
            f"  {fair.scoreable_count}/{fair.total_count} scoreable"
        )
        print(
            f"    transparency  {trans.status.value:<14s}"
            f"  {trans.scoreable_count}/{trans.total_count} scoreable"
        )
    print()


# Metric scores


def _print_metrics(
    records_by_supplier: dict[str, list],
    red_team_data: dict,
) -> None:
    print("Metric Scores:")
    print(_DIVIDER)

    adv_metric = AdversarialResistanceMetric()
    fair_metric = FairnessParityMetric()
    attr_metric = AttributionTraceabilityMetric()

    for supplier_id, records in records_by_supplier.items():
        print(f"  {supplier_id}:")
        for metric, kwargs in (
            (adv_metric, {"red_team_dataset": red_team_data}),
            (fair_metric, {}),
            (attr_metric, {}),
        ):
            try:
                result = metric.score(records, **kwargs)
                score_str = (
                    f"{result.score:.4f}"
                    if result.score is not None
                    else "  N/A "
                )
                print(
                    f"    {result.metric_name:<32s}"
                    f"  score={score_str}  [{result.threshold_result}]"
                )
            except RuntimeError as exc:
                name = type(metric).__name__
                print(f"    {name:<32s}  SKIPPED — {exc}")
    print()


# Red-team pipeline


def _print_redteam(supplier_a_records: list[CanonicalInteraction]) -> None:
    print("Red-Team Pipeline (Supplier A):")
    print(_DIVIDER)
    if not _RED_TEAM_DATASET.exists():
        print(f"  [SKIPPED] Dataset not found at: {_RED_TEAM_DATASET}")
        print()
        return

    try:
        search = SemanticSearch()
        search.load_dataset(str(_RED_TEAM_DATASET))
        pipeline = RedTeamPipeline(semantic_search=search)
        report = pipeline.run(supplier_a_records)
        report.print_report()
    except RuntimeError as exc:
        print(f"  [SKIPPED] {exc}")
    print()


# Entry point


def main() -> None:
    print("=== RAIT Evaluation Pipeline ===")
    print()

    print("Loading synthetic data …")
    supplier_a, supplier_b, supplier_c = _load_suppliers()
    print(
        f"  supplier_a={len(supplier_a)} records, "
        f"supplier_b={len(supplier_b)} records, "
        f"supplier_c={len(supplier_c)} records"
    )
    print()

    # Load the red-team dataset once; pass to metrics and pipeline.
    red_team_data: dict = {}
    if _RED_TEAM_DATASET.exists():
        with open(_RED_TEAM_DATASET) as f:
            red_team_data = json.load(f)
    else:
        print(
            f"  [WARNING] Red-team dataset not found at {_RED_TEAM_DATASET}. "
            "Adversarial metric will return CANNOT_SCORE."
        )
        print()

    records_by_supplier = {
        "supplier_a": supplier_a,
        "supplier_b": supplier_b,
        "supplier_c": supplier_c,
    }

    _print_coverage(records_by_supplier)
    _print_metrics(records_by_supplier, red_team_data)
    _print_redteam(supplier_a)

    print("=== End of Report ===")


if __name__ == "__main__":
    main()
