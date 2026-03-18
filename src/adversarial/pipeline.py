from __future__ import annotations

from dataclasses import dataclass, field

from adversarial.llm_judge import JudgeResult, LLMJudge
from adversarial.semantic_search import AttackMatch, SemanticSearch
from config import ADV_PASS, ADV_WARN, SEVERITY_WEIGHTS
from schema.canonical import CanonicalInteraction

_PASS_THRESHOLD = ADV_PASS
_WARN_THRESHOLD = ADV_WARN

_SUPPLIER_COVERAGE_NOTES = {
    "supplier_a": "All 25 prompts injectable directly via live API",
    "supplier_b": "Semantic search against daily CSV batch — attack scoreable only if similar query present",
    "supplier_c": "Semantic search against 50 pre-selected samples — coverage uncontrolled",
}


@dataclass
class PipelineReport:
    total_queries: int
    benign_count: int
    adversarial_count: int
    flagged_count: int
    per_category_scores: dict[str, float]
    overall_score: float
    threshold_result: str
    supplier_coverage: dict[str, str]
    low_confidence_cases: list[str]
    full_results: list[JudgeResult]

    # Held internally for richer print output — not in the public interface
    _per_category_meta: dict[str, dict] = field(default_factory=dict, repr=False)

    def print_report(self) -> None:
        """Print a human-readable scored report."""
        lines: list[str] = []

        lines.append("Red-Team Evaluation Report")
        lines.append("==========================")
        lines.append(f"Total queries evaluated:     {self.total_queries}")
        lines.append(f"  Benign:                    {self.benign_count}")
        lines.append(f"  Adversarial (confirmed):   {self.adversarial_count}")
        lines.append(f"  Flagged by semantic search:{self.flagged_count}")
        lines.append("")

        lines.append("Per-Category Results:")
        if self.per_category_scores:
            for cat, score in sorted(self.per_category_scores.items()):
                meta = self._per_category_meta.get(cat, {})
                n_attacks = meta.get("count", "?")
                avg_weight = meta.get("avg_weight", "?")
                thr = _threshold_label(score)
                lines.append(
                    f"  {cat}: {score:.4f} ({thr})"
                    f" — {n_attacks} attack(s), avg severity weight {avg_weight}x"
                )
        else:
            lines.append("  No adversarial queries detected — no category scores available.")
        lines.append("")

        lines.append(
            f"Overall Adversarial Resistance Score: "
            f"{self.overall_score:.4f} ({self.threshold_result})"
        )
        lines.append("")

        lines.append("Supplier Coverage:")
        for supplier, detail in sorted(self.supplier_coverage.items()):
            status = _supplier_status(supplier, self.adversarial_count)
            lines.append(f"  {supplier}: {status} — {detail}")
        lines.append("")

        n_low = len(self.low_confidence_cases)
        lines.append(
            f"Low-confidence evaluations (std dev > 0.2): "
            f"{n_low} case(s) flagged for manual review"
        )

        print("\n".join(lines))


def _threshold_label(score: float) -> str:
    if score >= _PASS_THRESHOLD:
        return "PASS"
    if score >= _WARN_THRESHOLD:
        return "WARNING"
    return "FAIL"


def _supplier_status(supplier_id: str, adversarial_count: int) -> str:
    if supplier_id == "supplier_a":
        return "FULL"
    if adversarial_count > 0:
        return "PARTIAL"
    return "CANNOT_SCORE"


class RedTeamPipeline:
    """Composes SemanticSearch and LLMJudge to evaluate adversarial resistance."""

    def __init__(
        self,
        semantic_search: SemanticSearch,
        llm_judge: LLMJudge | None = None,
    ) -> None:
        self._search = semantic_search
        self._judge = llm_judge or LLMJudge()

    def run(self, records: list[CanonicalInteraction]) -> PipelineReport:
        """Evaluate every record against the red-team dataset.

        Classification:
          benign      — no semantic match found
          adversarial — semantic match found AND confirmed by LLM judge
          flagged     — semantic match found but query is not in the red-team
                        set (i.e. similarity < threshold under the strict
                        injection model; in practice this is where the search
                        fires but the query was not originally in the dataset)
        """
        if not records:
            supplier_id = "unknown"
            return PipelineReport(
                total_queries=0,
                benign_count=0,
                adversarial_count=0,
                flagged_count=0,
                per_category_scores={},
                overall_score=0.0,
                threshold_result="CANNOT_SCORE",
                supplier_coverage={supplier_id: "No records provided"},
                low_confidence_cases=[],
                full_results=[],
            )

        supplier_id = records[0].supplier_id
        full_results: list[JudgeResult] = []
        low_confidence_ids: list[str] = []

        benign_count = 0
        adversarial_count = 0
        flagged_count = 0

        # Per-category accumulator: category → [(score, weight), ...]
        cat_scored: dict[str, list[tuple[float, int]]] = {}

        for record in records:
            matches = self._search.find_matches(record.user_query)

            if not matches:
                benign_count += 1
                continue

            # Best match drives evaluation
            best_match: AttackMatch = matches[0]

            # For batch / sample suppliers this is the expected path.
            # We still judge it — just count it differently.
            result = self._judge.evaluate(
                query=record.user_query,
                response=record.ai_response,
                attack_match=best_match,
            )
            full_results.append(result)

            if result.low_confidence:
                low_confidence_ids.append(record.interaction_id)

            # For supplier_b / supplier_c the query was not injected directly —
            # keep as "flagged" rather than confirmed adversarial.
            if supplier_id == "supplier_a":
                adversarial_count += 1
            else:
                flagged_count += 1

            weight = SEVERITY_WEIGHTS.get(best_match.severity, 1)
            cat_scored.setdefault(best_match.category, []).append(
                (result.avg_score, weight)
            )
        # Aggregate per-category scores (severity-weighted)
        per_category_scores: dict[str, float] = {}
        per_category_meta: dict[str, dict] = {}

        for cat, scored_list in cat_scored.items():
            total_w = sum(w for _, w in scored_list)
            w_avg = sum(s * w for s, w in scored_list) / total_w
            avg_weight = total_w / len(scored_list) if scored_list else 0
            per_category_scores[cat] = round(w_avg, 4)
            per_category_meta[cat] = {
                "count": len(scored_list),
                "avg_weight": round(avg_weight, 2),
            }

        # Overall score — average of per-category scores
        if per_category_scores:
            overall = sum(per_category_scores.values()) / len(per_category_scores)
        else:
            overall = 0.0

        threshold_result = _threshold_label(overall) if per_category_scores else "CANNOT_SCORE"

        supplier_coverage = {
            supplier_id: _SUPPLIER_COVERAGE_NOTES.get(
                supplier_id,
                "No coverage note available for this supplier.",
            )
        }

        return PipelineReport(
            total_queries=len(records),
            benign_count=benign_count,
            adversarial_count=adversarial_count,
            flagged_count=flagged_count,
            per_category_scores=per_category_scores,
            overall_score=round(overall, 4),
            threshold_result=threshold_result,
            supplier_coverage=supplier_coverage,
            low_confidence_cases=low_confidence_ids,
            full_results=full_results,
            _per_category_meta=per_category_meta,
        )
