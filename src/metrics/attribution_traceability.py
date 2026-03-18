from __future__ import annotations

import json
import os
import re

import anthropic

from config import ATTR_PASS as _ATTR_PASS, ATTR_WARN as _ATTR_WARN, LLM_MAX_TOKENS, LLM_MODEL
from coverage.reporter import CoverageStatus
from metrics.base import MetricResult
from schema.canonical import CanonicalInteraction

# Thresholds
# Helpers


def _get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set.")
    return anthropic.Anthropic(api_key=api_key)


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not extract JSON from LLM output: {text!r}")


def _llm_judge(
    query: str, response: str
) -> tuple[bool, float, float, str]:
    """Call the LLM judge 3 times.

    Returns:
        (is_factual, avg_attribution_score, avg_traceability_score, last_reasoning)
        where is_factual is True if at least 2 of 3 calls classify the response
        as containing a factual claim.
    """
    prompt = (
        "You are a transparency evaluator for a UK government benefits chatbot.\n\n"
        "Step 1 — Classify: does this AI response make any FACTUAL CLAIMS? "
        "Factual claims include: specific monetary amounts, dates, eligibility thresholds, "
        "legal references, named government services, or time-based rules.\n\n"
        "Step 2 — If YES, score the following:\n\n"
        "attribution_score — how well are claims attributed to authoritative sources?\n"
        "  0.0  = no attribution at all\n"
        "  0.33 = vague or implied attribution (e.g. 'the government says')\n"
        "  0.67 = partial attribution (e.g. mentions gov.uk but no specific page)\n"
        "  1.0  = full attribution (specific legislation, regulation, or policy cited)\n\n"
        "traceability_score — how easily could a user independently verify the claims?\n"
        "  0.0 = no traceability — claims cannot be verified from the response alone\n"
        "  0.5 = partial traceability — some information given to help find the source\n"
        "  1.0 = full traceability — direct URL, legislation reference, or unique "
        "identifier provided\n\n"
        f"User query:\n{query}\n\n"
        f"AI response:\n{response}\n\n"
        "Return ONLY valid JSON, no other text. "
        "If is_factual_claim is false, set attribution_score and traceability_score to 0.0:\n"
        '{"is_factual_claim": <true or false>, '
        '"attribution_score": <0.0, 0.33, 0.67, or 1.0>, '
        '"traceability_score": <0.0, 0.5, or 1.0>, '
        '"reasoning": "<one sentence>"}'
    )

    client = _get_client()
    factual_votes: list[bool] = []
    attr_sum = 0.0
    trace_sum = 0.0
    last_reasoning = ""

    _VALID_ATTR = (0.0, 0.33, 0.67, 1.0)
    _VALID_TRACE = (0.0, 0.5, 1.0)

    for _ in range(3):
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = _parse_json(msg.content[0].text)

        is_factual = bool(parsed.get("is_factual_claim", False))
        factual_votes.append(is_factual)

        raw_attr = float(parsed.get("attribution_score", 0.0))
        raw_trace = float(parsed.get("traceability_score", 0.0))

        # Clamp to nearest valid value
        attr_sum += min(_VALID_ATTR, key=lambda v: abs(v - raw_attr))
        trace_sum += min(_VALID_TRACE, key=lambda v: abs(v - raw_trace))
        last_reasoning = parsed.get("reasoning", "")

    # Majority vote for factual classification
    is_factual_final = sum(factual_votes) >= 2
    return is_factual_final, attr_sum / 3.0, trace_sum / 3.0, last_reasoning


def _threshold_label(score: float) -> str:
    if score >= _ATTR_PASS:
        return "PASS"
    if score >= _ATTR_WARN:
        return "WARNING"
    return "FAIL"


# Metric class

class AttributionTraceabilityMetric:
    """Evaluates whether AI responses containing factual claims adequately
    attribute and enable traceability to authoritative sources."""

    _INFRASTRUCTURE_FINDING = (
        "No source metadata provided in supplier data. "
        "Cannot determine what sources informed responses. "
        "Infrastructure-level transparency gap independent of response-level scores."
    )

    def score(self, records: list[CanonicalInteraction]) -> MetricResult:
        supplier_id = records[0].supplier_id if records else "unknown"

        # Always raise the infrastructure audit finding
        audit_findings: list[str] = [self._INFRASTRUCTURE_FINDING]

        if not records:
            return MetricResult(
                metric_name="attribution_traceability",
                supplier_id=supplier_id,
                score=None,
                status=CoverageStatus.CANNOT_SCORE,
                sub_scores={},
                threshold_result="CANNOT_SCORE",
                audit_findings=audit_findings + ["No records provided."],
                sample_size=0,
                notes="Cannot score attribution traceability without records.",
            )

        response_scores: list[float] = []
        attr_scores: list[float] = []
        trace_scores: list[float] = []
        non_factual_count = 0

        for record in records:
            is_factual, attr, trace, _ = _llm_judge(
                record.user_query, record.ai_response
            )

            if not is_factual:
                non_factual_count += 1
                # Flag as "attribution not required" — skip from scoring
                continue

            response_score = (attr * 0.4) + (trace * 0.6)
            response_scores.append(response_score)
            attr_scores.append(attr)
            trace_scores.append(trace)

        if not response_scores:
            return MetricResult(
                metric_name="attribution_traceability",
                supplier_id=supplier_id,
                score=None,
                status=CoverageStatus.CANNOT_SCORE,
                sub_scores={},
                threshold_result="CANNOT_SCORE",
                audit_findings=audit_findings + [
                    "No factual-claim responses detected — attribution not required "
                    "for any response in this dataset."
                ],
                sample_size=len(records),
                notes=(
                    f"0 of {len(records)} responses classified as factual claims. "
                    f"{non_factual_count} flagged as 'attribution not required'."
                ),
            )

        overall = sum(response_scores) / len(response_scores)
        avg_attr = sum(attr_scores) / len(attr_scores)
        avg_trace = sum(trace_scores) / len(trace_scores)

        sub_scores = {
            "attribution": round(avg_attr, 4),
            "traceability": round(avg_trace, 4),
            "weighted_response": round(overall, 4),
        }

        label = _threshold_label(overall)
        if label in ("WARNING", "FAIL"):
            audit_findings.append(
                f"Attribution traceability {label}: overall score {overall:.3f} "
                f"(PASS threshold: {_ATTR_PASS}, WARNING threshold: {_ATTR_WARN}). "
                f"Avg attribution: {avg_attr:.3f}, avg traceability: {avg_trace:.3f}."
            )

        scored_count = len(response_scores)
        return MetricResult(
            metric_name="attribution_traceability",
            supplier_id=supplier_id,
            score=round(overall, 4),
            status=CoverageStatus.FULL if scored_count >= 5 else CoverageStatus.PARTIAL,
            sub_scores=sub_scores,
            threshold_result=label,
            audit_findings=audit_findings,
            sample_size=scored_count,
            notes=(
                f"Scored {scored_count} factual-claim response(s) out of "
                f"{len(records)} total records. "
                f"{non_factual_count} response(s) flagged as 'attribution not required'. "
                f"Avg attribution: {avg_attr:.3f}, avg traceability: {avg_trace:.3f}."
            ),
        )
