from __future__ import annotations

from typing import Any

from adapters.base import BaseAdapter
from schema.canonical import CanonicalInteraction, CoverageIndicator

_OPTIONAL_FIELDS = [
    "timestamp",
    "model_name",
    "model_version",
    "prompt_tokens",
    "response_tokens",
    "confidence_score",
]


class SupplierCAdapter(BaseAdapter):
    """Adapter for Supplier C — consumes a dict with a list of sampled interactions."""

    def ingest(self, source: Any) -> list[CanonicalInteraction]:
        records: list[dict] = source if isinstance(source, list) else source["interactions"]
        month: str = source.get("month", "") if isinstance(source, dict) else ""
        interactions: list[CanonicalInteraction] = []
        for index, record in enumerate(records):
            record_month = record.get("month", month)
            interactions.append(
                CanonicalInteraction(
                    interaction_id=f"supplier_c_{record_month}_{index}",
                    supplier_id="supplier_c",
                    user_query=record["query"],
                    ai_response=record["response"],
                    timestamp=None,
                    model_name=None,
                    model_version=None,
                    prompt_tokens=None,
                    response_tokens=None,
                    confidence_score=None,
                )
            )
        return interactions

    def get_coverage(self, records: list[CanonicalInteraction]) -> list[CoverageIndicator]:
        return [
            CoverageIndicator(
                field_name=field_name,
                available=False,
                reason="Supplier C provides query and response only",
            )
            for field_name in _OPTIONAL_FIELDS
        ]
