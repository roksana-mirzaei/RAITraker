from __future__ import annotations

from datetime import datetime
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


class SupplierAAdapter(BaseAdapter):
    """Adapter for Supplier A — consumes a list of JSON API response dicts."""

    def ingest(self, source: Any) -> list[CanonicalInteraction]:
        records: list[dict] = source if isinstance(source, list) else source["records"]
        interactions: list[CanonicalInteraction] = []
        for record in records:
            timestamp: datetime | None = None
            raw_ts = record.get("timestamp")
            if raw_ts:
                timestamp = datetime.fromisoformat(raw_ts)

            interactions.append(
                CanonicalInteraction(
                    interaction_id=f"supplier_a_{record['id']}",
                    supplier_id="supplier_a",
                    user_query=record["prompt"],
                    ai_response=record["response"],
                    timestamp=timestamp,
                    model_name=record.get("model_name"),
                    model_version=record.get("model_version"),
                    prompt_tokens=record.get("prompt_tokens"),
                    response_tokens=record.get("response_tokens"),
                    confidence_score=None,  # Supplier A does not provide this
                )
            )
        return interactions

    def get_coverage(self, records: list[CanonicalInteraction]) -> list[CoverageIndicator]:
        if not records:
            return [
                CoverageIndicator(field_name=f, available=False, reason="No records provided")
                for f in _OPTIONAL_FIELDS
            ]

        coverage: list[CoverageIndicator] = []
        for field_name in _OPTIONAL_FIELDS:
            if field_name == "confidence_score":
                coverage.append(
                    CoverageIndicator(
                        field_name=field_name,
                        available=False,
                        reason="Supplier A does not provide confidence scores",
                    )
                )
                continue
            available = any(getattr(r, field_name) is not None for r in records)
            coverage.append(CoverageIndicator(field_name=field_name, available=available))
        return coverage
