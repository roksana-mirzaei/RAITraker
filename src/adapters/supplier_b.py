from __future__ import annotations

import csv
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

_UNAVAILABLE_FIELDS = {"model_name", "model_version", "prompt_tokens", "response_tokens"}


class SupplierBAdapter(BaseAdapter):
    """Adapter for Supplier B — consumes a path to a CSV file."""

    def ingest(self, source: Any) -> list[CanonicalInteraction]:
        csv_path: str = source
        interactions: list[CanonicalInteraction] = []
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for index, row in enumerate(reader):
                timestamp: datetime | None = None
                raw_ts = row.get("timestamp", "").strip()
                if raw_ts:
                    timestamp = datetime.fromisoformat(raw_ts)

                raw_confidence = row.get("confidence_score", "").strip()
                confidence_score: float | None = float(raw_confidence) if raw_confidence else None

                date_part = row.get("date", raw_ts[:10] if raw_ts else "").strip()

                interactions.append(
                    CanonicalInteraction(
                        interaction_id=f"supplier_b_{date_part}_{index}",
                        supplier_id="supplier_b",
                        user_query=row["user_query"],
                        ai_response=row["system_response"],
                        timestamp=timestamp,
                        model_name=None,
                        model_version=None,
                        prompt_tokens=None,
                        response_tokens=None,
                        confidence_score=confidence_score,
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
            if field_name in _UNAVAILABLE_FIELDS:
                coverage.append(
                    CoverageIndicator(
                        field_name=field_name,
                        available=False,
                        reason="Supplier B does not provide this field",
                    )
                )
                continue
            available = any(getattr(r, field_name) is not None for r in records)
            coverage.append(CoverageIndicator(field_name=field_name, available=available))
        return coverage
