from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from coverage.reporter import CoverageStatus


@dataclass
class MetricResult:
    metric_name: str
    supplier_id: str
    score: Optional[float]
    status: CoverageStatus
    sub_scores: dict[str, float]
    threshold_result: str  # "PASS", "WARNING", "FAIL", or "CANNOT_SCORE"
    audit_findings: list[str]
    sample_size: int
    notes: str
