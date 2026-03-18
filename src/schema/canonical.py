from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class CanonicalInteraction:
    # Required fields
    interaction_id: str
    supplier_id: str
    user_query: str
    ai_response: str

    # Optional fields
    timestamp: Optional[datetime] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    prompt_tokens: Optional[int] = None
    response_tokens: Optional[int] = None
    confidence_score: Optional[float] = None

    def __post_init__(self) -> None:
        for fname, value in (
            ("interaction_id", self.interaction_id),
            ("supplier_id", self.supplier_id),
            ("user_query", self.user_query),
            ("ai_response", self.ai_response),
        ):
            if value == "":
                raise ValueError(f"'{fname}' must not be an empty string")


@dataclass
class CoverageIndicator:
    field_name: str
    available: bool
    reason: Optional[str] = None
