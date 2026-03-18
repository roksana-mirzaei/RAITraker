from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from schema.canonical import CanonicalInteraction, CoverageIndicator


class BaseAdapter(ABC):
    """Abstract base class for all supplier adapters."""

    @abstractmethod
    def ingest(self, source: Any) -> list[CanonicalInteraction]:
        """Normalise raw supplier data into canonical interactions."""

    @abstractmethod
    def get_coverage(self, records: list[CanonicalInteraction]) -> list[CoverageIndicator]:
        """Return field-level coverage indicators for the given records."""
