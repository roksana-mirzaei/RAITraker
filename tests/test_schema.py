"""Unit tests for schema.canonical — CanonicalInteraction and CoverageIndicator."""
from __future__ import annotations

from datetime import datetime

import pytest

from schema.canonical import CanonicalInteraction, CoverageIndicator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(**kwargs) -> CanonicalInteraction:
    defaults = dict(
        interaction_id="test-001",
        supplier_id="supplier_test",
        user_query="What is Universal Credit?",
        ai_response="Universal Credit is a monthly payment.",
    )
    defaults.update(kwargs)
    return CanonicalInteraction(**defaults)


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------

class TestCanonicalInteractionRequiredFields:
    def test_valid_creation(self):
        ci = _make()
        assert ci.interaction_id == "test-001"
        assert ci.supplier_id == "supplier_test"
        assert ci.user_query == "What is Universal Credit?"
        assert ci.ai_response == "Universal Credit is a monthly payment."

    def test_empty_interaction_id_raises(self):
        with pytest.raises(ValueError, match="interaction_id"):
            _make(interaction_id="")

    def test_empty_supplier_id_raises(self):
        with pytest.raises(ValueError, match="supplier_id"):
            _make(supplier_id="")

    def test_empty_user_query_raises(self):
        with pytest.raises(ValueError, match="user_query"):
            _make(user_query="")

    def test_empty_ai_response_raises(self):
        with pytest.raises(ValueError, match="ai_response"):
            _make(ai_response="")

    def test_whitespace_only_is_valid(self):
        # Only the empty string "" is rejected — whitespace-only strings pass
        ci = _make(user_query="   ", ai_response="\t")
        assert ci.user_query == "   "

    def test_single_character_is_valid(self):
        ci = _make(user_query="?", ai_response=".")
        assert ci.user_query == "?"


# ---------------------------------------------------------------------------
# Optional field defaults
# ---------------------------------------------------------------------------

class TestCanonicalInteractionOptionalFields:
    def test_all_optional_fields_default_to_none(self):
        ci = _make()
        assert ci.timestamp is None
        assert ci.model_name is None
        assert ci.model_version is None
        assert ci.prompt_tokens is None
        assert ci.response_tokens is None
        assert ci.confidence_score is None

    def test_optional_fields_accept_values(self):
        ci = _make(
            timestamp=datetime(2026, 3, 17, 12, 0, 0),
            model_name="claude-sonnet-4-20250514",
            model_version="2025-05-14",
            prompt_tokens=42,
            response_tokens=128,
            confidence_score=0.95,
        )
        assert ci.timestamp == datetime(2026, 3, 17, 12, 0, 0)
        assert ci.model_name == "claude-sonnet-4-20250514"
        assert ci.model_version == "2025-05-14"
        assert ci.prompt_tokens == 42
        assert ci.response_tokens == 128
        assert ci.confidence_score == pytest.approx(0.95)

    def test_partial_optional_population(self):
        ci = _make(model_name="gpt-4o")
        assert ci.model_name == "gpt-4o"
        assert ci.timestamp is None
        assert ci.confidence_score is None


# ---------------------------------------------------------------------------
# CoverageIndicator
# ---------------------------------------------------------------------------

class TestCoverageIndicator:
    def test_available_true_no_reason(self):
        ci = CoverageIndicator(field_name="timestamp", available=True)
        assert ci.available is True
        assert ci.reason is None

    def test_available_false_with_reason(self):
        ci = CoverageIndicator(
            field_name="confidence_score",
            available=False,
            reason="Supplier A does not provide confidence scores",
        )
        assert ci.available is False
        assert "Supplier A" in ci.reason

    def test_reason_defaults_to_none(self):
        ci = CoverageIndicator(field_name="model_name", available=False)
        assert ci.reason is None

    def test_field_name_stored_correctly(self):
        ci = CoverageIndicator(field_name="prompt_tokens", available=True)
        assert ci.field_name == "prompt_tokens"
