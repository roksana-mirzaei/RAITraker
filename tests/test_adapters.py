"""Unit tests for all three supplier adapters against their synthetic data files."""
from __future__ import annotations

import json
from pathlib import Path

from adapters.supplier_a import SupplierAAdapter
from adapters.supplier_b import SupplierBAdapter
from adapters.supplier_c import SupplierCAdapter
from schema.canonical import CanonicalInteraction

_DATA_ROOT = Path(__file__).parent.parent / "data"
_SUPPLIER_A_JSON = _DATA_ROOT / "supplier_a" / "synthetic.json"
_SUPPLIER_B_CSV = _DATA_ROOT / "supplier_b" / "synthetic.csv"
_SUPPLIER_C_JSON = _DATA_ROOT / "supplier_c" / "synthetic.json"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_all_valid(records: list[CanonicalInteraction]) -> None:
    """Assert every record is a non-empty CanonicalInteraction."""
    for r in records:
        assert isinstance(r, CanonicalInteraction)
        assert r.interaction_id, "interaction_id must not be empty"
        assert r.supplier_id, "supplier_id must not be empty"
        assert r.user_query, "user_query must not be empty"
        assert r.ai_response, "ai_response must not be empty"


# ---------------------------------------------------------------------------
# Supplier A — JSON dict list
# ---------------------------------------------------------------------------

class TestSupplierAAdapter:
    def setup_method(self):
        with open(_SUPPLIER_A_JSON) as f:
            data = json.load(f)
        self.adapter = SupplierAAdapter()
        self.records = self.adapter.ingest(data)

    def test_record_count(self):
        assert len(self.records) == 30

    def test_all_valid_canonical_interactions(self):
        _assert_all_valid(self.records)

    def test_interaction_id_namespace(self):
        for r in self.records:
            assert r.interaction_id.startswith("supplier_a_"), (
                f"Expected 'supplier_a_' prefix, got: {r.interaction_id!r}"
            )

    def test_supplier_id_field(self):
        assert all(r.supplier_id == "supplier_a" for r in self.records)

    def test_confidence_score_always_none(self):
        assert all(r.confidence_score is None for r in self.records)

    def test_timestamps_populated(self):
        records_with_ts = [r for r in self.records if r.timestamp is not None]
        assert len(records_with_ts) > 0

    def test_coverage_confidence_unavailable(self):
        coverage = {c.field_name: c for c in self.adapter.get_coverage(self.records)}
        assert coverage["confidence_score"].available is False

    def test_coverage_timestamp_available(self):
        coverage = {c.field_name: c for c in self.adapter.get_coverage(self.records)}
        assert coverage["timestamp"].available is True

    def test_ingest_accepts_wrapped_dict(self):
        with open(_SUPPLIER_A_JSON) as f:
            raw = json.load(f)
        wrapped = {"records": raw}
        records = self.adapter.ingest(wrapped)
        assert len(records) == 30


# ---------------------------------------------------------------------------
# Supplier B — CSV file path
# ---------------------------------------------------------------------------

class TestSupplierBAdapter:
    def setup_method(self):
        self.adapter = SupplierBAdapter()
        self.records = self.adapter.ingest(str(_SUPPLIER_B_CSV))

    def test_record_count(self):
        assert len(self.records) == 30

    def test_all_valid_canonical_interactions(self):
        _assert_all_valid(self.records)

    def test_interaction_id_namespace(self):
        for r in self.records:
            assert r.interaction_id.startswith("supplier_b_"), (
                f"Expected 'supplier_b_' prefix, got: {r.interaction_id!r}"
            )

    def test_supplier_id_field(self):
        assert all(r.supplier_id == "supplier_b" for r in self.records)

    def test_confidence_scores_present_and_valid(self):
        for r in self.records:
            assert r.confidence_score is not None, (
                f"Expected confidence_score on {r.interaction_id}"
            )
            assert 0.0 <= r.confidence_score <= 1.0

    def test_model_fields_always_none(self):
        for r in self.records:
            assert r.model_name is None
            assert r.model_version is None

    def test_coverage_confidence_available(self):
        coverage = {c.field_name: c for c in self.adapter.get_coverage(self.records)}
        assert coverage["confidence_score"].available is True

    def test_coverage_model_name_unavailable(self):
        coverage = {c.field_name: c for c in self.adapter.get_coverage(self.records)}
        assert coverage["model_name"].available is False


# ---------------------------------------------------------------------------
# Supplier C — JSON with interactions list
# ---------------------------------------------------------------------------

class TestSupplierCAdapter:
    def setup_method(self):
        with open(_SUPPLIER_C_JSON) as f:
            data = json.load(f)
        self.adapter = SupplierCAdapter()
        self.records = self.adapter.ingest(data)

    def test_record_count(self):
        assert len(self.records) == 50

    def test_all_valid_canonical_interactions(self):
        _assert_all_valid(self.records)

    def test_interaction_id_namespace(self):
        for r in self.records:
            assert r.interaction_id.startswith("supplier_c_"), (
                f"Expected 'supplier_c_' prefix, got: {r.interaction_id!r}"
            )

    def test_supplier_id_field(self):
        assert all(r.supplier_id == "supplier_c" for r in self.records)

    def test_all_optional_fields_none(self):
        for r in self.records:
            assert r.timestamp is None
            assert r.model_name is None
            assert r.model_version is None
            assert r.prompt_tokens is None
            assert r.response_tokens is None
            assert r.confidence_score is None

    def test_coverage_all_fields_unavailable(self):
        coverage = self.adapter.get_coverage(self.records)
        for indicator in coverage:
            assert indicator.available is False, (
                f"Expected all fields unavailable for Supplier C, "
                f"but {indicator.field_name} is available"
            )

    def test_coverage_includes_reason(self):
        coverage = self.adapter.get_coverage(self.records)
        for indicator in coverage:
            assert indicator.reason is not None
