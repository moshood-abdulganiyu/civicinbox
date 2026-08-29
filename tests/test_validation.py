import pytest
from pydantic import ValidationError

from app.models.schema import Entities, TriageResult


def test_valid_triage_result():
    result = TriageResult(
        category="financial_aid",
        urgency="high",
        requested_action="Check status of scholarship application",
        entities=Entities(student_id="S12345", email="jane@example.edu"),
        missing_information=[],
        draft_response="Thank you for reaching out. We are reviewing your scholarship status...",
        confidence=0.87,
    )
    assert result.category == "financial_aid"
    assert result.confidence == 0.87


def test_invalid_category_rejected():
    with pytest.raises(ValidationError):
        TriageResult(
            category="billing_issue",  # not in the taxonomy
            urgency="high",
            requested_action="Check status",
            entities=Entities(),
            missing_information=[],
            draft_response="...",
            confidence=0.5,
        )


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValidationError):
        TriageResult(
            category="registration",
            urgency="low",
            requested_action="Add a course",
            entities=Entities(),
            missing_information=[],
            draft_response="...",
            confidence=1.5,  # out of [0,1] bound
        )


def test_missing_required_field_rejected():
    with pytest.raises(ValidationError):
        TriageResult(
            category="registration",
            urgency="low",
            # requested_action missing entirely
            entities=Entities(),
            missing_information=[],
            draft_response="...",
            confidence=0.5,
        )
