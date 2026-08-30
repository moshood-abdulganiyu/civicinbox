import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.core.preprocessing import normalize_text


def test_baseline_eval_runs():
    """
    Runs scripts/evaluate_baseline.py end-to-end and checks the JSON report
    it writes has the expected keys and sane value ranges.

    Precondition: data/models/baseline.joblib and data/models/test_split.json
    must already exist (produced by scripts/train_baseline.py — Step 8).
    This test does NOT regenerate them; it assumes the dev has already run
    the training script locally. If those files are missing, this test
    should fail loudly rather than silently pass or auto-train.
    """
    model_path = Path("data/models/baseline.joblib")
    split_path = Path("data/models/test_split.json")
    assert model_path.exists(), (
        "data/models/baseline.joblib not found — run "
        "scripts/train_baseline.py first (Step 8)."
    )
    assert split_path.exists(), (
        "data/models/test_split.json not found — run "
        "scripts/train_baseline.py first (Step 8)."
    )

    result = subprocess.run(
        [sys.executable, "scripts/evaluate_baseline.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"Script failed:\n{result.stderr}"

    results_dir = Path("evals/results")
    report_files = sorted(results_dir.glob("baseline_*.json"))
    assert report_files, "No baseline_<date>.json report was written."

    report = json.loads(report_files[-1].read_text())

    # Top-level schema
    for key in (
        "date",
        "model",
        "n_test_examples",
        "macro_f1",
        "accuracy",
        "per_class",
    ):
        assert key in report, f"Missing key: {key}"

    # Value sanity
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert 0.0 <= report["accuracy"] <= 1.0
    assert report["n_test_examples"] > 0
    assert isinstance(report["per_class"], dict)
    assert len(report["per_class"]) > 0

    # Per-class schema
    for cat, vals in report["per_class"].items():
        for k in ("precision", "recall", "f1", "support"):
            assert k in vals, f"Missing '{k}' for category '{cat}'"
        assert 0.0 <= vals["precision"] <= 1.0
        assert 0.0 <= vals["recall"] <= 1.0
        assert 0.0 <= vals["f1"] <= 1.0


def test_normalize_text_whitespace_and_case():
    raw = "  My Scholarship Application   Has Not Been Reviewed!!  \n"
    expected = "my scholarship application has not been reviewed!"
    assert normalize_text(raw) == expected


def test_normalize_text_internal_newlines_collapse():
    raw = "HELLO\n\n\nWorld"
    expected = "hello world"
    assert normalize_text(raw) == expected


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


def test_normalize_text_repeated_punctuation():
    assert normalize_text("Is this urgent???") == "is this urgent?"
    assert normalize_text("please help....") == "please help."


def test_llm_client_mocked():
    """Mocks get_client() entirely, so this test never touches the
    network and never costs API credit -- required, since the master
    plan's budget-aware rule says real LLM calls happen only where I
    deliberately choose to spend, never as a side effect of running
    the test suite.

    Confirms call_llm() correctly navigates the OpenAI response shape
    (response.choices[0].message.content) and returns it as plain text.
    """
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="mocked reply"))]

    with patch("app.services.llm_client.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = fake_response
        mock_get_client.return_value = mock_client

        from app.services.llm_client import call_llm

        result = call_llm("test prompt")

    assert result == "mocked reply"
    mock_client.chat.completions.create.assert_called_once()


# ---------------------------------------------------------------------------
# STEP 12 — LLM classifier unit tests (mocked)
# ---------------------------------------------------------------------------

import pytest

from app.services.llm_classifier import LLMClassificationFailed, classify_with_llm

VALID_RESPONSE = """{
    "category": "financial_aid",
    "urgency": "high",
    "requested_action": "review application status",
    "entities": {"student_id": "12345"},
    "missing_information": [],
    "draft_response": "Thank you for reaching out, we are reviewing your application.",
    "confidence": 0.9
}"""

MARKDOWN_WRAPPED_RESPONSE = f"""```json
                                {VALID_RESPONSE}
                                ```"""

MISSING_FIELD_RESPONSE = """{
                                "category": "financial_aid",
                                "urgency": "high"
                            }"""

VALID_RESPONSE_DOCUMENT_REQUEST = """{
    "category": "document_request",
    "urgency": "low",
    "requested_action": "provide submission instructions",
    "entities": {},
    "missing_information": [],
    "draft_response": "You can submit the document through the student portal.",
    "confidence": 0.85
}"""


def test_classify_with_llm_recovers_after_malformed_then_valid():
    """First attempt returns markdown-fenced JSON (JSONDecodeError).
    Second attempt returns valid JSON. Should recover and return a
    valid TriageResult without exhausting all 3 attempts."""
    with patch(
        "app.services.llm_classifier.call_llm",
        side_effect=[MARKDOWN_WRAPPED_RESPONSE, VALID_RESPONSE],
    ) as mock_call:
        result = classify_with_llm("My scholarship has not been reviewed.")

        assert result.category == "financial_aid"
        assert result.urgency == "high"
        assert mock_call.call_count == 2


def test_classify_with_llm_recovers_after_validation_error_then_valid():
    """First attempt is valid JSON but missing required fields
    (ValidationError). Second attempt is fully valid. Should recover."""
    with patch(
        "app.services.llm_classifier.call_llm",
        side_effect=[MISSING_FIELD_RESPONSE, VALID_RESPONSE_DOCUMENT_REQUEST],
    ) as mock_call:
        result = classify_with_llm("Where do I submit this document?")

        assert result.category == "document_request"
        assert mock_call.call_count == 2


def test_classify_with_llm_fails_gracefully_after_exhausting_attempts(caplog):
    """All attempts return malformed/invalid output. Should raise
    LLMClassificationFailed (not crash with an unhandled exception),
    and log the failure."""
    with patch(
        "app.services.llm_classifier.call_llm",
        side_effect=[
            MARKDOWN_WRAPPED_RESPONSE,
            MISSING_FIELD_RESPONSE,
            "not json at all",
        ],
    ) as mock_call:
        with pytest.raises(LLMClassificationFailed) as exc_info:
            classify_with_llm("The water pump is broken.", max_attempts=3)

        assert mock_call.call_count == 3
        assert exc_info.value.last_error is not None
        assert "LLM classification failed after 3 attempts" in caplog.text


def test_classify_with_llm_repair_prompt_reflects_most_recent_error():
    """Confirms the repair prompt sent on the final attempt reflects
    the error from the attempt immediately before it, not an
    accumulated history of all prior errors (per our earlier design
    decision)."""
    with patch(
        "app.services.llm_classifier.call_llm",
        side_effect=[MARKDOWN_WRAPPED_RESPONSE, MISSING_FIELD_RESPONSE, VALID_RESPONSE],
    ) as mock_call:
        classify_with_llm("Test message.", max_attempts=3)

        third_call_prompt = mock_call.call_args_list[2].args[0]
        assert (
            "field" in third_call_prompt.lower()
            or "missing" in third_call_prompt.lower()
        )


# ---------------------------------------------------------------------------
# STEP 13 — Live pipeline structural check (mocked)
# ---------------------------------------------------------------------------


def test_classify_with_llm_all_fields_non_null():
    """Structural assertion: classify_with_llm() must return a TriageResult
    with every required field populated (not None), regardless of content
    quality. This mocks call_llm so it runs free in CI — the real 5-message
    live check lives in scripts/step13_spot_check.py and is run manually."""
    from unittest.mock import patch

    from app.services.llm_classifier import classify_with_llm

    valid_response = json.dumps(
        {
            "category": "general_inquiry",
            "urgency": "low",
            "requested_action": "Provide office hours information",
            "entities": {},
            "missing_information": [],
            "draft_response": "Our office hours are Monday to Friday, 9 AM to 5 PM.",
            "confidence": 0.9,
        }
    )

    with patch("app.services.llm_classifier.call_llm", return_value=valid_response):
        result = classify_with_llm("What are the office hours for the registrar this week?")

    assert result.category is not None
    assert result.urgency is not None
    assert result.requested_action is not None
    assert result.entities is not None
    assert result.missing_information is not None
    assert result.draft_response is not None
    assert result.confidence is not None