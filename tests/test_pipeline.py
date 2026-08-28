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
