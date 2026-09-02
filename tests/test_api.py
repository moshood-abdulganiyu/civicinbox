from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_baseline():
    """Requires data/models/baseline.joblib to already exist locally
    (run scripts/train_baseline.py first) — same precondition as
    test_baseline_eval_runs. Not yet handled in CI; flagged for Step 23.
    """
    response = client.post(
        "/classify/baseline",
        json={"message": "My scholarship application has not been reviewed yet."},
    )
    assert response.status_code == 200

    body = response.json()
    assert body["category"] in {
        "academic_records",
        "financial_aid",
        "registration",
        "document_request",
        "general_inquiry",
        "complaint_escalation",
    }
    assert 0.0 <= body["confidence"] <= 1.0


from app.models.schema import Entities, TriageResult
from app.services.llm_classifier import (
    LLMClassificationFailed,
    LLMClassificationOutcome,
)


def test_classify_llm(monkeypatch):
    fake_result = TriageResult(
        category="financial_aid",
        urgency="medium",
        requested_action="review application status",
        entities=Entities(),
        missing_information=[],
        draft_response="Thank you for reaching out...",
        confidence=0.9,
    )
    fake_outcome = LLMClassificationOutcome(result=fake_result, attempts_used=1)

    monkeypatch.setattr(
        "app.main.classify_with_llm",
        lambda message: fake_outcome,
    )

    response = client.post(
        "/classify/llm",
        json={"message": "My scholarship application has not been reviewed yet."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["attempts_used"] == 1
    assert body["status"] in ("auto_approved", "needs_review")
    assert body["result"]["category"] == "financial_aid"


def test_classify_llm_failure_returns_503(monkeypatch):
    def fake_raise(message):
        raise LLMClassificationFailed(
            "Failed after 3 attempts: ValidationError",
            last_error=None,
        )

    monkeypatch.setattr("app.main.classify_with_llm", fake_raise)

    response = client.post(
        "/classify/llm",
        json={"message": "My scholarship application has not been reviewed yet."},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail == "Failed after 3 attempts: ValidationError"
    assert "input_value" not in detail