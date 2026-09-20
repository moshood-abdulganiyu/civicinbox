import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models.db import get_db
from app.models.db_models import Base, Prediction, Request

client = TestClient(app)


@pytest.fixture(autouse=True)
def isolated_db():
    """Every test in this module gets its own throwaway in-memory
    database via dependency override — none of them should ever
    touch the real civicinbox.db file. Autouse because Step 18 wired
    persistence into both classify endpoints, so any endpoint test
    now exercises the DB whether or not that's the thing under test.
    """

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(engine)
    TestSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestSessionLocal
    app.dependency_overrides.clear()
    engine.dispose()


def test_empty_message_baseline_returns_422():
    response = client.post("/classify/baseline", json={"message": ""})
    assert response.status_code == 422


def test_empty_message_llm_returns_422():
    response = client.post("/classify/llm", json={"message": ""})
    assert response.status_code == 422


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_classify_baseline():
    """Requires data/models/baseline.joblib to already exist locally
    (run scripts/train_baseline.py first)
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


from app.main import app


def test_classify_llm_persists_to_db(monkeypatch, isolated_db):
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
    monkeypatch.setattr("app.main.classify_with_llm", lambda message: fake_outcome)

    response = client.post(
        "/classify/llm",
        json={"message": "My scholarship application has not been reviewed yet."},
    )
    assert response.status_code == 200
    body = response.json()

    request_id = body["request_id"]
    prediction_id = body["prediction_id"]
    assert request_id is not None
    assert prediction_id is not None

    check_session = isolated_db()
    try:
        db_request = check_session.get(Request, request_id)
        db_prediction = check_session.get(Prediction, prediction_id)

        assert db_request is not None
        assert db_request.message_text == (
            "My scholarship application has not been reviewed yet."
        )
        assert db_prediction is not None
        assert db_prediction.request_id == request_id
        assert db_prediction.model_type == "llm"
        assert db_prediction.category == "financial_aid"
        assert db_prediction.urgency == "medium"
        assert db_prediction.draft_response == "Thank you for reaching out..."
        assert db_prediction.confidence == 0.9
        assert db_prediction.attempts_used == 1
        assert db_prediction.status == body["status"]
    finally:
        check_session.close()
