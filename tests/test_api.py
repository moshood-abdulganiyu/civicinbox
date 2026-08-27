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
