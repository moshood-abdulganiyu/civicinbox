from app.core.db import SessionLocal

from app.models.db_models import Prediction, Request

db = SessionLocal()
req = Request(
    message_text="My scholarship application has not been reviewed since March."
)
db.add(req)
db.commit()
db.refresh(req)

db.add(
    Prediction(
        request_id=req.id,
        model_type="baseline",
        category="financial_aid",
        urgency="medium",
        requested_action="Check application status",
        draft_response="",
        confidence=0.71,
        status="auto_approved",
    )
)
db.add(
    Prediction(
        request_id=req.id,
        model_type="llm",
        category="financial_aid",
        urgency="high",
        requested_action="Escalate to financial aid office",
        draft_response="Thank you for reaching out...",
        confidence=0.92,
        attempts_used=1,
        status="needs_review",
    )
)
db.commit()
db.close()
print(f"Seeded request_id={req.id}")
