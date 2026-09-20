"""
Review-action functions: approve, edit, reject a Prediction.

Every action does two things in one transaction:
  1. Writes a ReviewerCorrection row -- the audit trail. Even an
     "approve as-is" writes one (all corrected_* fields left None),
     so "approved" and "never looked at" are distinguishable in the DB.
  2. Updates Prediction.status -- the single source of truth for
     "what state is this prediction in right now", so a review-queue
     count is a plain `WHERE status = ...` query, not a join against
     reviewer_corrections.

Prediction's own fields (category, urgency, etc.) are never mutated
here -- the original model output must survive a correction for the
Step 12 active-learning dataset (see ReviewerCorrection's docstring
in db_models.py).
"""

from sqlalchemy.orm import Session

from app.models.db_models import Prediction, ReviewerCorrection


def approve_prediction(db: Session, prediction_id: int) -> Prediction:
    """Reviewer agrees with the prediction as-is."""
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise ValueError(f"No prediction with id={prediction_id}")

    db.add(ReviewerCorrection(
        prediction_id=prediction_id,
        reviewer_notes="Approved as-is.",
    ))
    prediction.status = "reviewer_approved"
    db.commit()
    db.refresh(prediction)
    return prediction


def edit_prediction(
    db: Session,
    prediction_id: int,
    *,
    corrected_category: str | None = None,
    corrected_urgency: str | None = None,
    corrected_action: str | None = None,
    reviewer_notes: str | None = None,
) -> Prediction:
    """Reviewer changes one or more fields."""
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise ValueError(f"No prediction with id={prediction_id}")

    if not any([corrected_category, corrected_urgency, corrected_action]):
        raise ValueError("edit_prediction called with no corrected fields")

    db.add(ReviewerCorrection(
        prediction_id=prediction_id,
        corrected_category=corrected_category,
        corrected_urgency=corrected_urgency,
        corrected_action=corrected_action,
        reviewer_notes=reviewer_notes,
    ))
    prediction.status = "reviewer_approved"
    db.commit()
    db.refresh(prediction)
    return prediction


def reject_prediction(
    db: Session, prediction_id: int, reviewer_notes: str | None = None
) -> Prediction:
    """Reviewer rejects the prediction outright."""
    prediction = db.get(Prediction, prediction_id)
    if prediction is None:
        raise ValueError(f"No prediction with id={prediction_id}")

    db.add(ReviewerCorrection(
        prediction_id=prediction_id,
        reviewer_notes=reviewer_notes or "Rejected.",
    ))
    prediction.status = "rejected"
    db.commit()
    db.refresh(prediction)
    return prediction