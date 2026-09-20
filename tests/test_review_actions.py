"""Tests for review_actions.py -- approve/edit/reject write correct DB state."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.db_models import Base, Prediction, Request, ReviewerCorrection
from app.services.review_actions import (
    approve_prediction,
    edit_prediction,
    reject_prediction,
)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def make_prediction(db_session, **overrides):
    request = Request(message_text="test message")
    db_session.add(request)
    db_session.commit()
    db_session.refresh(request)

    defaults = dict(
        request_id=request.id,
        model_type="llm",
        category="financial_aid",
        urgency="medium",
        requested_action="check status",
        draft_response="draft",
        status="needs_review",
    )
    defaults.update(overrides)
    prediction = Prediction(**defaults)
    db_session.add(prediction)
    db_session.commit()
    db_session.refresh(prediction)
    return prediction


def test_approve_sets_status_and_writes_correction(db_session):
    pred = make_prediction(db_session)
    approve_prediction(db_session, pred.id)

    db_session.refresh(pred)
    assert pred.status == "reviewer_approved"

    corrections = (
        db_session.query(ReviewerCorrection).filter_by(prediction_id=pred.id).all()
    )
    assert len(corrections) == 1
    assert corrections[0].corrected_category is None


def test_edit_writes_only_changed_fields(db_session):
    pred = make_prediction(db_session)
    edit_prediction(db_session, pred.id, corrected_category="infrastructure")

    db_session.refresh(pred)
    assert pred.status == "reviewer_approved"

    correction = (
        db_session.query(ReviewerCorrection).filter_by(prediction_id=pred.id).one()
    )
    assert correction.corrected_category == "infrastructure"
    assert correction.corrected_urgency is None


def test_edit_with_no_fields_raises(db_session):
    pred = make_prediction(db_session)
    with pytest.raises(ValueError):
        edit_prediction(db_session, pred.id)


def test_reject_sets_status(db_session):
    pred = make_prediction(db_session)
    reject_prediction(db_session, pred.id, reviewer_notes="Wrong category")

    db_session.refresh(pred)
    assert pred.status == "rejected"

    correction = (
        db_session.query(ReviewerCorrection).filter_by(prediction_id=pred.id).one()
    )
    assert correction.reviewer_notes == "Wrong category"


def test_action_on_nonexistent_prediction_raises(db_session):
    with pytest.raises(ValueError):
        approve_prediction(db_session, 9999)