from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from app.core.routing import determine_review_status
from app.models.db import get_db
from app.models.db_models import Prediction
from app.models.db_models import Request as RequestRow
from app.models.schema import (
    BaselineClassification,
    ClassifyRequest,
    LLMClassificationResponse,
)
from app.services.baseline_classifier import predict_baseline
from app.services.llm_classifier import LLMClassificationFailed, classify_with_llm

app = FastAPI(title="CivicInbox", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _get_or_create_request(db: Session, request: ClassifyRequest) -> RequestRow:
    """
    Resolve the Request row a Prediction should attach to.

    If request.request_id is given, look it up and 404 if it doesn't
    exist -- the caller made a specific claim ("attach this to request
    N") and we don't silently fall back to creating a new one if that
    claim is wrong. If request_id is omitted, always create a new
    Request row (see Step 18 design decision: no auto-dedupe by text).
    """
    if request.request_id is not None:
        row = db.get(RequestRow, request.request_id)
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"request_id {request.request_id} not found",
            )
        return row

    row = RequestRow(message_text=request.message)
    db.add(row)
    db.flush()  # populates row.id without committing yet
    return row


@app.post("/classify/baseline", response_model=BaselineClassification)
def classify_baseline(
    request: ClassifyRequest, db: Session = Depends(get_db)
) -> BaselineClassification:
    try:
        result = predict_baseline(request.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    request_row = _get_or_create_request(db, request)

    prediction_row = Prediction(
        request_id=request_row.id,
        model_type="baseline",
        category=result.category,
        # Baseline has no urgency/action/draft_response -- it's a category
        # classifier only (see BaselineClassification docstring). Store
        # empty string rather than None: these columns are non-nullable
        # because LLM predictions always populate them, and splitting the
        # column into nullable-for-baseline/required-for-LLM isn't worth
        # the schema complexity for a Step 18 wiring pass.
        urgency="",
        requested_action="",
        draft_response="",
        confidence=result.confidence,
        attempts_used=None,
        status="auto_approved",  # baseline has no review-routing concept yet
    )
    db.add(prediction_row)
    db.commit()
    db.refresh(prediction_row)

    return BaselineClassification(
        category=result.category,
        confidence=result.confidence,
        request_id=request_row.id,
        prediction_id=prediction_row.id,
    )


@app.post("/classify/llm", response_model=LLMClassificationResponse)
def classify_llm(
    request: ClassifyRequest, db: Session = Depends(get_db)
) -> LLMClassificationResponse:
    try:
        outcome = classify_with_llm(request.message)
    except LLMClassificationFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    status = determine_review_status(
        outcome.result,
        llm_attempts_used=outcome.attempts_used,
    )

    request_row = _get_or_create_request(db, request)

    prediction_row = Prediction(
        request_id=request_row.id,
        model_type="llm",
        category=outcome.result.category,
        urgency=outcome.result.urgency,
        requested_action=outcome.result.requested_action,
        draft_response=outcome.result.draft_response,
        confidence=outcome.result.confidence,
        attempts_used=outcome.attempts_used,
        status=status,
    )
    db.add(prediction_row)
    db.commit()
    db.refresh(prediction_row)

    return LLMClassificationResponse(
        result=outcome.result,
        status=status,
        attempts_used=outcome.attempts_used,
        request_id=request_row.id,
        prediction_id=prediction_row.id,
    )
