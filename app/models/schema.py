from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "academic_records",
    "financial_aid",
    "registration",
    "document_request",
    "general_inquiry",
    "complaint_escalation",
]

Urgency = Literal["low", "medium", "high"]


class Entities(BaseModel):
    """Contact/identifying info extracted from the message.
    All fields optional: absence of a field is itself signal,
    captured separately in `missing_information`.
    """

    student_name: str | None = None
    student_id: str | None = None
    email: str | None = None
    phone: str | None = None
    date_mentioned: str | None = None


class TriageResult(BaseModel):
    category: Category
    urgency: Urgency
    requested_action: str = Field(
        ...,
        min_length=1,
        description="What the requester wants done, in plain language.",
    )
    entities: Entities
    missing_information: list[str] = Field(default_factory=list)
    draft_response: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

# ------------------------------
# STEP 10
# ------------------------------


class ClassifyRequest(BaseModel):
    """Input for both /classify/baseline and /classify/llm."""

    message: str = Field(
        ..., min_length=1, description="Raw, unprocessed incoming message text."
    )
    request_id: int | None = Field(
        default=None,
        description="Reuse an existing Request row (e.g. to attach a second "
        "model's prediction to the same submission). Omit to create a new Request.",
    )


class BaselinePrediction(BaseModel):
    """Raw output of the TF-IDF + LogisticRegression baseline model itself.

    Deliberately narrower than TriageResult: the baseline is a category
    classifier, not an extractor. It has no way to produce urgency,
    entities, requested_action, missing_information, or a draft_response
    — those require language generation, which is what the LLM endpoint
    (Step 15) exists for.

    This type is DB-agnostic and API-agnostic on purpose — it's what
    predict_baseline() returns, and it's also used directly by the
    Step 9 eval script and Step 16 comparison script, neither of which
    touches the database. See BaselineClassification for the API
    response shape, which wraps this with request_id/prediction_id
    after a DB write (Step 18).
    """

    category: Category
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Max class probability from predict_proba()."
    )


class BaselineClassification(BaselinePrediction):
    """API response shape for POST /classify/baseline.

    Extends BaselinePrediction with the DB identifiers created during
    that endpoint's Prediction insert (Step 18). Never constructed
    outside app/main.py — anything computing a bare category+confidence
    prediction (eval scripts, the baseline classifier itself) should
    use BaselinePrediction instead.
    """

    request_id: int
    prediction_id: int
    
from typing import Literal

ReviewStatus = Literal["auto_approved", "needs_review"]


class LLMClassificationResponse(BaseModel):
    result: TriageResult
    status: ReviewStatus
    attempts_used: int
    request_id: int
    prediction_id: int