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


class BaselineClassification(BaseModel):
    """Output of the TF-IDF + LogisticRegression baseline.

    Deliberately narrower than TriageResult: the baseline is a category
    classifier, not an extractor. It has no way to produce urgency,
    entities, requested_action, missing_information, or a draft_response
    — those require language generation, which is what the LLM endpoint
    (Step 15) exists for. Forcing baseline output into TriageResult with
    null placeholders would misrepresent what this model actually does.
    """

    category: Category
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Max class probability from predict_proba()."
    )


# ------------------------------
# STEP 15 
# ------------------------------

from typing import Literal

ReviewStatus = Literal["auto_approved", "needs_review"]

class LLMClassificationResponse(BaseModel):
    result: TriageResult
    status: ReviewStatus
    attempts_used: int