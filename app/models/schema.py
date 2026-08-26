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
        ..., min_length=1, description="What the requester wants done, in plain language."
    )
    entities: Entities
    missing_information: list[str] = Field(default_factory=list)
    draft_response: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)