from typing import Literal, Optional

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
    student_name: Optional[str] = None
    student_id: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date_mentioned: Optional[str] = None


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