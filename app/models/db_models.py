"""
Database table definitions.

Three tables, no relationship() objects yet (see Step 17 design notes) --
joins between them are written explicitly at the query site (Step 18+)
rather than hidden behind ORM lazy-loading, so it stays visible what
each query is actually doing.

Foreign keys:
    predictions.request_id          -> requests.id
    reviewer_corrections.prediction_id -> predictions.id
"""

from datetime import UTC, datetime

from sqlalchemy import ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Request(Base):
    """One row per inbound message submitted to the system."""

    __tablename__ = "requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_text: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class Prediction(Base):
    """
    One row per classification run against a request.

    model_type distinguishes baseline vs LLM runs -- a single request
    can have two Prediction rows (one per model) during comparison work,
    which is why this isn't just columns bolted onto Request.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True)
    request_id: Mapped[int] = mapped_column(ForeignKey("requests.id"))

    model_type: Mapped[str]  # "baseline" or "llm"

    category: Mapped[str]
    urgency: Mapped[str]
    requested_action: Mapped[str]
    draft_response: Mapped[str]

    # Nullable: baseline may not populate confidence the same way LLM does.
    confidence: Mapped[float | None] = mapped_column(default=None)

    # Nullable: only meaningful for the LLM path (LLMClassificationOutcome).
    attempts_used: Mapped[int | None] = mapped_column(default=None)

    # Output of determine_review_status(): "auto_approved" or "needs_review"
    status: Mapped[str]

    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))


class ReviewerCorrection(Base):
    """
    One row per human edit to a prediction.

    Kept separate from Prediction (rather than mutating it in place) so
    the original model output survives for eval purposes even after a
    human corrects it -- this is the Step 30 active-learning dataset.
    """

    __tablename__ = "reviewer_corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))

    # Nullable individually: a reviewer may only correct one field.
    corrected_category: Mapped[str | None] = mapped_column(default=None)
    corrected_urgency: Mapped[str | None] = mapped_column(default=None)
    corrected_action: Mapped[str | None] = mapped_column(default=None)

    reviewer_notes: Mapped[str | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))