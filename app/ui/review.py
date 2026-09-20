"""
Read-only Streamlit review screen for CivicInbox.

Pulls every Request and its Predictions (baseline and/or LLM) and
renders them side by side so a reviewer can compare the two models'
output for the same inbound message.

Read-only: Step 2 of the finish plan (old Step 20) adds the
approve/edit/reject actions that write to reviewer_corrections. This
screen only queries and displays.
"""

from collections import defaultdict

import streamlit as st

from app.models.db import SessionLocal
from app.models.db_models import Prediction, Request
from app.services.review_actions import (
    approve_prediction,
    edit_prediction,
    reject_prediction,
)

st.set_page_config(page_title="CivicInbox Review", layout="wide")
st.title("CivicInbox — Review Queue")

STATUS_ICONS = {
    "auto_approved": "🟢",
    "needs_review": "🟡",
    "reviewer_approved": "✅",
    "rejected": "🔴",
}


def handle_approve(prediction_id: int) -> None:
    db = SessionLocal()
    try:
        approve_prediction(db, prediction_id)
    finally:
        db.close()


def handle_reject(prediction_id: int) -> None:
    db = SessionLocal()
    try:
        reject_prediction(db, prediction_id)
    finally:
        db.close()


def handle_edit(prediction_id: int, **fields) -> None:
    db = SessionLocal()
    try:
        edit_prediction(db, prediction_id, **fields)
    finally:
        db.close()


def load_requests_with_predictions() -> list[
    tuple[Request, dict[str, Prediction | None]]
]:
    """
    Two flat queries + a Python groupby, deliberately not a JOIN.

    A JOIN between requests and predictions would return one row per
    (request, prediction) pair -- so a request with both a baseline and
    an LLM prediction comes back as two duplicated rows, and you'd have
    to de-duplicate the Request side in Python anyway before rendering.
    Fetching each table separately and grouping by request_id in Python
    is simpler to reason about, and cheap at this data scale.
    """
    db = SessionLocal()
    try:
        requests = db.query(Request).order_by(Request.created_at.desc()).all()
        request_ids = [r.id for r in requests]

        predictions = (
            db.query(Prediction).filter(Prediction.request_id.in_(request_ids)).all()
        )

        # Keyed by model_type ("baseline"/"llm") rather than a plain list,
        # since there are only ever these two possible predictions per
        # request -- this makes the side-by-side render a direct dict
        # lookup instead of a scan.
        by_request: dict[int, dict[str, Prediction | None]] = defaultdict(
            lambda: {"baseline": None, "llm": None}
        )
        for p in predictions:
            by_request[p.request_id][p.model_type] = p

        return [(r, by_request[r.id]) for r in requests]
    finally:
        db.close()


def render_prediction_column(label: str, pred: Prediction | None) -> None:
    st.markdown(f"**{label}**")
    if pred is None:
        st.caption("No prediction yet.")
        return

    st.write(f"Category: `{pred.category}`")
    st.write(f"Urgency: `{pred.urgency}`")
    st.write(f"Requested action: {pred.requested_action}")
    st.write(
        f"Confidence: {pred.confidence:.2f}"
        if pred.confidence is not None
        else "Confidence: n/a"
    )

    icon = STATUS_ICONS.get(pred.status, "⚪")
    st.write(f"Status: {icon} `{pred.status}`")

    st.text_area(
        "Draft response",
        value=pred.draft_response,
        height=120,
        disabled=True,
        key=f"draft_{pred.id}",
    )

    if pred.status in ("reviewer_approved", "rejected"):
        return  # already finalized -- no further action

    edit_key = f"editing_{pred.id}"
    st.session_state.setdefault(edit_key, False)

    col_a, col_e, col_r = st.columns(3)
    with col_a:
        if st.button("Approve", key=f"approve_btn_{pred.id}"):
            handle_approve(pred.id)
            st.rerun()
    with col_e:
        if st.button("Edit", key=f"edit_btn_{pred.id}"):
            st.session_state[edit_key] = True
    with col_r:
        if st.button("Reject", key=f"reject_btn_{pred.id}"):
            handle_reject(pred.id)
            st.rerun()

    if st.session_state[edit_key]:
        with st.form(key=f"edit_form_{pred.id}"):
            new_category = st.text_input("Corrected category")
            new_urgency = st.text_input("Corrected urgency")
            new_action = st.text_input("Corrected requested action")
            notes = st.text_area("Reviewer notes")
            if st.form_submit_button("Save correction"):
                handle_edit(
                    pred.id,
                    corrected_category=new_category or None,
                    corrected_urgency=new_urgency or None,
                    corrected_action=new_action or None,
                    reviewer_notes=notes or None,
                )
                st.session_state[edit_key] = False
                st.rerun()


rows = load_requests_with_predictions()

if not rows:
    st.info("No requests in the database yet.")
else:
    for request, preds in rows:
        with st.container(border=True):
            st.subheader(f"Request #{request.id}")
            st.caption(request.created_at.isoformat())
            st.write(request.message_text)

            col_baseline, col_llm = st.columns(2)
            with col_baseline:
                render_prediction_column("Baseline", preds["baseline"])
            with col_llm:
                render_prediction_column("LLM", preds["llm"])
