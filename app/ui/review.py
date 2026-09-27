"""
Read-only Streamlit review screen for CivicInbox.

Pulls every Request and its Predictions (baseline and/or LLM) and
renders them as glass cards in a 3-column grid, so a reviewer can scan
several requests at once and scroll for more.

Styling/layout: glass-card panels (blur + transparency), color-coded
review actions (green/yellow/red), 3-per-row grid, compact message
preview with a "show full" expander to keep card heights consistent --
added in the demo-polish pass. Functionally unchanged from the
original approve/edit/reject screen: reviewer_actions.py still owns
all the write logic; this file only changed presentation.
"""

from collections import defaultdict

import streamlit as st

DATABASE_URL_KEY = "DATABASE_URL"
if DATABASE_URL_KEY in st.secrets:
    import os
    os.environ[DATABASE_URL_KEY] = st.secrets[DATABASE_URL_KEY]

from app.models.db import SessionLocal
from app.models.db_models import Prediction, Request
from app.services.review_actions import (
    approve_prediction,
    edit_prediction,
    reject_prediction,
)

st.set_page_config(page_title="CivicInbox Review", layout="wide")

GRID_COLUMNS = 3
MESSAGE_PREVIEW_CHARS = 110

STATUS_ICONS = {
    "auto_approved": "🟢",
    "needs_review": "🟡",
    "reviewer_approved": "✅",
    "rejected": "🔴",
}

# ---------------------------------------------------------------------------
# Styling
#
# Streamlit's own DOM structure (data-testid names, the "st-key-<key>"
# class it adds to a widget's wrapper when you pass key=...) is an
# internal implementation detail that can shift between versions. If
# the button colors or card glass effect don't render after upgrading
# Streamlit, use your browser's inspector on a button/container to find
# the current selector and adjust the block below -- the logic (approve/
# edit/reject) is untouched either way, only presentation would break.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 50%, #1e1b4b 100%);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        padding: 0.75rem !important;
    }

    .model-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.4rem 0 0.3rem 0;
    }
    .model-badge .icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 1.4rem;
        height: 1.4rem;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.25);
        font-size: 0.8rem;
    }
    .model-divider {
        border: none;
        border-top: 1px solid rgba(255, 255, 255, 0.15);
        margin: 0.5rem 0;
    }
    .field-line {
        font-size: 0.85rem;
        margin: 0.1rem 0;
    }

    div[class*="st-key-approve_btn_"] button {
        background-color: #22c55e !important;
        color: #ffffff !important;
        border: none !important;
    }
    div[class*="st-key-edit_btn_"] button {
        background-color: #eab308 !important;
        color: #1f2937 !important;
        border: none !important;
    }
    div[class*="st-key-reject_btn_"] button {
        background-color: #ef4444 !important;
        color: #ffffff !important;
        border: none !important;
    }
    div[class*="st-key-approve_btn_"] button:hover,
    div[class*="st-key-edit_btn_"] button:hover,
    div[class*="st-key-reject_btn_"] button:hover {
        filter: brightness(1.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("CivicInbox — Review Queue")


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

        by_request: dict[int, dict[str, Prediction | None]] = defaultdict(
            lambda: {"baseline": None, "llm": None}
        )
        for p in predictions:
            by_request[p.request_id][p.model_type] = p

        return [(r, by_request[r.id]) for r in requests]
    finally:
        db.close()


def render_prediction_block(label: str, icon: str, pred: Prediction | None) -> None:
    st.markdown(
        f'<div class="model-badge"><span class="icon">{icon}</span>{label}</div>',
        unsafe_allow_html=True,
    )
    if pred is None:
        st.caption("No prediction yet.")
        return

    st.markdown(f"<div class='field-line'>Category: <code>{pred.category}</code></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field-line'>Urgency: <code>{pred.urgency}</code></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='field-line'>Action: {pred.requested_action}</div>", unsafe_allow_html=True)
    confidence_text = f"{pred.confidence:.2f}" if pred.confidence is not None else "n/a"
    st.markdown(f"<div class='field-line'>Confidence: {confidence_text}</div>", unsafe_allow_html=True)

    status_icon = STATUS_ICONS.get(pred.status, "⚪")
    st.markdown(
        f"<div class='field-line'>Status: {status_icon} <code>{pred.status}</code></div>",
        unsafe_allow_html=True,
    )

    st.text_area(
        "Draft response",
        value=pred.draft_response,
        height=90,
        disabled=True,
        key=f"draft_{pred.id}",
        label_visibility="collapsed",
    )

    if pred.status in ("reviewer_approved", "rejected"):
        return  # already finalized -- no further action

    edit_key = f"editing_{pred.id}"
    st.session_state.setdefault(edit_key, False)

    col_a, col_e, col_r = st.columns(3)
    with col_a:
        if st.button("Approve", key=f"approve_btn_{pred.id}", use_container_width=True):
            handle_approve(pred.id)
            st.rerun()
    with col_e:
        if st.button("Edit", key=f"edit_btn_{pred.id}", use_container_width=True):
            st.session_state[edit_key] = True
    with col_r:
        if st.button("Reject", key=f"reject_btn_{pred.id}", use_container_width=True):
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


def render_request_card(request: Request, preds: dict[str, Prediction | None]) -> None:
    with st.container(border=True):
        st.markdown(f"**Request #{request.id}**")
        st.caption(request.created_at.isoformat())

        message = request.message_text
        if len(message) > MESSAGE_PREVIEW_CHARS:
            st.write(message[:MESSAGE_PREVIEW_CHARS].rstrip() + "…")
            with st.expander("Show full message"):
                st.write(message)
        else:
            st.write(message)

        render_prediction_block("Baseline", "📊", preds["baseline"])
        st.markdown("<hr class='model-divider'>", unsafe_allow_html=True)
        render_prediction_block("LLM", "🤖", preds["llm"])


rows = load_requests_with_predictions()

if not rows:
    st.info("No requests in the database yet.")
else:
    st.caption(f"{len(rows)} total requests")

    for row_start in range(0, len(rows), GRID_COLUMNS):
        chunk = rows[row_start : row_start + GRID_COLUMNS]
        grid_cols = st.columns(GRID_COLUMNS)
        for col, (request, preds) in zip(grid_cols, chunk):
            with col:
                render_request_card(request, preds)