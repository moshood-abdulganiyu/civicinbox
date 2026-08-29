# app/core/preprocessing.py
"""
Text preprocessing for CivicInbox.

Scope note (intentional, not an oversight):
This module handles TEXT NORMALIZATION only — case folding and
whitespace cleanup so both the TF-IDF baseline and the LLM pipeline
see consistent input. It does NOT scrub or redact PII (names, student
IDs, emails, phone numbers). Raw messages may contain PII and it is
preserved as-is through this function. PII handling, if added later,
should be a separate, explicit step — not silently bundled into
normalization.
"""

import re


def normalize_text(text: str) -> str:
    """
    Normalize a raw message for downstream classification.

    Steps:
    1. Strip leading/trailing whitespace.
    2. Collapse any run of whitespace (spaces, tabs, newlines) to a
       single space — internal structure (paragraph breaks etc.) is
       not meaningful for classification, so we don't preserve it.
    3. Lowercase everything, so TF-IDF and prompt matching don't treat
       "Scholarship" and "scholarship" as different tokens.
    4. Strip repeated trailing punctuation (e.g. "!!" -> "!", "??" -> "?")
       down to a single mark, since urgency-signaling punctuation
       ("!!!", "???") is common in this dataset and we want it to
       normalize to one consistent form rather than being either lost
       or treated as N distinct tokens.

    This is intentionally lossy — we are not trying to preserve
    formatting, only to produce a consistent string for a classifier
    to consume.
    """
    if not text:
        return ""

    normalized = text.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.lower()
    normalized = re.sub(r"([!?.,])\1+", r"\1", normalized)

    return normalized
