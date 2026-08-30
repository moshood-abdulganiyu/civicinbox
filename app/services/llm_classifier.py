import json
import logging

from pydantic import ValidationError

from app.models.schema import TriageResult
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)


class LLMClassificationFailed(Exception):
    """Raised when the LLM fails to produce a schema-valid TriageResult
    after all retry attempts are exhausted."""

    def __init__(self, message: str, last_error: Exception | None = None):
        super().__init__(message)
        self.last_error = last_error


SCHEMA_INSTRUCTIONS = """Return ONLY a JSON object with exactly these fields, no prose, no markdown fences:
{
  "category": string, must be exactly one of: "academic_records", "financial_aid", "registration", "document_request", "general_inquiry", "complaint_escalation",
  "urgency": string, must be exactly one of: "low", "medium", "high",
  "requested_action": string,
  "entities": object,
  "missing_information": array of strings,
  "draft_response": string,
  "confidence": float between 0 and 1
}

Category selection rule: if the message explicitly asks to escalate, file a complaint, or expresses frustration about an unresolved delay, classify as "complaint_escalation" even if the underlying subject is a document, registration, or academic matter."""

def build_prompt(message: str) -> str:
    return f"""Classify the following message from a university student affairs/registrar inbox.

Message: \"\"\"{message}\"\"\"

{SCHEMA_INSTRUCTIONS}"""


def build_repair_prompt(message: str, raw_output: str, error: Exception) -> str:
    return f"""Your previous response could not be used: {error}

Your previous output was:
\"\"\"{raw_output[:500]}\"\"\"

Classify the following message again from a university student affairs/registrar inbox.

Message: \"\"\"{message}\"\"\"

{SCHEMA_INSTRUCTIONS}"""


def classify_with_llm(message: str, max_attempts: int = 3) -> TriageResult:
    prompt = build_prompt(message)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        raw = call_llm(prompt)
        try:
            data = json.loads(raw)
            return TriageResult.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning(f"LLM classify attempt {attempt}/{max_attempts} failed: {e}")
            prompt = build_repair_prompt(message, raw, e)

    logger.error(
        f"LLM classification failed after {max_attempts} attempts: {last_error}"
    )
    raise LLMClassificationFailed(
        f"Failed after {max_attempts} attempts: {last_error}",
        last_error=last_error,
    )
