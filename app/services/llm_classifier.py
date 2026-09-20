import json
import logging

from openai import OpenAIError
from pydantic import BaseModel, ValidationError

from app.models.schema import TriageResult
from app.services.llm_client import call_llm

logger = logging.getLogger(__name__)


class LLMClassificationOutcome(BaseModel):
    result: TriageResult
    attempts_used: int


class LLMClassificationFailed(Exception):
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


def classify_with_llm(message: str, max_attempts: int = 3) -> LLMClassificationOutcome:
    prompt = build_prompt(message)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            raw = call_llm(prompt)
        except RuntimeError as e:
            # Missing/misconfigured API key -- a local config problem, not
            # a transient failure. Retrying won't fix it and would only
            # waste attempts (and, once a key IS present, real API cost)
            # on a call that's guaranteed to fail every time. Fail fast.
            logger.error(f"LLM client misconfigured, not retrying: {e}")
            raise LLMClassificationFailed(
                f"LLM client configuration error: {e}", last_error=e
            ) from e
        except OpenAIError as e:
            # Transient (or at least retry-worth-trying) failures: network
            # timeout, connection error, rate limit, a 5xx from OpenAI's
            # side, etc. Worth spending one of our max_attempts on.
            last_error = e
            logger.warning(f"LLM classify attempt {attempt}/{max_attempts} — API error: {e}")
            continue

        try:
            data = json.loads(raw)
            result = TriageResult.model_validate(data)
            return LLMClassificationOutcome(result=result, attempts_used=attempt)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            logger.warning(f"LLM classify attempt {attempt}/{max_attempts} — invalid output: {e}")
            prompt = build_repair_prompt(message, raw, e)

    logger.error(
        f"LLM classification failed after {max_attempts} attempts: {last_error}"
    )
    raise LLMClassificationFailed(
        f"Failed after {max_attempts} attempts: {type(last_error).__name__}",
        last_error=last_error,
    )