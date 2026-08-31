from typing import Literal

from app.models.schema import TriageResult  

# Placeholder pending Step 16's full-dataset baseline-vs-LLM comparison.
# Spot check (n=5) showed LLM confidence sitting in 0.85-0.95 regardless of
# correctness, so this threshold is set low deliberately -- it should almost
# never fire on its own right now. The retry-count check below is doing the
# real work until Step 16 gives us a real distribution to calibrate against.
LLM_CONFIDENCE_THRESHOLD_PLACEHOLDER = 0.5


def determine_review_status(
    result: TriageResult,
    llm_attempts_used: int,
    confidence_threshold: float = LLM_CONFIDENCE_THRESHOLD_PLACEHOLDER,
) -> Literal["auto_approved", "needs_review"]:
    """Decide whether an LLM classification can be auto-approved.

    Routes to needs_review if the LLM needed more than one attempt to
    produce valid output (a real, observed quality signal from Step 13),
    or if confidence falls below the placeholder threshold.
    """
    if llm_attempts_used > 1:
        return "needs_review"
    if result.confidence < confidence_threshold:
        return "needs_review"
    return "auto_approved"