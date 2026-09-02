from fastapi import FastAPI, HTTPException

from app.models.schema import (
    BaselineClassification,
    ClassifyRequest,
    LLMClassificationResponse,
)
from app.services.baseline_classifier import predict_baseline
from app.services.llm_classifier import classify_with_llm, LLMClassificationFailed
from app.core.routing import determine_review_status

app = FastAPI(title="CivicInbox", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify/baseline", response_model=BaselineClassification)
def classify_baseline(request: ClassifyRequest) -> BaselineClassification:
    try:
        return predict_baseline(request.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/classify/llm", response_model=LLMClassificationResponse)
def classify_llm(request: ClassifyRequest) -> LLMClassificationResponse:
    try:
        outcome = classify_with_llm(request.message)
    except LLMClassificationFailed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    status = determine_review_status(
        outcome.result,
        llm_attempts_used=outcome.attempts_used,
    )

    return LLMClassificationResponse(
        result=outcome.result,
        status=status,
        attempts_used=outcome.attempts_used,
    )