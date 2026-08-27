from fastapi import FastAPI, HTTPException

from app.models.schema import BaselineClassification, ClassifyRequest
from app.services.baseline_classifier import predict_baseline

app = FastAPI(title="CivicInbox", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify/baseline", response_model=BaselineClassification)
def classify_baseline(request: ClassifyRequest) -> BaselineClassification:
    try:
        return predict_baseline(request.message)
    except FileNotFoundError as exc:
        # Model artifact missing (not trained yet, or not present on this
        # deploy target) — a 503 tells the caller "service unavailable,
        # try again later," not "your request was wrong" (4xx) or "we
        # crashed" (bare 500). Real failure-path hardening for other
        # cases (bad input, DB errors) is Step 21, not this step.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
