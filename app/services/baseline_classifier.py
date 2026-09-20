"""
Loads the Step 8 baseline pipeline (data/models/baseline.joblib) and
exposes a single predict function returning category + confidence.

Loaded once at import time, not per-request — joblib.load() reads a
~450-message TF-IDF vocabulary + LogisticRegression weights from disk,
which is wasted work to repeat on every POST.
"""

from pathlib import Path

import joblib

from app.core.preprocessing import normalize_text

MODEL_PATH = Path("data/models/baseline.joblib")

_pipeline = None


def _get_pipeline():
    """Lazy singleton load, so importing this module doesn't fail at
    collection time (e.g. during test discovery) if the model hasn't
    been trained yet — the error only surfaces when a prediction is
    actually requested.
    """
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Baseline model not found at {MODEL_PATH}. "
                "Run `python scripts/train_baseline.py` first."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline


from app.models.schema import BaselinePrediction


def predict_baseline(raw_message: str) -> BaselinePrediction:
    """
    Normalize raw_message the same way Step 8 normalized training
    data, then predict category + confidence (max class probability).
    """
    pipeline = _get_pipeline()
    normalized = normalize_text(raw_message)

    category = pipeline.predict([normalized])[0]
    confidence = float(pipeline.predict_proba([normalized])[0].max())

    return BaselinePrediction(category=category, confidence=confidence)
