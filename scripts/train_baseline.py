"""
Train the CivicInbox baseline classifier: TF-IDF + Logistic Regression.

Why this baseline (Step 8 of civicinbox-steps.md):
- Interpretable: you can inspect which words drive each category.
- Fast and zero API cost: no network calls, trains in seconds on 450 rows.
- Sets the bar the LLM-based classifier (Steps 11-13) must beat.

Usage:
    python scripts/train_baseline.py

Output:
    data/models/baseline.joblib   (a fitted sklearn Pipeline: vectorizer + classifier)
"""

import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Make app/ importable when running this script directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.core.preprocessing import normalize_text

DATA_PATH = Path("data/labeled_messages.jsonl")
MODEL_DIR = Path("data/models")
MODEL_PATH = MODEL_DIR / "baseline.joblib"

RANDOM_STATE = 42  # fixed so Step 9's eval script can reuse the exact same split


def load_dataset(path: Path) -> tuple[list[str], list[str]]:
    """Load messages and labels from the labeled_messages.jsonl file."""
    texts, labels = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            labels.append(row["category"])
    return texts, labels


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Expected labeled dataset at {DATA_PATH}. "
            "Run scripts/validate_dataset.py first if unsure it's in place."
        )

    print(f"Loading dataset from {DATA_PATH} ...")
    texts, labels = load_dataset(DATA_PATH)
    print(f"Loaded {len(texts)} messages across {len(set(labels))} categories.")

    # Normalize BEFORE vectorizing — TF-IDF is case/punctuation sensitive,
    # and this is the exact reason Step 7's normalize_text() exists.
    normalized_texts = [normalize_text(t) for t in texts]

    # Stratified split: preserves class balance in both train and test sets.
    # With 75/category (450 total), 20% test = 15/category held out.
    X_train, X_test, y_train, y_test = train_test_split(
        normalized_texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)} (held out, untouched)")

    # Pipeline bundles vectorizer + classifier into one object so Step 10's
    # FastAPI endpoint can load ONE file and call .predict() directly —
    # no risk of a vectorizer/classifier mismatch.
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),  # unigrams + bigrams: captures phrases
                    # like "not working", not just single words
                    min_df=2,  # ignore words appearing in only 1 message
                    # (likely noise, not a real signal at this size)
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,  # TF-IDF vectors can need more iterations
                    # to converge than sklearn's default (100)
                    class_weight="balanced",  # protects against any residual
                    # category imbalance after your 450-row expansion
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("Fitting pipeline on training set...")
    pipeline.fit(X_train, y_train)

    # Smoke test: predict on ONE held-out example, per Step 8's test criterion.
    # This is NOT the full evaluation (that's Step 9) — just a sanity check
    # that the saved model can make a valid prediction at all.
    sample_text, sample_true_label = X_test[0], y_test[0]
    sample_pred = pipeline.predict([sample_text])[0]
    print("\nSmoke test — held-out example:")
    print(f"  True category:      {sample_true_label}")
    print(f"  Predicted category: {sample_pred}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved fitted pipeline to {MODEL_PATH}")

    # Also persist the exact test split so Step 9 evaluates on the SAME
    # held-out rows this script trained without seeing — not a fresh split.
    split_path = MODEL_DIR / "test_split.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump({"X_test": X_test, "y_test": y_test}, f, indent=2)
    print(f"Saved held-out test split to {split_path} (for Step 9's eval script)")


if __name__ == "__main__":
    main()
