"""
Step 16 — Baseline vs LLM comparison.

Reuses the EXACT test split from scripts/train_baseline.py (same RANDOM_STATE=42,
test_size=0.2, stratify=labels) so both models are evaluated on the same held-out
messages. train_test_split's index assignment depends only on array length, order,
and stratify labels — not on the actual text content — so recreating it from
(indices, labels) with identical params reproduces the identical split.

Usage:
    python evals/run_comparison.py --sample 5      # smoke test, no real cost concern
    python evals/run_comparison.py                 # full test split, real API calls
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.routing import determine_review_status
from app.services.baseline_classifier import predict_baseline
from app.services.llm_classifier import LLMClassificationFailed, classify_with_llm

RANDOM_STATE = 42  # must match scripts/train_baseline.py exactly
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled_messages.jsonl"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "evals" / "results"


def load_records() -> list[dict]:
    with open(DATA_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_test_split(records: list[dict]) -> list[dict]:
    """Recreate Step 9's held-out test set by index, not by content."""
    labels = [r["category"] for r in records]
    indices = list(range(len(records)))
    _, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    return [records[i] for i in test_idx]


def run_baseline(test_records: list[dict]) -> dict:
    y_true, y_pred = [], []
    for r in test_records:
        pred = predict_baseline(r["text"])
        y_true.append(r["category"])
        y_pred.append(pred.category)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "n_test_examples": len(test_records),
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "per_class": {
            k: v for k, v in report.items() if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }


def run_llm(test_records: list[dict], sample: int | None) -> dict:
    records = test_records[:sample] if sample else test_records
    y_true, y_pred = [], []
    n_failed = 0
    n_needs_review = 0
    n_auto_approved = 0

    for i, r in enumerate(records, start=1):
        print(f"  [{i}/{len(records)}] {r['id']}...", flush=True)
        try:
            outcome = classify_with_llm(r["text"])
        except LLMClassificationFailed:
            n_failed += 1
            n_needs_review += 1  # a failure is a de facto review case, not auto-approvable
            continue

        status = determine_review_status(outcome.result, outcome.attempts_used)
        if status == "needs_review":
            n_needs_review += 1
        else:
            n_auto_approved += 1

        y_true.append(r["category"])
        y_pred.append(outcome.result.category)

    total = len(records)
    report = (
        classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        if y_true
        else {"accuracy": None, "macro avg": {"f1-score": None}}
    )

    return {
        "n_test_examples": total,
        "n_scored": len(y_true),
        "n_invalid_json_failures": n_failed,
        "invalid_json_rate": n_failed / total if total else None,
        "n_needs_review": n_needs_review,
        "n_auto_approved": n_auto_approved,
        "pct_routed_to_review": (n_needs_review / total * 100) if total else None,
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report.get("accuracy"),
        "per_class": {
            k: v
            for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit LLM run to first N test messages (smoke test before full spend). "
        "Baseline always runs on the full test split (no API cost).",
    )
    args = parser.parse_args()

    today = datetime.now(tz=UTC).date()
    records = load_records()
    test_records = get_test_split(records)
    print(f"Test split: {len(test_records)} messages (of {len(records)} total, 20% held out)")

    print("\nRunning baseline (no API cost)...")
    baseline_results = run_baseline(test_records)
    print(f"  baseline macro-F1: {baseline_results['macro_f1']:.4f}")

    n_llm = args.sample or len(test_records)
    print(f"\nRunning LLM on {n_llm} messages (real API calls, retries possible)...")
    llm_results = run_llm(test_records, args.sample)
    if llm_results["macro_f1"] is not None:
        print(f"  llm macro-F1: {llm_results['macro_f1']:.4f}")
    print(f"  llm invalid-JSON rate: {llm_results['invalid_json_rate']:.2%}")
    print(f"  llm % routed to review: {llm_results['pct_routed_to_review']:.1f}%")

    out = {
        "date": str(today),
        "test_split_size": len(test_records),
        "sample_limit_used": args.sample,
        "baseline": baseline_results,
        "llm": llm_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"comparison_{today}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
