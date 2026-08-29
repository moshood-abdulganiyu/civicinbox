"""
Step 9 — Evaluate the baseline TF-IDF + LogisticRegression classifier.

Loads the model and held-out test split saved by scripts/train_baseline.py
(does NOT re-split — reuses the exact rows saved in Step 8 to keep the
evaluation valid). Computes macro-F1 and per-class precision/recall,
writes a dated JSON report to evals/results/.

ASSUMPTION TO VERIFY: X_test in test_split.json is expected to already be
normalized (normalize_text() applied before the Step 8 split was taken).
If that's wrong, predictions here won't reflect real incoming-message
behavior — check train_baseline.py before trusting these numbers.
"""

import json
from datetime import date
from pathlib import Path

import joblib
from sklearn.metrics import classification_report, f1_score

MODEL_PATH = Path("data/models/baseline.joblib")
SPLIT_PATH = Path("data/models/test_split.json")
NAIVE_FLOOR = 0.05  # macro-F1 for "always guess one class" on 6 balanced classes


def main():
    pipeline = joblib.load(MODEL_PATH)

    with open(SPLIT_PATH) as f:
        split = json.load(f)
    X_test = split["X_test"]
    y_test = split["y_test"]

    y_pred = pipeline.predict(X_test)

    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    if macro_f1 < NAIVE_FLOOR + 0.05:
        print(
            f"WARNING: macro-F1 ({macro_f1:.3f}) is close to the naive-guess "
            f"floor (~{NAIVE_FLOOR}). Something is likely broken, not just "
            f"'weak model'."
        )

    for cat in ["general_inquiry", "complaint_escalation"]:
        if cat in report_dict:
            print(
                f"{cat}: recall={report_dict[cat]['recall']:.3f}, "
                f"precision={report_dict[cat]['precision']:.3f}"
            )

    output = {
        "date": str(date.today()),
        "model": "tfidf_logreg_baseline",
        "n_test_examples": len(y_test),
        "macro_f1": macro_f1,
        "accuracy": report_dict["accuracy"],
        "per_class": {
            cat: {
                "precision": vals["precision"],
                "recall": vals["recall"],
                "f1": vals["f1-score"],
                "support": vals["support"],
            }
            for cat, vals in report_dict.items()
            if cat not in ("accuracy", "macro avg", "weighted avg")
        },
    }

    out_path = Path(f"evals/results/baseline_{output['date']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"Wrote {out_path}, macro_f1={macro_f1:.3f}")


if __name__ == "__main__":
    main()
