"""
scripts/validate_dataset.py

Validates data/labeled_messages.jsonl against the CivicInbox taxonomy.
Run: python scripts/validate_dataset.py
Exits non-zero on any failure so it can be wired into CI later if wanted.
"""

import json
import sys
from collections import Counter
from pathlib import Path

EXPECTED_CATEGORIES = {
    "academic_records",
    "financial_aid",
    "registration",
    "document_request",
    "general_inquiry",
    "complaint_escalation",
}
EXPECTED_URGENCIES = {"low", "medium", "high"}
EXPECTED_ROW_COUNT = 150

DATA_PATH = Path("data/labeled_messages.jsonl")


def main() -> int:
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.")
        return 1

    rows = []
    with DATA_PATH.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"ERROR: line {lineno} is not valid JSON: {e}")
                return 1
            rows.append(row)

    errors = []

    # Row count
    if len(rows) != EXPECTED_ROW_COUNT:
        errors.append(f"Expected {EXPECTED_ROW_COUNT} rows, found {len(rows)}.")

    # Required fields + empty label checks
    seen = set()
    dupes = []
    category_counts = Counter()
    urgency_counts = Counter()

    for row in rows:
        rid = row.get("id", "<missing id>")
        text = row.get("text", "")
        category = row.get("category", "")
        urgency = row.get("urgency", "")

        if not text.strip():
            errors.append(f"{rid}: empty message text.")
        if not category:
            errors.append(f"{rid}: empty category label.")
        elif category not in EXPECTED_CATEGORIES:
            errors.append(f"{rid}: category '{category}' not in taxonomy.")
        if not urgency:
            errors.append(f"{rid}: empty urgency label.")
        elif urgency not in EXPECTED_URGENCIES:
            errors.append(f"{rid}: urgency '{urgency}' not in {EXPECTED_URGENCIES}.")

        category_counts[category] += 1
        urgency_counts[urgency] += 1

        key = (text.strip().lower(), category)
        if key in seen:
            dupes.append(rid)
        seen.add(key)

    if dupes:
        errors.append(f"Duplicate message+category rows: {dupes}")

    # Report
    print("=== Category distribution ===")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    print("\n=== Urgency distribution ===")
    for urg, count in sorted(urgency_counts.items()):
        print(f"  {urg}: {count}")

    print("\n=== Category set check ===")
    missing = EXPECTED_CATEGORIES - set(category_counts.keys())
    extra = set(category_counts.keys()) - EXPECTED_CATEGORIES
    if missing:
        errors.append(f"Categories missing from dataset entirely: {missing}")
    if extra:
        errors.append(f"Unexpected categories found: {extra}")
    if not missing and not extra:
        print("  OK — matches taxonomy exactly.")

    print()
    if errors:
        print(f"FAILED — {len(errors)} issue(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print(f"PASSED — {len(rows)} rows, taxonomy matches, no duplicates, no empty labels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
