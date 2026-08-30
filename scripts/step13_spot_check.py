"""
Step 13 — Live API spot check.

Runs classify_with_llm() against 5 hand-picked real messages from
data/labeled_messages.jsonl. This makes 5 real, paid OpenAI API calls
(more if retries fire — max_attempts=3 default, so up to 15 worst case).

This is a manual verification script, not an automated test. Run it,
read the output, eyeball whether each TriageResult looks reasonable.
"""

from app.services.llm_classifier import LLMClassificationFailed, classify_with_llm

# The 5 messages, locked in after manual selection from data/labeled_messages.jsonl
SAMPLES = [
    {
        "label": "1. academic_records (weakest baseline class)",
        "message": "Can I get an official transcript sent to my new employer?",
        "true_category": "academic_records",
        "true_urgency": "low",
    },
    {
        "label": "2. complaint_escalation, non-urgent (category/urgency confound)",
        "message": (
            "I would like to formally escalate the delay in processing my "
            "document request. It has now exceeded the stated turnaround "
            "time by over two weeks with no update."
        ),
        "true_category": "complaint_escalation",
        "true_urgency": "medium",
    },
    {
        "label": "3. messy/multi-intent (accepted overlap with academic_records)",
        "message": (
            "I submitted a grade appeal for my ENG 301 course three weeks "
            "ago and I still haven't heard anything back. This is affecting "
            "my GPA calculation for a scholarship renewal deadline coming up."
        ),
        "true_category": "academic_records",
        "true_urgency": None,  # fill in from your dataset if labeled
    },
    {
        "label": "4. short/low-context",
        "message": "scholarship status pls",
        "true_category": "financial_aid",
        "true_urgency": None,
    },
    {
        "label": "5. clean control",
        "message": "What are the office hours for the registrar this week?",
        "true_category": "general_inquiry",
        "true_urgency": "low",
    },
]


def main() -> None:
    for sample in SAMPLES:
        print("=" * 70)
        print(sample["label"])
        print(f"Message: {sample['message']}")
        print(f"Ground truth: category={sample['true_category']}, urgency={sample['true_urgency']}")
        print("-" * 70)

        try:
            result = classify_with_llm(sample["message"])
        except LLMClassificationFailed as e:
            print(f"FAILED: {e}")
            print(f"Last error: {e.last_error}")
            continue

        print(f"category:             {result.category}")
        print(f"urgency:              {result.urgency}")
        print(f"requested_action:     {result.requested_action}")
        print(f"entities:             {result.entities}")
        print(f"missing_information:  {result.missing_information}")
        print(f"draft_response:       {result.draft_response}")
        print(f"confidence:           {result.confidence}")
        print()


if __name__ == "__main__":
    main()