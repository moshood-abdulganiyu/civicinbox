# CivicInbox — Baseline vs. LLM Evaluation Report

**Date:** 2026-09-20
**Test set:** 90 messages (20% stratified held-out split, `RANDOM_STATE=42`, seeded and reproducible via `evals/run_comparison.py`)

---

## 1. Dataset

- 450 labeled messages total, 6-category taxonomy: `academic_records`, `financial_aid`, `registration`, `document_request`, `general_inquiry`, `complaint_escalation`
- Composition: 150 LLM-drafted + 300 user-authored messages, 75 messages per category
- Domain: university student affairs / registrar office
- **Limitation:** LLM-drafted messages (1/3 of the dataset) may be more stylistically uniform than real user-authored text, which could make both models' scores optimistic relative to a fully organic dataset. Disclosed here per the master evaluation standard.
- **Known confound:** urgency correlates strongly with category in this dataset (e.g. `complaint_escalation` skews high-urgency). Category-classification metrics below should not be read as independent evidence about urgency-detection quality.


## 2. Baseline: TF-IDF + Logistic Regression

**Macro-F1: 0.8349** | Accuracy: 0.8333

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| academic_records | 0.625 | 0.667 | 0.645 | 15 |
| complaint_escalation | 0.867 | 0.867 | 0.867 | 15 |
| document_request | 0.778 | 0.933 | 0.848 | 15 |
| financial_aid | 1.000 | 0.800 | 0.889 | 15 |
| general_inquiry | 0.875 | 0.933 | 0.903 | 15 |
| registration | 0.923 | 0.800 | 0.857 | 15 |

Weakest class: `academic_records` (F1 0.645), consistent with the original Step 9 measurement.

## 3. LLM: Structured Classification (OpenAI, `temperature=0`)

**Macro-F1: 0.679 (mean of 5 runs)** | This run: 0.682, accuracy 0.711

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| academic_records | 1.000 | 0.133 | 0.235 | 15 |
| complaint_escalation | 0.500 | 1.000 | 0.667 | 15 |
| document_request | 0.688 | 0.733 | 0.710 | 15 |
| financial_aid | 1.000 | 0.733 | 0.846 | 15 |
| general_inquiry | 0.778 | 0.933 | 0.848 | 15 |
| registration | 0.846 | 0.733 | 0.786 | 15 |

Invalid-JSON rate: 0.00% | Routed to review: 0.0% | Auto-approved: 90/90

By far the LLM's weakest class: `academic_records` (F1 0.235 — the single largest driver of the macro-F1 gap vs. the baseline).


### Run-to-run variance (transparency note)

Five repeated runs on the identical held-out split, with `temperature=0` explicitly set, produced macro-F1 values ranging 0.669–0.692 (spread 0.023). This is consistent with OpenAI's documented behavior: `temperature=0` reduces but does not eliminate output variance, due to floating-point non-associativity in their inference infrastructure. It is not a defect in this codebase's classification or retry logic. An earlier single-run measurement (Step 16, pre-Step-3 failure-path changes) recorded 0.724; given the now-confirmed variance, that figure is directional and superseded by the 5-run mean above, not treated as a fixed baseline.

## 4. Comparison

| Metric | Baseline (TF-IDF + LogReg) | LLM (structured, temp=0) |
|---|---:|---:|
| Macro-F1 | **0.8349** | 0.679 (mean of 5) |
| Invalid output rate | n/a | 0.00% |
| % routed to human review | 0% (never gates) | 0.0–1.1% |
| Per-request cost | ~$0 | real API cost per call |

**The baseline outperforms the LLM by roughly 15 points of macro-F1 on this dataset**, holding across every repeated LLM run. This is the central finding: assuming an LLM is automatically better than a simple, interpretable, free baseline is not supported by the evidence here.


## 5. Error Analysis

- **`academic_records` is the LLM's dominant failure mode.** Precision 1.0 / recall 0.133 means the LLM correctly labeled only 2 of 15 true `academic_records` messages — whenever it *did* predict the category, it was right, but it very rarely predicted it at all.
- **These misclassifications land almost entirely in `complaint_escalation`.** That category shows recall 1.0 (all 15 true complaints caught) but precision 0.5 (30 total predictions, half wrong) — meaning roughly 15 false-positive `complaint_escalation` predictions exist, and 13 of the 15 true `academic_records` messages were misrouted somewhere. The two numbers together strongly suggest most of that 13 landed specifically in `complaint_escalation`, consistent with the original Step 16 finding (12/15) on an earlier code state.
- **Review routing does not catch this.** The LLM produces schema-valid, confident-looking output even when wrong — 0/90 messages were routed to `needs_review` in this run, and confidence scores cluster 0.85–0.95 regardless of correctness (Step 14 finding). A wrong `academic_records`→`complaint_escalation` classification looks, structurally, identical to a correct one.
- **Transient API failure, recovered correctly:** in an earlier run, `msg_376` hit a connection error on attempt 1 and recovered via Step 3's retry logic — concrete evidence the failure-path hardening works under real conditions, separate from the classification-accuracy issue above.

**Hypothesis for the `academic_records`/`complaint_escalation` confusion:** messages describing an unresolved academic issue (e.g. "my transcript request has been pending for weeks") plausibly read to the LLM as *complaints about* an academic process rather than *requests related to* academic records — a genuine semantic boundary ambiguity in the taxonomy itself, not an obviously fixable prompt bug. Worth noting as a taxonomy-design limitation, not purely a model failure.


## 6. Limitations

- Dataset composition (150 LLM-drafted / 300 user-authored) may inflate both models' scores relative to fully organic messages
- Urgency and category are confounded in this dataset; urgency-extraction quality is not independently validated here
- LLM evaluation numbers reflect `temperature=0` but are not bit-reproducible run-to-run due to OpenAI API-level non-determinism; report uses a 5-run mean rather than a single measurement
- Baseline predictions leave `urgency`, `requested_action`, and `draft_response` empty (non-nullable schema fields the baseline cannot populate) — this affects any full-field-completeness comparison, though it does not affect the macro-F1 category-classification numbers above
- Confidence score is not a reliable routing signal on its own (clusters 0.85–0.95 regardless of correctness); `attempts_used` is used as a secondary signal in routing logic instead