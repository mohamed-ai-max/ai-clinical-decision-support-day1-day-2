"""
evaluation_set.py — Day 2 Reference Test Set
Covers the 2 premier WHO clinical guideline sources in data/.

Labels are grounded in indexed PDF text (not circular retrieval output).
Includes positive clinical queries and out-of-scope negative controls.
"""

GUIDELINE = "Guideline for the pharmacological treatment of hypertension in adults.pdf"
HEARTS = "WHO-NMH-NVI-18.2-eng.pdf"

EVAL_SET = [
    # ── Guideline for the pharmacological treatment of hypertension in adults.pdf ──
    {
        "question": "What is the target blood pressure for a patient with cardiovascular disease?",
        "expected_document": GUIDELINE,
        "expected_pages": [28],
        "difficulty": "easy",
        "notes": "Section 3.6 target blood pressure — anchor question (top score ~0.796)",
    },
    {
        "question": "Which antihypertensive drug classes are recommended as first-line therapy?",
        "expected_document": GUIDELINE,
        "expected_pages": [10, 32, 33],
        "difficulty": "medium",
        "notes": "First-line drug class recommendations (sections 3.4–3.5)",
    },
    {
        "question": "What are the blood pressure thresholds for initiating antihypertensive treatment in adults?",
        "expected_document": GUIDELINE,
        "expected_pages": [9, 19],
        "difficulty": "medium",
        "notes": "Initiation thresholds — multiple valid recommendation pages",
    },
    {
        "question": "What lifestyle modifications are recommended alongside pharmacological treatment for hypertension?",
        "expected_document": GUIDELINE,
        "expected_pages": [14, 15],
        "difficulty": "hard",
        "notes": "Non-pharmacological / lifestyle intervention pages",
    },
    {
        "question": "When should combination therapy be initiated for patients with high blood pressure?",
        "expected_document": GUIDELINE,
        "expected_pages": [26, 38, 39],
        "difficulty": "medium",
        "notes": "Recalibrated: p.26 (monotherapy vs combination evidence), p.38–39 (combination algorithms Fig. 3–4)",
    },
    {
        "question": "What is the recommended follow-up interval for blood pressure re-assessment after initiating treatment?",
        "expected_document": GUIDELINE,
        "expected_pages": [30, 37, 38],
        "difficulty": "medium",
        "notes": "Recalibrated: p.30 (follow-up interval RCT), p.37–38 (implementation pathway)",
    },

    # ── WHO-NMH-NVI-18.2-eng.pdf (HEARTS Technical Package) ──
    {
        "question": "What is the HEARTS treatment protocol for hypertension in primary health care?",
        "expected_document": HEARTS,
        "expected_pages": [1, 3, 11],
        "difficulty": "easy",
        "notes": "HEARTS protocol overview and introduction pages",
    },
    {
        "question": "How does the HEARTS module recommend organizing hypertension care at the clinic level?",
        "expected_document": HEARTS,
        "expected_pages": [11, 28],
        "difficulty": "hard",
        "notes": "Clinic organisation and service delivery pages",
    },
    {
        "question": "What simplified medication titration algorithm is recommended under the HEARTS primary care module?",
        "expected_document": HEARTS,
        "expected_pages": [1, 3, 8, 11, 12, 13, 17, 18, 19, 29],
        "difficulty": "easy",
        "notes": "Recalibrated: step/intensification algorithm pages across HEARTS module",
    },
    {
        "question": "How should cardiovascular risk assessment be integrated into primary healthcare hypertension protocols under HEARTS?",
        "expected_document": HEARTS,
        "expected_pages": [1, 3, 4, 5, 9, 11, 13, 14],
        "difficulty": "hard",
        "notes": "Recalibrated: CVD risk integration across HEARTS screening and treatment pages",
    },

    # ── Out-of-scope negative controls ──
    {
        "question": "What screening interval does this guideline recommend for breast cancer?",
        "expected_document": None,
        "expected_pages": [],
        "difficulty": "negative",
        "max_top_score": 0.65,
        "notes": "Out-of-scope — hypertension corpus should not produce a confident match",
    },
    {
        "question": "What is the recommended antibiotic regimen for community-acquired pneumonia?",
        "expected_document": None,
        "expected_pages": [],
        "difficulty": "negative",
        "max_top_score": 0.65,
        "notes": "Out-of-scope — unrelated clinical domain",
    },
]

POSITIVE_EVAL_SET = [q for q in EVAL_SET if q.get("expected_document")]
NEGATIVE_EVAL_SET = [q for q in EVAL_SET if not q.get("expected_document")]

_DIFFICULTY_DIST = {
    d: sum(1 for q in EVAL_SET if q["difficulty"] == d)
    for d in ("easy", "medium", "hard", "negative")
}


def export_eval_csv(path: str = "eval/Day2_Evaluation_Test_Set.csv") -> None:
    """Write EVAL_SET to the hackathon CSV format for submission."""
    import csv
    from pathlib import Path

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["Question", "Expected Source (Document / Section / Page)"])
        for item in EVAL_SET:
            if item.get("expected_document"):
                pages = ", ".join(str(p) for p in item["expected_pages"])
                source = f"{item['expected_document']} / {item['notes']} / Page {pages}"
            else:
                source = "Not covered in hypertension guidelines / Out-of-scope control question"
            writer.writerow([item["question"], source])


if __name__ == "__main__":
    print(f"EVAL_SET loaded: {len(EVAL_SET)} questions")
    print(f"  Positive (scored): {len(POSITIVE_EVAL_SET)}")
    print(f"  Negative (out-of-scope): {len(NEGATIVE_EVAL_SET)}")
    print(f"Difficulty distribution: {_DIFFICULTY_DIST}")
    docs = {q["expected_document"] for q in POSITIVE_EVAL_SET}
    print(f"Active premier documents covered: {len(docs)}/2")
    for doc in sorted(docs):
        print(f"  - {doc}")
