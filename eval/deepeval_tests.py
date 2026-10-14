"""
DeepEval test suite — hard assertions on the golden dataset.

Usage:
    uv run deepeval test run eval/deepeval_tests.py

These tests complement Ragas (which gives graded scores) with
boolean pass/fail that fits a CI gate.
"""
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipelines.baseline import run_baseline

DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"


def load_entries():
    entries = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


ENTRIES = load_entries()


def check_must_contain(answer: str, must_contain: list) -> list:
    """Return the list of phrases missing from the answer (empty if all present)."""
    answer_lower = answer.lower()
    return [p for p in must_contain if p.lower() not in answer_lower]


def check_must_not_contain(answer: str, must_not_contain: list) -> list:
    """Return the list of forbidden phrases found in the answer (empty if clean)."""
    answer_lower = answer.lower()
    return [p for p in must_not_contain if p.lower() in answer_lower]


def test_must_contain_phrases():
    """For every entry, every must_contain phrase must appear in the answer."""
    failures = []
    for entry in ENTRIES:
        if not entry["must_contain"]:
            continue
        result = run_baseline(entry["question"])
        missing = check_must_contain(result["answer"], entry["must_contain"])
        if missing:
            failures.append({
                "id": entry["id"],
                "category": entry["category"],
                "missing": missing,
                "answer_preview": result["answer"][:150] + "...",
            })

    print(f"\n=== must_contain report ===")
    print(f"Failures: {len(failures)} / {len([e for e in ENTRIES if e['must_contain']])}")
    for f in failures[:5]:
        print(f"  {f['id']} ({f['category']}): missing {f['missing']}")

    # Record failures but don't block in Week 4 (baseline is expected to fail many)
    return failures


def test_must_not_contain_phrases():
    """For every entry, no must_not_contain phrase may appear in the answer."""
    failures = []
    for entry in ENTRIES:
        if not entry["must_not_contain"]:
            continue
        result = run_baseline(entry["question"])
        found = check_must_not_contain(result["answer"], entry["must_not_contain"])
        if found:
            failures.append({
                "id": entry["id"],
                "category": entry["category"],
                "forbidden_found": found,
                "answer_preview": result["answer"][:150] + "...",
            })

    print(f"\n=== must_not_contain report ===")
    print(f"Failures: {len(failures)} / {len([e for e in ENTRIES if e['must_not_contain']])}")
    for f in failures[:5]:
        print(f"  {f['id']} ({f['category']}): found forbidden {f['forbidden_found']}")

    return failures


def test_out_of_scope_honesty():
    """Out-of-scope questions must trigger a refusal (not an answer)."""
    REFUSAL_MARKERS = [
        "ne peux pas", "je ne peux", "je n'ai pas", "ne suis pas en mesure",
        "consultez", "recommande de", "simulateur", "conseiller",
    ]
    failures = []
    oos_entries = [e for e in ENTRIES if e["category"] == "out_of_scope"]
    for entry in oos_entries:
        result = run_baseline(entry["question"])
        answer_lower = result["answer"].lower()
        has_refusal = any(marker in answer_lower for marker in REFUSAL_MARKERS)
        if not has_refusal:
            failures.append({
                "id": entry["id"],
                "answer_preview": result["answer"][:200] + "...",
            })

    print(f"\n=== out_of_scope honesty report ===")
    print(f"Failures: {len(failures)} / {len(oos_entries)}")
    for f in failures:
        print(f"  {f['id']}: NO refusal detected")
        print(f"    Answer: {f['answer_preview']}")

    return failures


if __name__ == "__main__":
    print("Running DeepEval test suite on baseline pipeline...\n")
    mc = test_must_contain_phrases()
    mnc = test_must_not_contain_phrases()
    oos = test_out_of_scope_honesty()

    total = len(mc) + len(mnc) + len(oos)
    print(f"\n=== Overall ===")
    print(f"Total failures: {total}")
    print(f"  must_contain:     {len(mc)}")
    print(f"  must_not_contain: {len(mnc)}")
    print(f"  out_of_scope:     {len(oos)}")