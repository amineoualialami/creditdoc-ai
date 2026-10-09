"""
Validate the structure of golden_dataset.jsonl.
Run: uv run python eval/validate_dataset.py
"""
import json
from pathlib import Path
from collections import Counter

DATASET_PATH = Path(__file__).parent / "golden_dataset.jsonl"

REQUIRED_FIELDS = {
    "id", "question", "category", "difficulty",
    "expected_answer", "expected_sources",
    "must_contain", "must_not_contain", "notes"
}

VALID_CATEGORIES = {"definition", "calculation", "regulatory", "nuanced", "out_of_scope"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate():
    entries = []
    errors = []

    with open(DATASET_PATH, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON — {e}")
                continue

            missing = REQUIRED_FIELDS - set(entry.keys())
            if missing:
                errors.append(f"Line {i} ({entry.get('id', '?')}): missing fields {missing}")

            if entry.get("category") not in VALID_CATEGORIES:
                errors.append(f"Line {i} ({entry.get('id')}): invalid category {entry.get('category')}")

            if entry.get("difficulty") not in VALID_DIFFICULTIES:
                errors.append(f"Line {i} ({entry.get('id')}): invalid difficulty {entry.get('difficulty')}")

            for field in ("expected_sources", "must_contain", "must_not_contain"):
                if not isinstance(entry.get(field), list):
                    errors.append(f"Line {i} ({entry.get('id')}): {field} must be a list")

            entries.append(entry)

    print(f"\n=== Golden Dataset Validation ===")
    print(f"Total entries: {len(entries)}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    else:
        print("All entries valid.")

    ids = [e["id"] for e in entries]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        print(f"\nDuplicate IDs: {duplicates}")

    print(f"\n--- Distribution ---")
    print("By category:")
    for cat, count in Counter(e["category"] for e in entries).most_common():
        print(f"  {cat:15} {count}")
    print("\nBy difficulty:")
    for diff, count in Counter(e["difficulty"] for e in entries).most_common():
        print(f"  {diff:10} {count}")

    print(f"\n--- Target distribution for a healthy dataset ---")
    print("  definition:    8-12")
    print("  calculation:   6-10")
    print("  regulatory:    8-12")
    print("  nuanced:       6-10")
    print("  out_of_scope:  4-6")
    print("  Total:         30-50")

    return len(errors) == 0


if __name__ == "__main__":
    ok = validate()
    exit(0 if ok else 1)