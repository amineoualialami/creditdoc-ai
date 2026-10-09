# Golden Dataset — CreditDoc AI

## Purpose

Reference set of question-answer pairs that defines what "correct"
means for CreditDoc AI. All RAG iterations and production changes
are evaluated against this dataset.

## File format

JSONL (one JSON object per line) at `golden_dataset.jsonl`.

## Schema

Each entry has:

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable identifier, format `g001`, `g002`... |
| `question` | string | The question in French, as a credit advisor would phrase it |
| `category` | string | One of: `definition`, `calculation`, `regulatory`, `nuanced`, `out_of_scope` |
| `difficulty` | string | `easy`, `medium`, `hard` |
| `expected_answer` | string | The reference answer. For `out_of_scope`, this is the refusal + redirect |
| `expected_sources` | list of strings | Document identifiers that should be retrieved |
| `must_contain` | list of strings | Phrases/numbers the answer MUST mention |
| `must_not_contain` | list of strings | Phrases the answer MUST NOT contain (e.g. personal advice, invented regulations) |
| `notes` | string | Author notes, edge cases, context |

## Categories

- **definition** — "What is X?" factual lookup
- **calculation** — "Compute X given Y and Z"
- **regulatory** — Legal and compliance questions
- **nuanced** — Ambiguous situations requiring judgment within domain
- **out_of_scope** — Questions the assistant should refuse (personal advice, predictions)

## Versioning

The dataset is versioned in Git. Every change requires:
1. A commit message explaining what was added/changed
2. An entry in CHANGELOG.md with date and rationale
3. Re-running the eval against the previous RAG version to detect drift