# Evaluation report — advanced_rag

- **Timestamp**: 2026-11-04T14:16:52.176744Z
- **Entries**: 25
- **Judge model**: ollama/llama3:8b
- **Pipeline model**: ollama/phi4:latest

## Scores (0.0 to 1.0, higher = better)

| Metric | Score |
| --- | --- |
| faithfulness | 0.313 |
| answer_relevancy | 0.533 |
| context_precision | 0.534 |
| context_recall | 0.583 |

## Notes

- `faithfulness`: does the answer stay grounded in the retrieved context?
- `answer_relevancy`: does the answer address the question?
- `context_precision`: do the retrieved chunks contain the right info?
- `context_recall`: do the retrieved chunks cover all the needed info?

For the baseline (no retrieval), context metrics are degenerate — ignore them.