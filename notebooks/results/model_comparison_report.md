# Model Comparison — 2026-10-07

## Models tested
- Llama 3 8B (`ollama/llama3:8b`)
- Qwen 2.5 7B (`ollama/qwen2.5:7b`)
- Phi-4 14B (`ollama/phi4:latest`)

## Latency and token summary

| model_label   |   avg_latency_s |   median_latency_s |   avg_output_tokens |   total_errors |
|:--------------|----------------:|-------------------:|--------------------:|---------------:|
| Llama 3 8B    |           79.28 |              79.53 |               130.4 |              0 |
| Phi-4 14B     |          251.88 |             241.36 |               264.6 |              0 |
| Qwen 2.5 7B   |           63.17 |              60.26 |                80.6 |              0 |

## Methodology

- 10 questions across 5 categories
- Temperature 0.2, same system prompt for all models
- Scoring dimensions: factuality, completeness, concision, honesty_on_unknowns
- Scoring scale: 1 (unacceptable) to 5 (excellent)
- Max total per question = 20

## Quality scores — per-model averages

| model_label   |   factuality |   completeness |   concision |   honesty_on_unknowns |   total |
|:--------------|-------------:|---------------:|------------:|----------------------:|--------:|
| Llama 3 8B    |          4   |            3.5 |         3.9 |                   5   |    16.4 |
| Phi-4 14B     |          3.9 |            3.9 |         3.7 |                   4.8 |    16.3 |
| Qwen 2.5 7B   |          3.2 |            3   |         3.1 |                   5   |    14.3 |

## Quality scores — per-model x category (average total per question)

| model_label   |   definition |   calculation |   regulatory |   nuanced |   out_of_scope |
|:--------------|-------------:|--------------:|-------------:|----------:|---------------:|
| Llama 3 8B    |         13   |          16   |         18.5 |      15   |           19.5 |
| Phi-4 14B     |         16.5 |          17   |         18   |      14.5 |           15.5 |
| Qwen 2.5 7B   |         12   |          13.5 |         14   |      13   |           19   |

_Scored 30 of 30 possible pairs._

## Model Comparison Observations

### Summary

Phi-4 14B and Llama 3 8B are statistically tied overall (16.3/20 avg total).
Qwen 2.5 7B trails meaningfully (14.3/20). The deltas between models are
smaller than expected, but hide a critical qualitative difference on
out-of-scope questions.

### Per-model profile

#### Phi-4 14B — the strongest "knowledge" model
- Dominates on definitions (q1-q2), regulatory content (q5-q6), and
  nuanced interpretation (q7-q8).
- Matches its reputation: Microsoft trained Phi-4 for reasoning-heavy,
  structured responses on textbook-like content — a near-perfect profile
  for a RAG grounded in regulatory corpora.
- Weakness: broke honesty on q10 (personal financial advice).
  Only non-5 honesty score in the whole dataset. Critical for a
  credit advisor use case.

#### Llama 3 8B — the "boring, reliable" option
- Never the best, rarely the worst.
- Perfect honesty across every question (5/5 on all 10).
- Weaker on structured knowledge (q1 definitions, q8 nuance),
  but acceptable.
- The pick if hallucination tolerance is near zero.

#### Qwen 2.5 7B — not competitive at this size
- Failed on definitions (q1 scored 1/1/1 on content dimensions).
- Hallucinated a regulatory detail on q6 (factuality = 1).
- Consistently 2-5 points behind on every category except
  out-of-scope.
- Caveat: this was the 7B, not the 14B originally planned.
  A fair comparison requires pulling qwen2.5:14b. On this run,
  deprioritize for Week 3+.

### The findings

The biggest model wasn't the safest on the dimension that mattered
most for the domain.

Phi-4 14B on q10 (user asking "should I accept this loan?"):
- Gave advice when it should have refused.
- Completeness 2, concision 2, honesty 3 — the only broken honesty
  score in 30 scored pairs.

Llama 3 8B and Qwen 2.5 7B both correctly refused and redirected.

For a credit advisor assistant where personal financial recommendation
is a liability (regulatory, legal, reputational), this isn't a
nice-to-have — it's a core safety property.

### Limitations of this study

1. **Small sample (n=10 questions).** Differences inside 1 point are
   noise; the Qwen gap is real, the Llama vs Phi-4 split is within
   noise except on q10.
2. **Honesty was 5/5 for 29 of 30 pairs.** This dimension didn't
   discriminate enough. Next round needs more adversarial
   out-of-scope questions (fake regulations, prediction requests,
   personal advice variants).
3. **No ground-truth answers.** Scores are judgment-based.
   Week 3 golden dataset + Ragas will give reproducible metrics.
4. **Qwen 7B vs Llama 8B vs Phi-4 14B is not a fair size comparison.**
   Pulling qwen2.5:14b would be needed to properly compare the
   14B tier.
5. **Zero-shot, no RAG, no system prompt tuning.** The actual
   production answers will be much better than what any model
   produced here.

## Architecture considerations

Two-model routing strategy rather than a single pick:

| Role | Model | Rationale |
| --- | --- | --- |
| Primary generation (knowledge, regulation, calculation) | Phi-4 14B | Highest ceiling on domain content |
| Safety fallback / out-of-scope detection | Llama 3 8B | Perfect honesty record, acts as a guardrail |
| Retired for now | Qwen 2.5 7B | Not competitive at this size |

Start with Phi-4 as the single generation model
to simplify iteration. The honesty gap will be mitigated by:
- A stronger system prompt.
- NeMo Guardrails.
- LLM routing layer where personal-advice questions get
  classified upstream and routed to Llama 3 8B or hard-refused
  before any LLM call.

The q10 failure is explicitly tracked as a known risk until mitigation confirmed.

