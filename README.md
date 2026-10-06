# CreditDoc AI

Local-first RAG assistant for consumer credit advisors — a reference
implementation for architects building production-grade GenAI systems
in financial services.


## Target users

A consumer credit advisor at a retail bank or a specialized credit
institution. They need to answer client
questions about credit products, regulation, and internal policies
without flipping through dozens of PDFs.

## Jobs to be done

- Answer factual questions about credit products (TAEG, amortization,
  insurance, eligibility criteria) with cited sources.
- Clarify regulatory requirements (Bank Al-Maghrib circulars, ACPR
  recommendations, EU directives) in plain French.
- Compute simple financial metrics (monthly payment, total cost of
  credit, debt-to-income ratio).
- Escalate out-of-scope or high-stakes questions to a human.

## Non-goals

- Not a decision-making system — the assistant informs, never decides
  on loan approval, pricing, or risk scoring.
- Not a chatbot for end customers — the advisor is the user.
- Not a replacement for the core lending system (CLS) — it reads
  reference content, it doesn't write to transactional systems.
- Not production software — this is a reference implementation and
  commercial demo.

## Assumptions

- Corpus is public reference material (regulatory texts, published
  product sheets) plus synthetic policy documents.
- All PII in the corpus is synthetic or redacted.
- The advisor is French-speaking; the assistant answers in French.
- Local inference is acceptable for development; production would
  use managed inference (Bedrock, Azure OpenAI).
- Single-tenant deployment — multi-tenancy is out of scope for v1.

## Architecture

See `docs/architecture.md` for the full picture. Short version:
Streamlit UI → LangGraph agent → LiteLLM → Ollama (local) or Bedrock
(cloud) + Qdrant for retrieval + NeMo Guardrails + Langfuse for
observability.


## Stack

| Layer | Tool |
| --- | --- |
| LLM runtime | Ollama (local) / AWS Bedrock (cloud) |
| LLM abstraction | LiteLLM |
| Orchestration | LangGraph |
| Vector store | Qdrant |
| Guardrails | NeMo Guardrails + Llama Guard 3 |
| Observability | Langfuse |
| Evaluation | Ragas + DeepEval |
| UI | Streamlit |

## Getting started

See `docs/setup.md`.

## License

MIT. Reference implementation — fork, learn, adapt.

## Author

Amine Ouali Alami — Technology Architect