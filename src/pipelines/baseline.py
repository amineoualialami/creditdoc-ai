"""
Baseline pipeline — LLM only, no RAG, no retrieval.
Used as the Week 4 reference point. Every subsequent pipeline
(Naive RAG, Advanced RAG, Agent) must beat this.
"""
import os
import time
from typing import Dict, Any
from dotenv import load_dotenv
import litellm

load_dotenv()

SYSTEM_PROMPT = (
    "You are an assistant specialized in French consumer and mortgage credit. "
    "Answer in French, concisely and factually. "
    "If you don't know the answer or if it depends on future events or personal advice, "
    "say so explicitly instead of guessing."
)


def run_baseline(question: str) -> Dict[str, Any]:
    """Call the LLM alone with no retrieved context."""
    start = time.time()
    response = litellm.completion(
        model=os.getenv("LITELLM_MODEL"),
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    return {
        "answer": response.choices[0].message.content,
        "retrieved_contexts": [],  # no retrieval in baseline
        "latency_s": round(time.time() - start, 2),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


if __name__ == "__main__":
    # Quick smoke test
    result = run_baseline("Quelle est la différence entre un TAEG et un TEG ?")
    print(result["answer"])
    print(f"\nLatency: {result['latency_s']}s, tokens: {result['completion_tokens']}")