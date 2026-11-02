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
from src.observability.langfuse_helper import new_trace, span, log_generation

OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "creditdoc")

load_dotenv()

SYSTEM_PROMPT = (
    "You are an assistant specialized in French consumer and mortgage credit. "
    "Answer in French, concisely and factually. "
    "If you don't know the answer or if it depends on future events or personal advice, "
    "say so explicitly instead of guessing."
)


def run_baseline(question: str) -> Dict[str, Any]:
    start = time.time()
    model_name = os.getenv("LITELLM_MODEL", "ollama/phi4")

    trace = new_trace(name="baseline", question=question,
                      pipeline="baseline", tags=["no_rag"])

    with span(trace, "prompt_build", {"n_chunks": 0}) as s:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        s["output"] = {"prompt_chars": sum(len(m["content"]) for m in messages)}

    response = litellm.completion(
        model=model_name, api_base=OLLAMA_BASE,
        messages=messages, temperature=0.2,
    )
    answer = response.choices[0].message.content
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }
    log_generation(trace, "llm_answer", model=model_name,
                   messages=messages, answer=answer, usage=usage)

    latency = time.time() - start
    trace.update(output={"answer": answer},
                 metadata={"latency_s": latency, "n_contexts": 0})

    return {
        "answer": answer,
        "retrieved_contexts": [],
        "latency_s": latency,
        "model": model_name,
    }


if __name__ == "__main__":
    # Quick smoke test
    result = run_baseline("Quelle est la différence entre un TAEG et un TEG ?")
    print(result["answer"])
    print(f"\nLatency: {result['latency_s']}s, tokens: {result['completion_tokens']}")