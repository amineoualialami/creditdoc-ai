"""
Naive RAG pipeline — top-k retrieval + context-aware prompt + LLM.

No reranking, no hybrid search, no query rewriting.
This is the "just add retrieval" baseline for comparing against Week 4.
"""
import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import litellm

from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "creditdoc")
EMBED_MODEL = os.getenv("LITELLM_EMBEDDING_MODEL", "ollama/nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
TOP_K = 5

qdrant = QdrantClient(url=QDRANT_URL)

SYSTEM_PROMPT = """You are an assistant specialized in French consumer and mortgage credit.
Answer in French, concisely and factually.

Rules:
- Base your answer ONLY on the provided context when it is relevant.
- If the context does not contain the answer, say so explicitly instead of guessing.
- If the question asks for personal advice or future predictions, refuse politely and redirect to a professional advisor.
- Cite sources inline when you can (e.g. "selon le document X").
"""

USER_TEMPLATE = """Question: {question}

Contexte (extraits de documents de référence) :
{context}

Réponds à la question en te basant sur ce contexte."""


def embed_query(text: str) -> List[float]:
    """Embed a single query string."""
    response = litellm.embedding(
        model=EMBED_MODEL,
        input=[text],
        api_base=OLLAMA_BASE,
    )
    return response.data[0]["embedding"]


def retrieve(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Top-k vector search over Qdrant."""
    query_vector = embed_query(query)
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
    )
    return [
        {
            "text": hit.payload["text"],
            "source": hit.payload["source"],
            "score": hit.score,
        }
        for hit in response.points
    ]


def format_context(hits: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks for the prompt."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[Source {i} — {hit['source']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def run_naive_rag(question: str) -> Dict[str, Any]:
    """End-to-end: retrieve, augment prompt, generate answer."""
    start = time.time()

    # 1. Retrieve
    hits = retrieve(question, k=TOP_K)
    retrieved_contexts = [hit["text"] for hit in hits]

    # 2. Augment prompt
    context_str = format_context(hits)
    user_message = USER_TEMPLATE.format(question=question, context=context_str)

    # 3. Generate
    response = litellm.completion(
        model=os.getenv("LITELLM_MODEL", "ollama/phi4"),
        api_base=OLLAMA_BASE,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )

    return {
        "answer": response.choices[0].message.content,
        "retrieved_contexts": retrieved_contexts,
        "latency_s": round(time.time() - start, 2),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


if __name__ == "__main__":
    question = "Quelle est la différence entre un TAEG et un TEG ?"
    print(f"Q: {question}\n")
    result = run_naive_rag(question)
    print(f"A: {result['answer']}\n")
    print(f"Retrieved {len(result['retrieved_contexts'])} contexts in {result['latency_s']}s")