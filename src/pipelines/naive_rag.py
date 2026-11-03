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
from src.observability.langfuse_helper import new_trace, span, log_generation

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "creditdoc")
EMBED_MODEL = os.getenv("LITELLM_EMBEDDING_MODEL", "ollama/nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
TOP_K = 5

qdrant = QdrantClient(url=QDRANT_URL)

SYSTEM_PROMPT = """Tu es un assistant spécialisé dans le crédit à la consommation et le crédit immobilier (France).
Tu réponds en français, de façon concise et factuelle.

RÈGLES DE RÉPONSE — à appliquer strictement :

1. UTILISE EN PRIORITÉ le contexte fourni. Si la réponse factuelle est présente dans le contexte,
   cite-la sans paraphraser au point d'en changer le sens. Référence la source entre crochets, par
   exemple [Source 2].

2. SI LE CONTEXTE EST INCOMPLET MAIS CONTIENT LES DONNÉES D'ENTRÉE (par exemple pour un calcul),
   tu peux effectuer le raisonnement arithmétique ou logique nécessaire — c'est considéré comme
   grounded dans le contexte.

3. SI LE CONTEXTE NE CONTIENT PAS LA RÉPONSE :
   - Dis-le explicitement en une phrase : "Les documents de référence consultés ne traitent pas
     spécifiquement de X."
   - N'INVENTE PAS de chiffre, de date, de nom de loi, de nom d'article de code que tu n'aurais
     pas trouvé dans le contexte.
   - Si tu connais la réponse par connaissance générale et qu'elle est factuellement stable
     (ex : règle réglementaire bien établie), tu peux l'ajouter, mais PRÉFIXE-LA de "[Hors sources] "
     pour que l'utilisateur sache que l'information n'a pas été vérifiée dans les documents.
   - Termine par une recommandation de vérification auprès d'une source officielle.

4. QUESTIONS HORS PÉRIMÈTRE (conseil personnel, prédiction, recommandation d'offre) :
   Refuse poliment et redirige vers un conseiller professionnel ou un simulateur officiel.
   N'ajoute PAS de "[Hors sources]" — c'est un refus, pas une réponse.

5. Reste focalisé sur la question. Pas d'introduction, pas de récapitulation, pas de digression
   sur des sujets adjacents présents dans le contexte mais non demandés.
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
    """End-to-end: retrieve, augment prompt, generate answer — fully traced."""
    start = time.time()
    model_name = os.getenv("LITELLM_MODEL", "ollama/phi4")

    # One trace per query
    trace = new_trace(
        name="naive_rag",
        question=question,
        pipeline="naive_rag",
        tags=["chunking+topk"],
    )

    # 1. Retrieve
    with span(trace, "vector_search", {"k": TOP_K}) as s:
        hits = retrieve(question, k=TOP_K)
        s["output"] = {
            "n_results": len(hits),
            "top_sources": [h["source"] for h in hits[:3]],
            "scores": [round(h["score"], 3) for h in hits],
        }
    retrieved_contexts = [hit["text"] for hit in hits]

    # 2. Augment prompt
    with span(trace, "prompt_build", {"n_chunks": len(hits)}) as s:
        context_str = format_context(hits)
        user_message = USER_TEMPLATE.format(question=question, context=context_str)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        s["output"] = {
            "context_chars": len(context_str),
            "prompt_chars": len(user_message),
        }

    # 3. Generate
    response = litellm.completion(
        model=model_name,
        api_base=OLLAMA_BASE,
        messages=messages,
        temperature=0.2,
        timeout=1600,
    )
    answer = response.choices[0].message.content
    usage = {
        "input": response.usage.prompt_tokens,
        "output": response.usage.completion_tokens,
        "total": response.usage.total_tokens,
    }

    # Log the LLM call as a Langfuse generation
    log_generation(trace, "llm_answer", model=model_name,
                   messages=messages, answer=answer, usage=usage)

    # Close the trace
    trace.update(
        output={"answer": answer},
        metadata={
            "latency_s": round(time.time() - start, 2),
            "n_contexts": len(retrieved_contexts),
        },
    )

    return {
        "answer": answer,
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