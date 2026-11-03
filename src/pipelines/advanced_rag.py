"""
Advanced RAG pipeline — hybrid search (BM25 + vector) + cross-encoder reranking.

Flow:
  1. Query -> embed + BM25 tokenize
  2. Retrieve top-20 from vector (Qdrant) + top-20 from BM25
  3. Fuse the two ranked lists with Reciprocal Rank Fusion
  4. Rerank top-20 candidates with bge-reranker-v2-m3
  5. Keep top-5 -> inject into prompt -> LLM
"""
import os
import time
import re
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv
import litellm

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.observability.langfuse_helper import new_trace, span, log_generation

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "creditdoc")
EMBED_MODEL = os.getenv("LITELLM_EMBEDDING_MODEL", "ollama/nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Retrieval params
CANDIDATES_K = 20   # how many we fetch from each retriever before reranking
FINAL_K = 5         # how many make it into the LLM prompt
RRF_K_CONSTANT = 60 # reciprocal rank fusion constant (standard)

qdrant = QdrantClient(url=QDRANT_URL)

# Load the reranker once at import time — it's slow to initialize
print("Loading reranker model (first run downloads ~500MB)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

# Load all corpus chunks into memory for BM25 (fine for small POC corpora)
print("Loading corpus into BM25 index...")
_all_points = qdrant.scroll(
    collection_name=COLLECTION,
    limit=10000,
    with_payload=True,
    with_vectors=False,
)[0]
_corpus_chunks = [
    {"id": str(p.id), "text": p.payload["text"], "source": p.payload["source"]}
    for p in _all_points
]
_tokenized_corpus = [re.findall(r"\w+", c["text"].lower()) for c in _corpus_chunks]
bm25 = BM25Okapi(_tokenized_corpus)
print(f"BM25 index built on {len(_corpus_chunks)} chunks.\n")

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

Contexte (extraits de documents de référence, les plus pertinents en premier) :
{context}

Réponds à la question en te basant sur ce contexte."""


def embed_query(text: str) -> List[float]:
    response = litellm.embedding(model=EMBED_MODEL, input=[text], api_base=OLLAMA_BASE)
    return response.data[0]["embedding"]


def vector_search(query: str, k: int) -> List[Dict[str, Any]]:
    """Top-k vector search via Qdrant."""
    query_vector = embed_query(query)
    response = qdrant.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=k,
    )
    return [
        {"id": str(hit.id), "text": hit.payload["text"],
         "source": hit.payload["source"], "score": hit.score}
        for hit in response.points
    ]


def bm25_search(query: str, k: int) -> List[Dict[str, Any]]:
    """Top-k BM25 keyword search on the in-memory corpus."""
    tokens = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokens)
    ranked_indices = scores.argsort()[::-1][:k]
    results = []
    for idx in ranked_indices:
        if scores[idx] <= 0:
            continue
        chunk = _corpus_chunks[idx]
        results.append({**chunk, "score": float(scores[idx])})
    return results


def reciprocal_rank_fusion(result_lists: List[List[Dict[str, Any]]],
                           k_constant: int = RRF_K_CONSTANT) -> List[Dict[str, Any]]:
    """Merge several ranked lists via RRF. Returns unique chunks sorted by RRF score."""
    rrf_scores = {}
    chunk_by_id = {}
    for ranked_list in result_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k_constant + rank + 1)
            chunk_by_id[cid] = chunk
    fused = sorted(chunk_by_id.values(),
                   key=lambda c: rrf_scores[c["id"]], reverse=True)
    for c in fused:
        c["rrf_score"] = rrf_scores[c["id"]]
    return fused


def rerank(query: str, candidates: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Cross-encoder reranking: scores (query, chunk) pairs, keeps top_k."""
    if not candidates:
        return []
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    reranked = []
    for chunk, score in scored[:top_k]:
        reranked.append({**chunk, "rerank_score": float(score)})
    return reranked


def format_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[Source {i} — {hit['source']}]\n{hit['text']}")
    return "\n\n".join(blocks)


def run_advanced_rag(question: str) -> Dict[str, Any]:
    """End-to-end: hybrid retrieve, rerank, generate — fully traced."""
    start = time.time()
    model_name = os.getenv("LITELLM_MODEL", "ollama/phi4")

    # One trace per query
    trace = new_trace(
        name="advanced_rag",
        question=question,
        pipeline="advanced_rag",
        tags=["week-6", "hybrid+rerank"],
    )

    # 1. Vector search
    with span(trace, "vector_search", {"k": CANDIDATES_K}) as s:
        vector_hits = vector_search(question, k=CANDIDATES_K)
        s["output"] = {"n_results": len(vector_hits),
                       "top_sources": [h["source"] for h in vector_hits[:3]]}

    # 2. BM25 search
    with span(trace, "bm25_search", {"k": CANDIDATES_K}) as s:
        bm25_hits = bm25_search(question, k=CANDIDATES_K)
        s["output"] = {"n_results": len(bm25_hits),
                       "top_sources": [h["source"] for h in bm25_hits[:3]]}

    # 3. RRF fusion
    with span(trace, "rrf_fusion", {"inputs": ["vector", "bm25"]}) as s:
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
        s["output"] = {"n_unique": len(fused)}

    # 4. Rerank
    with span(trace, "rerank",
              {"candidates": len(fused[:CANDIDATES_K]), "top_k": FINAL_K}) as s:
        top_hits = rerank(question, fused[:CANDIDATES_K], top_k=FINAL_K)
        s["output"] = {
            "kept": len(top_hits),
            "final_chunks": [
                {"source": h["source"], "rerank_score": h["rerank_score"],
                 "text_preview": h["text"][:140]}
                for h in top_hits
            ],
        }

    retrieved_contexts = [hit["text"] for hit in top_hits]

    # 5. Generate
    context_str = format_context(top_hits)
    user_message = USER_TEMPLATE.format(question=question, context=context_str)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

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

    log_generation(trace, "llm_answer", model=model_name,
                   messages=messages, answer=answer, usage=usage)

    # Close the trace with the final output
    trace.update(output={"answer": answer}, metadata={
        "latency_s": round(time.time() - start, 2),
        "n_contexts": len(retrieved_contexts),
    })

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
    result = run_advanced_rag(question)
    print(f"A: {result['answer']}\n")
    print(f"Retrieved {len(result['retrieved_contexts'])} contexts in {result['latency_s']}s")