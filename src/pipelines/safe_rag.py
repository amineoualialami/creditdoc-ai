"""
Safe RAG pipeline — advanced RAG with two guardrails:

  1. Intent classifier (pre-pipeline) — a small LLM classifies the query
     into {in_scope, personal_advice, prediction, out_of_domain} and
     short-circuits with a structured refusal for anything but in_scope.

  2. Reranker confidence threshold (post-retrieval) — if the top-1 rerank
     score is below RERANK_THRESHOLD, we assume the corpus does not cover
     the question and return a structured "insufficient context" response
     instead of forcing a generation.

Reuses the retrieval + rerank + generation helpers from advanced_rag so
there is a single source of truth for hybrid search.

Return shape is identical to run_advanced_rag so the eval harness works
unchanged.
"""
import os
import time
from typing import Dict, Any, List
from dotenv import load_dotenv
import litellm

from src.observability.langfuse_helper import new_trace, span, log_generation
from src.pipelines.advanced_rag import (
    vector_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank,
    format_context,
    CANDIDATES_K,
    FINAL_K,
    USER_TEMPLATE,
)

load_dotenv()

OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
PIPELINE_MODEL = os.getenv("LITELLM_MODEL", "ollama/phi4")
INTENT_MODEL = os.getenv("INTENT_MODEL", "ollama/llama3:8b")

# Threshold on the reranker's top-1 score. bge-reranker-v2-m3 returns
# logits roughly in [-10, 10]; positive = likely relevant, negative = irrelevant.
# 0.0 is permissive (lets through anything the reranker thinks is non-negative);
# increase to 1.0 or 2.0 for stricter rejection. To calibrate rigorously, see
# phase 1.2 of the W7 plan (score distribution over golden questions).
RERANK_THRESHOLD = float(os.getenv("RERANK_THRESHOLD", "0.0"))

# ──────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────

INTENT_SYSTEM_PROMPT = """Tu es un classificateur d'intent pour un assistant crédit français.

Classe la question de l'utilisateur en UNE SEULE catégorie :

- in_scope : question factuelle sur le crédit à la consommation ou le crédit immobilier en France (définitions, calculs TAEG, règles réglementaires, procédures, droits de l'emprunteur…)
- personal_advice : demande un conseil personnel ("devrais-je accepter…", "est-ce une bonne idée…", "quelle offre choisir…")
- prediction : demande de prévision future ("quel sera le taux…", "va-t-il baisser…", "évolution future de…")
- out_of_domain : sujet hors crédit français (santé, recettes, autre pays, immobilier non lié au crédit, politique…)

Réponds UNIQUEMENT par le nom de la catégorie, en un seul mot, sans explication."""


ALLOWED_INTENTS = {"in_scope", "personal_advice", "prediction", "out_of_domain"}


def classify_intent(question: str) -> str:
    """Zero-shot intent classification with a small LLM.

    Returns one of ALLOWED_INTENTS. Falls back to 'in_scope' if the
    classifier output cannot be parsed — safer to let a borderline query
    through than to over-block a legitimate one.
    """
    messages = [
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    try:
        response = litellm.completion(
            model=INTENT_MODEL,
            api_base=OLLAMA_BASE,
            messages=messages,
            temperature=0.0,
            timeout=60,
            max_tokens=15,
        )
        raw = response.choices[0].message.content.strip().lower()
        # Robust parse: look for any allowed intent word in the response
        for intent in ALLOWED_INTENTS:
            if intent in raw:
                return intent
        return "in_scope"  # fallback — let it through
    except Exception:
        return "in_scope"  # fallback — do not block on classifier error


# ──────────────────────────────────────────────────────────────────────────
# STRUCTURED REFUSAL TEMPLATES
# ──────────────────────────────────────────────────────────────────────────

REFUSAL_TEMPLATES = {
    "personal_advice": (
        "Je ne peux pas vous conseiller personnellement sur l'acceptation, le refus "
        "ou le choix d'une offre de crédit. Cette décision dépend de votre situation "
        "financière globale, de vos autres engagements et de votre capacité de "
        "remboursement. Je vous recommande de consulter un conseiller bancaire ou "
        "d'utiliser un simulateur officiel comme celui de la Banque de France."
    ),
    "prediction": (
        "Je ne peux pas prédire l'évolution future des taux d'intérêt, des règles "
        "réglementaires ou des décisions de la Banque centrale européenne. Pour des "
        "projections à jour, consultez les publications officielles de la BCE, de la "
        "Banque de France, ou les notes de conjoncture récentes."
    ),
    "out_of_domain": (
        "Votre question sort du domaine que je couvre (crédit à la consommation et "
        "crédit immobilier en France). Je ne peux pas y répondre de façon fiable."
    ),
}

INSUFFICIENT_CONTEXT_RESPONSE = (
    "Les documents de référence que j'ai consultés ne contiennent pas d'information "
    "directement applicable à votre question. Je préfère ne pas répondre plutôt que "
    "de risquer une imprécision. Pour une réponse fiable, consultez le Code de la "
    "consommation (Légifrance) ou un conseiller bancaire."
)


# ──────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT V2 (durci sur les refus)
# ──────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_V2 = """Tu es un assistant spécialisé dans le crédit à la consommation et le crédit immobilier (France).
Tu réponds en français, de façon concise et factuelle.

RÈGLES DE RÉPONSE — à appliquer strictement :

1. UTILISE EN PRIORITÉ le contexte fourni. Si la réponse factuelle est présente dans
   le contexte, cite-la sans paraphraser au point d'en changer le sens. Référence la
   source entre crochets, par exemple [Source 2].

2. SI LE CONTEXTE CONTIENT LES DONNÉES D'ENTRÉE POUR UN CALCUL (montants, taux,
   durées), tu peux effectuer le raisonnement arithmétique nécessaire — c'est
   considéré comme grounded.

3. SI LE CONTEXTE NE CONTIENT PAS LA RÉPONSE :
   - Dis-le explicitement : "Les documents de référence consultés ne traitent pas
     spécifiquement de X."
   - N'INVENTE PAS de chiffre, de date, de nom de loi ou d'article de code.
   - Termine par une recommandation de vérification auprès d'une source officielle.
   - N'AJOUTE PAS de [Hors sources] sur un domaine régulé — mieux vaut dire "je ne
     sais pas" que de deviner.

4. NE DONNE JAMAIS DE CONSEIL PERSONNEL. Si la question demande "devrais-je…",
   "est-ce une bonne idée…", "quelle offre choisir…" : refuse en une phrase et
   redirige vers un conseiller. Ne liste PAS d'arguments pour/contre — c'est
   déjà un conseil déguisé.

5. NE PRÉDIS JAMAIS. Les questions sur l'évolution future des taux, des règles ou
   du marché doivent être refusées — pas d'extrapolation, pas de "il est probable
   que…", pas de scénarios.

6. Reste focalisé sur la question. Pas d'introduction, pas de récapitulation, pas
   de digression sur des sujets adjacents présents dans le contexte mais non
   demandés.
"""


# ──────────────────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────────────────

def _refusal_result(reason: str, answer: str, start: float, trace) -> Dict[str, Any]:
    """Build a return dict matching run_advanced_rag's shape for a refusal path."""
    latency = round(time.time() - start, 2)
    trace.update(
        output={"answer": answer},
        metadata={"latency_s": latency, "refused": True, "reason": reason, "n_contexts": 0},
    )
    return {
        "answer": answer,
        "retrieved_contexts": [],
        "latency_s": latency,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }


def run_safe_rag(question: str) -> Dict[str, Any]:
    """End-to-end: classify intent → (refuse early) → hybrid retrieve+rerank
    → (refuse low-context) → generate. Fully traced in Langfuse.
    """
    start = time.time()

    trace = new_trace(
        name="safe_rag",
        question=question,
        pipeline="safe_rag",
        tags=["week-7", "guarded", "hybrid+rerank"],
    )

    # ① Intent classifier (pre-pipeline guard)
    with span(trace, "intent_classify", {"model": INTENT_MODEL}) as s:
        intent = classify_intent(question)
        s["output"] = {"intent": intent}

    if intent != "in_scope":
        answer = REFUSAL_TEMPLATES[intent]
        return _refusal_result(f"intent={intent}", answer, start, trace)

    # ② Retrieval
    with span(trace, "vector_search", {"k": CANDIDATES_K}) as s:
        vector_hits = vector_search(question, k=CANDIDATES_K)
        s["output"] = {"n_results": len(vector_hits)}

    with span(trace, "bm25_search", {"k": CANDIDATES_K}) as s:
        bm25_hits = bm25_search(question, k=CANDIDATES_K)
        s["output"] = {"n_results": len(bm25_hits)}

    with span(trace, "rrf_fusion", {"inputs": ["vector", "bm25"]}) as s:
        fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
        s["output"] = {"n_unique": len(fused)}

    with span(trace, "rerank",
              {"candidates": len(fused[:CANDIDATES_K]), "top_k": FINAL_K}) as s:
        top_hits = rerank(question, fused[:CANDIDATES_K], top_k=FINAL_K)
        top1_score = top_hits[0]["rerank_score"] if top_hits else float("-inf")
        s["output"] = {
            "kept": len(top_hits),
            "top1_rerank_score": top1_score,
            "all_rerank_scores": [h["rerank_score"] for h in top_hits],
        }

    # ③ Confidence threshold (post-retrieval guard)
    with span(trace, "confidence_check",
              {"threshold": RERANK_THRESHOLD, "top1_score": top1_score}) as s:
        passes_threshold = top1_score >= RERANK_THRESHOLD
        s["output"] = {"passes_threshold": passes_threshold}

    if not passes_threshold:
        return _refusal_result(
            f"low_confidence (top1={top1_score:.2f} < {RERANK_THRESHOLD})",
            INSUFFICIENT_CONTEXT_RESPONSE, start, trace,
        )

    # ④ Generation (reached only if intent=in_scope AND top1 ≥ threshold)
    retrieved_contexts = [hit["text"] for hit in top_hits]
    context_str = format_context(top_hits)
    user_message = USER_TEMPLATE.format(question=question, context=context_str)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_message},
    ]

    response = litellm.completion(
        model=PIPELINE_MODEL,
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

    log_generation(trace, "llm_answer", model=PIPELINE_MODEL,
                   messages=messages, answer=answer, usage=usage)

    latency = round(time.time() - start, 2)
    trace.update(
        output={"answer": answer},
        metadata={"latency_s": latency, "refused": False, "n_contexts": len(retrieved_contexts)},
    )

    return {
        "answer": answer,
        "retrieved_contexts": retrieved_contexts,
        "latency_s": latency,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
    }


if __name__ == "__main__":
    # Three canonical smoke tests covering all code paths
    cases = [
        ("Quelle est la durée du droit de rétractation d'un crédit à la consommation ?",
         "expected: in_scope → normal generation"),
        ("Devrais-je accepter cette offre de crédit de 15 000 EUR à 6% TAEG ?",
         "expected: personal_advice → structured refusal"),
        ("Comment faire une tarte aux pommes ?",
         "expected: out_of_domain → structured refusal"),
    ]
    for q, note in cases:
        print(f"\n{'='*70}\nQ: {q}\n({note})\n{'='*70}")
        result = run_safe_rag(q)
        print(f"A: {result['answer'][:400]}")
        print(f"[{result['latency_s']}s, {len(result['retrieved_contexts'])} contexts]")
