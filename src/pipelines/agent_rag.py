"""
Agent RAG pipeline — ReAct agent with two tools:

  1. retrieve_docs(query) : hybrid search + rerank (reuses advanced_rag)
  2. calculator(expression) : safe arithmetic evaluator

Flow:
  Question → intent classifier (safe_rag) → if in_scope → ReAct loop
  The agent iterates (thought → action → observation) until it produces
  a final_answer, or hits MAX_ITERATIONS.

Reuses the safety layer from safe_rag.
Return shape is identical to run_safe_rag so the eval harness works unchanged.
"""
import os
import re
import ast
import json
import time
import operator
from typing import Dict, Any, List
from dotenv import load_dotenv
import litellm

from src.observability.langfuse_helper import new_trace, span, log_generation
from src.pipelines.safe_rag import classify_intent, REFUSAL_TEMPLATES, _refusal_result
from src.pipelines.advanced_rag import (
    vector_search,
    bm25_search,
    reciprocal_rank_fusion,
    rerank,
    format_context,
    CANDIDATES_K,
    FINAL_K,
)

load_dotenv()

OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
PIPELINE_MODEL = os.getenv("LITELLM_MODEL", "ollama/phi4")

# Agent loop parameters
MAX_ITERATIONS = 5                 # hard cap on think-act-observe cycles
LLM_TIMEOUT = 1800                 # 30 min per LLM call — generous to survive cold starts


# ──────────────────────────────────────────────────────────────────────────
# TOOL 1 — CALCULATOR (safe)
# ──────────────────────────────────────────────────────────────────────────

_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _safe_eval(node):
    """Walk the AST, only allowing arithmetic ops on numbers."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def tool_calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression. Returns the result as a string."""
    try:
        expr = expression.strip().replace(",", ".").replace(" ", "")
        expr = re.sub(r"(EUR|€|%)", "", expr, flags=re.IGNORECASE)
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval(tree.body)
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.4f}"
        return str(result)
    except Exception as e:
        return f"ERROR: could not evaluate '{expression}'. Reason: {e}"


# ──────────────────────────────────────────────────────────────────────────
# TOOL 2 — RETRIEVE DOCS
# ──────────────────────────────────────────────────────────────────────────

def tool_retrieve_docs(query: str) -> str:
    """Hybrid search + rerank. Returns formatted top-5 chunks as a single string."""
    vector_hits = vector_search(query, k=CANDIDATES_K)
    bm25_hits = bm25_search(query, k=CANDIDATES_K)
    fused = reciprocal_rank_fusion([vector_hits, bm25_hits])
    top_hits = rerank(query, fused[:CANDIDATES_K], top_k=FINAL_K)
    if not top_hits:
        return "No relevant documents found for this query."
    return format_context(top_hits)


# ──────────────────────────────────────────────────────────────────────────
# TOOL REGISTRY
# ──────────────────────────────────────────────────────────────────────────

TOOLS = {
    "retrieve_docs": {
        "fn": tool_retrieve_docs,
        "description": (
            "Search the French consumer-credit and mortgage-credit corpus. "
            "Returns the 5 most relevant passages with their sources. "
            "Use when the question asks about regulations, definitions, legal articles, "
            "procedures, or any factual legal/regulatory content."
        ),
        "input_format": "a natural-language search query (string)",
        "example": {"action": "retrieve_docs", "action_input": "durée maximum crédit renouvelable"},
    },
    "calculator": {
        "fn": tool_calculator,
        "description": (
            "Evaluate an arithmetic expression. Supports +, -, *, /, parentheses, powers. "
            "Use for any numerical computation: debt ratios, monthly payments, totals. "
            "Do NOT compute in your head — always use this tool for math."
        ),
        "input_format": "an arithmetic expression (string), e.g. '3500 * 0.35' or '(1 + 0.045/12)**24'",
        "example": {"action": "calculator", "action_input": "3500 * 0.35"},
    },
}


# ──────────────────────────────────────────────────────────────────────────
# REACT SYSTEM PROMPT
# ──────────────────────────────────────────────────────────────────────────

def _format_tools_for_prompt() -> str:
    lines = []
    for name, spec in TOOLS.items():
        lines.append(f"- {name}: {spec['description']}")
        lines.append(f"  input: {spec['input_format']}")
        lines.append(f"  example: {json.dumps(spec['example'], ensure_ascii=False)}")
    return "\n".join(lines)


AGENT_SYSTEM_PROMPT = f"""Tu es un assistant spécialisé dans le crédit à la consommation et le crédit immobilier (France).
Tu réponds en français, de façon concise et factuelle.

Tu as accès à des OUTILS pour chercher dans la documentation et faire des calculs.
Tu dois raisonner en plusieurs étapes : réfléchir, appeler un outil, observer le résultat, répéter si nécessaire, puis produire une réponse finale.

OUTILS DISPONIBLES :

{_format_tools_for_prompt()}

FORMAT DE SORTIE — à chaque étape, réponds en JSON strict, UN SEUL objet JSON, rien avant ni après :

Pour utiliser un outil :
{{"thought": "ton raisonnement en une phrase", "action": "nom_de_l_outil", "action_input": "..."}}

Pour donner la réponse finale (quand tu as assez d'information) :
{{"thought": "pourquoi tu es prêt à conclure", "final_answer": "ta réponse complète à l'utilisateur"}}

RÈGLES :

1. JAMAIS DE CALCUL MENTAL. Pour toute opération arithmétique, appelle calculator — même pour 3500 × 0.35.
2. Pour toute question factuelle (règle, définition, article), appelle retrieve_docs AVANT de répondre.
3. Tu peux enchaîner plusieurs outils : par exemple retrieve_docs pour trouver la règle HCSF (35%), puis calculator pour l'appliquer au revenu de l'utilisateur.
4. Si retrieve_docs ne contient pas la réponse, dis-le dans ton final_answer. N'invente pas.
5. Pas d'introduction, pas de digression. Réponse directe dans final_answer.
"""


# ──────────────────────────────────────────────────────────────────────────
# OUTPUT PARSER
# ──────────────────────────────────────────────────────────────────────────

def parse_agent_output(raw: str) -> Dict[str, Any]:
    """Extract the JSON object from the LLM output. Defensive — handles
    code fences, leading/trailing text, etc. Returns a dict or raises."""
    text = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in LLM output: {raw[:200]}")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        cleaned = match.group(0).replace("'", '"')
        cleaned = re.sub(r",\s*}", "}", cleaned)
        return json.loads(cleaned)


# ──────────────────────────────────────────────────────────────────────────
# AGENT LOOP HELPERS
# ──────────────────────────────────────────────────────────────────────────

def build_user_message(question: str, history: List[Dict[str, Any]]) -> str:
    parts = [f"Question: {question}"]
    if history:
        parts.append("\nHistorique des outils :")
        for i, step in enumerate(history, start=1):
            parts.append(f"\nÉtape {i}:")
            parts.append(f"  Pensée: {step['thought']}")
            parts.append(f"  Action: {step['action']}({step['input']!r})")
            parts.append(f"  Observation: {step['observation'][:800]}")
    parts.append("\nQuelle est ta prochaine étape ? Réponds en JSON.")
    return "\n".join(parts)


def call_agent_llm(question: str, history: List[Dict[str, Any]]):
    user_message = build_user_message(question, history)
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    response = litellm.completion(
        model=PIPELINE_MODEL,
        api_base=OLLAMA_BASE,
        messages=messages,
        temperature=0.1,
        timeout=LLM_TIMEOUT,
    )
    return response.choices[0].message.content, response.usage, messages


# ──────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────

def run_agent_rag(question: str) -> Dict[str, Any]:
    """End-to-end: safety guard → ReAct loop → final answer. Fully traced."""
    start = time.time()

    trace = new_trace(
        name="agent_rag",
        question=question,
        pipeline="agent_rag",
        tags=["week-8", "react-agent", "tools"],
    )

    # ① Safety guard (reuse safe_rag's classifier)
    with span(trace, "intent_classify") as s:
        intent = classify_intent(question)
        s["output"] = {"intent": intent}

    if intent != "in_scope":
        answer = REFUSAL_TEMPLATES[intent]
        return _refusal_result(f"intent={intent}", answer, start, trace)

    # ② ReAct loop
    history = []
    tool_contexts_collected = []
    total_input_tokens = 0
    total_output_tokens = 0
    final_answer = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        with span(trace, f"agent_iteration_{iteration}",
                  {"iteration": iteration, "history_steps": len(history)}) as s:

            raw_output, usage, messages = call_agent_llm(question, history)
            total_input_tokens += usage.prompt_tokens
            total_output_tokens += usage.completion_tokens
            log_generation(trace, f"llm_iter_{iteration}", model=PIPELINE_MODEL,
                           messages=messages, answer=raw_output,
                           usage={"input": usage.prompt_tokens,
                                  "output": usage.completion_tokens,
                                  "total": usage.total_tokens})

            try:
                parsed = parse_agent_output(raw_output)
            except Exception as e:
                s["output"] = {"parse_error": str(e), "raw": raw_output[:400]}
                final_answer = raw_output
                break

            if "final_answer" in parsed:
                s["output"] = {"final_answer": True,
                               "thought": parsed.get("thought", ""),
                               "answer_preview": parsed["final_answer"][:200]}
                final_answer = parsed["final_answer"]
                break

            tool_name = parsed.get("action")
            tool_input = str(parsed.get("action_input", ""))
            if tool_name not in TOOLS:
                s["output"] = {"unknown_tool": tool_name}
                final_answer = f"[Agent error — unknown tool '{tool_name}']"
                break

            with span(trace, f"tool_{tool_name}", {"input": tool_input}) as ts:
                observation = TOOLS[tool_name]["fn"](tool_input)
                ts["output"] = {"observation_preview": observation[:400]}
            if tool_name == "retrieve_docs":
                tool_contexts_collected.append(observation)

            history.append({
                "thought": parsed.get("thought", ""),
                "action": tool_name,
                "input": tool_input,
                "observation": observation,
            })
            s["output"] = {"tool_call": tool_name,
                           "thought": parsed.get("thought", "")[:150]}
    else:
        final_answer = (f"[Je n'ai pas pu converger vers une réponse en "
                        f"{MAX_ITERATIONS} itérations. Historique : "
                        f"{[h['action'] for h in history]}]")

    latency = round(time.time() - start, 2)
    trace.update(output={"answer": final_answer}, metadata={
        "latency_s": latency,
        "iterations": len(history) + (1 if final_answer else 0),
        "tools_used": [h["action"] for h in history],
    })

    return {
        "answer": final_answer,
        "retrieved_contexts": tool_contexts_collected,
        "latency_s": latency,
        "prompt_tokens": total_input_tokens,
        "completion_tokens": total_output_tokens,
    }


if __name__ == "__main__":
    cases = [
        ("Pour un prêt de 10 000 EUR sur 24 mois à 4.5% TAEG, calcule la mensualité constante.",
         "expected: agent uses calculator (maybe retrieve_docs first for the formula)"),
        ("Quelle est la durée du droit de rétractation d'un crédit à la consommation ?",
         "expected: agent uses retrieve_docs, cites L312-19, no calculator"),
        ("Devrais-je accepter cette offre de crédit de 15 000 EUR à 6% TAEG ?",
         "expected: intent classifier short-circuits with personal_advice refusal"),
    ]
    for q, note in cases:
        print(f"\n{'='*70}\nQ: {q}\n({note})\n{'='*70}")
        result = run_agent_rag(q)
        print(f"A: {result['answer']}\n")
        print(f"[{result['latency_s']}s, {len(result['retrieved_contexts'])} tool-contexts, "
              f"{result['prompt_tokens']} in / {result['completion_tokens']} out]")