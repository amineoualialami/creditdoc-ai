"""
Chainlit UI pour CreditDoc AI.
Permet d'interroger safe_rag ou agent_rag en mode chat, avec visualisation
des décisions (intent, outils appelés, latence).

Lancer : chainlit run ui/app.py
"""
import asyncio
import os
import sys
from pathlib import Path

# Rendre src.* importable quand lancé depuis la racine du repo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chainlit as cl
from dotenv import load_dotenv

from src.pipelines.safe_rag import run_safe_rag
from src.pipelines.agent_rag import run_agent_rag

load_dotenv()

LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "http://localhost:3000")

PIPELINES = {
    "agent_rag (ReAct + tools)": run_agent_rag,
    "safe_rag (hybrid + rerank + safety)": run_safe_rag,
}


# ─────────────────────────────────────────────────
# Setup — sélecteur de pipeline au démarrage
# ─────────────────────────────────────────────────
@cl.on_chat_start
async def start():
    # Default pipeline
    cl.user_session.set("pipeline_name", "agent_rag (ReAct + tools)")

    settings = await cl.ChatSettings([
        cl.input_widget.Select(
            id="pipeline",
            label="Pipeline",
            values=list(PIPELINES.keys()),
            initial_index=0,
        ),
    ]).send()

    await cl.Message(
        content=(
            "**CreditDoc AI — assistant crédit français**\n\n"
            "Pose une question sur le crédit à la consommation ou le crédit immobilier. "
            "L'assistant utilise le RAG local et, selon la question, appelle des outils "
            "(recherche documentaire, calculator) ou refuse si la question est hors-périmètre.\n\n"
            f"*Pipeline actif : {cl.user_session.get('pipeline_name')}*"
        ),
    ).send()


@cl.on_settings_update
async def update_settings(settings):
    cl.user_session.set("pipeline_name", settings["pipeline"])
    await cl.Message(content=f"✓ Pipeline changé pour : **{settings['pipeline']}**").send()


# ─────────────────────────────────────────────────
# Chat handler
# ─────────────────────────────────────────────────
@cl.on_message
async def main(message: cl.Message):
    pipeline_name = cl.user_session.get("pipeline_name")
    pipeline_fn = PIPELINES[pipeline_name]

    # "Thinking" indicator while the pipeline runs
    async with cl.Step(name=f"Running {pipeline_name}", type="llm") as step:
        step.input = message.content
        # run_agent_rag / run_safe_rag sont bloquants — on les wrap dans un thread
        result = await asyncio.to_thread(pipeline_fn, message.content)
        step.output = f"Done in {result['latency_s']}s"

    # Format et envoi de la réponse principale
    answer = result["answer"]
    n_contexts = len(result["retrieved_contexts"])
    tokens = result["prompt_tokens"] + result["completion_tokens"]

    # Metadata affichée en bas de la réponse
    metadata_md = (
        f"\n\n---\n"
        f"⏱️ {result['latency_s']}s · "
        f"📚 {n_contexts} contextes récupérés · "
        f"🎫 {tokens} tokens"
    )
    if tokens == 0 and n_contexts == 0:
        metadata_md += "\n\n🛡️ *Court-circuit par le classifier d'intent — aucun appel LLM ni retrieval.*"

    metadata_md += f"\n\n🔍 [Voir les traces complètes dans Langfuse]({LANGFUSE_HOST})"

    await cl.Message(content=answer + metadata_md).send()