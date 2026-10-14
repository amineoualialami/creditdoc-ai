"""
End-to-end stack test: Ollama via LiteLLM, trace in Langfuse v2.
Run: uv run python src/creditdoc_ai/hello_llm.py
"""
import os
import time
from dotenv import load_dotenv
from langfuse import Langfuse
import litellm

load_dotenv()
langfuse = Langfuse()


def ask_llm(question: str) -> str:
    """Ask the local LLM, manually create a trace in Langfuse v2."""
    trace = langfuse.trace(
        name="credit-qa-hello",
        tags=["hello-world", "setup-validation"],
        input={"question": question},
    )

    start = time.time()
    response = litellm.completion(
        model=os.getenv("LITELLM_MODEL", "ollama/llama3:8b"),
        api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434"),
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an assistant specialized in consumer and mortgage credit. "
                    "Answer in French, concisely and factually."
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    latency_ms = int((time.time() - start) * 1000)

    trace.generation(
        name="llm-call",
        model=os.getenv("LITELLM_MODEL"),
        input=question,
        output=answer,
        metadata={"latency_ms": latency_ms},
    )
    trace.update(output={"answer": answer})
    return answer


if __name__ == "__main__":
    question = "Quelle est la difference entre un TAEG et un TEG ?"
    print(f"Q: {question}\n")
    answer = ask_llm(question)
    print(f"A: {answer}\n")

    langfuse.flush()
    print("Trace sent to Langfuse. Check: http://localhost:3000")