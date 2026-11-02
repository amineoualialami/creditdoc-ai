"""
Thin wrapper over Langfuse v2 SDK — provides a context manager
for creating a trace with nested spans, kept pipeline-agnostic.
"""
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

# Instantiate once at import — Langfuse SDK handles batching internally
_langfuse = Langfuse()


def new_trace(name: str, question: str, pipeline: str, tags: list | None = None):
    """Create a top-level trace for a single RAG query."""
    return _langfuse.trace(
        name=name,
        input={"question": question},
        tags=(tags or []) + [pipeline],
        metadata={"pipeline": pipeline},
    )


@contextmanager
def span(trace, name: str, input_data: dict | None = None):
    """Context manager for a nested span. Auto-closes with output."""
    s = trace.span(name=name, input=input_data or {})
    captured = {}
    try:
        yield captured
    finally:
        s.end(output=captured.get("output", {}))


def log_generation(trace, name: str, model: str, messages: list,
                   answer: str, usage: dict | None = None):
    """Log an LLM generation as a Langfuse generation (special span type)."""
    return trace.generation(
        name=name,
        model=model,
        input=messages,
        output=answer,
        usage=usage or {},
    )


def flush():
    """Flush pending traces to Langfuse. Call once per batch run."""
    _langfuse.flush()