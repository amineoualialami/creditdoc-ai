"""
Ragas-based evaluation harness.

Usage:
    uv run python eval/ragas_eval.py --pipeline baseline

Loads golden_dataset.jsonl, runs the specified pipeline against each
question, computes Ragas metrics, saves JSON and Markdown reports in
eval/results/.
"""
import argparse
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Ragas imports
from datasets import Dataset
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "golden_dataset.jsonl"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_golden_dataset() -> list:
    """Load the golden dataset from JSONL."""
    entries = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_ragas_dataset(entries: list, pipeline_fn) -> Dataset:
    """Run the pipeline on every entry and build the Ragas-compatible dataset."""
    rows = []
    for i, entry in enumerate(entries, start=1):
        print(f"[{i}/{len(entries)}] {entry['id']} ({entry['category']})...")
        try:
            result = pipeline_fn(entry["question"])
            rows.append({
                "question": entry["question"],
                "answer": result["answer"],
                "contexts": result["retrieved_contexts"] or [""],
                "ground_truth": entry["expected_answer"],
                # extra metadata we'll use later
                "id": entry["id"],
                "category": entry["category"],
                "difficulty": entry["difficulty"],
                "latency_s": result["latency_s"],
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "question": entry["question"],
                "answer": f"[ERROR: {e}]",
                "contexts": [""],
                "ground_truth": entry["expected_answer"],
                "id": entry["id"],
                "category": entry["category"],
                "difficulty": entry["difficulty"],
                "latency_s": 0,
            })
    return Dataset.from_list(rows)


def configure_ragas_judge():
    """Wire Ragas to use a local Ollama model as judge + embeddings."""
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "ollama/llama3:8b").replace("ollama/", "")
    embed_model = os.getenv("LITELLM_EMBEDDING_MODEL", "ollama/nomic-embed-text").replace("ollama/", "")
    base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

    judge_llm = LangchainLLMWrapper(
        ChatOllama(model=judge_model, base_url=base, temperature=0.0, timeout=1200, num_ctx=4096,)
    )
    embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=embed_model, base_url=base)
    )
    return judge_llm, embeddings


def run_evaluation(pipeline_name: str) -> dict:
    """End-to-end eval: load dataset, run pipeline, score with Ragas."""
    from ragas.run_config import RunConfig

    # Direct import for baseline (robust across Python path quirks)
    if pipeline_name == "baseline":
        from src.pipelines.baseline import run_baseline as pipeline_fn
    elif pipeline_name == "naive_rag":
        from src.pipelines.naive_rag import run_naive_rag as pipeline_fn
    else:
        pipeline_module = importlib.import_module(f"src.pipelines.{pipeline_name}")
        pipeline_fn = getattr(pipeline_module, f"run_{pipeline_name}")

    print(f"\n=== Loading golden dataset ===")
    entries = load_golden_dataset()
    print(f"Loaded {len(entries)} entries.\n")

    print(f"=== Running pipeline: {pipeline_name} ===")
    dataset = build_ragas_dataset(entries, pipeline_fn)

    print(f"\n=== Configuring Ragas judge ===")
    judge_llm, embeddings = configure_ragas_judge()

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    run_config = RunConfig(
        timeout=1200,
        max_retries=2,
        max_wait=120,
        max_workers=1,
    )

    print(f"=== Scoring with Ragas (sequential, 10-30 min expected) ===")
    scores = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
        show_progress=True,
    )

    # Safer extraction — handles the case where a metric returned all-nan
    import numpy as np
    scores_df = scores.to_pandas()
    scores_summary = {}
    for metric_name in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if metric_name in scores_df.columns:
            values = scores_df[metric_name].dropna()
            scores_summary[metric_name] = float(values.mean()) if len(values) > 0 else None
        else:
            scores_summary[metric_name] = None

    return {
        "pipeline": pipeline_name,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "n_entries": len(entries),
        "scores_summary": scores_summary,
        "scores_per_entry": scores_df.to_dict(orient="records"),
    }


def save_results(results: dict):
    """Save JSON + Markdown report."""
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    pipeline = results["pipeline"]

    json_path = RESULTS_DIR / f"{pipeline}_{date_str}.json"
    md_path = RESULTS_DIR / f"{pipeline}_{date_str}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Markdown report
    summary = results["scores_summary"]
    lines = [
        f"# Evaluation report — {pipeline}",
        "",
        f"- **Timestamp**: {results['timestamp']}",
        f"- **Entries**: {results['n_entries']}",
        f"- **Judge model**: {os.getenv('RAGAS_JUDGE_MODEL', 'ollama/llama3:8b')}",
        f"- **Pipeline model**: {os.getenv('LITELLM_MODEL', 'ollama/phi4')}",
        "",
        "## Scores (0.0 to 1.0, higher = better)",
        "",
        "| Metric | Score |",
        "| --- | --- |",
    ]
    for metric, value in summary.items():
        v = f"{value:.3f}" if value is not None else "N/A"
        lines.append(f"| {metric} | {v} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `faithfulness`: does the answer stay grounded in the retrieved context?")
    lines.append("- `answer_relevancy`: does the answer address the question?")
    lines.append("- `context_precision`: do the retrieved chunks contain the right info?")
    lines.append("- `context_recall`: do the retrieved chunks cover all the needed info?")
    lines.append("")
    lines.append("For the baseline (no retrieval), context metrics are degenerate — ignore them.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved:")
    print(f"  JSON: {json_path}")
    print(f"  Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", default="baseline",
                        help="Pipeline name (module in src/pipelines/, e.g. 'baseline')")
    args = parser.parse_args()

    # Make src/ importable
    sys.path.insert(0, str(Path(__file__).parent.parent))

    results = run_evaluation(args.pipeline)

    print(f"\n=== Summary ===")
    for metric, value in results["scores_summary"].items():
        v = f"{value:.3f}" if value is not None else "N/A"
        print(f"  {metric:20} {v}")

    save_results(results)


if __name__ == "__main__":
    main()