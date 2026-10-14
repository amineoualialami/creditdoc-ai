"""
Ingestion pipeline — read documents from data/corpus/,
chunk them, embed with Ollama, upsert into Qdrant.

Usage:
    uv run python src\ingestion\ingest.py
"""
import os
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv
import litellm

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

load_dotenv()

# Config
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "corpus"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "creditdoc")
EMBED_MODEL = os.getenv("LITELLM_EMBEDDING_MODEL", "ollama/nomic-embed-text")
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

# Naive chunking params
CHUNK_SIZE = 512          # chars (not tokens — simpler for naive v1)
CHUNK_OVERLAP = 64
EMBED_DIM = 768           # nomic-embed-text output dim

qdrant = QdrantClient(url=QDRANT_URL)


def read_file(path: Path) -> str:
    """Dispatch by extension. Returns raw text."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif ext in {".html", ".htm"}:
        from bs4 import BeautifulSoup
        with open(path, encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    elif ext in {".md", ".txt"}:
        return path.read_text(encoding="utf-8")
    else:
        print(f"  Skipping unsupported file: {path.name}")
        return ""


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Naive fixed-size chunking with character overlap."""
    text = " ".join(text.split())  # collapse whitespace
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a batch via LiteLLM/Ollama."""
    response = litellm.embedding(
        model=EMBED_MODEL,
        input=texts,
        api_base=OLLAMA_BASE,
    )
    return [item["embedding"] for item in response.data]


def ensure_collection():
    """Create the Qdrant collection if it doesn't exist."""
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION in existing:
        print(f"Collection '{COLLECTION}' exists. Dropping and recreating for clean ingest.")
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION}' created (dim={EMBED_DIM}, cosine).")


def doc_id_for(source: str, chunk_idx: int) -> str:
    """Deterministic UUID so re-ingesting the same doc overwrites."""
    key = f"{source}:{chunk_idx}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def main():
    if not CORPUS_DIR.exists():
        print(f"ERROR: Corpus directory not found: {CORPUS_DIR}")
        return

    files = sorted([p for p in CORPUS_DIR.iterdir() if p.is_file() and not p.name.startswith(".")])
    print(f"Found {len(files)} files in corpus.\n")

    ensure_collection()

    total_chunks = 0
    for i, path in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] {path.name}")
        text = read_file(path)
        if not text.strip():
            print("  (empty or unreadable)")
            continue

        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")

        # Embed in batches of 16 (Ollama handles small batches best)
        BATCH = 16
        points = []
        for b_start in range(0, len(chunks), BATCH):
            batch = chunks[b_start:b_start + BATCH]
            vectors = embed_batch(batch)
            for j, (chunk, vec) in enumerate(zip(batch, vectors)):
                idx = b_start + j
                points.append(PointStruct(
                    id=doc_id_for(path.name, idx),
                    vector=vec,
                    payload={
                        "source": path.name,
                        "chunk_index": idx,
                        "text": chunk,
                    },
                ))

        if points:
            qdrant.upsert(collection_name=COLLECTION, points=points)
            total_chunks += len(points)
            print(f"  Upserted {len(points)} chunks.")

    print(f"\n=== Done ===")
    print(f"Total chunks indexed: {total_chunks}")
    print(f"Collection: {COLLECTION}")
    print(f"Qdrant dashboard: {QDRANT_URL}/dashboard")


if __name__ == "__main__":
    main()