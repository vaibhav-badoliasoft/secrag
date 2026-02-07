import json
import numpy as np
from pathlib import Path

from utils.embeddings import embed_query


def load_chunks(chunks_path: Path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list")
    return data


def load_embeddings(embeddings_path: Path):
    emb = np.load(embeddings_path).astype(np.float32)
    if emb.ndim != 2:
        raise ValueError("Embeddings must be 2D (num_chunks, dim)")
    return emb


def retrieve_top_k(
    chunks_path: Path,
    embeddings_path: Path,
    query: str,
    top_k: int = 5,
    min_score=None
):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    chunks = load_chunks(chunks_path)
    embeddings = load_embeddings(embeddings_path)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Mismatch: chunks count != embeddings rows")

    q = embed_query(query)      # (384,)
    scores = embeddings @ q     # cosine similarity (because normalized)

    k = min(top_k, scores.shape[0])
    top_idx = np.argpartition(-scores, k - 1)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    results = []
    for idx in top_idx:
        score = float(scores[idx])
        if min_score is not None and score < min_score:
            continue

        ch = chunks[int(idx)]
        results.append({
            "score": score,
            "chunk_id": ch.get("chunk_id"),
            "filename": ch.get("filename"),
            "content": ch.get("content"),
            "metadata": {
                "source_path": ch.get("source_path"),
                "created_at": ch.get("created_at"),
                "char_start": ch.get("char_start"),
                "char_end": ch.get("char_end"),
            }
        })

    return results
