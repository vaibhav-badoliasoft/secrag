import json
import numpy as np
from pathlib import Path
from utils.embeddings import embed_query

_CACHE = {
    "chunks": {},
    "embeddings": {},
}

def _mtime(p: Path) -> float:
    return p.stat().st_mtime


def load_chunks(chunks_path: Path):
    p = Path(chunks_path)
    m = _mtime(p)

    cached = _CACHE["chunks"].get(str(p))
    if cached and cached["mtime"] == m:
        return cached["data"]

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Chunks JSON must be a list")

    _CACHE["chunks"][str(p)] = {"mtime": m, "data": data}
    return data


def load_embeddings(embeddings_path: Path):
    p = Path(embeddings_path)
    m = _mtime(p)

    cached = _CACHE["embeddings"].get(str(p))
    if cached and cached["mtime"] == m:
        return cached["data"]

    emb = np.load(p).astype(np.float32)
    if emb.ndim != 2:
        raise ValueError("Embeddings must be 2D (num_chunks, dim)")

    _CACHE["embeddings"][str(p)] = {"mtime": m, "data": emb}
    return emb


def retrieve_top_k(chunks_path: Path, embeddings_path: Path, query: str, top_k: int = 5, min_score=None):
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    chunks = load_chunks(chunks_path)
    embeddings = load_embeddings(embeddings_path)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Mismatch: chunks count != embeddings rows")

    q = embed_query(query)
    scores = embeddings @ q

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


def cache_status():
    return {
        "chunks_cached": len(_CACHE["chunks"]),
        "embeddings_cached": len(_CACHE["embeddings"]),
    }


def load_meta(meta_path: Path):
    p = Path(meta_path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)
