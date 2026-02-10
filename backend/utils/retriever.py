import json
import numpy as np
from pathlib import Path

from utils.embeddings import embed_query
from utils.bm25 import build_bm25, bm25_scores


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


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    if mx - mn < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def _build_results(chunks: list, indices: np.ndarray, scores: np.ndarray, min_score=None):
    results = []
    for idx in indices:
        sc = float(scores[int(idx)])
        if min_score is not None and sc < float(min_score):
            continue
        ch = chunks[int(idx)]
        results.append({
            "score": sc,
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


def retrieve_top_k(
    chunks_path: Path,
    embeddings_path: Path,
    query: str,
    top_k: int = 5,
    min_score=None,
    mode: str = "hybrid",
    alpha: float = 0.7,
    candidate_mult: int = 5
):

    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    mode = (mode or "hybrid").lower().strip()
    if mode not in {"hybrid", "semantic", "bm25"}:
        raise ValueError("mode must be one of: hybrid, semantic, bm25")

    chunks = load_chunks(chunks_path)

    n = len(chunks)
    if n == 0:
        return []

    bm25_norm = None
    if mode in {"bm25", "hybrid"}:
        bm25 = build_bm25(chunks)
        bm25_raw = np.array(bm25_scores(bm25, query), dtype=np.float32)
        bm25_norm = _minmax_norm(bm25_raw)

    emb_norm = None
    if mode in {"semantic", "hybrid"}:
        embeddings = load_embeddings(embeddings_path)
        if n != embeddings.shape[0]:
            raise ValueError("Mismatch: chunks count != embeddings rows")

        q = embed_query(query)
        emb_scores = (embeddings @ q).astype(np.float32)
        emb_norm = ((emb_scores + 1.0) / 2.0).clip(0.0, 1.0).astype(np.float32)

    if mode == "semantic":
        k = min(top_k, n)
        idx = np.argpartition(-emb_norm, k - 1)[:k]
        idx = idx[np.argsort(-emb_norm[idx])]
        return _build_results(chunks, idx, emb_norm, min_score=min_score)

    if mode == "bm25":
        k = min(top_k, n)
        idx = np.argpartition(-bm25_norm, k - 1)[:k]
        idx = idx[np.argsort(-bm25_norm[idx])]
        return _build_results(chunks, idx, bm25_norm, min_score=min_score)

    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be between 0 and 1")
    if candidate_mult <= 0:
        raise ValueError("candidate_mult must be > 0")

    k_candidates = min(n, top_k * candidate_mult)

    idx_emb = np.argpartition(-emb_norm, k_candidates - 1)[:k_candidates]
    idx_bm = np.argpartition(-bm25_norm, k_candidates - 1)[:k_candidates]
    candidate_set = set(map(int, idx_emb)) | set(map(int, idx_bm))
    candidate_idx = np.array(list(candidate_set), dtype=np.int32)

    final_scores = alpha * emb_norm[candidate_idx] + (1.0 - alpha) * bm25_norm[candidate_idx]

    order = np.argsort(-final_scores)
    chosen = candidate_idx[order][: min(top_k, len(order))]

    final_full = np.zeros((n,), dtype=np.float32)
    final_full[candidate_idx] = final_scores.astype(np.float32)

    return _build_results(chunks, chosen, final_full, min_score=min_score)
