"""
reranker.py  —  Cross-encoder reranker that cuts top-20 candidates to top-5.

Two backends:
  1. cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, free, local)
     pip install sentence-transformers  (already in your env)
  2. LLM-as-judge fallback (uses OpenAI if cross-encoder unavailable)

Usage:
    from utils.reranker import rerank

    # candidates: list of dicts with "content" key (output from retrieve_top_k)
    reranked = rerank(query, candidates, top_k=5)
    # returns same list shape, sorted by rerank score, top_k items
"""

from __future__ import annotations

import os
import logging
from typing import Any

logger = logging.getLogger("secrag.reranker")

# ---------------------------------------------------------------------------
# Cross-encoder backend
# ---------------------------------------------------------------------------

_ce_model = None
_ce_available = None


def _get_cross_encoder():
    global _ce_model, _ce_available
    if _ce_available is None:
        try:
            from sentence_transformers import CrossEncoder
            _ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            _ce_available = True
            logger.info("Reranker: cross-encoder loaded")
        except Exception as e:
            _ce_available = False
            logger.warning(f"Reranker: cross-encoder unavailable ({e}), falling back to LLM-as-judge")
    return _ce_model if _ce_available else None


def _rerank_cross_encoder(query: str, candidates: list[dict]) -> list[dict]:
    model = _get_cross_encoder()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)
    for c, score in zip(candidates, scores):
        c["rerank_score"] = float(score)
        c["rerank_method"] = "cross-encoder"
    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


# ---------------------------------------------------------------------------
# LLM-as-judge fallback
# ---------------------------------------------------------------------------

def _rerank_llm(query: str, candidates: list[dict]) -> list[dict]:
    """
    Ask GPT-4o-mini to score each chunk's relevance to the query (1-10).
    Batches all chunks in a single request to keep costs low.
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logger.error(f"Reranker LLM fallback failed to init OpenAI: {e}")
        # Last resort: return original order with original scores
        for c in candidates:
            c["rerank_score"] = c.get("score", 0.0)
            c["rerank_method"] = "passthrough"
        return candidates

    numbered = "\n\n".join(
        f"[{i}] {c['content'][:400]}" for i, c in enumerate(candidates)
    )
    prompt = (
        f"Query: {query}\n\n"
        f"Rate each passage's relevance to the query on a scale from 1 to 10.\n"
        f"Respond ONLY as a JSON array of integers in the same order, e.g. [8,3,7,...]\n\n"
        f"{numbered}"
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0,
        )
        import json
        scores = json.loads(resp.choices[0].message.content.strip())
        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise ValueError("Score list length mismatch")
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)
            c["rerank_method"] = "llm-judge"
    except Exception as e:
        logger.warning(f"LLM reranker scoring failed ({e}), using original order")
        for c in candidates:
            c["rerank_score"] = c.get("score", 0.0)
            c["rerank_method"] = "passthrough"

    return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank `candidates` for `query`. Returns at most `top_k` items.

    Each candidate dict must have a "content" key.
    The returned dicts gain two new keys:
      - rerank_score: float
      - rerank_method: "cross-encoder" | "llm-judge" | "passthrough"
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    if model is not None:
        ranked = _rerank_cross_encoder(query, candidates)
    else:
        ranked = _rerank_llm(query, candidates)

    return ranked[:top_k]
