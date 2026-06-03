"""
chunking_strategies.py  —  Three pluggable chunking strategies.

Strategies:
  "fixed"     — character-based fixed-size with overlap (your original baseline)
  "sentence"  — sentence-aware splitting (your existing chunking_v2 logic, preserved exactly)
  "semantic"  — splits on embedding similarity drops (topic-boundary detection)

All strategies return List[Tuple[char_start, char_end, chunk_text, strategy_name]].

Usage:
    from utils.chunking_strategies import chunk_text, STRATEGIES

    chunks = chunk_text(text, strategy="sentence")   # default
    # or
    comparison = compare_strategies(text)            # runs all three, returns dict
"""

import re
from typing import List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size with overlap  (baseline)
# ---------------------------------------------------------------------------

def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[int, int, str, str]]:
    """Character-based fixed-size chunks with character overlap."""
    if not text.strip():
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk, "fixed"))
        if end == length:
            break
        start = end - overlap
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-based (your existing logic, verbatim)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r".+?(?:[.!?]+(?=\s|$)|$)", re.DOTALL)


def _split_sentences_with_spans(text: str) -> List[Tuple[int, int, str]]:
    spans = []
    for m in _SENTENCE_RE.finditer(text):
        s = m.group(0)
        if not s or not s.strip():
            continue
        spans.append((m.start(), m.end(), s.strip()))
    return spans


def chunk_sentence(text: str, chunk_size: int = 500, overlap_sentences: int = 1) -> List[Tuple[int, int, str, str]]:
    """Sentence-aware chunking (preserves your chunking_v2.py logic exactly)."""
    sents = _split_sentences_with_spans(text)
    if not sents:
        return []

    chunks = []
    i = 0
    n = len(sents)
    while i < n:
        start = sents[i][0]
        current_text_parts = []
        j = i
        while j < n:
            candidate = (" ".join(current_text_parts + [sents[j][2]])).strip()
            if len(candidate) <= chunk_size or not current_text_parts:
                current_text_parts.append(sents[j][2])
                j += 1
            else:
                break
        end = sents[j - 1][1]
        chunk_text = " ".join(current_text_parts).strip()
        chunks.append((start, end, chunk_text, "sentence"))
        if j >= n:
            break
        i = max(j - overlap_sentences, i + 1)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Semantic chunking (topic-boundary detection via embeddings)
# ---------------------------------------------------------------------------

def chunk_semantic(
    text: str,
    chunk_size: int = 500,
    similarity_threshold: float = 0.75,
) -> List[Tuple[int, int, str, str]]:
    """
    Split text on topic boundaries detected by cosine similarity drops between
    consecutive sentence embeddings.

    Falls back to sentence chunking if sentence-transformers is not available
    or the text has fewer than 3 sentences.
    """
    try:
        from utils.embeddings import embed_texts
    except ImportError:
        return chunk_sentence(text, chunk_size)

    sents = _split_sentences_with_spans(text)
    if len(sents) < 3:
        return chunk_sentence(text, chunk_size)

    sent_texts = [s[2] for s in sents]
    vectors = embed_texts(sent_texts)  # shape (n, dim), normalized

    # Cosine similarity between consecutive sentences
    sims = np.einsum("id,id->i", vectors[:-1], vectors[1:]).astype(float)

    # Identify boundary positions where similarity drops below threshold
    boundaries = set()
    for i, sim in enumerate(sims):
        if sim < similarity_threshold:
            boundaries.add(i + 1)  # split before sentence i+1

    # Build chunks from boundary-delimited sentence groups
    chunks = []
    group_start_idx = 0
    group_indices = sorted(boundaries) + [len(sents)]

    for boundary in group_indices:
        group = sents[group_start_idx:boundary]
        if not group:
            group_start_idx = boundary
            continue

        char_start = group[0][0]
        char_end = group[-1][1]
        group_text = " ".join(s[2] for s in group).strip()

        # If the group is too large, further split with fixed strategy
        if len(group_text) > chunk_size * 1.5:
            sub = chunk_fixed(group_text, chunk_size=chunk_size, overlap=50)
            for sub_start, sub_end, sub_text, _ in sub:
                # Re-map character offsets into original text
                chunks.append((char_start + sub_start, char_start + sub_end, sub_text, "semantic"))
        else:
            chunks.append((char_start, char_end, group_text, "semantic"))

        group_start_idx = boundary

    return chunks if chunks else chunk_sentence(text, chunk_size)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

STRATEGIES = ("fixed", "sentence", "semantic")


def chunk_text(
    text: str,
    strategy: str = "sentence",
    chunk_size: int = 500,
    **kwargs,
) -> List[Tuple[int, int, str, str]]:
    """
    Route to the requested chunking strategy.
    Returns List[Tuple[char_start, char_end, chunk_text, strategy_name]].
    """
    strategy = strategy.lower().strip()
    if strategy == "fixed":
        return chunk_fixed(text, chunk_size=chunk_size, overlap=kwargs.get("overlap", 50))
    elif strategy == "sentence":
        return chunk_sentence(text, chunk_size=chunk_size, overlap_sentences=kwargs.get("overlap_sentences", 1))
    elif strategy == "semantic":
        return chunk_semantic(text, chunk_size=chunk_size, similarity_threshold=kwargs.get("similarity_threshold", 0.75))
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {STRATEGIES}")


def compare_strategies(text: str, chunk_size: int = 500) -> dict:
    """
    Run all three strategies on the same text and return a comparison dict.
    Useful for the /ingest?compare=true endpoint and the eval framework.
    """
    results = {}
    for strategy in STRATEGIES:
        chunks = chunk_text(text, strategy=strategy, chunk_size=chunk_size)
        results[strategy] = {
            "count": len(chunks),
            "avg_length": round(sum(len(c[2]) for c in chunks) / max(len(chunks), 1), 1),
            "min_length": min((len(c[2]) for c in chunks), default=0),
            "max_length": max((len(c[2]) for c in chunks), default=0),
            "chunks": chunks,
        }
    return results
