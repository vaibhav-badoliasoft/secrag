"""
eval_framework.py  —  Golden Q&A dataset and automated evaluation metrics.

Three things in one file:
  1. GoldenDataset  — load/save/add hand-curated Q&A pairs
  2. run_eval()     — score a RAG pipeline against the dataset
  3. compare_chunking_strategies()  — run the same eval with all three strategies

Golden dataset format (JSON):
[
  {
    "id": "q001",
    "question": "What is the purpose of X?",
    "expected_answer": "X is used for ...",
    "source_chunks": ["chunk text that should be retrieved"],
    "difficulty": "easy" | "moderate" | "hard",
    "category": "factual" | "multi-hop" | "no-answer",
    "notes": "optional"
  }
]

Usage:
    from utils.eval_framework import GoldenDataset, run_eval

    ds = GoldenDataset("data/eval/golden_qa.json")
    ds.add(question="...", expected_answer="...", difficulty="easy")
    ds.save()

    report = run_eval(
        dataset=ds,
        pdf_name="my_doc.pdf",
        retrieve_fn=retrieve_top_k,   # your existing function
        answer_fn=generate_answer,
        verify_fn=verify_citations,   # optional
    )
    print(report["summary"])
"""

from __future__ import annotations

import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

logger = logging.getLogger("secrag.eval")


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------

class GoldenDataset:
    def __init__(self, path: str = "data/eval/golden_qa.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: list[dict] = []
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    @property
    def items(self) -> list[dict]:
        return self._data

    def add(
        self,
        question: str,
        expected_answer: str,
        source_chunks: list[str] | None = None,
        difficulty: str = "moderate",
        category: str = "factual",
        notes: str = "",
    ) -> str:
        item_id = f"q{str(uuid.uuid4())[:8]}"
        self._data.append({
            "id": item_id,
            "question": question,
            "expected_answer": expected_answer,
            "source_chunks": source_chunks or [],
            "difficulty": difficulty,
            "category": category,
            "notes": notes,
            "created_at": datetime.utcnow().isoformat(),
        })
        return item_id

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        logger.info(f"Saved {len(self._data)} golden Q&A pairs to {self.path}")

    def __len__(self):
        return len(self._data)


# ---------------------------------------------------------------------------
# LLM-as-judge scorer
# ---------------------------------------------------------------------------

def _score_answer(question: str, expected: str, actual: str) -> dict:
    """
    Uses GPT-4o-mini to score actual answer correctness vs expected answer.
    Returns {score: 1-5, reasoning: str}
    """
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        return {"score": 3, "reasoning": "OpenAI unavailable — skipped scoring"}

    prompt = (
        f"Question: {question}\n\n"
        f"Expected Answer: {expected}\n\n"
        f"Actual Answer: {actual}\n\n"
        "Score the Actual Answer from 1 to 5 based on correctness and completeness:\n"
        "5 = correct and complete\n"
        "4 = mostly correct, minor omissions\n"
        "3 = partially correct\n"
        "2 = mostly wrong\n"
        "1 = completely wrong or hallucinated\n\n"
        'Respond ONLY as JSON: {"score": <int>, "reasoning": "<one sentence>"}'
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0,
        )
        return json.loads(resp.choices[0].message.content.strip())
    except Exception as e:
        logger.warning(f"Scoring failed: {e}")
        return {"score": 3, "reasoning": "Scoring error"}


def _score_retrieval(expected_chunks: list[str], retrieved_chunks: list[dict]) -> float:
    """
    Simple retrieval relevance: fraction of expected_chunks that appear in retrieved content.
    Uses substring matching (fast, no LLM needed).
    """
    if not expected_chunks:
        return 1.0
    retrieved_text = " ".join(c.get("content", "") for c in retrieved_chunks).lower()
    hits = sum(1 for ec in expected_chunks if ec.lower()[:100] in retrieved_text)
    return round(hits / len(expected_chunks), 3)


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_eval(
    dataset: GoldenDataset,
    pdf_name: str,
    retrieve_fn: Callable,
    answer_fn: Callable,
    verify_fn: Callable | None = None,
    retrieval_mode: str = "hybrid",
    top_k: int = 5,
    chunk_strategy: str = "sentence",
    run_label: str | None = None,
) -> dict:
    """
    Run the full eval pipeline against the golden dataset.

    retrieve_fn signature: (chunks_path, embeddings_path, query, top_k, ...) -> list[dict]
    answer_fn signature:   (query, retrieved_chunks) -> str
    verify_fn signature:   (answer_text, retrieved_chunks) -> dict  (optional)

    Returns a structured report dict.
    """
    label = run_label or f"{chunk_strategy}_{retrieval_mode}_{datetime.utcnow().strftime('%H%M%S')}"
    results = []

    for item in dataset.items:
        q = item["question"]
        expected = item["expected_answer"]

        # Retrieve
        try:
            retrieved = retrieve_fn(query=q, top_k=top_k, mode=retrieval_mode)
        except Exception as e:
            logger.warning(f"Retrieval failed for '{q[:50]}': {e}")
            retrieved = []

        # Answer
        try:
            answer = answer_fn(q, retrieved) if retrieved else "No relevant context found."
        except Exception as e:
            logger.warning(f"Answer failed for '{q[:50]}': {e}")
            answer = ""

        # Score answer quality
        quality = _score_answer(q, expected, answer)

        # Score retrieval relevance
        retrieval_score = _score_retrieval(item.get("source_chunks", []), retrieved)

        # Citation verification (optional)
        citation_accuracy = None
        if verify_fn and retrieved:
            try:
                vr = verify_fn(answer, retrieved)
                citation_accuracy = vr.get("citation_accuracy")
            except Exception:
                pass

        results.append({
            "id": item["id"],
            "question": q,
            "expected_answer": expected,
            "actual_answer": answer,
            "answer_score": quality.get("score", 0),
            "answer_reasoning": quality.get("reasoning", ""),
            "retrieval_relevance": retrieval_score,
            "citation_accuracy": citation_accuracy,
            "difficulty": item.get("difficulty", ""),
            "category": item.get("category", ""),
            "retrieved_count": len(retrieved),
        })

    # Build summary
    scores = [r["answer_score"] for r in results]
    retrieval_scores = [r["retrieval_relevance"] for r in results]
    citation_scores = [r["citation_accuracy"] for r in results if r["citation_accuracy"] is not None]

    summary = {
        "run_label": label,
        "chunk_strategy": chunk_strategy,
        "retrieval_mode": retrieval_mode,
        "total_questions": len(results),
        "avg_answer_score": round(sum(scores) / max(len(scores), 1), 2),
        "avg_retrieval_relevance": round(sum(retrieval_scores) / max(len(retrieval_scores), 1), 3),
        "avg_citation_accuracy": round(sum(citation_scores) / max(len(citation_scores), 1), 3) if citation_scores else None,
        "score_distribution": {
            str(i): scores.count(i) for i in range(1, 6)
        },
        "per_difficulty": _group_scores(results, "difficulty"),
        "per_category": _group_scores(results, "category"),
        "timestamp": datetime.utcnow().isoformat(),
    }

    return {"summary": summary, "results": results}


def _group_scores(results: list[dict], key: str) -> dict:
    groups: dict[str, list] = {}
    for r in results:
        k = r.get(key, "unknown")
        groups.setdefault(k, []).append(r["answer_score"])
    return {k: round(sum(v) / len(v), 2) for k, v in groups.items()}


# ---------------------------------------------------------------------------
# Chunking strategy comparison
# ---------------------------------------------------------------------------

def compare_chunking_strategies(
    dataset: GoldenDataset,
    pdf_name: str,
    retrieve_fn_factory: Callable,   # takes strategy name, returns retrieve_fn
    answer_fn: Callable,
) -> dict:
    """
    Run the eval suite for all three chunking strategies and return a comparison.
    retrieve_fn_factory("sentence") should return a retrieve_fn that uses sentence-chunked data.
    """
    comparison = {}
    from utils.chunking_strategies import STRATEGIES
    for strategy in STRATEGIES:
        try:
            retrieve_fn = retrieve_fn_factory(strategy)
            report = run_eval(
                dataset=dataset,
                pdf_name=pdf_name,
                retrieve_fn=retrieve_fn,
                answer_fn=answer_fn,
                chunk_strategy=strategy,
                run_label=strategy,
            )
            comparison[strategy] = report["summary"]
        except Exception as e:
            comparison[strategy] = {"error": str(e)}

    # Find winner per metric
    strategies_with_data = [s for s in STRATEGIES if "error" not in comparison.get(s, {})]
    if strategies_with_data:
        best_answer = max(strategies_with_data, key=lambda s: comparison[s].get("avg_answer_score", 0))
        best_retrieval = max(strategies_with_data, key=lambda s: comparison[s].get("avg_retrieval_relevance", 0))
        comparison["_winners"] = {
            "best_answer_quality": best_answer,
            "best_retrieval_relevance": best_retrieval,
        }

    return comparison
