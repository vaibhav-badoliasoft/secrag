import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")
    return OpenAI(api_key=api_key)


def generate_answer(query: str, retrieved_chunks: list):
    client = _get_client()

    context = "\n\n".join(
        f"[Chunk {c['chunk_id']} | Score {round(c['score'],3)}]\n{c['content']}"
        for c in retrieved_chunks
    )

    prompt = f"""
You are a security expert assistant.
Answer the question using ONLY the provided context.
If the answer is not present, say you do not know.

Context:
{context}

Question:
{query}
"""

    response = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=500,
    )

    return response.output_text.strip()


def generate_sample_questions(filename: str, context: str, max_output_tokens: int = 220) -> list[str]:
    """
    Returns 6 content-aware sample questions as list[str].
    Tries JSON parse first; falls back to line parsing.
    """
    client = _get_client()

    prompt = f"""
You are helping build a RAG demo UI.

Generate 6 short, helpful sample questions a user can ask about this document.
Questions must be specific to the content and NOT generic.
Do not mention "chunks" or "context".
Do not include numbering like "1.".

Document: {filename}

Context:
{context}

Return ONLY a JSON array of strings.
Example:
["Question 1?", "Question 2?", "Question 3?"]
"""

    resp = client.responses.create(
        model="gpt-4.1",
        input=prompt,
        max_output_tokens=max_output_tokens,
    )

    text = (resp.output_text or "").strip()

    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            cleaned = []
            for x in arr:
                s = str(x).strip()
                if s:
                    cleaned.append(s)
            return cleaned[:6]
    except Exception:
        pass

    lines = []
    for line in text.splitlines():
        s = line.strip().lstrip("-•").strip()
        if not s:
            continue
        if len(s) > 2 and (s[0].isdigit() and s[1] in [".", ")", "-"]):
            s = s[2:].strip()
        if s:
            lines.append(s)

    return lines[:6]
