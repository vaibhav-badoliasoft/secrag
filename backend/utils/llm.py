import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_answer(query: str, retrieved_chunks: list):
    """
    Calls OpenAI using retrieved chunks as context.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set.")

    client = OpenAI(api_key=api_key)

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
