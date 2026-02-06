from fastapi import FastAPI, UploadFile, File
from utils.embeddings import embed_texts
from utils.chunking import chunk_text
from datetime import datetime
from pypdf import PdfReader
import numpy as np
import json
import os

app = FastAPI()

DATA_DIR = "../data"

os.makedirs(DATA_DIR, exist_ok=True)

@app.get("/health")
def health_check():
    return {"status": "SecRAG backend is running"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    temp_path = os.path.join(DATA_DIR, file.filename)
    with open(temp_path, "wb") as f:
        f.write(contents)

    reader = PdfReader(temp_path)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text() or ""
    
    text_filename = file.filename.replace(".pdf", ".txt")
    text_path = os.path.join(DATA_DIR, text_filename)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    chunks = chunk_text(extracted_text)
    created_at = datetime.utcnow().isoformat()

    chunk_data = []
    for index, (char_start, char_end, chunk) in enumerate(chunks):
        chunk_data.append({
        "chunk_id": index,
        "filename": file.filename,
        "source_path": temp_path,
        "created_at": created_at,
        "char_start": char_start,
        "char_end": char_end,
        "content": chunk
        })

    chunk_filename = file.filename.replace(".pdf", "_chunks.json")
    chunk_path = os.path.join(DATA_DIR, chunk_filename)
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)

    texts = [c["content"] for c in chunk_data]
    vectors = embed_texts(texts)

    emb_filename = file.filename.replace(".pdf", "_embeddings.npy")
    emb_path = os.path.join(DATA_DIR, emb_filename)
    np.save(emb_path, vectors)

    return {
        "filename": file.filename,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunk_data),
        "embedding_dim": int(vectors.shape[1]),
        "first_chunk_preview": chunk_data[0]["content"][:200] if chunk_data else ""
    }