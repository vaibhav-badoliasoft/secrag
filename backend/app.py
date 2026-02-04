from fastapi import FastAPI, UploadFile, File
from utils.chunking import chunk_text
from pypdf import PdfReader
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

    chunk_data = []
    for index, chunk in enumerate(chunks):
        chunk_data.append({
        "chunk_id": index,
        "content": chunk
        })

    chunk_filename = file.filename.replace(".pdf", "_chunks.json")
    chunk_path = os.path.join(DATA_DIR, chunk_filename)
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)

    return {
        "filename": file.filename,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunk_data),
        "first_chunk_preview": chunk_data[0]["content"][:200] if chunk_data else ""
    }