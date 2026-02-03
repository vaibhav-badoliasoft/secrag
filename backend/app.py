from fastapi import FastAPI, UploadFile, File
from pypdf import PdfReader
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

    return {
        "filename": file.filename,
        "text_preview": extracted_text[:500],
        "total_characters": len(extracted_text)
    }