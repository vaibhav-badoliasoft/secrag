from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pathlib import Path
from utils.uploader import process_pdf_upload
from utils.retriever import retrieve_top_k

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (BASE_DIR / ".." / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health_check():
    return {"status": "SecRAG backend is running"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    pdf_path = DATA_DIR / file.filename

    try:
        pdf_path.write_bytes(contents)
        result = process_pdf_upload(str(pdf_path), str(DATA_DIR))
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class RetrieveRequest(BaseModel):
    filename: str
    query: str
    top_k: int = 5
    min_score: float | None = None


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    filename = req.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename cannot be empty")

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    chunk_filename = filename.replace(".pdf", "_chunks.json")
    emb_filename = filename.replace(".pdf", "_embeddings.npy")

    chunk_path = DATA_DIR / chunk_filename
    emb_path = DATA_DIR / emb_filename

    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunks file not found. Upload PDF first.")
    if not emb_path.exists():
        raise HTTPException(status_code=404, detail="Embeddings file not found. Upload PDF first.")

    try:
        results = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query=req.query,
            top_k=req.top_k,
            min_score=req.min_score
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    return {
        "filename": filename,
        "query": req.query,
        "top_k": req.top_k,
        "results": results
    }
