from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from utils.uploader import process_pdf_upload
from utils.retriever import retrieve_top_k
from utils.naming import get_artifact_paths
from utils.llm import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        return process_pdf_upload(str(pdf_path), str(DATA_DIR))
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

    chunk_path, emb_path = get_artifact_paths(filename, DATA_DIR)

    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail="Chunks file not found. Upload PDF first.")
    if not emb_path.exists():
        raise HTTPException(status_code=404, detail="Embedding file not found. Upload PDF first.")

    try:
        results = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query=req.query,
            top_k=req.top_k,
            min_score=req.min_score
        )
        return {"filename": filename, "query": req.query, "top_k": req.top_k, "results": results}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")


class AnswerRequest(BaseModel):
    filename: str
    query: str
    top_k: int = 5
    min_score: float | None = None


@app.post("/answer")
def answer(req: AnswerRequest):
    filename = req.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename cannot be empty")

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    chunk_path, emb_path = get_artifact_paths(filename, DATA_DIR)

    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    try:
        retrieved = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query=req.query,
            top_k=req.top_k,
            min_score=req.min_score
        )

        if not retrieved:
            return {"filename": filename, "query": req.query, "answer": "No relevant context found.", "citations": []}

        answer_text = generate_answer(req.query, retrieved)

        return {
            "filename": filename,
            "query": req.query,
            "top_k": req.top_k,
            "answer": answer_text,
            "citations": [
                {
                    "chunk_id": c["chunk_id"],
                    "score": c["score"],
                    "char_range": [c["metadata"]["char_start"], c["metadata"]["char_end"]],
                }
                for c in retrieved
            ],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_docs")
def list_docs():
    pdfs = [f.name for f in DATA_DIR.glob("*.pdf")]
    return {"documents": pdfs}
