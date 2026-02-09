from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

from utils.uploader import process_pdf_upload
from utils.retriever import retrieve_top_k, load_chunks, load_meta, cache_status
from utils.naming import get_artifact_paths
from utils.llm import generate_answer, generate_sample_questions
from utils.summarizer import summarize_from_chunks

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

    chunk_path, emb_path, _meta_path = get_artifact_paths(filename, DATA_DIR)

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

    chunk_path, emb_path, _meta_path = get_artifact_paths(filename, DATA_DIR)

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

        citations = []
        for c in retrieved:
            content = c.get("content", "") or ""
            citations.append({
                "chunk_id": c.get("chunk_id"),
                "score": c.get("score"),
                "char_range": [c["metadata"]["char_start"], c["metadata"]["char_end"]],
                "preview": content[:240]
            })

        return {
            "filename": filename,
            "query": req.query,
            "top_k": req.top_k,
            "answer": answer_text,
            "citations": citations,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class SummarizeRequest(BaseModel):
    filename: str
    intro_chunks: int = 3
    top_k: int = 5
    max_output_tokens: int = 350
    min_score: float | None = None


@app.post("/summarize")
def summarize(req: SummarizeRequest):
    filename = req.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename cannot be empty")

    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    if req.intro_chunks <= 0 or req.intro_chunks > 10:
        raise HTTPException(status_code=400, detail="intro_chunks must be between 1 and 10")
    if req.top_k <= 0 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    chunk_path, emb_path, _meta_path = get_artifact_paths(filename, DATA_DIR)

    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    try:
        all_chunks = load_chunks(chunk_path)
        intro = all_chunks[: req.intro_chunks]

        retrieved = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query="Summarize this document.",
            top_k=req.top_k,
            min_score=req.min_score
        )

        merged = {}

        for c in intro:
            cid = c.get("chunk_id")
            if cid is None:
                continue
            merged[cid] = {
                "chunk_id": cid,
                "content": c.get("content", ""),
                "score": 0.0,
                "metadata": {
                    "char_start": c.get("char_start"),
                    "char_end": c.get("char_end"),
                },
                "source": "intro"
            }

        for r in retrieved:
            cid = r.get("chunk_id")
            if cid is None:
                continue
            score = float(r.get("score", 0.0))
            if cid in merged:
                if score > float(merged[cid].get("score", 0.0)):
                    merged[cid]["score"] = score
                merged[cid]["source"] = "hybrid"
            else:
                merged[cid] = {
                    "chunk_id": cid,
                    "content": r.get("content", ""),
                    "score": score,
                    "metadata": {
                        "char_start": r.get("metadata", {}).get("char_start"),
                        "char_end": r.get("metadata", {}).get("char_end"),
                    },
                    "source": "retrieved"
                }

        intro_ids = [c.get("chunk_id") for c in intro if c.get("chunk_id") is not None]
        retrieved_ids_sorted = sorted(
            [cid for cid in merged.keys() if cid not in intro_ids],
            key=lambda cid: float(merged[cid].get("score", 0.0)),
            reverse=True
        )

        final_ids = intro_ids + retrieved_ids_sorted
        final_chunks = [merged[cid] for cid in final_ids if cid in merged]

        if not final_chunks:
            return {"filename": filename, "summary": "I do not know.", "citations": []}

        summary_text = summarize_from_chunks(
            filename=filename,
            chunks=final_chunks,
            max_output_tokens=req.max_output_tokens
        )

        citations = []
        for c in final_chunks:
            content = c.get("content", "") or ""
            citations.append({
                "chunk_id": c["chunk_id"],
                "score": c.get("score", 0.0),
                "source": c.get("source", ""),
                "char_range": [
                    c.get("metadata", {}).get("char_start"),
                    c.get("metadata", {}).get("char_end"),
                ],
                "preview": content[:240]
            })

        return {
            "filename": filename,
            "intro_chunks": req.intro_chunks,
            "top_k": req.top_k,
            "summary": summary_text,
            "citations": citations
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list_docs")
def list_docs():
    pdfs = [f.name for f in DATA_DIR.glob("*.pdf")]
    return {"documents": pdfs}


@app.get("/stats")
def stats(filename: str):
    filename = (filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename cannot be empty")
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    chunk_path, emb_path, meta_path = get_artifact_paths(filename, DATA_DIR)

    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    meta = load_meta(meta_path) or {}

    embedding_dim = meta.get("embedding_dim")
    total_chunks = meta.get("total_chunks")
    created_at = meta.get("created_at")

    if total_chunks is None:
        try:
            total_chunks = len(load_chunks(chunk_path))
        except Exception:
            total_chunks = None

    return {
        "filename": filename,
        "total_chunks": total_chunks,
        "embedding_dim": embedding_dim,
        "last_ingested_at": created_at,
        "cache": cache_status(),
        "artifacts": {
            "chunks": chunk_path.name,
            "embedding": emb_path.name,
            "meta": meta_path.name if meta_path.exists() else None,
        }
    }

class SampleQuestionsRequest(BaseModel):
    filename: str
    intro_chunks: int = 2
    top_k: int = 4
    max_output_tokens: int = 220
    min_score: float | None = None

@app.post("/sample_questions")
def sample_questions(req: SampleQuestionsRequest):
    filename = req.filename.strip()
    if not filename:
        raise HTTPException(status_code=400, detail="filename cannot be empty")
    if not filename.lower().endswith(".pdf"):
        filename += ".pdf"

    if req.intro_chunks <= 0 or req.intro_chunks > 10:
        raise HTTPException(status_code=400, detail="intro_chunks must be between 1 and 10")
    if req.top_k <= 0 or req.top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")

    chunk_path, emb_path, _meta_path = get_artifact_paths(filename, DATA_DIR)
    if not chunk_path.exists() or not emb_path.exists():
        raise HTTPException(status_code=404, detail="Artifacts not found. Upload PDF first.")

    try:
        all_chunks = load_chunks(chunk_path)
        intro = all_chunks[: req.intro_chunks]

        retrieved = retrieve_top_k(
            chunks_path=chunk_path,
            embeddings_path=emb_path,
            query="What are the main topics of this document?",
            top_k=req.top_k,
            min_score=req.min_score
        )

        merged = []
        seen = set()

        for c in intro:
            cid = c.get("chunk_id")
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            merged.append({"chunk_id": cid, "content": c.get("content", "")})

        for r in retrieved:
            cid = r.get("chunk_id")
            if cid is None or cid in seen:
                continue
            seen.add(cid)
            merged.append({"chunk_id": cid, "content": r.get("content", "")})

        context = "\n\n".join(
            f"[Chunk {c['chunk_id']}]\n{c['content']}"
            for c in merged
        )

        qs = generate_sample_questions(
            filename=filename,
            context=context,
            max_output_tokens=req.max_output_tokens
        )

        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        return {"filename": filename, "questions": qs[:6]}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))