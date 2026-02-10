## ✅ Current Capabilities

### Backend
- PDF upload and storage
- Text extraction using `pypdf`
- **Sentence-aware chunking** (packs full sentences into chunks)
  - chunk_size ≈ 500 chars
  - overlap by last 1 sentence
- 384-dimensional normalized embeddings using MiniLM (`all-MiniLM-L6-v2`)
- Vector similarity search using NumPy dot product (cosine via normalized vectors)
- **BM25 keyword search** using `rank-bm25`
- **Hybrid retrieval (BM25 + embeddings)** with weighted scoring
- Retrieval modes supported (toggleable):
  - `semantic` → embeddings only
  - `bm25` → keyword only
  - `hybrid` → weighted merge
- `/retrieve` endpoint (top-k chunks)
- `/answer` endpoint (grounded response + citations)
- `/summarize` endpoint (Hybrid summarize: intro chunks + retrieved chunks)
- Logging middleware:
  - request id
  - latency
  - `X-Request-ID` response header

### Citation Metadata
- chunk_id
- similarity score (0–1 in Day 10 hybrid scoring)
- character range (char_start, char_end)
- source type for summarize (intro / retrieved / hybrid)

---

## 🔥 Architecture (Day 10)

React Client  
→ FastAPI Backend  
→ PDF Extraction (`pypdf`)  
→ Sentence Chunking  
→ Embeddings (MiniLM)  
→ Retrieval (Semantic / BM25 / Hybrid)  
→ OpenAI (Answer / Summary)  
→ Citations + Confidence  