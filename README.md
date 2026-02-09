## Current Capabilities

### Backend
- PDF upload and persistent storage
- Text extraction using `pypdf`
- Character-based chunking (size = 500, overlap = 100)
- 384-dimensional normalized embeddings using MiniLM (`all-MiniLM-L6-v2`)
- Cosine similarity search via NumPy dot product
- In-memory caching of:
  - Chunks (JSON)
  - Embeddings (NumPy)
- `_meta.json` artifact per document
- `/retrieve` endpoint (top-k similarity results)
- `/answer` endpoint (grounded responses + citations + preview)
- `/summarize` endpoint (Hybrid strategy: intro + retrieved chunks)
- `/sample_questions` endpoint (dynamic, document-aware question generation)
- `/stats` endpoint (document stats + cache status)
- Citation metadata:
  - chunk_id
  - similarity score
  - character range
  - preview text
  - source type (intro / retrieved / hybrid)

### Frontend (React)
- Document upload and dynamic selection
- Chat-style interface
- Confidence indicator (based on top similarity score)
- Right-side citation drawer (enterprise-style)
- Copy answer button
- Hybrid document summarization
- Dynamic sample questions (generated per document)
- Loading skeletons
- Toast-style error notifications
- Clear chat option

---

## Architecture

React Client  
→ FastAPI Backend  
→ PDF Extraction (`pypdf`)  
→ Chunking (500 / 100 overlap)  
→ Embedding Generation (MiniLM)  
→ Vector Similarity Search (NumPy)  
→ Retrieval  
→ OpenAI (Answer / Summary / Sample Questions)  
→ Structured Citations + UI Rendering  

---

## Design Philosophy

- Transparent  
  - Every answer is backed by chunk-level citations  
  - Confidence signal derived from similarity score  

- Modular  
  - `app.py` → API routing  
  - `uploader.py` → ingestion  
  - `retriever.py` → similarity search + caching  
  - `llm.py` → OpenAI interaction  
  - `summarizer.py` → hybrid summarization logic  
  - `naming.py` → artifact consistency  

- Cost-efficient  
  - Local embeddings (MiniLM)  
  - No vector database required  
  - OpenAI used only for generation (not embeddings)  

- Performance-aware  
  - In-memory caching of embeddings and chunks  
  - Auto invalidation via file modification time  
  - Hybrid retrieval for stable summarization  

- Interview-ready  
  - Manual RAG implementation (no LangChain)  
  - Clear data flow  
  - Transparent cosine similarity  
  - Hybrid reasoning strategy  
  - Stats endpoint for credibility  

---

## Hybrid Strategies Used

### Hybrid Summarization
- First N intro chunks (stable context framing)
- Top-k retrieved chunks (document-wide relevance)
- Deduplicated by `chunk_id`

### Hybrid Sample Questions
- Intro chunks
- Top retrieved topic chunks
- Context-aware OpenAI prompt
- Returns 6 document-specific questions

---

## API Endpoints

GET `/health`  
GET `/list_docs`  
GET `/stats?filename=...`  

POST `/upload`  
POST `/retrieve`  
POST `/answer`  
POST `/summarize`  
POST `/sample_questions`

---

## Artifacts Per Document

For `example.pdf`, the system generates:

- `example_chunks.json`
- `example_embedding.npy`
- `example_meta.json`

These enable:
- Retrieval
- Stats reporting
- Cache validation
- Reproducibility
