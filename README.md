## Current Capabilities

### Backend
- PDF upload and storage
- Text extraction using `pypdf`
- Character-based chunking (size = 500, overlap = 100)
- 384-dimensional normalized embeddings using MiniLM
- Cosine similarity search via NumPy dot product
- `/retrieve` endpoint (top-k similarity results)
- `/answer` endpoint (grounded responses + citations)
- `/summarize` endpoint (Hybrid strategy: intro + retrieved chunks)
- Citation metadata:
  - chunk_id
  - similarity score
  - character range
  - source type (intro / retrieved / hybrid)

### Frontend (React)
- Document upload and dynamic selection
- Chat-style interface
- Confidence indicator (based on similarity score)
- Expandable citation viewer
- Copy answer button
- Hybrid document summarization
- Clear chat option
- Sample prompt buttons

---

## Architecture

React Client  
→ FastAPI Backend  
→ PDF Extraction  
→ Chunking  
→ Embedding Generation (MiniLM)  
→ Vector Similarity Search (NumPy)  
→ Retrieval  
→ OpenAI  
→ Answer / Summary + Citations  

---

## Design Philosophy

- Transparent (citation-backed responses)
- Modular (clear separation of API, retrieval, generation)
- Cost-efficient (local embeddings)
- Interview-ready (manual RAG implementation without heavy abstraction frameworks)