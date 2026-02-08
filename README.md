# SecRAG

SecRAG is a modular Retrieval-Augmented Generation (RAG) system for asking questions over PDF documents.

It uploads PDFs, extracts text, chunks it, generates embeddings, performs similarity search, and produces grounded answers with citations using OpenAI.

---

## Current Capabilities (Day 7)

- Upload PDF documents  
- Extract text using `pypdf`  
- Split text into overlapping character-based chunks  
  - chunk size = 500  
  - overlap = 100  
- Store chunk metadata (char offsets, timestamps, source path)  
- Generate 384-dimensional normalized embeddings using MiniLM  
- Save embeddings as `_embedding.npy`  
- Perform cosine similarity search using dot product  
- Retrieve top-k chunks via `/retrieve`  
- Generate grounded answers via `/answer`  
- Return citation metadata:
  - chunk_id  
  - similarity score  
  - character range  

### Frontend (React UI)

- Upload documents from sidebar  
- Select document dynamically  
- Chat interface  
- Confidence scoring (based on top similarity score)  
- Expandable citation viewer  
- Copy answer button  
- Clear chat option  
- Sample question buttons  
- Auto-scroll behavior  

---

## Architecture

Client (React)  
→ FastAPI  
→ PDF Upload  
→ Text Extraction  
→ Chunking  
→ Embedding Generation  
→ Vector Storage  
→ Query Embedding  
→ Similarity Search  
→ Top-K Retrieval  
→ OpenAI  
→ Answer + Citations