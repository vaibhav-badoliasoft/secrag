# SecRAG

**SecRAG** (Secure Retrieval-Augmented Generation) is a private, production-ready RAG system for querying PDF documents using semantic, keyword, and hybrid retrieval.

---

# Overview

SecRAG allows users to:

- Upload PDF documents
- Automatically extract and chunk text
- Generate embeddings
- Perform semantic, keyword, or hybrid retrieval
- Generate grounded answers using LLMs
- Run locally or inside Docker
- Operate fully private for interview demos

This project focuses on clean architecture, modularity, explainability, and production-readiness.

---

# Core Capabilities

## 1. PDF Processing

- Upload PDF files
- Extract text using `pypdf`
- Character-based chunking with overlap
- Store chunk metadata:
  - char_start
  - char_end
  - created_at
  - source_path

## 2. Embedding Generation

- Model: `sentence-transformers` (MiniLM)
- 384-dimensional normalized embeddings
- Stored as `.npy` files
- One embedding per chunk

## 3. Retrieval Modes

SecRAG supports three retrieval strategies:

### Semantic (Vector Only)
- Cosine similarity
- Embedding-based ranking
- Best for conceptual queries

### Keyword (BM25)
- Uses `rank-bm25`
- Sparse lexical scoring
- Best for exact-term queries

### Hybrid (Semantic + Keyword)
- Combines semantic similarity and BM25 score
- Weighted fusion ranking
- Best overall performance

Retrieval mode is toggleable from the frontend.

## 4. LLM Answer Generation

- Uses OpenAI API
- Grounded answer generation
- Injects retrieved context only
- Returns citations per chunk

## 5. Frontend Features

- Light/Dark mode toggle
- Document count badge
- Delete document with custom confirmation UI
- Retrieval mode selector
- Real-time backend health indicator
- Chat interface with system/user roles

## 6. Production Features

- Dockerized backend + frontend
- Environment variable configuration
- Request logging middleware
- API key authentication
- Upload size validation
- Health endpoint
- Private deployment mode

---

# Architecture

## High-Level Flow

Client  
→ FastAPI Backend  
→ PDF Upload  
→ Text Extraction  
→ Chunking  
→ Embedding Generation  
→ Artifact Storage  
→ Retrieval (Semantic / Keyword / Hybrid)  
→ Context Injection  
→ LLM Answer Generation  
→ JSON Response  

---

## Backend Structure

```
backend/
│
├── app.py
├── utils/
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── bm25.py
│   └── artifacts.py
│
├── data/
│   ├── uploads/
│   ├── chunks/
│   ├── embeddings/
│
├── requirements.txt
├── Dockerfile
```

Design decisions:

- Modular utilities for separation of concerns
- Artifacts stored locally for fast reload
- Retrieval layer abstracted for mode flexibility
- No database required for demo simplicity
- Designed for future vector DB migration

---

## Frontend Structure

```
frontend/
│
├── src/
│   ├── App.jsx
│   ├── components/
│
├── Dockerfile
```

- Built with React + Vite
- Environment-based API configuration
- Clean two-panel layout (control + chat)
- Dark/light theme toggle
- Production-ready build support

---

# Scalable Options

SecRAG can be extended easily.

## 1. Vector Database

Replace local `.npy` storage with:

- FAISS
- Pinecone
- Weaviate
- Qdrant
- Milvus

This enables:
- Faster retrieval
- Large-scale document handling
- Persistent storage

## 2. Storage

Instead of local file system:
- AWS S3
- Azure Blob Storage
- GCP Cloud Storage

## 3. Authentication

Current:
- Static API key

Upgradeable to:
- JWT-based auth
- OAuth
- Role-based access control

## 4. Deployment

Currently:
- Docker Compose

Scalable to:
- AWS ECS
- Kubernetes
- Azure Container Apps
- Render / Fly.io

## 5. Caching Layer

Can integrate:
- Redis for embedding caching
- Retrieval result caching
- LLM response caching

---

# Environment Configuration

Backend:

```
OPENAI_API_KEY=
SECRAG_API_KEY=
ALLOWED_ORIGINS=http://localhost:5173
MAX_UPLOAD_MB=25
```

Frontend:

```
VITE_API_BASE=http://localhost:8000
VITE_API_KEY=
```

---

# Running Locally

### Development

Backend:
```
uvicorn app:app --reload
```

Frontend:
```
npm run dev
```

### Docker

```
docker compose up --build
```

Access:
- Backend: http://localhost:8000
- Frontend: http://localhost:5173

---

# Why the Name "SecRAG"

SecRAG = Secure Retrieval-Augmented Generation.

- "Sec" represents privacy, security, and controlled environments.
- "RAG" represents the retrieval-augmented architecture.
- Built intentionally for private AI systems rather than public SaaS.

---

# Design Philosophy

- Keep it modular.
- Keep it explainable.
- Keep it private.
- Make it interview-defensible.
- Make it scalable without rewriting the system.

---

# Project Status

Day 11 — Production Ready

- Hybrid retrieval implemented
- Dockerized deployment
- UI enhancements complete
- Logging and validation added
- Private demo mode enabled

---

SecRAG is not just a demo app.  
It is a clean, scalable RAG architecture designed to show understanding of:

- Information retrieval
- Vector search
- Hybrid ranking
- LLM grounding
- Production engineering
- Deployment strategy