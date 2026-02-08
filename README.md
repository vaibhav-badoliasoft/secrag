## Current Capabilities
- Upload PDF documents
- Extract text using pypdf
- Split text into overlapping character-based chunks (size=500, overlap=100)
- Store structured chunk metadata (char offsets, timestamps, source path)
- Generate 384-dimensional normalized embeddings using MiniLM
- Persist embeddings as NumPy binary files `_embedding.npy`
- Perform cosine similarity search using dot product
- Retrieve top-k relevant chunks via `/retrieve` endpoint
- Generate grounded answers using OpenAI via `/answer` endpoint
- Return citation metadata (chunk_id, similarity score, char range)

## Architecture (Current Phase)
Client → FastAPI → PDF Upload → Text Extraction → Chunking → Embedding Generation → Vector Storage → Query Embedding → Similarity Search → Top-K Retrieval → OpenAI Generation → Answer + Citations

## Next Milestone
- Build ChatGPT-style React UI
- Add enterprise-style citation panel
- Implement confidence scoring
- Improve chunking strategy (sentence-aware)
- Optimize retrieval performance