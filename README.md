## Current Capabilities
- Upload PDF documents
- Extract text using pypdf
- Split text into overlapping character-based chunks
- Store structured chunk metadata (char offsets, timestamps, source path)
- Generate 384-dimensional normalized embeddings using MiniLM
- Persist embeddings as NumPy binary files `_embeddings.npy`
- Perform cosine similarity search using dot product
- Retrieve top-k relevant chunks via `/search` endpoint

## Architecture (Current Phase)
Client → FastAPI → PDF Upload → Text Extraction → Chunking → Embedding Generation → Vector Storage → Query Embedding → Similarity Search → Top-K Retrieval

## Next Milestone
- Integrate LLM response generation
- Add citation-style answers using retrieved chunks
- Improve chunking strategy (sentence-aware)
- Optimize retrieval performance
