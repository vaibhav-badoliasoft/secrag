## Current Capabilities
- Upload PDF documents
- Extract text from PDFs using pypdf
- Persist original PDF and extracted text locally
- Split extracted text into overlapping character-based chunks
- Store structured chunk data with metadata:
  - char_start
  - char_end
  - created_at
  - source_path
- Generate embeddings (384-dimensional) for each chunk
- Save embeddings as NumPy binary files (`_embeddings.npy`)
- Return structured JSON response with total characters, total chunks, and embedding dimension

## Architecture (Current Phase)
Client → FastAPI → File Upload → Save PDF → PDF Parsing → Save Text → Chunking (with overlap + metadata) → Embedding Generation → Save Embeddings → JSON Response

## Next Milestone
- Implement vector similarity search
- Add query embedding endpoint
- Retrieve top-k relevant chunks
- Integrate retrieval with LLM response generation
