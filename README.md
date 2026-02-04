## Current Capabilities
- Upload PDF documents
- Extract text from PDFs using pypdf
- Persist original PDF and extracted text locally
- Split extracted text into overlapping character-based chunks
- Save structured chunk data as JSON (`_chunks.json`)
- Return structured JSON response with total characters and total chunks

## Architecture (Current Phase)
Client → FastAPI → File Upload → Save PDF → PDF Parsing → Save Text → Chunking (with overlap) → Save Chunks JSON → JSON Response

## Next Milestone
- Add chunk metadata (char_start, char_end, created_at, source_path)
- Generate embeddings for each chunk
- Implement vector similarity search
