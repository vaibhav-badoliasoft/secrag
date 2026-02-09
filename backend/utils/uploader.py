import os
import json
import numpy as np
from datetime import datetime
from pypdf import PdfReader
from pathlib import Path

from utils.chunking import chunk_text
from utils.embeddings import embed_texts


def process_pdf_upload(file_path: str, data_dir: str):
    reader = PdfReader(file_path)

    extracted_text = ""
    for page in reader.pages:
        extracted_text += (page.extract_text() or "") + "\n"

    text_filename = os.path.basename(file_path).replace(".pdf", ".txt")
    text_path = os.path.join(data_dir, text_filename)
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)

    chunks = chunk_text(extracted_text)
    created_at = datetime.utcnow().isoformat()

    chunk_data = []
    for index, (char_start, char_end, chunk) in enumerate(chunks):
        chunk_data.append({
            "chunk_id": index,
            "filename": os.path.basename(file_path),
            "source_path": file_path,
            "created_at": created_at,
            "char_start": char_start,
            "char_end": char_end,
            "content": chunk
        })

    chunk_filename = os.path.basename(file_path).replace(".pdf", "_chunks.json")
    chunk_path = os.path.join(data_dir, chunk_filename)
    with open(chunk_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2)

    texts = [c["content"] for c in chunk_data]
    vectors = embed_texts(texts)

    stem = Path(file_path).stem
    emb_filename = f"{stem}_embedding.npy"
    emb_path = os.path.join(data_dir, emb_filename)
    np.save(emb_path, vectors)

    meta = {
        "filename": os.path.basename(file_path),
        "stem": stem,
        "created_at": created_at,
        "total_characters": len(extracted_text),
        "total_chunks": len(chunk_data),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "chunk_file": os.path.basename(chunk_path),
        "embedding_file": os.path.basename(emb_path),
    }
    meta_path = os.path.join(data_dir, f"{stem}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return {
        "filename": os.path.basename(file_path),
        "total_characters": len(extracted_text),
        "total_chunks": len(chunk_data),
        "embedding_dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "created_at": created_at,
        "first_chunk_preview": chunk_data[0]["content"][:200] if chunk_data else ""
    }