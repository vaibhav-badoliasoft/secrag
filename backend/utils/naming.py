from pathlib import Path

def get_artifact_paths(filename: str, data_dir: Path):
    stem = Path(filename).stem
    chunk_path = data_dir / f"{stem}_chunks.json"
    embedding_path = data_dir / f"{stem}_embedding.npy"
    meta_path = data_dir / f"{stem}_meta.json"
    return chunk_path, embedding_path, meta_path