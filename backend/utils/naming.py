from pathlib import Path


def get_artifact_paths(filename: str, data_dir: Path):
    """
    Given a filename like 'abc.pdf',
    returns paths for:
      - chunks.json
      - embedding.npy
    using Path.stem for safety.
    """
    stem = Path(filename).stem

    chunk_path = data_dir / f"{stem}_chunks.json"
    embedding_path = data_dir / f"{stem}_embedding.npy"

    return chunk_path, embedding_path
