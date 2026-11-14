"""Bundle the exact multilingual embedder so production needs no model download."""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
from pathlib import Path
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "models" / "minilm"
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", revision="e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
    assert model.get_sentence_embedding_dimension() == 384
    model.save(str(target))
    print("Bundled multilingual MiniLM (384 dimensions)")
