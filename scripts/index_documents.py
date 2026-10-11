"""Build/update the local index while the backend is stopped."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import CORPUS_DIR, VECTOR_DIR
from backend.retrieval.embeddings import LocalEmbedder
from backend.retrieval.store import VectorStore
from backend.retrieval.indexer import index_corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    store = VectorStore(VECTOR_DIR)
    try:
        print(index_corpus(CORPUS_DIR, store, LocalEmbedder(), rebuild=args.rebuild))
    finally:
        store.close()
