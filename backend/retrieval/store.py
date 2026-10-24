"""Explicit-vector Chroma persistence and cross-process ownership."""
import json
from pathlib import Path
from .types import Hit


class VectorStore:
    def __init__(self, path: Path, *, reader=False):
        import chromadb
        from chromadb.config import Settings
        from filelock import FileLock
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path / "ownership.lock"))
        self.lock.acquire(timeout=0)
        try:
            if reader and ((self.path / "incomplete").exists() or not (self.path / "manifest.json").exists()):
                raise RuntimeError("Index missing or incomplete; run index_documents.py")
            self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
            self.collection = self.client.get_or_create_collection("shopbot", metadata={"hnsw:space": "cosine"}, embedding_function=None)
            self.revision = self.manifest.get("revision", "")
        except Exception:
            self.lock.release()
            raise

    @property
    def manifest(self):
        p = self.path / "manifest.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def upsert(self, chunks, embeddings):
        if not chunks:
            return
        self.collection.upsert(ids=[c.chunk_id for c in chunks], embeddings=embeddings,
            documents=[c.text for c in chunks], metadatas=[dict(doc_id=c.document.doc_id,
            source=c.document.source,title=c.document.title,doc_type=c.document.doc_type,
            sku=c.document.sku,order_id=c.document.order_id) for c in chunks])

    def delete_ids(self, ids):
        if ids:
            self.collection.delete(ids=ids)

    def delete_document(self, doc_id):
        self.collection.delete(where={"doc_id": doc_id})

    def search(self, embedding, *, top_k, filters):
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k,
                                       where=filters or None, include=["documents", "metadatas", "distances"])
        return [Hit(id, m["source"], m["title"], text, 1-float(distance), m["doc_type"],m["sku"],m["order_id"])
                for id,m,text,distance in zip(result["ids"][0],result["metadatas"][0],result["documents"][0],result["distances"][0])]

    def close(self):
        self.lock.release()
