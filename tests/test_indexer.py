import json
import pytest
from backend.retrieval.indexer import index_corpus
from tests.test_corpus import Words


class MemoryStore:
    def __init__(self, path):
        self.path, self.chunks = path, {}

    def upsert(self, chunks, embeddings):
        self.chunks.update({c.chunk_id: c for c in chunks})

    def delete_ids(self, ids):
        for id in ids:
            self.chunks.pop(id, None)


class Embedder:
    tokenizer = Words()
    revision = "test"
    dimension = 3

    def encode(self, texts):
        return [[1., 0., 0.] for t in texts]


def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "one.md").write_text("first " * 500)
    (root / "manifest.json").write_text(json.dumps({"documents": [dict(doc_id="one",title="One",source="one.md",doc_type="product",provenance="test")]}))
    return root


def test_index_rerun_and_shortening_remove_obsolete_chunks(tmp_path):
    root = corpus(tmp_path)
    store = MemoryStore(tmp_path / "index")
    report = index_corpus(root, store, Embedder())
    assert report.added == 1
    count = len(store.chunks)
    assert index_corpus(root, store, Embedder()).unchanged == 1
    assert len(store.chunks) == count
    (root / "one.md").write_text("shorter")
    assert index_corpus(root, store, Embedder()).updated == 1
    assert len(store.chunks) == 1
    (root / "one.md").unlink()
    (root / "manifest.json").write_text('{"documents": []}')
    assert index_corpus(root, store, Embedder()).deleted == 1
    assert not store.chunks


def test_embedding_failure_preserves_complete_revision(tmp_path):
    root = corpus(tmp_path)
    store = MemoryStore(tmp_path / "index")
    index_corpus(root, store, Embedder())
    before = dict(store.chunks)
    (root / "one.md").write_text("changed")
    class Broken(Embedder):
        def encode(self, texts):
            raise RuntimeError("embedding failed")
    with pytest.raises(RuntimeError):
        index_corpus(root, store, Broken())
    assert store.chunks == before
    assert not (store.path / "incomplete").exists()
