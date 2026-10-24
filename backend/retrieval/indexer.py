"""Incremental document indexing with pre-mutation embedding and recovery marker."""
import hashlib
import json
from .corpus import CHUNK_VERSION, load_documents, chunk_document
from .types import IndexReport


def index_corpus(corpus_dir, store, embedder, *, rebuild=False):
    path = store.path
    path.mkdir(parents=True, exist_ok=True)
    file = path / "manifest.json"
    old = json.loads(file.read_text()) if file.exists() else {}
    settings = dict(model_revision=embedder.revision,dimension=embedder.dimension,chunker=CHUNK_VERSION)
    if old and old["settings"] != settings and not rebuild:
        raise ValueError("Model/chunker changed; pass --rebuild")
    if (path / "incomplete").exists() and not rebuild:
        raise ValueError("Incomplete index; pass --rebuild")
    previous = old.get("documents", {})
    documents = load_documents(corpus_dir)
    current, pending, changed, unchanged, added, updated = {}, [], [], 0, 0, 0
    for d in documents:
        fingerprint = hashlib.sha256(json.dumps(d.__dict__,sort_keys=True).encode()).hexdigest()
        prior = previous.get(d.doc_id)
        if prior and prior["hash"] == fingerprint and not rebuild:
            current[d.doc_id] = prior
            unchanged += 1
            continue
        chunks = chunk_document(d, embedder.tokenizer)
        pending.extend(chunks)
        changed.append(d.doc_id)
        current[d.doc_id] = dict(hash=fingerprint, chunks=[c.chunk_id for c in chunks])
        if prior:
            updated += 1
        else:
            added += 1
    deleted = set(previous) - set(current)
    if not changed and not deleted and not rebuild:
        return IndexReport(0,0,0,unchanged,old.get("revision",""))
    embeddings = embedder.encode([c.text for c in pending]) if pending else []
    if len(embeddings) != len(pending):
        raise ValueError("Embedding count mismatch")
    marker = path / "incomplete"
    marker.write_text("Index update in progress; rebuild if interrupted")
    if rebuild and hasattr(store,"collection"):
        all_ids = store.collection.get()["ids"]
        store.delete_ids(all_ids)
    store.upsert(pending, embeddings)
    obsolete = [id for doc_id in changed + list(deleted) for id in previous.get(doc_id,{}).get("chunks",[])
                if id not in current.get(doc_id,{}).get("chunks",[])]
    store.delete_ids(obsolete)
    revision = hashlib.sha256(json.dumps([settings,current],sort_keys=True).encode()).hexdigest()
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(dict(settings=settings,documents=current,revision=revision),indent=2))
    temp.replace(file)
    marker.unlink()
    store.revision = revision
    return IndexReport(added,updated,len(deleted),unchanged,revision)
