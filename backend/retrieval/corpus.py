"""Validated sources and deterministic, tokenizer-bounded chunks."""
import hashlib
import json
import re
from pathlib import Path

from .types import Chunk, Document

CHUNK_VERSION = "sections-v2-original-220-32"


def load_documents(root: Path) -> list[Document]:
    root = root.resolve()
    entries = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["documents"]
    documents, seen = [], set()
    for entry in entries:
        if not all(entry.get(k) for k in ("doc_id", "title", "source", "doc_type", "provenance")):
            raise ValueError("Missing document metadata")
        target = (root / entry["source"]).resolve()
        if not target.is_relative_to(root) or target.suffix != ".md":
            raise ValueError("Invalid corpus path")
        if entry["doc_id"] in seen:
            raise ValueError("Duplicate document ID")
        seen.add(entry["doc_id"])
        text = target.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError("Empty document")
        documents.append(Document(**{k: entry.get(k, "") for k in
                                     ("doc_id", "title", "source", "doc_type", "sku", "order_id")}, text=text))
    if {d.source for d in documents} != {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*.md")}:
        raise ValueError("Manifest must include every Markdown source exactly once")
    return documents


def chunk_document(document, tokenizer, *, max_tokens=220, overlap=32):
    prefix = f"{document.title} {document.order_id or document.sku}\n"
    capacity = max_tokens - len(tokenizer.encode(prefix, add_special_tokens=False)) - 4
    if capacity <= overlap:
        raise ValueError("Title leaves no chunk capacity")
    chunks = []
    for section in re.split(r"(?m)(?=^## )", document.text):
        tokens = tokenizer.encode(section, add_special_tokens=False)
        if getattr(tokenizer, 'is_fast', False):
            offsets = tokenizer(section, add_special_tokens=False,
                                return_offsets_mapping=True)['offset_mapping']
        else:
            # Word-tokenizer doubles used by the offline indexing tests.
            offsets = [(m.start(), m.end()) for m in re.finditer(r'\S+', section)]
            if len(offsets) != len(tokens):
                raise ValueError('Chunking requires a tokenizer with character offsets')
        start = 0
        while start < len(tokens):
            end = min(start + capacity, len(tokens))
            def source_slice(end):
                left = 0 if start == 0 else offsets[start][0]
                right = len(section) if end == len(tokens) else offsets[end-1][1]
                return prefix + section[left:right]
            text = source_slice(end)
            while len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
                end -= 1
                if end <= start:
                    raise ValueError('Source token cannot fit chunk budget')
                text = source_slice(end)
            digest = hashlib.sha256((CHUNK_VERSION + text).encode()).hexdigest()[:20]
            ordinal = len(chunks)
            chunks.append(Chunk(f"{document.doc_id}:{ordinal}:{digest}", document, text, ordinal))
            if end == len(tokens):
                break
            start = max(start + 1, end - overlap)
    return chunks
