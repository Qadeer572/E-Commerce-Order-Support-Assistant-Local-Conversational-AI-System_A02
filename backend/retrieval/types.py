from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    source: str
    text: str
    doc_type: str
    sku: str = ""
    order_id: str = ""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: Document
    text: str
    ordinal: int


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    source: str
    title: str
    text: str
    similarity: float
    doc_type: str
    sku: str = ""
    order_id: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    status: Literal["ok", "no_match", "unavailable", "timeout", "busy"]
    hits: list[Hit]
    elapsed_ms: float
    cache_hit: bool = False
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexReport:
    added: int
    updated: int
    deleted: int
    unchanged: int
    revision: str
