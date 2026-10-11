from pathlib import Path

from backend.retrieval.corpus import load_documents, chunk_document
from backend.retrieval.types import Document

ROOT = Path(__file__).resolve().parents[1]


class Words:
    def encode(self, text, **kwargs):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


def test_corpus_has_63_unique_documents():
    documents = load_documents(ROOT / "data/corpus")
    assert len(documents) == 63
    assert len({d.doc_id for d in documents}) == 63


def test_chunks_preserve_order_id_and_fit_embedding_limit():
    documents = load_documents(ROOT / "data/corpus")
    for document in documents:
        chunks = chunk_document(document, Words())
        assert chunks
        assert all(len(Words().encode(c.text)) <= 220 for c in chunks)
        if document.order_id:
            assert all(document.order_id in c.text for c in chunks)


def test_chunk_text_preserves_source_prices_case_and_line_breaks():
    class LossyWords(Words):
        def decode(self, tokens, **kwargs):
            return ' '.join(tokens).lower().replace('$9.99', '$ 9. 99')
    text = '# CablePro\n\n- ACC-002 | CablePro | $9.99 | Accessories'
    doc = Document('catalog', 'Catalog', 'catalog.md', text, 'catalog')
    chunks = chunk_document(doc, LossyWords())
    assert text in chunks[0].text


def test_oversized_chunks_preserve_original_source_substrings():
    class LossyWords(Words):
        def decode(self, tokens, **kwargs):
            return ' '.join(tokens).lower()
    text = ' '.join(f'Price{i}: $9.99' for i in range(100))
    doc = Document('prices', 'Prices', 'prices.md', text, 'product')
    chunks = chunk_document(doc, LossyWords(), max_tokens=30, overlap=5)
    assert len(chunks) > 1
    assert all(c.text.split('\n', 1)[1] in text for c in chunks)
