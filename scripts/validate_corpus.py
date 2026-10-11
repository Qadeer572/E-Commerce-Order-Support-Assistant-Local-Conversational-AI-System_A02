"""Validate source document metadata and fixture consistency."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.config import DATA_DIR
from backend.retrieval.corpus import load_documents


def validate():
    documents = load_documents(DATA_DIR / "corpus")
    products = {p["sku"]: p for p in json.loads((DATA_DIR / "products.json").read_text(encoding="utf-8"))}
    orders = {o["order_id"]: o for o in json.loads((DATA_DIR / "orders.json").read_text(encoding="utf-8"))}
    for d in documents:
        if d.doc_type == "product":
            p = products[d.sku]
            assert f"Price: ${p['price']:.2f}" in d.text
            assert f"Stock: {p['stock']} units" in d.text
            assert p["name"] in d.text
        elif d.doc_type == "order":
            o = orders[d.order_id]
            assert f"Status: {o['status']}" in d.text
            assert str(o["total"]) in d.text
        elif d.doc_type == "catalog":
            for line in d.text.splitlines():
                if line.startswith('- '):
                    sku, name, price, category = line[2:].split(' | ')
                    p = products[sku]
                    assert (name, price, category) == (p['name'], f"${p['price']:.2f}", p['category'])
    assert 50 <= len(documents) <= 100
    print(f"Validated {len(documents)} unique documents and fixture facts")


if __name__ == "__main__":
    validate()
