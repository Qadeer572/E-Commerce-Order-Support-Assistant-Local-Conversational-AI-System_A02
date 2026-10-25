"""SQLite copy of the product fixtures that the catalog_query tool reads.

RAG documents are static snapshots; this database is the "live" source for price and
stock. It is seeded from data/products.json and can be edited with scripts/set_stock.py.
"""
import json
import sqlite3
from pathlib import Path

from backend.config import PRODUCTS_FILE, SHOP_DB_FILE

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
    price REAL NOT NULL, stock INTEGER NOT NULL, description TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS variants (
    sku TEXT NOT NULL REFERENCES products(sku), label TEXT NOT NULL, stock INTEGER NOT NULL,
    PRIMARY KEY (sku, label)
);
"""


def connect(db_path: Path = SHOP_DB_FILE) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def variant_label(variant: dict) -> str:
    """'Midnight Black' or 'Black / Large': every variant field except stock."""
    return " / ".join(str(value) for key, value in variant.items() if key != "stock")


def build_database(db_path: Path = SHOP_DB_FILE, products_file: Path = PRODUCTS_FILE) -> int:
    """Create the schema and upsert every product. Returns the number of products."""
    products = json.loads(Path(products_file).read_text(encoding="utf-8"))
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for item in products:
            conn.execute(
                "INSERT INTO products(sku, name, category, price, stock, description) VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(sku) DO UPDATE SET name=excluded.name, category=excluded.category, "
                "price=excluded.price, stock=excluded.stock, description=excluded.description",
                (item["sku"], item["name"], item["category"], item["price"], item["stock"],
                 item.get("description", "")))
            for variant in item.get("variants", []):
                conn.execute(
                    "INSERT INTO variants(sku, label, stock) VALUES (?,?,?) "
                    "ON CONFLICT(sku, label) DO UPDATE SET stock=excluded.stock",
                    (item["sku"], variant_label(variant), variant["stock"]))
    conn.close()
    return len(products)


def ensure_database(db_path: Path = SHOP_DB_FILE, products_file: Path = PRODUCTS_FILE) -> None:
    """Seed the database only when it does not exist, so manual stock edits survive restarts."""
    if not Path(db_path).exists():
        build_database(db_path, products_file)


def set_stock(sku: str, label: str, stock: int, db_path: Path = SHOP_DB_FILE) -> bool:
    """Set one variant's stock and recompute the product total. Returns False if not found."""
    with connect(db_path) as conn:
        changed = conn.execute("UPDATE variants SET stock=? WHERE sku=? AND label=?", (stock, sku, label)).rowcount
        if changed:
            conn.execute("UPDATE products SET stock=(SELECT SUM(stock) FROM variants WHERE sku=?) WHERE sku=?",
                         (sku, sku))
    conn.close()
    return bool(changed)
