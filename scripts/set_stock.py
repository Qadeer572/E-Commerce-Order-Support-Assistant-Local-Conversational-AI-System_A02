"""CLI tool to view, update, or rebuild the live SQLite shop catalog (data/shop.db).

Usage:
  python scripts/set_stock.py --list
  python scripts/set_stock.py --sku ELEC-001 --stock 0
  python scripts/set_stock.py --sku ELEC-001 --label "Midnight Black" --stock 5
  python scripts/set_stock.py --rebuild
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.shop_db import connect, build_database, ensure_database, set_stock
from backend.config import SHOP_DB_FILE


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def list_catalog(db_path=SHOP_DB_FILE):
    ensure_database(db_path)
    with connect(db_path) as conn:
        products = conn.execute("SELECT sku, name, price, stock FROM products ORDER BY sku").fetchall()
        print(f"\n{'SKU':<12} {'Price':<8} {'Stock':<8} {'Name'}")
        print("-" * 60)
        for p in products:
            print(f"{p['sku']:<12} ${p['price']:<7.2f} {p['stock']:<8} {p['name']}")
            variants = conn.execute("SELECT label, stock FROM variants WHERE sku=? ORDER BY label",
                                    (p["sku"],)).fetchall()
            for v in variants:
                print(f"  |- {v['label']}: {v['stock']} in stock")
        print("-" * 60)
        print(f"Total: {len(products)} products\n")


def update_stock(sku: str, stock: int, label: str | None = None, db_path=SHOP_DB_FILE):
    ensure_database(db_path)
    with connect(db_path) as conn:
        prod = conn.execute("SELECT sku, name, stock FROM products WHERE sku=?", (sku,)).fetchone()
        if not prod:
            print(f"Error: SKU '{sku}' not found in database.")
            sys.exit(1)

        variants = conn.execute("SELECT label, stock FROM variants WHERE sku=?", (sku,)).fetchall()

        if label:
            changed = set_stock(sku, label, stock, db_path=db_path)
            if not changed:
                print(f"Error: Variant '{label}' not found for SKU '{sku}'.")
                sys.exit(1)
            print(f"Updated {sku} ('{label}') stock to {stock}.")
        elif variants:
            # If product has variants and no specific variant specified, update all or prompt
            if len(variants) == 1:
                set_stock(sku, variants[0]["label"], stock, db_path=db_path)
                print(f"Updated {sku} ('{variants[0]['label']}') stock to {stock}.")
            else:
                conn.execute("UPDATE products SET stock=? WHERE sku=?", (stock, sku))
                print(f"Updated {sku} product stock to {stock} (variants retain individual counts).")
        else:
            conn.execute("UPDATE products SET stock=? WHERE sku=?", (stock, sku))
            print(f"Updated {sku} product stock to {stock}.")


def main():
    parser = argparse.ArgumentParser(description="View or update live stock in data/shop.db")
    parser.add_argument("--list", action="store_true", help="List all products and their variant stock")
    parser.add_argument("--sku", type=str, help="Product SKU to update (e.g., ELEC-001)")
    parser.add_argument("--label", type=str, default=None, help="Specific variant label (e.g., 'Midnight Black')")
    parser.add_argument("--stock", type=int, help="New stock count")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild shop.db from data/products.json")

    args = parser.parse_args()

    if args.rebuild:
        count = build_database()
        print(f"Successfully rebuilt {SHOP_DB_FILE} with {count} products.")
        return

    if args.list or (args.sku is None and args.stock is None):
        list_catalog()
        return

    if args.sku and args.stock is not None:
        update_stock(args.sku.upper(), args.stock, args.label)
    else:
        print("Error: Both --sku and --stock are required to update stock. Run with --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
