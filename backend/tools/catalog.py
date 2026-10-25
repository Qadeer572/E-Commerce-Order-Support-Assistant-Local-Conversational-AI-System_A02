"""catalog_query tool: parameterized queries over the live SQLite product database."""
import asyncio
import re
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from backend import shop_db
from backend.config import SHOP_DB_FILE
from .base import Tool, ToolContext, ToolResult

CATEGORIES = ("Accessories", "Apparel", "Books", "Electronics", "Home", "Sports")
_STOPWORDS = {"the", "a", "an", "is", "are", "in", "of", "for", "still", "any", "do", "you", "have",
              "stock", "available", "price", "how", "many", "left", "much", "does", "cost", "what",
              "show", "me", "your", "it", "there", "color", "colour", "colors", "and", "to", "with"}
_SORTS = {"price_asc": "price ASC", "price_desc": "price DESC", "name": "name ASC"}


class CatalogArgs(BaseModel):
    query: Optional[str] = Field(default=None, max_length=80, description="Product name or keywords")
    sku: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3,5}-\d{3}$", description="Exact SKU")
    category: Optional[Literal["Accessories", "Apparel", "Books", "Electronics", "Home", "Sports"]] = None
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    in_stock_only: bool = False
    sort: Literal["price_asc", "price_desc", "name"] = "price_asc"
    limit: int = Field(default=5, ge=1, le=10)


def _keywords(query: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", query.lower())
    return [w[:-1] if w.endswith("s") and len(w) > 3 else w for w in words if w not in _STOPWORDS]


def search(args: CatalogArgs, db_path: Path) -> list[dict]:
    """Build a parameterized query. User text only ever travels as bound parameters."""
    clauses, params = [], []
    sku_match = re.search(r"\b([A-Z]{3,5}-\d{3})\b", (args.query or ""), re.I)
    target_sku = args.sku or (sku_match.group(1).upper() if sku_match else None)
    if target_sku:
        clauses.append("sku = ?")
        params.append(target_sku)
    if args.category:
        clauses.append("category = ?")
        params.append(args.category)
    if args.min_price is not None:
        clauses.append("price >= ?")
        params.append(args.min_price)
    if args.max_price is not None:
        clauses.append("price <= ?")
        params.append(args.max_price)
    if args.in_stock_only:
        clauses.append("stock > 0")
    search_text = args.query if (not sku_match and args.query) else ""
    for word in _keywords(search_text):
        clauses.append("(sku LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\' OR category LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\')")
        pattern = "%" + word.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        params.extend([pattern] * 4)
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    sql = f"SELECT sku, name, category, price, stock FROM products{where} ORDER BY {_SORTS[args.sort]} LIMIT ?"
    conn = shop_db.connect(db_path)
    try:
        rows = conn.execute(sql, [*params, args.limit]).fetchall()
        result = []
        for row in rows:
            variants = conn.execute("SELECT label, stock FROM variants WHERE sku = ? ORDER BY label",
                                    (row["sku"],)).fetchall()
            result.append({"sku": row["sku"], "name": row["name"], "category": row["category"],
                           "price": row["price"], "total_stock": row["stock"],
                           "variants": [{"label": v["label"], "stock": v["stock"]} for v in variants]})
        return result
    finally:
        conn.close()


class CatalogTool(Tool):
    name = "catalog_query"
    description = ("Looks up products in the live store database: price, stock, variants, category filters. "
                   "Use for 'is it in stock', 'how many left', 'cheapest X', 'under $N' questions. "
                   "Do NOT use for shipping or return policy questions.")
    args_model = CatalogArgs
    timeout_s = 2.0
    timeout_message = "The stock lookup did not respond in time. Please try again, or check the product page."
    gate_pattern = re.compile(
        r"\b(?:in stock|stock|how many (?:are )?left|available|availability|price of|cheaper than|cheapest|"
        r"under \$?\d+|below \$?\d+|less than \$?\d+|how much (?:is|does|are|do))\b", re.I)
    examples = [
        ("is the ProSound earbuds in navy blue still in stock", {"query": "ProSound earbuds"}),
        ("show electronics under $50 sorted by price", {"category": "Electronics", "max_price": 50}),
    ]

    def __init__(self, db_path: Path = SHOP_DB_FILE):
        self.db_path = Path(db_path)

    def fallback_arguments(self, text: str, ctx: ToolContext) -> Optional[dict]:
        args: dict = {}
        sku = re.search(r"\b([A-Z]{3,5}-\d{3})\b", text.upper())
        if sku:
            args["sku"] = sku.group(1)
        under = re.search(r"\b(?:under|below|less than)\s*\$?(\d+(?:\.\d+)?)", text, re.I)
        if under:
            args["max_price"] = float(under.group(1))
        for category in CATEGORIES:
            if category.lower() in text.lower():
                args["category"] = category
        if not args:
            words = " ".join(_keywords(text))
            if words:
                args["query"] = words
        return args or None

    async def run(self, args: CatalogArgs, ctx: ToolContext) -> ToolResult:
        shop_db.ensure_database(self.db_path)
        rows = await asyncio.to_thread(search, args, self.db_path)
        if not rows:
            return ToolResult(ok=True, data={"count": 0, "products": [], "note": "No matching products."})
        return ToolResult(ok=True, data={"count": len(rows), "products": rows})
