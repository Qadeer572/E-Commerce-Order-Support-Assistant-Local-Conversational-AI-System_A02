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
_CATEGORY_ALIASES = {'home goods': 'Home', 'home and kitchen': 'Home',
                     'sports equipment': 'Sports', 'clothing': 'Apparel',
                     **{category.lower(): category for category in CATEGORIES}}
_STOPWORDS = {"the", "a", "an", "is", "are", "in", "of", "for", "still", "any", "do", "you", "have",
              "stock", "available", "price", "how", "many", "left", "much", "does", "cost", "what",
              "show", "me", "your", "it", "there", "color", "colour", "colors", "and", "to", "with"}
_BROWSE_WORDS = {'please', 'provide', 'give', 'tell', 'list', 'all', 'products', 'product', 'items', 'item',
                 'store', 'shop', 'catalog', 'catalogue', 'currently', 'current', 'that', 'this', 'i',
                 'am', 'telling', 'want', 'need', 'can', 'could', 'would', 'now', 'only', 'options',
                 'cheap', 'cheapest', 'affordable', 'budget', 'inexpensive', 'sorted', 'sort', 'by',
                 'ascending', 'descending', 'first', 'most', 'expensive', 'highest', 'lowest'}
_BROWSE_WORDS.update({'recommend', 'suggest', 'something', 'buy', 'purchase', 'sell', 'carry',
                      'offer', 'offering', 'selection', 'range', 'option', 'some', 'priced',
                      'costing', 'full', 'complete', 'entire', 'at', 'here', 'our', 'right'})
_SORTS = {"price_asc": "price ASC", "price_desc": "price DESC", "name": "name ASC"}


class CatalogArgs(BaseModel):
    query: Optional[str] = Field(default=None, max_length=80, description="Product name or keywords")
    sku: Optional[str] = Field(default=None, pattern=r"^[A-Z]{3,5}-\d{3}$", description="Exact SKU")
    category: Optional[Literal["Accessories", "Apparel", "Books", "Electronics", "Home", "Sports"]] = None
    categories: list[Literal["Accessories", "Apparel", "Books", "Electronics", "Home", "Sports"]] = Field(default_factory=list, max_length=6)
    min_price: Optional[float] = Field(default=None, ge=0)
    max_price: Optional[float] = Field(default=None, ge=0)
    in_stock_only: bool = False
    out_of_stock_only: bool = False
    sort: Literal["price_asc", "price_desc", "name"] = "price_asc"
    limit: int = Field(default=5, ge=1, le=20)


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
    if args.categories:
        clauses.append('category IN (' + ','.join('?' for _ in args.categories) + ')')
        params.extend(args.categories)
    if args.min_price is not None:
        clauses.append("price >= ?")
        params.append(args.min_price)
    if args.max_price is not None:
        clauses.append("price <= ?")
        params.append(args.max_price)
    if args.in_stock_only:
        clauses.append("stock > 0")
    if args.out_of_stock_only:
        clauses.append("stock <= 0")
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

    def gate(self, text: str) -> bool:
        from backend.conversation_manager.intent_triage import detect_intent, is_catalog_request
        return detect_intent(text) == 'PRODUCT_QUERY' and (super().gate(text) or is_catalog_request(text))

    def direct_arguments(self, text: str, ctx: ToolContext) -> Optional[dict]:
        from backend.conversation_manager.intent_triage import detect_intent, is_catalog_request, normalize_shopping_text
        text = normalize_shopping_text(text)
        if detect_intent(text) != 'PRODUCT_QUERY':
            return None
        sku = re.search(r'\b([a-z]{3,5}-\d{3})\b', text)
        if not sku and not is_catalog_request(text):
            return None
        unavailable = re.search(r'\b(?:not in stock|out of stock|sold out|unavailable)\b', text)
        args = {'limit': 1 if sku else 20,
                'in_stock_only': not sku and not unavailable and bool(re.search(r'\b(?:available|in stock)\b', text)),
                'out_of_stock_only': bool(unavailable)}
        if sku:
            args['sku'] = sku[1].upper()
        under = re.search(r'\b(?:under|below|less than|up to|at most)\s*\$?(\d+(?:\.\d+)?)', text)
        over = re.search(r'\b(?:over|above|more than|at least)\s*\$?(\d+(?:\.\d+)?)', text)
        price_range = re.search(r'\bbetween\s*\$?(\d+(?:\.\d+)?)\s+and\s*\$?(\d+(?:\.\d+)?)', text)
        if price_range:
            args['min_price'], args['max_price'] = float(price_range[1]), float(price_range[2])
        if under:
            args['max_price'] = float(under[1])
        if over:
            args['min_price'] = float(over[1])
        category_matches, categories = [], []
        for alias, category in _CATEGORY_ALIASES.items():
            category_match = re.search(r'\b' + re.escape(alias) + r'\b', text)
            if category_match:
                category_matches.append(category_match)
                if category not in categories:
                    categories.append(category)
        if len(categories) == 1:
            args['category'] = categories[0]
        elif categories:
            args['categories'] = categories
        if re.search(r'\b(?:most expensive|highest price|price descending)\b', text):
            args['sort'] = 'price_desc'
        if re.search(r'\bcheapest\b', text):
            args['limit'] = 1
        explicit_limit = re.search(r'\b(?:top|first|show|list)\s+(\d+)\b', text)
        if explicit_limit:
            args['limit'] = min(20, max(1, int(explicit_limit[1])))
        query = text
        for match in (under, over, price_range, explicit_limit, sku, unavailable, *category_matches):
            if match:
                query = query.replace(match[0], ' ')
        words = [word for word in re.findall(r'[a-z0-9]+', query)
                 if word not in _STOPWORDS and word not in _BROWSE_WORDS
                 and word not in {category.lower() for category in CATEGORIES}]
        if words:
            args['query'] = ' '.join(words)[:80]
        return args

    def fallback_arguments(self, text: str, ctx: ToolContext) -> Optional[dict]:
        direct = self.direct_arguments(text, ctx)
        if direct is not None:
            return direct
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


def render_catalog_result(result: ToolResult) -> str:
    """Render live facts directly so generation cannot contradict a successful lookup."""
    if not result.ok:
        return result.user_message or 'The catalog lookup failed. Please try again shortly.'
    products = result.data.get('products', [])
    if not products:
        return "I couldn't find matching products in the live store catalog. Try a product name, category, or different budget."
    lines = ['Here are the matching products in our store:', '']
    for product in products:
        stock = int(product['total_stock'])
        availability = f'In stock: {stock}' if stock > 0 else 'Out of stock'
        lines.append(f"- {product['name']} ({product['sku']}) — ${product['price']:.2f} | {availability}")
    if len(products) == 1 and products[0].get('variants'):
        lines += ['', 'Variants:']
        lines += [f"- {variant['label']}: {variant['stock']} in stock" for variant in products[0]['variants']]
    lines += ['', 'Source: live store catalog.']
    return '\n'.join(lines)
