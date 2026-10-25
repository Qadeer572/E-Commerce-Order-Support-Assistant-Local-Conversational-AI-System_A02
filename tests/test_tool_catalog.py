import pytest
from backend.tools.base import ToolContext
from backend.tools.catalog import CatalogTool, CatalogArgs, search
from backend import shop_db


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "shop_test.db"
    shop_db.build_database(db_file)
    return db_file


def test_catalog_search_exact_sku(test_db):
    res = search(CatalogArgs(sku="ELEC-001"), test_db)
    assert len(res) == 1
    assert res[0]["sku"] == "ELEC-001"
    assert res[0]["price"] == 49.99
    assert len(res[0]["variants"]) == 3


def test_catalog_search_category_and_price(test_db):
    res = search(CatalogArgs(category="Electronics", max_price=50.0), test_db)
    assert len(res) >= 1
    for item in res:
        assert item["category"] == "Electronics"
        assert item["price"] <= 50.0


def test_catalog_sql_injection_is_inert(test_db):
    malicious = "'; DROP TABLE products; --"
    res = search(CatalogArgs(query=malicious), test_db)
    assert isinstance(res, list)
    # Check table is still intact
    res_after = search(CatalogArgs(sku="ELEC-001"), test_db)
    assert len(res_after) == 1


@pytest.mark.asyncio
async def test_catalog_tool_run_ok(test_db):
    tool = CatalogTool(db_path=test_db)
    ctx = ToolContext(owner_id="user1", session_id="s1")
    res = await tool.run(CatalogArgs(query="earbuds"), ctx)
    assert res.ok is True
    assert res.data["count"] >= 1
    assert "ProSound" in res.data["products"][0]["name"]


@pytest.mark.asyncio
async def test_catalog_tool_no_results(test_db):
    tool = CatalogTool(db_path=test_db)
    ctx = ToolContext(owner_id="user1", session_id="s1")
    res = await tool.run(CatalogArgs(query="nonexistent_product_xyz"), ctx)
    assert res.ok is True
    assert res.data["count"] == 0
    assert res.data["products"] == []
