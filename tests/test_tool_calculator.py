import pytest
from backend.tools.base import ToolContext
from backend.tools.calculator import CalculatorTool, CalculatorArgs, calculate, CalculatorError


def test_calculate_basic_math():
    res = calculate("49.99 * 3")
    assert res["result"] == "149.97"


def test_calculate_discount_and_tax():
    # 49.99 * 3 * 0.85 = 127.4745 -> 127.47
    res = calculate("49.99*3*(1-0.15)")
    assert res["result"] == "127.47"


def test_calculate_percent_normalization():
    res = calculate("15% off 200")
    assert res["result"] == "170.00"

    res2 = calculate("15% of 200")
    assert res2["result"] == "30.00"


def test_calculate_division_by_zero():
    with pytest.raises(CalculatorError) as exc:
        calculate("100 / 0")
    assert exc.value.code == "division_by_zero"


def test_calculate_huge_exponent():
    with pytest.raises(CalculatorError) as exc:
        calculate("2 ** 50")
    assert exc.value.code == "too_large"


def test_calculate_rejects_names_and_code():
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('dir')")
    with pytest.raises(CalculatorError):
        calculate("open('data/products.json').read()")


@pytest.mark.asyncio
async def test_calculator_tool_run():
    tool = CalculatorTool()
    ctx = ToolContext(owner_id="user1", session_id="s1")
    res = await tool.run(CalculatorArgs(expression="10 + 20 * 3"), ctx)
    assert res.ok is True
    assert res.data["result"] == "70.00"


@pytest.mark.asyncio
async def test_calculator_tool_fallback_args():
    tool = CalculatorTool()
    ctx = ToolContext(owner_id="user1", session_id="s1")
    args = tool.fallback_arguments("what is 15% off 49.99 times 3", ctx)
    assert args is not None
    assert "49.99" in args["expression"]
