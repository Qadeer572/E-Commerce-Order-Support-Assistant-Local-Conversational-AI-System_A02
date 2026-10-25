"""calculator tool: safe AST arithmetic with Decimal, never eval()."""
import ast
import operator
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

from pydantic import BaseModel, Field

from .base import Tool, ToolContext, ToolResult

MAX_EXPONENT = 10
MAX_ABS_RESULT = Decimal("1e15")
_BINARY = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_FUNCTIONS = {"round": round, "abs": abs, "min": min, "max": max}


class CalculatorError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code, self.message = code, message


class CalculatorArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=200,
                            description="Arithmetic expression, e.g. 49.99*3*(1-0.15)")


def normalize_expression(expression: str) -> str:
    """Rewrite '15% of 200' / '15% off 200' / '8%' left in the expression by the LLM."""
    text = expression.replace("$", "").replace(",", "").strip()
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)", r"(\1/100)*\2", text, flags=re.I)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*off\s*(\d+(?:\.\d+)?)", r"\2*(1-\1/100)", text, flags=re.I)
    return re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", text)


def _evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalculatorError("too_large", "That exponent is too large for me to calculate.")
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculatorError("division_by_zero", "That calculation divides by zero.")
        return _BINARY[type(node.op)](left, right)
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS and not node.keywords):
        return _FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args])
    raise CalculatorError("unsupported", "I can only calculate plain arithmetic expressions.")


def calculate(expression: str) -> dict:
    """Evaluate an expression and return {expression, result, rounded_to}."""
    cleaned = normalize_expression(expression)
    try:
        value = _evaluate(ast.parse(cleaned, mode="eval"))
    except CalculatorError:
        raise
    except (SyntaxError, ValueError, ArithmeticError, TypeError, RecursionError):
        raise CalculatorError("invalid_expression", "I could not understand that expression.")
    if abs(value) > MAX_ABS_RESULT:
        raise CalculatorError("too_large", "That result is too large for me to report.")
    rounded = Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return {"expression": cleaned, "result": str(rounded), "rounded_to": 2}


_WORD_OPERATORS = [
    (r"\b(?:multiplied by|times)\b", "*"), (r"\bdivided by\b", "/"),
    (r"\bplus\b", "+"), (r"\bminus\b", "-"),
]


class CalculatorTool(Tool):
    name = "calculator"
    description = ("Evaluates an arithmetic expression for order totals, discounts, tax and shipping math. "
                   "Use when the customer asks for a number to be computed. "
                   "Do NOT use for price lookups or stock questions.")
    args_model = CalculatorArgs
    timeout_s = 1.0
    timeout_message = "The calculator did not respond in time. Please try again."
    gate_pattern = re.compile(
        r"\d\s*[-+*/x%]\s*\d|\d\s*%|\b(?:percent|discount|tax|total|times|multiplied|divided|plus|minus)\b"
        r"|\bhow much for \w+|\b\d+\s*(?:items?|units?|pieces?)\b", re.I)
    examples = [
        ("what is 15% off 49.99", {"expression": "49.99*(1-0.15)"}),
        ("calculate (12.99 * 3) + 5.50", {"expression": "(12.99*3)+5.50"}),
        ("total for 4 items at $12.50 plus 8% tax", {"expression": "4*12.50*(1+0.08)"}),
    ]

    def fallback_arguments(self, text: str, ctx: ToolContext) -> Optional[dict]:
        lowered = text.lower().replace(",", "")
        for pattern, symbol in _WORD_OPERATORS:
            lowered = re.sub(pattern, symbol, lowered)
        off = re.search(r"(\d+(?:\.\d+)?)\s*%\s*off\s*\$?(\d+(?:\.\d+)?)(?:\s*[*x]\s*(\d+(?:\.\d+)?))?", lowered)
        if off:
            quantity = off.group(3) or "1"
            return {"expression": f"{off.group(2)}*{quantity}*(1-{off.group(1)}/100)"}
        match = re.search(r"\$?\d+(?:\.\d+)?(?:\s*[-+*/x]\s*\$?\d+(?:\.\d+)?)+", lowered)
        if match:
            return {"expression": match.group(0).replace("x", "*").replace("$", "")}
        return None

    async def run(self, args: CalculatorArgs, ctx: ToolContext) -> ToolResult:
        try:
            return ToolResult(ok=True, data=calculate(args.expression))
        except CalculatorError as error:
            return ToolResult(ok=False, error_code=error.code, user_message=error.message)
