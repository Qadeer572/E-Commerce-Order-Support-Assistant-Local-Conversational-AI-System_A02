"""Tool registry: lookup by name and compact descriptions for the decision prompt."""
from typing import Optional

from .base import Tool


def compact_schema(tool: Tool) -> str:
    """One line per argument, rendered from the same pydantic model that validates it."""
    schema = tool.schema()
    required = set(schema.get("required", []))
    lines = []
    for name, spec in schema.get("properties", {}).items():
        options = spec.get("enum")
        if options is None:
            for branch in spec.get("anyOf", []):
                options = options or branch.get("enum")
        kind = "|".join(options) if options else spec.get("type") or next(
            (b.get("type") for b in spec.get("anyOf", []) if b.get("type") != "null"), "string")
        hint = f" ({spec['description']})" if spec.get("description") else ""
        lines.append(f"  {name}: {kind}{' [required]' if name in required else ''}{hint}")
    return "\n".join(lines)


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def describe(self, names: list[str]) -> str:
        parts = []
        for name in names:
            tool = self._tools[name]
            parts.append(f"- {tool.name}: {tool.description}\n{compact_schema(tool)}")
        return "\n".join(parts)
