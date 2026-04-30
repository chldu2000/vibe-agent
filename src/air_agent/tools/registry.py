from __future__ import annotations

import inspect
import json
from typing import Any, Callable, Awaitable

from air_agent.tools.base import Tool


def _python_type_to_json_schema(annotation: Any) -> dict[str, Any]:
    origin = getattr(annotation, "__origin__", None)
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is list or origin is list:
        return {"type": "array"}
    if annotation is dict or origin is dict:
        return {"type": "object"}
    return {}


def _extract_parameters(func: Callable) -> dict[str, Any]:
    sig = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        schema = _python_type_to_json_schema(param.annotation)
        if not schema and param.annotation is not inspect.Parameter.empty:
            schema = {"type": "string"}
        if not schema:
            schema = {"type": "string"}
        properties[param_name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    result: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        result["required"] = required
    return result


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, func: Callable, name: str | None = None, description: str = "") -> None:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""
        parameters = _extract_parameters(func)
        self._tools[tool_name] = Tool(
            name=tool_name,
            description=tool_desc,
            parameters=parameters,
            handler=func,
            is_mcp=False,
        )

    def register_mcp_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[dict], Awaitable[Any]],
    ) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            is_mcp=True,
        )

    def get_openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments_json: str) -> str:
        if name not in self._tools:
            raise KeyError(f"Tool not found: {name}")
        tool = self._tools[name]
        args = json.loads(arguments_json)
        if tool.is_mcp:
            result = await tool.handler(args)
        else:
            result = await tool.handler(**args)
        return str(result)

    def has_tool(self, name: str) -> bool:
        return name in self._tools
