from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agentory.tools.views import Tool

if TYPE_CHECKING:
    from agentory.mcp.server import MCPServer

logger = logging.getLogger(__name__)


class Tools:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._context: Any = None

    def set_context(self, context: Any) -> None:
        self._context = context

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def action(
        self, description: str, name: str | None = None, status: str | None = None
    ) -> Callable:
        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._register(
                Tool(
                    name=name or fn.__name__,
                    description=description,
                    fn=fn,
                    status=status,
                )
            )
            return fn

        return decorator

    def _register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    async def execute(self, name: str, args: dict) -> str:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool '{name}'. Available: {list(self._tools)}")
        try:
            return await tool.execute(args, context=self._context)
        except Exception as e:
            logger.exception("Tool '%s' raised an exception", name)
            return f"Error: {e}"

    async def register_mcp_server(self, server: MCPServer) -> None:
        tools = await server.list_tools()
        for tool in tools:
            self._register(tool)

    def to_schema(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]