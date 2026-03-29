from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, get_type_hints

from agentory.tools.schema_builder import is_context_type
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
            return await tool.execute(self._resolve_args(tool, args))
        except Exception as e:
            logger.exception("Tool '%s' raised an exception", name)
            return f"Error: {e}"

    def _resolve_args(self, tool: Tool, args: dict) -> dict:
        if self._context is None:
            return dict(args)
        kwargs = dict(args)
        hints = get_type_hints(tool.fn)
        for param_name in inspect.signature(tool.fn).parameters:
            if param_name in ("self", "cls"):
                continue
            hint = hints.get(param_name)
            if hint is not None and is_context_type(hint):
                try:
                    if isinstance(self._context, hint):
                        kwargs[param_name] = self._context
                        break
                except TypeError:
                    pass
        return kwargs

    async def register_mcp_server(self, server: MCPServer) -> None:
        tools = await server.list_tools()
        for tool in tools:
            self._register(tool)

    def to_schema(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]
