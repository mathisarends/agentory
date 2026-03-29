from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin, get_type_hints

from agentory.tools.inject import Inject
from agentory.tools.views import Tool

if TYPE_CHECKING:
    from agentory.mcp.server import MCPServer

logger = logging.getLogger(__name__)


class Tools:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._context: list[Any] = []

    def provide(self, *dependencies: Any) -> Tools:
        self._context.extend(dependencies)
        return self

    def clear_dependencies(self) -> Tools:
        self._context.clear()
        return self

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
        if not self._context:
            return dict(args)
        kwargs = dict(args)
        hints = get_type_hints(tool.fn, include_extras=True)
        for param_name in inspect.signature(tool.fn).parameters:
            hint = hints.get(param_name)
            if hint is None or not self._is_injectable(hint):
                continue
            actual_type = get_args(hint)[0]
            for ctx in self._context:
                if isinstance(ctx, actual_type):
                    kwargs[param_name] = ctx
                    break
        return kwargs

    @staticmethod
    def _is_injectable(hint: Any) -> bool:
        if get_origin(hint) is not Annotated:
            return False
        return Inject in get_args(hint)

    async def register_mcp_server(self, server: MCPServer) -> None:
        tools = await server.list_tools()
        for tool in tools:
            self._register(tool)

    def to_schema(self) -> list[dict]:
        return [t.to_schema() for t in self._tools.values()]
