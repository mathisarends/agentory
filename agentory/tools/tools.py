from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    TypeVar,
    overload,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from agentory.tools.di import _InjectMarker, ToolContext
from agentory.tools.views import Tool, ToolSchema, DoneParams, DONE_TOOL_NAME

if TYPE_CHECKING:
    from agentory.mcp.server import MCPServer

logger = logging.getLogger(__name__)

_P = TypeVar("_P", bound=BaseModel)


class Tools:
    def __init__(
        self, context: ToolContext | None = None, use_done_tool: bool = False
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self._context = context or ToolContext()
        self._use_done_tool = use_done_tool

        if use_done_tool:
            self._register_done_tool()

    def is_done_tool(self, name: str) -> bool:
        return self._use_done_tool and name == DONE_TOOL_NAME

    def _register_done_tool(self) -> None:
        @self.action(
            name=DONE_TOOL_NAME,
            description="Signal that you are finished and return the final answer to the user.",
            params=DoneParams,
        )
        async def done(params: DoneParams) -> str:
            return params.output

    def set_context(self, context: ToolContext) -> Tools:
        self._context = context
        return self

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    @overload
    def action(
        self,
        description: str,
        name: str | None = None,
        *,
        status: str | Callable[[_P], str] | None = None,
        status_label: str | Callable[[_P], str] | None = None,
        params: type[_P],
    ) -> Callable: ...

    @overload
    def action(
        self,
        description: str,
        name: str | None = None,
        *,
        status: str | None = None,
        status_label: str | None = None,
        params: None = None,
    ) -> Callable: ...

    def action(
        self,
        description: str,
        name: str | None = None,
        status: str | Callable[[BaseModel], str] | None = None,
        status_label: str | Callable[[BaseModel], str] | None = None,
        params: type[BaseModel] | None = None,
    ) -> Callable:
        if status is not None and status_label is not None:
            raise ValueError("Use either 'status' or 'status_label', not both")

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._register(
                Tool(
                    name=name or fn.__name__,
                    description=description,
                    fn=fn,
                    status=status,
                    status_label=status_label,
                    param_model=params,
                )
            )
            return fn

        return decorator

    def _register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    async def execute(self, name: str, args: dict) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool '{name}'. Available: {list(self._tools)}")
        try:
            return await tool.execute(self._resolve_args(tool, args))
        except Exception as e:
            logger.exception("Tool '%s' raised an exception", name)
            return tool.format_error(e)

    def _resolve_args(self, tool: Tool, args: dict) -> dict[str, Any]:
        kwargs = self._resolve_non_injected_args(tool, args)
        hints = get_type_hints(tool.fn, include_extras=True)
        signature = inspect.signature(tool.fn)

        for param_name, param in signature.parameters.items():
            hint = hints.get(param_name)
            if hint is None or not self._is_injectable(hint):
                continue

            actual_type = get_args(hint)[0]
            dependency = self._context.resolve(actual_type)
            if dependency is None:
                if param.default is inspect.Parameter.empty:
                    raise ValueError(
                        f"Missing injected dependency for parameter '{param_name}' of type '{actual_type.__name__}'"
                    )
                continue
            kwargs[param_name] = dependency

        return kwargs

    def _resolve_non_injected_args(self, tool: Tool, args: dict) -> dict[str, Any]:
        if tool.param_model is None:
            return dict(args)

        model_instance = tool.param_model.model_validate(args)
        hints = get_type_hints(tool.fn, include_extras=True)
        signature = inspect.signature(tool.fn)
        target = self._find_param_model_parameter(
            signature=signature,
            hints=hints,
            param_model=tool.param_model,
        )
        if target is None:
            raise ValueError(
                f"Tool '{tool.name}' uses params model '{tool.param_model.__name__}' "
                "but function has no parameter that can receive it"
            )
        return {target: model_instance}

    def _find_param_model_parameter(
        self,
        signature: inspect.Signature,
        hints: dict[str, Any],
        param_model: type[BaseModel],
    ) -> str | None:
        candidates: list[str] = []
        for param_name in signature.parameters:
            if param_name in ("self", "cls"):
                continue
            hint = hints.get(param_name)
            if hint is not None and self._is_injectable(hint):
                continue
            candidates.append(param_name)
            if hint == param_model:
                return param_name

        if len(candidates) == 1:
            return candidates[0]
        return None

    @staticmethod
    def _is_injectable(hint: Any) -> bool:
        if get_origin(hint) is not Annotated:
            return False
        return any(isinstance(a, _InjectMarker) for a in get_args(hint))

    async def register_mcp_server(self, server: MCPServer) -> None:
        tools = await server.list_tools()
        for tool in tools:
            self._register(tool)

    def to_schema(self) -> list[ToolSchema]:
        return [t.to_schema() for t in self._tools.values()]
