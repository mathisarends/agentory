from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from agentory.tools.views import Tool


class ToolResultAdapter(Protocol):
    """Converts tool execution outcomes to values written into agent history."""

    def on_success(self, *, tool: Tool, result: Any) -> Any: ...

    def on_error(self, *, tool: Tool, error: Exception) -> Any: ...


class DefaultToolResultAdapter:
    """Default adapter returning raw results and string errors."""

    def on_success(self, *, tool: Tool, result: Any) -> Any:
        return result if result is not None else ""

    def on_error(self, *, tool: Tool, error: Exception) -> str:
        return f"Error: {error}"
