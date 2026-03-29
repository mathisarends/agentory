import inspect
from collections.abc import Callable
from typing import Any

from agentory.tools.schema_builder import ToolSchemaBuilder


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        status: str | Callable[[dict[str, Any]], str] | None = None,
        schema: dict | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self._schema = schema
        self._schema_builder = None if schema else ToolSchemaBuilder(self.fn)
        self.status = status
        if callable(self.status) and self._schema_builder is not None:
            self._validate_status_keys()

    def _validate_status_keys(self) -> None:
        params = set(inspect.signature(self.fn).parameters.keys())
        accessed: set[str] = set()

        class _Tracker:
            def __getitem__(self, key: str) -> str:
                accessed.add(key)
                return ""

            def get(self, key: str, default: Any = None) -> Any:
                accessed.add(key)
                return default

        assert callable(self.status)
        self.status(_Tracker())

        if unknown := accessed - params:
            raise ValueError(
                f"Tool '{self.name}' status references unknown args: {unknown}. "
                f"Available: {params}"
            )

    def render_status(self, args: dict[str, Any]) -> str | None:
        if self.status is None:
            return None
        if callable(self.status):
            return self.status(args)
        return self.status

    async def execute(self, args: dict[str, Any], context: Any = None) -> str:
        kwargs = dict(args)
        if "context" in inspect.signature(self.fn).parameters:
            kwargs["context"] = context
        if inspect.iscoroutinefunction(self.fn):
            result = await self.fn(**kwargs)
        else:
            result = self.fn(**kwargs)
        return str(result) if result is not None else ""

    def to_schema(self) -> dict:
        parameters = self._schema if self._schema else self._schema_builder.build()
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def __eq__(self, other: object) -> bool:
        return self.name == other.name if isinstance(other, Tool) else NotImplemented

    def __hash__(self) -> int:
        return hash(self.name)