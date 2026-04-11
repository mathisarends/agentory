import inspect
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from agentory.tools.result import DefaultToolResultAdapter, ToolResultAdapter
from agentory.tools.schema_builder import ToolSchemaBuilder


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        status: str
        | Callable[[dict[str, Any]], str]
        | Callable[[BaseModel], str]
        | None = None,
        schema: dict | None = None,
        param_model: type[BaseModel] | None = None,
        result_adapter: ToolResultAdapter | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.fn = fn
        self._schema = schema
        self.param_model = param_model
        self.result_adapter = result_adapter or DefaultToolResultAdapter()
        self._schema_builder = (
            None if schema else ToolSchemaBuilder(self.fn, param_model)
        )
        self.status = status
        if callable(self.status):
            if self.param_model is not None:
                self._validate_status_for_model()
            elif self._schema_builder is not None:
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

    def _validate_status_for_model(self) -> None:
        assert self.param_model is not None
        assert callable(self.status)

        dummy = _make_dummy(self.param_model)
        try:
            result = self.status(dummy)
            if not isinstance(result, str):
                raise ValueError(
                    f"Tool '{self.name}' status callable must return str, "
                    f"got {type(result).__name__}"
                )
        except AttributeError as e:
            raise ValueError(
                f"Tool '{self.name}' status callable references unknown params-model field: {e}"
            ) from e
        except ValueError:
            raise
        except Exception:
            # Runtime failures are allowed for dummy placeholders.
            pass

    def render_status(self, args: dict[str, Any]) -> str | None:
        if self.status is None:
            return None
        if callable(self.status):
            if self.param_model is not None:
                try:
                    params = self.param_model.model_validate(args)
                    return self.status(params)
                except Exception:
                    pass
            return self.status(args)
        return self.status

    async def execute(self, args: dict[str, Any]) -> Any:
        try:
            if inspect.iscoroutinefunction(self.fn):
                result = await self.fn(**args)
            else:
                result = self.fn(**args)
            return self.result_adapter.on_success(tool=self, result=result)
        except Exception as e:
            return self.result_adapter.on_error(tool=self, error=e)

    def format_error(self, error: Exception) -> Any:
        return self.result_adapter.on_error(tool=self, error=error)

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


def _make_dummy(param_model: type[BaseModel]) -> BaseModel:
    """Builds a model instance with placeholder values for status callable checks."""
    defaults: dict[str, Any] = {}
    for field_name, field_info in param_model.model_fields.items():
        annotation = field_info.annotation
        if annotation is str:
            defaults[field_name] = "placeholder"
        elif annotation is int:
            defaults[field_name] = 0
        elif annotation is float:
            defaults[field_name] = 0.0
        elif annotation is bool:
            defaults[field_name] = False
        else:
            defaults[field_name] = None
    return param_model.model_construct(**defaults)
