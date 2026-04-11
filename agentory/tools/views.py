import inspect
from collections.abc import Callable
from typing import Any, TypeVar, TypedDict, overload

from pydantic import BaseModel, Field

from agentory.tools.schema_builder import ToolSchemaBuilder


_P = TypeVar("_P", bound=BaseModel)


class FunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict


class ToolSchema(TypedDict):
    type: str
    function: FunctionSchema


class Tool:
    @overload
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        status: str | Callable[[_P], str] | None = None,
        status_label: str | Callable[[_P], str] | None = None,
        schema: dict | None = None,
        param_model: type[_P] | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        status: str | None = None,
        status_label: str | None = None,
        schema: dict | None = None,
        param_model: None = None,
    ) -> None: ...

    def __init__(
        self,
        name: str,
        description: str,
        fn: Callable[..., Any],
        status: str | Callable[[BaseModel], str] | None = None,
        status_label: str | Callable[[BaseModel], str] | None = None,
        schema: dict | None = None,
        param_model: type[BaseModel] | None = None,
    ) -> None:
        if status is not None and status_label is not None:
            raise ValueError(
                f"Tool '{name}' received both 'status' and 'status_label'. Use only one."
            )

        self.name = name
        self.description = description
        self.fn = fn
        self._schema = schema
        self.param_model = param_model
        self._schema_builder = (
            None if schema else ToolSchemaBuilder(self.fn, param_model)
        )
        self._status_label = status_label if status_label is not None else status

        if callable(self._status_label):
            if self.param_model is None:
                raise ValueError(
                    f"Tool '{self.name}' status_label callable requires a params model"
                )
            self._validate_status_for_model()

    def _validate_status_for_model(self) -> None:
        assert self.param_model is not None
        assert callable(self._status_label)

        dummy = _make_dummy(self.param_model)
        try:
            result = self._status_label(dummy)
            if not isinstance(result, str):
                raise ValueError(
                    f"Tool '{self.name}' status_label callable must return str, "
                    f"got {type(result).__name__}"
                )
        except AttributeError as e:
            raise ValueError(
                f"Tool '{self.name}' status_label callable references unknown params-model field: {e}"
            ) from e
        except ValueError:
            raise
        except Exception:
            # Runtime failures are allowed for dummy placeholders.
            pass

    def render_status(self, args: dict[str, Any]) -> str | None:
        if self._status_label is None:
            return None
        if callable(self._status_label):
            if self.param_model is None:
                return None
            try:
                params = self.param_model.model_validate(args)
                return self._status_label(params)
            except Exception:
                return None
        return self._status_label

    async def execute(self, args: dict[str, Any]) -> Any:
        try:
            if inspect.iscoroutinefunction(self.fn):
                result = await self.fn(**args)
            else:
                result = self.fn(**args)
            return result if result is not None else ""
        except Exception as e:
            return f"Error: {e}"

    def format_error(self, error: Exception) -> Any:
        return f"Error: {error}"

    def to_schema(self) -> ToolSchema:
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


class DoneParams(BaseModel):
    output: str = Field(description="The final answer or result to return to the user")


DONE_TOOL_NAME = "done"
