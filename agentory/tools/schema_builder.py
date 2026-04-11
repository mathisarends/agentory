import collections.abc
import inspect
import types
from collections.abc import Callable
from enum import Enum
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel

from agentory.tools.inject import _InjectMarker


def is_injectable(hint: Any) -> bool:
    """Return *True* if *hint* is ``Inject[T]`` i.e. ``Annotated[T, _InjectMarker]``."""
    if get_origin(hint) is not Annotated:
        return False
    return any(isinstance(a, _InjectMarker) for a in get_args(hint))


class ToolSchemaBuilder:
    _PRIMITIVE_TYPES: ClassVar[dict[type, str]] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    _COLLECTION_TYPES: ClassVar[tuple[type, ...]] = (
        collections.abc.Sequence,
        collections.abc.Iterable,
        collections.abc.Collection,
    )

    def __init__(
        self,
        function: Callable,
        param_model: type[BaseModel] | None = None,
    ) -> None:
        self._function = function
        self._param_model = param_model

    def build(self) -> dict:
        if self._param_model is not None and self._is_pydantic_model(self._param_model):
            return self._build_from_pydantic_model(self._param_model)

        sig = inspect.signature(self._function)
        hints = get_type_hints(self._function, include_extras=True)

        properties: dict[str, dict] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            hint = hints.get(param_name, str)
            if is_injectable(hint):
                continue
            actual_type, description = self._extract_type_and_description(hint)
            properties[param_name] = self._to_json_property(actual_type, description)

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _extract_type_and_description(self, hint: Any) -> tuple[Any, str | None]:
        if get_origin(hint) is not Annotated:
            return hint, None
        args = get_args(hint)
        desc = next((a for a in args[1:] if isinstance(a, str)), None)
        return args[0], desc

    def _unwrap_optional(self, hint: Any) -> Any | None:
        origin = get_origin(hint)
        if origin is Union or isinstance(hint, types.UnionType):
            non_none = [a for a in get_args(hint) if a is not type(None)]
            return non_none[0] if len(non_none) == 1 else None
        return None

    def _to_json_property(
        self, python_type: Any, description: str | None = None
    ) -> dict:
        prop: dict[str, Any] = {}
        if description:
            prop["description"] = description

        unwrapped = self._unwrap_optional(python_type)
        if unwrapped is not None:
            return self._to_json_property(unwrapped, description)

        origin = get_origin(python_type)
        if origin is Literal:
            return {
                **prop,
                "type": "string",
                "enum": [str(v) for v in get_args(python_type)],
            }
        if origin is list or origin in self._COLLECTION_TYPES:
            args = get_args(python_type)
            item_type = args[0] if args else str
            return {**prop, "type": "array", "items": self._to_json_property(item_type)}
        if origin is dict:
            return {**prop, "type": "object"}

        if self._is_pydantic_model(python_type):
            return self._build_model_property(python_type, description)

        if self._is_enum(python_type):
            return {
                **prop,
                "type": "string",
                "enum": [member.value for member in python_type],
            }

        json_type = self._PRIMITIVE_TYPES.get(python_type, "string")
        return {**prop, "type": json_type}

    def _build_from_pydantic_model(self, model: type[BaseModel]) -> dict:
        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for field_name, field_info in model.model_fields.items():
            properties[field_name] = self._to_json_property(
                field_info.annotation,
                field_info.description,
            )
            if not field_info.is_required() and field_info.default is not None:
                properties[field_name]["default"] = field_info.default
            if field_info.is_required():
                required.append(field_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _build_model_property(
        self, model: type[BaseModel], description: str | None
    ) -> dict[str, Any]:
        schema = self._build_from_pydantic_model(model)
        if description:
            schema["description"] = description
        return schema

    @staticmethod
    def _is_pydantic_model(model: Any) -> bool:
        try:
            return isinstance(model, type) and issubclass(model, BaseModel)
        except TypeError:
            return False

    @staticmethod
    def _is_enum(model: Any) -> bool:
        try:
            return isinstance(model, type) and issubclass(model, Enum)
        except TypeError:
            return False
