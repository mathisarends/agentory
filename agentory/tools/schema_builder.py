import collections.abc
import inspect
import types
from collections.abc import Callable
from typing import Annotated, Any, ClassVar, Union, get_args, get_origin, get_type_hints

from agentory.tools.inject import Inject


def is_injectable(hint: Any) -> bool:
    """Return *True* if *hint* is marked with ``Annotated[..., Inject]``."""
    if get_origin(hint) is not Annotated:
        return False
    return Inject in get_args(hint)


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

    def __init__(self, function: Callable) -> None:
        self._function = function

    def build(self) -> dict:
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
        if origin is list or origin in self._COLLECTION_TYPES:
            args = get_args(python_type)
            item_type = args[0] if args else str
            return {**prop, "type": "array", "items": self._to_json_property(item_type)}
        if origin is dict:
            return {**prop, "type": "object"}

        json_type = self._PRIMITIVE_TYPES.get(python_type, "string")
        return {**prop, "type": json_type}
