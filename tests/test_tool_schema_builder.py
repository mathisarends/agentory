import collections.abc
from typing import Annotated

from agentory.tools.inject import Inject
from agentory.tools.schema_builder import ToolSchemaBuilder


class SimpleModel:
    pass


def _build(fn) -> dict:
    return ToolSchemaBuilder(fn).build()


class TestPrimitiveTypes:
    def test_str_parameter(self) -> None:
        def fn(name: str) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["name"] == {"type": "string"}
        assert schema["required"] == ["name"]

    def test_int_parameter(self) -> None:
        def fn(count: int) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["count"] == {"type": "integer"}

    def test_float_parameter(self) -> None:
        def fn(price: float) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["price"] == {"type": "number"}

    def test_bool_parameter(self) -> None:
        def fn(active: bool) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["active"] == {"type": "boolean"}


class TestCollectionTypes:
    def test_list_of_str(self) -> None:
        def fn(items: list[str]) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }

    def test_list_of_int(self) -> None:
        def fn(ids: list[int]) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["ids"] == {
            "type": "array",
            "items": {"type": "integer"},
        }

    def test_bare_list(self) -> None:
        def fn(items: list) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["items"]["type"] == "array"

    def test_dict_parameter(self) -> None:
        def fn(data: dict) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["data"] == {"type": "object"}

    def test_sequence_type(self) -> None:
        def fn(items: collections.abc.Sequence[str]) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["items"] == {
            "type": "array",
            "items": {"type": "string"},
        }


class TestOptionalTypes:
    def test_optional_str(self) -> None:
        def fn(name: str | None) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["name"] == {"type": "string"}

    def test_optional_int(self) -> None:
        def fn(count: int | None) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["count"] == {"type": "integer"}


class TestAnnotatedTypes:
    def test_annotated_with_description(self) -> None:
        def fn(city: Annotated[str, "The city name"]) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["city"] == {
            "type": "string",
            "description": "The city name",
        }

    def test_annotated_without_description(self) -> None:
        def fn(city: Annotated[str, 42]) -> None: ...

        schema = _build(fn)
        assert schema["properties"]["city"] == {"type": "string"}


class TestRequiredAndDefaults:
    def test_required_params(self) -> None:
        def fn(a: str, b: int) -> None: ...

        schema = _build(fn)
        assert schema["required"] == ["a", "b"]

    def test_optional_param_not_required(self) -> None:
        def fn(a: str, b: int = 5) -> None: ...

        schema = _build(fn)
        assert schema["required"] == ["a"]

    def test_no_params(self) -> None:
        def fn() -> None: ...

        schema = _build(fn)
        assert schema["properties"] == {}
        assert schema["required"] == []


class TestSkippedParams:
    def test_self_is_skipped(self) -> None:
        def fn(self, name: str) -> None: ...

        schema = _build(fn)
        assert "self" not in schema["properties"]
        assert "name" in schema["properties"]

    def test_inject_param_is_skipped(self) -> None:
        def fn(svc: Annotated[SimpleModel, Inject], name: str) -> None: ...

        schema = _build(fn)
        assert "svc" not in schema["properties"]
        assert "name" in schema["properties"]

    def test_bare_custom_type_is_not_skipped(self) -> None:
        """A custom class WITHOUT Inject should appear in the schema (not auto-skipped)."""

        def fn(svc: SimpleModel, name: str) -> None: ...

        schema = _build(fn)
        assert "svc" in schema["properties"]
        assert "name" in schema["properties"]

    def test_cls_is_skipped(self) -> None:
        def fn(cls, name: str) -> None: ...

        schema = _build(fn)
        assert "cls" not in schema["properties"]
        assert "name" in schema["properties"]
