import inspect
from typing import Any

import pytest

from agentory.tools.views import Tool


def _sync_fn(name: str) -> str:
    return f"hello {name}"


async def _async_fn(name: str) -> str:
    return f"async hello {name}"


def _fn_with_context(query: str, context: Any = None) -> str:
    return f"{query} ctx={context}"


class TestToolExecute:
    @pytest.mark.asyncio
    async def test_sync_function_execution(self) -> None:
        tool = Tool(name="greet", description="greets", fn=_sync_fn)
        result = await tool.execute({"name": "world"})
        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_async_function_execution(self) -> None:
        tool = Tool(name="greet", description="greets", fn=_async_fn)
        result = await tool.execute({"name": "world"})
        assert result == "async hello world"

    @pytest.mark.asyncio
    async def test_context_passed_to_function(self) -> None:
        tool = Tool(name="search", description="search", fn=_fn_with_context)
        result = await tool.execute({"query": "test"}, context="my_ctx")
        assert result == "test ctx=my_ctx"

    @pytest.mark.asyncio
    async def test_none_return_gives_empty_string(self) -> None:
        def fn(x: str) -> None:
            pass

        tool = Tool(name="noop", description="noop", fn=fn)
        result = await tool.execute({"x": "a"})
        assert result == ""


class TestToolSchema:
    def test_to_schema_structure(self) -> None:
        tool = Tool(name="greet", description="Say hi", fn=_sync_fn)
        schema = tool.to_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "greet"
        assert schema["function"]["description"] == "Say hi"
        assert "parameters" in schema["function"]

    def test_to_schema_with_explicit_schema(self) -> None:
        custom = {"type": "object", "properties": {"x": {"type": "string"}}}
        tool = Tool(name="t", description="d", fn=_sync_fn, schema=custom)
        schema = tool.to_schema()
        assert schema["function"]["parameters"] == custom


class TestToolRenderStatus:
    def test_static_status(self) -> None:
        tool = Tool(name="t", description="d", fn=_sync_fn, status="Loading…")
        assert tool.render_status({}) == "Loading…"

    def test_callable_status(self) -> None:
        tool = Tool(
            name="t",
            description="d",
            fn=_sync_fn,
            status=lambda args: f"Greeting {args['name']}",
        )
        assert tool.render_status({"name": "Alice"}) == "Greeting Alice"

    def test_no_status(self) -> None:
        tool = Tool(name="t", description="d", fn=_sync_fn)
        assert tool.render_status({}) is None


class TestToolStatusValidation:
    def test_invalid_status_key_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown args"):
            Tool(
                name="t",
                description="d",
                fn=_sync_fn,
                status=lambda args: args["nonexistent"],
            )


class TestToolEquality:
    def test_tools_equal_by_name(self) -> None:
        a = Tool(name="t", description="d1", fn=_sync_fn)
        b = Tool(name="t", description="d2", fn=_async_fn)
        assert a == b

    def test_tools_not_equal_by_name(self) -> None:
        a = Tool(name="a", description="d", fn=_sync_fn)
        b = Tool(name="b", description="d", fn=_sync_fn)
        assert a != b

    def test_tool_hash_based_on_name(self) -> None:
        a = Tool(name="t", description="d1", fn=_sync_fn)
        b = Tool(name="t", description="d2", fn=_async_fn)
        assert hash(a) == hash(b)

    def test_not_equal_to_non_tool(self) -> None:
        tool = Tool(name="t", description="d", fn=_sync_fn)
        assert tool != "not a tool"
