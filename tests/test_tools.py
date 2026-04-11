from typing import Any
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from agentory.tools.context import ToolContext
from agentory.tools.inject import Inject
from agentory.tools.tools import Tools
from agentory.tools.views import Tool


def _dummy_fn(x: str) -> str:
    return f"result:{x}"


async def _async_dummy(x: str) -> str:
    return f"async:{x}"


class _FakeContext:
    def __init__(self, value: str) -> None:
        self.value = value


class _AnotherService:
    def __init__(self, name: str) -> None:
        self.name = name


class _EchoParams(BaseModel):
    text: str


class _DictResultAdapter:
    def on_success(self, *, tool: Tool, result: Any) -> dict[str, Any]:
        return {"tool": tool.name, "ok": True, "result": result}

    def on_error(self, *, tool: Tool, error: Exception) -> dict[str, Any]:
        return {"tool": tool.name, "ok": False, "error": str(error)}


class TestToolsRegisterAndGet:
    def test_register_via_action_decorator(self) -> None:
        tools = Tools()

        @tools.action(description="A test tool")
        def my_tool(x: str) -> str:
            return x

        assert tools.get("my_tool") is not None
        assert tools.get("my_tool").name == "my_tool"

    def test_register_with_custom_name(self) -> None:
        tools = Tools()

        @tools.action(description="A test tool", name="custom_name")
        def my_tool(x: str) -> str:
            return x

        assert tools.get("custom_name") is not None
        assert tools.get("my_tool") is None

    def test_get_unknown_tool_returns_none(self) -> None:
        tools = Tools()
        assert tools.get("nonexistent") is None


class TestToolsExecute:
    @pytest.mark.asyncio
    async def test_execute_registered_tool(self) -> None:
        tools = Tools()

        @tools.action(description="echo")
        def echo(x: str) -> str:
            return f"echo:{x}"

        result = await tools.execute("echo", {"x": "hello"})
        assert result == "echo:hello"

    @pytest.mark.asyncio
    async def test_execute_unknown_tool_raises(self) -> None:
        tools = Tools()
        with pytest.raises(ValueError, match="Unknown tool 'missing'"):
            await tools.execute("missing", {})

    @pytest.mark.asyncio
    async def test_execute_failing_tool_returns_error(self) -> None:
        tools = Tools()

        @tools.action(description="fails")
        def bad_tool(x: str) -> str:
            raise RuntimeError("boom")

        result = await tools.execute("bad_tool", {"x": "a"})
        assert "Error" in result
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_execute_with_params_model(self) -> None:
        tools = Tools()

        @tools.action(description="echo model", params=_EchoParams)
        def echo(params: _EchoParams) -> str:
            return f"echo:{params.text}"

        result = await tools.execute("echo", {"text": "hello"})
        assert result == "echo:hello"

    @pytest.mark.asyncio
    async def test_execute_with_params_model_and_injected_dependency(self) -> None:
        tools = Tools()

        @tools.action(description="echo model", params=_EchoParams)
        def echo(
            params: _EchoParams,
            svc: Inject[_AnotherService],
        ) -> str:
            return f"echo:{params.text}:{svc.name}"

        tools.provide(_AnotherService("svc"))
        result = await tools.execute("echo", {"text": "hello"})
        assert result == "echo:hello:svc"

    @pytest.mark.asyncio
    async def test_execute_uses_custom_result_adapter(self) -> None:
        tools = Tools()

        @tools.action(
            description="echo",
            result_adapter=_DictResultAdapter(),
        )
        def echo(x: str) -> str:
            return f"echo:{x}"

        result = await tools.execute("echo", {"x": "hello"})
        assert result == {"tool": "echo", "ok": True, "result": "echo:hello"}


class TestToolsProvide:
    @pytest.mark.asyncio
    async def test_single_dependency_injected(self) -> None:
        tools = Tools()

        @tools.action(description="uses context")
        def ctx_tool(x: str, ctx: Inject[_FakeContext] = None) -> str:
            return f"{x}:{ctx.value}"

        tools.provide(_FakeContext("my_context"))
        result = await tools.execute("ctx_tool", {"x": "val"})
        assert result == "val:my_context"

    @pytest.mark.asyncio
    async def test_multiple_dependencies_injected(self) -> None:
        tools = Tools()

        @tools.action(description="needs both")
        def multi(
            x: str,
            ctx: Inject[_FakeContext] = None,
            svc: Inject[_AnotherService] = None,
        ) -> str:
            return f"{x}:{ctx.value}:{svc.name}"

        tools.provide(_FakeContext("fc"), _AnotherService("as"))
        result = await tools.execute("multi", {"x": "hi"})
        assert result == "hi:fc:as"

    @pytest.mark.asyncio
    async def test_tool_only_gets_matching_dependency(self) -> None:
        tools = Tools()

        @tools.action(description="only needs one")
        def single(x: str, svc: Inject[_AnotherService] = None) -> str:
            return f"{x}:{svc.name}"

        tools.provide(_FakeContext("ignored"), _AnotherService("used"))
        result = await tools.execute("single", {"x": "hi"})
        assert result == "hi:used"

    @pytest.mark.asyncio
    async def test_no_inject_params_unaffected(self) -> None:
        tools = Tools()

        @tools.action(description="plain tool")
        def plain(x: str) -> str:
            return f"plain:{x}"

        tools.provide(_FakeContext("ctx"))
        result = await tools.execute("plain", {"x": "a"})
        assert result == "plain:a"

    def test_provide_extends_not_replaces(self) -> None:
        tools = Tools()
        tools.provide(_FakeContext("a"))
        tools.provide(_AnotherService("b"))
        assert len(tools._context) == 2

    def test_provide_returns_self_for_chaining(self) -> None:
        tools = Tools()
        result = tools.provide(_FakeContext("a"))
        assert result is tools

    def test_clear_dependencies(self) -> None:
        tools = Tools()
        tools.provide(_FakeContext("a"), _AnotherService("b"))
        tools.clear_dependencies()
        assert len(tools._context) == 0

    def test_clear_dependencies_returns_self(self) -> None:
        tools = Tools()
        result = tools.clear_dependencies()
        assert result is tools

    def test_set_context_replaces_existing_context(self) -> None:
        tools = Tools()
        tools.provide(_FakeContext("a"))

        new_context = ToolContext().provide(_FakeContext("b"))
        tools.set_context(new_context)

        assert tools._context.resolve(_FakeContext).value == "b"

    @pytest.mark.asyncio
    async def test_unannotated_custom_type_not_injected(self) -> None:
        """A custom-class param WITHOUT Inject is NOT injected — it's explicit opt-in."""
        tools = Tools()

        @tools.action(description="bare custom type")
        def bare(x: str, ctx: _FakeContext = None) -> str:
            return f"{x}:{ctx}"

        tools.provide(_FakeContext("should_not_appear"))
        result = await tools.execute("bare", {"x": "hi"})
        assert result == "hi:None"


class TestToolsSchema:
    def test_to_schema_empty(self) -> None:
        tools = Tools()
        assert tools.to_schema() == []

    def test_to_schema_returns_list_of_schemas(self) -> None:
        tools = Tools()

        @tools.action(description="tool a")
        def tool_a(x: str) -> str:
            return x

        @tools.action(description="tool b")
        def tool_b(y: int) -> str:
            return str(y)

        schemas = tools.to_schema()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"tool_a", "tool_b"}


class TestToolsMCPRegistration:
    @pytest.mark.asyncio
    async def test_register_mcp_server_adds_tools(self) -> None:
        mock_tool = Tool(name="mcp_tool", description="from mcp", fn=_dummy_fn)
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = [mock_tool]

        tools = Tools()
        await tools.register_mcp_server(mock_server)

        assert tools.get("mcp_tool") is not None
        assert tools.get("mcp_tool").name == "mcp_tool"
