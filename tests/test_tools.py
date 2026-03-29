from unittest.mock import AsyncMock

import pytest

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


class TestToolsContext:
    @pytest.mark.asyncio
    async def test_single_context_forwarded_to_tool(self) -> None:
        tools = Tools()

        @tools.action(description="uses context")
        def ctx_tool(x: str, ctx: _FakeContext = None) -> str:
            return f"{x}:{ctx.value}"

        tools.set_context(_FakeContext("my_context"))
        result = await tools.execute("ctx_tool", {"x": "val"})
        assert result == "val:my_context"

    @pytest.mark.asyncio
    async def test_multiple_contexts_injected(self) -> None:
        tools = Tools()

        @tools.action(description="needs both")
        def multi(x: str, ctx: _FakeContext = None, svc: _AnotherService = None) -> str:
            return f"{x}:{ctx.value}:{svc.name}"

        tools.set_context(_FakeContext("fc"), _AnotherService("as"))
        result = await tools.execute("multi", {"x": "hi"})
        assert result == "hi:fc:as"

    @pytest.mark.asyncio
    async def test_tool_only_gets_matching_context(self) -> None:
        tools = Tools()

        @tools.action(description="only needs one")
        def single(x: str, svc: _AnotherService = None) -> str:
            return f"{x}:{svc.name}"

        tools.set_context(_FakeContext("ignored"), _AnotherService("used"))
        result = await tools.execute("single", {"x": "hi"})
        assert result == "hi:used"

    @pytest.mark.asyncio
    async def test_no_context_params_unaffected(self) -> None:
        tools = Tools()

        @tools.action(description="plain tool")
        def plain(x: str) -> str:
            return f"plain:{x}"

        tools.set_context(_FakeContext("ctx"))
        result = await tools.execute("plain", {"x": "a"})
        assert result == "plain:a"


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
