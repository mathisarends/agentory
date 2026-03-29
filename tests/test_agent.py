import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llmify.messages import Function, ToolCall

from agentory.agent import Agent
from agentory.skills import Skill
from agentory.tools.tools import Tools
from agentory.views import ToolCallEvent


def _make_llm_response(completion: str | None = None, tool_calls: list | None = None):
    resp = MagicMock()
    resp.completion = completion
    resp.tool_calls = tool_calls
    return resp


def _make_tool_call(call_id: str, name: str, arguments: dict) -> ToolCall:
    return ToolCall(
        id=call_id,
        function=Function(name=name, arguments=json.dumps(arguments)),
    )


class TestAgentInit:
    def test_default_tools_created(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="test", llm=llm)
        assert agent.tools is not None

    def test_system_prompt_set(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="You are a helper.", llm=llm)
        assert agent._history[0].content == "You are a helper."

    def test_system_prompt_with_skills(self) -> None:
        llm = AsyncMock()
        skill = Skill(name="s", description="d", instructions="Do X")
        agent = Agent(instructions="Base", llm=llm, skills=[skill])
        prompt = agent._history[0].content
        assert "Base" in prompt
        assert "<skills>" in prompt
        assert "Do X" in prompt


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hello!")

        agent = Agent(instructions="test", llm=llm)
        events: list = []
        async for event in agent.run("Hi"):
            events.append(event)

        assert events == ["Hello!"]

    @pytest.mark.asyncio
    async def test_tool_call_flow(self) -> None:
        tools = Tools()

        @tools.action(description="add numbers")
        def add(a: int, b: int) -> str:
            return str(a + b)

        tool_call = _make_tool_call("c1", "add", {"a": 1, "b": 2})
        llm = AsyncMock()
        llm.invoke.side_effect = [
            _make_llm_response(tool_calls=[tool_call]),
            _make_llm_response(completion="The sum is 3"),
        ]

        agent = Agent(instructions="test", llm=llm, tools=tools)
        events: list = []
        async for event in agent.run("add 1+2"):
            events.append(event)

        assert any(isinstance(e, ToolCallEvent) and e.tool_name == "add" for e in events)
        assert "The sum is 3" in events

    @pytest.mark.asyncio
    async def test_max_iterations_stops_loop(self) -> None:
        tool_call = _make_tool_call("c1", "echo", {"x": "hi"})
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(tool_calls=[tool_call])

        tools = Tools()

        @tools.action(description="echo")
        def echo(x: str) -> str:
            return x

        agent = Agent(instructions="test", llm=llm, tools=tools, max_iterations=2)
        events: list = []
        async for event in agent.run("loop"):
            events.append(event)

        assert "[max iterations reached]" in events

    @pytest.mark.asyncio
    async def test_tool_exception_captured_as_error(self) -> None:
        tools = Tools()

        @tools.action(description="fails")
        def bad_tool(x: str) -> str:
            raise RuntimeError("broken")

        tool_call = _make_tool_call("c1", "bad_tool", {"x": "a"})
        llm = AsyncMock()
        llm.invoke.side_effect = [
            _make_llm_response(tool_calls=[tool_call]),
            _make_llm_response(completion="Done"),
        ]

        agent = Agent(instructions="test", llm=llm, tools=tools)
        events: list = []
        async for event in agent.run("do it"):
            events.append(event)

        assert "Done" in events


class TestAgentReset:
    def test_reset_clears_history(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="test", llm=llm)
        agent._history.append(MagicMock())
        agent._history.append(MagicMock())
        assert len(agent._history) > 1

        agent.reset()
        assert len(agent._history) == 1
        assert agent._history[0].content == "test"


class TestAgentAsyncContext:
    @pytest.mark.asyncio
    async def test_aenter_connects_mcp_servers(self) -> None:
        llm = AsyncMock()
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = []

        agent = Agent(instructions="test", llm=llm, mcp_servers=[mock_server])
        async with agent:
            mock_server.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aexit_cleans_up_mcp_servers(self) -> None:
        llm = AsyncMock()
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = []

        agent = Agent(instructions="test", llm=llm, mcp_servers=[mock_server])
        async with agent:
            pass
        mock_server.cleanup.assert_awaited_once()
