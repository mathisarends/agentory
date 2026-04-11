import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from llmify.messages import Function, ToolCall, ToolResultMessage, UserMessage
from pydantic import BaseModel

from agentory.agent import Agent
from agentory.skills import Skill
from agentory.tools.tools import Tools
from agentory.views import AgentResult, ToolCallEvent


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


class _RecordingHistoryManager:
    def __init__(self) -> None:
        self.items: list = []
        self.reset_calls = 0

    def append(self, message) -> None:
        self.items.append(message)

    def messages(self):
        return self.items

    def reset(self, system_message) -> None:
        self.reset_calls += 1
        self.items = [system_message]


class TestAgentInit:
    def test_default_tools_created(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="test", llm=llm)
        assert agent.tools is not None

    def test_system_prompt_set(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="You are a helper.", llm=llm)
        assert list(agent._message_store.messages())[0].content == "You are a helper."

    def test_system_prompt_with_skills(self) -> None:
        llm = AsyncMock()
        skill = Skill(name="s", description="d", instructions="Do X")
        agent = Agent(instructions="Base", llm=llm, skills=[skill])
        prompt = list(agent._message_store.messages())[0].content
        assert "Base" in prompt
        assert "<skills>" in prompt
        assert '<skill name="s">' in prompt
        assert "<description>d</description>" in prompt
        # Level 1: only metadata in system prompt, NOT full instructions
        assert "Do X" not in prompt


class TestAgentSkillTools:
    @pytest.mark.asyncio
    async def test_read_skill_tool_returns_full_instructions(self) -> None:
        llm = AsyncMock()
        skill = Skill(
            name="research",
            description="How to research",
            instructions="Step 1\nStep 2",
        )
        agent = Agent(instructions="Base", llm=llm, skills=[skill])

        result = await agent.tools.execute("read_skill", {"skill_name": "research"})
        assert "Step 1" in result
        assert "Step 2" in result

    @pytest.mark.asyncio
    async def test_read_skill_tool_unknown_skill(self) -> None:
        llm = AsyncMock()
        skill = Skill(name="research", description="d", instructions="x")
        agent = Agent(instructions="Base", llm=llm, skills=[skill])

        result = await agent.tools.execute("read_skill", {"skill_name": "nope"})
        assert "Unknown skill" in result
        assert "research" in result

    @pytest.mark.asyncio
    async def test_read_skill_file_returns_content(self, tmp_path) -> None:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: my-skill\ndescription: d\n---\nMain instructions.",
            encoding="utf-8",
        )
        (tmp_path / "FORMS.md").write_text("Form instructions here.", encoding="utf-8")
        skill = Skill.from_directory(tmp_path)

        llm = AsyncMock()
        agent = Agent(instructions="Base", llm=llm, skills=[skill])

        result = await agent.tools.execute(
            "read_skill_file", {"skill_name": "my-skill", "filename": "FORMS.md"}
        )
        assert "Form instructions here." in result

    @pytest.mark.asyncio
    async def test_read_skill_file_missing_file(self) -> None:
        llm = AsyncMock()
        skill = Skill(name="s", description="d", instructions="x")
        agent = Agent(instructions="Base", llm=llm, skills=[skill])

        result = await agent.tools.execute(
            "read_skill_file", {"skill_name": "s", "filename": "nope.md"}
        )
        assert "not found" in result.lower() or "no source directory" in result.lower()

    @pytest.mark.asyncio
    async def test_read_skill_lists_bundled_files(self, tmp_path) -> None:
        (tmp_path / "SKILL.md").write_text(
            "---\nname: pdf\ndescription: PDF stuff\n---\nInstructions.",
            encoding="utf-8",
        )
        (tmp_path / "FORMS.md").write_text("forms", encoding="utf-8")
        (tmp_path / "REFERENCE.md").write_text("ref", encoding="utf-8")
        skill = Skill.from_directory(tmp_path)

        llm = AsyncMock()
        agent = Agent(instructions="Base", llm=llm, skills=[skill])

        result = await agent.tools.execute("read_skill", {"skill_name": "pdf"})
        assert "FORMS.md" in result
        assert "REFERENCE.md" in result

    def test_no_skill_tools_when_no_skills(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="Base", llm=llm)
        assert agent.tools.get("read_skill") is None
        assert agent.tools.get("read_skill_file") is None

    @pytest.mark.asyncio
    async def test_skill_tools_in_schema(self) -> None:
        llm = AsyncMock()
        skill = Skill(name="s", description="d", instructions="x")
        agent = Agent(instructions="Base", llm=llm, skills=[skill])
        schema = agent.tools.to_schema()
        tool_names = [t["function"]["name"] for t in schema]
        assert "read_skill" in tool_names
        assert "read_skill_file" in tool_names


class TestAgentRun:
    @pytest.mark.asyncio
    async def test_simple_text_response(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hello!")

        agent = Agent(instructions="test", llm=llm)
        result = await agent.run("Hi")

        assert isinstance(result, AgentResult)
        assert result.output == "Hello!"

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
        async for event in agent.stream("add 1+2"):
            events.append(event)

        assert any(
            isinstance(e, ToolCallEvent) and e.tool_name == "add" for e in events
        )
        assert any(
            isinstance(e, AgentResult) and e.output == "The sum is 3" for e in events
        )

    @pytest.mark.asyncio
    async def test_tool_call_event_renders_status_label(self) -> None:
        tools = Tools()

        class _EchoParams(BaseModel):
            x: str

        @tools.action(
            description="echo",
            params=_EchoParams,
            status_label=lambda p: f"Echo {p.x}",
        )
        def echo(params: _EchoParams) -> str:
            return params.x

        tool_call = _make_tool_call("c1", "echo", {"x": "hi"})
        llm = AsyncMock()
        llm.invoke.side_effect = [
            _make_llm_response(tool_calls=[tool_call]),
            _make_llm_response(completion="done"),
        ]

        agent = Agent(instructions="test", llm=llm, tools=tools)
        events: list = []
        async for event in agent.stream("echo hi"):
            events.append(event)

        assert any(
            isinstance(e, ToolCallEvent)
            and e.tool_name == "echo"
            and e.status == "Echo hi"
            for e in events
        )

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
        async for event in agent.stream("loop"):
            events.append(event)

        assert any(
            isinstance(e, AgentResult) and e.finish_reason == "max_iterations_reached"
            for e in events
        )

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
        result = await agent.run("do it")

        assert result.output == "Done"

    @pytest.mark.asyncio
    async def test_non_string_tool_result_is_serialized_for_history(self) -> None:
        tools = Tools()

        @tools.action(description="returns object")
        def to_object(x: int) -> dict:
            return {"value": x}

        tool_call = _make_tool_call("c1", "to_object", {"x": 7})
        llm = AsyncMock()
        llm.invoke.side_effect = [
            _make_llm_response(tool_calls=[tool_call]),
            _make_llm_response(completion="ok"),
        ]

        agent = Agent(instructions="test", llm=llm, tools=tools)
        await agent.run("run")

        tool_messages = [
            m
            for m in agent._message_store.messages()
            if isinstance(m, ToolResultMessage)
        ]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == '{"value": 7}'

    @pytest.mark.asyncio
    async def test_agent_uses_custom_history_manager(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hello")
        history_manager = _RecordingHistoryManager()

        agent = Agent(
            instructions="test",
            llm=llm,
            message_store=history_manager,
        )
        await agent.run("Hi")

        assert history_manager.reset_calls == 1
        assert any(isinstance(msg, UserMessage) for msg in history_manager.items)


class TestAgentReset:
    def test_reset_clears_history(self) -> None:
        llm = AsyncMock()
        agent = Agent(instructions="test", llm=llm)
        agent._message_store.append(MagicMock())
        agent._message_store.append(MagicMock())
        assert len(list(agent._message_store.messages())) > 1

        agent.reset()
        assert len(list(agent._message_store.messages())) == 1
        assert list(agent._message_store.messages())[0].content == "test"


class TestAgentMCP:
    @pytest.mark.asyncio
    async def test_run_connects_mcp_servers(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hi")
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = []

        agent = Agent(instructions="test", llm=llm, mcp_servers=[mock_server])
        await agent.run("Hi")

        mock_server.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_connects_only_once(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hi")
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = []

        agent = Agent(instructions="test", llm=llm, mcp_servers=[mock_server])
        await agent.run("Hi")
        await agent.run("Hi again")

        mock_server.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_cleans_up_mcp_servers(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hi")
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = []

        agent = Agent(instructions="test", llm=llm, mcp_servers=[mock_server])
        await agent.run("Hi")
        await agent.close()

        mock_server.cleanup.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_async_context_manager_still_works(self) -> None:
        llm = AsyncMock()
        llm.invoke.return_value = _make_llm_response(completion="Hi")
        mock_server = AsyncMock()
        mock_server.list_tools.return_value = []

        agent = Agent(instructions="test", llm=llm, mcp_servers=[mock_server])
        async with agent:
            mock_server.connect.assert_awaited_once()
        mock_server.cleanup.assert_awaited_once()
