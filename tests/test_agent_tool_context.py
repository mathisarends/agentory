import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from llmify.messages import Function, ToolCall, ToolResultMessage

from agentory.agent import Agent
from agentory.history import MessageStore
from agentory.tools import Inject, ToolContext, Tools


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


class _AgentContext:
    def __init__(self, name: str) -> None:
        self.name = name


class _Service:
    def __init__(self, name: str) -> None:
        self.name = name


class _LegacyDependency:
    pass


class TestAgentToolContextWiring:
    def test_agent_sets_tool_context_with_message_store(self) -> None:
        llm = AsyncMock()
        tools = Tools()

        agent = Agent(instructions="test", llm=llm, tools=tools)

        injected_store = tools._context.resolve(MessageStore)
        assert injected_store is agent._message_store

    def test_agent_replaces_existing_tool_context(self) -> None:
        llm = AsyncMock()
        tools = Tools(context=ToolContext(_LegacyDependency()))

        agent = Agent(
            instructions="test",
            llm=llm,
            tools=tools,
            injectables=[_Service("svc")],
        )

        assert tools._context.resolve(_LegacyDependency) is None
        assert tools._context.resolve(_Service) is not None
        assert tools._context.resolve(_Service).name == "svc"
        assert tools._context.resolve(MessageStore) is agent._message_store

    @pytest.mark.asyncio
    async def test_agent_passes_context_and_flat_injectables_to_tool_execution(
        self,
    ) -> None:
        tools = Tools()

        @tools.action(description="resolve deps")
        def resolve_deps(
            context: Inject[_AgentContext],
            service: Inject[_Service],
        ) -> str:
            return f"{context.name}:{service.name}"

        tool_call = _make_tool_call("c1", "resolve_deps", {})
        llm = AsyncMock()
        llm.invoke.side_effect = [
            _make_llm_response(tool_calls=[tool_call]),
            _make_llm_response(completion="done"),
        ]

        agent = Agent(
            instructions="test",
            llm=llm,
            tools=tools,
            context=_AgentContext("ctx"),
            injectables=[_Service("svc")],
        )

        await agent.run("run")

        tool_messages = [
            msg
            for msg in agent._message_store.messages()
            if isinstance(msg, ToolResultMessage)
        ]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "ctx:svc"
