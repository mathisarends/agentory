import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Self

from llmify import ChatModel
from llmify.messages import (
    AssistantMessage,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)
from pydantic import BaseModel

from agentory.history import MessageStore, InMemoryMessageStore
from agentory.skills import Skill
from agentory.tools import ToolContext, Tools
from agentory.tools.views import DoneParams
from agentory.views import AgentResult, StreamEvent, ToolCallEvent

from agentory.mcp.server import MCPServer

logger = logging.getLogger(__name__)


class Agent[Context]:
    def __init__(
        self,
        instructions: str,
        llm: ChatModel,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        skills: list[Skill] | None = None,
        max_iterations: int = 10,
        context: Context | None = None,
        message_store: MessageStore | None = None,
        injectables: tuple[Any, ...] | list[Any] | None = None,
        use_done_tool: bool = False,
    ) -> None:
        self.llm = llm
        self._instructions = instructions
        self.tools = tools or Tools(use_done_tool=use_done_tool)
        self._mcp_servers = mcp_servers or []
        self._skills = skills or []
        self._max_iterations = max_iterations
        self._context = context
        self._message_store = message_store or InMemoryMessageStore()
        self._injectables = tuple(injectables or ())

        self._wire_tool_context()

        self._mcp_connected = False

        self._message_store.reset(SystemMessage(content=self._build_system_prompt()))

    def _wire_tool_context(self) -> None:
        dependencies: list[Any] = [self._message_store]
        if self._context is not None:
            dependencies.append(self._context)
        dependencies.extend(self._injectables)
        self.tools.set_context(ToolContext(*dependencies))

    def _build_system_prompt(self) -> str:
        if not self._skills:
            return self._instructions
        skills_block = "\n\n".join(skill.render() for skill in self._skills)
        return f"{self._instructions}\n\n<skills>\n{skills_block}\n</skills>"

    async def __aenter__(self) -> Self:
        await self._connect_mcp_servers()
        return self

    async def __aexit__(self, *_) -> None:
        await self._cleanup_mcp_servers()

    async def run(self, task: str) -> AsyncGenerator[StreamEvent]:
        await self._connect_mcp_servers()
        self._message_store.append(UserMessage(content=task))
        schema = self.tools.to_schema() or None
        iterations = 0

        while True:
            if iterations >= self._max_iterations:
                yield AgentResult(output="", finish_reason="max_iterations_reached")
                return
            iterations += 1

            response = await self.llm.invoke(
                list(self._message_store.messages()), tools=schema
            )

            if not response.tool_calls:
                content = response.completion or ""
                self._message_store.append(AssistantMessage(content=content))
                yield AgentResult(output=content, finish_reason="done")
                return

            self._message_store.append(
                AssistantMessage(content=None, tool_calls=response.tool_calls)
            )

            for call in response.tool_calls:
                function = call.function
                tool_args = json.loads(function.arguments)

                if self.tools.is_done_tool(function.name):
                    params = DoneParams.model_validate(tool_args)
                    yield AgentResult(output=params.output, finish_reason="done")
                    return

                tool = self.tools.get(function.name)
                yield ToolCallEvent(
                    tool_name=function.name,
                    status=tool.render_status(tool_args) if tool else None,
                )
                try:
                    result = await self.tools.execute(function.name, tool_args)
                except Exception as e:
                    result = json.dumps({"error": str(e), "tool": function.name})

                self._message_store.append(
                    ToolResultMessage(
                        tool_call_id=call.id,
                        content=self._serialize_tool_result(result),
                    )
                )

    async def _connect_mcp_servers(self) -> None:
        if self._mcp_connected:
            return
        for server in self._mcp_servers:
            await server.connect()
            await self.tools.register_mcp_server(server)
        self._mcp_connected = True

    async def _cleanup_mcp_servers(self) -> None:
        if not self._mcp_connected:
            return
        for server in self._mcp_servers:
            await server.cleanup()
        self._mcp_connected = False

    async def close(self) -> None:
        await self._cleanup_mcp_servers()

    def reset(self) -> None:
        self._message_store.reset(SystemMessage(content=self._build_system_prompt()))

    @staticmethod
    def _serialize_tool_result(result: Any) -> str:
        if isinstance(result, BaseModel):
            return result.model_dump_json()
        if isinstance(result, str):
            return result
        if result is None:
            return ""
        try:
            return json.dumps(result)
        except TypeError:
            return str(result)
