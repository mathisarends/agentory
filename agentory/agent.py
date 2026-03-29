from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from llmify import ChatModel
from llmify.messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolResultMessage,
    UserMessage,
)

from agentory.skills import Skill
from agentory.tools.tools import Tools
from agentory.views import StreamEvent, ToolCallEvent

if TYPE_CHECKING:
    from agentory.mcp.server import MCPServer


logger = logging.getLogger(__name__)


class Agent[T]:
    def __init__(
        self,
        instructions: str,
        llm: ChatModel,
        tools: Tools | None = None,
        mcp_servers: list[MCPServer] | None = None,
        skills: list[Skill] | None = None,
        max_iterations: int = 10,
        context: T | None = None,
    ) -> None:
        self.llm = llm
        self._instructions = instructions
        self.tools = tools or Tools()
        self.tools.set_context(context)

        self._mcp_servers = mcp_servers or []
        self._skills = skills or []
        self._max_iterations = max_iterations
        self._context: T | None = None
        
        self._history: list[Message] = [SystemMessage(content=self._build_system_prompt())]

    def _build_system_prompt(self) -> str:
        if not self._skills:
            return self._instructions
        skills_block = "\n\n".join(skill.render() for skill in self._skills)
        return f"{self._instructions}\n\n<skills>\n{skills_block}\n</skills>"
    
    async def __aenter__(self) -> Agent[T]:
        await self._connect_mcp_servers()
        return self
        
    async def __aexit__(self, *_) -> None:
        await self._cleanup_mcp_servers()

    async def run(self, task: str) -> AsyncIterator[StreamEvent]:
        self._history.append(UserMessage(content=task))
        schema = self.tools.to_schema() or None
        iterations = 0

        while True:
            if iterations >= self._max_iterations:
                yield "[max iterations reached]"
                return
            iterations += 1

            response = await self.llm.invoke(self._history, tools=schema)

            if not response.tool_calls:
                content = response.completion or ""
                self._history.append(AssistantMessage(content=content))
                yield content
                return

            self._history.append(
                AssistantMessage(content=None, tool_calls=response.tool_calls)
            )

            for call in response.tool_calls:
                tool = self.tools.get(call.function.name)
                tool_args = json.loads(call.function.arguments)
                yield ToolCallEvent(
                    tool_name=call.function.name,
                    status=tool.render_status(tool_args) if tool else None,
                )
                result = await self.tools.execute(
                    call.function.name, tool_args
                )
                self._history.append(
                    ToolResultMessage(tool_call_id=call.id, content=result)
                )

    async def _connect_mcp_servers(self) -> None:
        for server in self._mcp_servers:
            await server.connect()
            await self.tools.register_mcp_server(server)

    async def _cleanup_mcp_servers(self) -> None:
        for server in self._mcp_servers:
            await server.cleanup()

    def reset(self) -> None:
        self._history = [SystemMessage(content=self._instructions)]