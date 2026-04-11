"""Agent backed by an MCP server (filesystem tools via npx)."""

import asyncio
from pathlib import Path

from llmify import ChatOpenAI

from agentory import Agent, MCPServerStdio, ToolCallEvent
from agentory.views import AgentResult

from dotenv import load_dotenv

load_dotenv(override=True)

WORK_DIR = Path.home() / "agentory-demo"
WORK_DIR.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    llm = ChatOpenAI(model="gpt-5.4-mini")
    server = MCPServerStdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", WORK_DIR],
        allowed_tools=["read_file", "write_file", "list_directory"],
    )

    agent = Agent(
        instructions=f"You are a file assistant. The working directory is {WORK_DIR}.",
        llm=llm,
        mcp_servers=[server],
    )

    async for event in agent.run(f"List the files in {WORK_DIR}."):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status or ''}")
        elif isinstance(event, AgentResult):
            print(event.output)

    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
