"""Basic agent with a single tool."""

import asyncio

from llmify import ChatOpenAI
from pydantic import BaseModel

from agentory import Agent, ToolCallEvent, Tools
from agentory.views import AgentResult
from dotenv import load_dotenv

load_dotenv(override=True)

tools = Tools()


class AddParams(BaseModel):
    x: float
    y: float


@tools.action(
    description="Add two numbers together",
    params=AddParams,
    status_label=lambda p: f"Adding {p.x} + {p.y}",
)
def add(params: AddParams) -> float:
    return params.x + params.y


@tools.action(description="Multiply two numbers together")
def multiply(x: float, y: float) -> float:
    return x * y


async def main() -> None:
    llm = ChatOpenAI(model="gpt-5.4-mini")
    agent = Agent(
        instructions="You are a helpful math assistant.",
        llm=llm,
        tools=tools,
    )

    async for event in agent.run("What is (3 + 4) * 12?"):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        elif isinstance(event, AgentResult):
            print(event.output)


if __name__ == "__main__":
    asyncio.run(main())
