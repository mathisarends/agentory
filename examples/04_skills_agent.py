"""Agent with skills — reusable prompt fragments injected into the system prompt."""

import asyncio

from llmify import ChatOpenAI
from pydantic import BaseModel

from agentory import Agent, Skill, ToolCallEvent, Tools
from agentory.views import AgentResult

from dotenv import load_dotenv

load_dotenv(override=True)

tools = Tools()


class WebSearchParams(BaseModel):
    query: str


@tools.action(
    description="Search the web for a query",
    params=WebSearchParams,
    status_label=lambda p: f"Searching: {p.query!r}",
)
async def web_search(params: WebSearchParams) -> str:
    # Stub — replace with a real search call.
    return f"[search results for '{params.query}']"


RESEARCH_SKILL = Skill(
    name="research",
    description="How to conduct research tasks",
    instructions=(
        "When asked to research a topic:\n"
        "1. Break the topic into 2-3 focused search queries.\n"
        "2. Run each query with web_search.\n"
        "3. Synthesize results into a concise summary."
    ),
)


async def main() -> None:
    llm = ChatOpenAI(model="gpt-5.4-mini")
    agent = Agent(
        instructions="You are a research assistant.",
        llm=llm,
        tools=tools,
        skills=[RESEARCH_SKILL],
    )

    async for event in agent.run("Research the history of the Model Context Protocol."):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        elif isinstance(event, AgentResult):
            print(event.output)


if __name__ == "__main__":
    asyncio.run(main())
