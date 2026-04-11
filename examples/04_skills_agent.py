"""Agent with skills — progressive disclosure of expertise.

Skills use Anthropic's 3-level progressive disclosure:
  Level 1: Only name + description in the system prompt (metadata)
  Level 2: Full SKILL.md instructions loaded on-demand via read_skill tool
  Level 3: Additional bundled files loaded via read_skill_file tool
"""

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

    async for event in agent.stream(
        "Research the history of the Model Context Protocol."
    ):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        elif isinstance(event, AgentResult):
            print(event.output)


if __name__ == "__main__":
    asyncio.run(main())
