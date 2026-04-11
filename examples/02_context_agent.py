"""Agent with dependency injection — tools receive shared state via Inject."""

import asyncio
from dataclasses import dataclass, field

from llmify import ChatOpenAI
from pydantic import BaseModel

from agentory import Agent, Inject, ToolCallEvent, Tools
from agentory.views import AgentResult
from dotenv import load_dotenv

load_dotenv(override=True)

tools = Tools()


@dataclass
class AppContext:
    notes: list[str] = field(default_factory=list)


class SaveNoteParams(BaseModel):
    text: str


@tools.action(
    description="Save a note",
    params=SaveNoteParams,
    status_label=lambda p: f"Saving note: {p.text!r}",
)
def save_note(params: SaveNoteParams, context: Inject[AppContext]) -> str:
    context.notes.append(params.text)
    return f"Saved: {params.text!r}"


@tools.action(description="List all saved notes")
def list_notes(context: Inject[AppContext]) -> str:
    if not context.notes:
        return "No notes saved yet."
    return "\n".join(f"- {n}" for n in context.notes)


async def main() -> None:
    ctx = AppContext()

    llm = ChatOpenAI(model="gpt-5.4-mini")
    agent = Agent(
        instructions="You are a note-taking assistant.",
        llm=llm,
        tools=tools,
        context=ctx,
    )

    async for event in agent.stream(
        "Save a note saying 'Buy milk', then list all notes."
    ):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        elif isinstance(event, AgentResult):
            print(event.output)

    print(f"\nFinal notes in context: {ctx.notes}")


if __name__ == "__main__":
    asyncio.run(main())
