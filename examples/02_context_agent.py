"""Agent with typed context — tools receive shared state via the context argument."""

import asyncio
from dataclasses import dataclass, field

from llmify import ChatOpenAI

from agentory import Agent, ToolCallEvent, Tools
from dotenv import load_dotenv
load_dotenv(override=True)

tools = Tools()


@dataclass
class AppContext:
    notes: list[str] = field(default_factory=list)


@tools.action(description="Save a note", status=lambda a: f"Saving note: {a['text']!r}")
def save_note(text: str, context: AppContext) -> str:
    context.notes.append(text)
    return f"Saved: {text!r}"


@tools.action(description="List all saved notes")
def list_notes(context: AppContext) -> str:
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

    async for event in agent.run("Save a note saying 'Buy milk', then list all notes."):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        else:
            print(event)

    print(f"\nFinal notes in context: {ctx.notes}")


if __name__ == "__main__":
    asyncio.run(main())
