# agentory

A lightweight Python library for building tool-calling agents.

## Installation

```bash
pip install agentory
```

Install only the providers you actually need

```bash
pip install "agentory[openai]" # + azure
pip install "agentory[anthropic]"
pip install "agentory[all]"
```

Requires Python 3.12+.

## Quickstart

```python
import asyncio
from llmify import ChatOpenAI
from agentory import Agent, Tools, ToolCallEvent
from agentory.views import AgentResult

tools = Tools()

@tools.action("Return the current UTC time as an ISO-8601 string.")
def get_time() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

async def main():
    llm = ChatOpenAI(model="gpt-5.4-mini")
    agent = Agent(
        instructions="You are a helpful assistant.",
        llm=llm,
        tools=tools,
    )
    async for event in agent.stream("What time is it?"):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        elif isinstance(event, AgentResult):
            print(event.output)

asyncio.run(main())
```

## Core API

### `Agent`

```python
Agent(
    instructions: str,
    llm: ChatOpenAI | ChatAzureOpenAI | ChatAnthropic,
    tools: Tools | None = None,
    mcp_servers: list[MCPServer] | None = None,
    skills: list[Skill] | None = None,
    max_iterations: int = 10,
    context: Any | None = None,
    message_store: MessageStore | None = None,
    injectables: tuple[Any, ...] | list[Any] | None = None,
    use_done_tool: bool = False,
)
```

The main agent class provides two execution methods:

- `await agent.run(task)` returns a final `AgentResult`.
- `agent.stream(task)` returns an `AsyncIterator[StreamEvent]` yielding `ToolCallEvent` and the final `AgentResult`.

- `context` is an optional shared dependency object injected into tools via `Inject[YourContextType]`.
- `message_store` is an optional custom message store implementation.
- `injectables` lets you register additional dependencies for `Inject[...]` resolution.
- `use_done_tool` registers a built-in `done` tool that agents can call to terminate explicitly.

MCP servers are connected automatically on the first `run()` call. Call `await agent.close()` when done to clean up MCP connections. The async context manager (`async with`) is also supported as an alternative.

```python
agent = Agent(instructions="...", llm=llm, mcp_servers=[server])
async for event in agent.stream("Do something"):
    print(event)
await agent.close()
```

### `Tools`

A registry that turns plain functions into LLM-callable tools.

```python
tools = Tools()

@tools.action("Fetch the content of a URL.", status_label="Fetching URL")
async def fetch(url: str) -> str:
    ...
```

- **`description`** – shown to the LLM in the tool schema.
- **`name`** – overrides the function name.
- **`status_label`** – either a string, or a callable that receives the typed `params` model and returns a human-readable status shown during streaming.
- **`status`** – backward-compatible alias for `status_label`.
- **`params`** – optional Pydantic model used for argument validation/schema generation.

Type hints on parameters are automatically converted to JSON Schema. Use `Annotated[str, "description"]` to attach per-parameter descriptions.

When `params` is provided, tool functions can receive a single typed model parameter:

```python
from pydantic import BaseModel


class SearchParams(BaseModel):
    query: str
    limit: int = 5


@tools.action(
    "Search docs.",
    params=SearchParams,
    status_label=lambda p: f"Searching {p.query}",
)
def search(params: SearchParams) -> str:
    ...
```

#### Dependency Injection with `Inject`

Use `Inject[...]` to mark parameters for dependency injection. The `Agent` wires tool context automatically and always provides the active message store as injectable. You can add your own dependencies as flat `injectables=[...]` on `Agent`.

```python
from agentory import Agent, Inject, Tools
from agentory.history import MessageStore

class SpotifyClient: ...
class UnsplashClient: ...

tools = Tools()

@tools.action("Search tracks on Spotify.")
async def search_tracks(spotify: Inject[SpotifyClient], query: str) -> str:
    ...

@tools.action("Search photos on Unsplash.")
async def search_photos(unsplash: Inject[UnsplashClient], query: str) -> str:
    ...

@tools.action("Count stored messages.")
def message_count(store: Inject[MessageStore]) -> int:
    return len(store.messages())

agent = Agent(
    instructions="...",
    llm=llm,
    tools=tools,
    injectables=[SpotifyClient(), UnsplashClient()],
)
```

For explicit context management, you can still set or replace a `ToolContext` directly on `Tools`:

```python
from agentory import ToolContext

context = ToolContext().provide(spotify_client, unsplash_client)
tools.set_context(context)
```

### `Tool`

Low-level dataclass representing a single tool. Usually created via `Tools.action`; useful when constructing tools manually or from MCP servers.

### `Skill`

A piece of reusable instructions injected into the system prompt inside a `<skill>` block.

```python
from pathlib import Path
from agentory import Skill

skill = Skill.from_path(Path("my_skill.md"))
# or load SKILL.md from a directory:
skill = Skill.from_directory(Path("skills/my_skill/"))

agent = Agent(instructions="...", llm=llm, skills=[skill])
```

Skill files use optional YAML frontmatter for `name` and `description`:

```markdown
---
name: web-search
description: Search the web for information
---

Use the search tool whenever the user asks about recent events...
```

### `MCPServerStdio`

Connects to any [Model Context Protocol](https://modelcontextprotocol.io) server over stdio and exposes its tools to the agent.

```python
from agentory import Agent, MCPServerStdio

server = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

agent = Agent(instructions="...", llm=llm, mcp_servers=[server])
async for event in agent.stream("List files in /tmp"):
    print(event)
await agent.close()
```

**Options:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `command` | — | Executable to spawn |
| `args` | `[]` | Arguments for the command |
| `env` | `None` | Extra environment variables (inherits current env when `None`) |
| `cache_tools_list` | `True` | Cache tool discovery after first call |
| `allowed_tools` | `None` | Whitelist of tool names to expose |

### `StreamEvent`

```python
type StreamEvent = ToolCallEvent | AgentResult
```

Events yielded by `agent.run()`. A `ToolCallEvent` signals that a tool is being called. `AgentResult` is emitted when the run finishes.

```python
@dataclass
class ToolCallEvent:
    tool_name: str
    status: str | None


@dataclass
class AgentResult:
    output: str
    finish_reason: Literal["done", "max_iterations_reached"]
```
