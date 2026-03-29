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
    async for event in agent.run("What time is it?"):
        if isinstance(event, ToolCallEvent):
            print(f"[tool] {event.tool_name}: {event.status}")
        else:
            print(event)

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
    context: T | None = None,
)
```

The main agent class. Call `agent.run(task)` to get an `AsyncIterator[StreamEvent]` that yields either plain `str` chunks or `ToolCallEvent` objects.

`context` is an arbitrary value (or tuple/list of values) injected into every tool function that declares a matching parameter type. Parameters with non-JSON types (custom classes) are automatically detected and filled from the context.

```python
from agentory import Agent, Tools

class SpotifyClient: ...
class UnsplashClient: ...

tools = Tools()

@tools.action("Search tracks on Spotify.")
async def search_tracks(spotify: SpotifyClient, query: str) -> str:
    ...

@tools.action("Search photos on Unsplash.")
async def search_photos(unsplash: UnsplashClient, query: str) -> str:
    ...

agent = Agent(
    instructions="...",
    llm=llm,
    tools=tools,
    context=(SpotifyClient(), UnsplashClient()),
)
```

Each tool only receives the context objects whose types match its parameter annotations. Primitive-typed parameters (`str`, `int`, `bool`, `float`, `list`, `dict`) are treated as normal LLM-provided arguments.

### `Tools`

A registry that turns plain functions into LLM-callable tools.

```python
tools = Tools()

@tools.action("Fetch the content of a URL.", status=lambda a: a["url"])
async def fetch(url: str) -> str:
    ...
```

- **`description`** – shown to the LLM in the tool schema.
- **`name`** – overrides the function name.
- **`status`** – a string or callable producing a human-readable status shown during streaming.

Type hints on parameters are automatically converted to JSON Schema. Use `Annotated[str, "description"]` to attach per-parameter descriptions.

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

async with server:
    agent = Agent(instructions="...", llm=llm, mcp_servers=[server])
    async for event in agent.run("List files in /tmp"):
        print(event)
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
type StreamEvent = str | ToolCallEvent
```

Events yielded by `agent.run()`. A plain `str` is a text chunk from the LLM; a `ToolCallEvent` signals that a tool is being called.

```python
@dataclass
class ToolCallEvent:
    tool_name: str
    status: str | None
```
