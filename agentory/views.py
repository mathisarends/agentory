from dataclasses import dataclass
from typing import Literal


@dataclass
class ToolCallEvent:
    tool_name: str
    status: str | None


@dataclass
class AgentResult:
    output: str
    finish_reason: Literal["done", "max_iterations_reached"]


type StreamEvent = ToolCallEvent | AgentResult
