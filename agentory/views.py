from dataclasses import dataclass


@dataclass
class ToolCallEvent:
    tool_name: str
    status: str | None


type StreamEvent = str | ToolCallEvent
