from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MessageStore(Protocol):
    def append(self, message: Any) -> None: ...
    def messages(self) -> Sequence[Any]: ...
    def reset(self, system_message: Any) -> None: ...


class InMemoryMessageStore:
    def __init__(self) -> None:
        self._messages: list[Any] = []

    def append(self, message: Any) -> None:
        self._messages.append(message)

    def messages(self) -> Sequence[Any]:
        return self._messages

    def reset(self, system_message: Any) -> None:
        self._messages = [system_message]
