"""Marker for dependency-injected tool parameters."""

from typing import final


@final
class _InjectMarker:
    """Marker for dependency-injected parameters. Use via the ``Inject`` singleton."""

    _instance: "_InjectMarker | None" = None

    def __new__(cls) -> "_InjectMarker":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "Inject"


Inject = _InjectMarker()
"""Use with ``Annotated[SomeType, Inject]`` to mark a parameter for injection via ``tools.provide()``."""
