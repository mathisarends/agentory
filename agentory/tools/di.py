from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Self, TypeVar, final

T = TypeVar("T")


@final
class ToolContext:
    """Dependency container for tool functions.

    Usage::

        ctx = ToolContext(notion_client, llm)

        @tools.action("Create page", param_model=CreatePageParams)
        async def create_page(
            params: CreatePageParams,
            client: Inject[NotionClient],
        ) -> CreatePageResult:
            ...
    """

    # --- Inject marker (inner class so it lives on ToolContext) ---

    _marker_instance: _InjectMarker | None = None  # set after class body

    if TYPE_CHECKING:
        type Inject[T] = T
    else:

        class Inject:
            def __class_getitem__(cls, item: Any) -> Any:
                return Annotated[item, ToolContext._get_marker()]

    @staticmethod
    def _get_marker() -> _InjectMarker:
        if ToolContext._marker_instance is None:
            ToolContext._marker_instance = _InjectMarker()
        return ToolContext._marker_instance

    # --- Container ---

    def __init__(self, *dependencies: Any) -> None:
        self._dependencies: list[Any] = list(dependencies)

    def provide(self, *dependencies: Any) -> Self:
        self._dependencies.extend(dependencies)
        return self

    def clear(self) -> Self:
        self._dependencies.clear()
        return self

    def resolve(self, expected_type: type[T]) -> T | None:
        for dep in self._dependencies:
            try:
                if isinstance(dep, expected_type):
                    return dep
            except TypeError:
                continue
        return None

    def __len__(self) -> int:
        return len(self._dependencies)


@final
class _InjectMarker:
    def __repr__(self) -> str:
        return "ToolContext.Inject"


Inject = ToolContext.Inject
