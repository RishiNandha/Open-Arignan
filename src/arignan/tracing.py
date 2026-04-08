from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(slots=True)
class ModelCallTrace:
    component: str
    task: str
    model_name: str
    backend: str
    status: str = "ok"
    item_count: int | None = None
    detail: str | None = None


@dataclass(slots=True)
class ModelTraceCollector:
    _calls: list[ModelCallTrace] = field(default_factory=list)
    on_record: Callable[[ModelCallTrace], None] | None = None

    def clear(self) -> None:
        self._calls.clear()

    def record(
        self,
        *,
        component: str,
        task: str,
        model_name: str,
        backend: str,
        status: str = "ok",
        item_count: int | None = None,
        detail: str | None = None,
    ) -> None:
        call = ModelCallTrace(
            component=component,
            task=task,
            model_name=model_name,
            backend=backend,
            status=status,
            item_count=item_count,
            detail=detail,
        )
        self._calls.append(call)
        if self.on_record is not None:
            self.on_record(call)

    def snapshot(self) -> list[ModelCallTrace]:
        return list(self._calls)
