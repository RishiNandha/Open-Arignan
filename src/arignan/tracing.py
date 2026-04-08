from __future__ import annotations

from dataclasses import dataclass, field


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
        self._calls.append(
            ModelCallTrace(
                component=component,
                task=task,
                model_name=model_name,
                backend=backend,
                status=status,
                item_count=item_count,
                detail=detail,
            )
        )

    def snapshot(self) -> list[ModelCallTrace]:
        return list(self._calls)
