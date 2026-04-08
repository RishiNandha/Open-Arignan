from __future__ import annotations

import json
from pathlib import Path

from arignan.models import LoadEvent


class IngestionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(self, event: LoadEvent) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            json.dump(event.to_dict(), handle)
            handle.write("\n")

    def read_all(self) -> list[LoadEvent]:
        events: list[LoadEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                events.append(LoadEvent.from_dict(json.loads(stripped)))
        return events

    def find_by_load_id(self, load_id: str) -> list[LoadEvent]:
        return [event for event in self.read_all() if event.load_id == load_id]
