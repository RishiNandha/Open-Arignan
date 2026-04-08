from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from arignan.paths import resolve_app_home

HAT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def validate_hat_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("hat name must not be empty")
    if cleaned == "auto":
        raise ValueError("'auto' is a runtime selector and cannot be used as a stored hat name")
    if not HAT_NAME_PATTERN.match(cleaned):
        raise ValueError(f"invalid hat name: {name!r}")
    return cleaned


@dataclass(frozen=True, slots=True)
class HatLayout:
    name: str
    root: Path
    vector_index_dir: Path
    bm25_index_dir: Path
    summaries_dir: Path
    map_path: Path

    def ensure(self) -> "HatLayout":
        self.vector_index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.map_path.parent.mkdir(parents=True, exist_ok=True)
        self.map_path.touch(exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class StorageLayout:
    root: Path
    settings_path: Path
    ingestion_log_path: Path
    hats_dir: Path
    global_map_path: Path

    @classmethod
    def from_home(cls, app_home: Path | None = None) -> "StorageLayout":
        root = resolve_app_home(app_home=app_home)
        return cls(
            root=root,
            settings_path=root / "settings.json",
            ingestion_log_path=root / "ingestion_log.jsonl",
            hats_dir=root / "hats",
            global_map_path=root / "hats" / "global_map.md",
        )

    def hat(self, name: str) -> HatLayout:
        validated = validate_hat_name(name)
        hat_root = self.hats_dir / validated
        return HatLayout(
            name=validated,
            root=hat_root,
            vector_index_dir=hat_root / "vector_index",
            bm25_index_dir=hat_root / "bm25_index",
            summaries_dir=hat_root / "summaries",
            map_path=hat_root / "map.md",
        )

    def ensure(self, include_default_hat: bool = True) -> "StorageLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        self.hats_dir.mkdir(parents=True, exist_ok=True)
        self.ingestion_log_path.touch(exist_ok=True)
        self.global_map_path.touch(exist_ok=True)
        if include_default_hat:
            self.hat("default").ensure()
        return self
