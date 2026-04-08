from __future__ import annotations

import os
from pathlib import Path

DEFAULT_HOME_DIRNAME = ".arignan"
APP_HOME_POINTER_FILENAME = ".arignan-home"


def app_home_pointer_path(home: Path | None = None) -> Path:
    return ((home or Path.home()).expanduser().resolve() / APP_HOME_POINTER_FILENAME).resolve()


def read_persisted_app_home(home: Path | None = None) -> Path | None:
    pointer_path = app_home_pointer_path(home=home)
    if not pointer_path.exists():
        return None
    value = pointer_path.read_text(encoding="utf-8").strip()
    if not value:
        return None
    return Path(value).expanduser().resolve()


def write_persisted_app_home(app_home: Path, home: Path | None = None) -> Path:
    pointer_path = app_home_pointer_path(home=home)
    pointer_path.write_text(str(Path(app_home).expanduser().resolve()) + "\n", encoding="utf-8")
    return pointer_path


def resolve_app_home(
    app_home: Path | None = None,
    environ: dict[str, str] | None = None,
) -> Path:
    if app_home is not None:
        return Path(app_home).expanduser().resolve()

    env = environ or os.environ
    if env.get("ARIGNAN_HOME"):
        return Path(env["ARIGNAN_HOME"]).expanduser().resolve()

    persisted = read_persisted_app_home()
    if persisted is not None:
        return persisted

    return (Path.home() / DEFAULT_HOME_DIRNAME).resolve()


def resolve_settings_path(
    settings_path: Path | None = None,
    app_home: Path | None = None,
) -> Path:
    if settings_path is not None:
        return Path(settings_path).expanduser().resolve()
    return resolve_app_home(app_home=app_home) / "settings.json"
