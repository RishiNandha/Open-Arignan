from __future__ import annotations

from pathlib import Path

from arignan.paths import (
    APP_HOME_POINTER_FILENAME,
    DEFAULT_HOME_DIRNAME,
    app_home_pointer_path,
    read_persisted_app_home,
    resolve_app_home,
    resolve_settings_path,
    write_persisted_app_home,
)


def test_resolve_app_home_prefers_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom-home"

    resolved = resolve_app_home(app_home=explicit, environ={"ARIGNAN_HOME": "ignored"})

    assert resolved == explicit.resolve()


def test_resolve_app_home_uses_environment_override(tmp_path: Path) -> None:
    env_home = tmp_path / "env-home"

    resolved = resolve_app_home(environ={"ARIGNAN_HOME": str(env_home)})

    assert resolved == env_home.resolve()


def test_resolve_app_home_uses_persisted_pointer(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    persisted_home = tmp_path / "persisted-home"
    write_persisted_app_home(persisted_home)

    resolved = resolve_app_home(environ={})

    assert resolved == persisted_home.resolve()


def test_resolve_app_home_falls_back_to_user_home() -> None:
    original_home = Path.home
    tmp_home = Path.cwd() / ".tmp-test-home"
    tmp_home.mkdir(parents=True, exist_ok=True)
    Path.home = staticmethod(lambda: tmp_home)
    resolved = resolve_app_home(environ={})
    try:
        assert resolved.name == DEFAULT_HOME_DIRNAME
    finally:
        pointer = tmp_home / APP_HOME_POINTER_FILENAME
        if pointer.exists():
            pointer.unlink()
        Path.home = original_home


def test_pointer_helpers_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    custom_home = tmp_path / "custom-home"

    pointer_path = write_persisted_app_home(custom_home)

    assert pointer_path == tmp_path / APP_HOME_POINTER_FILENAME
    assert app_home_pointer_path() == pointer_path
    assert read_persisted_app_home() == custom_home.resolve()


def test_resolve_settings_path_defaults_to_settings_json(app_home: Path) -> None:
    resolved = resolve_settings_path(app_home=app_home)

    assert resolved == app_home.resolve() / "settings.json"
