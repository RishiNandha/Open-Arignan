from __future__ import annotations

import threading
import time

from arignan.mcp.server import _LazyArignanApp


def test_lazy_mcp_app_background_loads_retrieval_models_without_timer() -> None:
    class FakeComponent:
        def __init__(self) -> None:
            self.release_calls = 0

        def release_device_memory(self) -> bool:
            self.release_calls += 1
            return True

    class FakeApp:
        def __init__(self) -> None:
            self.embedder = FakeComponent()
            self.reranker = FakeComponent()

    messages: list[str] = []
    fake_app = FakeApp()

    state = _LazyArignanApp(
        app=None,
        app_factory=lambda: fake_app,
        progress_sink=messages.append,
    )

    state.background_load_retrieval_models()

    assert any("Starting background MCP retrieval-model load" in message for message in messages)
    assert any("App init started" in message for message in messages)
    assert any("Background retrieval-model load finished" in message for message in messages)
    assert not any("offload" in message.lower() for message in messages)


def test_lazy_mcp_app_can_release_retrieval_models_on_demand() -> None:
    class FakeComponent:
        def __init__(self) -> None:
            self.release_calls = 0

        def release_device_memory(self) -> bool:
            self.release_calls += 1
            return True

    class FakeApp:
        def __init__(self) -> None:
            self.embedder = FakeComponent()
            self.reranker = FakeComponent()

    messages: list[str] = []
    fake_app = FakeApp()
    state = _LazyArignanApp(
        app=fake_app,
        app_factory=None,
        progress_sink=messages.append,
    )

    state.release_retrieval_models("Releasing retrieval models before local LLM use")

    assert fake_app.embedder.release_calls == 1
    assert fake_app.reranker.release_calls == 1
    assert any("Releasing retrieval models before local LLM use" in message for message in messages)
    assert any("Released embedding model from GPU" in message for message in messages)
    assert any("Released reranking model from GPU" in message for message in messages)


def test_lazy_mcp_app_serializes_overlapping_retrieval_usage() -> None:
    created: list[object] = []
    messages: list[str] = []
    entered: list[str] = []
    first_entered = threading.Event()
    release_first = threading.Event()

    class FakeApp:
        pass

    def build_app():
        app = FakeApp()
        created.append(app)
        return app

    state = _LazyArignanApp(
        app=None,
        app_factory=build_app,
        progress_sink=messages.append,
    )

    def first_worker() -> None:
        with state.retrieval_usage("first") as app:
            entered.append("first")
            first_entered.set()
            release_first.wait(timeout=1.0)
            assert app is created[0]

    def second_worker() -> None:
        with state.retrieval_usage("second") as app:
            entered.append("second")
            assert app is created[0]

    thread_one = threading.Thread(target=first_worker)
    thread_two = threading.Thread(target=second_worker)
    thread_one.start()
    assert first_entered.wait(timeout=1.0)
    thread_two.start()
    time.sleep(0.05)
    assert entered == ["first"]
    release_first.set()
    thread_one.join(timeout=1.0)
    thread_two.join(timeout=1.0)

    assert entered == ["first", "second"]
    assert len(created) == 1
    assert any("waiting for retrieval gate" in message for message in messages)
    assert any("acquired retrieval gate" in message for message in messages)
    assert any("released retrieval gate" in message for message in messages)


def test_lazy_mcp_app_does_not_hold_state_lock_during_slow_init() -> None:
    messages: list[str] = []
    created: list[object] = []
    init_started = threading.Event()
    allow_finish = threading.Event()
    first_done = threading.Event()
    second_done = threading.Event()
    resolved: list[str] = []

    class FakeApp:
        pass

    def build_app():
        init_started.set()
        allow_finish.wait(timeout=1.0)
        app = FakeApp()
        created.append(app)
        return app

    state = _LazyArignanApp(
        app=None,
        app_factory=build_app,
        progress_sink=messages.append,
    )

    def first_worker() -> None:
        app = state.resolve()
        assert app is created[0]
        resolved.append("first")
        first_done.set()

    def second_worker() -> None:
        assert init_started.wait(timeout=1.0)
        app = state.resolve()
        assert app is created[0]
        resolved.append("second")
        second_done.set()

    thread_one = threading.Thread(target=first_worker)
    thread_two = threading.Thread(target=second_worker)
    thread_one.start()
    assert init_started.wait(timeout=1.0)
    thread_two.start()
    time.sleep(0.05)
    assert resolved == []
    allow_finish.set()
    assert first_done.wait(timeout=1.0)
    assert second_done.wait(timeout=1.0)
    thread_one.join(timeout=1.0)
    thread_two.join(timeout=1.0)

    assert resolved == ["first", "second"] or resolved == ["second", "first"]
    assert len(created) == 1
    assert any("App init started" in message for message in messages)
    assert any("Waiting for app init to finish" in message for message in messages)
    assert any("App init finished; reusing initialized app" in message for message in messages)
