from __future__ import annotations

from arignan.mcp.server import _LazyArignanApp


def test_lazy_mcp_app_prewarms_and_releases_retrieval_models(monkeypatch) -> None:
    scheduled: list[tuple[float, object, object]] = []

    class FakeTimer:
        def __init__(self, interval, callback) -> None:
            self.interval = interval
            self.callback = callback
            self.cancelled = False
            scheduled.append((interval, callback, self))

        def start(self) -> None:
            return None

        def cancel(self) -> None:
            self.cancelled = True

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
    monkeypatch.setattr("arignan.mcp.server.threading.Timer", FakeTimer)

    state = _LazyArignanApp(
        app=None,
        app_factory=lambda: fake_app,
        progress_sink=messages.append,
        retrieval_keep_alive_seconds=12,
    )

    state.prewarm_retrieval_models()

    assert messages[0] == "Loading Arignan app"
    assert "Embedding and reranking models warm-started" in messages[-1]
    assert scheduled and scheduled[0][0] == 12

    callback = scheduled[0][1]
    callback()

    assert fake_app.embedder.release_calls == 1
    assert fake_app.reranker.release_calls == 1
    assert any("Released embedding model from GPU" in message for message in messages)
    assert any("Released reranking model from GPU" in message for message in messages)
