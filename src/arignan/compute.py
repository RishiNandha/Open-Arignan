from __future__ import annotations

import gc
from typing import Any


def preferred_torch_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def release_torch_cuda_memory() -> bool:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return False
    if not torch.cuda.is_available():
        return False
    released = False
    for action_name in ("empty_cache", "ipc_collect"):
        action = getattr(torch.cuda, action_name, None)
        if callable(action):
            try:
                action()
                released = True
            except Exception:  # pragma: no cover - depends on local torch runtime
                continue
    return released


def torch_cuda_memory_snapshot() -> dict[str, float] | None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None
    if not torch.cuda.is_available():
        return None
    total = float(torch.cuda.get_device_properties(0).total_memory)
    allocated = float(torch.cuda.memory_allocated())
    reserved = float(torch.cuda.memory_reserved())
    return {
        "allocated_gib": allocated / (1024 ** 3),
        "reserved_gib": reserved / (1024 ** 3),
        "total_gib": total / (1024 ** 3),
    }


def format_torch_cuda_memory(label: str) -> str | None:
    snapshot = torch_cuda_memory_snapshot()
    if snapshot is None:
        return None
    return (
        f"{label}: torch cuda allocated={snapshot['allocated_gib']:.2f} GiB, "
        f"reserved={snapshot['reserved_gib']:.2f} GiB, total={snapshot['total_gib']:.2f} GiB"
    )

