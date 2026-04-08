from __future__ import annotations


def preferred_torch_device() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

