from __future__ import annotations

import os
from collections.abc import MutableMapping

TEXT_RUNTIME_ENVIRONMENT = {
    "TRANSFORMERS_NO_TF": "1",
    "USE_TF": "0",
    "TRANSFORMERS_NO_FLAX": "1",
    "USE_FLAX": "0",
}


def configure_text_runtime_environment(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Force Transformers into the text-only PyTorch path used by Arignan."""

    env = os.environ if environ is None else environ
    applied: dict[str, str] = {}
    for key, value in TEXT_RUNTIME_ENVIRONMENT.items():
        env[key] = value
        applied[key] = value
    return applied
