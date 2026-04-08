"""Open Arignan."""

from arignan.runtime_env import configure_text_runtime_environment
from arignan.config import AppConfig, load_config

configure_text_runtime_environment()

__all__ = ["AppConfig", "load_config"]
