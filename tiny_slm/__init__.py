"""TinySLM — a from-scratch small language model for low-RAM devices."""

from .config import TinySLMConfig
from .model import TinySLM

__all__ = ["TinySLM", "TinySLMConfig"]
