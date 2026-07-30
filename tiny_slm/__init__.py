"""TinySLM — a from-scratch small language model for low-RAM devices."""

from .config import TinySLMConfig
from .model import TinySLM

__version__ = "0.2.4"
__all__ = ["TinySLM", "TinySLMConfig", "__version__"]
