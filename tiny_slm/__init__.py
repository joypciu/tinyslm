"""TinySLM — a from-scratch small language model for low-RAM devices."""

from .config import TinySLMConfig
from .model import TinySLM

__version__ = "0.3.0"
__all__ = ["TinySLM", "TinySLMConfig", "__version__"]
