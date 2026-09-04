"""
angvey-robust
=============

Lightweight robustness layer for AI agents.

Core idea:
    Task → Execute → Reflect → Store learning → Retrieve & improve next time

Makes agents notice their mistakes, remember them, and get better over time.
"""

from .core import RobustAgent, Reflection, ReflectionStore
from .models import TaskResult, FailureCategory, ReflectionRecord

__version__ = "0.1.0"
__all__ = [
    "RobustAgent",
    "Reflection",
    "ReflectionStore",
    "TaskResult",
    "FailureCategory",
    "ReflectionRecord",
]
