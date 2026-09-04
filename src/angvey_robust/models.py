"""Data models for angvey-robust."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class FailureCategory(str, Enum):
    """Common failure categories observed in agentic systems."""

    NONE = "none"
    REASONING = "reasoning"          # Bad planning / logic error
    TOOL_USE = "tool_use"           # Wrong tool, bad arguments, tool failure
    HALLUCINATION = "hallucination" # Fabricated facts or data
    CONTEXT = "context"             # Context overflow, lost information
    PLANNING = "planning"           # Poor task decomposition
    EXECUTION = "execution"         # Runtime / environment error
    UNKNOWN = "unknown"


@dataclass
class TaskResult:
    """Outcome of an agent task execution."""

    task: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    tools_used: List[str] = field(default_factory=list)
    steps: int = 0
    duration_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReflectionRecord:
    """Structured self-reflection produced after a task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    task: str = ""
    success: bool = False
    confidence: float = 0.5          # 0.0 – 1.0
    what_worked: str = ""
    what_failed: str = ""
    root_cause: FailureCategory = FailureCategory.NONE
    lessons: str = ""
    suggested_improvement: str = ""
    tools_used: List[str] = field(default_factory=list)
    steps: int = 0
    duration_seconds: Optional[float] = None
    raw_output: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["root_cause"] = self.root_cause.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReflectionRecord":
        data = dict(data)
        if "root_cause" in data and isinstance(data["root_cause"], str):
            data["root_cause"] = FailureCategory(data["root_cause"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ImprovementSuggestion:
    """A concrete, versioned improvement proposal."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    title: str = ""
    description: str = ""
    category: str = "prompt"  # prompt | tool | memory | process
    priority: str = "medium"  # low | medium | high
    based_on_reflections: List[str] = field(default_factory=list)
    status: str = "proposed"  # proposed | accepted | rejected | applied
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
