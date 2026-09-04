"""Core robustness engine."""

from __future__ import annotations

from typing import Optional, List, Dict, Any, Callable
import re

from .models import (
    TaskResult,
    ReflectionRecord,
    FailureCategory,
    ImprovementSuggestion,
)
from .store import ReflectionStore


class Reflection:
    """
    Helper that creates structured reflections.

    Can work with or without an LLM.
    When no LLM is provided, it uses simple heuristics.
    """

    def __init__(self, llm_callable: Optional[Callable[[str], str]] = None):
        """
        Args:
            llm_callable: Optional function that takes a prompt and returns text.
                          Signature: (prompt: str) -> str
        """
        self.llm = llm_callable

    def create(
        self,
        result: TaskResult,
        extra_context: str = "",
    ) -> ReflectionRecord:
        """Create a structured reflection from a task result."""
        if self.llm is not None:
            return self._create_with_llm(result, extra_context)
        return self._create_heuristic(result)

    def _create_heuristic(self, result: TaskResult) -> ReflectionRecord:
        """Rule-based reflection (no LLM required)."""
        success = result.success
        confidence = 0.8 if success else 0.4

        if success:
            what_worked = f"Task completed successfully. Output length: {len(result.output)} chars."
            what_failed = ""
            root_cause = FailureCategory.NONE
            lessons = "Continue using the same approach for similar tasks."
            suggested = ""
        else:
            error_text = (result.error or result.output or "").lower()
            what_worked = "Some steps may have progressed before failure."
            what_failed = result.error or "Task did not complete successfully."

            # Simple root-cause heuristics
            if any(k in error_text for k in ["tool", "function", "argument", "api"]):
                root_cause = FailureCategory.TOOL_USE
            elif any(k in error_text for k in ["hallucin", "fabricat", "made up", "invented"]):
                root_cause = FailureCategory.HALLUCINATION
            elif any(k in error_text for k in ["context", "token", "overflow", "too long"]):
                root_cause = FailureCategory.CONTEXT
            elif any(k in error_text for k in ["plan", "step", "decompose"]):
                root_cause = FailureCategory.PLANNING
            elif any(k in error_text for k in ["reason", "logic", "incorrect"]):
                root_cause = FailureCategory.REASONING
            else:
                root_cause = FailureCategory.UNKNOWN

            lessons = f"Detected possible {root_cause.value} issue. Review tool usage and intermediate steps."
            suggested = (
                f"Consider adding stricter validation for {root_cause.value} "
                "and retrieving similar past failures before retrying."
            )

        return ReflectionRecord(
            task=result.task,
            success=success,
            confidence=confidence,
            what_worked=what_worked,
            what_failed=what_failed,
            root_cause=root_cause,
            lessons=lessons,
            suggested_improvement=suggested,
            tools_used=result.tools_used,
            steps=result.steps,
            duration_seconds=result.duration_seconds,
            raw_output=result.output[:2000] if result.output else "",
            metadata=result.metadata.copy(),
        )

    def _create_with_llm(
        self, result: TaskResult, extra_context: str = ""
    ) -> ReflectionRecord:
        """Use an LLM to produce a richer reflection."""
        prompt = f"""You are a careful AI agent performing self-reflection.

Task:
{result.task}

Success: {result.success}
Output / Error:
{result.output or result.error or "(none)"}

Tools used: {", ".join(result.tools_used) or "none"}
Steps taken: {result.steps}
{extra_context}

Produce a structured reflection in exactly this format:

CONFIDENCE: <float between 0.0 and 1.0>
WHAT_WORKED: <short text>
WHAT_FAILED: <short text or "none">
ROOT_CAUSE: <one of: none, reasoning, tool_use, hallucination, context, planning, execution, unknown>
LESSONS: <1-2 sentences>
SUGGESTED_IMPROVEMENT: <concrete next action or "none">
"""

        try:
            response = self.llm(prompt)
            return self._parse_llm_response(result, response)
        except Exception as e:
            # Fallback to heuristic if LLM fails
            record = self._create_heuristic(result)
            record.metadata["llm_error"] = str(e)
            return record

    def _parse_llm_response(
        self, result: TaskResult, response: str
    ) -> ReflectionRecord:
        def extract(key: str, default: str = "") -> str:
            pattern = rf"{key}:\s*(.+?)(?:\n[A-Z_]+:|$)"
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
            return default

        confidence_str = extract("CONFIDENCE", "0.5")
        try:
            confidence = float(re.search(r"[\d.]+", confidence_str).group())
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = 0.5

        root_str = extract("ROOT_CAUSE", "unknown").lower().replace(" ", "_")
        try:
            root_cause = FailureCategory(root_str)
        except ValueError:
            root_cause = FailureCategory.UNKNOWN

        return ReflectionRecord(
            task=result.task,
            success=result.success,
            confidence=confidence,
            what_worked=extract("WHAT_WORKED", "Not specified"),
            what_failed=extract("WHAT_FAILED", "none" if result.success else "Unknown failure"),
            root_cause=root_cause,
            lessons=extract("LESSONS", ""),
            suggested_improvement=extract("SUGGESTED_IMPROVEMENT", ""),
            tools_used=result.tools_used,
            steps=result.steps,
            duration_seconds=result.duration_seconds,
            raw_output=result.output[:2000] if result.output else "",
            metadata={"llm_raw": response[:1000]},
        )


class RobustAgent:
    """
    Main entry point for adding robustness to any agent.

    Usage:
        robust = RobustAgent(db_path="my_agent.db")
        # after a task finishes:
        record = robust.reflect(task_result)
        # before a new similar task:
        past = robust.retrieve_similar("new task description")
    """

    def __init__(
        self,
        db_path: str = "angvey_robust.db",
        llm_callable: Optional[Callable[[str], str]] = None,
    ):
        self.store = ReflectionStore(db_path)
        self.reflection = Reflection(llm_callable=llm_callable)

    def reflect(
        self,
        result: TaskResult,
        extra_context: str = "",
        tags: Optional[List[str]] = None,
    ) -> ReflectionRecord:
        """
        Create and persist a reflection for a completed task.
        Returns the stored ReflectionRecord.
        """
        record = self.reflection.create(result, extra_context=extra_context)
        if tags:
            record.tags = tags
        self.store.add_reflection(record)
        return record

    def reflect_from_values(
        self,
        task: str,
        success: bool,
        output: str = "",
        error: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        steps: int = 0,
        **kwargs,
    ) -> ReflectionRecord:
        """Convenience wrapper when you don't want to build a TaskResult yourself."""
        result = TaskResult(
            task=task,
            success=success,
            output=output,
            error=error,
            tools_used=tools_used or [],
            steps=steps,
            metadata=kwargs.get("metadata", {}),
        )
        return self.reflect(result, extra_context=kwargs.get("extra_context", ""))

    def retrieve_similar(self, task: str, limit: int = 5) -> List[ReflectionRecord]:
        """Retrieve past reflections most similar to the given task."""
        return self.store.search_similar(task, limit=limit)

    def get_stats(self) -> Dict[str, Any]:
        """Return high-level performance statistics."""
        return self.store.stats()

    def suggest_improvements(self, min_failures: int = 3) -> List[ImprovementSuggestion]:
        """
        Generate simple improvement suggestions based on recent failure patterns.
        Heuristic only (no LLM required).
        """
        failures = self.store.list_reflections(limit=100, success=False)
        if len(failures) < min_failures:
            return []

        # Group by root_cause
        from collections import Counter
        cause_counts = Counter(r.root_cause for r in failures)
        suggestions = []

        for cause, count in cause_counts.most_common(5):
            if count < 2:
                continue
            title = f"Reduce {cause.value} failures"
            description = (
                f"Observed {count} failures categorized as '{cause.value}'. "
                f"Review recent reflections and consider adding explicit checks, "
                f"better tool schemas, or stronger planning instructions."
            )
            related_ids = [r.id for r in failures if r.root_cause == cause][:5]
            sug = ImprovementSuggestion(
                title=title,
                description=description,
                category="process",
                priority="high" if count >= 5 else "medium",
                based_on_reflections=related_ids,
            )
            self.store.add_suggestion(sug)
            suggestions.append(sug)

        return suggestions

    def format_context_for_agent(
        self, task: str, limit: int = 3
    ) -> str:
        """
        Produce a short text block that can be injected into an agent prompt
        so it can learn from past similar experiences.
        """
        similar = self.retrieve_similar(task, limit=limit)
        if not similar:
            return "No relevant past experiences found."

        lines = ["### Relevant past experiences:"]
        for i, ref in enumerate(similar, 1):
            status = "SUCCESS" if ref.success else "FAILURE"
            lines.append(
                f"{i}. [{status}] {ref.task[:120]}\n"
                f"   Lesson: {ref.lessons or ref.what_failed or 'n/a'}"
            )
        return "\n".join(lines)
