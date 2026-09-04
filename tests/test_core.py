"""Basic tests for angvey-robust."""

import tempfile
from pathlib import Path

import pytest

from angvey_robust import RobustAgent, TaskResult, FailureCategory
from angvey_robust.models import ReflectionRecord


@pytest.fixture
def agent(tmp_path):
    db = tmp_path / "test.db"
    return RobustAgent(db_path=str(db))


def test_successful_reflection(agent):
    result = TaskResult(
        task="Write a hello world function",
        success=True,
        output="def hello(): print('hello')",
        tools_used=["editor"],
        steps=2,
    )
    record = agent.reflect(result)
    assert record.success is True
    assert record.root_cause == FailureCategory.NONE
    assert record.id is not None


def test_failed_reflection_detects_tool_issue(agent):
    result = TaskResult(
        task="Call the weather API",
        success=False,
        error="Tool 'weather.get' failed with invalid argument",
        tools_used=["weather.get"],
    )
    record = agent.reflect(result)
    assert record.success is False
    assert record.root_cause == FailureCategory.TOOL_USE


def test_retrieve_similar(agent):
    agent.reflect(
        TaskResult(task="Book a flight to Paris", success=True, output="Booked")
    )
    agent.reflect(
        TaskResult(task="Book a hotel in Lyon", success=False, error="no availability")
    )
    agent.reflect(
        TaskResult(task="Calculate 2+2", success=True, output="4")
    )

    similar = agent.retrieve_similar("Book accommodation in France")
    assert len(similar) >= 1
    assert any("Book" in r.task for r in similar)


def test_stats(agent):
    agent.reflect(TaskResult(task="ok task", success=True, output="done"))
    agent.reflect(TaskResult(task="bad task", success=False, error="fail"))
    stats = agent.get_stats()
    assert stats["total_reflections"] == 2
    assert stats["successes"] == 1
    assert stats["failures"] == 1
    assert 0.0 <= stats["success_rate"] <= 1.0


def test_format_context(agent):
    agent.reflect(
        TaskResult(
            task="Send email to client",
            success=False,
            error="SMTP auth failed",
        )
    )
    ctx = agent.format_context_for_agent("Send email to customer")
    assert "past experiences" in ctx.lower() or "No relevant" in ctx


def test_suggest_improvements(agent):
    for i in range(4):
        agent.reflect(
            TaskResult(
                task=f"API call {i}",
                success=False,
                error="tool timeout error",
                tools_used=["http"],
            )
        )
    suggestions = agent.suggest_improvements(min_failures=2)
    assert isinstance(suggestions, list)
    # Should have generated at least one suggestion
    assert len(suggestions) >= 1


def test_persist_and_reload(tmp_path):
    db = tmp_path / "persist.db"
    agent1 = RobustAgent(db_path=str(db))
    agent1.reflect(TaskResult(task="persistent task", success=True, output="ok"))

    agent2 = RobustAgent(db_path=str(db))
    stats = agent2.get_stats()
    assert stats["total_reflections"] == 1
    refs = agent2.store.list_reflections(limit=5)
    assert refs[0].task == "persistent task"
