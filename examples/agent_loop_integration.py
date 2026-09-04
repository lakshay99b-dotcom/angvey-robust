#!/usr/bin/env python3
"""
Shows how to integrate angvey-robust into a simple agent loop.

This is a minimal "toy agent" that uses the robustness layer
before and after every task.
"""

from angvey_robust import RobustAgent, TaskResult
import time


def toy_agent_execute(task: str) -> TaskResult:
    """Simulate a very simple agent. Replace with your real agent logic."""
    print(f"  → Agent working on: {task[:60]}...")
    time.sleep(0.3)

    # Simulate different outcomes
    if "calendar" in task.lower() or "meeting" in task.lower():
        return TaskResult(
            task=task,
            success=False,
            error="calendar tool authentication failed",
            tools_used=["calendar"],
            steps=2,
        )
    if "summary" in task.lower() or "report" in task.lower():
        return TaskResult(
            task=task,
            success=True,
            output="Summary generated successfully.",
            tools_used=["read", "summarize"],
            steps=3,
        )
    return TaskResult(
        task=task,
        success=True,
        output="Done.",
        tools_used=[],
        steps=1,
    )


def run_agent_with_robustness(tasks: list[str]):
    robust = RobustAgent(db_path="toy_agent.db")

    for task in tasks:
        print(f"\n=== New task: {task} ===")

        # 1. Before acting: retrieve past lessons
        past_context = robust.format_context_for_agent(task)
        print("Past experiences injected into prompt:")
        print(past_context)

        # 2. Execute the agent (your real agent goes here)
        result = toy_agent_execute(task)

        # 3. After acting: reflect and store
        reflection = robust.reflect(result)
        print(f"Reflection stored (success={reflection.success}, "
              f"cause={reflection.root_cause.value})")

    print("\n=== Final stats ===")
    print(robust.get_stats())


if __name__ == "__main__":
    sample_tasks = [
        "Summarize the Q3 sales report",
        "Book a meeting with Alice next Tuesday",
        "Generate a summary of customer feedback",
        "Schedule a meeting with the design team",
        "Create a short report on open tickets",
    ]
    run_agent_with_robustness(sample_tasks)
