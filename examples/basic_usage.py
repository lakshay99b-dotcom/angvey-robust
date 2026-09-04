#!/usr/bin/env python3
"""
Basic usage example for angvey-robust.

Run from the project root after installing:
    pip install -e .
    python examples/basic_usage.py
"""

from angvey_robust import RobustAgent, TaskResult


def main():
    # Create a robustness layer (stores data in a local SQLite file)
    robust = RobustAgent(db_path="example_agent.db")

    print("=== Example 1: Successful task ===")
    result_ok = TaskResult(
        task="Summarize the quarterly sales report and extract top 3 products",
        success=True,
        output="Top products: A, B, C. Revenue grew 12%.",
        tools_used=["read_file", "calculate"],
        steps=4,
    )
    reflection = robust.reflect(result_ok)
    print(f"Reflection ID: {reflection.id}")
    print(f"Confidence: {reflection.confidence}")
    print(f"Lessons: {reflection.lessons}")
    print()

    print("=== Example 2: Failed task ===")
    result_fail = TaskResult(
        task="Book a meeting with the client for next Tuesday at 3pm",
        success=False,
        error="Tool 'calendar.create' returned 401 Unauthorized – invalid token",
        tools_used=["calendar.create", "email.send"],
        steps=2,
    )
    reflection2 = robust.reflect(result_fail)
    print(f"Root cause detected: {reflection2.root_cause.value}")
    print(f"Suggested improvement: {reflection2.suggested_improvement}")
    print()

    print("=== Example 3: Retrieve similar past experiences ===")
    similar = robust.retrieve_similar(
        "Schedule a call with the customer next week"
    )
    for r in similar:
        status = "SUCCESS" if r.success else "FAILURE"
        print(f"  [{status}] {r.task}")
        print(f"      → {r.lessons or r.what_failed}")
    print()

    print("=== Example 4: Inject past lessons into a new prompt ===")
    context = robust.format_context_for_agent(
        "Schedule a call with the customer next week"
    )
    print(context)
    print()

    print("=== Example 5: Performance stats ===")
    stats = robust.get_stats()
    print(stats)
    print()

    print("=== Example 6: Generate improvement suggestions ===")
    # Add a couple more failures so suggestions can appear
    for i in range(3):
        robust.reflect(
            TaskResult(
                task=f"Call external API endpoint #{i}",
                success=False,
                error="Tool timeout / connection error",
                tools_used=["http_request"],
            )
        )
    suggestions = robust.suggest_improvements(min_failures=2)
    for s in suggestions:
        print(f"  [{s.priority}] {s.title}")
        print(f"      {s.description[:100]}...")


if __name__ == "__main__":
    main()
