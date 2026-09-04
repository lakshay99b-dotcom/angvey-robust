"""Simple command-line interface for angvey-robust."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import RobustAgent
from .models import TaskResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="angvey-robust",
        description="Lightweight robustness layer for AI agents",
    )
    parser.add_argument(
        "--db",
        default="angvey_robust.db",
        help="Path to SQLite database (default: angvey_robust.db)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # reflect
    p_reflect = sub.add_parser("reflect", help="Record a reflection for a finished task")
    p_reflect.add_argument("--task", required=True, help="Task description")
    p_reflect.add_argument("--success", action="store_true", help="Mark as successful")
    p_reflect.add_argument("--failure", action="store_true", help="Mark as failed")
    p_reflect.add_argument("--output", default="", help="Agent output or error message")
    p_reflect.add_argument("--tools", default="", help="Comma-separated list of tools used")
    p_reflect.add_argument("--steps", type=int, default=0)

    # stats
    sub.add_parser("stats", help="Show performance statistics")

    # similar
    p_sim = sub.add_parser("similar", help="Find past reflections similar to a task")
    p_sim.add_argument("--task", required=True)
    p_sim.add_argument("--limit", type=int, default=5)

    # suggest
    sub.add_parser("suggest", help="Generate improvement suggestions from failures")

    # list
    p_list = sub.add_parser("list", help="List recent reflections")
    p_list.add_argument("--limit", type=int, default=10)
    p_list.add_argument("--failures-only", action="store_true")

    args = parser.parse_args(argv)
    agent = RobustAgent(db_path=args.db)

    if args.command == "reflect":
        if args.success == args.failure:
            print("Error: specify exactly one of --success or --failure", file=sys.stderr)
            return 1
        success = args.success
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
        result = TaskResult(
            task=args.task,
            success=success,
            output=args.output,
            tools_used=tools,
            steps=args.steps,
        )
        record = agent.reflect(result)
        print(json.dumps(record.to_dict(), indent=2))
        return 0

    if args.command == "stats":
        stats = agent.get_stats()
        print(json.dumps(stats, indent=2))
        return 0

    if args.command == "similar":
        refs = agent.retrieve_similar(args.task, limit=args.limit)
        for r in refs:
            print(f"[{'OK' if r.success else 'FAIL'}] {r.task[:80]}")
            print(f"   → {r.lessons or r.what_failed}")
            print()
        return 0

    if args.command == "suggest":
        suggestions = agent.suggest_improvements()
        if not suggestions:
            print("Not enough failure data yet to generate suggestions.")
            return 0
        for s in suggestions:
            print(f"[{s.priority.upper()}] {s.title}")
            print(f"   {s.description}")
            print()
        return 0

    if args.command == "list":
        success_filter = False if args.failures_only else None
        refs = agent.store.list_reflections(limit=args.limit, success=success_filter)
        for r in refs:
            status = "OK" if r.success else "FAIL"
            print(f"[{status}] {r.timestamp[:19]} | {r.root_cause.value:12} | {r.task[:60]}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
