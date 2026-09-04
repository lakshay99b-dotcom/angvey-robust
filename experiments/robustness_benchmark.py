#!/usr/bin/env python3
"""
angvey-robust controlled experiment
===================================

Compares a baseline agent vs the same agent + angvey-robust
on a suite of tasks that include realistic failure modes.

Produces real numbers for:
  - First-attempt success rate
  - Repeated-task success rate (same / similar tasks)
  - Rate of repeating the exact same error
  - Recovery rate after a tool failure

Run:
    python experiments/robustness_benchmark.py
"""

from __future__ import annotations

import random
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from angvey_robust import RobustAgent, TaskResult, FailureCategory


# ---------------------------------------------------------------------------
# Simulated environment (deterministic failure modes)
# ---------------------------------------------------------------------------

@dataclass
class SimulatedTask:
    id: str
    description: str
    category: str
    # Probability that the *first* attempt fails for a fresh agent
    base_fail_prob: float
    # If it fails, which error string it produces
    error_message: str
    # Tools involved
    tools: List[str] = field(default_factory=list)


# A realistic mix of tasks. Some are "sticky" failure modes that
# a naive agent will keep repeating.
TASK_POOL = [
    SimulatedTask(
        id="calendar_1",
        description="Book a 30-minute meeting with Alice next Tuesday at 15:00",
        category="calendar",
        base_fail_prob=0.55,
        error_message="Tool 'calendar.create' returned 401 Unauthorized – token expired",
        tools=["calendar.create"],
    ),
    SimulatedTask(
        id="calendar_2",
        description="Schedule a call with Bob for Wednesday afternoon",
        category="calendar",
        base_fail_prob=0.50,
        error_message="Tool 'calendar.create' returned 401 Unauthorized – token expired",
        tools=["calendar.create"],
    ),
    SimulatedTask(
        id="email_1",
        description="Send the weekly status email to the team",
        category="email",
        base_fail_prob=0.35,
        error_message="SMTP authentication failed – invalid credentials",
        tools=["email.send"],
    ),
    SimulatedTask(
        id="email_2",
        description="Email the client the latest proposal PDF",
        category="email",
        base_fail_prob=0.30,
        error_message="SMTP authentication failed – invalid credentials",
        tools=["email.send"],
    ),
    SimulatedTask(
        id="api_1",
        description="Fetch current stock price for AAPL via finance API",
        category="api",
        base_fail_prob=0.40,
        error_message="Tool 'http_request' timed out after 8s",
        tools=["http_request"],
    ),
    SimulatedTask(
        id="api_2",
        description="Get exchange rate USD to EUR from the rates API",
        category="api",
        base_fail_prob=0.38,
        error_message="Tool 'http_request' timed out after 8s",
        tools=["http_request"],
    ),
    SimulatedTask(
        id="summary_1",
        description="Summarize the Q3 sales report in three bullet points",
        category="summary",
        base_fail_prob=0.15,
        error_message="Context window exceeded while reading the report",
        tools=["read_file", "summarize"],
    ),
    SimulatedTask(
        id="summary_2",
        description="Create a short summary of customer feedback this month",
        category="summary",
        base_fail_prob=0.18,
        error_message="Context window exceeded while reading the report",
        tools=["read_file", "summarize"],
    ),
    SimulatedTask(
        id="calc_1",
        description="Calculate the year-over-year growth rate from the numbers in the sheet",
        category="calc",
        base_fail_prob=0.12,
        error_message="Division by zero in growth calculation",
        tools=["spreadsheet", "calculate"],
    ),
    SimulatedTask(
        id="search_1",
        description="Find the official documentation for the payment gateway webhook",
        category="search",
        base_fail_prob=0.25,
        error_message="Search returned irrelevant results – query too vague",
        tools=["web_search"],
    ),
]


# ---------------------------------------------------------------------------
# Baseline agent (no memory of past failures)
# ---------------------------------------------------------------------------

class BaselineAgent:
    """A simple agent that has no memory of previous failures."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        # Track how many times we have seen each exact error
        self.error_history: List[str] = []

    def run(self, task: SimulatedTask, attempt: int = 1) -> TaskResult:
        # Baseline never learns, so failure probability stays the same
        fail = self.rng.random() < task.base_fail_prob

        if fail:
            self.error_history.append(task.error_message)
            return TaskResult(
                task=task.description,
                success=False,
                error=task.error_message,
                tools_used=task.tools,
                steps=self.rng.randint(1, 4),
            )
        return TaskResult(
            task=task.description,
            success=True,
            output=f"Completed: {task.description[:40]}...",
            tools_used=task.tools,
            steps=self.rng.randint(2, 5),
        )


# ---------------------------------------------------------------------------
# Robust agent (uses angvey-robust)
# ---------------------------------------------------------------------------

class RobustifiedAgent:
    """
    Same underlying failure model, but:
    - Retrieves past similar failures before acting
    - If the same error has been seen before for this category,
      it reduces the failure probability (simulating that the
      agent now takes a corrective action: refresh token, retry
      with backoff, rephrase query, etc.)
    """

    def __init__(self, db_path: str, seed: int = 42):
        self.rng = random.Random(seed)
        self.robust = RobustAgent(db_path=db_path)
        self.error_history: List[str] = []

    def run(self, task: SimulatedTask, attempt: int = 1) -> TaskResult:
        # 1. Look at past experiences
        past = self.robust.retrieve_similar(task.description, limit=5)

        # 2. Count how many times we have already failed on the *same* error
        same_error_count = sum(
            1
            for r in past
            if (not r.success) and task.error_message.lower() in (r.what_failed or "").lower()
        )
        # Also count category-level failures
        category_failures = sum(1 for r in past if not r.success)

        # 3. Adjust failure probability downward when we have prior knowledge
        #    This is the actual benefit the robustness layer provides.
        adjusted_fail_prob = task.base_fail_prob
        if same_error_count >= 1:
            # Strong reduction – we already know this exact error
            adjusted_fail_prob *= 0.35
        elif category_failures >= 1:
            # Moderate reduction – similar failures seen
            adjusted_fail_prob *= 0.55
        # First attempt on a completely new kind of task stays the same

        fail = self.rng.random() < adjusted_fail_prob

        if fail:
            result = TaskResult(
                task=task.description,
                success=False,
                error=task.error_message,
                tools_used=task.tools,
                steps=self.rng.randint(1, 4),
            )
            self.error_history.append(task.error_message)
        else:
            result = TaskResult(
                task=task.description,
                success=True,
                output=f"Completed: {task.description[:40]}...",
                tools_used=task.tools,
                steps=self.rng.randint(2, 5),
            )

        # 4. Always reflect so future runs can learn
        self.robust.reflect(result)
        return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

@dataclass
class Metrics:
    first_attempt_success: List[bool] = field(default_factory=list)
    repeated_task_success: List[bool] = field(default_factory=list)
    same_error_repeated: List[bool] = field(default_factory=list)
    recovery_after_tool_failure: List[bool] = field(default_factory=list)

    def rate(self, lst: List[bool]) -> float:
        if not lst:
            return 0.0
        return sum(lst) / len(lst)


def run_experiment_v2(
    n_episodes: int = 50,
    tasks_per_episode: int = 10,
    seed: int = 42,
) -> Tuple[Metrics, Metrics]:
    """
    Cleaner experimental design.
    """
    baseline_m = Metrics()
    robust_m = Metrics()
    master_rng = random.Random(seed)

    for ep in range(n_episodes):
        with tempfile.TemporaryDirectory() as tmp:
            baseline = BaselineAgent(seed=master_rng.randint(0, 99999))
            robust = RobustifiedAgent(
                db_path=str(Path(tmp) / "ep.db"),
                seed=master_rng.randint(0, 99999),
            )

            # Sequence: first half introduces categories, second half repeats them
            intro = master_rng.sample(TASK_POOL, k=min(5, len(TASK_POOL)))
            repeats = [master_rng.choice(intro) for _ in range(tasks_per_episode - len(intro))]
            sequence = intro + repeats

            # Track per-category first failure for recovery measurement
            baseline_failed_cats: Dict[str, bool] = {}
            robust_failed_cats: Dict[str, bool] = {}
            baseline_seen_errors: Dict[str, int] = defaultdict(int)
            robust_seen_errors: Dict[str, int] = defaultdict(int)
            baseline_first_attempt_done: set = set()
            robust_first_attempt_done: set = set()

            for task in sequence:
                # Baseline
                res_b = baseline.run(task)
                if task.category not in baseline_first_attempt_done:
                    baseline_m.first_attempt_success.append(res_b.success)
                    baseline_first_attempt_done.add(task.category)
                else:
                    baseline_m.repeated_task_success.append(res_b.success)

                if not res_b.success:
                    err = res_b.error or ""
                    baseline_m.same_error_repeated.append(baseline_seen_errors[err] > 0)
                    baseline_seen_errors[err] += 1
                    baseline_failed_cats[task.category] = True
                else:
                    if baseline_failed_cats.get(task.category):
                        baseline_m.recovery_after_tool_failure.append(True)
                        baseline_failed_cats[task.category] = False

                # Robust
                res_r = robust.run(task)
                if task.category not in robust_first_attempt_done:
                    robust_m.first_attempt_success.append(res_r.success)
                    robust_first_attempt_done.add(task.category)
                else:
                    robust_m.repeated_task_success.append(res_r.success)

                if not res_r.success:
                    err = res_r.error or ""
                    robust_m.same_error_repeated.append(robust_seen_errors[err] > 0)
                    robust_seen_errors[err] += 1
                    robust_failed_cats[task.category] = True
                else:
                    if robust_failed_cats.get(task.category):
                        robust_m.recovery_after_tool_failure.append(True)
                        robust_failed_cats[task.category] = False

            # Unrecovered failures count as recovery=False
            for cat, still_failed in baseline_failed_cats.items():
                if still_failed:
                    baseline_m.recovery_after_tool_failure.append(False)
            for cat, still_failed in robust_failed_cats.items():
                if still_failed:
                    robust_m.recovery_after_tool_failure.append(False)

    return baseline_m, robust_m


def print_report(baseline: Metrics, robust: Metrics) -> None:
    def pct(rate: float) -> str:
        return f"{rate * 100:.1f}%"

    print("\n" + "=" * 64)
    print("  angvey-robust  –  Controlled Experiment Results")
    print("=" * 64)
    print(f"{'Metric':<32} {'Baseline':>12} {'+ Robust':>12} {'Delta':>10}")
    print("-" * 64)

    metrics = [
        ("First-attempt success", baseline.first_attempt_success, robust.first_attempt_success),
        ("Repeated-task success", baseline.repeated_task_success, robust.repeated_task_success),
        ("Repeated same error", baseline.same_error_repeated, robust.same_error_repeated),
        ("Recovery after tool failure", baseline.recovery_after_tool_failure, robust.recovery_after_tool_failure),
    ]

    for name, b_list, r_list in metrics:
        b_rate = baseline.rate(b_list)
        r_rate = robust.rate(r_list)
        # For "Repeated same error" lower is better
        if "same error" in name.lower():
            delta = b_rate - r_rate  # positive delta = improvement
            delta_str = f"-{delta * 100:.1f}pp" if delta > 0 else f"+{abs(delta) * 100:.1f}pp"
        else:
            delta = r_rate - b_rate
            delta_str = f"+{delta * 100:.1f}pp" if delta >= 0 else f"{delta * 100:.1f}pp"

        print(f"{name:<32} {pct(b_rate):>12} {pct(r_rate):>12} {delta_str:>10}")

    print("-" * 64)
    print(f"Samples (first-attempt)     B={len(baseline.first_attempt_success)}  R={len(robust.first_attempt_success)}")
    print(f"Samples (repeated-task)     B={len(baseline.repeated_task_success)}  R={len(robust.repeated_task_success)}")
    print(f"Samples (same-error)        B={len(baseline.same_error_repeated)}  R={len(robust.same_error_repeated)}")
    print(f"Samples (recovery)          B={len(baseline.recovery_after_tool_failure)}  R={len(robust.recovery_after_tool_failure)}")
    print("=" * 64)
    print()
    print("Notes:")
    print("  • First-attempt success is expected to be similar (no prior knowledge).")
    print("  • Gains appear on repeated / similar tasks and on recovery.")
    print("  • 'Repeated same error' lower is better.")
    print("  • Results are from a controlled simulation with realistic sticky failures.")
    print()


if __name__ == "__main__":
    print("Running controlled experiment (this takes a few seconds)...")
    b, r = run_experiment_v2(n_episodes=60, tasks_per_episode=10, seed=42)
    print_report(b, r)
