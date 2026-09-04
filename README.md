# angvey-robust

**Lightweight robustness layer for AI agents**

> Task → Execute → Reflect → Store learning → Improve next time

`angvey-robust` helps any agentic system notice its own mistakes, remember them, and get better over time. It is deliberately small, dependency-free (stdlib only), and easy to drop into existing agent loops.

Built and maintained by **angvey int**.

---

## Why this exists

Most agents today just execute and finish. They rarely examine *why* they succeeded or failed, and they almost never reuse that knowledge on the next similar task.

This leads to:

- Repeated mistakes
- Silent performance drift
- Hard-to-debug failures
- Manual, guess-based prompt engineering

`angvey-robust` adds a thin, practical metacognitive loop that works with almost any agent framework (or no framework at all).

---

## Measured impact (multi-seed controlled experiment)

We ran a controlled simulation of realistic agent workloads (calendar, email, API, summary, search tasks) with sticky failure modes that naive agents tend to repeat.

**Protocol:** 20 random seeds × 50 episodes × 10 tasks ≈ **20,000 agent-task executions**.

| Metric                        | Baseline (mean ± std) | + angvey-robust (mean ± std) | Delta (mean)     | Delta range across seeds |
|-------------------------------|-----------------------|------------------------------|------------------|--------------------------|
| First-attempt success         | 69.0% ± 2.8           | 72.1% ± 2.0                  | +3.0 pp          | −3.0 … +8.9              |
| Repeated-task success         | 67.0% ± 2.8           | **80.5% ± 2.2**              | **+13.4 pp**     | +5.4 … +18.6             |
| Repeated same error           | 31.6% ± 2.3           | **14.9% ± 2.7**              | **−16.8 pp**     | −24.3 … −11.3            |
| Recovery after tool failure   | 48.2% ± 3.6           | **60.9% ± 2.7**              | **+12.7 pp**     | +4.1 … +18.1             |

**How to read this**
- First-attempt success stays roughly the same (little or no prior knowledge).
- Once the agent has seen similar failures, repeated-task success rises consistently (~13 pp on average).
- The rate of repeating the *exact same error* drops sharply (~17 pp).
- Recovery after a tool failure improves by ~13 pp on average.
- The effect is stable across 20 different random seeds (see ranges).

Reproduce yourself:

```bash
PYTHONPATH=src python experiments/multi_seed_benchmark.py
# or the single-seed version:
PYTHONPATH=src python experiments/robustness_benchmark.py
```

These are simulation results under a defined failure model, not production A/B tests. They demonstrate the direction, magnitude, and stability of the effect the library is designed to produce.

---

## Features (v0.1)

- **Structured self-reflection** after every task (success or failure)
- **Local persistent store** (SQLite) of reflections and improvement suggestions
- **Simple similarity retrieval** so an agent can see relevant past experiences
- **Heuristic root-cause detection** (tool errors, hallucination, context issues, etc.)
- **Optional LLM-powered richer reflections** (you supply the LLM callable)
- **Drift / failure pattern summaries** and basic improvement suggestions
- **Zero required dependencies** — pure Python + stdlib
- **CLI** for quick inspection and recording

---

## Installation

```bash
pip install -e .
# or later: pip install angvey-robust
```

Optional (for LLM-powered reflections):

```bash
pip install angvey-robust[llm]
```

---

## Quick Start

```python
from angvey_robust import RobustAgent, TaskResult

robust = RobustAgent(db_path="my_agent.db")

# After your agent finishes a task:
result = TaskResult(
    task="Summarize the sales report and list top 3 products",
    success=True,
    output="Top products: A, B, C. Growth 12%.",
    tools_used=["read_file", "calculate"],
    steps=4,
)
reflection = robust.reflect(result)
print(reflection.lessons)

# Before a new similar task — give the agent memory of the past:
context = robust.format_context_for_agent(
    "Create a short summary of the latest sales numbers"
)
# → inject `context` into your agent prompt

# High-level stats
print(robust.get_stats())
```

---

## Integration into an agent loop

```python
robust = RobustAgent(db_path="production_agent.db")

def run_task(task: str):
    # 1. Retrieve past lessons
    past = robust.format_context_for_agent(task)

    # 2. Build prompt that includes past experiences
    prompt = f"{past}\n\nCurrent task: {task}\n..."

    # 3. Run your real agent
    output, success, tools, steps = your_agent.run(prompt)

    # 4. Reflect
    robust.reflect(TaskResult(
        task=task,
        success=success,
        output=output,
        tools_used=tools,
        steps=steps,
    ))
```

See `examples/agent_loop_integration.py` for a complete runnable toy agent.

---

## CLI

```bash
# Record a reflection
angvey-robust reflect --task "Book meeting with client" --failure --output "Auth error"

# Show stats
angvey-robust stats

# Find similar past experiences
angvey-robust similar --task "Schedule a call next week"

# List recent reflections
angvey-robust list --failures-only

# Generate improvement suggestions
angvey-robust suggest
```

---

## Optional LLM reflections

If you want richer, natural-language reflections instead of pure heuristics:

```python
def my_llm(prompt: str) -> str:
    # Call OpenAI, local model, Anthropic, etc.
    return client.chat.completions.create(...).choices[0].message.content

robust = RobustAgent(db_path="agent.db", llm_callable=my_llm)
```

The library falls back to heuristics automatically if the LLM call fails.

---

## Project status

- **v0.1.0** — Core reflection, store, retrieval, stats, CLI, examples, tests
- Designed to be extended (embeddings-based similarity, richer evaluators, multi-agent support, etc.)

---

## Contributing

Contributions are welcome — especially:

- Better similarity / retrieval
- Additional failure categories or detectors
- Integrations with popular agent frameworks
- Real-world evaluation of the improvement loop

Please open an issue or PR on the GitHub repository.

---

## License

MIT © 2026 angvey int

---

**angvey-robust** — making agents more reliable, one reflection at a time.
