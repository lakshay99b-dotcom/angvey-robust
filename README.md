# Angvey Robust

**An experimental framework for improving AI-agent reliability under difficult and repeated tasks.**

> Agents that *notice* their mistakes, *remember* them, and *stop repeating them*.

Built by **angvey int**.

---

## Results at a glance

Controlled multi-seed simulation · **20 seeds × 50 episodes × 10 tasks ≈ 20,000 agent-task executions**

| Metric | Baseline | + Angvey Robust | Delta |
|--------|----------|-----------------|-------|
| **Repeated-task success** | 67.0% ± 2.8 | **80.5% ± 2.2** | **+13.4 pp** |
| **Repeated same error** | 31.6% ± 2.3 | **14.9% ± 2.7** | **−16.8 pp** |
| **Recovery after tool failure** | 48.2% ± 3.6 | **60.9% ± 2.7** | **+12.7 pp** |
| First-attempt success | 69.0% ± 2.8 | 72.1% ± 2.0 | +3.0 pp |

**Takeaway in one line:** first tries stay similar; once the agent has failed, the next time on a similar task it succeeds more often and repeats the *same* error far less.

Reproduce:

```bash
git clone https://github.com/lakshay99b-dotcom/angvey-robust.git
cd angvey-robust
pip install -e .
PYTHONPATH=src python experiments/multi_seed_benchmark.py
```

These are simulation results under a defined failure model (not a production A/B test). They show direction, size, and stability of the effect across 20 seeds.

---

## What is Angvey Robust?

A **lightweight robustness layer** you drop into any agent loop:

```
Task → Execute → Reflect → Store learning → Improve next time
```

It does four things:

1. **Reflect** after every task (success or failure)
2. **Store** structured lessons in a local SQLite DB
3. **Retrieve** similar past experiences before the next task
4. **Surface** failure patterns and simple improvement suggestions

Stdlib only. No required cloud APIs. Works with any agent framework — or none.

---

## What problem does it solve?

Most agents:

- execute once and forget
- repeat the same tool / auth / format mistakes
- give no structured signal when they are drifting

That shows up as:

- sticky failures on repeated work
- weak recovery after tool errors
- manual prompt hacking instead of measured learning

Angvey Robust adds a thin metacognitive loop so the agent can use its *own* history.

---

## What we tested

**Environment:** simulated agent workloads (calendar, email, API, summary, search) with sticky failure modes that naive agents tend to repeat.

**Protocol:**

- 20 independent random seeds  
- 50 episodes per seed  
- 10 task types  
- ≈ **20,000** agent-task executions total  
- Metrics: first-attempt success, repeated-task success, rate of repeating the *same* error, recovery after tool failure  

**Finding:** the largest gains appear on *repeated* work and recovery — exactly where memory of past failures should help.

---

## Quick start

```bash
git clone https://github.com/lakshay99b-dotcom/angvey-robust.git
cd angvey-robust
pip install -e .
```

Minimal usage:

```python
from angvey_robust import RobustAgent, TaskResult

robust = RobustAgent(db_path="agent_memory.db")

# After a task finishes:
result = TaskResult(
    task="Summarize Q3 sales report",
    success=True,
    output="Top products: A, B, C. Growth 12%.",
    tools_used=["read_file", "calculate"],
    steps=4,
)
reflection = robust.reflect(result)
print(reflection.lessons)

# Before a similar new task — inject past lessons:
context = robust.format_context_for_agent(
    "Create a short summary of the latest sales numbers"
)
# → put `context` into your agent prompt
```

Full loop pattern:

```python
robust = RobustAgent(db_path="production_agent.db")

def run_task(task: str):
    past = robust.format_context_for_agent(task)
    prompt = f"{past}\n\nCurrent task: {task}\n..."
    output, success, tools, steps = your_agent.run(prompt)
    robust.reflect(TaskResult(
        task=task,
        success=success,
        output=output,
        tools_used=tools,
        steps=steps,
    ))
```

See `examples/basic_usage.py` and `examples/agent_loop_integration.py`.

---

## Features (v0.1)

- Structured self-reflection after every task  
- Local persistent store (SQLite)  
- Similarity retrieval of past experiences  
- Heuristic root-cause tags (tool errors, hallucination, context, …)  
- Optional LLM-powered richer reflections (you pass a callable)  
- Drift / failure-pattern summaries  
- CLI for inspect / record / suggest  
- **Zero required dependencies** (pure Python + stdlib)

---

## CLI

```bash
angvey-robust reflect --task "Book meeting with client" --failure --output "Auth error"
angvey-robust stats
angvey-robust similar --task "Schedule a call next week"
angvey-robust list --failures-only
angvey-robust suggest
```

---

## Optional LLM reflections

```python
def my_llm(prompt: str) -> str:
    return client.chat.completions.create(...).choices[0].message.content

robust = RobustAgent(db_path="agent.db", llm_callable=my_llm)
```

Falls back to heuristics if the LLM call fails.

---

## Project status

- **v0.1.0** — core reflection, store, retrieval, stats, CLI, examples, tests, multi-seed benchmark  
- Next: stronger similarity, framework adapters, real-world evals  

---

## License

MIT © 2026 angvey int

---

**Angvey Robust** — measured reliability gains for agents that learn from their own mistakes.
