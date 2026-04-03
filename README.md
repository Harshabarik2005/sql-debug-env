# SQL Debug Environment

An **OpenEnv-compliant RL environment** where an AI agent learns to identify and fix broken SQL queries against a live SQLite database. Five real-world tasks span **easy → hard** — all graded deterministically with partial credit (0.0–1.0) and dense per-step reward signals.

---

## Motivation

SQL bugs are among the most common and costly defects in production codebases. An agent capable of diagnosing and fixing query errors — syntax typos, wrong JOINs, missing GROUP BY, misplaced HAVING, and broken subqueries — has immediate value for developer tooling, code review assistants, and automated database maintenance systems.

This environment provides a **controlled, reproducible** setting to train and evaluate such agents.

---

## Environment Design

```
Agent ──── test_query ──▶  SQLite DB ──▶ table output (0.05 reward if runs)
     └──── submit_fix ──▶  Grader   ──▶ score breakdown + feedback
```

**Episode flow**
1. `reset(task_id=...)` — load task, populate a fresh in-memory SQLite DB, return observation.
2. Agent calls `test_query` to probe the DB (up to 5 rewarded calls) and `submit_fix` to attempt the fix (up to 5 attempts).
3. Episode ends when the agent scores ≥ 0.9 on a `submit_fix`, or attempts are exhausted.

---

## Tasks

| Task ID  | Difficulty | Bug Type             | Description |
|----------|------------|----------------------|-------------|
| `easy`   | easy       | Syntax typos         | Fix misspelled SQL keywords (SELCT, FORM, WERE) |
| `medium` | medium     | Missing GROUP BY     | Aggregate query returns 1 row instead of per-dept totals |
| `medium2`| medium     | Wrong JOIN condition | `ON e.id = d.id` should be `ON e.dept_id = d.id` |
| `hard`   | hard       | WHERE vs HAVING      | `WHERE AVG(salary) > 70000` must become `HAVING` |
| `hard2`  | hard       | Scalar subquery      | Subquery returns multiple rows — needs `MAX()` wrapper |

---

## Action Space

```json
{
  "action_type": "test_query | submit_fix",
  "sql_query": "<SQL string>",
  "explanation": "<optional reasoning>"
}
```

| `action_type` | Effect | Counts as attempt? |
|---------------|--------|--------------------|
| `test_query`  | Execute SQL, return formatted table output | No |
| `submit_fix`  | Grade the query; ends episode if score ≥ 0.9 | Yes (max 5) |

---

## Observation Space

| Field               | Type    | Description |
|---------------------|---------|-------------|
| `task_id`           | str     | Active task identifier |
| `difficulty`        | str     | easy / medium / hard |
| `task_description`  | str     | Natural-language goal |
| `broken_query`      | str     | The broken SQL to fix |
| `db_schema`         | str     | CREATE TABLE DDL |
| `error_message`     | str     | Last SQL execution error (empty if none) |
| `feedback`          | str     | Grader feedback after submit_fix |
| `last_test_output`  | str     | Formatted table from last test_query |
| `attempt_number`    | int     | Submit attempts used |
| `max_attempts`      | int     | 5 |
| `steps_taken`       | int     | Total steps in episode |
| `hint`              | str     | Shown after 2+ failed submissions |
| `score_breakdown`   | dict    | {syntax_ok, columns_ok, rows_ok, values_ok} |
| `current_best_score`| float   | Highest score achieved this episode |
| `is_solved`         | bool    | True when score ≥ 0.9 |

---

## Reward Function

Rewards provide a **dense signal throughout the episode**, not just at termination.

### `test_query`
| Condition | Reward |
|-----------|--------|
| Query executes without error (first 5 calls) | +0.05 |
| Query produces a new (unseen) error | +0.02 |

### `submit_fix` — grader components
| Component     | Max points | Condition |
|---------------|-----------|-----------|
| `syntax_ok`   | 0.30 | Query runs without SQL error |
| `columns_ok`  | 0.20 | Returned columns match expected |
| `rows_ok`     | 0.10 | Row count matches expected |
| `values_ok`   | 0.40 | Row values match (partial credit per row) |

**Attempt multiplier** applied to the raw component sum:

| Attempt | Multiplier |
|---------|-----------|
| 1 | ×1.00 |
| 2 | ×0.95 |
| 3 | ×0.90 |
| 4 | ×0.85 |
| 5 | ×0.80 |

Maximum achievable score per episode: **1.0** (perfect fix on first attempt).

---

## Database Schema

A shared 4-table HR/projects schema is seeded fresh for each episode:

```sql
employees(id, name, dept_id, salary, hire_year)   -- 8 rows
departments(id, name, budget)                      -- 3 rows
projects(id, name, dept_id, status)                -- 4 rows
employee_projects(employee_id, project_id, hours)  -- 9 rows
```

---

## Baseline Scores

| Agent | easy | medium | medium2 | hard | hard2 | Average |
|-------|------|--------|---------|------|-------|---------|
| Oracle (correct query) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** |
| Qwen2.5-72B (baseline) | 1.00 | 0.95 | 0.90 | 0.90 | 0.85 | **0.92** |
| Random (no fix) | 0.30 | 0.05 | 0.05 | 0.00 | 0.00 | **0.08** |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start a new episode — body: `{"task_id": "easy"}` |
| `/step`  | POST | Take one step — body: SQLAction JSON |
| `/state` | GET  | Current episode metadata |
| `/health`| GET  | Health check |
| `/tasks` | GET  | List all tasks with schemas |
| `/grader`| POST | Grade a query without a full episode |
| `/baseline` | GET | Oracle scores on all tasks |
| `/docs`  | GET  | Swagger UI |

---

## Quick Start

### Python (async)

```python
import asyncio
from client import SQLDebugEnv
from models import SQLAction

async def main():
    async with SQLDebugEnv(base_url="https://YOUR-SPACE.hf.space") as env:
        # Start an easy episode
        result = await env.reset(task_id="easy")
        obs = result.observation
        print("Broken query:", obs.broken_query)

        # Test a candidate fix
        result = await env.step(SQLAction(
            action_type="test_query",
            sql_query="SELECT name, salary FROM employees WHERE dept_id = 1",
        ))
        print("Test output:", result.observation.last_test_output)

        # Submit the fix
        result = await env.step(SQLAction(
            action_type="submit_fix",
            sql_query="SELECT name, salary FROM employees WHERE dept_id = 1",
            explanation="Fixed keywords: SELCT→SELECT, FORM→FROM, WERE→WHERE",
        ))
        print(f"Score: {result.reward:.2f}")
        print(result.observation.feedback)

asyncio.run(main())
```

### Python (sync)

```python
from client import SQLDebugEnv
from models import SQLAction

with SQLDebugEnv(base_url="https://YOUR-SPACE.hf.space").sync() as env:
    result = env.reset(task_id="hard")
    print(result.observation.broken_query)
    result = env.step(SQLAction(
        action_type="submit_fix",
        sql_query="SELECT dept_id, AVG(salary) AS avg_salary FROM employees GROUP BY dept_id HAVING AVG(salary) > 70000 ORDER BY avg_salary DESC",
    ))
    print(result.reward)
```

---

## Setup and Local Development

```bash
# Clone from HF Spaces
git clone https://huggingface.co/spaces/YOUR_USERNAME/sql-debug-env
cd sql-debug-env

# Install dependencies
pip install -r requirements.txt

# Run server (from repo root)
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# Verify
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
```

### Docker

```bash
docker build -t sql-debug-env .
docker run -p 7860:7860 sql-debug-env

# Verify
curl http://localhost:7860/health
```

### Run baseline inference

```bash
export HF_TOKEN=your_hf_token
export SQL_DEBUG_ENV_URL=http://localhost:7860
python inference.py
```

---

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
PYTHONPATH=. pytest tests/ -v
```

---

## Links

- [OpenEnv GitHub](https://github.com/meta-pytorch/OpenEnv)
- [Environment Hub](https://huggingface.co/openenv)
- [Competition](https://huggingface.co/spaces/open-env-project/openenv-challenge)
