"""
FastAPI server for SQL Debug Environment.

Endpoints
---------
POST /reset          Start a new episode. Body: {"task_id": "easy"}
POST /step           Take a step.         Body: SQLAction JSON
GET  /state          Current episode state.
GET  /health         Health check.
GET  /tasks          List all available tasks.
POST /grader         Grade a query without a full episode.
GET  /baseline       Run the oracle agent on all tasks; return all scores.
GET  /docs           Auto-generated Swagger UI.

Port: 7860  (Hugging Face Spaces default)
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the path when uvicorn is called from anywhere
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import SQLAction, SQLObservation, SQLState, StepResult
from server.environment import SQLDebugEnvironment
from server.tasks import ALL_TASKS, TASK_BY_ID

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SQL Debug Environment",
    description=(
        "An OpenEnv-compliant RL environment where an agent learns to identify "
        "and fix broken SQL queries. Five tasks spanning easy → hard cover the "
        "most common real-world SQL bug categories."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global single-session environment (one agent at a time)
_env = SQLDebugEnvironment()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


class GraderRequest(BaseModel):
    task_id: str
    sql_query: str


# ---------------------------------------------------------------------------
# Core OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=StepResult, summary="Reset the environment to a new episode")
def reset(body: ResetRequest = ResetRequest()) -> StepResult:
    """
    Start a new episode.

    - **task_id**: one of ``easy``, ``medium``, ``medium2``, ``hard``, ``hard2``
    """
    task_id = body.task_id or "easy"
    if task_id not in TASK_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_BY_ID)}")
    obs = _env.reset(task_id=task_id)
    return StepResult(observation=obs, reward=0.0, done=False)


@app.post("/step", response_model=StepResult, summary="Take one step in the environment")
def step(action: SQLAction) -> StepResult:
    """
    Execute an action.

    - **action_type** ``test_query``: run SQL to inspect output (no attempt consumed)
    - **action_type** ``submit_fix``: grade the query (consumes one attempt)
    """
    if _env._task is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    obs, reward, done, info = _env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=SQLState, summary="Get current episode state")
def state() -> SQLState:
    """Return episode-level metadata without advancing the environment."""
    if _env._task is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.state


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    """Returns ``{"status": "healthy"}`` when the server is up."""
    return {"status": "healthy"}


# ---------------------------------------------------------------------------
# Bonus endpoints
# ---------------------------------------------------------------------------

@app.get("/tasks", summary="List all available tasks")
def list_tasks() -> Dict[str, Any]:
    """Enumerate all 5 tasks with descriptions, difficulty, and schema."""
    return {
        "tasks": SQLDebugEnvironment.list_tasks(),
        "action_schema": SQLAction.model_json_schema(),
        "observation_schema": SQLObservation.model_json_schema(),
    }


@app.post("/grader", summary="Grade a SQL query without a full episode")
def standalone_grader(body: GraderRequest) -> Dict[str, Any]:
    """
    Run the grader on a submitted query for any task without consuming
    episode attempts.  Useful for debugging and tooling integration.
    """
    if body.task_id not in TASK_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{body.task_id}'")

    # Spin up a temporary environment for this task
    tmp = SQLDebugEnvironment()
    tmp.reset(task_id=body.task_id)

    from server.graders import grade
    result = grade(
        conn=tmp._conn,
        submitted_query=body.sql_query,
        expected_cols=tmp._expected_cols,
        expected_rows=tmp._expected_rows,
        attempt_number=1,
    )
    # Don't expose internal row data
    result.pop("actual_rows", None)
    return {"task_id": body.task_id, **result}


@app.get("/baseline", summary="Run oracle agent on all tasks and return scores")
def baseline_scores() -> Dict[str, Any]:
    """
    Runs the correct (oracle) query on every task and reports scores.
    Useful for sanity-checking that the grader works correctly.
    Expected: all scores ≈ 1.0.
    """
    tmp = SQLDebugEnvironment()
    results = []
    for task in ALL_TASKS:
        r = tmp.oracle_score(task.id)
        results.append(r)
    avg = round(sum(r["score"] for r in results) / len(results), 4)
    return {"scores": results, "average": avg}


# ---------------------------------------------------------------------------
# Entry point (for local dev without uvicorn CLI)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
