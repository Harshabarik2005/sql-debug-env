"""
FastAPI server for SQL Debug Environment.

Endpoints
---------
POST /reset      Start a new episode. Body: {"task_id": "easy"}
POST /step       Take a step.         Body: SQLAction JSON
GET  /state      Current episode state.
GET  /health     Health check.
GET  /tasks      List all available tasks.
POST /grader     Grade a query without a full episode.
GET  /baseline   Run oracle agent on all tasks; return all scores.
GET  /docs       Swagger UI.

Port: 7860  (Hugging Face Spaces default)
"""

from __future__ import annotations

import os
import sys

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

_env = SQLDebugEnvironment()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


class GraderRequest(BaseModel):
    task_id: str
    sql_query: str


@app.post("/reset", response_model=StepResult, summary="Reset the environment to a new episode")
def reset(body: ResetRequest = ResetRequest()) -> StepResult:
    task_id = body.task_id or "easy"
    if task_id not in TASK_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{task_id}'. Valid: {list(TASK_BY_ID)}")
    obs = _env.reset(task_id=task_id)
    return StepResult(observation=obs, reward=0.0, done=False)


@app.post("/step", response_model=StepResult, summary="Take one step in the environment")
def step(action: SQLAction) -> StepResult:
    if _env._task is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    obs, reward, done, info = _env.step(action)
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=SQLState, summary="Get current episode state")
def state() -> SQLState:
    if _env._task is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.state


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    return {"status": "healthy"}


@app.get("/tasks", summary="List all available tasks")
def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": SQLDebugEnvironment.list_tasks(),
        "action_schema": SQLAction.model_json_schema(),
        "observation_schema": SQLObservation.model_json_schema(),
    }


@app.post("/grader", summary="Grade a SQL query without a full episode")
def standalone_grader(body: GraderRequest) -> Dict[str, Any]:
    if body.task_id not in TASK_BY_ID:
        raise HTTPException(status_code=400, detail=f"Unknown task_id '{body.task_id}'")
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
    result.pop("actual_rows", None)
    return {"task_id": body.task_id, **result}


@app.get("/baseline", summary="Run oracle agent on all tasks and return scores")
def baseline_scores() -> Dict[str, Any]:
    tmp = SQLDebugEnvironment()
    results = [tmp.oracle_score(task.id) for task in ALL_TASKS]
    avg = round(sum(r["score"] for r in results) / len(results), 4)
    return {"scores": results, "average": avg}


# FIX: single clean main() — removed duplicate if __name__ == "__main__" block
def main():
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
