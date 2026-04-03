"""
Typed models for SQL Debug Environment.
Action, Observation, and State follow the OpenEnv spec.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Try to inherit from openenv-core base classes; fall back to BaseModel.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State as _BaseState  # type: ignore

    _ActionBase = Action
    _ObsBase = Observation
    _StateBase = _BaseState
except Exception:
    _ActionBase = BaseModel  # type: ignore
    _ObsBase = BaseModel  # type: ignore
    _StateBase = BaseModel  # type: ignore


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class SQLAction(_ActionBase):
    """
    Agent action in the SQL Debug environment.

    Three action types are available:

    * ``test_query``   – Run a SQL query against the live DB and observe results.
                        Does NOT count as a graded attempt.
    * ``submit_fix``   – Submit the corrected query for grading. Counted as an
                        attempt. Episode ends when score ≥ 0.9 or attempts
                        exhausted.
    """

    action_type: Literal["test_query", "submit_fix"] = Field(
        ...,
        description=(
            "test_query: run SQL to inspect results without using an attempt; "
            "submit_fix: grade the corrected query (uses one attempt)"
        ),
    )
    sql_query: str = Field(
        ...,
        description="The SQL query to run or submit. Must be valid SQLite syntax.",
    )
    explanation: Optional[str] = Field(
        None,
        description="Optional free-text explanation of the fix approach.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "action_type": "submit_fix",
                "sql_query": "SELECT name, salary FROM employees WHERE dept_id = 1",
                "explanation": "Fixed misspelled keywords: SELCT→SELECT, FORM→FROM, WERE→WHERE",
            }
        }


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class SQLObservation(_ObsBase):
    """Full observation returned after every step and reset."""

    # Task info
    task_id: str = Field(..., description="Identifier of the current task.")
    difficulty: str = Field(..., description="Difficulty tier: easy | medium | hard.")
    task_description: str = Field(
        ..., description="Natural-language description of what the query must produce."
    )

    # Query under repair
    broken_query: str = Field(
        ..., description="The original broken SQL query that must be fixed."
    )
    db_schema: str = Field(
        ..., description="CREATE TABLE statements for every table in the database."
    )

    # Feedback from last action
    error_message: str = Field(
        "", description="SQL execution error from the most recent action, or empty string."
    )
    feedback: str = Field(
        "", description="Detailed grader feedback after a submit_fix action."
    )
    last_test_output: str = Field(
        "",
        description=(
            "Formatted table output from the most recent test_query action. "
            "Empty before any test_query is executed."
        ),
    )

    # Episode progress
    attempt_number: int = Field(
        0, description="Number of submit_fix attempts made so far."
    )
    max_attempts: int = Field(5, description="Maximum submit_fix attempts allowed.")
    steps_taken: int = Field(0, description="Total steps (test + submit) taken.")
    hint: str = Field(
        "", description="A hint shown automatically after 2 failed submit attempts."
    )

    # Score breakdown (populated after each submit_fix)
    score_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Partial score components after the most recent submit_fix: "
            "{syntax_ok, columns_ok, rows_ok, values_ok}. Empty before first submission."
        ),
    )
    current_best_score: float = Field(
        0.0, description="Highest grader score achieved in this episode so far."
    )
    is_solved: bool = Field(
        False,
        description="True once a submit_fix achieves score ≥ 0.9.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "task_id": "easy",
                "difficulty": "easy",
                "task_description": "Get names and salaries of all employees in department 1",
                "broken_query": "SELCT name, salary FORM employees WERE dept_id = 1",
                "db_schema": "CREATE TABLE employees (id INTEGER, name TEXT, dept_id INTEGER, salary REAL, hire_year INTEGER);",
                "error_message": "",
                "feedback": "",
                "last_test_output": "",
                "attempt_number": 0,
                "max_attempts": 5,
                "steps_taken": 0,
                "hint": "",
                "score_breakdown": {},
                "current_best_score": 0.0,
                "is_solved": False,
            }
        }


# ---------------------------------------------------------------------------
# State  (returned by GET /state)
# ---------------------------------------------------------------------------
class SQLState(_StateBase):
    """Episode-level metadata."""

    episode_id: str = Field(..., description="Unique ID for the current episode.")
    task_id: str = Field(..., description="Active task identifier.")
    difficulty: str = Field(..., description="Task difficulty.")
    attempt_number: int = Field(0, description="Submit attempts so far.")
    steps_taken: int = Field(0, description="Total steps so far.")
    is_solved: bool = Field(False, description="Whether the episode is solved.")
    current_best_score: float = Field(0.0, description="Best grader score so far.")
    max_attempts: int = Field(5, description="Attempt cap.")


# ---------------------------------------------------------------------------
# StepResult (used internally by the client)
# ---------------------------------------------------------------------------
class StepResult(BaseModel):
    observation: SQLObservation
    reward: float = 0.0
    done: bool = False
    info: Dict = Field(default_factory=dict)
