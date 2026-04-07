"""
Typed models for SQL Debug Environment.
Action, Observation, and State follow the OpenEnv spec.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Try to inherit from openenv-core base classes; fall back to BaseModel.
# ---------------------------------------------------------------------------
try:
    from openenv.core.env_server.types import Action, Observation, State as _BaseState  # type: ignore
    _ActionBase = Action
    _ObsBase    = Observation
    _StateBase  = _BaseState
except Exception:
    _ActionBase = BaseModel  # type: ignore
    _ObsBase    = BaseModel  # type: ignore
    _StateBase  = BaseModel  # type: ignore


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
class SQLAction(_ActionBase):
    """Agent action in the SQL Debug environment."""

    # FIX: extra='ignore' so any unknown fields from openenv-core never crash
    model_config = ConfigDict(extra="ignore")

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


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class SQLObservation(_ObsBase):
    """Full observation returned after every step and reset."""

    # FIX: extra='ignore' prevents crash when openenv-core base class adds
    # extra fields like 'done', 'reward', 'metadata' to the serialized response
    model_config = ConfigDict(extra="ignore")

    task_id: str              = Field(...,  description="Identifier of the current task.")
    difficulty: str           = Field(...,  description="Difficulty tier: easy | medium | hard.")
    task_description: str     = Field(...,  description="Natural-language description of what the query must produce.")
    broken_query: str         = Field(...,  description="The original broken SQL query that must be fixed.")
    db_schema: str            = Field(...,  description="CREATE TABLE statements for every table in the database.")
    error_message: str        = Field("",  description="SQL execution error from the most recent action, or empty string.")
    feedback: str             = Field("",  description="Detailed grader feedback after a submit_fix action.")
    last_test_output: str     = Field("",  description="Formatted table output from the most recent test_query action.")
    attempt_number: int       = Field(0,   description="Number of submit_fix attempts made so far.")
    max_attempts: int         = Field(5,   description="Maximum submit_fix attempts allowed.")
    steps_taken: int          = Field(0,   description="Total steps (test + submit) taken.")
    hint: str                 = Field("",  description="A hint shown automatically after 2 failed submit attempts.")
    score_breakdown: Dict[str, float] = Field(default_factory=dict, description="Partial score components.")
    current_best_score: float = Field(0.0, description="Highest grader score achieved in this episode so far.")
    is_solved: bool           = Field(False, description="True once a submit_fix achieves score >= 0.9.")


# ---------------------------------------------------------------------------
# State  (returned by GET /state)
# ---------------------------------------------------------------------------
class SQLState(_StateBase):
    """Episode-level metadata."""

    # FIX: extra='ignore' for same reason as SQLObservation
    model_config = ConfigDict(extra="ignore")

    episode_id: str           = Field(...,  description="Unique ID for the current episode.")
    task_id: str              = Field(...,  description="Active task identifier.")
    difficulty: str           = Field(...,  description="Task difficulty.")
    attempt_number: int       = Field(0,   description="Submit attempts so far.")
    steps_taken: int          = Field(0,   description="Total steps so far.")
    is_solved: bool           = Field(False, description="Whether the episode is solved.")
    current_best_score: float = Field(0.0, description="Best grader score so far.")
    max_attempts: int         = Field(5,   description="Attempt cap.")


# ---------------------------------------------------------------------------
# StepResult (used internally by the client)
# ---------------------------------------------------------------------------
class StepResult(BaseModel):
    # FIX: extra='ignore' so StepResult never crashes on unexpected server fields
    model_config = ConfigDict(extra="ignore")

    observation: SQLObservation
    reward: float = 0.0
    done: bool    = False
    info: Dict    = Field(default_factory=dict)
