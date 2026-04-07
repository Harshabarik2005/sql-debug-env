"""
Core environment logic for SQL Debug Environment.

Each call to reset() creates a fresh in-memory SQLite DB and initialises
the task state.  step() dispatches to either _test_query or _submit_fix.

Reward shaping
--------------
test_query (does not count as attempt):
    +0.05  if query executes without SQL error  (first 5 test calls only)
    +0.02  if query errors but is a NEW error   (encourages exploration)

submit_fix (counts as attempt):
    grader_score × attempt_multiplier           see graders.py for full breakdown

Episode terminates when:
    is_solved  (submit_fix score ≥ 0.9), OR
    attempt_number ≥ max_attempts,        OR
    steps_taken ≥ MAX_STEPS (20)
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any, Dict, List, Optional, Tuple

# Internal imports
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SQLAction, SQLObservation, SQLState
from server.tasks import ALL_TASKS, SHARED_SCHEMA_DDL, SHARED_SEED_SQL, TASK_BY_ID, Task
from server.graders import compute_expected, grade, run_test_query


MAX_STEPS = 20
MAX_ATTEMPTS = 5
SOLVE_THRESHOLD = 0.9
MAX_TEST_REWARD_CALLS = 5   # cap on rewarded test_query calls


class SQLDebugEnvironment:
    """
    Stateful SQL debugging environment.

    Each instance manages one episode at a time.  Call reset() to start a
    new episode; call step() to advance it.
    """

    # ------------------------------------------------------------------
    # Initialise
    # ------------------------------------------------------------------
    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._task: Optional[Task] = None
        self._episode_id: str = ""
        self._attempt_number: int = 0
        self._steps_taken: int = 0
        self._is_solved: bool = False
        self._current_best_score: float = 0.01
        self._expected_cols: List[str] = []
        self._expected_rows: List[Dict] = []
        self._last_error: str = ""
        self._last_feedback: str = ""
        self._last_test_output: str = ""
        self._rewarded_test_calls: int = 0
        self._seen_errors: set = set()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, task_id: str = "easy", **kwargs) -> SQLObservation:
        """Start a new episode for the given task_id."""
        task_id = str(task_id).strip()
        if task_id not in TASK_BY_ID:
            task_id = "easy"

        self._task = TASK_BY_ID[task_id]
        self._episode_id = str(uuid.uuid4())
        self._attempt_number = 0
        self._steps_taken = 0
        self._is_solved = False
        self._current_best_score = 0.01
        self._last_error = ""
        self._last_feedback = ""
        self._last_test_output = ""
        self._rewarded_test_calls = 0
        self._seen_errors = set()

        # Build a fresh in-memory SQLite DB
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SHARED_SCHEMA_DDL)
        self._conn.executescript(SHARED_SEED_SQL)
        self._conn.commit()

        # Pre-compute expected results from the reference query
        self._expected_cols, self._expected_rows = compute_expected(
            self._conn, self._task.correct_query
        )

        return self._build_observation()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: SQLAction) -> Tuple[SQLObservation, float, bool, Dict]:
        """Advance the episode by one step."""
        if self._task is None:
            raise RuntimeError("Call reset() before step()")

        self._steps_taken += 1
        done = False
        reward = 0.0
        info: Dict[str, Any] = {}

        if action.action_type == "test_query":
            reward, done = self._test_query(action.sql_query or "")
        elif action.action_type == "submit_fix":
            reward, done = self._submit_fix(action.sql_query or "")
        else:
            self._last_error = f"Unknown action_type: '{action.action_type}'"

        # Safety: end episode if max steps reached
        if self._steps_taken >= MAX_STEPS and not done:
            done = True
            info["reason"] = "max_steps_reached"

        obs = self._build_observation()
        # Clamp reward to (0.01, 0.99) — Phase 2 validator rejects exact 0.0 or 1.0
        reward = max(0.01, min(0.99, round(reward, 4)))
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------
    @property
    def state(self) -> SQLState:
        return SQLState(
            episode_id=self._episode_id,
            task_id=self._task.id if self._task else "",
            difficulty=self._task.difficulty if self._task else "",
            attempt_number=self._attempt_number,
            steps_taken=self._steps_taken,
            is_solved=self._is_solved,
            current_best_score=self._current_best_score,
            max_attempts=MAX_ATTEMPTS,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _test_query(self, sql: str) -> Tuple[float, bool]:
        """Execute sql, observe results, reward exploration."""
        success, output = run_test_query(self._conn, sql)
        self._last_test_output = output
        self._last_feedback = ""

        reward = 0.0
        if success:
            if self._rewarded_test_calls < MAX_TEST_REWARD_CALLS:
                reward = 0.05
                self._rewarded_test_calls += 1
            self._last_error = ""
        else:
            # Extract error string from output
            err = output.replace("SQL Error: ", "")
            self._last_error = err
            # Small reward for hitting a new error (encourages exploration)
            if err not in self._seen_errors:
                self._seen_errors.add(err)
                reward = 0.02

        return reward, False   # test_query never ends the episode directly

    def _submit_fix(self, sql: str) -> Tuple[float, bool]:
        """Grade the submitted query, update attempt counter."""
        self._attempt_number += 1
        self._last_test_output = ""

        result = grade(
            conn=self._conn,
            submitted_query=sql,
            expected_cols=self._expected_cols,
            expected_rows=self._expected_rows,
            attempt_number=self._attempt_number,
        )

        reward = result["score"]
        self._last_error = result["error_message"]
        self._last_feedback = result["feedback"]

        if result["raw_score"] > self._current_best_score:
            self._current_best_score = result["raw_score"]

        # Breakdown for observation
        self._last_breakdown = {
            "syntax_ok": result["syntax_ok"],
            "columns_ok": result["columns_ok"],
            "rows_ok": result["rows_ok"],
            "values_ok": result["values_ok"],
        }

        # Episode termination
        done = False
        if result["raw_score"] >= SOLVE_THRESHOLD:
            self._is_solved = True
            done = True
        elif self._attempt_number >= MAX_ATTEMPTS:
            done = True

        return reward, done

    def _build_observation(self) -> SQLObservation:
        hint = ""
        if self._attempt_number >= 2 and not self._is_solved:
            hint = self._task.hint

        breakdown = getattr(self, "_last_breakdown", {})

        return SQLObservation(
            task_id=self._task.id,
            difficulty=self._task.difficulty,
            task_description=self._task.description,
            broken_query=self._task.broken_query,
            db_schema=SHARED_SCHEMA_DDL,
            error_message=self._last_error,
            feedback=self._last_feedback,
            last_test_output=self._last_test_output,
            attempt_number=self._attempt_number,
            max_attempts=MAX_ATTEMPTS,
            steps_taken=self._steps_taken,
            hint=hint,
            score_breakdown=breakdown,
            current_best_score=self._current_best_score,
            is_solved=self._is_solved,
        )

    # ------------------------------------------------------------------
    # Utility: list all tasks (used by /tasks endpoint)
    # ------------------------------------------------------------------
    @staticmethod
    def list_tasks() -> List[Dict]:
        return [
            {
                "id": t.id,
                "difficulty": t.difficulty,
                "description": t.description,
                "schema_ddl": t.schema_ddl,
                "broken_query": t.broken_query,
                "error_pattern": t.error_pattern,
                "hint": t.hint,
            }
            for t in ALL_TASKS
        ]

    # ------------------------------------------------------------------
    # Oracle grader: runs the correct query and returns perfect score
    # ------------------------------------------------------------------
    def oracle_score(self, task_id: str) -> Dict:
        """
        Utility used by the /baseline endpoint.
        Resets the env to the task, submits the correct query, returns score.
        """
        self.reset(task_id=task_id)
        task = TASK_BY_ID.get(task_id)
        if not task:
            return {"task_id": task_id, "score": 0.0, "error": "unknown task_id"}
        action = SQLAction(action_type="submit_fix", sql_query=task.correct_query)
        _, reward, _, _ = self.step(action)
        return {"task_id": task_id, "score": reward}
