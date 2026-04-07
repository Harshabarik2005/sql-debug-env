"""
Tests for SQL Debug Environment.

Run with:
    PYTHONPATH=. pytest tests/ -v
"""

from __future__ import annotations

import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from models import SQLAction, SQLObservation, SQLState
from server.environment import SQLDebugEnvironment
from server.graders import grade, compute_expected, run_test_query
from server.tasks import ALL_TASKS, TASK_BY_ID, SHARED_SCHEMA_DDL, SHARED_SEED_SQL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_db():
    conn = sqlite3.connect(":memory:")
    conn.executescript(SHARED_SCHEMA_DDL)
    conn.executescript(SHARED_SEED_SQL)
    conn.commit()
    return conn


@pytest.fixture
def env():
    return SQLDebugEnvironment()


# ---------------------------------------------------------------------------
# Grader unit tests
# ---------------------------------------------------------------------------

class TestGrader:

    def test_perfect_score_easy(self, fresh_db):
        task = TASK_BY_ID["easy"]
        expected_cols, expected_rows = compute_expected(fresh_db, task.correct_query)
        result = grade(fresh_db, task.correct_query, expected_cols, expected_rows, attempt_number=1)
        assert result["score"] == pytest.approx(0.99, abs=0.001)  # clamped max
        assert result["syntax_ok"] == pytest.approx(0.30, abs=0.01)
        assert result["columns_ok"] == pytest.approx(0.20, abs=0.01)
        assert result["rows_ok"] == pytest.approx(0.10, abs=0.01)
        assert result["values_ok"] == pytest.approx(0.40, abs=0.01)

    def test_syntax_error_scores_zero(self, fresh_db):
        task = TASK_BY_ID["easy"]
        expected_cols, expected_rows = compute_expected(fresh_db, task.correct_query)
        result = grade(fresh_db, "SELCT name FORM employees WERE dept_id = 1",
                       expected_cols, expected_rows, attempt_number=1)
        assert result["syntax_ok"] == 0.0
        assert result["score"] == pytest.approx(0.01, abs=0.001)  # clamped min
        assert result["error_message"] != ""

    def test_columns_mismatch_partial_credit(self, fresh_db):
        task = TASK_BY_ID["easy"]
        expected_cols, expected_rows = compute_expected(fresh_db, task.correct_query)
        # Return only 1 of 2 expected columns
        result = grade(fresh_db, "SELECT name FROM employees WHERE dept_id = 1",
                       expected_cols, expected_rows, attempt_number=1)
        assert result["syntax_ok"] == pytest.approx(0.30, abs=0.01)
        assert result["columns_ok"] < 0.20  # partial or zero

    def test_wrong_filter_partial_credit(self, fresh_db):
        task = TASK_BY_ID["easy"]
        expected_cols, expected_rows = compute_expected(fresh_db, task.correct_query)
        # Correct columns but wrong filter — different rows
        result = grade(fresh_db, "SELECT name, salary FROM employees WHERE dept_id = 2",
                       expected_cols, expected_rows, attempt_number=1)
        assert result["syntax_ok"] == pytest.approx(0.30, abs=0.01)
        assert result["columns_ok"] == pytest.approx(0.20, abs=0.01)
        # dept_id=2 has same row count (3) as dept_id=1, so rows_ok can be 0.10
        assert result["values_ok"] < 0.40  # Carol/Dave/Hank ≠ Alice/Bob/Grace

    def test_attempt_multiplier(self, fresh_db):
        task = TASK_BY_ID["easy"]
        expected_cols, expected_rows = compute_expected(fresh_db, task.correct_query)
        r1 = grade(fresh_db, task.correct_query, expected_cols, expected_rows, attempt_number=1)
        r3 = grade(fresh_db, task.correct_query, expected_cols, expected_rows, attempt_number=3)
        assert r1["attempt_mult"] == pytest.approx(0.99, abs=0.001)  # clamped max
        assert r3["attempt_mult"] == 0.90
        assert r1["score"] > r3["score"]

    def test_score_in_range(self, fresh_db):
        task = TASK_BY_ID["medium"]
        expected_cols, expected_rows = compute_expected(fresh_db, task.correct_query)
        for query in [task.correct_query, task.broken_query, "SELECT 1", "BROKEN SQL @@"]:
            result = grade(fresh_db, query, expected_cols, expected_rows, attempt_number=1)
            assert 0.0 <= result["score"] <= 1.0, f"Score out of range for: {query!r}"

    def test_run_test_query_success(self, fresh_db):
        ok, output = run_test_query(fresh_db, "SELECT name FROM employees LIMIT 3")
        assert ok is True
        assert "Alice" in output or "name" in output.lower()

    def test_run_test_query_error(self, fresh_db):
        ok, output = run_test_query(fresh_db, "SELCT * FORM employees")
        assert ok is False
        assert "SQL Error" in output


# ---------------------------------------------------------------------------
# Environment unit tests
# ---------------------------------------------------------------------------

class TestEnvironment:

    def test_reset_returns_observation(self, env):
        obs = env.reset(task_id="easy")
        assert isinstance(obs, SQLObservation)
        assert obs.task_id == "easy"
        assert obs.difficulty == "easy"
        assert obs.attempt_number == 0
        assert obs.steps_taken == 0
        assert obs.is_solved is False
        assert obs.broken_query != ""
        assert obs.db_schema != ""

    def test_reset_unknown_task_falls_back(self, env):
        obs = env.reset(task_id="nonexistent")
        assert obs.task_id == "easy"

    def test_state_after_reset(self, env):
        env.reset(task_id="medium")
        s = env.state
        assert isinstance(s, SQLState)
        assert s.task_id == "medium"
        assert s.attempt_number == 0

    def test_test_query_does_not_consume_attempt(self, env):
        env.reset(task_id="easy")
        action = SQLAction(action_type="test_query",
                           sql_query="SELECT name FROM employees LIMIT 1")
        obs, reward, done, _ = env.step(action)
        assert obs.attempt_number == 0
        assert done is False
        assert reward >= 0.0

    def test_test_query_gives_reward_for_valid_sql(self, env):
        env.reset(task_id="easy")
        action = SQLAction(action_type="test_query",
                           sql_query="SELECT name, salary FROM employees WHERE dept_id = 1")
        _, reward, _, _ = env.step(action)
        assert reward == pytest.approx(0.05, abs=0.001)

    def test_test_query_shows_output(self, env):
        env.reset(task_id="easy")
        action = SQLAction(action_type="test_query",
                           sql_query="SELECT * FROM departments")
        obs, _, _, _ = env.step(action)
        assert obs.last_test_output != ""
        assert "Engineering" in obs.last_test_output

    def test_submit_fix_correct_solution_easy(self, env):
        env.reset(task_id="easy")
        action = SQLAction(
            action_type="submit_fix",
            sql_query="SELECT name, salary FROM employees WHERE dept_id = 1",
        )
        obs, reward, done, _ = env.step(action)
        assert reward >= 0.9
        assert done is True
        assert obs.is_solved is True
        assert obs.attempt_number == 1

    def test_submit_fix_correct_solution_medium(self, env):
        env.reset(task_id="medium")
        action = SQLAction(
            action_type="submit_fix",
            sql_query=(
                "SELECT d.name, SUM(e.salary) AS total_salary "
                "FROM departments d JOIN employees e ON d.id = e.dept_id "
                "GROUP BY d.id, d.name ORDER BY d.name"
            ),
        )
        obs, reward, done, _ = env.step(action)
        assert reward >= 0.9
        assert done is True

    def test_submit_fix_correct_solution_hard(self, env):
        env.reset(task_id="hard")
        action = SQLAction(
            action_type="submit_fix",
            sql_query=(
                "SELECT dept_id, AVG(salary) AS avg_salary FROM employees "
                "GROUP BY dept_id HAVING AVG(salary) > 70000 ORDER BY avg_salary DESC"
            ),
        )
        obs, reward, done, _ = env.step(action)
        assert reward >= 0.9
        assert done is True

    def test_submit_fix_correct_solution_hard2(self, env):
        env.reset(task_id="hard2")
        action = SQLAction(
            action_type="submit_fix",
            sql_query=(
                "SELECT name, salary FROM employees "
                "WHERE salary > (SELECT MAX(salary) FROM employees WHERE dept_id = 3) "
                "ORDER BY salary DESC"
            ),
        )
        obs, reward, done, _ = env.step(action)
        assert reward >= 0.9
        assert done is True

    def test_submit_broken_syntax_no_solve(self, env):
        env.reset(task_id="easy")
        action = SQLAction(action_type="submit_fix", sql_query="SELCT name FORM employees")
        obs, reward, done, _ = env.step(action)
        assert reward == pytest.approx(0.01, abs=0.001)  # clamped min — validator rejects 0.0
        assert done is False
        assert obs.error_message != ""

    def test_attempt_limit_ends_episode(self, env):
        env.reset(task_id="medium")
        bad = SQLAction(action_type="submit_fix", sql_query="SELECT 1")
        done = False
        for _ in range(5):
            obs, _, done, _ = env.step(bad)
        assert done is True
        assert obs.attempt_number == 5

    def test_hint_appears_after_two_failures(self, env):
        env.reset(task_id="hard")
        bad = SQLAction(action_type="submit_fix", sql_query="SELECT 1")
        env.step(bad)
        env.step(bad)
        obs, _, _, _ = env.step(SQLAction(action_type="test_query",
                                           sql_query="SELECT 1"))
        assert obs.hint != ""

    def test_score_breakdown_populated_after_submit(self, env):
        env.reset(task_id="easy")
        action = SQLAction(
            action_type="submit_fix",
            sql_query="SELECT name, salary FROM employees WHERE dept_id = 1",
        )
        obs, _, _, _ = env.step(action)
        assert "syntax_ok" in obs.score_breakdown
        assert "values_ok" in obs.score_breakdown

    def test_rewards_always_in_range(self, env):
        for task in ALL_TASKS:
            env.reset(task_id=task.id)
            action = SQLAction(action_type="submit_fix", sql_query=task.correct_query)
            _, reward, _, _ = env.step(action)
            assert 0.0 <= reward <= 1.0, f"Reward {reward} out of range for task {task.id}"

    def test_all_tasks_have_valid_correct_queries(self, env):
        """Oracle should score >= 0.9 on every task."""
        for task in ALL_TASKS:
            result = env.oracle_score(task.id)
            assert result["score"] >= 0.9, (
                f"Oracle score {result['score']:.2f} < 0.9 for task '{task.id}'"
            )

    def test_multiple_resets_are_independent(self, env):
        env.reset(task_id="easy")
        env.step(SQLAction(action_type="submit_fix", sql_query="SELECT 1"))
        obs2 = env.reset(task_id="medium")
        assert obs2.task_id == "medium"
        assert obs2.attempt_number == 0

    def test_max_steps_terminates_episode(self, env):
        from server.environment import MAX_STEPS
        env.reset(task_id="easy")
        done = False
        for _ in range(MAX_STEPS):
            _, _, done, info = env.step(
                SQLAction(action_type="test_query", sql_query="SELECT 1")
            )
            if done:
                break
        assert done is True


# ---------------------------------------------------------------------------
# Task catalogue tests
# ---------------------------------------------------------------------------

class TestTaskCatalogue:

    def test_task_count(self):
        assert len(ALL_TASKS) == 5

    def test_difficulty_coverage(self):
        difficulties = {t.difficulty for t in ALL_TASKS}
        assert "easy" in difficulties
        assert "medium" in difficulties
        assert "hard" in difficulties

    def test_all_tasks_have_required_fields(self):
        for t in ALL_TASKS:
            assert t.id, f"Task missing id: {t}"
            assert t.broken_query, f"Task {t.id} missing broken_query"
            assert t.correct_query, f"Task {t.id} missing correct_query"
            assert t.hint, f"Task {t.id} missing hint"
            assert t.error_pattern, f"Task {t.id} missing error_pattern"

    def test_broken_differs_from_correct(self):
        for t in ALL_TASKS:
            assert t.broken_query.strip() != t.correct_query.strip(), (
                f"Task {t.id}: broken_query equals correct_query"
            )
