"""
Deterministic grader for SQL Debug Environment.

Score components (summing to 1.0):
  syntax_ok   0.30  — query executes without error
  columns_ok  0.20  — returned column set matches expected (order-insensitive)
  rows_ok     0.10  — row count matches expected
  values_ok   0.40  — cell values match (partial credit: score per row)

Attempt multiplier (applied to the raw 0-1 score):
  attempt 1   × 1.00
  attempt 2   × 0.95
  attempt 3   × 0.90
  attempt 4   × 0.85
  attempt 5   × 0.80
"""

from __future__ import annotations

import sqlite3
import math
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_to_dicts(cursor: sqlite3.Cursor) -> Tuple[List[str], List[Dict]]:
    """Return (column_names, list_of_row_dicts) from a cursor after execute."""
    cols = [d[0].lower() for d in cursor.description] if cursor.description else []
    rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
    return cols, rows


def _normalise_value(v: Any) -> Any:
    """Normalise a cell value for comparison."""
    if v is None:
        return None
    if isinstance(v, float):
        # Round to 2 decimal places to avoid floating-point noise
        return round(v, 2)
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _rows_match(actual_rows: List[Dict], expected_rows: List[Dict]) -> float:
    """
    Return fraction of expected rows that appear in actual_rows.
    Comparison is order-insensitive; values are normalised before comparison.
    Partial credit: 1 point per matched expected row / len(expected_rows).
    """
    if not expected_rows:
        return 1.0 if not actual_rows else 0.0

    # Normalise both sets — sort by str(value) to handle mixed float/str columns
    def norm_row(r: Dict) -> tuple:
        return tuple(sorted((_normalise_value(v) for v in r.values()), key=lambda x: str(x)))

    actual_set: List[tuple] = [norm_row(r) for r in actual_rows]
    expected_set: List[tuple] = [norm_row(r) for r in expected_rows]

    # Count how many expected rows are covered (multiset matching)
    remaining = list(actual_set)
    matched = 0
    for exp in expected_set:
        if exp in remaining:
            remaining.remove(exp)
            matched += 1

    return matched / len(expected_rows)


# ---------------------------------------------------------------------------
# Public grader
# ---------------------------------------------------------------------------

def grade(
    conn: sqlite3.Connection,
    submitted_query: str,
    expected_cols: List[str],
    expected_rows: List[Dict],
    attempt_number: int = 1,
) -> Dict:
    """
    Grade a submitted SQL query against expected results.

    Parameters
    ----------
    conn            : sqlite3.Connection with the task DB pre-populated
    submitted_query : SQL string submitted by the agent
    expected_cols   : column names from the reference query (lower-cased)
    expected_rows   : list of row-dicts from the reference query
    attempt_number  : 1-indexed attempt count (affects multiplier)

    Returns
    -------
    dict with keys:
        score           float  final weighted score in [0, 1]
        raw_score       float  score before attempt multiplier
        attempt_mult    float  multiplier applied
        syntax_ok       float  0.0 or 0.30
        columns_ok      float  0.0 or 0.20
        rows_ok         float  0.0 or 0.10
        values_ok       float  in [0.0, 0.40]
        error_message   str    SQL error if any
        feedback        str    human-readable feedback
        actual_rows     list   rows returned by the submitted query
    """
    breakdown = {
        "syntax_ok": 0.0,
        "columns_ok": 0.0,
        "rows_ok": 0.0,
        "values_ok": 0.0,
    }
    error_message = ""
    actual_rows: List[Dict] = []
    actual_cols: List[str] = []

    # --- Syntax / execution check (0.30) ---
    try:
        cur = conn.execute(submitted_query)
        actual_cols, actual_rows = _rows_to_dicts(cur)
        breakdown["syntax_ok"] = 0.30
    except sqlite3.Error as exc:
        error_message = str(exc)
        raw_score = 0.0
        feedback = (
            f"Your query raised an SQL error: {error_message}\n"
            "Fix the syntax or logic before re-submitting."
        )
        return _build_result(breakdown, raw_score, attempt_number, error_message, feedback, actual_rows)

    # --- Column check (0.20) ---
    if set(actual_cols) == set(expected_cols):
        breakdown["columns_ok"] = 0.20
    elif set(actual_cols).issuperset(set(expected_cols)):
        breakdown["columns_ok"] = 0.10  # partial: extra columns
    elif set(actual_cols).issubset(set(expected_cols)) and actual_cols:
        breakdown["columns_ok"] = 0.05  # partial: missing some columns

    # --- Row count check (0.10) ---
    if len(actual_rows) == len(expected_rows):
        breakdown["rows_ok"] = 0.10
    elif len(expected_rows) > 0:
        ratio = min(len(actual_rows), len(expected_rows)) / max(len(actual_rows), len(expected_rows))
        breakdown["rows_ok"] = round(0.10 * ratio, 4)

    # --- Value match check (0.40) ---
    if expected_rows:
        match_frac = _rows_match(actual_rows, expected_rows)
        breakdown["values_ok"] = round(0.40 * match_frac, 4)
    else:
        breakdown["values_ok"] = 0.40 if not actual_rows else 0.0

    raw_score = sum(breakdown.values())

    # --- Build feedback string ---
    feedback = _build_feedback(breakdown, actual_cols, expected_cols, actual_rows, expected_rows)

    return _build_result(breakdown, raw_score, attempt_number, error_message, feedback, actual_rows)


def _build_result(
    breakdown: Dict,
    raw_score: float,
    attempt_number: int,
    error_message: str,
    feedback: str,
    actual_rows: List[Dict],
) -> Dict:
    multipliers = {1: 1.00, 2: 0.95, 3: 0.90, 4: 0.85, 5: 0.80}
    mult = multipliers.get(attempt_number, 0.75)
    final_score = round(raw_score * mult, 4)
    return {
        "score": final_score,
        "raw_score": round(raw_score, 4),
        "attempt_mult": mult,
        "syntax_ok": breakdown["syntax_ok"],
        "columns_ok": breakdown["columns_ok"],
        "rows_ok": breakdown["rows_ok"],
        "values_ok": breakdown["values_ok"],
        "error_message": error_message,
        "feedback": feedback,
        "actual_rows": actual_rows,
    }


def _build_feedback(
    breakdown: Dict,
    actual_cols: List[str],
    expected_cols: List[str],
    actual_rows: List[Dict],
    expected_rows: List[Dict],
) -> str:
    lines = []

    if breakdown["syntax_ok"] >= 0.30:
        lines.append("✓ Query executed without errors.")
    else:
        lines.append("✗ Query failed to execute.")
        return "\n".join(lines)

    # Columns
    if breakdown["columns_ok"] >= 0.20:
        lines.append("✓ Columns match expected output.")
    else:
        missing = sorted(set(expected_cols) - set(actual_cols))
        extra = sorted(set(actual_cols) - set(expected_cols))
        if missing:
            lines.append(f"✗ Missing expected columns: {missing}")
        if extra:
            lines.append(f"  Extra columns returned (not required): {extra}")

    # Row count
    if breakdown["rows_ok"] >= 0.10:
        lines.append(f"✓ Row count correct ({len(actual_rows)} rows).")
    else:
        lines.append(
            f"✗ Row count mismatch: got {len(actual_rows)}, expected {len(expected_rows)}."
        )

    # Values
    pct = round(breakdown["values_ok"] / 0.40 * 100)
    if pct == 100:
        lines.append("✓ All row values match.")
    elif pct > 0:
        lines.append(f"~ {pct}% of expected rows matched correctly.")
    else:
        lines.append("✗ Row values do not match expected output.")

    score = sum(breakdown.values())
    lines.append(f"\nRaw score: {score:.2f}/1.00")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test-query executor (does NOT grade — just returns formatted output)
# ---------------------------------------------------------------------------

def run_test_query(conn: sqlite3.Connection, sql: str, max_rows: int = 10) -> Tuple[bool, str]:
    """
    Execute a SQL query and return (success, formatted_output).
    Used for test_query action — does not affect attempt count.
    """
    try:
        cur = conn.execute(sql)
        cols, rows = _rows_to_dicts(cur)
        if not cols:
            return True, "(Query executed successfully, no rows returned)"
        # Format as a simple ASCII table
        lines = [" | ".join(cols)]
        lines.append("-" * len(lines[0]))
        for row in rows[:max_rows]:
            lines.append(" | ".join(str(row.get(c, "")) for c in cols))
        if len(rows) > max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows not shown)")
        return True, "\n".join(lines)
    except sqlite3.Error as exc:
        return False, f"SQL Error: {exc}"


# ---------------------------------------------------------------------------
# Pre-compute expected results for a task
# ---------------------------------------------------------------------------

def compute_expected(conn: sqlite3.Connection, correct_query: str) -> Tuple[List[str], List[Dict]]:
    """Run the reference (correct) query and return (cols, rows)."""
    cur = conn.execute(correct_query)
    return _rows_to_dicts(cur)
