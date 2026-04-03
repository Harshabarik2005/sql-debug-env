"""
Task catalogue for SQL Debug Environment.

Every task specifies:
  - id, difficulty, description
  - db_schema  : CREATE TABLE DDL (shared schema, subset of tables used)
  - seed_sql   : INSERT statements to populate the DB
  - broken_query
  - correct_query  : reference answer used to compute expected output
  - hint           : shown after 2 failed attempts
  - error_pattern  : type of bug ('syntax' | 'aggregate' | 'join' | 'having' | 'subquery')
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Shared DDL — a realistic HR / projects schema
# ---------------------------------------------------------------------------
SHARED_SCHEMA_DDL = """
CREATE TABLE employees (
    id         INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    dept_id    INTEGER NOT NULL,
    salary     REAL    NOT NULL,
    hire_year  INTEGER NOT NULL
);

CREATE TABLE departments (
    id      INTEGER PRIMARY KEY,
    name    TEXT    NOT NULL,
    budget  REAL    NOT NULL
);

CREATE TABLE projects (
    id      INTEGER PRIMARY KEY,
    name    TEXT    NOT NULL,
    dept_id INTEGER NOT NULL,
    status  TEXT    NOT NULL   -- 'active' | 'completed' | 'paused'
);

CREATE TABLE employee_projects (
    employee_id  INTEGER NOT NULL,
    project_id   INTEGER NOT NULL,
    hours_worked INTEGER NOT NULL,
    PRIMARY KEY (employee_id, project_id)
);
""".strip()

SHARED_SEED_SQL = """
INSERT INTO departments VALUES
  (1, 'Engineering',   500000),
  (2, 'Data Science',  400000),
  (3, 'Support',       200000);

INSERT INTO employees VALUES
  (1, 'Alice',  1, 75000, 2019),
  (2, 'Bob',    1, 55000, 2021),
  (3, 'Carol',  2, 90000, 2018),
  (4, 'Dave',   2, 85000, 2020),
  (5, 'Eve',    3, 45000, 2022),
  (6, 'Frank',  3, 60000, 2019),
  (7, 'Grace',  1, 95000, 2017),
  (8, 'Hank',   2, 70000, 2021);

INSERT INTO projects VALUES
  (1, 'Project Alpha', 1, 'active'),
  (2, 'Project Beta',  2, 'completed'),
  (3, 'Project Gamma', 1, 'active'),
  (4, 'Project Delta', 3, 'paused');

INSERT INTO employee_projects VALUES
  (1, 1, 120), (1, 3, 80),
  (2, 1,  60),
  (3, 2, 200),
  (4, 2, 150),
  (5, 4,  40),
  (6, 4,  30),
  (7, 3, 100),
  (8, 2,  90);
""".strip()


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------
@dataclass
class Task:
    id: str
    difficulty: str               # 'easy' | 'medium' | 'hard'
    description: str
    schema_ddl: str               # subset of SHARED_SCHEMA_DDL used in task description
    broken_query: str
    correct_query: str
    hint: str
    error_pattern: str            # 'syntax' | 'aggregate' | 'join' | 'having' | 'subquery'
    tables_used: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Task 1 — EASY: Syntax typos
# ---------------------------------------------------------------------------
TASK_EASY = Task(
    id="easy",
    difficulty="easy",
    description=(
        "Retrieve the name and salary of every employee who works in department 1 "
        "(Engineering). Return all matching rows."
    ),
    schema_ddl=(
        "employees(id INTEGER, name TEXT, dept_id INTEGER, salary REAL, hire_year INTEGER)"
    ),
    broken_query="SELCT name, salary FORM employees WERE dept_id = 1",
    correct_query="SELECT name, salary FROM employees WHERE dept_id = 1",
    hint="Look carefully at the SQL keywords — check for typos in SELECT, FROM, and WHERE.",
    error_pattern="syntax",
    tables_used=["employees"],
)

# ---------------------------------------------------------------------------
# Task 2 — MEDIUM: Missing GROUP BY
# ---------------------------------------------------------------------------
TASK_MEDIUM_GROUPBY = Task(
    id="medium",
    difficulty="medium",
    description=(
        "For each department, return the department name and the total salary "
        "of all its employees. Order the results by department name."
    ),
    schema_ddl=(
        "employees(id INTEGER, name TEXT, dept_id INTEGER, salary REAL, hire_year INTEGER)\n"
        "departments(id INTEGER, name TEXT, budget REAL)"
    ),
    broken_query=(
        "SELECT d.name, SUM(e.salary) AS total_salary\n"
        "FROM departments d\n"
        "JOIN employees e ON d.id = e.dept_id\n"
        "ORDER BY d.name"
    ),
    correct_query=(
        "SELECT d.name, SUM(e.salary) AS total_salary\n"
        "FROM departments d\n"
        "JOIN employees e ON d.id = e.dept_id\n"
        "GROUP BY d.id, d.name\n"
        "ORDER BY d.name"
    ),
    hint=(
        "When you use SUM() alongside non-aggregate columns, every non-aggregate "
        "column in SELECT must appear in a GROUP BY clause."
    ),
    error_pattern="aggregate",
    tables_used=["employees", "departments"],
)

# ---------------------------------------------------------------------------
# Task 3 — MEDIUM: Wrong JOIN ON condition
# ---------------------------------------------------------------------------
TASK_MEDIUM_JOIN = Task(
    id="medium2",
    difficulty="medium",
    description=(
        "List every employee's name together with the name of the department they "
        "belong to. Include all employees. Order by employee name."
    ),
    schema_ddl=(
        "employees(id INTEGER, name TEXT, dept_id INTEGER, salary REAL, hire_year INTEGER)\n"
        "departments(id INTEGER, name TEXT, budget REAL)"
    ),
    broken_query=(
        "SELECT e.name AS employee_name, d.name AS dept_name\n"
        "FROM employees e\n"
        "JOIN departments d ON e.id = d.id\n"
        "ORDER BY e.name"
    ),
    correct_query=(
        "SELECT e.name AS employee_name, d.name AS dept_name\n"
        "FROM employees e\n"
        "JOIN departments d ON e.dept_id = d.id\n"
        "ORDER BY e.name"
    ),
    hint=(
        "Check the JOIN condition carefully — which column in `employees` stores "
        "the department reference? Is it `id` or `dept_id`?"
    ),
    error_pattern="join",
    tables_used=["employees", "departments"],
)

# ---------------------------------------------------------------------------
# Task 4 — HARD: WHERE vs HAVING with aggregate
# ---------------------------------------------------------------------------
TASK_HARD_HAVING = Task(
    id="hard",
    difficulty="hard",
    description=(
        "Find all departments where the average employee salary exceeds 70 000. "
        "Return the department id and the average salary, ordered by avg_salary descending."
    ),
    schema_ddl=(
        "employees(id INTEGER, name TEXT, dept_id INTEGER, salary REAL, hire_year INTEGER)"
    ),
    broken_query=(
        "SELECT dept_id, AVG(salary) AS avg_salary\n"
        "FROM employees\n"
        "WHERE AVG(salary) > 70000\n"
        "GROUP BY dept_id\n"
        "ORDER BY avg_salary DESC"
    ),
    correct_query=(
        "SELECT dept_id, AVG(salary) AS avg_salary\n"
        "FROM employees\n"
        "GROUP BY dept_id\n"
        "HAVING AVG(salary) > 70000\n"
        "ORDER BY avg_salary DESC"
    ),
    hint=(
        "Aggregate functions like AVG() cannot appear in a WHERE clause because WHERE "
        "filters rows *before* grouping. Use HAVING to filter *after* grouping."
    ),
    error_pattern="having",
    tables_used=["employees"],
)

# ---------------------------------------------------------------------------
# Task 5 — HARD: Scalar subquery returning multiple rows
# ---------------------------------------------------------------------------
TASK_HARD_SUBQUERY = Task(
    id="hard2",
    difficulty="hard",
    description=(
        "Find all employees whose salary is higher than the highest salary "
        "in the Support department (dept_id = 3). Return their names and salaries, "
        "ordered by salary descending."
    ),
    schema_ddl=(
        "employees(id INTEGER, name TEXT, dept_id INTEGER, salary REAL, hire_year INTEGER)"
    ),
    broken_query=(
        "SELECT name, salary\n"
        "FROM employees\n"
        "WHERE salary > (SELECT salary FROM employees WHERE dept_id = 3)\n"
        "ORDER BY salary DESC"
    ),
    correct_query=(
        "SELECT name, salary\n"
        "FROM employees\n"
        "WHERE salary > (SELECT MAX(salary) FROM employees WHERE dept_id = 3)\n"
        "ORDER BY salary DESC"
    ),
    hint=(
        "The subquery returns multiple rows (one per dept-3 employee), but the > "
        "operator expects a single scalar value. Wrap the subquery with MAX() to "
        "get the single highest salary."
    ),
    error_pattern="subquery",
    tables_used=["employees"],
)

# ---------------------------------------------------------------------------
# Public catalogue
# ---------------------------------------------------------------------------
ALL_TASKS: List[Task] = [
    TASK_EASY,
    TASK_MEDIUM_GROUPBY,
    TASK_MEDIUM_JOIN,
    TASK_HARD_HAVING,
    TASK_HARD_SUBQUERY,
]

TASK_BY_ID = {t.id: t for t in ALL_TASKS}

# Canonical ordering for the baseline inference script
TASK_IDS_ORDERED = ["easy", "medium", "medium2", "hard", "hard2"]
