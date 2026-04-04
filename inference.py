"""
Baseline inference script for SQL Debug Environment.
=====================================================
MANDATORY environment variables:
    API_BASE_URL        The LLM API endpoint (default: HuggingFace router)
    MODEL_NAME          Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN            Hugging Face / API key
    IMAGE_NAME          Docker image name (optional; set to use from_docker_image)
    SQL_DEBUG_ENV_URL   Direct server URL (default: http://localhost:7860)

STDOUT FORMAT (strict — any deviation causes eval failure):
    [START] task=<task_id> env=sql_debug_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from client import SQLDebugEnv
from models import SQLAction
from server.tasks import TASK_IDS_ORDERED

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL = os.getenv("SQL_DEBUG_ENV_URL", "http://localhost:7860")

MAX_STEPS_PER_EPISODE = 15
TEMPERATURE = 0.3
MAX_TOKENS = 512
SOLVE_THRESHOLD = 0.9

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert SQL debugging agent. You receive a broken SQL query and must fix it.

AVAILABLE ACTIONS:
1. test_query: Run a SQL query to inspect results (no penalty, up to 5 free test calls).
2. submit_fix: Submit your corrected query for grading.

STRATEGY:
- First, carefully read the broken query and identify the bug type.
- Optionally use test_query to verify your fix before submitting.
- Use submit_fix when confident in your answer.

RESPONSE FORMAT — respond with EXACTLY one of these two JSON objects:

For testing:
{"action_type": "test_query", "sql_query": "<your test SQL>"}

For submitting:
{"action_type": "submit_fix", "sql_query": "<your corrected SQL>", "explanation": "<brief reason>"}

Output ONLY the JSON object. No prose, no markdown, no extra text.
""").strip()


def build_user_prompt(obs_dict: dict, step: int, history: List[str]) -> str:
    hist_block = "\n".join(history[-6:]) if history else "None"
    hint_line = f"\nHINT: {obs_dict['hint']}" if obs_dict.get("hint") else ""
    feedback_line = f"\nLAST FEEDBACK:\n{obs_dict['feedback']}" if obs_dict.get("feedback") else ""
    test_out_line = (
        f"\nLAST TEST OUTPUT:\n{obs_dict['last_test_output']}"
        if obs_dict.get("last_test_output")
        else ""
    )
    error_line = f"\nERROR: {obs_dict['error_message']}" if obs_dict.get("error_message") else ""

    return textwrap.dedent(f"""
        STEP: {step}
        TASK ({obs_dict['difficulty']}): {obs_dict['task_description']}

        BROKEN QUERY:
        {obs_dict['broken_query']}

        DB SCHEMA:
        {obs_dict['db_schema']}

        ATTEMPTS USED: {obs_dict['attempt_number']} / {obs_dict['max_attempts']}{hint_line}{error_line}{feedback_line}{test_out_line}

        PREVIOUS STEPS:
        {hist_block}

        What is your next action?
    """).strip()


def call_llm(client: OpenAI, user_prompt: str) -> dict:
    import json

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.rstrip("`").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        # Fallback: echo the broken query (will score low, but won't crash)
        return {"action_type": "submit_fix", "sql_query": "SELECT 1"}


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------
async def run_episode(env: SQLDebugEnv, task_id: str, llm_client: OpenAI) -> dict:
    """Run one full episode for a task. Returns {task_id, success, steps, score, rewards}."""
    log_start(task=task_id, env="sql_debug_env", model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    history: List[str] = []

    try:
        result = await env.reset(task_id=task_id)
        obs = result.observation

        for step in range(1, MAX_STEPS_PER_EPISODE + 1):
            if result.done:
                break

            obs_dict = obs.model_dump()
            user_prompt = build_user_prompt(obs_dict, step, history)
            action_dict = call_llm(llm_client, user_prompt)

            # Force submit after 3 test queries to avoid infinite loops
            test_count = sum(1 for h in history if "[test_query]" in h)
            if test_count >= 3 and action_dict.get("action_type") == "test_query":
                action_dict["action_type"] = "submit_fix"

            # Build the action object
            try:
                action = SQLAction(**action_dict)
            except Exception:
                action = SQLAction(action_type="submit_fix", sql_query=action_dict.get("sql_query", "SELECT 1"))

            result = await env.step(action)
            obs = result.observation

            reward = result.reward
            done = result.done
            error = obs.error_message or None

            rewards.append(reward)
            steps_taken = step

            # Action summary for logging
            action_str = f"{action.action_type}({repr(action.sql_query[:80])})"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(
                f"Step {step} [{action.action_type}]: {(action.sql_query or '')[:60]!r} → reward={reward:.2f}"
            )

            if done:
                score = obs.current_best_score
                success = obs.is_solved
                break

        if not rewards:
            score = 0.0
            success = False
        else:
            score = obs.current_best_score

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task_id": task_id, "success": success, "steps": steps_taken, "score": score}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if IMAGE_NAME:
        print(f"[DEBUG] Starting container from image: {IMAGE_NAME}", flush=True)
        env = await SQLDebugEnv.from_docker_image(IMAGE_NAME)
    else:
        env = SQLDebugEnv(base_url=ENV_URL)
        import aiohttp

        env._session = aiohttp.ClientSession()

    all_results = []
    try:
        for task_id in TASK_IDS_ORDERED:
            result = await run_episode(env, task_id, llm_client)
            all_results.append(result)
    finally:
        await env.close()

    # Summary
    print("\n[SUMMARY]", flush=True)
    for r in all_results:
        print(
            f"  task={r['task_id']} success={r['success']} score={r['score']:.3f} steps={r['steps']}",
            flush=True,
        )
    avg = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"  average_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
