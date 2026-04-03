"""
Python client for SQL Debug Environment.

Usage (async — recommended for training loops):

    from client import SQLDebugEnv
    from models import SQLAction

    async with SQLDebugEnv(base_url="https://your-space.hf.space") as env:
        result = await env.reset(task_id="easy")
        print(result.observation.broken_query)

        result = await env.step(SQLAction(
            action_type="submit_fix",
            sql_query="SELECT name, salary FROM employees WHERE dept_id = 1",
        ))
        print(result.reward, result.observation.feedback)

Usage (sync wrapper):

    with SQLDebugEnv(base_url="...").sync() as env:
        result = env.reset(task_id="medium")
        result = env.step(SQLAction(...))

Usage (from docker image):

    env = await SQLDebugEnv.from_docker_image("sql-debug-env:latest")
    async with env:
        result = await env.reset()
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any, Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SQLAction, SQLObservation, SQLState, StepResult

# ---------------------------------------------------------------------------
# Try to use openenv-core HTTPEnvClient; fall back to a custom aiohttp client
# ---------------------------------------------------------------------------
try:
    import aiohttp as _aiohttp_check  # noqa: F401

    _HAS_AIOHTTP = True
except ImportError:
    _HAS_AIOHTTP = False


class SQLDebugEnv:
    """
    Async HTTP client for the SQL Debug Environment.

    Connects to a running FastAPI server via HTTP.
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self._base_url = base_url.rstrip("/")
        self._session: Any = None
        self._container_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "SQLDebugEnv":
        import aiohttp

        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    async def reset(self, task_id: str = "easy", **kwargs: Any) -> StepResult:
        """Reset the environment.  Returns a StepResult with the initial observation."""
        import aiohttp

        body = {"task_id": task_id}
        try:
            async with self._session.post(
                f"{self._base_url}/reset",
                json=body,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            raise RuntimeError(f"reset() failed: {exc}") from exc
        return self._parse_result(data)

    async def step(self, action: SQLAction) -> StepResult:
        """Take one step.  Returns a StepResult."""
        import aiohttp

        body = action.model_dump()
        try:
            async with self._session.post(
                f"{self._base_url}/step",
                json=body,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            raise RuntimeError(f"step() failed: {exc}") from exc
        return self._parse_result(data)

    async def state(self) -> SQLState:
        """Return the current episode state."""
        import aiohttp

        try:
            async with self._session.get(
                f"{self._base_url}/state",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            raise RuntimeError(f"state() failed: {exc}") from exc
        return SQLState(**data)

    async def close(self) -> None:
        """Close the HTTP session and stop the Docker container if any."""
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        if self._container_id:
            try:
                subprocess.run(
                    ["docker", "stop", self._container_id],
                    capture_output=True,
                    timeout=15,
                )
                subprocess.run(
                    ["docker", "rm", self._container_id],
                    capture_output=True,
                    timeout=15,
                )
            except Exception:
                pass
            self._container_id = None

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------
    def sync(self) -> "_SyncWrapper":
        return _SyncWrapper(self)

    # ------------------------------------------------------------------
    # Factory: from Docker image
    # ------------------------------------------------------------------
    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        host_port: int = 7860,
        container_port: int = 7860,
        startup_timeout: int = 60,
    ) -> "SQLDebugEnv":
        """
        Start a Docker container from *image_name* and return a connected client.

        Parameters
        ----------
        image_name      : Docker image tag, e.g. "sql-debug-env:latest"
        host_port       : Port to expose on localhost
        container_port  : Port the server listens on inside the container
        startup_timeout : Seconds to wait for the health endpoint
        """
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "-p", f"{host_port}:{container_port}",
                image_name,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        base_url = f"http://localhost:{host_port}"

        # Wait for health endpoint
        import aiohttp

        deadline = time.time() + startup_timeout
        last_exc: Optional[Exception] = None
        while time.time() < deadline:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(
                        f"{base_url}/health",
                        timeout=aiohttp.ClientTimeout(total=3),
                    ) as r:
                        if r.status == 200:
                            break
            except Exception as exc:
                last_exc = exc
            await asyncio.sleep(2)
        else:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm", container_id], capture_output=True)
            raise RuntimeError(
                f"Container did not become healthy within {startup_timeout}s. "
                f"Last error: {last_exc}"
            )

        client = cls(base_url=base_url)
        client._container_id = container_id
        # Open session
        client._session = aiohttp.ClientSession()
        return client

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_result(data: Dict[str, Any]) -> StepResult:
        obs = SQLObservation(**data["observation"])
        return StepResult(
            observation=obs,
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )


# ---------------------------------------------------------------------------
# Synchronous wrapper
# ---------------------------------------------------------------------------
class _SyncWrapper:
    """Synchronous interface around the async SQLDebugEnv client."""

    def __init__(self, client: SQLDebugEnv) -> None:
        self._client = client
        self._loop = asyncio.new_event_loop()

    def __enter__(self) -> "_SyncWrapper":
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *args: Any) -> None:
        self._loop.run_until_complete(self._client.__aexit__(*args))
        self._loop.close()

    def reset(self, task_id: str = "easy", **kwargs: Any) -> StepResult:
        return self._loop.run_until_complete(self._client.reset(task_id=task_id, **kwargs))

    def step(self, action: SQLAction) -> StepResult:
        return self._loop.run_until_complete(self._client.step(action))

    def state(self) -> SQLState:
        return self._loop.run_until_complete(self._client.state())
