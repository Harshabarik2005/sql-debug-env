"""
Python client for SQL Debug Environment.
"""

from __future__ import annotations

import asyncio
import subprocess
import time
from typing import Any, Dict, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import SQLAction, SQLObservation, SQLState, StepResult


class SQLDebugEnv:
    """Async HTTP client for the SQL Debug Environment."""

    # FIX: default URL is live HF Space, not localhost
    def __init__(self, base_url: str = "https://harshabarik2005-sql-debug-env.hf.space") -> None:
        self._base_url = base_url.rstrip("/")
        self._session: Any = None
        self._container_id: Optional[str] = None

    async def __aenter__(self) -> "SQLDebugEnv":
        import aiohttp
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    async def reset(self, task_id: str = "easy", **kwargs: Any) -> StepResult:
        import aiohttp
        body = {"task_id": task_id}
        try:
            async with self._session.post(
                f"{self._base_url}/reset",
                json=body,
                # FIX: increased timeout from 30s to 60s for cold HF Space starts
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            raise RuntimeError(f"reset() failed: {exc}") from exc
        return self._parse_result(data)

    async def step(self, action: SQLAction) -> StepResult:
        import aiohttp
        # FIX: exclude_none=True so optional 'explanation' field is not sent as null
        body = action.model_dump(exclude_none=True)
        try:
            async with self._session.post(
                f"{self._base_url}/step",
                json=body,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            raise RuntimeError(f"step() failed: {exc}") from exc
        return self._parse_result(data)

    async def state(self) -> SQLState:
        import aiohttp
        try:
            async with self._session.get(
                f"{self._base_url}/state",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
        except Exception as exc:
            raise RuntimeError(f"state() failed: {exc}") from exc
        # FIX: use model_validate instead of ** unpacking (handles extra fields safely)
        return SQLState.model_validate(data)

    async def close(self) -> None:
        if self._session:
            try:
                await self._session.close()
            except Exception:
                pass
            self._session = None
        if self._container_id:
            try:
                subprocess.run(["docker", "stop", self._container_id], capture_output=True, timeout=15)
                subprocess.run(["docker", "rm",   self._container_id], capture_output=True, timeout=15)
            except Exception:
                pass
            self._container_id = None

    def sync(self) -> "_SyncWrapper":
        return _SyncWrapper(self)

    @classmethod
    async def from_docker_image(
        cls,
        image_name: str,
        host_port: int = 7860,
        container_port: int = 7860,
        startup_timeout: int = 60,
    ) -> "SQLDebugEnv":
        result = subprocess.run(
            ["docker", "run", "-d", "-p", f"{host_port}:{container_port}", image_name],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"docker run failed: {result.stderr.strip()}")

        container_id = result.stdout.strip()
        base_url     = f"http://localhost:{host_port}"

        import aiohttp
        deadline  = time.time() + startup_timeout
        last_exc: Optional[Exception] = None
        while time.time() < deadline:
            try:
                async with aiohttp.ClientSession() as s:
                    async with s.get(f"{base_url}/health", timeout=aiohttp.ClientTimeout(total=3)) as r:
                        if r.status == 200:
                            break
            except Exception as exc:
                last_exc = exc
            await asyncio.sleep(2)
        else:
            subprocess.run(["docker", "stop", container_id], capture_output=True)
            subprocess.run(["docker", "rm",   container_id], capture_output=True)
            raise RuntimeError(f"Container did not become healthy within {startup_timeout}s. Last error: {last_exc}")

        client = cls(base_url=base_url)
        client._container_id = container_id
        client._session = aiohttp.ClientSession()
        return client

    @staticmethod
    def _parse_result(data: Dict[str, Any]) -> StepResult:
        # FIX: model_validate handles extra fields safely (extra='ignore' in models)
        obs = SQLObservation.model_validate(data["observation"])
        return StepResult(
            observation=obs,
            # FIX: guard against None reward from server
            reward=float(data.get("reward") or 0.0),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )


class _SyncWrapper:
    def __init__(self, client: SQLDebugEnv) -> None:
        self._client = client
        self._loop   = asyncio.new_event_loop()

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
