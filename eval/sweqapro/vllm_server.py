"""Subprocess-managed local vLLM OpenAI-compatible server.

Used by `scripts/run_agent.py` and `scripts/run_direct.py` for vllm-local
models. Lifetime is scoped to a single context-manager block.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import requests

DEFAULT_STARTUP_TIMEOUT = 600
POLL_INTERVAL = 2.0


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def _bool_flag(args: list, key: str, value: Any) -> None:
    if value is True:
        args.append(_flag(key))


class VLLMServer:
    """Context manager that launches `vllm serve <model>` and tears it down on exit.

    `vllm` and `agent_vllm` are dicts from the model spec. The vllm block is
    always applied; `agent_vllm` is only applied when `with_tools=True` (the
    block configures tool-call parsing, which is irrelevant for direct mode).
    """

    def __init__(
        self,
        model_id: str,
        *,
        with_tools: bool,
        vllm: Optional[Dict[str, Any]] = None,
        agent_vllm: Optional[Dict[str, Any]] = None,
        host: str = "127.0.0.1",
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    ):
        self.model_id = model_id
        self.with_tools = with_tools
        self.vllm = dict(vllm or {})
        self.agent_vllm = dict(agent_vllm or {})
        self.host = host
        self.startup_timeout = startup_timeout
        self.port: Optional[int] = None
        self.proc: Optional[subprocess.Popen] = None
        self.base_url: Optional[str] = None

    # ---------------------------------------------------------------- internals

    def _build_command(self) -> list[str]:
        args: list[str] = [
            "vllm", "serve", self.model_id,
            "--host", self.host,
            "--port", str(self.port),
            "--trust-remote-code",
        ]
        for key, value in self.vllm.items():
            if isinstance(value, bool):
                _bool_flag(args, key, value)
            else:
                args.extend([_flag(key), str(value)])
        if self.with_tools and self.agent_vllm:
            for key, value in self.agent_vllm.items():
                if isinstance(value, bool):
                    _bool_flag(args, key, value)
                else:
                    args.extend([_flag(key), str(value)])
        return args

    def _wait_for_ready(self) -> None:
        url = f"http://{self.host}:{self.port}/v1/models"
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                code = self.proc.returncode
                raise RuntimeError(f"vLLM server exited early with code {code}")
            try:
                r = requests.get(url, timeout=5)
                if r.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(POLL_INTERVAL)
        raise TimeoutError(f"vLLM did not become ready within {self.startup_timeout}s")

    # ---------------------------------------------------------------- ctxmgr

    def __enter__(self) -> "VLLMServer":
        self.port = _pick_free_port()
        cmd = self._build_command()
        print(f"[VLLMServer] launching: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            start_new_session=True,
        )
        try:
            self._wait_for_ready()
        except BaseException:
            self._terminate()
            raise
        self.base_url = f"http://{self.host}:{self.port}/v1"
        print(f"[VLLMServer] ready at {self.base_url}", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._terminate()

    def _terminate(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=10)
        finally:
            print("[VLLMServer] terminated", flush=True)
