"""Minimal ComfyUI HTTP client used by the Qwen RGB weather runner.

Talks to a local (or remote) ComfyUI server over REST:
  GET  /system_stats | /object_info | /   — health
  GET  /queue                           — queue depth
  POST /prompt                          — enqueue a workflow
  GET  /history/{prompt_id}             — completion check
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import requests


class ComfyClient:
    """Thin session wrapper around ComfyUI's prompt / history APIs."""

    def __init__(self, base_url: str = "http://127.0.0.1:8188", session: Optional[requests.Session] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self._last_qcheck = 0.0
        self._cached_qsize = 0

    def health(self) -> bool:
        """Return True if any known health endpoint responds with HTTP 200."""
        for path in ("/system_stats", "/object_info", "/"):
            try:
                r = self.session.get(f"{self.base_url}{path}", timeout=5)
                if r.status_code == 200:
                    return True
            except OSError:
                pass
        return False

    def queue_size(self, interval: float = 2.0) -> int:
        """Cached queue depth (running + pending), refreshed at most every ``interval`` seconds."""
        now = time.time()
        if now - self._last_qcheck > interval:
            try:
                r = self.session.get(f"{self.base_url}/queue", timeout=5)
                r.raise_for_status()
                payload = r.json()
                self._cached_qsize = len(payload.get("queue_running", [])) + len(
                    payload.get("queue_pending", [])
                )
            except Exception:
                self._cached_qsize = 0
            self._last_qcheck = now
        return self._cached_qsize

    def queue_prompt(self, workflow: dict[str, Any]) -> Optional[str]:
        """POST a workflow graph; return ``prompt_id`` or None on failure."""
        payload = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        try:
            r = self.session.post(f"{self.base_url}/prompt", json=payload, timeout=30)
            r.raise_for_status()
            return r.json().get("prompt_id")
        except Exception as exc:
            print(f"Queue error: {exc}", flush=True)
            return None

    def is_done(self, prompt_id: str) -> bool:
        """True when history for ``prompt_id`` contains outputs."""
        try:
            r = self.session.get(f"{self.base_url}/history/{prompt_id}", timeout=10)
            r.raise_for_status()
            hist = r.json()
            return bool(hist.get(prompt_id, {}).get("outputs"))
        except Exception:
            return False
