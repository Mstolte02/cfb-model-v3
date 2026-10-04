"""Minimal TypeSafe/Jev API client."""
from __future__ import annotations

import os
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class JevClient:
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.typesafe.ai", model: str | None = None, timeout: float = 30.0, max_attempts: int = 5) -> None:
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model or os.getenv("JEV_MODEL", "jev-latest")
        self.timeout = timeout
        self.max_attempts = max(1, int(max_attempts))

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for live Jev calls")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def models(self) -> dict[str, Any]:
        request = Request(f"{self.base_url}/v1/models", headers=self._headers())
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)

    def evaluate(self, state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        payload = {"state": state, "model": model or self.model, "questions": questions}
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.max_attempts + 1):
            request = Request(f"{self.base_url}/v1/systemone", headers=self._headers(),
                              data=body, method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == self.max_attempts:
                    raise
            except URLError:
                if attempt == self.max_attempts:
                    raise
            time.sleep(2 ** (attempt - 1))
        raise RuntimeError("unreachable Jev retry state")

    def payload(self, state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        return {"state": state, "model": model or self.model, "questions": questions}
