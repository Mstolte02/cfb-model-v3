"""Minimal TypeSafe/Jev API client."""
from __future__ import annotations

import os
from typing import Any
import requests


class JevClient:
    def __init__(self, api_key: str | None = None, base_url: str = "https://api.typesafe.ai", model: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model or os.getenv("JEV_MODEL", "jev-latest")
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("TYPESAFE_API_KEY is required for live Jev calls")
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def models(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def evaluate(self, state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        payload = {"state": state, "model": model or self.model, "questions": questions}
        response = requests.post(f"{self.base_url}/v1/systemone", headers=self._headers(), json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def payload(self, state: Any, questions: dict[str, Any], model: str | None = None) -> dict[str, Any]:
        return {"state": state, "model": model or self.model, "questions": questions}
