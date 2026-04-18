"""Ollama HTTP API client for prompt-lab."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class ModelResult:
    model: str
    response: str
    elapsed_seconds: float
    error: str | None = None


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base = base_url.rstrip("/")

    def list_model_names(self) -> list[str]:
        with httpx.Client(base_url=self._base, timeout=30) as c:
            resp = c.get("/api/tags")
            resp.raise_for_status()
            data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    def generate(self, model: str, prompt: str) -> ModelResult:
        start = time.perf_counter()
        try:
            with httpx.Client(base_url=self._base, timeout=300) as c:
                resp = c.post(
                    "/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                resp.raise_for_status()
                data = resp.json()
            elapsed = time.perf_counter() - start
            return ModelResult(
                model=model,
                response=data.get("response", "").strip(),
                elapsed_seconds=elapsed,
            )
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return ModelResult(
                model=model,
                response="",
                elapsed_seconds=elapsed,
                error=str(exc),
            )
