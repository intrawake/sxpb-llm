"""OpenAI-compatible embedding client."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any


class EmbeddingError(RuntimeError):
    """An embedding request failed or returned malformed data."""


@dataclass(frozen=True)
class EmbeddingClient:
    api_url: str
    model: str
    batch_size: int = 10
    timeout: float | None = None
    api_key: str | None = None

    def _target_url(self) -> str:
        url = self.api_url.rstrip("/")
        return url if url.endswith("/embeddings") else f"{url}/embeddings"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in batches and return vectors in input order."""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"input": texts, "model": self.model}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self._target_url(), data=payload, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body: Any = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
        try:
            items = sorted(body["data"], key=lambda item: item["index"])
            if [item["index"] for item in items] != list(range(len(texts))):
                raise ValueError("response indices do not match request")
            vectors = [item["embedding"] for item in items]
            if not all(isinstance(vector, list) and vector for vector in vectors):
                raise ValueError("empty or invalid embedding vector")
            return vectors
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(f"malformed embedding response: {exc}") from exc
