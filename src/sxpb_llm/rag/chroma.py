"""Minimal ChromaDB v2 HTTP client."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterator


class ChromaError(RuntimeError):
    """A ChromaDB operation failed."""


@dataclass(frozen=True)
class ChromaClient:
    base_url: str
    tenant: str = "default_tenant"
    database: str = "default_database"
    timeout: float | None = 30

    @property
    def database_url(self) -> str:
        return (
            f"{self.base_url.rstrip('/')}/tenants/{self.tenant}"
            f"/databases/{self.database}"
        )

    def request(self, path: str, method: str = "GET", data: dict | None = None) -> Any:
        encoded = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = urllib.request.Request(
            f"{self.database_url}{path}", data=encoded, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                if response.status == 204:
                    return None
                raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else None
        except Exception as exc:
            raise ChromaError(f"{method} {path} failed: {exc}") from exc

    def list_collections(self) -> list[dict]:
        return self.request("/collections") or []

    def get_collection(self, name: str) -> dict | None:
        return next(
            (item for item in self.list_collections() if item.get("name") == name), None
        )

    def ensure_collection(self, name: str, metadata: dict | None = None) -> str:
        existing = self.get_collection(name)
        if existing:
            return str(existing["id"])
        result = self.request(
            "/collections",
            "POST",
            {"name": name, "metadata": metadata or {"hnsw:space": "cosine"}},
        )
        if not isinstance(result, dict) or "id" not in result:
            raise ChromaError(f"could not create collection {name!r}")
        return str(result["id"])

    def collection_id(self, name: str) -> str:
        collection = self.get_collection(name)
        if not collection:
            raise ChromaError(f"collection {name!r} does not exist")
        return str(collection["id"])

    def get(self, collection_id: str, **payload: Any) -> dict:
        return self.request(f"/collections/{collection_id}/get", "POST", payload) or {}

    def iter_get(
        self, collection_id: str, *, batch_size: int = 100, **payload: Any
    ) -> Iterator[dict]:
        offset = 0
        while True:
            page = self.get(collection_id, limit=batch_size, offset=offset, **payload)
            ids = page.get("ids") or []
            if not ids:
                return
            yield page
            if len(ids) < batch_size:
                return
            offset += len(ids)

    def add(self, collection_id: str, payload: dict) -> None:
        self.request(f"/collections/{collection_id}/add", "POST", payload)

    def upsert(self, collection_id: str, payload: dict) -> None:
        self.request(f"/collections/{collection_id}/upsert", "POST", payload)

    def delete(self, collection_id: str, payload: dict) -> None:
        self.request(f"/collections/{collection_id}/delete", "POST", payload)

    def query(self, collection_id: str, payload: dict) -> dict:
        return (
            self.request(f"/collections/{collection_id}/query", "POST", payload) or {}
        )

    def count(self, collection_id: str) -> int:
        result = self.request(f"/collections/{collection_id}/count")
        return int(result or 0)

    def delete_collection(self, name: str) -> None:
        self.request(f"/collections/{name}", "DELETE")
