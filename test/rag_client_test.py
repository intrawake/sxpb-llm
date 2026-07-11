import json
import urllib.request
from unittest.mock import patch

import pytest

from sxpb_llm.rag.chroma import ChromaClient, ChromaError
from sxpb_llm.rag.embed import EmbeddingClient, EmbeddingError


class FakeResponse:
    def __init__(self, body, status: int = 200):
        self.body = json.dumps(body).encode()
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return self.body


def test_embedding_client_orders_vectors_and_sends_authorization():
    body = {
        "data": [
            {"index": 1, "embedding": [2.0]},
            {"index": 0, "embedding": [1.0]},
        ]
    }
    with patch.object(
        urllib.request, "urlopen", return_value=FakeResponse(body)
    ) as urlopen:
        vectors = EmbeddingClient(
            "https://example.test/v1",
            "embed-model",
            api_key="secret",
        ).embed(["one", "two"])

    assert vectors == [[1.0], [2.0]]
    request = urlopen.call_args.args[0]
    assert request.full_url == "https://example.test/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer secret"
    assert json.loads(request.data) == {
        "input": ["one", "two"],
        "model": "embed-model",
    }


def test_embedding_client_rejects_incomplete_indices():
    body = {"data": [{"index": 1, "embedding": [2.0]}]}
    with patch.object(urllib.request, "urlopen", return_value=FakeResponse(body)):
        with pytest.raises(EmbeddingError, match="indices"):
            EmbeddingClient("https://example.test", "model").embed(["one"])


def test_chroma_client_builds_v2_database_url_and_creates_collection():
    responses = [
        FakeResponse([]),
        FakeResponse({"id": "new-id", "name": "docs"}),
    ]
    with patch.object(urllib.request, "urlopen", side_effect=responses) as urlopen:
        collection_id = ChromaClient(
            "https://chroma.test/api/v2", tenant="tenant", database="database"
        ).ensure_collection("docs")

    assert collection_id == "new-id"
    first_request = urlopen.call_args_list[0].args[0]
    second_request = urlopen.call_args_list[1].args[0]
    expected = (
        "https://chroma.test/api/v2/tenants/tenant/databases/database/collections"
    )
    assert first_request.full_url == expected
    assert second_request.full_url == expected
    assert json.loads(second_request.data)["name"] == "docs"


def test_chroma_client_wraps_transport_errors():
    with patch.object(urllib.request, "urlopen", side_effect=OSError("offline")):
        with pytest.raises(ChromaError, match="offline"):
            ChromaClient("https://chroma.test/api/v2").list_collections()
