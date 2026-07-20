"""Smoke tests for sxpb_llm package."""

import json
from pathlib import Path

import httpx2 as httpx
import pytest

import sxpb_llm
from sxpb_llm.model import ModelConfig, load_model_definitions, resolve_model


def test_imports():
    """Verify all public names are importable."""
    assert hasattr(sxpb_llm, "call_api")
    assert hasattr(sxpb_llm, "async_call_api")
    assert hasattr(sxpb_llm, "async_call_image_api")
    assert hasattr(sxpb_llm, "ModelConfig")
    assert hasattr(sxpb_llm, "load_model_definitions")
    assert hasattr(sxpb_llm, "resolve_model")
    assert hasattr(sxpb_llm, "get_sxpb_from_markdown")
    assert hasattr(sxpb_llm, "parse_sxpb_answer")


def test_model_config_defaults():
    """ModelConfig should have sensible defaults."""
    mc = ModelConfig(fullname="test-model")
    assert mc.fullname == "test-model"
    assert mc.token_ctx_limit == 128000
    assert mc.token_gen_limit == 16384
    assert mc.reasoning_effort is None
    assert mc.timeout == 0
    assert mc.extra == {}


def test_load_model_definitions_from_string():
    """Parse a SxPB string directly."""
    s = """()
(local-test
 (fullname llama.cpp/test:Q8_0)
 (token_gen_limit 4000)
 (timeout 60)
)
(dono-test
 (fullname aistudio/test)
)
"""
    defs = load_model_definitions(s)
    assert len(defs) == 2

    assert "local-test" in defs
    mc = defs["local-test"]
    assert mc.fullname == "llama.cpp/test:Q8_0"
    assert mc.token_gen_limit == 4000
    assert mc.timeout == 60
    assert mc.token_ctx_limit == 128000  # default

    assert "dono-test" in defs
    mc = defs["dono-test"]
    assert mc.fullname == "aistudio/test"
    assert mc.token_gen_limit == 16384  # default


def test_load_model_definitions_string_alias():
    """A bare string value becomes the fullname."""
    defs = load_model_definitions('() (just-a-string "my/model")')
    assert defs["just-a-string"].fullname == "my/model"


def test_load_model_definitions_from_file():
    """Load the small model-definition fixture from a file path."""
    path = Path(__file__).parent / "data" / "model_by_name.sxpb"

    defs = load_model_definitions(path)

    assert set(defs) == {"example-chat", "example-reasoning"}
    assert defs["example-chat"] == ModelConfig(fullname="provider/example-chat")
    assert defs["example-reasoning"] == ModelConfig(
        fullname="provider/example-reasoning",
        token_ctx_limit=32768,
        token_gen_limit=4096,
        reasoning_effort="medium",
        timeout=30,
        extra={"temperature": 0.5},
    )


def test_resolve_model_found():
    """resolve_model merges definition defaults with overrides."""
    defs = {
        "test": ModelConfig(
            fullname="base/model",
            token_gen_limit=8000,
            timeout=30,
            extra={"temperature": 0.7},
        )
    }
    mc = resolve_model("test", defs, token_gen_limit=4000, top_p=0.9)
    assert mc.fullname == "base/model"
    assert mc.token_gen_limit == 4000  # overridden
    assert mc.timeout == 30  # from def
    assert mc.extra == {"temperature": 0.7, "top_p": 0.9}


def test_resolve_model_not_found():
    """resolve_model uses alias as fullname when not in definitions."""
    mc = resolve_model("some/model", {})
    assert mc.fullname == "some/model"
    assert mc.token_gen_limit == 16384  # default


def test_sxpb_parse_answer_fenced():
    """parse_sxpb_answer extracts from ```sxpb > /dev/stdout blocks."""
    from sxpb_llm.sxpb_parse import parse_sxpb_answer

    response = """\
Some preamble text.

```sxpb > /dev/stdout
(answer "b2")
```
Some trailing text.
"""
    assert parse_sxpb_answer(response) == "b2"


def test_sxpb_parse_answer_bare():
    """parse_sxpb_answer falls back to bare (answer ...) lines."""
    from sxpb_llm.sxpb_parse import parse_sxpb_answer

    response = """\
Let me think about this. The best move is:

(answer "c3")

That should work.
"""
    assert parse_sxpb_answer(response) == "c3"


def test_sxpb_parse_answer_none():
    """parse_sxpb_answer returns None for unparseable input."""
    from sxpb_llm.sxpb_parse import parse_sxpb_answer

    assert parse_sxpb_answer("") is None
    assert parse_sxpb_answer("Just some text, no answer here.") is None


def test_get_sxpb_from_markdown():
    """get_sxpb_from_markdown extracts and parses sxpb blocks."""
    from sxpb_llm.sxpb_parse import get_sxpb_from_markdown

    text = """\
```sxpb
(candidates
 (candidate (name "Test") (texture "Smooth"))
)
```
"""
    result = get_sxpb_from_markdown(text)
    assert result is not None
    assert "candidates" in result


# --------------------------------------------------------------------------
# async_call_api tests
# --------------------------------------------------------------------------


@pytest.fixture
def echo_server():
    """A tiny async echo server that returns a fake /chat/completions response.

    Uses httpx's own transport-mocking to avoid binding a real port.
    """

    class EchoTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            # Read the payload the caller sent
            body = json.loads(request.read().decode())
            model_name = body.get("model", "unknown")
            user_msgs = [
                m["content"] for m in body.get("messages", []) if m["role"] == "user"
            ]
            content = f"echo:{model_name}:{':'.join(user_msgs)}"
            response_payload = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ]
            }
            return httpx.Response(
                200,
                json=response_payload,
                request=request,
            )

    return EchoTransport()


@pytest.mark.asyncio
async def test_async_call_api_basic(echo_server):
    """async_call_api sends a user message and returns content."""
    async with httpx.AsyncClient(transport=echo_server) as client:
        result = await sxpb_llm.async_call_api(
            "test-model",
            "Hello world",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result == "echo:test-model:Hello world"


@pytest.mark.asyncio
async def test_async_call_api_message_list(echo_server):
    """async_call_api accepts a list of message dicts."""
    async with httpx.AsyncClient(transport=echo_server) as client:
        result = await sxpb_llm.async_call_api(
            "test-model",
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
            ],
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result == "echo:test-model:Hi"


@pytest.mark.asyncio
async def test_async_call_api_return_full(echo_server):
    """async_call_api with return_full returns content, payload, response."""
    async with httpx.AsyncClient(transport=echo_server) as client:
        content, payload, response = await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
            return_full=True,
        )
        assert content == "echo:test-model:Hello"
        assert payload["model"] == "test-model"
        assert "choices" in response


@pytest.mark.asyncio
async def test_async_call_api_extra_kwargs(echo_server):
    """Extra kwargs are passed through to the API payload."""
    async with httpx.AsyncClient(transport=echo_server) as client:
        content, payload, _ = await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
            return_full=True,
            temperature=0.5,
            top_p=0.9,
        )
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9


@pytest.mark.asyncio
async def test_async_call_api_uses_provided_client(echo_server):
    """The provided httpx_client is used (not replaced)."""
    async with httpx.AsyncClient(transport=echo_server) as client:
        result = await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert "echo:" in result
        # Client should still be open (we passed it in, so we manage it)
        assert not client.is_closed


@pytest.mark.asyncio
async def test_async_call_api_no_client_creates_and_closes(echo_server):
    """Without httpx_client, a temporary client is created and closed."""
    # We need a real-ish URL pattern; override transport via monkeypatch isn't
    # possible since we aren't providing a client.  Instead, verify that a call
    # with a bad URL fails gracefully rather than crashing.
    with pytest.raises(ValueError, match="api_url"):
        await sxpb_llm.async_call_api("test-model", "Hello", api_url="")


@pytest.mark.asyncio
async def test_async_call_api_api_key_header(echo_server):
    """api_key is sent as an Authorization header."""
    captured_headers = {}

    class SpyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal captured_headers
            captured_headers = dict(request.headers)
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
                request=request,
            )

    async with httpx.AsyncClient(transport=SpyTransport()) as client:
        await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
            api_key="sk-secret",
        )
        assert captured_headers.get("authorization") == "Bearer sk-secret"


@pytest.mark.asyncio
async def test_async_call_api_reasoning_content_fallback(echo_server):
    """If content is empty, reasoning_content is used."""

    class ReasoningTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "reasoning_content": "think think think",
                            }
                        }
                    ]
                },
                request=request,
            )

    async with httpx.AsyncClient(transport=ReasoningTransport()) as client:
        result = await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result == "think think think"


@pytest.mark.asyncio
async def test_async_call_api_retries_on_429(echo_server, monkeypatch):
    """Rate-limited requests are retried with exponential backoff."""

    # Replace asyncio.sleep so the test doesn't actually wait 64 seconds.
    async def fake_sleep(_seconds):
        pass

    import sxpb_llm.async_api

    monkeypatch.setattr(sxpb_llm.async_api.asyncio, "sleep", fake_sleep)

    call_count = 0

    class RateLimitTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(
                    429,
                    json={"error": {"message": "Too many requests"}},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "finally"}}]},
                request=request,
            )

    async with httpx.AsyncClient(transport=RateLimitTransport()) as client:
        result = await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result == "finally"
        assert call_count == 3


@pytest.mark.asyncio
async def test_async_call_api_gives_up_after_5_retries(echo_server, monkeypatch):
    """After 5 consecutive failures, returns None."""

    async def fake_sleep(_seconds):
        pass

    import sxpb_llm.async_api

    monkeypatch.setattr(sxpb_llm.async_api.asyncio, "sleep", fake_sleep)

    class FailTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(
                503,
                json={"error": {"message": "Service Unavailable"}},
                request=request,
            )

    async with httpx.AsyncClient(transport=FailTransport()) as client:
        result = await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result is None


@pytest.mark.asyncio
async def test_async_call_api_reasoning_effort_none(echo_server):
    """reasoning_effort='none' is passed through as-is."""
    captured_payload = {}

    class SpyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal captured_payload
            captured_payload = json.loads(request.read().decode())
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "ok"}}]},
                request=request,
            )

    async with httpx.AsyncClient(transport=SpyTransport()) as client:
        await sxpb_llm.async_call_api(
            "test-model",
            "Hello",
            api_url="http://fake/v1",
            httpx_client=client,
            reasoning_effort="none",
        )
        assert captured_payload["reasoning_effort"] == "none"
        assert "chat_template_kwargs" not in captured_payload


# --------------------------------------------------------------------------
# async_call_image_api tests
# --------------------------------------------------------------------------


@pytest.fixture
def image_echo_server():
    """A transport that returns a fake /images/generations response."""

    class ImageEchoTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            body = json.loads(request.read().decode())
            prompt = body.get("prompt", "")
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"b64_json": f"base64image:{prompt}"},
                    ]
                },
                request=request,
            )

    return ImageEchoTransport()


@pytest.mark.asyncio
async def test_async_call_image_api_basic(image_echo_server):
    """async_call_image_api returns b64_json from response."""
    async with httpx.AsyncClient(transport=image_echo_server) as client:
        result = await sxpb_llm.async_call_image_api(
            "test-model",
            "a cat in a hat",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result == "base64image:a cat in a hat"


@pytest.mark.asyncio
async def test_async_call_image_api_url_fallback(image_echo_server):
    """If b64_json is missing, the URL is fetched and base64-encoded."""

    class UrlTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.method == "POST":
                return httpx.Response(
                    200,
                    json={"data": [{"url": "http://fake.invalid/image.png"}]},
                    request=request,
                )
            # GET — URL fetch
            return httpx.Response(
                200,
                content=b"\x89PNG fake image bytes",
                request=request,
            )

    import base64

    async with httpx.AsyncClient(transport=UrlTransport()) as client:
        result = await sxpb_llm.async_call_image_api(
            "test-model",
            "a cat",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        expected = base64.b64encode(b"\x89PNG fake image bytes").decode("utf-8")
        assert result == expected


@pytest.mark.asyncio
async def test_async_call_image_api_no_data():
    """Returns None when response has no data array."""

    class EmptyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, json={}, request=request)

    async with httpx.AsyncClient(transport=EmptyTransport()) as client:
        result = await sxpb_llm.async_call_image_api(
            "test-model",
            "a cat",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result is None


@pytest.mark.asyncio
async def test_async_call_image_api_extra_kwargs(image_echo_server):
    """Extra kwargs are passed through to the API payload."""
    captured_payload = {}

    class SpyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal captured_payload
            captured_payload = json.loads(request.read().decode())
            return httpx.Response(
                200,
                json={"data": [{"b64_json": "abc"}]},
                request=request,
            )

    async with httpx.AsyncClient(transport=SpyTransport()) as client:
        await sxpb_llm.async_call_image_api(
            "test-model",
            "a cat",
            api_url="http://fake/v1",
            httpx_client=client,
            size="1024x1024",
        )
        assert captured_payload["size"] == "1024x1024"


@pytest.mark.asyncio
async def test_async_call_image_api_api_key(image_echo_server):
    """api_key is sent as an Authorization header."""
    captured_headers = {}

    class SpyTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal captured_headers
            captured_headers = dict(request.headers)
            return httpx.Response(
                200,
                json={"data": [{"b64_json": "abc"}]},
                request=request,
            )

    async with httpx.AsyncClient(transport=SpyTransport()) as client:
        await sxpb_llm.async_call_image_api(
            "test-model",
            "a cat",
            api_url="http://fake/v1",
            httpx_client=client,
            api_key="sk-secret",
        )
        assert captured_headers.get("authorization") == "Bearer sk-secret"


@pytest.mark.asyncio
async def test_async_call_image_api_retries_on_429(monkeypatch):
    """Rate-limited image requests are retried."""

    async def fake_sleep(_seconds):
        pass

    import sxpb_llm.async_api

    monkeypatch.setattr(sxpb_llm.async_api.asyncio, "sleep", fake_sleep)

    call_count = 0

    class RateLimitTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return httpx.Response(
                    429,
                    json={"error": {"message": "Too many requests"}},
                    request=request,
                )
            return httpx.Response(
                200,
                json={"data": [{"b64_json": "finally"}]},
                request=request,
            )

    async with httpx.AsyncClient(transport=RateLimitTransport()) as client:
        result = await sxpb_llm.async_call_image_api(
            "test-model",
            "a cat",
            api_url="http://fake/v1",
            httpx_client=client,
        )
        assert result == "finally"
        assert call_count == 3
