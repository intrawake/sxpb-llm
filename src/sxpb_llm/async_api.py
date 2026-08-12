"""Async OpenAI-compatible API caller with exponential backoff.

Mirrors ``api.py::call_api`` but uses ``httpx2`` for async HTTP and
``await asyncio.sleep()`` for non-blocking retries.

Requires ``sxpb-llm[async]`` (installs ``httpx2``).
"""

import asyncio
import base64
import json
import sys

import httpx2 as httpx


_HTTPX_IMPORT_ERROR_MSG = (
    "httpx2 is required for async_call_api.  Install with: pip install sxpb-llm[async]"
)


async def async_call_api(
    model: str,
    prompt,
    *,
    api_url: str,
    token_gen_limit: int | None = None,
    token_ctx_limit: int | None = None,
    reasoning_effort: str | None = None,
    timeout: int | None = 0,
    log_file: str | None = None,
    return_full: bool = False,
    api_key: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    **extra,
):
    """Call an OpenAI-compatible /chat/completions endpoint — async.

    Args:
        model: Model fullname string (e.g. ``"aistudio/gemma-4-31b-it"``).
        prompt: Either a string (wrapped as a user message) or a list of
                message dicts.
        api_url: Base URL of the OpenAI-compatible API (required).
        token_gen_limit: Max tokens to generate (sent as ``max_tokens``), or
                         ``None`` to leave the provider default unspecified.
        token_ctx_limit: Context window size (unused in payload; informational
                         for callers that inspect ``return_full``).
        reasoning_effort: Optional reasoning effort hint sent to the API.
        timeout: HTTP timeout in seconds, or ``None`` to wait forever.
        log_file: If given, appends request/response JSON to this file.
        return_full: If True, returns ``(content, payload, response_json)``
                     instead of just the content string.
        api_key: Optional API key sent as ``Authorization: Bearer <key>``.
        httpx_client: Optional shared ``httpx.AsyncClient`` for connection
                      reuse.  If not given, a temporary client is created.
        **extra: Additional key-value pairs passed through to the API payload.

    Returns:
        The response content string, or a 3-tuple if ``return_full`` is True.
        Returns ``None`` (or ``(None, None, None)``) on all failures.
    """
    if timeout is not None and timeout <= 0:
        timeout = None

    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]

    payload: dict = {
        "model": model,
        "messages": messages,
    }
    if token_gen_limit is not None and token_gen_limit > 0:
        payload["max_tokens"] = token_gen_limit
    payload.update(extra)

    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort

    def _log(text: str) -> None:
        if not log_file:
            return
        try:
            with open(log_file, "a") as f:
                f.write(text + "\n")
        except Exception as e:
            sys.stderr.write(f"Logging Error: {e}\n")

    if log_file:
        _log(f"\n--- API Request ---\n{json.dumps(payload, indent=2)}")

    if not api_url:
        raise ValueError("api_url must be provided to async_call_api")

    target_url = api_url
    if not target_url.endswith("/chat/completions"):
        if not target_url.endswith("/"):
            target_url += "/"
        target_url += "chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _request_once(client: httpx.AsyncClient):
        response = await client.post(
            target_url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout) if timeout is not None else None,
        )
        response.raise_for_status()
        return response.json()

    async def _run_retries(client: httpx.AsyncClient):
        for attempt in range(5):
            try:
                res_data = await _request_once(client)
                choice = res_data["choices"][0]
                content = choice["message"].get("content") or ""
                content = content.strip()

                if log_file:
                    _log(f"\n--- API Response ---\n{content}")
                if return_full:
                    return content, payload, res_data
                return content

            except httpx.HTTPStatusError as e:
                err_body = e.response.text[:500] if e.response else ""
                status = e.response.status_code if e.response else 0

                if (
                    status == 429
                    or "too many requests" in err_body.lower()
                    or "rate limit" in err_body.lower()
                ):
                    if attempt == 4:
                        sys.stderr.write("Rate limited on final attempt. Giving up.\n")
                        break
                    sleep_time = 32 if attempt < 2 else 64
                    sys.stdout.write(
                        f"Rate limited (429) on attempt {attempt + 1}. "
                        f"Retrying in {sleep_time}s...\n"
                    )
                    sys.stdout.flush()
                    if log_file:
                        _log(f"Rate limited: {err_body}")
                    await asyncio.sleep(sleep_time)

                elif status in (500, 502, 503, 504) or "timeout" in err_body.lower():
                    sys.stdout.write(
                        f"Server Error/Timeout ({status}) on attempt "
                        f"{attempt + 1}. Retrying...\n"
                    )
                    sys.stdout.flush()
                    if log_file:
                        _log(f"Server Error ({status}): {err_body}")
                    await asyncio.sleep(2)

                else:
                    sys.stderr.write(
                        f"API Error ({status}): {e.response.reason_phrase}\n"
                        f"Body: {err_body}\n"
                    )
                    break

            except httpx.TimeoutException:
                sys.stdout.write(f"Timeout on attempt {attempt + 1}. Retrying...\n")
                sys.stdout.flush()
                if log_file:
                    _log(f"Timeout on attempt {attempt + 1}. Retrying...")
                await asyncio.sleep(2)

            except httpx.RequestError as e:
                sys.stderr.write(f"Request Error: {e}\n")
                break

            except Exception as e:
                sys.stderr.write(f"Request Error: {e}\n")
                break

        if return_full:
            return None, None, None
        return None

    close_client = httpx_client is None
    client = httpx_client or httpx.AsyncClient()
    try:
        return await _run_retries(client)
    finally:
        if close_client:
            await client.aclose()


async def async_call_image_api(
    model: str,
    prompt: str,
    *,
    api_url: str,
    n: int = 1,
    timeout: int | None = 0,
    log_file: str | None = None,
    api_key: str | None = None,
    httpx_client: httpx.AsyncClient | None = None,
    **extra,
) -> str | None:
    """Call an OpenAI-compatible /images/generations endpoint — async.

    Args:
        model: Model fullname string.
        prompt: Image generation prompt string.
        api_url: Base URL of the OpenAI-compatible API (required).
        n: Number of images to generate (default 1; only the first is returned).
        timeout: HTTP timeout in seconds, or ``None`` to wait forever.
        log_file: If given, appends request/response JSON to this file.
        api_key: Optional API key sent as ``Authorization: Bearer <key>``.
        httpx_client: Optional shared ``httpx.AsyncClient`` for connection
                      reuse.  If not given, a temporary client is created.
        **extra: Additional key-value pairs passed through to the API payload.

    Returns:
        A raw base64-encoded image string, or ``None`` on failure.
    """
    if timeout is not None and timeout <= 0:
        timeout = None

    payload: dict = {
        "model": model,
        "prompt": prompt,
        "n": n,
        "response_format": "b64_json",
    }
    payload.update(extra)

    def _log(text: str) -> None:
        if not log_file:
            return
        try:
            with open(log_file, "a") as f:
                f.write(text + "\n")
        except Exception as e:
            sys.stderr.write(f"Logging Error: {e}\n")

    if log_file:
        _log(f"\n--- Image API Request ---\n{json.dumps(payload, indent=2)}")

    if not api_url:
        raise ValueError("api_url must be provided to async_call_image_api")

    target_url = api_url
    if not target_url.endswith("/images/generations"):
        if not target_url.endswith("/"):
            target_url += "/"
        target_url += "images/generations"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def _request_once(client: httpx.AsyncClient):
        response = await client.post(
            target_url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout) if timeout is not None else None,
        )
        response.raise_for_status()
        return response.json()

    async def _extract_image(res_data: dict, client: httpx.AsyncClient) -> str | None:
        """Extract base64 image from response, fetching URL if needed."""
        if "data" not in res_data or not res_data["data"]:
            return None
        item = res_data["data"][0]
        if item.get("b64_json"):
            return item["b64_json"]
        if item.get("url"):
            img_resp = await client.get(item["url"])
            img_resp.raise_for_status()
            return base64.b64encode(img_resp.content).decode("utf-8")
        return None

    close_client = httpx_client is None
    client = httpx_client or httpx.AsyncClient()

    try:
        for attempt in range(5):
            try:
                res_data = await _request_once(client)
                result = await _extract_image(res_data, client)
                if log_file:
                    _log(
                        f"\n--- Image API Response ---\n"
                        f"{'(image)' if result else 'no image data'}"
                    )
                return result

            except httpx.HTTPStatusError as e:
                err_body = e.response.text[:500] if e.response else ""
                status = e.response.status_code if e.response else 0

                if (
                    status == 429
                    or "too many requests" in err_body.lower()
                    or "rate limit" in err_body.lower()
                ):
                    if attempt == 4:
                        sys.stderr.write(
                            "Image API rate limited on final attempt. Giving up.\n"
                        )
                        break
                    sleep_time = 32 if attempt < 2 else 64
                    sys.stdout.write(
                        f"Image API rate limited (429) on attempt {attempt + 1}. "
                        f"Retrying in {sleep_time}s...\n"
                    )
                    sys.stdout.flush()
                    if log_file:
                        _log(f"Rate limited: {err_body}")
                    await asyncio.sleep(sleep_time)

                elif status in (500, 502, 503, 504) or "timeout" in err_body.lower():
                    sys.stdout.write(
                        f"Image API Server Error/Timeout ({status}) on attempt "
                        f"{attempt + 1}. Retrying...\n"
                    )
                    sys.stdout.flush()
                    if log_file:
                        _log(f"Server Error ({status}): {err_body}")
                    await asyncio.sleep(2)

                else:
                    sys.stderr.write(
                        f"Image API Error ({status}): "
                        f"{e.response.reason_phrase}\n"
                        f"Body: {err_body}\n"
                    )
                    break

            except httpx.TimeoutException:
                sys.stdout.write(
                    f"Image API Timeout on attempt {attempt + 1}. Retrying...\n"
                )
                sys.stdout.flush()
                if log_file:
                    _log(f"Timeout on attempt {attempt + 1}. Retrying...")
                await asyncio.sleep(2)

            except httpx.RequestError as e:
                sys.stderr.write(f"Image API Request Error: {e}\n")
                break

            except Exception as e:
                sys.stderr.write(f"Image API Request Error: {e}\n")
                break

        return None

    finally:
        if close_client:
            await client.aclose()
