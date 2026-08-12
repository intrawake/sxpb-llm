"""Robust OpenAI-compatible API caller with exponential backoff.

Ported from sxpb-game's eval/utils.py::call_api, generalized for reuse.
"""

import json
import socket
import sys
import time
import urllib.error
import urllib.request


def call_api(
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
    **extra,
):
    """Call an OpenAI-compatible /chat/completions endpoint.

    Args:
        model: Model fullname string (e.g. "aistudio/gemma-4-31b-it").
        prompt: Either a string (wrapped as a user message) or a list of
                message dicts.
        api_url: Base URL of the OpenAI-compatible API (required).
        token_gen_limit: Max tokens to generate (sent as ``max_tokens``), or
                         ``None`` to leave the provider default unspecified.
        token_ctx_limit: Context window size (unused in payload; informational
                         for callers that inspect `return_full`).
        reasoning_effort: Optional reasoning effort hint sent to the API.
        timeout: HTTP timeout in seconds, or ``None`` to wait forever.
        log_file: If given, appends request/response JSON to this file.
        return_full: If True, returns ``(content, payload, response_json)``
                     instead of just the content string.
        api_key: Optional API key sent as ``Authorization: Bearer <key>``.
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

    data = json.dumps(payload).encode("utf-8")
    if not api_url:
        raise ValueError("api_url must be provided to call_api")

    target_url = api_url
    if not target_url.endswith("/chat/completions"):
        if not target_url.endswith("/"):
            target_url += "/"
        target_url += "chat/completions"

    req = urllib.request.Request(
        target_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )

    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                choice = res_data["choices"][0]
                content = choice["message"].get("content") or ""
                content = content.strip()

                if log_file:
                    _log(f"\n--- API Response ---\n{content}")
                if return_full:
                    return content, payload, res_data
                return content

        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8")
            except Exception:
                pass

            if (
                e.code == 429
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
                time.sleep(sleep_time)

            elif (
                e.code in [500, 502, 503, 504]
                or "timeout" in str(e).lower()
                or "timeout" in err_body.lower()
            ):
                sys.stdout.write(
                    f"Server Error/Timeout ({e.code}) on attempt {attempt + 1}. "
                    "Retrying...\n"
                )
                sys.stdout.flush()
                if log_file:
                    _log(f"Server Error ({e.code}): {err_body}")
                time.sleep(2)

            else:
                sys.stderr.write(
                    f"API Error ({e.code}): {e.reason}\nBody: {err_body}\n"
                )
                break

        except urllib.error.URLError as e:
            if isinstance(e.reason, socket.timeout) or "timeout" in str(e).lower():
                sys.stdout.write(f"URL Timeout on attempt {attempt + 1}. Retrying...\n")
                sys.stdout.flush()
                if log_file:
                    _log(f"URL Timeout on attempt {attempt + 1}. Retrying...")
                time.sleep(2)
            else:
                sys.stderr.write(f"URL Error: {e}\n")
                break

        except TimeoutError:
            sys.stdout.write(f"Timeout Error on attempt {attempt + 1}. Retrying...\n")
            sys.stdout.flush()
            if log_file:
                _log(f"Timeout Error on attempt {attempt + 1}. Retrying...")
            time.sleep(2)

        except Exception as e:
            sys.stderr.write(f"Request Error: {e}\n")
            break

    if return_full:
        return None, None, None
    return None
