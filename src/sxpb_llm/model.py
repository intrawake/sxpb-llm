"""Model definition loading and alias resolution.

Parses ``model_by_name.sxpb`` files and resolves short aliases to full
ModelConfig instances with provider-specific overrides.

Format
------
A ``model_by_name.sxpb`` file looks like::

    ()
    (example-chat
     (fullname provider/example-chat)
    )
    (example-reasoning
     (fullname provider/example-reasoning)
     (token_gen_limit 4096)
    )

Every alias record is a dict with at least ``fullname``.
Optional fields:

* ``token_gen_limit``: int — max tokens to generate.
* ``token_ctx_limit``: int — context window size (informational).
* ``reasoning_effort``: str — passed through to the API.
* ``timeout``: int — HTTP timeout in seconds (0 = wait forever).
* Any other key is passed through as ``extra`` kwargs to ``call_api``.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import sxpb


@dataclass
class ModelConfig:
    """Resolved model configuration."""

    fullname: str
    """The full model identifier sent to the API (e.g. ``aistudio/gemma-4-31b-it``)."""

    token_ctx_limit: int = 128000
    """Context window size (informational, not sent in payload)."""

    token_gen_limit: int = 16384
    """Max tokens to generate (sent as ``max_tokens``)."""

    reasoning_effort: str | None = None
    """Reasoning effort hint sent to the API."""

    timeout: int = 0
    """HTTP timeout in seconds. 0 means wait forever."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Additional keyword arguments passed through to ``call_api``."""


def load_model_definitions(source: str | Path) -> dict[str, ModelConfig]:
    """Parse a ``model_by_name.sxpb`` file or SxPB string into a dict of aliases.

    Args:
        source: A file path (str or Path) or a raw SxPB string that starts
                with ``(``.

    Returns:
        A dict mapping alias names to ``ModelConfig`` instances.  Returns an
        empty dict if the source cannot be parsed.
    """
    if isinstance(source, Path):
        source = str(source)

    parsed: Any = {}
    if source.strip().startswith("("):
        parsed = sxpb.loads(source)
    else:
        try:
            parsed = sxpb.loads(Path(source).read_text())
        except (FileNotFoundError, OSError):
            return {}

    if not isinstance(parsed, dict):
        return {}
    raw: dict = cast(dict, parsed)

    result: dict[str, ModelConfig] = {}
    for alias, entry in raw.items():
        alias = str(alias)
        if isinstance(entry, str):
            result[alias] = ModelConfig(fullname=entry)
        elif isinstance(entry, dict):
            entry = dict(entry)
            fullname = str(entry.pop("fullname", alias))
            config = ModelConfig(
                fullname=fullname,
                token_ctx_limit=int(entry.pop("token_ctx_limit", 128000)),
                token_gen_limit=int(entry.pop("token_gen_limit", 16384)),
                reasoning_effort=entry.pop("reasoning_effort", None),
                timeout=int(entry.pop("timeout", 0)),
                extra=entry,  # remaining keys become extra kwargs
            )
            result[alias] = config
    return result


def resolve_model(
    alias: str,
    definitions: dict[str, ModelConfig] | None = None,
    *,
    token_ctx_limit: int | None = None,
    token_gen_limit: int | None = None,
    reasoning_effort: str | None = None,
    timeout: int | None = None,
    **extra,
) -> ModelConfig:
    """Resolve a model alias to a ``ModelConfig``.

    If the alias is found in *definitions*, its values are used as defaults
    and then overridden by any explicit keyword arguments.  If the alias is
    *not* found, the alias string itself is used as the ``fullname``.

    Args:
        alias: Short name (e.g. ``"dono-gemma4-31b"``) or full model string.
        definitions: Dict from ``load_model_definitions``.
        token_ctx_limit: Override context window size.
        token_gen_limit: Override generation token limit.
        reasoning_effort: Override reasoning effort.
        timeout: Override HTTP timeout.
        **extra: Additional overrides merged into ``ModelConfig.extra``.

    Returns:
        A resolved ``ModelConfig``.
    """
    base = ModelConfig(fullname=alias)

    if definitions and alias in definitions:
        base = definitions[alias]

    overrides: dict[str, Any] = {}
    if token_ctx_limit is not None:
        overrides["token_ctx_limit"] = token_ctx_limit
    if token_gen_limit is not None:
        overrides["token_gen_limit"] = token_gen_limit
    if reasoning_effort is not None:
        overrides["reasoning_effort"] = reasoning_effort
    if timeout is not None:
        overrides["timeout"] = timeout

    merged_extra = dict(base.extra)
    merged_extra.update(extra)

    return ModelConfig(
        fullname=base.fullname,
        token_ctx_limit=overrides.get("token_ctx_limit", base.token_ctx_limit),
        token_gen_limit=overrides.get("token_gen_limit", base.token_gen_limit),
        reasoning_effort=overrides.get("reasoning_effort", base.reasoning_effort),
        timeout=overrides.get("timeout", base.timeout),
        extra=merged_extra,
    )
