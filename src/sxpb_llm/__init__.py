"""Shared LLM calling, model definition parsing, and SxPB extraction utilities.

Used by sxpb-game, subllminal, and other projects needing a consistent
OpenAI-compatible API caller with model alias resolution.
"""

from sxpb_llm.api import call_api
from sxpb_llm.block_parse import CodeBlock, CodeBlockError, parse_code_blocks
from sxpb_llm.model import ModelConfig, load_model_definitions, resolve_model
from sxpb_llm.sxpb_parse import get_sxpb_from_markdown, parse_sxpb_answer


# Lazy import to avoid requiring httpx2 unless actually used.
def __getattr__(name: str):
    if name == "async_call_api":
        from sxpb_llm.async_api import async_call_api as _fn

        return _fn
    if name == "async_call_image_api":
        from sxpb_llm.async_api import async_call_image_api as _fn

        return _fn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "call_api",
    "async_call_api",
    "async_call_image_api",
    "ModelConfig",
    "load_model_definitions",
    "resolve_model",
    "get_sxpb_from_markdown",
    "parse_sxpb_answer",
    "CodeBlock",
    "parse_code_blocks",
    "CodeBlockError",
]
