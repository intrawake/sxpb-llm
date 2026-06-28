"""Shared LLM calling, model definition parsing, and SxPB extraction utilities.

Used by sxpb-game, subllminal, and other projects needing a consistent
OpenAI-compatible API caller with model alias resolution.
"""

from sxpb_llm.api import call_api
from sxpb_llm.async_api import async_call_api
from sxpb_llm.model import (
    ModelConfig,
    load_model_definitions,
    resolve_model,
)
from sxpb_llm.sxpb_parse import get_sxpb_from_markdown, parse_sxpb_answer

__all__ = [
    "call_api",
    "async_call_api",
    "ModelConfig",
    "load_model_definitions",
    "resolve_model",
    "get_sxpb_from_markdown",
    "parse_sxpb_answer",
]
