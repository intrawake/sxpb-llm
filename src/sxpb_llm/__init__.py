"""Shared LLM calling, model definition parsing, and SxPB extraction utilities.

Submodules are loaded lazily so independent facilities such as ``sxpb_llm.rag``
do not import parser or async dependencies they do not use.
"""

from importlib import import_module
from typing import Any

_LAZY_ATTRS = {
    "call_api": ("sxpb_llm.api", "call_api"),
    "async_call_api": ("sxpb_llm.async_api", "async_call_api"),
    "async_call_image_api": ("sxpb_llm.async_api", "async_call_image_api"),
    "CodeBlock": ("sxpb_llm.block_parse", "CodeBlock"),
    "CodeBlockError": ("sxpb_llm.block_parse", "CodeBlockError"),
    "parse_code_blocks": ("sxpb_llm.block_parse", "parse_code_blocks"),
    "ModelConfig": ("sxpb_llm.model", "ModelConfig"),
    "load_model_definitions": ("sxpb_llm.model", "load_model_definitions"),
    "resolve_model": ("sxpb_llm.model", "resolve_model"),
    "get_sxpb_from_markdown": ("sxpb_llm.sxpb_parse", "get_sxpb_from_markdown"),
    "parse_sxpb_answer": ("sxpb_llm.sxpb_parse", "parse_sxpb_answer"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = list(_LAZY_ATTRS)
