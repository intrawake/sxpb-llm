"""Extract SxPB content from LLM responses.

Merges the two complementary approaches from sxpb-game and subllminal:

1. **Regex-first** (sxpb-game): Looks for ```sxpb > /dev/stdout``` fenced
   blocks — the exact format sxpb-game's harness instructs LLMs to use.
2. **Line-growth fallback** (subllminal): Scans for bare ``(answer ...)``
   lines and grows them until a valid SxPB parse is found.
"""

import re
from typing import Any, cast

import sxpb


def get_sxpb_from_markdown(text: str) -> dict | None:
    """Extract content from a ```sxpb ... ``` code block and parse it.

    If no ``sxpb``-tagged block is found, falls back to the first untagged
    ``` block.

    Returns:
        The parsed SxPB dict, or ``None`` if no block was found or parsing
        failed.
    """
    if not text:
        return None

    start_marker = "```sxpb"
    end_marker = "```"

    start_idx = text.find(start_marker)
    if start_idx == -1:
        start_marker = "```"
        start_idx = text.find(start_marker)
        if start_idx == -1:
            return None

    content_start = start_idx + len(start_marker)
    end_idx = text.find(end_marker, content_start)

    if end_idx == -1:
        content = text[content_start:].strip()
    else:
        content = text[content_start:end_idx].strip()

    if not content:
        return None

    try:
        result: Any = sxpb.loads(content)
        return cast(dict, result) if isinstance(result, dict) else None
    except Exception:
        return None


def parse_sxpb_answer(response: str) -> str | None:
    """Parse an ``(answer "...")`` value from an LLM response.

    Strategy (tries in order, returns first success):

    1. **Fenced block** (sxpb-game format):
       Searches for ```sxpb > /dev/stdout ... ``` blocks and looks for an
       ``(answer ...)`` record inside.

    2. **Line-growth** (subllminal format):
       Finds lines starting with ``(answer`` and grows them until a valid
       SxPB parse yields an ``answer`` key.

    Returns:
        The answer string, or ``None`` if nothing could be parsed.
    """
    if not response:
        return None

    # ── Strategy 1: fenced sxpb > /dev/stdout blocks ───────────────────
    matches = list(
        re.finditer(
            r"(?:^|\n)[ \t]*```[ \t]*sxpb[ \t]*>[ \t]*/dev/stdout[ \t]*\r?\n"
            r"(.*?)\r?\n[ \t]*```[ \t]*(?:\r?\n|$)",
            response,
            re.DOTALL,
        )
    )
    if matches:
        block_content = matches[-1].group(1)
        try:
            parsed_block = sxpb.loads(block_content)
            if isinstance(parsed_block, dict) and "answer" in parsed_block:
                answer = parsed_block["answer"]
                if answer and str(answer).strip():
                    return str(answer).strip()
        except Exception:
            pass

    # ── Strategy 2: bare (answer ...) line-growth ──────────────────────
    lines = response.splitlines()
    candidate_indices = [
        i
        for i, line in enumerate(lines)
        if line.startswith("(answer ") or line.startswith('(answer"')
    ]

    for start_idx in reversed(candidate_indices):
        for end_idx in range(start_idx, len(lines)):
            if not lines[end_idx].strip().endswith(")"):
                continue

            candidate_text = "\n".join(lines[start_idx : end_idx + 1])
            try:
                parsed = sxpb.loads(candidate_text)
                if isinstance(parsed, dict) and "answer" in parsed:
                    content = parsed["answer"]
                    if content and str(content).strip():
                        return str(content).strip()
                break  # parsed, but not an answer dict — stop growing
            except Exception:
                continue  # keep growing

    return None
