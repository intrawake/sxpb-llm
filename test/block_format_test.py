import re

import pytest

from sxpb_llm import format_code_block, parse_code_blocks


@pytest.mark.parametrize(
    "content",
    [
        "",
        "x",
        "x\n",
        "x\n\n",
        "x\r\n",
        "x\r",
        " \tx  \n\n ",
        "é🐸\u2028\u2029\x85\v\f",
        "a`b",
        "a``b",
        "```sxpb > state\n(x 1)\n```\n````",
    ],
)
def test_content_and_collision_safe_boundaries(content):
    formatted = format_code_block(
        content, language="py", operation="<", filepath="a file.py"
    )
    (block,) = parse_code_blocks(formatted)
    assert block.error is None
    assert (block.language, block.operation, block.filepath) == ("py", "<", "a file.py")
    expected = content + "\n" if content and not content.endswith("\n") else content
    assert block.content == expected
    fence = block.closing
    assert fence is not None and set(fence) == {"`"} and len(fence) >= 3
    assert all(len(run) < len(fence) for run in re.findall(r"`+", content))
    assert block.opening == f"{fence}py < a file.py\n"
    assert block.opening + block.content + fence == formatted


@pytest.mark.parametrize(
    "language, operation, filepath, info",
    [
        ("", None, None, ""),
        ("py", None, None, "py"),
        ("text", "<", "notes.txt", "text < notes.txt"),
        ("sxpb", ">", "state.sxpb", "sxpb > state.sxpb"),
        ("text", None, "dir/my é file.txt", "text dir/my é file.txt"),
    ],
)
def test_header_fields_roundtrip(language, operation, filepath, info):
    formatted = format_code_block(
        "data", language=language, operation=operation, filepath=filepath
    )
    assert formatted == f"```{info}\ndata\n```"
    (block,) = parse_code_blocks(formatted)
    assert (block.language, block.operation, block.filepath) == (
        language,
        operation,
        filepath,
    )


@pytest.mark.parametrize(
    "fields",
    [
        {"language": "py\ninjected"},
        {"operation": ">\rinjected"},
        {"filepath": "a`file.txt"},
    ],
)
def test_fields_cannot_break_the_opening_fence(fields):
    with pytest.raises(ValueError):
        format_code_block("data", **fields)
