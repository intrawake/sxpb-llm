"""Exact fence tokenization is shared; action authority belongs to callers."""

import random

import pytest

from sxpb_llm import CodeBlockError, parse_code_blocks


def reconstruct(blocks):
    return "".join((b.opening or "") + b.content + (b.closing or "") for b in blocks)


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
@pytest.mark.parametrize(
    "body",
    ["", "x", "x  ", "x\t", "x\n\n", "x\r", "a\r\nb\n", "é🐸\u2028\u2029\x85\v\f"],
)
def test_exact_content_and_boundaries(eol, body):
    opening = "```py > a.py" + eol
    closing = "```" + eol
    source = "prose\n" + opening + body + eol + closing + "after"
    blocks = parse_code_blocks(source)
    assert reconstruct(blocks) == source
    assert len(blocks) == 3
    block = blocks[1]
    assert (block.language, block.operation, block.filepath) == ("py", ">", "a.py")
    assert block.opening == opening and block.closing == closing
    assert block.content == body + eol
    assert block.error is None


@pytest.mark.parametrize(
    "source",
    [
        "",
        "\n",
        "\r",
        "prose",
        "```py",
        "```py\n",
        "```py\nx  ",
        "```py\n```",
        "```py\n```\n",
        "```py\n\n```",
        "```py\r\n```\r\n",
        "~~~py > a.py\nx\n~~~",
        "  ```py > a.py  \nx\n  ````  \n",
        "```py\nx\u2028```\ny\n```",
        "```py\nx\r```\ny\n```",
        "```py\nx\n```\r",
        "```py\nx\n```\u2028",
        "```py\nx\n```\x85",
        ">a.py\nx  \n\n<in.py\ny\n",
        "```py\nx\n```\n```text\ny\n```",
    ],
)
def test_reconstruction(source):
    assert reconstruct(parse_code_blocks(source)) == source


def test_bare_operations_remain_exact_prose():
    source = "before\n>out.py\nx  \n\n<in.py\ny\nafter\n"
    blocks = parse_code_blocks(source)
    assert len(blocks) == 1
    assert blocks[0].error == CodeBlockError.NOT_FENCED
    assert blocks[0].operation is None
    assert blocks[0].opening is blocks[0].closing is None
    assert blocks[0].content == source


@pytest.mark.parametrize("separator", ["\r", "\v", "\f", "\x85", "\u2028", "\u2029"])
def test_only_lf_delimits_fences(separator):
    source = "```py\nx" + separator + "```\ny\n```"
    blocks = parse_code_blocks(source)
    assert len(blocks) == 1 and blocks[0].error is None
    assert blocks[0].content == "x" + separator + "```\ny\n"
    assert reconstruct(blocks) == source


def test_unmatched_keeps_boundary_and_all_content():
    source = "```py > a.py\r\nx  \n\t"
    (block,) = parse_code_blocks(source)
    assert block.opening == "```py > a.py\r\n"
    assert block.closing is None
    assert block.error == CodeBlockError.UNMATCHED_FENCE
    assert block.content == "x  \n\t"
    assert reconstruct([block]) == source


def test_raw_shorter_fences_stay_inside():
    source = "`````py > a.py\n```sxpb > state\n(x 1)\n```\n````\n`````"
    (block,) = parse_code_blocks(source)
    assert block.content == "```sxpb > state\n(x 1)\n```\n````\n"
    assert reconstruct([block]) == source


def test_default_preserves_content_and_leaves_unfenced_operations_as_prose():
    (block,) = parse_code_blocks("```py > a.py\nx  \n```")
    assert block.content == "x  \n"
    assert block.opening == "```py > a.py\n" and block.closing == "```"
    (block,) = parse_code_blocks(">a.py\nx  \n")
    assert block.error is CodeBlockError.NOT_FENCED
    assert block.filepath is block.operation is None
    assert block.content == ">a.py\nx  \n"


def test_optional_sxpb_parsing_does_not_change_raw_content():
    (block,) = parse_code_blocks(
        "```sxpb > state\n(x 1)  \n```", parse_sxpb=True, precise=True
    )
    assert block.sxpb == {"x": 1}
    assert block.sxpb_error is None and block.content == "(x 1)  \n"


def test_deterministic_mixed_input_reconstruction():
    rng = random.Random(0)
    pieces = [
        "",
        "x",
        " ",
        "\t",
        "\n",
        "\r\n",
        "\r",
        "\u2028",
        "\x85",
        "```",
        "~~~",
        "````py > a.py\n",
        "```sxpb > state\n",
        "\n```\n",
        ">a.py\n",
        "<in.py\n",
        "(x 1)",
        "é🐸",
    ]
    for _ in range(300):
        source = "".join(rng.choice(pieces) for _ in range(rng.randrange(30)))
        assert reconstruct(parse_code_blocks(source)) == source
