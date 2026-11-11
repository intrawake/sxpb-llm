"""Fenced metadata and exact source preservation; unfenced text stays prose."""

import pytest

from sxpb_llm.block_parse import CodeBlock, CodeBlockError, parse_code_blocks


def test_codeblock_fields_and_defaults():
    block = CodeBlock("sxpb", ">", "out.sxpb", '(answer "42")')
    assert (block.language, block.operation, block.filepath, block.content) == (
        "sxpb",
        ">",
        "out.sxpb",
        '(answer "42")',
    )
    assert block.error is block.sxpb is block.sxpb_error is None
    assert block.opening is block.closing is None
    assert "out.sxpb" in repr(block)
    assert CodeBlockError.UNMATCHED_FENCE.value == "unmatched_fence"
    assert CodeBlockError.NOT_FENCED.value == "not_fenced"


def test_empty_input():
    assert parse_code_blocks("") == []


@pytest.mark.parametrize(
    ("opening", "content", "closing", "metadata"),
    [
        ("```python\n", 'print("hi")\n', "```", ("python", None, None)),
        (
            "```sxpb > /dev/stdout\n",
            '(answer "42")\n',
            "```",
            ("sxpb", ">", "/dev/stdout"),
        ),
        (
            "```sxpb >assistant.sxpb\n",
            "(name Bot)\n",
            "```",
            ("sxpb", ">", "assistant.sxpb"),
        ),
        (
            "```sxpb < /dev/stdin\n",
            "(question why)\n",
            "```",
            ("sxpb", "<", "/dev/stdin"),
        ),
        ("```sxpb <input.sxpb\n", "(read)\n", "```", ("sxpb", "<", "input.sxpb")),
        ("```text output.txt\n", "Hello world\n", "```", ("text", None, "output.txt")),
        ("```sxpb >\n", "(data)\n", "```", ("sxpb", ">", None)),
        ("```\n", "preformatted text\n", "```", ("", None, None)),
        ("~~~sh\n", "echo hi\n", "~~~", ("sh", None, None)),
        ("````python\n", "code with ``` inside\n", "````", ("python", None, None)),
        ("```py\n", "pass\n", "`````\n", ("py", None, None)),
        ("   ```sxpb > out\n", "(x)\n", "   ```", ("sxpb", ">", "out")),
        ("\t```py   \n", "1\n", "\t```  \n", ("py", None, None)),
        (
            "```sxpb > <> arrows.txt\n",
            "content\n",
            "```",
            ("sxpb", ">", "<> arrows.txt"),
        ),
        ("```> file.txt\n", "content\n", "```", (">", None, "file.txt")),
        ("```py\n", "", "```", ("py", None, None)),
        ("```py\r\n", "x  \r\n", "```\r\n", ("py", None, None)),
        ("```text\n", "first\n\nsecond  \t\n\n", "```", ("text", None, None)),
        ("```py\n", "```sxpb\nnot a closer\n", "```", ("py", None, None)),
    ],
)
def test_fenced_metadata_and_original_text(opening, content, closing, metadata):
    language, operation, filepath = metadata
    assert parse_code_blocks(opening + content + closing) == [
        CodeBlock(
            language,
            operation,
            filepath,
            content,
            opening=opening,
            closing=closing,
        )
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Just some plain text.",
        " \n\t",
        ">f.sxpb\n(data)\n",
        ">f.sxpb\n(data)",
        "Before.\n<in.txt\n(read)\nAfter.\n",
        ">a.sxpb\n(a)\n>b.sxpb\n(b)\n",
        ">f.sxpb\nline1\nline2\n\nprose\n",
        ">empty.sxpb\n\n\nMore prose.\n",
        ">f.sxpb\n(data)  \n\n",
        "  > f.sxpb\n(x)\n",
        "\t>\tf.sxpb\n(x)\n",
        ">README\ncontent\n",
        ">sub/dir/file.sxpb\n(x)\n",
        "<in.txt\n(read)\n>out.txt\n(write)\n",
        "This is a > test with an arrow mid-line.\n",
        ">file with spaces.sxpb\ncontent\n",
    ],
)
def test_unfenced_text_is_exact_prose_not_an_operation(text):
    assert parse_code_blocks(text, parse_sxpb=True) == [
        CodeBlock("", None, None, text, error=CodeBlockError.NOT_FENCED)
    ]


def test_prose_and_fenced_blocks_preserve_order_and_separators():
    text = (
        "Before.\n>not-an-artifact.sxpb\n(inert)\n"
        "```sxpb > a.sxpb  \n(a)  \n```  \n"
        "\n"
        "```sxpb > b.sxpb\n(b)\n```\n"
        "After.\n"
    )
    blocks = parse_code_blocks(text)
    assert [(block.language, block.content) for block in blocks] == [
        ("", "Before.\n>not-an-artifact.sxpb\n(inert)\n"),
        ("sxpb", "(a)  \n"),
        ("", "\n"),
        ("sxpb", "(b)\n"),
        ("", "After.\n"),
    ]
    assert [block.error for block in blocks] == [
        CodeBlockError.NOT_FENCED,
        None,
        CodeBlockError.NOT_FENCED,
        None,
        CodeBlockError.NOT_FENCED,
    ]
    assert (
        "".join(
            (block.opening or "") + block.content + (block.closing or "")
            for block in blocks
        )
        == text
    )


def test_adjacent_fences_do_not_invent_prose():
    blocks = parse_code_blocks("```py\n1\n```\n```py\n2\n```")
    assert len(blocks) == 2
    assert [block.content for block in blocks] == ["1\n", "2\n"]
    assert all(block.error is None for block in blocks)


@pytest.mark.parametrize(
    ("opening", "content", "metadata"),
    [
        ("```py", "", ("py", None, None)),
        ("```py\n", "", ("py", None, None)),
        ("```sxpb > output.sxpb\n", "(incomplete", ("sxpb", ">", "output.sxpb")),
        ("````\n", "hi\n```", ("", None, None)),
        ("~~~sh\n", "hi\n```\nstill in tilde block", ("sh", None, None)),
        ("```py\r\n", "x  \n\t", ("py", None, None)),
    ],
)
def test_unmatched_fence_preserves_the_remaining_source(opening, content, metadata):
    language, operation, filepath = metadata
    assert parse_code_blocks(opening + content) == [
        CodeBlock(
            language,
            operation,
            filepath,
            content,
            error=CodeBlockError.UNMATCHED_FENCE,
            opening=opening,
        )
    ]
