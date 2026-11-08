"""Targeted opt-in repairs, never general extraction or acceptance policy."""

from unittest.mock import patch

import pytest
import sxpb

from sxpb_llm import SxpbFixup, SxpbParseResult, parse_lenient_sxpb
from sxpb_llm.block_parse import parse_code_blocks


@pytest.mark.parametrize(
    "source",
    [
        '(answer "ok")',
        '(answer\n "ok"\n)\n',
        '(text "literal ) remains")\n',
        '(text "first\n)\nlast")\n',
        '(answer "ok") ; comment )\n',
        "",
        "; comment only\n",
    ],
)
def test_valid_input_unchanged_single_parse(source):
    with patch("sxpb_llm.sxpb_parse.sxpb.loads", wraps=sxpb.loads) as loads:
        result = parse_lenient_sxpb(source)
    assert isinstance(result, SxpbParseResult)
    assert result.fixups == ()
    assert loads.call_count == 1
    assert result.value == sxpb.loads(source)


@pytest.mark.parametrize(
    ("clean", "extra"),
    [
        ('(guess "A teacup with a whistle in its handle")\n', ")"),
        ('(answer\n "hello world"\n)\n', ")\n"),
        ('(answer "ok")\n', "  )  \n\n"),
        ('(answer "ok")\r\n', "\t)\t\r\n\r\n"),
        ('; header\n(answer "ok") ; literal ) in comment\n', ")\n"),
        ('(text "first\n)\nlast")\n', ")\n"),
        ('(text "☃ and )")\n', ")\n"),
    ],
)
def test_remove_one_standalone_paren_with_exact_character_receipt(clean, extra):
    source = clean + extra
    offset = len(clean) + extra.index(")")
    expected = source[:offset] + source[offset + 1 :]
    with patch("sxpb_llm.sxpb_parse.sxpb.loads", wraps=sxpb.loads) as loads:
        result = parse_lenient_sxpb(source)
    assert loads.call_count == 2
    assert result.value == sxpb.loads(clean)
    assert loads.call_args_list[-1].args[0] == expected
    assert result.fixups == (SxpbFixup("remove_trailing_close_paren", offset, ")"),)
    # Receipt identifies the exact deletion without returning a second source copy.
    fix = result.fixups[0]
    assert source[fix.offset : fix.offset + len(fix.removed)] == fix.removed
    assert source[: fix.offset] + source[fix.offset + len(fix.removed) :] == expected


def test_precise_kinds_survive_both_parses():
    clean = '(bag (()) one two)(variant ()(name "value"))(many (()) (x 1))\n'
    expected = sxpb.loads(clean, precise=True)
    assert isinstance(expected, sxpb.Mesg)
    for source in [clean, clean + ")"]:
        result = parse_lenient_sxpb(source, precise=True)
        assert type(result.value) is type(expected)
        assert sxpb.dumps(result.value) == sxpb.dumps(expected)
        assert isinstance(result.value["bag"], sxpb.List)
        assert type(result.value["variant"]) is type(expected["variant"])
        assert isinstance(result.value["many"], sxpb.Many)
        canonical = sxpb.dumps(result.value)
        reparsed = parse_lenient_sxpb(canonical, precise=True)
        assert not reparsed.fixups
        assert sxpb.dumps(reparsed.value) == canonical
        for key in ("bag", "variant", "many"):
            assert type(reparsed.value[key]) is type(result.value[key])


@pytest.mark.parametrize(
    "source",
    [
        ")",
        " \n\t)\n",
        "; comment-only prefix must not become an empty document\n)",
        "; even a comment with ( is not an expression\n)",
        '(x "ok"))',
        '(x "ok")\n)\n)',
        '(x "ok")\n)\n)\n)',
        '(x "ok")\n)\n(y "later")',
        '(x "ok")\n) ; do not broaden to trailing comments',
        '(x "unterminated\n)',
        '(x "valid")\n(broken (inner\n)',
        '(x "ok")\n(unterminated',
        '```sxpb > state\n(x "ok")\n)\n```',
    ],
)
def test_reject_other_errors_no_iterative_balancing_or_empty_repair(source):
    with pytest.raises(sxpb.SxpbParseError) as original:
        sxpb.loads(source)
    with pytest.raises(sxpb.SxpbParseError) as actual:
        parse_lenient_sxpb(source)
    assert str(actual.value) == str(original.value)


def test_failed_candidate_raises_original_exception():
    original = sxpb.SxpbParseError("original")
    candidate = sxpb.SxpbParseError("candidate")
    with patch(
        "sxpb_llm.sxpb_parse.sxpb.loads", side_effect=[original, candidate]
    ) as loads:
        with pytest.raises(sxpb.SxpbParseError) as actual:
            parse_lenient_sxpb("(x y)\n)")
    assert actual.value is original and loads.call_count == 2


def test_non_syntax_failure_not_caught_or_retried():
    with patch(
        "sxpb_llm.sxpb_parse.sxpb.loads", side_effect=RuntimeError("unexpected")
    ) as loads:
        with pytest.raises(RuntimeError, match="unexpected"):
            parse_lenient_sxpb("(x y)\n)")
    assert loads.call_count == 1


def test_old_block_parser_remains_strict_and_keeps_raw_body():
    body = '(x "ok")\n)'
    block = parse_code_blocks(f"```sxpb > state\n{body}\n```", parse_sxpb=True)[0]
    assert block.content == body and block.sxpb is None
    assert isinstance(block.sxpb_error, sxpb.SxpbParseError)
    assert parse_lenient_sxpb(block.content).value == {"x": "ok"}


def test_helper_does_not_apply_application_schema_or_permissions():
    result = parse_lenient_sxpb('(name "shell")(query "not permitted")\n)')
    assert result.value == {"name": "shell", "query": "not permitted"}
    # Parsing records syntax only. Callers must still reject unauthorized actions.
    assert result.fixups
