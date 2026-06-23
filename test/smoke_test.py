"""Smoke tests for sxpb_llm package."""

import sxpb_llm
from sxpb_llm.model import ModelConfig, load_model_definitions, resolve_model


def test_imports():
    """Verify all public names are importable."""
    assert hasattr(sxpb_llm, "call_api")
    assert hasattr(sxpb_llm, "ModelConfig")
    assert hasattr(sxpb_llm, "load_model_definitions")
    assert hasattr(sxpb_llm, "resolve_model")
    assert hasattr(sxpb_llm, "get_sxpb_from_markdown")
    assert hasattr(sxpb_llm, "parse_sxpb_answer")


def test_model_config_defaults():
    """ModelConfig should have sensible defaults."""
    mc = ModelConfig(fullname="test-model")
    assert mc.fullname == "test-model"
    assert mc.token_ctx_limit == 128000
    assert mc.token_gen_limit == 16384
    assert mc.reasoning_effort is None
    assert mc.timeout == 0
    assert mc.extra == {}


def test_load_model_definitions_from_string():
    """Parse a SxPB string directly."""
    s = """()
(local-test
 (fullname llama.cpp/test:Q8_0)
 (token_gen_limit 4000)
 (timeout 60)
)
(dono-test
 (fullname aistudio/test)
)
"""
    defs = load_model_definitions(s)
    assert len(defs) == 2

    assert "local-test" in defs
    mc = defs["local-test"]
    assert mc.fullname == "llama.cpp/test:Q8_0"
    assert mc.token_gen_limit == 4000
    assert mc.timeout == 60
    assert mc.token_ctx_limit == 128000  # default

    assert "dono-test" in defs
    mc = defs["dono-test"]
    assert mc.fullname == "aistudio/test"
    assert mc.token_gen_limit == 16384  # default


def test_load_model_definitions_string_alias():
    """A bare string value becomes the fullname."""
    defs = load_model_definitions('() (just-a-string "my/model")')
    assert defs["just-a-string"].fullname == "my/model"


def test_resolve_model_found():
    """resolve_model merges definition defaults with overrides."""
    defs = {
        "test": ModelConfig(
            fullname="base/model",
            token_gen_limit=8000,
            timeout=30,
            extra={"temperature": 0.7},
        )
    }
    mc = resolve_model("test", defs, token_gen_limit=4000, top_p=0.9)
    assert mc.fullname == "base/model"
    assert mc.token_gen_limit == 4000  # overridden
    assert mc.timeout == 30  # from def
    assert mc.extra == {"temperature": 0.7, "top_p": 0.9}


def test_resolve_model_not_found():
    """resolve_model uses alias as fullname when not in definitions."""
    mc = resolve_model("some/model", {})
    assert mc.fullname == "some/model"
    assert mc.token_gen_limit == 16384  # default


def test_sxpb_parse_answer_fenced():
    """parse_sxpb_answer extracts from ```sxpb > /dev/stdout blocks."""
    from sxpb_llm.sxpb_parse import parse_sxpb_answer

    response = """\
Some preamble text.

```sxpb > /dev/stdout
(answer "b2")
```
Some trailing text.
"""
    assert parse_sxpb_answer(response) == "b2"


def test_sxpb_parse_answer_bare():
    """parse_sxpb_answer falls back to bare (answer ...) lines."""
    from sxpb_llm.sxpb_parse import parse_sxpb_answer

    response = """\
Let me think about this. The best move is:

(answer "c3")

That should work.
"""
    assert parse_sxpb_answer(response) == "c3"


def test_sxpb_parse_answer_none():
    """parse_sxpb_answer returns None for unparseable input."""
    from sxpb_llm.sxpb_parse import parse_sxpb_answer

    assert parse_sxpb_answer("") is None
    assert parse_sxpb_answer("Just some text, no answer here.") is None


def test_get_sxpb_from_markdown():
    """get_sxpb_from_markdown extracts and parses sxpb blocks."""
    from sxpb_llm.sxpb_parse import get_sxpb_from_markdown

    text = """\
```sxpb
(candidates
 (candidate (name "Test") (texture "Smooth"))
)
```
"""
    result = get_sxpb_from_markdown(text)
    assert result is not None
    assert "candidates" in result
