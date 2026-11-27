# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.parser.engine.replay_harness import (
    DUMMY_TOOLS,
    MockTokenizer,
    _test_request,
    collect_output,
    replay_streaming,
)
from vllm.parser.deepseek_v41 import deepseek_v41_config
from vllm.parser.parser_manager import ParserManager

CALLS = (
    '\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">杭州</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="count" string="false">42</｜DSML｜ parameter>\n'
    '</｜DSML｜ invoke>\n<｜DSML｜ invoke name="add">\n'
    '<｜DSML｜ parameter name="x" string="false">1.5</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="y" string="false">2.25</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)

# V4's ``tool_calls`` wrapper name with V4.1's spacing (#62051).
V4_NAMED_CALLS = CALLS.replace("｜DSML｜ calls>", "｜DSML｜ tool_calls>")


def tokenizer_for(text, special_calls):
    vocab = {"<think>": 50, "</think>": 51, "<｜User｜>": 54, "<｜Assistant｜>": 55}
    if special_calls:
        vocab |= {"<｜DSML｜ calls>": 52, "</｜DSML｜ calls>": 53}
    tokens: list[tuple[int, str]] = []
    while text:
        special = next((marker for marker in vocab if text.startswith(marker)), None)
        piece = special or text[0]
        tokens.append((vocab[piece] if special else 100 + len(tokens), piece))
        text = text[len(piece) :]
    return MockTokenizer(vocab, tokens), tokens


def parser_for(tokenizer, controls):
    return parser_for_name("deepseek_v41", tokenizer, controls)


def parser_for_name(parser_name, tokenizer, controls):
    cls = ParserManager.get_parser(
        tool_parser_name=parser_name,
        reasoning_parser_name=parser_name,
        enable_auto_tools=True,
    )
    assert cls is not None
    return cls(tokenizer, chat_template_kwargs=controls)


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("special_calls", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_registered_adapters_parse_parallel_calls_across_chunks(
    thinking,
    special_calls,
    chunk_size,
):
    text = ("Plan.</think>" if thinking else "") + "Checking." + CALLS
    tokenizer, tokens = tokenizer_for(text, special_calls)
    parser = parser_for(tokenizer, {"thinking": thinking})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    assert output.reasoning == ("Plan." if thinking else "")
    assert output.content.strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [json.loads(call["arguments"]) for call in output.tool_calls] == [
        {"city": "杭州", "count": 42},
        {"x": 1.5, "y": 2.25},
    ]


@pytest.mark.parametrize("thinking", [False, True])
def test_registered_adapters_parse_complete_output(thinking):
    text = ("Plan.</think>" if thinking else "") + CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": thinking})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert (reasoning or "") == ("Plan." if thinking else "")
    assert not (content or "").strip()
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert json.loads(calls[0].arguments) == {"city": "杭州", "count": 42}


@pytest.mark.parametrize("thinking", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_v4_wrapper_name_opens_tool_block(thinking, chunk_size):
    text = ("Plan.</think>" if thinking else "") + "Checking." + V4_NAMED_CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": thinking}),
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50 if thinking else 51],
        )
    )
    reasoning, content, calls = parser_for(tokenizer, {"thinking": thinking}).parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert output.reasoning == (reasoning or "") == ("Plan." if thinking else "")
    assert output.content.strip() == (content or "").strip() == "Checking."
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]
    assert [call.name for call in calls] == ["get_weather", "add"]
    assert (
        [json.loads(call["arguments"]) for call in output.tool_calls]
        == [json.loads(call.arguments) for call in calls]
        == [{"city": "杭州", "count": 42}, {"x": 1.5, "y": 2.25}]
    )


def test_v4_wrapper_name_ends_reasoning():
    text = "Plan." + V4_NAMED_CALLS
    tokenizer, tokens = tokenizer_for(text, False)
    output = collect_output(
        replay_streaming(
            parser_for(tokenizer, {"thinking": True}),
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[50],
        )
    )
    assert output.reasoning == "Plan."
    assert not output.content.strip()
    assert [call["name"] for call in output.tool_calls] == ["get_weather", "add"]


@pytest.mark.parametrize(
    ("controls", "text", "expected_reasoning", "reasoning_tokens"),
    [
        ({"thinking": False}, "12", "", 0),
        ({"enable_thinking": False}, "12", "", 0),
        ({"thinking": True, "reasoning_effort": "none"}, "12", "", 0),
        ({}, "</think>12", "", 0),
        ({}, "Plan.</think>12", "Plan.", 5),
    ],
)
def test_reasoning_adapter_controls_and_usage(
    controls,
    text,
    expected_reasoning,
    reasoning_tokens,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, controls)
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
        )
    )
    assert output.reasoning == expected_reasoning
    assert output.content == "12"
    assert (
        parser.reasoning_parser.count_reasoning_tokens([tid for tid, _ in tokens])
        == reasoning_tokens
    )
    assert parser.is_reasoning_end([50, 100, 51])
    assert not parser.is_reasoning_end([51, 100, 50])


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    ("prompt_token_ids", "text", "expected_reasoning", "expected_content"),
    [
        # continue_final_message: the current turn already closed reasoning.
        ([54, 100, 55, 50, 101, 51, 102], "12", "", "12"),
        # A closed block in an earlier turn does not end this turn's reasoning.
        (
            [54, 100, 55, 50, 101, 51, 102, 54, 103, 55, 50],
            "Plan.</think>12",
            "Plan.",
            "12",
        ),
    ],
)
def test_prompt_closed_reasoning_is_scoped_to_current_turn(
    streaming,
    prompt_token_ids,
    text,
    expected_reasoning,
    expected_content,
):
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": True})
    output_ids = [tid for tid, _ in tokens]
    if streaming:
        output = collect_output(
            replay_streaming(
                parser,
                tokens,
                chunk_size=1,
                finished_on_last=True,
                prompt_token_ids=prompt_token_ids,
            )
        )
        reasoning, content = output.reasoning, output.content
    else:
        parser.set_prompt_token_ids(prompt_token_ids)
        reasoning, content, _ = parser.parse(
            text, _test_request(), model_output_token_ids=output_ids
        )
    assert (reasoning or "") == expected_reasoning
    assert content == expected_content
    assert (parser.reasoning_parser.count_reasoning_tokens(output_ids) == 0) == (
        not expected_reasoning
    )


def test_python_argument_conversion_and_partial_values():
    converter = deepseek_v41_config().arg_converter
    assert converter is not None
    raw = (
        '<｜DSML｜ parameter name="object" string="false">'
        '{"a": [true, null]}</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bad" string="false">[broken</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="bare">7</｜DSML｜ parameter>'
        '<｜DSML｜ parameter name="text" string="true">  a<b'
    )
    assert json.loads(converter(raw, True)) == {
        "object": {"a": [True, None]},
        "bad": "[broken",
        "bare": 7,
        "text": "  a<b",
    }


# The checkpoint drops the space that separates V4.1 tags from the V4 ones, and
# at long context emits the unspaced V4 wrapper outright. deepseek_v41_config
# rebuilds ``terminals`` wholesale, so it has to restate the variant tuples that
# deepseek_v4_config declares; otherwise a non-canonical spelling has no
# transition at all and the whole block leaks into content with no tool call.
def _spaced_block(wrapper):
    return (
        f"\n\n{wrapper}\n"
        '<｜DSML｜ invoke name="get_weather">\n'
        '<｜DSML｜ parameter name="city" string="true">杭州'
        "</｜DSML｜ parameter>\n"
        "</｜DSML｜ invoke>\n"
        "</｜DSML｜ calls>"
    )


# A block that fell back to the V4 family is unspaced throughout: tolerating
# only its opener would strand the parser in TOOL_PREAMBLE.
V4_FAMILY_CALLS = (
    '\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name="get_weather">\n'
    '<｜DSML｜parameter name="city" string="true">杭州</｜DSML｜parameter>\n'
    "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
)

TOLERATED_BLOCKS = [
    pytest.param(_spaced_block("<｜DSML｜ calls>"), id="canonical-spaced"),
    pytest.param(_spaced_block("<｜DSML｜calls>"), id="wrapper-space-dropped"),
    pytest.param(_spaced_block("<｜DSML｜ call>"), id="wrapper-truncated"),
    pytest.param(V4_FAMILY_CALLS, id="unspaced-v4-family"),
]


@pytest.mark.parametrize("block", TOLERATED_BLOCKS)
@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
def test_streaming_recovers_every_tolerated_wrapper_spelling(block, chunk_size):
    text = "Checking." + block
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    output = collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=chunk_size,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[51],
        )
    )
    assert [call["name"] for call in output.tool_calls] == ["get_weather"]
    assert json.loads(output.tool_calls[0]["arguments"]) == {"city": "杭州"}
    assert output.content.strip() == "Checking."
    assert "DSML" not in output.content


@pytest.mark.parametrize("block", TOLERATED_BLOCKS)
def test_non_streaming_recovers_every_tolerated_wrapper_spelling(block):
    text = "Checking." + block
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    reasoning, content, calls = parser.parse(
        text,
        _test_request(tools=DUMMY_TOOLS),
        enable_auto_tools=True,
        model_output_token_ids=[tid for tid, _ in tokens],
    )
    assert [call.name for call in calls] == ["get_weather"]
    assert json.loads(calls[0].arguments) == {"city": "杭州"}
    assert (content or "").strip() == "Checking."
    assert "DSML" not in (content or "")


def test_variants_never_displace_the_canonical_spelling():
    """The first declared spelling is the one the format reports.

    ``terminal_literal`` backs the renderer, structural-tag grammars and any
    stop string handed to the sampler, so a tolerated misspelling must never
    become the spelling vLLM itself emits.
    """
    config = deepseek_v41_config()
    assert config.terminal_literal("TOOL_START") == "<｜DSML｜ calls>"
    assert config.terminal_literal("TOOL_END") == "</｜DSML｜ calls>"
    assert config.terminal_literal("INVOKE_PREFIX") == '<｜DSML｜ invoke name="'


def test_token_id_terminals_stay_single_canonical_strings():
    """Token-id matching has no variant form.

    ``token_id_terminals`` is typed ``dict[str, str]``: a special token is
    either in the vocab or it is not, so widening the text terminals must not
    leak tuples into it.
    """
    config = deepseek_v41_config()
    assert all(isinstance(v, str) for v in config.token_id_terminals.values())
    assert config.token_id_terminals["TOOL_START"] == "<｜DSML｜ calls>"
    assert config.token_id_terminals["TOOL_END"] == "</｜DSML｜ calls>"


def test_v4_misspelling_tolerance_is_not_regressed_by_v41_overrides():
    """The V4 parser keeps its own #56141 variants independently."""
    from vllm.parser.deepseek_v4 import deepseek_v4_config

    v4 = deepseek_v4_config().terminals["TOOL_START"]
    assert isinstance(v4, tuple)
    assert "<｜DSML｜tool_calls>" in v4
    assert "<｜DSML｜toolcalls>" in v4
    assert "<｜DSML｜tool>" in v4


# Prose that merely talks about the markup. The tolerated spellings enter the
# ordinary TOOL_START transition rather than the validated orphan-recovery
# path, so quoting one must never be enough to synthesise a tool call.
PROSE_FIXTURES = [
    pytest.param(
        "Wrap the call in <｜DSML｜tool_calls> and close it again.",
        id="mentions-v4-wrapper",
    ),
    pytest.param(
        "The V4.1 spelling is <｜DSML｜ calls>, not <｜DSML｜tool_calls>.",
        id="mentions-both-spellings",
    ),
    pytest.param(
        "Never emit <｜DSML｜tool> or <｜DSML｜toolcalls> as literal text.",
        id="mentions-misspellings",
    ),
    pytest.param(
        'An example is <｜DSML｜ invoke name="get_weather"> in prose.',
        id="mentions-orphan-invoke",
    ),
]


def _stream_prose(parser_name, prose):
    tokenizer, tokens = tokenizer_for(prose, False)
    parser = parser_for_name(parser_name, tokenizer, {"thinking": False})
    return collect_output(
        replay_streaming(
            parser,
            tokens,
            chunk_size=1,
            finished_on_last=True,
            tools=DUMMY_TOOLS,
            prompt_token_ids=[51],
        )
    )


@pytest.mark.parametrize("prose", PROSE_FIXTURES)
def test_tolerated_variants_do_not_fabricate_calls_from_prose(prose):
    """Widening TOOL_START must not turn quoted markup into a tool call.

    An opener with no matching close is swallowed rather than emitted, which is
    long-standing V4 behaviour and the reason the strict-admission gate exists;
    it is not what this asserts. What must hold is that an unclosed mention
    never yields a call, and that V4.1 does not diverge from the parser whose
    variants it now shares.
    """
    outputs = {
        name: _stream_prose(name, prose) for name in ("deepseek_v41", "deepseek_v4")
    }
    for name, output in outputs.items():
        assert output.tool_calls == [], f"{name}: fabricated {output.tool_calls}"
    v41, v4 = outputs["deepseek_v41"], outputs["deepseek_v4"]
    assert bool(v41.content.strip()) == bool(v4.content.strip()), (
        "V4.1 and V4 disagree on whether the mention survives: "
        f"{v41.content!r} vs {v4.content!r}"
    )


# A closed block naming a tool the request never declared is committed by both
# parsers: only the orphan-invoke recovery carries ``validate_tool_name``, so a
# properly wrapped block is trusted. That is pre-existing upstream behaviour and
# not this change's to fix -- but widening the wrapper spellings must not make
# V4.1 answer differently from V4 on the same shape.
UNDECLARED_BLOCK = {
    "deepseek_v4": (
        "Example:\n<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="delete_everything">\n'
        '<｜DSML｜parameter name="path" string="true">/</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
    ),
    "deepseek_v41": (
        "Example:\n<｜DSML｜ calls>\n"
        '<｜DSML｜ invoke name="delete_everything">\n'
        '<｜DSML｜ parameter name="path" string="true">/</｜DSML｜ parameter>\n'
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    ),
}


def test_wrapped_block_handling_matches_v4():
    results = {
        name: [c["name"] for c in _stream_prose(name, block).tool_calls]
        for name, block in UNDECLARED_BLOCK.items()
    }
    assert results["deepseek_v41"] == results["deepseek_v4"], results


@pytest.mark.parametrize(
    "partial_tag", ['<｜DSML｜ parameter name="j', "<｜DSML｜ parameter na"]
)
def test_partial_tag_after_gt_is_held_back(partial_tag):
    config = deepseek_v41_config()
    assert config.arg_converter is not None
    assert config.arg_value_extender is not None
    raw = '<｜DSML｜ parameter name="k" string="true">a > b' + partial_tag
    assert json.loads(config.arg_converter(raw, True)) == {"k": "a > b"}
    assert config.arg_value_extender(raw) is None


def test_long_string_argument_streams_before_close():
    # Regression for #52846: a string body has no ">" so it used to be
    # buffered until the closing parameter tag arrived.
    body = "A" * 4096
    text = (
        '<｜DSML｜ calls>\n<｜DSML｜ invoke name="emit">\n'
        f'<｜DSML｜ parameter name="text" string="true">{body}'
        "</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "emit",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                },
            },
        }
    ]
    tokenizer, tokens = tokenizer_for(text, False)
    parser = parser_for(tokenizer, {"thinking": False})
    results = replay_streaming(
        parser, tokens, chunk_size=32, finished_on_last=True, tools=tools
    )
    arg_chunks = [
        tool_call.function.arguments
        for result in results
        if result
        for tool_call in result.tool_calls or []
        if tool_call.function and tool_call.function.arguments
    ]
    assert len(arg_chunks) > len(body) // 64
    assert json.loads("".join(arg_chunks)) == {"text": body}
