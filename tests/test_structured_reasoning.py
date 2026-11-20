"""Issue #2: dict response schemas, per-request settings/reasoning, deep-merged
overrides, and thinking-token usage."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from llm_kit import (
    BatchItem,
    LLMClient,
    Message,
    Reasoning,
    ReasoningEffort,
    RequestSettings,
    StreamEnd,
    UsageLedger,
    ValidationError,
)
from llm_kit.batch.jsonl import write_results_to_jsonl
from llm_kit.config.llm import MessageFormat, StructuredOutputStyle
from llm_kit.llm.response import BatchResult, LLMResponse, TokenUsage
from llm_kit.messages.formatter import AnthropicFormatter, GeminiFormatter, OpenAIFormatter
from llm_kit.messages.formatter._shared import (
    EFFORT_TO_BUDGET,
    effort_from_budget,
    resolve_budget,
    resolve_effort,
)
from llm_kit.provider_batch.driver import serialize_items
from llm_kit.provider_batch.job import BatchJob

from conftest import (
    anthropic_sse,
    chat_response,
    gemini_sse,
    make_config,
    mock_client,
    stream_mock_client,
)
from test_provider_batch_openai import FakeOpenAIBatch

SCHEMA: dict[str, Any] = {
    "title": "Verdict",
    "type": "object",
    "properties": {"label": {"type": "string"}, "score": {"type": "number"}},
    "required": ["label"],
}


def _gemini_llm(handler) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.GEMINI
    cfg.llm.chat_completions_path = "/v1beta/models/gemini-test:generateContent"
    return LLMClient(cfg, client=mock_client(handler), owns_client=True)


def _gemini_stream_llm(body: bytes) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.GEMINI
    cfg.llm.chat_completions_path = "/v1beta/models/gemini-test:generateContent"
    return LLMClient(cfg, client=stream_mock_client(body), owns_client=True)


def _anthropic_llm(handler=None, client=None) -> LLMClient:
    cfg = make_config()
    cfg.llm.message_format = MessageFormat.ANTHROPIC
    cfg.llm.chat_completions_path = "/v1/messages"
    return LLMClient(cfg, client=client or mock_client(handler), owns_client=True)


def _gemini_response(text: str = '{"label": "ok"}', **usage: int) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}],
        "usageMetadata": usage or {"promptTokenCount": 1, "candidatesTokenCount": 1},
        "modelVersion": "gemini-test",
    }


# ------------------------------------------------------- schema dict + reasoning in body


def _build(fmt, **kw):
    base = dict(
        model="m",
        messages=[],
        system=None,
        temperature=None,
        top_p=None,
        max_tokens=None,
        extra_sampling={},
        response_model=SCHEMA,
    )
    base.update(kw)
    return fmt.build_body(**base)


def test_openai_json_schema_dict_and_reasoning() -> None:
    body = _build(
        OpenAIFormatter(structured_output_style=StructuredOutputStyle.OPENAI_JSON_SCHEMA),
        reasoning=Reasoning(effort=ReasoningEffort.HIGH),
    )
    assert body["reasoning_effort"] == "high"
    js = body["response_format"]["json_schema"]
    assert js["name"] == "Verdict"
    assert js["strict"] is True
    assert js["schema"]["additionalProperties"] is False
    assert js["schema"]["required"] == ["label", "score"]  # strictified copy


def test_openai_tool_use_style_dict_and_reasoning() -> None:
    body = _build(
        OpenAIFormatter(structured_output_style=StructuredOutputStyle.TOOL_USE),
        reasoning=Reasoning(budget_tokens=2000),
    )
    assert body["reasoning_effort"] == "low"
    assert body["tools"][0]["function"]["name"] == "Verdict"
    assert body["tools"][0]["function"]["parameters"] == SCHEMA
    assert body["tool_choice"]["function"]["name"] == "Verdict"


def test_dict_schema_without_title_is_named_response() -> None:
    fmt = OpenAIFormatter(structured_output_style=StructuredOutputStyle.OPENAI_JSON_SCHEMA)
    fields = fmt.structured_output_fields({"type": "object", "properties": {}})
    assert fields["response_format"]["json_schema"]["name"] == "response"


def test_anthropic_dict_and_reasoning() -> None:
    body = _build(AnthropicFormatter(), reasoning=Reasoning(effort=ReasoningEffort.MEDIUM))
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 8192}
    schema = body["output_config"]["format"]["schema"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["label"] == {"type": "string"}


def test_anthropic_explicit_budget_wins_and_temperature_untouched() -> None:
    body = _build(
        AnthropicFormatter(),
        temperature=0.3,
        max_tokens=100,
        reasoning=Reasoning(effort=ReasoningEffort.LOW, budget_tokens=5000),
    )
    assert body["thinking"]["budget_tokens"] == 5000
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 100


def test_gemini_dict_and_reasoning() -> None:
    body = _build(GeminiFormatter(), reasoning=Reasoning(effort=ReasoningEffort.MINIMAL))
    gc = body["generationConfig"]
    assert gc["responseMimeType"] == "application/json"
    assert gc["responseJsonSchema"] == SCHEMA
    assert gc["thinkingConfig"] == {"thinkingLevel": "minimal"}
    assert "responseSchema" not in gc
    assert "thinkingBudget" not in gc["thinkingConfig"]


def test_gemini_budget_only_maps_to_thinking_level() -> None:
    body = _build(GeminiFormatter(), reasoning=Reasoning(budget_tokens=20000))
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


def test_gemini_reasoning_keeps_extra_thinking_config() -> None:
    body = _build(
        GeminiFormatter(),
        extra_sampling={"thinkingConfig": {"includeThoughts": True}},
        reasoning=Reasoning(effort=ReasoningEffort.LOW),
        response_model=None,
    )
    assert body["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "low",
    }


def test_no_reasoning_emits_nothing() -> None:
    assert "reasoning_effort" not in _build(
        OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE),
        reasoning=Reasoning(),
    )
    assert "thinking" not in _build(AnthropicFormatter())


def test_gemini_pydantic_nested_model_keeps_defs() -> None:
    class Inner(BaseModel):
        n: int

    class Outer(BaseModel):
        inner: Inner
        items: list[Inner]

    gc = _build(GeminiFormatter(), response_model=Outer)["generationConfig"]
    schema = gc["responseJsonSchema"]
    assert "Inner" in schema["$defs"]
    assert schema["properties"]["inner"] == {"$ref": "#/$defs/Inner"}
    assert "responseSchema" not in gc


# ------------------------------------------------------------- mapping tables


@pytest.mark.parametrize(
    ("budget", "effort"),
    [(0, "minimal"), (1024, "minimal"), (1025, "low"), (4096, "low"), (4097, "medium"),
     (16384, "medium"), (16385, "high"), (100000, "high")],
)
def test_budget_to_effort_thresholds(budget: int, effort: str) -> None:
    assert effort_from_budget(budget) == effort
    assert resolve_effort(Reasoning(budget_tokens=budget)) == effort


def test_effort_to_budget_table() -> None:
    assert EFFORT_TO_BUDGET == {
        ReasoningEffort.MINIMAL: 1024,
        ReasoningEffort.LOW: 2048,
        ReasoningEffort.MEDIUM: 8192,
        ReasoningEffort.HIGH: 24576,
    }
    assert resolve_budget(Reasoning(effort=ReasoningEffort.HIGH)) == 24576
    assert resolve_budget(Reasoning()) is None and resolve_budget(None) is None
    assert resolve_effort(None) is None


# ------------------------------------------------------ dict parsing / validation


async def test_dict_schema_parses_to_plain_json_and_invalid_is_validation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(content='{"label": "a", "extra": [1, 2]}'))

    llm = LLMClient(make_config(), client=mock_client(handler), owns_client=True)
    async with llm:
        resp = await llm.invoke([Message.user("hi")], response_model=SCHEMA)
    assert resp.parsed == {"label": "a", "extra": [1, 2]}  # unvalidated: extra key kept
    assert type(resp.parsed) is dict

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_response(content="not json {"))

    llm = LLMClient(make_config(), client=mock_client(bad), owns_client=True)
    async with llm:
        with pytest.raises(ValidationError) as ei:
            await llm.invoke([Message.user("hi")], response_model=SCHEMA)
        assert ei.value.raw_content == "not json {"
        results = [
            r async for r in llm.batch_stream(
                [BatchItem(id="x", messages=[Message.user("hi")], response_model=SCHEMA)]
            )
        ]
    assert isinstance(results[0].error, ValidationError)


async def test_dict_schema_streaming_assembles_parsed() -> None:
    body = gemini_sse(
        [
            {"candidates": [{"content": {"parts": [{"text": '{"label":'}]}}]},
            {
                "candidates": [{"content": {"parts": [{"text": ' "z"}'}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2},
            },
        ]
    )
    llm = _gemini_stream_llm(body)
    async with llm:
        events = [e async for e in llm.invoke_stream([Message.user("q")], response_model=SCHEMA)]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.parsed == {"label": "z"}


async def test_callers_schema_dict_is_never_mutated() -> None:
    snapshot = copy.deepcopy(SCHEMA)
    for fmt in (
        OpenAIFormatter(structured_output_style=StructuredOutputStyle.OPENAI_JSON_SCHEMA),
        OpenAIFormatter(structured_output_style=StructuredOutputStyle.TOOL_USE),
        AnthropicFormatter(),
        GeminiFormatter(),
    ):
        body = _build(fmt, reasoning=Reasoning(effort=ReasoningEffort.LOW))
        # mutate the output; the caller's dict must be unaffected
        json.dumps(body)
    assert SCHEMA == snapshot
    body = _build(GeminiFormatter())
    body["generationConfig"]["responseJsonSchema"]["properties"]["label"]["type"] = "X"
    assert SCHEMA == snapshot


def test_jsonl_serialises_dict_and_list_parsed(tmp_path: Path) -> None:
    async def gen():
        for i, parsed in enumerate([{"a": 1}, [1, 2], None]):
            yield BatchResult(id=f"i{i}", response=LLMResponse(text="t", parsed=parsed))

    import asyncio

    path = tmp_path / "o.jsonl"
    asyncio.run(write_results_to_jsonl(path, gen()))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["response"]["parsed"] for r in rows] == [{"a": 1}, [1, 2], None]


# ------------------------------------------------- overrides: deep merge


async def test_override_of_one_generation_config_key_keeps_the_rest() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=_gemini_response())

    llm = _gemini_llm(handler)
    async with llm:
        await llm.invoke(
            [Message.user("hi")],
            response_model=SCHEMA,
            settings=RequestSettings(max_tokens=50, reasoning=Reasoning(effort=ReasoningEffort.LOW)),
            overrides={"generationConfig": {"temperature": 0.9}},
        )
    gc = seen["generationConfig"]
    assert gc["temperature"] == 0.9
    assert gc["responseJsonSchema"] == SCHEMA
    assert gc["maxOutputTokens"] == 50
    assert gc["thinkingConfig"] == {"thinkingLevel": "low"}


async def test_override_lists_replace_and_config_not_mutated() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    cfg = make_config()
    cfg.llm.extra_sampling = {"metadata": {"a": 1, "b": 2}, "stop": ["x"]}
    llm = LLMClient(cfg, client=mock_client(handler), owns_client=True)
    async with llm:
        await llm.invoke(
            [Message.user("hi")], overrides={"metadata": {"b": 3}, "stop": ["y", "z"]}
        )
    assert seen["metadata"] == {"a": 1, "b": 3}
    assert seen["stop"] == ["y", "z"]
    assert cfg.llm.extra_sampling == {"metadata": {"a": 1, "b": 2}, "stop": ["x"]}


async def test_precedence_config_then_settings_then_overrides() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=chat_response())

    cfg = make_config()
    cfg.llm.temperature = 0.1
    cfg.llm.max_tokens = 10
    cfg.llm.top_p = 0.5
    llm = LLMClient(cfg, client=mock_client(handler), owns_client=True)
    async with llm:
        await llm.invoke([Message.user("a")])
        await llm.invoke([Message.user("b")], settings=RequestSettings(temperature=0.7, max_tokens=99))
        await llm.invoke(
            [Message.user("c")],
            settings=RequestSettings(temperature=0.7),
            overrides={"temperature": 1.5},
        )
    assert (bodies[0]["temperature"], bodies[0]["max_tokens"], bodies[0]["top_p"]) == (0.1, 10, 0.5)
    assert (bodies[1]["temperature"], bodies[1]["max_tokens"], bodies[1]["top_p"]) == (0.7, 99, 0.5)
    assert bodies[2]["temperature"] == 1.5 and bodies[2]["max_tokens"] == 10


# ------------------------------------ per-item settings (batch + provider batch)


async def test_two_batch_items_carry_different_settings() -> None:
    bodies: dict[str, dict[str, Any]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies[body["messages"][-1]["content"]] = body
        return httpx.Response(200, json=chat_response())

    llm = LLMClient(make_config(), client=mock_client(handler), owns_client=True)
    items = [
        BatchItem(id="a", messages=[Message.user("a")], settings=RequestSettings(temperature=0.0, max_tokens=5)),
        BatchItem(
            id="b",
            messages=[Message.user("b")],
            settings=RequestSettings(temperature=1.0, reasoning=Reasoning(effort=ReasoningEffort.HIGH)),
        ),
    ]
    async with llm:
        results = [r async for r in llm.batch_stream(items)]
    assert all(r.ok for r in results)
    assert (bodies["a"]["temperature"], bodies["a"]["max_tokens"]) == (0.0, 5)
    assert "reasoning_effort" not in bodies["a"]
    assert bodies["b"]["temperature"] == 1.0 and bodies["b"]["reasoning_effort"] == "high"
    assert "max_tokens" not in bodies["b"]


async def test_provider_batch_serialize_items_per_item_settings_and_schema() -> None:
    llm = LLMClient(make_config(), client=mock_client(FakeOpenAIBatch()), owns_client=True)
    items = [
        BatchItem(id="a", messages=[Message.user("a")], settings=RequestSettings(temperature=0.2),
                  response_model=SCHEMA, metadata={"k": 1}),
        BatchItem(id="b", messages=[Message.user("b")], settings=RequestSettings(temperature=0.8, max_tokens=7)),
    ]
    async with llm:
        requests, manifest = await serialize_items(llm._build_request_body, items, max_requests=10)
    by_id = {r.custom_id: r.body for r in requests}
    assert by_id["a"]["temperature"] == 0.2 and "max_tokens" not in by_id["a"]
    assert by_id["a"]["response_format"]["json_schema"]["name"] == "Verdict"
    assert by_id["b"]["temperature"] == 0.8 and by_id["b"]["max_tokens"] == 7
    assert "response_format" not in by_id["b"]
    assert manifest["a"] == {"metadata": {"k": 1}, "response_schema": SCHEMA}
    assert manifest["b"] == {"metadata": {}}


async def test_dict_schema_manifest_round_trip_validates_without_items(tmp_path: Path) -> None:
    handler = FakeOpenAIBatch(content_for=lambda cid: '{"label": "persisted"}')
    llm = LLMClient(make_config(), client=mock_client(handler), owns_client=True)
    items = [BatchItem(id="a", messages=[Message.user("q")], response_model=SCHEMA)]
    path = tmp_path / "job.json"
    async with llm:
        job = await llm.submit_batch(items)
        job.save(path)
        reloaded = BatchJob.load(path)
        assert reloaded.response_schema_for("a") == SCHEMA
        results = [r async for r in llm.fetch_batch_results(reloaded)]  # no items
    assert results[0].ok and results[0].response.parsed == {"label": "persisted"}


async def test_dict_schema_invalid_json_in_provider_batch_is_error_result() -> None:
    handler = FakeOpenAIBatch(content_for=lambda cid: "nope")
    llm = LLMClient(make_config(), client=mock_client(handler), owns_client=True)
    items = [BatchItem(id="a", messages=[Message.user("q")], response_model=SCHEMA)]
    async with llm:
        job = await llm.submit_batch(items)
        results = [r async for r in llm.fetch_batch_results(job)]
    assert isinstance(results[0].error, ValidationError)


def test_old_job_file_without_response_schema_still_loads(tmp_path: Path) -> None:
    p = tmp_path / "old.json"
    p.write_text(json.dumps({"provider": "openai", "batch_id": "b", "input_file_id": None,
                             "manifest": {"x": {"metadata": {"i": 1}}}}))
    job = BatchJob.load(p)
    assert job.metadata_for("x") == {"i": 1}
    assert job.response_schema_for("x") is None


# --------------------------------------------------------------- usage / thinking


GEMINI_THINKING = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {"text": "let me think", "thought": True},
                    {"text": '{"label": "ok"}'},
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 5,
        "thoughtsTokenCount": 40,
        "totalTokenCount": 55,
    },
}


async def test_gemini_thinking_tokens_counted_and_thought_text_excluded() -> None:
    llm = _gemini_llm(lambda r: httpx.Response(200, json=GEMINI_THINKING))
    async with llm:
        resp = await llm.invoke([Message.user("hi")], response_model=SCHEMA)
    assert resp.text == '{"label": "ok"}'
    assert resp.parsed == {"label": "ok"}
    u = resp.usage
    assert (u.prompt_tokens, u.completion_tokens, u.reasoning_tokens, u.total_tokens) == (10, 5, 40, 55)


def test_gemini_total_computed_when_absent() -> None:
    data = copy.deepcopy(GEMINI_THINKING)
    del data["usageMetadata"]["totalTokenCount"]
    assert GeminiFormatter().extract_usage(data) == (10, 5, 55, 40)


async def test_gemini_streaming_skips_thought_parts_and_counts_thinking() -> None:
    body = gemini_sse(
        [
            {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}]}}]},
            {"candidates": [{"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}],
             "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "thoughtsTokenCount": 9}},
        ]
    )
    llm = _gemini_stream_llm(body)
    async with llm:
        events = [e async for e in llm.invoke_stream([Message.user("q")])]
    end = events[-1]
    assert isinstance(end, StreamEnd)
    assert end.response.text == "answer"
    assert [type(e).__name__ for e in events] == ["TextChunk", "StreamEnd"]
    assert end.response.usage.reasoning_tokens == 9
    assert end.response.usage.total_tokens == 14


def test_openai_reasoning_tokens_parsed() -> None:
    data = chat_response()
    data["usage"] = {
        "prompt_tokens": 5,
        "completion_tokens": 30,
        "total_tokens": 35,
        "completion_tokens_details": {"reasoning_tokens": 22},
    }
    fmt = OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE)
    assert fmt.extract_usage(data) == (5, 30, 35, 22)
    assert fmt.extract_usage(chat_response())[3] == 0


async def test_openai_stream_usage_reasoning_tokens() -> None:
    chunks = [
        {"model": "m", "choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 6, "total_tokens": 7,
                                   "completion_tokens_details": {"reasoning_tokens": 4}}},
    ]
    body = ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n").encode()
    llm = LLMClient(make_config(), client=stream_mock_client(body), owns_client=True)
    async with llm:
        events = [e async for e in llm.invoke_stream([Message.user("q")])]
    assert events[-1].response.usage.reasoning_tokens == 4


async def test_anthropic_thinking_blocks_excluded_from_text() -> None:
    data = {
        "content": [
            {"type": "thinking", "thinking": "secret", "signature": "sig"},
            {"type": "redacted_thinking", "data": "xxx"},
            {"type": "text", "text": '{"label": "ok"}'},
        ],
        "usage": {"input_tokens": 4, "output_tokens": 20},
        "stop_reason": "end_turn",
    }
    llm = _anthropic_llm(lambda r: httpx.Response(200, json=data))
    async with llm:
        resp = await llm.invoke([Message.user("hi")], response_model=SCHEMA)
    assert resp.text == '{"label": "ok"}'
    assert resp.parsed == {"label": "ok"}
    assert resp.usage.reasoning_tokens == 0 and resp.usage.total_tokens == 24


async def test_anthropic_stream_ignores_thinking_events() -> None:
    events = [
        {"type": "message_start", "message": {"model": "c", "usage": {"input_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "s"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "done"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 8}},
        {"type": "message_stop"},
    ]
    llm = _anthropic_llm(client=stream_mock_client(anthropic_sse(events)))
    async with llm:
        out = [e async for e in llm.invoke_stream([Message.user("q")])]
    assert out[-1].response.text == "done"
    assert [type(e).__name__ for e in out] == ["TextChunk", "StreamEnd"]


def test_ledger_and_jsonl_carry_reasoning_tokens() -> None:
    ledger = UsageLedger()
    for _ in range(2):
        ledger.add(
            BatchResult(
                id="x",
                response=LLMResponse(text="t", usage=TokenUsage(1, 2, 13, reasoning_tokens=10)),
            )
        )
    summary = ledger.to_summary()
    assert summary.reasoning_tokens == 20 and summary.total_tokens == 26

    from llm_kit.batch.jsonl import _usage_to_dict

    assert _usage_to_dict(TokenUsage(1, 2, 13, 10))["reasoning_tokens"] == 10


def test_cost_prices_gemini_thinking_as_output_without_double_counting() -> None:
    gem = GeminiFormatter().extract_usage(GEMINI_THINKING)  # 10 / 5 / 55 / 40
    oai = OpenAIFormatter(structured_output_style=StructuredOutputStyle.NONE).extract_usage(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 45, "total_tokens": 55,
                   "completion_tokens_details": {"reasoning_tokens": 40}}}
    )
    ant = AnthropicFormatter().extract_usage({"usage": {"input_tokens": 10, "output_tokens": 45}})
    ledger = UsageLedger()
    for p, c, t, r in (gem, oai, ant):
        ledger.add(BatchResult(id="x", response=LLMResponse(text="t", usage=TokenUsage(p, c, t, r))))
    summary = ledger.to_summary(prompt_cost_per_1m=1_000_000, completion_cost_per_1m=1_000_000)
    assert TokenUsage(*gem).output_tokens == 45  # 5 candidates + 40 thoughts
    assert summary.output_tokens == 135  # 45 each: same billable output for all three
    assert summary.estimated_cost == 30 + 135
    # missing total_tokens must not undercount
    assert TokenUsage(10, 7, 0).output_tokens == 7
