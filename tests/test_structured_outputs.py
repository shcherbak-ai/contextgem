# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict JSON Schema extraction responses."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from litellm import UnsupportedParamsError
from litellm.types.decisions import (
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
)

from contextgem import (
    BooleanConcept,
    DateConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    JsonObjectConcept,
    LabelConcept,
    NumericalConcept,
    StringConcept,
)
from contextgem.internal.base import llms as llms_module
from contextgem.internal.data_models import _ModelCall, _ModelUsage
from contextgem.internal.llm_output_structs.strict_structured_output import (
    _build_strict_extraction_response_format,
    _unwrap_strict_extraction_response,
)


def _strict_format(*, with_metadata: bool) -> dict:
    """Build the schema used by the concept-extraction tests."""
    return _build_strict_extraction_response_format(
        extracted_item_type="concept",
        concepts=[
            StringConcept(name="title", description="A title"),
            BooleanConcept(name="enabled", description="Whether it is enabled"),
        ],
        identifiers=["C0", "C1"],
        add_justifications=with_metadata,
        add_references=with_metadata,
        reference_depth="paragraphs",
    )


def _assert_no_unconstrained_schema(schema: dict) -> None:
    """Assert that the generated schema has no unconstrained object nodes."""
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        assert schema.get("additionalProperties") is False
        assert set(schema.get("required", [])) == set(properties)
    for key, value in schema.items():
        if isinstance(value, dict):
            assert value, f"Unconstrained schema at {key}"
            _assert_no_unconstrained_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _assert_no_unconstrained_schema(item)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_strict_concept_schema_is_wrapped_and_typed(with_metadata: bool) -> None:
    """The strict schema wraps the list and constrains each concept value."""
    response_format = _strict_format(with_metadata=with_metadata)

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["type"] == "object"
    assert schema["required"] == ["items"]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["items"]["type"] == "array"

    concept_model = schema["$defs"]["ConceptModel"]
    assert concept_model["properties"]["concept_id"]["enum"] == ["C0", "C1"]
    if with_metadata:
        value_schema = schema["$defs"]["ExtractedItemModel"]["properties"]["value"]
    else:
        value_schema = concept_model["properties"]["extracted_items"]["items"]
    assert value_schema == {"anyOf": [{"type": "string"}, {"type": "boolean"}]}
    _assert_no_unconstrained_schema(schema)

    if with_metadata:
        assert (
            "reference_paragraph_ids"
            in schema["$defs"]["ExtractedItemModel"]["properties"]
        )
        assert "justification" in schema["$defs"]["ExtractedItemModel"]["properties"]
    else:
        assert "ExtractedItemModel" not in schema["$defs"]


def test_strict_schema_is_valid_json_for_litellm() -> None:
    """LiteLLM can serialize the response-format dictionary."""
    response_format = _strict_format(with_metadata=True)

    # LiteLLM accepts a JSON-serializable OpenAI response_format dictionary.
    assert json.loads(json.dumps(response_format)) == response_format


def test_strict_schema_supports_contextgem_concept_value_types() -> None:
    """Known ContextGem value types produce closed, constrained schemas."""
    concepts = [
        DateConcept(name="date", description="A date"),
        LabelConcept(
            name="kind",
            description="A kind",
            labels=["one", "two"],
            classification_type="multi_class",
        ),
        NumericalConcept(name="count", description="A count", numeric_type="int"),
        JsonObjectConcept(
            name="details",
            description="Fixed details",
            structure={"title": str, "description": str, "default": str, "count": int},
        ),
    ]
    response_format = _build_strict_extraction_response_format(
        extracted_item_type="concept",
        concepts=concepts,
        identifiers=["C0", "C1", "C2", "C3"],
        add_justifications=False,
        add_references=False,
        reference_depth="paragraphs",
    )

    schema = response_format["json_schema"]["schema"]
    _assert_no_unconstrained_schema(schema)
    value_schema = schema["$defs"]["ConceptModel"]["properties"]["extracted_items"][
        "items"
    ]
    assert value_schema["anyOf"]
    assert {item.get("type") for item in value_schema["anyOf"]} >= {
        "string",
        "integer",
    }
    object_properties = [
        set(definition["properties"])
        for definition in schema["$defs"].values()
        if isinstance(definition.get("properties"), dict)
        and set(definition["properties"])
        == {"title", "description", "default", "count"}
    ]
    assert object_properties == [
        {
            "title",
            "description",
            "default",
            "count",
        }
    ]


@pytest.mark.parametrize("with_metadata", [False, True])
def test_strict_schema_does_not_constrain_user_concept_id(with_metadata: bool) -> None:
    """User data fields named concept_id are not ContextGem extraction IDs."""
    concept = JsonObjectConcept(
        name="details",
        description="Details",
        structure={"concept_id": str, "asset": str},
    )
    response_format = _build_strict_extraction_response_format(
        extracted_item_type="concept",
        concepts=[concept],
        identifiers=["C0"],
        add_justifications=with_metadata,
        add_references=with_metadata,
        reference_depth="paragraphs",
    )

    schema = response_format["json_schema"]["schema"]
    if with_metadata:
        user_value_schema = schema["$defs"]["ExtractedItemModel"]["properties"]["value"]
    else:
        user_value_schema = schema["$defs"]["ConceptModel"]["properties"][
            "extracted_items"
        ]["items"]

    assert user_value_schema["properties"]["concept_id"] == {"type": "string"}
    assert schema["$defs"]["ConceptModel"]["properties"]["concept_id"]["enum"] == ["C0"]


def test_strict_aspect_schema_constrains_ids() -> None:
    """Aspect extraction uses the same strict object envelope and ID bounds."""
    response_format = _build_strict_extraction_response_format(
        extracted_item_type="aspect",
        concepts=[],
        identifiers=["A0", "A1"],
        add_justifications=False,
        add_references=True,
        reference_depth="paragraphs",
    )

    schema = response_format["json_schema"]["schema"]
    _assert_no_unconstrained_schema(schema)
    assert schema["properties"]["items"]["items"]["$ref"].startswith("#/$defs/")
    aspect = schema["$defs"]["AspectModel"]
    assert aspect["properties"]["aspect_id"]["enum"] == ["A0", "A1"]


@pytest.mark.parametrize(
    "structure, message",
    [
        (
            {"metadata": dict[str, str]},
            "does not support arbitrary object keys",
        ),
        (
            {"optional": str | None},
            "requires every object field to be required",
        ),
    ],
)
def test_strict_schema_rejects_unsupported_json_object_shapes(
    structure: dict, message: str
) -> None:
    """Unsupported map and optional-field schemas fail before an API call."""
    concept = JsonObjectConcept(
        name="details", description="Details", structure=structure
    )

    with pytest.raises(ValueError, match=message):
        _build_strict_extraction_response_format(
            extracted_item_type="concept",
            concepts=[concept],
            identifiers=["C0"],
            add_justifications=False,
            add_references=False,
            reference_depth="paragraphs",
        )


def test_strict_envelope_is_unwrapped_before_contextgem_validation() -> None:
    """Only the exact strict response envelope is unwrapped."""
    parsed = {"items": [{"concept_id": "C0", "extracted_items": ["A title"]}]}

    assert _unwrap_strict_extraction_response(parsed) == parsed["items"]
    assert _unwrap_strict_extraction_response([parsed]) is None
    assert _unwrap_strict_extraction_response({"items": [], "extra": True}) is None


def test_strict_aspect_schema_uses_prompt_ids() -> None:
    """The strict aspect schema restricts IDs to those shown in the prompt."""
    response_format = _build_strict_extraction_response_format(
        extracted_item_type="aspect",
        concepts=[],
        identifiers=["A0", "A1"],
        add_justifications=False,
        add_references=True,
        reference_depth="sentences",
    )

    schema = response_format["json_schema"]["schema"]
    aspect_model = schema["$defs"]["AspectModel"]
    assert aspect_model["properties"]["aspect_id"]["enum"] == ["A0", "A1"]
    assert "paragraphs" in aspect_model["properties"]


def test_extraction_prompts_switch_to_the_strict_object_envelope() -> None:
    """Strict prompts describe the object envelope; default prompts stay arrays."""
    llm = DocumentLLM(model="openai/gpt-4o-mini", auto_pricing=False)

    for prompt in (llm._extract_concept_items_prompt, llm._extract_aspect_items_prompt):
        default = prompt.render(
            strict_structured_output=False,
            structured_output_prefix="[",
            structured_output_suffix="]",
        )
        strict = prompt.render(
            strict_structured_output=True,
            structured_output_prefix='{"items": [',
            structured_output_suffix="]}",
        )
        assert '"items": [' not in default
        assert '"items": [' in strict
        answer_format = strict[
            strict.index("ANSWER FORMAT (MANDATORY):") : strict.rindex(
                "</answer_format>"
            )
        ]
        assert answer_format.rstrip().endswith("]}")


def test_query_llm_keeps_response_format_in_litellm_request(monkeypatch) -> None:
    """The per-call response format reaches the LLM request configuration."""
    llm = DocumentLLM(model="openai/gpt-4o-mini", auto_pricing=False)
    response_format = _strict_format(with_metadata=False)
    captured: dict = {}

    async def capture_send(**kwargs):
        """Capture the request configuration passed to the send helper."""
        captured.update(kwargs["request_config"])
        return "{}", False

    monkeypatch.setattr(llm, "_validate_input_tokens", lambda messages: None)
    monkeypatch.setattr(llm, "_validate_output_tokens", lambda: None)
    monkeypatch.setattr(llm, "_get_thinking_blocks_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        llm,
        "_messages_to_request",
        lambda messages, thinking_blocks_key: [{"role": "user", "content": "test"}],
    )
    monkeypatch.setattr(llm, "_send_messages_with_tools", capture_send)

    asyncio.run(
        llm._query_llm(
            messages=[llm._build_user_message("test")],
            model_call_obj=_ModelCall(prompt_kwargs={}, prompt="test"),
            response_format=response_format,
        )
    )

    assert captured["response_format"] == response_format


def test_strict_response_format_rejects_global_litellm_drop_params(monkeypatch) -> None:
    """Strict output fails before sending when LiteLLM globally drops params."""
    llm = DocumentLLM(model="openai/gpt-4o-mini", auto_pricing=False)
    monkeypatch.setattr(llms_module.litellm, "drop_params", True)

    async def unexpected_send(**kwargs):
        raise AssertionError("strict request must fail before reaching LiteLLM")

    monkeypatch.setattr(llm, "_send_messages_with_tools", unexpected_send)

    with pytest.raises(ValueError, match="global `drop_params` setting"):
        asyncio.run(
            llm._query_llm(
                messages=[llm._build_user_message("test")],
                model_call_obj=_ModelCall(prompt_kwargs={}, prompt="test"),
                response_format=_strict_format(with_metadata=False),
            )
        )


@pytest.mark.parametrize("strict", [False, True])
def test_gpt5_mini_discards_rejected_temperature(monkeypatch, strict: bool) -> None:
    """GPT-5 mini reaches HTTP without temperature and retains strict output if set."""
    llm = DocumentLLM(
        model="openai/gpt-5-mini",
        api_key="DUMMY",
        temperature=0.3,
        strict_structured_output=strict,
        auto_pricing=False,
    )
    document = Document(raw_text="The title is A title.")
    document.add_concepts([StringConcept(name="Title", description="The title")])
    requests: list[dict[str, Any]] = []
    extracted_items = [{"concept_id": "C0", "extracted_items": ["A title"]}]

    async def send_stub(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        """Capture the body after LiteLLM and the OpenAI SDK transform it."""
        assert request.url.host == "api.openai.com"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            request=request,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-5-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {"items": extracted_items}
                                if strict
                                else extracted_items
                            ),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "send", send_stub)
    llm.extract_concepts_from_document(document)

    assert document.concepts[0].extracted_items[0].value == "A title"
    (request,) = requests
    assert "temperature" not in request
    if strict:
        assert request["response_format"] == _build_strict_extraction_response_format(
            extracted_item_type="concept",
            concepts=document.concepts,
            identifiers=["C0"],
            add_justifications=False,
            add_references=False,
            reference_depth="paragraphs",
        )
    else:
        assert "response_format" not in request
    (record,) = llm.get_usage()
    assert (record.usage.input, record.usage.output) == (10, 5)
    assert len(record.usage.calls) == 1


@pytest.mark.parametrize("reject_retry", [False, True])
def test_strict_retry_drops_only_rejected_sampling_params(
    monkeypatch, reject_retry: bool
) -> None:
    """LiteLLM's list of unsupported kwargs permits a single selective retry."""
    llm = DocumentLLM(
        model="openai/gpt-4o-mini",
        auto_pricing=False,
        temperature=0.3,
        top_p=0.8,
        seed=13,
        max_tokens=128,
    )
    request = llm._build_request_config()
    request["response_format"] = _strict_format(with_metadata=False)
    calls: list[dict[str, Any]] = []
    answer = object()

    async def completion_stub(**kwargs: Any) -> object:
        """Reject a list of sampling options using the pinned LiteLLM message."""
        calls.append(kwargs)
        if len(calls) == 1:
            raise UnsupportedParamsError(
                message=(
                    "openai does not support parameters: ['temperature', 'seed'], "
                    "for model=gpt-4o-mini. To drop these, set `litellm.drop_params=True`"
                ),
                llm_provider="openai",
                model="gpt-4o-mini",
            )
        if reject_retry:
            raise UnsupportedParamsError(
                message=(
                    "openai does not support parameters: ['top_p'], "
                    "for model=gpt-4o-mini. To drop these, set `litellm.drop_params=True`"
                ),
                llm_provider="openai",
                model="gpt-4o-mini",
            )
        return answer

    monkeypatch.setattr(llms_module.litellm, "acompletion", completion_stub)
    completion = llm._send_non_streaming_completion(
        request_config=request,
        messages_payload=[{"role": "user", "content": "test"}],
    )
    if reject_retry:
        with pytest.raises(UnsupportedParamsError, match="top_p"):
            asyncio.run(completion)
    else:
        assert asyncio.run(completion) is answer

    first, second = calls
    assert second == {
        key: value for key, value in first.items() if key not in {"temperature", "seed"}
    }
    assert second["drop_params"] is False
    assert second["max_tokens"] == 128
    assert second["response_format"] == request["response_format"]


@pytest.mark.parametrize(
    "message",
    [
        "response_format is not supported",
        (
            "openai does not support parameters: ['temperature', 'response_format'], "
            "for model=gpt-4o-mini."
        ),
        (
            "openai does not support parameters: ['temperature', 'max_tokens'], "
            "for model=gpt-4o-mini."
        ),
        "Unsupported combination of request parameters",
    ],
)
def test_strict_unsupported_params_are_not_retried_without_response_format(
    monkeypatch,
    message: str,
) -> None:
    """Fail when a rejected parameter is required or cannot be identified."""
    from contextgem.internal.exceptions import ModelAPIError

    llm = DocumentLLM(
        model="openai/gpt-4o-mini", auto_pricing=False, temperature=0.3, max_tokens=128
    )
    response_format = _strict_format(with_metadata=False)
    drop_params_values: list[bool] = []

    async def reject_strict_format(**kwargs):
        """Simulate a provider rejecting a strict response format."""
        drop_params_values.append(kwargs["drop_params"])
        raise UnsupportedParamsError(
            message=message,
            llm_provider="openai",
            model="gpt-4o-mini",
        )

    monkeypatch.setattr(llm, "_validate_input_tokens", lambda messages: None)
    monkeypatch.setattr(llm, "_validate_output_tokens", lambda: None)
    monkeypatch.setattr(llm, "_get_thinking_blocks_key", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        llm,
        "_messages_to_request",
        lambda messages, thinking_blocks_key: [{"role": "user", "content": "test"}],
    )
    monkeypatch.setattr(llms_module.litellm, "acompletion", reject_strict_format)

    with pytest.raises(ModelAPIError):
        asyncio.run(
            llm._query_llm(
                messages=[llm._build_user_message("test")],
                model_call_obj=_ModelCall(prompt_kwargs={}, prompt="test"),
                response_format=response_format,
            )
        )

    assert drop_params_values == [False]


def test_litellm_completion_receives_response_format(monkeypatch) -> None:
    """The completion request passes the response format to LiteLLM unchanged."""
    llm = DocumentLLM(model="openai/gpt-4o-mini", auto_pricing=False)
    response_format = _strict_format(with_metadata=False)
    captured: dict = {}

    async def capture_completion(**kwargs):
        """Capture the arguments passed to LiteLLM's completion call."""
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llms_module.litellm, "acompletion", capture_completion)

    asyncio.run(
        llm._send_non_streaming_completion(
            request_config={"model": llm.model, "response_format": response_format},
            messages_payload=[{"role": "user", "content": "test"}],
        )
    )

    assert captured["response_format"] == response_format


def test_strict_extraction_unwraps_and_validates_with_contextgem(monkeypatch) -> None:
    """ContextGem unwraps strict output and performs its existing validation."""
    llm = DocumentLLM(
        model="openai/gpt-4o-mini",
        auto_pricing=False,
        strict_structured_output=True,
        max_retries_invalid_data=0,
    )
    document = Document(raw_text="The title is A title.")
    document.add_concepts([StringConcept(name="Title", description="The title")])
    captured: dict = {}

    async def fake_query(**kwargs):
        """Return a valid strict envelope without making a model request."""
        captured.update(kwargs)
        return (
            '{"items":[{"concept_id":"C0","extracted_items":["A title"]}]}',
            _ModelUsage(),
        )

    monkeypatch.setattr(llm, "_query_llm", fake_query)
    llm.extract_concepts_from_document(document, overwrite_existing=True)

    assert captured["response_format"]["json_schema"]["strict"] is True
    assert document.concepts[0].extracted_items[0].value == "A title"


def test_fallback_llm_must_match_strict_structured_output_setting() -> None:
    """A fallback cannot silently switch off strict response formatting."""
    main = DocumentLLM(
        model="openai/gpt-4o-mini",
        auto_pricing=False,
        strict_structured_output=True,
    )
    fallback = DocumentLLM(model="openai/gpt-4o-mini", auto_pricing=False)

    with pytest.raises(ValueError, match="same `strict_structured_output`"):
        main._validate_kind_specific_fallback(fallback)


def test_strict_extraction_uses_decision_model_fallback(monkeypatch) -> None:
    """A decision model can still extract a boolean after a strict LLM call fails."""
    fallback = DocumentDecisionModel(
        model="typesafe/jev-latest", api_key="DUMMY", is_fallback=True
    )
    llm = DocumentLLM(
        model="openai/gpt-4o-mini",
        api_key="DUMMY",
        auto_pricing=False,
        strict_structured_output=True,
        fallback_models=[fallback],
    )
    document = Document(raw_text="The application is enabled.")
    document.add_concepts(
        [BooleanConcept(name="Enabled", description="The application is enabled.")]
    )
    requests: list[str] = []

    async def reject_strict_format(**kwargs: Any) -> None:
        """Reject the schema so ContextGem proceeds to the decision endpoint."""
        requests.append(kwargs["model"])
        assert kwargs["response_format"]["json_schema"]["strict"] is True
        raise UnsupportedParamsError(
            message="response_format is not supported",
            llm_provider="openai",
            model="gpt-4o-mini",
        )

    async def decisions_stub(**kwargs: Any) -> OpenAIDecisionResponse:
        """Return the expected predicate answer using the Decisions API's types."""
        requests.append(kwargs["model"])
        assert "response_format" not in kwargs
        (question,) = kwargs["questions"]
        assert question["type"] == "predicate"
        return OpenAIDecisionResponse(
            model="jev-1.13.0",
            answers=(OpenAIPredicateAnswer(name=question["name"], probability=0.93),),
            usage=OpenAIDecisionUsage(
                input_tokens=100, output_tokens=2, total_tokens=102
            ),
        )

    monkeypatch.setattr(llms_module.litellm, "acompletion", reject_strict_format)
    monkeypatch.setattr(llms_module.litellm, "adecisions", decisions_stub)
    llm.extract_concepts_from_document(document)

    assert requests == ["openai/gpt-4o-mini", "typesafe/jev-latest"]
    assert document.concepts[0].extracted_items[0].value is True
