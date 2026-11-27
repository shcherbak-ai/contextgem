# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Tests for strict JSON Schema extraction responses."""

from __future__ import annotations

import asyncio
import json

import pytest
from litellm import UnsupportedParamsError

from contextgem import (
    BooleanConcept,
    DateConcept,
    Document,
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


def test_strict_unsupported_params_are_not_retried_without_response_format(
    monkeypatch,
) -> None:
    """A strict response-format error is surfaced, never retried with dropped params."""
    from contextgem.internal.exceptions import ModelAPIError

    llm = DocumentLLM(model="openai/gpt-4o-mini", auto_pricing=False)
    response_format = _strict_format(with_metadata=False)
    drop_params_values: list[bool] = []

    async def reject_strict_format(**kwargs):
        """Simulate a provider rejecting a strict response format."""
        drop_params_values.append(kwargs["drop_params"])
        raise UnsupportedParamsError(
            message="response_format is not supported",
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
    monkeypatch.setattr(llm, "_send_messages_with_tools", reject_strict_format)

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
