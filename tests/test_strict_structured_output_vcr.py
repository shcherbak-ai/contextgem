# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""Record strict concept extraction against the supported provider APIs."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from vcr.cassette import Cassette

from contextgem import Document, DocumentLLM, StringConcept
from tests.base import BaseTest


def _strict_extraction_fixture(
    model: str, api_key: str | None
) -> tuple[DocumentLLM, Document]:
    """Build the same extraction for both providers with a bounded token budget."""
    document = Document(raw_text="The project code is ORCHID-42.")
    document.add_concepts(
        [
            StringConcept(
                name="Project code",
                description="The exact project code, without the ending period.",
                singular_occurrence=True,
                add_justifications=True,
                add_references=True,
            )
        ]
    )
    llm = DocumentLLM(
        model=model,
        api_key=api_key,
        auto_pricing=False,
        max_tokens=1024,
        max_retries_failed_request=0,
        max_retries_invalid_data=0,
        strict_structured_output=True,
    )
    return llm, document


def _extract_and_get_request(
    model: str, api_key: str | None, cassette: Cassette
) -> dict[str, Any]:
    """Check the extracted item and return the request body recorded by VCR."""
    llm, document = _strict_extraction_fixture(model, api_key)
    llm.extract_concepts_from_document(document)

    (item,) = document.concepts[0].extracted_items
    assert item.value == "ORCHID-42"
    assert item.justification and item.justification.strip()
    assert [document.get_paragraph_index(p) for p in item.reference_paragraphs] == [0]

    (request,) = cassette.requests
    body = json.loads(request.body)
    assert body["model"] == model.split("/", 1)[1]
    return body


def _assert_closed_extraction_envelope(schema: dict[str, Any]) -> None:
    """Verify the provider receives the closed extraction envelope."""
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["items"]
    assert schema["properties"]["items"]["type"] == "array"


class TestStrictStructuredOutputVCR(BaseTest):
    """Provider integration tests for native JSON Schema extraction."""

    @pytest.mark.vcr
    def test_openai_strict_extraction(self, vcr: Cassette) -> None:
        """OpenAI accepts the strict format and returns the expected item."""
        body = _extract_and_get_request(
            "openai/gpt-6-luna", os.getenv("CONTEXTGEM_OPENAI_API_KEY"), vcr
        )
        response_format = body["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        _assert_closed_extraction_envelope(response_format["json_schema"]["schema"])
