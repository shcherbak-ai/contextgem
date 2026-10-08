#
# ContextGem
#
# Copyright 2025 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Module defining tests for model groups, model roles and fallback models.
"""

from __future__ import annotations

import json
import types
from typing import Any
from unittest.mock import patch

import pytest

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.llms import _DocumentLLM
from contextgem.internal.exceptions import ModelAPIError
from contextgem.public import (
    Aspect,
    BooleanConcept,
    ChatSession,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    StringConcept,
)
from tests.base import BaseTest
from tests.utils import get_test_document_text


class TestModelGroups(BaseTest):
    """
    Tests for model groups, the routing of aspects and concepts to models
    by role, and fallback models.
    """

    def test_init_model_group(self):
        """
        Tests the behavior of the `DocumentModelGroup` class initialization.
        """

        # Invalid params
        with pytest.raises(ValueError):
            DocumentModelGroup()
        with pytest.raises(ValueError):
            DocumentModelGroup(models=[])
        with pytest.raises(ValueError):
            DocumentModelGroup(models=[self.llm_reasoner])
        with pytest.raises(ValueError):
            DocumentModelGroup(models=[1, True])
        # Duplicate roles
        with pytest.raises(ValueError):
            DocumentModelGroup(
                models=[
                    self.llm_extractor,
                    self.llm_extractor,
                    self.llm_reasoner,
                ]
            )
        with pytest.raises(ValueError):
            DocumentModelGroup(
                models=[self.llm_extractor, self.llm_reasoner],
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            # Non-consistent output languages of the LLMs in the group
            DocumentModelGroup(
                models=[
                    self.llm_extractor,
                    self.llm_extractor_non_eng,  # output lang set to "adapt"
                ],
                output_language="en",
            )
        document_model_group = DocumentModelGroup(
            models=[self.llm_extractor, self.llm_reasoner]
        )
        assert document_model_group.is_group is True
        # Validate assignment
        with pytest.raises(ValueError):
            document_model_group.models = [self.llm_extractor, self.llm_extractor]
        assert document_model_group.models == [
            self.llm_extractor,
            self.llm_reasoner,
        ]

        # No serialization for LLM instances
        with pytest.raises(NotImplementedError):
            document_model_group.model_dump()
        with pytest.raises(NotImplementedError):
            document_model_group.model_dump_json()

    def test_model_roles_and_concept_modality(self):
        """
        Tests for routing aspects/concepts to LLMs by the free-form model roles, and for
        extracting concepts from the content specified by their modality.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        def create_llm(model: str, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with params that keep the stubbed calls deterministic.
            """
            return DocumentLLM(
                model=model,
                api_key="test_api_key",
                system_message="",
                max_retries_invalid_data=0,
                **kwargs,
            )

        def create_document() -> Document:
            """
            Creates a document with text and an image, as well as with aspects
            and concepts that have different model roles and modalities.
            """
            return Document(
                raw_text="Non-disclosure agreement\n\nThe parties agree as follows.",
                images=[self.test_img_png_invoice],
                aspects=[
                    Aspect(name="Parties", description="Contract parties"),
                    Aspect(
                        name="Term",
                        description="Contract term",
                        model_role="complex",
                        concepts=[
                            StringConcept(
                                name="Duration",
                                description="Contract duration",
                                model_role="complex",
                            )
                        ],
                    ),
                ],
                concepts=[
                    StringConcept(name="Title", description="Contract title"),
                    StringConcept(
                        name="Type", description="Contract type", model_role="complex"
                    ),
                    StringConcept(
                        name="Stamp",
                        description="Text of the stamp",
                        model_role="vision",
                        modality="image",
                    ),
                    StringConcept(
                        name="Date", description="Contract date", modality="multimodal"
                    ),
                ],
            )

        # (model, content type, names of the aspects/concepts in the call)
        llm_calls: list[tuple[str, str, list[str]]] = []

        def stub_model_api(llm: DocumentLLM) -> None:
            """
            Stubs the LLM API call, recording the aspects/concepts sent to the LLM.
            """

            async def _send_messages_stub(self, *, model_call_obj: Any, **kwargs: Any):
                """
                Simulates a successful LLM API call, which returns an extracted item
                for each aspect/concept in the prompt.

                Same params as `llm._send_messages_with_tools`.
                """
                prompt_kwargs = model_call_obj.prompt_kwargs
                if "aspects" in prompt_kwargs:
                    instances = prompt_kwargs["aspects"]
                    content_type = "text"
                    answer = [
                        {"aspect_id": f"A{i}", "paragraph_ids": ["P0"]}
                        for i in range(len(instances))
                    ]
                else:
                    instances = prompt_kwargs["concepts"]
                    content_type = prompt_kwargs["data_type"]
                    answer = [
                        {"concept_id": f"C{i}", "extracted_items": ["Item"]}
                        for i in range(len(instances))
                    ]
                llm_calls.append(
                    (self.model, content_type, [i.name for i in instances])
                )
                return json.dumps(answer), False

            llm._send_messages_with_tools = types.MethodType(_send_messages_stub, llm)  # ty: ignore[invalid-assignment]

        def check_all_extracted(document: Document) -> None:
            """
            Checks that all aspects and concepts of the document are extracted.
            """
            for aspect in document.aspects:
                assert aspect.extracted_items
                assert all(i.extracted_items for i in aspect.concepts)
            assert all(i.extracted_items for i in document.concepts)
            # A multimodal concept is extracted from both the text and the images
            assert len(document.get_concept_by_name("Date").extracted_items) == 2

        default_model = "openai/gpt-6-luna"
        complex_model = "openai/gpt-6.1-sol"
        vision_model = "openai/gpt-5.6-luna"

        # === Roles are free-form labels, and modality is a param of a concept ===
        llm_no_role = create_llm(default_model)
        assert llm_no_role.role is None
        assert llm_no_role.list_roles == [None]
        llm_complex = create_llm(complex_model, role=" complex ")
        assert llm_complex.role == "complex"  # stripped
        llm_complex.role = "advanced"
        llm_complex.role = "complex"
        llm_vision = create_llm(vision_model, role="vision")
        # A role is a non-empty string of up to 64 printable characters
        max_length_role = "r" * 64
        assert create_llm(default_model, role=max_length_role).role == max_length_role
        for valid_role in [
            "Legal review (complex)",
            "\u042e\u0440\u0438\u0441\u0442",
            "\U0001f9e0",
        ]:
            assert create_llm(default_model, role=valid_role).role == valid_role
        for invalid_role in [
            "",
            "  ",
            1,
            max_length_role + "r",  # too long
            "line\nbreak",
            "tab\tinside",
            "control\x00character",
            "zero\u200bwidth",  # invisible character
        ]:
            with pytest.raises(ValueError):
                create_llm(default_model, role=invalid_role)
            with pytest.raises(ValueError):
                Aspect(name="Parties", description="Parties", model_role=invalid_role)  # ty: ignore[invalid-argument-type]
            with pytest.raises(ValueError):
                StringConcept(
                    name="Title",
                    description="Title",
                    model_role=invalid_role,  # ty: ignore[invalid-argument-type]
                )
        concept = StringConcept(name="Title", description="Contract title")
        assert concept.model_role is None
        assert concept.modality == "text"
        concept.model_role = "complex"
        concept.modality = "multimodal"
        assert (concept.model_role, concept.modality) == ("complex", "multimodal")
        with pytest.raises(ValueError):
            concept.modality = "video"  # ty: ignore[invalid-assignment]
        # Concepts extracted from images do not support references
        with pytest.raises(ValueError, match="do not support references"):
            StringConcept(
                name="Title",
                description="Contract title",
                modality="image",
                add_references=True,
            )
        concept.add_references = True
        with pytest.raises(ValueError, match="do not support references"):
            concept.modality = "image"
        assert concept.modality == "multimodal"  # no change due to invalid assignment
        # Roles assigned to the instances of a document, including sub-aspects
        document = create_document()
        document.aspects[0].aspects = [
            Aspect(name="Signatories", description="Signatories", model_role="deep")
        ]
        assert document.model_roles == {None, "complex", "vision", "deep"}
        assert document.aspects[1].model_roles == {"complex"}

        # === Standalone LLM extracts all aspects/concepts, regardless of their roles ===
        for standalone_llm in [llm_no_role, llm_complex]:
            stub_model_api(standalone_llm)
            llm_calls.clear()
            document = create_document()
            standalone_llm.extract_all(document)
            check_all_extracted(document)
            assert sorted(llm_calls) == sorted(
                [
                    (standalone_llm.model, "text", ["Parties", "Term"]),
                    (standalone_llm.model, "text", ["Duration"]),
                    (standalone_llm.model, "text", ["Title", "Type"]),
                    (standalone_llm.model, "image", ["Stamp"]),
                    (standalone_llm.model, "text", ["Date"]),
                    (standalone_llm.model, "image", ["Date"]),
                ]
            )

        # === model group routes each aspect/concept to the LLM with the same role ===
        stub_model_api(llm_vision)
        model_group = DocumentModelGroup(models=[llm_no_role, llm_complex, llm_vision])
        assert model_group.list_roles == [None, "complex", "vision"]
        llm_calls.clear()
        document = create_document()
        model_group.extract_all(document)
        check_all_extracted(document)
        assert sorted(llm_calls) == sorted(
            [
                (default_model, "text", ["Parties"]),
                (default_model, "text", ["Title"]),
                (default_model, "text", ["Date"]),
                (default_model, "image", ["Date"]),
                (complex_model, "text", ["Term"]),
                (complex_model, "text", ["Duration"]),
                (complex_model, "text", ["Type"]),
                (vision_model, "image", ["Stamp"]),
            ]
        )
        assert [i.role for i in model_group.get_usage(model_role="vision")] == [
            "vision"
        ]
        assert [i.role for i in model_group.get_usage()] == [None, "complex", "vision"]

        # Roles must be unique in the group, and only one LLM can have no role
        with pytest.raises(ValueError, match="different roles"):
            DocumentModelGroup(models=[llm_no_role, create_llm(complex_model)])
        with pytest.raises(ValueError, match="different roles"):
            DocumentModelGroup(
                models=[llm_complex, create_llm(default_model, role="complex")]
            )
        # This is also checked when extraction starts, as the role of an LLM
        # can be changed after the LLM is added to a group
        llm_changed_role = create_llm(vision_model, role="changed")
        stub_model_api(llm_changed_role)
        model_group_changed_role = DocumentModelGroup(
            models=[llm_no_role, llm_changed_role]
        )
        document = Document(
            raw_text="Non-disclosure agreement",
            aspects=[Aspect(name="Parties", description="Contract parties")],
            concepts=[StringConcept(name="Title", description="Contract title")],
        )
        llm_changed_role.role = None
        llm_calls.clear()
        for extraction_method in [
            model_group_changed_role.extract_all,
            model_group_changed_role.extract_aspects_from_document,
            model_group_changed_role.extract_concepts_from_document,
        ]:
            with pytest.raises(
                ValueError,
                match=r"different roles.*several LLMs: None \(no role\)",
            ):
                extraction_method(document)
        assert not llm_calls  # no LLM calls are made
        llm_changed_role.role = "changed"
        model_group_changed_role.extract_all(document)
        assert sorted(llm_calls) == sorted(
            [
                (default_model, "text", ["Parties"]),
                (default_model, "text", ["Title"]),
            ]
        )

        # Aspects/concepts that cannot be routed to an LLM in the group are not skipped
        llm_calls.clear()
        model_group_no_vision = DocumentModelGroup(models=[llm_no_role, llm_complex])
        document = create_document()
        with pytest.raises(
            ValueError, match="no LLM in the group has the role: `vision`"
        ):
            model_group_no_vision.extract_all(document)
        with pytest.raises(ValueError, match="`vision`"):
            model_group_no_vision.extract_concepts_from_document(document)
        model_group_no_default = DocumentModelGroup(models=[llm_complex, llm_vision])
        with pytest.raises(ValueError, match=r"None \(no role\)"):
            model_group_no_default.extract_aspects_from_document(document)
        assert not llm_calls  # no LLM calls are made
        # While the routable ones can still be extracted
        routable_concepts = [
            i
            for i in document.concepts
            if i.model_role in model_group_no_vision.list_roles
        ]
        model_group_no_vision.extract_concepts_from_document(
            document, from_concepts=routable_concepts
        )
        assert all(i.extracted_items for i in routable_concepts)
        assert not document.get_concept_by_name("Stamp").extracted_items

        # === Vision support is checked before any LLM calls are made ===
        text_only_model = "cerebras/llama-3.3-70b"
        llm_text_only = create_llm(text_only_model)
        assert not llm_text_only._supports_vision
        stub_model_api(llm_text_only)
        # Concepts cannot be extracted from images by an LLM that does not support vision
        llm_calls.clear()
        document = create_document()
        for extraction_method in [
            llm_text_only.extract_all,
            llm_text_only.extract_concepts_from_document,
        ]:
            with pytest.raises(
                ValueError,
                match=f"^Model `{text_only_model}` does not support vision"
                ".*from images: `Stamp`, `Date`",
            ):
                extraction_method(document)
        assert not llm_calls  # no LLM calls are made
        # While the concepts to be extracted from text can still be extracted
        llm_text_only.extract_concepts_from_document(
            document,
            from_concepts=[i for i in document.concepts if i.modality == "text"],
        )
        assert llm_calls == [(text_only_model, "text", ["Title", "Type"])]
        # A multimodal concept is extracted from images only if the document has them
        document_without_images = Document(
            raw_text="Non-disclosure agreement",
            concepts=[
                StringConcept(
                    name="Date", description="Contract date", modality="multimodal"
                )
            ],
        )
        llm_text_only.extract_all(document_without_images)
        assert document_without_images.concepts[0].extracted_items

        # A fallback LLM is sent the same content as the LLM it replaces,
        # i.e. it must support vision as well
        llm_text_only_fallback = create_llm(text_only_model, is_fallback=True)
        llm_with_text_only_fallback = create_llm(
            default_model, fallback_models=[llm_text_only_fallback]
        )
        stub_model_api(llm_text_only_fallback)
        stub_model_api(llm_with_text_only_fallback)
        llm_calls.clear()
        document = create_document()
        for extraction_method in [
            llm_with_text_only_fallback.extract_all,
            llm_with_text_only_fallback.extract_concepts_from_document,
        ]:
            with pytest.raises(
                ValueError,
                match=f"^Fallback model `{text_only_model}` does not support vision",
            ):
                extraction_method(document)
        # Same for the images sent in a chat
        with pytest.raises(
            ValueError,
            match=f"^Fallback model `{text_only_model}` does not support vision",
        ):
            llm_with_text_only_fallback.chat(
                "What is in the image?", images=[self.test_img_png_invoice]
            )
        assert not llm_calls  # no LLM calls are made
        # Unless the vision support of the model is declared explicitly
        llm_text_only_fallback._supports_vision = True
        llm_with_text_only_fallback.extract_all(document)
        check_all_extracted(document)

        # In a model group, vision support is required only of the LLMs that
        # the concepts to be extracted from images are routed to
        llm_text_only_complex = create_llm(text_only_model, role="complex")
        llm_text_only_vision = create_llm(text_only_model, role="vision")
        stub_model_api(llm_text_only_complex)
        stub_model_api(llm_text_only_vision)
        document = create_document()
        DocumentModelGroup(
            models=[llm_no_role, llm_text_only_complex, llm_vision]
        ).extract_all(document)
        check_all_extracted(document)
        llm_calls.clear()
        document = create_document()
        with pytest.raises(
            ValueError,
            match=f"^Model `{text_only_model}` does not support vision"
            ".*from images: `Stamp`\\.",
        ):
            DocumentModelGroup(
                models=[llm_no_role, llm_complex, llm_text_only_vision]
            ).extract_all(document)
        assert not llm_calls  # no LLM calls are made

    @pytest.mark.parametrize("use_concurrency", [False, True])
    def test_nested_extraction_role_preflight(self, use_concurrency: bool):
        """
        Tests that, when extracting with a model group, the roles of the nested
        aspects/concepts to be processed by the extraction method are validated before
        any LLM calls are made, i.e. before the state of the document is changed, while
        the roles of the aspects/concepts that are not selected or are skipped are not.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        group = DocumentModelGroup(
            models=[
                DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key"),
                DocumentLLM(
                    model="openai/gpt-6.1-sol", api_key="test_api_key", role="other"
                ),
            ]
        )
        document = Document(
            raw_text="The parties agree to these terms.",
            aspects=[
                Aspect(
                    name="Parent",
                    description="Parent section",
                    concepts=[
                        StringConcept(name="Selected", description="Selected concept"),
                        StringConcept(
                            name="Unselected",
                            description="Unselected concept",
                            model_role="unselected",
                        ),
                    ],
                    aspects=[
                        Aspect(
                            name="Child",
                            description="Child section",
                            model_role="missing_aspect",
                            concepts=[
                                StringConcept(
                                    name="Nested",
                                    description="Nested concept",
                                    model_role="missing_concept",
                                )
                            ],
                        ),
                        Aspect(
                            name="Empty",
                            description="Absent section",
                            concepts=[
                                StringConcept(
                                    name="Skipped",
                                    description="Skipped concept",
                                    model_role="skipped",
                                )
                            ],
                        ),
                    ],
                )
            ],
        )

        async def send_messages_stub(self, *, model_call_obj: Any, **kwargs: Any):
            """
            Simulates a successful LLM API call, which returns an extracted item for
            each aspect/concept in the prompt, except for the aspect with no content.

            Same params as `llm._send_messages_with_tools`.
            """
            params = model_call_obj.prompt_kwargs
            if "aspects" in params:
                answer = [
                    {
                        "aspect_id": f"A{i}",
                        "paragraph_ids": [] if aspect.name == "Empty" else ["P0"],
                    }
                    for i, aspect in enumerate(params["aspects"])
                ]
            else:
                answer = [
                    {"concept_id": f"C{i}", "extracted_items": ["Item"]}
                    for i, _ in enumerate(params["concepts"])
                ]
            return json.dumps(answer), False

        with patch.object(
            _DocumentLLM,
            "_send_messages_with_tools",
            autospec=True,
            side_effect=send_messages_stub,
        ) as send_messages:
            before = document.to_dict()
            with pytest.raises(ValueError, match="`missing_aspect`"):
                group.extract_aspects_from_document(
                    document, use_concurrency=use_concurrency
                )
            send_messages.assert_not_called()
            assert document.to_dict() == before

            # A selected aspect includes its descendants, but no concept roles
            # and no unrelated document aspects are needed for aspect extraction.
            document.add_aspects(
                [
                    Aspect(
                        name="Unrelated",
                        description="Unrelated section",
                        model_role="unrelated",
                    )
                ]
            )
            parent = document.aspects[0]
            before = document.to_dict()
            with pytest.raises(ValueError, match="`missing_aspect`"):
                group.extract_aspects_from_document(
                    document, from_aspects=[parent], use_concurrency=use_concurrency
                )
            send_messages.assert_not_called()
            assert document.to_dict() == before
            parent.aspects[0].model_role = None
            group.extract_aspects_from_document(
                document, from_aspects=[parent], use_concurrency=use_concurrency
            )
            assert parent._is_processed
            assert not document.aspects[1]._is_processed

            # Selecting a parent concept still extracts concepts in its non-empty
            # descendants. Unknown roles there must fail before the parent call.
            send_messages.reset_mock()
            before = document.to_dict()
            with pytest.raises(ValueError, match="`missing_concept`"):
                group.extract_concepts_from_aspect(
                    parent,
                    document,
                    from_concepts=[parent.concepts[0]],
                    use_concurrency=use_concurrency,
                )
            send_messages.assert_not_called()
            assert document.to_dict() == before
            parent.aspects[0].concepts[0].model_role = None
            group.extract_concepts_from_aspect(
                parent,
                document,
                from_concepts=[parent.concepts[0]],
                use_concurrency=use_concurrency,
            )
            assert parent.concepts[0].extracted_items[0].value == "Item"
            assert parent.aspects[0].concepts[0].extracted_items[0].value == "Item"
            assert not parent.concepts[1]._is_processed
            assert not parent.aspects[1].concepts[0]._is_processed

    @pytest.mark.vcr
    def test_extract_with_fallback(self):
        """
        Tests for retrying extraction with a fallback LLM.
        """

        document = Document(raw_text=get_test_document_text())
        document_concepts = [
            StringConcept(
                name="Title",
                description="Contract title",
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
            ),
        ]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        self.config_llm_async_limiter_for_mock_responses(
            self.invalid_llm_with_valid_fallback
        )

        self.invalid_llm_with_valid_fallback.extract_all(document)
        self.check_instance_container_states(
            original_container=document_concepts,  # ty: ignore[invalid-argument-type]
            assigned_container=document.concepts,  # ty: ignore[invalid-argument-type]
            assigned_instance_class=_Concept,
        )

        # Check serialization of an LLM with fallback
        self._check_deserialized_llm_config_eq(self.invalid_llm_with_valid_fallback)

        # Check usage tokens
        self.check_usage(self.invalid_llm_with_valid_fallback)
        # Check cost
        self.check_cost(self.invalid_llm_with_valid_fallback)

    def test_multiple_fallback_models(self):
        """
        Tests for configuring multiple fallback LLMs, which are tried in the given
        order.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        def create_llm(model: str, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with params that keep the stubbed calls deterministic.
            """
            return DocumentLLM(
                model=model,
                api_key="test_api_key",
                system_message="",
                max_retries_invalid_data=0,
                **kwargs,
            )

        primary = create_llm("openai/gpt-6-luna")
        fallback_1 = create_llm("openai/gpt-5.6-luna", is_fallback=True)
        fallback_2 = create_llm("openai/gpt-6.1-sol", is_fallback=True)
        fallback_3 = create_llm("openai/gpt-5.6-sol", is_fallback=True)
        fallbacks = [fallback_1, fallback_2, fallback_3]

        # === Config ===
        assert primary.fallback_models == []
        primary.fallback_models = fallbacks  # ty: ignore[invalid-assignment]
        assert primary.fallback_models == fallbacks
        assert all(
            assigned is original
            for assigned, original in zip(
                primary.fallback_models, fallbacks, strict=True
            )
        )
        assert (
            create_llm("openai/gpt-6-luna", fallback_models=fallbacks).fallback_models
            == fallbacks
        )

        # Each fallback LLM is validated
        with pytest.raises(ValueError, match="is_fallback=True"):
            primary.fallback_models = [fallback_1, create_llm("openai/gpt-5.6-terra")]
        with pytest.raises(ValueError, match="same role"):
            create_llm(
                "openai/gpt-6-luna",
                fallback_models=[
                    fallback_1,
                    create_llm(
                        "openai/gpt-5.6-terra", role="reasoner", is_fallback=True
                    ),
                ],
            )
        with pytest.raises(ValueError, match="same output language"):
            create_llm(
                "openai/gpt-6-luna",
                fallback_models=[
                    fallback_1,
                    create_llm(
                        "openai/gpt-5.6-terra",
                        output_language="adapt",
                        is_fallback=True,
                    ),
                ],
            )
        with pytest.raises(ValueError, match="same `system_message`"):
            create_llm(
                "openai/gpt-6-luna",
                fallback_models=[
                    fallback_1,
                    # The default system message, while the main LLM has none
                    DocumentLLM(
                        model="openai/gpt-5.6-terra",
                        api_key="test_api_key",
                        is_fallback=True,
                    ),
                ],
            )
        with pytest.raises(ValueError, match="same config params as the main LLM"):
            create_llm(
                "openai/gpt-6-luna",
                fallback_models=[
                    fallback_1,
                    create_llm("openai/gpt-6-luna", is_fallback=True),
                ],
            )
        with pytest.raises(ValueError, match="same config params as each other"):
            create_llm(
                "openai/gpt-6-luna",
                fallback_models=[
                    fallback_1,
                    create_llm("openai/gpt-5.6-luna", is_fallback=True),
                ],
            )
        with pytest.raises(ValueError, match="same config params as each other"):
            create_llm("openai/gpt-6-luna", fallback_models=[fallback_1, fallback_1])
        with pytest.raises(ValueError, match="cannot have its own fallback LLMs"):
            create_llm(
                "openai/gpt-5.6-terra", is_fallback=True, fallback_models=[fallback_1]
            )

        # Usage and cost are tracked for the primary LLM and each of its fallbacks,
        # in the order they are tried
        expected_models = [llm.model for llm in [primary, *fallbacks]]
        assert [i.model for i in primary.get_usage()] == expected_models
        assert [i.model for i in primary.get_cost()] == expected_models
        assert [i.is_fallback for i in primary.get_usage()] == [False, True, True, True]

        # Serialization
        primary_dict = primary.to_dict()
        assert len(primary_dict["fallback_models"]) == len(fallbacks)
        self._check_deserialized_llm_config_eq(primary)

        # A single `fallback_llm` is rejected, i.e. not silently ignored,
        # both as a param and in the serialized data
        with pytest.raises(ValueError, match="fallback_llm\n"):
            create_llm("openai/gpt-6-luna", fallback_llm=fallback_1)
        primary_dict["fallback_llm"] = primary_dict.pop("fallback_models")[0]
        with pytest.raises(ValueError, match="fallback_llm\n"):
            DocumentLLM.from_dict(primary_dict)

        # === Order of LLM calls ===
        called_models: list[str] = []

        def stub_model_api(llm: DocumentLLM, answer: str | None) -> None:
            """
            Stubs the LLM API call, which fails if no answer is provided.
            """

            async def _send_messages_stub(self, **kwargs: Any):
                """
                Simulates a failed or a successful LLM API call.

                Same params as `llm._send_messages_with_tools`.
                """
                called_models.append(self.model)
                if answer is None:
                    raise RuntimeError(f"Simulated API error for {self.model}")
                return answer, False

            llm._send_messages_with_tools = types.MethodType(_send_messages_stub, llm)  # ty: ignore[invalid-assignment]

        # Chat: fallbacks are tried in the given order until one of them succeeds
        stub_model_api(primary, None)
        stub_model_api(fallback_1, None)
        stub_model_api(fallback_2, "4")
        stub_model_api(fallback_3, "5")
        assert primary.chat("What's 2+2?") == "4"
        assert called_models == [primary.model, fallback_1.model, fallback_2.model]
        usage_calls = [i.usage.calls for i in primary.get_usage()]
        assert usage_calls[0] and usage_calls[1]  # failed calls are recorded
        assert not usage_calls[3]  # the last fallback was not tried

        called_models.clear()
        chat_session = ChatSession()
        assert primary.chat("What's 2+2?", chat_session=chat_session) == "4"
        assert called_models == [primary.model, fallback_1.model, fallback_2.model]
        # Ensure no duplicate messages in chat despite the calls to several LLMs
        assert [i.role for i in chat_session.messages] == ["user", "assistant"]
        assert chat_session.messages[0]._response_succeeded is True

        # Chat: the error is raised only when the last fallback fails
        called_models.clear()
        stub_model_api(fallback_2, None)
        stub_model_api(fallback_3, None)
        with pytest.raises(ModelAPIError):
            primary.chat("What's 2+2?")
        assert called_models == expected_models

        # Extraction: fallbacks are tried in the given order until one of them succeeds
        document = Document(
            raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
        )
        document.concepts = [
            StringConcept(name="Title", description="Contract title"),
            StringConcept(name="Parties", description="Contract parties"),
        ]
        concepts_answer = (
            '[{"concept_id": "C0", "extracted_items": ["Item 1"]}, '
            '{"concept_id": "C1", "extracted_items": ["Item 2"]}]'
        )
        called_models.clear()
        primary.reset_usage_and_cost()
        assert not any(i.usage.calls for i in primary.get_usage())
        stub_model_api(fallback_2, concepts_answer)
        primary.extract_concepts_from_document(document)
        assert called_models == [primary.model, fallback_1.model, fallback_2.model]
        assert [i.extracted_items[0].value for i in document.concepts] == [
            "Item 1",
            "Item 2",
        ]

        # Same with concurrency, i.e. with each concept processed in a separate call
        concept_answer = '[{"concept_id": "C0", "extracted_items": ["Item"]}]'
        called_models.clear()
        stub_model_api(fallback_2, None)
        stub_model_api(fallback_3, concept_answer)
        primary.extract_concepts_from_document(
            document, overwrite_existing=True, use_concurrency=True
        )
        assert sorted(called_models) == sorted(expected_models * 2)
        assert [i.extracted_items[0].value for i in document.concepts] == [
            "Item",
            "Item",
        ]

        # Extraction: the error is raised only when the last fallback fails
        called_models.clear()
        stub_model_api(fallback_3, None)
        with pytest.raises(ModelAPIError):
            primary.extract_concepts_from_document(document, overwrite_existing=True)
        assert called_models == expected_models
        # But a warning is issued if `raise_exception_on_extraction_error` is False
        called_models.clear()
        document_unprocessed = Document(raw_text=document.raw_text)
        document_unprocessed.concepts = [
            StringConcept(name="Title", description="Contract title")
        ]
        with pytest.warns(UserWarning, match="LLM API"):
            primary.extract_concepts_from_document(
                document_unprocessed,
                raise_exception_on_extraction_error=False,
            )
        assert called_models == expected_models
        assert not any(i.extracted_items for i in document_unprocessed.concepts)

    def test_deserialize_fallback_models_differing_in_credentials(self):
        """
        Tests that an LLM can be deserialized if its fallback LLMs differ from it,
        or from each other, only in the API credentials.

        The API credentials are removed from the serialized data, so such LLMs have
        the same config params in that data, while a fallback LLM with the same config
        params as the main LLM, or as another fallback LLM, is rejected when
        the fallback LLMs are set.

        No LLM API calls are made, i.e. no cassette is required.
        """

        from contextgem.internal.base.serialization import (
            _is_deserialization_in_progress,
        )

        api_base_1 = "https://endpoint-1.example.com/v1"
        api_base_2 = "https://endpoint-2.example.com/v1"

        def create_llm(**kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM of the same model with the given params.
            """
            return DocumentLLM(model="openai/gpt-6-luna", **kwargs)

        # Fallback LLM that differs from the main LLM only in the API base,
        # e.g. the same model served from another endpoint
        llm_other_api_base = create_llm(
            api_key="api_key_1",
            api_base=api_base_1,
            fallback_models=[
                create_llm(api_key="api_key_1", api_base=api_base_2, is_fallback=True)
            ],
        )
        # Fallback LLM that differs from the main LLM only in the API key
        llm_other_api_key = create_llm(
            api_key="api_key_1",
            api_base=api_base_1,
            fallback_models=[
                create_llm(api_key="api_key_2", api_base=api_base_1, is_fallback=True)
            ],
        )
        # Fallback LLMs that differ only in the API credentials from the main LLM
        # and from each other
        llm_fallbacks_other_credentials = create_llm(
            api_key="api_key_1",
            api_base=api_base_1,
            role="reasoner",
            fallback_models=[
                create_llm(
                    api_key="api_key_1",
                    api_base=api_base_2,
                    role="reasoner",
                    is_fallback=True,
                ),
                create_llm(
                    api_key="api_key_2",
                    api_base=api_base_2,
                    role="reasoner",
                    is_fallback=True,
                ),
            ],
        )
        model_group = DocumentModelGroup(
            models=[llm_other_api_base, llm_fallbacks_other_credentials]
        )

        # Deserialized from JSON, from a dict and from disk
        for instance in [
            llm_other_api_base,
            llm_other_api_key,
            llm_fallbacks_other_credentials,
            model_group,
        ]:
            self._check_deserialized_llm_config_eq(instance)

        # The other params of the fallback LLMs are preserved, and the API credentials
        # are not, i.e. the deserialized LLMs have the same config params
        llm_deserialized = DocumentLLM.from_json(
            llm_fallbacks_other_credentials.to_json()
        )
        assert len(llm_deserialized.fallback_models) == 2
        for llm in [llm_deserialized, *llm_deserialized.fallback_models]:
            assert llm.model == "openai/gpt-6-luna"
            assert llm.role == "reasoner"
            assert llm.api_key is None
            assert llm.api_base is None
        assert [i.is_fallback for i in llm_deserialized.fallback_models] == [True, True]

        # The API credentials can be set again in any order: the fallback LLMs are
        # not checked for the same config params when another param is assigned,
        # as the LLMs have the same config params until all the credentials are set
        fallback_llm_1, fallback_llm_2 = llm_deserialized.fallback_models
        llm_deserialized.api_key = "api_key_1"  # both fallback LLMs are still the same
        fallback_llm_1.api_key = "api_key_1"  # same as the main LLM
        fallback_llm_2.api_base = api_base_2
        llm_deserialized.api_base = api_base_1
        fallback_llm_1.api_base = api_base_2
        llm_deserialized.timeout = 60  # an unrelated param
        fallback_llm_2.api_key = "api_key_2"
        original_llms = [
            llm_fallbacks_other_credentials,
            *llm_fallbacks_other_credentials.fallback_models,
        ]
        for llm, original_llm in zip(
            [llm_deserialized, fallback_llm_1, fallback_llm_2],
            original_llms,
            strict=True,
        ):
            assert llm.api_key == original_llm.api_key
            assert llm.api_base == original_llm.api_base

        # The other requirements for the fallback LLMs still apply to the
        # serialized data, e.g. the role of the main LLM
        llm_dict = llm_other_api_base.to_dict()
        llm_dict["fallback_models"][0]["role"] = "another_role"
        with pytest.raises(ValueError, match="must have the same role"):
            DocumentLLM.from_dict(llm_dict)
        # ... and a failed deserialization does not affect the validation that follows
        assert not _is_deserialization_in_progress()

        # Fallback LLMs with the same config params are still rejected when
        # created or assigned, including for a deserialized LLM
        with pytest.raises(ValueError, match="same config params as the main LLM"):
            create_llm(
                api_key="api_key_1",
                api_base=api_base_1,
                fallback_models=[
                    create_llm(
                        api_key="api_key_1", api_base=api_base_1, is_fallback=True
                    )
                ],
            )
        with pytest.raises(ValueError, match="same config params as each other"):
            create_llm(
                api_key="api_key_1",
                api_base=api_base_1,
                fallback_models=[
                    create_llm(api_key="api_key_2", is_fallback=True),
                    create_llm(api_key="api_key_2", is_fallback=True),
                ],
            )
        llm_deserialized = DocumentLLM.from_json(llm_other_api_base.to_json())
        llm_deserialized.api_key = "api_key_1"
        with pytest.raises(ValueError, match="same config params as the main LLM"):
            llm_deserialized.fallback_models = [
                create_llm(api_key="api_key_1", is_fallback=True)
            ]
