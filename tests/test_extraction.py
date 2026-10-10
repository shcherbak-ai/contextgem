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
Module defining tests for the extraction of aspects and concepts.
"""

from __future__ import annotations

import json
import os
import re
import types
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter

from contextgem.internal.base import documents as documents_module
from contextgem.internal.base import llms as llms_module
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.data_models import _Message, _ModelCall, _ModelUsage
from contextgem.internal.exceptions import ModelAPIError, ModelExtractionError
from contextgem.internal.loggers import logger
from contextgem.internal.utils import _split_text_into_paragraphs
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    JsonObjectConcept,
    LabelConcept,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    Sentence,
    StringConcept,
    StringExample,
)
from tests.base import BaseTest
from tests.utils import get_project_root_path, get_test_document_text


class TestExtraction(BaseTest):
    """
    Tests for the extraction of aspects and concepts from documents with LLMs,
    and for the handling of extraction errors.
    """

    # Shared objects passed to the `parametrize` decorators of the tests below.
    # The decorators are evaluated in the class body, where the attributes inherited
    # from the base class are not in scope, so the names are bound here.
    document = BaseTest.document
    document_ua = BaseTest.document_ua
    document_zh = BaseTest.document_zh
    llm_extractor = BaseTest.llm_extractor
    model_group = BaseTest.model_group

    @pytest.mark.vcr
    def test_model_extraction_error_exception(self):
        """
        Tests for raising an exception when a model extraction error occurs.
        """

        @contextmanager
        def capture_logger_warnings():
            """
            Context manager to capture logger WARNING messages
            """
            import logging as stdlib_logging

            captured_logs = []

            class CaptureHandler(stdlib_logging.Handler):
                """
                Handler to capture log messages for testing.
                """

                def emit(self, record):
                    """
                    Emits a log message.
                    """
                    captured_logs.append(record.getMessage())

            # Create and add the capture handler
            capture_handler = CaptureHandler()
            capture_handler.setLevel(stdlib_logging.WARNING)
            logger.addHandler(capture_handler)

            try:
                yield captured_logs
            finally:
                logger.removeHandler(capture_handler)

        document = Document(raw_text=get_test_document_text())
        aspect = Aspect(name="Liability", description="Liability clauses")
        aspect_concept = StringConcept(
            name="Liability cap",
            description="Liability cap",
        )
        aspect.add_concepts([aspect_concept])
        document_concept = NumericalConcept(
            name="Contract term", description="Contract term"
        )
        document.add_aspects([aspect])
        document.add_concepts([document_concept])

        # === With retries ===
        # raise_exception_on_extraction_error is True by default
        llm = DocumentLLM(
            model="ollama/llama3.1:8b",
            api_base="http://localhost:11434",
            max_retries_invalid_data=1,  # default is 3
        )

        async def _invalid_json_query_stub(
            self,
            messages: list[_Message],
            model_call_obj: _ModelCall,
            num_retries_failed_request: int = 3,
            max_retries_failed_request: int = 0,
            async_limiter: AsyncLimiter | None = None,
            drop_params: bool = False,
            raise_exception_on_model_api_error: bool = True,
            has_fallback_pending: bool = False,
        ):
            """
            Simulates a successful call that returns invalid JSON.

            Same params as `llm._query_llm`.
            """
            return "<<<not-json>>>", _ModelUsage()

        _orig_query_llm = llm._query_llm
        llm._query_llm = types.MethodType(_invalid_json_query_stub, llm)  # ty: ignore[invalid-assignment]
        try:
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*1 retries"):
                llm.extract_aspects_from_document(
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
                # Aspect has a concept, and since the aspect was not extracted,
                # there's no aspect context to extract the concept from
                llm.extract_concepts_from_aspect(
                    document.aspects[0],
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*1 retries"):
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*1 retries"):
                llm.extract_all(
                    document,
                    overwrite_existing=True,
                )

            # But should issue a warning if `raise_exception_on_extraction_error` is False
            with capture_logger_warnings() as captured_logs:
                llm.extract_aspects_from_document(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                assert any(
                    re.search(r"invalid JSON.*1 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid JSON.*1 retries' not found in logs: {captured_logs}"
                )

            with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
                # Aspect has a concept, and since the aspect was not extracted,
                # there's no aspect context to extract the concept from
                llm.extract_concepts_from_aspect(
                    document.aspects[0],
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )

            with capture_logger_warnings() as captured_logs:
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                assert any(
                    re.search(r"invalid JSON.*1 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid JSON.*1 retries' not found in logs: {captured_logs}"
                )

            with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
                # Aspect has a concept, and since the aspect was not extracted,
                # there's no aspect context to extract the concept from
                llm.extract_all(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
        finally:
            llm._query_llm = _orig_query_llm  # ty: ignore[invalid-assignment]

        # === Without retries ===
        # raise_exception_on_extraction_error is True by default
        llm = DocumentLLM(
            model="ollama/llama3.1:8b",
            api_base="http://localhost:11434",
            max_retries_invalid_data=0,
        )

        _orig_query_llm_no_retry = llm._query_llm
        llm._query_llm = types.MethodType(_invalid_json_query_stub, llm)  # ty: ignore[invalid-assignment]
        try:
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*0 retries"):
                llm.extract_aspects_from_document(
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
                # Aspect has a concept, and since the aspect was not extracted,
                # there's no aspect context to extract the concept from
                llm.extract_concepts_from_aspect(
                    document.aspects[0],
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*0 retries"):
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*0 retries"):
                llm.extract_all(
                    document,
                    overwrite_existing=True,
                )

            # But should issue a warning if `raise_exception_on_extraction_error` is False
            with capture_logger_warnings() as captured_logs:
                llm.extract_aspects_from_document(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                assert any(
                    re.search(r"invalid JSON.*0 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid JSON.*0 retries' not found in logs: {captured_logs}"
                )

            with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
                # Aspect has a concept, and since the aspect was not extracted,
                # there's no aspect context to extract the concept from
                llm.extract_concepts_from_aspect(
                    document.aspects[0],
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )

            with capture_logger_warnings() as captured_logs:
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                assert any(
                    re.search(r"invalid JSON.*0 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid JSON.*0 retries' not found in logs: {captured_logs}"
                )

            with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
                # Aspect has a concept, and since the aspect was not extracted,
                # there's no aspect context to extract the concept from
                llm.extract_all(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
        finally:
            llm._query_llm = _orig_query_llm_no_retry  # ty: ignore[invalid-assignment]

        # === Test invalid LLM without fallback LLM ===
        # raise_exception_on_extraction_error is True by default
        # (an invalid model is not a transient error, i.e. the request is not retried)
        with pytest.raises(ModelAPIError, match=r"LLM API.*0 retries"):
            self.llm_invalid.extract_aspects_from_document(
                document,
                overwrite_existing=True,
            )
        with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
            # Aspect has a concept, and since the aspect was not extracted,
            # there's no aspect context to extract the concept from
            self.llm_invalid.extract_concepts_from_aspect(
                document.aspects[0],
                document,
                overwrite_existing=True,
            )
        with pytest.raises(ModelAPIError, match=r"LLM API.*0 retries"):
            self.llm_invalid.extract_concepts_from_document(
                document,
                overwrite_existing=True,
            )
        with pytest.raises(ModelAPIError, match=r"LLM API.*0 retries"):
            self.llm_invalid.extract_all(
                document,
                overwrite_existing=True,
            )

        # But should issue a warning if `raise_exception_on_extraction_error` is False
        with capture_logger_warnings() as captured_logs:
            self.llm_invalid.extract_aspects_from_document(
                document,
                overwrite_existing=True,
                raise_exception_on_extraction_error=False,
            )
            assert any(re.search(r"LLM API", log) for log in captured_logs), (
                f"Expected warning pattern 'LLM API' not found in logs: {captured_logs}"
            )

        with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
            # Aspect has a concept, and since the aspect was not extracted,
            # there's no aspect context to extract the concept from
            self.llm_invalid.extract_concepts_from_aspect(
                document.aspects[0],
                document,
                overwrite_existing=True,
                raise_exception_on_extraction_error=False,
            )

        with capture_logger_warnings() as captured_logs:
            self.llm_invalid.extract_concepts_from_document(
                document,
                overwrite_existing=True,
                raise_exception_on_extraction_error=False,
            )
            assert any(re.search(r"LLM API", log) for log in captured_logs), (
                f"Expected warning pattern 'LLM API' not found in logs: {captured_logs}"
            )

        with pytest.raises(ValueError, match=r"Aspect.*not yet processed"):
            # Aspect has a concept, and since the aspect was not extracted,
            # there's no aspect context to extract the concept from
            self.llm_invalid.extract_all(
                document,
                overwrite_existing=True,
                raise_exception_on_extraction_error=False,
            )

        # === Test with fallback LLM, with valid (populated) extraction results ===
        # raise_exception_on_extraction_error is True by default
        extracted_aspects = (
            self.invalid_llm_with_valid_fallback.extract_aspects_from_document(
                document,
                overwrite_existing=True,
            )
        )
        assert all(i.extracted_items for i in extracted_aspects)
        extracted_concepts = (
            self.invalid_llm_with_valid_fallback.extract_concepts_from_document(
                document,
                overwrite_existing=True,
            )
        )
        assert all(i.extracted_items for i in extracted_concepts)
        processed_document = self.invalid_llm_with_valid_fallback.extract_all(
            document,
            overwrite_existing=True,
        )
        assert all(i.extracted_items for i in processed_document.aspects)
        assert all(i.extracted_items for i in processed_document.concepts)
        assert all(i.extracted_items for i in document.aspects[0].concepts)

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_extract_aspects_from_document(self, llm: DocumentModelGroup | DocumentLLM):
        """
        Tests the aspects extraction functionality of LLMs.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        document = self.document.clone()  # clone for method-scoped state modification

        # Check for unsupported modality (aspects are extracted from text only)
        with pytest.raises(ValueError):
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
                modality="image",  # ty: ignore[unknown-argument]
            )

        document_aspects = [
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
            ),
        ]
        document.aspects = document_aspects
        assert document.aspects is not document_aspects

        # Aspects validation

        # Attachment to document
        with pytest.raises(ValueError):
            detached_aspects = [
                Aspect(name="Random", description="Random"),
            ]
            llm.extract_aspects_from_document(document, from_aspects=detached_aspects)

        # Invalid aspect types
        invalid_aspects = [1, True]
        with pytest.raises(ValueError):
            document.aspects = invalid_aspects  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError):
            llm.extract_aspects_from_document(
                document,
                from_aspects=invalid_aspects,  # ty: ignore[invalid-argument-type]
            )

        # Invalid sequence types for "from_aspects"
        with pytest.raises(ValueError, match="list"):
            llm.extract_aspects_from_document(
                document,
                from_aspects=tuple(document_aspects),
            )
        with pytest.raises(ValueError, match="list"):
            llm.extract_aspects_from_document(
                document,
                from_aspects=set([1, 2, 3]),  # type: ignore
            )

        # Duplicates check
        duplicate_aspects = [
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
            ),
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
                model_role="reasoner",
                add_justifications=True,
            ),
        ]
        with pytest.raises(ValueError):
            document.aspects = duplicate_aspects
        duplicate_name_aspects = [
            Aspect(name="Parties", description="Desc 1"),
            Aspect(name="Parties", description="Desc 2"),
        ]
        with pytest.raises(ValueError):
            document.aspects = duplicate_name_aspects
        duplicate_description_aspects = [
            Aspect(name="Name 1", description="Description"),
            Aspect(name="Name 2", description="Description"),
        ]
        with pytest.raises(ValueError):
            document.aspects = duplicate_description_aspects

        # No partial update check
        assert document.aspects == document_aspects
        assert document.aspects is not document_aspects

        # LLM extraction
        document_aspects = [
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
            ),
            Aspect(
                name="Term",
                description="Clauses addressing term of the contract",
            ),
            Aspect(
                name="Exclusions from confidential information",
                description="Clauses addressing exclusions from the confidential information.",
                reference_depth="sentences",
                aspects=[  # sub-aspects
                    Aspect(
                        name="Information in possession of the receiving party",
                        description="Information already in possession of the receiving party",
                    ),
                    Aspect(
                        name="Information received from a third party",
                        description="Information received from a third party",
                    ),
                    Aspect(
                        name="Independently developed information",
                        description="Information developed independently by the receiving party",
                        reference_depth="sentences",
                        add_justifications=True,
                        justification_depth="balanced",
                    ),
                ],
            ),
            Aspect(
                name="Definition of confidential information",
                description="Clauses defining confidential information",
            ),
            Aspect(
                name="Purpose",
                description="Purpose of the agreement",
            ),
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
                reference_depth="sentences",
            ),
            Aspect(
                name="Non-compete",
                description="Clauses containing non-compete provisions",
                model_role="reasoner",
            ),
            Aspect(
                name="Penalties",
                description="Clauses describing penalties for breaching the contract",
                model_role="reasoner",
            ),
        ]
        document.aspects = document_aspects

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(3 if llm.is_group else 2) * 3
            + 2,  # *3 due to 30 (1/3) paras per call, +2 due to sub-aspects
            expected_n_calls_1_item_per_call=len(document.aspects) * 3
            + 3,  # *3 due to 30 (1/3) paras per call, +3 due to sub-aspects
            expected_n_calls_with_concurrency=len(document.aspects) * 3
            + 3,  # *3 due to 30 (1/3) paras per call, +3 due to sub-aspects
            func=llm.extract_aspects_from_document,
            func_kwargs={
                "document": document,
                "max_paragraphs_to_analyze_per_call": 30,
            },
            original_container=document_aspects,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
            compare_sequential_1_item_in_call=False,
        )
        check_aspects = document.get_aspects_by_names(
            [
                "Parties",
                "Term",
                "Exclusions from confidential information",
                "Definition of confidential information",
            ]
        )
        for aspect in check_aspects:
            self.check_extra_data_in_extracted_items(aspect)  # ty: ignore[invalid-argument-type]

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_aspects_from_document(document)
        with pytest.raises(ValueError):
            llm.extract_aspects_from_document(document, from_aspects=document_aspects)
        overwrite_aspects = [document.get_aspect_by_name("Parties")]
        extracted_aspects = llm.extract_aspects_from_document(
            document,
            from_aspects=overwrite_aspects,
            overwrite_existing=True,
        )
        assert overwrite_aspects == extracted_aspects
        self.check_instance_container_states(
            original_container=[document_aspects[0]],
            assigned_container=extracted_aspects,  # ty: ignore[invalid-argument-type]
            assigned_instance_class=Aspect,
        )

        # Log sub-aspects for testing
        self.log_extracted_items_for_instance(
            document.get_aspect_by_name(  # ty: ignore[invalid-argument-type]
                "Exclusions from confidential information"
            ).get_aspect_by_name("Information in possession of the receiving party"),
            full_repr=False,
        )
        self.log_extracted_items_for_instance(
            document.get_aspect_by_name(  # ty: ignore[invalid-argument-type]
                "Exclusions from confidential information"
            ).get_aspect_by_name("Information received from a third party"),
            full_repr=False,
        )
        self.log_extracted_items_for_instance(
            document.get_aspect_by_name(  # ty: ignore[invalid-argument-type]
                "Exclusions from confidential information"
            ).get_aspect_by_name("Independently developed information"),
            full_repr=False,
        )

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_extract_concepts_from_aspect(self, llm: DocumentModelGroup | DocumentLLM):
        """
        Tests for concept extraction from aspect.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        document = self.document.clone()  # clone for method-scoped state modification

        aspect_concepts = [
            BooleanConcept(
                name="Information in all forms",
                description="Confidential information can be conveyed in all forms.",
                add_references=True,
                reference_depth="sentences",
            ),
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
            ),
            StringConcept(
                name="Technical Information",
                description="Categories of Technical Information",
            ),
            StringConcept(
                name="Customer and Market Information",
                description="Categories of Customer and Market Information",
            ),
            StringConcept(
                name="Personnel and Internal Information",
                description="Categories of Personnel and Internal Information",
            ),
            StringConcept(
                name="Intellectual Property",
                description="Categories of Intellectual Property covered in the document",
                examples=[
                    StringExample(content="patents"),
                    StringExample(content="source code"),
                ],
            ),
            BooleanConcept(
                name="Software code",
                description="Whether software code is explicitly included into the confidential information.",
                add_references=True,
                reference_depth="sentences",
            ),
        ]
        aspects = [
            Aspect(
                name="Confidential information",
                description="Clauses describing confidential information covered by the NDA",
                concepts=aspect_concepts,
            )
        ]

        # Concepts validation

        # Attachment to aspect
        with pytest.raises(ValueError):
            document.aspects = []
            llm.extract_concepts_from_aspect(aspect=aspects[0], document=document)

        # LLM extraction
        document.aspects = aspects
        assert document.aspects is not aspects
        attached_aspect = document.get_aspect_by_name("Confidential information")

        # Attempt concept extraction first, without pre-extraction of the aspect.
        assert not attached_aspect.extracted_items
        with pytest.raises(ValueError):
            llm.extract_concepts_from_aspect(aspect=attached_aspect, document=document)

        # Aspects first
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=1,
            expected_n_calls_1_item_per_call=1,
            expected_n_calls_with_concurrency=1,
            func=llm.extract_aspects_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=aspects,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
        )
        # Then concepts
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=2,
            expected_n_calls_1_item_per_call=7,
            expected_n_calls_with_concurrency=7,
            func=llm.extract_concepts_from_aspect,
            func_kwargs={
                "aspect": attached_aspect,
                "document": document,
            },
            original_container=list(aspects[0].concepts),
            assigned_container=list(attached_aspect.concepts),
            assigned_instance_class=_Concept,
            compare_sequential_1_item_in_call=False,
        )

        # Aspect-scoped concept references resolve to positions in the document,
        # in document order (the LLM sees only the aspect's reference paragraphs
        # as context for this extraction path)
        for concept_name in ("Information in all forms", "Software code"):
            concept_with_refs = attached_aspect.get_concept_by_name(concept_name)
            assert concept_with_refs.extracted_items
            for extracted_item in concept_with_refs.extracted_items:
                assert extracted_item.reference_paragraphs
                item_ref_para_indices = [
                    document.get_paragraph_index(p)
                    for p in extracted_item.reference_paragraphs
                ]
                assert item_ref_para_indices == sorted(item_ref_para_indices)
                assert extracted_item.reference_sentences
                item_ref_sent_positions = [
                    document.get_sentence_index(s)
                    for s in extracted_item.reference_sentences
                ]
                assert item_ref_sent_positions == sorted(item_ref_sent_positions)

        # No concepts defined for aspect. Logger warning and empty list returned.
        aspects_without_concepts = [
            Aspect(
                name="Term and termination",
                description="All clauses addressing term and termination of the contract",
                concepts=[],
                add_justifications=True,
            ),
        ]
        document.aspects = aspects_without_concepts
        assert document.aspects is not aspects_without_concepts
        llm.extract_aspects_from_document(document)
        extracted_concepts = llm.extract_concepts_from_aspect(
            aspect=document.get_aspect_by_name("Term and termination"),
            document=document,
        )
        self.check_instance_container_states(
            original_container=aspects_without_concepts,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
        )
        assert extracted_concepts == []

        # Invalid concepts passed
        document.aspects = aspects
        assert document.aspects is not aspects
        llm.extract_aspects_from_document(document)
        attached_aspect = document.get_aspect_by_name("Confidential information")
        with pytest.raises(ValueError):
            no_concepts = []
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=document,
                from_concepts=no_concepts,
            )
        with pytest.raises(ValueError):
            invalid_concepts = [1, True]
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=document,
                from_concepts=invalid_concepts,  # ty: ignore[invalid-argument-type]
            )
        with pytest.raises(ValueError):
            detached_concepts = [
                StringConcept(name="Random", description="Random"),
            ]
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=document,
                from_concepts=detached_concepts,
            )

        # Invalid sequence types for "from_concepts"
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=document,
                from_concepts=tuple(attached_aspect.concepts),
            )
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=document,
                from_concepts=set([1, 2, 3]),  # type: ignore
            )

        # Duplicates check
        duplicate_concepts = [
            StringConcept(
                name="Title",
                description="Contract title",
                add_justifications=True,
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                model_role="reasoner",
            ),
        ]
        with pytest.raises(ValueError):
            attached_aspect.concepts = duplicate_concepts
        duplicate_name_concepts = [
            StringConcept(
                name="Title",
                description="Desc 1",
            ),
            BooleanConcept(
                name="Title",
                description="Desc 2",
            ),
        ]
        with pytest.raises(ValueError):
            attached_aspect.concepts = duplicate_name_concepts
        duplicate_description_concepts = [
            StringConcept(name="Name 1", description="Description"),
            NumericalConcept(
                name="Name 2", description="Description", numeric_type="int"
            ),
        ]
        with pytest.raises(ValueError):
            attached_aspect.concepts = duplicate_description_concepts

        # No partial update check
        assert attached_aspect.concepts == aspect_concepts
        assert attached_aspect.concepts is not aspect_concepts

        # LLM extraction
        extracted_concepts = llm.extract_concepts_from_aspect(
            aspect=attached_aspect, document=document
        )
        assert attached_aspect.concepts == extracted_concepts
        self.check_instance_container_states(
            original_container=aspect_concepts,  # ty: ignore[invalid-argument-type]
            assigned_container=list(attached_aspect.concepts),
            assigned_instance_class=_Concept,
        )

        # Extract with different paragraph chunks for context
        extracted_concepts = llm.extract_concepts_from_aspect(
            aspect=attached_aspect,
            document=document,
            overwrite_existing=True,
            max_paragraphs_to_analyze_per_call=30,
        )
        assert attached_aspect.concepts == extracted_concepts
        self.check_instance_container_states(
            original_container=aspect_concepts,  # ty: ignore[invalid-argument-type]
            assigned_container=list(attached_aspect.concepts),
            assigned_instance_class=_Concept,
        )

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_concepts_from_aspect(aspect=attached_aspect, document=document)
        with pytest.raises(ValueError):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=document,
                from_concepts=attached_aspect.concepts,
            )
        overwrite_concepts = [attached_aspect.concepts[0]]
        extracted_concepts = llm.extract_concepts_from_aspect(
            aspect=attached_aspect,
            document=document,
            from_concepts=overwrite_concepts,
            overwrite_existing=True,
        )
        assert attached_aspect.concepts[0] == extracted_concepts[0]
        self.check_instance_container_states(
            original_container=[aspects[0].concepts[0]],
            assigned_container=extracted_concepts,
            assigned_instance_class=_Concept,
        )

        # Test concepts extraction from sub-aspects
        # Define an aspect with optional concept(s), using natural language
        logger.info("Testing concepts extraction from sub-aspects")
        main_aspect = Aspect(
            name="Liability",
            description="Clauses defining the liability of the Supplier, "
            "including total liability and penalties for breach",
        )
        sub_aspects = [
            Aspect(
                name="Total liability",
                description="Total liability",
                concepts=[
                    StringConcept(
                        name="Liability cap",
                        description="Total liability cap amount",
                    ),
                    NumericalConcept(
                        name="Liability amount period",
                        description="Period (in months) that applies as the basis for calculating the liability amount",
                        numeric_type="int",
                    ),
                ],
            ),
            Aspect(
                name="Penalties for breach",
                description="Penalties for breach",
                concepts=[
                    StringConcept(
                        name="Penalty amount",
                        description="Penalty amount",
                    )
                ],
            ),
            Aspect(
                name="Cookie recipe",  # intentionally empty, out-of-scope
                description="Cookie recipe",
                concepts=[
                    StringConcept(
                        name="Pasta recipe",
                        description="Pasta recipe",
                    )
                ],
            ),
        ]
        main_aspect.add_aspects(sub_aspects)  # ty: ignore[invalid-argument-type]
        document = self.document.clone()
        document.aspects = [main_aspect]
        llm.extract_all(document)
        total_liability_sub_aspect = document.get_aspect_by_name(
            "Liability"
        ).get_aspect_by_name("Total liability")
        assert all(i.extracted_items for i in total_liability_sub_aspect.concepts)
        self.log_extracted_items_for_instance(total_liability_sub_aspect)  # ty: ignore[invalid-argument-type]
        penalties_for_breach_sub_aspect = document.get_aspect_by_name(
            "Liability"
        ).get_aspect_by_name("Penalties for breach")
        assert all(i.extracted_items for i in penalties_for_breach_sub_aspect.concepts)
        self.log_extracted_items_for_instance(penalties_for_breach_sub_aspect)  # ty: ignore[invalid-argument-type]
        assert (
            not document.get_aspect_by_name("Liability")
            .get_aspect_by_name("Cookie recipe")
            .extracted_items
        )

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_extract_concepts_from_document(
        self, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for concept extraction from document.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        document = self.document.clone()  # clone for method-scoped state modification

        document_concepts = [
            StringConcept(
                name="Title",
                description="Contract title",
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                model_role="reasoner",
                add_justifications=True,
            ),
        ]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        # Concepts validation

        # Attachment to document
        with pytest.raises(ValueError):
            detached_concepts = [
                StringConcept(name="Random", description="Random"),
            ]
            llm.extract_concepts_from_document(
                document, from_concepts=detached_concepts
            )

        # Invalid types
        with pytest.raises(ValueError):
            invalid_concepts = [1, True]
            document.concepts = invalid_concepts  # ty: ignore[invalid-assignment]

        # Invalid sequence types for "from_concepts"
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_document(
                document,
                from_concepts=tuple(document.concepts),
            )
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_document(
                document,
                from_concepts=set([1, 2, 3]),  # type: ignore
            )

        # No partial update check
        assert document.concepts == document_concepts
        assert document.concepts is not document_concepts

        document_concepts = [
            DateConcept(
                name="Effective Date",
                description="Contract start date",
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                add_justifications=True,
            ),
            StringConcept(
                name="Title",
                description="Contract title",
            ),
            StringConcept(
                name="Parties",
                description="Names of contract parties",
                add_references=True,
            ),
            StringConcept(
                name="Duration",
                description="Contract duration",
                add_references=True,
            ),
            NumericalConcept(
                name="Duration in years",
                description="Contract duration in years",
                model_role="reasoner",
                numeric_type="int",
            ),
            BooleanConcept(
                name="Contract type check",
                description="Contract is a shareholder's agreement",
                add_justifications=True,
            ),
            RatingConcept(
                name="Contract quality",
                description="Contract quality from NDA best practice perspective.",
                rating_scale=(1, 10),
                add_justifications=True,
                justification_depth="balanced",
                justification_max_sents=5,
            ),
        ]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        # Duplicates check
        duplicate_concepts = [
            StringConcept(
                name="Title",
                description="Contract title",
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                model_role="reasoner",
                add_justifications=True,
            ),
        ]
        with pytest.raises(ValueError):
            document.concepts = duplicate_concepts
        duplicate_name_concepts = [
            StringConcept(
                name="Title",
                description="Desc 1",
            ),
            BooleanConcept(
                name="Title",
                description="Desc 2",
            ),
        ]
        with pytest.raises(ValueError):
            document.concepts = duplicate_name_concepts
        duplicate_description_concepts = [
            StringConcept(name="Name 1", description="Description"),
            NumericalConcept(
                name="Name 2", description="Description", numeric_type="int"
            ),
        ]
        with pytest.raises(ValueError):
            document.concepts = duplicate_description_concepts

        # No partial update check
        assert document.concepts == document_concepts
        assert document.concepts is not document_concepts

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(5 if llm.is_group else 4),
            expected_n_calls_with_concurrency=len(document.concepts),
            expected_n_calls_1_item_per_call=len(document.concepts),
            func=llm.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )
        self.check_extra_data_in_extracted_items(document)

        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Effective Date")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Contract quality")
        )

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_concepts_from_document(document)
        with pytest.raises(ValueError):
            llm.extract_concepts_from_document(
                document, from_concepts=document.concepts
            )
        overwrite_concepts = [document.concepts[0]]
        extracted_concepts = llm.extract_concepts_from_document(
            document, from_concepts=overwrite_concepts, overwrite_existing=True
        )
        assert overwrite_concepts == extracted_concepts
        self.check_instance_container_states(
            original_container=[document_concepts[0]],
            assigned_container=extracted_concepts,
            assigned_instance_class=_Concept,
        )

        # Extract with different paragraph chunks for context
        extracted_concepts = llm.extract_concepts_from_document(
            document,
            from_concepts=overwrite_concepts,
            overwrite_existing=True,
            max_paragraphs_to_analyze_per_call=20,
        )
        assert overwrite_concepts == extracted_concepts
        self.check_instance_container_states(
            original_container=[document_concepts[0]],
            assigned_container=extracted_concepts,
            assigned_instance_class=_Concept,
        )

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "document",
        [
            document,
            document_ua,
            document_zh,
        ],
    )
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_extract_all(
        self, document: Document, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for extracting all aspects and concepts from the document and its aspects.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        self.config_llms_for_output_lang(document, llm)

        is_non_eng_document = document in [self.document_ua, self.document_zh]
        document = document.clone()  # clone for method-scoped state modification

        # Standard scenario - concepts are defined
        document_aspects = [
            Aspect(
                name="Term and termination",
                description="All clauses addressing term and termination of the contract.",
                concepts=[
                    StringConcept(
                        name="Contract term",
                        description="Contract term",
                    ),
                    NumericalConcept(
                        name="Duration",
                        description="Contract duration in years",
                        numeric_type="int",
                        add_justifications=True,
                        add_references=True,
                        reference_depth="sentences",
                    ),
                ],
            ),
            Aspect(
                name="Parties",
                description="Parties to the agreement",
                concepts=[
                    StringConcept(
                        name="Contract party name",
                        description="Contract party name in the contract. "
                        "Exclude information on individual signatories.",
                    ),
                    StringConcept(
                        name="Contract party name and role",
                        description="Contract party name and role in the contract. "
                        "Exclude information on individual signatories.",
                        examples=[StringExample(content="Company ABC (Supplier)")],
                    ),
                ],
                add_justifications=True,
            ),
        ]
        document_concepts = [
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                model_role="reasoner",
            ),
            StringConcept(
                name="Title",
                description="Contract title",
            ),
            StringConcept(
                name="Start date",
                description="Contract start date",
                add_justifications=True,
            ),
            JsonObjectConcept(
                name="Key aspects",
                description="Names of key aspects covered in the document",
                structure={"aspect_names": list[str]},
            ),
        ]
        document.aspects = document_aspects
        assert document.aspects is not document_aspects
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(8 if llm.is_group else 7),
            expected_n_calls_with_concurrency=10,
            expected_n_calls_1_item_per_call=10,
            func=llm.extract_all,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_aspects),
            assigned_container=list(document.aspects),
            assigned_instance_class=Aspect,
        )
        if is_non_eng_document:
            # Skip further processing for non-ENG docs.
            return

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_all(document)
        document = llm.extract_all(document, overwrite_existing=True)  # ty: ignore[invalid-assignment]
        self.check_instance_container_states(
            original_container=list(document_aspects),
            assigned_container=list(document.aspects),  # ty: ignore[invalid-argument-type]
            assigned_instance_class=Aspect,
        )
        self.check_instance_container_states(
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )

        # No defined concepts - just a logger message must appear and no error thrown, empty list returned
        document_aspects = [
            Aspect(
                name="Term and termination",
                description="All clauses addressing term and termination of the contract.",
                concepts=[],
            ),
            Aspect(
                name="Parties",
                description="Parties to the agreement",
                concepts=[],
            ),
        ]
        document_concepts = []
        document.aspects = document_aspects
        assert document.aspects is not document_aspects
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        # LLM extraction
        document = llm.extract_all(document)  # ty: ignore[invalid-assignment]
        self.check_instance_container_states(
            original_container=document_aspects,
            assigned_container=document.aspects,  # ty: ignore[invalid-argument-type]
            assigned_instance_class=Aspect,
        )

        # No aspects, just document concepts
        document.aspects = []
        document_concepts = [
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                model_role="reasoner",
            ),
            StringConcept(
                name="Title",
                description="Contract title",
            ),
            StringConcept(
                name="Start date",
                description="Contract start date",
            ),
        ]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts
        document = llm.extract_all(document)  # ty: ignore[invalid-assignment]
        self.check_instance_container_states(
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )

        with pytest.raises(ValueError):
            # Must override existing concepts
            llm.extract_all(document)

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    def test_aspect_extraction_from_paragraphs(
        self, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for extracting aspects if only paragraphs are provided for a document.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        paragraphs = _split_text_into_paragraphs(get_test_document_text())
        paragraphs = [Paragraph(raw_text=i) for i in paragraphs]
        document = Document(paragraphs=paragraphs)  # ty: ignore[invalid-argument-type]
        assert document.raw_text  # to be populated from paragraphs
        aspects_to_extract = [
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
                add_justifications=True,
            ),
            Aspect(
                name="Term and termination",
                description="Term and termination clauses",
                reference_depth="sentences",
            ),
            Aspect(
                name="Anomalies",
                description="Anomalies in the contract. A contract anomaly is any deviation, inconsistency, "
                "or irregularity in a contract's terms, structure, or execution - such as bizarre "
                "clauses about intergalactic jurisdiction, mandatory sock-wearing policies, "
                "or payment in rare seashells - that may indicate errors, conflicts, or potential risks.",
                reference_depth="sentences",
            ),
        ]
        document.aspects = aspects_to_extract
        extracted_aspects = llm.extract_aspects_from_document(
            document, use_concurrency=True
        )
        assert document.aspects == extracted_aspects
        self.check_instance_container_states(
            original_container=aspects_to_extract,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
        )
        for aspect in extracted_aspects:
            self.check_extra_data_in_extracted_items(aspect)  # ty: ignore[invalid-argument-type]

        # Extract with different paragraph chunks for context
        extracted_aspects = llm.extract_aspects_from_document(
            document,
            overwrite_existing=True,
            max_paragraphs_to_analyze_per_call=15,
            use_concurrency=True,
        )
        assert document.aspects == extracted_aspects
        self.check_instance_container_states(
            original_container=aspects_to_extract,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
        )

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

    @pytest.mark.vcr
    def test_extract_with_additional_context(self):
        """
        Tests for LLM extraction from a document with additional context for paragraphs.
        """

        doc = Document.from_disk(self.test_doc_with_additional_context_path)
        assert all(paragraph.additional_context for paragraph in doc.paragraphs)

        # Create a new model group with new usage stats,
        # as we need to isolate the prompts for inspection.
        # The LLM without a role extracts from text, while the concepts to be
        # extracted from images are routed to a separate LLM by the role.
        model_group = DocumentModelGroup(
            models=[
                DocumentLLM(**self._llm_extractor_kwargs_openai),  # ty: ignore[invalid-argument-type]
                DocumentLLM(
                    **self._llm_extractor_kwargs_openai,  # ty: ignore[invalid-argument-type]
                    role="vision",
                ),
            ]
        )

        def check_additional_context_in_prompt(prompt_kwargs_key: str) -> None:
            """
            Validates that the additional context is included in the prompts that list
            paragraphs, and is not included in the prompts with the text as a whole.

            :param prompt_kwargs_key: The key in prompt_kwargs to check ('text' or 'paragraphs')
            :type prompt_kwargs_key: str
            """
            # Only check text extraction calls, made by the LLM without a role
            call_objs = next(
                i for i in model_group.get_usage() if i.role is None
            ).usage.calls
            text_call_objs = [
                c
                for c in call_objs
                if (
                    "data_type" not in c.prompt_kwargs
                    or c.prompt_kwargs["data_type"] == "text"
                )
                and prompt_kwargs_key in c.prompt_kwargs
            ]
            assert text_call_objs, f"No text call objects found for {prompt_kwargs_key}"

            # We may have multiple text extraction calls when chunking document text,
            # i.e. submitting text fragments for extraction in the same pipeline.
            # We need to check each call separately.
            for text_call_obj in text_call_objs:
                if prompt_kwargs_key == "paragraphs":
                    assert text_call_obj.prompt_kwargs.get(
                        "additional_context_for_paras_or_sents"
                    )
                    assert "additional context information" in text_call_obj.prompt
                    assert all(
                        paragraph.additional_context in text_call_obj.prompt
                        for paragraph in text_call_obj.prompt_kwargs["paragraphs"]
                    )
                else:
                    assert (
                        "additional_context_for_paras_or_sents"
                        not in text_call_obj.prompt_kwargs
                    )
                    assert "<additional_context>" not in text_call_obj.prompt

        # Test concept extraction
        doc.concepts = [
            NumericalConcept(
                name="Hidden gems count",
                description="Number of hidden gems in the document",
                numeric_type="int",
            ),
            StringConcept(
                name="Invoice total amount",
                description="Total amount of the invoice",
                model_role="vision",
                modality="image",
            ),
        ]

        # Test extraction from full text
        extracted_concepts = model_group.extract_concepts_from_document(doc)
        assert extracted_concepts[0].extracted_items
        logger.debug(extracted_concepts[0].extracted_items[0].value)
        assert extracted_concepts[0].extracted_items[0].value == 3
        assert extracted_concepts[1].extracted_items
        logger.debug(extracted_concepts[1].extracted_items[0].value)
        assert "4800" in extracted_concepts[1].extracted_items[0].value
        check_additional_context_in_prompt(prompt_kwargs_key="text")

        # Test extraction with max paragraphs
        extracted_concepts = model_group.extract_concepts_from_document(
            doc, max_paragraphs_to_analyze_per_call=25, overwrite_existing=True
        )
        assert extracted_concepts[0].extracted_items
        check_additional_context_in_prompt(prompt_kwargs_key="text")

        # Test aspect extraction

        # Test with paragraph-level refs (default)
        doc.aspects = [
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
            )
        ]
        extracted_aspects = model_group.extract_aspects_from_document(doc)
        assert extracted_aspects[0].extracted_items
        assert extracted_aspects[0].extracted_items[0].reference_paragraphs
        # Aspect references resolve to positions in the document, in document order
        aspect_ref_para_indices = [
            doc.get_paragraph_index(p)
            for p in extracted_aspects[0].reference_paragraphs
        ]
        assert aspect_ref_para_indices == sorted(aspect_ref_para_indices)
        check_additional_context_in_prompt(prompt_kwargs_key="paragraphs")

        # Test with sentence-level refs
        doc.aspects = [
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
                reference_depth="sentences",
            )
        ]
        extracted_aspects = model_group.extract_aspects_from_document(doc)
        assert extracted_aspects[0].extracted_items
        assert extracted_aspects[0].extracted_items[0].reference_sentences
        # Aspect references resolve to positions in the document, in document order
        aspect_ref_sent_positions = [
            doc.get_sentence_index(s) for s in extracted_aspects[0].reference_sentences
        ]
        assert aspect_ref_sent_positions == sorted(aspect_ref_sent_positions)
        # Each sentence inherits the additional context of its paragraph
        assert all(
            sentence.additional_context == paragraph.additional_context
            for paragraph in doc.paragraphs
            for sentence in paragraph.sentences
        )
        check_additional_context_in_prompt(prompt_kwargs_key="paragraphs")

        # Test serialization and cloning after LLM extraction
        self.check_instance_serialization_and_cloning(doc)
        for aspect in doc.aspects:
            self.check_instance_serialization_and_cloning(aspect)
        for concept in doc.concepts:
            self.check_instance_serialization_and_cloning(concept)
        for paragraph in doc.paragraphs:
            self.check_instance_serialization_and_cloning(paragraph)
            for sentence in paragraph.sentences:
                self.check_instance_serialization_and_cloning(sentence)
        for sentence in doc.sentences:
            self.check_instance_serialization_and_cloning(sentence)
        for image in doc.images:
            self.check_instance_serialization_and_cloning(image)

    @pytest.mark.vcr
    def test_very_long_doc_extraction(self):
        """
        Tests for very long document extraction (200+ pages).
        """
        # Load the test text file as a Document
        with open(
            os.path.join(
                get_project_root_path(),
                "tests",
                "other_files",
                "gdpr_modified_for_testing.txt",
            ),
            encoding="utf-8",
        ) as f:
            text_content = f.read()
        doc_concepts = [
            StringConcept(
                name="Document title",
                description="Title of the current document",
                singular_occurrence=True,
            ),
            LabelConcept(
                name="Document type",
                description="Type of the current document",
                labels=["legislation", "contract", "other"],
                singular_occurrence=True,
            ),
            StringConcept(
                name="Entry into force date",
                description=(
                    "Date of the entry into force of the current document. "
                    "Only focus on the entry into force date for the current document, "
                    "not for other documents. If no specific date is mentioned, "
                    "look for the provision on how this date is determined. "
                    "Note that the entry into force date may not be the same as "
                    "the publication date."
                ),
            ),  # entry into force date is modified in the test doc
            StringConcept(
                name="Anomalies",
                description="Anomalies in the document: unusual or unexpected content",
            ),  # anomaly is in the middle of the document
        ]
        doc = Document(raw_text=text_content)
        doc.concepts = doc_concepts

        # First, test hitting the max input tokens limit
        # A suggestion to use the optimization guide is included in the error message
        too_long_doc = Document(raw_text=text_content * 5)
        too_long_doc.concepts = doc_concepts
        llm_short_context = DocumentLLM(
            model="anthropic/claude-haiku-4-5",  # 200k context window
            api_key="...",  # dummy key (pre-call validation will fail)
        )
        with pytest.raises(
            ValueError,
            match="https://contextgem.dev/optimizations/optimization_long_docs/",
        ):
            llm_short_context.extract_concepts_from_document(
                too_long_doc,
            )

        # Use params optimized for very long documents (200+ pages)
        extracted_concepts = self.llm_extractor.extract_concepts_from_document(
            doc,
            max_paragraphs_to_analyze_per_call=500,  # split into paragraph chunks
            use_concurrency=True,
        )

        assert extracted_concepts[0].extracted_items
        self.log_extracted_items_for_instance(extracted_concepts[0])
        assert extracted_concepts[1].extracted_items
        self.log_extracted_items_for_instance(extracted_concepts[1])
        assert extracted_concepts[2].extracted_items
        self.log_extracted_items_for_instance(extracted_concepts[2])
        assert extracted_concepts[3].extracted_items
        self.log_extracted_items_for_instance(extracted_concepts[3])

        # We intentionally modified the entry into force date in the GDPR test doc,
        # to check that the LLM does not rely on the general (pre-trained) knowledge alone,
        # but instead actually uses the document text for extraction
        match_found = False
        for extracted_item in extracted_concepts[2].extracted_items:
            if "thirtieth" in extracted_item.value or "30" in extracted_item.value:
                match_found = True
                break
        assert match_found, "No modified entry into force date found"

        # We intentionally added an anomaly in the middle of the document.
        match_found = False
        for extracted_item in extracted_concepts[3].extracted_items:
            if "Texas" in extracted_item.value:
                match_found = True
                break
        assert match_found, "No anomaly found in the middle of the document"

    def test_repeated_instances_are_extracted_once(self):
        """
        Tests that an aspect or a concept that is passed to an extraction method
        more than once is extracted once: it is listed in one prompt, once, gets
        its items once, and is returned once, with and without concurrency.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        # The IDs of the aspects or concepts that each prompt lists
        listed_ids: list[list[str]] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: each aspect that the prompt lists is found in
            the first paragraph, and each concept listed gets one item.
            """
            prompt = kwargs["messages"][-1]["content"]
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            aspect_ids = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            listed_ids.append(concept_ids or aspect_ids)
            content = (
                [
                    {"concept_id": concept_id, "extracted_items": ["Item"]}
                    for concept_id in concept_ids
                ]
                if concept_ids
                else [
                    {"aspect_id": aspect_id, "paragraph_ids": ["P0"]}
                    for aspect_id in aspect_ids
                ]
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_document() -> Document:
            """
            Creates a document with two aspects, each with a sub-aspect and
            a concept, and with two document-level concepts.
            """
            document = Document(
                raw_text="Services agreement\n\nThe parties agree as follows."
            )
            document.aspects = [
                Aspect(
                    name=f"Aspect {i}",
                    description=f"Clauses on topic {i}",
                    aspects=[
                        Aspect(
                            name=f"Sub-aspect {i}",
                            description=f"Clauses on a part of topic {i}",
                        )
                    ],
                    concepts=[
                        StringConcept(
                            name=f"Aspect concept {i}",
                            description=f"Detail of topic {i}",
                        )
                    ],
                )
                for i in range(2)
            ]
            document.concepts = [
                StringConcept(name=f"Concept {i}", description=f"Detail {i}")
                for i in range(2)
            ]
            return document

        for extraction_kwargs in [{}, {"use_concurrency": True}]:
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            # With concurrency, each instance is listed in a prompt of its own
            in_separate_prompts = bool(extraction_kwargs)

            with patch.object(litellm, "acompletion", new=acompletion_stub):
                # === Aspects of the document ===
                document = create_document()
                aspect, other_aspect = document.aspects
                listed_ids.clear()
                extracted_aspects = llm.extract_aspects_from_document(
                    document,
                    from_aspects=[aspect, other_aspect, aspect],
                    **extraction_kwargs,
                )
                # The aspects, then the sub-aspect of each of them
                assert sorted(listed_ids) == (
                    [["A0"]] * 4
                    if in_separate_prompts
                    else [["A0"], ["A0"], ["A0", "A1"]]
                )
                # The distinct aspects, in the order given
                assert [id(i) for i in extracted_aspects] == [
                    id(aspect),
                    id(other_aspect),
                ]
                for extracted_aspect in extracted_aspects:
                    assert len(extracted_aspect.extracted_items) == 1
                    assert len(extracted_aspect.reference_paragraphs) == 1
                    assert len(extracted_aspect.aspects[0].extracted_items) == 1

                # === Concepts of an aspect ===
                (concept,) = aspect.concepts
                listed_ids.clear()
                extracted_concepts = llm.extract_concepts_from_aspect(
                    aspect,
                    document,
                    from_concepts=[concept, concept, concept],
                    **extraction_kwargs,
                )
                assert listed_ids == [["C0"]]
                assert [id(i) for i in extracted_concepts] == [id(concept)]
                assert [item.value for item in concept.extracted_items] == ["Item"]

                # === Concepts of the document ===
                concept, other_concept = document.concepts
                listed_ids.clear()
                extracted_concepts = llm.extract_concepts_from_document(
                    document,
                    from_concepts=[other_concept, concept, other_concept, concept],
                    **extraction_kwargs,
                )
                assert sorted(listed_ids) == (
                    [["C0"], ["C0"]] if in_separate_prompts else [["C0", "C1"]]
                )
                assert [id(i) for i in extracted_concepts] == [
                    id(other_concept),
                    id(concept),
                ]
                for extracted_concept in extracted_concepts:
                    assert [
                        item.value for item in extracted_concept.extracted_items
                    ] == ["Item"]

            # A request for each prompt, and no other
            usage = llm.get_usage()[0].usage
            assert len(usage.calls) == (4 + 1 + 2 if in_separate_prompts else 3 + 1 + 1)

        # === In a model group ===
        model_group = DocumentModelGroup(
            models=[
                DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key"),
                DocumentLLM(
                    model="openai/gpt-5.6-luna", api_key="test_api_key", role="other"
                ),
            ]
        )
        document = create_document()
        concept, other_concept = document.concepts
        other_concept.model_role = "other"
        listed_ids.clear()
        with patch.object(litellm, "acompletion", new=acompletion_stub):
            extracted_concepts = model_group.extract_concepts_from_document(
                document, from_concepts=[concept, other_concept, concept, other_concept]
            )
        # Each concept is extracted once, by the LLM with its role
        assert listed_ids == [["C0"], ["C0"]]
        assert [id(i) for i in extracted_concepts] == [id(concept), id(other_concept)]
        for extracted_concept in extracted_concepts:
            assert [item.value for item in extracted_concept.extracted_items] == [
                "Item"
            ]

    def test_overwrite_existing_resets_aspect_references(self):
        """
        Tests that an aspect extracted again with ``overwrite_existing=True`` has
        the reference paragraphs and sentences of its new items only, so that its
        sub-aspects and its concepts are extracted from them, and that
        the references of an aspect are still added to without it.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        texts = [
            "Services agreement",
            "The client shall pay a fee of 5,000 euros every month.",
            "Either party may terminate this agreement with a notice of 30 days.",
            "This agreement is governed by the laws of Norway.",
        ]
        responses: list[str] = []
        prompts: list[str] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: records the prompt, and returns the next
            response of the queue.
            """
            prompts.append(kwargs["messages"][-1]["content"])
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=responses.pop(0))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_document() -> Document:
            """
            Creates a document with one sentence in each paragraph, and an aspect
            whose items are sentences, with a sub-aspect and a concept.
            """
            document = Document(
                paragraphs=[
                    Paragraph(raw_text=text, sentences=[Sentence(raw_text=text)])
                    for text in texts
                ]
            )
            document.aspects = [
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    reference_depth="sentences",
                    aspects=[Aspect(name="Notices", description="Clauses on notices")],
                    concepts=[
                        StringConcept(
                            name="Notice period",
                            description="Length of the notice period",
                        )
                    ],
                )
            ]
            return document

        def get_aspect_response(paragraph_positions: list[int]) -> str:
            """
            Returns the response of the LLM with the first sentence of each of
            the given paragraphs extracted for the aspect.
            """
            paragraphs_json = ", ".join(
                f'{{"paragraph_id": "P{position}", '
                f'"sentences": [{{"sentence_id": "P{position}-S0"}}]}}'
                for position in paragraph_positions
            )
            return f'[{{"aspect_id": "A0", "paragraphs": [{paragraphs_json}]}}]'

        sub_aspect_response = '[{"aspect_id": "A0", "paragraph_ids": ["P0"]}]'
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        document = create_document()
        (aspect,) = document.aspects
        paragraphs = document.paragraphs

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # === The first extraction ===
            responses.extend([get_aspect_response([1, 2]), sub_aspect_response])
            llm.extract_aspects_from_document(document)
            assert [i.value for i in aspect.extracted_items] == texts[1:3]
            assert [id(i) for i in aspect.reference_paragraphs] == [
                id(paragraphs[1]),
                id(paragraphs[2]),
            ]
            assert [id(i) for i in aspect.reference_sentences] == [
                id(paragraphs[1].sentences[0]),
                id(paragraphs[2].sentences[0]),
            ]
            # The sub-aspect is extracted from the paragraphs of the aspect
            assert texts[1] in prompts[1] and texts[2] in prompts[1]
            assert texts[0] not in prompts[1] and texts[3] not in prompts[1]

            # === The aspect is extracted again, without one of the paragraphs ===
            prompts.clear()
            responses.extend([get_aspect_response([2]), sub_aspect_response])
            llm.extract_aspects_from_document(document, overwrite_existing=True)
            assert [i.value for i in aspect.extracted_items] == [texts[2]]
            # The references are those of the new items only
            assert [id(i) for i in aspect.reference_paragraphs] == [id(paragraphs[2])]
            assert [id(i) for i in aspect.reference_sentences] == [
                id(paragraphs[2].sentences[0])
            ]
            assert [id(i) for i in aspect.reference_paragraphs] == [
                id(paragraph)
                for item in aspect.extracted_items
                for paragraph in item.reference_paragraphs
            ]
            assert [id(i) for i in aspect.reference_sentences] == [
                id(sentence)
                for item in aspect.extracted_items
                for sentence in item.reference_sentences
            ]
            # The sub-aspect is extracted from the new paragraphs only
            assert len(prompts) == 2
            assert texts[2] in prompts[1]
            assert texts[1] not in prompts[1]
            assert [i.value for i in aspect.aspects[0].extracted_items] == [texts[2]]
            # ... and so is the concept of the aspect
            prompts.clear()
            responses.append('[{"concept_id": "C0", "extracted_items": ["30 days"]}]')
            llm.extract_concepts_from_aspect(aspect, document)
            (concept_prompt,) = prompts
            assert texts[2] in concept_prompt
            assert texts[1] not in concept_prompt
            assert aspect.concepts[0].extracted_items[0].value == "30 days"

            # An aspect that nothing is extracted for any more has no references
            responses.append(get_aspect_response([]))
            llm.extract_aspects_from_document(document, overwrite_existing=True)
            assert aspect._is_processed
            assert not aspect.extracted_items
            assert not aspect.reference_paragraphs
            assert not aspect.reference_sentences

            # === Without overwriting ===
            # The references that an aspect has are kept, and those of its items
            # are added to them, as before
            document = create_document()
            (aspect,) = document.aspects
            paragraphs = document.paragraphs
            aspect.reference_paragraphs = [paragraphs[0]]
            aspect.reference_sentences = [paragraphs[0].sentences[0]]
            responses.extend([get_aspect_response([2]), sub_aspect_response])
            llm.extract_aspects_from_document(document)
            assert [id(i) for i in aspect.reference_paragraphs] == [
                id(paragraphs[0]),
                id(paragraphs[2]),
            ]
            assert [id(i) for i in aspect.reference_sentences] == [
                id(paragraphs[0].sentences[0]),
                id(paragraphs[2].sentences[0]),
            ]
        assert not responses

    def test_negative_max_per_call_params_are_rejected(self):
        """
        Tests that a negative max number of paragraphs or of images to analyze
        per call is rejected by each extraction method that takes it, before any
        request is sent and before anything else is checked, for an LLM,
        a decision model and a model group. With a negative max number of
        paragraphs, an LLM was sent nothing and extracted nothing, without
        an error.

        The APIs are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        requests_sent: list[dict[str, Any]] = []

        async def api_stub(**kwargs: Any):
            """
            Records a request, which is sent where none is expected.
            """
            requests_sent.append(kwargs)
            raise AssertionError("A request was sent")

        def create_document() -> Document:
            """
            Creates a document with an aspect that has a concept, and with
            a document-level concept, all of which a decision model can extract.
            """
            document = Document(
                raw_text="Services agreement\n\nThe parties agree as follows."
            )
            document.aspects = [
                Aspect(
                    name="Parties",
                    description="Clauses on the parties",
                    concepts=[
                        BooleanConcept(
                            name="Parties named",
                            description="The parties are named",
                        )
                    ],
                )
            ]
            document.concepts = [
                BooleanConcept(
                    name="Agreement", description="The document is an agreement"
                )
            ]
            return document

        models = [
            DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key"),
            DocumentDecisionModel(model="typesafe/jev-latest", api_key="test_api_key"),
            DocumentModelGroup(
                models=[
                    DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key"),
                    DocumentDecisionModel(
                        model="typesafe/jev-latest",
                        api_key="test_api_key",
                        role="other",
                    ),
                ]
            ),
        ]
        with (
            patch.object(litellm, "acompletion", new=api_stub),
            patch.object(litellm, "adecisions", new=api_stub),
        ):
            for model in models:
                for value in (-1, -50):
                    # === Max number of paragraphs ===
                    document = create_document()
                    (aspect,) = document.aspects
                    error = (
                        rf"^`max_paragraphs_to_analyze_per_call` cannot be negative: "
                        rf"got {value}\. Use 0, the default, to set no limit\.$"
                    )
                    with pytest.raises(ValueError, match=error):
                        model.extract_all(
                            document, max_paragraphs_to_analyze_per_call=value
                        )
                    with pytest.raises(ValueError, match=error):
                        model.extract_aspects_from_document(
                            document, max_paragraphs_to_analyze_per_call=value
                        )
                    # The aspect is not extracted yet, which is checked only
                    # after the max number
                    with pytest.raises(ValueError, match=error):
                        model.extract_concepts_from_aspect(
                            aspect, document, max_paragraphs_to_analyze_per_call=value
                        )
                    with pytest.raises(ValueError, match=error):
                        model.extract_concepts_from_document(
                            document, max_paragraphs_to_analyze_per_call=value
                        )

                    # === Max number of images ===
                    error = (
                        rf"^`max_images_to_analyze_per_call` cannot be negative: "
                        rf"got {value}\. Use 0, the default, to set no limit\.$"
                    )
                    with pytest.raises(ValueError, match=error):
                        model.extract_all(
                            document, max_images_to_analyze_per_call=value
                        )
                    with pytest.raises(ValueError, match=error):
                        model.extract_concepts_from_document(
                            document, max_images_to_analyze_per_call=value
                        )

                    # Nothing is extracted, and nothing is marked as processed
                    instances = [aspect, *aspect.concepts, *document.concepts]
                    assert not any(i.extracted_items for i in instances)
                    assert not any(i._is_processed for i in instances)
        # No request is sent
        assert not requests_sent

    def test_concepts_with_sentence_references_after_aspects_are_extracted(self):
        """
        Tests that a concept with references to sentences is extracted from
        an aspect, and from a sub-aspect, also when the sentences of the document
        are segmented only after the aspects were extracted: when the concept is
        attached after that, when the document is restored from its serialized
        form in between, and when another call segmented the sentences in
        between.

        The paragraphs of a sub-aspect, and of any aspect of a restored document,
        are copies of the paragraphs of the document, which the sentences
        segmented later did not reach: the extraction from a sub-aspect failed
        with an error that the sub-aspect is not assigned to the document, and
        the prompt for an aspect of a restored document had no text.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        paragraph_texts = [
            "The supplier delivers the goods. Delivery takes 30 days.",
            "The buyer pays the price. Payment is due in 14 days.",
            "This agreement is governed by the laws of Norway.",
        ]
        sentence_texts = [re.split(r"(?<=\.) ", text) for text in paragraph_texts]
        # The texts that the sentence segmentation is run on, by each run
        segmented_texts: list[list[str]] = []
        prompts: list[str] = []

        class SentenceSplitter:
            """
            Replaces the SaT model: splits a text after each full stop.
            """

            def split(self, texts: list[str]) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                segmented_texts.append(texts)
                return [re.split(r"(?<=\.) ", text) for text in texts]

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: the first two paragraphs that a prompt lists
            are extracted for each aspect, and each concept gets one item, with
            the second sentence of the first paragraph listed as its reference.
            """
            prompt = kwargs["messages"][-1]["content"]
            prompts.append(prompt)
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            aspect_ids = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            content = (
                [
                    {
                        "concept_id": concept_id,
                        "extracted_items": [
                            {
                                "value": "30 days",
                                "reference_paragraphs": [
                                    {
                                        "reference_paragraph_id": "P0",
                                        "reference_sentence_ids": ["P0-S1"],
                                    }
                                ],
                            }
                        ],
                    }
                    for concept_id in concept_ids
                ]
                if concept_ids
                else [
                    {"aspect_id": aspect_id, "paragraph_ids": ["P0", "P1"]}
                    for aspect_id in aspect_ids
                ]
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_concept() -> StringConcept:
            """
            Creates a concept with references to sentences.
            """
            return StringConcept(
                name="Number of days",
                description="Number of days of a deadline",
                add_references=True,
                reference_depth="sentences",
            )

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(
                documents_module,
                "_load_sat_model",
                new=lambda model_id: SentenceSplitter(),
            ),
        ):
            for scenario in ("attached_after", "restored", "segmented_by_other_call"):
                llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
                document = Document(raw_text="\n\n".join(paragraph_texts))
                document.aspects = [
                    Aspect(
                        name="Duties",
                        description="Duties of the parties",
                        aspects=[
                            Aspect(
                                name="Deadlines",
                                description="Deadlines of the duties",
                            )
                        ],
                    )
                ]
                # Nothing requires sentences when the aspects are extracted
                llm.extract_aspects_from_document(document)
                assert not document.sentences
                assert not segmented_texts

                if scenario == "restored":
                    document = Document.from_dict(document.to_dict())
                    # The paragraphs of the aspect are not those of the document
                    assert not any(
                        paragraph is document_paragraph
                        for paragraph in document.aspects[0].reference_paragraphs
                        for document_paragraph in document.paragraphs
                    )
                elif scenario == "segmented_by_other_call":
                    document.concepts = [create_concept()]
                    llm.extract_concepts_from_document(document)
                    assert document.sentences

                (aspect,) = document.aspects
                aspect.concepts = [create_concept()]
                aspect.aspects[0].concepts = [create_concept()]
                prompts.clear()
                llm.extract_concepts_from_aspect(aspect, document)

                # The sentences are segmented once, for the paragraphs of
                # the document, and not for the copies of the paragraphs
                assert segmented_texts == [paragraph_texts], scenario
                # The aspect and the sub-aspect have the first two paragraphs:
                # the prompt for the concept of each lists their sentences
                assert len(prompts) == 2, scenario
                for prompt in prompts:
                    for paragraph_position, texts in enumerate(sentence_texts[:2]):
                        for sentence_position, text in enumerate(texts):
                            sentence_id = f"P{paragraph_position}-S{sentence_position}"
                            assert f"<sentence_id>{sentence_id}</sentence_id>" in prompt
                            assert text in prompt
                    assert paragraph_texts[2] not in prompt

                for instance in (aspect, aspect.aspects[0]):
                    (concept,) = instance.concepts
                    (item,) = concept.extracted_items
                    assert item.value == "30 days"
                    # The references of the item are found in the document
                    (reference_paragraph,) = item.reference_paragraphs
                    (reference_sentence,) = item.reference_sentences
                    assert reference_sentence.raw_text == sentence_texts[0][1]
                    assert document.get_paragraph_index(reference_paragraph) == 0
                    assert document.get_sentence_index(reference_sentence) == (0, 1)
                    # The paragraphs of the aspect have the sentences of
                    # the paragraphs of the document
                    assert len(instance.reference_paragraphs) == 2
                    for paragraph in instance.reference_paragraphs:
                        document_paragraph = document.paragraphs[
                            document.get_paragraph_index(paragraph)
                        ]
                        assert paragraph == document_paragraph
                        assert [s.unique_id for s in paragraph.sentences] == [
                            s.unique_id for s in document_paragraph.sentences
                        ]
                segmented_texts.clear()

    def test_aspect_reference_sentences_are_those_of_its_items(self):
        """
        Tests that the reference sentences of an aspect with sentences as its
        items (``reference_depth="sentences"``) are the reference sentences of
        its extracted items, in the order of the items, each once: when
        the aspect is extracted, when it is extracted again with
        ``overwrite_existing=True``, and when the references of its items are
        added to those that the aspect already has.

        The aspect had all the sentences of each of its reference paragraphs
        instead, i.e. also those that were not extracted: five reference
        sentences for a paragraph of five sentences of which one was extracted.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        # The sentences of each paragraph of the document
        sentence_texts = [
            ["Services agreement"],
            [
                "The client shall pay a fee of 5,000 euros every month.",
                "The fee is paid in advance.",
                "Invoices are sent by email.",
                "Late payments bear interest.",
                "The supplier reports on the services every quarter.",
            ],
            [
                "Either party may terminate this agreement with a notice of 30 days.",
                "The notice shall be given in writing.",
                "Termination does not affect the accrued rights.",
            ],
            [
                "This agreement is governed by the laws of Norway.",
                "The courts of Oslo have jurisdiction.",
            ],
        ]
        responses: list[str] = []
        prompts: list[str] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: records the prompt, and returns the next
            response of the queue.
            """
            prompts.append(kwargs["messages"][-1]["content"])
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=responses.pop(0))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_document() -> Document:
            """
            Creates a document whose paragraphs have their sentences, up to
            five in a paragraph, and an aspect whose items are sentences, with
            a concept.
            """
            document = Document(
                paragraphs=[
                    Paragraph(
                        raw_text=" ".join(texts),
                        sentences=[Sentence(raw_text=text) for text in texts],
                    )
                    for texts in sentence_texts
                ]
            )
            document.aspects = [
                Aspect(
                    name="Duties",
                    description="Duties of the parties",
                    reference_depth="sentences",
                    concepts=[
                        StringConcept(
                            name="Notice period",
                            description="Length of the notice period",
                        )
                    ],
                )
            ]
            return document

        def get_aspect_response(sentence_positions: list[tuple[int, int]]) -> str:
            """
            Returns the response of the LLM with the given sentences extracted
            for the aspect, each given by the positions of its paragraph and of
            the sentence in it.
            """
            sentences_of_paragraphs: dict[str, list[dict[str, str]]] = {}
            for paragraph_position, sentence_position in sentence_positions:
                paragraph_id = f"P{paragraph_position}"
                sentences_of_paragraphs.setdefault(paragraph_id, []).append(
                    {"sentence_id": f"{paragraph_id}-S{sentence_position}"}
                )
            paragraphs_listed = [
                {"paragraph_id": paragraph_id, "sentences": sentences}
                for paragraph_id, sentences in sentences_of_paragraphs.items()
            ]
            return json.dumps([{"aspect_id": "A0", "paragraphs": paragraphs_listed}])

        def check_references(
            document: Document,
            paragraph_positions: list[int],
            sentence_positions: list[tuple[int, int]],
        ) -> None:
            """
            Checks that the references of the aspect of the document are
            the given paragraphs and the given sentences of the document,
            in the given order.
            """
            (aspect,) = document.aspects
            paragraphs = document.paragraphs
            assert [id(i) for i in aspect.reference_paragraphs] == [
                id(paragraphs[position]) for position in paragraph_positions
            ]
            assert [id(i) for i in aspect.reference_sentences] == [
                id(paragraphs[paragraph_position].sentences[sentence_position])
                for paragraph_position, sentence_position in sentence_positions
            ]

        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        document = create_document()
        (aspect,) = document.aspects

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # === The first extraction ===
            # One of the five sentences of the second paragraph, and two of
            # the three sentences of the third one
            extracted_sentences = [(1, 0), (2, 0), (2, 2)]
            responses.append(get_aspect_response(extracted_sentences))
            llm.extract_aspects_from_document(document)
            assert [i.value for i in aspect.extracted_items] == [
                sentence_texts[paragraph_position][sentence_position]
                for paragraph_position, sentence_position in extracted_sentences
            ]
            # The reference sentences of the aspect are the three extracted
            # sentences, and not the eight sentences of its two reference
            # paragraphs. A sentence is a reference sentence also when its
            # paragraph is the one of a sentence extracted before it.
            check_references(document, [1, 2], extracted_sentences)
            # ... i.e. they are those of the items of the aspect
            assert [id(i) for i in aspect.reference_sentences] == [
                id(sentence)
                for item in aspect.extracted_items
                for sentence in item.reference_sentences
            ]
            # The concept of the aspect is extracted from the reference
            # paragraphs of the aspect, i.e. also from their sentences that
            # are not extracted for the aspect
            prompts.clear()
            responses.append('[{"concept_id": "C0", "extracted_items": ["30 days"]}]')
            llm.extract_concepts_from_aspect(aspect, document)
            (concept_prompt,) = prompts
            assert all(
                text in concept_prompt
                for text in [*sentence_texts[1], *sentence_texts[2]]
            )
            assert not any(
                text in concept_prompt
                for text in [*sentence_texts[0], *sentence_texts[3]]
            )

            # === The aspect is extracted again, with other sentences ===
            extracted_sentences = [(1, 4), (3, 1)]
            responses.append(get_aspect_response(extracted_sentences))
            llm.extract_aspects_from_document(document, overwrite_existing=True)
            assert [i.value for i in aspect.extracted_items] == [
                sentence_texts[paragraph_position][sentence_position]
                for paragraph_position, sentence_position in extracted_sentences
            ]
            # The references are those of the new items only: the sentence of
            # the second paragraph that was extracted before is not among them,
            # while its paragraph is a reference paragraph again
            check_references(document, [1, 3], extracted_sentences)

            # === Without overwriting ===
            # The references that an aspect has are kept, and those of its items
            # are added to them: a sentence that the aspect already has is not
            # added again, and a new sentence is added also when its paragraph
            # is a reference paragraph already
            document = create_document()
            (aspect,) = document.aspects
            paragraphs = document.paragraphs
            aspect.reference_paragraphs = [paragraphs[2]]
            aspect.reference_sentences = [paragraphs[2].sentences[1]]
            responses.append(get_aspect_response([(1, 3), (2, 1), (2, 2)]))
            llm.extract_aspects_from_document(document)
            assert len(aspect.extracted_items) == 3
            check_references(document, [2, 1], [(2, 1), (1, 3), (2, 2)])
        assert not responses
