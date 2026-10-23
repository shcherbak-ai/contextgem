# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the extraction of aspects and concepts.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import types
import warnings
from contextlib import contextmanager
from typing import Any, Literal
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter

from contextgem.internal.base import documents as documents_module
from contextgem.internal.base import llms as llms_module
from contextgem.internal.base import models as models_module
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.data_models import _Message, _ModelCall, _ModelUsage
from contextgem.internal.exceptions import (
    ContextGemWarning,
    ModelAPIError,
    ModelExtractionError,
    ModelRefusalError,
)
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
from tests.utils import (
    capture_logs,
    get_project_root_path,
    get_test_document_text,
)


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
            refusal_msg_suffix: str = "",
        ):
            """
            Simulates a successful call that returns invalid JSON.

            Same params as `llm._query_llm`.
            """
            return "<<<not-json>>>", _ModelUsage()

        _orig_query_llm = llm._query_llm
        llm._query_llm = types.MethodType(_invalid_json_query_stub, llm)  # ty: ignore[invalid-assignment]
        try:
            with pytest.raises(ModelExtractionError, match=r"invalid data.*1 retries"):
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
            with pytest.raises(ModelExtractionError, match=r"invalid data.*1 retries"):
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ModelExtractionError, match=r"invalid data.*1 retries"):
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
                    re.search(r"invalid data.*1 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid data.*1 retries' not found in logs: {captured_logs}"
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
                    re.search(r"invalid data.*1 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid data.*1 retries' not found in logs: {captured_logs}"
                )

            # The concept of the aspect is not extracted, with a warning, as the
            # aspect was not extracted: there's no aspect context to extract it
            # from. The concept of the document is still extracted.
            with capture_logger_warnings() as captured_logs:
                llm.extract_all(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                assert any(
                    "Concepts of aspect `Liability`" in log for log in captured_logs
                ), captured_logs
                assert any(
                    re.search(
                        r"Failed concepts: \['Contract term'\] \(1 retries\)", log
                    )
                    for log in captured_logs
                ), captured_logs
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
            with pytest.raises(ModelExtractionError, match=r"invalid data.*0 retries"):
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
            with pytest.raises(ModelExtractionError, match=r"invalid data.*0 retries"):
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                )
            with pytest.raises(ModelExtractionError, match=r"invalid data.*0 retries"):
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
                    re.search(r"invalid data.*0 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid data.*0 retries' not found in logs: {captured_logs}"
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
                    re.search(r"invalid data.*0 retries", log) for log in captured_logs
                ), (
                    f"Expected warning pattern 'invalid data.*0 retries' not found in logs: {captured_logs}"
                )

            # The concept of the aspect is not extracted, with a warning, as the
            # aspect was not extracted: there's no aspect context to extract it
            # from. The concept of the document is still extracted.
            with capture_logger_warnings() as captured_logs:
                llm.extract_all(
                    document,
                    overwrite_existing=True,
                    raise_exception_on_extraction_error=False,
                )
                assert any(
                    "Concepts of aspect `Liability`" in log for log in captured_logs
                ), captured_logs
                assert any(
                    re.search(
                        r"Failed concepts: \['Contract term'\] \(0 retries\)", log
                    )
                    for log in captured_logs
                ), captured_logs
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

        # The concept of the aspect is not extracted, with a warning, as the aspect
        # was not extracted, and the failure of the request for the concept of
        # the document is a warning too
        with capture_logger_warnings() as captured_logs:
            self.llm_invalid.extract_all(
                document,
                overwrite_existing=True,
                raise_exception_on_extraction_error=False,
            )
            assert any(
                "Concepts of aspect `Liability`" in log for log in captured_logs
            ), captured_logs
            assert any(re.search(r"LLM API", log) for log in captured_logs), (
                captured_logs
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
                from_aspects={1, 2, 3},  # type: ignore
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
            expected_n_calls_with_concurrency=7,
            func=llm.extract_concepts_from_aspect,
            func_kwargs={
                "aspect": attached_aspect,
                "document": document,
            },
            original_container=list(aspects[0].concepts),
            assigned_container=list(attached_aspect.concepts),
            assigned_instance_class=_Concept,
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
                from_concepts={1, 2, 3},  # type: ignore
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
                from_concepts={1, 2, 3},  # type: ignore
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
        document = Document(paragraphs=paragraphs)
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
                DocumentLLM(**self._llm_extractor_kwargs_openai),
                DocumentLLM(
                    **self._llm_extractor_kwargs_openai,
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
        # The prompt shows the context that a sentence shares with its paragraph
        # once, as that of the paragraph
        sentence_call_objs = [
            c
            for c in next(
                i for i in model_group.get_usage() if i.role is None
            ).usage.calls
            if c.prompt_kwargs.get("reference_depth") == "sentences"
        ]
        assert sentence_call_objs
        for call_obj in sentence_call_objs:
            shown = [
                context
                for context in re.findall(
                    r"<additional_context>\s*(.*?)\s*</additional_context>",
                    call_obj.prompt,
                    re.DOTALL,
                )
                if context
            ]
            assert len(shown) == len(call_obj.prompt_kwargs["paragraphs"])

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
            match="https://contextgem.shcherbak.ai/optimizations/optimization_long_docs/",
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

    def test_error_raised_in_concurrent_call_is_raised_as_it_is(self):
        """
        Tests that an error raised in an extraction call, other than a failure
        of the model that is reported through the result of the call (a failed
        API request, a cut-off response, an output that cannot be used), reaches
        the calling code as it is with concurrency, as it does without it:
        whatever the retries for invalid data are, with the concepts extracted
        in one call and in several calls, whether or not an exception is to be
        raised on an extraction error, and for an LLM, a decision model and
        a model group. Such an error is the one for a prompt that exceeds
        the context window of an LLM, when there is no fallback LLM left to try,
        and any other error.

        With concurrency and several calls, such an error was taken for
        an output of the model that cannot be used: the call was made again,
        or, with no retries for invalid data and no fallback model,
        the extraction failed with an error on invalid JSON, or only warned
        of it.

        The APIs are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"
        n_concepts = 4
        get_model_info = litellm.get_model_info
        validate_parsed_llm_output = models_module._validate_parsed_llm_output
        # The models that are given a context window too small for any prompt
        small_context_window_models: set[str] = set()
        requested_models: list[str] = []

        def get_model_info_stub(model: str, *args: Any, **kwargs: Any) -> Any:
            """
            Returns the info of the model, with a context window of 10 tokens
            for a model that is given a small one.
            """
            model_info = dict(get_model_info(model, *args, **kwargs))
            if model in small_context_window_models:
                model_info["max_input_tokens"] = 10
            return model_info

        def respond(request: dict[str, Any]) -> Any:
            """
            Creates the response of the LLM API to the request: an item for
            each concept that the prompt lists.
            """
            prompt = request["messages"][-1]["content"]
            content = [
                {"concept_id": concept_id, "extracted_items": [True]}
                for concept_id in re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            ]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: records the model of the request, and
            responds to the request.
            """
            requested_models.append(kwargs["model"])
            return respond(kwargs)

        async def adecisions_stub(**kwargs: Any):
            """
            Records a request to a decision model, which is sent where none
            is expected.
            """
            requested_models.append(kwargs["model"])
            raise AssertionError("A request was sent")

        def create_llm(
            max_retries_invalid_data: int, with_fallback: bool = False
        ) -> DocumentLLM:
            """
            Creates an LLM with the given number of retries for invalid data,
            optionally with a fallback LLM.
            """
            llm = DocumentLLM(
                model=primary_model,
                api_key="test_api_key",
                max_retries_invalid_data=max_retries_invalid_data,
            )
            if with_fallback:
                llm.fallback_models = [
                    DocumentLLM(
                        model=fallback_model, api_key="test_api_key", is_fallback=True
                    )
                ]
            return llm

        def create_document(with_text: bool = True) -> Document:
            """
            Creates a document that has a text, or only an image, with concepts
            that both an LLM and a decision model can extract from a text. In
            a model group, half of the concepts are routed to the model with
            the role "other".
            """
            document = (
                Document(raw_text="Services agreement\n\nThe parties agree as follows.")
                if with_text
                else Document(images=[self.test_img_png_invoice])
            )
            document.concepts = [
                BooleanConcept(
                    name=f"Concept {i}",
                    description=f"Topic {i} is covered",
                    model_role="other" if i % 2 else None,
                )
                for i in range(n_concepts)
            ]
            return document

        def get_context_window_error(model: str) -> str:
            """
            Returns the pattern of the error for a prompt that exceeds
            the context window of the model.
            """
            return (
                r"^Input messages contain \d+ tokens, which exceeds the model's "
                rf"maximum input tokens of 10 for model `{re.escape(model)}`\. "
            )

        def check_error_is_raised_as_it_is(
            model: DocumentLLM | DocumentDecisionModel | DocumentModelGroup,
            error: str,
            document_has_text: bool = True,
        ) -> None:
            """
            Checks that the extraction of the concepts fails with the given
            error, which is raised in a call before any request is sent, in
            the same way with and without concurrency: with each concept
            extracted in a call of its own and with all the concepts of a model
            extracted in one call, and whether or not an exception is to be
            raised on an extraction error.
            """
            for raise_exception in (True, False):
                for max_items_per_call in (1, n_concepts):
                    error_messages = []
                    for use_concurrency in (False, True):
                        document = create_document(with_text=document_has_text)
                        requested_models.clear()
                        with pytest.raises(ValueError, match=error) as exc_info:
                            model.extract_concepts_from_document(
                                document,
                                max_items_per_call=max_items_per_call,
                                use_concurrency=use_concurrency,
                                raise_exception_on_extraction_error=raise_exception,
                            )
                        error_messages.append(str(exc_info.value))
                        assert not requested_models
                        assert not any(
                            i.extracted_items or i._is_processed
                            for i in document.concepts
                        )
                    # The same error with and without concurrency
                    assert error_messages[0] == error_messages[1]

        async def check_other_calls_are_stopped(
            llm: DocumentLLM, raise_exception: bool
        ) -> None:
            """
            Extracts the concepts concurrently, each in a call of its own, with
            an error raised when the response for the first concept is
            processed, and checks that the error is raised as it is when
            the other calls are stopped. By the time of the error, the call of
            the second concept is completed, and the requests of the other
            calls are sent and not answered.
            """
            document = create_document()
            failed, completed, *in_flight = document.concepts
            error = RuntimeError("Simulated error")
            requests: list[tuple[str, str]] = []
            cancelled_requests: list[str] = []

            async def answer_in_turn(**kwargs: Any):
                """
                Simulates the LLM API: the request for the second concept is
                answered at once, the request for the first concept when
                the second concept is extracted, and the other requests are
                not answered.
                """
                (name,) = re.findall(
                    r"<concept_name>(Concept \d)</concept_name>",
                    kwargs["messages"][-1]["content"],
                )
                requests.append((kwargs["model"], name))
                if name == failed.name:
                    for _ in range(1000):
                        if completed._is_processed:
                            break
                        await asyncio.sleep(0)
                elif name != completed.name:
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        cancelled_requests.append(name)
                        raise
                return respond(kwargs)

            def validate_or_fail(**kwargs: Any) -> Any:
                """
                Fails the processing of the response for the first concept,
                i.e. of the response that is received when the second concept
                is extracted, and validates the response for the second one.
                """
                if completed._is_processed:
                    raise error
                return validate_parsed_llm_output(**kwargs)

            with (
                patch.object(litellm, "acompletion", new=answer_in_turn),
                patch.object(
                    models_module, "_validate_parsed_llm_output", new=validate_or_fail
                ),
                pytest.raises(RuntimeError) as exc_info,
            ):
                await llm.extract_concepts_from_document_async(
                    document,
                    use_concurrency=True,
                    raise_exception_on_extraction_error=raise_exception,
                )
            assert exc_info.value is error
            # The requests in flight are cancelled, and no task is left running
            assert sorted(cancelled_requests) == [i.name for i in in_flight]
            current_task = asyncio.current_task()
            assert not [
                task
                for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            ]
            # No call is made again, and no fallback LLM is tried
            assert sorted(requests) == [
                (primary_model, i.name) for i in document.concepts
            ]
            # The concept extracted by then keeps its item. The usage counts
            # the request for it, and the request whose response the error was
            # raised for; the requests cancelled in flight are not counted.
            assert [i.value for i in completed.extracted_items] == [True]
            assert not any(
                i.extracted_items or i._is_processed for i in (failed, *in_flight)
            )
            usage = llm.get_usage()[0].usage
            assert (usage.input, usage.output) == (20, 10)
            assert len(usage.calls) == 2

        # Without and with the retries for invalid data
        llms = [create_llm(0), create_llm(3)]
        llms_with_fallback = [
            create_llm(0, with_fallback=True),
            create_llm(3, with_fallback=True),
        ]
        decision_model = DocumentDecisionModel(
            model="typesafe/jev-latest", api_key="test_api_key"
        )
        model_group = DocumentModelGroup(
            models=[
                create_llm(0),
                DocumentDecisionModel(
                    model="typesafe/jev-latest", api_key="test_api_key", role="other"
                ),
            ]
        )
        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(litellm, "adecisions", new=adecisions_stub),
            patch.object(litellm, "get_model_info", new=get_model_info_stub),
        ):
            # === The prompt exceeds the context window of the LLM ===
            # No fallback LLM: the error of the LLM is raised
            small_context_window_models.add(primary_model)
            for llm in llms:
                check_error_is_raised_as_it_is(
                    llm, error=get_context_window_error(primary_model)
                )
            # A call that fails in this way is not made again: with concurrency,
            # each call is recorded once at most, as unfinished
            llm = llms[1]
            llm.reset_usage_and_cost()
            with pytest.raises(ValueError):
                llm.extract_concepts_from_document(
                    create_document(), use_concurrency=True
                )
            calls = llm.get_usage()[0].usage.calls
            assert 1 <= len(calls) <= n_concepts
            assert all(call.timestamp_received is None for call in calls)

            # A fallback LLM with a larger context window takes over
            for llm in llms_with_fallback:
                for max_items_per_call in (1, n_concepts):
                    for use_concurrency in (False, True):
                        document = create_document()
                        requested_models.clear()
                        llm.extract_concepts_from_document(
                            document,
                            max_items_per_call=max_items_per_call,
                            use_concurrency=use_concurrency,
                        )
                        # A request for each call, to the fallback LLM only
                        assert requested_models == [fallback_model] * (
                            n_concepts // max_items_per_call
                        )
                        assert all(i.extracted_items for i in document.concepts)

            # A fallback LLM whose context window is too small as well:
            # the error of the fallback LLM, the last one to try, is raised
            small_context_window_models.add(fallback_model)
            for llm in llms_with_fallback:
                check_error_is_raised_as_it_is(
                    llm, error=get_context_window_error(fallback_model)
                )
            small_context_window_models.clear()

            # === The document has no text to extract the concepts from ===
            # The error is raised in each call, by a model of any kind
            for model in [*llms, decision_model, model_group]:
                check_error_is_raised_as_it_is(
                    model,
                    error=r"^Document lacks text or paragraphs for concept "
                    r"extraction\.$",
                    document_has_text=False,
                )

            # === An error of another type, raised when a response is processed ===
            for llm in [*llms, llms_with_fallback[0]]:
                for raise_exception in (True, False):
                    llm.reset_usage_and_cost()
                    asyncio.run(check_other_calls_are_stopped(llm, raise_exception))

    def test_failed_concurrent_call_stops_other_calls(self):
        """
        Tests that, with concurrency, a call that fails the extraction stops
        the other calls of the same model before its error is raised, as an error
        that is not a failure of the model does: the requests in flight are
        cancelled, no request is sent afterwards, no task is left running, and
        the items and the usage of the calls completed by then are kept. Such
        a call is one whose API request fails, or whose response is cut off at
        the max output tokens, with no fallback model left to try, among
        the first calls and among the calls made again, for an LLM, a decision
        model and in a model group. The same applies to a call whose data is
        invalid and that is not made again: a call made again whose data is still
        invalid after the last retry, and a call of a model that has no retries
        for invalid data and no fallback model, i.e. of an LLM with
        ``max_retries_invalid_data=0`` or of a decision model.

        A failure that does not fail the extraction changes nothing: the other
        calls go on when a fallback model is yet to be tried, and when
        the failure is only warned of.

        The other calls were completed, and the requests not yet sent were
        sent, before the error of such a call was raised.

        The APIs are stubbed, i.e. no cassette is required.
        """
        # The types of the responses of litellm to a decision model, which only
        # this test of the module builds
        from litellm.types import decisions as decision_types

        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"
        decision_model = "typesafe/jev-latest"
        concept_names = [f"Concept {i}" for i in range(4)]
        invalid_answer = "<<<not-json>>>"
        cut_off_answer = '[{"concept_id": "C0", "extracted_items": [tr'
        api_error = litellm.AuthenticationError(
            message="Simulated API error", model=primary_model, llm_provider="openai"
        )
        # The outcome of a request that is not answered, i.e. stays in flight
        in_flight = object()
        # The outcomes of the requests to a model for a concept, in turn, by
        # the model and the name of the concept: an error to raise, a text for
        # the LLM to return, or a request in flight, optionally with a condition
        # that the request waits for. A request with no outcome set is answered
        # at once, with an item for the concept.
        outcomes: dict[tuple[str, str], list[Any]] = {}
        # The requests sent, answered, failed and cancelled in flight, each as
        # the model and the name of the concept
        requests: list[tuple[str, str]] = []
        responses: list[tuple[str, str]] = []
        failed_requests: list[tuple[str, str]] = []
        cancelled_requests: list[tuple[str, str]] = []

        async def send_request(model: str, text: str) -> Any:
            """
            Records the request of the model for the concept that the text names,
            and gives its outcome, when the condition of the outcome is met:
            raises the error, or returns the text to answer with, or None for
            the answer with an item. A request in flight is answered only if it
            is not cancelled within five seconds.
            """
            (name,) = set(re.findall(r"Concept \d", text))
            request = (model, name)
            requests.append(request)
            outcome = outcomes[request].pop(0) if outcomes.get(request) else None
            if isinstance(outcome, tuple):
                condition, outcome = outcome
                for _ in range(1000):
                    if condition():
                        break
                    await asyncio.sleep(0)
            if outcome is in_flight:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    cancelled_requests.append(request)
                    raise
                outcome = None
            if isinstance(outcome, Exception):
                failed_requests.append(request)
                raise outcome
            responses.append(request)
            return outcome

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API. A response uses 10 input tokens, and is cut
            off if its text is the cut-off answer.
            """
            content = await send_request(
                kwargs["model"], kwargs["messages"][-1]["content"]
            )
            if content is None:
                content = json.dumps([{"concept_id": "C0", "extracted_items": [True]}])
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        finish_reason="length" if content == cut_off_answer else "stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        async def adecisions_stub(**kwargs: Any):
            """
            Simulates the API of the decision model, which answers "yes" to
            the question of the request, or gives no answer to it if the outcome
            of the request is the invalid answer. A response uses 10 input
            tokens.
            """
            (question,) = kwargs["questions"]
            outcome = await send_request(kwargs["model"], question["instructions"])
            return decision_types.OpenAIDecisionResponse(
                model="jev-1.13.0",
                answers=(
                    ()
                    if outcome == invalid_answer
                    else (
                        decision_types.OpenAIPredicateAnswer(
                            name=question["name"], probability=0.9
                        ),
                    )
                ),
                usage=decision_types.OpenAIDecisionUsage(
                    input_tokens=10,
                    output_tokens=0,
                    total_tokens=10,
                    input_tokens_details=decision_types.OpenAIDecisionInputTokensDetails(
                        cached_tokens=0, cache_write_tokens=0
                    ),
                ),
            )

        def create_llm(with_fallback: bool = False, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM, optionally with a fallback LLM.
            """
            llm = DocumentLLM(model=primary_model, api_key="test_api_key", **kwargs)
            if with_fallback:
                llm.fallback_models = [
                    DocumentLLM(
                        model=fallback_model, api_key="test_api_key", is_fallback=True
                    )
                ]
            return llm

        def create_decision_model(**kwargs: Any) -> DocumentDecisionModel:
            """
            Creates a decision model.
            """
            return DocumentDecisionModel(
                model=decision_model, api_key="test_api_key", **kwargs
            )

        def create_document() -> Document:
            """
            Creates a document of two paragraphs with four concepts that both
            an LLM and a decision model can extract. In a model group, each second
            concept is routed to the model with the role "other".
            """
            document = Document(
                raw_text="Services agreement\n\nThe parties agree as follows."
            )
            document.concepts = [
                BooleanConcept(
                    name=name,
                    description=f"Topic {i} is covered",
                    model_role="other" if i % 2 else None,
                )
                for i, name in enumerate(concept_names)
            ]
            return document

        def set_outcomes(request_outcomes: dict[tuple[str, str], list[Any]]) -> None:
            """
            Sets the outcomes of the requests to a model for a concept, and
            resets the records.
            """
            outcomes.clear()
            outcomes.update(request_outcomes)
            for record in (requests, responses, failed_requests, cancelled_requests):
                record.clear()

        async def extract(
            model: DocumentLLM | DocumentDecisionModel | DocumentModelGroup,
            document: Document,
            **kwargs: Any,
        ) -> Exception | None:
            """
            Extracts the concepts of the document concurrently, each in a call
            of its own, and returns the error raised, if any. Checks that
            the extraction leaves no task running, i.e. that nothing is sent
            after it.
            """
            tasks_before = asyncio.all_tasks()
            error = None
            try:
                await model.extract_concepts_from_document_async(
                    document, use_concurrency=True, **kwargs
                )
            except Exception as e:  # noqa: BLE001 - the error is returned to the test
                error = e
            assert not [
                task for task in asyncio.all_tasks() - tasks_before if not task.done()
            ]
            return error

        def check_outcome(
            model: DocumentLLM | DocumentDecisionModel | DocumentModelGroup,
            document: Document,
            expected_requests: list[tuple[str, str]],
            cancelled: list[tuple[str, str]],
            extracted: list[str],
        ) -> None:
            """
            Checks the requests sent, the requests cancelled in flight, and
            the concepts extracted, by their names. Checks that the usage recorded
            for the model, with its fallback models, or for the models of
            the group, is that of the requests completed: each request that got
            a response with its tokens, and each request that failed as a call
            without a response. A request cancelled in flight is not recorded.
            """
            assert sorted(requests) == sorted(expected_requests)
            assert sorted(cancelled_requests) == sorted(cancelled)
            for concept in document.concepts:
                is_extracted = concept.name in extracted
                assert bool(concept.extracted_items) == is_extracted
                assert concept._is_processed == is_extracted
            for usage_info in model.get_usage():
                usage = usage_info.usage
                n_responses = [m for m, _ in responses].count(usage_info.model)
                n_failed = [m for m, _ in failed_requests].count(usage_info.model)
                assert usage.input == 10 * n_responses
                assert len(usage.calls) == n_responses + n_failed
                assert [
                    call.timestamp_received is not None for call in usage.calls
                ].count(True) == n_responses

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(litellm, "adecisions", new=adecisions_stub),
        ):
            # === A failed API request and a cut-off response of an LLM ===
            # Each call takes two requests, one for each paragraph. The request for
            # the first concept fails, or is answered with a cut-off response,
            # when the call of the second concept is completed. By then, the LLM
            # returned invalid data for the third concept, and the first request
            # for the fourth concept is in flight.
            for failure, error_message in [
                (api_error, r"^Exception occurred while calling LLM API"),
                (cut_off_answer, r"was cut off at the max output tokens"),
            ]:
                llm = create_llm()
                document = create_document()
                failed, completed, invalid, unanswered = concept_names
                set_outcomes(
                    {
                        (primary_model, failed): [
                            (
                                lambda c=document.concepts[1]: c._is_processed,
                                failure,
                            )
                        ],
                        (primary_model, invalid): [invalid_answer],
                        (primary_model, unanswered): [in_flight],
                    }
                )
                error = asyncio.run(
                    extract(llm, document, max_paragraphs_to_analyze_per_call=1)
                )
                # The error is the one that such a failure raises
                assert isinstance(error, ModelAPIError)
                assert re.search(error_message, str(error))
                if failure is api_error:
                    assert error.original_error is api_error
                    assert error.retry_count == 0
                # The request in flight is cancelled, and no request is sent
                # afterwards: neither the second request for the fourth concept,
                # nor the request for the third concept again
                check_outcome(
                    llm,
                    document,
                    expected_requests=[
                        (primary_model, failed),
                        (primary_model, completed),
                        (primary_model, completed),
                        (primary_model, invalid),
                        (primary_model, unanswered),
                    ],
                    cancelled=[(primary_model, unanswered)],
                    extracted=[completed],
                )

            # === A failed API request of a decision model ===
            # The request for the first concept fails when the call of the second
            # concept is completed, with the other requests in flight
            model = create_decision_model()
            document = create_document()
            failed, completed, *unanswered_concepts = concept_names
            set_outcomes(
                {
                    (decision_model, failed): [
                        (lambda c=document.concepts[1]: c._is_processed, api_error)
                    ],
                    **{(decision_model, i): [in_flight] for i in unanswered_concepts},
                }
            )
            error = asyncio.run(extract(model, document))
            assert isinstance(error, ModelAPIError)
            assert re.search(
                r"^Exception occurred while calling decision model API", str(error)
            )
            assert error.original_error is api_error
            assert error.retry_count == 0
            check_outcome(
                model,
                document,
                expected_requests=[(decision_model, i) for i in concept_names],
                cancelled=[(decision_model, i) for i in unanswered_concepts],
                extracted=[completed],
            )

            # === A call made again: the fallback LLM fails as well ===
            # The requests of the LLM for the first two concepts fail. The request
            # of the fallback LLM for the first concept fails when its request
            # for the second concept is in flight.
            llm = create_llm(with_fallback=True)
            document = create_document()
            failed, unanswered, *completed_concepts = concept_names
            set_outcomes(
                {
                    (primary_model, failed): [api_error],
                    (primary_model, unanswered): [api_error],
                    (fallback_model, failed): [
                        (lambda: (fallback_model, unanswered) in requests, api_error)
                    ],
                    (fallback_model, unanswered): [in_flight],
                }
            )
            error = asyncio.run(extract(llm, document))
            assert isinstance(error, ModelAPIError)
            assert error.original_error is api_error
            check_outcome(
                llm,
                document,
                expected_requests=[
                    *((primary_model, i) for i in concept_names),
                    (fallback_model, failed),
                    (fallback_model, unanswered),
                ],
                cancelled=[(fallback_model, unanswered)],
                extracted=completed_concepts,
            )

            # === A call made again: the data is still invalid after the last retry ===
            # The LLM returns invalid data for the first two concepts. Requested
            # again, it returns invalid data for the first concept when
            # the request sent again for the second concept is in flight.
            llm = create_llm(max_retries_invalid_data=1)
            document = create_document()
            failed, unanswered, *completed_concepts = concept_names
            set_outcomes(
                {
                    (primary_model, failed): [
                        invalid_answer,
                        (
                            lambda: requests.count((primary_model, unanswered)) == 2,
                            invalid_answer,
                        ),
                    ],
                    (primary_model, unanswered): [invalid_answer, in_flight],
                }
            )
            error = asyncio.run(extract(llm, document))
            assert isinstance(error, ModelExtractionError)
            assert "invalid data" in str(error)
            assert error.retry_count == 1
            check_outcome(
                llm,
                document,
                expected_requests=[
                    *((primary_model, i) for i in concept_names),
                    (primary_model, failed),
                    (primary_model, unanswered),
                ],
                cancelled=[(primary_model, unanswered)],
                extracted=completed_concepts,
            )

            # === Invalid data, and the call is not made again ===
            # The model has no retries for invalid data and no fallback model. It
            # returns invalid data for the first concept when the call of
            # the second concept is completed, with the other requests in flight.
            for create_model, model_name, error_message in [
                (
                    lambda: create_llm(max_retries_invalid_data=0),
                    primary_model,
                    "invalid data returned by LLM",
                ),
                (
                    create_decision_model,
                    decision_model,
                    "an invalid response returned by decision model",
                ),
            ]:
                failed, completed, *unanswered_concepts = concept_names
                # Without concurrency, the error is raised for the first call
                model = create_model()
                set_outcomes({(model_name, failed): [invalid_answer]})
                with pytest.raises(ModelExtractionError) as exc_info:
                    model.extract_concepts_from_document(
                        create_document(), max_items_per_call=1
                    )
                assert requests == [(model_name, failed)]

                model = create_model()
                document = create_document()
                set_outcomes(
                    {
                        (model_name, failed): [
                            (
                                lambda c=document.concepts[1]: c._is_processed,
                                invalid_answer,
                            )
                        ],
                        **{(model_name, i): [in_flight] for i in unanswered_concepts},
                    }
                )
                error = asyncio.run(extract(model, document))
                # The error is the one raised without concurrency
                assert isinstance(error, ModelExtractionError)
                assert error_message in str(error)
                assert str(error) == str(exc_info.value)
                assert error.retry_count == 0
                check_outcome(
                    model,
                    document,
                    expected_requests=[(model_name, i) for i in concept_names],
                    cancelled=[(model_name, i) for i in unanswered_concepts],
                    extracted=[completed],
                )

            # === A model group ===
            # The request of the LLM for its first concept fails when its request
            # for its other concept, and the requests of the decision model, are
            # in flight: the other call of the LLM is stopped, as the calls of
            # the decision model are
            group = DocumentModelGroup(
                models=[create_llm(), create_decision_model(role="other")]
            )
            document = create_document()
            failed_request, *unanswered_requests = [
                (decision_model if concept.model_role else primary_model, concept.name)
                for concept in document.concepts
            ]
            set_outcomes(
                {
                    failed_request: [(lambda: len(requests) == 4, api_error)],
                    **{request: [in_flight] for request in unanswered_requests},
                }
            )
            error = asyncio.run(extract(group, document))
            assert isinstance(error, ModelAPIError)
            assert error.original_error is api_error
            check_outcome(
                group,
                document,
                expected_requests=[failed_request, *unanswered_requests],
                cancelled=unanswered_requests,
                extracted=[],
            )

            # === A failure that does not fail the extraction ===
            # The request of the LLM for the first concept fails when the other
            # requests are in flight, and they are answered afterwards
            for with_fallback in (True, False):
                llm = create_llm(with_fallback=with_fallback)
                document = create_document()
                failed, *other_concepts = concept_names
                set_outcomes(
                    {
                        (primary_model, failed): [
                            (lambda: len(requests) == 4, api_error)
                        ],
                        **{
                            (primary_model, i): [(lambda: bool(failed_requests), None)]
                            for i in other_concepts
                        },
                    }
                )
                if with_fallback:
                    # The fallback LLM is tried for the first concept
                    error = asyncio.run(extract(llm, document))
                else:
                    # A warning is logged for the first concept
                    with patch.object(
                        logger, "warning", wraps=logger.warning
                    ) as log_warning:
                        error = asyncio.run(
                            extract(
                                llm, document, raise_exception_on_extraction_error=False
                            )
                        )
                    assert any(
                        "Exception occurred while calling LLM API" in str(call.args[0])
                        for call in log_warning.call_args_list
                    )
                assert error is None
                check_outcome(
                    llm,
                    document,
                    expected_requests=[
                        *((primary_model, i) for i in concept_names),
                        *([(fallback_model, failed)] if with_fallback else []),
                    ],
                    cancelled=[],
                    extracted=concept_names if with_fallback else other_concepts,
                )

            # The same for invalid data that an LLM without retries for it returns
            # for the first concept: the other requests are answered after it
            for with_fallback in (True, False):
                llm = create_llm(
                    with_fallback=with_fallback, max_retries_invalid_data=0
                )
                document = create_document()
                failed, *other_concepts = concept_names
                set_outcomes(
                    {
                        (primary_model, failed): [
                            (lambda: len(requests) == 4, invalid_answer)
                        ],
                        **{
                            (primary_model, i): [(lambda: bool(responses), None)]
                            for i in other_concepts
                        },
                    }
                )
                if with_fallback:
                    # The fallback LLM is tried for the first concept
                    error = asyncio.run(extract(llm, document))
                else:
                    # A warning is logged for the first concept
                    with patch.object(
                        logger, "warning", wraps=logger.warning
                    ) as log_warning:
                        error = asyncio.run(
                            extract(
                                llm, document, raise_exception_on_extraction_error=False
                            )
                        )
                    assert any(
                        re.search(
                            r"invalid data.*\(0 retries\)\. If you want to raise",
                            str(call.args[0]),
                        )
                        for call in log_warning.call_args_list
                    )
                assert error is None
                check_outcome(
                    llm,
                    document,
                    expected_requests=[
                        *((primary_model, i) for i in concept_names),
                        *([(fallback_model, failed)] if with_fallback else []),
                    ],
                    cancelled=[],
                    extracted=concept_names if with_fallback else other_concepts,
                )

    def test_log_of_failed_call_tells_what_happened(self):
        """
        Tests that the log of an extraction tells what happened to a call that
        failed, for aspects and for concepts, with and without concurrency, for
        an LLM and a decision model, each with a fallback model of either kind:

        - a call is logged as one whose LLM did not return valid JSON only when
          the LLM returned a response that is not such a JSON;
        - a call is logged as retried for invalid data only when it is made again
          with the model that returned invalid data for it, numbered among
          the retries of that model;
        - the fallback model to be tried is not announced as an LLM, which
          a fallback decision model is not.

        A prompt that exceeds the context window of the LLM, and a failed API
        request, were also logged as an LLM that did not return valid JSON, when
        a fallback model was yet to be tried or a warning was issued instead of
        an exception, although the LLM returned nothing. A response cut off at
        the max output tokens was also logged as a JSON that failed
        the validation. The first call of a fallback model was logged as a retry
        for the invalid output of that model, whatever the failure it was tried
        for. A request that still failed for its params after they were dropped
        was not logged as failed.

        The APIs are stubbed, i.e. no cassette is required.
        """
        import logging as stdlib_logging

        # The types of the responses of litellm to a decision model
        from litellm.types import decisions as decision_types

        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"
        decision_model = "typesafe/jev-latest"
        fallback_decision_model = "typesafe/jev-1.13.0"
        n_instances = 2
        invalid_answer = "<<<not-json>>>"
        cut_off_answer = '[{"concept_id": "C0", "extracted_items": [tr'
        api_error = litellm.AuthenticationError(
            message="Simulated API error", model=primary_model, llm_provider="openai"
        )
        unsupported_params_error = litellm.UnsupportedParamsError(
            message="Simulated error", model=primary_model, llm_provider="openai"
        )
        get_model_info = litellm.get_model_info
        # Whether the LLM is given a context window too small for any prompt
        small_context_window: list[bool] = [False]
        # The outcomes of the requests that a model is sent for an aspect or
        # a concept, in turn, by the model: an error to raise, or a text for
        # the model to return. A request with no outcome set is answered with
        # an item for the aspect or the concept.
        outcomes: dict[str, list[Any]] = {}
        # The number of the requests sent, by the model and the name of the aspect
        # or the concept
        n_requests: dict[tuple[str, str], int] = {}

        @contextmanager
        def capture_logs():
            """
            Captures the lines logged for the calls of an extraction, i.e. at
            the INFO, WARNING and ERROR levels, each as its level and its message.
            """
            captured_logs: list[str] = []

            class CaptureHandler(stdlib_logging.Handler):
                """
                Handler to capture log messages for testing.
                """

                def emit(self, record):
                    """
                    Adds the level and the message of the record to the captured ones.
                    """
                    if record.levelname in ("INFO", "WARNING", "ERROR"):
                        captured_logs.append(
                            f"{record.levelname}: {record.getMessage()}"
                        )

            capture_handler = CaptureHandler()
            capture_handler.setLevel(stdlib_logging.INFO)
            logger.addHandler(capture_handler)
            try:
                yield captured_logs
            finally:
                logger.removeHandler(capture_handler)

        def get_model_info_stub(model: str, *args: Any, **kwargs: Any) -> Any:
            """
            Returns the info of the model, with a context window of 10 tokens
            for the LLM when it is given a small one.
            """
            model_info = dict(get_model_info(model, *args, **kwargs))
            if model == primary_model and small_context_window[0]:
                model_info["max_input_tokens"] = 10
            return model_info

        def get_outcome(model: str, text: str) -> Any:
            """
            Gives the outcome of the next request of the model for the aspect or
            the concept that the text names: raises the error, or returns
            the text to answer with, or None for the answer with an item.
            """
            (name,) = set(re.findall(r"Topic \d", text))
            position = n_requests.get((model, name), 0)
            n_requests[(model, name)] = position + 1
            model_outcomes = outcomes.get(model, [])
            outcome = (
                model_outcomes[position] if position < len(model_outcomes) else None
            )
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API. A response is cut off if its text is
            the cut-off answer.
            """
            prompt = kwargs["messages"][-1]["content"]
            content = get_outcome(kwargs["model"], prompt)
            if content is None:
                content = json.dumps(
                    [{"concept_id": "C0", "extracted_items": [True]}]
                    if re.search(r"<concept_id>C0</concept_id>", prompt)
                    else [{"aspect_id": "A0", "paragraph_ids": ["P0"]}]
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        finish_reason="length" if content == cut_off_answer else "stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        async def adecisions_stub(**kwargs: Any):
            """
            Simulates the API of the decision model, which answers "yes" to
            the question of the request, or gives no answer to it if the outcome
            of the request is the invalid answer.
            """
            (question,) = kwargs["questions"]
            outcome = get_outcome(kwargs["model"], question["instructions"])
            return decision_types.OpenAIDecisionResponse(
                model="jev-1.13.0",
                answers=(
                    ()
                    if outcome == invalid_answer
                    else (
                        decision_types.OpenAIPredicateAnswer(
                            name=question["name"], probability=0.9
                        ),
                    )
                ),
                usage=decision_types.OpenAIDecisionUsage(
                    input_tokens=10,
                    output_tokens=0,
                    total_tokens=10,
                    input_tokens_details=decision_types.OpenAIDecisionInputTokensDetails(
                        cached_tokens=0, cache_write_tokens=0
                    ),
                ),
            )

        def create_model(
            model: str, **kwargs: Any
        ) -> DocumentLLM | DocumentDecisionModel:
            """
            Creates an LLM or a decision model, by the provider of the model.
            """
            model_class = (
                DocumentDecisionModel if model.startswith("typesafe/") else DocumentLLM
            )
            return model_class(model=model, api_key="test_api_key", **kwargs)

        def create_document(instance_type: str) -> Document:
            """
            Creates a document of one paragraph with two aspects, or with two
            concepts, that both an LLM and a decision model can extract.
            """
            document = Document(raw_text="The parties agree as follows.")
            if instance_type == "aspect":
                document.aspects = [
                    Aspect(name=f"Topic {i}", description=f"Clauses on Topic {i}")
                    for i in range(n_instances)
                ]
            else:
                document.concepts = [
                    BooleanConcept(
                        name=f"Topic {i}", description=f"Topic {i} is covered"
                    )
                    for i in range(n_instances)
                ]
            return document

        def check_log(
            models: list[str],
            expected_log: list[str],
            request_outcomes: dict[str, list[Any]] | None = None,
            with_small_context_window: bool = False,
            raise_exception: bool = True,
            extracted: bool = True,
            **model_kwargs: Any,
        ) -> None:
            """
            Extracts the aspects, and the concepts, of a document with the first
            of the models, which has the other ones as its fallback models, each
            aspect or concept in a call of its own, without and with concurrency.
            Checks that each call is logged with the expected lines, and that
            nothing else is logged. A line is expected by its level and
            the beginning of its message, with the type of what is extracted
            in place of ``{type}``. The events of the calls are logged only,
            i.e. no warning of ContextGem is issued for them.
            """
            for instance_type in ("aspect", "concept"):
                expected = [line.format(type=instance_type) for line in expected_log]
                for use_concurrency in (False, True):
                    outcomes.clear()
                    outcomes.update(request_outcomes or {})
                    n_requests.clear()
                    small_context_window[0] = with_small_context_window
                    model = create_model(models[0], **model_kwargs)
                    model.fallback_models = [
                        create_model(fallback, is_fallback=True)
                        for fallback in models[1:]
                    ]
                    document = create_document(instance_type)
                    extract = (
                        model.extract_aspects_from_document
                        if instance_type == "aspect"
                        else model.extract_concepts_from_document
                    )
                    extraction_kwargs: dict[str, Any] = {
                        "max_items_per_call": 1,
                        "use_concurrency": use_concurrency,
                        "raise_exception_on_extraction_error": raise_exception,
                    }
                    with (
                        capture_logs() as captured_logs,
                        warnings.catch_warnings(record=True) as caught_warnings,
                    ):
                        warnings.simplefilter("always")
                        extract(document, **extraction_kwargs)
                    assert not [
                        caught_warning
                        for caught_warning in caught_warnings
                        if issubclass(caught_warning.category, ContextGemWarning)
                    ]
                    # Each line logged, as the expected line it begins with, if any
                    logged = [
                        next(
                            (
                                line
                                for line in sorted(expected, key=len, reverse=True)
                                if captured_line.startswith(line)
                            ),
                            captured_line,
                        )
                        for captured_line in captured_logs
                    ]
                    # The lines of each call, and no other lines. The calls are
                    # logged one after another without concurrency.
                    if use_concurrency:
                        assert sorted(logged) == sorted(expected * n_instances), (
                            captured_logs
                        )
                    else:
                        assert logged == expected * n_instances, captured_logs
                    instances = (
                        document.aspects
                        if instance_type == "aspect"
                        else document.concepts
                    )
                    assert all(bool(i.extracted_items) == extracted for i in instances)

        # The lines logged for the failures, and for what follows them
        prompt_not_sent = "ERROR: The request is not sent to the LLM: Input messages "
        llm_api_error = "Exception occurred while calling LLM API: litellm.Auth"
        cut_off_response = (
            f"ERROR: The response of LLM `{primary_model}` was cut off at the max "
            f"output tokens"
        )
        # ... and the warning that the call failed, when no exception is raised
        cut_off_warning = (
            f"WARNING: The response of LLM `{primary_model}` was cut off at the "
            f"max output tokens"
        )
        # An LLM that returned a response that is not a valid JSON
        not_valid_json = [
            "ERROR: Error when validating parsed JSON: parsed_json is None",
            "ERROR: LLM did not return valid JSON for {type}s",
        ]
        fallback_pending = "INFO: The call will be retried with a fallback model."
        trying_fallback_llm = f"INFO: Trying with fallback LLM {fallback_model} (1/1)"
        trying_fallback_decision_model = (
            f"INFO: Trying with fallback decision model {fallback_decision_model} (1/1)"
        )

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(litellm, "adecisions", new=adecisions_stub),
            patch.object(litellm, "get_model_info", new=get_model_info_stub),
        ):
            # === The prompt exceeds the context window of the LLM ===
            # No request is sent, i.e. the LLM returns nothing, and the fallback
            # model is called for the first time
            for fallback, trying_fallback in [
                (fallback_model, trying_fallback_llm),
                (fallback_decision_model, trying_fallback_decision_model),
            ]:
                check_log(
                    [primary_model, fallback],
                    [prompt_not_sent, fallback_pending, trying_fallback],
                    with_small_context_window=True,
                )

            # === The API request of the LLM fails ===
            for fallback, trying_fallback in [
                (fallback_model, trying_fallback_llm),
                (fallback_decision_model, trying_fallback_decision_model),
            ]:
                check_log(
                    [primary_model, fallback],
                    [f"ERROR: {llm_api_error}", fallback_pending, trying_fallback],
                    request_outcomes={primary_model: [api_error]},
                )
            # The request of the fallback LLM fails too, and a warning is logged
            check_log(
                [primary_model, fallback_model],
                [
                    f"ERROR: {llm_api_error}",
                    fallback_pending,
                    trying_fallback_llm,
                    f"ERROR: {llm_api_error}",
                    f"WARNING: {llm_api_error}",
                ],
                request_outcomes={
                    primary_model: [api_error],
                    fallback_model: [api_error],
                },
                raise_exception=False,
                extracted=False,
            )
            # No fallback model, and a warning is logged
            check_log(
                [primary_model],
                [f"ERROR: {llm_api_error}", f"WARNING: {llm_api_error}"],
                request_outcomes={primary_model: [api_error]},
                raise_exception=False,
                extracted=False,
            )
            # The params of the request are not supported for the model, and
            # the request fails for them also when it is sent without them
            check_log(
                [primary_model, fallback_model],
                [
                    (
                        f"WARNING: The request to LLM `{primary_model}` has params that "
                        f"are not supported for the model, and is sent without them: "
                    ),
                    (
                        "ERROR: Exception occurred while calling LLM API with "
                        "drop_params=True: "
                    ),
                    fallback_pending,
                    trying_fallback_llm,
                ],
                request_outcomes={primary_model: [unsupported_params_error] * 2},
            )

            # === The response of the LLM is cut off at the max output tokens ===
            check_log(
                [primary_model, fallback_model],
                [cut_off_response, fallback_pending, trying_fallback_llm],
                request_outcomes={primary_model: [cut_off_answer]},
            )
            # No fallback model, and a warning is logged
            check_log(
                [primary_model],
                [cut_off_response, cut_off_warning],
                request_outcomes={primary_model: [cut_off_answer]},
                raise_exception=False,
                extracted=False,
            )

            # === The LLM returns invalid data ===
            # The call is made again twice with the LLM, and then with
            # the fallback LLM, which returns invalid data for its first call,
            # i.e. is called again: its first retry of the three it has by
            # default, after its first call, as a primary LLM has
            check_log(
                [primary_model, fallback_model],
                [
                    *not_valid_json,
                    "INFO: Retrying {type}s with invalid data (1/2)",
                    *not_valid_json,
                    "INFO: Retrying {type}s with invalid data (2/2)",
                    *not_valid_json,
                    trying_fallback_llm,
                    *not_valid_json,
                    "INFO: Retrying {type}s with invalid data (1/3)",
                ],
                request_outcomes={
                    primary_model: [invalid_answer] * 3,
                    fallback_model: [invalid_answer],
                },
                max_retries_invalid_data=2,
            )
            # The call is not made again with the LLM, and the fallback decision
            # model, which did not return an invalid response, is called once
            check_log(
                [primary_model, fallback_decision_model],
                [*not_valid_json, trying_fallback_decision_model],
                request_outcomes={primary_model: [invalid_answer]},
                max_retries_invalid_data=0,
            )

            # === A decision model ===
            # Its API request fails, and the fallback LLM is called for the first
            # time, i.e. not for a JSON that is not valid
            check_log(
                [decision_model, fallback_model],
                [
                    "ERROR: Exception occurred while calling decision model API: ",
                    fallback_pending,
                    trying_fallback_llm,
                ],
                request_outcomes={decision_model: [api_error]},
            )
            # It returns an invalid response, and the fallback decision model is
            # called once
            check_log(
                [decision_model, fallback_decision_model],
                [
                    "ERROR: Decision model `"
                    + decision_model
                    + "` returned an invalid response for {type} `Topic ",
                    trying_fallback_decision_model,
                ],
                request_outcomes={decision_model: [invalid_answer]},
            )

    def test_fallback_model_has_its_own_retries_for_invalid_data(self):
        """
        Tests that a fallback model that returns invalid data during extraction is
        called as a primary model with the same setting is: once, and then again
        as many times as it has retries for invalid data, i.e. a fallback LLM up
        to ``1 + max_retries_invalid_data`` times, and a fallback decision model,
        which has no such retries, once. Also tests that the number of retries
        reported when the data stays invalid, in ``ModelExtractionError`` and in
        the warning issued instead of it, is the number of times that the model
        the report is for, i.e. the last one tried, was called again, and that
        each of these calls is logged as a retry of that model.

        A fallback LLM was called one time fewer than a primary LLM with the same
        ``max_retries_invalid_data``: three times with the default of 3, and not
        again at all with 1. The first call of a fallback model was reported as
        a retry, e.g. one retry for a fallback decision model, which is called
        once.

        Checked for aspects and for concepts, with and without concurrency, with
        an exception raised and with a warning issued instead of it. The APIs are
        stubbed, i.e. no cassette is required.
        """
        import logging as stdlib_logging

        # The types of the responses of litellm to a decision model
        from litellm.types import decisions as decision_types

        litellm = llms_module.litellm
        llm_model = "openai/gpt-6-luna"
        decision_model = "typesafe/jev-latest"
        n_instances = 2
        invalid_answer = "<<<not-json>>>"
        cut_off_answer = '[{"concept_id": "C0", "extracted_items": [tr'
        api_error = litellm.AuthenticationError(
            message="Simulated API error", model=llm_model, llm_provider="openai"
        )
        # The outcomes of the requests that a model is sent for an aspect or
        # a concept, in turn, by the position of the model in the chain (0 for
        # the primary model): an error to raise, a text for the model to return,
        # or None for an answer with an item. The last outcome is that of all
        # the further requests.
        outcomes: dict[int, list[Any]] = {}
        # The requests sent, each as the position of the model in the chain and
        # the name of the aspect or the concept
        requests: list[tuple[int, str]] = []
        # The lines logged on the fallback models tried and on the retries
        logged: list[str] = []

        class CaptureHandler(stdlib_logging.Handler):
            """
            Handler to capture log messages for testing.
            """

            def emit(self, record):
                """
                Adds the message of the record to the captured ones, if it tells
                that a fallback model is tried or that a call is made again.
                """
                message = record.getMessage()
                if message.startswith(("Trying with fallback ", "Retrying ")):
                    logged.append(message)

        def get_outcome(api_key: str, text: str) -> Any:
            """
            Records the request of a model for the aspect or the concept that
            the text names, and gives its outcome: raises the error, or returns
            the text to answer with, or None for the answer with an item.
            The API key of a model tells its position in the chain.
            """
            position = int(api_key.removeprefix("key-"))
            (name,) = set(re.findall(r"Topic \d", text))
            n_previous_requests = requests.count((position, name))
            requests.append((position, name))
            model_outcomes = outcomes[position]
            outcome = model_outcomes[min(n_previous_requests, len(model_outcomes) - 1)]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API. A response is cut off if its text is
            the cut-off answer.
            """
            prompt = kwargs["messages"][-1]["content"]
            content = get_outcome(kwargs["api_key"], prompt)
            if content is None:
                content = json.dumps(
                    [{"concept_id": "C0", "extracted_items": [True]}]
                    if re.search(r"<concept_id>C0</concept_id>", prompt)
                    else [{"aspect_id": "A0", "paragraph_ids": ["P0"]}]
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        finish_reason="length" if content == cut_off_answer else "stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        async def adecisions_stub(**kwargs: Any):
            """
            Simulates the API of the decision model, which answers "yes" to
            the question of the request, or gives no answer to it if the outcome
            of the request is the invalid answer.
            """
            (question,) = kwargs["questions"]
            outcome = get_outcome(kwargs["api_key"], question["instructions"])
            return decision_types.OpenAIDecisionResponse(
                model="jev-1.13.0",
                answers=(
                    ()
                    if outcome == invalid_answer
                    else (
                        decision_types.OpenAIPredicateAnswer(
                            name=question["name"], probability=0.9
                        ),
                    )
                ),
                usage=decision_types.OpenAIDecisionUsage(
                    input_tokens=10,
                    output_tokens=0,
                    total_tokens=10,
                    input_tokens_details=decision_types.OpenAIDecisionInputTokensDetails(
                        cached_tokens=0, cache_write_tokens=0
                    ),
                ),
            )

        def create_model(
            chain: list[int | None],
        ) -> DocumentLLM | DocumentDecisionModel:
            """
            Creates the primary model of a chain, with the other models of
            the chain as its fallback models. A model of the chain is given as
            the ``max_retries_invalid_data`` of an LLM, or as None for a decision
            model, which has no retries for invalid data.
            """
            models: list[DocumentLLM | DocumentDecisionModel] = []
            for position, retries in enumerate(chain):
                model_kwargs: dict[str, Any] = {
                    "api_key": f"key-{position}",
                    "is_fallback": position > 0,
                }
                model = (
                    DocumentDecisionModel(model=decision_model, **model_kwargs)
                    if retries is None
                    else DocumentLLM(
                        model=llm_model,
                        max_retries_invalid_data=retries,
                        **model_kwargs,
                    )
                )
                # The requests to the stubbed APIs are not to be paced
                model.async_limiter = AsyncLimiter(1000, 1)
                models.append(model)
            models[0].fallback_models = [*models[1:]]
            return models[0]

        def create_document(instance_type: str) -> Document:
            """
            Creates a document of one paragraph with two aspects, or with two
            concepts, that both an LLM and a decision model can extract.
            """
            document = Document(raw_text="The parties agree as follows.")
            if instance_type == "aspect":
                document.aspects = [
                    Aspect(name=f"Topic {i}", description=f"Clauses on Topic {i}")
                    for i in range(n_instances)
                ]
            else:
                document.concepts = [
                    BooleanConcept(
                        name=f"Topic {i}", description=f"Topic {i} is covered"
                    )
                    for i in range(n_instances)
                ]
            return document

        def extract(
            chain: list[int | None],
            instance_type: str,
            use_concurrency: bool,
            raise_exception: bool = True,
            request_outcomes: dict[int, list[Any]] | None = None,
        ) -> Document:
            """
            Extracts the aspects, or the concepts, of a document with the primary
            model of the chain, each aspect or concept in a call of its own.
            Each model returns invalid data for each request, unless other
            outcomes of the requests are given.
            """
            outcomes.clear()
            outcomes.update(
                request_outcomes
                or {position: [invalid_answer] for position in range(len(chain))}
            )
            requests.clear()
            logged.clear()
            model = create_model(chain)
            document = create_document(instance_type)
            extract_instances = (
                model.extract_aspects_from_document
                if instance_type == "aspect"
                else model.extract_concepts_from_document
            )
            extract_instances(
                document,
                max_items_per_call=1,
                use_concurrency=use_concurrency,
                raise_exception_on_extraction_error=raise_exception,
            )
            return document

        def get_calls() -> list[list[int]]:
            """
            Returns the positions of the models in the order of the requests
            sent to them, for each aspect or concept, i.e. for each call.
            """
            return [
                [position for position, name in requests if name == f"Topic {i}"]
                for i in range(n_instances)
            ]

        def get_expected_call(chain: list[int | None]) -> list[int]:
            """
            Returns the positions of the models in the order of the requests of
            a call for which each model of the chain returns invalid data: each
            model in turn is sent a request, and then as many requests again as
            it has retries for invalid data.
            """
            return [
                position
                for position, retries in enumerate(chain)
                for _ in range(1 + (retries or 0))
            ]

        def get_expected_log(chain: list[int | None], instance_type: str) -> list[str]:
            """
            Returns the lines logged for such a call: a line for each fallback
            model tried, and a line for each retry of a model, numbered among
            the retries of that model.
            """
            expected_log: list[str] = []
            for position, retries in enumerate(chain):
                if position > 0:
                    fallback = (
                        f"decision model {decision_model}"
                        if retries is None
                        else f"LLM {llm_model}"
                    )
                    expected_log.append(
                        f"Trying with fallback {fallback} ({position}/{len(chain) - 1})"
                    )
                expected_log += [
                    f"Retrying {instance_type}s with invalid data ({i + 1}/{retries})"
                    for i in range(retries or 0)
                ]
            return expected_log

        chains: list[list[int | None]] = [
            # A primary model on its own: an LLM with 0, 1 or 3 retries for
            # invalid data, or a decision model
            *[[primary] for primary in (0, 1, 3, None)],
            # A fallback LLM with 0, 1 or 3 retries
            *[
                [primary, fallback]
                for primary in (0, 1, 3, None)
                for fallback in (0, 1, 3)
            ],
            # A fallback decision model
            *[[primary, None] for primary in (0, 1, 3, None)],
            # Two fallback models
            [0, 0, 0],
            [1, 1, 1],
            [3, 3, 3],
            [3, 1, 0],
            [1, None, 3],
            [1, 3, None],
            [None, 1, 3],
            [None, None, 1],
        ]
        combinations = [
            (instance_type, use_concurrency)
            for instance_type in ("aspect", "concept")
            for use_concurrency in (False, True)
        ]

        capture_handler = CaptureHandler()
        capture_handler.setLevel(stdlib_logging.INFO)
        logger.addHandler(capture_handler)
        try:
            with (
                patch.object(litellm, "acompletion", new=acompletion_stub),
                patch.object(litellm, "adecisions", new=adecisions_stub),
            ):
                # === Each model of the chain returns invalid data ===
                for chain in chains:
                    expected_call = get_expected_call(chain)
                    # The model that failed last, which the report is for:
                    # the kind of the model, what it returned, and the number
                    # of times that it was called again
                    last_retries = chain[-1] or 0
                    reported = (
                        "due to an invalid response returned by decision model"
                        if chain[-1] is None
                        else "due to invalid data returned by LLM"
                    )
                    for instance_type, use_concurrency in combinations:
                        expected_log = get_expected_log(chain, instance_type)

                        # An exception is raised for the first call that fails
                        with pytest.raises(
                            ModelExtractionError,
                            match=rf"{reported}\. .*\(after {last_retries} retries\)$",
                        ) as exc_info:
                            extract(chain, instance_type, use_concurrency)
                        assert exc_info.value.retry_count == last_retries
                        if use_concurrency:
                            # The call that failed is made with each model of
                            # the chain, and is logged in full. The other call
                            # is stopped at some point.
                            assert expected_call in get_calls()
                            assert all(
                                call == expected_call[: len(call)]
                                for call in get_calls()
                            )
                            assert set(logged) == set(expected_log)
                        else:
                            # The other call is not made
                            assert get_calls() == [expected_call, []]
                            assert logged == expected_log

                        # A warning is logged for each call instead
                        with patch.object(
                            logger, "warning", wraps=logger.warning
                        ) as log_warning:
                            document = extract(
                                chain,
                                instance_type,
                                use_concurrency,
                                raise_exception=False,
                            )
                        warning_messages = [
                            str(call.args[0])
                            for call in log_warning.call_args_list
                            if "could not be processed" in str(call.args[0])
                        ]
                        assert len(warning_messages) == n_instances
                        assert all(
                            re.search(
                                rf"{reported}\. .*\] \({last_retries} retries\)\. ",
                                message,
                            )
                            for message in warning_messages
                        ), warning_messages
                        assert get_calls() == [expected_call] * n_instances
                        if use_concurrency:
                            assert sorted(logged) == sorted(expected_log * n_instances)
                        else:
                            assert logged == expected_log * n_instances
                        instances = (
                            document.aspects
                            if instance_type == "aspect"
                            else document.concepts
                        )
                        assert not any(i.extracted_items for i in instances)

                # === A fallback LLM returns valid data in the last call it has ===
                # I.e. in its last retry: nothing fails, and nothing is reported
                valid_last_call_chains: list[list[int | None]] = [
                    [0, 1],
                    [3, 3],
                    [None, 1],
                    [1, 1, 3],
                ]
                for chain in valid_last_call_chains:
                    last_position = len(chain) - 1
                    request_outcomes: dict[int, list[Any]] = {
                        position: [invalid_answer] for position in range(last_position)
                    }
                    request_outcomes[last_position] = [
                        *([invalid_answer] * (chain[-1] or 0)),
                        None,
                    ]
                    for instance_type, use_concurrency in combinations:
                        document = extract(
                            chain,
                            instance_type,
                            use_concurrency,
                            request_outcomes=request_outcomes,
                        )
                        assert get_calls() == [get_expected_call(chain)] * n_instances
                        instances = (
                            document.aspects
                            if instance_type == "aspect"
                            else document.concepts
                        )
                        assert all(i.extracted_items for i in instances)

                # === The failures other than invalid data ===
                # A fallback LLM whose API request fails, or whose response is
                # cut off at the max output tokens, is not called again, whether
                # this happens in its first call or in a retry
                fallback_outcomes_and_n_calls: list[tuple[list[Any], int]] = [
                    ([api_error], 1),
                    ([cut_off_answer], 1),
                    ([invalid_answer, api_error], 2),
                    ([invalid_answer, cut_off_answer], 2),
                ]
                for (
                    fallback_outcomes,
                    n_fallback_calls,
                ) in fallback_outcomes_and_n_calls:
                    for instance_type, use_concurrency in combinations:
                        with pytest.raises(ModelAPIError):
                            extract(
                                [1, 3],
                                instance_type,
                                use_concurrency,
                                request_outcomes={
                                    0: [invalid_answer],
                                    1: fallback_outcomes,
                                },
                            )
                        assert [0, 0] + [1] * n_fallback_calls in get_calls()
                        assert all(
                            call.count(1) <= n_fallback_calls for call in get_calls()
                        )
                # The primary LLM that fails in such a way is not called again
                # either, and the fallback LLM has its retries for invalid data
                # whatever the failure that it is tried for
                for primary_outcome in (api_error, cut_off_answer):
                    for instance_type, use_concurrency in combinations:
                        with pytest.raises(
                            ModelExtractionError, match=r"\(after 1 retries\)$"
                        ) as exc_info:
                            extract(
                                [3, 1],
                                instance_type,
                                use_concurrency,
                                request_outcomes={
                                    0: [primary_outcome],
                                    1: [invalid_answer],
                                },
                            )
                        assert exc_info.value.retry_count == 1
                        assert [0, 1, 1] in get_calls()
                        assert all(call.count(0) <= 1 for call in get_calls())
        finally:
            logger.removeHandler(capture_handler)

    def test_concepts_with_sentence_references_after_aspects_are_extracted(self):
        """
        Tests that a concept with references to sentences is extracted from
        an aspect, and from a sub-aspect, also when the sentences of the document
        are segmented only after the aspects were extracted: when the concept is
        attached after that, when the document is restored from its serialized
        form in between, and when another call segmented the sentences in
        between.

        The paragraphs of a sub-aspect, and of any aspect of a restored document,
        were copies of the paragraphs of the document, which the sentences
        segmented later did not reach: the extraction from a sub-aspect failed
        with an error that the sub-aspect is not assigned to the document, and
        the prompt for an aspect of a restored document had no text. They are
        the paragraphs of the document itself.

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

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, e.g. the batch
                    size, which the stub does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                segmented_texts.append(sorted(texts))
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
                "_get_sat_model",
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
                    # The paragraphs of the aspect are those of the document
                    assert [
                        id(paragraph)
                        for paragraph in document.aspects[0].reference_paragraphs
                    ] == [id(paragraph) for paragraph in document.paragraphs[:2]]
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
                # the document, which are those of the aspect and of the sub-aspect
                assert segmented_texts == [sorted(paragraph_texts)], scenario
                # The aspect and the sub-aspect have the first two paragraphs:
                # the prompt for the concept of each lists their sentences
                assert len(prompts) == 2, scenario
                for prompt in prompts:
                    for paragraph_position, texts in enumerate(sentence_texts[:2]):
                        for sentence_position, text in enumerate(texts):
                            sentence_id = f"P{paragraph_position}-S{sentence_position}"
                            assert f'<sentence id="{sentence_id}">' in prompt
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
                    # The paragraphs of the aspect are the paragraphs of
                    # the document, i.e. they have their sentences
                    assert len(instance.reference_paragraphs) == 2
                    for paragraph in instance.reference_paragraphs:
                        document_paragraph = document.paragraphs[
                            document.get_paragraph_index(paragraph)
                        ]
                        assert paragraph is document_paragraph
                        assert paragraph.sentences
                    # ... as are the references of the item
                    assert reference_paragraph is document.paragraphs[0]
                    assert reference_sentence is document.paragraphs[0].sentences[1]
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

    def test_malformed_sentence_id_in_llm_output_is_invalid_data(self):
        """
        Tests that an ID of a sentence that is not a number, in the output of
        an LLM for an aspect with sentences as its items, is handled as any other
        invalid data returned by the LLM: the call is made again, up to
        ``max_retries_invalid_data`` times, and ``ModelExtractionError`` is
        raised, or a warning is issued, if the data stays invalid. Such output
        raised ``ValueError`` from the code that reads it, without the call being
        made again.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price."],
        ]
        responses: list[str] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: returns the next response of the queue.
            """
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=responses.pop(0))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def get_response(sentence_id: str) -> str:
            """
            Returns the response of the LLM with a sentence of the first
            paragraph extracted for the aspect.
            """
            return json.dumps(
                [
                    {
                        "aspect_id": "A0",
                        "paragraphs": [
                            {
                                "paragraph_id": "P0",
                                "sentences": [{"sentence_id": sentence_id}],
                            }
                        ],
                    }
                ]
            )

        def create_document() -> Document:
            """
            Creates a document with an aspect with sentences as its items.
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
                    name="Delivery",
                    description="Clauses on the delivery",
                    reference_depth="sentences",
                )
            ]
            return document

        malformed_response = get_response("P0-Sx")
        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # === The call is made again, and the valid data is extracted ===
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            assert llm.max_retries_invalid_data == 3
            document = create_document()
            responses.extend([malformed_response, get_response("P0-S1")])
            (aspect,) = llm.extract_aspects_from_document(document)
            assert not responses
            (item,) = aspect.extracted_items
            assert item.reference_sentences == [document.paragraphs[0].sentences[1]]
            assert len(llm.get_usage()[0].usage.calls) == 2

            # === Without retries, the extraction fails for invalid data ===
            llm = DocumentLLM(
                model="openai/gpt-6-luna",
                api_key="test_api_key",
                max_retries_invalid_data=0,
            )
            document = create_document()
            responses.append(malformed_response)
            with pytest.raises(ModelExtractionError, match="invalid data"):
                llm.extract_aspects_from_document(document)
            assert not responses

            # ... or a warning is logged, and nothing is extracted
            document = create_document()
            responses.append(malformed_response)
            with patch.object(logger, "warning", wraps=logger.warning) as log_warning:
                (aspect,) = llm.extract_aspects_from_document(
                    document, raise_exception_on_extraction_error=False
                )
            assert any(
                "invalid data" in str(call.args[0])
                for call in log_warning.call_args_list
            )
            assert not responses
            assert not aspect.extracted_items

    def test_ids_repeated_in_llm_output_are_used_once(self):
        """
        Tests that an ID of a paragraph or of a sentence that an LLM lists more
        than once, among those of an aspect or among the references of an item
        of a concept, is used once, as the LLM is instructed to list each ID
        once: an aspect gets one item for the paragraph or the sentence, and
        an item of a concept references it once. The output is taken as it is,
        i.e. the call is not made again. An aspect got an item for each time that
        the ID was listed, and for a concept ``ValueError`` was raised.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price.", "Payment is due in 14 days."],
            ["This agreement is governed by the laws of Norway."],
        ]
        responses: list[str] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: returns the next response of the queue.
            """
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
            Creates a document of three paragraphs with their sentences given.
            """
            return Document(
                paragraphs=[
                    Paragraph(
                        raw_text=" ".join(texts),
                        sentences=[Sentence(raw_text=text) for text in texts],
                    )
                    for texts in sentence_texts
                ]
            )

        def extract_aspect(response: list[dict[str, Any]], **aspect_kwargs: Any):
            """
            Extracts an aspect with the given response of the LLM, in one request,
            and returns the document and the aspect.
            """
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            document = create_document()
            document.aspects = [
                Aspect(
                    name="Deadlines",
                    description="Clauses on deadlines",
                    **aspect_kwargs,
                )
            ]
            responses.append(json.dumps(response))
            (aspect,) = llm.extract_aspects_from_document(document)
            assert not responses
            # The output is used as it is: the call is not made again
            assert len(llm.get_usage()[0].usage.calls) == 1
            return document, aspect

        def extract_concept(
            references: dict[str, Any],
            reference_depth: Literal["paragraphs", "sentences"],
        ):
            """
            Extracts a concept with references, with an item that has the given
            references in the response of the LLM, in one request, and returns
            the document and the item.
            """
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            document = create_document()
            document.concepts = [
                StringConcept(
                    name="Delivery time",
                    description="Time that the delivery takes",
                    add_references=True,
                    reference_depth=reference_depth,
                )
            ]
            responses.append(
                json.dumps(
                    [
                        {
                            "concept_id": "C0",
                            "extracted_items": [{"value": "30 days", **references}],
                        }
                    ]
                )
            )
            (concept,) = llm.extract_concepts_from_document(document)
            assert not responses
            assert len(llm.get_usage()[0].usage.calls) == 1
            (item,) = concept.extracted_items
            return document, item

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # === Aspect, paragraphs as items: a paragraph listed twice ===
            document, aspect = extract_aspect(
                [{"aspect_id": "A0", "paragraph_ids": ["P1", "P0", "P1", "P0"]}]
            )
            paragraphs = document.paragraphs
            assert [item.value for item in aspect.extracted_items] == [
                paragraphs[0].raw_text,
                paragraphs[1].raw_text,
            ]
            assert aspect.reference_paragraphs == paragraphs[:2]

            # ... also when the aspect itself is listed twice
            document, aspect = extract_aspect(
                [
                    {"aspect_id": "A0", "paragraph_ids": ["P0", "P1"]},
                    {"aspect_id": "A0", "paragraph_ids": ["P1", "P2"]},
                ]
            )
            assert [item.value for item in aspect.extracted_items] == [
                paragraph.raw_text for paragraph in document.paragraphs
            ]

            # ... and with justifications: that of the first listing is kept
            document, aspect = extract_aspect(
                [
                    {
                        "aspect_id": "A0",
                        "paragraphs": [
                            {"paragraph_id": "P0", "justification": "First"},
                            {"paragraph_id": "P0", "justification": "Second"},
                            {"paragraph_id": "P1", "justification": "Third"},
                        ],
                    }
                ],
                add_justifications=True,
            )
            assert [
                (item.value, item.justification) for item in aspect.extracted_items
            ] == [
                (document.paragraphs[0].raw_text, "First"),
                (document.paragraphs[1].raw_text, "Third"),
            ]

            # === Aspect, sentences as items: a sentence listed twice, for one
            # listing of its paragraph and for two ===
            document, aspect = extract_aspect(
                [
                    {
                        "aspect_id": "A0",
                        "paragraphs": [
                            {
                                "paragraph_id": "P0",
                                "sentences": [
                                    {"sentence_id": "P0-S1"},
                                    {"sentence_id": "P0-S1"},
                                ],
                            },
                            {
                                "paragraph_id": "P1",
                                "sentences": [{"sentence_id": "P1-S1"}],
                            },
                            {
                                "paragraph_id": "P1",
                                "sentences": [
                                    {"sentence_id": "P1-S1"},
                                    {"sentence_id": "P1-S0"},
                                ],
                            },
                        ],
                    }
                ],
                reference_depth="sentences",
            )
            paragraphs = document.paragraphs
            assert [item.value for item in aspect.extracted_items] == [
                sentence_texts[0][1],
                sentence_texts[1][1],
                sentence_texts[1][0],
            ]
            assert aspect.reference_paragraphs == paragraphs[:2]
            assert aspect.reference_sentences == [
                paragraphs[0].sentences[1],
                paragraphs[1].sentences[1],
                paragraphs[1].sentences[0],
            ]

            # === Concept, references to paragraphs: a paragraph listed twice ===
            document, item = extract_concept(
                {"reference_paragraph_ids": ["P0", "P1", "P0"]}, "paragraphs"
            )
            assert item.reference_paragraphs == document.paragraphs[:2]
            assert not item.reference_sentences

            # === Concept, references to sentences: a sentence listed twice, for
            # one listing of its paragraph and for two ===
            document, item = extract_concept(
                {
                    "reference_paragraphs": [
                        {
                            "reference_paragraph_id": "P0",
                            "reference_sentence_ids": ["P0-S1", "P0-S1"],
                        },
                        {
                            "reference_paragraph_id": "P1",
                            "reference_sentence_ids": ["P1-S0"],
                        },
                        {
                            "reference_paragraph_id": "P1",
                            "reference_sentence_ids": ["P1-S0", "P1-S1"],
                        },
                    ]
                },
                "sentences",
            )
            paragraphs = document.paragraphs
            assert item.reference_paragraphs == paragraphs[:2]
            assert item.reference_sentences == [
                paragraphs[0].sentences[1],
                paragraphs[1].sentences[0],
                paragraphs[1].sentences[1],
            ]

    @pytest.mark.parametrize(
        "flow",
        [
            "separate_methods",
            "extract_all",
            "overwrite_existing",
            "added_instances",
            "sentences_segmented_late",
            "decision_model",
        ],
    )
    @pytest.mark.parametrize("use_concurrency", [False, True])
    def test_references_are_paragraphs_and_sentences_of_document(
        self, flow: str, use_concurrency: bool
    ):
        """
        Tests that the reference paragraphs and reference sentences of the aspects
        of a document, of their sub-aspects, and of the extracted items of
        the aspects and of the concepts at both levels, are the paragraphs and
        sentences of the document itself, i.e. the same objects, after
        an extraction: by the separate methods and by ``extract_all()``, with and
        without concurrency, again with ``overwrite_existing=True``, of the aspects
        and concepts added to a processed document, and with the sentences
        segmented after the aspects were extracted. The same holds for the aspects
        extracted by a decision model, and in a clone of the document, for
        the paragraphs and sentences of the clone.

        The sub-aspects and the items of their concepts held equal copies of
        the paragraphs and sentences instead, with the same unique IDs, and so did
        the aspects and concepts that a processed document already had, once
        others were added to them.

        Model API calls are stubbed, i.e. no cassette is required.
        """
        from litellm.types import decisions as decision_types

        litellm = llms_module.litellm
        # The sentences of each paragraph of the document
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price.", "Payment is due in 14 days."],
            ["Either party may terminate.", "The notice period is 60 days."],
            ["This agreement is governed by the laws of Norway."],
        ]
        paragraph_texts = [" ".join(texts) for texts in sentence_texts]

        class SentenceSplitter:
            """
            Replaces the SaT model: splits a text after each full stop.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, e.g. the batch
                    size, which the stub does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                return [re.split(r"(?<=\.) ", text) for text in texts]

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API. An aspect gets two of the paragraphs that
            the prompt lists, starting with the one at the position of the aspect
            in the prompt, each with its first sentence if the prompt lists
            sentences. A concept gets one item, which references the first
            paragraph that the prompt lists, and its first sentence if
            the prompt lists sentences.
            """
            # As with a real request, the call waits for the response: the calls
            # made concurrently are all started before any of them is completed
            await asyncio.sleep(0)
            prompt = kwargs["messages"][-1]["content"]
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            aspect_ids = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            n_paragraphs = len(re.findall(r'<paragraph id="P\d+">', prompt))
            assert n_paragraphs >= 2
            with_sentences = bool(re.search(r'<sentence id="P\d+-S\d+">', prompt))
            content: list[dict[str, Any]] = []
            for concept_id in concept_ids:
                references: dict[str, Any] = (
                    {
                        "reference_paragraphs": [
                            {
                                "reference_paragraph_id": "P0",
                                "reference_sentence_ids": ["P0-S0"],
                            }
                        ]
                    }
                    if with_sentences
                    else {"reference_paragraph_ids": ["P0"]}
                )
                content.append(
                    {
                        "concept_id": concept_id,
                        "extracted_items": [{"value": "30 days", **references}],
                    }
                )
            for aspect_position, aspect_id in enumerate(aspect_ids):
                paragraph_ids = sorted(
                    f"P{(aspect_position + shift) % n_paragraphs}" for shift in range(2)
                )
                content.append(
                    {
                        "aspect_id": aspect_id,
                        "paragraphs": [
                            {
                                "paragraph_id": paragraph_id,
                                "sentences": [{"sentence_id": f"{paragraph_id}-S0"}],
                            }
                            for paragraph_id in paragraph_ids
                        ],
                    }
                    if with_sentences
                    else {"aspect_id": aspect_id, "paragraph_ids": paragraph_ids}
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        async def adecisions_stub(**kwargs: Any):
            """
            Simulates the API of the decision model, which answers "yes" to
            each question of a request: each paragraph, and each sentence,
            belongs to each aspect.
            """
            return decision_types.OpenAIDecisionResponse(
                model="jev-1.13.0",
                answers=tuple(
                    decision_types.OpenAIPredicateAnswer(
                        name=question["name"], probability=0.9
                    )
                    for question in kwargs["questions"]
                ),
                usage=decision_types.OpenAIDecisionUsage(
                    input_tokens=10,
                    output_tokens=0,
                    total_tokens=10,
                    input_tokens_details=decision_types.OpenAIDecisionInputTokensDetails(
                        cached_tokens=0, cache_write_tokens=0
                    ),
                ),
            )

        def create_concept(
            name: str, reference_depth: Literal["paragraphs", "sentences"]
        ) -> StringConcept:
            """
            Creates a concept with references of the given depth.
            """
            return StringConcept(
                name=name,
                description=f"Description of {name}",
                add_references=True,
                reference_depth=reference_depth,
            )

        def create_aspect(
            name: str,
            reference_depth: Literal["paragraphs", "sentences"],
            with_concepts: bool = True,
        ) -> Aspect:
            """
            Creates an aspect with a sub-aspect, each with a concept, unless
            they are to be created without concepts.
            """
            return Aspect(
                name=name,
                description=f"Description of {name}",
                reference_depth=reference_depth,
                concepts=[create_concept(f"Concept of {name}", reference_depth)]
                if with_concepts
                else [],
                aspects=[
                    Aspect(
                        name=f"Sub-aspect of {name}",
                        description=f"Description of the sub-aspect of {name}",
                        reference_depth=reference_depth,
                        concepts=[
                            create_concept(
                                f"Concept of the sub-aspect of {name}", reference_depth
                            )
                        ]
                        if with_concepts
                        else [],
                    )
                ],
            )

        def create_document(
            reference_depth: Literal["paragraphs", "sentences"],
        ) -> Document:
            """
            Creates a document of four paragraphs with their sentences, with
            two aspects, each with a sub-aspect, and with a concept of
            the document, of each aspect and of each sub-aspect.
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
                create_aspect(f"Aspect {i}", reference_depth) for i in range(2)
            ]
            document.concepts = [create_concept("Document concept", reference_depth)]
            return document

        def check_references(document: Document) -> tuple[int, int, int]:
            """
            Checks that each reference paragraph and each reference sentence of
            the aspects of the document, at both levels, and of the extracted
            items of the aspects and of the concepts, is a paragraph or
            a sentence of the document itself, i.e. the same object. Returns
            the numbers of the aspects and items checked, of their reference
            paragraphs, and of their reference sentences.
            """
            holders: list[Any] = []
            concepts = list(document.concepts)
            for aspect in document.aspects:
                for instance in (aspect, *aspect.aspects):
                    assert instance._is_processed
                    holders.extend([instance, *instance.extracted_items])
                    concepts.extend(instance.concepts)
            for concept in concepts:
                holders.extend(concept.extracted_items)
            n_paragraphs = n_sentences = 0
            for holder in holders:
                for paragraph in holder.reference_paragraphs:
                    position = document.get_paragraph_index(paragraph)
                    assert paragraph is document.paragraphs[position]
                    n_paragraphs += 1
                for sentence in holder.reference_sentences:
                    paragraph_position, position = document.get_sentence_index(sentence)
                    document_paragraph = document.paragraphs[paragraph_position]
                    assert sentence is document_paragraph.sentences[position]
                    n_sentences += 1
            return len(holders), n_paragraphs, n_sentences

        def check_clone(document: Document) -> None:
            """
            Checks that the references in a clone of the document are
            the paragraphs and sentences of the clone, and not of the document.
            """
            clone = document.clone()
            assert clone == document
            assert check_references(clone) == check_references(document)
            assert not any(
                paragraph is document_paragraph
                for paragraph, document_paragraph in zip(
                    clone.paragraphs, document.paragraphs, strict=True
                )
            )

        def create_llm() -> DocumentLLM:
            """
            Creates an LLM whose requests are not held back by the limiter.
            """
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            llm.async_limiter = AsyncLimiter(1000, 1)
            return llm

        def extract_by_separate_methods(
            llm: DocumentLLM, document: Document, **kwargs: Any
        ) -> None:
            """
            Extracts the aspects of the document, then the concepts of each
            aspect, then the concepts of the document.
            """
            llm.extract_aspects_from_document(document, **kwargs)
            for aspect in document.aspects:
                llm.extract_concepts_from_aspect(aspect, document, **kwargs)
            llm.extract_concepts_from_document(document, **kwargs)

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(litellm, "adecisions", new=adecisions_stub),
            patch.object(
                documents_module,
                "_get_sat_model",
                new=lambda model_id: SentenceSplitter(),
            ),
        ):
            if flow == "sentences_segmented_late":
                # Nothing requires sentences when the aspects are extracted: they
                # are segmented when the concepts, attached afterwards, need them
                llm = create_llm()
                document = Document(raw_text="\n\n".join(paragraph_texts))
                document.aspects = [
                    create_aspect(f"Aspect {i}", "paragraphs", with_concepts=False)
                    for i in range(2)
                ]
                llm.extract_aspects_from_document(
                    document, use_concurrency=use_concurrency
                )
                assert not document.sentences
                # 2 aspects and 2 sub-aspects, each with 2 paragraphs as its items
                assert check_references(document) == (12, 16, 0)
                document.concepts = [create_concept("Document concept", "sentences")]
                for aspect in document.aspects:
                    for instance in (aspect, *aspect.aspects):
                        instance.concepts = [
                            create_concept(f"Concept of {instance.name}", "sentences")
                        ]
                for aspect in document.aspects:
                    llm.extract_concepts_from_aspect(
                        aspect, document, use_concurrency=use_concurrency
                    )
                llm.extract_concepts_from_document(
                    document, use_concurrency=use_concurrency
                )
                assert len(document.sentences) == 7
                # ... and 5 concepts, each with an item that references a sentence
                assert check_references(document) == (17, 21, 5)
                check_clone(document)
                return

            for reference_depth in ("paragraphs", "sentences"):
                # With sentences as the references, each item of an aspect and of
                # a concept references a sentence, and each aspect the sentences
                # of its two items
                with_sentences = reference_depth == "sentences"
                kwargs: dict[str, Any] = {"use_concurrency": use_concurrency}

                if flow == "decision_model":
                    # A decision model extracts the aspects and the sub-aspects:
                    # each of the 4 paragraphs, or of the 7 sentences, is an item
                    # of each of the 2 aspects and 2 sub-aspects
                    decision_model = DocumentDecisionModel(
                        model="typesafe/jev-latest", api_key="test_api_key"
                    )
                    decision_model.async_limiter = AsyncLimiter(1000, 1)
                    document = create_document(reference_depth)
                    document.remove_all_concepts()
                    document.aspects = [
                        create_aspect(
                            f"Aspect {i}", reference_depth, with_concepts=False
                        )
                        for i in range(2)
                    ]
                    expected = (32, 44, 56) if with_sentences else (20, 32, 0)
                    decision_model.extract_aspects_from_document(document, **kwargs)
                    assert check_references(document) == expected
                    decision_model.extract_aspects_from_document(
                        document, overwrite_existing=True, **kwargs
                    )
                    assert check_references(document) == expected
                    check_clone(document)
                    continue

                llm = create_llm()
                document = create_document(reference_depth)
                # 2 aspects and 2 sub-aspects, each with 2 items, and 5 concepts,
                # each with an item
                all_extracted = (17, 21, 21 if with_sentences else 0)

                if flow == "separate_methods":
                    llm.extract_aspects_from_document(document, **kwargs)
                    assert check_references(document) == (
                        12,
                        16,
                        16 if with_sentences else 0,
                    )
                    for aspect in document.aspects:
                        llm.extract_concepts_from_aspect(aspect, document, **kwargs)
                    llm.extract_concepts_from_document(document, **kwargs)
                    assert check_references(document) == all_extracted

                elif flow == "extract_all":
                    llm.extract_all(document, **kwargs)
                    assert check_references(document) == all_extracted

                elif flow == "overwrite_existing":
                    llm.extract_all(document, **kwargs)
                    extract_by_separate_methods(
                        llm, document, overwrite_existing=True, **kwargs
                    )
                    assert check_references(document) == all_extracted
                    llm.extract_all(document, overwrite_existing=True, **kwargs)
                    assert check_references(document) == all_extracted

                elif flow == "added_instances":
                    llm.extract_all(document, **kwargs)
                    # An aspect with a sub-aspect, and a concept at each level, are
                    # added to the processed document: the aspects and concepts
                    # that it already has are copied with those that are added
                    document.add_aspects(
                        [create_aspect("Added aspect", reference_depth)]
                    )
                    document.add_concepts(
                        [create_concept("Added document concept", reference_depth)]
                    )
                    for instance in (
                        document.aspects[0],
                        document.aspects[0].aspects[0],
                    ):
                        instance.add_concepts(
                            [
                                create_concept(
                                    f"Added concept of {instance.name}", reference_depth
                                )
                            ]
                        )
                    added_aspect = document.get_aspect_by_name("Added aspect")
                    llm.extract_aspects_from_document(
                        document, from_aspects=[added_aspect], **kwargs
                    )
                    llm.extract_concepts_from_aspect(added_aspect, document, **kwargs)
                    llm.extract_concepts_from_document(
                        document,
                        from_concepts=[
                            document.get_concept_by_name("Added document concept")
                        ],
                        **kwargs,
                    )
                    # The concepts of the first aspect and of its sub-aspect are
                    # extracted again, with those that are added to them
                    llm.extract_concepts_from_aspect(
                        document.aspects[0], document, overwrite_existing=True, **kwargs
                    )
                    # 3 aspects and 3 sub-aspects, each with 2 items, and
                    # 10 concepts, each with an item
                    assert check_references(document) == (
                        28,
                        34,
                        34 if with_sentences else 0,
                    )

                check_clone(document)

    def test_document_relinks_references_to_its_paragraphs_and_sentences(self):
        """
        Tests that a document replaces the copies of its paragraphs and sentences,
        among the references of its aspects and of the extracted items, with
        the paragraphs and sentences themselves: each copy is replaced with
        the one that has the same unique ID, in place, also when the copy has
        other content, e.g. no sentences; a reference that the document has no
        paragraph or sentence with the same unique ID for is left as it is; and
        the serialized document is the same before and after, when the copies are
        equal to what they are copies of.

        Such copies are made when aspects and concepts are assigned, as they are
        copied with what they reference, and are replaced when the next extraction
        starts: here, the aspect of a document is assigned to it again before
        the sentences are segmented for a concept that is added to the aspect.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price.", "Payment is due in 14 days."],
            ["This agreement is governed by the laws of Norway."],
        ]

        def create_document(with_sentences: bool = True) -> Document:
            """
            Creates a document of three paragraphs, with their sentences unless
            it is to be created without them, with an aspect and a concept that
            have the references that an extraction gives them: the aspect has
            the first two paragraphs and the second sentence of each, and
            the item of the concept has the third paragraph and its sentence.
            """
            document = Document(
                paragraphs=[
                    Paragraph(
                        raw_text=" ".join(texts),
                        sentences=[Sentence(raw_text=text) for text in texts]
                        if with_sentences
                        else [],
                    )
                    for texts in sentence_texts
                ]
            )
            document.aspects = [
                Aspect(
                    name="Duties",
                    description="Duties of the parties",
                    reference_depth="sentences" if with_sentences else "paragraphs",
                )
            ]
            document.concepts = [
                StringConcept(
                    name="Governing law",
                    description="Law that governs the agreement",
                    add_references=True,
                    reference_depth="sentences" if with_sentences else "paragraphs",
                )
            ]
            (aspect,) = document.aspects
            (concept,) = document.concepts
            aspect_items = []
            for paragraph in document.paragraphs[:2]:
                item = aspect._item_class(value=paragraph.raw_text)
                item.reference_paragraphs = [paragraph]
                item.reference_sentences = paragraph.sentences[1:]
                aspect_items.append(item)
            aspect.extracted_items = aspect_items  # ty: ignore[invalid-assignment]
            aspect.reference_paragraphs = document.paragraphs[:2]
            aspect.reference_sentences = [
                sentence
                for item in aspect_items
                for sentence in item.reference_sentences
            ]
            concept_item = concept._item_class(value="Norway")
            concept_item.reference_paragraphs = document.paragraphs[2:]
            concept_item.reference_sentences = list(document.paragraphs[2].sentences)
            concept.extracted_items = [concept_item]
            for instance in (aspect, concept):
                instance._is_processed = True
            return document

        def get_holders(document: Document) -> list[Any]:
            """
            Returns the aspect of the document, its items and the item of
            the concept, i.e. what holds references.
            """
            (aspect,) = document.aspects
            (concept,) = document.concepts
            return [aspect, *aspect.extracted_items, *concept.extracted_items]

        def get_reference_lists(document: Document) -> list[list[Any]]:
            """
            Returns the lists of the reference paragraphs and of the reference
            sentences of the aspect and of the items of the document.
            """
            return [
                references
                for holder in get_holders(document)
                for references in (
                    holder.reference_paragraphs,
                    holder.reference_sentences,
                )
            ]

        def replace_references_with_copies(document: Document) -> None:
            """
            Replaces each reference of the document with a copy of it.
            """
            for holder in get_holders(document):
                holder.reference_paragraphs = [
                    paragraph.clone() for paragraph in holder.reference_paragraphs
                ]
                holder.reference_sentences = [
                    sentence.clone() for sentence in holder.reference_sentences
                ]

        def get_reference_positions(document: Document) -> list[Any]:
            """
            Checks that each reference of the document is a paragraph or
            a sentence of the document itself, i.e. the same object, and returns
            the positions of the references of each holder in the document.
            """
            positions = []
            for holder in get_holders(document):
                paragraph_positions = []
                for paragraph in holder.reference_paragraphs:
                    position = document.get_paragraph_index(paragraph)
                    assert paragraph is document.paragraphs[position]
                    paragraph_positions.append(position)
                sentence_positions = []
                for sentence in holder.reference_sentences:
                    paragraph_position, position = document.get_sentence_index(sentence)
                    document_paragraph = document.paragraphs[paragraph_position]
                    assert sentence is document_paragraph.sentences[position]
                    sentence_positions.append((paragraph_position, position))
                positions.append((paragraph_positions, sentence_positions))
            return positions

        expected_positions = [
            ([0, 1], [(0, 1), (1, 1)]),
            ([0], [(0, 1)]),
            ([1], [(1, 1)]),
            ([2], [(2, 0)]),
        ]

        # === Equal copies are replaced, in place ===
        document = create_document()
        assert get_reference_positions(document) == expected_positions
        serialized = document.to_dict()
        replace_references_with_copies(document)
        (aspect,) = document.aspects
        assert aspect.reference_paragraphs[0] is not document.paragraphs[0]
        assert aspect.reference_sentences[0] is not document.paragraphs[0].sentences[1]
        # Equal copies are serialized as what they are copies of
        assert document.to_dict() == serialized
        reference_lists = get_reference_lists(document)
        document._relink_references()
        assert get_reference_positions(document) == expected_positions
        assert document.to_dict() == serialized
        assert document.to_json() == json.dumps(serialized, ensure_ascii=False)
        # The copies are replaced in the lists that the aspect and the items have
        assert all(
            references is references_before
            for references, references_before in zip(
                get_reference_lists(document), reference_lists, strict=True
            )
        )
        # Nothing changes when there is nothing to replace
        document._relink_references()
        assert get_reference_positions(document) == expected_positions
        assert document.to_dict() == serialized

        # === A reference with no counterpart in the document is left as it is ===
        document = create_document()
        replace_references_with_copies(document)
        (aspect,) = document.aspects
        other_paragraph = Paragraph(
            raw_text="Another text. Of another document.",
            sentences=[
                Sentence(raw_text="Another text."),
                Sentence(raw_text="Of another document."),
            ],
        )
        aspect.reference_paragraphs = [
            aspect.reference_paragraphs[0],
            other_paragraph,
            aspect.reference_paragraphs[1],
        ]
        aspect.reference_sentences = [
            other_paragraph.sentences[1],
            *aspect.reference_sentences,
        ]
        document._relink_references()
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(document.paragraphs[0]),
            id(other_paragraph),
            id(document.paragraphs[1]),
        ]
        assert [id(sentence) for sentence in aspect.reference_sentences] == [
            id(other_paragraph.sentences[1]),
            id(document.paragraphs[0].sentences[1]),
            id(document.paragraphs[1].sentences[1]),
        ]

        # === A copy with other content is replaced as well ===
        # The copies have what the paragraphs and sentences of the document had
        # when they were copied, e.g. the custom data that is changed afterwards
        document = create_document()
        replace_references_with_copies(document)
        document.paragraphs[0].custom_data = {"page": 1}
        document.paragraphs[0].sentences[1].custom_data = {"page": 1}
        (aspect,) = document.aspects
        assert aspect.reference_paragraphs[0] != document.paragraphs[0]
        assert aspect.reference_paragraphs[0].custom_data == {}
        document._relink_references()
        assert get_reference_positions(document) == expected_positions
        assert aspect.reference_paragraphs[0].custom_data == {"page": 1}
        assert aspect.extracted_items[0].reference_sentences[0].custom_data == {
            "page": 1
        }

        # === Paragraphs without sentences, and a document without text ===
        document = create_document(with_sentences=False)
        serialized = document.to_dict()
        replace_references_with_copies(document)
        document._relink_references()
        assert get_reference_positions(document) == [
            ([0, 1], []),
            ([0], []),
            ([1], []),
            ([2], []),
        ]
        assert document.to_dict() == serialized
        image_document = Document(images=[self.test_img_png_invoice])
        image_document.concepts = [
            StringConcept(name="Title", description="Title of the document")
        ]
        image_document._relink_references()

        # === Copies made by an assignment are replaced by the next extraction ===
        # The aspect is assigned again, i.e. copied with its reference paragraphs,
        # before the sentences are segmented: the copies would be left without
        # the sentences that the concept of the aspect is extracted with
        prompts: list[str] = []

        class SentenceSplitter:
            """
            Replaces the SaT model: splits a text after each full stop.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, e.g. the batch
                    size, which the stub does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                return [re.split(r"(?<=\.) ", text) for text in texts]

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: the concept gets an item that references
            the second sentence of the first paragraph that the prompt lists.
            """
            prompts.append(kwargs["messages"][-1]["content"])
            content = [
                {
                    "concept_id": "C0",
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
            ]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        document = create_document(with_sentences=False)
        document.remove_all_concepts()
        document.aspects = document.aspects
        (aspect,) = document.aspects
        aspect.concepts = [
            StringConcept(
                name="Delivery time",
                description="Time that the delivery takes",
                add_references=True,
                reference_depth="sentences",
            )
        ]
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(
                documents_module,
                "_get_sat_model",
                new=lambda model_id: SentenceSplitter(),
            ),
        ):
            (concept,) = llm.extract_concepts_from_aspect(aspect, document)
        # The prompt lists the sentences of the paragraphs of the aspect
        (prompt,) = prompts
        assert all(text in prompt for text in [*sentence_texts[0], *sentence_texts[1]])
        (item,) = concept.extracted_items
        assert item.reference_paragraphs[0] is document.paragraphs[0]
        assert item.reference_sentences[0] is document.paragraphs[0].sentences[1]
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraph) for paragraph in document.paragraphs[:2]
        ]
        for aspect_item, paragraph in zip(
            aspect.extracted_items, document.paragraphs[:2], strict=True
        ):
            assert aspect_item.reference_paragraphs[0] is paragraph

    def test_aspect_references_are_told_apart_by_unique_id(self):
        """
        Tests that the references of the extracted items of an aspect are added
        to the references of the aspect by their unique IDs: a paragraph or
        a sentence that the aspect already has a copy of, with the same unique ID
        and other content, e.g. without the sentences of the original, is not
        added again.

        Such a copy is not equal to the original, so it was taken for another
        reference: the list of the references of the aspect was extended in place
        and then rejected, with ``ValueError`` for duplicates, and kept
        the duplicate.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        sentence_texts = [
            ["The supplier delivers the goods.", "Delivery takes 30 days."],
            ["The buyer pays the price.", "Payment is due in 14 days."],
        ]

        def create_document() -> Document:
            """
            Creates a document of two paragraphs with their sentences, with
            an aspect that has sentences as its items.
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
                    name="Deadlines",
                    description="Clauses on deadlines",
                    reference_depth="sentences",
                )
            ]
            return document

        def create_copies(document: Document) -> tuple[Any, Any]:
            """
            Returns a copy of the first paragraph of the document without its
            sentences, and a copy of its second sentence with other custom
            data: each has the unique ID of the original, and is not equal to it.
            """
            paragraph = document.paragraphs[0]
            paragraph_copy = Paragraph.from_dict(
                {**paragraph.to_dict(), "sentences": []}
            )
            sentence_copy = paragraph.sentences[1].clone()
            sentence_copy.custom_data = {"page": 1}
            for original, copy in (
                (paragraph, paragraph_copy),
                (paragraph.sentences[1], sentence_copy),
            ):
                assert copy.unique_id == original.unique_id
                assert copy != original
                assert copy not in [original]
            return paragraph_copy, sentence_copy

        # === An aspect that has the copies as its references is extracted ===
        # The references that an aspect has are kept, and those of its items
        # are added to them
        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: the second sentence of each paragraph is
            extracted for the aspect.
            """
            content = [
                {
                    "aspect_id": "A0",
                    "paragraphs": [
                        {
                            "paragraph_id": f"P{position}",
                            "sentences": [{"sentence_id": f"P{position}-S1"}],
                        }
                        for position in range(2)
                    ],
                }
            ]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        document = create_document()
        (aspect,) = document.aspects
        paragraphs = document.paragraphs
        paragraph_copy, sentence_copy = create_copies(document)
        aspect.reference_paragraphs = [paragraph_copy]
        aspect.reference_sentences = [sentence_copy]
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        with patch.object(litellm, "acompletion", new=acompletion_stub):
            llm.extract_aspects_from_document(document)
        assert [item.value for item in aspect.extracted_items] == [
            sentence_texts[0][1],
            sentence_texts[1][1],
        ]
        # Each paragraph and sentence is a reference of the aspect once: that
        # of the document, which replaces its copy when the extraction starts
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraphs[0]),
            id(paragraphs[1]),
        ]
        assert [id(sentence) for sentence in aspect.reference_sentences] == [
            id(paragraphs[0].sentences[1]),
            id(paragraphs[1].sentences[1]),
        ]

        # === The references of the items are added to an aspect that has copies ===
        # Here, the copies are not replaced first, as they are when an extraction
        # starts: the references of the items, some of which several items have,
        # are added to those of the aspect as they are
        document = create_document()
        (aspect,) = document.aspects
        paragraphs = document.paragraphs
        paragraph_copy, sentence_copy = create_copies(document)
        aspect.reference_paragraphs = [paragraph_copy]
        aspect.reference_sentences = [sentence_copy]
        references_before = (aspect.reference_paragraphs, aspect.reference_sentences)
        items = []
        for paragraph_position, sentence_position in [(1, 0), (0, 1), (0, 0), (1, 0)]:
            sentence = paragraphs[paragraph_position].sentences[sentence_position]
            item = aspect._item_class(value=sentence.raw_text)
            item.reference_paragraphs = [paragraphs[paragraph_position]]
            item.reference_sentences = [sentence]
            items.append(item)
        aspect.extracted_items = items  # ty: ignore[invalid-assignment]
        aspect._add_references_of_extracted_items()
        # The copies that the aspect had are kept, and what they are copies of
        # is not added: the other references follow in the order of the items
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraph_copy),
            id(paragraphs[1]),
        ]
        assert [id(sentence) for sentence in aspect.reference_sentences] == [
            id(sentence_copy),
            id(paragraphs[1].sentences[0]),
            id(paragraphs[0].sentences[0]),
        ]
        # The lists that the aspect had are not extended in place
        assert references_before == ([paragraph_copy], [sentence_copy])

    def test_concepts_of_aspects_that_reference_another_document(self):
        """
        Tests that concepts are extracted from an aspect and from its sub-aspect
        whose reference paragraphs are not paragraphs of the document, as when
        the aspects were extracted from another document and assigned to this
        one. Such references have no counterpart in the document and are left as
        they are, also when a reference of an item of the sub-aspect is not equal
        to the reference of the sub-aspect with the same unique ID. The items of
        the concepts reference the paragraphs of the aspect and of the sub-aspect.

        The concepts of a sub-aspect are extracted with a temporary document,
        built from the paragraphs of the sub-aspect, which holds a copy of
        the sub-aspect: the sub-aspect is to stay equal to that copy, so
        the references of the copy are not replaced with the paragraphs of
        the temporary document.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: each aspect gets the first two paragraphs that
            the prompt lists, and each concept an item that references the first
            one.
            """
            prompt = kwargs["messages"][-1]["content"]
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            aspect_ids = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            content = [
                {
                    "concept_id": concept_id,
                    "extracted_items": [
                        {"value": "30 days", "reference_paragraph_ids": ["P0"]}
                    ],
                }
                for concept_id in concept_ids
            ] or [
                {"aspect_id": aspect_id, "paragraph_ids": ["P0", "P1"]}
                for aspect_id in aspect_ids
            ]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            for with_unequal_reference in (False, True):
                llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
                source_document = Document(
                    raw_text="The supplier delivers the goods in 30 days.\n\n"
                    "The buyer pays the price in 14 days.\n\n"
                    "This agreement is governed by the laws of Norway."
                )
                source_document.aspects = [
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
                llm.extract_aspects_from_document(source_document)

                # The aspects are assigned to a document with other paragraphs
                document = Document(raw_text="Another text.\n\nOf another document.")
                document.aspects = source_document.aspects
                (aspect,) = document.aspects
                (sub_aspect,) = aspect.aspects
                unequal_reference = None
                if with_unequal_reference:
                    # A reference of an item of the sub-aspect has other content
                    # than the reference of the sub-aspect with the same unique ID
                    item = sub_aspect.extracted_items[0]
                    unequal_reference = item.reference_paragraphs[0].clone()
                    unequal_reference.custom_data = {"page": 1}
                    item.reference_paragraphs = [unequal_reference]
                    assert unequal_reference != sub_aspect.reference_paragraphs[0]
                for instance in (aspect, sub_aspect):
                    instance.concepts = [
                        StringConcept(
                            name=f"Delivery time in {instance.name}",
                            description="Time that the delivery takes",
                            add_references=True,
                        )
                    ]
                references_before = [
                    [id(paragraph) for paragraph in instance.reference_paragraphs]
                    for instance in (aspect, sub_aspect)
                ]
                llm.extract_concepts_from_aspect(aspect, document)

                for instance, reference_ids in zip(
                    (aspect, sub_aspect), references_before, strict=True
                ):
                    # The references of the aspect are left as they are: none of
                    # them is a paragraph of the document
                    assert len(instance.reference_paragraphs) == 2
                    assert [
                        id(paragraph) for paragraph in instance.reference_paragraphs
                    ] == reference_ids
                    for paragraph in instance.reference_paragraphs:
                        with pytest.raises(ValueError, match="was not found"):
                            document.get_paragraph_index(paragraph)
                    # The item of the concept references the first of them
                    (concept,) = instance.concepts
                    (concept_item,) = concept.extracted_items
                    (reference_paragraph,) = concept_item.reference_paragraphs
                    assert reference_paragraph is instance.reference_paragraphs[0]
                if unequal_reference is not None:
                    (reference_paragraph,) = sub_aspect.extracted_items[
                        0
                    ].reference_paragraphs
                    assert reference_paragraph is unequal_reference

    def test_concepts_of_sub_aspect_extracted_in_several_calls(self):
        """
        Tests that the concepts of a sub-aspect are extracted also when it takes
        more than one call: a call per concept (``max_items_per_call=1``), a call
        for each group of concepts with the same extraction params, and a call
        that is made again after another call for the sub-aspect is completed.

        Each call checks that the aspect is assigned to the document. For
        a sub-aspect, the document is one built from the paragraphs of its parent
        aspect, which holds a copy of the sub-aspect made before the extraction.
        The sub-aspect is told from the other aspects by its unique ID. It was
        compared with the copy, which it was no longer equal to once a call had
        given a concept its items: ``ValueError`` was raised for the next call.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        justified_concept_name = "Notice period"
        # The names of the concepts that are extracted with justifications
        justified_concept_names: set[str] = set()
        # The names of the concepts of each request, and the names of the concepts
        # that the first response is invalid for
        requests: list[list[str]] = []
        invalid_first_response_for: set[str] = set()

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: each aspect gets the paragraphs of the prompt,
            and each concept gets one item, with a justification for the concept
            that is extracted with justifications. As for a real request,
            the call is suspended until the response arrives.
            """
            await asyncio.sleep(0)
            prompt = kwargs["messages"][-1]["content"]
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            concept_names = re.findall(r"<concept_name>(.+?)</concept_name>", prompt)
            if concept_ids:
                requests.append(concept_names)
                item: Any = "30 days"
                if justified_concept_names.intersection(concept_names):
                    item = {"value": "30 days", "justification": "Stated in the text."}
                content = json.dumps(
                    [
                        {"concept_id": concept_id, "extracted_items": [item]}
                        for concept_id in concept_ids
                    ]
                )
                invalid_names = invalid_first_response_for.intersection(concept_names)
                if invalid_names:
                    invalid_first_response_for.difference_update(invalid_names)
                    content = "<<<not-json>>>"
            else:
                content = json.dumps(
                    [
                        {"aspect_id": aspect_id, "paragraph_ids": ["P0", "P1"]}
                        for aspect_id in re.findall(
                            r"<aspect_id>(A\d+)</aspect_id>", prompt
                        )
                    ]
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_document(with_justified_concept: bool = False) -> Document:
            """
            Creates a document with an aspect that has a concept and a sub-aspect
            with two concepts, the second of which is optionally extracted with
            justifications, i.e. in a call of its own.
            """
            justified_concept_names.clear()
            if with_justified_concept:
                justified_concept_names.add(justified_concept_name)
            document = Document(
                raw_text="The supplier delivers the goods in 30 days.\n\n"
                "Either party may terminate with a notice of 30 days."
            )
            document.aspects = [
                Aspect(
                    name="Duties",
                    description="Duties of the parties",
                    concepts=[
                        StringConcept(
                            name="Parties", description="Parties that have duties"
                        )
                    ],
                    aspects=[
                        Aspect(
                            name="Deadlines",
                            description="Deadlines of the duties",
                            concepts=[
                                StringConcept(
                                    name="Delivery time",
                                    description="Time that the delivery takes",
                                ),
                                StringConcept(
                                    name=justified_concept_name,
                                    description="Period of the notice of termination",
                                    add_justifications=with_justified_concept,
                                ),
                            ],
                        )
                    ],
                )
            ]
            return document

        def check_concepts_extracted(document: Document) -> None:
            """
            Checks that each concept of the aspect and of its sub-aspect has
            its item.
            """
            (aspect,) = document.aspects
            (sub_aspect,) = aspect.aspects
            for concept in [*aspect.concepts, *sub_aspect.concepts]:
                (item,) = concept.extracted_items
                assert item.value == "30 days", concept.name
                assert (item.justification is not None) == concept.add_justifications

        sub_aspect_requests = [["Delivery time"], [justified_concept_name]]
        with patch.object(litellm, "acompletion", new=acompletion_stub):
            for use_extract_all in (False, True):

                def extract(document: Document, **kwargs: Any) -> None:
                    """
                    Extracts the concepts of the aspect and of its sub-aspect,
                    after the aspects, with the methods of each step or with
                    the method that does all of them.
                    """
                    llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
                    requests.clear()
                    if use_extract_all:  # noqa: B023
                        llm.extract_all(document, **kwargs)
                    else:
                        llm.extract_aspects_from_document(document, **kwargs)
                        llm.extract_concepts_from_aspect(
                            document.aspects[0], document, **kwargs
                        )

                # === A call per concept ===
                document = create_document()
                extract(document, max_items_per_call=1)
                check_concepts_extracted(document)
                assert requests == [["Parties"], *sub_aspect_requests]

                # === A call for each group of concepts with the same params ===
                document = create_document(with_justified_concept=True)
                extract(document)
                check_concepts_extracted(document)
                assert requests == [["Parties"], *sub_aspect_requests]

                # === A call that is made again when another call is completed ===
                # With concurrency, each concept is extracted in a call of its
                # own, and the calls with invalid data are made again when all
                # the calls are made
                document = create_document()
                invalid_first_response_for.add("Delivery time")
                extract(document, use_concurrency=True)
                assert not invalid_first_response_for
                check_concepts_extracted(document)
                assert sorted(requests) == sorted(
                    [["Parties"], *sub_aspect_requests, ["Delivery time"]]
                )

            # === An aspect of another document is rejected, as before ===
            document = create_document()
            other_document = create_document()
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            llm.extract_aspects_from_document(other_document)
            llm.extract_aspects_from_document(document)
            requests.clear()
            with pytest.raises(ValueError, match="must be assigned to the document"):
                llm.extract_concepts_from_aspect(other_document.aspects[0], document)
            assert not requests

    def test_call_made_again_continues_from_part_that_failed(self):
        """
        Tests that a call whose content is sent to an LLM in parts, i.e. with
        a max number of paragraphs per call, is continued from the part that
        the LLM returned invalid data for when the call is made again:
        the parts that were answered with valid data are not sent again, and what
        was extracted from them is kept. Nothing of the failed attempt for a part
        is kept. The retries are those of the call, not of each part, and
        a fallback LLM extracts from all the parts, with retries continued in
        the same way.

        Each time such a call was made again, all its parts were sent again.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"
        n_parts = 4
        paragraph_texts = [
            f"Clause {number}: a duty of the parties." for number in range(1, 5)
        ]
        invalid = "<<<not-json>>>"
        # The responses to the requests of a model for an aspect or a concept and
        # a part, in turn, by the model, the name of the aspect or concept and
        # the number of the clause of the part: the invalid text, or a response
        # to give instead of the default one, with an item for the part
        responses: dict[tuple[str, str, int], list[Any]] = {}
        # The requests sent: the model, the name, and the number of the clause
        requests: list[tuple[str, str, int]] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, which is sent one paragraph in each request.
            """
            await asyncio.sleep(0)
            prompt = kwargs["messages"][-1]["content"]
            model = kwargs["model"]
            (clause,) = {int(number) for number in re.findall(r"Clause (\d)", prompt)}
            concept_names = re.findall(r"<concept_name>(.+?)</concept_name>", prompt)
            aspect_names = re.findall(r"<aspect_name>(.+?)</aspect_name>", prompt)
            (name,) = concept_names or aspect_names
            request = (model, name, clause)
            requests.append(request)
            content = responses[request].pop(0) if responses.get(request) else None
            if content is None:
                content = (
                    [{"concept_id": "C0", "extracted_items": [f"{model}: {clause}"]}]
                    if concept_names
                    else [{"aspect_id": "A0", "paragraph_ids": ["P0"]}]
                )
            if content != invalid:
                content = json.dumps(content)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        def create_llm(with_fallback: bool = False, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM, optionally with a fallback LLM.
            """
            llm = DocumentLLM(model=primary_model, api_key="test_api_key", **kwargs)
            if with_fallback:
                llm.fallback_models = [
                    DocumentLLM(
                        model=fallback_model, api_key="test_api_key", is_fallback=True
                    )
                ]
            return llm

        def create_document(**concept_kwargs: Any) -> Document:
            """
            Creates a document of four paragraphs with a concept.
            """
            document = Document(raw_text="\n\n".join(paragraph_texts))
            document.concepts = [
                StringConcept(
                    name="Duty", description="Duty of the parties", **concept_kwargs
                )
            ]
            return document

        def set_responses(
            request_responses: dict[tuple[str, str, int], list[Any]],
        ) -> None:
            """
            Sets the responses to give instead of the default ones, and resets
            the record of the requests.
            """
            responses.clear()
            responses.update(request_responses)
            requests.clear()

        def sent_parts(model: str, name: str = "Duty") -> list[int]:
            """
            Returns the numbers of the clauses of the parts sent to the model
            for the aspect or concept, in the order of the requests.
            """
            return [
                clause
                for request_model, request_name, clause in requests
                if (request_model, request_name) == (model, name)
            ]

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # === Invalid data for a part: the call is continued from that part ===
            llm = create_llm()
            document = create_document()
            set_responses({(primary_model, "Duty", 3): [invalid]})
            (concept,) = llm.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=1
            )
            assert sent_parts(primary_model) == [1, 2, 3, 3, 4]
            # An item from each part, in the order of the parts
            assert [item.value for item in concept.extracted_items] == [
                f"{primary_model}: {clause}" for clause in range(1, n_parts + 1)
            ]
            usage = llm.get_usage()[0].usage
            assert len(usage.calls) == 5
            assert (usage.input, usage.output) == (50, 25)

            # === Nothing of a failed attempt for a part is kept ===
            # The response for the second part gives an item, and is then found
            # invalid for a paragraph that the part does not have
            llm = create_llm()
            document = Document(raw_text="\n\n".join(paragraph_texts))
            document.aspects = [
                Aspect(
                    name="Duties",
                    description="Duties of the parties",
                    add_justifications=True,
                )
            ]

            def get_aspect_response(*paragraphs: tuple[str, str]) -> list[Any]:
                """
                Returns the response with the given paragraphs for the aspect,
                each as its ID and the justification.
                """
                return [
                    {
                        "aspect_id": "A0",
                        "paragraphs": [
                            {"paragraph_id": paragraph_id, "justification": text}
                            for paragraph_id, text in paragraphs
                        ],
                    }
                ]

            set_responses(
                {
                    (primary_model, "Duties", clause): [
                        get_aspect_response(("P0", f"Justification {clause}"))
                    ]
                    for clause in (1, 3, 4)
                }
                | {
                    (primary_model, "Duties", 2): [
                        get_aspect_response(("P0", "Failed attempt"), ("P7", "None")),
                        get_aspect_response(("P0", "Justification 2")),
                    ]
                }
            )
            (aspect,) = llm.extract_aspects_from_document(
                document, max_paragraphs_to_analyze_per_call=1
            )
            assert sent_parts(primary_model, "Duties") == [1, 2, 2, 3, 4]
            assert [
                (item.value, item.justification) for item in aspect.extracted_items
            ] == [
                (text, f"Justification {position + 1}")
                for position, text in enumerate(paragraph_texts)
            ]
            assert aspect.reference_paragraphs == document.paragraphs

            # === The retries are those of the call, not of each part ===
            llm = create_llm(max_retries_invalid_data=1)
            document = create_document()
            set_responses(
                {
                    (primary_model, "Duty", 2): [invalid],
                    (primary_model, "Duty", 3): [invalid],
                }
            )
            with pytest.raises(ModelExtractionError) as exc_info:
                llm.extract_concepts_from_document(
                    document, max_paragraphs_to_analyze_per_call=1
                )
            assert exc_info.value.retry_count == 1
            assert sent_parts(primary_model) == [1, 2, 2, 3]
            assert not document.concepts[0].extracted_items

            # === A concept with a single occurrence ===
            # The first part has no item; the parts after the one that gives
            # the item are not sent, as before
            llm = create_llm()
            document = create_document(singular_occurrence=True)
            set_responses(
                {
                    (primary_model, "Duty", 1): [
                        [{"concept_id": "C0", "extracted_items": []}]
                    ],
                    (primary_model, "Duty", 2): [invalid],
                }
            )
            (concept,) = llm.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=1
            )
            assert sent_parts(primary_model) == [1, 2, 2]
            assert [item.value for item in concept.extracted_items] == [
                f"{primary_model}: 2"
            ]

            # === A fallback LLM extracts from all the parts ===
            # ... and is continued from the part that it fails for
            llm = create_llm(with_fallback=True, max_retries_invalid_data=0)
            document = create_document()
            set_responses(
                {
                    (primary_model, "Duty", 2): [invalid],
                    (fallback_model, "Duty", 3): [invalid],
                }
            )
            (concept,) = llm.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=1
            )
            assert sent_parts(primary_model) == [1, 2]
            assert sent_parts(fallback_model) == [1, 2, 3, 3, 4]
            # The items are those of the fallback LLM only
            assert [item.value for item in concept.extracted_items] == [
                f"{fallback_model}: {clause}" for clause in range(1, n_parts + 1)
            ]

            # === With concurrency: each call is continued on its own ===
            llm = create_llm()
            document = Document(raw_text="\n\n".join(paragraph_texts))
            document.concepts = [
                StringConcept(name=name, description=f"{name} of the parties")
                for name in ("Duty", "Right")
            ]
            set_responses(
                {
                    (primary_model, "Duty", 2): [invalid],
                    (primary_model, "Right", 4): [invalid],
                }
            )
            llm.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=1, use_concurrency=True
            )
            assert sent_parts(primary_model, "Duty") == [1, 2, 2, 3, 4]
            assert sent_parts(primary_model, "Right") == [1, 2, 3, 4, 4]
            for concept in document.concepts:
                assert [item.value for item in concept.extracted_items] == [
                    f"{primary_model}: {clause}" for clause in range(1, n_parts + 1)
                ]

    @pytest.mark.parametrize(
        "method_name",
        [
            "extract_all_async",
            "extract_aspects_from_document_async",
            "extract_concepts_from_aspect_async",
            "extract_concepts_from_document_async",
        ],
    )
    def test_sentences_segmented_without_blocking_event_loop(self, method_name: str):
        """
        Tests that the async extraction methods segment the sentences of a document
        in a worker thread: the other tasks of the event loop run while the SaT
        model segments them.

        No LLM API calls are made, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        segmentation_started = threading.Event()
        loop_ran_during_segmentation = threading.Event()

        class WaitingSentenceSplitter:
            """
            Replaces the SaT model: splits a text after each full stop once another
            task of the event loop has run, which it cannot while the segmentation
            blocks the event loop.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, e.g. the batch
                    size, which the stub does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                segmentation_started.set()
                assert loop_ran_during_segmentation.wait(timeout=10), (
                    "The event loop was blocked while the sentences were segmented"
                )
                return [re.split(r"(?<=\.) ", text) for text in texts]

        async def acompletion_stub(**kwargs: Any) -> Any:
            """
            Simulates the LLM API: the first paragraph is extracted for an aspect
            with paragraphs as its references, and nothing else is extracted.
            """
            prompt = kwargs["messages"][-1]["content"]
            concept_ids = re.findall(r"<concept_id>(C\d+)</concept_id>", prompt)
            aspect_ids = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            if concept_ids:
                content = [
                    {"concept_id": i, "extracted_items": []} for i in concept_ids
                ]
            elif "<sentence id=" in prompt:
                content = [{"aspect_id": i, "paragraphs": []} for i in aspect_ids]
            else:
                content = [
                    {"aspect_id": i, "paragraph_ids": ["P0"]} for i in aspect_ids
                ]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=json.dumps(content))
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        async def run_during_segmentation() -> None:
            """
            Runs on the event loop once the segmentation has started.
            """
            while not segmentation_started.is_set():
                await asyncio.sleep(0.01)
            loop_ran_during_segmentation.set()

        def create_concept() -> StringConcept:
            """
            Creates a concept with references to sentences.
            """
            return StringConcept(
                name="Duty",
                description="A duty of a party",
                add_references=True,
                reference_depth="sentences",
            )

        async def run() -> Document:
            """
            Runs the extraction method, and the task that is to run meanwhile.
            """
            document = Document(raw_text="First duty. Second duty.\n\nThird duty.")
            llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
            if method_name == "extract_concepts_from_aspect_async":
                # The aspect is extracted first, when nothing requires sentences:
                # they are segmented for the concept of the aspect, attached after
                document.aspects = [
                    Aspect(name="Duties", description="Duties of the parties")
                ]
                await llm.extract_aspects_from_document_async(document)
                assert not document.sentences
                (aspect,) = document.aspects
                aspect.concepts = [create_concept()]
                extraction = llm.extract_concepts_from_aspect_async(aspect, document)
            else:
                document.aspects = [
                    Aspect(
                        name="Duties",
                        description="Duties of the parties",
                        reference_depth="sentences",
                        concepts=[create_concept()],
                    )
                ]
                document.concepts = [create_concept()]
                extraction = getattr(llm, method_name)(document)
            await asyncio.gather(extraction, run_during_segmentation())
            return document

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(
                documents_module,
                "_get_sat_model",
                new=lambda model_id: WaitingSentenceSplitter(),
            ),
        ):
            document = asyncio.run(run())
        assert [len(p.sentences) for p in document.paragraphs] == [2, 1]

    def test_llm_refusal_is_retried_and_falls_back(self):
        """
        Tests that an extraction that the LLM or its provider declines to answer
        is handled as invalid data is: the refused answer is not used, also if it is
        valid JSON, the request is sent again up to ``max_retries_invalid_data``
        times, as the verdict can differ on another request, and then to the fallback
        models. If none of them answers, ``ModelRefusalError``, a subclass of
        ``ModelExtractionError``, is raised, or a warning is logged instead, and each
        refused request is recorded in the usage with its refusal:

        - the ``content_filter`` finish reason, which litellm reports for
          a refusal of a model and for the content filter of a provider;
        - the refusal of OpenAI's ``refusal`` field of the message;
        - a request that the provider blocked on its input, which litellm raises
          as ``ContentPolicyViolationError``. It is handled as a failed request with
          a non-transient error, as before refusals had an error of their own: it is
          not sent again to the LLM, the fallback LLM is tried, and the error of
          litellm is the ``original_error`` of ``ModelRefusalError``.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"

        def answer(value: str) -> str:
            """
            Returns an answer with one item for the concept of the prompt.
            """
            return json.dumps([{"concept_id": "C0", "extracted_items": [value]}])

        def create_completion(
            content: str | None,
            finish_reason: str = "stop",
            refusal: str | None = None,
        ) -> Any:
            """
            Creates a completion as litellm returns it, with the text, the finish
            reason, and OpenAI's refusal field, which litellm keeps in
            the provider-specific fields of the message.
            """
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=content,
                            provider_specific_fields={"refusal": refusal},
                        ),
                        finish_reason=finish_reason,
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        refusals: list[tuple[str, str]] = [
            (
                "content_filter",
                (
                    f"The answer of LLM `{primary_model}` was stopped with "
                    f"the `content_filter` finish reason: the model refused to "
                    f"answer, or the content filter of its provider stopped "
                    f"the answer."
                ),
            ),
            ("refusal", f"LLM `{primary_model}` refused to answer."),
            (
                "input_blocked",
                (
                    f"The provider of LLM `{primary_model}` blocked the request "
                    f"for its content."
                ),
            ),
        ]

        def refused_outcome(refusal: str) -> Any:
            """
            Returns what the LLM API gives for a refusal of the kind: the answer
            stopped with the `content_filter` finish reason is valid JSON, which
            is not to be used.
            """
            if refusal == "content_filter":
                return create_completion(
                    answer("Refused value"), finish_reason="content_filter"
                )
            if refusal == "refusal":
                return create_completion(None, refusal="I can't help with that.")
            return input_blocked_error

        def create_stub(outcomes: list[Any], requested_models: list[str]) -> Any:
            """
            Creates a stub of the LLM API: the main LLM gives the outcomes in turn,
            the last one to each request after them, raised if it is an error, and
            the fallback LLM answers with an item of its own. The models of
            the requests are added to the list given.
            """
            queue = list(outcomes)

            async def acompletion_stub(**kwargs: Any) -> Any:
                """
                Simulates the LLM API.
                """
                requested_models.append(kwargs["model"])
                if kwargs["model"] == fallback_model:
                    return create_completion(answer("Fallback value"))
                outcome = queue.pop(0) if len(queue) > 1 else queue[0]
                if isinstance(outcome, Exception):
                    raise outcome
                return outcome

            return acompletion_stub

        def create_document() -> Document:
            """
            Creates a document with one concept.
            """
            document = Document(raw_text="Acme Ltd and Beta LLC agree as follows.")
            document.concepts = [
                StringConcept(name="Parties", description="Parties of the agreement")
            ]
            return document

        def create_llm(max_retries: int, with_fallback: bool) -> DocumentLLM:
            """
            Creates the main LLM, with a fallback LLM or without.
            """
            return DocumentLLM(
                model=primary_model,
                api_key="test_api_key",
                max_retries_invalid_data=max_retries,
                fallback_models=(
                    [
                        DocumentLLM(
                            model=fallback_model,
                            api_key="test_api_key",
                            is_fallback=True,
                        )
                    ]
                    if with_fallback
                    else []
                ),
            )

        input_blocked_error = litellm.ContentPolicyViolationError(
            message="The request was rejected by the safety system",
            model=primary_model,
            llm_provider="openai",
        )
        for refusal, refusal_sentence in refusals:
            # Whether the refused request is sent again: an answer refused, not
            # a request blocked on its input
            sent_again = refusal != "input_blocked"

            # === A retry that is answered ===
            if sent_again:
                requested_models: list[str] = []
                llm = create_llm(max_retries=2, with_fallback=True)
                document = create_document()
                acompletion_stub = create_stub(
                    [refused_outcome(refusal), create_completion(answer("Acme"))],
                    requested_models,
                )
                with (
                    patch.object(litellm, "acompletion", new=acompletion_stub),
                    capture_logs() as messages,
                ):
                    llm.extract_concepts_from_document(document)
                assert requested_models == [primary_model, primary_model]
                assert [
                    item.value for item in document.concepts[0].extracted_items
                ] == ["Acme"]
                assert "Retrying concepts after a refusal (1/2)" in messages
                first_call, second_call = llm.get_usage()[0].usage.calls
                assert first_call._refusal == refusal
                assert second_call._refusal is None

            # === Refused on each retry, answered by the fallback LLM ===
            requested_models = []
            llm = create_llm(max_retries=2, with_fallback=True)
            document = create_document()
            acompletion_stub = create_stub([refused_outcome(refusal)], requested_models)
            with patch.object(litellm, "acompletion", new=acompletion_stub):
                llm.extract_concepts_from_document(document)
            n_primary_requests = 3 if sent_again else 1
            assert requested_models == (
                [primary_model] * n_primary_requests + [fallback_model]
            )
            # The refused answer is not used, also when it is valid JSON
            assert [item.value for item in document.concepts[0].extracted_items] == [
                "Fallback value"
            ]
            primary_usage, fallback_usage = llm.get_usage()
            assert [call._refusal for call in primary_usage.usage.calls] == [
                refusal
            ] * n_primary_requests
            assert len(fallback_usage.usage.calls) == 1
            # A blocked request has no response, and no tokens
            assert primary_usage.usage.input == (
                0 if refusal == "input_blocked" else 30
            )

            # === Refused on each retry, without a fallback LLM ===
            for max_retries in [1, 0]:
                n_retries = max_retries if sent_again else 0
                expected_msg = (
                    f"{refusal_sentence} Failed concepts: ['Parties']"
                    if sent_again
                    else refusal_sentence
                )
                for raise_exception in [True, False]:
                    requested_models = []
                    llm = create_llm(max_retries=max_retries, with_fallback=False)
                    document = create_document()
                    acompletion_stub = create_stub(
                        [refused_outcome(refusal)], requested_models
                    )
                    with (
                        patch.object(litellm, "acompletion", new=acompletion_stub),
                        patch.object(
                            logger, "warning", wraps=logger.warning
                        ) as log_warning,
                    ):
                        if raise_exception:
                            # Caught as a ModelExtractionError, as before refusals
                            # had an error of their own
                            with pytest.raises(ModelExtractionError) as exc_info:
                                llm.extract_concepts_from_document(document)
                            assert isinstance(exc_info.value, ModelRefusalError)
                            assert exc_info.value.message == expected_msg
                            assert exc_info.value.retry_count == n_retries
                            assert exc_info.value.original_error is (
                                None if sent_again else input_blocked_error
                            )
                        else:
                            llm.extract_concepts_from_document(
                                document, raise_exception_on_extraction_error=False
                            )
                            assert any(
                                str(call.args[0]).startswith(
                                    f"{expected_msg} ({n_retries} retries)."
                                )
                                for call in log_warning.call_args_list
                            ), log_warning.call_args_list
                    assert requested_models == [primary_model] * (1 + n_retries)
                    (concept,) = document.concepts
                    assert not concept._is_processed
                    assert not concept.extracted_items
                    assert [
                        call._refusal for call in llm.get_usage()[0].usage.calls
                    ] == [refusal] * (1 + n_retries)

        # With concurrency, a refusal without a retry or a fallback model stops
        # the extraction as other final errors do, and each request sent is in
        # the usage
        requested_models = []

        async def acompletion_stub_concurrent(**kwargs: Any) -> Any:
            """
            Simulates the LLM API: declines to answer the request for the first
            concept, and answers the others with an item.
            """
            requested_models.append(kwargs["model"])
            if "Parties" in kwargs["messages"][-1]["content"]:
                return create_completion(
                    answer("Refused value"), finish_reason="content_filter"
                )
            return create_completion(answer("Acme"))

        llm = DocumentLLM(
            model=primary_model, api_key="test_api_key", max_retries_invalid_data=0
        )
        document = Document(raw_text="Acme Ltd and Beta LLC agree as follows.")
        document.concepts = [
            StringConcept(name="Parties", description="Parties of the agreement"),
            StringConcept(name="Governing law", description="Law of the agreement"),
        ]
        with (
            patch.object(litellm, "acompletion", new=acompletion_stub_concurrent),
            pytest.raises(ModelRefusalError, match=r"Failed concepts: \['Parties'\]"),
        ):
            llm.extract_concepts_from_document(document, use_concurrency=True)
        assert set(requested_models) == {primary_model}
        assert len(llm.get_usage()[0].usage.calls) == len(requested_models)
        assert not document.concepts[0]._is_processed
