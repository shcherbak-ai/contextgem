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
Module defining tests for the framework.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import types
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional  # noqa: UP035
from unittest.mock import patch

import pytest
from _pytest.nodes import Item as PytestItem
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator

from contextgem.internal.base import llms as llms_module
from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.attrs import (
    _AssignedAspectsProcessor,
    _AssignedConceptsProcessor,
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.base.concepts import (
    _BooleanConcept,
    _Concept,
    _DateConcept,
    _JsonObjectConcept,
    _LabelConcept,
    _NumericalConcept,
    _RatingConcept,
    _StringConcept,
)
from contextgem.internal.base.data_models import _ModelPricing
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.examples import _JsonObjectExample, _StringExample
from contextgem.internal.base.images import _Image
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.base.llms import (
    _DocumentLLM,
    _DocumentModelGroup,
)
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.base.pipelines import _ExtractionPipeline
from contextgem.internal.base.utils import _JsonObjectClassStruct
from contextgem.internal.data_models import (
    _Message,
    _ModelCall,
    _ModelCost,
    _ModelUsage,
)
from contextgem.internal.exceptions import (
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelExtractionError,
)
from contextgem.internal.items import (
    _BooleanItem,
    _DateItem,
    _FloatItem,
    _IntegerItem,
    _IntegerOrFloatItem,
    _JsonObjectItem,
    _LabelItem,
    _StringItem,
)
from contextgem.internal.loggers import (
    LOGGER_LEVEL_ENV_VAR_NAME,
    logger,
)
from contextgem.internal.registry import (
    _publicize,
    _resolve_public_type,
)
from contextgem.internal.utils import (
    _get_llm_cache_token_counts,
    _get_model_api_retry_delay,
    _get_template,
    _is_retryable_model_api_error,
    _load_sat_model,
    _split_text_into_paragraphs,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    ChatSession,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    ExtractionPipeline,
    Image,
    JsonObjectClassStruct,
    JsonObjectConcept,
    JsonObjectExample,
    LabelConcept,
    ModelPricing,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    Sentence,
    StringConcept,
    StringExample,
    register_tool,
    reload_logger_settings,
)
from contextgem.public.utils import create_image, image_to_base64
from tests.benchmark.core import run_benchmark_for_module
from tests.conftest import VCR_REDACTION_MARKER
from tests.memory_profiling import check_locals_memory_usage, memory_profile_and_capture
from tests.utils import (
    VCR_FILTER_HEADERS,
    TestUtils,
    get_project_root_path,
    get_test_document_text,
    get_test_img,
    set_dummy_env_variables_for_testing_from_cassettes,
    vcr_before_record_request,
    vcr_before_record_response,
)


# If .env exists locally, it'll be loaded. If it doesn't exist (e.g. in CI),
# no error is raised. Then fallback to environment variables set by the workflow.

if not load_dotenv():  # Returns False if no .env file is found
    set_dummy_env_variables_for_testing_from_cassettes()


@pytest.fixture(scope="module")
def vcr_config():
    """
    Pytest fixture providing VCR configuration for recording and replaying HTTP interactions.

    This fixture configures VCR to record HTTP requests and responses during test execution,
    allowing tests to replay recorded interactions without making actual network calls.

    :return: Dictionary containing VCR configuration options including header filtering,
             request/response processing hooks, matching criteria, and cassette settings.
    :rtype: dict
    """
    return {
        "filter_headers": [(i, VCR_REDACTION_MARKER) for i in VCR_FILTER_HEADERS],
        "before_record_request": vcr_before_record_request,
        "before_record_response": vcr_before_record_response,
        "match_on": ["method", "host", "path", "body"],
        "record_mode": "once",
        "cassette_library_dir": "tests/cassettes",
        "ignore_localhost": False,
        "ignore_hosts": ["huggingface.co", "hf.co"],
    }


class TestAll(TestUtils):
    """
    Test cases for validating the functionality and error handling of the framework's
    core classes and methods, particularly for document analysis workflows.

    Variables are initialized at a class level to be used in the test methods.
    """

    # Default system messages
    default_system_message_en = _get_template(  # ty: ignore[unresolved-attribute]
        "default_system_message",
        template_type="system",
        template_extension="j2",
    ).render({"output_language": "en"})  # type: ignore[attr-defined]
    default_system_message_non_en = _get_template(  # ty: ignore[unresolved-attribute]
        "default_system_message",
        template_type="system",
        template_extension="j2",
    ).render({"output_language": "adapt"})  # type: ignore[attr-defined]

    # Documents
    # From raw texts
    document = Document(raw_text=get_test_document_text())
    document_ua = Document(raw_text=get_test_document_text(lang="ua"))
    document_zh = Document(raw_text=get_test_document_text(lang="zh"))
    # Serialized document with additional context for each paragraph
    test_doc_with_additional_context_path = os.path.join(
        get_project_root_path(),
        "tests",
        "other_files",
        "doc_with_additional_context.json",
    )

    # Pre-load SaT model to cache it before VCR starts intercepting requests.
    # This prevents VCR's broken async handling from interfering with HuggingFace Hub downloads.
    _sat_model_preload = _load_sat_model("sat-3l-sm")
    del (
        _sat_model_preload
    )  # Free memory, model will be re-loaded from cache when needed

    # Extraction pipeline
    extraction_pipeline = ExtractionPipeline(
        aspects=[
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
            ),
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                add_justifications=True,
                concepts=[
                    StringConcept(
                        name="Liability cap",
                        description="Provisions on liability cap",
                        add_justifications=True,
                    )
                ],
            ),
        ],
        concepts=[
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
                add_justifications=True,
                add_references=True,
            ),
            BooleanConcept(
                name="Invoice number check",
                description="Invoice number is present in the document",
                modality="image",
                add_justifications=True,
            ),
        ],
    )

    # LLMs

    # Extractor (LLM without a role, i.e. the default one in the model group)
    _llm_extractor_kwargs_openai = {
        "model": "openai/gpt-6-luna",
        "api_key": os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        "pricing_details": ModelPricing(  # Explicitly set pricing details
            **{
                "input_per_1m_tokens": 0.10,
                "output_per_1m_tokens": 0.50,
            }
        ),
    }

    # Reasoner (LLM with a role, i.e. the one that the aspects/concepts
    # with the same role are routed to in the model group)
    _llm_reasoner_kwargs_openai = {
        "model": "openai/gpt-5.6-luna",
        "api_key": os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        "role": "reasoner",
        "auto_pricing": True,  # use auto-pricing
        "auto_pricing_refresh": False,  # avoid network auto-refresh during tests
    }

    # Extractor
    llm_extractor = DocumentLLM(**_llm_extractor_kwargs_openai)  # ty: ignore[invalid-argument-type]

    # Reasoner
    llm_reasoner = DocumentLLM(**_llm_reasoner_kwargs_openai)  # ty: ignore[invalid-argument-type]

    # Model group
    _model_group_llms = [
        DocumentLLM(**_llm_extractor_kwargs_openai),  # ty: ignore[invalid-argument-type]
        DocumentLLM(**_llm_reasoner_kwargs_openai),  # ty: ignore[invalid-argument-type]
    ]  # newly initialized LLMs in group for separate usage and cost tracking)
    model_group = DocumentModelGroup(models=_model_group_llms)  # ty: ignore[invalid-argument-type]

    # LLM with fallback
    _invalid_llm_kwargs = {
        "model": "helloworld/helloworld-v1",
        "api_key": "invalid_api_key",
    }
    llm_invalid = DocumentLLM(**_invalid_llm_kwargs)  # ty: ignore[invalid-argument-type]
    invalid_llm_with_valid_fallback = DocumentLLM(**_invalid_llm_kwargs)  # ty: ignore[invalid-argument-type]
    invalid_llm_with_valid_fallback.fallback_models = [
        DocumentLLM(
            **_llm_extractor_kwargs_openai,  # ty: ignore[invalid-argument-type]
            is_fallback=True,
        )
    ]

    # LLM with a non-EN output language setting
    llm_extractor_non_eng = DocumentLLM(
        **_llm_extractor_kwargs_openai,  # ty: ignore[invalid-argument-type]
        output_language="adapt",
    )

    # Images
    # Invoices
    test_img_png_invoice = get_test_img("invoice.png", "invoices")
    test_img_jpg_invoice = get_test_img("invoice.jpg", "invoices")
    test_img_jpg_2_invoice = get_test_img("invoice2.jpg", "invoices")
    test_img_webp_invoice = get_test_img("invoice.webp", "invoices")
    # Other
    test_img_png_apt_plan = get_test_img("apt_plan.png", "other")

    # Document with both text and an image
    document_multimodal = Document(
        raw_text=(
            "Invoices:\n\n#0001 (1 May 2025)\n\n#0002 (20 May 2025)\n\n"
            "#0005 (16 June 2025)"
        ),
        images=[test_img_png_invoice],
    )

    # Memory profiling
    memory_baseline: float = 0.0  # baseline memory usage
    memory_profiles: dict[str, str] = {}  # memory usage profiles
    memory_deltas: dict[str, float] = {}  # memory usage deltas (relative to baseline)

    @pytest.fixture(autouse=True)
    def restore_llms_output_language(self):
        """
        Restores the default output language of the LLMs shared by the tests.

        The tests that process documents in other languages change the output language
        of these LLMs. Restoring it after each test, including a failed one, keeps
        the LLM requests of the other tests independent of the order of the tests.
        """
        yield
        if self.model_group.output_language != "en":
            self.model_group.group_update_output_language("en")
        if self.llm_extractor.output_language != "en":
            self.llm_extractor.output_language = "en"

    @memory_profile_and_capture
    def test_establish_memory_baseline(self):
        """
        Establishes a memory usage baseline for subsequent per-method memory calculations.

        This test captures the initial memory footprint including module imports,
        variable initialization, and framework setup. The memory profile from this
        test serves as a baseline against which memory deltas of other test methods
        can be measured.
        """
        logger.info("Memory usage baseline captured.")

    @memory_profile_and_capture
    def test_prompt_templates(self):
        """
        Tests for content validity and consistency in the prompt templates.
        """
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        prompts_folder_path = project_root / "contextgem" / "internal" / "prompts"
        j2_files_found = False
        for root, _, files in os.walk(prompts_folder_path):
            for file in files:
                if file.endswith(".j2"):
                    j2_files_found = True
                    file_path = os.path.join(root, file)
                    with open(file_path, encoding="utf-8") as f:
                        raw_content = f.read()
                    # For testing, apply the same check as for the rendered prompt
                    self.check_rendered_prompt(raw_content)
        assert j2_files_found, prompts_folder_path

    @memory_profile_and_capture
    def test_registry(self):
        """
        Sanity checks for internal -> public mappings via the registry.
        """

        pub_aspect = _publicize(
            _Aspect,
            name="Test Aspect",
            description="Test Description",
            model_role="reasoner",
            reference_depth="sentences",
        )
        assert isinstance(pub_aspect, Aspect)
        assert pub_aspect.model_role == "reasoner"
        assert pub_aspect.reference_depth == "sentences"
        assert _resolve_public_type(_Aspect) is Aspect

        pub_concept = _publicize(
            _StringConcept,
            name="Test Concept",
            description="Test Description",
            add_references=True,
        )
        assert isinstance(pub_concept, StringConcept)
        assert pub_concept.add_references
        assert _resolve_public_type(_StringConcept) is StringConcept

        pub_doc = _publicize(_Document, raw_text="Hello")
        assert isinstance(pub_doc, Document)
        assert _resolve_public_type(_Document) is Document
        pub_doc.add_aspects([pub_aspect])
        assert all(isinstance(a, Aspect) for a in pub_doc.aspects)
        pub_doc.add_concepts([pub_concept])
        assert all(isinstance(c, StringConcept) for c in pub_doc.concepts)

        check_locals_memory_usage(locals(), test_name="test_registry")

    @memory_profile_and_capture
    def test_attr_models(self):
        """
        Tests for attribute validation models.
        """
        # Usage data model
        _ModelUsage(input=30, output=10)
        _ModelUsage(input=30, output=10, cache_read=20, cache_write=5)
        with pytest.raises(ValueError):
            _ModelUsage(input=-1, output=2)
        with pytest.raises(ValueError):
            _ModelUsage(input=30, output=10, cache_read=-1)
        with pytest.raises(ValueError):
            _ModelUsage(input=30, output=10, cache_write=-1)
        # Pricing data model
        ModelPricing(input_per_1m_tokens=0.00015, output_per_1m_tokens=0.0006)
        ModelPricing(
            input_per_1m_tokens=0.00015,
            output_per_1m_tokens=0.0006,
            cache_read_per_1m_tokens=0.000075,
            cache_write_per_1m_tokens=0.0001875,
        )
        with pytest.raises(ValueError):
            ModelPricing(input_per_1m_tokens=-1, output_per_1m_tokens=0)
        with pytest.raises(ValueError):
            ModelPricing(input_per_1m_tokens=0.1)
        # Cost data model
        _ModelCost(input=Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03"))
        with pytest.raises(ValueError):
            _ModelCost(
                input=-Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03")
            )
        with pytest.raises(ValueError):
            _ModelCost(input=-Decimal("0.001"))

    @memory_profile_and_capture
    def test_init_instance_bases(self):
        """
        Tests for direct initialization of the base classes (internal classes).
        """

        # Base instance classes
        base_instance_classes = [
            _Aspect,
            _StringConcept,
            _BooleanConcept,
            _DateConcept,
            _NumericalConcept,
            _RatingConcept,
            _JsonObjectConcept,
            _LabelConcept,
            _ModelPricing,
            _StringExample,
            _JsonObjectExample,
            _Image,
            _Document,
            _Paragraph,
            _Sentence,
            _ExtractionPipeline,
            _DocumentLLM,
            _DocumentModelGroup,
            _JsonObjectClassStruct,
        ]
        for base_class in base_instance_classes:
            with pytest.raises(
                TypeError, match="internal and cannot be instantiated directly"
            ):
                base_class()

        # Shared attribute classes
        base_attr_classes = [
            _AssignedAspectsProcessor,
            _AssignedConceptsProcessor,
            _AssignedInstancesProcessor,
            _ExtractedItemsAttributeProcessor,
            _RefParasAndSentsAttrituteProcessor,
        ]
        for base_class in base_attr_classes:

            class TestNoRequiredAttrs(base_class):  # ty: ignore[unsupported-base]
                value: str = "Test"  # no required attributes

                @property
                def _item_class(self) -> type:
                    return PytestItem

            with pytest.raises(AttributeError):
                TestNoRequiredAttrs()  # initialized with no required attributes

    @memory_profile_and_capture(
        max_memory=500.0
    )  # higher limit due to multiple SaT models loading
    def test_sat_model_no_cache(self):
        """
        Tests that the SaT model is not cached.
        """
        sat_model_id = "sat-3l-sm"
        assert _load_sat_model(sat_model_id) != _load_sat_model(sat_model_id)
        assert id(_load_sat_model(sat_model_id)) != id(_load_sat_model(sat_model_id))

        check_locals_memory_usage(locals(), test_name="test_sat_model_no_cache")

    @memory_profile_and_capture
    def test_local_sat_model(self):
        """
        Tests the loading of a local SAT model.
        """

        # Test nonexistent path
        with pytest.raises(ValueError) as exc_info:
            non_existent_path = "/nonexistent/path/to/model"
            _load_sat_model(non_existent_path)
            assert "does not exist or is not a directory" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(ValueError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=non_existent_path,
                )

        # Test file path (not a directory)
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError) as exc_info:
                _load_sat_model(temp_file.name)
            assert "does not exist or is not a directory" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(ValueError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=temp_file.name,
                )

        # Test valid path but invalid model
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(RuntimeError) as exc_info:
                _load_sat_model(temp_dir)
            assert "does not contain a valid SaT model" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(RuntimeError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=temp_dir,
                )

        check_locals_memory_usage(locals(), test_name="test_local_sat_model")

    @memory_profile_and_capture
    def test_requires_sentence_segmentation(self):
        """
        Tests the `_requires_sentence_segmentation` method of the Document class.
        """
        # === Direct aspect/concept assignment ===

        document = Document(raw_text="This is a sentence.")

        # Aspects
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    reference_depth="paragraphs",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()
        document.remove_all_aspects()

        # Concepts
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_concepts()
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=False,
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()
        document.remove_all_concepts()
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()

        # Sub-aspects
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    aspects=[
                        Aspect(
                            name="Sub-aspect 1",
                            description="Sub-aspect 1",
                            reference_depth="sentences",
                        ),
                    ],
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    aspects=[
                        Aspect(
                            name="Sub-aspect 1",
                            description="Sub-aspect 1",
                            reference_depth="paragraphs",
                        ),
                    ],
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()

        # === Assignment of aspects/concepts via ExtractionPipeline ===

        document = Document(raw_text="This is a sentence.")

        # Aspects
        extraction_pipeline = ExtractionPipeline(
            aspects=[
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()

        # Concepts
        extraction_pipeline = ExtractionPipeline(
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert document._requires_sentence_segmentation()

        check_locals_memory_usage(
            locals(), test_name="test_requires_sentence_segmentation"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture(
        max_memory=500.0
    )  # higher limit due to multiple SaT models loading
    def test_sat_model_deferred_segmentation(self):
        """
        Tests for the SaT model deferred segmentation.
        """

        # Trigger based on param
        document = Document(
            raw_text=get_test_document_text(),
            pre_segment_sentences=True,
        )
        assert document.paragraphs
        assert document.sentences

        # Trigger manually
        document = Document(raw_text=get_test_document_text())
        assert document.paragraphs
        # By default, sentences are not segmented.
        assert not document.sentences
        document._segment_sents()  # called when sentence-level refs are needed for aspects/concepts
        assert document.sentences

        # Selective segmentation of paragraphs
        document = Document(
            paragraphs=[
                Paragraph(
                    raw_text="This is sentence 1. This is sentence 2.",
                    sentences=[
                        Sentence(raw_text="This is sentence 1."),
                        Sentence(raw_text="This is sentence 2."),
                    ],
                ),
                Paragraph(
                    raw_text="This is a short sentence. And this is a bit longer sentence."
                ),  # this paragraph will be segmented
            ],
            pre_segment_sentences=True,
        )
        assert len(document.paragraphs[1].sentences) == 2
        assert len(document.sentences) == 4

        # === Segmentation is triggered when LLM extraction method is called for aspects/concepts
        # that require sentence segmentation ===

        # Some aspects require sentence segmentation (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            aspects=[
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ],
        )
        assert not document.sentences

        # Some concepts require sentence segmentation (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        assert not document.sentences

        # Some sub-aspects require sentence segmentation (assignment during Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.aspects = [
            Aspect(
                name="Liability",
                description="Liability",
                aspects=[
                    Aspect(
                        name="Liability cap",
                        description="Total liability cap",
                        reference_depth="sentences",
                    ),
                ],
            ),
        ]
        assert not document.sentences

        # Mixed - some aspects and concepts require sentence segmentation, some do not
        # (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            aspects=[
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    reference_depth="paragraphs",
                ),
                Aspect(
                    name="Aspect 2", description="Aspect 2", reference_depth="sentences"
                ),
            ],
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
                StringConcept(
                    name="Concept 2",
                    description="Concept 2",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        assert not document.sentences

        # Some aspects require sentence segmentation (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.aspects = [
            Aspect(
                name="Liability", description="Liability", reference_depth="sentences"
            ),
        ]
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_aspects_from_document(document)
        assert document.sentences

        # Some concepts require sentence segmentation (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.concepts = [
            StringConcept(
                name="Liability cap",
                description="Liability cap",
                add_references=True,
                reference_depth="sentences",
            ),
        ]
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_concepts_from_document(document)

        # Some sub-aspects require sentence segmentation (assignment during Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Liability",
                    aspects=[
                        Aspect(
                            name="Liability cap",
                            description="Total liability cap",
                            reference_depth="sentences",
                        ),
                    ],
                ),
            ]
        )
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_aspects_from_document(document)
        assert document.sentences

        # Mixed - some aspects and concepts require sentence segmentation, some do not
        # (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Confidentiality",
                    description="Confidentiality",
                    reference_depth="paragraphs",
                ),
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ]
        )
        document.add_concepts(
            [
                StringConcept(
                    name="Confidential information",
                    description="Confidential information",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_all(document)
        assert document.sentences

        # Test "from_aspects" and "from_concepts" methods
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document.sentences
        self.llm_extractor.extract_aspects_from_document(
            document, from_aspects=[document.aspects[0]]
        )
        assert document.sentences
        document = Document(raw_text=get_test_document_text())
        document.add_concepts(
            [
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                )
            ]
        )
        assert not document.sentences
        self.llm_extractor.extract_concepts_from_document(
            document, from_concepts=[document.concepts[0]]
        )
        assert document.sentences

        # === Assignment via ExtractionPipeline ===

        document = Document(raw_text=get_test_document_text())
        extraction_pipeline = ExtractionPipeline(
            aspects=[
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ],
            concepts=[
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert not document.sentences
        self.llm_extractor.extract_all(document)
        assert document.sentences

        check_locals_memory_usage(
            locals(), test_name="test_sat_model_deferred_segmentation"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_local_llms_text(self):
        """
        Tests for initialization of and getting a response from local LLMs,
        e.g. models run on Ollama local server. Limited to text processing.

        Note that using `ollama_chat/` (preferred in live mode) instead of
        `ollama/` prefix for Ollama models does not work with VCR.
        """

        # Ollama
        # Non-reasoning LLM
        llm_non_reasoning_ollama = DocumentLLM(
            model="ollama/mistral-small:24b",
            api_base="http://localhost:11434",
            seed=123,
        )
        self.extract_with_local_llm(llm_non_reasoning_ollama)
        # Reasoning (CoT-capable) LLM
        llm_reasoning_ollama = DocumentLLM(
            model="ollama/deepseek-r1:32b",
            api_base="http://localhost:11434",
            seed=123,
        )
        llm_reasoning_ollama._supports_reasoning = True
        self.extract_with_local_llm(llm_reasoning_ollama)

        # LM Studio
        # Non-reasoning LLM
        llm_non_reasoning_lm_studio = DocumentLLM(
            model="lm_studio/mistralai/mistral-small-3.2",
            api_base="http://localhost:1234/v1",
            api_key="random-key",  # required for LM Studio API
            seed=123,
        )
        self.extract_with_local_llm(llm_non_reasoning_lm_studio)

        check_locals_memory_usage(locals(), test_name="test_local_llms_text")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_local_llms_vision(self):
        """
        Tests for initialization of and getting a response from local LLMs
        with vision capabilities.
        """
        document_concepts = [
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                modality="image",
                add_justifications=True,
            ),
        ]
        document = Document(
            images=[self.test_img_png_invoice], concepts=document_concepts
        )
        # Warn about using ollama/ prefix when images are sent to a local model
        llm_ollama_chat = DocumentLLM(
            model="ollama_chat/gemma3:27b",
            api_base="http://localhost:11434",
        )
        llm_ollama_chat._supports_vision = True
        with pytest.warns(UserWarning, match="use `ollama/` prefix instead"):
            llm_ollama_chat._ensure_vision_support_for_messages(
                [
                    llm_ollama_chat._build_user_message(
                        message_text="Test", images=[self.test_img_png_invoice]
                    )
                ]
            )

        # Ollama
        llm_ollama = DocumentLLM(
            model="ollama/gemma3:27b",
            api_base="http://localhost:11434",
        )
        # The model is not detected as vision-capable, i.e. the extraction from images
        # fails until the capability is declared explicitly
        # TODO: remove this check once litellm supports vision for this model
        with pytest.raises(ValueError, match="does not support vision"):
            llm_ollama.extract_concepts_from_document(document)
        llm_ollama._supports_vision = True
        extracted_concepts = llm_ollama.extract_concepts_from_document(document)
        assert extracted_concepts[0].extracted_items
        # Log the extracted item for debugging
        self.log_extracted_items_for_instance(extracted_concepts[0])

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm_ollama)

        check_locals_memory_usage(locals(), test_name="test_local_llms_vision")

    @pytest.mark.vcr
    @memory_profile_and_capture
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

        check_locals_memory_usage(
            locals(), test_name="test_model_extraction_error_exception"
        )

    @memory_profile_and_capture
    def test_init_api_llm(self):
        """
        Tests the behaviour of the `DocumentLLM` class initialization.
        """
        DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )

        # Optional params
        DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            top_p=None,
            temperature=None,
        )

        # Default: temperature is set, top_p is not (Claude 4.x rejects both)
        llm_defaults = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert llm_defaults.temperature == 0.3
        assert llm_defaults.top_p is None

        # Reasoning (CoT-capable) model: the sampling params are not sent,
        # and the output is limited by `max_completion_tokens`
        assert llm_defaults._supports_reasoning
        default_request_config = llm_defaults._build_request_config()
        assert default_request_config["max_completion_tokens"] == 16000
        for param in ["temperature", "top_p", "max_tokens"]:
            assert param not in default_request_config

        # Model without reasoning support: temperature is sent, top_p is omitted
        # unless set, and the output is limited by `max_tokens`
        non_reasoning_model = "cerebras/llama-3.3-70b"
        llm_non_reasoning = DocumentLLM(
            model=non_reasoning_model, api_key="test_api_key"
        )
        assert not llm_non_reasoning._supports_reasoning
        non_reasoning_request_config = llm_non_reasoning._build_request_config()
        assert non_reasoning_request_config["temperature"] == 0.3
        assert non_reasoning_request_config["max_tokens"] == 4096
        for param in ["top_p", "max_completion_tokens"]:
            assert param not in non_reasoning_request_config

        llm_with_top_p = DocumentLLM(
            model=non_reasoning_model, api_key="test_api_key", top_p=0.3
        )
        assert llm_with_top_p._build_request_config()["top_p"] == 0.3

        # Unsupported methods
        with pytest.raises(NotImplementedError):
            self.model_group.model_dump()
        with pytest.raises(NotImplementedError):
            self.model_group.model_dump_json()
        with pytest.raises(NotImplementedError):
            self.llm_extractor.model_dump()
        with pytest.raises(NotImplementedError):
            self.llm_extractor.model_dump_json()

        # Invalid params
        with pytest.raises(ValueError):
            DocumentLLM()
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )

        # Fallback LLM validation
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                fallback_models=[
                    self.llm_reasoner
                ],  # inconsistent role of the fallback LLM
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                fallback_models=[
                    DocumentLLM(
                        model="openai/gpt-6-luna",
                        api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                        is_fallback=True,
                    )
                ],  # same params as the main LLM, except for `is_fallback` flag
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                is_fallback=True,
                fallback_models=[
                    self.llm_extractor
                ],  # a fallback LLM is passed to fallback LLM
            )
        # Validate fallback LLMs assignment
        document_llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert document_llm.is_group is False
        assert document_llm.fallback_models == []
        document_llm.is_fallback = True
        with pytest.raises(ValueError):
            document_llm.fallback_models = [
                self.llm_extractor
            ]  # a fallback LLM is passed to fallback LLM
        assert document_llm.fallback_models == []  # No change due to invalid assignment
        document_llm.is_fallback = False
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                is_fallback=False,
                fallback_models=[document_llm],  # non-fallback LLM passed as a fallback
            )

        with pytest.raises(ValueError):
            DocumentLLM(
                model="gpt-6-luna",  # no provider prefix in model name
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="",  # invalid role (empty label)
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                temperature=-1.0,  # negative temperature
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                max_tokens=-4000,  # negative value
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    **{  # invalid pricing details keys
                        "in": 0.1,
                        "out": 0.2,
                    }
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    **{  # invalid pricing details values
                        "input_per_1m_tokens": -0.00015,
                        "output_per_1m_tokens": 0.0006,
                    }
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key="",  # empty api key
            )

        # Warnings
        with pytest.warns(
            UserWarning, match="`output_language` parameter will take no effect"
        ):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="",
            )
        with pytest.warns(
            UserWarning, match="`output_language` parameter will take no effect"
        ):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="You are a helpful assistant.",
            )

        check_locals_memory_usage(locals(), test_name="test_init_api_llm")

    @memory_profile_and_capture
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

        check_locals_memory_usage(locals(), test_name="test_init_model_group")

    @memory_profile_and_capture
    def test_pre_v1_names_rejected(self):
        """
        Tests that the names of the classes and parameters that were renamed in v1.0
        are rejected, i.e. not silently ignored, both in the code and in the
        serialized data.
        """

        import contextgem
        from contextgem.internal import exceptions

        # Renamed classes are not available under the old names
        for module, old_name in [
            (contextgem, "DocumentLLMGroup"),
            (contextgem, "LLMPricing"),
            (exceptions, "LLMAPIError"),
            (exceptions, "LLMExtractionError"),
        ]:
            assert not hasattr(module, old_name)

        # Renamed params
        with pytest.raises(ValueError, match="llm_role\n"):
            Aspect(
                name="Liability",
                description="Liability clauses",
                llm_role="reasoner",  # ty: ignore[unknown-argument]
            )
        with pytest.raises(ValueError, match="llm_role\n"):
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="reasoner",  # ty: ignore[unknown-argument]
            )
        with pytest.raises(ValueError, match="llms\n"):
            DocumentModelGroup(
                llms=[self.llm_extractor, self.llm_reasoner],  # ty: ignore[unknown-argument]
            )
        model_group = DocumentModelGroup(models=[self.llm_extractor, self.llm_reasoner])
        for method in [
            model_group.get_usage,
            model_group.get_cost,
            model_group.reset_usage_and_cost,
        ]:
            with pytest.raises(TypeError, match="llm_role"):
                method(llm_role=self.llm_reasoner.role)  # ty: ignore[unknown-argument]

        # Renamed keys and class names in the serialized data
        concept = StringConcept(
            name="Title", description="Contract title", model_role="reasoner"
        )
        concept_dict = concept.to_dict()
        concept_dict["llm_role"] = concept_dict.pop("model_role")
        with pytest.raises(ValueError, match="llm_role\n"):
            StringConcept.from_dict(concept_dict)
        model_group_dict = model_group.to_dict()
        model_group_dict["llms"] = model_group_dict.pop("models")
        with pytest.raises(ValueError, match="llms\n"):
            DocumentModelGroup.from_dict(model_group_dict)
        model_group_dict = model_group.to_dict()
        model_group_dict["__class__"] = "DocumentLLMGroup"
        with pytest.raises(TypeError, match="DocumentLLMGroup"):
            DocumentModelGroup.from_dict(model_group_dict)
        pricing_dict = ModelPricing(
            input_per_1m_tokens=0.1, output_per_1m_tokens=0.5
        ).to_dict()
        pricing_dict["__class__"] = "LLMPricing"
        with pytest.raises(TypeError, match="LLMPricing"):
            ModelPricing.from_dict(pricing_dict)

        check_locals_memory_usage(locals(), test_name="test_pre_v1_names_rejected")

    @memory_profile_and_capture
    def test_rejected_assignment_keeps_instance_unchanged(self):
        """
        Tests that an instance is left unchanged when a value assigned to its param
        is rejected by the validation that involves the other params of the instance,
        i.e. the rejected value is not kept.

        No LLM API calls are made, i.e. no cassette is required.
        """

        def check_rejected(
            instance: Any, param: str, value: Any, error_match: str
        ) -> None:
            """
            Checks that the value is rejected for the param of the instance,
            and that the params of the instance are the same as before.
            """
            params_before = {
                name: getattr(instance, name) for name in type(instance).model_fields
            }
            params_set_before = set(instance.model_fields_set)
            with pytest.raises(ValueError, match=error_match):
                setattr(instance, param, value)
            for name, value_before in params_before.items():
                assert getattr(instance, name) is value_before
            assert instance.model_fields_set == params_set_before

        def create_llm(model: str = "openai/gpt-6-luna", **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with the given params.
            """
            return DocumentLLM(model=model, api_key="test_api_key", **kwargs)

        def create_llm_with_fallback(**kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with a fallback LLM, both with the given params.
            """
            return create_llm(
                fallback_models=[
                    create_llm("openai/gpt-6.1-sol", is_fallback=True, **kwargs)
                ],
                **kwargs,
            )

        @register_tool
        def rollback_test_tool_1(order_id: str) -> str:
            """
            Returns the status of the order.
            """
            return f"Status of {order_id}"

        @register_tool
        def rollback_test_tool_2(order_id: str) -> str:
            """
            Returns the total of the order.
            """
            return f"Total of {order_id}"

        # === LLM ===
        check_rejected(
            create_llm_with_fallback(role="extractor"),
            "role",
            "reasoner",
            "must have the same role",
        )
        check_rejected(
            create_llm_with_fallback(),
            "output_language",
            "adapt",
            "must have the same output language",
        )
        check_rejected(
            create_llm_with_fallback(),
            "is_fallback",
            True,
            "cannot have its own fallback LLMs",
        )
        check_rejected(
            create_llm(
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.1, output_per_1m_tokens=0.5
                )
            ),
            "auto_pricing",
            True,
            "cannot be set together with `auto_pricing=True`",
        )
        llm = create_llm_with_fallback()
        fallback_llm = llm.fallback_models[0]
        check_rejected(
            llm,
            "fallback_models",
            [fallback_llm, create_llm(is_fallback=True)],
            "same config params as the main LLM",
        )
        assert llm.fallback_models == [fallback_llm]
        # The rejected fallback LLMs are not taken for the validated ones,
        # i.e. are rejected again
        check_rejected(
            llm,
            "fallback_models",
            [fallback_llm, create_llm(is_fallback=True)],
            "same config params as the main LLM",
        )
        # The LLM remains valid, i.e. its other params can be assigned
        llm.temperature = 0.5
        llm.fallback_models = []
        assert llm.fallback_models == []

        # The state derived from the params is not changed either
        llm_with_tools = create_llm_with_fallback(tools=[rollback_test_tool_1])
        other_tools = create_llm(tools=[rollback_test_tool_2]).tools
        assert list(llm_with_tools._tool_registry) == ["rollback_test_tool_1"]
        check_rejected(
            llm_with_tools,
            "tools",
            other_tools,
            "must have the same `tools` configuration",
        )
        assert list(llm_with_tools._tool_registry) == ["rollback_test_tool_1"]

        # === Model group ===
        check_rejected(
            DocumentModelGroup(
                models=[create_llm(), create_llm("openai/gpt-6.1-sol", role="reasoner")]
            ),
            "output_language",
            "adapt",
            "must have the same value of `output_language`",
        )

        # === Concept ===
        json_object_concept = JsonObjectConcept(
            name="Party",
            description="Party details",
            structure={"name": str},
            examples=[JsonObjectExample(content={"name": "Acme Inc."})],
        )
        check_rejected(
            json_object_concept,
            "examples",
            [JsonObjectExample(content={"name": 1})],
            "Invalid JSON object structure",
        )
        check_rejected(
            json_object_concept,
            "structure",
            {"name": int},
            "Invalid JSON object structure",
        )
        # The concept validates the extracted values against the structure it has
        assert json_object_concept._process_item_value({"name": "Other Inc."}) == {
            "name": "Other Inc."
        }

        # === Paragraph ===
        check_rejected(
            Paragraph(raw_text="The liability of the parties is capped."),
            "sentences",
            [Sentence(raw_text="This sentence is not in the paragraph.")],
            "Not all sentences were matched",
        )

        # === Aspect ===
        sub_aspect = Aspect(name="Liability caps", description="Liability caps")
        aspect = Aspect(
            name="Liability", description="Liability clauses", aspects=[sub_aspect]
        )
        check_rejected(
            aspect,
            "aspects",
            [sub_aspect, Aspect(name="Liability", description="Liability clauses")],
            "cannot have the same name as parent aspect",
        )
        assert [i.name for i in aspect.aspects] == ["Liability caps"]

        check_locals_memory_usage(
            locals(), test_name="test_rejected_assignment_keeps_instance_unchanged"
        )

    @memory_profile_and_capture
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

        check_locals_memory_usage(
            locals(), test_name="test_model_roles_and_concept_modality"
        )

    @pytest.mark.parametrize("use_concurrency", [False, True])
    @memory_profile_and_capture
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

        check_locals_memory_usage(
            locals(), test_name="test_nested_extraction_role_preflight"
        )

    @memory_profile_and_capture
    def test_model_group_usage_and_cost_by_role(self):
        """
        Tests for retrieving and resetting the usage and cost of the LLMs in a group
        by the model role, which also applies to the fallback LLMs of the LLM with such role.
        """

        fallback = DocumentLLM(
            model="openai/gpt-5.6-luna",
            api_key="test_api_key",
            role="named",
            is_fallback=True,
        )
        primary = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key="test_api_key",
            role="named",
            fallback_models=[fallback],
        )
        llm_no_role = DocumentLLM(model="openai/gpt-6.1-sol", api_key="test_api_key")
        group = DocumentModelGroup(models=[primary, llm_no_role])
        for llm in [primary, fallback, llm_no_role]:
            llm._usage = _ModelUsage(input=10, output=5)
            llm._cost = _ModelCost(input=Decimal("1"), total=Decimal("1"))

        invalid_roles: list[Any] = ["", "  ", 1, "line\nbreak", "r" * 65]
        for getter in [group.get_usage, group.get_cost]:
            # The LLM with the role is selected together with its fallback LLMs
            selected = getter(model_role="named")
            assert [i.model for i in selected] == [primary.model, fallback.model]
            assert [i.is_fallback for i in selected] == [False, True]
            # All LLMs are selected if no role is specified
            assert [i.role for i in getter()] == ["named", "named", None]
            assert len(getter(model_role=None)) == 3
            with pytest.raises(ValueError, match="`missing`"):
                getter(model_role="missing")
            # The role is validated in the same way as the roles assigned to LLMs
            assert len(getter(model_role=" named ")) == 2
            for invalid_role in invalid_roles:
                with pytest.raises(ValueError, match="Model role must"):
                    getter(model_role=invalid_role)

        for invalid_role in invalid_roles:
            with pytest.raises(ValueError, match="Model role must"):
                group.reset_usage_and_cost(model_role=invalid_role)
        assert all(i.usage.input == 10 for i in group.get_usage())  # no change
        group.reset_usage_and_cost(model_role=" named ")
        assert [i.usage.input for i in group.get_usage()] == [0, 0, 10]
        assert [i.cost.total for i in group.get_cost()] == [0, 0, 1]
        with pytest.raises(ValueError, match="`missing`"):
            group.reset_usage_and_cost(model_role="missing")
        group.reset_usage_and_cost()
        assert all(i.usage.input == 0 for i in group.get_usage())
        assert all(i.cost.total == 0 for i in group.get_cost())

        check_locals_memory_usage(
            locals(), test_name="test_model_group_usage_and_cost_by_role"
        )

    @memory_profile_and_capture
    def test_cached_tokens_usage_and_cost(self):
        """
        Tests for tracking the input tokens read from and written to the prompt cache
        of the LLM provider, and for pricing such tokens when calculating the costs.
        """

        litellm = llms_module.litellm
        answer = '[{"concept_id": "C0", "extracted_items": ["Item"]}]'

        # === Cache token counts reported by litellm ===
        usage_cases: list[tuple[dict[str, Any], tuple[int, int]]] = [
            # No cache tokens reported
            ({}, (0, 0)),
            ({"prompt_tokens_details": {"cached_tokens": None}}, (0, 0)),
            # As reported e.g. for OpenAI, Gemini
            ({"prompt_tokens_details": {"cached_tokens": 1500}}, (1500, 0)),
            # As reported e.g. for Anthropic, Bedrock
            ({"cache_read_input_tokens": 1200}, (1200, 0)),
            ({"cache_creation_input_tokens": 500}, (0, 500)),
            (
                {"cache_read_input_tokens": 1200, "cache_creation_input_tokens": 500},
                (1200, 500),
            ),
            # As reported e.g. for DeepSeek
            ({"prompt_cache_hit_tokens": 800}, (800, 0)),
            # Cache tokens are a part of the prompt tokens, i.e. cannot exceed them
            ({"prompt_tokens_details": {"cached_tokens": 3000}}, (2000, 0)),
            (
                {"cache_read_input_tokens": 1500, "cache_creation_input_tokens": 900},
                (1500, 500),
            ),
        ]
        for usage_kwargs, expected_counts in usage_cases:
            completion_usage = litellm.Usage(
                prompt_tokens=2000, completion_tokens=100, **usage_kwargs
            )
            assert _get_llm_cache_token_counts(completion_usage) == expected_counts
        # Usage data without the prompt token details
        assert _get_llm_cache_token_counts(
            types.SimpleNamespace(prompt_tokens=2000, completion_tokens=100)
        ) == (0, 0)

        # === Usage and cost of an LLM ===
        completion_usages: list[Any] = []  # usage data of the upcoming completions

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, with the usage data of each completion
            taken from the queue.
            """
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content=answer))
                ],
                usage=completion_usages.pop(0),
            )

        def create_llm(model: str = "openai/gpt-6-luna", **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with the specified pricing params.
            """
            return DocumentLLM(
                model=model, api_key="test_api_key", system_message="", **kwargs
            )

        def queue_completion_usages() -> None:
            """
            Queues the usage data of two completions: the first one writes
            a part of the prompt to the cache, the second one reads it from the cache.
            """
            completion_usages[:] = [
                litellm.Usage(
                    prompt_tokens=200_000,
                    completion_tokens=10_000,
                    cache_creation_input_tokens=150_000,
                ),
                litellm.Usage(
                    prompt_tokens=200_000,
                    completion_tokens=10_000,
                    prompt_tokens_details={"cached_tokens": 150_000},
                ),
            ]

        def calc_auto_prices(
            model: str, **usage_kwargs: int
        ) -> tuple[Decimal, Decimal]:
            """
            Calculates the input and output prices of the usage with genai-prices.
            """
            provider_id, model_ref = model.split("/")
            price = llms_module._calc_auto_price(
                llms_module._GPUsage(**usage_kwargs),
                model_ref=model_ref,
                provider_id=provider_id,
            )
            return price.input_price, price.output_price

        def check_usage(llm: DocumentLLM) -> None:
            """
            Checks the usage of the LLM after the two queued completions.
            """
            assert not completion_usages  # both completions are received
            usage = llm.get_usage()[0].usage
            assert (usage.input, usage.output) == (400_000, 20_000)
            assert (usage.cache_read, usage.cache_write) == (150_000, 150_000)
            assert len(usage.calls) == 2

        document = Document(
            raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
        )
        document.concepts = [StringConcept(name="Title", description="Contract title")]

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # Cache tokens are priced at the cache prices, if set
            for pricing_kwargs, expected_input_cost in [
                # 100K regular * 1.0 + 150K read * 0.1 + 150K written * 1.25
                (
                    {
                        "cache_read_per_1m_tokens": 0.1,
                        "cache_write_per_1m_tokens": 1.25,
                    },
                    Decimal("0.3025"),
                ),
                # 250K regular and written * 1.0 + 150K read * 0.1
                ({"cache_read_per_1m_tokens": 0.1}, Decimal("0.265")),
                # 250K regular and read * 1.0 + 150K written * 1.25
                ({"cache_write_per_1m_tokens": 1.25}, Decimal("0.4375")),
                # 150K read * 0.0 (cache tokens can be free of charge)
                ({"cache_read_per_1m_tokens": 0.0}, Decimal("0.25")),
                # ...and as regular input tokens otherwise: 400K * 1.0
                ({}, Decimal("0.4")),
            ]:
                llm = create_llm(
                    pricing_details=ModelPricing(
                        input_per_1m_tokens=1.0,
                        output_per_1m_tokens=4.0,
                        **pricing_kwargs,
                    )
                )
                # Each paragraph is processed in a separate request,
                # with the usage of the requests merged
                queue_completion_usages()
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=1,
                )
                check_usage(llm)
                cost = llm.get_cost()[0].cost
                assert cost.input == expected_input_cost
                assert cost.output == Decimal("0.08")  # 20K * 4.0
                assert cost.total == cost.input + cost.output
                self._check_deserialized_llm_config_eq(llm)

            # The same applies to chat
            llm = create_llm(
                pricing_details=ModelPricing(
                    input_per_1m_tokens=1.0,
                    output_per_1m_tokens=4.0,
                    cache_read_per_1m_tokens=0.1,
                    cache_write_per_1m_tokens=1.25,
                )
            )
            queue_completion_usages()
            llm.chat("Test")
            llm.chat("Test")
            check_usage(llm)
            assert llm.get_cost()[0].cost.input == Decimal("0.3025")

            llm.reset_usage_and_cost()
            usage = llm.get_usage()[0].usage
            assert (usage.cache_read, usage.cache_write) == (0, 0)

            # Tokens are counted whether or not pricing is configured
            llm = create_llm()
            queue_completion_usages()
            llm.extract_concepts_from_document(
                document, overwrite_existing=True, max_paragraphs_to_analyze_per_call=1
            )
            check_usage(llm)
            cost = llm.get_cost()[0].cost
            assert (cost.input, cost.output, cost.total) == (0, 0, 0)

            # Auto-pricing: cache tokens are priced at the cache prices of the model.
            # Each request is priced on its own, also when the usage of the requests
            # is merged, as the price tier of a model with tiered prices depends
            # on the number of input tokens in a request.
            for model, has_price_tiers in [
                ("anthropic/claude-haiku-4-5", False),
                ("openai/gpt-6-luna", True),  # higher prices above 272K
            ]:
                with pytest.warns(
                    UserWarning, match="prices will not be 100% accurate."
                ):
                    llm = create_llm(model, auto_pricing=True)
                queue_completion_usages()
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=1,
                )
                check_usage(llm)
                request_prices = [
                    calc_auto_prices(
                        model,
                        input_tokens=200_000,
                        output_tokens=10_000,
                        **cache_tokens,
                    )
                    for cache_tokens in [
                        {"cache_write_tokens": 150_000},
                        {"cache_read_tokens": 150_000},
                    ]
                ]
                cost = llm.get_cost()[0].cost
                assert cost.input == sum(i[0] for i in request_prices)
                assert cost.output == sum(i[1] for i in request_prices)
                # ...which differs from the price of the same tokens without the cache
                assert (
                    cost.input
                    != 2
                    * calc_auto_prices(
                        model, input_tokens=200_000, output_tokens=10_000
                    )[0]
                )
                # ...and from the price of the merged usage, if it gets to another tier
                merged_usage_prices = calc_auto_prices(
                    model,
                    input_tokens=400_000,
                    output_tokens=20_000,
                    cache_read_tokens=150_000,
                    cache_write_tokens=150_000,
                )
                assert (
                    merged_usage_prices != (cost.input, cost.output)
                ) == has_price_tiers

        # === Missing pricing data ===
        # The warning suggests refreshing the pricing data, as the bundled data
        # may not include a recently released model
        with pytest.warns(UserWarning, match="auto_pricing_refresh=True"):
            llm = create_llm("openai/gptXX", auto_pricing=True)
        # ...unless the refresh is already enabled
        llm.auto_pricing_refresh = True
        assert "auto_pricing_refresh" not in llm._get_missing_auto_pricing_advice()

        # === Pricing model ===
        invalid_pricing_cases: list[dict[str, Any]] = [
            {"cache_read_per_1m_tokens": -0.1},
            {"cache_write_per_1m_tokens": -0.1},
            {"cache_read_per_1m_tokens": "0.1"},
        ]
        for invalid_pricing_kwargs in invalid_pricing_cases:
            with pytest.raises(ValueError):
                ModelPricing(
                    input_per_1m_tokens=1.0,
                    output_per_1m_tokens=4.0,
                    **invalid_pricing_kwargs,
                )
        # Pricing serialized without the cache prices can still be deserialized
        pricing = ModelPricing(input_per_1m_tokens=1.0, output_per_1m_tokens=4.0)
        pricing_dict = pricing.to_dict()
        for key in ["cache_read_per_1m_tokens", "cache_write_per_1m_tokens"]:
            assert pricing_dict.pop(key) is None
        assert ModelPricing.from_dict(pricing_dict) == pricing

        check_locals_memory_usage(
            locals(), test_name="test_cached_tokens_usage_and_cost"
        )

    @memory_profile_and_capture
    def test_update_default_prompt(self):
        """
        Tests for updating the default prompt for the LLM.
        """
        # Test custom prompts assignment
        llm_with_updated_prompts = DocumentLLM(
            **self._llm_extractor_kwargs_openai,  # ty: ignore[invalid-argument-type]
        )
        project_root = get_project_root_path()
        prompt_aspects_without_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_aspects_no_tags.j2"
        )
        prompt_concepts_without_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_concepts_no_tags.j2"
        )
        prompt_aspects_with_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_aspects_with_tags.j2"
        )
        prompt_concepts_with_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_concepts_with_tags.j2"
        )
        with pytest.raises(ValueError, match="no Jinja2 tags"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_aspects_without_tags_path,
                prompt_type="aspect",  # type: ignore
            )
        with pytest.raises(ValueError, match="no Jinja2 tags"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_concepts_without_tags_path,
                prompt_type="concept",  # type: ignore
            )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_aspects_with_tags_path,
            prompt_type="aspect",  # type: ignore
        )
        assert (
            "test custom prompt template for aspects"
            in llm_with_updated_prompts._extract_aspect_items_prompt.render()
        )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_concepts_with_tags_path,
            prompt_type="concept",  # type: ignore
        )
        assert (
            "test custom prompt template for concepts"
            in llm_with_updated_prompts._extract_concept_items_prompt.render()
        )

        check_locals_memory_usage(locals(), test_name="test_update_default_prompt")

    @pytest.mark.parametrize(
        "image", [test_img_png_invoice, test_img_jpg_invoice, test_img_webp_invoice]
    )
    @memory_profile_and_capture
    def test_init_and_attach_image(self, image: Image):
        """
        Tests for constructing a Image instance and attaching it to a Document instance.
        """

        with pytest.raises(ValueError):
            Image()
        with pytest.raises(ValueError):
            Image(
                mime_type="image/weoighwe",  # invalid MIME type  # type: ignore
                base64_data="base 64 encoded string",
            )
        with pytest.raises(ValueError):
            Image(
                mime_type="image/png",
                base64_data="base 64 encoded string",
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        document = Document(images=[image])
        with pytest.raises(ValueError):
            Document(images=[image, image])  # duplicate images with same base64 string

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(document)

        check_locals_memory_usage(locals(), test_name="test_init_and_attach_image")

    @pytest.mark.parametrize(
        "test_image_path",
        [
            "tests/images/invoices/invoice.png",
            "tests/images/invoices/invoice.jpg",
            "tests/images/invoices/invoice.webp",
        ],
    )
    @memory_profile_and_capture
    def test_create_image(self, test_image_path: str):
        """
        Tests for the `create_image` utility function with various input types.
        """
        project_root = get_project_root_path()
        full_image_path = project_root / test_image_path

        # Ensure test image exists
        assert full_image_path.exists(), f"Test image not found: {full_image_path}"

        # Test 1: Create Image from file path (string)
        img_from_str_path = create_image(str(full_image_path))
        assert isinstance(img_from_str_path, Image)
        assert img_from_str_path.base64_data
        assert img_from_str_path.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 2: Create Image from file path (Path object)
        img_from_path_obj = create_image(full_image_path)
        assert isinstance(img_from_path_obj, Image)
        assert img_from_path_obj.base64_data
        assert img_from_path_obj.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 3: Create Image from PIL Image object
        pil_image = PILImage.open(full_image_path)
        img_from_pil = create_image(pil_image)
        assert isinstance(img_from_pil, Image)
        assert img_from_pil.base64_data
        assert img_from_pil.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 4: Create Image from file handle
        with open(full_image_path, "rb") as f:
            img_from_file_handle = create_image(f)
            assert isinstance(img_from_file_handle, Image)
            assert img_from_file_handle.base64_data
            assert img_from_file_handle.mime_type in [
                "image/png",
                "image/jpeg",
                "image/webp",
            ]

        # Test 5: Create Image from raw bytes
        with open(full_image_path, "rb") as f:
            image_bytes = f.read()
        img_from_bytes = create_image(image_bytes)
        assert isinstance(img_from_bytes, Image)
        assert img_from_bytes.base64_data
        assert img_from_bytes.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 6: Create Image from BytesIO
        buffer = BytesIO(image_bytes)
        img_from_bytesio = create_image(buffer)
        assert isinstance(img_from_bytesio, Image)
        assert img_from_bytesio.base64_data
        assert img_from_bytesio.mime_type in ["image/png", "image/jpeg", "image/webp"]

        # Test 7: Verify all methods produce equivalent results
        # (they should have the same MIME type and base64 data)
        expected_mime_type = img_from_str_path.mime_type
        assert img_from_path_obj.mime_type == expected_mime_type
        assert img_from_pil.mime_type == expected_mime_type
        assert img_from_file_handle.mime_type == expected_mime_type
        assert img_from_bytes.mime_type == expected_mime_type
        assert img_from_bytesio.mime_type == expected_mime_type

        # All should have same base64 data
        expected_base64 = img_from_str_path.base64_data
        assert img_from_path_obj.base64_data == expected_base64
        # assert img_from_pil.base64_data == expected_base64  # ignore due to PIL compression
        assert img_from_file_handle.base64_data == expected_base64
        assert img_from_bytes.base64_data == expected_base64
        assert img_from_bytesio.base64_data == expected_base64

        # Test error scenarios
        # Test 1: Non-existent file path
        with pytest.raises(FileNotFoundError):
            create_image("non_existent_image.jpg")

        # Test 2: PIL Image without format
        pil_image_no_format = PILImage.new("RGB", (100, 100), color="red")
        with pytest.raises(ValueError, match="Cannot determine image format"):
            create_image(pil_image_no_format)

        # Test 3: Invalid bytes data
        with pytest.raises(OSError, match="Cannot open image from bytes data"):
            create_image(b"invalid image data")

        # Test 4: Invalid file-like object
        invalid_buffer = BytesIO(b"not an image")
        with pytest.raises(OSError, match="Cannot open image from file-like object"):
            create_image(invalid_buffer)

        check_locals_memory_usage(locals(), test_name="test_create_image")

    @pytest.mark.parametrize(
        "test_image_path",
        [
            "tests/images/invoices/invoice.png",
            "tests/images/invoices/invoice.jpg",
            "tests/images/invoices/invoice.webp",
        ],
    )
    @memory_profile_and_capture
    def test_image_to_base64(self, test_image_path: str):
        """
        Tests for the `image_to_base64` utility function with various input types.
        """
        project_root = get_project_root_path()
        full_image_path = project_root / test_image_path

        # Ensure test image exists
        assert full_image_path.exists(), f"Test image not found: {full_image_path}"

        # Test 1: Convert from file path (string)
        base64_from_str_path = image_to_base64(str(full_image_path))
        assert isinstance(base64_from_str_path, str)

        # Test 2: Convert from file path (Path object)
        base64_from_path_obj = image_to_base64(full_image_path)
        assert isinstance(base64_from_path_obj, str)

        # Test 3: Convert from file handle
        with open(full_image_path, "rb") as f:
            base64_from_file_handle = image_to_base64(f)
            assert isinstance(base64_from_file_handle, str)

        # Test 4: Convert from raw bytes
        with open(full_image_path, "rb") as f:
            image_bytes = f.read()
        base64_from_bytes = image_to_base64(image_bytes)
        assert isinstance(base64_from_bytes, str)

        # Test 5: Convert from BytesIO
        buffer = BytesIO(image_bytes)
        base64_from_bytesio = image_to_base64(buffer)
        assert isinstance(base64_from_bytesio, str)

        # Test 6: Verify all methods produce identical results
        # (they should all have the same base64 data for the same source)
        assert base64_from_str_path == base64_from_path_obj
        assert base64_from_str_path == base64_from_file_handle
        assert base64_from_str_path == base64_from_bytes
        assert base64_from_str_path == base64_from_bytesio

        # Test error scenarios
        # Test 1: Non-existent file path
        with pytest.raises(FileNotFoundError):
            image_to_base64("non_existent_image.jpg")

        # Test 2: Invalid file-like object that can't be read
        invalid_buffer = BytesIO(b"some data")
        invalid_buffer.close()  # Close the buffer to make it unreadable
        with pytest.raises(OSError, match="Cannot read from file-like object"):
            image_to_base64(invalid_buffer)

        check_locals_memory_usage(locals(), test_name="test_image_to_base64")

    @memory_profile_and_capture
    def test_init_paragraph(self):
        """
        Tests for constructing a Paragraph instance and attaching it to a Document instance.
        """
        paragraph = Paragraph(raw_text="Test paragraph")
        assert paragraph.raw_text
        document = Document(paragraphs=[paragraph])
        assert document.raw_text  # to be populated from paragraphs
        with pytest.raises(ValueError):
            Paragraph()
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="Test paragraph",
                additional_context=1,  # type: ignore
            )
        sentence = Sentence(raw_text="Test sentence")
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="XYZ",
                sentences=[sentence],  # not matched in paragraph text
            )
        with pytest.raises(ValueError):
            Paragraph(
                raw_text=sentence.raw_text,
                sentences=[sentence, sentence],  # sentences with the same ID
            )
        # Warning is logged if linebreaks occur in additional context
        Paragraph(
            raw_text="Test paragraph",
            additional_context="""
            Test
            linebreaks
            """,
        )
        # Test with non-empty text but containing only control chars
        with pytest.raises(ValueError, match="control characters"):
            Paragraph(raw_text=" \u200c ")  # zero-width non-joiner

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(paragraph)

        check_locals_memory_usage(locals(), test_name="test_init_paragraph")

    @memory_profile_and_capture
    def test_init_sentence(self):
        """
        Tests for constructing a Sentence instance and attaching it to a Paragraph instance.
        """
        sentence = Sentence(raw_text="Test sentence")
        assert sentence.raw_text
        Paragraph(raw_text=sentence.raw_text, sentences=[sentence])
        # Warning is logged if linebreaks occur in additional context
        with pytest.raises(ValueError):
            Sentence()
        Sentence(
            raw_text="Test sentence",
            additional_context="""
            Test
            linebreaks
            """,
        )
        # Test with non-empty text but containing only control chars
        with pytest.raises(ValueError, match="control characters"):
            Sentence(raw_text=" \u200c ")  # zero-width non-joiner

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(sentence)

        check_locals_memory_usage(locals(), test_name="test_init_sentence")

    @memory_profile_and_capture
    def test_get_paragraph_and_sentence_index(self):
        """
        Tests for looking up paragraph/sentence positions in a document by object,
        covering duplicate texts, unsupported in-place list mutations, lazy
        sentence population, empty documents, and serialization round-trips.
        """
        # Paragraph lookup, with duplicate paragraph texts (adjacent and
        # separated) resolving to their specific occurrences
        para_texts = [
            "Duplicate clause.",
            "Duplicate clause.",
            "Intro.",
            "Duplicate clause.",
            "End.",
        ]
        document = Document(paragraphs=[Paragraph(raw_text=i) for i in para_texts])
        for idx, paragraph in enumerate(document.paragraphs):
            assert document.get_paragraph_index(paragraph) == idx
            # Repeated lookups are consistent
            assert document.get_paragraph_index(paragraph) == idx
        # Sorting references by position works with duplicate texts
        refs = [
            document.paragraphs[3],
            document.paragraphs[0],
            document.paragraphs[1],
        ]
        assert sorted(refs, key=document.get_paragraph_index) == [
            document.paragraphs[0],
            document.paragraphs[1],
            document.paragraphs[3],
        ]
        # Single-paragraph document
        single_para_document = Document(paragraphs=[Paragraph(raw_text="Only one.")])
        assert (
            single_para_document.get_paragraph_index(single_para_document.paragraphs[0])
            == 0
        )
        # Foreign paragraph is not found, even with identical text
        with pytest.raises(ValueError, match="not found"):
            document.get_paragraph_index(Paragraph(raw_text="Duplicate clause."))
        # Paragraphs of a same-text twin document are not found either
        # (lookup is keyed by unique ID, not text equality)
        twin_document = Document(paragraphs=[Paragraph(raw_text=i) for i in para_texts])
        with pytest.raises(ValueError, match="not found"):
            document.get_paragraph_index(twin_document.paragraphs[0])
        # Document without paragraphs (image-only)
        image_document = Document(images=[self.test_img_png_invoice])
        with pytest.raises(ValueError, match="not found"):
            image_document.get_paragraph_index(document.paragraphs[0])
        with pytest.raises(ValueError, match="not found"):
            image_document.get_sentence_index(Sentence(raw_text="Sent A."))

        # Unsupported in-place mutations of the paragraphs list must not
        # yield stale positions (lookups always reflect the live list state)
        mutated_document = Document(
            paragraphs=[Paragraph(raw_text=i) for i in ["P0.", "P1.", "P2."]]
        )
        first_para = mutated_document.paragraphs[0]
        last_para = mutated_document.paragraphs[-1]
        assert mutated_document.get_paragraph_index(last_para) == 2
        # Insertion shifts positions
        inserted_para = Paragraph(raw_text="Inserted.")
        mutated_document.paragraphs.insert(0, inserted_para)
        assert mutated_document.get_paragraph_index(first_para) == 1
        assert mutated_document.get_paragraph_index(last_para) == 3
        assert mutated_document.get_paragraph_index(inserted_para) == 0
        # Removed paragraph no longer resolves
        del mutated_document.paragraphs[-1]
        with pytest.raises(ValueError, match="not found"):
            mutated_document.get_paragraph_index(last_para)
        assert mutated_document.get_paragraph_index(first_para) == 1
        # Removal from the start shifts positions
        del mutated_document.paragraphs[0]
        assert mutated_document.get_paragraph_index(first_para) == 0

        # Sentence lookup, with duplicate sentence texts within and
        # across paragraphs resolving to their specific occurrences
        document_with_sents = Document(
            paragraphs=[
                Paragraph(
                    raw_text="Sent A. Sent A. Sent B.",
                    sentences=[
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent B."),
                    ],
                ),
                Paragraph(
                    raw_text="Sent A. Sent C.",
                    sentences=[
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent C."),
                    ],
                ),
            ]
        )
        for para_idx, paragraph in enumerate(document_with_sents.paragraphs):
            for sent_idx, sentence in enumerate(paragraph.sentences):
                assert document_with_sents.get_sentence_index(sentence) == (
                    para_idx,
                    sent_idx,
                )
                # Repeated lookups are consistent
                assert document_with_sents.get_sentence_index(sentence) == (
                    para_idx,
                    sent_idx,
                )
        # Positions sort in document order across paragraphs, matching the
        # flattened `document.sentences` order
        all_sents = document_with_sents.sentences
        assert (
            sorted(all_sents, key=document_with_sents.get_sentence_index) == all_sents
        )
        # Foreign sentence is not found, even with identical text
        with pytest.raises(ValueError, match="not found"):
            document_with_sents.get_sentence_index(Sentence(raw_text="Sent A."))

        # Lookup error hints that sentences are not segmented yet
        with pytest.raises(ValueError, match="no sentences segmented"):
            document.get_sentence_index(Sentence(raw_text="Intro."))

        # Sentences populated after earlier lookups resolve correctly,
        # including partial (per-paragraph) population
        document.paragraphs[2].sentences = [Sentence(raw_text="Intro.")]
        assert document.get_sentence_index(document.paragraphs[2].sentences[0]) == (
            2,
            0,
        )
        document.paragraphs[4].sentences = [Sentence(raw_text="End.")]
        assert document.get_sentence_index(document.paragraphs[4].sentences[0]) == (
            4,
            0,
        )
        # Previously populated sentences still resolve
        assert document.get_sentence_index(document.paragraphs[2].sentences[0]) == (
            2,
            0,
        )
        # Once some sentences exist, the not-segmented hint no longer applies
        with pytest.raises(ValueError, match="not found") as exc_info:
            document.get_sentence_index(Sentence(raw_text="Intro."))
        assert "no sentences segmented" not in str(exc_info.value)

        # Unsupported in-place mutations of a paragraph's sentences list must not
        # yield stale positions (lookups always reflect the live list state)
        mutable_sents_para = document_with_sents.paragraphs[1]
        tracked_sent = mutable_sents_para.sentences[-1]  # "Sent C." at (1, 1)
        assert document_with_sents.get_sentence_index(tracked_sent) == (1, 1)
        # Insertion shifts positions within the paragraph
        inserted_sent = Sentence(raw_text="Sent A.")
        mutable_sents_para.sentences.insert(0, inserted_sent)
        assert document_with_sents.get_sentence_index(tracked_sent) == (1, 2)
        assert document_with_sents.get_sentence_index(inserted_sent) == (1, 0)
        # Removed sentence no longer resolves
        del mutable_sents_para.sentences[-1]  # removes tracked_sent
        with pytest.raises(ValueError, match="not found"):
            document_with_sents.get_sentence_index(tracked_sent)
        # Other paragraphs' sentences are unaffected
        assert document_with_sents.get_sentence_index(
            document_with_sents.paragraphs[0].sentences[0]
        ) == (0, 0)

        # Unique IDs survive serialization round-trips, so positions resolve
        # across original/deserialized document boundaries, in both directions
        deserialized_document = Document.from_dict(document.to_dict())
        for idx in (0, 1, 3):
            assert (
                deserialized_document.get_paragraph_index(document.paragraphs[idx])
                == idx
            )
            assert (
                document.get_paragraph_index(deserialized_document.paragraphs[idx])
                == idx
            )
        assert deserialized_document.get_sentence_index(
            document.paragraphs[4].sentences[0]
        ) == (4, 0)
        assert document.get_sentence_index(
            deserialized_document.paragraphs[2].sentences[0]
        ) == (2, 0)
        # Deserialized document's own objects resolve within itself
        for idx, paragraph in enumerate(deserialized_document.paragraphs):
            assert deserialized_document.get_paragraph_index(paragraph) == idx

        # Performing lookups (which build internal position caches) must not
        # affect document equality, serialization, or cloning
        assert Document.from_dict(document.to_dict()) == document
        assert Document.from_dict(document_with_sents.to_dict()) == document_with_sents
        self.check_instance_serialization_and_cloning(document)

        check_locals_memory_usage(
            locals(), test_name="test_get_paragraph_and_sentence_index"
        )

    @memory_profile_and_capture
    def test_init_message(self):
        """
        Tests for constructing a _Message instance and validating its content schema.
        """
        # Simple text message
        msg = _Message(role="user", content="Hello")
        msg._response_succeeded = False
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.unique_id, str) and msg.unique_id
        assert isinstance(msg.time_created, datetime) and msg.time_created
        assert msg._to_message_dict() == {"role": "user", "content": "Hello"}
        assert msg._response_succeeded is False

        # Validation errors for frozen fields
        with pytest.raises(ValueError):
            msg.content = "New content"
        with pytest.raises(ValueError):
            msg.role = "assistant"
        # Check for no overrides
        assert msg.role == "user"
        assert msg.content == "Hello"

        # Multimodal content (text + image_url)
        multimodal = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
        msg_mm = _Message(role="user", content=multimodal)
        assert isinstance(msg_mm.content, list) and len(msg_mm.content) == 2
        assert msg_mm._response_succeeded is None

        # Validation errors for malformed parts
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"text": "missing type"}],
            )
        with pytest.raises(ValueError):
            _Message(role="user", content=[{"type": "text"}])
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "text", "text": ""}],
            )
        with pytest.raises(ValueError):
            _Message(role="user", content=[{"type": "image_url"}])
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "image_url", "image_url": {}}],
            )
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "image_url", "image_url": {"url": ""}}],
            )
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "audio", "url": "x"}],
            )

        # Assignment validation (validate_assignment=True)
        msg2 = _Message(role="assistant", content="ok")
        with pytest.raises(ValueError):
            msg2.content = [{"type": "text", "text": ""}]  # invalid after init
        # `_response_succeeded` defaults and assignment rules
        assert msg2._response_succeeded is None
        with pytest.raises(AttributeError):
            msg2._response_succeeded = True

        # System role should also forbid assignment
        msg_sys = _Message(role="system", content="hi")
        assert msg_sys._response_succeeded is None
        with pytest.raises(AttributeError):
            msg_sys._response_succeeded = False

        # User role allows assignment
        msg._response_succeeded = True
        assert msg._response_succeeded is True

        # Serialization roundtrip (dict/json) preserves fields and private unique_id
        d = msg.to_dict()
        msg_from_d = _Message.from_dict(d)
        assert msg_from_d == msg
        assert msg_from_d.role == msg.role
        assert msg_from_d.content == msg.content
        assert msg_from_d.unique_id == msg.unique_id
        assert msg_from_d.time_created == msg.time_created
        assert msg_from_d._response_succeeded == msg._response_succeeded

        j = msg.to_json()
        msg_from_j = _Message.from_json(j)
        assert msg_from_j == msg
        assert msg_from_j.role == msg.role
        assert msg_from_j.content == msg.content
        assert msg_from_j.unique_id == msg.unique_id
        assert msg_from_j.time_created == msg.time_created
        assert msg_from_j._response_succeeded == msg._response_succeeded

        msg._response_succeeded = None  # allow for re-setting to None

        # Disabled pydantic dump methods
        with pytest.raises(NotImplementedError):
            msg.model_dump()
        with pytest.raises(NotImplementedError):
            msg.model_dump_json()

        # Tool-related messages
        tool_msg = _Message(
            role="tool",
            content="result payload",
            tool_call_id="call_123",
            name="function_name",
        )
        tool_dict = tool_msg._to_message_dict()
        assert tool_dict == {
            "role": "tool",
            "content": "result payload",
            "tool_call_id": "call_123",
            "name": "function_name",
        }
        assert "tool_calls" not in tool_dict

        with pytest.raises(
            ValueError, match="must be set for messages with role 'tool'"
        ):
            _Message(role="tool", content="ok")

        with pytest.raises(
            ValueError, match="`name` must be set for messages with role 'tool'"
        ):
            _Message(role="tool", content="ok", tool_call_id="id_missing_name")

        with pytest.raises(
            ValueError, match="Tool message content must be a non-empty string"
        ):
            _Message(
                role="tool",
                content=[{"type": "text", "text": "x"}],
                tool_call_id="id1",
            )

        with pytest.raises(
            ValueError,
            match="`tool_calls` is only allowed for messages with role 'assistant'",
        ):
            _Message(
                role="tool",
                content="ok",
                name="fn",
                tool_call_id="id2",
                tool_calls=[
                    {"type": "function", "function": {"name": "fn", "arguments": "{}"}}
                ],
            )

        with pytest.raises(
            ValueError,
            match="`tool_call_id` is only allowed for messages with role 'tool'",
        ):
            _Message(role="user", content="hello", tool_call_id="some_id")

        tool_calls = [
            {
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "weather"}'},
            }
        ]
        asst_msg = _Message(
            role="assistant", content="thinking...", tool_calls=tool_calls
        )
        asst_dict = asst_msg._to_message_dict()
        assert asst_dict["role"] == "assistant"
        assert asst_dict["content"] == "thinking..."
        assert asst_dict["tool_calls"] == tool_calls
        assert "tool_call_id" not in asst_dict

        with pytest.raises(
            ValueError,
            match="`tool_calls` is only allowed for messages with role 'assistant'",
        ):
            _Message(role="user", content="hi", tool_calls=tool_calls)

        with pytest.raises(
            ValueError,
            match="`tool_call_id` is only allowed for messages with role 'tool'",
        ):
            _Message(role="assistant", content="ok", tool_call_id="id3")

        # Serialization roundtrip tests for tool and assistant-with-tools messages
        # Tool message roundtrip (dict/json)
        tool_d = tool_msg.to_dict()
        tool_from_d = _Message.from_dict(tool_d)
        assert tool_from_d == tool_msg
        assert tool_from_d.role == "tool"
        assert tool_from_d.content == "result payload"
        assert tool_from_d.tool_call_id == "call_123"
        assert tool_from_d.name == "function_name"
        assert tool_from_d.unique_id == tool_msg.unique_id
        assert tool_from_d.time_created == tool_msg.time_created
        assert "_response_succeeded" not in tool_d

        tool_j = tool_msg.to_json()
        tool_from_j = _Message.from_json(tool_j)
        assert tool_from_j == tool_msg
        assert tool_from_j.role == "tool"
        assert tool_from_j.content == "result payload"
        assert tool_from_j.tool_call_id == "call_123"
        assert tool_from_j.name == "function_name"
        assert tool_from_j.unique_id == tool_msg.unique_id
        assert tool_from_j.time_created == tool_msg.time_created

        # Assistant message with tool_calls roundtrip (dict/json)
        asst_d = asst_msg.to_dict()
        asst_from_d = _Message.from_dict(asst_d)
        assert asst_from_d == asst_msg
        assert asst_from_d.role == "assistant"
        assert asst_from_d.content == "thinking..."
        assert asst_from_d.tool_calls == tool_calls
        assert asst_from_d.tool_call_id is None
        assert asst_from_d.unique_id == asst_msg.unique_id
        assert asst_from_d.time_created == asst_msg.time_created
        assert "_response_succeeded" not in asst_d

        asst_j = asst_msg.to_json()
        asst_from_j = _Message.from_json(asst_j)
        assert asst_from_j == asst_msg
        assert asst_from_j.role == "assistant"
        assert asst_from_j.content == "thinking..."
        assert asst_from_j.tool_calls == tool_calls
        assert asst_from_j.tool_call_id is None
        assert asst_from_j.unique_id == asst_msg.unique_id
        assert asst_from_j.time_created == asst_msg.time_created

        check_locals_memory_usage(locals(), test_name="test_init_message")

    @memory_profile_and_capture
    def test_init_chat_session(self):
        """
        Tests for initializing `ChatSession` and enforcing message history rules.
        """

        # 1) System + user message order is preserved, and messages property returns a copy
        cs = ChatSession()
        system_msg = _Message(role="system", content="You are ContextGem.")
        user_msg = _Message(role="user", content="Hello")
        cs._set_messages([system_msg, user_msg])
        msgs = cs.messages
        assert len(msgs) == 2
        assert msgs[0].role == "system" and msgs[0].content == "You are ContextGem."
        assert msgs[1] == user_msg

        # messages property should return a copy
        msgs.append(_Message(role="assistant", content="Hi"))
        assert len(cs.messages) == 2

        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs)

        # reset clears all messages
        cs.reset()
        assert cs.messages == []

        # 2) Duplicate messages are not allowed
        dup = _Message(role="user", content="dup")
        # LLM instance is not required for chat session tests now
        cs2 = ChatSession()
        with pytest.raises(ValueError, match="contain duplicates"):
            cs2._extend_messages([dup, dup])

        # 3) A single system message must be the first message
        cs3 = ChatSession()
        with pytest.raises(ValueError, match="must be the first"):
            cs3._extend_messages(
                [
                    _Message(role="user", content="first"),
                    _Message(role="system", content="sys"),
                ]
            )

        # 4) Multiple system messages are not allowed
        cs4 = ChatSession()
        with pytest.raises(ValueError, match="Multiple 'system' messages"):
            cs4._extend_messages(
                [
                    _Message(role="system", content="one"),
                    _Message(role="system", content="another"),
                ]
            )

        check_locals_memory_usage(locals(), test_name="test_init_chat_session")

    @memory_profile_and_capture
    def test_init_aspect(self):
        """
        Tests the initializing and error handling of the `Aspect` class.
        """

        with pytest.raises(ValueError):
            Aspect()
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                model_role="",  # invalid model role (empty label)
            )
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                modality="image",  # ty: ignore[unknown-argument]  # aspects are extracted from text only
            )
        aspect = Aspect(
            name="Categories of confidential information",
            description="Clauses describing confidential information covered by the NDA",
            add_justifications=True,
        )
        aspect_concepts = [
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
            ),
            StringConcept(
                name="Technical Information",
                description="Categories of Technical Information",
                model_role="reasoner",
            ),
        ]
        aspect.concepts = aspect_concepts
        assert aspect.concepts is not aspect_concepts
        # Validate assignment
        with pytest.raises(ValueError, match="non-empty list"):
            aspect.add_aspects([])  # empty list
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [  # ty: ignore[invalid-argument-type]
                    1,  # invalid type
                ]
            )
        with pytest.raises(ValueError, match="non-empty list"):
            aspect.add_concepts([])  # empty list
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [
                    StringConcept(  # duplicate name and description
                        name="Business Information",
                        description="Categories of Business Information",
                        add_justifications=True,
                    )
                ]
            )
        with pytest.raises(ValueError, match="extracted from text only"):
            aspect.add_concepts(
                [
                    StringConcept(
                        name="Random",
                        description="Random",
                        modality="image",  # invalid modality for aspect-level concept
                    )
                ]
            )
        with pytest.raises(ValueError, match="extracted from text only"):
            aspect.add_concepts(
                [
                    StringConcept(
                        name="Random",
                        description="Random",
                        modality="multimodal",  # invalid modality for aspect-level concept
                    )
                ]
            )

        # Invalid sequence types
        with pytest.raises(ValueError, match="list"):
            aspect.add_concepts(
                (
                    StringConcept(
                        name="Random",
                        description="Random",
                    ),
                )  # tuple instead of list  # type: ignore
            )
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = (
                StringConcept(
                    name="Random",
                    description="Random",
                ),
            )  # tuple instead of list
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = set([1, 2, 3])  # set instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = range(10)  # range instead of list  # type: ignore

        # Test adding a sub-aspect with the same name or description as the main aspect
        sub_aspect_same_name = Aspect(
            name="Categories of confidential information",  # Same name as parent aspect
            description="Different description",
        )
        with pytest.raises(ValueError):
            aspect.add_aspects([sub_aspect_same_name])

        sub_aspect_same_description = Aspect(
            name="Different name",
            description="Clauses describing confidential information covered by the NDA",  # Same description as parent aspect
        )
        with pytest.raises(ValueError):
            aspect.add_aspects([sub_aspect_same_description])

        # Test that only one level of nesting is allowed for sub-aspects
        main_aspect = Aspect(
            name="Main Aspect",
            description="Main aspect description",
        )
        sub_aspect = Aspect(
            name="Sub Aspect",
            description="Sub aspect description",
            model_role="reasoner",
        )
        sub_sub_aspect = Aspect(
            name="Sub-Sub Aspect",
            description="Sub-sub aspect description",
        )

        # Adding a sub-aspect to the main aspect should work
        main_aspect.add_aspects([sub_aspect])
        assert len(main_aspect.aspects) == 1
        assert main_aspect._nesting_level == 0

        sub_aspect = main_aspect.aspects[0]
        assert sub_aspect._nesting_level == 1

        # Trying to add a sub-sub-aspect to the sub-aspect will raise ValueError
        with pytest.raises(ValueError, match="maximum nesting level"):
            sub_aspect.add_aspects(
                [sub_sub_aspect]
            )  # would be nesting level 2, max is 1

        # Verify no changes occurred due to invalid assignment
        logger.debug(sub_aspect.aspects)  # ty: ignore[invalid-argument-type]
        assert not sub_aspect.aspects

        # Verify no changes occurred due to invalid assignments
        assert not aspect.aspects

        # Invalid extracted items
        with pytest.raises(ValueError):
            aspect.extracted_items = [1, True]  # type: ignore
        # No changes due to invalid assignments
        assert aspect.concepts == aspect_concepts
        assert aspect.concepts is not aspect_concepts
        with pytest.raises(ValueError):
            aspect.get_concept_by_name("Non-existent")  # not assigned concept

        # Adding concepts
        aspect.add_concepts(
            [
                StringConcept(
                    name="Personnel Information",
                    description="Categories of information on personnel and staff",
                    add_justifications=True,
                )
            ]
        )
        assert len(aspect.concepts) == 3
        assert aspect.get_concepts_by_names(
            ["Business Information", "Technical Information", "Personnel Information"]
        )

        # Removal of concepts
        aspect.remove_concept_by_name("Technical Information")
        assert len(aspect.concepts) == 2
        assert len(aspect_concepts) == 2
        with pytest.raises(ValueError):
            aspect.get_concept_by_name("Technical Information")  # not assigned concept
        aspect.remove_all_concepts()
        assert len(aspect.concepts) == 0
        assert len(aspect_concepts) == 2

        # List field items' unique IDs
        concept = aspect_concepts[0]
        with pytest.raises(ValueError):
            aspect.add_concepts([concept, concept])

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(aspect)

        check_locals_memory_usage(locals(), test_name="test_init_aspect")

    @memory_profile_and_capture
    def test_init_and_validate_string_concept(self):
        """
        Tests the initialization of the StringConcept class with valid and invalid input parameters.
        """
        # Valid initialization
        string_concept = StringConcept(
            name="Business Information",
            description="Categories of Business Information",
            add_justifications=True,
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            StringConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                extra_param=True,  # type: ignore
            )

        # Test attaching extracted items
        string_concept.extracted_items = [_StringItem(value="Confidential data")]  # ty: ignore[invalid-assignment]

        # Test with invalid items
        with pytest.raises(ValueError):
            string_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _BooleanItem(value=True)
            ]  # must be _StringItem

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(string_concept)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_string_concept"
        )

    @memory_profile_and_capture
    def test_init_and_validate_boolean_concept(self):
        """
        Tests the initialization of the BooleanConcept class with valid and invalid input parameters.
        """
        # Valid initialization
        boolean_concept = BooleanConcept(
            name="Contract type check",
            description="Contract is an NDA",
            add_justifications=True,
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            BooleanConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                extra_param=True,  # type: ignore
            )

        # Test attaching extracted items
        boolean_concept.extracted_items = [_BooleanItem(value=True)]  # ty: ignore[invalid-assignment]

        # Test with invalid items
        with pytest.raises(ValueError):
            boolean_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _StringItem(value="True")
            ]  # must be _BooleanItem
        with pytest.raises(ValueError):
            boolean_concept.extracted_items = [  # type: ignore
                _BooleanItem(value=None)  # type: ignore
            ]  # must be _BooleanItem

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(boolean_concept)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_boolean_concept"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_extract_bool_values(self):
        """
        Tests the extraction of a BooleanConcept instance.
        """
        concept_positive_1 = BooleanConcept(
            name="NDA check",
            description="Document is an NDA",
            add_justifications=True,
        )
        concept_positive_2 = BooleanConcept(
            name="Liability section check",
            description="Document has a section on liability",
            add_justifications=False,
            add_references=True,
        )
        concept_negative_1 = BooleanConcept(
            name="Supplier agreement check",
            description="Document is a supplier agreement",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )
        concept_negative_2 = BooleanConcept(
            name="Payment terms section check",
            description="Document includes a section on payment terms",
            add_justifications=False,
        )
        concept_impossible_to_determine_1 = BooleanConcept(
            name="Bitcoin price check",
            description="Bitcoin price is above $100,000",
            add_justifications=True,
        )
        concept_impossible_to_determine_2 = BooleanConcept(
            name="Weather check",
            description="Weather in Oslo is above 30 degrees today",
            add_justifications=False,
        )
        document = self.document.clone()
        document.add_concepts(
            [
                concept_positive_1,
                concept_positive_2,
                concept_negative_1,
                concept_negative_2,
                concept_impossible_to_determine_1,
                concept_impossible_to_determine_2,
            ]
        )
        (
            concept_positive_1,
            concept_positive_2,
            concept_negative_1,
            concept_negative_2,
            concept_impossible_to_determine_1,
            concept_impossible_to_determine_2,
        ) = self.llm_extractor.extract_concepts_from_document(
            document, use_concurrency=True
        )
        assert concept_positive_1.extracted_items
        assert concept_positive_2.extracted_items
        assert concept_negative_1.extracted_items
        assert concept_negative_2.extracted_items
        assert concept_positive_1.extracted_items[0].value is True
        assert concept_positive_2.extracted_items[0].value is True
        assert concept_negative_1.extracted_items[0].value is False
        assert concept_negative_2.extracted_items[0].value is False
        assert (
            not concept_impossible_to_determine_1.extracted_items
            or concept_impossible_to_determine_1.extracted_items[0].value is False
        )
        assert (
            not concept_impossible_to_determine_2.extracted_items
            or concept_impossible_to_determine_2.extracted_items[0].value is False
        )

        # Test multiple items of the same concept
        text = """
        CONSOLIDATED AUDIT REPORT - Q3 2024

        SUBSIDIARY COMPLIANCE REVIEW:

        1. TechCorp North America: The audit identified several GDPR compliance gaps 
        in data processing procedures. Immediate remediation required.

        2. XYZ Europe Ltd: All regulatory requirements met. No compliance issues 
        identified during the review period.

        3. ABC Asia Pacific: GDPR compliance satisfactory, however, local data 
        residency requirements in Singapore market not fully addressed.
        """
        document = Document(raw_text=text)
        concept_multiple_items = BooleanConcept(
            name="Has Regulatory Compliance Issues",
            description="Entity has outstanding regulatory compliance violations or issues. "
            "Analyze and extract the value for each mentioned entity.",
            singular_occurrence=False,  # Allow multiple items (default)
            add_references=True,
            reference_depth="sentences",
        )
        document.add_concepts([concept_multiple_items])
        (concept_multiple_items,) = self.llm_extractor.extract_concepts_from_document(
            document
        )
        assert len(concept_multiple_items.extracted_items) == 3
        assert concept_multiple_items.extracted_items[0].value is True
        assert concept_multiple_items.extracted_items[1].value is False
        assert concept_multiple_items.extracted_items[2].value is True

        # Check that we have identical behaviour when bool is a return type
        # in a JsonObjectConcept field
        json_obj_concept_positive = JsonObjectConcept(
            name="NDA check",
            description="Document is an NDA",
            add_justifications=True,
            structure={
                "is_nda": bool,
            },
        )
        json_obj_concept_negative = JsonObjectConcept(
            name="Supplier agreement check",
            description="Document is a supplier agreement",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
            structure={
                "is_supplier_agreement": bool,
            },
        )
        json_obj_concept_impossible_to_determine = JsonObjectConcept(
            name="Bitcoin price check",
            description="Bitcoin price is above $100,000",
            add_justifications=True,
            structure={
                "bitcoin_price_is_above_100k": bool,
            },
        )
        document = self.document.clone()
        document.add_concepts(
            [
                json_obj_concept_positive,
                json_obj_concept_negative,
                json_obj_concept_impossible_to_determine,
            ]
        )
        (
            json_obj_concept_positive,
            json_obj_concept_negative,
            json_obj_concept_impossible_to_determine,
        ) = self.llm_extractor.extract_concepts_from_document(
            document, use_concurrency=True
        )
        assert json_obj_concept_positive.extracted_items
        assert json_obj_concept_negative.extracted_items
        assert json_obj_concept_positive.extracted_items[0].value["is_nda"] is True
        assert (
            json_obj_concept_negative.extracted_items[0].value["is_supplier_agreement"]
            is False
        )
        assert (
            not json_obj_concept_impossible_to_determine.extracted_items
            or json_obj_concept_impossible_to_determine.extracted_items[0].value[
                "bitcoin_price_is_above_100k"
            ]
            is False
        )

        check_locals_memory_usage(locals(), test_name="test_extract_boolean_concept")

    @memory_profile_and_capture
    def test_init_and_validate_numerical_concept(self):
        """
        Tests for initialization and usage of NumericalConcept with different types.
        """
        # Test initialization with different numeric_type values
        int_concept = NumericalConcept(
            name="Integer concept",
            description="Test integer concept",
            numeric_type="int",
        )
        float_concept = NumericalConcept(
            name="Float concept", description="Test float concept", numeric_type="float"
        )
        any_concept = NumericalConcept(
            name="Any numeric concept",
            description="Test any numeric concept",
            numeric_type="any",
        )

        # Test with invalid numeric_type
        with pytest.raises(ValueError):
            NumericalConcept(
                name="Invalid type",
                description="Test invalid type",
                numeric_type="string",  # type: ignore
            )

        # Test attaching extracted items
        # Integer concept should accept only integer items
        int_item = _IntegerItem(value=42)
        int_concept.extracted_items = [int_item]  # ty: ignore[invalid-assignment]

        # Should reject float items for int concept
        with pytest.raises(ValueError):
            int_concept.extracted_items = [_FloatItem(value=42.5)]  # ty: ignore[invalid-assignment]

        # Float concept should accept only float items
        float_item = _FloatItem(value=3.14)
        float_concept.extracted_items = [float_item]  # ty: ignore[invalid-assignment]

        # Should reject integer items for float concept
        with pytest.raises(ValueError):
            float_concept.extracted_items = [_IntegerItem(value=3)]  # ty: ignore[invalid-assignment]

        # Any numeric concept should accept both integer and float items
        any_concept.extracted_items = [_IntegerOrFloatItem(value=20)]  # ty: ignore[invalid-assignment]
        any_concept.extracted_items = [_IntegerOrFloatItem(value=20.5)]  # ty: ignore[invalid-assignment]
        # Should reject integer items for any numeric concept
        with pytest.raises(ValueError):
            any_concept.extracted_items = [_IntegerItem(value=10)]  # ty: ignore[invalid-assignment]
        # Should reject float items for any numeric concept
        with pytest.raises(ValueError):
            any_concept.extracted_items = [_FloatItem(value=10.5)]  # ty: ignore[invalid-assignment]

        # Test with justifications and references
        concept_with_refs = NumericalConcept(
            name="Concept with refs",
            description="Test concept with references",
            numeric_type="any",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(int_concept)
        self.check_instance_serialization_and_cloning(float_concept)
        self.check_instance_serialization_and_cloning(any_concept)
        self.check_instance_serialization_and_cloning(concept_with_refs)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_numerical_concept"
        )

    @memory_profile_and_capture
    def test_init_and_validate_rating_concept(self):
        """
        Tests the initialization of the RatingConcept class with valid and invalid input parameters.
        """

        # Valid initialization
        valid_rating_concept = RatingConcept(
            name="Customer Satisfaction",
            description="Rating of customer satisfaction",
            rating_scale=(1, 5),
        )

        # Test property access
        assert valid_rating_concept._rating_start == 1
        assert valid_rating_concept._rating_end == 5

        # Invalid initialization
        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(),  # empty tuple  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(-1, 5),  # negative start
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(1, 0),  # zero end
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(5, 5),  # equal start and end
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(10, 5),  # start greater than end
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(
                    1,
                ),  # tuple with invalid length (1 element)  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(
                    1,
                    1,
                    2,
                ),  # tuple with invalid length (3 elements)  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=(1.5, 3),  # float element  # type: ignore
            )

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale={"start": 1, "end": 5},  # not a tuple  # type: ignore
            )

        # Test attaching extracted items
        rating_item = _IntegerItem(value=3)
        valid_rating_concept.extracted_items = [rating_item]

        # Duplicated items
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [rating_item, rating_item]

        # Test with out-of-range values
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [
                _IntegerItem(value=0)
            ]  # below min (1)

        with pytest.raises(ValueError):
            valid_rating_concept._process_item_value(0)

        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [
                _IntegerItem(value=6)
            ]  # above max (5)

        with pytest.raises(ValueError):
            valid_rating_concept._process_item_value(6)

        # Test with non-integer items
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [_FloatItem(value=3.5)]  # type: ignore

        # Test with justifications and references
        concept_with_refs = RatingConcept(
            name="Concept with refs",
            description="Test concept with references",
            rating_scale=(1, 10),
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(valid_rating_concept)
        self.check_instance_serialization_and_cloning(concept_with_refs)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_rating_concept"
        )

    @memory_profile_and_capture
    def test_init_and_validate_json_object_concept(self):
        """
        Tests the initialization of the JsonObjectConcept class with valid and invalid input parameters.
        """
        # Base class direct initialization
        with pytest.raises(TypeError):
            _Concept(name="Test", description="Test")
        with pytest.raises(ValueError):
            JsonObjectConcept()
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "test": str,
                },
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "_test": str,  # cannot start with underscores as these attrs will not be validated by pydantic
                    "info": str,
                },
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "test": str,
                },
                modality="image",
                add_references=True,
                # references are not supported in vision concepts
            )

        # Define all structures to test
        # 1. Simple dictionary structure
        dict_struct = {
            "category": str,
            "descriptions": List[str],  # noqa: UP006
        }

        # 2. Class structure
        class Struct:
            category: str
            descriptions: list[str]

        # 3. Pydantic structure
        class PydanticStruct(BaseModel):
            category: str
            descriptions: list[str]

        # 4. Structure with optional types
        optional_struct = {
            "required_field": str,
            "optional_str": Optional[str],  # noqa: UP045
            "optional_int": Optional[int],  # noqa: UP045
            "optional_float": float | None,
            "optional_bool": bool | None,
            "union_syntax": str | None,
        }

        # 5. Structure with literal types
        literal_struct = {
            "status": Literal["active", "inactive", "pending"],
            "role": Literal["admin", "user", "guest"],
            "priority": Literal[1, 2, 3],
        }

        # 6. Nested dictionary structure
        nested_dict_struct = {
            "name": str,
            "age": int,
            "contact": {
                "email": str,
                "phone": str,
                "address": {"street": str, "city": str, "country": str},
            },
        }

        # 7. Structure with list of dictionaries
        list_of_dicts_struct = {
            "name": str,
            "skills": [{"name": str, "level": int}],
        }

        # 8. Classes with type annotations for nested structures
        # We also add fields to test warnings (field logic will be discarded)
        @dataclass
        class Address(JsonObjectClassStruct):
            street: str = field(metadata={"description": "Street address"})
            city: str = field(repr=True)
            country: str = field(compare=True)

        @dataclass
        class Contact(JsonObjectClassStruct):
            email: str = field(metadata={"description": "test"})
            phone: str = field(metadata={"format": "international"})
            address: Address = field(repr=False)
            contact_type: Optional[  # noqa: UP045
                Literal["primary", "secondary", "emergency", None]
                | Literal["union", "literal", None]
            ] = field(default=None)  # intentionally messed up type hint for testing

        @dataclass
        class Person(JsonObjectClassStruct):
            name: str = field(compare=True)
            age: int = field(metadata={"min": 0, "max": 120})
            contact: Contact = field(repr=False)

        # 9. Pydantic models for nested structures
        class AddressModel(BaseModel, JsonObjectClassStruct):
            street: str
            city: str
            country: str

        class ContactModel(BaseModel, JsonObjectClassStruct):
            email: str
            phone: str
            address: AddressModel

        class PersonModel(BaseModel, JsonObjectClassStruct):
            name: str
            age: int
            contact: ContactModel

        # 10. Super complex structure combining everything
        super_complex_struct = {
            "user": PersonModel,
            "status": Literal["active", "inactive"],
            "permissions": [
                {"resource": str, "access": Literal["read", "write", "admin"]}
            ],
            "settings": {
                "theme": str,
                "notifications": {
                    "email": bool,
                    "sms": bool,
                    "frequency": Literal["daily", "weekly", "monthly"],
                },
            },
        }

        # 11. Structure with list[cls]
        @dataclass
        class Item(JsonObjectClassStruct):
            id: int
            name: str
            active: bool

        list_of_class_struct = {"title": str, "items": list[Item]}

        # Class structure containing a list[cls] type hint
        @dataclass
        class ClassWithListOfClasses(JsonObjectClassStruct):
            name: str
            description: str
            items: list[Item]

        # 12. Structure with Dict mapping
        dict_mapping_struct = {
            "name": str,
            "scores": Dict[str, int],  # noqa: UP006
        }

        # 13. Pydantic model with Field() validations
        # We add fields to test warnings (field logic will be discarded)
        class ValidatedModel(BaseModel, JsonObjectClassStruct):
            name: str = Field(..., min_length=3, max_length=50)
            age: int = Field(..., ge=18, le=100)
            email: str = Field(
                ..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
            )
            tags: list[str] = Field(default_factory=list, max_length=5)
            score: float = Field(default=0.0, ge=0.0, le=10.0)

        validated_model_struct = ValidatedModel

        # 14. Nested structure with dict having union type values
        @dataclass
        class ConfigSettings(JsonObjectClassStruct):
            enabled: bool
            values: dict[
                str, int | float | str | bool | None
            ]  # Union type as dict value
            description: str

        @dataclass
        class AppConfiguration(JsonObjectClassStruct):
            app_name: str
            version: str
            settings: ConfigSettings

        # 15. Pydantic model with field validators
        class UserProfile(BaseModel, JsonObjectClassStruct):
            name: str
            email: str
            age: int

            @field_validator("email")
            @classmethod
            def validate_email(cls, v):
                if "@" not in v:
                    raise ValueError("Email must contain @ symbol")
                return v

            @field_validator("age")
            @classmethod
            def validate_age(cls, v):
                if v < 0 or v > 120:
                    raise ValueError("Age must be between 0 and 120")
                return v

        # 16. Structure with list instances using primitive types
        list_instance_primitives_struct = {
            "product_names": [Literal["laptop", "mouse", "keyboard"]],
            "scores": [int],
            "prices": [float],
            "active_flags": [bool],
            "mixed_numbers": [int | float],
            "optional_tags": [str | None],
        }

        # 17. Structure with list instances using classes
        @dataclass
        class Product(JsonObjectClassStruct):
            id: int
            name: str
            price: float

        list_instance_classes_struct = {
            "title": str,
            "products": [Product],
            "categories": [Literal["electronics", "books", "clothing"]],
        }

        # Define valid examples for each structure
        valid_examples = {
            "ClassWithListOfClasses": {
                "name": "Collection of Items",
                "description": "A collection of multiple items",
                "items": [
                    {"id": 1, "name": "First Item", "active": True},
                    {"id": 2, "name": "Second Item", "active": False},
                    {"id": 3, "name": "Third Item", "active": True},
                ],
            },
            "dict_struct": {
                "category": "test",
                "descriptions": ["test1", "test2", "test3"],
            },
            "Struct": {
                "category": "test",
                "descriptions": ["test1", "test2", "test3"],
            },
            "PydanticStruct": {
                "category": "test",
                "descriptions": ["test1", "test2", "test3"],
            },
            "optional_struct": {
                "required_field": "value",
                "optional_str": "string value",
                "optional_int": 123,
                "optional_float": 45.67,
                "optional_bool": True,
                "union_syntax": None,
            },
            "literal_struct": {
                "status": "active",
                "role": "admin",
                "priority": 1,
            },
            "nested_dict_struct": {
                "name": "John Doe",
                "age": 30,
                "contact": {
                    "email": "john@example.com",
                    "phone": "123-456-7890",
                    "address": {
                        "street": "123 Main St",
                        "city": "Anytown",
                        "country": "USA",
                    },
                },
            },
            "list_of_dicts_struct": {
                "name": "Jane Smith",
                "skills": [
                    {"name": "Python", "level": 5},
                    {"name": "JavaScript", "level": 4},
                ],
            },
            "Person": {
                "name": "Alice Johnson",
                "age": 28,
                "contact": {
                    "email": "alice@example.com",
                    "phone": "987-654-3210",
                    "address": {
                        "street": "456 Oak Ave",
                        "city": "Somewhere",
                        "country": "Canada",
                    },
                    "contact_type": "primary",
                },
            },
            "PersonModel": {
                "name": "Bob Brown",
                "age": 35,
                "contact": {
                    "email": "bob@example.com",
                    "phone": "555-123-4567",
                    "address": {
                        "street": "789 Pine St",
                        "city": "Elsewhere",
                        "country": "UK",
                    },
                },
            },
            "super_complex_struct": {
                "user": {
                    "name": "Charlie Davis",
                    "age": 42,
                    "contact": {
                        "email": "charlie@example.com",
                        "phone": "111-222-3333",
                        "address": {
                            "street": "101 Maple Dr",
                            "city": "Nowhere",
                            "country": "Australia",
                        },
                    },
                },
                "status": "active",
                "permissions": [
                    {"resource": "files", "access": "read"},
                    {"resource": "users", "access": "admin"},
                ],
                "settings": {
                    "theme": "dark",
                    "notifications": {
                        "email": True,
                        "sms": False,
                        "frequency": "weekly",
                    },
                },
            },
            "list_of_class_struct": {
                "title": "My Item Collection",
                "items": [
                    {"id": 1, "name": "First Item", "active": True},
                    {"id": 2, "name": "Second Item", "active": False},
                ],
            },
            "dict_mapping_struct": {
                "name": "Student Results",
                "scores": {"math": 95, "science": 87, "history": 78},
            },
            "ValidatedModel": {
                "name": "John Doe",
                "age": 35,
                "email": "john.doe@example.com",
                "tags": ["developer", "python"],
                "score": 8.5,
            },
            "AppConfiguration": {
                "app_name": "TestApp",
                "version": "1.0.0",
                "settings": {
                    "enabled": True,
                    "values": {
                        "timeout": 30,
                        "rate_limit": 5.5,
                        "api_key": "abc123",
                        "debug_mode": False,
                        "cache": None,
                    },
                    "description": "Test configuration settings",
                },
            },
            "UserProfile": {
                "name": "John Smith",
                "email": "john@example.com",
                "age": 30,
            },
            "list_instance_primitives_struct": {
                "product_names": ["laptop", "mouse", "keyboard"],
                "scores": [95, 87, 78],
                "prices": [999.99, 25.50, 75.00],
                "active_flags": [True, False, True],
                "mixed_numbers": [10, 15.5, 20],
                "optional_tags": ["sale", None, "premium"],
            },
            "list_instance_classes_struct": {
                "title": "Product Catalog",
                "products": [
                    {"id": 1, "name": "Laptop", "price": 999.99},
                    {"id": 2, "name": "Mouse", "price": 25.50},
                    {"id": 3, "name": "Keyboard", "price": 75.00},
                ],
                "categories": ["electronics", "electronics", "electronics"],
            },
        }

        # Define invalid examples for each structure
        invalid_examples = {
            "ClassWithListOfClasses": [
                {"name": "Test"},  # Missing required fields
                {"name": "Test", "description": "Test"},  # Missing items field
                {"name": "Test", "items": []},  # Missing description field
                {
                    "name": "Test",
                    "description": "Test",
                    "items": {},
                },  # Wrong type for items
                {
                    "name": "Test",
                    "description": "Test",
                    "items": [{"id": "1", "name": "Test", "active": True}],
                },  # Wrong type for id (must be int)
                {
                    "name": "Test",
                    "description": "Test",
                    "items": [{"name": "Test", "active": True}],
                },  # Missing required field in item
            ],
            "dict_struct": [
                {"category": None},  # Missing required field
                {"descriptions": ["test"]},  # Missing required field
                {"category": "test", "descriptions": "not a list"},  # Wrong type
                [],  # Not a dict
            ],
            "Struct": [
                {"category": None},  # Missing required field
                {"descriptions": ["test"]},  # Missing required field
                {"category": "test", "descriptions": "not a list"},  # Wrong type
                [],  # Not a dict
            ],
            "PydanticStruct": [
                {"category": None},  # Missing required field
                {"descriptions": ["test"]},  # Missing required field
                {"category": "test", "descriptions": "not a list"},  # Wrong type
                [],  # Not a dict
            ],
            "optional_struct": [
                {},  # Missing required field
                {"optional_str": "test"},  # Missing required field
                {"required_field": 123},  # Wrong type for required field
                {
                    "required_field": "value",
                    "optional_str": 123,
                },  # Wrong type for optional field
            ],
            "literal_struct": [
                {"status": "unknown"},  # Invalid literal value
                {"role": "superuser"},  # Invalid literal value
                {"priority": 4},  # Invalid literal value
                {
                    "status": "active",
                    "role": "admin",
                    "priority": "1",
                },  # Wrong type for literal
            ],
            "nested_dict_struct": [
                {"name": "John", "contact": {}},  # Missing required field
                {"name": "John", "age": "30"},  # Wrong type
                {
                    "name": "John",
                    "age": 30,
                    "contact": {"email": "john@example.com"},
                },  # Incomplete nested structure
            ],
            "list_of_dicts_struct": [
                {"name": "Jane", "skills": "Python"},  # Wrong type for list
                {
                    "name": "Jane",
                    "skills": [{"name": "Python"}],
                },  # Missing required field in list item
                {
                    "name": "Jane",
                    "skills": [{"level": 5}],
                },  # Missing required field in list item
            ],
            "Person": [
                {"name": "Alice", "age": "28"},  # Wrong type
                {"name": "Alice", "contact": {}},  # Incomplete nested structure
                {
                    "name": "Alice",
                    "age": 28,
                    "contact": {"email": "alice@example.com"},
                },  # Incomplete nested structure
            ],
            "PersonModel": [
                {"name": "Bob", "age": "35"},  # Wrong type
                {"name": "Bob", "contact": {}},  # Incomplete nested structure
                {
                    "name": "Bob",
                    "age": 35,
                    "contact": {"email": "bob@example.com"},
                },  # Incomplete nested structure
            ],
            "super_complex_struct": [
                {"user": {}, "status": "active"},  # Incomplete user structure
                {
                    "user": {"name": "Charlie", "age": 42},
                    "status": "unknown",
                },  # Invalid literal value
                {
                    "user": {"name": "Charlie", "age": 42, "contact": {}},
                    "status": "active",
                },  # Incomplete contact
            ],
            "list_of_class_struct": [
                {"title": "Collection"},  # Missing items field
                {"items": []},  # Missing title field
                {"title": "Collection", "items": {}},  # Wrong type for items
                {
                    "title": "Collection",
                    "items": [{"id": "1", "name": "Item", "active": True}],
                },  # Wrong type for id (must be int)
                {
                    "title": "Collection",
                    "items": [{"name": "Item", "active": True}],
                },  # Missing required field in item
            ],
            "dict_mapping_struct": [
                {"name": "Results"},  # Missing required fields
                {"name": "Results", "scores": []},  # Wrong type for scores
                {
                    "name": "Results",
                    "scores": {"math": "A"},
                },  # Wrong value type in scores
            ],
            "ValidatedModel": [
                {
                    "name": "John Doe",
                    "age": 35,
                    "email": "john.doe@example.com",
                },  # some fields are missing
                {
                    "name": 1,
                    "age": 17,
                    "email": "john.doe@example.com",
                    "tags": ["one", "two", "three", "four", "five", "six"],
                    "score": 11.0,
                },  # name field is of wrong type
            ],
            "AppConfiguration": [
                {
                    "app_name": "TestApp",
                    "version": "1.0.0",
                },  # Incomplete nested structure, missing fields
                {
                    "app_name": "TestApp",
                    "version": "1.0.0",
                    "settings": {
                        "enabled": True,
                        "description": "Test settings",
                    },
                },  # Missing values dict
            ],
            "UserProfile": [
                {"name": "John", "age": 30},  # Missing email
            ],
            "list_instance_primitives_struct": [
                {"product_names": "not a list"},  # Wrong type for list field
                {"scores": [95, "invalid"]},  # Wrong type in list element
                {"prices": [999.99, 25.50]},
                {"active_flags": ["True", False]},  # Wrong type in list element
                {"mixed_numbers": [10, "invalid"]},  # Wrong type in union
                {
                    "product_names": [],
                    "scores": [],
                    "prices": [],
                    "active_flags": [],
                    "mixed_numbers": [],
                    "optional_tags": "not a list",
                },  # Wrong type for optional list
            ],
            "list_instance_classes_struct": [
                {"products": []},  # Missing required fields
                {"title": "Catalog"},  # Missing required fields
                {"title": "Catalog", "products": {}},  # Wrong type for products
                {
                    "title": "Catalog",
                    "products": [{"id": "1", "name": "Test", "price": 10.0}],
                },  # Wrong type for id
                {
                    "title": "Catalog",
                    "products": [{"name": "Test", "price": 10.0}],
                },  # Missing required field in product
                {
                    "title": "Catalog",
                    "products": [],
                    "categories": ["invalid"],
                },  # Invalid literal value
            ],
        }

        # Map structure objects to their names
        structure_map = {
            "dict_struct": dict_struct,
            "Struct": Struct,
            "PydanticStruct": PydanticStruct,
            "optional_struct": optional_struct,
            "literal_struct": literal_struct,
            "nested_dict_struct": nested_dict_struct,
            "list_of_dicts_struct": list_of_dicts_struct,
            "Person": Person,
            "PersonModel": PersonModel,
            "super_complex_struct": super_complex_struct,
            "list_of_class_struct": list_of_class_struct,
            "ClassWithListOfClasses": ClassWithListOfClasses,
            "dict_mapping_struct": dict_mapping_struct,
            "ValidatedModel": validated_model_struct,
            "AppConfiguration": AppConfiguration,
            "UserProfile": UserProfile,
            "list_instance_primitives_struct": list_instance_primitives_struct,
            "list_instance_classes_struct": list_instance_classes_struct,
        }

        # Utility function to test a structure
        def test_structure(struct_name: str, structure: Any) -> JsonObjectConcept:
            """
            Utility function to test a JsonObjectConcept structure.

            :param struct_name: Name of the structure being tested.
            :type struct_name: str
            :param structure: The structure definition to test.
            :type structure: Any
            :return: The JsonObjectConcept instance.
            :rtype: JsonObjectConcept
            """
            # Create concept with the structure
            concept = JsonObjectConcept(
                name=f"{struct_name} Concept",
                description=f"Testing {struct_name} structure",
                structure=structure,
                examples=[JsonObjectExample(content=valid_examples[struct_name])],
            )

            # Test instance serialization and cloning
            self.check_instance_serialization_and_cloning(concept)

            # Test valid extracted data
            validator = concept._get_structure_validator()
            valid_data = validator.model_validate(valid_examples[struct_name])
            assert valid_data is not None

            # Test invalid extracted data
            for invalid_example in invalid_examples[struct_name]:
                with pytest.raises(ValueError):
                    validator.model_validate(invalid_example)

            # Test name assignment
            concept.name = f"Updated {struct_name}"
            with pytest.raises(ValueError):
                concept.name = True  # type: ignore
            assert concept.name == f"Updated {struct_name}"

            # Test invalid extracted items
            with pytest.raises(ValueError):
                concept.extracted_items = [1, True]  # type: ignore

            return concept

        # Test each structure
        for struct_name, structure in structure_map.items():
            test_structure(struct_name, structure)

        # Test invalid structure cases
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={str: str},  # invalid mapping  # type: ignore
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={int: "integer"},  # invalid mapping  # type: ignore
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={},  # empty mapping
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={"random": Exception},  # invalid value type
            )

        # Test with invalid type hints as structure values
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Optional Type",
                description="Test with invalid Optional type",
                structure={"field": Optional[Person]},  # noqa: UP045
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Dict Type",
                description="Test with invalid dict type",
                structure={"field": dict[str, Item]},  # cls as value
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Dict Type",
                description="Test with invalid dict type",
                structure={"field": dict[Item, str]},  # cls as key
            )

        # Test with vision extractor and references (should fail)
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Vision Concept with References",
                description="Testing vision concept with references",
                structure={"test": str},
                modality="image",
                add_references=True,
            )

        # Test with Any type (should fail)
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Any Concept",
                description="Testing Any type",
                structure={"test": Any},  # unclear type
            )

        # Test with raw list and dicts (should fail)
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Raw List and Dicts Concept",
                description="Testing raw list and dicts",
                structure={"test": list},  # unclear type
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Raw List and Dicts Concept",
                description="Testing raw list and dicts",
                structure={"test": dict},  # unclear type
            )

        # Test with unclear types in dataclasses (should fail)
        with pytest.raises(ValueError):

            @dataclass
            class StudentWithUnclearAny(JsonObjectClassStruct):
                name: str
                courses: Any  # unclear type - should fail

            JsonObjectConcept(
                name="Unclear Any Class Concept",
                description="Testing unclear any type in dataclass",
                structure=StudentWithUnclearAny,
            )

        with pytest.raises(ValueError):

            @dataclass
            class StudentWithUnclearDict(JsonObjectClassStruct):
                name: str
                grades: dict  # unclear type - should fail

            JsonObjectConcept(
                name="Unclear Dict Class Concept",
                description="Testing unclear dict type in dataclass",
                structure=StudentWithUnclearDict,
            )

        # Test with vision extractor and justifications (should work)
        vision_concept = JsonObjectConcept(
            name="Vision Concept with Justifications",
            description="Testing vision concept with justifications",
            structure={
                "category": str,
                "description": str,
                "score": int | float | None,
            },
            modality="image",
            add_justifications=True,
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(vision_concept)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_json_object_concept"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_extract_complex_json_object_concept(self):
        """
        Tests the extraction of a complex JsonObjectConcept from a text file, validating
        that the structure representation is properly included in the prompt and that
        the concept is correctly extracted.
        """

        @dataclass
        class Address(JsonObjectClassStruct):
            street: str
            city: str
            country: str

        @dataclass
        class Contact(JsonObjectClassStruct):
            email: str
            phone: str
            address: Address
            contact_type: (
                Literal["primary", "secondary", "emergency"]
                | Literal["union", "literal"]
                | None
            )  # intentionally messed up type hint for testing

        @dataclass
        class PersonModel(JsonObjectClassStruct):
            name: str
            age: int
            contact: Contact

        # Define an Item class for use in list[cls]
        @dataclass
        class UserItem(JsonObjectClassStruct):
            id: int
            name: str
            description: str
            is_active: bool

        # Define the complex structure
        super_complex_struct = {
            "user": PersonModel,
            "status": Literal["active", "inactive"],
            "permissions": [
                {"resource": str, "access": Literal["read", "write", "admin"]}
            ],
            "settings": {
                "theme": str,
                "notifications": {
                    "email": bool,
                    "sms": bool,
                    "frequency": Literal["daily", "weekly", "monthly"],
                },
            },
            "related_items": [
                UserItem
            ],  # Add [cls] nested structure (equivalent to list[cls])
            "security_level": Literal["Basic", "Advanced", "Enterprise"] | None,
        }

        # Create a JsonObjectConcept with this structure
        complex_concept = JsonObjectConcept(
            name="User System Profile",
            description="Comprehensive user profile with system access permissions and settings",
            structure=super_complex_struct,
            examples=[
                JsonObjectExample(
                    content={
                        "user": {
                            "name": "Charlie Davis",
                            "age": 42,
                            "contact": {
                                "email": "charlie@example.com",
                                "phone": "111-222-3333",
                                "address": {
                                    "street": "101 Maple Dr",
                                    "city": "Nowhere",
                                    "country": "Australia",
                                },
                                "contact_type": "primary",
                            },
                        },
                        "status": "active",
                        "permissions": [
                            {"resource": "files", "access": "read"},
                            {"resource": "users", "access": "admin"},
                        ],
                        "settings": {
                            "theme": "dark",
                            "notifications": {
                                "email": True,
                                "sms": False,
                                "frequency": "weekly",
                            },
                        },
                        "related_items": [
                            {
                                "id": 1,
                                "name": "Document Review",
                                "description": "Annual document review task",
                                "is_active": True,
                            },
                            {
                                "id": 2,
                                "name": "Access Audit",
                                "description": "System access verification",
                                "is_active": False,
                            },
                        ],
                        "security_level": "Advanced",
                    }
                )
            ],
        )

        # Load the test text file as a Document
        with open(
            os.path.join(
                get_project_root_path(),
                "tests",
                "other_files",
                "complex_user_profile.txt",
            ),
            encoding="utf-8",
        ) as f:
            text_content = f.read()

        # Create a Document from the text content and add the complex concept
        document = Document(raw_text=text_content)
        document.concepts = [complex_concept]

        # Get the structure representation that should be included in the prompt
        structure_str = complex_concept._format_structure_in_prompt()

        # Verify that the structure representation contains key elements
        assert '"user"' in structure_str
        assert '"name": str' in structure_str
        assert '"age": int' in structure_str
        assert '"status": "active" or "inactive"' in structure_str
        assert '"permissions"' in structure_str
        assert '"resource": str' in structure_str
        assert '"access": "read" or "write" or "admin"' in structure_str
        assert '"theme": str' in structure_str
        assert '"email": bool' in structure_str
        assert '"frequency": "daily" or "weekly" or "monthly"' in structure_str
        assert (
            '"security_level": "Basic" or "Advanced" or "Enterprise" or null'
            in structure_str
        )
        assert (
            '"contact_type": "primary" or "secondary" or "emergency" or "union" or "literal" or null'
            in structure_str
        )

        # Verify the list[cls] structure representation
        assert '"related_items"' in structure_str
        assert '"id": int' in structure_str
        assert '"description": str' in structure_str
        assert '"is_active": bool' in structure_str

        # Extract the concept
        extracted_concepts = self.llm_extractor.extract_concepts_from_document(document)

        # Verify prompt content from the LLM call log
        prompt_string = self.llm_extractor.get_usage()[-1].usage.calls[-1].prompt
        assert structure_str in prompt_string

        # Validate extraction results
        assert len(extracted_concepts) == 1
        extracted_concept = extracted_concepts[0]
        assert extracted_concept.extracted_items
        extracted_item = extracted_concept.extracted_items[0]
        extracted_data = extracted_item.value

        # Validate the extracted data structure
        assert "user" in extracted_data
        assert extracted_data["user"]["name"] == "Charlie Davis"
        assert extracted_data["user"]["age"] == 42
        assert extracted_data["user"]["contact"]["email"] == "charlie@example.com"
        assert extracted_data["user"]["contact"]["address"]["city"] == "Nowhere"
        assert extracted_data["user"]["contact"]["contact_type"] == "primary"
        assert extracted_data["status"] == "active"
        assert isinstance(extracted_data["permissions"], list)
        assert len(extracted_data["permissions"]) >= 1
        assert "resource" in extracted_data["permissions"][0]
        assert "access" in extracted_data["permissions"][0]
        assert "settings" in extracted_data
        assert "theme" in extracted_data["settings"]
        assert "notifications" in extracted_data["settings"]
        assert "email" in extracted_data["settings"]["notifications"]
        assert "frequency" in extracted_data["settings"]["notifications"]
        assert "security_level" in extracted_data
        assert extracted_data["security_level"] == "Advanced"

        # Validate the list[cls] structure in the extracted data
        assert "related_items" in extracted_data
        assert isinstance(extracted_data["related_items"], list)
        assert len(extracted_data["related_items"]) >= 1
        assert "id" in extracted_data["related_items"][0]
        assert "name" in extracted_data["related_items"][0]
        assert "description" in extracted_data["related_items"][0]
        assert "is_active" in extracted_data["related_items"][0]

        # Log the extracted item for debugging
        self.log_extracted_items_for_instance(extracted_concept)

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(complex_concept)
        self.check_instance_serialization_and_cloning(document)

        check_locals_memory_usage(
            locals(), test_name="test_extract_complex_json_object_concept"
        )

    @memory_profile_and_capture
    def test_init_and_validate_date_concept(self):
        """
        Tests the initialization of the DateConcept class with valid and invalid input parameters.
        """

        # Valid initialization with default date format
        default_date_concept = DateConcept(
            name="Contract Date",
            description="Date when the contract was signed",
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            DateConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                extra_param=True,  # type: ignore
            )

        # Test attaching extracted items
        # Create a date string that matches the default format
        date_str = "02-03-2025"
        date_obj = default_date_concept._string_to_date(date_str)
        default_date_concept.extracted_items = [_DateItem(value=date_obj)]  # ty: ignore[invalid-assignment]

        # Test with invalid date string
        with pytest.raises(ValueError):
            date_str = "2025-01-01"
            date_obj = default_date_concept._string_to_date(date_str)

        # Test with invalid items
        with pytest.raises(ValueError):
            default_date_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _StringItem(value="15/01/2023")
            ]  # must be _DateItem

        with pytest.raises(ValueError):
            _DateItem(value="01-01-2025")  # must be a date object  # type: ignore

        # Test with justifications and references
        date_concept_with_refs = DateConcept(
            name="Date with refs",
            description="Test date concept with references",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(default_date_concept)
        self.check_instance_serialization_and_cloning(date_concept_with_refs)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_date_concept"
        )

    @memory_profile_and_capture
    def test_init_and_validate_label_concept(self):
        """
        Tests the initialization of the LabelConcept class with valid and invalid input parameters.
        """
        # Valid initialization with multi-class classification
        multi_class_concept = LabelConcept(
            name="Document Type",
            description="Classify the type of legal document",
            labels=["NDA", "Consultancy Agreement", "Privacy Policy", "Other"],
            classification_type="multi_class",
            add_justifications=True,
        )

        # Valid initialization with multi-label classification
        multi_label_concept = LabelConcept(
            name="Content Topics",
            description="Identify all relevant topics covered in the document",
            labels=["Finance", "Legal", "Technology", "HR", "Operations", "Marketing"],
            classification_type="multi_label",
            model_role="reasoner",
            add_references=True,
            reference_depth="sentences",
        )

        # Test with invalid extra parameters
        with pytest.raises(ValueError):
            LabelConcept(
                name="Invalid Params",
                description="Test invalid parameters",
                labels=["A", "B", "C"],
                extra_param=True,  # type: ignore
            )

        # Test with insufficient labels (less than 2)
        with pytest.raises(ValueError):
            LabelConcept(
                name="Too Few Labels",
                description="Test with only one label",
                labels=["OnlyOne"],
            )

        # Test with duplicate labels
        with pytest.raises(ValueError):
            LabelConcept(
                name="Duplicate Labels",
                description="Test with duplicate labels",
                labels=["A", "B", "a", "C"],  # duplicate labels (case-insensitive)
            )

        # Test with invalid classification type
        with pytest.raises(ValueError):
            LabelConcept(
                name="Invalid Classification",
                description="Test invalid classification type",
                labels=["A", "B", "C"],
                classification_type="invalid_type",  # type: ignore
            )

        # Test with invalid label types (non-string types in labels list)
        with pytest.raises(ValueError):
            LabelConcept(
                name="Invalid Label Types",
                description="Test with non-string label types",
                labels=["A", 123, "C"],  # Integer in labels list  # type: ignore
            )

        # Test attaching extracted items for multi-class
        multi_class_item = _LabelItem(value=["NDA"])
        multi_class_concept.extracted_items = [multi_class_item]

        # Test attaching extracted items for multi-label
        multi_label_item = _LabelItem(value=["Finance", "Legal"])
        multi_label_concept.extracted_items = [multi_label_item]

        # Test with invalid items (wrong type)
        with pytest.raises(ValueError):
            multi_class_concept.extracted_items = [  # ty: ignore[invalid-assignment]
                _StringItem(value="NDA")
            ]  # must be _LabelItem

        # Test with invalid labels in extracted item
        with pytest.raises(ValueError):
            multi_class_concept.extracted_items = [
                _LabelItem(value=["InvalidLabel"])
            ]  # label not in predefined set

        # Test multi-class constraint validation (only one label allowed)
        with pytest.raises(ValueError):
            multi_class_concept._process_item_value({"labels": ["NDA", "Other"]})

        # Test multi-label allows multiple labels
        processed_labels = multi_label_concept._process_item_value(
            {"labels": ["Finance", "Legal", "Technology"]}
        )
        assert processed_labels == ["Finance", "Legal", "Technology"]

        # Test _format_labels_in_prompt property
        assert (
            multi_class_concept._format_labels_in_prompt
            == '["NDA", "Consultancy Agreement", "Privacy Policy", "Other"]'
        )
        assert (
            multi_label_concept._format_labels_in_prompt
            == '["Finance", "Legal", "Technology", "HR", "Operations", "Marketing"]'
        )

        # Test _item_type_in_prompt property
        assert multi_class_concept._item_type_in_prompt == "dict"
        assert multi_label_concept._item_type_in_prompt == "dict"

        # Test with justifications and references
        concept_with_refs = LabelConcept(
            name="Concept with refs",
            description="Test concept with references",
            labels=["Category1", "Category2", "Category3"],
            classification_type="multi_label",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
            singular_occurrence=True,
        )

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(multi_class_concept)
        self.check_instance_serialization_and_cloning(multi_label_concept)
        self.check_instance_serialization_and_cloning(concept_with_refs)

        check_locals_memory_usage(
            locals(), test_name="test_init_and_validate_label_concept"
        )

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    @memory_profile_and_capture
    def test_extract_label_concept(self, llm: DocumentModelGroup | DocumentLLM):
        """
        Tests for label concept extraction from document using LLMs.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        # Create a document with content suitable for label classification
        document = Document(raw_text=get_test_document_text())

        # Test multi-class classification
        document_type_concept = LabelConcept(
            name="Document Type",
            description="Classify the type of legal document",
            labels=[
                "NDA",
                "Consultancy Agreement",
                "Privacy Policy",
                "Employment Contract",
                "Other",
            ],
            classification_type="multi_class",
            add_justifications=True,
            add_references=True,
            reference_depth="paragraphs",
            singular_occurrence=True,
        )

        # Test multi-label classification
        content_topics_concept = LabelConcept(
            name="Content Topics",
            description="Identify all relevant topics covered in this legal document",
            labels=[
                "Confidentiality",
                "Technology",
                "Finance",
                "Legal Compliance",
                "Third Party Relations",
                "Contract Terms",
            ],
            classification_type="multi_label",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Assign concepts to document
        document_concepts = [document_type_concept, content_topics_concept]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(2 if llm.is_group else 2),
            expected_n_calls_with_concurrency=(2 if llm.is_group else 2),
            expected_n_calls_1_item_per_call=(2 if llm.is_group else 2),
            func=llm.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )
        self.check_extra_data_in_extracted_items(document)

        # Verify extraction results
        extracted_document_type = document.get_concept_by_name("Document Type")
        extracted_content_topics = document.get_concept_by_name("Content Topics")

        # Check that items were extracted
        assert extracted_document_type.extracted_items
        assert extracted_content_topics.extracted_items

        # Verify multi-class constraint (only one label)
        document_type_item = extracted_document_type.extracted_items[0]
        assert len(document_type_item.value) == 1
        assert document_type_item.value[0] in document_type_concept.labels

        # Verify multi-label allows multiple labels
        content_topics_item = extracted_content_topics.extracted_items[0]
        assert len(content_topics_item.value) >= 1
        for label in content_topics_item.value:
            assert label in content_topics_concept.labels

        # Verify justifications and references are present
        assert document_type_item.justification
        assert document_type_item.reference_paragraphs
        assert content_topics_item.justification
        assert content_topics_item.reference_sentences

        # Verify that extraction references resolve to positions in the document,
        # in document order
        ref_para_indices = [
            document.get_paragraph_index(p)
            for p in document_type_item.reference_paragraphs
        ]
        assert ref_para_indices == sorted(ref_para_indices)
        ref_sent_positions = [
            document.get_sentence_index(s)
            for s in content_topics_item.reference_sentences
        ]
        assert ref_sent_positions == sorted(ref_sent_positions)

        # Log extracted items for verification
        self.log_extracted_items_for_instance(extracted_document_type)
        self.log_extracted_items_for_instance(extracted_content_topics)

        # Test scenario where no labels apply
        # Note: The framework allows empty extracted_items when multi-label classification
        # is used and no predefined labels apply. This is the expected behavior
        # as documented in LabelConcept.
        irrelevant_concept = LabelConcept(
            name="Technical Specifications",
            description="Identify technical specifications mentioned in the document",
            labels=[
                "Cooking Recipes",
                "Nutrition Information",
            ],
            classification_type="multi_label",
        )

        document.add_concepts([irrelevant_concept])
        llm.extract_concepts_from_document(
            document,
            from_concepts=[document.get_concept_by_name("Technical Specifications")],
        )

        # Should have empty extracted_items when no labels apply
        assert not document.get_concept_by_name(
            "Technical Specifications"
        ).extracted_items

        # For multi-class classification, a label is always returned, even if
        # it does not perfectly fit the content. This is the expected behavior
        # as documented in LabelConcept.
        irrelevant_concept = LabelConcept(
            name="Technical Specifications",
            description="Identify technical specifications mentioned in the document",
            labels=[
                "Cooking Recipes",
                "Nutrition Information",
            ],
            classification_type="multi_class",
        )
        document.concepts = [irrelevant_concept]
        llm.extract_concepts_from_document(document)
        technical_specifications_concept = document.get_concept_by_name(
            "Technical Specifications"
        )
        assert technical_specifications_concept.extracted_items
        assert (
            len(technical_specifications_concept.extracted_items[0].value) == 1
        )  # should always return a single label
        assert (
            technical_specifications_concept.extracted_items[0].value[0]
            in irrelevant_concept.labels
        )

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_concepts_from_document(document)

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(document)

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

        check_locals_memory_usage(locals(), test_name="test_extract_label_concept")

    @memory_profile_and_capture
    def test_init_example(self):
        """
        Tests the initialization of the example classes.
        """
        example = StringExample(content="Test")
        StringConcept(
            name="test",
            description="test",
            examples=[example],
        )
        with pytest.raises(ValueError):
            # Examples with same unique ID
            StringConcept(
                name="test",
                description="test",
                examples=[example, example],
            )
        with pytest.raises(ValueError):
            StringExample(content=1)  # type: ignore
        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(example)

        example = JsonObjectExample(content={"category": "test", "description": "test"})
        JsonObjectConcept(
            name="test",
            description="test",
            structure={"category": str, "description": str},
            examples=[example],
        )
        with pytest.raises(ValueError):
            # Examples with same unique ID
            JsonObjectConcept(
                name="test",
                structure={"category": str, "description": str},
                description="test",
                examples=[example, example],
            )
        with pytest.raises(ValueError):
            # Example structure does not align with the expected structure of the concept
            JsonObjectConcept(
                name="test",
                structure={"category": int, "value": float},
                description="test",
                examples=[example],
            )
        with pytest.raises(ValueError):
            JsonObjectExample(content={"category": str, "valid": bool})
        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(example)

        check_locals_memory_usage(locals(), test_name="test_init_example")

    @memory_profile_and_capture
    def test_init_item(self):
        """
        Tests the initialization of the extracted item classes.
        """
        _StringItem(value="Random string")
        _BooleanItem(value=True)
        _IntegerItem(value=1)
        _FloatItem(value=1.0)
        _IntegerOrFloatItem(value=1)
        _IntegerOrFloatItem(value=1.0)
        _JsonObjectItem(value={"hello": "world"})
        _LabelItem(value=["NDA"])
        with pytest.raises(TypeError):
            _ExtractedItem(value=1)
        with pytest.raises(ValueError):
            _StringItem()
        with pytest.raises(ValueError):
            _BooleanItem(value=1)  # type: ignore
        with pytest.raises(ValueError):
            _BooleanItem()
        with pytest.raises(ValueError):
            _BooleanItem(value="True")  # type: ignore
        with pytest.raises(ValueError):
            _IntegerOrFloatItem()
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=int)  # type: ignore
        with pytest.raises(ValueError):
            _JsonObjectItem()
        with pytest.raises(ValueError):
            _JsonObjectItem(value={})
        with pytest.raises(ValueError):
            _JsonObjectItem(value={"hello": {}})
        with pytest.raises(ValueError):
            _StringItem(
                value="Random string",
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            _LabelItem(value=[])
        with pytest.raises(ValueError):
            _LabelItem(value="NDA")  # type: ignore

        # List field items' unique IDs
        para = Paragraph(raw_text="Test")
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=1.0, reference_paragraphs=[para, para, para])  # type: ignore

        # Frozen state
        item = _IntegerOrFloatItem(value=1)
        with pytest.raises(ValueError):
            item.value = 2.0

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(item)

        check_locals_memory_usage(locals(), test_name="test_init_item")

    @pytest.mark.parametrize("context", [document, extraction_pipeline])
    @memory_profile_and_capture(max_memory=2500.0)  # for testing larger SaT models
    def test_init_document_and_extraction_pipeline(
        self, context: Document | ExtractionPipeline
    ):
        """
        Tests different initialization scenarios and validations associated with
        the `Document` and `ExtractionPipeline` classes.
        """
        context = context.clone()  # clone for method-scoped state modification

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(context)

        # Document initialization
        if isinstance(context, Document):
            for sat_model_id in [
                "sat-3l-sm",
                # "sat-6l-sm",
                # "sat-12l-sm",
            ]:
                for lang in ["en", "ua", "zh"]:  # EN and other langs
                    document = Document(
                        raw_text=get_test_document_text(lang=lang),
                        sat_model_id=sat_model_id,
                        paragraph_segmentation_mode="sat",
                        pre_segment_sentences=True,
                    )
                    assert document.paragraphs  # to be segmented from text
                    assert all(
                        i.sentences for i in document.paragraphs
                    )  # segmented from paragraphs since `pre_segment_sentences` is True
            Document(
                raw_text="Random text",
                paragraphs=[
                    Paragraph(raw_text="Random text"),
                ],
            )
            document = Document(
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                    Paragraph(raw_text="Random text 3"),
                ],
            )
            assert document.raw_text  # to be populated from paragraphs
            assert not any(
                i.sentences for i in document.paragraphs
            )  # sentences not segmented yet since `pre_segment_sentences` is False (default)
            with pytest.raises(ValueError):
                document.raw_text = "Random text 1"  # cannot be set once populated
            with pytest.raises(ValueError):
                document.paragraphs = [
                    Paragraph(raw_text="Random text 1"),
                ]  # cannot be set once populated
            Document(
                images=[
                    self.test_img_png_invoice,
                ]
            )
            with pytest.raises(ValueError):
                Document()
            with pytest.raises(ValueError):
                # Paragraphs do not match the text
                Document(
                    raw_text="XYZ",
                    paragraphs=[
                        Paragraph(raw_text="Random text 1"),
                        Paragraph(raw_text="Random text 2"),
                        Paragraph(raw_text="Random text 3"),
                    ],
                )
            with pytest.raises(ValueError):
                # Paragraphs are not ordered according to their appearance in the text
                Document(
                    raw_text="AAA BBB CCC",
                    paragraphs=[
                        Paragraph(raw_text="CCC"),
                        Paragraph(raw_text="BBB"),
                        Paragraph(raw_text="AAA"),
                    ],
                )
            # Paragraphs are ordered according to their appearance in the text,
            # but some paragraphs are duplicated
            Document(
                raw_text="AAA BBB CCC AAA DDD EEE BBB FFF",
                paragraphs=[
                    Paragraph(raw_text="AAA"),
                    Paragraph(raw_text="BBB"),
                    Paragraph(raw_text="CCC"),
                    Paragraph(raw_text="AAA"),
                    Paragraph(raw_text="DDD"),
                    Paragraph(raw_text="EEE"),
                    Paragraph(raw_text="BBB"),
                    Paragraph(raw_text="FFF"),
                ],
            )
            with pytest.raises(ValueError):
                # Paragraph duplicates exist but not all duplicates are matched
                Document(
                    raw_text="AAA BBB CCC",
                    paragraphs=[
                        Paragraph(raw_text="AAA"),
                        Paragraph(raw_text="BBB"),
                        Paragraph(raw_text="CCC"),
                        Paragraph(raw_text="BBB"),
                    ],
                )
            with pytest.raises(ValueError):
                Document(
                    raw_text="Random text",
                    extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
                )
            Document(
                raw_text="Random text 1\n\nRandom text 2",
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                ],
                images=[
                    self.test_img_png_invoice,
                ],
            )
            # List field items' unique IDs
            para = Paragraph(raw_text="Random text")
            with pytest.raises(ValueError):
                Document(paragraphs=[para, para, para])
            with pytest.raises(ValueError):
                Document(
                    images=[
                        self.test_img_png_invoice,
                        self.test_img_png_invoice,
                    ]
                )
            # Test with non-empty text but containing only control chars
            with pytest.raises(ValueError, match="control characters"):
                Document(raw_text=" \u200c ")  # zero-width non-joiner

        # Extraction pipeline initialization
        elif isinstance(context, ExtractionPipeline):
            ExtractionPipeline()  # works as we can interactive add aspects and concepts after initialization
            # Pipeline assignment
            document = self.document.clone()
            document.assign_pipeline(context)
            with pytest.raises(RuntimeError):
                document.assign_pipeline(
                    context
                )  # document already has aspects and concepts
            document.aspects = []
            document.concepts = []
            document.assign_pipeline(context)
            assert context.aspects is not document.aspects
            assert context.concepts is not document.concepts
            # Pipeline params
            with pytest.raises(ValueError):
                ExtractionPipeline(
                    aspects=[
                        Aspect(
                            name="Liability",
                            description="Clauses describing liability of the parties",
                        )
                    ],
                    extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
                )
            ExtractionPipeline(
                aspects=[
                    Aspect(
                        name="Liability",
                        description="Clauses describing liability of the parties",
                        add_justifications=True,
                    )
                ]
            )
            ExtractionPipeline(
                concepts=[
                    StringConcept(
                        name="Business Information",
                        description="Categories of Business Information",
                        add_justifications=True,
                    ),
                ]
            )

        # Document and extraction pipeline initialization
        # Document aspects
        document_aspects = [
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
            ),
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                add_justifications=True,
            ),
        ]
        context.aspects = document_aspects
        assert context.aspects is not document_aspects

        # Validate assignment
        with pytest.raises(ValueError, match="non-empty list"):
            context.add_aspects([])  # empty list
        with pytest.raises(ValueError, match="non-empty list"):
            context.add_concepts([])  # empty list
        with pytest.raises(ValueError):
            context.add_aspects(
                [  # ty: ignore[invalid-argument-type]
                    "Random string",
                ]  # invalid aspect type
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="liability",
                        description="clauses describing liability of the parties",
                        add_justifications=True,
                    )
                ]  # duplicate aspect (case-insensitive)
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="Business Information",
                        description="Categories of Business Information",
                        modality="image",  # ty: ignore[unknown-argument]  # aspects are extracted from text only
                    )
                ]
            )

        # Invalid sequence types
        with pytest.raises(ValueError, match="list"):
            context.add_concepts(
                (
                    StringConcept(
                        name="Random",
                        description="Random",
                    ),
                )  # tuple instead of list  # type: ignore
            )
        with pytest.raises(ValueError, match="list"):
            context.concepts = (
                StringConcept(
                    name="Random",
                    description="Random",
                ),
            )  # tuple instead of list
        with pytest.raises(ValueError, match="list"):
            context.concepts = set([1, 2, 3])  # set instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            context.concepts = range(10)  # range instead of list  # type: ignore

        assert context.aspects == document_aspects
        assert context.aspects is not document_aspects
        context.get_aspect_by_name("Liability")
        with pytest.raises(ValueError):
            context.get_aspect_by_name("Non-existent")
        # Document concepts
        document_concepts = [
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
                add_justifications=True,
            ),
            BooleanConcept(
                name="Contract type check",
                description="Contract is an NDA",
                modality="image",
                add_justifications=True,
            ),
        ]
        context.concepts = document_concepts
        assert context.concepts is not document_concepts
        # Validate assignment
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [  # ty: ignore[invalid-assignment]
                123
            ]  # invalid concept type
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [
                StringConcept(
                    name="business information",
                    description="categories of business information",
                ),
            ]  # duplicate concept (case-insensitive)
        assert context.concepts == document_concepts
        assert context.concepts is not document_concepts
        context.get_concept_by_name("Business Information")
        with pytest.raises(ValueError):
            context.get_concept_by_name("Non-existent")

        # Adding instances
        context.add_aspects(
            [
                Aspect(
                    name="Penalties",
                    description="Clauses describing contractual penalties",
                    add_justifications=True,
                    model_role="reasoner",
                ),
                Aspect(
                    name="Term and duration",
                    description="Clauses describing term and duration of the contract",
                ),
            ]
        )
        assert context.get_aspects_by_names(["Penalties", "Term and duration"])
        context.remove_aspects_by_names(["Penalties", "Term and duration"])
        context.add_concepts(
            [
                BooleanConcept(
                    name="NDA type check",
                    description="Is NDA one-way or mutual?",
                )
            ]
        )
        assert context.get_concepts_by_names(["NDA type check"])
        context.remove_concepts_by_names(["NDA type check"])

        # Removal of instances
        # Aspects
        context.remove_aspect_by_name("Liability")
        assert len(context.aspects) == 1
        assert len(document_aspects) == 2
        with pytest.raises(ValueError):
            context.get_aspect_by_name("Liability")  # not assigned aspect
        context.remove_all_aspects()
        assert len(context.aspects) == 0
        assert len(document_aspects) == 2
        # Concepts
        context.remove_concept_by_name("Contract type check")
        assert len(context.concepts) == 1
        assert len(document_concepts) == 2
        with pytest.raises(ValueError):
            context.get_concept_by_name("Contract type check")  # not assigned concept
        context.remove_all_concepts()
        assert len(context.concepts) == 0
        assert len(document_concepts) == 2
        # All instances
        context.aspects = document_aspects
        context.concepts = document_concepts
        context.remove_all_instances()
        assert len(context.aspects) == 0
        assert len(document_aspects) == 2
        assert len(context.concepts) == 0
        assert len(document_concepts) == 2

        # List field items' unique IDs
        context.remove_all_instances()
        assert not context.aspects and not context.concepts
        aspect = Aspect(
            name="Test",
            description="Test",
        )
        with pytest.raises(ValueError):
            context.add_aspects([aspect, aspect])
        concept = StringConcept(
            name="Test",
            description="Test",
        )
        with pytest.raises(ValueError):
            context.add_concepts([concept, concept])

        check_locals_memory_usage(
            locals(), test_name="test_init_document_and_extraction_pipeline"
        )

    @memory_profile_and_capture
    def test_input_output_token_validation(self):
        """
        Tests for max input and max output token validation.
        """

        # Test 1: Max input tokens validation
        llm = DocumentLLM(
            model="anthropic/claude-haiku-4-5",  # 200k context window
            api_key="...",  # dummy key (no LLM calls are made)
        )

        # Create a very long message that should exceed input limits
        long_message = "This is a test message. " * 50000
        messages = [
            {"role": "user", "content": long_message},
        ]

        with pytest.raises(
            ValueError, match="exceeds the model's maximum input tokens"
        ):
            llm._validate_input_tokens(messages)

        # Test 2: Max output tokens validation
        llm_excessive_output = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            max_completion_tokens=999999,  # excessive output tokens
        )

        with pytest.raises(
            ValueError, match="exceeds the model's maximum output tokens"
        ):
            llm_excessive_output._validate_output_tokens()

        check_locals_memory_usage(
            locals(), test_name="test_input_output_token_validation"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_system_messages(self):
        """
        Tests the system messages functionality of LLMs.
        """
        system_message = "When asked, introduce yourself as ContextGem."

        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message=system_message,
        )
        model.chat("What's your name?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "ContextGem" in response
        logger.debug(response)

        # Test empty system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )
        assert model.system_message == ""

        # Test None system message (set to default system message) with default output language (en)
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert (
            model.system_message is not None
            and model.system_message == self.default_system_message_en
        )

        # Test None system message (set to default system message) with non-en output language (adapt)
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            output_language="adapt",
        )
        assert (
            model.system_message is not None
            and model.system_message == self.default_system_message_non_en
        )

        # Test with custom system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="You are a helpful assistant that can answer questions and help with tasks.",
        )
        assert (
            model.system_message
            == "You are a helpful assistant that can answer questions and help with tasks."
        )

        check_locals_memory_usage(locals(), test_name="test_system_messages")

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    @memory_profile_and_capture
    def test_extract_aspects_from_document(self, llm: DocumentModelGroup | DocumentLLM):
        """
        Tests the aspects extraction functionality of LLMs.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

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
        self.document.aspects = document_aspects
        assert self.document.aspects is not document_aspects

        # Aspects validation

        # Attachment to document
        with pytest.raises(ValueError):
            detached_aspects = [
                Aspect(name="Random", description="Random"),
            ]
            llm.extract_aspects_from_document(
                self.document, from_aspects=detached_aspects
            )

        # Invalid aspect types
        invalid_aspects = [1, True]
        with pytest.raises(ValueError):
            self.document.aspects = invalid_aspects  # ty: ignore[invalid-assignment]
        with pytest.raises(ValueError):
            llm.extract_aspects_from_document(
                self.document,
                from_aspects=invalid_aspects,  # ty: ignore[invalid-argument-type]
            )

        # Invalid sequence types for "from_aspects"
        with pytest.raises(ValueError, match="list"):
            llm.extract_aspects_from_document(
                self.document,
                from_aspects=tuple(document_aspects),
            )
        with pytest.raises(ValueError, match="list"):
            llm.extract_aspects_from_document(
                self.document,
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
            self.document.aspects = duplicate_aspects
        duplicate_name_aspects = [
            Aspect(name="Parties", description="Desc 1"),
            Aspect(name="Parties", description="Desc 2"),
        ]
        with pytest.raises(ValueError):
            self.document.aspects = duplicate_name_aspects
        duplicate_description_aspects = [
            Aspect(name="Name 1", description="Description"),
            Aspect(name="Name 2", description="Description"),
        ]
        with pytest.raises(ValueError):
            self.document.aspects = duplicate_description_aspects

        # No partial update check
        assert self.document.aspects == document_aspects
        assert self.document.aspects is not document_aspects

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
        self.document.aspects = document_aspects

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(3 if llm.is_group else 2) * 3
            + 2,  # *3 due to 30 (1/3) paras per call, +2 due to sub-aspects
            expected_n_calls_1_item_per_call=len(self.document.aspects) * 3
            + 3,  # *3 due to 30 (1/3) paras per call, +3 due to sub-aspects
            expected_n_calls_with_concurrency=len(self.document.aspects) * 3
            + 3,  # *3 due to 30 (1/3) paras per call, +3 due to sub-aspects
            func=llm.extract_aspects_from_document,
            func_kwargs={
                "document": self.document,
                "max_paragraphs_to_analyze_per_call": 30,
            },
            original_container=document_aspects,
            assigned_container=self.document.aspects,
            assigned_instance_class=Aspect,
            compare_sequential_1_item_in_call=False,
        )
        check_aspects = self.document.get_aspects_by_names(
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
            llm.extract_aspects_from_document(self.document)
        with pytest.raises(ValueError):
            llm.extract_aspects_from_document(
                self.document, from_aspects=document_aspects
            )
        overwrite_aspects = [self.document.get_aspect_by_name("Parties")]
        extracted_aspects = llm.extract_aspects_from_document(
            self.document,
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
            self.document.get_aspect_by_name(  # ty: ignore[invalid-argument-type]
                "Exclusions from confidential information"
            ).get_aspect_by_name("Information in possession of the receiving party"),
            full_repr=False,
        )
        self.log_extracted_items_for_instance(
            self.document.get_aspect_by_name(  # ty: ignore[invalid-argument-type]
                "Exclusions from confidential information"
            ).get_aspect_by_name("Information received from a third party"),
            full_repr=False,
        )
        self.log_extracted_items_for_instance(
            self.document.get_aspect_by_name(  # ty: ignore[invalid-argument-type]
                "Exclusions from confidential information"
            ).get_aspect_by_name("Independently developed information"),
            full_repr=False,
        )

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

        check_locals_memory_usage(
            locals(), test_name="test_extract_aspects_from_document"
        )

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    @memory_profile_and_capture
    def test_extract_concepts_from_aspect(self, llm: DocumentModelGroup | DocumentLLM):
        """
        Tests for concept extraction from aspect.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

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
            self.document.aspects = []
            llm.extract_concepts_from_aspect(aspect=aspects[0], document=self.document)

        # LLM extraction
        self.document.aspects = aspects
        assert self.document.aspects is not aspects
        attached_aspect = self.document.get_aspect_by_name("Confidential information")

        # Attempt concept extraction first, without pre-extraction of the aspect.
        assert not attached_aspect.extracted_items
        with pytest.raises(ValueError):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect, document=self.document
            )

        # Aspects first
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=1,
            expected_n_calls_1_item_per_call=1,
            expected_n_calls_with_concurrency=1,
            func=llm.extract_aspects_from_document,
            func_kwargs={
                "document": self.document,
            },
            original_container=aspects,
            assigned_container=self.document.aspects,
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
                "document": self.document,
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
                    self.document.get_paragraph_index(p)
                    for p in extracted_item.reference_paragraphs
                ]
                assert item_ref_para_indices == sorted(item_ref_para_indices)
                assert extracted_item.reference_sentences
                item_ref_sent_positions = [
                    self.document.get_sentence_index(s)
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
        self.document.aspects = aspects_without_concepts
        assert self.document.aspects is not aspects_without_concepts
        llm.extract_aspects_from_document(self.document)
        extracted_concepts = llm.extract_concepts_from_aspect(
            aspect=self.document.get_aspect_by_name("Term and termination"),
            document=self.document,
        )
        self.check_instance_container_states(
            original_container=aspects_without_concepts,
            assigned_container=self.document.aspects,
            assigned_instance_class=Aspect,
        )
        assert extracted_concepts == []

        # Invalid concepts passed
        self.document.aspects = aspects
        assert self.document.aspects is not aspects
        llm.extract_aspects_from_document(self.document)
        attached_aspect = self.document.get_aspect_by_name("Confidential information")
        with pytest.raises(ValueError):
            no_concepts = []
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=self.document,
                from_concepts=no_concepts,
            )
        with pytest.raises(ValueError):
            invalid_concepts = [1, True]
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=self.document,
                from_concepts=invalid_concepts,  # ty: ignore[invalid-argument-type]
            )
        with pytest.raises(ValueError):
            detached_concepts = [
                StringConcept(name="Random", description="Random"),
            ]
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=self.document,
                from_concepts=detached_concepts,
            )

        # Invalid sequence types for "from_concepts"
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=self.document,
                from_concepts=tuple(attached_aspect.concepts),
            )
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=self.document,
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
            aspect=attached_aspect, document=self.document
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
            document=self.document,
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
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect, document=self.document
            )
        with pytest.raises(ValueError):
            llm.extract_concepts_from_aspect(
                aspect=attached_aspect,
                document=self.document,
                from_concepts=attached_aspect.concepts,
            )
        overwrite_concepts = [attached_aspect.concepts[0]]
        extracted_concepts = llm.extract_concepts_from_aspect(
            aspect=attached_aspect,
            document=self.document,
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

        check_locals_memory_usage(
            locals(), test_name="test_extract_concepts_from_aspect"
        )

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    @memory_profile_and_capture
    def test_extract_concepts_from_document(
        self, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for concept extraction from document.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

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
        self.document.concepts = document_concepts
        assert self.document.concepts is not document_concepts

        # Concepts validation

        # Attachment to document
        with pytest.raises(ValueError):
            detached_concepts = [
                StringConcept(name="Random", description="Random"),
            ]
            llm.extract_concepts_from_document(
                self.document, from_concepts=detached_concepts
            )

        # Invalid types
        with pytest.raises(ValueError):
            invalid_concepts = [1, True]
            self.document.concepts = invalid_concepts  # ty: ignore[invalid-assignment]

        # Invalid sequence types for "from_concepts"
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_document(
                self.document,
                from_concepts=tuple(self.document.concepts),
            )
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_document(
                self.document,
                from_concepts=set([1, 2, 3]),  # type: ignore
            )

        # No partial update check
        assert self.document.concepts == document_concepts
        assert self.document.concepts is not document_concepts

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
        self.document.concepts = document_concepts
        assert self.document.concepts is not document_concepts

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
            self.document.concepts = duplicate_concepts
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
            self.document.concepts = duplicate_name_concepts
        duplicate_description_concepts = [
            StringConcept(name="Name 1", description="Description"),
            NumericalConcept(
                name="Name 2", description="Description", numeric_type="int"
            ),
        ]
        with pytest.raises(ValueError):
            self.document.concepts = duplicate_description_concepts

        # No partial update check
        assert self.document.concepts == document_concepts
        assert self.document.concepts is not document_concepts

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(5 if llm.is_group else 4),
            expected_n_calls_with_concurrency=len(self.document.concepts),
            expected_n_calls_1_item_per_call=len(self.document.concepts),
            func=llm.extract_concepts_from_document,
            func_kwargs={
                "document": self.document,
            },
            original_container=list(document_concepts),
            assigned_container=list(self.document.concepts),
            assigned_instance_class=_Concept,
        )
        self.check_extra_data_in_extracted_items(self.document)

        self.log_extracted_items_for_instance(
            self.document.get_concept_by_name("Effective Date")
        )
        self.log_extracted_items_for_instance(
            self.document.get_concept_by_name("Contract quality")
        )

        # Overwrite check
        with pytest.raises(ValueError):
            llm.extract_concepts_from_document(self.document)
        with pytest.raises(ValueError):
            llm.extract_concepts_from_document(
                self.document, from_concepts=self.document.concepts
            )
        overwrite_concepts = [self.document.concepts[0]]
        extracted_concepts = llm.extract_concepts_from_document(
            self.document, from_concepts=overwrite_concepts, overwrite_existing=True
        )
        assert overwrite_concepts == extracted_concepts
        self.check_instance_container_states(
            original_container=[document_concepts[0]],
            assigned_container=extracted_concepts,
            assigned_instance_class=_Concept,
        )

        # Extract with different paragraph chunks for context
        extracted_concepts = llm.extract_concepts_from_document(
            self.document,
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

        check_locals_memory_usage(
            locals(), test_name="test_extract_concepts_from_document"
        )

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
    @memory_profile_and_capture
    def test_extract_all(
        self, document: Document, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for extracting all aspects and concepts from the document and its aspects.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        self.config_llms_for_output_lang(document, llm)

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
        if document in [self.document_ua, self.document_zh]:
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

        check_locals_memory_usage(locals(), test_name="test_extract_all")

    @pytest.mark.vcr
    @memory_profile_and_capture
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

        check_locals_memory_usage(locals(), test_name="test_extract_with_fallback")

    @memory_profile_and_capture
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

        check_locals_memory_usage(locals(), test_name="test_multiple_fallback_models")

    @memory_profile_and_capture
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

        check_locals_memory_usage(
            locals(),
            test_name="test_deserialize_fallback_models_differing_in_credentials",
        )

    @memory_profile_and_capture
    def test_model_api_request_retries(self):
        """
        Tests for retrying failed LLM API requests: only transient errors are retried,
        with a backoff between the retries, and each LLM, including each fallback LLM,
        applies its own retry params.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm

        def create_api_error(error_class: Any) -> Exception:
            """
            Creates a model API error of the specified litellm exception class.
            """
            return error_class(
                message="Simulated API error",
                model="openai/gpt-6-luna",
                llm_provider="openai",
            )

        rate_limit_error = create_api_error(litellm.RateLimitError)
        auth_error = create_api_error(litellm.AuthenticationError)

        # === Transient vs non-transient errors ===
        for error in [
            create_api_error(litellm.Timeout),
            rate_limit_error,
            create_api_error(litellm.InternalServerError),
            create_api_error(litellm.ServiceUnavailableError),
            create_api_error(litellm.APIConnectionError),
            RuntimeError("Error not classified by litellm"),
        ]:
            assert _is_retryable_model_api_error(error)
        for error in [
            create_api_error(litellm.BadRequestError),
            auth_error,
            create_api_error(litellm.NotFoundError),
            create_api_error(litellm.ContextWindowExceededError),
            create_api_error(litellm.ContentPolicyViolationError),
        ]:
            assert not _is_retryable_model_api_error(error)

        # === Backoff with jitter ===
        for retry_number, backoff in [
            (1, 1.0),
            (2, 2.0),
            (3, 4.0),
            (4, 8.0),
            (9, 10.0),
        ]:
            delays = {_get_model_api_retry_delay(retry_number) for _ in range(50)}
            assert all(backoff / 2 <= delay <= backoff for delay in delays)
            assert len(delays) > 1  # delays are randomized

        # === Retries of the requests ===
        primary_model = "openai/gpt-6-luna"
        fallback_model_1 = "openai/gpt-5.6-luna"
        fallback_model_2 = "openai/gpt-6.1-sol"
        invalid_answer = "<<<not-json>>>"
        concepts_answer = '[{"concept_id": "C0", "extracted_items": ["Item"]}]'

        requests: list[tuple[str, int]] = []  # (model, provider-specific retries)
        retry_numbers: list[int] = []
        outcomes: dict[str, list[Exception | str]] = {}

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, with the outcomes of the requests to each model
            taken from the queue of that model. The last outcome on the queue is repeated.
            """
            assert kwargs["num_retries"] == 0  # retries are not delegated to litellm
            model = kwargs["model"]
            requests.append((model, kwargs["max_retries"]))
            queue = outcomes[model]
            outcome = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(outcome, Exception):
                raise outcome
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=outcome)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def retry_delay_stub(retry_number: int) -> float:
            """
            Records the retry and removes the delay before it.
            """
            retry_numbers.append(retry_number)
            return 0.0

        def set_outcomes(model_outcomes: dict[str, list[Exception | str]]) -> None:
            """
            Sets the outcomes of the requests to each model and resets the records.
            """
            outcomes.clear()
            outcomes.update(model_outcomes)
            requests.clear()
            retry_numbers.clear()

        def requested_models() -> list[str]:
            """
            Returns the models in the order the requests were sent to them.
            """
            return [model for model, _ in requests]

        def create_llm(model: str = primary_model, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM, with the default retry params unless specified.
            """
            return DocumentLLM(
                model=model, api_key="test_api_key", system_message="", **kwargs
            )

        def create_llm_with_fallbacks() -> DocumentLLM:
            """
            Creates an LLM with two fallback LLMs, each with different retry params.
            """
            return create_llm(
                num_retries_failed_request=1,
                fallback_models=[
                    create_llm(
                        fallback_model_1,
                        num_retries_failed_request=2,
                        is_fallback=True,
                    ),
                    create_llm(
                        fallback_model_2,
                        num_retries_failed_request=3,
                        is_fallback=True,
                    ),
                ],
            )

        def create_document(n_concepts: int = 1) -> Document:
            """
            Creates a document with the specified number of concepts to extract.
            """
            document = Document(
                raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
            )
            document.concepts = [
                StringConcept(name=f"Concept {i}", description=f"Description {i}")
                for i in range(n_concepts)
            ]
            return document

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(
                llms_module, "_get_model_api_retry_delay", new=retry_delay_stub
            ),
        ):
            # Transient errors are retried until the request succeeds
            set_outcomes({primary_model: [rate_limit_error, rate_limit_error, "4"]})
            assert create_llm().chat("What's 2+2?") == "4"
            assert requested_models() == [primary_model] * 3
            assert retry_numbers == [1, 2]

            # ...or until the retries are exhausted
            for num_retries in [3, 1, 0]:
                set_outcomes({primary_model: [rate_limit_error]})
                with pytest.raises(ModelAPIError) as exc_info:
                    create_llm(num_retries_failed_request=num_retries).chat("Test")
                assert exc_info.value.retry_count == num_retries
                assert exc_info.value.original_error is rate_limit_error
                assert requested_models() == [primary_model] * (1 + num_retries)
                assert retry_numbers == list(range(1, num_retries + 1))

            # Non-transient errors are not retried
            set_outcomes({primary_model: [auth_error]})
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm().chat("Test")
            assert exc_info.value.retry_count == 0
            assert exc_info.value.original_error is auth_error
            assert requested_models() == [primary_model]
            assert retry_numbers == []
            # ...also when they follow a transient error
            set_outcomes({primary_model: [rate_limit_error, auth_error]})
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm().chat("Test")
            assert exc_info.value.retry_count == 1
            assert requested_models() == [primary_model] * 2

            # Provider-specific retries apply to the first attempt only
            set_outcomes({primary_model: [rate_limit_error, "4"]})
            assert create_llm(max_retries_failed_request=2).chat("Test") == "4"
            assert requests == [(primary_model, 2), (primary_model, 0)]

            # Chat: each LLM applies its own retry params before the next one is tried
            expected_requested_models = (
                [primary_model] * 2 + [fallback_model_1] * 3 + [fallback_model_2] * 3
            )
            set_outcomes(
                {
                    primary_model: [rate_limit_error],
                    fallback_model_1: [rate_limit_error],
                    fallback_model_2: [rate_limit_error, rate_limit_error, "4"],
                }
            )
            assert create_llm_with_fallbacks().chat("What's 2+2?") == "4"
            assert requested_models() == expected_requested_models

            # Extraction: same, and a failed request is not sent to the same LLM again
            set_outcomes(
                {
                    primary_model: [rate_limit_error],
                    fallback_model_1: [rate_limit_error],
                    fallback_model_2: [
                        rate_limit_error,
                        rate_limit_error,
                        concepts_answer,
                    ],
                }
            )
            document = create_document()
            create_llm_with_fallbacks().extract_concepts_from_document(document)
            assert requested_models() == expected_requested_models
            assert document.concepts[0].extracted_items[0].value == "Item"

            # Extraction: the last fallback LLM also retries its request before failing
            set_outcomes(
                {
                    primary_model: [auth_error],
                    fallback_model_1: [auth_error],
                    fallback_model_2: [rate_limit_error],
                }
            )
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm_with_fallbacks().extract_concepts_from_document(
                    create_document()
                )
            assert exc_info.value.retry_count == 3
            assert requested_models() == (
                [primary_model, fallback_model_1] + [fallback_model_2] * 4
            )

            # Extraction: invalid data is requested again without a delay
            set_outcomes({primary_model: [invalid_answer]})
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*2 retries"):
                create_llm(max_retries_invalid_data=2).extract_concepts_from_document(
                    create_document()
                )
            assert requested_models() == [primary_model] * 3
            assert retry_numbers == []
            # ...while a request that fails when repeated is passed to a fallback LLM
            set_outcomes(
                {
                    primary_model: [invalid_answer, rate_limit_error],
                    fallback_model_1: [concepts_answer],
                }
            )
            document = create_document()
            create_llm_with_fallbacks().extract_concepts_from_document(document)
            assert requested_models() == [primary_model] * 3 + [fallback_model_1]
            assert document.concepts[0].extracted_items[0].value == "Item"

            # Extraction with concurrency: a final API error is raised as is,
            # without sending the failed requests again
            set_outcomes({primary_model: [auth_error]})
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm().extract_concepts_from_document(
                    create_document(n_concepts=2), use_concurrency=True
                )
            assert exc_info.value.original_error is auth_error
            assert requested_models() == [primary_model] * 2  # one per concept

            # Extraction: a warning is issued if `raise_exception_on_extraction_error`
            # is False, without sending the failed request again
            set_outcomes({primary_model: [rate_limit_error]})
            document = create_document()
            with pytest.warns(UserWarning, match=r"LLM API.*3 retries"):
                create_llm().extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=False
                )
            assert requested_models() == [primary_model] * 4
            assert not document.concepts[0].extracted_items

        check_locals_memory_usage(locals(), test_name="test_model_api_request_retries")

    @memory_profile_and_capture
    def test_async_limiter_per_llm(self):
        """
        Tests that, with concurrency enabled, the requests to each LLM, including
        each fallback LLM, are paced by the async limiter of that LLM.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"

        llm = DocumentLLM(
            model=primary_model,
            api_key="test_api_key",
            fallback_models=[
                DocumentLLM(
                    model=fallback_model, api_key="test_api_key", is_fallback=True
                )
            ],
        )
        llm.async_limiter = AsyncLimiter(10, 5)
        llm.fallback_models[0].async_limiter = AsyncLimiter(20, 5)

        acquisitions: list[tuple[float, float]] = []

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the config of the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.
                """
                acquisitions.append((self.max_rate, self.time_period))
                await super().acquire(amount)

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: requests to the primary LLM fail,
            requests to the fallback LLM succeed.
            """
            if kwargs["model"] == primary_model:
                raise litellm.AuthenticationError(
                    message="Simulated API error",
                    llm_provider="openai",
                    model=primary_model,
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"concept_id": "C0", "extracted_items": ["Item"]}]'
                        )
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        document = Document(
            raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
        )
        document.concepts = [
            StringConcept(name="Title", description="Contract title"),
            StringConcept(name="Parties", description="Contract parties"),
        ]

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(llms_module, "AsyncLimiter", new=RecordingAsyncLimiter),
        ):
            # With concurrency, each concept is processed in a separate request:
            # first to the primary LLM, then to the fallback LLM
            llm.extract_concepts_from_document(document, use_concurrency=True)
            assert sorted(acquisitions) == [(10, 5)] * 2 + [(20, 5)] * 2
            assert all(i.extracted_items for i in document.concepts)

            # Without concurrency, requests are sent one by one, with no limiter
            acquisitions.clear()
            llm.extract_concepts_from_document(document, overwrite_existing=True)
            assert acquisitions == []
            assert all(i.extracted_items for i in document.concepts)

        check_locals_memory_usage(locals(), test_name="test_async_limiter_per_llm")

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
    @memory_profile_and_capture
    def test_serialization_and_cloning(
        self, document: Document, llm: DocumentModelGroup | DocumentLLM
    ):
        """
        Tests for custom serialization, deserialization, and cloning of application class instances,
        with preservation of the relevant private attributes.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        self.config_llms_for_output_lang(document, llm)

        # Extraction pipeline serialization
        self.check_instance_serialization_and_cloning(self.extraction_pipeline)
        for aspect in self.extraction_pipeline.aspects:
            self.check_instance_serialization_and_cloning(aspect)
            for concept in aspect.concepts:
                self.check_instance_serialization_and_cloning(concept)
        for concept in self.extraction_pipeline.concepts:
            self.check_instance_serialization_and_cloning(concept)

        # Processed document serialization
        document_aspects = [
            Aspect(
                name="Term and termination",
                description="All clauses addressing term and termination of the contract.",
                concepts=[
                    NumericalConcept(
                        name="Duration",
                        description="Contract duration",
                        numeric_type="int",
                        add_justifications=True,
                        add_references=True,
                        reference_depth="sentences",
                        singular_occurrence=True,
                    ),
                    JsonObjectConcept(
                        name="Duration in years",
                        description="Contract duration in years, if specified.",
                        structure={"years": int | None},
                        examples=[
                            JsonObjectExample(content={"years": 10}),
                            JsonObjectExample(content={"years": None}),
                        ],
                        singular_occurrence=True,
                    ),
                    RatingConcept(
                        name="NDA term adequacy",
                        description="Rate the adequacy of the length of the NDA term, "
                        "from a best practice perspective.",
                        rating_scale=(1, 5),
                        add_justifications=True,
                        justification_depth="balanced",
                        justification_max_sents=5,
                        singular_occurrence=True,
                    ),
                ],
            ),
            Aspect(
                name="Parties",
                description="Parties to the agreement",
                concepts=[
                    StringConcept(
                        name="Contract party name and role",
                        description="Contract party name and role in the contract. "
                        "Exclude information on individual signatories. "
                        "Example: Company ABC (Supplier).",
                        singular_occurrence=False,
                    ),
                    LabelConcept(
                        name="Party types",
                        description="The types of parties involved in the agreement.",
                        labels=[
                            "Individual",
                            "Corporation",
                            "Government entity",
                            "Non-profit organization",
                            "Partnership",
                        ],
                        classification_type="multi_label",
                    ),
                ],
                reference_depth="sentences",
            ),
        ]
        document_images = [
            self.test_img_jpg_invoice,
            self.test_img_jpg_2_invoice,
        ]
        document_concepts = [
            DateConcept(
                name="Start date",
                description="Contract start date",
                add_justifications=True,
                add_references=True,
                reference_depth="sentences",
                singular_occurrence=True,
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                model_role="reasoner",
                add_justifications=True,
                singular_occurrence=True,
            ),
            LabelConcept(
                name="Contract type",
                description="The type(s) of contract represented by this document",
                labels=[
                    "Non-disclosure agreement",
                    "Employment contract",
                    "Service agreement",
                    "Sales contract",
                    "Lease agreement",
                    "Licensing agreement",
                ],
                classification_type="multi_label",
                model_role="reasoner",
                singular_occurrence=True,  # global document classification
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                singular_occurrence=True,
            ),
            NumericalConcept(
                name="Duration in years",
                description="Contract duration in years",
                model_role="reasoner",
                numeric_type="int",
                add_justifications=True,
                singular_occurrence=True,
            ),
            JsonObjectConcept(
                name="Key aspects",
                description="Names and summaries of key aspects covered in the document",
                structure={
                    "aspects_names_and_summaries": [
                        {
                            "aspect_name": str,
                            "aspect_summary": str,
                        }
                    ]  # intentionally overcomplicated structure
                },
                singular_occurrence=True,
            ),
            RatingConcept(
                name="Contract quality",
                description="Rate contract quality from the perspective "
                "of its adherence to best practices in NDA drafting.",
                rating_scale=(1, 10),
                add_justifications=True,
                justification_depth="comprehensive",
                justification_max_sents=10,
                singular_occurrence=True,
            ),
            StringConcept(
                name="Section titles",
                description="Section titles in the contract",
                singular_occurrence=False,
            ),
            # Vision-related
            BooleanConcept(
                name="Invoice check",
                description="Whether the attached images contain an invoice",
                model_role="reasoner",
                modality="image",
                singular_occurrence=True,
            ),
            StringConcept(
                name="Invoice amount",
                description="The total amount of the attached invoice",
                modality="image",
                singular_occurrence=False,  # multiple invoices in the document
            ),
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                modality="image",
                singular_occurrence=False,  # multiple invoices in the document
            ),
            LabelConcept(
                name="Invoice type",
                description="The type of invoice shown in the image",
                labels=[
                    "Commercial invoice",
                    "Proforma invoice",
                    "Tax invoice",
                    "Credit note",
                    "Debit note",
                ],
                classification_type="multi_class",
                modality="image",
                singular_occurrence=False,  # multiple invoices in the document
            ),
        ]
        document.aspects = document_aspects
        document.images = document_images  # ty: ignore[invalid-assignment]
        document.concepts = document_concepts
        document = llm.extract_all(  # ty: ignore[invalid-assignment]
            document,
            max_paragraphs_to_analyze_per_call=30,  # process contract in chunks to test for singular-occurrence concepts
            max_images_to_analyze_per_call=1,  # process invoices 1 by 1, to test for vision concepts in each
        )

        self.log_extracted_items_for_instance(
            document.get_aspect_by_name("Term and termination").get_concept_by_name(
                "NDA term adequacy"
            )
        )
        self.log_extracted_items_for_instance(
            document.get_aspect_by_name("Parties").get_concept_by_name("Party types")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Start date")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Contract quality")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Section titles")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Contract type")
        )
        # Vision-related
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice check")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice amount")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice number")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Key aspects")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice type")
        )

        # Document serialization and deserialization (post-extraction)
        self.check_instance_serialization_and_cloning(document)

        # Aspect serialization and deserialization
        for i, aspect in enumerate(document.aspects):
            self.check_extra_data_in_extracted_items(aspect)  # ty: ignore[invalid-argument-type]
            self.check_instance_serialization_and_cloning(aspect)

            for extracted_item in aspect.extracted_items:
                self.check_instance_serialization_and_cloning(extracted_item)
                for para in extracted_item.reference_paragraphs:
                    self.check_instance_serialization_and_cloning(para)
                    for sentence in para.sentences:
                        self.check_instance_serialization_and_cloning(sentence)

            for concept in aspect.concepts:
                # Aspect concept serialization and deserialization
                aspect_concept_dict = concept.to_dict()
                if i == 1:
                    with pytest.raises(TypeError):
                        # Test invalid class to reconstruct from dict
                        NumericalConcept.from_dict(aspect_concept_dict)
                self.check_instance_serialization_and_cloning(concept)
                for extracted_item in concept.extracted_items:
                    self.check_instance_serialization_and_cloning(extracted_item)
                    for para in extracted_item.reference_paragraphs:
                        self.check_instance_serialization_and_cloning(para)
                        for sentence in para.sentences:
                            self.check_instance_serialization_and_cloning(sentence)

        # Document concept serialization and deserialization
        for i, concept in enumerate(document.concepts):
            concept_dict = concept.to_dict()
            if i == 1:
                with pytest.raises(TypeError):
                    # Test invalid class to reconstruct from dict
                    NumericalConcept.from_dict(concept_dict)
            self.check_instance_serialization_and_cloning(concept)
            for extracted_item in concept.extracted_items:
                self.check_instance_serialization_and_cloning(extracted_item)
                for para in extracted_item.reference_paragraphs:
                    self.check_instance_serialization_and_cloning(para)
                    for sentence in para.sentences:
                        self.check_instance_serialization_and_cloning(sentence)

        # Document image serialization and deserialization
        for _i, image in enumerate(document.images):
            self.check_instance_serialization_and_cloning(image)

        # Check with extraction pipeline assignment
        with pytest.raises(RuntimeError):
            document.assign_pipeline(self.extraction_pipeline)
        document.remove_all_instances()
        document.assign_pipeline(self.extraction_pipeline)
        llm.extract_all(document)
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice number check")
        )
        self.check_instance_container_states(
            original_container=self.extraction_pipeline.aspects,  # ty: ignore[invalid-argument-type]
            assigned_container=document.aspects,  # ty: ignore[invalid-argument-type]
            assigned_instance_class=Aspect,
        )
        self.check_instance_container_states(
            original_container=list(self.extraction_pipeline.concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )
        self.check_instance_serialization_and_cloning(document)
        self.check_instance_serialization_and_cloning(self.extraction_pipeline)

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm)

        # Check serialization of an LLM with fallback
        self._check_deserialized_llm_config_eq(self.invalid_llm_with_valid_fallback)

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

        check_locals_memory_usage(locals(), test_name="test_serialization_and_cloning")

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [model_group, llm_extractor])
    @memory_profile_and_capture
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

        check_locals_memory_usage(
            locals(), test_name="test_aspect_extraction_from_paragraphs"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_vision(self):
        """
        Tests for data extraction from document images using vision API.
        """
        with pytest.raises(ValueError):
            Document(images=[])

        # Invoice
        document_concepts = [
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                modality="image",
                add_justifications=True,
            ),
            JsonObjectConcept(
                name="Service items",
                description="Service items, their quantity and price",
                structure={
                    "service_name": str,
                    "quantity": int,
                    "price": str,
                },
                examples=[
                    JsonObjectExample(
                        content={
                            "service_name": "Service 1",
                            "quantity": 1,
                            "price": "100.00 EUR",
                        }
                    )
                ],
                model_role="reasoner",
                modality="image",
                add_justifications=True,
            ),
            JsonObjectConcept(
                name="Invoice amounts",
                description="Invoice amounts as they appear in the invoice",
                structure={
                    "line": str,
                    "amount": str,
                },
                modality="image",
            ),
            NumericalConcept(
                name="Total invoice amount",
                description="Total amount of the invoice",
                modality="image",
                numeric_type="float",
                add_justifications=True,
            ),
        ]
        document = Document(
            images=[self.test_img_png_invoice], concepts=document_concepts
        )

        self.config_llm_async_limiter_for_mock_responses(self.model_group)

        self.compare_with_concurrent_execution(
            llm=self.model_group,
            expected_n_calls_no_concurrency=3,
            expected_n_calls_with_concurrency=4,
            expected_n_calls_1_item_per_call=4,
            func=self.model_group.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )

        # Other
        document_concepts = [
            NumericalConcept(
                name="Number of bedrooms",
                description="Number of bedrooms in the floor plan",
                model_role="reasoner",
                modality="image",
                numeric_type="int",
                add_justifications=True,
            ),
            NumericalConcept(
                name="Number of balconies",
                description="Number of balconies in the floor plan",
                model_role="reasoner",
                modality="image",
                numeric_type="int",
                add_justifications=True,
            ),
            StringConcept(
                name="Room names",
                description="Names of the rooms in the floor plan",
                modality="image",
            ),
        ]
        document = Document(
            images=[self.test_img_png_apt_plan], concepts=document_concepts
        )

        self.compare_with_concurrent_execution(
            llm=self.model_group,
            expected_n_calls_no_concurrency=2,
            expected_n_calls_with_concurrency=3,
            expected_n_calls_1_item_per_call=3,
            func=self.model_group.extract_concepts_from_document,
            func_kwargs={
                "document": document,
            },
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
        )

        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Number of bedrooms")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Number of balconies")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Room names")
        )

        # Check usage tokens
        self.check_usage(self.model_group)
        # Check cost
        self.check_cost(self.model_group)

        # Test error when model is not vision-capable
        llm_not_vision_capable = DocumentLLM(
            model="cerebras/llama-3.3-70b",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        with pytest.raises(ValueError, match="does not support vision"):
            llm_not_vision_capable.chat(
                "What's the type of this document?", images=[self.test_img_png_invoice]
            )

        check_locals_memory_usage(
            locals(), test_name="test_vision", max_obj_memory=5.0
        )  # higher value for images

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [llm_extractor, llm_reasoner, model_group])
    @memory_profile_and_capture
    def test_multimodal_concepts(self, llm: DocumentLLM | DocumentModelGroup):
        """
        Tests for multimodal concept extraction from multimodal document (text + images).
        """

        # ===Text + images===
        self.document_multimodal.remove_all_instances()
        self.document_multimodal.add_concepts(
            [
                StringConcept(
                    name="Invoice number",
                    description="Number of the invoice",
                    modality="multimodal",
                    add_references=True,
                    add_justifications=True,
                ),
                DateConcept(
                    name="Invoice date",
                    description="Date of the invoice",
                    modality="multimodal",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        with pytest.warns(
            UserWarning,
            match="When extracting from images, references will not be added",
        ):
            # Document has both text and images - warns that refs will be added only for text
            llm.extract_concepts_from_document(self.document_multimodal)
        concept_invoice_number = self.document_multimodal.get_concept_by_name(
            "Invoice number"
        )
        concept_invoice_date = self.document_multimodal.get_concept_by_name(
            "Invoice date"
        )
        if llm is not self.llm_reasoner:
            self.log_extracted_items_for_instance(concept_invoice_number)
            assert concept_invoice_number.extracted_items
            assert concept_invoice_number.extracted_items[0].reference_paragraphs
            assert concept_invoice_number.extracted_items[0].justification
            assert not all(
                i.reference_paragraphs for i in concept_invoice_number.extracted_items
            )  # refs will be added only for text content, not for images
            assert all(i.justification for i in concept_invoice_number.extracted_items)
        else:
            self.log_extracted_items_for_instance(concept_invoice_date)
            assert concept_invoice_date.extracted_items
            assert concept_invoice_date.extracted_items[0].reference_paragraphs
            assert concept_invoice_date.extracted_items[0].reference_sentences
            assert not all(
                i.reference_paragraphs for i in concept_invoice_date.extracted_items
            )  # refs will be added only for text content, not for images

        # Test together with aspects and other concepts routed to different LLMs
        if isinstance(llm, DocumentModelGroup):
            self.document_multimodal.add_aspects(
                [
                    Aspect(
                        name="Invoices",
                        description="Provisions on invoices in the contract",
                        concepts=[
                            NumericalConcept(
                                name="Invoice number",
                                description="Number of the invoice",
                            )
                        ],
                    )
                ]
            )
            llm.extract_aspects_from_document(self.document_multimodal)
            assert self.document_multimodal.aspects[0].extracted_items
            self.document_multimodal.add_concepts(
                [
                    StringConcept(
                        name="Document type",
                        description="Guess the type of the document",
                        model_role="reasoner",
                        add_justifications=True,
                    )
                ]
            )
            llm.extract_all(self.document_multimodal, overwrite_existing=True)
            document_type_concept = self.document_multimodal.get_concept_by_name(
                "Document type"
            )
            assert document_type_concept.extracted_items
            self.log_extracted_items_for_instance(document_type_concept)
            assert self.document_multimodal.aspects[0].concepts[0].extracted_items

        # ===Text only===
        document = Document(raw_text=get_test_document_text())
        document.add_concepts(
            [
                StringConcept(
                    name="Document type",
                    description="Type of the document",
                    modality="multimodal",
                )
            ]
        )
        llm.extract_concepts_from_document(document)
        document_type_concept = document.get_concept_by_name("Document type")
        assert document_type_concept.extracted_items

        # ===Images only===
        document = Document(images=[self.test_img_png_invoice])
        document.add_concepts(
            [
                StringConcept(
                    name="Invoice number",
                    description="Number of the invoice",
                    modality="multimodal",
                )
            ]
        )
        llm.extract_concepts_from_document(document)
        invoice_number_concept = document.get_concept_by_name("Invoice number")
        assert invoice_number_concept.extracted_items

        check_locals_memory_usage(locals(), test_name="test_multimodal_concepts")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_chat_no_session(self):
        """
        Tests for the chat method.
        """
        for model, with_image in [
            (self.llm_extractor, False),
            (self.llm_extractor, True),
            (self.invalid_llm_with_valid_fallback, False),
        ]:
            with pytest.raises(ValueError):
                model.chat("")
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=1)  # type: ignore
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=[1])  # type: ignore
            with pytest.raises(TypeError):
                model.chat(images=self.test_img_png_invoice)  # type: ignore
            if with_image:
                # Check with text + image
                with pytest.warns(UserWarning, match="default system message"):
                    model.chat(
                        "What's the type of this document?",
                        images=[self.test_img_png_invoice],
                    )
                response = model.get_usage()[0].usage.calls[-1].response
                assert response is not None
                assert "invoice" in response.lower()
            else:
                # Check with text
                with pytest.warns(UserWarning, match="default system message"):
                    model.chat("What's the result of 2+2?")
                if model == self.invalid_llm_with_valid_fallback:
                    assert model.fallback_models
                    response = (
                        model.fallback_models[0].get_usage()[0].usage.calls[-1].response
                    )
                else:
                    response = model.get_usage()[0].usage.calls[-1].response
                assert response is not None
                assert "4" in response
            logger.debug(response)

        # Test for non-vision model
        text_only_model = DocumentLLM(model="cerebras/llama-3.3-70b")
        with pytest.raises(ValueError, match="vision"):
            text_only_model.chat(
                "What's the type of this document?", images=[self.test_img_png_invoice]
            )

        # Test for empty system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )
        assert model.system_message == ""
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")  # Capture all warnings
            model.chat("What's the result of 10+10?")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "Expected no warning, but got: " + str(
            user_warnings
        )
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "20" in response

        # Test for None system message (output language "en", which is the default)
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert model.system_message == self.default_system_message_en
        with pytest.warns(UserWarning, match="default system message"):
            model.chat("What's the result of 10+10?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "20" in response

        # Test for None system message (output language "adapt")
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            output_language="adapt",
        )
        assert model.system_message == self.default_system_message_non_en
        with pytest.warns(UserWarning, match="default system message"):
            model.chat("Hva er resultatet av 30+10?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "40" in response

        # Test with custom system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="Your name is John Doe. Introduce yourself as such.",
        )
        assert (
            model.system_message == "Your name is John Doe. Introduce yourself as such."
        )
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")  # Capture all warnings
            model.chat("What's your name?")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "Expected no warning, but got: " + str(
            user_warnings
        )
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "John Doe" in response

        check_locals_memory_usage(locals(), test_name="test_chat_no_session")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_chat_with_session(self):
        """
        Tests for the chat method with a chat session that preserves message history.
        """
        # ChatSession: text-only flow
        cs = ChatSession()
        # With default system message, a warning should be emitted on first call
        with pytest.warns(UserWarning, match="default system message"):
            reply1 = self.llm_extractor.chat(
                "What's the result of 3+5?",
                chat_session=cs,
            )
        assert reply1 and "8" in reply1
        # System + user + assistant should be in history
        msgs = cs.messages
        assert msgs[0].role == "system"
        assert (
            msgs[1].role == "user"
            and "3+5" in msgs[1].content
            and msgs[1]._response_succeeded is True
        )
        assert msgs[2].role == "assistant" and "8" in msgs[2].content
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs)

        # Second turn within same session
        with pytest.warns(UserWarning, match="default system message"):
            reply2 = self.llm_extractor.chat(
                "And what's 10-3?",
                chat_session=cs,
            )
        assert reply2 and "7" in reply2
        msgs2 = cs.messages
        # Expect two more messages appended (user, assistant)
        assert len(msgs2) == len(msgs) + 2
        assert (
            msgs2[-2].role == "user"
            and "10-3" in msgs2[-2].content
            and msgs2[-2]._response_succeeded is True
        )
        assert msgs2[-1].role == "assistant" and "7" in msgs2[-1].content
        # Follow-up: ask about previous answer to ensure history is preserved
        with pytest.warns(UserWarning, match="default system message"):
            reply3 = self.llm_extractor.chat(
                "What was your answer to my last question? Reply with only the number.",
                chat_session=cs,
            )
        assert reply3 and "7" in reply3
        msgs3 = cs.messages
        assert len(msgs3) == len(msgs2) + 2
        assert msgs3[-2].role == "user" and msgs3[-2]._response_succeeded is True
        assert msgs3[-1].role == "assistant"
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs)

        # ChatSession: vision model with images
        cs_v = ChatSession()
        with pytest.warns(UserWarning, match="default system message"):
            r_img = self.llm_extractor.chat(
                "What's the type of this document?",
                images=[self.test_img_png_invoice],
                chat_session=cs_v,
            )
        assert r_img and "invoice" in r_img.lower()
        msgs_v = cs_v.messages
        assert len(msgs_v) == 3
        assert msgs_v[0].role == "system"
        assert (
            msgs_v[1].role == "user"
            and "type of this document" in msgs_v[1].content[0]["text"]  # type: ignore
            and msgs_v[1]._response_succeeded is True
        )
        # user message content is multimodal list when images are supplied
        assert isinstance(msgs_v[1].content, list) and len(msgs_v[1].content) == 2
        assert any(
            isinstance(i, dict) and i.get("type") == "image_url"
            for i in msgs_v[1].content
        )
        assert msgs_v[2].role == "assistant" and "invoice" in msgs_v[2].content.lower()  # type: ignore
        # Follow-up in the same vision session without re-sending images: ensure history is preserved
        with pytest.warns(UserWarning, match="default system message"):
            r_img2 = self.llm_extractor.chat(
                "What type of document did you say it was earlier?",
                chat_session=cs_v,
            )
        assert r_img2 and "invoice" in r_img2.lower()
        msgs_v2 = cs_v.messages
        assert len(msgs_v2) == len(msgs_v) + 2
        assert msgs_v2[-2].role == "user" and msgs_v2[-2]._response_succeeded is True
        assert (
            msgs_v2[-1].role == "assistant" and "invoice" in msgs_v2[-1].content.lower()  # type: ignore
        )
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_v)

        # Non-vision model must still fail with images even with a ChatSession
        text_only_model = DocumentLLM(model="cerebras/llama-3.3-70b")
        cs_empty = ChatSession()
        with pytest.raises(ValueError, match="vision"):
            text_only_model.chat(
                "What's the type of this document?",
                images=[self.test_img_png_invoice],
                chat_session=ChatSession(),
            )
        assert not cs_empty.messages
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_empty)

        # ChatSession: empty system message should not warn
        no_sys_model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )
        cs_no_sys = ChatSession()
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")
            resp = no_sys_model.chat("What's 9+1?", chat_session=cs_no_sys)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
        assert resp and "10" in resp
        # history: no system auto-injected when system_message is ""
        assert (
            len(cs_no_sys.messages) == 2
            and cs_no_sys.messages[0].role == "user"
            and "9+1" in cs_no_sys.messages[0].content
            and cs_no_sys.messages[0]._response_succeeded is True
            and cs_no_sys.messages[1].role == "assistant"
            and "10" in cs_no_sys.messages[1].content
        )
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_no_sys)

        # ChatSession: custom system message should not warn and be first
        custom_sys = "Your name is John Doe. Introduce yourself as such."
        m_custom = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message=custom_sys,
        )
        cs_custom = ChatSession()
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")
            m_custom.chat("What's your name?", chat_session=cs_custom)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
        assert cs_custom.messages and cs_custom.messages[0].role == "system"
        assert cs_custom.messages[0].content == custom_sys
        assert (
            cs_custom.messages[1].role == "user"
            and "what's your name?" in cs_custom.messages[1].content.lower()  # type: ignore
            and cs_custom.messages[1]._response_succeeded is True
        )
        assert (
            cs_custom.messages[2].role == "assistant"
            and "John Doe" in cs_custom.messages[2].content
        )
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_custom)

        # ChatSession: fallback path
        cs_fb = ChatSession()
        with pytest.warns(UserWarning, match="default system message"):
            self.invalid_llm_with_valid_fallback.chat("What's 2+2?", chat_session=cs_fb)
        # Ensure no duplicate messages in chat despite retries
        assert len(cs_fb.messages) == 3
        assert cs_fb.messages[0].role == "system"
        assert (
            cs_fb.messages[1].role == "user"
            and "what's 2+2?" in cs_fb.messages[1].content.lower()  # type: ignore
        )
        assert cs_fb.messages[1]._response_succeeded is True
        assert (
            cs_fb.messages[2].role == "assistant" and "4" in cs_fb.messages[2].content
        )

        # If the LLM call fails with an API error, the pending user message in the
        # provided ChatSession should remain with _response_succeeded == False and
        # no assistant message should be appended.
        cs_err = ChatSession()
        with pytest.raises(ModelAPIError):
            self.llm_invalid.chat("Will this fail?", chat_session=cs_err)

        msgs = cs_err.messages
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert (
            msgs[1].role == "user"
            and "will this fail" in msgs[1].content.lower()  # type: ignore
            and msgs[1]._response_succeeded is False
        )

        # Second attempt with the same prompt should not duplicate the user message
        with pytest.raises(ModelAPIError):
            self.llm_invalid.chat("Will this fail?", chat_session=cs_err)
        msgs2 = cs_err.messages
        assert len(msgs2) == 2
        assert (
            msgs2[1].role == "user"
            and "will this fail" in msgs2[1].content.lower()  # type: ignore
            and msgs2[1]._response_succeeded is False
        )

        # If both primary and fallback LLMs fail, the pending user message should
        # remain with _response_succeeded == False and no assistant message should
        # be appended.
        cs_err_fb = ChatSession()
        bad_primary = DocumentLLM(**self._invalid_llm_kwargs)  # ty: ignore[invalid-argument-type]
        invalid_llm_kwargs_2 = deepcopy(self._invalid_llm_kwargs)
        invalid_llm_kwargs_2["model"] = "newmodel/123"
        bad_primary.fallback_models = [
            DocumentLLM(
                **invalid_llm_kwargs_2,  # ty: ignore[invalid-argument-type]
                is_fallback=True,
            )
        ]

        with pytest.raises(ModelAPIError):
            bad_primary.chat("Another failing call", chat_session=cs_err_fb)

        msgs = cs_err_fb.messages
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user" and msgs[1]._response_succeeded is False

        check_locals_memory_usage(locals(), test_name="test_chat_with_session")

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "parallel_tool_calls, use_session",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    @memory_profile_and_capture
    def test_chat_with_tools(self, parallel_tool_calls: bool, use_session: bool):
        """
        Test chat with tools.
        """

        # Track invocations per tool
        calls = {"invoice": 0, "discount": 0, "surcharge": 0, "final": 0}

        # Tool 1: compute invoice total from structured list of items
        @register_tool
        def compute_invoice_total(items: list[dict]) -> str:
            """
            Computes invoice total as sum(qty*price) over items.
            """
            calls["invoice"] += 1
            total = 0
            for it in items:
                qty = int(it.get("qty", 0))
                price = int(it.get("price", 0))
                total += qty * price
            return str(total)

        # Tool 2: discount amount lookup by opaque order id (LLM cannot infer value)
        @register_tool
        def get_discount_amount(order_id: str) -> str:
            """
            Returns discount amount for a given order id.
            """
            calls["discount"] += 1
            mapping = {"ORD-123": "4"}
            return mapping.get(str(order_id), "0")

        # Tool 3: surcharge lookup by opaque order id
        @register_tool
        def get_surcharge(order_id: str) -> str:
            """
            Returns surcharge amount for a given order id.
            """
            calls["surcharge"] += 1
            mapping = {"ORD-123": "2"}
            return mapping.get(str(order_id), "0")

        # Tool 4: compute final amount from previous tool outputs
        @register_tool
        def compute_final_amount(
            invoice_total: str, discount: str, surcharge: str
        ) -> str:
            """
            Computes final amount as (invoice_total - discount) + surcharge.
            """
            calls["final"] += 1
            try:
                res = (int(invoice_total) - int(discount)) + int(surcharge)
            except Exception:
                # Be robust if model passes non-ints
                res = (int(float(invoice_total)) - int(float(discount))) + int(
                    float(surcharge)
                )
            return str(res)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "compute_invoice_total",
                    "description": "Compute invoice total as sum(qty*price) over items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "qty": {"type": "integer"},
                                        "price": {"type": "integer"},
                                    },
                                    "required": ["qty", "price"],
                                },
                                "minItems": 1,
                            }
                        },
                        "required": ["items"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_discount_amount",
                    "description": "Return discount amount for a given order id",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_surcharge",
                    "description": "Return surcharge amount for a given order id",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compute_final_amount",
                    "description": "Compute final amount as (invoice_total - discount) + surcharge",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "invoice_total": {"type": "string"},
                            "discount": {"type": "string"},
                            "surcharge": {"type": "string"},
                        },
                        "required": ["invoice_total", "discount", "surcharge"],
                    },
                },
            },
        ]

        # Configure LLM for tools; leave tool_choice to auto so it can call both
        llm_fallback = DocumentLLM(
            model="openai/gpt-5.6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            is_fallback=True,
        )
        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Test for `supports_tools` mismatch between main and fallback LLMs
        llm._supports_tools = False
        with pytest.raises(ValueError, match="supports_tools"):
            llm.fallback_models = [llm_fallback]
        llm._supports_tools = True
        llm.fallback_models = [llm_fallback]

        # Mark support exolicitly due to litellm bug where litellm.supports_parallel_function_calling()
        # returns False for all models: https://github.com/BerriAI/litellm/issues/13980
        # TODO: remove this once litellm bug is fixed
        llm._supports_parallel_tool_calls = True

        # Optional chat session to preserve history
        cs = ChatSession() if use_session else None
        prompt = (
            "Use the available tools to compute a final amount. "
            'First call `compute_invoice_total` with exactly items=[{"qty":2,"price":3},{"qty":1,"price":3}]. '
            "Then call `get_discount_amount` with order_id='ORD-123'. "
            "Then call `get_surcharge` with order_id='ORD-123'. "
            "Then call `compute_final_amount` using the exact outputs returned by the previous tools as "
            "invoice_total, discount, surcharge. "
            "Reply only with the number returned by `compute_final_amount`, nothing else. "
            "Do not guess any values; you must call the tools."
        )
        if parallel_tool_calls:
            llm._supports_parallel_tool_calls = False  # set to False to test error
            with pytest.raises(ValueError, match="parallel tool calls"):
                llm.chat(prompt, chat_session=cs)
            llm._supports_parallel_tool_calls = True
        else:
            llm.tool_max_rounds = 1
            with pytest.raises(LLMToolLoopLimitError):
                # compute_invoice_total will be called first, then error
                llm.chat(prompt, chat_session=cs)
            llm.tool_max_rounds = (
                4  # now this should pass, as we have 4 tools, 1 round per tool
            )
        answer = llm.chat(prompt, chat_session=cs)
        logger.debug(
            f"Answer with tools (parallel tool calls: {parallel_tool_calls}):\n"
            + answer
        )

        # Expected: (2*3 + 1*3) - 4 + 2 = 7
        assert answer and "7" in answer

        # Only user + assistant in primary history (when session is used)
        if use_session:
            assert cs is not None
            assert (
                len(cs.messages) == 2
                and cs.messages[0].role == "user"
                and cs.messages[1].role == "assistant"
            )

        # Each tool must be invoked exactly once
        assert (
            calls["invoice"] == 1 if parallel_tool_calls else 2
        )  # twice when parallel_tool_calls=False because of the caught LLMToolLoopLimitError above
        assert calls["discount"] == 1
        assert calls["surcharge"] == 1
        assert calls["final"] == 1

        # Usage must include at least two provider calls (tool round-trip + follow-up)
        usage_calls = llm.get_usage()[0].usage.calls
        logger.info(f"Usage calls: {len(usage_calls)}")
        assert len(usage_calls) == 3 if parallel_tool_calls else 5

        # Tool messages are stored separately when session is used
        if use_session:
            assert cs is not None
            tool_msgs = cs._tool_messages
            assert tool_msgs and any(
                m.role == "assistant" and m.tool_calls for m in tool_msgs
            )
            assert any(m.role == "tool" for m in tool_msgs)

        # Check serialization of LLM with registered tools
        self._check_deserialized_llm_config_eq(llm)

        # Warning should be emitted that tools are ignored during extraction workflows
        with pytest.warns(UserWarning, match="ignored during extraction workflows"):
            llm.extract_all(document=Document(raw_text="test"))

        check_locals_memory_usage(locals(), test_name="test_chat_with_tools")

    @memory_profile_and_capture
    def test_invalid_tool_registrations(self):
        """
        Validate errors for invalid LLM tool registrations and tool configs.
        """

        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )

        valid_schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }

        # 1) register_tool with non-callable
        with pytest.raises(TypeError, match="Tool handler must be callable"):
            register_tool(123)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        # 2) register_tool with positional-only parameter
        def define_bad_handler():
            """
            Defines invalid handler with positional-only parameter.
            """

            @register_tool
            def bad(a, /, b: int) -> str:  # positional-only arg is invalid
                """
                Invalid handler with positional-only parameter.
                """
                return str(b)

        with pytest.raises(ValueError, match="keyword arguments"):
            define_bad_handler()

        # 3) LLM.tools missing `function` dict
        with pytest.raises(
            ValueError, match="Tool function must be a non-empty dictionary"
        ):
            llm.tools = [{"type": "function"}]
        assert llm.tools is None

        # 4) Empty tool name
        with pytest.raises(ValueError, match="Tool name must be a non-empty string"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {"name": " ", "parameters": valid_schema},
                }
            ]
        assert llm.tools is None

        # 5) Parameters not a dict
        with pytest.raises(ValueError, match="Expected a JSON object"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {"name": "foo", "parameters": "not a dict"},
                }
            ]
        assert llm.tools is None

        # 6) Parameters with wrong `type`
        with pytest.raises(ValueError, match="type == 'object'"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "array",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 7) Parameters with non-dict properties
        with pytest.raises(ValueError, match="must include a 'properties' object"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "object",
                            "properties": [],
                            "required": [],
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 8) Parameters with invalid `required`
        with pytest.raises(ValueError, match="'required' must be a list of strings"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": "x",
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 9) Parameters with missing required keys in properties
        with pytest.raises(ValueError, match="missing in properties: x"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": ["x"],
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 10) Name does not match any registered handler
        with pytest.raises(
            ValueError, match="No registered handler found for tool 'unknown_tool'"
        ):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "unknown_tool",
                        "parameters": valid_schema,
                    },
                }
            ]
        assert llm.tools is None

        check_locals_memory_usage(locals(), test_name="test_invalid_tool_registrations")

    def test_logger_disabled(self, monkeypatch, capsys):
        """
        Tests for disabling the logger.
        """

        # 1) Set environment variable to disable logger
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "OFF")
        reload_logger_settings()

        # 2) Attempt to log a message
        logger.debug("This message should NOT appear.")

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Assert that the message is indeed missing
        assert "This message should NOT appear." not in captured.out

    def test_logger_enabled(self, monkeypatch, capsys):
        """
        Tests for enabling the logger.
        """

        # 1) Set environment variable to enable logger
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "DEBUG")
        reload_logger_settings()

        # 2) Log a message
        logger.debug("This message SHOULD appear.")

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Assert that the message is in the output
        assert "This message SHOULD appear." in captured.out

    @pytest.mark.parametrize(
        "env_level,should_debug,should_info,should_success,should_warning,should_error,should_critical",
        [
            ("DEBUG", True, True, True, True, True, True),
            ("INFO", False, True, True, True, True, True),
            ("SUCCESS", False, False, True, True, True, True),
            ("WARNING", False, False, False, True, True, True),
            ("ERROR", False, False, False, False, True, True),
            ("CRITICAL", False, False, False, False, False, True),
        ],
    )
    def test_log_levels(
        self,
        monkeypatch,
        capsys,
        env_level,
        should_debug,
        should_info,
        should_success,
        should_warning,
        should_error,
        should_critical,
    ):
        """
        Tests for outputting log messages of different log levels.

        For each env_level, verifies which messages appear in stdout.
        We test debug, info, success, warning, error, critical logging calls.

        Also handy for debugging format changes in the logger's output,
        e.g. overall design, icon width, etc.
        """

        # 1) Set the log level.
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, env_level)
        reload_logger_settings()

        # 2) Emit one message per level:
        def log_messages():
            """
            Helper function to emit log messages at all levels for testing.
            """
            logger.debug("DEBUG message")
            logger.info("INFO message")
            logger.success("SUCCESS message")
            logger.warning("WARNING message")
            logger.error("ERROR message")
            logger.critical("CRITICAL message")

        log_messages()

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Check presence/absence of each message
        debug_present = "DEBUG message" in captured.out
        info_present = "INFO message" in captured.out
        success_present = "SUCCESS message" in captured.out
        warning_present = "WARNING message" in captured.out
        error_present = "ERROR message" in captured.out
        critical_present = "CRITICAL message" in captured.out

        # 5) Compare actual results with expected booleans
        assert debug_present == should_debug, f"DEBUG unexpected for level {env_level}"
        assert info_present == should_info, f"INFO unexpected for level {env_level}"
        assert success_present == should_success, (
            f"SUCCESS unexpected for level {env_level}"
        )
        assert warning_present == should_warning, (
            f"WARNING unexpected for level {env_level}"
        )
        assert error_present == should_error, f"ERROR unexpected for level {env_level}"
        assert critical_present == should_critical, (
            f"CRITICAL unexpected for level {env_level}"
        )

        # 6) After the last check, output all messages for visual formatting check
        if env_level == "CRITICAL":
            monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "DEBUG")
            reload_logger_settings()
            print()
            log_messages()

    @pytest.mark.vcr
    @memory_profile_and_capture(
        max_memory=1000.0
    )  # higher limit for multiple SaT models loading and splitting
    def test_usage_examples(self):
        """
        Tests for usage examples in project's documentation and README.md.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        """
        from dev.usage_examples.docs.advanced import (
            advanced_aspects_and_concepts_document,  # noqa: F401
            advanced_aspects_with_concepts,  # noqa: F401
            advanced_multiple_docs_pipeline,  # noqa: F401
        )
        from dev.usage_examples.docs.aspects import (
            aspect_with_concepts,  # noqa: F401
            aspect_with_justifications,  # noqa: F401
            aspect_with_sub_aspects,  # noqa: F401
            basic_aspect,  # noqa: F401
            complex_hierarchy,  # noqa: F401
        )
        from dev.usage_examples.docs.concepts.boolean_concept import (
            boolean_concept,  # noqa: F401
            refs_and_justifications,
        )
        from dev.usage_examples.docs.concepts.date_concept import (
            date_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.json_object_concept import (
            adding_examples,  # noqa: F401
            json_object_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.json_object_concept.structure import (
            nested_class_structure,  # noqa: F401
            nested_structure,  # noqa: F401
            simple_class_structure,  # noqa: F401
            simple_structure,  # noqa: F401
        )
        from dev.usage_examples.docs.concepts.label_concept import (
            document_aspect_analysis,  # noqa: F401
            label_concept,  # noqa: F401
            multi_label_classification,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.numerical_concept import (
            numerical_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.rating_concept import (
            multiple_ratings,  # noqa: F401
            rating_concept,  # noqa: F401
            refs_and_justifications,  # noqa: F401,F811
        )
        from dev.usage_examples.docs.concepts.string_concept import (
            adding_examples,  # noqa: F401,F811
            refs_and_justifications,  # noqa: F401,F811
            string_concept,  # noqa: F401
        )
        from dev.usage_examples.docs.llm_config import (
            chat_with_tools,  # noqa: F401
            cost_tracking,  # noqa: F401
            detailed_usage,  # noqa: F401
            fallback_models,  # noqa: F401
            llm_api,  # noqa: F401
            llm_local,  # noqa: F401
            model_group,  # noqa: F401
            reasoning_params,  # noqa: F401
            tracking_usage_and_cost,  # noqa: F401
        )
        from dev.usage_examples.docs.llms.llm_extraction_methods import (
            extract_all,  # noqa: F401
            extract_aspects_from_document,  # noqa: F401
            extract_concepts_from_aspect,  # noqa: F401
            extract_concepts_from_document,  # noqa: F401
        )
        from dev.usage_examples.docs.llms.llm_init import (
            llm_api,  # noqa: F401,F811
            llm_local,  # noqa: F401,F811
            lm_studio_connection_error_fix,  # noqa: F401
        )
        from dev.usage_examples.docs.optimizations import (
            optimization_accuracy,  # noqa: F401
            optimization_choosing_llm,  # noqa: F401
            optimization_cost,  # noqa: F401
            optimization_long_docs,  # noqa: F401
            optimization_speed,  # noqa: F401
        )
        from dev.usage_examples.docs.quickstart import (
            quickstart_aspect,  # noqa: F401
            quickstart_concept_aspect,  # noqa: F401
            quickstart_concept_document_text,  # noqa: F401
            quickstart_concept_document_vision,  # noqa: F401
            quickstart_sub_aspect,  # noqa: F401
        )
        from dev.usage_examples.docs.serialization import serialization  # noqa: F401
        from dev.usage_examples.readme import (
            llm_chat,  # noqa: F401
            quickstart_aspect,  # noqa: F401,F811
            quickstart_concept,  # noqa: F401
        )

        check_locals_memory_usage(locals(), test_name="test_usage_examples")

    @memory_profile_and_capture
    def test_docstring_examples(self):
        """
        Tests for examples in docstrings.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        """
        from dev.usage_examples.docstrings.aspects import def_aspect  # noqa: F401
        from dev.usage_examples.docstrings.concepts import (
            def_boolean_concept,  # noqa: F401
            def_date_concept,  # noqa: F401
            def_json_object_concept,  # noqa: F401
            def_label_concept,  # noqa: F401
            def_numerical_concept,  # noqa: F401
            def_rating_concept,  # noqa: F401
            def_string_concept,  # noqa: F401
        )
        from dev.usage_examples.docstrings.data_models import (
            def_model_pricing,  # noqa: F401
        )
        from dev.usage_examples.docstrings.documents import def_document  # noqa: F401
        from dev.usage_examples.docstrings.examples import (
            def_example_json_object,  # noqa: F401
            def_example_string,  # noqa: F401
        )
        from dev.usage_examples.docstrings.images import def_image  # noqa: F401
        from dev.usage_examples.docstrings.llms import (  # noqa: F401
            def_llm,
            def_model_group,
        )
        from dev.usage_examples.docstrings.paragraphs import def_paragraph  # noqa: F401
        from dev.usage_examples.docstrings.pipelines import def_pipeline  # noqa: F401
        from dev.usage_examples.docstrings.sentences import def_sentence  # noqa: F401
        from dev.usage_examples.docstrings.utils import (
            json_object_cls_struct,  # noqa: F401
            reload_logger_settings,  # noqa: F401
        )

        check_locals_memory_usage(locals(), test_name="test_docstring_examples")

    @pytest.mark.vcr
    @memory_profile_and_capture
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

        check_locals_memory_usage(
            locals(), test_name="test_extract_with_additional_context"
        )

    @pytest.mark.vcr
    @memory_profile_and_capture(max_memory=1000.0)
    # higher value to allocate memory for SaT model sentence splitting in a very long document
    # (2000+ sentences)
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

        check_locals_memory_usage(
            locals(), test_name="test_very_long_doc_extraction", max_obj_memory=5.0
        )  # higher value for a very long document (200+ pages)

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_benchmarks(self):
        """
        Executes extraction benchmarks to evaluate the accuracy and reliability
        of the extraction pipeline.

        This test runs a suite of benchmark scenarios and asserts that the total benchmark
        scores meet or exceed a set minimum score.

        The benchmarks are particularly important for validating the impact of prompt changes
        and ensuring that extraction quality remains high as the system evolves. If a benchmark
        score falls below the threshold, the test fails, signaling a potential regression in
        extraction performance.
        """

        run_benchmark_for_module(
            llm=self.llm_extractor,
            judge_llm=self.llm_reasoner,
            module_name="tests.benchmark.configs.dev_contract",
            benchmark_name="Dev Contract (Text Only)",
        )

        check_locals_memory_usage(locals(), test_name="test_benchmarks")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_auto_pricing(self):
        """
        Tests for optional auto-pricing using genai-prices when no manual pricing is provided.
        """

        # Use an LLM with auto pricing enabled and no manual pricing details
        with pytest.warns(UserWarning, match="prices will not be 100% accurate."):
            llm = DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                auto_pricing=True,
                auto_pricing_refresh=True,  # test with auto-refresh
            )
        assert (
            llm._auto_pricing_refresh_attempted
        )  # since we use dummy call to check auto-pricing support for provider/model
        document = Document(raw_text=get_test_document_text())
        document.concepts = [
            RatingConcept(
                name="NDA compliance rating",
                description="Rating of the NDA's compliance with best practices",
                rating_scale=(1, 10),
                add_justifications=True,
                justification_max_sents=5,
            ),
        ]
        llm.extract_concepts_from_document(document)
        cost_info = llm.get_cost()
        assert len(cost_info) == 1
        cost = cost_info[0].cost
        assert cost.input > Decimal("0")
        assert cost.output > Decimal("0")
        assert cost.total == cost.input + cost.output
        logger.debug(
            f"Costs for model {llm.model} (auto-pricing): "
            f"input {cost.input}, output {cost.output}, total {cost.total}"
        )

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm)

        # Test errors

        # Local LLMs
        with pytest.raises(ValueError, match="local models"):
            llm = DocumentLLM(
                model="lm_studio/mistralai/mistral-small-3.2",
                api_base="http://localhost:1234/v1",
                api_key="random-key",  # required for LM Studio API
                auto_pricing=True,
            )
        with pytest.raises(ValueError, match="local models"):
            DocumentLLM(
                model="ollama_chat/mistral-small:24b",
                api_base="http://localhost:11434",
                auto_pricing=True,
            )
        with pytest.raises(ValueError, match="local models"):
            DocumentLLM(
                model="ollama/mistral-small:24b",
                api_base="http://localhost:11434",
                auto_pricing=True,
            )

        # Setting auto-pricing and ModelPricing together
        with pytest.raises(ValueError, match="auto_pricing=True"):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.00015, output_per_1m_tokens=0.0006
                ),
                auto_pricing=True,
            )

        # Test warnings

        # Unknown model initialization
        with pytest.warns(UserWarning, match="Unable to fetch pricing data for model"):
            llm = DocumentLLM(
                model="openai/gptXX",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                auto_pricing=True,
            )

        check_locals_memory_usage(locals(), test_name="test_auto_pricing")
