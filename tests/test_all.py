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

import os
import platform
import re
import sys
import tempfile
import warnings
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional  # noqa: UP035

import pytest
from _pytest.nodes import Item as PytestItem
from dotenv import load_dotenv
from lxml import etree
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator

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
from contextgem.internal.base.data_models import _LLMPricing, _RatingScale
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.examples import _JsonObjectExample, _StringExample
from contextgem.internal.base.images import _Image
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.base.llms import (
    _DocumentLLM,
    _DocumentLLMGroup,
)
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.base.pipelines import _DocumentPipeline, _ExtractionPipeline
from contextgem.internal.base.utils import _JsonObjectClassStruct
from contextgem.internal.converters.docx import _DocxPackage
from contextgem.internal.converters.docx.utils import WORD_XML_NAMESPACES
from contextgem.internal.data_models import _LLMCost, _LLMUsage
from contextgem.internal.exceptions import (
    DocxConverterError,
    DocxFormatError,
    LLMAPIError,
    LLMExtractionError,
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
    dedicated_stream,
    logger,
)
from contextgem.internal.registry import (
    _publicize,
    _resolve_public_type,
)
from contextgem.internal.utils import (
    _get_template,
    _load_sat_model,
    _split_text_into_paragraphs,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentLLMGroup,
    DocxConverter,
    ExtractionPipeline,
    Image,
    JsonObjectClassStruct,
    JsonObjectConcept,
    JsonObjectExample,
    LabelConcept,
    LLMPricing,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    RatingScale,
    Sentence,
    StringConcept,
    StringExample,
    reload_logger_settings,
)
from contextgem.public.pipelines import DocumentPipeline
from contextgem.public.utils import create_image, image_to_base64
from tests.benchmark.core import run_benchmark_for_module
from tests.conftest import VCR_REDACTION_MARKER
from tests.memory_profiling import check_locals_memory_usage, memory_profile_and_capture
from tests.url_security import validate_existing_cassettes_urls_security
from tests.utils import (
    VCR_FILTER_HEADERS,
    TestUtils,
    get_project_root_path,
    get_test_document_text,
    get_test_img,
    read_text_file,
    remove_file,
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
    default_system_message_en = _get_template(
        "default_system_message",
        template_type="system",
        template_extension="j2",
    ).render({"output_language": "en"})  # type: ignore[attr-defined]
    default_system_message_non_en = _get_template(
        "default_system_message",
        template_type="system",
        template_extension="j2",
    ).render({"output_language": "adapt"})  # type: ignore[attr-defined]

    # Documents
    # From raw texts
    document = Document(raw_text=get_test_document_text())
    document_ua = Document(raw_text=get_test_document_text(lang="ua"))
    document_zh = Document(raw_text=get_test_document_text(lang="zh"))
    # From DOCX files
    test_docx_nda_path = os.path.join(
        get_project_root_path(), "tests", "docx_files", "en_nda_with_anomalies.docx"
    )
    test_docx_nda_ua_path = os.path.join(
        get_project_root_path(), "tests", "docx_files", "ua_nda_with_anomalies.docx"
    )
    test_docx_badly_formatted_path = os.path.join(
        get_project_root_path(), "tests", "docx_files", "badly_formatted.docx"
    )
    document_docx = DocxConverter().convert(test_docx_nda_path)
    document_docx_ua = DocxConverter().convert(test_docx_nda_ua_path)
    # Match badly formatted converted DOCX content (default mode - all content is included)
    test_badly_formatted_converted_md_text = read_text_file(
        os.path.join(
            get_project_root_path(),
            "tests",
            "docx_converted",
            "badly_formatted_md.txt",
        )
    )
    test_badly_formatted_converted_raw_text = read_text_file(
        os.path.join(
            get_project_root_path(),
            "tests",
            "docx_converted",
            "badly_formatted_raw.txt",
        )
    )
    test_badly_formatted_converted_md_paras_text = read_text_file(
        os.path.join(
            get_project_root_path(),
            "tests",
            "docx_converted",
            "badly_formatted_paras_md.txt",
        )
    )
    test_badly_formatted_converted_raw_paras_text = read_text_file(
        os.path.join(
            get_project_root_path(),
            "tests",
            "docx_converted",
            "badly_formatted_paras_raw.txt",
        )
    )

    # Extraction pipeline
    extraction_pipeline = ExtractionPipeline(
        aspects=[
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                llm_role="extractor_text",
            ),
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                llm_role="extractor_text",
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
                llm_role="extractor_text",
                add_justifications=True,
                add_references=True,
            ),
            BooleanConcept(
                name="Invoice number check",
                description="Invoice number is present in the document",
                llm_role="extractor_vision",
                add_justifications=True,
            ),
        ],
    )
    # Deprecated DocumentPipeline class (will be removed in v1.0.0)
    document_pipeline = DocumentPipeline(
        aspects=extraction_pipeline.aspects,
        concepts=extraction_pipeline.concepts,
    )

    # LLMs

    # Extractor text
    _llm_extractor_text_kwargs_openai = {
        "model": "azure/gpt-4.1-mini",
        "api_key": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
        "role": "extractor_text",
        "pricing_details": LLMPricing(  # Explicitly set pricing details
            **{
                "input_per_1m_tokens": 0.40,
                "output_per_1m_tokens": 1.60,
            }
        ),
    }

    # Reasoner text
    _llm_reasoner_text_kwargs_openai = {
        "model": "azure/gpt-5-mini",
        "api_key": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
        "api_version": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
        "api_base": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
        "role": "reasoner_text",
        "auto_pricing": True,  # use auto-pricing
        "auto_pricing_refresh": False,  # avoid network auto-refresh during tests
    }

    # Extractor text
    llm_extractor_text = DocumentLLM(**_llm_extractor_text_kwargs_openai)

    # Reasoner text
    llm_reasoner_text = DocumentLLM(**_llm_reasoner_text_kwargs_openai)

    # Extractor vision
    _llm_extractor_vision_kwargs_openai = deepcopy(_llm_extractor_text_kwargs_openai)
    _llm_extractor_vision_kwargs_openai["role"] = "extractor_vision"
    llm_extractor_vision = DocumentLLM(**_llm_extractor_vision_kwargs_openai)

    # Reasoner vision
    _llm_reasoner_vision_kwargs_openai = deepcopy(_llm_reasoner_text_kwargs_openai)
    _llm_reasoner_vision_kwargs_openai["role"] = "reasoner_vision"
    llm_reasoner_vision = DocumentLLM(**_llm_reasoner_vision_kwargs_openai)

    # LLM group
    _llm_group_llms = [
        DocumentLLM(**_llm_extractor_text_kwargs_openai),
        DocumentLLM(**_llm_reasoner_text_kwargs_openai),
        DocumentLLM(**_llm_extractor_vision_kwargs_openai),
        DocumentLLM(**_llm_reasoner_vision_kwargs_openai),
    ]  # newly initialized LLMs in group for separate usage and cost tracking)
    llm_group = DocumentLLMGroup(llms=_llm_group_llms)

    # LLM with fallback
    _invalid_llm_kwargs = {
        "model": "helloworld/helloworld-v1",
        "api_key": "invalid_api_key",
        "role": "extractor_text",
    }
    llm_invalid = DocumentLLM(**_invalid_llm_kwargs)
    invalid_llm_with_valid_fallback = DocumentLLM(**_invalid_llm_kwargs)
    invalid_llm_with_valid_fallback.fallback_llm = DocumentLLM(
        **_llm_extractor_text_kwargs_openai,
        is_fallback=True,
    )

    # LLM with a non-EN output language setting
    llm_extractor_text_non_eng = DocumentLLM(
        **_llm_extractor_text_kwargs_openai, output_language="adapt"
    )

    # Images
    # Invoices
    test_img_png_invoice = get_test_img("invoice.png", "invoices")
    test_img_jpg_invoice = get_test_img("invoice.jpg", "invoices")
    test_img_jpg_2_invoice = get_test_img("invoice2.jpg", "invoices")
    test_img_webp_invoice = get_test_img("invoice.webp", "invoices")
    # Other
    test_img_png_apt_plan = get_test_img("apt_plan.png", "other")

    # Memory profiling
    memory_baseline: float = 0.0  # baseline memory usage
    memory_profiles: dict[str, str] = {}  # memory usage profiles
    memory_deltas: dict[str, float] = {}  # memory usage deltas (relative to baseline)

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
            llm_role="reasoner_text",
            reference_depth="sentences",
        )
        assert isinstance(pub_aspect, Aspect)
        assert pub_aspect.llm_role == "reasoner_text"
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
        # LLM usage model
        _LLMUsage(input=30, output=10)
        with pytest.raises(ValueError):
            _LLMUsage(input=-1, output=2)
        # LLM pricing model
        LLMPricing(input_per_1m_tokens=0.00015, output_per_1m_tokens=0.0006)
        with pytest.raises(ValueError):
            LLMPricing(input_per_1m_tokens=-1, output_per_1m_tokens=0)
        with pytest.raises(ValueError):
            LLMPricing(input_per_1m_tokens=0.1)  # type: ignore
        # LLM cost model
        _LLMCost(input=Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03"))
        with pytest.raises(ValueError):
            _LLMCost(
                input=-Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03")
            )
        with pytest.raises(ValueError):
            _LLMCost(input=-Decimal("0.001"))

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
            _LLMPricing,
            _RatingScale,
            _StringExample,
            _JsonObjectExample,
            _Image,
            _Document,
            _Paragraph,
            _Sentence,
            _ExtractionPipeline,
            _DocumentPipeline,
            _DocumentLLM,
            _DocumentLLMGroup,
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

            class TestNoRequiredAttrs(base_class):
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
        self.llm_extractor_text.extract_aspects_from_document(document)
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
        self.llm_extractor_text.extract_concepts_from_document(document)

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
        self.llm_extractor_text.extract_aspects_from_document(document)
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
        self.llm_extractor_text.extract_all(document)
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
        self.llm_extractor_text.extract_aspects_from_document(
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
        self.llm_extractor_text.extract_concepts_from_document(
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
        self.llm_extractor_text.extract_all(document)
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

        def extract_with_local_llm(llm: DocumentLLM):
            """
            Helper function to test extraction with a local LLM model.

            :param llm: The DocumentLLM instance to test extraction with.
            :type llm: DocumentLLM
            """
            # Configure document
            document = Document(
                raw_text=get_test_document_text()[:1000],
            )
            concept = StringConcept(
                name="Contract title",
                description="The title of the contract.",
                llm_role=llm.role,
            )
            document.add_concepts([concept])

            # Run extraction
            self.config_llm_async_limiter_for_mock_responses(llm)
            extracted_concepts = llm.extract_concepts_from_document(document)
            assert llm.get_usage()[0].usage.calls[-1].prompt
            assert llm.get_usage()[0].usage.calls[-1].response
            extracted_items = extracted_concepts[0].extracted_items
            assert len(extracted_items), (
                f"No extracted items returned with local LLM {llm.model}"
            )
            self.log_extracted_items_for_instance(extracted_concepts[0])

            # Check serialization of LLM
            self._check_deserialized_llm_config_eq(llm)

        # Ollama
        # Non-reasoning LLM
        llm_non_reasoning_ollama = DocumentLLM(
            model="ollama/mistral-small:24b",
            api_base="http://localhost:11434",
            seed=123,
        )
        extract_with_local_llm(llm_non_reasoning_ollama)
        # Reasoning (CoT-capable) LLM
        llm_reasoning_ollama = DocumentLLM(
            model="ollama/deepseek-r1:32b",
            api_base="http://localhost:11434",
            seed=123,
            role="reasoner_text",
        )
        llm_reasoning_ollama._supports_reasoning = True
        extract_with_local_llm(llm_reasoning_ollama)

        # LM Studio
        # Non-reasoning LLM
        llm_non_reasoning_lm_studio = DocumentLLM(
            model="lm_studio/mistralai/mistral-small-3.2",
            api_base="http://localhost:1234/v1",
            api_key="random-key",  # required for LM Studio API
            seed=123,
        )
        extract_with_local_llm(llm_non_reasoning_lm_studio)

        check_locals_memory_usage(locals(), test_name="test_local_llms_text")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_local_llms_text_gpt_oss(self):
        """
        Tests for initialization of and getting a response from gpt-oss models.
        """

        def extract_with_local_llm(llm: DocumentLLM):
            """
            Test extraction with local LLM.

            :param llm: The DocumentLLM instance to test.
            """

            # Configure document
            document = Document(raw_text=get_test_document_text())
            aspect = Aspect(
                name="Liability",
                description="Liability clauses",
                llm_role=llm.role,
            )
            aspect_concept = StringConcept(
                name="Liability cap",
                description="Liability cap",
                llm_role=llm.role,
            )
            aspect.add_concepts([aspect_concept])
            document_concept = NumericalConcept(
                name="Contract term",
                description="Contract term in years",
                numeric_type="float",
                llm_role=llm.role,
            )
            document.add_aspects([aspect])
            document.add_concepts([document_concept])

            # Run extraction
            llm.extract_all(document)
            assert document.aspects[0].extracted_items
            self.log_extracted_items_for_instance(document.aspects[0])
            assert document.concepts[0].extracted_items
            self.log_extracted_items_for_instance(document.concepts[0])
            assert document.aspects[0].concepts[0].extracted_items
            self.log_extracted_items_for_instance(document.aspects[0].concepts[0])

            # Check serialization of LLM
            self._check_deserialized_llm_config_eq(llm)

        # gpt-oss works with Ollama
        llm_gpt_oss_ollama = DocumentLLM(
            model="ollama_chat/gpt-oss:20b",
            api_base="http://localhost:11434",
            role="reasoner_text",
        )
        llm_gpt_oss_ollama._supports_reasoning = True
        extract_with_local_llm(llm_gpt_oss_ollama)

        # TODO: Remove this once LiteLLM's `lm_studio` gpt-oss support is fixed
        # But does not work with `lm_studio/` prefix
        with pytest.raises((LLMAPIError, LLMExtractionError)):
            with pytest.warns(UserWarning, match="gpt-oss"):
                llm_gpt_oss_lm_studio = DocumentLLM(
                    model="lm_studio/openai/gpt-oss-20b",
                    api_base="http://localhost:1234/v1",
                    api_key="random-key",  # required for LM Studio API
                    role="reasoner_text",
                )
            llm_gpt_oss_lm_studio._supports_reasoning = True
            extract_with_local_llm(llm_gpt_oss_lm_studio)

        check_locals_memory_usage(locals(), test_name="test_local_llms_text_gpt_oss")

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
                llm_role="extractor_vision",
                add_justifications=True,
            ),
        ]
        document = Document(
            images=[self.test_img_png_invoice], concepts=document_concepts
        )
        # Warn about using ollama/ prefix for local vision models
        with pytest.warns(UserWarning, match="use `ollama/` prefix instead"):
            DocumentLLM(
                model="ollama_chat/gemma3:27b",
                api_base="http://localhost:11434",
                role="extractor_vision",
            )

        # Ollama
        # Warn about the model not being detected as vision-capable
        # TODO: remove this warning once litellm supports vision for this model
        with pytest.warns(UserWarning, match="vision-capable"):
            llm_ollama = DocumentLLM(
                model="ollama/gemma3:27b",
                api_base="http://localhost:11434",
                role="extractor_vision",
            )
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
    def test_llm_extraction_error_exception(self):
        """
        Tests for raising an exception when an LLM extraction error occurs.
        """

        @contextmanager
        def capture_logger_warnings():
            """
            Context manager to capture logger WARNING messages
            """
            captured_logs = []

            def capture_warning(message):
                """
                Captures a logger WARNING message.
                """
                captured_logs.append(message.record["message"])

            handler_id = logger.add(
                capture_warning, level="WARNING", format="{message}"
            )
            try:
                yield captured_logs
            finally:
                logger.remove(handler_id)

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
        with pytest.raises(LLMExtractionError, match=r"invalid JSON.*1 retries"):
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
        with pytest.raises(LLMExtractionError, match=r"invalid JSON.*1 retries"):
            llm.extract_concepts_from_document(
                document,
                overwrite_existing=True,
            )
        with pytest.raises(LLMExtractionError, match=r"invalid JSON.*1 retries"):
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

        # === Without retries ===
        # raise_exception_on_extraction_error is True by default
        llm = DocumentLLM(
            model="ollama/llama3.1:8b",
            api_base="http://localhost:11434",
            max_retries_invalid_data=0,
        )
        with pytest.raises(LLMExtractionError, match=r"invalid JSON.*0 retries"):
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
        with pytest.raises(LLMExtractionError, match=r"invalid JSON.*0 retries"):
            llm.extract_concepts_from_document(
                document,
                overwrite_existing=True,
            )
        with pytest.raises(LLMExtractionError, match=r"invalid JSON.*0 retries"):
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

        # === Test invalid LLM without fallback LLM ===
        # raise_exception_on_extraction_error is True by default
        with pytest.raises(LLMAPIError, match=r"LLM API.*3 retries"):
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
        with pytest.raises(LLMAPIError, match=r"LLM API.*3 retries"):
            self.llm_invalid.extract_concepts_from_document(
                document,
                overwrite_existing=True,
            )
        with pytest.raises(LLMAPIError, match=r"LLM API.*3 retries"):
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
            locals(), test_name="test_llm_extraction_error_exception"
        )

    @memory_profile_and_capture
    def test_init_api_llm(self):
        """
        Tests the behaviour of the `DocumentLLM` class initialization.
        """
        DocumentLLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )

        # Optional params
        DocumentLLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            top_p=None,
            temperature=None,
        )

        # Unsupported methods
        with pytest.raises(NotImplementedError):
            self.llm_group.model_dump()
        with pytest.raises(NotImplementedError):
            self.llm_group.model_dump_json()
        with pytest.raises(NotImplementedError):
            self.llm_extractor_text.model_dump()
        with pytest.raises(NotImplementedError):
            self.llm_extractor_text.model_dump_json()

        # Invalid params
        with pytest.raises(ValueError):
            DocumentLLM()
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                extra=True,  # extra fields not permitted
            )
        with pytest.warns(UserWarning, match="vision-capable"):
            DocumentLLM(
                model="openai/gpt-3.5-turbo",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_vision",  # model does not support vision
            )

        # Fallback LLM validation
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                fallback_llm=self.llm_reasoner_vision,  # inconsistent role of the fallback LLM
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                fallback_llm=DocumentLLM(
                    model="openai/gpt-4o-mini",
                    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                    role="extractor_text",
                    is_fallback=True,
                ),  # same params as the main LLM, except for `is_fallback` flag
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                is_fallback=True,
                fallback_llm=self.llm_extractor_text,  # a fallback LLM is passed to fallback LLM
            )
        # Validate fallback LLM assignment
        document_llm = DocumentLLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert document_llm.is_group is False
        document_llm.is_fallback = True
        with pytest.raises(ValueError):
            document_llm.fallback_llm = (
                self.llm_extractor_text
            )  # a fallback LLM is passed to fallback LLM
        assert document_llm.fallback_llm is None  # No change due to invalid assignment
        document_llm.is_fallback = False
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                is_fallback=False,
                fallback_llm=document_llm,  # non-fallback LLM passed as a fallback
            )

        with pytest.raises(ValueError):
            DocumentLLM(
                model="gpt-4o-mini",  # no provider prefix in model name
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="processor",  # invalid role
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                temperature=-1.0,  # negative temperature
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                max_tokens=-4000,  # negative value
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                pricing_details=LLMPricing(
                    **{  # invalid pricing details keys
                        "in": 0.1,
                        "out": 0.2,
                    }
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_text",
                pricing_details=LLMPricing(
                    **{  # invalid pricing details values
                        "input_per_1m_tokens": -0.00015,
                        "output_per_1m_tokens": 0.0006,
                    }
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key="",  # empty api key
                role="extractor_text",
            )

        check_locals_memory_usage(locals(), test_name="test_init_api_llm")

    @memory_profile_and_capture
    def test_init_llm_group(self):
        """
        Tests the behavior of the `DocumentLLMGroup` class initialization.
        """

        # Invalid params
        with pytest.raises(ValueError):
            DocumentLLMGroup()  # type: ignore
        with pytest.raises(ValueError):
            DocumentLLMGroup(llms=[])
        with pytest.raises(ValueError):
            DocumentLLMGroup(llms=[self.llm_reasoner_text])
        with pytest.raises(ValueError):
            DocumentLLMGroup(llms=[1, True])  # type: ignore
        # Duplicate roles
        with pytest.raises(ValueError):
            DocumentLLMGroup(
                llms=[
                    self.llm_extractor_text,
                    self.llm_extractor_text,
                    self.llm_reasoner_text,
                ]
            )
        with pytest.raises(ValueError):
            DocumentLLMGroup(
                llms=[self.llm_extractor_text, self.llm_reasoner_text],
                extra=True,  # extra fields not permitted # type: ignore
            )
        with pytest.raises(ValueError):
            # Non-consistent output languages of the LLMs in the group
            DocumentLLMGroup(
                llms=[
                    self.llm_extractor_text,
                    self.llm_extractor_text_non_eng,  # output lang set to "adapt"
                ],
                output_language="en",
            )
        document_llm_group = DocumentLLMGroup(
            llms=[self.llm_extractor_text, self.llm_reasoner_text]
        )
        assert document_llm_group.is_group is True
        # Validate assignment
        with pytest.raises(ValueError):
            document_llm_group.llms = [self.llm_extractor_text, self.llm_extractor_text]
        assert document_llm_group.llms == [
            self.llm_extractor_text,
            self.llm_reasoner_text,
        ]

        # No serialization for LLM instances
        with pytest.raises(NotImplementedError):
            document_llm_group.model_dump()
        with pytest.raises(NotImplementedError):
            document_llm_group.model_dump_json()

        check_locals_memory_usage(locals(), test_name="test_init_llm_group")

    @memory_profile_and_capture
    def test_update_default_prompt(self):
        """
        Tests for updating the default prompt for the LLM.
        """
        # Test custom prompts assignment
        llm_with_updated_prompts = DocumentLLM(
            **self._llm_extractor_text_kwargs_openai,
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
            Image()  # type: ignore
        with pytest.raises(ValueError):
            Image(
                mime_type="image/weoighwe",  # invalid MIME type  # type: ignore
                base64_data="base 64 encoded string",
            )
        with pytest.raises(ValueError):
            Image(
                mime_type="image/png",
                base64_data="base 64 encoded string",
                extra=True,  # extra fields not permitted  # type: ignore
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
        assert not paragraph._md_text  # markdown text is not populated from raw text
        document = Document(paragraphs=[paragraph])
        assert document.raw_text  # to be populated from paragraphs
        assert not document._md_text  # markdown text is not populated from paragraphs
        with pytest.raises(ValueError):
            Paragraph()  # type: ignore
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
        assert not hasattr(
            sentence, "_md_text"
        )  # markdown text does not apply to sentences
        Paragraph(raw_text=sentence.raw_text, sentences=[sentence])
        # Warning is logged if linebreaks occur in additional context
        with pytest.raises(ValueError):
            Sentence()  # type: ignore
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
    def test_init_aspect(self):
        """
        Tests the initializing and error handling of the `Aspect` class.
        """

        with pytest.raises(ValueError):
            Aspect()  # type: ignore
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                extra=True,  # extra fields not permitted  # type: ignore
            )
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                llm_role="extractor_vision",  # invalid LLM role  # type: ignore
            )
        aspect = Aspect(
            name="Categories of confidential information",
            description="Clauses describing confidential information covered by the NDA",
            llm_role="extractor_text",
            add_justifications=True,
        )
        aspect_concepts = [
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
                llm_role="extractor_text",
            ),
            StringConcept(
                name="Technical Information",
                description="Categories of Technical Information",
                llm_role="reasoner_text",
            ),
        ]
        aspect.concepts = aspect_concepts
        assert aspect.concepts is not aspect_concepts
        # Validate assignment
        with pytest.raises(ValueError, match="non-empty list"):
            aspect.add_aspects([])  # empty list
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [
                    1,  # invalid type  # type: ignore
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
                        llm_role="extractor_text",
                        add_justifications=True,
                    )
                ]
            )
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [
                    StringConcept(
                        name="Random",
                        description="Random",
                        llm_role="extractor_vision",  # invalid llm role
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
            )  # tuple instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = set([1, 2, 3])  # set instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            aspect.concepts = range(10)  # range instead of list  # type: ignore

        # Test adding a sub-aspect with the same name or description as the main aspect
        sub_aspect_same_name = Aspect(
            name="Categories of confidential information",  # Same name as parent aspect
            description="Different description",
            llm_role="extractor_text",
        )
        with pytest.raises(ValueError):
            aspect.add_aspects([sub_aspect_same_name])

        sub_aspect_same_description = Aspect(
            name="Different name",
            description="Clauses describing confidential information covered by the NDA",  # Same description as parent aspect
            llm_role="extractor_text",
        )
        with pytest.raises(ValueError):
            aspect.add_aspects([sub_aspect_same_description])

        # Test that only one level of nesting is allowed for sub-aspects
        main_aspect = Aspect(
            name="Main Aspect",
            description="Main aspect description",
            llm_role="extractor_text",
        )
        sub_aspect = Aspect(
            name="Sub Aspect",
            description="Sub aspect description",
            llm_role="reasoner_text",
        )
        sub_sub_aspect = Aspect(
            name="Sub-Sub Aspect",
            description="Sub-sub aspect description",
            llm_role="extractor_text",
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
        logger.debug(sub_aspect.aspects)
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
                    llm_role="extractor_text",
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
            llm_role="extractor_text",
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
        string_concept.extracted_items = [_StringItem(value="Confidential data")]

        # Test with invalid items
        with pytest.raises(ValueError):
            string_concept.extracted_items = [
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
            llm_role="extractor_text",
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
        boolean_concept.extracted_items = [_BooleanItem(value=True)]

        # Test with invalid items
        with pytest.raises(ValueError):
            boolean_concept.extracted_items = [
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
        ) = self.llm_extractor_text.extract_concepts_from_document(
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
        (concept_multiple_items,) = (
            self.llm_extractor_text.extract_concepts_from_document(document)
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
        ) = self.llm_extractor_text.extract_concepts_from_document(
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
        int_concept.extracted_items = [int_item]

        # Should reject float items for int concept
        with pytest.raises(ValueError):
            int_concept.extracted_items = [_FloatItem(value=42.5)]

        # Float concept should accept only float items
        float_item = _FloatItem(value=3.14)
        float_concept.extracted_items = [float_item]

        # Should reject integer items for float concept
        with pytest.raises(ValueError):
            float_concept.extracted_items = [_IntegerItem(value=3)]

        # Any numeric concept should accept both integer and float items
        any_concept.extracted_items = [_IntegerOrFloatItem(value=20)]
        any_concept.extracted_items = [_IntegerOrFloatItem(value=20.5)]
        # Should reject integer items for any numeric concept
        with pytest.raises(ValueError):
            any_concept.extracted_items = [_IntegerItem(value=10)]
        # Should reject float items for any numeric concept
        with pytest.raises(ValueError):
            any_concept.extracted_items = [_FloatItem(value=10.5)]

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

        # Test deprecated RatingScale
        with pytest.warns(DeprecationWarning):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=RatingScale(start=1, end=5),
            )

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
            _Concept(name="Test", description="Test")  # type: ignore
        with pytest.raises(ValueError):
            JsonObjectConcept()  # type: ignore
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure={
                    "test": str,
                },
                extra=True,  # extra fields not permitted # type: ignore
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
                llm_role="extractor_vision",
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
                llm_role="extractor_text",
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
                llm_role="extractor_text",
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={int: "integer"},  # invalid mapping  # type: ignore
                llm_role="extractor_text",
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={},  # empty mapping
                llm_role="extractor_text",
            )

        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Invalid Structure",
                description="Testing invalid structure",
                structure={"random": Exception},  # invalid value type
                llm_role="extractor_text",
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
                llm_role="extractor_vision",
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
            llm_role="extractor_vision",
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
            llm_role="extractor_text",
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
        extracted_concepts = self.llm_extractor_text.extract_concepts_from_document(
            document
        )

        # Verify prompt content from the LLM call log
        prompt_string = self.llm_extractor_text.get_usage()[-1].usage.calls[-1].prompt
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
        default_date_concept.extracted_items = [_DateItem(value=date_obj)]

        # Test with invalid date string
        with pytest.raises(ValueError):
            date_str = "2025-01-01"
            date_obj = default_date_concept._string_to_date(date_str)

        # Test with invalid items
        with pytest.raises(ValueError):
            default_date_concept.extracted_items = [
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
            llm_role="extractor_text",
            add_justifications=True,
        )

        # Valid initialization with multi-label classification
        multi_label_concept = LabelConcept(
            name="Content Topics",
            description="Identify all relevant topics covered in the document",
            labels=["Finance", "Legal", "Technology", "HR", "Operations", "Marketing"],
            classification_type="multi_label",
            llm_role="reasoner_text",
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
            multi_class_concept.extracted_items = [
                _StringItem(value="NDA")  # type: ignore
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
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_extract_label_concept(self, llm: DocumentLLMGroup | DocumentLLM):
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
            llm_role="extractor_text",
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
            llm_role="extractor_text",
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
            llm_role="extractor_text",
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
            llm_role="extractor_text",
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
            _StringItem()  # type: ignore
        with pytest.raises(ValueError):
            _BooleanItem(value=1)  # type: ignore
        with pytest.raises(ValueError):
            _BooleanItem()  # type: ignore
        with pytest.raises(ValueError):
            _BooleanItem(value="True")  # type: ignore
        with pytest.raises(ValueError):
            _IntegerOrFloatItem()  # type: ignore
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=int)  # type: ignore
        with pytest.raises(ValueError):
            _JsonObjectItem()  # type: ignore
        with pytest.raises(ValueError):
            _JsonObjectItem(value={})
        with pytest.raises(ValueError):
            _JsonObjectItem(value={"hello": {}})
        with pytest.raises(ValueError):
            _StringItem(
                value="Random string",
                extra=True,  # extra fields not permitted  # type: ignore
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

    @pytest.mark.parametrize(
        "context", [document, extraction_pipeline, document_pipeline]
    )
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
                "sat-6l-sm",
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
            assert (
                not document._md_text
            )  # markdown text is not populated from paragraphs
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
                    extra=True,  # extra fields not permitted  # type: ignore
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
                    extra=True,  # extra fields not permitted  # type: ignore
                )
            ExtractionPipeline(
                aspects=[
                    Aspect(
                        name="Liability",
                        description="Clauses describing liability of the parties",
                        llm_role="extractor_text",
                        add_justifications=True,
                    )
                ]
            )
            ExtractionPipeline(
                concepts=[
                    StringConcept(
                        name="Business Information",
                        description="Categories of Business Information",
                        llm_role="extractor_text",
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
                llm_role="extractor_text",
            ),
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                llm_role="extractor_text",
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
                [
                    "Random string",  # type: ignore
                ]  # invalid aspect type
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="liability",
                        description="clauses describing liability of the parties",
                        llm_role="extractor_text",
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
                        llm_role="extractor_vision",  # unsupported llm role for aspect  # type: ignore
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
            )  # tuple instead of list  # type: ignore
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
                llm_role="extractor_text",
                add_justifications=True,
            ),
            BooleanConcept(
                name="Contract type check",
                description="Contract is an NDA",
                llm_role="extractor_vision",
                add_justifications=True,
            ),
        ]
        context.concepts = document_concepts
        assert context.concepts is not document_concepts
        # Validate assignment
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [123]  # invalid concept type
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [
                StringConcept(
                    name="business information",
                    description="categories of business information",
                    llm_role="extractor_text",
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
                    llm_role="reasoner_text",
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
            model="azure/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
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
            model="azure/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            max_tokens=999999,  # excessive output tokens
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
        non_reasoning_model = DocumentLLM(
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            system_message=system_message,
        )
        with pytest.warns(UserWarning, match="reasoning-capable"):
            # Default role "extractor_text" is used; warn that the model is reasoning-capable
            # and "reasoner_*" roles are recommended for higher quality responses
            o1_model = DocumentLLM(
                model="azure/o1",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                reasoning_effort="low",
                system_message=system_message,
            )
        with pytest.warns(UserWarning, match="reasoning-capable"):
            # Default role "extractor_text" is used; warn that the model is reasoning-capable
            # and "reasoner_*" roles are recommended for higher quality responses
            o3_mini_model = DocumentLLM(
                model="azure/o3-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                system_message=system_message,
            )
        with pytest.warns(UserWarning, match="reasoning-capable"):
            # Default role "extractor_text" is used; warn that the model is reasoning-capable
            # and "reasoner_*" roles are recommended for higher quality responses
            o4_mini_model = DocumentLLM(
                model="azure/o4-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                system_message=system_message,
                reasoning_effort="low",
            )
        for model in [non_reasoning_model, o1_model, o3_mini_model, o4_mini_model]:
            model.chat("What's your name?")
            response = model.get_usage()[0].usage.calls[-1].response
            assert response is not None
            assert "ContextGem" in response
            logger.debug(response)

        # Test empty system message
        model = DocumentLLM(
            model="azure/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            system_message="",
        )
        assert model.system_message == ""

        # Test None system message (set to default system message) with default output language (en)
        model = DocumentLLM(
            model="azure/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
        )
        assert (
            model.system_message is not None
            and model.system_message == self.default_system_message_en
        )

        # Test None system message (set to default system message) with non-en output language (adapt)
        model = DocumentLLM(
            model="azure/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            output_language="adapt",
        )
        assert (
            model.system_message is not None
            and model.system_message == self.default_system_message_non_en
        )

        # Test with custom system message
        model = DocumentLLM(
            model="azure/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            system_message="You are a helpful assistant that can answer questions and help with tasks.",
        )
        assert (
            model.system_message
            == "You are a helpful assistant that can answer questions and help with tasks."
        )

        check_locals_memory_usage(locals(), test_name="test_system_messages")

    @pytest.mark.vcr
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_extract_aspects_from_document(self, llm: DocumentLLMGroup | DocumentLLM):
        """
        Tests the aspects extraction functionality of LLMs.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        # Check for unsupported LLM roles
        with pytest.raises(ValueError):
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
                llm_role="extractor_vision",  # type: ignore
            )

        document_aspects = [
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
                llm_role="extractor_text",
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
            self.document.aspects = invalid_aspects
        with pytest.raises(ValueError):
            llm.extract_aspects_from_document(
                self.document, from_aspects=invalid_aspects
            )

        # Invalid sequence types for "from_aspects"
        with pytest.raises(ValueError, match="list"):
            llm.extract_aspects_from_document(
                self.document,
                from_aspects=tuple(document_aspects),  # type: ignore
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
                llm_role="extractor_text",
            ),
            Aspect(
                name="Parties",
                description="Information on the parties in the contract",
                llm_role="reasoner_text",
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
                llm_role="reasoner_text",
            ),
            Aspect(
                name="Penalties",
                description="Clauses describing penalties for breaching the contract",
                llm_role="reasoner_text",
            ),
        ]
        self.document.aspects = document_aspects

        # LLM extraction
        self.compare_with_concurrent_execution(
            llm=llm,
            expected_n_calls_no_concurrency=(3 if llm.is_group else 2) * 3
            + 2,  # *3 due to 30 (1/3) paras per call, +2 due to sub-aspects
            expected_n_calls_1_item_per_call=(
                len(self.document.aspects) if llm.is_group else 6
            )
            * 3
            + 3,  # *3 due to 30 (1/3) paras per call, +3 due to sub-aspects
            expected_n_calls_with_concurrency=(
                len(self.document.aspects) if llm.is_group else 6
            )
            * 3
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
            self.check_extra_data_in_extracted_items(aspect)

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
            assigned_container=extracted_aspects,
            assigned_instance_class=Aspect,
            llm_roles=llm.list_roles,
        )

        # Log sub-aspects for testing
        self.log_extracted_items_for_instance(
            self.document.get_aspect_by_name(
                "Exclusions from confidential information"
            ).get_aspect_by_name("Information in possession of the receiving party"),
            full_repr=False,
        )
        self.log_extracted_items_for_instance(
            self.document.get_aspect_by_name(
                "Exclusions from confidential information"
            ).get_aspect_by_name("Information received from a third party"),
            full_repr=False,
        )
        self.log_extracted_items_for_instance(
            self.document.get_aspect_by_name(
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
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_extract_concepts_from_aspect(self, llm: DocumentLLMGroup | DocumentLLM):
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
                llm_role="extractor_text",
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

        # No concepts defined for aspect. Logger warning and empty list returned.
        aspects_without_concepts = [
            Aspect(
                name="Term and termination",
                description="All clauses addressing term and termination of the contract",
                concepts=[],
                llm_role="extractor_text",
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
            llm_roles=llm.list_roles,
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
                from_concepts=invalid_concepts,
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
                from_concepts=tuple(attached_aspect.concepts),  # type: ignore
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
                llm_role="extractor_text",
                add_justifications=True,
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="reasoner_text",
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
            original_container=aspect_concepts,
            assigned_container=list(attached_aspect.concepts),
            assigned_instance_class=_Concept,
            llm_roles=llm.list_roles,
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
            original_container=aspect_concepts,
            assigned_container=list(attached_aspect.concepts),
            assigned_instance_class=_Concept,
            llm_roles=llm.list_roles,
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
            llm_roles=llm.list_roles,
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
        main_aspect.add_aspects(sub_aspects)
        document = self.document.clone()
        document.aspects = [main_aspect]
        llm.extract_all(document)
        total_liability_sub_aspect = document.get_aspect_by_name(
            "Liability"
        ).get_aspect_by_name("Total liability")
        assert all(i.extracted_items for i in total_liability_sub_aspect.concepts)
        self.log_extracted_items_for_instance(total_liability_sub_aspect)
        penalties_for_breach_sub_aspect = document.get_aspect_by_name(
            "Liability"
        ).get_aspect_by_name("Penalties for breach")
        assert all(i.extracted_items for i in penalties_for_breach_sub_aspect.concepts)
        self.log_extracted_items_for_instance(penalties_for_breach_sub_aspect)
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
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_extract_concepts_from_document(self, llm: DocumentLLMGroup | DocumentLLM):
        """
        Tests for concept extraction from document.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        document_concepts = [
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="extractor_text",
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                llm_role="reasoner_text",
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
            self.document.concepts = invalid_concepts

        # Invalid sequence types for "from_concepts"
        with pytest.raises(ValueError, match="list"):
            llm.extract_concepts_from_document(
                self.document,
                from_concepts=tuple(self.document.concepts),  # type: ignore
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
                llm_role="extractor_text",
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                llm_role="extractor_text",
                add_justifications=True,
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="extractor_text",
            ),
            StringConcept(
                name="Parties",
                description="Names of contract parties",
                llm_role="extractor_text",
                add_references=True,
            ),
            StringConcept(
                name="Duration",
                description="Contract duration",
                llm_role="extractor_text",
                add_references=True,
            ),
            NumericalConcept(
                name="Duration in years",
                description="Contract duration in years",
                llm_role="reasoner_text",
                numeric_type="int",
            ),
            BooleanConcept(
                name="Contract type check",
                description="Contract is a shareholder's agreement",
                llm_role="extractor_text",
                add_justifications=True,
            ),
            RatingConcept(
                name="Contract quality",
                description="Contract quality from NDA best practice perspective.",
                llm_role="extractor_text",
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
                llm_role="reasoner_text",
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
            expected_n_calls_with_concurrency=(
                len(self.document.concepts) if llm.is_group else 7
            ),
            expected_n_calls_1_item_per_call=(
                len(self.document.concepts) if llm.is_group else 7
            ),
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
            llm_roles=llm.list_roles,
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
            llm_roles=llm.list_roles,
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
            document_docx,
            document_ua,
            document_zh,
            document_docx_ua,
        ],
    )
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_extract_all(self, document: Document, llm: DocumentLLMGroup | DocumentLLM):
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
                        llm_role="extractor_text",
                    ),
                    NumericalConcept(
                        name="Duration",
                        description="Contract duration in years",
                        llm_role="extractor_text",
                        numeric_type="int",
                        add_justifications=True,
                        add_references=True,
                        reference_depth="sentences",
                    ),
                ],
                llm_role="extractor_text",
            ),
            Aspect(
                name="Parties",
                description="Parties to the agreement",
                concepts=[
                    StringConcept(
                        name="Contract party name",
                        description="Contract party name in the contract. "
                        "Exclude information on individual signatories.",
                        llm_role="extractor_text",
                    ),
                    StringConcept(
                        name="Contract party name and role",
                        description="Contract party name and role in the contract. "
                        "Exclude information on individual signatories.",
                        examples=[StringExample(content="Company ABC (Supplier)")],
                        llm_role="extractor_text",
                    ),
                ],
                llm_role="extractor_text",
                add_justifications=True,
            ),
        ]
        document_concepts = [
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                llm_role="reasoner_text",
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="extractor_text",
            ),
            StringConcept(
                name="Start date",
                description="Contract start date",
                llm_role="extractor_text",
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
            expected_n_calls_with_concurrency=(10 if llm.is_group else 9),
            expected_n_calls_1_item_per_call=(10 if llm.is_group else 9),
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
        document = llm.extract_all(document, overwrite_existing=True)
        self.check_instance_container_states(
            original_container=list(document_aspects),
            assigned_container=list(document.aspects),
            assigned_instance_class=Aspect,
            llm_roles=llm.list_roles,
        )
        self.check_instance_container_states(
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
            llm_roles=llm.list_roles,
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
        document = llm.extract_all(document)
        self.check_instance_container_states(
            original_container=document_aspects,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
            llm_roles=llm.list_roles,
        )

        # No aspects, just document concepts
        document.aspects = []
        document_concepts = [
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                llm_role="reasoner_text",
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="extractor_text",
            ),
            StringConcept(
                name="Start date",
                description="Contract start date",
                llm_role="extractor_text",
            ),
        ]
        document.concepts = document_concepts
        assert document.concepts is not document_concepts
        document = llm.extract_all(document)
        self.check_instance_container_states(
            original_container=list(document_concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
            llm_roles=llm.list_roles,
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
            original_container=document_concepts,
            assigned_container=document.concepts,
            assigned_instance_class=_Concept,
            llm_roles=self.invalid_llm_with_valid_fallback.list_roles,
        )

        # Check serialization of an LLM with fallback
        self._check_deserialized_llm_config_eq(self.invalid_llm_with_valid_fallback)

        # Check usage tokens
        self.check_usage(self.invalid_llm_with_valid_fallback)
        # Check cost
        self.check_cost(self.invalid_llm_with_valid_fallback)

        check_locals_memory_usage(locals(), test_name="test_extract_with_fallback")

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "document",
        [
            document,
            document_docx,
            document_ua,
            document_zh,
            document_docx_ua,
        ],
    )
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_serialization_and_cloning(
        self, document: Document, llm: DocumentLLMGroup | DocumentLLM
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
                        llm_role="extractor_text",
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
                llm_role="extractor_text",
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
                        llm_role="extractor_text",
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
                llm_role="extractor_text",
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
                llm_role="extractor_text",
                add_justifications=True,
                add_references=True,
                reference_depth="sentences",
                singular_occurrence=True,
            ),
            BooleanConcept(
                name="Is a contract",
                description="Document is a contract",
                llm_role="reasoner_text",
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
                llm_role="reasoner_text",
                singular_occurrence=True,  # global document classification
            ),
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="extractor_text",
                singular_occurrence=True,
            ),
            NumericalConcept(
                name="Duration in years",
                description="Contract duration in years",
                llm_role="reasoner_text",
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
                llm_role="extractor_text",
                singular_occurrence=False,
            ),
            # Vision-related
            BooleanConcept(
                name="Invoice check",
                description="Whether the attached images contain an invoice",
                llm_role="reasoner_vision",
                singular_occurrence=True,
            ),
            StringConcept(
                name="Invoice amount",
                description="The total amount of the attached invoice",
                llm_role="extractor_vision",
                singular_occurrence=False,  # multiple invoices in the document
            ),
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                llm_role="extractor_vision",
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
                llm_role="extractor_vision",
                singular_occurrence=False,  # multiple invoices in the document
            ),
        ]
        document.aspects = document_aspects
        document.images = document_images
        document.concepts = document_concepts
        document = llm.extract_all(
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
            self.check_extra_data_in_extracted_items(aspect)
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
            original_container=self.extraction_pipeline.aspects,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
            llm_roles=llm.list_roles,
        )
        self.check_instance_container_states(
            original_container=list(self.extraction_pipeline.concepts),
            assigned_container=list(document.concepts),
            assigned_instance_class=_Concept,
            llm_roles=llm.list_roles,
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
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])  # type: ignore
    @memory_profile_and_capture
    def test_aspect_extraction_from_paragraphs(
        self, llm: DocumentLLMGroup | DocumentLLM
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
                llm_role="extractor_text",
                add_justifications=True,
            ),
            Aspect(
                name="Term and termination",
                description="Term and termination clauses",
                llm_role="extractor_text",
                reference_depth="sentences",
            ),
            Aspect(
                name="Anomalies",
                description="Anomalies in the contract. A contract anomaly is any deviation, inconsistency, "
                "or irregularity in a contract's terms, structure, or execution - such as bizarre "
                "clauses about intergalactic jurisdiction, mandatory sock-wearing policies, "
                "or payment in rare seashells - that may indicate errors, conflicts, or potential risks.",
                llm_role="extractor_text",
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
            llm_roles=llm.list_roles,
        )
        for aspect in extracted_aspects:
            self.check_extra_data_in_extracted_items(aspect)

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
            llm_roles=llm.list_roles,
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
                llm_role="extractor_vision",
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
                llm_role="reasoner_vision",
                add_justifications=True,
            ),
            JsonObjectConcept(
                name="Invoice amounts",
                description="Invoice amounts as they appear in the invoice",
                structure={
                    "line": str,
                    "amount": str,
                },
                llm_role="extractor_vision",
            ),
            NumericalConcept(
                name="Total invoice amount",
                description="Total amount of the invoice",
                llm_role="extractor_vision",
                numeric_type="float",
                add_justifications=True,
            ),
        ]
        document = Document(
            images=[self.test_img_png_invoice], concepts=document_concepts
        )

        self.config_llm_async_limiter_for_mock_responses(self.llm_group)

        self.compare_with_concurrent_execution(
            llm=self.llm_group,
            expected_n_calls_no_concurrency=3,
            expected_n_calls_with_concurrency=4,
            expected_n_calls_1_item_per_call=4,
            func=self.llm_group.extract_concepts_from_document,
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
                llm_role="reasoner_vision",
                numeric_type="int",
                add_justifications=True,
            ),
            NumericalConcept(
                name="Number of balconies",
                description="Number of balconies in the floor plan",
                llm_role="reasoner_vision",
                numeric_type="int",
                add_justifications=True,
            ),
            StringConcept(
                name="Room names",
                description="Names of the rooms in the floor plan",
                llm_role="extractor_vision",
            ),
        ]
        document = Document(
            images=[self.test_img_png_apt_plan], concepts=document_concepts
        )

        self.compare_with_concurrent_execution(
            llm=self.llm_group,
            expected_n_calls_no_concurrency=2,
            expected_n_calls_with_concurrency=3,
            expected_n_calls_1_item_per_call=3,
            func=self.llm_group.extract_concepts_from_document,
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
        self.check_usage(self.llm_group)
        # Check cost
        self.check_cost(self.llm_group)

        # Test error when model is not vision-capable
        with pytest.warns(UserWarning, match="vision-capable"):
            llm_not_vision_capable = DocumentLLM(
                model="openai/gpt-3.5-turbo",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="extractor_vision",
            )
        with pytest.raises(ValueError, match="does not support vision"):
            llm_not_vision_capable.chat(
                "What's the type of this document?", images=[self.test_img_png_invoice]
            )

        check_locals_memory_usage(
            locals(), test_name="test_vision", max_obj_memory=5.0
        )  # higher value for images

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_chat(self):
        """
        Tests for the chat method.
        """
        for model in [
            self.llm_extractor_text,
            self.llm_extractor_vision,
            self.invalid_llm_with_valid_fallback,
        ]:
            with pytest.raises(ValueError):
                model.chat("")
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=1)  # type: ignore
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=[1])  # type: ignore
            with pytest.raises(TypeError):
                model.chat(images=self.test_img_png_invoice)  # type: ignore
            if model == self.llm_extractor_vision:
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
                    assert model.fallback_llm is not None
                    response = (
                        model.fallback_llm.get_usage()[0].usage.calls[-1].response
                    )
                else:
                    response = model.get_usage()[0].usage.calls[-1].response
                assert response is not None
                assert "4" in response
            logger.debug(response)
        # Test for non-vision model
        text_only_model = DocumentLLM(model="openai/gpt-3.5-turbo")
        with pytest.raises(ValueError, match="vision"):
            text_only_model.chat(
                "What's the type of this document?", images=[self.test_img_png_invoice]
            )

        # Test for empty system message
        model = DocumentLLM(
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            system_message="",
        )
        assert model.system_message == ""
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")  # Capture all warnings
            model.chat("What's the result of 10+10?")
        assert len(w) == 0, "Expected no warning, but got: " + str(w)
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "20" in response

        # Test for None system message (output language "en", which is the default)
        model = DocumentLLM(
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
        )
        assert model.system_message == self.default_system_message_en
        with pytest.warns(UserWarning, match="default system message"):
            model.chat("What's the result of 10+10?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "20" in response

        # Test for None system message (output language "adapt")
        model = DocumentLLM(
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
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
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            system_message="Your name is John Doe. Introduce yourself as such.",
        )
        assert (
            model.system_message == "Your name is John Doe. Introduce yourself as such."
        )
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")  # Capture all warnings
            model.chat("What's your name?")
        assert len(w) == 0, "Expected no warning, but got: " + str(w)
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "John Doe" in response

        check_locals_memory_usage(locals(), test_name="test_chat")

    # Do not memory-profile this test as we monkey patch sys.stdout
    def test_logger_disabled(self, monkeypatch, capsys):
        """
        Tests for disabling the logger.
        """

        # Ensure our dedicated stream uses the current (monkeypatched) sys.stdout:
        monkeypatch.setattr(dedicated_stream, "base", sys.stdout)

        # 1) Set environment variable to disable logger
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "OFF")
        reload_logger_settings()

        # 2) Attempt to log a message
        logger.debug("This message should NOT appear.")

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Assert that the message is indeed missing
        assert "This message should NOT appear." not in captured.out

    # Do not memory-profile this test as we monkey patch sys.stdout
    def test_logger_enabled(self, monkeypatch, capsys):
        """
        Tests for enabling the logger.
        """

        # Ensure our dedicated stream uses the current (monkeypatched) sys.stdout:
        monkeypatch.setattr(dedicated_stream, "base", sys.stdout)

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
    # Do not memory-profile this test as we monkey patch sys.stdout
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

        # Ensure our dedicated stream uses the current (monkeypatched) sys.stdout:
        monkeypatch.setattr(dedicated_stream, "base", sys.stdout)

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
            cost_tracking,  # noqa: F401
            detailed_usage,  # noqa: F401
            fallback_llm,  # noqa: F401
            llm_api,  # noqa: F401
            llm_group,  # noqa: F401
            llm_local,  # noqa: F401
            o1_o4,  # noqa: F401
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
            def_llm_pricing,  # noqa: F401
        )
        from dev.usage_examples.docstrings.documents import def_document  # noqa: F401
        from dev.usage_examples.docstrings.examples import (
            def_example_json_object,  # noqa: F401
            def_example_string,  # noqa: F401
        )
        from dev.usage_examples.docstrings.images import def_image  # noqa: F401
        from dev.usage_examples.docstrings.llms import (  # noqa: F401
            def_llm,
            def_llm_group,
        )
        from dev.usage_examples.docstrings.paragraphs import def_paragraph  # noqa: F401
        from dev.usage_examples.docstrings.pipelines import def_pipeline  # noqa: F401
        from dev.usage_examples.docstrings.sentences import def_sentence  # noqa: F401
        from dev.usage_examples.docstrings.utils import (
            json_object_cls_struct,  # noqa: F401
            reload_logger_settings,  # noqa: F401
        )

        check_locals_memory_usage(locals(), test_name="test_docstring_examples")

    @pytest.mark.parametrize("apply_markdown", [True, False])
    @pytest.mark.parametrize("strict_mode", [True, False])
    @pytest.mark.parametrize(
        "include_options",
        [
            "default",  # All options set to True
            "minimal",  # All options set to False
            "no_images",  # All options True except images
        ],
    )
    @memory_profile_and_capture
    def test_docx_converter(
        self, apply_markdown: bool, strict_mode: bool, include_options: str
    ):
        """
        Tests for the DocxConverter class, covering both convert() and convert_to_text_format() methods.

        This test uses parametrization to test different combinations of:
        - apply_markdown (True/False)
        - strict_mode (True/False)
        - include_options (default/minimal/no_images)

        It checks conversion from file path, file object, and BytesIO for compatibility.
        """

        # Define inclusion parameters based on the option
        include_params = {
            "default": {
                "include_tables": True,
                "include_comments": True,
                "include_footnotes": True,
                "include_headers": True,
                "include_footers": True,
                "include_textboxes": True,
                "include_links": True,
                "include_inline_formatting": True,
                "include_images": True,
            },
            "minimal": {
                "include_tables": False,
                "include_comments": False,
                "include_footnotes": False,
                "include_headers": False,
                "include_footers": False,
                "include_textboxes": False,
                "include_links": False,
                "include_inline_formatting": False,
                "include_images": False,
            },
            "no_images": {
                "include_tables": True,
                "include_comments": True,
                "include_footnotes": True,
                "include_headers": True,
                "include_footers": True,
                "include_textboxes": True,
                "include_links": True,
                "include_inline_formatting": True,
                "include_images": False,
            },
        }[include_options]

        # Utility function to verify all _DocxPackage attributes are populated
        def verify_docx_package_attributes(test_file_path_or_object):
            """
            Utility function to verify all _DocxPackage attributes are populated correctly.

            :param test_file_path_or_object: File path or file object to create DocxPackage from.
            :type test_file_path_or_object: str | Path | BinaryIO
            """
            package = _DocxPackage(test_file_path_or_object)
            try:
                # Verify all attributes that are present in test DOCX file are populated
                assert package.archive is not None, "Archive must be populated"
                assert package.rels != {}, "Relationships must be populated"
                assert package.main_document is not None, (
                    "Main document must be populated"
                )
                assert package.styles is not None, "Styles must be populated"
                assert package.numbering is not None, "Numbering must be populated"
                assert package.footnotes is not None, "Footnotes must be populated"
                assert package.comments is not None, "Comments must be populated"
                assert package.headers, "Headers must be populated"
                assert package.footers, "Footers must be populated"
                assert package.images, "Images must be populated"
                assert package.hyperlinks, "Hyperlinks must be populated"

                logger.debug("All _DocxPackage attributes successfully verified")
            finally:
                if package:
                    package.close()

        # Prepare file objects for different input methods
        with open(self.test_docx_badly_formatted_path, "rb") as f:
            file_bytes = f.read()

        # Create a BytesIO object that can be reset
        def create_bytesio():
            """
            Creates a BytesIO object from the test file bytes.

            :return: BytesIO object containing the test file data.
            :rtype: BytesIO
            """
            return BytesIO(file_bytes)

        # Function to open the file
        def open_file():
            """
            Opens the test DOCX file in binary read mode.

            :return: File object opened in binary mode.
            :rtype: BinaryIO
            """
            return open(self.test_docx_badly_formatted_path, "rb")

        # Create converter instance
        converter = DocxConverter()

        # Test with invalid file extension
        with pytest.raises(DocxConverterError):
            converter.convert("random_path.txt")

        # Helper function to verify Document objects
        def verify_document_equality(documents: list[Document]) -> None:
            """
            Helper function to verify Document objects have expected properties that are
            consistent across documents created from different input sources.

            :param documents: List of Document objects to verify.
            :type documents: list[Document]
            """
            # Check that all documents have the expected properties
            for doc in documents:
                assert isinstance(doc, Document)
                assert doc.raw_text, "Document should have raw text"
                if include_options == "default":  # when all content is included
                    assert (
                        doc.raw_text.strip()
                        == self.test_badly_formatted_converted_raw_text
                    ), "Raw text does not match"
                if apply_markdown:
                    assert doc._md_text, (
                        "Document should have markdown text when markdown is enabled"
                    )
                    if include_options == "default":  # when all content is included
                        assert (
                            doc._md_text.strip()
                            == self.test_badly_formatted_converted_md_text
                        ), "Markdown text does not match"
                    with pytest.raises(ValueError):
                        doc._md_text = "Random md text"  # cannot be set once populated
                else:
                    assert doc._md_text is None, (
                        "Document should not have markdown text when markdown is disabled"
                    )
                assert doc.paragraphs, "Document should have paragraphs"

                # Verify that each sentence inherits additional_context from its paragraph
                paragraphs_raw_text = ""
                paragraphs_md_text = ""
                for paragraph in doc.paragraphs:
                    assert paragraph.raw_text, "Paragraph should have raw text"
                    paragraphs_raw_text += paragraph.raw_text + "\n"
                    if apply_markdown:
                        assert paragraph._md_text, (
                            "Paragraph should have markdown text when markdown is enabled"
                        )
                        paragraphs_md_text += paragraph._md_text + "\n"
                        with pytest.raises(ValueError):
                            paragraph._md_text = (
                                "Random md text"  # cannot be set once populated
                            )
                    else:
                        assert paragraph._md_text is None, (
                            "Paragraph should not have markdown text when markdown is disabled"
                        )
                    assert paragraph.additional_context, (
                        "Paragraph should have additional context"
                    )
                    paragraphs_raw_text += paragraph.additional_context + "\n\n"
                    if apply_markdown:
                        paragraphs_md_text += paragraph.additional_context + "\n\n"
                    for sentence in paragraph.sentences:
                        assert (
                            sentence.additional_context == paragraph.additional_context
                        ), (
                            "Sentence additional_context should match its paragraph's additional_context"
                        )
                        assert sentence.custom_data == paragraph.custom_data, (
                            "Sentence custom_data should match its paragraph's custom_data"
                        )
                if include_options == "default":  # when all content is included
                    assert (
                        paragraphs_raw_text.strip()
                        == self.test_badly_formatted_converted_raw_paras_text
                    ), "Raw paragraphs text does not match"
                if apply_markdown:
                    if include_options == "default":  # when all content is included
                        assert (
                            paragraphs_md_text.strip()
                            == self.test_badly_formatted_converted_md_paras_text
                        ), "Markdown paragraphs text does not match"
                else:
                    assert paragraphs_md_text == "", (
                        "Markdown paragraphs text should be empty when markdown is disabled"
                    )

            # Check that all documents have the same content
            first_doc = documents[0]
            for i, doc in enumerate(documents[1:], 1):
                # Compare full texts
                assert doc.raw_text == first_doc.raw_text, (
                    f"Document {i} has different raw text"
                )
                if apply_markdown:
                    assert doc._md_text and doc._md_text == first_doc._md_text, (
                        f"Document {i} has different markdown text"
                    )
                assert len(doc.paragraphs) == len(first_doc.paragraphs), (
                    f"Document {i} has different paragraph count"
                )
                # Compare paragraphs
                for p_idx, para in enumerate(doc.paragraphs):
                    assert para.raw_text == first_doc.paragraphs[p_idx].raw_text, (
                        f"Paragraph {p_idx} has different raw text"
                    )
                    if apply_markdown:
                        assert (
                            para._md_text
                            and para._md_text == first_doc.paragraphs[p_idx]._md_text
                        ), f"Paragraph {p_idx} has different markdown text"
                    for sent_idx, sent in enumerate(para.sentences):
                        assert (
                            sent.raw_text
                            == first_doc.paragraphs[p_idx].sentences[sent_idx].raw_text
                        ), f"Sentence {sent_idx} has different raw text"
                        # markdown does not apply to sentences

                # Check images if they should be included
                if include_params["include_images"]:
                    assert len(doc.images) == len(first_doc.images), (
                        f"Document {i} has different image count"
                    )

        # Test 1: Test convert() method with different input sources
        documents = []

        # Convert from file path
        verify_docx_package_attributes(self.test_docx_badly_formatted_path)
        doc_from_path = converter.convert(
            self.test_docx_badly_formatted_path,
            apply_markdown=apply_markdown,
            strict_mode=strict_mode,
            **include_params,
        )
        documents.append(doc_from_path)

        # Convert from file object
        with open_file() as file_obj:
            verify_docx_package_attributes(file_obj)
            doc_from_obj = converter.convert(
                file_obj,
                apply_markdown=apply_markdown,
                strict_mode=strict_mode,
                **include_params,
            )
        documents.append(doc_from_obj)

        # Convert from BytesIO
        bytesio = create_bytesio()
        verify_docx_package_attributes(bytesio)
        doc_from_bytesio = converter.convert(
            bytesio,
            apply_markdown=apply_markdown,
            strict_mode=strict_mode,
            **include_params,
        )
        documents.append(doc_from_bytesio)

        # Verify all documents are equal content-wise
        verify_document_equality(documents)

        # Test 2: Test convert_to_text_format() method with different formats and input sources
        for output_format in ["markdown", "raw"]:
            text_results = []

            # Text conversion params (exclude include_images as it's not applicable to markdown)
            text_params = {
                k: v for k, v in include_params.items() if k != "include_images"
            }

            # Convert from file path
            text_from_path = converter.convert_to_text_format(
                self.test_docx_badly_formatted_path,
                output_format=output_format,  # type: ignore
                strict_mode=strict_mode,
                **text_params,
            )
            text_results.append(text_from_path)

            # Convert from file object
            with open_file() as file_obj:
                text_from_obj = converter.convert_to_text_format(
                    file_obj,
                    output_format=output_format,  # type: ignore
                    strict_mode=strict_mode,
                    **text_params,
                )
            text_results.append(text_from_obj)

            # Convert from BytesIO
            bytesio = create_bytesio()
            text_from_bytesio = converter.convert_to_text_format(
                bytesio,
                output_format=output_format,  # type: ignore
                strict_mode=strict_mode,
                **text_params,
            )
            text_results.append(text_from_bytesio)

            # Verify all text results are equal
            first_result = text_results[0]
            for i, result in enumerate(text_results[1:], 1):
                assert result == first_result, (
                    f"Text result {i} is different for format {output_format}"
                )

            # Match with the converted Document text content (must be the same)
            for text_result in text_results:
                if not include_params["include_images"]:
                    if output_format == "raw":
                        assert all(text_result == doc.raw_text for doc in documents)
                    if apply_markdown and output_format == "markdown":
                        assert all(text_result == doc._md_text for doc in documents)

        check_locals_memory_usage(locals(), test_name="test_docx_converter")

    @memory_profile_and_capture
    def test_docx_converter_include_params(self):
        """
        Test that disabling include parameters affects document text content.
        """
        converter = DocxConverter()

        # Define all include parameters to test
        # Note: include_images doesn't affect md or raw text content, so we skip it
        include_params = [
            "include_tables",
            "include_comments",
            "include_footnotes",
            "include_headers",
            "include_footers",
            "include_textboxes",
            "include_links",
            "include_inline_formatting",
        ]

        params_not_affecting_raw_text = [
            "include_links",
            "include_inline_formatting",
        ]

        # Try with markdown and without markdown
        for apply_markdown in [True, False]:
            # Convert with all parameters True (baseline default)
            baseline_doc = converter.convert(
                self.test_docx_badly_formatted_path, apply_markdown=apply_markdown
            )

            # Test each parameter by setting it to False while others remain True
            for param_name in include_params:
                kwargs = {param: True for param in include_params}
                kwargs[param_name] = False

                test_doc = converter.convert(
                    self.test_docx_badly_formatted_path,
                    apply_markdown=apply_markdown,
                    **kwargs,
                )

                # Assert that text content is different when parameter is disabled
                if param_name not in params_not_affecting_raw_text:
                    assert test_doc.raw_text != baseline_doc.raw_text, (
                        f"raw_text should differ when {param_name}=False"
                    )
                    test_doc_raw_paras_merged = "".join(
                        p.raw_text for p in test_doc.paragraphs
                    )
                    baseline_doc_raw_paras_merged = "".join(
                        p.raw_text for p in baseline_doc.paragraphs
                    )
                    assert test_doc_raw_paras_merged != baseline_doc_raw_paras_merged, (
                        f"Paragraphs' raw_text should differ when {param_name}=False"
                    )
                if apply_markdown:
                    assert test_doc._md_text != baseline_doc._md_text, (
                        f"md_text should differ when {param_name}=False"
                    )
                    test_doc_md_paras_merged = "".join(
                        p._md_text
                        for p in test_doc.paragraphs  # type: ignore
                    )
                    baseline_doc_md_paras_merged = "".join(
                        p._md_text
                        for p in baseline_doc.paragraphs  # type: ignore
                    )
                    assert test_doc_md_paras_merged != baseline_doc_md_paras_merged, (
                        f"Paragraphs' md_text should differ when {param_name}=False"
                    )

        check_locals_memory_usage(
            locals(), test_name="test_docx_converter_include_params"
        )

    @pytest.mark.vcr
    @pytest.mark.parametrize("apply_markdown", [True, False])
    @memory_profile_and_capture
    def test_docx_converter_llm_extract(self, apply_markdown: bool):
        """
        Tests for LLM extraction from DOCX files.
        """

        converter = DocxConverter()

        doc = converter.convert(
            self.test_docx_badly_formatted_path, apply_markdown=apply_markdown
        )
        if apply_markdown:
            assert doc._md_text, (
                "Document should have markdown text when markdown is enabled"
            )
            for para in doc.paragraphs:
                assert para._md_text, (
                    "Paragraph should have markdown text when markdown is enabled"
                )
        else:
            assert doc._md_text is None, (
                "Document should not have markdown text when markdown is disabled"
            )
            for para in doc.paragraphs:
                assert para._md_text is None, (
                    "Paragraph should not have markdown text when markdown is disabled"
                )

        # Create a new LLM group with new usage stats for each test iteration,
        # as we will supply markdown or raw text based on apply_markdown flag
        # and we need to isolate such prompts for inspection.
        llm_group = DocumentLLMGroup(
            llms=[
                DocumentLLM(**self._llm_extractor_text_kwargs_openai),
                DocumentLLM(**self._llm_extractor_vision_kwargs_openai),
            ]
        )

        md_test_chars = ["**", "|", "##"]

        def check_is_markdown(text: str, expect_newlines: bool = False) -> None:
            """
            Checks that the text contains markdown.
            """
            if expect_newlines:
                assert any(i in text for i in md_test_chars + ["\n"])
            else:
                assert any(i in text for i in md_test_chars)

        def check_not_markdown(text: str) -> None:
            """
            Checks that the text does not contain markdown.
            """
            assert not any(i in text for i in md_test_chars)

        def check_markdown_in_prompt(
            prompt_kwargs_key: str,
            expect_newlines: bool = False,
        ) -> None:
            """
            Validates that prompt content contains or excludes markdown formatting based on the apply_markdown flag.

            This function inspects LLM call objects to verify that text content submitted in prompts
            either contains markdown formatting (when apply_markdown=True) or excludes it (when
            apply_markdown=False). It checks both the specific prompt parameter content and the
            full rendered prompt.

            :param prompt_kwargs_key: The key in prompt_kwargs to check ('text' or 'paragraphs')
            :type prompt_kwargs_key: str
            :param expect_newlines: Whether to expect newlines in addition to markdown characters
            :type expect_newlines: bool
            :raises ValueError: If an invalid prompt_kwargs_key is provided
            """
            # Only check text extraction calls
            call_objs = llm_group.get_usage(llm_role="extractor_text")[0].usage.calls
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
                # Check for markdown flag in prompt kwargs
                if apply_markdown:
                    assert "is_markdown" in text_call_obj.prompt_kwargs
                    assert text_call_obj.prompt_kwargs["is_markdown"]
                    if (
                        text_call_obj.prompt_kwargs.get("reference_depth")
                        != "sentences"
                    ):
                        # md is not available for sentences
                        assert "markdown format" in text_call_obj.prompt
                    if text_call_obj.prompt_kwargs.get("paragraphs"):
                        assert (
                            "additional_context_for_paras_or_sents"
                            in text_call_obj.prompt_kwargs
                        )
                        assert text_call_obj.prompt_kwargs[
                            "additional_context_for_paras_or_sents"
                        ]
                        assert "additional context information" in text_call_obj.prompt
                else:
                    assert "is_markdown" not in text_call_obj.prompt_kwargs
                    assert "markdown format" not in text_call_obj.prompt
                    if text_call_obj.prompt_kwargs.get("paragraphs"):
                        # Additional context is still provided even when markdown is disabled
                        assert (
                            "additional_context_for_paras_or_sents"
                            in text_call_obj.prompt_kwargs
                        )
                        assert text_call_obj.prompt_kwargs[
                            "additional_context_for_paras_or_sents"
                        ]
                        assert "additional context information" in text_call_obj.prompt

                # Check the text of specific param in the prompt
                if prompt_kwargs_key == "text":
                    submitted_text_in_prompt = text_call_obj.prompt_kwargs[
                        prompt_kwargs_key
                    ]
                elif prompt_kwargs_key == "paragraphs":
                    if apply_markdown:
                        p_md_texts = [
                            p._md_text
                            for p in text_call_obj.prompt_kwargs[prompt_kwargs_key]
                        ]
                        # Check that some paragraphs have non-stripped markdown text,
                        # e.g. to keep indentation in lists
                        assert any(t != t.strip() for t in p_md_texts), (
                            "Expected some paragraphs to have non-stripped markdown text"
                        )
                        submitted_text_in_prompt = "".join(p_md_texts)
                    else:
                        submitted_text_in_prompt = "".join(
                            [
                                p.raw_text
                                for p in text_call_obj.prompt_kwargs[prompt_kwargs_key]
                            ]
                        )
                else:
                    raise ValueError(f"Invalid prompt_kwargs_key: {prompt_kwargs_key}")
                if apply_markdown:
                    check_is_markdown(
                        submitted_text_in_prompt,
                        expect_newlines=expect_newlines,
                    )
                else:
                    check_not_markdown(
                        submitted_text_in_prompt,
                    )
                # Separately check for paragraph sentences (raw text only)
                if prompt_kwargs_key == "paragraphs":
                    for paragraph in text_call_obj.prompt_kwargs[prompt_kwargs_key]:
                        for sentence in paragraph.sentences:
                            check_not_markdown(sentence.raw_text)

                # Check the full rendered prompt
                submitted_prompt = text_call_obj.prompt
                if apply_markdown:
                    if text_call_obj.prompt_kwargs["reference_depth"] == "sentences":
                        # Only sentences with raw text are passed to the prompt
                        check_not_markdown(
                            submitted_prompt,
                        )
                        # No markdown instructions in the prompt, as only sentences
                        # with raw text are passed as context
                        assert "markdown syntax" not in submitted_prompt.lower()
                    else:
                        check_is_markdown(
                            submitted_prompt,
                            expect_newlines=True,
                        )
                        # Check for markdown instructions in the prompt
                        assert "markdown syntax" in submitted_prompt.lower()
                else:
                    check_not_markdown(
                        submitted_prompt,
                    )

        # Test concept extraction
        doc_concepts = [
            NumericalConcept(
                name="Hidden gems count",
                description="Number of hidden gems in the document",
                numeric_type="int",
                llm_role="extractor_text",
            ),
            StringConcept(
                name="Invoice total amount",
                description="Total amount of the invoice",
                llm_role="extractor_vision",
            ),
        ]
        if apply_markdown:
            doc_concepts.append(
                StringConcept(
                    name="Links in the document",
                    description="URLs mentioned in the document",
                    llm_role="extractor_text",
                )
            )
        doc.concepts = doc_concepts

        # Test extraction from full text (markdown)
        extracted_concepts = llm_group.extract_concepts_from_document(doc)
        assert extracted_concepts[0].extracted_items
        logger.debug(extracted_concepts[0].extracted_items[0].value)
        assert extracted_concepts[0].extracted_items[0].value == 3
        assert extracted_concepts[1].extracted_items
        logger.debug(extracted_concepts[1].extracted_items[0].value)
        assert "4800" in extracted_concepts[1].extracted_items[0].value
        if apply_markdown:
            assert extracted_concepts[2].extracted_items
            logger.debug(extracted_concepts[2].extracted_items[0].value)
            assert "example" in extracted_concepts[2].extracted_items[0].value
        # Check the full text for markdown content
        check_markdown_in_prompt(prompt_kwargs_key="text", expect_newlines=True)

        # Test extraction with max paragraphs
        # (paragraphs' _md_text and raw_text are used based on apply_markdown flag)
        extracted_concepts = llm_group.extract_concepts_from_document(
            doc, max_paragraphs_to_analyze_per_call=25, overwrite_existing=True
        )
        assert extracted_concepts[0].extracted_items
        # Check the chunked text for markdown content
        check_markdown_in_prompt(prompt_kwargs_key="text", expect_newlines=True)

        # Test aspect extraction

        # Test with paragraph-level refs (default)
        doc.aspects = [
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
            )
        ]
        extracted_aspects = llm_group.extract_aspects_from_document(doc)
        assert extracted_aspects[0].extracted_items
        assert extracted_aspects[0].extracted_items[0].reference_paragraphs
        # Check the paragraphs' texts for markdown content
        check_markdown_in_prompt(prompt_kwargs_key="paragraphs", expect_newlines=False)

        # Test with sentence-level refs
        doc.aspects = [
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
                reference_depth="sentences",
            )
        ]
        extracted_aspects = llm_group.extract_aspects_from_document(doc)
        assert extracted_aspects[0].extracted_items
        assert extracted_aspects[0].extracted_items[0].reference_sentences
        # Check the paragraphs' texts for markdown content
        # Markdown does not apply to sentences that are passed to the aspects
        # extraction prompt when reference_depth="sentences"
        check_markdown_in_prompt(
            prompt_kwargs_key="paragraphs",
            expect_newlines=False,
        )

        # Test serialization and cloning after DOCX conversion and LLM extraction
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

        check_locals_memory_usage(locals(), test_name="test_docx_converter_llm_extract")

    @memory_profile_and_capture
    def test_docx_package_error_handling(self):
        """
        Tests for error handling in _DocxPackage initialization and XML loading.
        """

        # Test with invalid file path
        with pytest.raises(DocxFormatError, match="not found"):
            _DocxPackage("non_existent_file.docx")

        try:
            # Test with file that's not a zip
            with open("tests/temp_not_a_docx.txt", "w") as f:
                f.write("This is not a DOCX file")
            with pytest.raises(DocxFormatError, match="not a valid"):
                _DocxPackage("tests/temp_not_a_docx.txt")

            # Test with file that's a zip but not a valid docx
            with zipfile.ZipFile("tests/temp_invalid.docx", "w") as zip_file:
                zip_file.writestr("dummy.txt", "This is not a valid DOCX structure")

            with pytest.raises(DocxFormatError, match="missing word/document.xml"):
                _DocxPackage("tests/temp_invalid.docx")

        finally:
            # Clean up
            remove_file("tests/temp_not_a_docx.txt")
            remove_file("tests/temp_invalid.docx")

    @memory_profile_and_capture
    def test_docx_converter_extract_paragraph_text(self):
        """
        Tests for paragraph text extraction from DOCX elements.
        """

        converter = DocxConverter()

        # Create sample XML elements for testing using lxml
        def create_text_element(text_content: str):
            """
            Creates a Word XML text element with the given content.

            :param text_content: Text content to include in the element.
            :type text_content: str
            :return: XML element representing Word text.
            :rtype: Element
            """
            element = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}t")
            element.text = text_content
            return element

        def create_run_with_text(text_content: str):
            """
            Creates a Word XML run element containing a text element.

            :param text_content: Text content for the run.
            :type text_content: str
            :return: XML element representing a Word run with text.
            :rtype: Element
            """
            run = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}r")
            text_elem = create_text_element(text_content)
            run.append(text_elem)
            return run

        def create_paragraph_with_runs(runs):
            """
            Creates a Word XML paragraph element containing the given runs.

            :param runs: List of run elements to include in the paragraph.
            :type runs: list[Element]
            :return: XML element representing a Word paragraph.
            :rtype: Element
            """
            para = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}p")
            for run in runs:
                para.append(run)
            return para

        # Create a minimal mock package
        class MockPackage:
            def __init__(self):
                self.styles = None
                self.numbering = None

        package = MockPackage()

        # Basic paragraph with multiple runs
        runs = [
            create_run_with_text("First part. "),
            create_run_with_text("Second part."),
        ]
        para = create_paragraph_with_runs(runs)
        text = converter._extract_paragraph_text(para, package)  # type: ignore
        assert text == "First part. Second part."

        # Paragraph with line breaks
        br_run = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}r")
        br_run.append(etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}br"))
        runs_with_br = [
            create_run_with_text("Before break"),
            br_run,
            create_run_with_text("After break"),
        ]
        para = create_paragraph_with_runs(runs_with_br)
        text = converter._extract_paragraph_text(para, package)  # type: ignore
        assert text == "Before break\nAfter break"

        # Paragraph with footnote reference
        run_with_footnote = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}r")
        footnote_ref = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}footnoteReference")
        footnote_ref.attrib[f"{{{WORD_XML_NAMESPACES['w']}}}id"] = "1"
        run_with_footnote.append(footnote_ref)
        para = create_paragraph_with_runs(
            [create_run_with_text("Text with footnote "), run_with_footnote]
        )
        text = converter._extract_paragraph_text(para, package, markdown_mode=True)  # type: ignore
        assert text == "Text with footnote [Footnote 1]"
        text = converter._extract_paragraph_text(para, package, markdown_mode=False)  # type: ignore
        assert text == "Text with footnote 1"

        # Empty paragraph
        # Create a paragraph element that will result in empty text
        empty_para = etree.Element(f"{{{WORD_XML_NAMESPACES['w']}}}p")

        result = converter._process_paragraph(empty_para, package)  # type: ignore
        assert result is None

        check_locals_memory_usage(
            locals(), test_name="test_docx_converter_extract_paragraph_text"
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
                llm_role="extractor_text",
                singular_occurrence=True,
            ),
            LabelConcept(
                name="Document type",
                description="Type of the current document",
                labels=["legislation", "contract", "other"],
                llm_role="extractor_text",
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
                llm_role="extractor_text",
            ),  # entry into force date is modified in the test doc
            StringConcept(
                name="Anomalies",
                description="Anomalies in the document: unusual or unexpected content",
                llm_role="extractor_text",
            ),  # anomaly is in the middle of the document
        ]
        doc = Document(raw_text=text_content)
        doc.concepts = doc_concepts

        # First, test hitting the max input tokens limit
        # A suggestion to use the optimization guide is included in the error message
        too_long_doc = Document(raw_text=text_content * 5)
        too_long_doc.concepts = doc_concepts
        llm_short_context = DocumentLLM(
            model="azure/gpt-4o-mini",  # 128k context window
            api_key="...",  # dummy key (pre-call validation will fail)
            api_base="...",  # dummy base
            api_version="...",  # dummy version
        )
        with pytest.raises(
            ValueError,
            match="https://contextgem.dev/optimizations/optimization_long_docs.html",
        ):
            llm_short_context.extract_concepts_from_document(
                too_long_doc,
            )

        # Use params optimized for very long documents (200+ pages)
        extracted_concepts = self.llm_extractor_text.extract_concepts_from_document(
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
            llm=self.llm_extractor_text,
            judge_llm=self.llm_reasoner_text,
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
                model="azure/gpt-4.1-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
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

        # Now, check that the same pricing is calculated when setting LLMPricing explicitly
        llm_explicit_pricing = DocumentLLM(
            model="azure/gpt-4.1-mini",
            api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            pricing_details=LLMPricing(
                input_per_1m_tokens=0.40, output_per_1m_tokens=1.60
            ),
        )
        llm_explicit_pricing.extract_concepts_from_document(
            document, overwrite_existing=True
        )
        cost_info_explicit = llm_explicit_pricing.get_cost()
        assert len(cost_info_explicit) == 1
        cost_explicit = cost_info_explicit[0].cost
        assert cost_explicit.input == cost.input
        assert cost_explicit.output == cost.output
        assert cost_explicit.total == cost.total
        logger.debug(
            f"Costs for model {llm_explicit_pricing.model} (explicit pricing): "
            f"input {cost_explicit.input}, output {cost_explicit.output}, total {cost_explicit.total}"
        )

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

        # Setting auto-pricing and LLMPricing together
        with pytest.raises(ValueError, match="auto_pricing=True"):
            DocumentLLM(
                model="azure/gpt-4.1-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                pricing_details=LLMPricing(
                    input_per_1m_tokens=0.00015, output_per_1m_tokens=0.0006
                ),
                auto_pricing=True,
            )

        # Test warnings

        # Unknown model initialization
        with pytest.warns(UserWarning, match="Unable to fetch pricing data for model"):
            llm = DocumentLLM(
                model="azure/gptXX",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                auto_pricing=True,
            )

        check_locals_memory_usage(locals(), test_name="test_auto_pricing")

    def test_cassette_url_security(self):
        """
        Test that validates URL security in all existing cassette files.

        This test ensures that all URLs in VCR cassette files are from approved domains.
        If any violations are found, the test will fail with URLSecurityError.

        Skipped on Windows when coverage is running due to memory pressure causing
        access violations with large YAML files.
        """

        # Skip test on Windows when coverage is running to avoid access violations
        if platform.system() == "Windows" and any("--cov" in arg for arg in sys.argv):
            pytest.skip(
                "Skipping cassette URL security test on Windows under coverage due to "
                "access violation with large YAML files"
            )

        validate_existing_cassettes_urls_security()

    def test_total_cost_and_reset(self):
        """
        Runs last and outputs total cost details for the test run, as well
        as tests resetting the usage and cost for the test LLMs.
        """
        # Ensure logger is enabled for cost output (in case previous tests disabled it)
        os.environ[LOGGER_LEVEL_ENV_VAR_NAME] = "DEBUG"
        reload_logger_settings()

        # Output total cost of the test run
        self.output_test_costs()
        # Test resetting all usage and cost stats
        self.llm_group.reset_usage_and_cost()
        self.llm_extractor_text.reset_usage_and_cost()
        for usage_dict in (
            self.llm_group.get_usage() + self.llm_extractor_text.get_usage()
        ):
            assert usage_dict.usage.input == 0
            assert usage_dict.usage.output == 0
        for cost_dict in self.llm_group.get_cost() + self.llm_extractor_text.get_cost():
            assert cost_dict.cost.input == Decimal("0")
            assert cost_dict.cost.output == Decimal("0")
            assert cost_dict.cost.total == Decimal("0")
