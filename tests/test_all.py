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
import sys
import warnings
import xml.etree.ElementTree as ET
import zipfile
from copy import deepcopy
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import Literal

import pytest
from _pytest.nodes import Item
from dotenv import load_dotenv
from pydantic import BaseModel

from contextgem import *
from contextgem.internal.base.attrs import (
    _AssignedAspectsProcessor,
    _AssignedConceptsProcessor,
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.base.llms import _GenericLLMProcessor
from contextgem.internal.converters.docx import (
    WORD_XML_NAMESPACES,
    DocxFormatError,
    _DocxPackage,
)
from contextgem.internal.data_models import _LLMCost, _LLMUsage
from contextgem.internal.items import (
    _BooleanItem,
    _DateItem,
    _FloatItem,
    _IntegerItem,
    _IntegerOrFloatItem,
    _JsonObjectItem,
    _StringItem,
)
from contextgem.internal.loggers import (
    DISABLE_LOGGER_ENV_VAR_NAME,
    LOGGER_LEVEL_ENV_VAR_NAME,
    dedicated_stream,
    logger,
)
from contextgem.internal.utils import _split_text_into_paragraphs
from tests.utils import (
    VCR_FILTER_HEADERS,
    TestUtils,
    get_project_root_path,
    get_test_document_text,
    get_test_img,
    remove_file,
    vcr_before_record_request,
    vcr_before_record_response,
)

# If .env exists locally, it'll be loaded. If it doesn't exist (e.g. in CI),
# no error is raised. Then fallback to environment variables set by the workflow.

load_dotenv()  # This will silently skip if no .env is present


# Add other LLM providers for testing, when needed
TEST_LLM_PROVIDER: Literal["azure_openai", "openai"] = "azure_openai"


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": [(i, "DUMMY") for i in VCR_FILTER_HEADERS],
        "before_record_request": vcr_before_record_request,
        "before_record_response": vcr_before_record_response,
        "match_on": ["method", "host", "path", "body"],
    }


class TestAll(TestUtils):
    """
    Test cases for validating the functionality and error handling of the framework's
    core classes and methods, particularly for document analysis workflows.

    :ivar document: Instance of Document initialized with test data.
    :type document: Document
    :ivar document_pipeline: Instance of DocumentPipeline initialized with test data.
    :type document_pipeline: DocumentPipeline
    :ivar llm_extractor_text: LLM instance configured for text-based extraction tasks.
    :type llm_extractor_text: DocumentLLM
    :ivar llm_reasoner_text: LLM instance configured for text-based reasoning tasks.
    :type llm_reasoner_text: DocumentLLM
    :ivar llm_extractor_vision: LLM instance configured for vision-based extraction tasks.
    :type llm_extractor_vision: DocumentLLM
    :ivar llm_reasoner_vision: LLM instance configured for vision-based reasoning tasks.
    :type llm_reasoner_vision: DocumentLLM
    :ivar llm_group: Instance of DocumentLLMGroup initialized with multiple LLM instances.
    :type llm_group: DocumentLLMGroup
    :ivar llm_with_fallback: Instance of invalid DocumentLLM with a valid DocumentLLM fallback.
    :type llm_with_fallback: DocumentLLM
    :ivar test_img_png: Instance of a test Image (PNG format).
    :type test_img_png: Image
    :ivar test_img_jpg: Instance of a test Image (JPG format).
    :type test_img_jpg: Image
    :ivar test_img_webp: Instance of a test Image (WEBP format).
    :type test_img_webp: Image
    """

    # Documents
    # From raw texts
    document = Document(raw_text=get_test_document_text())
    document_ua = Document(raw_text=get_test_document_text(lang="ua"))
    document_zh = Document(raw_text=get_test_document_text(lang="zh"))
    # From DOCX files
    test_docx_nda_path = os.path.join(
        get_project_root_path(), "tests", "docx_files", "en_nda_with_anomalies.docx"
    )
    test_docx_badly_formatted_path = os.path.join(
        get_project_root_path(), "tests", "docx_files", "badly_formatted.docx"
    )
    document_docx = DocxConverter().convert(test_docx_nda_path)

    # Document pipeline
    document_pipeline = DocumentPipeline(
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

    # LLMs

    if TEST_LLM_PROVIDER == "azure_openai":

        # Extractor text
        _llm_extractor_text_kwargs_openai = {
            "model": "azure/gpt-4.1-mini",
            "api_key": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            "api_base": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            "role": "extractor_text",
            "pricing_details": LLMPricing(
                **{
                    "input_per_1m_tokens": 0.40,
                    "output_per_1m_tokens": 1.60,
                }
            ),
        }

        # Reasoner text
        _llm_reasoner_text_kwargs_openai = {
            "model": "azure/o4-mini",
            "api_key": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
            "api_version": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
            "api_base": os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
            "role": "reasoner_text",
            "pricing_details": LLMPricing(
                **{
                    "input_per_1m_tokens": 1.10,
                    "output_per_1m_tokens": 4.40,
                }
            ),
        }

    elif TEST_LLM_PROVIDER == "openai":

        # Extractor text
        _llm_extractor_text_kwargs_openai = {
            "model": "openai/gpt-4o-mini",
            "api_key": os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            "role": "extractor_text",
            "pricing_details": LLMPricing(
                **{
                    "input_per_1m_tokens": 0.150,
                    "output_per_1m_tokens": 0.600,
                }
            ),
        }

        # Reasoner text
        _llm_reasoner_text_kwargs_openai = {
            "model": "openai/o3-mini",
            "api_key": os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            "role": "reasoner_text",
            "pricing_details": LLMPricing(
                **{
                    "input_per_1m_tokens": 1.10,
                    "output_per_1m_tokens": 4.40,
                }
            ),
        }

    else:
        raise ValueError(f"Test LLM provider {TEST_LLM_PROVIDER} is not supported.")

    if TEST_LLM_PROVIDER in ["azure_openai", "openai"]:

        # Extractor text
        llm_extractor_text = DocumentLLM(**_llm_extractor_text_kwargs_openai)

        # Reasoner text
        llm_reasoner_text = DocumentLLM(**_llm_reasoner_text_kwargs_openai)

        # Extractor vision
        _llm_extractor_vision_kwargs_openai = deepcopy(
            _llm_extractor_text_kwargs_openai
        )
        _llm_extractor_vision_kwargs_openai["role"] = "extractor_vision"
        llm_extractor_vision = DocumentLLM(**_llm_extractor_vision_kwargs_openai)

        # Reasoner vision
        _llm_reasoner_vision_kwargs_openai = deepcopy(
            _llm_extractor_vision_kwargs_openai
        )
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
        llm_with_fallback = DocumentLLM(**_invalid_llm_kwargs)
        llm_with_fallback.fallback_llm = DocumentLLM(
            **_llm_extractor_text_kwargs_openai,
            is_fallback=True,
        )

        # LLM with a non-EN output language setting
        llm_extractor_text_non_eng = DocumentLLM(
            **_llm_extractor_text_kwargs_openai, output_language="adapt"
        )

    # Images
    test_img_png = get_test_img("invoice.png")
    test_img_jpg = get_test_img("invoice.jpg")
    test_img_jpg_2 = get_test_img("invoice2.jpg")
    test_img_webp = get_test_img("invoice.webp")

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
                    with open(file_path, "r", encoding="utf-8") as f:
                        raw_content = f.read()
                    # For testing, apply the same check as for the rendered prompt
                    self.check_rendered_prompt(raw_content)
        assert j2_files_found, prompts_folder_path

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
            LLMPricing(input_per_1m_tokens=0.1)
        # LLM cost model
        _LLMCost(input=Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03"))
        with pytest.raises(ValueError):
            _LLMCost(
                input=-Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03")
            )
        with pytest.raises(ValueError):
            _LLMCost(input=-Decimal("0.001"))

    def test_init_instance_bases(self):
        """
        Tests for initialization of the base classes.
        """
        # Base class direct initialization
        base_classes = [
            _AssignedAspectsProcessor,
            _AssignedConceptsProcessor,
            _AssignedInstancesProcessor,
            _ExtractedItemsAttributeProcessor,
            _RefParasAndSentsAttrituteProcessor,
        ]
        for base_class in base_classes:

            class TestNoRequiredAttrs(base_class):
                value: str = "Test"  # no required attributes

                @property
                def _item_class(self) -> type:
                    return Item

            with pytest.raises(AttributeError):
                TestNoRequiredAttrs()  # initialized with no required attributes

    @pytest.mark.vcr()
    def test_init_local_llm(self):
        """
        Tests for initialization of and getting a response from a local LLM,
        e.g. a model run on Ollama local server.
        """
        document = Document(
            raw_text="Non-disclosure agreement\n\n"
            "The parties shall keep the data confidential.",
        )
        concept = StringConcept(
            name="Contract title", description="The title of the contract."
        )
        document.add_concepts([concept])
        llm = DocumentLLM(
            model="ollama/llama3.1:8b",
            api_base="http://localhost:11434",
            seed=123,
        )

        self.config_llm_async_limiter_for_mock_responses(llm)

        extracted_concepts = llm.extract_concepts_from_document(document)
        self.log_extracted_items_for_instance(extracted_concepts[0])
        extracted_items = extracted_concepts[0].extracted_items
        if extracted_items:
            logger.debug(f"Extracted by local LLM: {extracted_items}")
        else:
            warnings.warn("Local LLM did not return any extracted items.")

    def test_init_api_llm(self):
        """
        Tests the behaviour of the `DocumentLLM` class initialization.
        """
        DocumentLLM(
            model="openai/gpt-4o-mini",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )

        # Base class direct initialization
        with pytest.raises(TypeError):
            _GenericLLMProcessor()

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
        with pytest.raises(ValueError):
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

    def test_init_llm_group(self):
        """
        Tests the behavior of the `DocumentLLMGroup` class initialization.
        """
        # Base class direct initialization
        with pytest.raises(TypeError):
            _GenericLLMProcessor()

        # Invalid params
        with pytest.raises(ValueError):
            DocumentLLMGroup()
        with pytest.raises(ValueError):
            DocumentLLMGroup(llms=[])
        with pytest.raises(ValueError):
            DocumentLLMGroup(llms=[self.llm_reasoner_text])
        with pytest.raises(ValueError):
            DocumentLLMGroup(llms=[1, True])
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
                extra=True,  # extra fields not permitted
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
                prompt_type="aspect",
            )
        with pytest.raises(ValueError, match="no Jinja2 tags"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_concepts_without_tags_path,
                prompt_type="concept",
            )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_aspects_with_tags_path,
            prompt_type="aspect",
        )
        assert (
            "test custom prompt template for aspects"
            in llm_with_updated_prompts._extract_aspect_items_prompt.render()
        )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_concepts_with_tags_path,
            prompt_type="concept",
        )
        assert (
            "test custom prompt template for concepts"
            in llm_with_updated_prompts._extract_concept_items_prompt.render()
        )

    @pytest.mark.parametrize("image", [test_img_png, test_img_jpg, test_img_webp])
    def test_init_and_attach_image(self, image: Image):
        """
        Tests for constructing a Image instance and attaching it to a Document instance.
        """

        with pytest.raises(ValueError):
            Image()
        with pytest.raises(ValueError):
            Image(
                mime_type="image/weoighwe",  # invalid MIME type
                base64_data="base 64 encoded string",
            )
        with pytest.raises(ValueError):
            Image(
                mime_type="image/png",
                base64_data="base 64 encoded string",
                extra=True,  # extra fields not permitted
            )
        image = Document(images=[image])
        with pytest.raises(ValueError):
            Document(images=[image, image])  # duplicate images with same base64 string
        self.check_custom_data_json_serializable(image)

    def test_init_paragraph(self):
        """
        Tests for constructing a Paragraph instance and attaching it to a Document instance.
        """
        paragraph = Paragraph(raw_text="Test paragraph")
        self.check_custom_data_json_serializable(paragraph)
        document = Document(paragraphs=[paragraph])
        assert document.raw_text  # to be populated from paragraphs
        with pytest.raises(ValueError):
            Paragraph()
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="Test paragraph",
                additional_context=1,
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

    def test_init_sentence(self):
        """
        Tests for constructing a Sentence instance and attaching it to a Paragraph instance.
        """
        sentence = Sentence(raw_text="Test sentence")
        self.check_custom_data_json_serializable(sentence)
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
                extra=True,  # extra fields not permitted
            )
        with pytest.raises(ValueError):
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
                llm_role="extractor_vision",  # invalid LLM role
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
        self.check_custom_data_json_serializable(aspect)
        aspect.concepts = aspect_concepts
        assert aspect.concepts is not aspect_concepts
        # Validate assignment
        with pytest.raises(ValueError):
            aspect.add_concepts(
                [
                    1,  # invalid type
                ]
            )
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
            aspect.extracted_items = [1, True]
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
                extra_param=True,
            )

        # Test attaching extracted items
        string_concept.extracted_items = [_StringItem(value="Confidential data")]

        # Test with invalid items
        with pytest.raises(ValueError):
            string_concept.extracted_items = [
                _BooleanItem(value=True)
            ]  # must be _StringItem

        # Verify custom data serialization works
        self.check_custom_data_json_serializable(string_concept)

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
                extra_param=True,
            )

        # Test attaching extracted items
        boolean_concept.extracted_items = [_BooleanItem(value=True)]

        # Test with invalid items
        with pytest.raises(ValueError):
            boolean_concept.extracted_items = [
                _StringItem(value="True")
            ]  # must be _BooleanItem

        # Verify custom data serialization works
        self.check_custom_data_json_serializable(boolean_concept)

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
                numeric_type="string",
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

        # Verify custom data serialization works
        self.check_custom_data_json_serializable(int_concept)
        self.check_custom_data_json_serializable(float_concept)
        self.check_custom_data_json_serializable(any_concept)

    def test_init_and_validate_rating_concept(self):
        """
        Tests the initialization of the RatingConcept class with valid and invalid input parameters.
        """
        # Valid initialization
        valid_rating_concept = RatingConcept(
            name="Customer Satisfaction",
            description="Rating of customer satisfaction",
            rating_scale=RatingScale(start=1, end=5),
        )

        # Test with default rating scale (0-10)
        default_scale_concept = RatingConcept(
            name="Product Quality",
            description="Rating of product quality",
            rating_scale=RatingScale(),
        )

        # Test with invalid rating scale parameters
        with pytest.raises(ValueError):
            RatingScale(start=5, end=5)  # start equals end

        with pytest.raises(ValueError):
            RatingScale(start=10, end=5)  # start greater than end

        with pytest.raises(ValueError):
            RatingScale(start=-1, end=5)  # negative start value

        with pytest.raises(ValueError):
            RatingScale(start=0, end=0)  # end must be greater than 0

        with pytest.raises(ValueError):
            RatingConcept(
                name="Invalid Concept",
                description="Test invalid concept",
                rating_scale=RatingScale(),
                extra=True,  # extra fields not permitted
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
            valid_rating_concept.extracted_items = [
                _IntegerItem(value=6)
            ]  # above max (5)

        # Test with non-integer items
        with pytest.raises(ValueError):
            valid_rating_concept.extracted_items = [_FloatItem(value=3.5)]

        # Test with justifications and references
        concept_with_refs = RatingConcept(
            name="Concept with refs",
            description="Test concept with references",
            rating_scale=RatingScale(start=1, end=10),
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Verify custom data serialization works
        self.check_custom_data_json_serializable(valid_rating_concept)
        self.check_custom_data_json_serializable(default_scale_concept)
        self.check_custom_data_json_serializable(concept_with_refs)

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
                extra=True,  # extra fields not permitted
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
        # From dict with type hints
        dict_struct = {
            "category": str,
            "descriptions": list[str],
        }

        # Class struct
        class Struct:
            category: str
            descriptions: list[str]

        # Pydantic struct
        class PydanticStruct(BaseModel):
            category: str
            descriptions: list[str]

        for struct in [dict_struct, Struct, PydanticStruct]:
            concept = JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information",
                structure=struct,
                examples=[
                    JsonObjectExample(
                        content={
                            "category": "test",
                            "descriptions": ["test", "test", "test"],
                        }
                    )
                ],
                llm_role="extractor_text",
            )
            # Validate assignment
            concept.name = "b2b information"
            with pytest.raises(ValueError):
                concept.name = True
            assert concept.name == "b2b information"
            valid_extracted_data = {
                "category": "test",
                "descriptions": [
                    "desc 1",
                    "desc 2",
                ],
            }
            invalid_extracted_data_1 = {
                "category": None,
            }
            invalid_extracted_data_2 = []
            invalid_extracted_data_3 = {
                "category": "test",
                "descriptions": True,
            }
            concept._get_structure_validator().model_validate(valid_extracted_data)
            for invalid_extracted_data in [
                invalid_extracted_data_1,
                invalid_extracted_data_2,
                invalid_extracted_data_3,
            ]:
                with pytest.raises(ValueError):
                    concept._get_structure_validator().model_validate(
                        invalid_extracted_data
                    )

        concept = JsonObjectConcept(
            name="Business Information",
            description="Categories of Business Information in the following JSON format: ...",
            structure={
                "category": str,
                "description": str,
                "score": int | float | None,
            },
            llm_role="extractor_vision",
            add_justifications=True,
        )
        self.check_custom_data_json_serializable(concept)
        # Invalid extracted items
        with pytest.raises(ValueError):
            concept.extracted_items = [1, True]
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information in the following JSON format: ...",
                structure={
                    str: str,
                },  # invalid mapping
                llm_role="extractor_text",
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information in the following JSON format: ...",
                structure={
                    int: "integer",
                },  # invalid mapping
                llm_role="extractor_text",
                add_justifications=True,
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information in the following JSON format: ...",
                structure={},  # empty mapping
                llm_role="extractor_text",
            )
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Business Information",
                description="Categories of Business Information in the following JSON format: ...",
                structure={
                    "random": Exception,
                },  # invalid value that cannot be represented in text format
                llm_role="extractor_text",
                add_justifications=True,
            )

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
                extra_param=True,
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
            _DateItem(value="01-01-2025")  # must be a date object

        # Test with justifications and references
        date_concept_with_refs = DateConcept(
            name="Date with refs",
            description="Test date concept with references",
            add_justifications=True,
            add_references=True,
            reference_depth="sentences",
        )

        # Verify custom data serialization works
        self.check_custom_data_json_serializable(default_date_concept)
        self.check_custom_data_json_serializable(date_concept_with_refs)

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
            StringExample(content=1)

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
        with pytest.raises(TypeError):
            _ExtractedItem(value=1)
        with pytest.raises(ValueError):
            _StringItem()
        with pytest.raises(ValueError):
            _BooleanItem(value=1)
        with pytest.raises(ValueError):
            _BooleanItem()
        with pytest.raises(ValueError):
            _BooleanItem(value="True")
        with pytest.raises(ValueError):
            _IntegerOrFloatItem()
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=int)
        with pytest.raises(ValueError):
            _JsonObjectItem()
        with pytest.raises(ValueError):
            _JsonObjectItem(value={})
        with pytest.raises(ValueError):
            _JsonObjectItem(value={"hello": {}})
        with pytest.raises(ValueError):
            _StringItem(
                value="Random string",
                extra=True,  # extra fields not permitted
            )

        # List field items' unique IDs
        para = Paragraph(raw_text="Test")
        with pytest.raises(ValueError):
            _IntegerOrFloatItem(value=1.0, reference_paragraphs=[para, para, para])

        # Frozen state
        item = _IntegerOrFloatItem(value=1)
        with pytest.raises(ValueError):
            item.value = 2.0
        self.check_custom_data_json_serializable(item)

    @pytest.mark.parametrize("context", [document, document_pipeline])
    def test_init_document_and_pipeline(self, context: Document | DocumentPipeline):
        """
        Tests different initialization scenarios and validations associated with
        the `Document` and `DocumentPipeline` classes.
        """
        context = context.clone()  # clone for method-scoped state modification
        self.check_custom_data_json_serializable(context)

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
                    )
                    assert document.paragraphs  # to be segmented from text
                    assert all(
                        i.sentences for i in document.paragraphs
                    )  # to be segmented from paragraphs
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
            assert all(
                i.sentences for i in document.paragraphs
            )  # to be segmented from paragraphs
            with pytest.raises(ValueError):
                document.raw_text = "Random text 1"  # cannot be set once populated
            with pytest.raises(ValueError):
                document.paragraphs = [
                    Paragraph(raw_text="Random text 1"),
                ]  # cannot be set once populated
            Document(
                images=[
                    self.test_img_png,
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
                    extra=True,  # extra fields not permitted
                )
            Document(
                raw_text="Random text 1\n\nRandom text 2",
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                ],
                images=[
                    self.test_img_png,
                ],
            )
            # List field items' unique IDs
            para = Paragraph(raw_text="Random text")
            with pytest.raises(ValueError):
                Document(paragraphs=[para, para, para])
            with pytest.raises(ValueError):
                Document(
                    images=[
                        self.test_img_png,
                        self.test_img_png,
                    ]
                )
        # Document pipeline initialization
        elif isinstance(context, DocumentPipeline):
            DocumentPipeline()  # works as we can interactive add aspects and concepts after initialization
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
                DocumentPipeline(
                    aspects=[
                        Aspect(
                            name="Liability",
                            description="Clauses describing liability of the parties",
                        )
                    ],
                    extra=True,  # extra fields not permitted
                )
            DocumentPipeline(
                aspects=[
                    Aspect(
                        name="Liability",
                        description="Clauses describing liability of the parties",
                        llm_role="extractor_text",
                        add_justifications=True,
                    )
                ]
            )
            DocumentPipeline(
                concepts=[
                    StringConcept(
                        name="Business Information",
                        description="Categories of Business Information",
                        llm_role="extractor_text",
                        add_justifications=True,
                    ),
                ]
            )

        # Document and document pipeline initialization
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
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    "Random string",
                ]  # invalid aspect type
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="Liability",
                        description="Clauses describing liability of the parties",
                        llm_role="extractor_text",
                        add_justifications=True,
                    )
                ]  # duplicate aspect
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="Business Information",
                        description="Categories of Business Information",
                        llm_role="extractor_vision",  # unsupported llm role for aspect
                    )
                ]
            )
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
                    name="Business Information",
                    description="Categories of Business Information",
                    llm_role="extractor_text",
                ),
            ]  # duplicate concept
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

    @pytest.mark.vcr()
    def test_system_messages(self):
        """
        Tests the system messages functionality of LLMs.
        """
        system_message = "When asked, introduce yourself as ContextGem."
        if TEST_LLM_PROVIDER == "azure_openai":
            non_reasoning_model = DocumentLLM(
                model="azure/gpt-4.1-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                system_message=system_message,
            )
            o1_model = DocumentLLM(
                model="azure/o1",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                reasoning_effort="low",
                system_message=system_message,
            )
            o3_mini_model = DocumentLLM(
                model="azure/o3-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                system_message=system_message,
            )
            o4_mini_model = DocumentLLM(
                model="azure/o4-mini",
                api_key=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_VERSION"),
                api_base=os.getenv("CONTEXTGEM_AZURE_OPENAI_API_BASE"),
                system_message=system_message,
            )
        elif TEST_LLM_PROVIDER == "openai":
            non_reasoning_model = DocumentLLM(
                model="openai/gpt-4o-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                system_message=system_message,
            )
            o1_model = DocumentLLM(
                model="openai/o1",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                system_message=system_message,
            )
            o3_mini_model = DocumentLLM(
                model="openai/o3-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                system_message=system_message,
            )
            o4_mini_model = DocumentLLM(
                model="openai/o4-mini",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                system_message=system_message,
            )
        for model in [non_reasoning_model, o1_model, o3_mini_model, o4_mini_model]:
            model.chat("What's your name?")
            response = model.get_usage()[0].usage.calls[-1].response
            assert "ContextGem" in response
            logger.debug(response)

    @pytest.mark.vcr()
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])
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
                llm_role="extractor_vision",
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

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])
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
                description="Categories of Intellectual Property",
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
            original_container=aspects[0].concepts,
            assigned_container=attached_aspect.concepts,
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
            assigned_container=attached_aspect.concepts,
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
            assigned_container=attached_aspect.concepts,
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

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])
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
                rating_scale=RatingScale(start=1, end=10),
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
            original_container=document_concepts,
            assigned_container=self.document.concepts,
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

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "document",
        [
            document,
            document_docx,
            document_ua,
            # document_zh
        ],
    )
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])
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
            original_container=document_aspects,
            assigned_container=document.aspects,
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
            original_container=document_aspects,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
            llm_roles=llm.list_roles,
        )
        self.check_instance_container_states(
            original_container=document_concepts,
            assigned_container=document.concepts,
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
            original_container=document_concepts,
            assigned_container=document.concepts,
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

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
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

        self.config_llm_async_limiter_for_mock_responses(self.llm_with_fallback)

        self.llm_with_fallback.extract_all(document)
        self.check_instance_container_states(
            original_container=document_concepts,
            assigned_container=document.concepts,
            assigned_instance_class=_Concept,
            llm_roles=self.llm_with_fallback.list_roles,
        )

        # Check serialization of an LLM with fallback
        self._check_deserialized_llm_config_eq(self.llm_with_fallback)

        # Check usage tokens
        self.check_usage(self.llm_with_fallback)
        # Check cost
        self.check_cost(self.llm_with_fallback)

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "document",
        [
            document,
            document_docx,
            document_ua,
            # document_zh
        ],
    )
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])
    def test_serialization_and_cloning(
        self, document: Document, llm: DocumentLLMGroup | DocumentLLM
    ):
        """
        Tests for custom serialization, deserialization, and cloning of application class instances,
        with preservation of the relevant private attributes.
        """

        self.config_llm_async_limiter_for_mock_responses(llm)

        self.config_llms_for_output_lang(document, llm)

        # Document pipeline serialization
        self.check_instance_serialization_and_cloning(self.document_pipeline)
        for aspect in self.document_pipeline.aspects:
            self.check_instance_serialization_and_cloning(aspect)
            for concept in aspect.concepts:
                self.check_instance_serialization_and_cloning(concept)
        for concept in self.document_pipeline.concepts:
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
                        rating_scale=RatingScale(start=1, end=5),
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
                ],
                llm_role="extractor_text",
                reference_depth="sentences",
            ),
        ]
        document_images = [
            self.test_img_jpg,
            self.test_img_jpg_2,
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
                description="Names of key aspects covered in the document",
                structure={"aspect_names": list[str]},
                singular_occurrence=True,
            ),
            RatingConcept(
                name="Contract quality",
                description="Rate contract quality from the perspective "
                "of its adherence to best practices in NDA drafting.",
                rating_scale=RatingScale(start=1, end=10),
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
            document.get_concept_by_name("Start date")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Contract quality")
        )
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Section titles")
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
        for i, image in enumerate(document.images):
            self.check_instance_serialization_and_cloning(image)

        # Check with document pipeline assignment
        with pytest.raises(RuntimeError):
            document.assign_pipeline(self.document_pipeline)
        document.remove_all_instances()
        document.assign_pipeline(self.document_pipeline)
        llm.extract_all(document)
        self.log_extracted_items_for_instance(
            document.get_concept_by_name("Invoice number check")
        )
        self.check_instance_container_states(
            original_container=self.document_pipeline.aspects,
            assigned_container=document.aspects,
            assigned_instance_class=Aspect,
            llm_roles=llm.list_roles,
        )
        self.check_instance_container_states(
            original_container=self.document_pipeline.concepts,
            assigned_container=document.concepts,
            assigned_instance_class=_Concept,
            llm_roles=llm.list_roles,
        )
        self.check_instance_serialization_and_cloning(document)
        self.check_instance_serialization_and_cloning(self.document_pipeline)

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm)

        # Check serialization of an LLM with fallback
        self._check_deserialized_llm_config_eq(self.llm_with_fallback)

        # Check usage tokens
        self.check_usage(llm)
        # Check cost
        self.check_cost(llm)

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    @pytest.mark.parametrize("llm", [llm_group, llm_extractor_text])
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

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    @pytest.mark.parametrize(
        "image",
        [
            test_img_png,
            # test_img_jpg,
            # test_img_webp
        ],
    )
    def test_vision(self, image: Image):
        """
        Tests for data extraction from document images using vision API.
        """
        with pytest.raises(ValueError):
            Document(images=[])
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
        document = Document(images=[image], concepts=document_concepts)

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
            original_container=document_concepts,
            assigned_container=document.concepts,
            assigned_instance_class=_Concept,
        )

        # Check usage tokens
        self.check_usage(self.llm_group)
        # Check cost
        self.check_cost(self.llm_group)

        # Log costs
        self.output_test_costs()

    @pytest.mark.vcr()
    def test_chat(self):
        """
        Tests for the chat method.
        """
        for model in [
            self.llm_extractor_text,
            self.llm_extractor_vision,
            self.llm_with_fallback,
        ]:
            with pytest.raises(ValueError):
                model.chat("")
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=1)
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=[1])
            with pytest.raises(TypeError):
                model.chat(images=self.test_img_png)
            if model == self.llm_extractor_vision:
                # Check with text + image
                model.chat(
                    "What's the type of this document?", images=[self.test_img_png]
                )
                response = model.get_usage()[0].usage.calls[-1].response.lower()
                assert "invoice" in response
            else:
                # Check with text
                model.chat("What's the result of 2+2?")
                if model == self.llm_with_fallback:
                    response = (
                        model.fallback_llm.get_usage()[0].usage.calls[-1].response
                    )
                else:
                    response = model.get_usage()[0].usage.calls[-1].response
                assert "4" in response
            logger.debug(response)
        # Test for non-vision model
        text_only_model = DocumentLLM(model="openai/gpt-3.5-turbo")
        with pytest.raises(ValueError, match="vision"):
            text_only_model.chat(
                "What's the type of this document?", images=[self.test_img_png]
            )

    def test_logger_disabled(self, monkeypatch, capsys):
        """
        Tests for disabling the logger.
        """

        # Ensure our dedicated stream uses the current (monkeypatched) sys.stdout:
        monkeypatch.setattr(dedicated_stream, "base", sys.stdout)

        # 1) Set environment variable to disable logger
        monkeypatch.setenv(DISABLE_LOGGER_ENV_VAR_NAME, "True")
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

        # Ensure our dedicated stream uses the current (monkeypatched) sys.stdout:
        monkeypatch.setattr(dedicated_stream, "base", sys.stdout)

        # 1) Unset environment variable or set to 'False'
        monkeypatch.delenv(DISABLE_LOGGER_ENV_VAR_NAME, raising=False)
        # or monkeypatch.setenv(DISABLE_LOGGER_ENV_VAR_NAME, "False")
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

        # Ensure our dedicated stream uses the current (monkeypatched) sys.stdout:
        monkeypatch.setattr(dedicated_stream, "base", sys.stdout)

        # 1) Ensure the logger is NOT disabled; only set the log level.
        monkeypatch.setenv(DISABLE_LOGGER_ENV_VAR_NAME, "False")
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, env_level)
        reload_logger_settings()

        # 2) Emit one message per level:
        def log_messages():
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
        assert (
            success_present == should_success
        ), f"SUCCESS unexpected for level {env_level}"
        assert (
            warning_present == should_warning
        ), f"WARNING unexpected for level {env_level}"
        assert error_present == should_error, f"ERROR unexpected for level {env_level}"
        assert (
            critical_present == should_critical
        ), f"CRITICAL unexpected for level {env_level}"

        # 6) After the last check, output all messages for visual formatting check
        if env_level == "CRITICAL":
            monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "DEBUG")
            reload_logger_settings()
            print()
            log_messages()

    @pytest.mark.vcr()
    def test_usage_examples(self):
        """
        Tests for usage examples in project's documentation and README.md.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        """
        from dev.usage_examples.docs.advanced import (
            advanced_aspects_and_concepts_document,
            advanced_aspects_with_concepts,
            advanced_multiple_docs_pipeline,
        )
        from dev.usage_examples.docs.optimizations import (
            optimization_accuracy,
            optimization_choosing_llm,
            optimization_cost,
            optimization_long_docs,
            optimization_speed,
        )
        from dev.usage_examples.docs.quickstart import (
            quickstart_aspect,
            quickstart_concept_aspect,
            quickstart_concept_document_text,
            quickstart_concept_document_vision,
            quickstart_sub_aspect,
        )
        from dev.usage_examples.docs.serialization import serialization
        from dev.usage_examples.readme import (
            llm_chat,
            quickstart_aspect,
            quickstart_concept,
        )

    def test_docstring_examples(self):
        """
        Tests for examples in docstrings.
        Note: Does not add to the tests costs calculation, as the code is executed in isolated modules.
        """
        from dev.usage_examples.docstrings.aspects import def_aspect
        from dev.usage_examples.docstrings.concepts import (
            def_boolean_concept,
            def_date_concept,
            def_json_object_concept,
            def_numerical_concept,
            def_rating_concept,
            def_string_concept,
        )
        from dev.usage_examples.docstrings.data_models import (
            def_llm_pricing,
            def_rating_scale,
        )
        from dev.usage_examples.docstrings.documents import def_document
        from dev.usage_examples.docstrings.examples import (
            def_example_json_object,
            def_example_string,
        )
        from dev.usage_examples.docstrings.images import def_image
        from dev.usage_examples.docstrings.llms import def_llm, def_llm_group
        from dev.usage_examples.docstrings.paragraphs import def_paragraph
        from dev.usage_examples.docstrings.pipelines import def_pipeline
        from dev.usage_examples.docstrings.sentences import def_sentence
        from dev.usage_examples.docstrings.utils import reload_logger_settings

    @pytest.mark.parametrize("raw_text_to_md", [True, False])
    @pytest.mark.parametrize("strict_mode", [True, False])
    @pytest.mark.parametrize(
        "include_options",
        [
            "default",  # All options set to True
            "minimal",  # All options set to False
            "no_images",  # All options True except images
        ],
    )
    def test_docx_converter(
        self, raw_text_to_md: bool, strict_mode: bool, include_options: str
    ):
        """
        Tests for the DocxConverter class, covering both convert() and convert_to_text_format() methods.

        This test uses parametrization to test different combinations of:
        - raw_text_to_md (True/False)
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
                "include_images": True,
            },
            "minimal": {
                "include_tables": False,
                "include_comments": False,
                "include_footnotes": False,
                "include_headers": False,
                "include_footers": False,
                "include_textboxes": False,
                "include_images": False,
            },
            "no_images": {
                "include_tables": True,
                "include_comments": True,
                "include_footnotes": True,
                "include_headers": True,
                "include_footers": True,
                "include_textboxes": True,
                "include_images": False,
            },
        }[include_options]

        # Utility function to verify all _DocxPackage attributes are populated
        def verify_docx_package_attributes(test_file_path_or_object):
            package = _DocxPackage(test_file_path_or_object)
            try:
                # Verify all attributes that are present in test DOCX file are populated
                assert package.archive is not None, "Archive must be populated"
                assert package.rels != {}, "Relationships must be populated"
                assert (
                    package.main_document is not None
                ), "Main document must be populated"
                assert package.styles is not None, "Styles must be populated"
                assert package.numbering is not None, "Numbering must be populated"
                assert package.footnotes is not None, "Footnotes must be populated"
                assert package.comments is not None, "Comments must be populated"
                assert package.headers, "Headers must be populated"
                assert package.footers, "Footers must be populated"
                assert package.images, "Images must be populated"

                logger.debug("All _DocxPackage attributes successfully verified")
            finally:
                if package:
                    package.close()

        # Prepare file objects for different input methods
        with open(self.test_docx_badly_formatted_path, "rb") as f:
            file_bytes = f.read()

        # Create a BytesIO object that can be reset
        def create_bytesio():
            return BytesIO(file_bytes)

        # Function to open the file
        def open_file():
            return open(self.test_docx_badly_formatted_path, "rb")

        # Create converter instance
        converter = DocxConverter()

        # Helper function to verify Document objects
        def verify_document_equality(documents):
            # Check that all documents have the expected properties
            for doc in documents:
                assert isinstance(doc, Document)
                assert doc.raw_text, "Document should have raw text"
                assert doc.paragraphs, "Document should have paragraphs"

                # Verify that each sentence inherits additional_context from its paragraph
                for paragraph in doc.paragraphs:
                    assert (
                        paragraph.additional_context
                    ), "Paragraph should have additional context"
                    for sentence in paragraph.sentences:
                        assert (
                            sentence.additional_context == paragraph.additional_context
                        ), f"Sentence additional_context should match its paragraph's additional_context"
                        assert (
                            sentence.custom_data == paragraph.custom_data
                        ), f"Sentence custom_data should match its paragraph's custom_data"

            # Check that all documents have the same content
            first_doc = documents[0]
            for i, doc in enumerate(documents[1:], 1):
                assert (
                    doc.raw_text == first_doc.raw_text
                ), f"Document {i} has different raw text"
                assert len(doc.paragraphs) == len(
                    first_doc.paragraphs
                ), f"Document {i} has different paragraph count"

                # Check images if they should be included
                if include_params["include_images"]:
                    assert len(doc.images) == len(
                        first_doc.images
                    ), f"Document {i} has different image count"

        # Test 1: Test convert() method with different input sources
        documents = []

        # Convert from file path
        verify_docx_package_attributes(self.test_docx_badly_formatted_path)
        doc_from_path = converter.convert(
            self.test_docx_badly_formatted_path,
            raw_text_to_md=raw_text_to_md,
            strict_mode=strict_mode,
            **include_params,
        )
        documents.append(doc_from_path)

        # Convert from file object
        with open_file() as file_obj:
            verify_docx_package_attributes(file_obj)
            doc_from_obj = converter.convert(
                file_obj,
                raw_text_to_md=raw_text_to_md,
                strict_mode=strict_mode,
                **include_params,
            )
        documents.append(doc_from_obj)

        # Convert from BytesIO
        bytesio = create_bytesio()
        verify_docx_package_attributes(bytesio)
        doc_from_bytesio = converter.convert(
            bytesio,
            raw_text_to_md=raw_text_to_md,
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
                output_format=output_format,
                strict_mode=strict_mode,
                **text_params,
            )
            text_results.append(text_from_path)

            # Convert from file object
            with open_file() as file_obj:
                text_from_obj = converter.convert_to_text_format(
                    file_obj,
                    output_format=output_format,
                    strict_mode=strict_mode,
                    **text_params,
                )
            text_results.append(text_from_obj)

            # Convert from BytesIO
            bytesio = create_bytesio()
            text_from_bytesio = converter.convert_to_text_format(
                bytesio,
                output_format=output_format,
                strict_mode=strict_mode,
                **text_params,
            )
            text_results.append(text_from_bytesio)

            # Verify all text results are equal
            first_result = text_results[0]
            for i, result in enumerate(text_results[1:], 1):
                assert (
                    result == first_result
                ), f"Text result {i} is different for format {output_format}"

    @pytest.mark.vcr()
    def test_docx_converter_llm_extract(self):
        """
        Tests for LLM extraction from DOCX files.
        """

        converter = DocxConverter()

        doc = converter.convert(self.test_docx_badly_formatted_path)

        md_test_chars = ["**", "|", "##"]

        def check_not_markdown(text: str) -> None:
            """
            Checks that the text does not contain markdown.
            """
            assert not any(i in text for i in md_test_chars)

        def check_is_markdown(text: str, expect_newlines: bool = False) -> None:
            """
            Checks that the text contains markdown.
            """
            if expect_newlines:
                assert any(i in text for i in md_test_chars + ["\n"])
            else:
                assert any(i in text for i in md_test_chars)

        # Test concept extraction
        doc.concepts = [
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

        # Test extraction from full text (markdown)
        extracted_concepts = self.llm_group.extract_concepts_from_document(doc)
        assert extracted_concepts[0].extracted_items
        logger.debug(extracted_concepts[0].extracted_items[0].value)
        assert extracted_concepts[0].extracted_items[0].value == 3
        assert extracted_concepts[1].extracted_items
        logger.debug(extracted_concepts[1].extracted_items[0].value)
        assert "4800" in extracted_concepts[1].extracted_items[0].value

        # Check that the text contains newlines and markdown, as full text is passed to LLM
        check_is_markdown(
            self.llm_group.get_usage(llm_role="extractor_text")[0]
            .usage.calls[-1]
            .prompt_kwargs["text"],
            expect_newlines=True,
        )

        # Test extraction with max paragraphs (no markdown)
        extracted_concepts = self.llm_group.extract_concepts_from_document(
            doc, max_paragraphs_to_analyze_per_call=25, overwrite_existing=True
        )
        assert extracted_concepts[0].extracted_items
        # Check that the text does not contain markdown, as we process paragraph chunks
        check_not_markdown(
            self.llm_group.get_usage(llm_role="extractor_text")[0]
            .usage.calls[-1]
            .prompt_kwargs["text"]
        )

        # Test aspect extraction

        # Test with paragraph-level refs (default)
        doc.aspects = [
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
            )
        ]
        extracted_aspects = self.llm_group.extract_aspects_from_document(doc)
        assert extracted_aspects[0].extracted_items
        assert extracted_aspects[0].extracted_items[0].reference_paragraphs
        # Check that the text does not contain markdown, as we process aspect paragraphs
        for paragraph in (
            self.llm_group.get_usage(llm_role="extractor_text")[0]
            .usage.calls[-1]
            .prompt_kwargs["paragraphs"]
        ):
            check_not_markdown(paragraph.raw_text)

        # Test with sentence-level refs
        doc.aspects = [
            Aspect(
                name="Obligations of the receiving party",
                description="Clauses describing the obligations of the receiving party",
                reference_depth="sentences",
            )
        ]
        extracted_aspects = self.llm_group.extract_aspects_from_document(
            doc, overwrite_existing=True
        )
        assert extracted_aspects[0].extracted_items
        assert extracted_aspects[0].extracted_items[0].reference_sentences
        for paragraph in (
            self.llm_group.get_usage(llm_role="extractor_text")[0]
            .usage.calls[-1]
            .prompt_kwargs["paragraphs"]
        ):
            # Check that the text does not contain markdown, as we process
            # aspect paragraphs and sentences
            check_not_markdown(paragraph.raw_text)
            for sentence in paragraph.sentences:
                check_not_markdown(sentence.raw_text)

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

    def test_docx_converter_extract_paragraph_text(self):
        """
        Tests for paragraph text extraction from DOCX elements.
        """

        converter = DocxConverter()

        # Create sample XML elements for testing
        def create_text_element(text_content):
            element = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}t")
            element.text = text_content
            return element

        def create_run_with_text(text_content):
            run = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}r")
            text_elem = create_text_element(text_content)
            run.append(text_elem)
            return run

        def create_paragraph_with_runs(runs):
            para = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}p")
            for run in runs:
                para.append(run)
            return para

        # Basic paragraph with multiple runs
        runs = [
            create_run_with_text("First part. "),
            create_run_with_text("Second part."),
        ]
        para = create_paragraph_with_runs(runs)
        text = converter._extract_paragraph_text(para)
        assert text == "First part. Second part."

        # Paragraph with line breaks
        br_run = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}r")
        br_run.append(ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}br"))
        runs_with_br = [
            create_run_with_text("Before break"),
            br_run,
            create_run_with_text("After break"),
        ]
        para = create_paragraph_with_runs(runs_with_br)
        text = converter._extract_paragraph_text(para)
        assert text == "Before break\nAfter break"

        # Paragraph with footnote reference
        run_with_footnote = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}r")
        footnote_ref = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}footnoteReference")
        footnote_ref.attrib[f"{{{WORD_XML_NAMESPACES['w']}}}id"] = "1"
        run_with_footnote.append(footnote_ref)
        para = create_paragraph_with_runs(
            [create_run_with_text("Text with footnote "), run_with_footnote]
        )
        text = converter._extract_paragraph_text(para)
        assert text == "Text with footnote [Footnote 1]"

        # Empty paragraph
        # Create a paragraph element that will result in empty text
        empty_para = ET.Element(f"{{{WORD_XML_NAMESPACES['w']}}}p")

        # Create a minimal mock package
        class MockPackage:
            def __init__(self):
                self.styles = None
                self.numbering = None

        package = MockPackage()

        result = converter._process_paragraph(empty_para, package)
        assert result is None

    def test_total_cost_and_reset(self):
        """
        Runs last and outputs total cost details for the test run, as well
        as tests resetting the usage and cost for the test LLMs.
        """
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
            assert cost_dict.cost.input == Decimal("0.00000")
            assert cost_dict.cost.output == Decimal("0.00000")
            assert cost_dict.cost.total == Decimal("0.00000")
