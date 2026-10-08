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
Module defining the base class of the feature and integration tests.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from contextgem.internal.utils import _get_template, _load_sat_model
from contextgem.public import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    ExtractionPipeline,
    ModelPricing,
    StringConcept,
)
from tests.utils import (
    TestUtils,
    get_project_root_path,
    get_test_document_text,
    get_test_img,
    set_dummy_env_variables_for_testing_from_cassettes,
)


# If .env exists locally, it'll be loaded. If it doesn't exist (e.g. in CI),
# no error is raised. Then fallback to environment variables set by the workflow.

if not load_dotenv():  # Returns False if no .env file is found
    set_dummy_env_variables_for_testing_from_cassettes()


class BaseTest(TestUtils):
    """
    Base class of the feature and integration tests, which validate the functionality
    and error handling of the framework's core classes and methods, particularly
    for document analysis workflows.

    Variables are initialized at a class level, once, to be used in the test methods
    of the test classes that inherit from this class. This class defines no tests.

    The objects are shared by all test classes, so a test must leave them unchanged.
    A test that changes a document works on its copy (``self.document.clone()``).
    The usage and cost recorded by the shared LLMs are reset before each test, and
    their output language is restored after each test (see the fixtures below).
    The tests do not depend on the order in which they are run.
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

    @pytest.fixture(autouse=True)
    def reset_llms_usage_and_cost(self):
        """
        Resets the usage and cost recorded by the LLMs shared by the tests.

        The reset before each test makes the usage and cost that a test reads those
        of its own LLM calls, independently of the tests run before it.
        """
        for llm in (
            self.llm_extractor,
            self.llm_reasoner,
            self.model_group,
            self.llm_invalid,
            self.invalid_llm_with_valid_fallback,
            self.llm_extractor_non_eng,
        ):
            llm.reset_usage_and_cost()

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
