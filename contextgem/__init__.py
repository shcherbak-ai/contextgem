# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
ContextGem - Extract structured data and insights from documents using LLMs and decision models

Documentation: https://contextgem.shcherbak.ai/

For coding agents and LLMs, https://contextgem.shcherbak.ai/llms.txt indexes
the documentation, with a Markdown copy of each page.

The documentation describes the latest release, and the docstrings of this
package describe the version that is installed (``contextgem.__version__``):
where the two differ, the docstrings apply. The classes and functions to use
are those that this module exports (``contextgem.__all__``).
"""

__version__ = "1.5.0"
__author__ = "Shcherbak AI AS"

from contextgem.public import (
    Aspect,
    BooleanConcept,
    ChatSession,
    ContextGemError,
    ContextGemWarning,
    DateConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    ExtractionPipeline,
    Image,
    JsonObjectClassStruct,
    JsonObjectConcept,
    JsonObjectExample,
    LabelConcept,
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelExtractionError,
    ModelPricing,
    ModelRefusalError,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    Sentence,
    StringConcept,
    StringExample,
    ToolCallError,
    UnknownPricesWarning,
    create_image,
    image_to_base64,
    register_tool,
    release_sat_model,
    reload_logger_settings,
)


__all__ = (  # noqa: RUF022 - grouped by kind
    # Aspects
    "Aspect",
    # Concepts
    "StringConcept",
    "BooleanConcept",
    "NumericalConcept",
    "RatingConcept",
    "JsonObjectConcept",
    "DateConcept",
    "LabelConcept",
    # Documents
    "Document",
    # Pipelines
    "ExtractionPipeline",
    # Paragraphs
    "Paragraph",
    # Sentences
    "Sentence",
    # Images
    "Image",
    # Examples
    "StringExample",
    "JsonObjectExample",
    # LLMs
    "ChatSession",
    "DocumentLLM",
    # Decision models
    "DocumentDecisionModel",
    # Model groups
    "DocumentModelGroup",
    # Data models
    "ModelPricing",
    # Utils
    "create_image",
    "image_to_base64",
    "reload_logger_settings",
    "release_sat_model",
    "JsonObjectClassStruct",
    # Decorators
    "register_tool",
    # Exceptions and warnings
    "ContextGemError",
    "ModelAPIError",
    "ModelExtractionError",
    "ModelRefusalError",
    "LLMToolLoopLimitError",
    "ToolCallError",
    "ContextGemWarning",
    "UnknownPricesWarning",
)
