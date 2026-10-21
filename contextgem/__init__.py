# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
ContextGem - Extract structured data and insights from documents using LLMs and decision models
"""

__version__ = "1.2.0"
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
    NumericalConcept,
    Paragraph,
    RatingConcept,
    Sentence,
    StringConcept,
    StringExample,
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
    "LLMToolLoopLimitError",
    "ContextGemWarning",
)
