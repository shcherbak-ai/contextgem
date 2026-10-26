# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

from contextgem.public.aspects import Aspect
from contextgem.public.concepts import (
    BooleanConcept,
    DateConcept,
    JsonObjectConcept,
    LabelConcept,
    NumericalConcept,
    RatingConcept,
    StringConcept,
)
from contextgem.public.data_models import ModelPricing
from contextgem.public.decision_models import DocumentDecisionModel
from contextgem.public.decorators import register_tool
from contextgem.public.documents import Document
from contextgem.public.examples import JsonObjectExample, StringExample
from contextgem.public.exceptions import (
    ContextGemError,
    ContextGemWarning,
    LLMToolLoopLimitError,
    ModelAPIError,
    ModelExtractionError,
    ModelRefusalError,
    ToolCallError,
    UnknownPricesWarning,
)
from contextgem.public.images import Image
from contextgem.public.llms import ChatSession, DocumentLLM
from contextgem.public.models import DocumentModelGroup
from contextgem.public.paragraphs import Paragraph
from contextgem.public.pipelines import ExtractionPipeline
from contextgem.public.sentences import Sentence
from contextgem.public.utils import (
    JsonObjectClassStruct,
    create_image,
    image_to_base64,
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
    "DocumentLLM",
    "ChatSession",
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
