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

from contextgem.public.aspects import Aspect
from contextgem.public.concepts import (
    BooleanConcept,
    DateConcept,
    JsonObjectConcept,
    NumericalConcept,
    RatingConcept,
    StringConcept,
)
from contextgem.public.converters import DocxConverter
from contextgem.public.data_models import LLMPricing, RatingScale
from contextgem.public.documents import Document
from contextgem.public.examples import JsonObjectExample, StringExample
from contextgem.public.images import Image
from contextgem.public.llms import DocumentLLM, DocumentLLMGroup
from contextgem.public.paragraphs import Paragraph
from contextgem.public.pipelines import DocumentPipeline
from contextgem.public.sentences import Sentence
from contextgem.public.utils import image_to_base64, reload_logger_settings

__all__ = [
    # Aspects
    "Aspect",
    # Concepts
    "StringConcept",
    "BooleanConcept",
    "NumericalConcept",
    "RatingConcept",
    "JsonObjectConcept",
    "DateConcept",
    # Documents
    "Document",
    # Pipelines
    "DocumentPipeline",
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
    "DocumentLLMGroup",
    # Data models
    "LLMPricing",
    "RatingScale",
    # Utils
    "image_to_base64",
    "reload_logger_settings",
    # Converters
    "DocxConverter",
]
