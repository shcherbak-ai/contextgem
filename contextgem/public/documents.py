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
Module for handling documents.

This module provides the Document class, which represents a structured or unstructured file
containing written or visual content. Documents can be processed to extract information,
analyze content, and organize data into paragraphs, sentences, aspects, and concepts.

The Document class supports various operations including:
- Managing raw text and structured paragraphs
- Handling embedded or attached images
- Organizing content into aspects for focused analysis
- Associating concepts for information extraction
- Processing documents through pipelines for automated analysis

Documents serve as the primary container for content analysis within the ContextGem framework,
enabling complex document understanding and information extraction workflows.
"""

from __future__ import annotations

import itertools
import warnings
from copy import deepcopy
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from contextgem.internal.base.attrs import _AssignedInstancesProcessor
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.decorators import _post_init_method, _timer_decorator
from contextgem.internal.loggers import logger
from contextgem.internal.typings.aliases import NonEmptyStr, SaTModelId, Self
from contextgem.internal.utils import _get_sat_model, _split_text_into_paragraphs
from contextgem.public.aspects import Aspect
from contextgem.public.images import Image
from contextgem.public.paragraphs import Paragraph
from contextgem.public.pipelines import DocumentPipeline
from contextgem.public.sentences import Sentence


class Document(_AssignedInstancesProcessor):
    """
    Represents a document containing textual and visual content for analysis.

    A document serves as the primary container for content analysis within the ContextGem framework,
    enabling complex document understanding and information extraction workflows.

    :ivar raw_text: The main text of the document as a single string.
        Defaults to None.
    :type raw_text: Optional[NonEmptyStr]
    :ivar paragraphs: List of Paragraph instances in consecutive order as they appear
        in the document. Defaults to an empty list.
    :type paragraphs: list[Paragraph]
    :ivar images: List of Image instances attached to or representing the document.
        Defaults to an empty list.
    :type images: list[Image]
    :ivar aspects: List of aspects associated with the document for focused analysis.
        Validated to ensure unique names and descriptions. Defaults to an empty list.
    :type aspects: list[Aspect]
    :ivar concepts: List of concepts associated with the document for information extraction.
        Validated to ensure unique names and descriptions. Defaults to an empty list.
    :type concepts: list[_Concept]
    :ivar paragraph_segmentation_mode: Mode for paragraph segmentation. When set to "sat",
        uses a SaT (Segment Any Text https://arxiv.org/abs/2406.16678) model. Defaults to "newlines".
    :type paragraph_segmentation_mode: Literal["newlines", "sat"]
    :ivar sat_model_id: SaT model ID for paragraph/sentence segmentation.
        Defaults to "sat-3l-sm". See https://github.com/segment-any-text/wtpsplit for the list of available models.
    :type sat_model_id: SaTModelId

    Note:
        Normally, you do not need to construct/populate paragraphs manually, as they are
        populated automatically from document's ``raw_text`` attribute. Only use this constructor
        for advanced use cases, such as when you have a custom paragraph segmentation tool.

    Example:
        .. literalinclude:: ../../../dev/usage_examples/docstrings/documents/def_document.py
            :language: python
            :caption: Document definition
    """

    raw_text: Optional[NonEmptyStr] = Field(default=None)
    paragraphs: list[Paragraph] = Field(default_factory=list)
    images: list[Image] = Field(default_factory=list)
    aspects: list[Aspect] = Field(default_factory=list)
    concepts: list[_Concept] = Field(default_factory=list)
    paragraph_segmentation_mode: Literal["newlines", "sat"] = Field(default="newlines")
    sat_model_id: SaTModelId = Field(default="sat-3l-sm")

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Sets the attribute of an instance, with additional restrictions on specific attributes.

        :param name: The name of the attribute to set.
        :type name: str
        :param value: The value to assign to the attribute.
        :return: None
        :raises ValueError: If attempting to reassign a restricted attribute
            after it has already been assigned to a *truthy* value.
        """
        if name in ["raw_text", "paragraphs"]:
            # Prevent raw_text/paragraphs reassignment once populated, to prevent inconsistencies in analysis.
            if getattr(self, name, None):
                raise ValueError(
                    f"The attribute `{name}` cannot be changed once populated."
                )
        super().__setattr__(name, value)

    @property
    def sentences(self) -> list[Sentence]:
        """
        Provides access to all sentences within the paragraphs of the document by flattening
        and combining sentences from each paragraph into a single list.

        :return: A list of Sentence objects that are contained within all paragraphs.
        :rtype: list[Sentence]
        """
        return list(itertools.chain.from_iterable(i.sentences for i in self.paragraphs))

    @_timer_decorator(
        "Document initialization",
    )
    @_post_init_method
    def _post_init(self, __context):
        self._set_text_from_paras()
        self._segment_paras_and_sents()

    def assign_pipeline(
        self,
        pipeline: DocumentPipeline,
        overwrite_existing: bool = False,
    ) -> Self:
        """
        Assigns a given pipeline to the document. The method deep-copies the input pipeline
        to prevent any modifications to the state of aspects or concepts in the original pipeline.
        If the aspects or concepts are already associated with the document, an error is raised
        unless the `overwrite_existing` parameter is explicitly set to `True`.

        :param pipeline: The DocumentPipeline object to attach to the document.
        :param overwrite_existing: A boolean flag. If set to True, any existing aspects and
            concepts assigned to the document will be overwritten by the new pipeline.
            Defaults to False.
        :return: Returns the current instance of the document after assigning the pipeline.
        """
        if (self.aspects or self.concepts) and not overwrite_existing:
            raise RuntimeError(
                "Document already has aspects and concepts assigned. "
                "Use `overwrite_existing=True` to overwrite existing aspects and concepts "
                "with the pipeline."
            )
        document_pipeline = deepcopy(
            pipeline
        )  # deep copy to avoid aspect/concept state modification of the pipeline
        self.aspects = document_pipeline.aspects
        self.concepts = document_pipeline.concepts
        logger.info("Pipeline assigned to the document")
        return self

    def _segment_paras_and_sents(self) -> None:
        """
        If no paragraphs are provided, but text exists, extracts paragraphs from text and assigns
        them on the document. The ``paragraph_segmentation_mode`` value determines whether the paragraphs
        will be segmented by newlines or using a SaT model.

        If paragraphs exist and some of them do not have extracted sentences, extracts sentences
        for such paragraphs and assigns them on the paragraphs. Sentences are always segmented
        using the SaT model.

        Does nothing if only images are provided without text or paragraphs.
        """

        if self.raw_text and not self.paragraphs:
            # Extract paragraphs from text, if text provided without paragraphs
            logger.info(
                "Text is being split into paragraphs, as no custom paragraphs were provided..."
            )
            if self.paragraph_segmentation_mode == "newlines":
                paragraphs: list[str] = _split_text_into_paragraphs(self.raw_text)
            elif self.paragraph_segmentation_mode == "sat":
                paragraphs: list[list[str]] = _get_sat_model(self.sat_model_id).split(
                    self.raw_text,
                    do_paragraph_segmentation=True,
                )
                paragraphs = ["".join(i) for i in paragraphs]
            else:
                raise ValueError(
                    f"Invalid paragraph segmentation mode: {self.paragraph_segmentation_mode}"
                )
            if not paragraphs:
                raise ValueError("No valid paragraphs in text")
            # Assign paragraphs on the document
            paragraphs: list[Paragraph] = [Paragraph(raw_text=i) for i in paragraphs]
            # Check that each paragraph is found in the document text
            # For duplicate paragraphs, verify each occurrence is matched in the document
            remaining_text = self.raw_text
            for paragraph in paragraphs:
                if paragraph.raw_text not in remaining_text:
                    raise ValueError(
                        "Not all segmented paragraphs were matched in document text."
                    )
                # Remove the first occurrence to handle duplicates correctly
                remaining_text = remaining_text.replace(paragraph.raw_text, "", 1)
            self.paragraphs = paragraphs

        if self.paragraphs:
            # Extract sentences for each paragraph without sentences provided
            if not all(i.sentences for i in self.paragraphs):
                logger.info("Paragraphs are being split into sentences...")
                if any(i.sentences for i in self.paragraphs):
                    warnings.warn(
                        "Some paragraphs already have sentences. "
                        "These will be used `as is`."
                    )
                split_sents_for_paras = _get_sat_model(self.sat_model_id).split(
                    [p.raw_text for p in self.paragraphs]
                )
                for paragraph, sent_group in zip(
                    self.paragraphs, split_sents_for_paras
                ):
                    if not paragraph.sentences:
                        # Filter out empty sents, if any
                        sent_group = [i.strip() for i in sent_group]
                        sent_group = [i for i in sent_group if len(i)]
                        assert all(
                            i in paragraph.raw_text for i in sent_group
                        ), "Not all segmented sentences were matched in paragraph text."
                        paragraph.sentences = [
                            Sentence(
                                raw_text=i,
                                custom_data=paragraph.custom_data,
                                additional_context=paragraph.additional_context,
                            )  # inherit custom data and additional context from paragraph object
                            for i in sent_group
                        ]

    def _set_text_from_paras(self) -> None:
        """
        Sets the text attribute for the object by combining text from paragraphs.

        This method checks if the `paragraphs` attribute of the object exists and
        is not empty, while the `text` attribute is empty. If these conditions
        are met, it merges the text content of all the paragraphs and assigns it
        to the `text` attribute.

        :return: None
        """
        if self.paragraphs and not self.raw_text:
            logger.info("Text is being set from paragraphs...")
            self.raw_text = "\n\n".join([i.raw_text for i in self.paragraphs])

    @field_validator("images")
    @classmethod
    def _validate_images(cls, images: list[Image]) -> list[Image]:
        """
        Validates the uniqueness of document images provided in the document.

        :param images: A list of `Image` objects to validate.
        :return: The original list of `Image` objects if all images are unique.
        :raises ValueError: If a duplicate image is found in the list, based on its
            `base64_data` content.
        """
        seen = set()
        for image in images:
            if image.base64_data in seen:
                raise ValueError(
                    f"Image already exists in the document. All images must be unique."
                )
            seen.add(image.base64_data)
        return images

    @model_validator(mode="before")
    @classmethod
    def _validate_document_pre(cls, data: Any) -> Any:
        """
        Validates the document's raw input data, which could be a dict with input values,
            an instance of the model, or another type depending on what is passed to the model.

        :raises ValueError:
            - If none of `raw_text`, `paragraphs`, or `images` are provided.
        :return:
            The validated data.
        """

        if isinstance(data, dict):
            if (
                not data.get("raw_text")
                and not data.get("paragraphs")
                and not data.get("images")
            ):
                raise ValueError(
                    "Either raw_text, paragraphs, or images must be provided for the document."
                )
        return data

    @model_validator(mode="after")
    def _validate_document_post(self) -> Self:
        """
        Validates the consistency between the `text` attribute and the `paragraphs` attribute
        of the instance. Specifically, verifies that if both `text` and `paragraphs` are provided,
        each paragraph's `text` must exist in the overall document's `text`.

        Does nothing if both `text` and `paragraphs` are not provided.

        :param self: The instance of the model being validated.

        :return: The validated instance of the model.
        :raises ValueError: If the `text` attribute exists, and not all paragraphs are matched in
            the overall document's text.
        """
        if self.raw_text and self.paragraphs:
            # Check that all paragraphs exist in the document text
            if not all(i.raw_text in self.raw_text for i in self.paragraphs):
                raise ValueError("Not all paragraphs were matched in document text.")

            # Check that paragraphs are ordered according to their appearance in the raw text
            # Handle case where paragraphs may have duplicate text content
            current_search_pos = 0
            for i in range(len(self.paragraphs) - 1):
                # Find current paragraph starting from the current search position
                current_pos = self.raw_text.find(
                    self.paragraphs[i].raw_text, current_search_pos
                )
                if current_pos == -1:  # This shouldn't happen due to earlier check
                    current_pos = self.raw_text.find(self.paragraphs[i].raw_text)

                # Update search position for next paragraph to start after current paragraph
                current_search_pos = current_pos + len(self.paragraphs[i].raw_text)

                # Find next paragraph starting from the current search position
                next_pos = self.raw_text.find(
                    self.paragraphs[i + 1].raw_text, current_search_pos
                )
                if (
                    next_pos == -1
                ):  # If not found from current position, check if it exists earlier
                    next_pos = self.raw_text.find(self.paragraphs[i + 1].raw_text)
                    if next_pos < current_search_pos:
                        raise ValueError(
                            "Paragraphs are not ordered according to their appearance in the document text."
                        )
        return self
