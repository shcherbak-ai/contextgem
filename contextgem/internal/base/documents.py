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
Module defining the base classes for Document class.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BeforeValidator, Field, field_validator, model_validator
from typing_extensions import Self

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.attrs import _AssignedInstancesProcessor
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.images import _Image
from contextgem.internal.base.md_text import _MarkdownTextAttributesProcessor
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.base.pipelines import _DocumentPipeline, _ExtractionPipeline
from contextgem.internal.decorators import (
    _disable_direct_initialization,
    _post_init_method,
    _timer_decorator,
)
from contextgem.internal.loggers import logger
from contextgem.internal.registry import _publicize
from contextgem.internal.typings.types import NonEmptyStr, SaTModelId
from contextgem.internal.typings.validators import _validate_sequence_is_list
from contextgem.internal.utils import (
    _check_paragraphs_match_in_text,
    _check_paragraphs_ordering_in_text,
    _is_text_content_empty,
    _load_sat_model,
    _split_text_into_paragraphs,
)


@_disable_direct_initialization
class _Document(_AssignedInstancesProcessor, _MarkdownTextAttributesProcessor):
    """
    Internal implementation of the Document class.
    """

    raw_text: NonEmptyStr | None = Field(
        default=None,
        description="Main document text as a single string (optional).",
    )
    paragraphs: list[_Paragraph] = Field(
        default_factory=list,
        description=(
            "Paragraphs in order of appearance. Usually auto-populated from raw_text."
        ),
    )
    images: list[_Image] = Field(
        default_factory=list,
        description="Images attached to or representing the document.",
    )
    aspects: Annotated[
        Sequence[_Aspect], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list,
        description=(
            "Aspects attached to the document. Validated to have unique names and descriptions."
        ),
    )
    concepts: Annotated[
        Sequence[_Concept], BeforeValidator(_validate_sequence_is_list)
    ] = Field(
        default_factory=list,
        description=(
            "Concepts attached to the document. Validated to have unique names and descriptions."
        ),
    )  # using Sequence field with list validator for type checking
    paragraph_segmentation_mode: Literal["newlines", "sat"] = Field(
        default="newlines",
        description=(
            "Paragraph segmentation mode. 'newlines' splits by newlines; 'sat' uses a SaT model."
        ),
    )
    sat_model_id: SaTModelId = Field(
        default="sat-3l-sm",
        description=(
            "SaT model ID for segmentation, or a local path to a SaT model directory."
        ),
    )
    pre_segment_sentences: bool = Field(
        default=False,
        description=(
            "If True, sentences are segmented during initialization using the SaT model."
        ),
    )

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
        if name in ["raw_text", "paragraphs", "_md_text"] and getattr(self, name, None):
            # Prevent raw_text/paragraphs/_md_text reassignment once populated,
            # to prevent inconsistencies in analysis.
            raise ValueError(
                f"The attribute `{name}` cannot be changed once populated."
            )
        if name == "_md_text":
            self._validate_md_text(value)
        super().__setattr__(name, value)

    @property
    def sentences(self) -> list[_Sentence]:
        """
        Provides access to all sentences within the paragraphs of the document by flattening
        and combining sentences from each paragraph into a single list.

        :return: A list of _Sentence objects that are contained within all paragraphs.
        :rtype: list[_Sentence]
        """
        return list(itertools.chain.from_iterable(i.sentences for i in self.paragraphs))

    @_timer_decorator(
        "Document initialization",
    )
    @_post_init_method
    def _post_init(self, __context: Any):
        """
        Post-initialization method that handles document text setup and segmentation.

        This method performs two key operations:
        1. Sets raw_text from paragraphs if raw_text is not provided
        2. Segments document text into paragraphs and optionally sentences

        :param __context: Pydantic context (unused).
        :type __context: Any
        """
        self._set_text_from_paras()
        self._segment_document_text()

    def assign_pipeline(
        self,
        pipeline: _ExtractionPipeline | _DocumentPipeline,  # type: ignore
        overwrite_existing: bool = False,
    ) -> Self:
        """
        Assigns a given pipeline to the document. The method deep-copies the input pipeline
        to prevent any modifications to the state of aspects or concepts in the original pipeline.
        If the aspects or concepts are already associated with the document, an error is raised
        unless the `overwrite_existing` parameter is explicitly set to `True`.

        :param pipeline: The ExtractionPipeline (or deprecated DocumentPipeline) object to
            attach to the document.
        :type pipeline: _ExtractionPipeline | _DocumentPipeline
        :param overwrite_existing: A boolean flag. If set to True, any existing aspects and
            concepts assigned to the document will be overwritten by the new pipeline.
            Defaults to False.
        :type overwrite_existing: bool
        :return: Returns the current instance of the document after assigning the pipeline.
        """
        if (self.aspects or self.concepts) and not overwrite_existing:
            raise RuntimeError(
                "Document already has aspects and concepts assigned. "
                "Use `overwrite_existing=True` to overwrite existing aspects and concepts "
                "with the pipeline."
            )
        extraction_pipeline = deepcopy(
            pipeline
        )  # deep copy to avoid aspect/concept state modification of the pipeline
        self.aspects = extraction_pipeline.aspects
        self.concepts = extraction_pipeline.concepts
        logger.info("Pipeline assigned to the document")
        return self

    def _segment_document_text(self) -> None:
        """
        If no paragraphs are provided, but text exists, segments paragraphs in text,
        creates _Paragraph instances, and assigns them on the document.
        The ``paragraph_segmentation_mode`` value determines whether the paragraphs
        will be segmented by newlines or using a SaT model.

        If ``pre_segment_sentences`` is True and paragraphs exist, segments sentences
        in paragraphs that don't have sentences already assigned.
        Sentences are segmented using the SaT model.

        Does nothing if only images are provided without text or paragraphs.
        """

        if self.raw_text and not self.paragraphs:
            # Segment paragraphs in text, if text provided without paragraphs
            logger.info(
                "Text is being segmented into paragraphs, as no Paragraph instances were provided..."
            )
            if self.paragraph_segmentation_mode == "newlines":
                paragraphs = _split_text_into_paragraphs(self.raw_text)
            elif self.paragraph_segmentation_mode == "sat":
                try:
                    paragraphs = _load_sat_model(self.sat_model_id).split(
                        self.raw_text,
                        do_paragraph_segmentation=True,
                    )
                except Exception as e:
                    logger.error(
                        f"Error splitting text into paragraphs using SaT model: {e}"
                    )
                    raise
                paragraphs = ["".join(i) for i in paragraphs]
            else:
                raise ValueError(
                    f"Invalid paragraph segmentation mode: {self.paragraph_segmentation_mode}"
                )
            if not paragraphs:
                raise ValueError("No valid paragraphs in text")
            # Assign paragraphs on the document
            paragraphs = [_publicize(_Paragraph, raw_text=i) for i in paragraphs]
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

        # Only segment sentences during initialization if `pre_segment_sentences` is True
        if self.pre_segment_sentences:
            self._segment_sents()

    def _segment_sents(self) -> None:
        """
        Segments sentences for paragraphs that don't have sentences already assigned.

        :return: None
        :raises RuntimeError: If no paragraphs exist or if segmented sentences cannot
            be matched in the parent paragraph's text
        :raises Exception: If the SaT model fails to process the paragraphs
        """
        if not self.paragraphs:
            raise RuntimeError("No paragraphs available for sentence segmentation")

        # Only process paragraphs that don't have sentences
        if all(i.sentences for i in self.paragraphs):
            return

        logger.info("Paragraphs are being split into sentences...")
        if any(i.sentences for i in self.paragraphs):
            logger.info(
                "Some paragraphs already have sentences. These will be used `as is`."
            )
        try:
            split_sents_for_paras = _load_sat_model(self.sat_model_id).split(
                [p.raw_text for p in self.paragraphs],
            )
        except Exception as e:
            logger.error(
                f"Error splitting paragraphs into sentences using SaT model: {e}"
            )
            raise
        for paragraph, sent_group in zip(
            self.paragraphs, split_sents_for_paras, strict=True
        ):
            if not paragraph.sentences:
                # Filter out empty sents, if any
                sent_group = [i.strip() for i in sent_group]
                sent_group = [i for i in sent_group if not _is_text_content_empty(i)]
                unmatched_sentences = [
                    i for i in sent_group if i not in paragraph.raw_text
                ]
                if unmatched_sentences:
                    raise ValueError(
                        f"Not all segmented sentences were matched in paragraph text.\n"
                        f"Paragraph text: {paragraph.raw_text}\n"
                        f"Unmatched sentences: {unmatched_sentences}"
                    )
                paragraph.sentences = [
                    _publicize(
                        _Sentence,
                        raw_text=i,
                        custom_data=paragraph.custom_data,
                        additional_context=paragraph.additional_context,
                    )  # inherit custom data and additional context from paragraph object
                    for i in sent_group
                ]

    def _requires_sentence_segmentation(self) -> bool:
        """
        Check if any aspect, sub-aspect, or concept attached to the document requires
        sentence-level segmentation.

        :return: True if any aspect, sub-aspect, or concept requires sentence-level segmentation,
            False otherwise
        :rtype: bool
        """

        # Sentences are already segmented
        if self.sentences:
            return False

        def _check_aspect_requires_sentences(aspect: _Aspect) -> bool:
            """
            Helper function to recursively check if an aspect or its sub-aspects
            require sentence segmentation.

            :param aspect: The aspect to check for sentence segmentation requirements
            :type aspect: _Aspect
            :return: True if the aspect or any of its sub-aspects require
                sentence segmentation
            :rtype: bool
            """
            # Check if this aspect requires sentence-level segmentation
            if aspect.reference_depth == "sentences":
                return True

            # Check if any sub-aspects require sentence-level segmentation
            for sub_aspect in aspect.aspects:
                if _check_aspect_requires_sentences(sub_aspect):
                    return True

            # Check if any concepts in this aspect require sentence-level segmentation
            for concept in aspect.concepts:
                if concept.add_references and concept.reference_depth == "sentences":
                    return True

            return False

        # Check if any aspect or sub-aspect requires sentence-level segmentation
        aspects_requiring_segmentation = [
            aspect
            for aspect in self.aspects
            if _check_aspect_requires_sentences(aspect)
        ]
        # Check document concepts
        concepts_requiring_segmentation = [
            concept
            for concept in self.concepts
            if concept.add_references and concept.reference_depth == "sentences"
        ]

        if aspects_requiring_segmentation or concepts_requiring_segmentation:
            if aspects_requiring_segmentation:
                aspect_names = [
                    aspect.name for aspect in aspects_requiring_segmentation
                ]
                logger.info(
                    f"Sentence-level segmentation is requested for aspects "
                    f"(including sub-aspects, if any): {aspect_names}."
                )

            if concepts_requiring_segmentation:
                concept_names = [
                    concept.name for concept in concepts_requiring_segmentation
                ]
                logger.info(
                    f"Sentence-level segmentation is requested for concepts: {concept_names}."
                )

            logger.info(
                "SaT model will be used for sentence segmentation of the document."
            )
            return True

        return False

    def _set_text_from_paras(self) -> None:
        """
        Sets the raw text attribute for the object by combining text from paragraphs.

        If the `paragraphs` attribute exists and is not empty, while the `raw_text`
        attribute is empty, it merges the text content of all paragraphs and assigns
        it to the `raw_text` attribute.

        :return: None
        """

        # Set raw text from paragraphs
        if self.paragraphs and not self.raw_text:
            logger.info("Raw text is being set from paragraphs...")
            self.raw_text = "\n\n".join([i.raw_text for i in self.paragraphs])

    @field_validator("images")
    @classmethod
    def _validate_images(cls, images: list[_Image]) -> list[_Image]:
        """
        Validates the uniqueness of document images provided in the document.

        :param images: A list of `_Image` objects to validate.
        :return: The original list of `_Image` objects if all images are unique.
        :raises ValueError: If a duplicate image is found in the list, based on its
            `base64_data` content.
        """
        seen = set()
        for image in images:
            if image.base64_data in seen:
                raise ValueError(
                    "Image already exists in the document. All images must be unique."
                )
            seen.add(image.base64_data)
        return images

    @field_validator("sat_model_id")
    @classmethod
    def _validate_sat_model_id(cls, sat_model_id: SaTModelId) -> str:
        """
        Validates and converts the sat_model_id to ensure it's a string.
        If a Path object is provided, it's converted to a string representation.
        This conversion ensures the document remains fully serializable.

        :param sat_model_id: The SaT model ID or path to validate
        :return: String representation of the model ID or path
        """
        if isinstance(sat_model_id, Path):
            return str(sat_model_id)
        return sat_model_id

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

        if isinstance(data, dict) and (
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
        Validates the consistency between document text attributes and paragraphs.

        Verifies that:
        - Document's `_md_text` cannot be populated without `raw_text` being set
        - Each paragraph's text exists in the corresponding document text
        - Paragraphs are ordered according to their appearance in the document text

        Does nothing if both `raw_text` and `paragraphs` are not provided.

        :param self: The instance of the model being validated.

        :return: The validated instance of the model.
        :raises ValueError: If paragraphs are not matched or ordered correctly in
            the document text, or if `_md_text` is provided without `raw_text`.
        """

        # Validate that _md_text cannot be populated without raw_text
        if self._md_text and not self.raw_text:
            raise ValueError(
                "Document's `_md_text` cannot be populated without `raw_text` being set."
            )

        if self.raw_text and self.paragraphs:
            # Check that all paragraphs exist in the document raw text
            unmatched_paragraphs = _check_paragraphs_match_in_text(
                self.paragraphs, self.raw_text, "raw"
            )
            if unmatched_paragraphs:
                unmatched_texts = [p.raw_text for p in unmatched_paragraphs]
                raise ValueError(
                    f"Not all paragraphs were matched in document raw text. "
                    f"Unmatched paragraphs: {unmatched_texts}"
                )
            # Check that paragraphs are ordered according to their appearance in the raw text
            _check_paragraphs_ordering_in_text(self.paragraphs, self.raw_text, "raw")

            # If both document and paragraphs have _md_text, validate matching in _md_text as well
            if self._md_text:
                unmatched_md_paragraphs = _check_paragraphs_match_in_text(
                    self.paragraphs, self._md_text, "markdown"
                )
                if unmatched_md_paragraphs:
                    unmatched_md_texts = [p._md_text for p in unmatched_md_paragraphs]
                    raise ValueError(
                        f"Paragraphs with _md_text were not matched in document _md_text. "
                        f"Unmatched _md_text: {unmatched_md_texts}"
                    )
                # Check that paragraphs are ordered according to their appearance in the markdown text
                _check_paragraphs_ordering_in_text(
                    self.paragraphs, self._md_text, "markdown"
                )

        return self
