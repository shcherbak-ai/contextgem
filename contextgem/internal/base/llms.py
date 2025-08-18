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
Module defining the base classes for LLM subclasses.

This module provides foundational class structures for LLM implementations
in the ContextGem framework. It includes abstract base classes and utility functions
that define the interface and common functionality for different types of LLMs,
enabling document analysis, information extraction, and reasoning capabilities
across the framework.
"""

from __future__ import annotations

import asyncio
import warnings
from abc import ABC, abstractmethod
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from aiolimiter import AsyncLimiter
from genai_prices import UpdatePrices as _GPUpdatePrices
from genai_prices import Usage as _GPUsage
from genai_prices import calc_price as _calc_auto_price
from jinja2 import Template
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.data_models import _LLMPricing
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.images import _Image
from contextgem.internal.base.mixins import _PostInitCollectorMixin
from contextgem.internal.base.paras_and_sents import _Paragraph
from contextgem.internal.base.serialization import _InstanceSerializer
from contextgem.internal.data_models import (
    _LLMCall,
    _LLMCost,
    _LLMCostOutputContainer,
    _LLMUsage,
    _LLMUsageOutputContainer,
)
from contextgem.internal.decorators import (
    _disable_direct_initialization,
    _post_init_method,
    _timer_decorator,
)
from contextgem.internal.exceptions import LLMAPIError, LLMExtractionError
from contextgem.internal.items import _ExtractedItem, _StringItem
from contextgem.internal.loggers import logger
from contextgem.internal.registry import _publicize
from contextgem.internal.typings.aliases import (
    AsyncCalsAndKwargs,
    DefaultPromptType,
    ExtractedInstanceType,
    JustificationDepth,
    LanguageRequirement,
    LLMRoleAny,
    NonEmptyStr,
    ReasoningEffort,
    ReferenceDepth,
)
from contextgem.internal.typings.validators import _validate_sequence_is_list
from contextgem.internal.utils import (
    _async_multi_executor,
    _chunk_list,
    _clean_text_for_llm_prompt,
    _get_template,
    _group_instances_by_fields,
    _llm_call_result_is_valid,
    _parse_llm_output_as_json,
    _remove_thinking_content_from_llm_output,
    _run_async_calls,
    _run_sync,
    _setup_jinja2_template,
    _suppress_litellm_warnings_context,
    _validate_parsed_llm_output,
)


with _suppress_litellm_warnings_context():
    import litellm


if TYPE_CHECKING:
    from contextgem.internal.base.llms import _DocumentLLM, _DocumentLLMGroup


litellm.suppress_debug_info = True


# Local model providers supported via liteLLM
_LOCAL_MODEL_PROVIDERS = [
    "ollama/",
    "ollama_chat/",
    "lm_studio/",
]

# Rounding precision for reporting LLM costs (quantize on access only)
_COST_QUANT = Decimal("0.00001")


class _GenericLLMProcessor(_PostInitCollectorMixin, _InstanceSerializer, ABC):
    """
    Base class that handles processing logic using LLMs.

    This abstract class provides the foundation for implementing LLM-based processing
    operations within the ContextGem framework. It defines the core interface and shared
    functionality for document analysis, information extraction, and content processing
    using various LLM backends.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    def extract_all(
        self,
        document: _Document,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> _Document:
        """
        Extracts all aspects and concepts from a document and its aspects.

        This method performs comprehensive extraction by processing the document for aspects
        and concepts, then extracting concepts from each aspect. The operation can be
        configured for concurrent processing and customized extraction parameters.

        This is the synchronous version of `extract_all_async()`.

        :param document: The document to analyze.
        :type document: _Document
        :param overwrite_existing: Whether to overwrite already processed aspects and concepts with
            newly extracted information. Defaults to False.
        :type overwrite_existing: bool, optional
        :param max_items_per_call: Maximum number of items with the same extraction params to process
            in each LLM call. Defaults to 0 (all items in one call). If concurrency is enabled, defaults to 1.
            For complex tasks, you should not set a high value, in order to avoid prompt overloading.
        :type max_items_per_call: int, optional
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool, optional
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to include in a
            single LLM prompt. Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int, optional
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images). Relevant only for document-level concepts.
        :type max_images_to_analyze_per_call: int, optional
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: The document with extracted aspects and concepts.
        :rtype: _Document
        """
        return _run_sync(
            self.extract_all_async(
                document=document,
                overwrite_existing=overwrite_existing,
                max_items_per_call=max_items_per_call,
                use_concurrency=use_concurrency,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                max_images_to_analyze_per_call=max_images_to_analyze_per_call,
                raise_exception_on_extraction_error=raise_exception_on_extraction_error,
            )
        )

    @_timer_decorator(process_name="All aspects and concepts extraction")
    async def extract_all_async(
        self,
        document: _Document,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> _Document:
        """
        Asynchronously extracts all aspects and concepts from a document and its aspects.

        This method performs comprehensive extraction by processing the document for aspects
        and concepts, then extracting concepts from each aspect. The operation can be
        configured for concurrent processing and customized extraction parameters.

        :param document: The document to analyze.
        :type document: _Document
        :param overwrite_existing: Whether to overwrite already processed aspects and concepts with
            newly extracted information. Defaults to False.
        :type overwrite_existing: bool, optional
        :param max_items_per_call: Maximum number of items with the same extraction params to process
            in each LLM call. Defaults to 0 (all items in one call). If concurrency is enabled,
            defaults to 1. For complex tasks, you should not set a high value, in order to avoid
            prompt overloading.
        :type max_items_per_call: int, optional
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool, optional
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to include in a
            single LLM prompt. Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int, optional
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images). Relevant only for document-level concepts.
        :type max_images_to_analyze_per_call: int, optional
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: The document with extracted aspects and concepts.
        :rtype: _Document
        """

        self._check_llm_roles_before_extract_all(document)

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        # Extract all aspects in the document
        await self.extract_aspects_from_document_async(
            document=document,
            overwrite_existing=overwrite_existing,
            max_items_per_call=max_items_per_call,
            use_concurrency=use_concurrency,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            raise_exception_on_extraction_error=raise_exception_on_extraction_error,
        )

        extract_concepts_kwargs = {
            "document": document,
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }
        aspect_kwargs_list = [
            {**extract_concepts_kwargs, "aspect": i} for i in document.aspects
        ]

        aspect_cals_and_kwargs = [
            (self.extract_concepts_from_aspect_async, i) for i in aspect_kwargs_list
        ]
        doc_concepts_cals_and_kwargs = [
            (
                self.extract_concepts_from_document_async,
                {
                    **extract_concepts_kwargs,
                    "max_images_to_analyze_per_call": max_images_to_analyze_per_call,
                },
            )
        ]
        # Safe cast: Concatenating two lists of (async_callable, kwargs) tuples
        # Type checker needs help to understand the result is still AsyncCalsAndKwargs
        cals_and_kwargs = cast(
            AsyncCalsAndKwargs, aspect_cals_and_kwargs + doc_concepts_cals_and_kwargs
        )
        await _run_async_calls(
            cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
        )

        return document

    def extract_aspects_from_document(
        self,
        document: _Document,
        from_aspects: Sequence[_Aspect]
        | None = None,  # using Sequence type with list validator for type checking
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> list[_Aspect]:
        """
        Extracts aspects from the provided document using predefined LLMs.

        If an aspect instance has ``extracted_items`` populated, the ``reference_paragraphs`` field will be
        automatically populated from these items.

        This is the synchronous version of `extract_aspects_from_document_async()`.

        :param document: The document from which aspects are to be extracted.
        :type document: _Document
        :param from_aspects: Existing aspects to use as a base for extraction. If None, uses all
            document's aspects.
        :type from_aspects: list[_Aspect] | None
        :param overwrite_existing: Whether to overwrite already processed aspects with newly extracted information.
            Defaults to False.
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum items with the same extraction params to process per LLM call.
            Defaults to 0 (all items in single call). For complex tasks, you should not set a value, to avoid
            prompt overloading. If concurrency is enabled, defaults to 1 (each item processed separately).
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM prompt.
            Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed _Aspect objects with extracted items.
        :rtype: list[_Aspect]
        """
        return _run_sync(
            self.extract_aspects_from_document_async(
                document=document,
                from_aspects=from_aspects,
                overwrite_existing=overwrite_existing,
                max_items_per_call=max_items_per_call,
                use_concurrency=use_concurrency,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                raise_exception_on_extraction_error=raise_exception_on_extraction_error,
            )
        )

    @_timer_decorator(process_name="Aspects extraction from document")
    async def extract_aspects_from_document_async(
        self,
        document: _Document,
        from_aspects: Sequence[_Aspect]
        | None = None,  # using Sequence type with list validator for type checking
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> list[_Aspect]:
        """
        Extracts aspects from the provided document using predefined LLMs asynchronously.

        If an aspect instance has ``extracted_items`` populated, the ``reference_paragraphs`` field will be
        automatically populated from these items.

        :param document: The document from which aspects are to be extracted.
        :type document: _Document
        :param from_aspects: Existing aspects to use as a base for extraction. If None, uses all
            document's aspects.
        :type from_aspects: list[_Aspect] | None
        :param overwrite_existing: Whether to overwrite already processed aspects with newly extracted information.
            Defaults to False.
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum number of items with the same extraction params to process
            per LLM call. Defaults to 0 (all items in one call). If concurrency is enabled,
            defaults to 1. For complex tasks, you should not set a high value, in order to avoid
            prompt overloading.
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM prompt.
            Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed _Aspect objects with extracted items.
        :rtype: list[_Aspect]
        """

        if from_aspects is not None:
            from_aspects = _validate_sequence_is_list(from_aspects)

        self._check_instances_and_llm_params(
            target=document,
            llm_or_group=self,
            instances_to_process=from_aspects,
            instance_type="aspect",
            overwrite_existing=overwrite_existing,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        extract_instances_kwargs = {
            "context": document,
            "instance_type": "aspect",
            "document": document,
            "from_instances": from_aspects,
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }

        if self.is_group:
            # Safe cast: self is a _DocumentLLMGroup because `is_group` is True
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in cast("_DocumentLLMGroup", self).llms
            ]
            # Safe cast: List comprehension creates tuples of (async_method, kwargs)
            # Type checker needs help to recognize this matches AsyncCalsAndKwargs format
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            # Safe cast: self is a _DocumentLLM because `is_group` is False
            await self._extract_instances(
                **extract_instances_kwargs, llm=cast("_DocumentLLM", self)
            )

        document_aspects = from_aspects if from_aspects else document.aspects

        # Extract sub-aspects
        extract_sub_aspects_kwargs = {
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }
        for aspect in document_aspects:
            if aspect.aspects:
                # Validate proper nesting level of sub-aspects
                self._validate_sub_aspects_nesting_level(aspect)
                logger.info(f"Extracting sub-aspects for aspect `{aspect.name}`")
                if not aspect.reference_paragraphs:
                    logger.info(
                        f"Aspect `{aspect.name}` has no extracted paragraphs. "
                        f"Sub-aspects will not be extracted."
                    )
                    continue
                # Treat an aspect as a document containing sub-aspects
                aspect_document = _publicize(
                    _Document,
                    paragraphs=aspect.reference_paragraphs,
                )
                aspect_document.add_aspects(aspect.aspects)
                await self.extract_aspects_from_document_async(
                    **extract_sub_aspects_kwargs, document=aspect_document
                )
                # Overwrite the sub-aspects with the newly extracted ones,
                # as the sub-aspects were deep-copied when attached to
                # the aspect document.
                aspect.aspects = aspect_document.aspects

        return document_aspects

    def extract_concepts_from_aspect(
        self,
        aspect: _Aspect,
        document: _Document,
        from_concepts: Sequence[_Concept]
        | None = None,  # using Sequence type with list validator for type checking
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> list[_Concept]:
        """
        Extracts concepts associated with a given aspect in a document.

        This method processes an aspect to extract related concepts using LLMs.
        If the aspect has not been previously processed, a ValueError is raised.

        This is the synchronous version of `extract_concepts_from_aspect_async()`.

        :param aspect: The aspect from which to extract concepts.
        :type aspect: _Aspect
        :param document: The document that contains the aspect.
        :type document: _Document
        :param from_concepts: List of existing concepts to process. Defaults to None.
        :type from_concepts: list[_Concept] | None
        :param overwrite_existing: Whether to overwrite already processed concepts with newly
            extracted information. Defaults to False.
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum number of items with the same extraction params to process
            in each LLM call. Defaults to 0 (all items in one call). If concurrency is enabled,
            defaults to 1. For complex tasks, you should not set a high value, in order to avoid
            prompt overloading.
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to include in a
            single LLM prompt. Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed concept objects.
        :rtype: list[_Concept]
        """
        return _run_sync(
            self.extract_concepts_from_aspect_async(
                aspect=aspect,
                document=document,
                from_concepts=from_concepts,
                overwrite_existing=overwrite_existing,
                max_items_per_call=max_items_per_call,
                use_concurrency=use_concurrency,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                raise_exception_on_extraction_error=raise_exception_on_extraction_error,
            )
        )

    @_timer_decorator(process_name="Concept extraction from aspect")
    async def extract_concepts_from_aspect_async(
        self,
        aspect: _Aspect,
        document: _Document,
        from_concepts: Sequence[_Concept]
        | None = None,  # using Sequence type with list validator for type checking
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> list[_Concept]:
        """
        Asynchronously extracts concepts from a specified aspect using LLMs.

        This method processes an aspect to extract related concepts using LLMs.
        If the aspect has not been previously processed, a ValueError is raised.

        :param aspect: The aspect from which to extract concepts.
        :type aspect: _Aspect
        :param document: The document that contains the aspect.
        :type document: _Document
        :param from_concepts: List of existing concepts to process. Defaults to None.
        :type from_concepts: list[_Concept] | None
        :param overwrite_existing: Whether to overwrite already processed concepts with newly
            extracted information. Defaults to False.
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum number of items with the same extraction params to process
            in each LLM call. Defaults to 0 (all items in one call). If concurrency is enabled,
            defaults to 1. For complex tasks, you should not set a high value, in order to avoid
            prompt overloading.
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to include in a
            single LLM prompt. Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed concept objects.
        :rtype: list[_Concept]
        """

        if from_concepts is not None:
            from_concepts = _validate_sequence_is_list(from_concepts)

        self._check_instances_and_llm_params(
            target=aspect,
            llm_or_group=self,
            instances_to_process=from_concepts,
            instance_type="concept",
            overwrite_existing=overwrite_existing,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        if not aspect._is_processed:
            if aspect.extracted_items:
                raise RuntimeError(
                    "Aspect is not marked as processed, yet it has extracted items."
                )
            raise ValueError(
                f"Aspect `{aspect.name}` is not yet processed. "
                f"Use `extract_aspects_from_document` first.`"
            )

        extract_instances_kwargs = {
            "context": aspect,
            "instance_type": "concept",
            "document": document,
            "from_instances": from_concepts,
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }

        if self.is_group:
            # Safe cast: self is a _DocumentLLMGroup because `is_group` is True
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in cast("_DocumentLLMGroup", self).llms
            ]
            # Safe cast: List comprehension creates tuples of (async_method, kwargs)
            # Type checker needs help to recognize this matches AsyncCalsAndKwargs format
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            # Safe cast: self is a _DocumentLLM because `is_group` is False
            await self._extract_instances(
                **extract_instances_kwargs, llm=cast("_DocumentLLM", self)
            )

        # Extract concepts from sub-aspects
        extract_concepts_from_sub_aspects_kwargs = {
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }
        if aspect.aspects:
            # Validate proper nesting level of sub-aspects
            self._validate_sub_aspects_nesting_level(aspect)
            logger.info(
                f"Extracting concepts from sub-aspects for aspect `{aspect.name}`"
            )
            for sub_aspect in aspect.aspects:
                if not sub_aspect.reference_paragraphs:
                    logger.info(
                        f"Sub-aspect `{sub_aspect.name}` has no paragraphs. "
                        f"Concepts will not be extracted."
                    )
                    continue
                # Treat an aspect as a document containing sub-aspects
                sub_aspect_document = _publicize(
                    _Document,
                    paragraphs=sub_aspect.reference_paragraphs,
                )
                sub_aspect_document.add_aspects([sub_aspect])
                sub_aspect_concepts = await self.extract_concepts_from_aspect_async(
                    **extract_concepts_from_sub_aspects_kwargs,
                    aspect=sub_aspect,
                    document=sub_aspect_document,
                )
                # Overwrite the sub-aspects with the newly extracted ones,
                # as the sub-aspects were deep-copied when attached to
                # the sub-aspect document.
                sub_aspect.concepts = sub_aspect_concepts

        # Explicitly return list of concepts for type checking
        return list(from_concepts) if from_concepts else list(aspect.concepts)

    def extract_concepts_from_document(
        self,
        document: _Document,
        from_concepts: Sequence[_Concept]
        | None = None,  # using Sequence type with list validator for type checking
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> list[_Concept]:
        """
        Extracts concepts from the provided document using predefined LLMs.

        This is the synchronous version of `extract_concepts_from_document_async()`.

        :param document: The document from which concepts are to be extracted.
        :type document: _Document
        :param from_concepts: Existing concepts to use as a base for extraction. If None, uses all
            document's concepts.
        :type from_concepts: list[_Concept] | None
        :param overwrite_existing: Whether to overwrite already processed concepts with
            newly extracted information. Defaults to False.
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum items with the same extraction params to process per LLM call.
            Defaults to 0 (all items in single call). For complex tasks, you should not set a value, to avoid
            prompt overloading. If concurrency is enabled, defaults to 1 (each item processed separately).
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM prompt.
            Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images).
        :type max_images_to_analyze_per_call: int, optional
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed Concept objects with extracted items.
        :rtype: list[_Concept]
        """
        return _run_sync(
            self.extract_concepts_from_document_async(
                document=document,
                from_concepts=from_concepts,
                overwrite_existing=overwrite_existing,
                max_items_per_call=max_items_per_call,
                use_concurrency=use_concurrency,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                max_images_to_analyze_per_call=max_images_to_analyze_per_call,
                raise_exception_on_extraction_error=raise_exception_on_extraction_error,
            )
        )

    @_timer_decorator(process_name="Concepts extraction from document")
    async def extract_concepts_from_document_async(
        self,
        document: _Document,
        from_concepts: Sequence[_Concept]
        | None = None,  # using Sequence type with list validator for type checking
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> list[_Concept]:
        """
        Extracts concepts from the provided document using predefined LLMs asynchronously.

        This method processes a document to extract concepts using configured LLMs.

        :param document: The document from which concepts are to be extracted.
        :type document: _Document
        :param from_concepts: Existing concepts to use as a base for extraction. If None, uses all
            document's concepts.
        :type from_concepts: list[_Concept] | None
        :param overwrite_existing: Whether to overwrite already processed concepts with
            newly extracted information. Defaults to False.
            Defaults to False.
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum number of items with the same extraction params to process
            per LLM call. Defaults to 0 (all items in one call). If concurrency is enabled,
            defaults to 1. For complex tasks, you should not set a high value, in order to avoid
            prompt overloading.
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM prompt.
            Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images).
        :type max_images_to_analyze_per_call: int, optional
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed Concept objects with extracted items.
        :rtype: list[_Concept]
        """

        if from_concepts is not None:
            from_concepts = _validate_sequence_is_list(from_concepts)

        self._check_instances_and_llm_params(
            target=document,
            llm_or_group=self,
            instances_to_process=from_concepts,
            instance_type="concept",
            overwrite_existing=overwrite_existing,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        extract_instances_kwargs = {
            "context": document,
            "instance_type": "concept",
            "document": document,
            "from_instances": from_concepts,
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "max_images_to_analyze_per_call": max_images_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }

        if self.is_group:
            # Safe cast: self is a _DocumentLLMGroup because `is_group` is True
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in cast("_DocumentLLMGroup", self).llms
            ]
            # Safe cast: List comprehension creates tuples of (async_method, kwargs)
            # Type checker needs help to recognize this matches AsyncCalsAndKwargs format
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            # Safe cast: self is a _DocumentLLM because `is_group` is False
            await self._extract_instances(
                **extract_instances_kwargs, llm=cast("_DocumentLLM", self)
            )

        # Explicitly return list of concepts for type checking
        return list(from_concepts) if from_concepts else list(document.concepts)

    def _check_llm_roles_before_extract_all(
        self,
        document: _Document,
    ) -> None:
        """
        Checks if all assigned LLM roles in the given document are present in the set
        of LLM roles of the current LLM / LLM group instance. If there are missing roles,
        a warning is logged. This process helps to check for completeness in data extraction
        when full extraction method extract_all() is called.

        :param document: The document object to check for LLM role assignments.
        :type document: _Document
        :return: None
        :rtype: None
        """

        if self.is_group:
            # Safe cast: self is a _DocumentLLMGroup because `is_group` is True
            llm_roles = {i.role for i in cast("_DocumentLLMGroup", self).llms}
        else:
            # Safe cast: self is a _DocumentLLM because `is_group` is False
            llm_roles = {cast("_DocumentLLM", self).role}
        missing_llm_roles = document.llm_roles.difference(llm_roles)
        if missing_llm_roles:
            warnings.warn(
                f"Document contains elements with LLM roles that are not found "
                f"in the current {'LLM group' if self.is_group else 'LLM'}: "
                f"{'LLM group roles' if self.is_group else 'LLM role'} {llm_roles}, "
                f"missing LLM roles {missing_llm_roles}. "
                f"Such elements will be ignored.",
                stacklevel=2,
            )

    def _check_instances_and_llm_params(
        self,
        target: _Document | _Aspect,  # type: ignore
        llm_or_group: _GenericLLMProcessor,
        instances_to_process: Sequence[_Aspect]
        | Sequence[_Concept]
        | None,  # using Sequence type with list validator for type checking
        instance_type: ExtractedInstanceType,
        overwrite_existing: bool = False,
    ) -> None:
        """
        Validates instances and LLM parameters, ensuring compatibility with the target
        and configurations provided.

        :param target: The target object, which should have an attribute corresponding
            to `instance_type` (e.g., 'aspects' for 'aspect', etc.). Expected to be an
            instance of either _Document or _Aspect.
        :param llm_or_group: The LLM or an LLM group to be validated. This may either
            be a standalone DocumentLLM or a DocumentLLMGroup to ensure compatibility.
        :param instances_to_process: A list of instances to process, which must match
            the specified instance-type (`aspect` or `concept`). If not provided,
            defaults to instances present in the target attribute.
        :param instance_type: Specifies the type of instances to validate ('aspect' or
            'concept'). This determines the expected type for validation.
        :param overwrite_existing: A flag indicating whether to overwrite the states of
            the existing processed instances. Defaults to False.

        :return: None. The function performs validation and raises errors in case of
            mismatches or invalid configurations.
        """

        if instances_to_process is not None:
            instances_to_process = _validate_sequence_is_list(instances_to_process)

        # Check instances
        instance_class_map = {
            "aspect": _Aspect,
            "concept": _Concept,
        }
        # Retrieve class or raise error
        instance_class = instance_class_map.get(instance_type)
        if not instance_class:
            raise ValueError(f"Unsupported instance_type '{instance_type}'.")
        # Check the target attribute
        if not hasattr(target, f"{instance_type}s"):
            raise AttributeError(
                f"Target object must have an attribute named '{instance_type}s'."
            )
        # Get instances to process, defaulting to target
        check_instances = (
            instances_to_process
            if instances_to_process is not None
            else getattr(target, f"{instance_type}s")
        )
        # Validate retrieved instances
        if not all(isinstance(i, instance_class) for i in check_instances):
            raise ValueError(
                f"All instances must be of type {instance_class.__name__}."
            )
        if not all(i in getattr(target, f"{instance_type}s") for i in check_instances):
            raise ValueError(
                f"All instances must be present in the target attribute "
                f"'{instance_type}s'."
            )
        self._check_instances_already_processed(
            instance_type=instance_type,
            instances=check_instances,
            overwrite_existing=overwrite_existing,
        )

        # Check DocumentLLMGroup
        if llm_or_group.is_group:
            # Safe cast: llm_or_group is a _DocumentLLMGroup because `is_group` is True
            llm_group = cast("_DocumentLLMGroup", llm_or_group)
            if not llm_group.llms:
                raise ValueError(
                    "The provided DocumentLLMGroup does not contain any defined LLMs."
                )

        # Check DocumentLLM
        else:
            # Safe cast: llm_or_group is a _DocumentLLM because `is_group` is False
            llm = cast("_DocumentLLM", llm_or_group)
            # Inform about inconsistent LLM roles
            if any(i.llm_role != llm.role for i in check_instances):
                logger.warning(
                    f"Some {instance_type}s rely on the LLM with a role different "
                    f"than the current LLM's role `{llm.role}`. "
                    f"This LLM will not extract such {instance_type}s."
                )

    @staticmethod
    def _check_instances_already_processed(
        instance_type: ExtractedInstanceType,
        instances: list[_Aspect] | list[_Concept],
        overwrite_existing: bool,
    ) -> None:
        """
        Checks whether the given instances of a specified type have already been processed.

        :param instance_type: The type of instances being checked.
        :param instances: A list of instances to be evaluated for processing status.
        :param overwrite_existing: Specifies whether to overwrite already processed instances.
        :return: None
        :raises ValueError: If `overwrite_existing` is False and one or more instances
            have already been processed.
        """
        if not overwrite_existing:
            already_processed_names = []
            for i in instances:
                if i._is_processed:
                    already_processed_names.append(i.name)
            if already_processed_names:
                raise ValueError(
                    f"Some {instance_type}s have already been processed: {already_processed_names}."
                    "Set `overwrite_existing=True` to overwrite them."
                )

    @staticmethod
    def _prepare_message_kwargs_list(
        extracted_instance_type: ExtractedInstanceType,
        source: _Document | _Aspect,  # type: ignore
        llm: _DocumentLLM,
        instances_to_process: list[_Aspect] | list[_Concept],
        document: _Document,
        add_justifications: bool = False,
        justification_depth: JustificationDepth = "brief",
        justification_max_sents: int = 2,
        add_references: bool = False,
        reference_depth: ReferenceDepth = "paragraphs",
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> list[dict[str, dict | list[_Paragraph] | list[_Image]]]:
        """
        Prepares the list of the message kwargs required for querying a LLM to extract aspects or concepts
        from a document or its associated entities like aspects.

        Each item on the list is a kwargs dict for each extraction, based on specific context chunks.
        E.g. if the text has 60 paragraphs, and `max_paragraphs_to_analyze_per_call` is set to 15, then
        each extraction type that uses all document paragraphs as context will have 4 items of kwargs
        dicts in the list, and the context of each message will include 15 paragraphs.

        :param extracted_instance_type: A string literal indicating the type of extracted instance.
            Accepted values are "aspect" and "concept".
        :param source: The input source, which can either be a `_Document` or an `_Aspect`
            instance, from which aspects or concepts are extracted.
        :param llm: The LLM instance used for extraction.
        :param instances_to_process: A list of instances (either `_Aspect` or `_Concept` instances)
            to process for extraction tasks.
        :param document: An instance of `_Document` containing paragraphs or text or images,
            which is used to provide additional context for the extraction process.
        :param add_justifications: A boolean flag. When set to `True`,
            justification for the extracted items is included in extraction result.
        :param justification_depth: The level of detail of justifications. Details to "brief".
        :param justification_max_sents: The maximum number of sentences in a justification.
            Defaults to 2.
        :param add_references: A boolean flag. When `True`, and if applicable,
            references aiding the extraction task are included in the extraction result.
        :param reference_depth: The structural depth of the references, i.e. whether to provide
            paragraphs as references or sentences as references. Defaults to "paragraphs".
            ``extracted_items`` will have values based on this parameter.
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the paragraphs are analyzed.
        :param max_images_to_analyze_per_call: The maximum number of images to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the images are analyzed.
        :return: A list of dictionaries containing prompt kwargs and context chunks for LLM queries.
        :rtype: list[dict[str, dict | list[_Paragraph] | list[_Image]]]
        """

        # Validate source data for extraction and set the extraction level
        def validate_source() -> str:
            """
            Validates the source based on the extracted instance type.
            """

            # Aspect (document-level)
            if extracted_instance_type == "aspect":
                if not isinstance(source, _Document):
                    raise ValueError(
                        "Aspect extraction is only supported for Document sources."
                    )
                if not (source.raw_text or source.paragraphs):
                    raise ValueError(
                        "Document lacks text or paragraphs for aspect extraction."
                    )
                return "aspect_document_text"

            # Concept (document- and aspect-levels)
            if extracted_instance_type == "concept":
                # Concept (document-level)
                if isinstance(source, _Document):
                    if llm.role.endswith("_text") and not (
                        source.raw_text or source.paragraphs
                    ):
                        raise ValueError(
                            "Document lacks text or paragraphs for concept extraction."
                        )
                    if llm.role.endswith("_vision") and not source.images:
                        raise ValueError("No images attached to the document.")
                    if llm.role.endswith("_vision") and add_references:
                        raise ValueError(
                            "Reference paragraphs are not supported for vision concepts."
                        )
                    return (
                        "concept_document_text"
                        if llm.role.endswith("_text")
                        else "concept_document_vision"
                    )
                # Concept (aspect-level)
                if isinstance(source, _Aspect):
                    return "concept_aspect_text"

            raise ValueError(
                f"Unsupported extracted item type: `{extracted_instance_type}`"
            )

        extraction_level = validate_source()
        message_kwargs_list = []

        # Text-based extraction
        if extraction_level.endswith("_text"):
            # Safe cast: source for the other extraction level is Aspect
            paragraphs = (
                document.paragraphs
                if extraction_level in {"aspect_document_text", "concept_document_text"}
                else cast(_Aspect, source).reference_paragraphs
            )
            if not paragraphs:
                raise ValueError("Context lacks paragraphs for text-based extraction.")
            max_paras_per_call = (
                min(len(paragraphs), max_paragraphs_to_analyze_per_call)
                if max_paragraphs_to_analyze_per_call
                else len(paragraphs)
            )
            paragraphs_chunks: list[list[_Paragraph]] = _chunk_list(
                paragraphs, max_paras_per_call
            )
            logger.debug(f"Processing {max_paras_per_call} paragraphs per LLM call.")

            for paragraphs_chunk in paragraphs_chunks:
                if not paragraphs_chunk:
                    raise RuntimeError("Paragraphs chunk cannot be empty.")

                # Aspect (document-level)
                if extraction_level == "aspect_document_text":
                    prompt_kwargs = {
                        "paragraphs": paragraphs_chunk,
                        "aspects": instances_to_process,
                        "add_justifications": add_justifications,
                        "justification_depth": justification_depth,
                        "justification_max_sents": justification_max_sents,
                        # For aspects, references will be populated from extracted items automatically
                        "reference_depth": reference_depth,
                        "output_language": llm.output_language,
                        "supports_reasoning": llm._supports_reasoning,
                    }
                    if any(p.additional_context for p in paragraphs_chunk) or any(
                        s.additional_context
                        for p in paragraphs_chunk
                        for s in p.sentences
                    ):
                        # Add guidance that the paragraphs or sentences have additional context,
                        # such as formatting, list context, or table position.
                        prompt_kwargs["additional_context_for_paras_or_sents"] = True
                    if all(p._md_text for p in paragraphs_chunk):
                        # Add guidance that the paragraphs are in markdown format, e.g. when document
                        # was converted from DOCX using DocxConverter in markdown mode.
                        prompt_kwargs["is_markdown"] = True

                # Concept (document- and aspect-levels)
                elif extraction_level in {
                    "concept_document_text",
                    "concept_aspect_text",
                }:
                    prompt_kwargs = {
                        "concepts": instances_to_process,
                        "add_justifications": add_justifications,
                        "justification_depth": justification_depth,
                        "justification_max_sents": justification_max_sents,
                        "data_type": "text",
                        "add_references": add_references,
                        "reference_depth": reference_depth,
                        "output_language": llm.output_language,
                        "supports_reasoning": llm._supports_reasoning,
                    }
                    if add_references:
                        # List of document/aspect paragraphs used for concept extraction with references
                        prompt_kwargs["paragraphs"] = paragraphs_chunk
                        if any(p.additional_context for p in paragraphs_chunk) or any(
                            s.additional_context
                            for p in paragraphs_chunk
                            for s in p.sentences
                        ):
                            # Add guidance that the paragraphs or sentences have additional context,
                            # such as formatting, list context, or table position.
                            prompt_kwargs["additional_context_for_paras_or_sents"] = (
                                True
                            )
                        if all(p._md_text for p in paragraphs_chunk):
                            # Add guidance that the paragraphs are in markdown format, e.g. when document
                            # was converted from DOCX using DocxConverter in markdown mode.
                            prompt_kwargs["is_markdown"] = True
                    else:
                        # Raw text of document/aspect used for concept extraction
                        if isinstance(source, _Document) and len(
                            paragraphs_chunk
                        ) == len(source.paragraphs):
                            # If the document is being processed as a whole, use _md_text of the document,
                            # if available (e.g. after being converted from DOCX using DocxConverter in markdown mode),
                            # otherwise use raw_text.
                            if source._md_text:
                                text = source._md_text
                                prompt_kwargs["is_markdown"] = True
                            else:
                                text = source.raw_text
                            if text is None:
                                raise ValueError(
                                    "Document lacks text for concept extraction."
                                )
                            prompt_kwargs["text"] = _clean_text_for_llm_prompt(
                                text,
                                preserve_linebreaks=True,
                                strip_text=True,
                            )  # markdown or raw text of the document
                        else:
                            # If an aspect is being processed, or if the document is being processed in chunks,
                            # use _md_text of the paragraphs, if available (e.g. when document was converted
                            # from DOCX using DocxConverter in markdown mode), otherwise use raw_text.
                            if all(p._md_text for p in paragraphs_chunk):
                                prompt_kwargs["text"] = "\n\n".join(
                                    [
                                        _clean_text_for_llm_prompt(
                                            p._md_text,  # type: ignore - p._md_text is checked above to not be None
                                            preserve_linebreaks=True,  # preserve markdown markers etc.
                                            strip_text=False,  # preserve markdown list indentation etc.
                                        )
                                        for p in paragraphs_chunk
                                    ]
                                )
                                prompt_kwargs["is_markdown"] = True
                            else:
                                prompt_kwargs["text"] = "\n\n".join(
                                    [
                                        _clean_text_for_llm_prompt(
                                            p.raw_text,
                                            preserve_linebreaks=False,
                                            strip_text=True,
                                        )
                                        for p in paragraphs_chunk
                                    ]
                                )

                else:
                    raise ValueError(
                        f"Unsupported extraction level for text-based extraction: `{extraction_level}`"
                    )

                message_kwargs = {
                    "prompt_kwargs": prompt_kwargs,
                    "paragraphs_chunk": paragraphs_chunk,
                }
                message_kwargs_list.append(message_kwargs)

        # Images-based extraction
        elif extraction_level.endswith("_vision"):
            # Safe cast: source for extracting data from images is always a _Document
            source = cast(_Document, source)
            max_images_per_call = (
                min(len(source.images), max_images_to_analyze_per_call)
                if max_images_to_analyze_per_call
                else len(source.images)
            )
            images_chunks = _chunk_list(source.images, max_images_per_call)
            logger.debug(f"Processing {max_images_per_call} images per LLM call.")

            for images_chunk in images_chunks:
                if not images_chunk:
                    raise RuntimeError("Images chunk cannot be empty.")

                # Concept (from images)
                if extraction_level == "concept_document_vision":
                    prompt_kwargs = {
                        "concepts": instances_to_process,
                        "add_justifications": add_justifications,
                        "justification_depth": justification_depth,
                        "justification_max_sents": justification_max_sents,
                        "data_type": "image",
                        "output_language": llm.output_language,
                        "supports_reasoning": llm._supports_reasoning,
                    }
                    message_kwargs = {
                        "prompt_kwargs": prompt_kwargs,
                        "images": images_chunk,
                    }

                else:
                    raise ValueError(
                        f"Unsupported extraction level for vision extraction: `{extraction_level}`"
                    )

                message_kwargs_list.append(message_kwargs)

        return message_kwargs_list

    async def _extract_items_from_instances(
        self,
        extracted_item_type: ExtractedInstanceType,
        source: _Document | _Aspect,  # type: ignore
        llm: _DocumentLLM,
        instances_to_process: list[_Aspect] | list[_Concept],
        document: _Document,
        overwrite_existing: bool = False,
        add_justifications: bool = False,
        justification_depth: JustificationDepth = "brief",
        justification_max_sents: int = 2,
        add_references: bool = False,
        reference_depth: ReferenceDepth = "paragraphs",
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        async_limiter: AsyncLimiter | None = None,
        raise_exception_on_extraction_error: bool = True,
    ) -> tuple[list[_Aspect] | list[_Concept] | None, _LLMUsage]:
        """
        Extracts items from either aspects or concepts using a specified LLM.
        This unified method handles extraction from both documents and aspects,
        supporting both text and vision LLMs.

        :param extracted_item_type: Type of the item(s) being extracted ("aspect" or "concept")
        :type extracted_item_type: ExtractedInstanceType
        :param source: The source to extract from (_Document or _Aspect)
        :type source: _Document | _Aspect
        :param llm: The LLM used for extraction
        :type llm: DocumentLLM
        :param instances_to_process: List of aspects or concepts to process
        :type instances_to_process: list[_Aspect] | list[_Concept]
        :param document: The document containing the source.
        :type document: _Document | None
        :param add_justifications: Whether to include justification for extracted items.
            Defaults to False.
        :type add_justifications: bool
        :param justification_depth: The level of detail of justifications. Details to "brief".
        :type justification_depth: JustificationDepth
        :param justification_max_sents: The maximum number of sentences in a justification.
            Defaults to 2.
        :type justification_max_sents: int
        :param add_references: Whether to include references for extracted items.
            Defaults to False.
        :type add_references: bool
        :param reference_depth: The structural depth of the references, i.e. whether to provide
            paragraphs as references or sentences as references. Defaults to "paragraphs".
            ``extracted_items`` will have values based on this parameter.
        :type reference_depth: ReferenceDepth
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the paragraphs are analyzed.
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: The maximum number of images to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the images are analyzed.
        :type max_images_to_analyze_per_call: int
        :param num_retries_failed_request: Optional number of retries when LLM request fails. Defaults to 3.
        :type num_retries_failed_request: int
        :param max_retries_failed_request: Specific to certain provider APIs (e.g. OpenAI). Optional number of
            retries when LLM request fails. Defaults to 0.
        :type max_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that controls the frequency of
            async LLM API requests, when concurrency is enabled for certain tasks. If not provided,
            such requests will be sent synchronously.
        :type async_limiter: AsyncLimiter | None
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional

        :return: A tuple containing:
            (0) List of processed instances with extracted items, or None if LLM processing fails
            (1) _LLMUsage instance with LLM usage information
        :rtype: tuple[list[_Aspect] | list[_Concept] | None, _LLMUsage]
        """

        def validate_source_in_document(
            source: _Document | _Aspect,  # type: ignore
            document: _Document,
        ) -> None:
            """
            Raises ValueError if an Aspect is not assigned to the given Document.
            """
            if isinstance(source, _Aspect) and source not in document.aspects:
                raise ValueError(
                    f"Aspect `{source.name}` must be assigned to the document"
                )

        def validate_vision_llm_usage(
            extracted_item_type: ExtractedInstanceType,
            llm: _DocumentLLM,
            source: _Document | _Aspect,  # type: ignore
            add_references: bool,
        ) -> None:
            """
            Raises ValueError if the LLM role is vision-based but the extraction type/source is unsupported.
            """
            if llm.role.endswith("_vision"):
                # Vision LLM: only document-level concept extraction is valid
                if extracted_item_type == "aspect" or (
                    extracted_item_type == "concept" and isinstance(source, _Aspect)
                ):
                    raise ValueError(
                        f"{extracted_item_type.capitalize()} extraction using vision LLMs is not supported. "
                        "Vision LLMs can be used only for document concept extraction."
                    )
                if extracted_item_type == "concept" and add_references:
                    raise ValueError(
                        "Reference paragraphs are not supported for vision concepts."
                    )

        def validate_aspect_for_concept_extraction(
            extracted_item_type: ExtractedInstanceType,
            source: _Document | _Aspect,  # type: ignore
        ) -> None:
            """
            If extracting concepts from an Aspect, validates that the aspect's extracted items have
            references.
            """
            if (
                extracted_item_type == "concept"
                and isinstance(source, _Aspect)
                and (
                    source.extracted_items
                    and (
                        not source.reference_paragraphs
                        or (
                            source.reference_depth
                            == "sentences"  # aspects do not have "add_references" attr
                            and not all(
                                p.sentences for p in source.reference_paragraphs
                            )
                        )
                    )
                )
            ):
                raise ValueError(
                    f"Aspect `{source.name}` has extracted items but no references"
                )

        # Perform validations
        validate_source_in_document(source=source, document=document)
        validate_vision_llm_usage(
            extracted_item_type=extracted_item_type,
            llm=llm,
            source=source,
            add_references=add_references,
        )
        validate_aspect_for_concept_extraction(
            extracted_item_type=extracted_item_type, source=source
        )

        # Skip processing if the aspect context is empty for aspect concept extraction
        if (
            extracted_item_type == "concept"
            and isinstance(source, _Aspect)
            and not source.extracted_items
        ):
            logger.info(
                f"Aspect `{source.name}` does not have any extracted items to extract concepts from"
            )
            return [], _LLMUsage()

        # Handle empty instances list
        if not instances_to_process:
            logger.info(
                f"No {extracted_item_type}s to process with LLM role `{llm.role}`"
            )
            return [], _LLMUsage()

        # Prepare message kwargs list based on extraction params
        message_kwargs_list = self._prepare_message_kwargs_list(
            extracted_instance_type=extracted_item_type,
            source=source,
            llm=llm,
            instances_to_process=instances_to_process,
            document=document,
            add_justifications=add_justifications,
            justification_depth=justification_depth,
            justification_max_sents=justification_max_sents,
            add_references=add_references,
            reference_depth=reference_depth,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            max_images_to_analyze_per_call=max_images_to_analyze_per_call,
        )

        def merge_usage_data(existing: _LLMUsage | None, new: _LLMUsage) -> _LLMUsage:
            """
            Aggregates usage data from all LLM calls.
            Returns:
                The latest aggregated usage data from all LLM calls.
            """
            if existing is None:
                return new
            existing.input += new.input
            existing.output += new.output
            existing.calls += new.calls
            return existing

        all_usage_data: _LLMUsage | None = None
        sources_mapper: dict[
            str, dict[str, _Aspect | _Concept | list[_ExtractedItem]]  # type: ignore
        ] = {}
        instances_enumerated = dict(enumerate(instances_to_process))

        # Each item on the list is a message kwargs dict for each extraction, based on specific context chunks.
        for idx, message_kwargs in enumerate(message_kwargs_list):
            logger.debug(
                f"Processing messages chunk {idx + 1}/{len(message_kwargs_list)}"
            )
            # Extract paragraphs and remove them from message_kwargs
            # Safe cast: "paragraphs_chunk" key, when present, always contains list[_Paragraph]
            paragraphs_chunk = cast(
                list["_Paragraph"] | None, message_kwargs.pop("paragraphs_chunk", None)
            )
            if paragraphs_chunk is not None:
                paragraphs_enumerated = dict(enumerate(paragraphs_chunk))
            else:
                paragraphs_enumerated = {}

            # Skip instances that have a single occurrence and already have extracted items
            discarded_instances = []
            if extracted_item_type == "concept":
                filtered_instances_to_process = []
                for i in instances_to_process:
                    i_in_source_mapper = sources_mapper.get(i.unique_id, None)
                    if (
                        i_in_source_mapper
                        # Safe cast: i is always a Concept because extracted_item_type is "concept"
                        and cast(_Concept, i).singular_occurrence
                        and i_in_source_mapper["extracted_items"]
                    ):
                        discarded_instances.append(i)
                        logger.debug(
                            f"Skipping {extracted_item_type} `{i.name}` because it has "
                            "singular occurrence enforced and already has extracted items"
                        )
                        continue
                    filtered_instances_to_process.append(i)
                if not filtered_instances_to_process:
                    logger.debug(
                        f"No {extracted_item_type}s left to process with LLM role `{llm.role}`"
                    )
                    continue
                # Do not include instances with a single occurrence that already have extracted items
                # in the prompt kwargs
                # Safe cast: "prompt_kwargs" key always contains a dict
                cast(dict, message_kwargs["prompt_kwargs"])["concepts"] = (
                    filtered_instances_to_process
                )
                if any(i in filtered_instances_to_process for i in discarded_instances):
                    raise RuntimeError(
                        "Discarded instances found in filtered instances list."
                    )
                logger.debug(
                    f"Total {extracted_item_type}s discarded: {len(discarded_instances)}"
                )
                # Re-enumerate the new instances to process, to match them against the indices in LLM responses
                instances_enumerated = dict(enumerate(filtered_instances_to_process))
                # Do not overwrite instances_to_process as we need to return all of the originally passed instances
                # even if they are discarded during one of the loop iterations

            # Construct a prompt from the message kwargs
            # Safe cast: "prompt_kwargs" key always contains a dict
            if extracted_item_type == "aspect":
                message = llm._extract_aspect_items_prompt.render(
                    **cast(dict, message_kwargs["prompt_kwargs"])
                )
            elif extracted_item_type == "concept":
                message = llm._extract_concept_items_prompt.render(
                    **cast(dict, message_kwargs["prompt_kwargs"])
                )
            else:
                raise ValueError(
                    f"Unsupported extracted item type: `{extracted_item_type}`"
                )

            # Safe cast: message_kwargs is always a dict
            cast(dict, message_kwargs)["message"] = message
            # Initialize the LLM call object to pass to the LLM query method
            cast(dict, message_kwargs)["llm_call_obj"] = _LLMCall(
                prompt_kwargs=cast(dict, message_kwargs["prompt_kwargs"]),
                prompt=message,
            )
            del message_kwargs["prompt_kwargs"]

            # Query LLM, process and validate results
            extracted_data, usage_data = await llm._query_llm(
                **cast(dict, message_kwargs),
                num_retries_failed_request=num_retries_failed_request,
                max_retries_failed_request=max_retries_failed_request,
                async_limiter=async_limiter,
                raise_exception_on_llm_api_error=raise_exception_on_extraction_error,
            )
            all_usage_data = merge_usage_data(all_usage_data, usage_data)
            extracted_data = _validate_parsed_llm_output(
                _parse_llm_output_as_json(
                    _remove_thinking_content_from_llm_output(extracted_data)
                ),
                extracted_item_type=extracted_item_type,
                justification_provided=add_justifications,
                references_provided=add_references,
                reference_depth=reference_depth,
            )
            if extracted_data is None:
                logger.error(
                    f"LLM did not return valid JSON for {extracted_item_type}s"
                )
                return None, all_usage_data

            # Aspect (document-level)
            if extracted_item_type == "aspect":
                for aspect_dict in extracted_data:
                    try:
                        relevant_aspect = instances_enumerated[
                            int(aspect_dict["aspect_id"].lstrip("A"))
                        ]
                    except (KeyError, ValueError):
                        logger.error("Aspect ID returned by LLM is invalid")
                        return None, all_usage_data
                    self._check_instances_already_processed(
                        instance_type=extracted_item_type,
                        # Safe cast: relevant_aspect is always an _Aspect instance
                        # because extracted_item_type is "aspect"
                        instances=[cast(_Aspect, relevant_aspect)],
                        overwrite_existing=overwrite_existing,
                    )
                    if relevant_aspect.unique_id not in sources_mapper:
                        sources_mapper[relevant_aspect.unique_id] = {
                            "source": relevant_aspect,
                            "extracted_items": [],
                        }
                    if add_justifications or reference_depth == "sentences":
                        # References are automatically included for aspects, as paragraphs/sents'
                        # texts are the values of extracted items for the aspects
                        try:
                            for para_dict in aspect_dict["paragraphs"]:
                                para_dict["paragraph_id"] = int(
                                    para_dict["paragraph_id"].lstrip("P")
                                )
                        except ValueError:
                            logger.error("Paragraph ID returned by LLM is invalid")
                            return None, all_usage_data
                        for para_dict in sorted(
                            aspect_dict["paragraphs"],
                            key=lambda x: x["paragraph_id"],
                        ):
                            para_id = para_dict["paragraph_id"]
                            try:
                                ref_para = paragraphs_enumerated[para_id]
                            except KeyError:
                                logger.error("Paragraph ID returned by LLM is invalid")
                                return None, all_usage_data
                            # Reference depth - sentence
                            # Each extracted item will have the reference sentence text as value,
                            # reference paragraph in the list of reference paragraphs, and reference sentence
                            # in the list of reference sentences
                            if reference_depth == "sentences":
                                para_sents_enumerated = dict(
                                    enumerate(ref_para.sentences)
                                )
                                for sent_dict in para_dict["sentences"]:
                                    sent_dict["sentence_id"] = int(
                                        sent_dict["sentence_id"].split("-S")[-1]
                                    )
                                for sent_dict in sorted(
                                    para_dict["sentences"],
                                    key=lambda x: x["sentence_id"],
                                ):
                                    sent_id = sent_dict["sentence_id"]
                                    try:
                                        ref_sent = para_sents_enumerated[sent_id]
                                    except KeyError:
                                        logger.error(
                                            f"Sentence ID returned by LLM "
                                            f"for paragraph ID {para_id} is invalid"
                                        )
                                        return None, all_usage_data
                                    extracted_item_kwargs = {
                                        "value": ref_sent.raw_text,
                                        "custom_data": ref_sent.custom_data,
                                        # inherit custom data from sentence object
                                    }
                                    if add_justifications:
                                        extracted_item_kwargs["justification"] = (
                                            sent_dict["justification"]
                                        )
                                    extracted_item = _StringItem(
                                        **extracted_item_kwargs
                                    )
                                    extracted_item.reference_paragraphs = [ref_para]
                                    extracted_item.reference_sentences = [ref_sent]
                                    # Safe cast: "extracted_items" key always contains a list
                                    cast(
                                        list,
                                        sources_mapper[relevant_aspect.unique_id][
                                            "extracted_items"
                                        ],
                                    ).append(extracted_item)
                            # Reference depth - paragraph
                            # Each extracted item will have the reference paragraph text as value,
                            # reference paragraph in the list of reference paragraphs, and no reference sentences
                            else:
                                extracted_item = _StringItem(
                                    value=ref_para.raw_text,
                                    justification=para_dict["justification"],
                                    custom_data=ref_para.custom_data,  # inherit custom data from paragraph object
                                )
                                extracted_item.reference_paragraphs = [ref_para]
                                # Safe cast: "extracted_items" key always contains a list
                                cast(
                                    list,
                                    sources_mapper[relevant_aspect.unique_id][
                                        "extracted_items"
                                    ],
                                ).append(extracted_item)
                    else:
                        # Reference depth - paragraph
                        # Each extracted item will have the reference paragraph text as value,
                        # reference paragraph in the list of reference paragraphs, and no reference sentences
                        try:
                            aspect_dict["paragraph_ids"] = [
                                int(para_id.lstrip("P"))
                                for para_id in aspect_dict["paragraph_ids"]
                            ]
                        except ValueError:
                            logger.error("Paragraph ID returned by LLM is invalid")
                            return None, all_usage_data
                        for para_id in sorted(aspect_dict["paragraph_ids"]):
                            try:
                                ref_para = paragraphs_enumerated[para_id]
                            except KeyError:
                                logger.error("Paragraph ID returned by LLM is invalid")
                                return None, all_usage_data
                            extracted_item = _StringItem(
                                value=ref_para.raw_text,
                                custom_data=ref_para.custom_data,  # inherit custom data from paragraph object
                            )
                            extracted_item.reference_paragraphs = [ref_para]
                            # Safe cast: "extracted_items" key always contains a list
                            cast(
                                list,
                                sources_mapper[relevant_aspect.unique_id][
                                    "extracted_items"
                                ],
                            ).append(extracted_item)

            # Concept (document- and aspect-levels)
            elif extracted_item_type == "concept":
                for concept_dict in extracted_data:
                    try:
                        relevant_concept = instances_enumerated[
                            int(concept_dict["concept_id"].lstrip("C"))
                        ]
                    except (KeyError, ValueError):
                        logger.error("Concept ID returned by LLM is invalid")
                        return None, all_usage_data
                    self._check_instances_already_processed(
                        instance_type=extracted_item_type,
                        # Safe cast: relevant_concept is always a _Concept
                        # because extracted_item_type is "concept"
                        instances=[cast(_Concept, relevant_concept)],
                        overwrite_existing=overwrite_existing,
                    )
                    if relevant_concept.unique_id not in sources_mapper:
                        sources_mapper[relevant_concept.unique_id] = {
                            "source": relevant_concept,
                            "extracted_items": [],
                        }

                    if add_justifications or add_references:
                        for i in concept_dict["extracted_items"]:
                            # Process the item value with a custom function on the concept
                            try:
                                # Safe cast: relevant_concept is always a _Concept
                                # because extracted_item_type is "concept"
                                i["value"] = cast(
                                    _Concept, relevant_concept
                                )._process_item_value(i["value"])
                            except ValueError as e:
                                logger.error(
                                    f"Error processing extracted item value: {e}"
                                )
                                return None, all_usage_data
                            concept_extracted_item_kwargs = {"value": i["value"]}
                            if add_justifications:
                                concept_extracted_item_kwargs["justification"] = i[
                                    "justification"
                                ]
                            if add_references:
                                concept_extracted_item_kwargs[
                                    "reference_paragraphs"
                                ] = []
                                concept_extracted_item_kwargs[
                                    "reference_sentences"
                                ] = []
                                # Reference depth - sentence
                                # Each extracted item will have reference paragraph in the list of
                                # reference paragraphs, and reference sentence in the list of reference sentences
                                if reference_depth == "sentences":
                                    reference_paragraphs_list = i[
                                        "reference_paragraphs"
                                    ]
                                    for para_dict in reference_paragraphs_list:
                                        try:
                                            para_dict["reference_paragraph_id"] = int(
                                                para_dict[
                                                    "reference_paragraph_id"
                                                ].lstrip("P")
                                            )
                                        except ValueError:
                                            logger.error(
                                                "Reference paragraph ID returned by LLM is invalid"
                                            )
                                            return None, all_usage_data
                                        try:
                                            para_dict["reference_sentence_ids"] = [
                                                int(i.split("-S")[-1])
                                                for i in para_dict[
                                                    "reference_sentence_ids"
                                                ]
                                            ]
                                        except ValueError:
                                            logger.error(
                                                "Reference sentence ID returned by LLM is invalid"
                                            )
                                            return None, all_usage_data
                                    reference_paragraphs_list = sorted(
                                        reference_paragraphs_list,
                                        key=lambda x: x["reference_paragraph_id"],
                                    )
                                # Reference depth - paragraph
                                # Each extracted item will have reference paragraph in the list of
                                # reference paragraphs, and no reference sentences
                                else:
                                    try:
                                        reference_paragraphs_list = sorted(
                                            [
                                                int(para_id.lstrip("P"))
                                                for para_id in i[
                                                    "reference_paragraph_ids"
                                                ]
                                            ]
                                        )
                                    except ValueError:
                                        logger.error(
                                            "Reference paragraph ID returned by LLM is invalid"
                                        )
                                        return None, all_usage_data
                                # Reference depth - paragraph or sentence
                                for para_obj_or_id in reference_paragraphs_list:
                                    try:
                                        if reference_depth == "sentences":
                                            # Safe cast: para_obj_or_id is always a dict
                                            # when reference_depth is "sentences"
                                            para_id = cast(dict, para_obj_or_id)[
                                                "reference_paragraph_id"
                                            ]
                                        else:
                                            para_id = para_obj_or_id
                                        ref_para = paragraphs_enumerated[para_id]
                                    except KeyError:
                                        logger.error(
                                            "Reference paragraph ID returned by LLM is invalid"
                                        )
                                        return None, all_usage_data
                                    concept_extracted_item_kwargs[
                                        "reference_paragraphs"
                                    ].append(ref_para)
                                    # Reference depth - sentence
                                    if reference_depth == "sentences":
                                        para_sents_enumerated = dict(
                                            enumerate(ref_para.sentences)
                                        )
                                        # Safe cast: para_obj_or_id is always a dict
                                        # when reference_depth is "sentences"
                                        reference_sentence_ids = sorted(
                                            cast(dict, para_obj_or_id)[
                                                "reference_sentence_ids"
                                            ]
                                        )
                                        for sent_id in reference_sentence_ids:
                                            try:
                                                ref_sent = para_sents_enumerated[
                                                    sent_id
                                                ]
                                            except KeyError:
                                                logger.error(
                                                    f"Sentence ID returned by LLM "
                                                    f"for paragraph ID {para_id} is invalid"
                                                )
                                                return None, all_usage_data
                                            concept_extracted_item_kwargs[
                                                "reference_sentences"
                                            ].append(ref_sent)

                            reference_paragraphs = concept_extracted_item_kwargs.pop(
                                "reference_paragraphs", []
                            )
                            reference_sentences = concept_extracted_item_kwargs.pop(
                                "reference_sentences", []
                            )
                            try:
                                extracted_item = relevant_concept._item_class(
                                    **concept_extracted_item_kwargs
                                )
                            except ValueError as e:
                                logger.error(
                                    f"Error creating {relevant_concept._item_class.__name__}: {e}"
                                )
                                return None, all_usage_data
                            extracted_item.reference_paragraphs = reference_paragraphs
                            extracted_item.reference_sentences = reference_sentences
                            # Safe cast: "extracted_items" key always contains a list
                            cast(
                                list,
                                sources_mapper[relevant_concept.unique_id][
                                    "extracted_items"
                                ],
                            ).append(extracted_item)
                    else:
                        for i in concept_dict["extracted_items"]:
                            # Process the item value with a custom function on the concept
                            try:
                                # Safe cast: relevant_concept is always a _Concept
                                # because extracted_item_type is "concept"
                                i = cast(
                                    _Concept, relevant_concept
                                )._process_item_value(i)
                            except ValueError as e:
                                logger.error(
                                    f"Error processing extracted item value: {e}"
                                )
                                return None, all_usage_data
                            try:
                                extracted_item = relevant_concept._item_class(value=i)
                            except ValueError as e:
                                logger.error(
                                    f"Error creating {relevant_concept._item_class.__name__}: {e}"
                                )
                                return None, all_usage_data
                            # Safe cast: "extracted_items" key always contains a list
                            cast(
                                list,
                                sources_mapper[relevant_concept.unique_id][
                                    "extracted_items"
                                ],
                            ).append(extracted_item)

            else:
                raise ValueError(
                    f"Unsupported extracted item type: `{extracted_item_type}`"
                )

        # Process all gathered results for all processed instances
        async with (
            llm._async_lock
        ):  # ensure atomicity of the instances' state check and modification
            for _source_id, source_data in sources_mapper.items():
                # Safe cast: "source" key always contains _Aspect or _Concept
                source_instance = cast(_Aspect | _Concept, source_data["source"])  # type: ignore
                self._check_instances_already_processed(
                    instance_type=extracted_item_type,
                    instances=[source_instance],  # type: ignore
                    overwrite_existing=overwrite_existing,
                )
                # Safe cast: "extracted_items" key always contains a list
                # of _ExtractedItem instances
                source_instance.extracted_items = cast(
                    list["_ExtractedItem"], source_data["extracted_items"]
                )
                if extracted_item_type == "aspect":
                    # References are automatically included for aspects, as paragraphs/sents'
                    # texts are the values of extracted items for the aspects
                    for ei in source_instance.extracted_items:
                        # Safe cast: source_instance is always an _Aspect instance
                        # because extracted_item_type is "aspect"
                        source_instance = cast(_Aspect, source_instance)
                        # Extracted items might have overlapping references
                        ref_paras = ei.reference_paragraphs
                        for ref_para in ref_paras:
                            if ref_para not in source_instance.reference_paragraphs:
                                source_instance.reference_paragraphs += [ref_para]
                                if source_instance.reference_depth == "sentences":
                                    for ref_sent in ref_para.sentences:
                                        if (
                                            ref_sent
                                            not in source_instance.reference_sentences
                                        ):
                                            source_instance.reference_sentences += [
                                                ref_sent
                                            ]
                source_instance._is_processed = True

        return instances_to_process, all_usage_data or _LLMUsage()

    async def _extract_instances(
        self,
        context: _Document | _Aspect,  # type: ignore
        llm: _DocumentLLM,
        instance_type: ExtractedInstanceType,
        document: _Document,
        from_instances: list[_Aspect] | list[_Concept] | None = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        raise_exception_on_extraction_error: bool = True,
    ) -> None:
        """
        Extracts aspects or concepts from a context (document or aspect) using a specified LLM.

        :param context: The context (document or aspect) from which to extract instances
        :type context: _Document | _Aspect
        :param llm: The LLM used for extraction
        :type llm: DocumentLLM
        :param instance_type: Type of instance to extract ("aspect" or "concept")
        :type instance_type: ExtractedInstanceType
        :param document: The document object associated with the context.
        :type document: _Document
        :param from_instances: List of specific instances to process. If None, all instances are processed
        :type from_instances: list[_Aspect] | list[_Concept] | None
        :param overwrite_existing: If True, overwrites existing instances with newly extracted information
        :type overwrite_existing: bool
        :param max_items_per_call: Maximum number of items with the same extraction params to process per
            LLM call. Defaults to 0, in which case all the items are processed in a single call.
            (If concurrency is enabled, defaults to 1, i.e. each item is processed in a separate call.)
            For complex tasks, you should not set a high value, in order to avoid prompt overloading.
        :type max_items_per_call: int
        :param use_concurrency: If True, enables concurrent processing of multiple items.
            Concurrency can considerably reduce processing time, but may cause rate limit errors
            with LLM providers. Use this option when API rate limits allow for multiple concurrent
            requests. Defaults to False.
        :type use_concurrency: bool
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the paragraphs are analyzed.
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: The maximum number of images to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the images are analyzed.
        :type max_images_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: None
        """

        # If concurrency is enabled, recreate the async limiter as a new instance
        # for the current event loop, presuming that we do not have intersecting
        # event loops, i.e. all event loops in the framework are run sequentially.
        async_limiter = (
            AsyncLimiter(
                max_rate=llm.async_limiter.max_rate,
                time_period=llm.async_limiter.time_period,
            )
            if use_concurrency
            else None
        )

        async def run_extraction(
            instances_to_process: list[_Aspect] | list[_Concept],
            add_justifications: bool = False,
            justification_depth: JustificationDepth = "brief",
            justification_max_sents: int = 2,
            add_references: bool = False,
            reference_depth: ReferenceDepth = "paragraphs",
        ) -> None:
            """
            Utility function for extracting instances using the provided LLM.

            :param instances_to_process: List of instances to extract
            :param add_justifications: Whether to provide justification for extracted items
            :param add_references: Whether to provide references for extracted items
            :param justification_depth: The level of detail of justifications. Details to "brief".
            :param justification_max_sents: The maximum number of sentences in a justification.
                Defaults to 2.
            :param reference_depth: The structural depth of the references, i.e. whether to provide
                paragraphs as references or sentences as references. Defaults to "paragraphs".
                ``extracted_items`` will have values based on this parameter.
            :return: None
            """
            if not instances_to_process:
                return None

            if llm is None:
                raise ValueError(
                    f"No LLM with role `{instances_to_process[0].llm_role}` is defined in the group, "
                    f"while some {instance_type}s rely on such LLM."
                )

            async def retry_processing_for_result(
                llm_instance: _DocumentLLM,
                res: tuple[list[_Aspect] | list[_Concept] | None, _LLMUsage],
                instances: list[_Aspect] | list[_Concept],
                n_retries: int = 0,
                retry_is_final: bool = False,
            ) -> bool:
                """
                Checks the processed result for validity and retries it if invalid.

                :param llm_instance: The LLM instance to process the data.
                :param res: Result to check and retry if invalid.
                :param instances: List of processed instances associated with the result.
                :param n_retries: Number of retries to perform.
                :param retry_is_final: Whether the retry is final and will not be repeated by a fallback LLM.
                :return: bool: True if retry was successful, False otherwise.
                """
                if n_retries <= 0:
                    return False
                if not _llm_call_result_is_valid(res):
                    for i in range(n_retries):
                        logger.info(
                            f"Retrying {instance_type}s with invalid JSON "
                            f"({i + 1}/{n_retries})"
                        )
                        res = await self._extract_items_from_instances(
                            extracted_item_type=instance_type,
                            source=context,
                            llm=llm_instance,
                            instances_to_process=instances,
                            document=document,
                            overwrite_existing=overwrite_existing,
                            add_justifications=add_justifications,
                            justification_depth=justification_depth,
                            justification_max_sents=justification_max_sents,
                            add_references=add_references,
                            reference_depth=reference_depth,
                            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                            max_images_to_analyze_per_call=max_images_to_analyze_per_call,
                            num_retries_failed_request=0,  # do not repeat retries already performed by LiteLLM
                            max_retries_failed_request=0,
                            async_limiter=async_limiter,
                            raise_exception_on_extraction_error=raise_exception_on_extraction_error,
                        )
                        # Update usage stats and cost
                        await llm_instance._update_usage_and_cost(res)
                        if res[0] is not None:
                            break
                    if _llm_call_result_is_valid(res):
                        return True
                    else:
                        if retry_is_final:
                            instance_names = [instance.name for instance in instances]
                            error_msg = (
                                f"Some {instance_type}s could not be processed due to invalid JSON returned by LLM. "
                                f"Failed {instance_type}s: {instance_names}"
                            )
                            if raise_exception_on_extraction_error:
                                logger.error(error_msg)
                                raise LLMExtractionError(
                                    error_msg, retry_count=n_retries
                                )
                            else:
                                warning_msg = (
                                    error_msg
                                    + f" ({n_retries} retries). "
                                    + " If you want to raise an exception instead, "
                                    "set `raise_exception_on_extraction_error` to True."
                                )
                                logger.warning(warning_msg)
                                warnings.warn(
                                    warning_msg,
                                    stacklevel=2,
                                )
                        return False
                return True

            if use_concurrency:
                # default - each item in a separate call
                max_items_per_call_inner = max(1, max_items_per_call)
            else:
                # default - all items in a single call
                max_items_per_call_inner = max(0, max_items_per_call)
                max_items_per_call_inner = (
                    len(instances_to_process)
                    if max_items_per_call_inner == 0
                    else max_items_per_call_inner
                )

            # Prepare data for processing
            data_chunks = _chunk_list(instances_to_process, max_items_per_call_inner)
            base_params = {
                "extracted_item_type": instance_type,
                "source": context,
                "llm": llm,
                "document": document,
                "overwrite_existing": overwrite_existing,
                "add_justifications": add_justifications,
                "justification_depth": justification_depth,
                "justification_max_sents": justification_max_sents,
                "add_references": add_references,
                "reference_depth": reference_depth,
                "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
                "max_images_to_analyze_per_call": max_images_to_analyze_per_call,
                "num_retries_failed_request": llm.num_retries_failed_request,
                "max_retries_failed_request": llm.max_retries_failed_request,
                "async_limiter": async_limiter,
                "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
            }
            data_list = [
                {**base_params, "instances_to_process": chunk} for chunk in data_chunks
            ]

            if (
                use_concurrency and len(data_list) > 1
            ):  # Disable overhead if a single call
                # Process chunks concurrently
                results = await _async_multi_executor(
                    func=self._extract_items_from_instances,
                    data_list=data_list,
                )
                if len(results) != len(data_chunks):
                    raise RuntimeError(
                        f"Number of results ({len(results)}) does not match "
                        f"number of data chunks ({len(data_chunks)})."
                    )

                # Update usage stats and cost
                for result in results:
                    await llm._update_usage_and_cost(result)

                # Retry failed chunks if needed
                for chunk, result in zip(data_chunks, results, strict=True):
                    if not _llm_call_result_is_valid(result):
                        retry_successful = False
                        if llm.max_retries_invalid_data > 0:
                            retry_successful = await retry_processing_for_result(
                                llm_instance=llm,
                                res=result,
                                instances=chunk,
                                n_retries=llm.max_retries_invalid_data,
                                retry_is_final=not llm.fallback_llm,
                            )
                        # Retry with fallback LLM if it is provided
                        if not retry_successful and llm.fallback_llm:
                            logger.info("Trying with fallback LLM")
                            retry_successful = await retry_processing_for_result(
                                llm_instance=llm.fallback_llm,
                                res=result,
                                instances=chunk,
                                n_retries=max(
                                    llm.fallback_llm.max_retries_invalid_data, 1
                                ),  # retry with fallback LLM at least once
                                retry_is_final=True,
                            )
                        if not retry_successful:
                            if llm.max_retries_invalid_data > 0 or llm.fallback_llm:
                                if raise_exception_on_extraction_error:
                                    # Final retry was already performed, therefore an exception should have
                                    # been raised in case of extraction error such as invalid JSON.
                                    raise RuntimeError(
                                        "Extraction failed after all retries with "
                                        "`raise_exception_on_extraction_error` set to True, "
                                        "yet no exception was raised."
                                    )
                            else:
                                instance_names = [instance.name for instance in chunk]
                                error_msg = (
                                    f"Some {instance_type}s could not be processed due to invalid JSON returned by LLM. "
                                    f"Failed {instance_type}s: {instance_names}"
                                )
                                if raise_exception_on_extraction_error:
                                    raise LLMExtractionError(error_msg, retry_count=0)
                                else:
                                    warning_msg = (
                                        error_msg
                                        + " (0 retries). "
                                        + " If you want to raise an exception instead, "
                                        "set `raise_exception_on_extraction_error` to True."
                                    )
                                    logger.warning(warning_msg)
                                    warnings.warn(
                                        warning_msg,
                                        stacklevel=2,
                                    )
            else:
                # Process sequentially
                for i, data in enumerate(data_list):
                    result = await self._extract_items_from_instances(**data)
                    logger.debug(
                        f"Result for chunk ({i + 1}/{len(data_list)}) processed."
                    )

                    # Update usage stats and cost
                    await llm._update_usage_and_cost(result)

                    # Retry if needed
                    if not _llm_call_result_is_valid(result):
                        retry_successful = False
                        if llm.max_retries_invalid_data > 0:
                            retry_successful = await retry_processing_for_result(
                                llm_instance=llm,
                                res=result,
                                instances=data["instances_to_process"],
                                n_retries=llm.max_retries_invalid_data,
                                retry_is_final=not llm.fallback_llm,
                            )
                        # Retry with fallback LLM if it is provided
                        if not retry_successful and llm.fallback_llm:
                            logger.info("Trying with fallback LLM")
                            retry_successful = await retry_processing_for_result(
                                llm_instance=llm.fallback_llm,
                                res=result,
                                instances=data["instances_to_process"],
                                n_retries=max(
                                    llm.fallback_llm.max_retries_invalid_data, 1
                                ),  # retry with fallback LLM at least once
                                retry_is_final=True,
                            )
                        if not retry_successful:
                            if llm.max_retries_invalid_data > 0 or llm.fallback_llm:
                                if raise_exception_on_extraction_error:
                                    # Final retry was already performed, therefore an exception should have
                                    # been raised in case of extraction error such as invalid JSON.
                                    raise RuntimeError(
                                        "Extraction failed after all retries with "
                                        "`raise_exception_on_extraction_error` set to True, "
                                        "yet no exception was raised."
                                    )
                            else:
                                instance_names = [
                                    instance.name
                                    for instance in data["instances_to_process"]
                                ]
                                error_msg = (
                                    f"Some {instance_type}s could not be processed due to invalid JSON returned by LLM. "
                                    f"Failed {instance_type}s: {instance_names}"
                                )
                                if raise_exception_on_extraction_error:
                                    raise LLMExtractionError(error_msg, retry_count=0)
                                else:
                                    warning_msg = (
                                        error_msg
                                        + " (0 retries). "
                                        + " If you want to raise an exception instead, "
                                        "set `raise_exception_on_extraction_error` to True."
                                    )
                                    logger.warning(warning_msg)
                                    warnings.warn(
                                        warning_msg,
                                        stacklevel=2,
                                    )

        # Group instances for processing, based on the relevant params, with the relevant prompts.
        if instance_type == "aspect":
            instance_class = _Aspect
            attribute_name = "aspects"
        elif instance_type == "concept":
            instance_class = _Concept
            attribute_name = "concepts"
        else:
            raise NotImplementedError(f"Unknown instance_type: {instance_type}")

        all_instances = getattr(context, attribute_name)

        # If `from_instances` is specified, make sure they are valid.
        if from_instances is not None:
            if (
                not isinstance(from_instances, list)
                or not from_instances
                or not all(isinstance(i, instance_class) for i in from_instances)
                or not all(i in all_instances for i in from_instances)
            ):
                raise ValueError(
                    f"`from_instances` must be a list of {instance_class.__name__} instances "
                    f"assigned to {context.__class__.__name__}. Use get_*() methods to retrieve such "
                    f"assigned instances from the object, e.g. document.get_aspect_by_name(name), etc."
                )
            instances_to_process = from_instances
        else:
            instances_to_process = all_instances

        # First, filter by the current LLM role.
        filtered_instances = [i for i in instances_to_process if i.llm_role == llm.role]

        # If we are not overwriting, check that none of the instances is already processed.
        self._check_instances_already_processed(
            instance_type=instance_type,
            # Safe cast: filtered_instances contains only _Aspect or _Concept instances
            # based on instance_type
            instances=cast(list[_Aspect] | list[_Concept], filtered_instances),
            overwrite_existing=overwrite_existing,
        )

        # Next, group instances with the same values for extraction params.
        fields_to_group_by = [
            "add_justifications",
            "justification_depth",
            "justification_max_sents",
            "add_references",  # references are always added to aspects automatically
            "reference_depth",
        ]
        # Safe cast: filtered_instances contains only _Aspect or _Concept instances
        # based on instance_type
        typed_instances = (
            cast(list[_Aspect], filtered_instances)
            if instance_type == "aspect"
            else cast(list[_Concept], filtered_instances)
        )
        instance_groups = _group_instances_by_fields(
            fields=fields_to_group_by, instances=typed_instances
        )

        # Run extraction for each group separately, as prompts will be rendered differently and
        # the expected JSON structures will be different based on the group's extraction params.
        cals_and_kwargs = []
        for group in instance_groups:
            if not group:
                continue
            # Build a dictionary of "params" from the group's first instance
            # because they all share the same values for these fields.
            params_combination = {
                field: getattr(group[0], field, False) for field in fields_to_group_by
            }
            logger.debug(
                f"Processing group of {instance_class.__name__}s ({len(group)}) "
                f"with LLM `{llm.role}` with params: {params_combination}"
            )
            cals_and_kwargs.append(
                (
                    run_extraction,
                    {
                        **params_combination,
                        "instances_to_process": group,
                    },
                )
            )
        await _run_async_calls(
            cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
        )

    @staticmethod
    def _validate_sub_aspects_nesting_level(
        parent_aspect: _Aspect,
    ) -> None:
        """
        Validates that all sub-aspects have the correct nesting level relative
        to their parent aspect.

        Args:
            parent_aspect: The parent aspect

        Raises:
            ValueError: If any sub-aspect of the parent aspect has
            an incorrect nesting level.
        """
        expected_level = parent_aspect._nesting_level + 1
        if not all(
            sub_aspect._nesting_level == expected_level
            for sub_aspect in parent_aspect.aspects
        ):
            raise ValueError(
                f"Sub-aspects must have a nesting level of `{expected_level}`. "
                f"Current nesting levels: "
                f"{[sub_aspect._nesting_level for sub_aspect in parent_aspect.aspects]}"
            )

    def _get_usage_or_cost(
        self,
        retrieval_type: Literal["usage", "cost"],
        llm_role: str | None = None,
    ) -> list[_LLMUsageOutputContainer | _LLMCostOutputContainer]:
        """
        Retrieves specified information (usage or cost) for either a single LLM or LLMs in a group.
        For groups, optionally filters the results by the specified LLM role.
        Iterates through primary LLMs and their fallback counterparts, collecting
        details about their usage or cost based on the specified retrieval type.

        :param retrieval_type: Determines the type of information to retrieve. Must be either
            "usage" to collect LLM usage statistics or "cost" to collect cost details.
        :param llm_role: The optional role of the LLM to filter the results. If provided,
            only results associated with LLMs matching this role are returned. If no LLM with
            the specified role exists, an exception is raised. If the matching LLM has
            a fallback LLM, its usage or cost details are also collected. Defaults to None.
        :return: A list of containers, each representing usage or cost information for
            a primary and fallback LLM, if it exists. The specific container type depends on
            the retrieval type.
        """
        info_containers = []
        # For individual LLM, use self and its fallback
        # For group, iterate through all LLMs in self.llms
        llms_to_process = self.llms if self.is_group else [self]  # type: ignore[attr-defined]
        # Safe cast: llms_to_process contains only _DocumentLLM instances
        llms_to_process = cast(list["_DocumentLLM"], llms_to_process)

        for primary_llm in llms_to_process:
            for llm in [primary_llm, primary_llm.fallback_llm]:
                if llm:  # fallback LLM may be missing
                    if retrieval_type == "usage":
                        info_container = _LLMUsageOutputContainer(
                            model=llm.model,
                            role=llm.role,
                            is_fallback=llm.is_fallback,
                            usage=llm._get_raw_usage(),
                        )
                    elif retrieval_type == "cost":
                        info_container = _LLMCostOutputContainer(
                            model=llm.model,
                            role=llm.role,
                            is_fallback=llm.is_fallback,
                            cost=llm._get_raw_cost(),
                        )
                    else:
                        raise ValueError(f"Invalid retrieval type `{retrieval_type}`")
                    info_containers.append(info_container)

        if llm_role:
            info_containers = [
                i for i in info_containers if i.role == llm_role
            ]  # for both main and fallback LLMs
            if not info_containers:
                raise ValueError(
                    f"No LLM with the given role `{llm_role}` was found in group."
                )
        return info_containers

    @property
    @abstractmethod
    def is_group(self) -> bool:
        """
        Abstract property, to be implemented by subclasses.

        Whether the LLM is a single instance or a group.
        """
        pass

    @property
    @abstractmethod
    def list_roles(self) -> list[LLMRoleAny]:
        """
        Abstract property, to be implemented by subclasses.

        Returns the list of all LLM roles in the LLM group or LLM.
        """
        pass

    @abstractmethod
    def _set_private_attrs(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Sets private attributes for the LLM group or LLM, e.g. prompts, capabilities, etc.
        """
        pass

    @abstractmethod
    def get_usage(self, *args, **kwargs) -> list[_LLMUsageOutputContainer]:
        """
        Abstract method, to be implemented by subclasses.

        Returns the usage data for the LLM group or LLM.
        """
        pass

    @abstractmethod
    def get_cost(self, *args, **kwargs) -> list[dict[str, str | _LLMCost]] | _LLMCost:
        """
        Abstract method, to be implemented by subclasses.

        Returns the cost data for the LLM group or LLM.
        """
        pass

    @abstractmethod
    def reset_usage_and_cost(self) -> None:
        """
        Abstract method, to be implemented by subclasses.

        Resets the usage and cost data for the LLM group or LLM.
        """
        pass


@_disable_direct_initialization
class _DocumentLLMGroup(_GenericLLMProcessor):
    """
    Internal implementation of the DocumentLLMGroup class.
    """

    llms: list[_DocumentLLM] = Field(
        ...,
        min_length=2,
        description=(
            "List of DocumentLLM instances assigned unique roles (e.g., "
            "'extractor_text', 'reasoner_text', 'extractor_vision', 'reasoner_vision'); "
            "minimum 2."
        ),
    )
    output_language: LanguageRequirement = Field(
        default="en",
        description=(
            "Language for generated textual output (justifications, explanations). "
            "'en' forces English; 'adapt' matches document/image language. "
            "Must be consistent across all LLMs in the group."
        ),
    )

    _llm_extractor_text: _DocumentLLM | None = PrivateAttr(default=None)  # type: ignore
    _llm_reasoner_text: _DocumentLLM | None = PrivateAttr(default=None)  # type: ignore
    _llm_extractor_vision: _DocumentLLM | None = PrivateAttr(default=None)  # type: ignore
    _llm_reasoner_vision: _DocumentLLM | None = PrivateAttr(default=None)  # type: ignore

    def _set_private_attrs(self) -> None:
        """
        Initialize and configure private attributes for the LLM group.

        :return: None
        :rtype: None
        """
        self._assign_role_specific_llms()

    @property
    def is_group(self) -> bool:
        """
        Returns True indicating this is a group of LLMs.

        :return: Always True for DocumentLLMGroup instances.
        :rtype: bool
        """
        return True

    @property
    def list_roles(self) -> list[LLMRoleAny]:
        """
        Returns a list of all roles assigned to the LLMs in this group.

        :return: A list of LLM role identifiers
        :rtype: list[LLMRoleAny]
        """
        return [i.role for i in self.llms]

    def group_update_output_language(
        self, output_language: LanguageRequirement
    ) -> None:
        """
        Updates the output language for all LLMs in the group.

        :param output_language: The new output language to set for all LLMs
        :type output_language: LanguageRequirement
        """
        for llm in self.llms:
            llm.output_language = output_language
        self.output_language = output_language
        logger.info(
            f"Updated output language for all LLMs in the group to `{output_language}`"
        )

    def _eq_deserialized_llm_config(
        self,
        other: _DocumentLLMGroup,
    ) -> bool:
        """
        Custom config equality method to compare this _DocumentLLMGroup with a deserialized instance.

        Uses the `_eq_deserialized_llm_config` method of the _DocumentLLM class to compare each LLM
        in the group, including fallbacks, if any.

        :param other: Another _DocumentLLMGroup instance to compare with
        :type other: _DocumentLLMGroup
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """
        for self_llm, other_llm in zip(self.llms, other.llms, strict=True):
            if not self_llm._eq_deserialized_llm_config(other_llm):
                return False
        return True

    @field_validator("llms")
    @classmethod
    def _validate_llms(cls, llms: list[_DocumentLLM]) -> list[_DocumentLLM]:
        """
        Validates the provided list of _DocumentLLMs ensuring that each
        LLM has a unique role within the group.

        :param llms: A list of `_DocumentLLM` instances to be validated.
        :return: A validated list of `_DocumentLLM` instances.
        :raises ValueError: If validation fails.
        """
        seen_roles = set()
        for llm in llms:
            if llm.role in seen_roles:
                raise ValueError("LLMs must have different roles.")
            seen_roles.add(llm.role)
        return llms

    def _assign_role_specific_llms(self) -> None:
        """
        Assigns specific LLMs to dedicated roles based on the role attribute of each LLM.

        :return: None
        """

        def get_llm_by_role(role: str) -> _DocumentLLM | None:  # type: ignore
            """
            Finds and returns the first LLM with the specified role.

            :param role: The role to search for among the LLMs.
            :type role: str
            :return: The _DocumentLLM with the matching role, or None if not found.
            :rtype: _DocumentLLM | None
            """
            return next((i for i in self.llms if i.role == role), None)

        self._llm_extractor_text = get_llm_by_role("extractor_text")
        self._llm_reasoner_text = get_llm_by_role("reasoner_text")
        self._llm_extractor_vision = get_llm_by_role("extractor_vision")
        self._llm_reasoner_vision = get_llm_by_role("reasoner_vision")

    def get_usage(self, llm_role: str | None = None) -> list[_LLMUsageOutputContainer]:
        """
        Retrieves the usage information of the LLMs in the group, filtered by the specified
        LLM role if provided.

        :param llm_role: Optional; A string representing the role of the LLM to filter
            the usage data. If None, returns usage for all LLMs in the group.
        :type llm_role: str | None
        :return: A list of usage statistics containers for the specified LLMs and their fallbacks.
        :rtype: list[_LLMUsageOutputContainer]
        :raises ValueError: If no LLM with the specified role exists in the group.
        """
        # Safe cast: _get_usage_or_cost with retrieval_type="usage"
        # returns only list of _LLMUsageOutputContainer
        return cast(
            list["_LLMUsageOutputContainer"],
            self._get_usage_or_cost(retrieval_type="usage", llm_role=llm_role),
        )

    def get_cost(self, llm_role: str | None = None) -> list[_LLMCostOutputContainer]:
        """
        Retrieves the accumulated cost information of the LLMs in the group, filtered by the specified
        LLM role if provided.

        :param llm_role: Optional; A string representing the role of the LLM to filter
            the cost data. If None, returns cost for all LLMs in the group.
        :type llm_role: str | None
        :return: A list of cost statistics containers for the specified LLMs and their fallbacks.
        :rtype: list[_LLMCostOutputContainer]
        :raises ValueError: If no LLM with the specified role exists in the group.
        """
        # Safe cast: _get_usage_or_cost with retrieval_type="cost"
        # returns only list of _LLMCostOutputContainer
        return cast(
            list["_LLMCostOutputContainer"],
            self._get_usage_or_cost(retrieval_type="cost", llm_role=llm_role),
        )

    def reset_usage_and_cost(self, llm_role: str | None = None) -> None:
        """
        Resets the usage and cost statistics for LLMs in the group.

        This method clears accumulated usage and cost data, which is useful when processing
        multiple documents sequentially and tracking metrics for each document separately.

        :param llm_role: Optional; A string representing the role of the LLM to reset statistics for.
            If None, resets statistics for all LLMs in the group.
        :type llm_role: str | None
        :raises ValueError: If no LLM with the specified role exists in the group.
        :return: None
        """
        if llm_role:
            try:
                llm = next(i for i in self.llms if i.role == llm_role)
                llm.reset_usage_and_cost()
            except StopIteration:
                raise ValueError(
                    f"No LLM with the given role `{llm_role}` was found in group."
                ) from None
        else:
            for llm in self.llms:
                llm.reset_usage_and_cost()

    @model_validator(mode="after")
    def _validate_document_llm_group_post(self) -> Self:
        """
        Validates the LLM group to ensure consistency of the `output_language`
        attribute across all LLMs within the group.

        Raises:
            ValueError: Raised if any LLM's `output_language` differs from the
            group's `output_language`.

        :return: The LLM group instance after successful validation.
        :rtype: Self
        """
        # Set private attributes before validation
        self._set_private_attrs()

        if any(i.output_language != self.output_language for i in self.llms):
            raise ValueError(
                "All LLMs in the group must have the same value of "
                "`output_language` attribute as the group."
            )
        return self


@_disable_direct_initialization
class _DocumentLLM(_GenericLLMProcessor):
    """
    Internal implementation of the DocumentLLM class.
    """

    # LLM config
    model: NonEmptyStr = Field(
        ...,
        description="Model identifier '<provider>/<model>' (e.g., 'openai/gpt-4o').",
    )
    deployment_id: NonEmptyStr | None = Field(
        default=None,
        description="Deployment ID for the model (e.g., for Azure OpenAI).",
    )
    api_key: NonEmptyStr | None = Field(
        default=None,
        description="API key for provider authentication; optional for local models.",
    )
    api_base: NonEmptyStr | None = Field(
        default=None,
        description="Base URL of the provider API endpoint.",
    )
    api_version: NonEmptyStr | None = Field(
        default=None,  # specific to Azure OpenAI
        description="API version; used by some providers (e.g., Azure OpenAI).",
    )
    role: LLMRoleAny = Field(
        default="extractor_text",
        description=(
            "LLM role for pipeline routing (e.g., 'extractor_text', 'reasoner_text', "
            "'extractor_vision', 'reasoner_vision')."
        ),
    )
    system_message: str | None = Field(
        default=None,
        description="System prompt to prime the model; defaults to framework message if unset.",
    )
    max_tokens: StrictInt = Field(
        default=4096,
        gt=0,
        description="Maximum output tokens for non-reasoning models.",
    )
    max_completion_tokens: StrictInt = Field(
        default=16000,
        gt=0,
        description="Maximum completion tokens for reasoning (CoT-capable) models.",
    )  # for reasoning (CoT-capable) models
    reasoning_effort: ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort for CoT models: 'low' | 'medium' | 'high'.",
    )  # for reasoning (CoT-capable) models
    num_retries_failed_request: StrictInt = Field(
        default=3,
        ge=0,
        description="Client-side retry count for failed requests.",
    )
    max_retries_failed_request: StrictInt = Field(
        default=0,
        ge=0,  # provider-specific
        description="Provider SDK retry count for failed requests.",
    )
    max_retries_invalid_data: StrictInt = Field(
        default=3,
        ge=0,
        description="Retries when the model returns invalid or unparsable data.",
    )
    timeout: StrictInt = Field(
        default=120,
        ge=0,
        description="Request timeout in seconds.",
    )
    pricing_details: _LLMPricing | None = Field(  # type: ignore
        default=None,
        description="Explicit pricing configuration for cost calculation.",
    )
    is_fallback: StrictBool = Field(
        default=False,
        description="Marks this LLM as a fallback model.",
    )
    fallback_llm: _DocumentLLM | None = Field(  # type: ignore
        default=None,
        description=(
            "Fallback LLM to use if this one fails; must have the same role and output_language."
        ),
    )
    output_language: LanguageRequirement = Field(
        default="en",
        description=(
            "Language for generated textual output (justifications, explanations). "
            "'en' forces English; 'adapt' matches document/image language."
        ),
    )
    temperature: StrictFloat | None = Field(
        default=0.3,
        ge=0,
        description="Sampling temperature [0..1]; higher values increase randomness.",
    )
    top_p: StrictFloat | None = Field(
        default=0.3,
        ge=0,
        description="Nucleus sampling [0..1]; alternative to temperature.",
    )
    seed: StrictInt | None = Field(
        default=None,
        description="Random seed for sampling (provider support dependent).",
    )

    # Auto-pricing via genai-prices (optional)
    auto_pricing: StrictBool = Field(
        default=False,
        description=(
            "Enable automatic price lookup via genai-prices when pricing_details is not set."
        ),
    )
    auto_pricing_refresh: StrictBool = Field(
        default=False,
        description="Allow genai-prices to auto-refresh cached pricing data.",
    )

    # Derived automatically from `model`
    _auto_pricing_provider_id: str | None = PrivateAttr(default=None)
    _auto_pricing_model_ref: str | None = PrivateAttr(default=None)
    _auto_pricing_refresh_attempted: bool = PrivateAttr(
        default=False
    )  # update pricing only once to avoid redundant updates during extraction

    # Prompts
    _extract_aspect_items_prompt: Template = PrivateAttr()
    _extract_concept_items_prompt: Template = PrivateAttr()

    # Async
    _async_limiter: AsyncLimiter = PrivateAttr()

    # Token counting
    _usage: _LLMUsage = PrivateAttr(default_factory=_LLMUsage)
    # Processing cost
    _cost: _LLMCost = PrivateAttr(default_factory=_LLMCost)

    # Async lock to guard shared state during async updates
    _async_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    # Capabilities (can be overridden by users if litellm.supports_*()
    # is not accurate for the model)
    _supports_vision: bool = PrivateAttr(default=False)
    _supports_reasoning: bool = PrivateAttr(default=False)

    def __init__(self, **data: Any):
        # Pop the async_limiter if provided; otherwise use a default.
        limiter = data.pop("async_limiter", None)
        super().__init__(**data)
        if limiter is not None:
            self.async_limiter = limiter
        else:
            self.async_limiter = AsyncLimiter(3, 10)

    @_post_init_method
    def _post_init(self, __context: Any):
        """
        Post-initialization method that logs model information and API configuration,
        as well as emits warnings for specific LLM configurations.

        :param __context: Pydantic context (unused).
        :type __context: Any
        """
        logger.info(f"Using model {self.model}")
        if self.api_key is None:
            logger.info("API key was not provided. Set `api_key`, if applicable.")
        if self.api_base is None:
            logger.info("API base was not provided. Set `api_base`, if applicable.")

        # Log helpful message for local model providers that may use smaller models
        if any(self.model.startswith(provider) for provider in _LOCAL_MODEL_PROVIDERS):
            logger.info(
                "Using local model provider. If you experience issues like JSON validation errors "
                "with smaller models, see our troubleshooting guide: "
                "https://contextgem.dev/optimizations/optimization_small_llm_troubleshooting.html"
            )

        # Recommend `ollama_chat` prefix for better responses for Ollama models (text-only processing)
        if self.model.startswith("ollama/") and not self.role.endswith("_vision"):
            logger.info(
                "For better responses with Ollama models, consider using "
                "'ollama_chat/' prefix instead of 'ollama/', as recommended by LiteLLM: "
                "https://docs.litellm.ai/docs/providers/ollama"
            )

        # Warn to use `ollama/` prefix for image processing when using local vision models,
        # as the ollama_chat/ does not yet support image inputs
        if self.model.startswith("ollama_chat/") and self.role.endswith("_vision"):
            warnings.warn(
                "Using `ollama_chat/` prefix for local vision models is not recommended, "
                "as it does not yet support image inputs. Please use `ollama/` prefix instead. "
                "See https://github.com/ollama/ollama/issues/10255 and "
                "https://github.com/ollama/ollama/issues/6451 for more details.",
                stacklevel=2,
            )

        # TODO: Remove this once LiteLLM's `lm_studio` gpt-oss support is fixed
        # Warn when gpt-oss models are used with LM Studio due to instability
        if self.model.startswith("lm_studio/") and "openai/gpt-oss" in self.model:
            warnings.warn(
                "Using OpenAI gpt-oss models with `lm_studio/` is currently unstable and may "
                "cause errors. For a stable alternative, consider using the model via Ollama "
                "instead (e.g., `ollama_chat/gpt-oss:20b`).",
                stacklevel=2,
            )

        # Warn about auto-pricing accuracy limitations
        if self.auto_pricing:
            warnings.warn(
                "Auto-pricing: these prices will not be 100% accurate. "
                "The price data cannot be exactly correct because model providers do not provide "
                "exact price information for their APIs in a format which can be reliably processed. "
                "See Pydantic's genai-prices https://github.com/pydantic/genai-prices for more details.",
                stacklevel=2,
            )

    def _set_private_attrs(self) -> None:
        """
        Initialize and configure private attributes for the LLM instance.

        :return: None
        :rtype: None
        """
        if self.system_message is None:
            self._set_system_message()
        self._set_prompts()
        self._set_capabilities()
        if self.auto_pricing:
            self._set_provider_and_model_for_auto_pricing()

    @property
    def async_limiter(self) -> AsyncLimiter:
        """
        Gets the async rate limiter for this LLM.

        :return: The AsyncLimiter instance controlling request rate limits.
        :rtype: AsyncLimiter
        """
        return self._async_limiter

    @async_limiter.setter
    def async_limiter(self, value: AsyncLimiter) -> None:
        """
        Sets the async rate limiter for this LLM.

        :param value: The AsyncLimiter instance to set.
        :type value: AsyncLimiter
        :raises TypeError: If value is not an AsyncLimiter instance.
        """
        if not isinstance(value, AsyncLimiter):
            raise TypeError("async_limiter must be an AsyncLimiter instance")
        self._async_limiter = value

    @property
    def is_group(self) -> bool:
        """
        Returns False indicating this is a single LLM, not a group.

        :return: Always False for DocumentLLM instances.
        :rtype: bool
        """
        return False

    @property
    def list_roles(self) -> list[LLMRoleAny]:
        """
        Returns a list containing the role of this LLM.

        (For a single LLM, this returns a list with just one element - the LLM's role.
        For LLM groups, the method implementation returns roles of all LLMs in the group.)

        :return: A list containing the role of this LLM.
        :rtype: list[LLMRoleAny]
        """
        return [self.role]

    def chat(self, prompt: str, images: list[_Image] | None = None) -> str:
        """
        Synchronously sends a prompt to the LLM and gets a response.
        For models supporting vision, attach images to the prompt if needed.

        This method allows direct interaction with the LLM by submitting your own prompt.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :return: The LLM's response
        :rtype: str
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises RuntimeError: If the LLM call fails and no fallback is available
        """
        return _run_sync(self.chat_async(prompt, images))

    async def chat_async(self, prompt: str, images: list[_Image] | None = None) -> str:
        """
        Asynchronously sends a prompt to the LLM and gets a response.
        For models supporting vision, attach images to the prompt if needed.

        This method allows direct interaction with the LLM by submitting your own prompt.

        :param prompt: The input prompt to send to the LLM
        :type prompt: str
        :param images: Optional list of Image instances for vision queries
        :type images: list[Image] | None
        :return: The LLM's response
        :rtype: str
        :raises ValueError: If the prompt is empty or not a string
        :raises ValueError: If images parameter is not a list of Image instances
        :raises ValueError: If images are provided but the model doesn't support vision
        :raises RuntimeError: If the LLM call fails and no fallback is available
        """

        # Validate prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        # Validate images
        if images and (
            not isinstance(images, list)
            or not all(isinstance(image, _Image) for image in images)
        ):
            raise ValueError("Images must be a list of Image instances")

        # Check for vision support
        if images and not self._supports_vision:
            raise ValueError(
                f"Model `{self.model}` does not support vision according to "
                f"litellm.supports_vision(). To override this detection, "
                f"manually set `_supports_vision=True` on the LLM instance."
            )

        # Create LLM call object to track the interaction
        llm_call = _LLMCall(prompt_kwargs={}, prompt=prompt)

        # Warn if using default system message
        # _get_template returns a Template object when template_extension == "j2"
        default_system_message = _get_template(
            "default_system_message",
            template_type="system",
            template_extension="j2",
        ).render({"output_language": self.output_language})  # type: ignore[attr-defined]
        if self.system_message == default_system_message:
            warnings.warn(
                "You are using the default system message optimized for extraction tasks. "
                "For simple chat interactions, consider setting system_message='' to disable it, "
                "or provide your own custom system message.",
                stacklevel=2,
            )

        # Send message to LLM
        result = await self._query_llm(
            message=prompt,
            llm_call_obj=llm_call,
            images=images,
            num_retries_failed_request=self.num_retries_failed_request,
            max_retries_failed_request=self.max_retries_failed_request,
            raise_exception_on_llm_api_error=True,  # always True for chat
        )

        # Update usage and cost statistics
        await self._update_usage_and_cost(result)

        response, _ = result

        # If response is None and fallback LLM is available, try with fallback LLM
        if response is None and self.fallback_llm:
            logger.info(f"Using fallback LLM {self.fallback_llm.model} for chat")
            return await self.fallback_llm.chat_async(prompt, images)
        elif response is None:
            raise RuntimeError(
                f"Failed to get response from LLM {self.model} and no fallback is available"
            )

        return response

    def _update_default_prompt(
        self, prompt_path: str | Path, prompt_type: DefaultPromptType
    ) -> None:
        """
        For advanced users only!

        Update the default Jinja2 prompt template for the LLM.

        This method allows you to replace the built-in prompt templates with custom ones
        for specific extraction types. The framework uses these templates to guide the LLM
        in extracting structured information from documents.

        The custom prompt must be a valid Jinja2 template and include all the necessary
        variables that are present in the default prompt. Otherwise, the extraction may fail.
        Default prompts are located under ``contextgem/internal/prompts/``

        IMPORTANT NOTES:

        The default prompts are complex and specifically designed for
        various steps of LLM extraction with the framework. Such prompts include the
        necessary instructions, template variables, nested structures and loops, etc.

        Only use custom prompts if you MUST have a deeper customization and adaptation of the
        default prompts to your specific use case. Otherwise, the default prompts should be
        sufficient for most use cases.

        Use at your own risk!

        :param prompt_path: Path to the Jinja2 template file (.j2 extension required)
        :type prompt_path: str | Path
        :param prompt_type: Type of prompt to update ("aspect" or "concept")
        :type prompt_type: DefaultPromptType
        """
        # Convert to string if Path object
        prompt_path_str = str(prompt_path)

        if not prompt_path_str.endswith(".j2"):
            raise ValueError("Prompt path must end with `.j2`.")

        with open(prompt_path, encoding="utf-8") as file:
            template_text = file.read().strip()
            if not template_text:
                raise ValueError("Prompt template is empty.")

        template = _setup_jinja2_template(template_text)

        if prompt_type == "aspect":
            self._extract_aspect_items_prompt = template
        elif prompt_type == "concept":
            self._extract_concept_items_prompt = template
        else:
            raise ValueError(f"Invalid prompt type: {prompt_type}")
        logger.info(
            f"Default prompt for {prompt_type} extraction updated with a custom template."
        )

    def _eq_deserialized_llm_config(
        self,
        other: _DocumentLLM,
    ) -> bool:
        """
        Custom config equality method to compare this _DocumentLLM with a deserialized instance.

        Compares the __dict__ of both instances and performs specific checks for
        certain attributes that require special handling.

        Note that, by default, the reconstructed deserialized _DocumentLLM will be only partially
        equal (==) to the original one, as the api credentials are redacted, and the attached prompt
        templates, async limiter, and async lock are not serialized and point to different objects
        in memory post-initialization. Also, usage and cost are reset by default pre-serialization.

        :param other: Another _DocumentLLM instance to compare with
        :type other: _DocumentLLM
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """

        # Create a copy of the dictionaries to modify
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()

        # Skips check for fallback LLM, if it is set
        if self.fallback_llm:
            self_fallback_llm = self_dict.pop("fallback_llm")
            other_fallback_llm = other_dict.pop("fallback_llm")
            if not other_fallback_llm:
                raise RuntimeError("Deserialized fallback LLM was not set")
            if not self_fallback_llm._eq_deserialized_llm_config(other_fallback_llm):
                logger.debug("Fallback LLM config of deserialized LLM is different.")
                return False

        # Skip checks for api_key and api_base that were redacted pre-serialization
        self_dict.pop("api_key")
        other_dict.pop("api_key")
        self_dict.pop("api_base")
        other_dict.pop("api_base")
        if not (other.api_key is None and other.api_base is None):
            raise RuntimeError(
                "Deserialized LLM has api_key or api_base set, "
                "while API credentials should have been redacted pre-serialization"
            )

        # Compare the modified dictionaries
        if self_dict != other_dict:
            logger.debug("LLM __dict__ of deserialized LLM is different.")
            return False

        # Special checks for specific private attributes

        # Check _extract_aspect_items_prompt
        if (
            self._extract_aspect_items_prompt.render()
            != other._extract_aspect_items_prompt.render()
        ):
            logger.debug(
                "Extract aspect items prompt of deserialized LLM is different."
            )
            return False

        # Check _extract_concept_items_prompt
        if (
            self._extract_concept_items_prompt.render()
            != other._extract_concept_items_prompt.render()
        ):
            logger.debug(
                "Extract concept items prompt of deserialized LLM is different."
            )
            return False

        # Check that usage and cost stats were reset pre-serialization
        if not (other._usage == _LLMUsage() and other._cost == _LLMCost()):
            raise RuntimeError(
                "Usage and cost stats were not properly reset during serialization"
            )

        # Check _async_limiter
        if (
            self._async_limiter.time_period != other._async_limiter.time_period
            or self._async_limiter.max_rate != other._async_limiter.max_rate
        ):
            logger.debug("Async limiter params of deserialized LLM are different.")
            return False

        # Check _async_lock
        if not (
            isinstance(self._async_lock, asyncio.Lock)
            and isinstance(other._async_lock, asyncio.Lock)
        ):
            logger.debug("Async lock of deserialized LLM is different.")
            return False

        # Check capabilities
        if self._supports_vision != other._supports_vision:
            logger.debug("`_supports_vision` of deserialized LLM is different.")
            return False
        if self._supports_reasoning != other._supports_reasoning:
            logger.debug("`_supports_reasoning` of deserialized LLM is different.")
            return False

        return True

    @field_validator("model")
    @classmethod
    def _validate_model(cls, model: str) -> str:
        """
        Validates the model identifier to ensure it conforms to the expected format.

        :param model: Model identifier string to validate.
        :type model: str
        :return: The validated model string.
        :rtype: str
        :raises ValueError: If `model` does not contain a forward slash ('/') to indicate
            the required format.
        """
        if "/" not in model:
            raise ValueError(
                "Model identifier must be in the form of `<model_provider>/<model_name>`. "
                "See https://docs.litellm.ai/docs/providers for the list of supported providers."
            )
        return model

    @field_validator("fallback_llm")
    @classmethod
    def _validate_fallback_llm(
        cls,
        fallback_llm: _DocumentLLM | None,  # type: ignore
    ) -> _DocumentLLM | None:  # type: ignore
        """
        Validates the ``fallback_llm`` input to ensure it meets the expected condition
        of being a fallback LLM model.

        :param fallback_llm: The _DocumentLLM instance to be validated.
        :type fallback_llm: _DocumentLLM
        :return: The valid fallback_llm that meets the expected criteria.
        :rtype: _DocumentLLM
        :raises ValueError: If the ``fallback_llm`` is not a fallback model, as
            indicated by the ``is_fallback`` attribute set to ``False``.
        """
        if fallback_llm is not None and not fallback_llm.is_fallback:
            raise ValueError(
                "Fallback LLM must be a fallback model. Use `is_fallback=True`."
            )
        return fallback_llm

    @model_validator(mode="after")
    def _validate_document_llm_post(self) -> Self:
        """
        Validate the integrity of the document LLM model after initialization.

        :return: Returns the instance of the current LLM model after successful validation.
        :rtype: Self
        :raises ValueError: If the LLM model is not properly configured.
        """

        # Set private attributes before validation
        self._set_private_attrs()

        # pricing_details and auto_pricing are mutually exclusive
        if self.pricing_details is not None and self.auto_pricing:
            raise ValueError(
                "`pricing_details` cannot be set together with `auto_pricing=True`. "
                "Provide manual LLMPricing or enable auto_pricing, not both."
            )

        if (
            any(self.model.startswith(provider) for provider in _LOCAL_MODEL_PROVIDERS)
            and self.auto_pricing
        ):
            raise ValueError(
                "`auto_pricing=True` is not supported for local models. "
                "Disable `auto_pricing` or provide explicit `pricing_details`."
            )

        # Fallback model validation
        if self.is_fallback and self.fallback_llm:
            raise ValueError(
                "Fallback LLM cannot have its own fallback LLM "
                "and must be attached to a non-fallback model."
            )

        if self.fallback_llm:
            # Check for the consistency of the fallback LLM role and output language
            if self.fallback_llm.role != self.role:
                raise ValueError(
                    f"The fallback LLM must have the same role `{self.role}` as the main one."
                )
            elif self.fallback_llm.output_language != self.output_language:
                raise ValueError(
                    f"The fallback LLM must have the same output language `{self.output_language}` as the main one."
                )

            # Check that the fallback LLM is not the replica of the main LLM, just with different
            # `is_fallback` and `fallback_llm` params
            main_llm_dict = {
                k: v
                for k, v in self.__dict__.items()
                if k not in ["is_fallback", "fallback_llm"]
            }
            fallback_llm_dict = {
                k: v
                for k, v in self.fallback_llm.__dict__.items()
                if k not in ["is_fallback", "fallback_llm"]
            }
            if main_llm_dict == fallback_llm_dict:
                raise ValueError(
                    "Fallback LLM must not have the exact same config params as the main LLM."
                )

        # Emit relevant warnings

        # Vision support check - when applicable
        if self.role.endswith("_vision") and not self._supports_vision:
            # Prompt the user to override _supports_vision if the model is known to support
            # vision while litellm does not detect it as vision-capable
            warnings.warn(
                f"Model `{self.model}` is assigned vision role `{self.role}` but "
                f"litellm does not detect it as vision-capable. This will cause "
                f"vision-related operations to fail. If you know this model supports "
                f"vision, manually set `_supports_vision=True` on the LLM instance.",
                stacklevel=2,
            )

        # Reasoning support check - when applicable
        if self.role.startswith("reasoner") and not self._supports_reasoning:
            # Prompt the user to override _supports_reasoning if the model is known to support
            # reasoning while litellm does not detect it as reasoning-capable
            warnings.warn(
                f"Model `{self.model}` is assigned reasoning role `{self.role}` but "
                f"litellm does not detect it as reasoning-capable. This will cause "
                f"reasoning-related prompt instructions to be skipped, which could lead to "
                f"lower quality responses. If you know this model supports reasoning, "
                f"manually set `_supports_reasoning=True` on the LLM instance.",
                stacklevel=2,
            )

        # Extractor role with reasoning-capable model - suggest using a reasoner role
        if self.role.startswith("extractor") and self._supports_reasoning:
            warnings.warn(
                f"Model `{self.model}` is assigned extractor role `{self.role}`, "
                f"while the model is reasoning-capable. Consider using a reasoner role "
                f"to enable reasoning-related instructions for higher quality responses.",
                stacklevel=2,
            )

        return self

    def _validate_input_tokens(self, messages: list[dict[str, str]]) -> None:
        """
        Validates that the input messages do not exceed the model's maximum input tokens.

        :param messages: List of message dictionaries to validate
        :type messages: list[dict[str, str]]
        :raises ValueError: If the messages exceed the model's maximum input tokens
        :return: None
        """
        context_exceeded = False
        try:
            # Get model information to check context window
            model_info = litellm.get_model_info(self.model)  # type: ignore[attr-defined]
            max_input_tokens = model_info.get("max_input_tokens")

            # If max_input_tokens is not available, skip validation
            if max_input_tokens is None:
                logger.warning(
                    f"Could not determine max_input_tokens for model `{self.model}`. Skipping validation."
                )
                return

            # Count tokens in the messages
            try:
                token_count = litellm.token_counter(model=self.model, messages=messages)  # type: ignore[attr-defined]
            except Exception as e:
                logger.warning(
                    f"Could not count tokens for model `{self.model}`: {e}. Skipping input token validation."
                )
                return

            # Check if we exceed the context window
            if token_count > max_input_tokens:
                context_exceeded = True
                raise ValueError(
                    f"Input messages contain {token_count} tokens, which exceeds the model's "
                    f"maximum input tokens of {max_input_tokens} for model `{self.model}`. "
                    f"For long documents, consider setting `max_paragraphs_to_analyze_per_call` "
                    f"(for text) or `max_images_to_analyze_per_call` (for images) to process the "
                    f"document in smaller chunks. "
                    f"See the optimization guide for long documents: "
                    f"https://contextgem.dev/optimizations/optimization_long_docs.html"
                )

            logger.debug(
                f"Input token validation passed: {token_count}/{max_input_tokens} tokens used"
            )

        except ValueError as e:
            if context_exceeded:
                # Re-raise our own ValueError
                raise
            # If it's a different ValueError, log and continue
            logger.warning(
                f"Could not validate max input tokens for model `{self.model}`: {e}"
            )
        except Exception as e:
            # If we can't get model info, log a warning but don't fail
            logger.warning(
                f"Could not validate max input tokens for model `{self.model}`: {e}"
            )

    def _validate_output_tokens(self) -> None:
        """
        Validates that the configured max_tokens or max_completion_tokens do not exceed
        the model's maximum output tokens.

        :raises ValueError: If the configured tokens exceed the model's maximum output tokens
        :return: None
        """
        output_exceeded = False
        try:
            # Get model information to check output token limits
            model_info = litellm.get_model_info(self.model)  # type: ignore[attr-defined]
            max_output_tokens = model_info.get("max_output_tokens")

            # If max_output_tokens is not available, fall back to max_tokens
            if max_output_tokens is None:
                max_output_tokens = model_info.get("max_tokens")

            # If we still don't have a limit, skip validation
            if max_output_tokens is None:
                logger.warning(
                    f"Could not determine max_output_tokens for model `{self.model}`. "
                    f"Skipping max output token validation."
                )
                return

            # Determine which token limit to check based on model type
            if self._supports_reasoning:
                configured_tokens = self.max_completion_tokens
                token_type = "max_completion_tokens"  # nosec B105 - not a password
            else:
                configured_tokens = self.max_tokens
                token_type = "max_tokens"  # nosec B105 - not a password

            # Check if configured tokens exceed the model's limit
            if configured_tokens > max_output_tokens:
                output_exceeded = True
                raise ValueError(
                    f"Configured {token_type} ({configured_tokens}) exceeds the model's "
                    f"maximum output tokens of {max_output_tokens} for model `{self.model}`. "
                    f"For long documents, consider setting `max_paragraphs_to_analyze_per_call` "
                    f"(for text) or `max_images_to_analyze_per_call` (for images) to process the "
                    f"document in smaller chunks. "
                    f"See the optimization guide for long documents: "
                    f"https://contextgem.dev/optimizations/optimization_long_docs.html"
                )

            logger.debug(
                f"Output token validation passed: {configured_tokens}/{max_output_tokens} "
                f"{token_type} configured"
            )

        except ValueError as e:
            if output_exceeded:
                # Re-raise our own ValueError
                raise
            # If it's a different ValueError, log and continue
            logger.warning(
                f"Could not validate max output tokens for model `{self.model}`: {e}"
            )
        except Exception as e:
            # If we can't get model info, log a warning but don't fail
            logger.warning(
                f"Could not validate max output tokens for model `{self.model}`: {e}"
            )

    async def _query_llm(
        self,
        message: str,
        llm_call_obj: _LLMCall,
        images: list[_Image] | None = None,
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        async_limiter: AsyncLimiter | None = None,
        drop_params: bool = False,
        raise_exception_on_llm_api_error: bool = True,
    ) -> tuple[str | None, _LLMUsage]:
        """
        Generates a response from an LLM based on the provided message, optional images,
        and system configuration. It formats the input messages according to the
        compatibility with different versions of the LLM, sends the request to the
        LLM API, and processes the generated response.

        :param message: The input message from the user intended for the LLM.
        :type message: str
        :param llm_call_obj: The _LLMCall object holding data on the initiated LLM call.
        :type llm_call_obj: _LLMCall
        :param images: Optional list of Image instances for vision queries.
            If provided, the query will be processed as a vision request.
        :type images: list[Image] | None
        :param num_retries_failed_request: Optional number of retries when LLM request fails. Defaults to 3.
            Note that this parameter may override the value set on the LLM instance to prevent
            accumulation of retries from failed requests and invalid data generation.
        :type num_retries_failed_request: int
        :param max_retries_failed_request: Specific to certain provider APIs (e.g. OpenAI). Optional number of
            retries when LLM request fails. Defaults to 0. This parameter may override the value set on
            the LLM instance to prevent accumulation of retries from failed requests and invalid data generation.
        :type max_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that controls the frequency of
            async LLM API requests, when concurrency is enabled for certain tasks. If not provided,
            such requests will be sent synchronously.
        :type async_limiter: AsyncLimiter | None
        :param drop_params: Whether to drop unsupported parameters when calling the LLM API.
            Used internally for automatic retry when UnsupportedParamsError occurs.
        :type drop_params: bool
        :param raise_exception_on_llm_api_error: Whether to raise an exception if the LLM call fails
            due to an error in the LLM API. If False, a warning will be issued instead, and no data
            will be returned. Defaults to True.
        :type raise_exception_on_llm_api_error: bool, optional
        :return: A tuple containing the LLM response and usage statistics.
            The LLM response is None if the LLM call fails.
        :rtype: tuple[str | None, _LLMUsage]
        """

        if images and not self._supports_vision:
            raise ValueError(
                f"Model `{self.model}` does not support vision according to "
                f"litellm.supports_vision(). To override this detection, "
                f"manually set `_supports_vision=True` on the LLM instance."
            )

        request_messages = []

        # Handle system message based on model type
        if self.system_message and self.system_message.strip():
            if not any(i in self.model for i in ["o1-preview", "o1-mini"]):
                # o1/o1-mini models don't support system/developer messages
                request_messages.append(
                    {
                        "role": "system",
                        "content": self.system_message,
                    }
                )
            else:
                warnings.warn(
                    f"System message ignored for the model `{self.model}`.",
                    stacklevel=2,
                )

        # Prepare user message content based on whether images are provided
        if images:
            user_message_content: list[dict[str, str | dict[str, str]]] = [
                {"type": "text", "text": message}
            ]
            for image in images:
                user_message_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image.mime_type};base64,{image.base64_data}",
                        },
                    },
                )
            request_messages.append(
                {
                    "role": "user",
                    "content": user_message_content,
                }
            )
        else:
            request_messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

        # Validate max input / output tokens before making the API call
        self._validate_input_tokens(request_messages)
        self._validate_output_tokens()

        # Prepare request dictionary with common parameters
        request_dict = {
            "model": self.model,
            "messages": request_messages,
        }

        # Add model-specific parameters
        if self._supports_reasoning:
            # Reasoning (CoT-capable) models
            model_params: list[str] | None = litellm.get_supported_openai_params(  # type: ignore[attr-defined]
                self.model
            )
            if model_params is not None:
                if "max_completion_tokens" in model_params:
                    if not (self.max_completion_tokens):
                        raise ValueError(
                            "`max_completion_tokens` must be set for reasoning (CoT-capable) models"
                        )
                    request_dict["max_completion_tokens"] = self.max_completion_tokens
                if "reasoning_effort" in model_params and self.reasoning_effort:
                    request_dict["reasoning_effort"] = self.reasoning_effort
            if self.temperature or self.top_p:
                logger.info(
                    "`temperature` and `top_p` parameters are ignored for reasoning (CoT-capable) models"
                )
        else:
            # Non-reasoning models
            request_dict["max_tokens"] = self.max_tokens

            # temperature and top_p might not be supported by some models
            if self.temperature is not None:
                request_dict["temperature"] = self.temperature
            if self.top_p is not None:
                request_dict["top_p"] = self.top_p

        if self.deployment_id:
            # Azure OpenAI-specific
            request_dict["deployment_id"] = self.deployment_id

        if self.seed:
            request_dict["seed"] = self.seed

        # Create an empty usage dict in case the call fails without the possibility to retrieve usage tokens
        usage = _LLMUsage()

        # Make API call and process response
        try:
            task = asyncio.create_task(
                litellm.acompletion(
                    **request_dict,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    api_version=self.api_version,
                    num_retries=num_retries_failed_request,
                    max_retries=max_retries_failed_request,
                    timeout=self.timeout,
                    stream=False,
                    drop_params=drop_params,
                )
            )
            if async_limiter:
                async with async_limiter:
                    chat_completion = await task
            else:
                chat_completion = await task
            answer = (
                chat_completion.choices[  # type: ignore[attr-defined]
                    0
                ].message.content  # type: ignore[attr-defined]
            )  # str, or None if invalid response
            usage.input = chat_completion.usage.prompt_tokens  # type: ignore[attr-defined]
            usage.output = chat_completion.usage.completion_tokens  # type: ignore[attr-defined]
            llm_call_obj._record_response_timestamp()
            llm_call_obj.response = answer
            usage.calls.append(llm_call_obj)  # record the call details (call finished)
            return answer, usage
        except litellm.UnsupportedParamsError as e:  # type: ignore[attr-defined]
            # Handle unsupported model parameters error
            if (
                not drop_params
            ):  # only retry if we haven't already tried with drop_params
                logger.error(f"Exception occurred while calling LLM API: {e}")
                logger.info("Retrying the call with unsupported parameters dropped...")

                # Recursively call with drop_params=True
                return await self._query_llm(
                    message=message,
                    llm_call_obj=llm_call_obj,
                    images=images,
                    num_retries_failed_request=num_retries_failed_request,
                    max_retries_failed_request=max_retries_failed_request,
                    async_limiter=async_limiter,
                    drop_params=True,
                    raise_exception_on_llm_api_error=raise_exception_on_llm_api_error,
                )
            else:
                # If drop_params was already True and we still got UnsupportedParamsError,
                # fall through to regular error handling
                warnings.warn(
                    f"Exception occurred while calling LLM API with drop_params=True: {e}",
                    stacklevel=2,
                )
                if self.fallback_llm:
                    logger.info(
                        "Call will be retried if retry params provided and/or a fallback LLM is configured."
                    )
                else:
                    n_retries = max(
                        num_retries_failed_request, max_retries_failed_request
                    )
                    if raise_exception_on_llm_api_error:
                        raise LLMAPIError(
                            "Exception occurred while calling LLM API",
                            retry_count=n_retries,
                            original_error=e,
                        ) from e
                    else:
                        warning_msg = (
                            f"Exception occurred while calling LLM API with drop_params=True: {e}"
                            + f" ({n_retries} retries)."
                        )
                        logger.warning(warning_msg)
                        warnings.warn(
                            warning_msg,
                            stacklevel=2,
                        )
        except Exception as e:
            # e.g. rate limit error
            logger.error(f"Exception occurred while calling LLM API: {e}")
            if self.fallback_llm:
                logger.info(
                    "Call will be retried if retry params provided and/or a fallback LLM is configured."
                )
            else:
                n_retries = max(num_retries_failed_request, max_retries_failed_request)
                if raise_exception_on_llm_api_error:
                    raise LLMAPIError(
                        "Exception occurred while calling LLM API",
                        retry_count=n_retries,
                        original_error=e,
                    ) from e
                else:
                    warning_msg = (
                        f"Exception occurred while calling LLM API: {e}"
                        + f" ({n_retries} retries)."
                    )
                    logger.warning(warning_msg)
                    warnings.warn(
                        warning_msg,
                        stacklevel=2,
                    )

        usage.calls.append(llm_call_obj)  # record the call details (call unfinished)
        return None, usage

    def _set_prompts(self) -> None:
        """
        Sets up prompt templates for various extraction tasks.

        :return: None
        """

        # Templates with placeholders
        # Extraction
        # Safe cast: _get_template with default params returns Template, not str
        self._extract_aspect_items_prompt = cast(
            Template, _get_template("extract_aspect_items")
        )
        self._extract_concept_items_prompt = cast(
            Template, _get_template("extract_concept_items")
        )

    def _set_capabilities(self) -> None:
        """
        Sets the capabilities of the LLM based on litellm.supports_*()
        functions.

        :return: None
        """
        self._supports_vision = litellm.supports_vision(self.model)  # type: ignore[attr-defined]
        self._supports_reasoning = litellm.supports_reasoning(self.model)  # type: ignore[attr-defined]

    def _set_system_message(self) -> None:
        """
        Sets the default system message for the LLM.
        :return: None
        """
        # _get_template returns a Template object when template_extension == "j2"
        self.system_message = _get_template(
            "default_system_message",
            template_type="system",
            template_extension="j2",
        ).render({"output_language": self.output_language})  # type: ignore[attr-defined]

    def _get_raw_usage(self) -> _LLMUsage:
        """
        Retrieves the raw usage information of the LLM.

        :return: _LLMUsage object containing usage data for the LLM.
        """

        return self._usage

    def _get_raw_cost(self) -> _LLMCost:
        """
        Retrieves the cost information of the LLM, quantized for reporting.

        The internal accumulators keep full precision. Quantization is applied
        only on access to ensure stable presentation while avoiding stepwise
        rounding errors during accumulation.

        :return: _LLMCost object containing quantized cost data for the LLM.
        """
        if self.pricing_details is None and not self.auto_pricing:
            logger.info(
                f"No pricing params provided for the LLM `{self.model}` "
                f"with role `{self.role}`. Costs for this LLM were not calculated."
            )

        # Quantize for presentation without mutating internal state
        input_q = self._cost.input.quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        output_q = self._cost.output.quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        total_q = (input_q + output_q).quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        return _LLMCost(input=input_q, output=output_q, total=total_q)

    def get_usage(self) -> list[_LLMUsageOutputContainer]:
        """
        Retrieves the usage information of the LLM and its fallback LLM if configured.

        This method collects token usage statistics for the current LLM instance and its
        fallback LLM (if configured), providing insights into API consumption.

        :return: A list of usage statistics containers for the LLM and its fallback.
        :rtype: list[_LLMUsageOutputContainer]
        """

        # Safe cast: _get_usage_or_cost with retrieval_type="usage"
        # returns only list of _LLMUsageOutputContainer
        return cast(
            list["_LLMUsageOutputContainer"],
            self._get_usage_or_cost(retrieval_type="usage"),
        )

    def get_cost(self) -> list[_LLMCostOutputContainer]:
        """
        Retrieves the accumulated cost information of the LLM and its fallback LLM if configured.

        This method collects cost statistics for the current LLM instance and its
        fallback LLM (if configured), providing insights into API usage expenses.

        :return: A list of cost statistics containers for the LLM and its fallback.
        :rtype: list[_LLMCostOutputContainer]
        """

        # Safe cast: _get_usage_or_cost with retrieval_type="cost"
        # returns only list of _LLMCostOutputContainer
        return cast(
            list["_LLMCostOutputContainer"],
            self._get_usage_or_cost(retrieval_type="cost"),
        )

    def reset_usage_and_cost(self) -> None:
        """
        Resets the usage and cost statistics for the LLM and its fallback LLM (if configured).

        This method clears accumulated usage and cost data, which is useful when processing
        multiple documents sequentially and tracking metrics for each document separately.

        :return: None
        """

        for llm in [self, self.fallback_llm]:
            if llm:
                llm._usage = _LLMUsage()
                llm._cost = _LLMCost()

    def _calculate_auto_pricing_costs(
        self, input_tokens: int, output_tokens: int
    ) -> tuple[Decimal, Decimal]:
        """
        Utility method to calculate costs using genai-prices for the given
        input/output token counts. Requires `_auto_pricing_provider_id` and
        `_auto_pricing_model_ref` to be set.

        :param input_tokens: Number of input tokens
        :param output_tokens: Number of output tokens
        :return: Tuple of (input_cost, output_cost) as Decimals
        :raises LookupError: If auto-pricing provider/model are not set
        :raises Exception: Propagates exceptions from genai-prices lookup/calculation
        """

        if (
            self._auto_pricing_provider_id is None
            or self._auto_pricing_model_ref is None
        ):
            raise LookupError(
                "Auto-pricing provider/model are not set for this LLM instance"
            )

        if self.auto_pricing_refresh and not self._auto_pricing_refresh_attempted:
            # Attempt price update only once, as it involves a network call
            self._auto_pricing_refresh_attempted = True
            with _GPUpdatePrices() as update_prices:
                logger.info(
                    f"Updating LLM API prices since `auto_pricing_refresh=True` for model `{self.model}`"
                )
                try:
                    update_prices.wait()
                    logger.info("Finished updating LLM API prices")
                except Exception as e:
                    logger.error(f"Error updating LLM API prices: {e}.")
                price_data = _calc_auto_price(
                    _GPUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                    model_ref=self._auto_pricing_model_ref,
                    provider_id=self._auto_pricing_provider_id,
                )
        else:
            price_data = _calc_auto_price(
                _GPUsage(input_tokens=input_tokens, output_tokens=output_tokens),
                model_ref=self._auto_pricing_model_ref,
                provider_id=self._auto_pricing_provider_id,
            )
        input_cost = Decimal(str(price_data.input_price))
        output_cost = Decimal(str(price_data.output_price))
        return input_cost, output_cost

    def _increment_cost(self, usage: _LLMUsage) -> None:
        """
        Calculates and increments the self._cost attribute values based on
        the additional usage details provided. Relevant only if the user has
        provided pricing details for the LLM or if auto-pricing is enabled.

        :param usage: _LLMUsage instance containing usage information on
                      additional number of input and output tokens processed.
        :type usage: _LLMUsage

        :return: None
        """

        if self.pricing_details or self.auto_pricing:
            mil_dec = Decimal("1000000")
            cost_input: Decimal = Decimal("0")
            cost_output: Decimal = Decimal("0")

            if self.pricing_details:
                cost_input = (Decimal(str(usage.input)) / mil_dec) * Decimal(
                    str(self.pricing_details.input_per_1m_tokens)
                )
                cost_output = (Decimal(str(usage.output)) / mil_dec) * Decimal(
                    str(self.pricing_details.output_per_1m_tokens)
                )

            elif self.auto_pricing:
                # Use genai-prices to calculate input/output costs
                if (
                    self._auto_pricing_provider_id is None
                    or self._auto_pricing_model_ref is None
                ):
                    # If not set, skip cost update
                    return
                try:
                    cost_input, cost_output = self._calculate_auto_pricing_costs(
                        usage.input, usage.output
                    )
                    if not isinstance(cost_input, Decimal) or not isinstance(
                        cost_output, Decimal
                    ):
                        raise RuntimeError("Auto-pricing returned non-Decimal values")
                except Exception as e:
                    warning_msg = (
                        f"Unable to fetch pricing data for model `{self.model}`: {e}. "
                        "Auto-pricing is skipped. "
                        "Consider setting `pricing_details=LLMPricing(...)` "
                        "or disabling auto-pricing."
                    )
                    warnings.warn(
                        warning_msg,
                        stacklevel=2,
                    )

            cost_total = cost_input + cost_output

            # Accumulate with full precision; quantization happens on read
            self._cost.input += cost_input
            self._cost.output += cost_output
            self._cost.total += cost_total

    async def _update_usage_and_cost(
        self, result: tuple[Any, _LLMUsage] | None
    ) -> None:
        """
        Updates the LLM usage and cost details based on the given processing result.
        This method  modifies the LLM instance's usage statistics and increments the associated
        cost if pricing params are provided.

        :param result: A tuple containing an optional value and usage data. The usage
            data is used to update the instance's input and output usage, as well as
            the total cost. If the result is None, the method does nothing.
        :type result: tuple[Any, _LLMUsage]
        :return: None
        """
        async with self._async_lock:
            if result is None:
                return
            new_usage = result[1]
            # Pricing data
            if self.pricing_details or self.auto_pricing:
                self._usage.input += new_usage.input
                self._usage.output += new_usage.output
                self._increment_cost(new_usage)
            # Calls data
            self._usage.calls += new_usage.calls

    def _set_provider_and_model_for_auto_pricing(self) -> None:
        """
        Sets (provider_id, model_ref) pair for genai-prices from `model` field.
        Does nothing when local models are detected.

        :return: None
        :rtype: None
        """

        if any(self.model.startswith(provider) for provider in _LOCAL_MODEL_PROVIDERS):
            # Local models are not supported for auto-pricing
            return

        # Derive from self.model formatted as "provider/model"
        parts = self.model.split("/", 1)
        if len(parts) != 2:
            warning_msg = (
                f"Could not derive auto-pricing provider/model from `{self.model}`. "
                f"Auto-pricing will be skipped. "
                f"Consider setting `pricing_details=LLMPricing(...)` "
                f"or disabling auto-pricing."
            )
            warnings.warn(
                warning_msg,
                stacklevel=2,
            )
            return

        provider_id = parts[0]
        model_ref = parts[1]
        self._auto_pricing_provider_id = provider_id
        self._auto_pricing_model_ref = model_ref

        # Probe calculation with dummy tokens.
        try:
            self._calculate_auto_pricing_costs(1000, 25000)
        except Exception as e:
            warning_msg = (
                f"Unable to fetch pricing data for model `{self.model}`: {e}. "
                "Auto-pricing will be skipped. "
                "Consider setting `pricing_details=LLMPricing(...)` "
                "or disabling auto-pricing."
            )
            warnings.warn(
                warning_msg,
                stacklevel=2,
            )
