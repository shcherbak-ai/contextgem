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

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal, Optional

from aiolimiter import AsyncLimiter
from pydantic import ConfigDict

from contextgem.internal.base.serialization import _InstanceSerializer

if TYPE_CHECKING:
    from contextgem.public.paragraphs import Paragraph
    from contextgem.public.images import Image
    from contextgem.public.llms import DocumentLLM, DocumentLLMGroup
    from contextgem.internal.data_models import _LLMCost
    from contextgem.internal.items import _ExtractedItem

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.mixins import _PostInitCollectorMixin
from contextgem.internal.data_models import (
    _LLMCall,
    _LLMCostOutputContainer,
    _LLMUsage,
    _LLMUsageOutputContainer,
)
from contextgem.internal.decorators import _timer_decorator
from contextgem.internal.items import _StringItem
from contextgem.internal.loggers import logger
from contextgem.internal.typings.aliases import (
    ExtractedInstanceType,
    JustificationDepth,
    LLMRoleAny,
    ReferenceDepth,
)
from contextgem.internal.utils import (
    _async_multi_executor,
    _chunk_list,
    _clean_text_for_llm_prompt,
    _group_instances_by_fields,
    _llm_call_result_is_valid,
    _parse_llm_output_as_json,
    _remove_thinking_content_from_llm_output,
    _run_async_calls,
    _run_sync,
    _validate_parsed_llm_output,
)
from contextgem.public.aspects import Aspect
from contextgem.public.documents import Document


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
        document: Document,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> Document:
        """
        Extracts all aspects and concepts from a document and its aspects.

        This method performs comprehensive extraction by processing the document for aspects
        and concepts, then extracting concepts from each aspect. The operation can be
        configured for concurrent processing and customized extraction parameters.

        This is the synchronous version of `extract_all_async()`.

        :param document: The document to analyze.
        :type document: Document
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
        :return: The document with extracted aspects and concepts.
        :rtype: Document
        """
        return _run_sync(
            self.extract_all_async(
                document=document,
                overwrite_existing=overwrite_existing,
                max_items_per_call=max_items_per_call,
                use_concurrency=use_concurrency,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                max_images_to_analyze_per_call=max_images_to_analyze_per_call,
            )
        )

    @_timer_decorator(process_name="All aspects and concepts extraction")
    async def extract_all_async(
        self,
        document: Document,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> Document:
        """
        Asynchronously extracts all aspects and concepts from a document and its aspects.

        This method performs comprehensive extraction by processing the document for aspects
        and concepts, then extracting concepts from each aspect. The operation can be
        configured for concurrent processing and customized extraction parameters.

        :param document: The document to analyze.
        :type document: Document
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
        :return: The document with extracted aspects and concepts.
        :rtype: Document
        """

        self._check_llm_roles_before_extract_all(document)

        # Extract all aspects in the document
        await self.extract_aspects_from_document_async(
            document=document,
            overwrite_existing=overwrite_existing,
            max_items_per_call=max_items_per_call,
            use_concurrency=use_concurrency,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        extract_concepts_kwargs = {
            "document": document,
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
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
        cals_and_kwargs = aspect_cals_and_kwargs + doc_concepts_cals_and_kwargs
        await _run_async_calls(
            cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
        )

        return document

    def extract_aspects_from_document(
        self,
        document: Document,
        from_aspects: Optional[list[Aspect]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> list[Aspect]:
        """
        Extracts aspects from the provided document using predefined LLMs.

        If an aspect instance has ``extracted_items`` populated, the ``reference_paragraphs`` field will be
        automatically populated from these items.

        This is the synchronous version of `extract_aspects_from_document_async()`.

        :param document: The document from which aspects are to be extracted.
        :type document: Document
        :param from_aspects: Existing aspects to use as a base for extraction. If None, uses all
            document's aspects.
        :type from_aspects: Optional[list[Aspect]]
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
        :return: List of processed Aspect objects with extracted items.
        :rtype: list[Aspect]
        """
        return _run_sync(
            self.extract_aspects_from_document_async(
                document=document,
                from_aspects=from_aspects,
                overwrite_existing=overwrite_existing,
                max_items_per_call=max_items_per_call,
                use_concurrency=use_concurrency,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            )
        )

    @_timer_decorator(process_name="Aspects extraction from document")
    async def extract_aspects_from_document_async(
        self,
        document: Document,
        from_aspects: Optional[list[Aspect]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> list[Aspect]:
        """
        Extracts aspects from the provided document using predefined LLMs asynchronously.

        If an aspect instance has ``extracted_items`` populated, the ``reference_paragraphs`` field will be
        automatically populated from these items.

        :param document: The document from which aspects are to be extracted.
        :type document: Document
        :param from_aspects: Existing aspects to use as a base for extraction. If None, uses all
            document's aspects.
        :type from_aspects: Optional[list[Aspect]]
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
        :return: List of processed Aspect objects with extracted items.
        :rtype: list[Aspect]
        """

        self._check_instances_and_llm_params(
            target=document,
            llm_or_group=self,
            instances_to_process=from_aspects,
            instance_type="aspect",
            overwrite_existing=overwrite_existing,
        )

        extract_instances_kwargs = {
            "context": document,
            "instance_type": "aspect",
            "document": document,
            "from_instances": from_aspects,
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
        }

        if self.is_group:
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in self.llms
            ]
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            await self._extract_instances(**extract_instances_kwargs, llm=self)

        document_aspects = document.aspects if not from_aspects else from_aspects

        # Extract sub-aspects
        extract_sub_aspects_kwargs = {
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
        }
        for aspect in document_aspects:
            if aspect.aspects:
                # Validate proper nesting level of sub-aspects
                self._validate_nesting_level(aspect, aspect.aspects, "sub-aspects")
                logger.info(f"Extracting sub-aspects for aspect `{aspect.name}`")
                if not aspect.reference_paragraphs:
                    logger.info(
                        f"Aspect `{aspect.name}` has no extracted paragraphs. "
                        f"Sub-aspects will not be extracted."
                    )
                    continue
                # Treat an aspect as a document containing sub-aspects
                aspect_document = Document(
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
        aspect: Aspect,
        document: Document,
        from_concepts: Optional[list[_Concept]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> list[_Concept]:
        """
        Extracts concepts associated with a given aspect in a document.

        This method processes an aspect to extract related concepts using LLMs.
        If the aspect has not been previously processed, a ValueError is raised.

        This is the synchronous version of `extract_concepts_from_aspect_async()`.

        :param aspect: The aspect from which to extract concepts.
        :type aspect: Aspect
        :param document: The document that contains the aspect.
        :type document: Document
        :param from_concepts: List of existing concepts to process. Defaults to None.
        :type from_concepts: Optional[list[_Concept]]
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
            )
        )

    @_timer_decorator(process_name="Concept extraction from aspect")
    async def extract_concepts_from_aspect_async(
        self,
        aspect: Aspect,
        document: Document,
        from_concepts: Optional[list[_Concept]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> list[_Concept]:
        """
        Asynchronously extracts concepts from a specified aspect using LLMs.

        This method processes an aspect to extract related concepts using LLMs.
        If the aspect has not been previously processed, a ValueError is raised.

        :param aspect: The aspect from which to extract concepts.
        :type aspect: Aspect
        :param document: The document that contains the aspect.
        :type document: Document
        :param from_concepts: List of existing concepts to process. Defaults to None.
        :type from_concepts: Optional[list[_Concept]]
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
        :return: List of processed concept objects.
        :rtype: list[_Concept]
        """

        self._check_instances_and_llm_params(
            target=aspect,
            llm_or_group=self,
            instances_to_process=from_concepts,
            instance_type="concept",
            overwrite_existing=overwrite_existing,
        )

        if not aspect._is_processed:
            assert (
                not aspect.extracted_items
            ), "Aspect is not marked as processed, yet it has extracted items."
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
        }

        if self.is_group:
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in self.llms
            ]
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            await self._extract_instances(**extract_instances_kwargs, llm=self)

        # Extract concepts from sub-aspects
        extract_concepts_from_sub_aspects_kwargs = {
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
        }
        if aspect.aspects:
            # Validate proper nesting level of sub-aspects
            self._validate_nesting_level(aspect, aspect.aspects, "sub-aspects")
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
                sub_aspect_document = Document(
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

        return aspect.concepts if not from_concepts else from_concepts

    def extract_concepts_from_document(
        self,
        document: Document,
        from_concepts: Optional[list[_Concept]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> list[_Concept]:
        """
        Extracts concepts from the provided document using predefined LLMs.

        This is the synchronous version of `extract_concepts_from_document_async()`.

        :param document: The document from which concepts are to be extracted.
        :type document: Document
        :param from_concepts: Existing concepts to use as a base for extraction. If None, uses all
            document's concepts.
        :type from_concepts: Optional[list[_Concept]]
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
            )
        )

    @_timer_decorator(process_name="Concepts extraction from document")
    async def extract_concepts_from_document_async(
        self,
        document: Document,
        from_concepts: Optional[list[_Concept]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> list[_Concept]:
        """
        Extracts concepts from the provided document using predefined LLMs asynchronously.

        This method processes a document to extract concepts using configured LLMs.

        :param document: The document from which concepts are to be extracted.
        :type document: Document
        :param from_concepts: Existing concepts to use as a base for extraction. If None, uses all
            document's concepts.
        :type from_concepts: Optional[list[_Concept]]
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
        :return: List of processed Concept objects with extracted items.
        :rtype: list[_Concept]
        """

        self._check_instances_and_llm_params(
            target=document,
            llm_or_group=self,
            instances_to_process=from_concepts,
            instance_type="concept",
            overwrite_existing=overwrite_existing,
        )

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
        }

        if self.is_group:
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in self.llms
            ]
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            await self._extract_instances(**extract_instances_kwargs, llm=self)

        return document.concepts if not from_concepts else from_concepts

    def _check_llm_roles_before_extract_all(
        self,
        document: Document,
    ) -> None:
        """
        Checks if all assigned LLM roles in the given document are present in the set
        of LLM roles of the current LLM / LLM group instance. If there are missing roles,
        a warning is logged. This process helps to check for completeness in data extraction
        when full extraction method extract_all() is called.

        :param document: The document object to check for LLM role assignments.
        :type document: Document
        :return: None
        :rtype: None
        """

        if self.is_group:
            llm_roles = {i.role for i in self.llms}
        else:
            llm_roles = {self.role}
        missing_llm_roles = document.llm_roles.difference(llm_roles)
        if missing_llm_roles:
            warnings.warn(
                f"Document contains elements with LLM roles that are not found "
                f"in the current {'LLM group' if self.is_group else 'LLM'}: "
                f"{'LLM group roles' if self.is_group else 'LLM role'} {llm_roles}, "
                f"missing LLM roles {missing_llm_roles}. "
                f"Such elements will be ignored."
            )

    def _check_instances_and_llm_params(
        self,
        target: Document | Aspect,
        llm_or_group: DocumentLLM | DocumentLLMGroup,
        instances_to_process: list[Aspect | _Concept] | None,
        instance_type: ExtractedInstanceType,
        overwrite_existing: bool = False,
    ) -> None:
        """
        Validates instances and LLM parameters, ensuring compatibility with the target
        and configurations provided.

        :param target: The target object, which should have an attribute corresponding
            to `instance_type` (e.g., 'aspects' for 'aspect', etc.). Expected to be an
            instance of either Document or Aspect.
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

        # Check instances
        instance_class_map = {
            "aspect": Aspect,
            "concept": _Concept,
        }
        # Retrieve class or raise error
        instance_class = instance_class_map.get(instance_type, None)
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
            if not llm_or_group.llms:
                raise ValueError(
                    "The provided DocumentLLMGroup does not contain any defined LLMs."
                )

        # Check DocumentLLM
        else:
            # Inform about inconsistent LLM roles
            if any(i.llm_role != llm_or_group.role for i in check_instances):
                logger.warning(
                    f"Some {instance_type}s rely on the LLM with a role different "
                    f"than the current LLM's role `{llm_or_group.role}`. "
                    f"This LLM will not extract such {instance_type}s."
                )

    @staticmethod
    def _check_instances_already_processed(
        instance_type: ExtractedInstanceType,
        instances: list[Aspect | _Concept],
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
        source: Document | Aspect,
        llm: DocumentLLM,
        instances_to_process: list[Aspect] | list[_Concept],
        document: Document,
        add_justifications: bool = False,
        justification_depth: JustificationDepth = "brief",
        justification_max_sents: int = 2,
        add_references: bool = False,
        reference_depth: ReferenceDepth = "paragraphs",
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> list[dict[str, dict | list[Paragraph | Image]]]:
        """
        Prepares the list of the message kwargs required for querying a LLM to extract aspects or concepts
        from a document or its associated entities like aspects.

        Each item on the list is a kwargs dict for each extraction, based on specific context chunks.
        E.g. if the text has 60 paragraphs, and `max_paragraphs_to_analyze_per_call` is set to 15, then
        each extraction type that uses all document paragraphs as context will have 4 items of kwargs
        dicts in the list, and the context of each message will include 15 paragraphs.

        :param extracted_instance_type: A string literal indicating the type of extracted instance.
            Accepted values are "aspect" and "concept".
        :param source: The input source, which can either be a `Document` or an `Aspect`
            instance, from which aspects or concepts are extracted.
        :param llm: The LLM instance used for extraction.
        :param instances_to_process: A list of instances (either `Aspect` or `_Concept` instances)
            to process for extraction tasks.
        :param document: An instance of `Document` containing paragraphs or text or images,
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
        """

        # Validate source data for extraction and set the extraction level
        def validate_source() -> str:
            """
            Validates the source based on the extracted instance type.
            """

            # Aspect (document-level)
            if extracted_instance_type == "aspect":
                if not (source.raw_text or source.paragraphs):
                    raise ValueError(
                        "Document lacks text or paragraphs for aspect extraction."
                    )
                return "aspect_document_text"

            # Concept (document- and aspect-levels)
            if extracted_instance_type == "concept":
                # Concept (document-level)
                if isinstance(source, Document):
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
                if isinstance(source, Aspect):
                    return "concept_aspect_text"

            raise ValueError(
                f"Unsupported extracted item type: `{extracted_instance_type}`"
            )

        extraction_level = validate_source()
        message_kwargs_list = []

        # Text-based extraction
        if extraction_level.endswith("_text"):
            paragraphs = (
                document.paragraphs
                if extraction_level in {"aspect_document_text", "concept_document_text"}
                else source.reference_paragraphs
            )
            max_paras_per_call = (
                min(len(paragraphs), max_paragraphs_to_analyze_per_call)
                if max_paragraphs_to_analyze_per_call
                else len(paragraphs)
            )
            paragraphs_chunks = _chunk_list(paragraphs, max_paras_per_call)
            logger.debug(f"Processing {max_paras_per_call} paragraphs per LLM call.")

            for paragraphs_chunk in paragraphs_chunks:
                assert len(paragraphs_chunk)

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
                    }

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
                    }
                    if add_references:
                        # List of document/aspect paragraphs used for concept extraction with references
                        prompt_kwargs["paragraphs"] = paragraphs_chunk
                    else:
                        # Raw text of document/aspect used for concept extraction
                        if isinstance(source, Document) and len(
                            paragraphs_chunk
                        ) == len(source.paragraphs):
                            # If the document is being processed as a whole, use the raw text of the document,
                            # which can be markdown (if converted from DOCX) or raw text.
                            prompt_kwargs["text"] = _clean_text_for_llm_prompt(
                                source.raw_text,
                                preserve_linebreaks=True,
                            )  # markdown or raw text of the document
                        else:
                            # If an aspect is being processed, or if the document is being processed in chunks,
                            # use the raw text of the paragraphs, which is cleaned of markdown and other formatting.
                            prompt_kwargs["text"] = "\n\n".join(
                                [
                                    _clean_text_for_llm_prompt(
                                        p.raw_text, preserve_linebreaks=False
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

            max_images_per_call = (
                min(len(source.images), max_images_to_analyze_per_call)
                if max_images_to_analyze_per_call
                else len(source.images)
            )
            images_chunks = _chunk_list(source.images, max_images_per_call)
            logger.debug(f"Processing {max_images_per_call} images per LLM call.")

            for images_chunk in images_chunks:
                assert len(images_chunk)

                # Concept (from images)
                if extraction_level == "concept_document_vision":
                    prompt_kwargs = {
                        "concepts": instances_to_process,
                        "add_justifications": add_justifications,
                        "justification_depth": justification_depth,
                        "justification_max_sents": justification_max_sents,
                        "data_type": "image",
                        "output_language": llm.output_language,
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
        source: Document | Aspect,
        llm: DocumentLLM,
        instances_to_process: list[Aspect] | list[_Concept],
        document: Document,
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
    ) -> tuple[list[Aspect] | list[_Concept] | None, _LLMUsage]:
        """
        Extracts items from either aspects or concepts using a specified LLM.
        This unified method handles extraction from both documents and aspects,
        supporting both text and vision LLMs.

        :param extracted_item_type: Type of the item(s) being extracted ("aspect" or "concept")
        :type extracted_item_type: ExtractedInstanceType
        :param source: The source to extract from (Document or Aspect)
        :type source: Document | Aspect
        :param llm: The LLM used for extraction
        :type llm: DocumentLLM
        :param instances_to_process: List of aspects or concepts to process
        :type instances_to_process: list[Aspect] | list[_Concept]
        :param document: The document containing the source.
        :type document: Optional[Document]
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

        :return: A tuple containing:
            (0) List of processed instances with extracted items, or None if LLM processing fails
            (1) _LLMUsage instance with LLM usage information
        :rtype: tuple[list[Aspect] | list[_Concept] | None, _LLMUsage]
        """

        def validate_source_in_document(
            source: Document | Aspect, document: Document
        ) -> None:
            """
            Raises ValueError if an Aspect is not assigned to the given Document.
            """
            if isinstance(source, Aspect) and source not in document.aspects:
                raise ValueError(
                    f"Aspect `{source.name}` must be assigned to the document"
                )

        def validate_vision_llm_usage(
            extracted_item_type: ExtractedInstanceType,
            llm: DocumentLLM,
            source: Document | Aspect,
            add_references: bool,
        ) -> None:
            """
            Raises ValueError if the LLM role is vision-based but the extraction type/source is unsupported.
            """
            if llm.role.endswith("_vision"):
                # Vision LLM: only document-level concept extraction is valid
                if extracted_item_type == "aspect" or (
                    extracted_item_type == "concept" and isinstance(source, Aspect)
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
            extracted_item_type: ExtractedInstanceType, source: Document | Aspect
        ) -> None:
            """
            If extracting concepts from an Aspect, validates that the aspect's extracted items have
            references.
            """
            if extracted_item_type == "concept" and isinstance(source, Aspect):
                if source.extracted_items:
                    if not source.reference_paragraphs or not all(
                        p.sentences for p in source.reference_paragraphs
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
            and isinstance(source, Aspect)
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
            str, dict[str, Aspect | _Concept | list[_ExtractedItem]]
        ] = {}
        instances_enumerated = dict(enumerate(instances_to_process))

        # Each item on the list is a message kwargs dict for each extraction, based on specific context chunks.
        for idx, message_kwargs in enumerate(message_kwargs_list):
            logger.debug(
                f"Processing messages chunk {idx + 1}/{len(message_kwargs_list)}"
            )
            # Extract paragraphs and remove them from message_kwargs
            paragraphs_chunk: list[Paragraph] = message_kwargs.pop(
                "paragraphs_chunk", None
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
                        and i.singular_occurrence
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
                message_kwargs["prompt_kwargs"][
                    "concepts"
                ] = filtered_instances_to_process
                assert not any(
                    i in filtered_instances_to_process for i in discarded_instances
                )
                logger.debug(
                    f"Total {extracted_item_type}s discarded: {len(discarded_instances)}"
                )
                # Re-enumerate the new instances to process, to match them against the indices in LLM responses
                instances_enumerated = dict(enumerate(filtered_instances_to_process))
                # Do not overwrite instances_to_process as we need to return all of the originally passed instances
                # even if they are discarded during one of the loop iterations

            # Construct a prompt from the message kwargs
            if extracted_item_type == "aspect":
                message = llm._extract_aspect_items_prompt.render(
                    **message_kwargs["prompt_kwargs"]
                )
            elif extracted_item_type == "concept":
                message = llm._extract_concept_items_prompt.render(
                    **message_kwargs["prompt_kwargs"]
                )
            else:
                raise ValueError(
                    f"Unsupported extracted item type: `{extracted_item_type}`"
                )

            message_kwargs["message"] = message
            # Initialize the LLM call object to pass to the LLM query method
            message_kwargs["llm_call_obj"] = _LLMCall(
                prompt_kwargs=message_kwargs["prompt_kwargs"],
                prompt=message,
            )
            del message_kwargs["prompt_kwargs"]

            # Query LLM, process and validate results
            extracted_data, usage_data = await llm._query_llm(
                **message_kwargs,
                num_retries_failed_request=num_retries_failed_request,
                max_retries_failed_request=max_retries_failed_request,
                async_limiter=async_limiter,
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
                    except KeyError:
                        logger.error("Aspect ID returned by LLM is invalid")
                        return None, all_usage_data
                    self._check_instances_already_processed(
                        instance_type=extracted_item_type,
                        instances=[relevant_aspect],
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
                        for para_dict in aspect_dict["paragraphs"]:
                            para_dict["paragraph_id"] = int(
                                para_dict["paragraph_id"].lstrip("P")
                            )
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
                                    sources_mapper[relevant_aspect.unique_id][
                                        "extracted_items"
                                    ].append(extracted_item)
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
                                sources_mapper[relevant_aspect.unique_id][
                                    "extracted_items"
                                ].append(extracted_item)
                    else:
                        # Reference depth - paragraph
                        # Each extracted item will have the reference paragraph text as value,
                        # reference paragraph in the list of reference paragraphs, and no reference sentences
                        aspect_dict["paragraph_ids"] = [
                            int(para_id.lstrip("P"))
                            for para_id in aspect_dict["paragraph_ids"]
                        ]
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
                            sources_mapper[relevant_aspect.unique_id][
                                "extracted_items"
                            ].append(extracted_item)

            # Concept (document- and aspect-levels)
            elif extracted_item_type == "concept":
                for concept_dict in extracted_data:
                    try:
                        relevant_concept = instances_enumerated[
                            int(concept_dict["concept_id"].lstrip("C"))
                        ]
                    except KeyError:
                        logger.error(f"Concept ID returned by LLM is invalid")
                        return None, all_usage_data
                    self._check_instances_already_processed(
                        instance_type=extracted_item_type,
                        instances=[relevant_concept],
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
                            i["value"] = relevant_concept._process_item_value(
                                i["value"]
                            )
                            concept_extracted_item_kwargs = {"value": i["value"]}
                            if add_justifications:
                                concept_extracted_item_kwargs["justification"] = i[
                                    "justification"
                                ]
                            if add_references:
                                concept_extracted_item_kwargs[
                                    "reference_paragraphs"
                                ] = []
                                concept_extracted_item_kwargs["reference_sentences"] = (
                                    []
                                )
                                # Reference depth - sentence
                                # Each extracted item will have reference paragraph in the list of
                                # reference paragraphs, and reference sentence in the list of reference sentences
                                if reference_depth == "sentences":
                                    reference_paragraphs_list = i[
                                        "reference_paragraphs"
                                    ]
                                    for para_dict in reference_paragraphs_list:
                                        para_dict["reference_paragraph_id"] = int(
                                            para_dict["reference_paragraph_id"].lstrip(
                                                "P"
                                            )
                                        )
                                        para_dict["reference_sentence_ids"] = [
                                            int(i.split("-S")[-1])
                                            for i in para_dict["reference_sentence_ids"]
                                        ]
                                    reference_paragraphs_list = sorted(
                                        reference_paragraphs_list,
                                        key=lambda x: x["reference_paragraph_id"],
                                    )
                                # Reference depth - paragraph
                                # Each extracted item will have reference paragraph in the list of
                                # reference paragraphs, and no reference sentences
                                else:
                                    reference_paragraphs_list = sorted(
                                        [
                                            int(para_id.lstrip("P"))
                                            for para_id in i["reference_paragraph_ids"]
                                        ]
                                    )
                                # Reference depth - paragraph or sentence
                                for para_obj_or_id in reference_paragraphs_list:
                                    try:
                                        if reference_depth == "sentences":
                                            para_id = para_obj_or_id[
                                                "reference_paragraph_id"
                                            ]
                                        else:
                                            para_id = para_obj_or_id
                                        ref_para = paragraphs_enumerated[para_id]
                                    except KeyError:
                                        logger.error(
                                            f"Reference paragraph ID returned by LLM is invalid"
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
                                        reference_sentence_ids = sorted(
                                            para_obj_or_id["reference_sentence_ids"]
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
                            extracted_item = relevant_concept._item_class(
                                **concept_extracted_item_kwargs
                            )
                            extracted_item.reference_paragraphs = reference_paragraphs
                            extracted_item.reference_sentences = reference_sentences
                            sources_mapper[relevant_concept.unique_id][
                                "extracted_items"
                            ].append(extracted_item)
                    else:
                        for i in concept_dict["extracted_items"]:
                            # Process the item value with a custom function on the concept
                            i = relevant_concept._process_item_value(i)
                            sources_mapper[relevant_concept.unique_id][
                                "extracted_items"
                            ].append(relevant_concept._item_class(value=i))

            else:
                raise ValueError(
                    f"Unsupported extracted item type: `{extracted_item_type}`"
                )

        # Process all gathered results for all processed instances
        async with (
            llm._async_lock
        ):  # ensure atomicity of the instances' state check and modification
            for source_id, source_data in sources_mapper.items():
                source_instance = source_data["source"]
                self._check_instances_already_processed(
                    instance_type=extracted_item_type,
                    instances=[source_instance],
                    overwrite_existing=overwrite_existing,
                )
                source_instance.extracted_items = source_data["extracted_items"]
                if extracted_item_type == "aspect":
                    # References are automatically included for aspects, as paragraphs/sents'
                    # texts are the values of extracted items for the aspects
                    for ei in source_instance.extracted_items:
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
        context: Document | Aspect,
        llm: DocumentLLM,
        instance_type: ExtractedInstanceType,
        document: Document,
        from_instances: Optional[list[Aspect] | list[_Concept]] = None,
        overwrite_existing: bool = False,
        max_items_per_call: int = 0,
        use_concurrency: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> None:
        """
        Extracts aspects or concepts from a context (document or aspect) using a specified LLM.

        :param context: The context (document or aspect) from which to extract instances
        :type context: Document | Aspect
        :param llm: The LLM used for extraction
        :type llm: DocumentLLM
        :param instance_type: Type of instance to extract ("aspect" or "concept")
        :type instance_type: ExtractedInstanceType
        :param document: The document object associated with the context.
        :type document: Document
        :param from_instances: List of specific instances to process. If None, all instances are processed
        :type from_instances: Optional[list[Aspect] | list[_Concept]]
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
            instances_to_process: list[Aspect] | list[_Concept],
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
                llm_instance: DocumentLLM,
                res: tuple[list[Aspect] | list[_Concept] | None, _LLMUsage],
                instances: list[Aspect] | list[_Concept],
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
                if not n_retries:
                    return False
                if not _llm_call_result_is_valid(res):
                    for i in range(n_retries):
                        logger.info(
                            f"Retrying {instance_type}s with invalid JSON "
                            f"({i+1}/{n_retries})"
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
                        )
                        # Update usage stats and cost
                        await llm_instance._update_usage_and_cost(res)
                        if res[0] is not None:
                            break
                    if _llm_call_result_is_valid(res):
                        return True
                    else:
                        if retry_is_final:
                            warnings.warn(
                                f"Some {instance_type}s could not be processed due to invalid JSON returned by LLM."
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
                assert len(results) == len(data_chunks)

                # Update usage stats and cost
                for result in results:
                    await llm._update_usage_and_cost(result)

                # Retry failed chunks if needed
                if any(not _llm_call_result_is_valid(result) for result in results):
                    for chunk, result in zip(data_chunks, results):
                        retry_successful = await retry_processing_for_result(
                            llm_instance=llm,
                            res=result,
                            instances=chunk,
                            n_retries=llm.max_retries_invalid_data,
                            retry_is_final=False if llm.fallback_llm else True,
                        )
                        # Retry with fallback LLM if it is provided
                        if not retry_successful and llm.fallback_llm:
                            logger.info("Trying with fallback LLM")
                            await retry_processing_for_result(
                                llm_instance=llm.fallback_llm,
                                res=result,
                                instances=chunk,
                                n_retries=max(
                                    llm.fallback_llm.max_retries_invalid_data, 1
                                ),  # retry with fallback LLM at least once
                                retry_is_final=True,
                            )
            else:
                # Process sequentially
                for i, data in enumerate(data_list):
                    result = await self._extract_items_from_instances(**data)
                    logger.debug(
                        f"Result for chunk ({i+1}/{len(data_list)}) processed."
                    )

                    # Update usage stats and cost
                    await llm._update_usage_and_cost(result)

                    # Retry if needed
                    if not _llm_call_result_is_valid(result):
                        retry_successful = await retry_processing_for_result(
                            llm_instance=llm,
                            res=result,
                            instances=data["instances_to_process"],
                            n_retries=llm.max_retries_invalid_data,
                            retry_is_final=False if llm.fallback_llm else True,
                        )
                        # Retry with fallback LLM if it is provided
                        if not retry_successful and llm.fallback_llm:
                            logger.info("Trying with fallback LLM")
                            await retry_processing_for_result(
                                llm_instance=llm.fallback_llm,
                                res=result,
                                instances=data["instances_to_process"],
                                n_retries=max(
                                    llm.fallback_llm.max_retries_invalid_data, 1
                                ),  # retry with fallback LLM at least once
                                retry_is_final=True,
                            )

        # Group instances for processing, based on the relevant params, with the relevant prompts.
        if instance_type == "aspect":
            instance_class = Aspect
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
            instances=filtered_instances,
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
        instance_groups = _group_instances_by_fields(
            fields=fields_to_group_by, instances=filtered_instances
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
    def _validate_nesting_level(
        parent_item: Aspect | _Concept,
        child_items: list[Aspect | _Concept],
        item_type_name: str = "sub-items",
    ) -> None:
        """
        Validates that all child items have the correct nesting level relative to their parent.

        Args:
            parent_item: The parent aspect or concept
            child_items: List of child aspects or concepts to validate
            item_type_name: Name of the item type for the error message (e.g., "sub-aspects")

        Raises:
            AssertionError: If any child item has an incorrect nesting level
        """
        expected_level = parent_item._nesting_level + 1
        assert all(item._nesting_level == expected_level for item in child_items), (
            f"{item_type_name.capitalize()} must have a nesting level of `{expected_level}`."
            f" Current nesting levels: {[item._nesting_level for item in child_items]}"
        )

    def _get_usage_or_cost(
        self,
        retrieval_type: Literal["usage", "cost"],
        llm_role: Optional[str] = None,
        is_group: bool = False,
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
        :param is_group: Boolean indicating whether this is being called on an LLM group (True)
            or individual LLM (False). Defaults to False.
        :return: A list of containers, each representing usage or cost information for
            a primary and fallback LLM, if it exists. The specific container type depends on
            the retrieval type.
        """
        info_containers = []
        # For individual LLM, use self and its fallback
        # For group, iterate through all LLMs in self.llms
        llms_to_process = self.llms if is_group else [self]

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
