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
Module defining the base classes for models and the model group.

This module provides the processing logic that single models and model groups share
(``_GenericModelProcessor``), the base class of single models of any kind
(``_DocumentModel``), on top of which a specific kind of model, such as an LLM,
is defined in a module of its own, and the group of models with different roles
(``_DocumentModelGroup``).
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from collections.abc import Sequence
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Any, Literal, cast

import tethered
from aiolimiter import AsyncLimiter
from genai_prices import UpdatePrices as _GPUpdatePrices
from genai_prices import Usage as _GPUsage
from genai_prices import calc_price as _calc_auto_price
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    StrictBool,
    StrictInt,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from contextgem.internal.base.abstract import _AbstractGenericModelProcessor
from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.data_models import _ModelPricing
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.images import _Image
from contextgem.internal.base.paras_and_sents import _Paragraph
from contextgem.internal.base.serialization import _is_deserialization_in_progress
from contextgem.internal.data_models import (
    _Message,
    _ModelCall,
    _ModelCost,
    _ModelCostOutputContainer,
    _ModelUsage,
    _ModelUsageOutputContainer,
)
from contextgem.internal.decorators import (
    _disable_direct_initialization,
    _post_init_method,
    _timer_decorator,
)
from contextgem.internal.exceptions import ModelAPIError, ModelExtractionError
from contextgem.internal.items import _ExtractedItem, _StringItem
from contextgem.internal.loggers import logger
from contextgem.internal.registry import _publicize
from contextgem.internal.typings.types import (
    AsyncCalsAndKwargs,
    ExtractedInstanceType,
    JustificationDepth,
    LanguageRequirement,
    Modality,
    ModelRole,
    NonEmptyStr,
    ReferenceDepth,
)
from contextgem.internal.typings.validators import (
    _validate_model_role,
    _validate_sequence_is_list,
)
from contextgem.internal.utils import (
    _async_multi_executor,
    _chunk_list,
    _clean_text_for_llm_prompt,
    _get_distinct_instances,
    _group_instances_by_fields,
    _model_call_result_is_api_failure,
    _model_call_result_is_output_truncated,
    _model_call_result_is_valid,
    _parse_llm_output_as_json,
    _remove_thinking_content_from_llm_output,
    _run_async_calls,
    _run_sync,
    _validate_parsed_llm_output,
)


if TYPE_CHECKING:
    # `_DocumentLLM` is defined in `llms.py`, which imports this module, so that
    # this module cannot import it at runtime. Here, it is used only in annotations
    # and `cast(...)` calls.
    from contextgem.internal.base.llms import _DocumentLLM

    # Self-import resolves the string forward references used by `cast(...)` calls
    # later in this module (cast("_DocumentModelGroup", self), etc.). ty doesn't
    # recognise the self-import idiom, but pyright/mypy do — keep it for them.
    from contextgem.internal.base.models import (
        _DocumentModelGroup,  # ty: ignore[unresolved-import]
    )


# Local model providers supported via liteLLM
_LOCAL_MODEL_PROVIDERS = [
    "ollama/",
    "ollama_chat/",
    "lm_studio/",
]

# Rounding precision for reporting model costs (quantize on access only)
_COST_QUANT = Decimal("0.00001")

# Network destinations required by genai-prices to refresh cached pricing data.
# Used by `tethered.scope` to constrain egress in `_calculate_auto_pricing_costs`,
# and imported by tests/conftest.py to keep the test allowlist in sync.
_GENAI_PRICES_REFRESH_HOSTS: tuple[str, ...] = ("raw.githubusercontent.com",)

# Name of the kind of the models in the messages on the models of different kinds
_ANY_KIND_NAME = "model"


class _LoopAsyncLimiters:
    """
    Keeps the limiters that pace the requests to a model: one limiter for each
    event loop in which requests are sent to the model.

    A limiter is bound to the event loop that it is first used in, so the limiter
    that is set for a model is not used for the requests itself: it gives the rate,
    and a limiter with this rate is created for each event loop. All the requests
    to the model that are sent in the same event loop are paced by the same limiter,
    i.e. they share the rate of the model, whichever extraction calls they belong
    to and however many of the calls run concurrently.

    A copy of an instance has no limiters: they are not copied with the model.
    """

    def __init__(self) -> None:
        # The limiters by their event loops
        self._limiters: dict[asyncio.AbstractEventLoop, AsyncLimiter] = {}
        # Event loops may run in different threads
        self._lock = threading.Lock()

    def __deepcopy__(self, memo: dict[int, Any]) -> _LoopAsyncLimiters:
        """
        Returns a new instance without limiters: a limiter cannot be copied
        together with the event loop that it is bound to.

        :param memo: Objects already copied (unused).
        :type memo: dict[int, Any]
        :return: A new instance without limiters.
        :rtype: _LoopAsyncLimiters
        """
        return type(self)()

    def get(self, max_rate: float, time_period: float) -> AsyncLimiter:
        """
        Returns the limiter of the running event loop, created on the first call
        in the event loop, and again when the rate changes.

        :param max_rate: The maximum number of requests in the time period.
        :type max_rate: float
        :param time_period: The time period, in seconds.
        :type time_period: float
        :return: The limiter of the running event loop, with the given rate.
        :rtype: AsyncLimiter
        :raises RuntimeError: If no event loop is running.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            # The limiters of the event loops that are closed are no longer needed
            for closed_loop in [i for i in self._limiters if i.is_closed()]:
                del self._limiters[closed_loop]
            limiter = self._limiters.get(loop)
            if (
                limiter is None
                or limiter.max_rate != max_rate
                or limiter.time_period != time_period
            ):
                limiter = AsyncLimiter(max_rate=max_rate, time_period=time_period)
                self._limiters[loop] = limiter
            return limiter


def _get_common_kind_name(models: Sequence[_DocumentModel]) -> str:
    """
    Returns the name of the kind of the given models for the messages that apply
    to all of them, e.g. to the models of a group.

    :param models: The models that a message applies to.
    :type models: Sequence[_DocumentModel]
    :return: The name of the kind that all the models are of, e.g. "LLM", or "model"
        if the models are of different kinds.
    :rtype: str
    """
    kind_names = {model._kind_name for model in models}
    return kind_names.pop() if len(kind_names) == 1 else _ANY_KIND_NAME


def _with_indefinite_article(kind_name: str) -> str:
    """
    Returns the name of a kind of models with the indefinite article.

    :param kind_name: The name of the kind of models, e.g. "LLM".
    :type kind_name: str
    :return: The name with its article, e.g. "an LLM" or "a decision model".
    :rtype: str
    """
    # "LLM" is spelled out, i.e. it starts with a vowel sound
    return f"an {kind_name}" if kind_name == "LLM" else f"a {kind_name}"


def _get_non_unique_model_roles_msg(kind_name: str) -> str:
    """
    Returns the error message for the models that have the same role in a model group.

    :param kind_name: The name of the kind of the models in the group, e.g. "LLM".
    :type kind_name: str
    :return: The error message.
    :rtype: str
    """
    msg = (
        f"{kind_name}s in a group must have different roles, "
        f"and only one {kind_name} in a group can have no role."
    )
    return msg[0].upper() + msg[1:]


class _GenericModelProcessor(_AbstractGenericModelProcessor):
    """
    Base class that handles processing logic using models.

    This class provides the foundation for implementing model-based processing
    operations within the ContextGem framework. It defines the core interface and shared
    functionality for document analysis, information extraction, and content processing
    using various model backends.
    """

    def extract_all(
        self,
        document: _Document,
        *,
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
            single LLM prompt. A decision model is sent each paragraph in a request of
            its own for aspects, whatever the value. For concepts, it is sent a text
            with more paragraphs in parts of at most that many paragraphs, each in
            a request of its own: a concept then gets an extracted item for each part,
            and a concept with ``singular_occurrence=True`` is rejected with
            ``ValueError``. Defaults to 0 (all paragraphs). Cannot be negative.
        :type max_paragraphs_to_analyze_per_call: int, optional
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images). Cannot be negative. Relevant only for document-level
            concepts.
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
        *,
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
            single LLM prompt. A decision model is sent each paragraph in a request of
            its own for aspects, whatever the value. For concepts, it is sent a text
            with more paragraphs in parts of at most that many paragraphs, each in
            a request of its own: a concept then gets an extracted item for each part,
            and a concept with ``singular_occurrence=True`` is rejected with
            ``ValueError``. Defaults to 0 (all paragraphs). Cannot be negative.
        :type max_paragraphs_to_analyze_per_call: int, optional
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images). Cannot be negative. Relevant only for document-level
            concepts.
        :type max_images_to_analyze_per_call: int, optional
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: The document with extracted aspects and concepts.
        :rtype: _Document
        """

        self._check_max_per_call_params(
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            max_images_to_analyze_per_call=max_images_to_analyze_per_call,
        )
        self._check_model_roles_before_extract_all(document)
        self._check_vision_support_for_concepts(
            document=document, concepts=document.concepts
        )
        # All aspects and concepts of the document are checked here, as the concepts
        # of the aspects are extracted only after the requests for the aspects are sent
        self._check_models_can_serve_instances(
            self._get_all_instances_of_document(document),
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        # Tools are not used in extraction workflows
        self._warn_tools_ignored_if_enabled()

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
        *,
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
        automatically populated from these items. For an aspect with the "sentences" reference depth,
        the ``reference_sentences`` field is populated from them as well: with the extracted sentences,
        and not with the other sentences of the reference paragraphs.

        This is the synchronous version of `extract_aspects_from_document_async()`.

        :param document: The document from which aspects are to be extracted.
        :type document: _Document
        :param from_aspects: Existing aspects to use as a base for extraction. If None, uses all
            document's aspects. An aspect that is given more than once is extracted once.
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
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM
            prompt. Changes nothing for a decision model, which is sent each paragraph in
            a request of its own. Defaults to 0 (all paragraphs). Cannot be negative.
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
        *,
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
        automatically populated from these items. For an aspect with the "sentences" reference depth,
        the ``reference_sentences`` field is populated from them as well: with the extracted sentences,
        and not with the other sentences of the reference paragraphs.

        :param document: The document from which aspects are to be extracted.
        :type document: _Document
        :param from_aspects: Existing aspects to use as a base for extraction. If None, uses all
            document's aspects. An aspect that is given more than once is extracted once.
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
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM
            prompt. Changes nothing for a decision model, which is sent each paragraph in
            a request of its own. Defaults to 0 (all paragraphs). Cannot be negative.
        :type max_paragraphs_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed _Aspect objects with extracted items.
        :rtype: list[_Aspect]
        """

        self._check_max_per_call_params(
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        if from_aspects is not None:
            from_aspects = _validate_sequence_is_list(from_aspects)

        self._check_instances_and_llm_params(
            target=document,
            llm_or_group=self,
            instances_to_process=from_aspects,
            instance_type="aspect",
            overwrite_existing=overwrite_existing,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        # Tools are not used in extraction workflows
        self._warn_tools_ignored_if_enabled()

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
            # Safe cast: self is a _DocumentModelGroup because `is_group` is True
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in cast("_DocumentModelGroup", self).models
            ]
            # Safe cast: List comprehension creates tuples of (async_method, kwargs)
            # Type checker needs help to recognize this matches AsyncCalsAndKwargs format
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            # Safe cast: self is a single model because `is_group` is False
            await self._extract_instances(
                **extract_instances_kwargs,  # ty: ignore[invalid-argument-type]
                llm=cast("_DocumentModel", self),
            )

        # An aspect that is given more than once is extracted once, with its
        # sub-aspects
        document_aspects = (
            _get_distinct_instances(from_aspects) if from_aspects else document.aspects
        )

        # Extract sub-aspects
        extract_sub_aspects_kwargs = {
            "overwrite_existing": overwrite_existing,
            "max_items_per_call": max_items_per_call,
            "use_concurrency": use_concurrency,
            "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
            "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
        }
        # Build per-aspect sub-aspect documents up front; validation and
        # skip-logging stay deterministic before any concurrent work starts.
        sub_aspect_jobs: list[tuple[_Aspect, _Document]] = []
        for aspect in document_aspects:
            if not aspect.aspects:
                continue
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
            # Keep the sub-aspects' nesting levels intact: the extraction engine
            # checks "aspect is assigned to document" via instance equality (which
            # includes `_nesting_level`), so the added aspects must stay equal to
            # the instances being processed rather than being normalized to 0.
            aspect_document._suppress_aspect_nesting_normalization = True
            aspect_document.add_aspects(aspect.aspects)  # ty: ignore[invalid-argument-type]
            sub_aspect_jobs.append((aspect, aspect_document))

        if sub_aspect_jobs:
            cals_and_kwargs = [
                (
                    self.extract_aspects_from_document_async,
                    {**extract_sub_aspects_kwargs, "document": doc},
                )
                for _, doc in sub_aspect_jobs
            ]
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
            for aspect, doc in sub_aspect_jobs:
                # Overwrite the sub-aspects with the newly extracted ones,
                # as the sub-aspects were deep-copied when attached to
                # the aspect document.
                aspect.aspects = doc.aspects

        return document_aspects  # ty: ignore[invalid-return-type]

    def extract_concepts_from_aspect(
        self,
        aspect: _Aspect,
        document: _Document,
        *,
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
            A concept that is given more than once is extracted once.
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
            single LLM prompt. A decision model is sent a text with more
            paragraphs in parts of at most that many paragraphs, each in a request of
            its own: a concept then gets an extracted item for each part, and
            a concept with ``singular_occurrence=True`` is rejected with ``ValueError``.
            Defaults to 0 (all paragraphs). Cannot be negative.
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
        *,
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
            A concept that is given more than once is extracted once.
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
            single LLM prompt. A decision model is sent a text with more
            paragraphs in parts of at most that many paragraphs, each in a request of
            its own: a concept then gets an extracted item for each part, and
            a concept with ``singular_occurrence=True`` is rejected with ``ValueError``.
            Defaults to 0 (all paragraphs). Cannot be negative.
        :type max_paragraphs_to_analyze_per_call: int
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed concept objects.
        :rtype: list[_Concept]
        """

        self._check_max_per_call_params(
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        if from_concepts is not None:
            from_concepts = _validate_sequence_is_list(from_concepts)

        self._check_instances_and_llm_params(
            target=aspect,
            llm_or_group=self,
            instances_to_process=from_concepts,
            instance_type="concept",
            overwrite_existing=overwrite_existing,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()
        # The paragraphs of the aspect and of its sub-aspects can be copies of
        # the paragraphs of the document, without the sentences that were segmented
        # after the aspect was extracted
        document._share_sents_with_reference_paragraphs(aspect)

        # Tools are not used in extraction workflows
        self._warn_tools_ignored_if_enabled()

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
            # Safe cast: self is a _DocumentModelGroup because `is_group` is True
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in cast("_DocumentModelGroup", self).models
            ]
            # Safe cast: List comprehension creates tuples of (async_method, kwargs)
            # Type checker needs help to recognize this matches AsyncCalsAndKwargs format
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            # Safe cast: self is a single model because `is_group` is False
            await self._extract_instances(
                **extract_instances_kwargs,  # ty: ignore[invalid-argument-type]
                llm=cast("_DocumentModel", self),
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
            sub_aspect_jobs: list[tuple[_Aspect, _Document]] = []
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
                # Keep the sub-aspect's nesting level intact: the extraction engine
                # checks "aspect is assigned to document" via instance equality
                # (which includes `_nesting_level`), so the added aspect must stay
                # equal to the instance being processed rather than normalized to 0.
                sub_aspect_document._suppress_aspect_nesting_normalization = True
                sub_aspect_document.add_aspects([sub_aspect])
                sub_aspect_jobs.append((sub_aspect, sub_aspect_document))

            if sub_aspect_jobs:
                cals_and_kwargs = [
                    (
                        self.extract_concepts_from_aspect_async,
                        {
                            **extract_concepts_from_sub_aspects_kwargs,
                            "aspect": sub_aspect,
                            "document": sub_aspect_document,
                        },
                    )
                    for sub_aspect, sub_aspect_document in sub_aspect_jobs
                ]
                cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
                results = await _run_async_calls(
                    cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
                )
                for (sub_aspect, _), sub_aspect_concepts in zip(
                    sub_aspect_jobs, results, strict=True
                ):
                    # Overwrite the sub-aspects with the newly extracted ones,
                    # as the sub-aspects were deep-copied when attached to
                    # the sub-aspect document.
                    sub_aspect.concepts = sub_aspect_concepts

        # Explicitly return list of concepts for type checking
        return (
            _get_distinct_instances(from_concepts)
            if from_concepts
            else list(aspect.concepts)
        )

    def extract_concepts_from_document(
        self,
        document: _Document,
        *,
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
            document's concepts. A concept that is given more than once is extracted once.
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
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM
            prompt. A decision model is sent a text with more
            paragraphs in parts of at most that many paragraphs, each in a request of
            its own: a concept then gets an extracted item for each part, and
            a concept with ``singular_occurrence=True`` is rejected with ``ValueError``.
            Defaults to 0 (all paragraphs). Cannot be negative.
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images). Cannot be negative.
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
        *,
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
            document's concepts. A concept that is given more than once is extracted once.
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
        :param max_paragraphs_to_analyze_per_call: Maximum paragraphs to analyze in a single LLM
            prompt. A decision model is sent a text with more
            paragraphs in parts of at most that many paragraphs, each in a request of
            its own: a concept then gets an extracted item for each part, and
            a concept with ``singular_occurrence=True`` is rejected with ``ValueError``.
            Defaults to 0 (all paragraphs). Cannot be negative.
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: Maximum images to include in a single LLM prompt.
            Defaults to 0 (all images). Cannot be negative.
        :type max_images_to_analyze_per_call: int, optional
        :param raise_exception_on_extraction_error: Whether to raise an exception if the extraction fails
            due to invalid data returned by an LLM or an error in the LLM API. If False, a warning will
            be issued instead, and no extracted items will be returned. Defaults to True.
        :type raise_exception_on_extraction_error: bool, optional
        :return: List of processed Concept objects with extracted items.
        :rtype: list[_Concept]
        """

        self._check_max_per_call_params(
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            max_images_to_analyze_per_call=max_images_to_analyze_per_call,
        )

        if from_concepts is not None:
            from_concepts = _validate_sequence_is_list(from_concepts)

        self._check_instances_and_llm_params(
            target=document,
            llm_or_group=self,
            instances_to_process=from_concepts,
            instance_type="concept",
            overwrite_existing=overwrite_existing,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        # Check if sentence segmentation is required for some aspects or concepts
        if document._requires_sentence_segmentation():
            document._segment_sents()

        # Tools are not used in extraction workflows
        self._warn_tools_ignored_if_enabled()

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
            # Safe cast: self is a _DocumentModelGroup because `is_group` is True
            cals_and_kwargs = [
                (self._extract_instances, {**extract_instances_kwargs, "llm": llm})
                for llm in cast("_DocumentModelGroup", self).models
            ]
            # Safe cast: List comprehension creates tuples of (async_method, kwargs)
            # Type checker needs help to recognize this matches AsyncCalsAndKwargs format
            cals_and_kwargs = cast(AsyncCalsAndKwargs, cals_and_kwargs)
            await _run_async_calls(
                cals_and_kwargs=cals_and_kwargs, use_concurrency=use_concurrency
            )
        else:
            # Safe cast: self is a single model because `is_group` is False
            await self._extract_instances(
                **extract_instances_kwargs, llm=cast("_DocumentModel", self)
            )

        # Explicitly return list of concepts for type checking
        return (
            _get_distinct_instances(from_concepts)
            if from_concepts
            else list(document.concepts)
        )

    def _check_model_roles_before_extract_all(
        self,
        document: _Document,
    ) -> None:
        """
        Checks, before the full extraction method extract_all() is run with a model group,
        that all aspects and concepts assigned to the given document, including sub-aspects
        and aspect-level concepts, can be routed to the LLMs in the group. This ensures
        that no LLM calls are made for the extraction that cannot be completed.

        The check does not apply to a standalone LLM, which extracts all aspects
        and concepts, regardless of their model roles.

        :param document: The document object to check for model role assignments.
        :type document: _Document
        :return: None
        :rtype: None
        :raises ValueError: If the document contains aspects or concepts with model roles
            that no LLM in the group has.
        """

        if self.is_group:
            self._check_model_roles_are_routable(document.model_roles)

    def _check_model_roles_are_routable(self, model_roles: set[str | None]) -> None:
        """
        Checks that each of the given model roles, assigned to aspects or concepts,
        is the role of exactly one LLM in the current model group. An aspect or concept
        is routed to the LLM in the group with the same role, and an aspect or concept
        without a role (None) is routed to the LLM without a role.

        :param model_roles: Model roles assigned to the aspects or concepts to be extracted.
        :type model_roles: set[str | None]
        :return: None
        :rtype: None
        :raises ValueError: If several LLMs in the group have the same role, or if any
            of the given model roles is not the role of an LLM in the group.
        """

        def format_roles(roles: set[str | None]) -> str:
            """
            Formats model roles for an error message.

            :param roles: Model roles to format.
            :type roles: set[str | None]
            :return: Comma-separated model roles, with None described as no role.
            :rtype: str
            """
            return ", ".join(
                "None (no role)" if role is None else f"`{role}`"
                for role in sorted(roles, key=lambda role: (role is not None, role))
            )

        # The models are named in the messages by their kind, e.g. as LLMs, unless
        # the group has models of different kinds.
        # Safe cast: the roles are routed only in a model group
        kind = _get_common_kind_name(cast("_DocumentModelGroup", self).models)

        # The roles of the LLMs are validated to be unique when a group is created
        # or its LLMs are reassigned. However, the role of an LLM can be changed after
        # the LLM is added to a group, which the group is not notified of.
        roles_of_llms = self.list_roles
        duplicate_roles = {
            role for role in roles_of_llms if roles_of_llms.count(role) > 1
        }
        if duplicate_roles:
            raise ValueError(
                f"{_get_non_unique_model_roles_msg(kind)} "
                f"Roles assigned to several {kind}s: {format_roles(duplicate_roles)}."
            )

        group_roles = set(roles_of_llms)
        missing_model_roles = model_roles.difference(group_roles)
        if missing_model_roles:
            raise ValueError(
                f"Some aspects/concepts cannot be routed to "
                f"{_with_indefinite_article(kind)} in the group, "
                f"as no {kind} in the group has the role: {format_roles(missing_model_roles)}. "
                f"Roles of the {kind}s in the group: {format_roles(group_roles)}. "
                f"An aspect/concept is extracted by the {kind} with the `role` equal to "
                f"the `model_role` of the aspect/concept. An aspect/concept without `model_role` "
                f"is extracted by the {kind} without `role`."
            )

    def _check_vision_support_for_concepts(
        self,
        document: _Document,
        concepts: Sequence[_Concept],
    ) -> None:
        """
        Checks, before any requests are sent, that each LLM that is to extract the given
        document-level concepts from the images of the document supports vision.
        The same is required of the fallback LLMs of such LLM, as a fallback LLM is sent
        the same content as the LLM it replaces. Otherwise, a fallback LLM that does not
        support vision would fail only when it is needed, i.e. after the requests
        to the primary LLM are sent.

        If the document has images, the concepts with the "image" and "multimodal"
        modalities are extracted from them. In a model group, each concept is extracted
        by the LLM that the concept is routed to by the role, i.e. the roles must be
        checked to be routable before this check.

        :param document: The document that the concepts are extracted from.
        :type document: _Document
        :param concepts: The document-level concepts to be extracted.
        :type concepts: Sequence[_Concept]
        :return: None
        :rtype: None
        :raises ValueError: If an LLM that is to extract concepts from images, or any
            of its fallback LLMs, does not support vision.
        """

        if not document.images:
            # No images are sent to the LLMs. Extraction of the concepts with the "image"
            # modality from such document fails for the lack of images.
            return

        image_concepts = [i for i in concepts if i.modality != "text"]
        if not image_concepts:
            return

        # Imported here, as the module that defines LLMs imports this module
        from contextgem.internal.base.llms import _DocumentLLM

        if self.is_group:
            # Safe cast: self is a _DocumentModelGroup because `is_group` is True
            llms = cast("_DocumentModelGroup", self).models
        else:
            # Safe cast: self is a single model because `is_group` is False
            llms = [cast("_DocumentModel", self)]

        for llm in llms:
            # In a model group, a concept is extracted by the LLM with the same role.
            # A standalone LLM extracts all the concepts, regardless of their roles.
            llm_image_concepts = [
                i
                for i in image_concepts
                if not self.is_group or i.model_role == llm.role
            ]
            if not llm_image_concepts:
                continue
            concept_names = ", ".join(f"`{i.name}`" for i in llm_image_concepts)
            for llm_instance in [llm, *llm.fallback_models]:
                if not isinstance(llm_instance, _DocumentLLM):
                    # Vision is a capability of LLMs. Whether a model of another
                    # kind can extract the concepts routed to it, e.g. from images,
                    # is checked separately (see `_check_models_can_serve_instances`).
                    continue
                if llm_instance._supports_vision:
                    continue
                if llm_instance.is_fallback:
                    subject = "Fallback model"
                    condition = ", if the LLM it is a fallback for fails"
                else:
                    subject = "Model"
                    condition = ""
                raise ValueError(
                    f"{subject} `{llm_instance.model}` does not support vision "
                    f"according to litellm.supports_vision(), but it is to extract "
                    f"the following concepts from images{condition}: {concept_names}. "
                    f"To override this detection, manually set `_supports_vision=True` "
                    f"on the LLM instance."
                )

    @staticmethod
    def _check_max_per_call_params(
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
    ) -> None:
        """
        Checks the max numbers of paragraphs and of images to analyze per call that
        an extraction method is called with. With a negative number, no paragraphs
        or images would be left to send to an LLM, i.e. nothing would be extracted,
        without an error.

        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs to
            analyze per call. Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: The max number of images to analyze
            per call. Defaults to 0 (all images).
        :type max_images_to_analyze_per_call: int
        :return: None
        :rtype: None
        :raises ValueError: If any of the numbers is negative.
        """
        for param_name, value in (
            ("max_paragraphs_to_analyze_per_call", max_paragraphs_to_analyze_per_call),
            ("max_images_to_analyze_per_call", max_images_to_analyze_per_call),
        ):
            if value < 0:
                raise ValueError(
                    f"`{param_name}` cannot be negative: got {value}. "
                    f"Use 0, the default, to set no limit."
                )

    def _check_instances_and_llm_params(
        self,
        target: _Document | _Aspect,
        llm_or_group: _DocumentModel | _DocumentModelGroup,
        instances_to_process: Sequence[_Aspect]
        | Sequence[_Concept]
        | None,  # using Sequence type with list validator for type checking
        instance_type: ExtractedInstanceType,
        overwrite_existing: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> None:
        """
        Validates instances and LLM parameters, ensuring compatibility with the target
        and configurations provided. For a model group, validates the roles of all
        descendants that this extraction will process before any requests are sent.
        Aspect extraction checks aspect roles only; concept extraction from an aspect
        checks concept roles in the selected aspect and its non-empty descendants.
        The same instances are checked to be extractable by the models that are
        to extract them, where a model extracts only some aspects/concepts,
        as a decision model does, with the given max number of paragraphs to
        analyze per call.

        :param target: The target object, which should have an attribute corresponding
            to `instance_type` (e.g., 'aspects' for 'aspect', etc.). Expected to be an
            instance of either _Document or _Aspect.
        :type target: _Document | _Aspect
        :param llm_or_group: The model or a model group to be validated. This may either
            be a standalone model, e.g. a DocumentLLM, or a DocumentModelGroup.
        :type llm_or_group: _DocumentModel | _DocumentModelGroup
        :param instances_to_process: A list of instances to process, which must match
            the specified instance-type (`aspect` or `concept`). If not provided,
            defaults to instances present in the target attribute.
        :type instances_to_process: Sequence[_Aspect] | Sequence[_Concept] | None
        :param instance_type: Specifies the type of instances to validate ('aspect' or
            'concept'). This determines the expected type for validation.
        :type instance_type: ExtractedInstanceType
        :param overwrite_existing: A flag indicating whether to overwrite the states of
            the existing processed instances. Defaults to False.
        :type overwrite_existing: bool
        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs to
            analyze per call that the extraction is run with. What a model can
            extract may depend on it. Defaults to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :return: None. The function performs validation and raises errors in case of
            mismatches or invalid configurations.
        :rtype: None
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

        # Include the descendants that the public extraction method processes
        # recursively, respecting the selection of top-level instances.
        instances_in_scope: list[_Aspect | _Concept] = list(check_instances)
        if instance_type == "aspect":
            pending_aspects = list(cast(Sequence[_Aspect], check_instances))
        elif isinstance(target, _Aspect):
            pending_aspects = list(target.aspects)
        else:
            pending_aspects = []
        while pending_aspects:
            aspect = pending_aspects.pop()
            if instance_type == "aspect":
                instances_in_scope.extend(aspect.aspects)
            else:
                # Concept extraction skips this whole subtree when the aspect
                # has no extracted paragraphs to provide as context.
                if not aspect.reference_paragraphs:
                    continue
                instances_in_scope.extend(aspect.concepts)
            pending_aspects.extend(aspect.aspects)

        # Check DocumentModelGroup. A standalone DocumentLLM extracts all the instances,
        # regardless of their model roles, i.e. requires no check of the model roles.
        if llm_or_group.is_group:
            # Safe cast: llm_or_group is a _DocumentModelGroup because `is_group` is True
            model_group = cast("_DocumentModelGroup", llm_or_group)
            if not model_group.models:
                raise ValueError(
                    "The provided DocumentModelGroup does not contain any defined LLMs."
                )
            model_group._check_model_roles_are_routable(
                {i.model_role for i in instances_in_scope}
            )

        # Concepts can be extracted from images only at the document level
        if instance_type == "concept" and isinstance(target, _Document):
            llm_or_group._check_vision_support_for_concepts(
                document=target,
                # Safe cast: check_instances contains only _Concept instances
                # because `instance_type` is "concept"
                concepts=cast(Sequence[_Concept], check_instances),
            )

        # A model of a kind that extracts only some aspects/concepts, e.g. a decision
        # model, must be able to extract what this extraction routes to it
        llm_or_group._check_models_can_serve_instances(
            instances_in_scope,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

    def _check_models_can_serve_instances(
        self,
        instances: Sequence[_Aspect | _Concept],
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> None:
        """
        Checks, before any requests are sent, that each model can extract the aspects
        and concepts that are to be extracted by it. The same is required of
        the fallback models of such model, as a fallback model is to extract what
        the model it replaces fails to extract. Otherwise, a fallback model that
        cannot extract an aspect or a concept would fail only when it is needed,
        i.e. after the requests to the primary model are sent.

        What a model can extract depends on its kind (see
        ``_DocumentModel._check_instances_servable``): an LLM extracts any aspect
        or concept, while a decision model extracts only some of them, which
        also depends on the max number of paragraphs to analyze per call.

        In a model group, each aspect or concept is extracted by the model that it
        is routed to by the role, i.e. the roles must be checked to be routable
        before this check. A standalone model extracts all the aspects and concepts,
        regardless of their roles.

        :param instances: The aspects and concepts to be extracted, including those
            extracted from the aspects that the extraction goes through.
        :type instances: Sequence[_Aspect | _Concept]
        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs to
            analyze per call that the extraction is run with. Defaults to 0
            (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :return: None
        :rtype: None
        :raises ValueError: If a model that is to extract some of the aspects or
            concepts, or any of its fallback models, cannot extract them.
        """

        if self.is_group:
            # Safe cast: self is a _DocumentModelGroup because `is_group` is True
            models = cast("_DocumentModelGroup", self).models
        else:
            # Safe cast: self is a single model because `is_group` is False
            models = [cast("_DocumentModel", self)]

        for model in models:
            routed_instances = [
                i for i in instances if not self.is_group or i.model_role == model.role
            ]
            if not routed_instances:
                continue
            for model_instance in [model, *model.fallback_models]:
                model_instance._check_instances_servable(
                    routed_instances,
                    routed_in_group=self.is_group,
                    max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                )

    @staticmethod
    def _get_all_instances_of_document(
        document: _Document,
    ) -> list[_Aspect | _Concept]:
        """
        Returns all aspects and concepts assigned to the document, including
        the sub-aspects and the concepts of the aspects at any level of nesting,
        i.e. everything that the full extraction method extract_all() is to extract.

        :param document: The document to get the aspects and concepts of.
        :type document: _Document
        :return: The concepts of the document, followed by each aspect with its
            concepts, the aspects of a level before their sub-aspects.
        :rtype: list[_Aspect | _Concept]
        """
        instances: list[_Aspect | _Concept] = list(document.concepts)
        pending_aspects = list(document.aspects)
        while pending_aspects:
            aspect = pending_aspects.pop(0)
            instances.append(aspect)
            instances.extend(aspect.concepts)
            pending_aspects.extend(aspect.aspects)
        return instances

    @staticmethod
    def _check_instances_already_processed(
        instance_type: ExtractedInstanceType,
        instances: list[_Aspect] | list[_Concept],
        overwrite_existing: bool,
    ) -> None:
        """
        Checks whether the given instances of a specified type have already been processed.

        :param instance_type: The type of instances being checked.
        :type instance_type: ExtractedInstanceType
        :param instances: A list of instances to be evaluated for processing status.
        :type instances: list[_Aspect] | list[_Concept]
        :param overwrite_existing: Specifies whether to overwrite already processed instances.
        :type overwrite_existing: bool
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
                    f"Some {instance_type}s have already been processed: {already_processed_names}. "
                    "Set `overwrite_existing=True` to overwrite them."
                )

    @staticmethod
    def _get_text_for_concept_extraction(
        source: _Document | _Aspect,
        paragraphs_chunk: list[_Paragraph],
    ) -> str:
        """
        Returns the text that concepts are extracted from, when it is sent to a model
        as a raw text, i.e. not as the paragraphs to be referenced.

        :param source: The source that the concepts are extracted from.
        :type source: _Document | _Aspect
        :param paragraphs_chunk: The paragraphs of the source to be sent in one
            call: all the paragraphs of the document or of the aspect, or a part
            of them if the source is processed in chunks.
        :type paragraphs_chunk: list[_Paragraph]
        :return: The raw text of the document, cleaned with its linebreaks preserved,
            if the document is processed as a whole; otherwise, the cleaned raw texts
            of the paragraphs, separated by a blank line.
        :rtype: str
        :raises ValueError: If the document is processed as a whole and has no text.
        """
        if isinstance(source, _Document) and len(paragraphs_chunk) == len(
            source.paragraphs
        ):
            # If the document is being processed as a whole,
            # use raw_text of the document.
            text = source.raw_text
            if text is None:
                raise ValueError("Document lacks text for concept extraction.")
            return _clean_text_for_llm_prompt(
                text,
                preserve_linebreaks=True,
                strip_text=True,
            )  # raw text of the document
        # If an aspect is being processed, or if the document is being
        # processed in chunks, use raw_text of the paragraphs.
        return "\n\n".join(
            [
                _clean_text_for_llm_prompt(
                    p.raw_text,
                    preserve_linebreaks=False,
                    strip_text=True,
                )
                for p in paragraphs_chunk
            ]
        )

    @staticmethod
    def _prepare_message_kwargs_list(
        extracted_instance_type: ExtractedInstanceType,
        source: _Document | _Aspect,
        llm: _DocumentLLM,
        instances_to_process: list[_Aspect] | list[_Concept],
        document: _Document,
        add_justifications: bool = False,
        justification_depth: JustificationDepth = "brief",
        justification_max_sents: int = 2,
        add_references: bool = False,
        reference_depth: ReferenceDepth = "paragraphs",
        modality: Modality = "text",
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
        :type extracted_instance_type: ExtractedInstanceType
        :param source: The input source, which can either be a `_Document` or an `_Aspect`
            instance, from which aspects or concepts are extracted.
        :type source: _Document | _Aspect
        :param llm: The LLM instance used for extraction.
        :type llm: _DocumentLLM
        :param instances_to_process: A list of instances (either `_Aspect` or `_Concept` instances)
            to process for extraction tasks.
        :type instances_to_process: list[_Aspect] | list[_Concept]
        :param document: An instance of `_Document` containing paragraphs or text or images,
            which is used to provide additional context for the extraction process.
        :type document: _Document
        :param add_justifications: A boolean flag. When set to `True`,
            justification for the extracted items is included in extraction result.
        :type add_justifications: bool
        :param justification_depth: The level of detail of justifications. Defaults to "brief".
        :type justification_depth: JustificationDepth
        :param justification_max_sents: The maximum number of sentences in a justification.
            Defaults to 2.
        :type justification_max_sents: int
        :param add_references: A boolean flag. When `True`, and if applicable,
            references aiding the extraction task are included in the extraction result.
        :type add_references: bool
        :param reference_depth: The structural depth of the references, i.e. whether to provide
            paragraphs as references or sentences as references. Defaults to "paragraphs".
            ``extracted_items`` will have values based on this parameter.
        :type reference_depth: ReferenceDepth
        :param modality: The document content the concepts are extracted from: "text", "image",
            or "multimodal" (text and/or images, whichever the document has). Relevant only for
            document-level concepts, as aspects and aspect-level concepts are always extracted
            from text. Defaults to "text".
        :type modality: Modality
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the paragraphs are analyzed.
        :type max_paragraphs_to_analyze_per_call: int
        :param max_images_to_analyze_per_call: The maximum number of images to analyze in a single
            LLM call (prompt). Defaults to 0, in which case all the images are analyzed.
        :type max_images_to_analyze_per_call: int
        :return: A list of dictionaries containing prompt kwargs and context chunks for LLM queries.
        :rtype: list[dict[str, dict | list[_Paragraph] | list[_Image]]]
        """

        # Validate source data for extraction and set the extraction level
        def validate_source_and_get_extraction_level() -> str:
            """
            Validates the source based on the extracted instance type.

            :return: A string indicating the extraction level or type.
            :rtype: str

            :raises ValueError: If the source is invalid for the given extraction type or
                required content (text, paragraphs, images) is missing.
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
                    has_text = bool(source.raw_text or source.paragraphs)
                    has_images = bool(source.images)

                    # Multimodal concepts can be extracted from text or images or both
                    if modality == "multimodal":
                        if not has_text and not has_images:
                            raise ValueError(
                                "Document lacks text/paragraphs or images for multimodal concept extraction."
                            )
                        # Multimodal concepts are flexible - they are extracted from whatever
                        # content is available, leveraging the existing text/vision paths
                        return "concept_document_multimodal"

                    # Concepts extracted from text
                    elif modality == "text":
                        if not has_text:
                            raise ValueError(
                                "Document lacks text or paragraphs for concept extraction."
                            )
                        return "concept_document_text"

                    # Concepts extracted from images
                    elif modality == "image":
                        if not has_images:
                            raise ValueError(
                                "Document lacks images to extract vision concepts from."
                            )
                        if add_references:
                            raise ValueError(
                                "References are not supported for vision concepts."
                            )
                        return "concept_document_vision"

                    raise ValueError(f"Unsupported modality: `{modality}`")

                # Concept (aspect-level)
                if isinstance(source, _Aspect):
                    return "concept_aspect_text"

            raise ValueError(
                f"Unsupported extracted item type: `{extracted_instance_type}`"
            )

        extraction_level = validate_source_and_get_extraction_level()
        message_kwargs_list = []

        # Text-based or multimodal extraction
        if extraction_level.endswith("_text") or (
            extraction_level.endswith("_multimodal")
            and isinstance(source, _Document)
            and bool(source.raw_text or source.paragraphs)
        ):
            # Safe cast: source for the other extraction level is Aspect
            paragraphs = (
                document.paragraphs
                if extraction_level
                in {
                    "aspect_document_text",
                    "concept_document_text",
                    "concept_document_multimodal",
                }
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

                # Concept (document- and aspect-levels)
                elif extraction_level in {
                    "concept_document_text",
                    "concept_aspect_text",
                    "concept_document_multimodal",
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
                        prompt_kwargs["paragraphs"] = paragraphs_chunk  # ty: ignore[invalid-assignment]
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
                    else:
                        # Raw text of document/aspect used for concept extraction
                        prompt_kwargs["text"] = (
                            _GenericModelProcessor._get_text_for_concept_extraction(
                                source=source, paragraphs_chunk=paragraphs_chunk
                            )
                        )

                else:
                    raise ValueError(
                        f"Unsupported extraction level for text-based extraction: `{extraction_level}`"
                    )

                message_kwargs = {
                    "prompt_kwargs": prompt_kwargs,
                    "paragraphs_chunk": paragraphs_chunk,
                    "references_apply": add_references,
                }
                message_kwargs_list.append(message_kwargs)

        # Images-based or multimodal extraction
        if extraction_level.endswith("_vision") or (
            extraction_level.endswith("_multimodal")
            and isinstance(source, _Document)
            and bool(source.images)
        ):
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
                if extraction_level == "concept_document_vision" or (
                    extraction_level == "concept_document_multimodal"
                ):
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
                        "references_apply": False,
                        # references are not supported for vision concepts,
                        # and will not be added for multimodal concepts when extracting
                        # from images only
                    }

                else:
                    raise ValueError(
                        f"Unsupported extraction level for vision/multimodal extraction: "
                        f"`{extraction_level}`"
                    )

                message_kwargs_list.append(message_kwargs)

        return message_kwargs_list

    async def _extract_items_from_instances(
        self,
        extracted_item_type: ExtractedInstanceType,
        source: _Document | _Aspect,
        llm: _DocumentModel,
        instances_to_process: list[_Aspect] | list[_Concept],
        document: _Document,
        overwrite_existing: bool = False,
        add_justifications: bool = False,
        justification_depth: JustificationDepth = "brief",
        justification_max_sents: int = 2,
        add_references: bool = False,
        reference_depth: ReferenceDepth = "paragraphs",
        modality: Modality = "text",
        max_paragraphs_to_analyze_per_call: int = 0,
        max_images_to_analyze_per_call: int = 0,
        num_retries_failed_request: int = 3,
        max_retries_failed_request: int = 0,
        async_limiter: AsyncLimiter | None = None,
        raise_exception_on_extraction_error: bool = True,
        has_fallback_pending: bool = False,
    ) -> tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]:
        """
        Extracts items from either aspects or concepts using a specified LLM.
        This unified method handles extraction from both documents and aspects,
        supporting both text and images as the content to extract from.

        The method is the single entry point of the extraction with a model of
        any kind. After the checks that apply to any model, the extraction with
        a decision model is passed to the implementation of that model, which
        returns the result under the same contract, while the extraction with
        an LLM is performed here.

        :param extracted_item_type: Type of the item(s) being extracted ("aspect" or "concept")
        :type extracted_item_type: ExtractedInstanceType
        :param source: The source to extract from (_Document or _Aspect)
        :type source: _Document | _Aspect
        :param llm: The model used for extraction: an LLM or a decision model
        :type llm: _DocumentModel
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
        :param modality: The document content the concepts are extracted from: "text", "image",
            or "multimodal" (text and/or images, whichever the document has). Relevant only for
            document-level concepts, as aspects and aspect-level concepts are always extracted
            from text. Defaults to "text".
        :type modality: Modality
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs to analyze in a single
            LLM call (prompt), or whose text is sent to a decision model in one request for concepts.
            Defaults to 0, in which case all the paragraphs are analyzed.
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
        :param has_fallback_pending: Whether a fallback LLM will be tried if the call to the
            specified LLM fails. If True, a model API error is logged without being raised or
            warned about, as it is not final. The same applies to a response cut off at
            the max output tokens. Defaults to False.
        :type has_fallback_pending: bool

        :return: A tuple containing:
            (0) List of processed instances with extracted items, or None if LLM processing fails
            (1) _ModelUsage instance with the model usage information that the usage and
                cost of the LLM are yet to be updated with, i.e. with the usage of
                the last LLM call made. If the context is processed in several LLM calls,
                the usage and cost of the LLM are updated with each of the previous calls
                before the next one is made.
        :rtype: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]
        """

        def validate_source_in_document(
            source: _Document | _Aspect,
            document: _Document,
        ) -> None:
            """
            Raises ValueError if an Aspect is not assigned to the given Document.

            :param source: The source to validate (_Document or _Aspect)
            :type source: _Document | _Aspect
            :param document: The document containing the source.
            :type document: _Document
            :return: None
            :rtype: None
            """
            if isinstance(source, _Aspect) and source not in document.aspects:
                raise ValueError(
                    f"Aspect `{source.name}` must be assigned to the document"
                )

        def validate_image_or_multimodal_extraction(
            extracted_item_type: ExtractedInstanceType,
            modality: Modality,
            source: _Document | _Aspect,
            add_references: bool,
        ) -> None:
            """
            Validates usage constraints for the extraction from images, i.e. for
            the "image" and "multimodal" modalities.

            - For the "image" and "multimodal" modalities, only document-level concept
              extraction is allowed.
            - Raises ValueError when attempting aspect extraction, or concept extraction
              from an Aspect, with the "image" or "multimodal" modality.
            - For concept extraction with ``add_references=True``:
              - "multimodal": references are permitted only when the Document has text.
                Emits warnings when both text and images are present (references will be
                added only for text) or when text is absent (no references will be added).
              - "image": references are not supported and a ValueError is raised.

            :param extracted_item_type: Target extraction type ("aspect" or "concept").
            :type extracted_item_type: ExtractedInstanceType
            :param modality: The document content to extract from.
            :type modality: Modality
            :param source: The extraction source (Document or Aspect).
            :type source: _Document | _Aspect
            :param add_references: Whether to add references for extracted concepts.
            :type add_references: bool
            :return: None
            :rtype: None
            """

            if modality != "text":
                # Image/multimodal extraction: only document-level concept extraction is valid
                if extracted_item_type == "aspect" or (
                    extracted_item_type == "concept" and isinstance(source, _Aspect)
                ):
                    raise ValueError(
                        "Extraction from images is supported only for document-level "
                        "concepts, i.e. the `image` and `multimodal` modalities cannot be "
                        "used for aspects and aspect-level concepts."
                    )
                if extracted_item_type == "concept" and add_references:
                    # Allow references for multimodal only when text is present
                    if modality == "multimodal":
                        has_text = isinstance(source, _Document) and bool(
                            source.raw_text or source.paragraphs
                        )
                        has_images = isinstance(source, _Document) and bool(
                            source.images
                        )
                        if has_text and has_images:
                            warnings.warn(
                                "For multimodal concepts, references will only be added "
                                "when extracting from text. When extracting from images, "
                                "references will not be added, as there is no text content "
                                "to reference.",
                                stacklevel=2,
                            )
                        elif has_images and not has_text:
                            warnings.warn(
                                "References will not be added for multimodal concepts "
                                "when extracting from images only, as there is no text "
                                "content to reference.",
                                stacklevel=2,
                            )
                    else:
                        # Vision-only concepts never support references
                        raise ValueError(
                            "References are not supported for vision concepts."
                        )

        def validate_aspect_for_concept_extraction(
            extracted_item_type: ExtractedInstanceType,
            source: _Document | _Aspect,
        ) -> None:
            """
            If extracting concepts from an Aspect, validates that the aspect's extracted items have
            references.

            :param extracted_item_type: Target extraction type ("aspect" or "concept").
            :type extracted_item_type: ExtractedInstanceType
            :param source: The extraction source (Document or Aspect).
            :type source: _Document | _Aspect
            :return: None
            :rtype: None
            :raises ValueError: If the aspect has extracted items but no references.
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
        validate_image_or_multimodal_extraction(
            extracted_item_type=extracted_item_type,
            modality=modality,
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
            return [], _ModelUsage()

        # Handle empty instances list
        if not instances_to_process:
            logger.info(
                f"No {extracted_item_type}s to process with {llm._kind_name} "
                f"`{llm.model}`"
            )
            return [], _ModelUsage()

        # A decision model is sent no prompt: it answers typed questions on the text
        # of the source, i.e. it builds its requests and the items from its answers
        # itself. The checks of the source above apply to a model of any kind.
        # Imported here, as the module that defines decision models imports this module.
        from contextgem.internal.base.decision_models import _DocumentDecisionModel

        if isinstance(llm, _DocumentDecisionModel):
            return await llm._extract_items_from_decisions(
                extracted_item_type=extracted_item_type,
                source=source,
                instances_to_process=instances_to_process,
                document=document,
                overwrite_existing=overwrite_existing,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                num_retries_failed_request=num_retries_failed_request,
                async_limiter=async_limiter,
                raise_exception_on_extraction_error=raise_exception_on_extraction_error,
                has_fallback_pending=has_fallback_pending,
            )

        # Safe cast: a model that is not a decision model is an LLM
        llm = cast("_DocumentLLM", llm)

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
            modality=modality,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            max_images_to_analyze_per_call=max_images_to_analyze_per_call,
        )

        def merge_usage_data(
            existing: _ModelUsage | None, new: _ModelUsage
        ) -> _ModelUsage:
            """
            Aggregates usage data from all LLM calls.

            :param existing: The existing usage data to merge with the new usage data.
            :type existing: _ModelUsage | None
            :param new: The new usage data to merge with the existing usage data.
            :type new: _ModelUsage
            :return: The latest aggregated usage data from all LLM calls.
            :rtype: _ModelUsage
            """

            if existing is None:
                return new
            existing.input += new.input
            existing.output += new.output
            existing.cache_read += new.cache_read
            existing.cache_write += new.cache_write
            existing.calls += new.calls
            return existing

        all_usage_data: _ModelUsage | None = None
        sources_mapper: dict[
            str, dict[str, _Aspect | _Concept | list[_ExtractedItem]]
        ] = {}
        instances_enumerated = dict(enumerate(instances_to_process))

        # Each item on the list is a message kwargs dict for each extraction, based on specific context chunks.
        for idx, message_kwargs in enumerate(message_kwargs_list):
            logger.debug(
                f"Processing messages chunk {idx + 1}/{len(message_kwargs_list)}"
            )

            # Safe cast: message_kwargs is always a dict
            message_kwargs = cast(dict, message_kwargs)

            # Extract prompt_kwargs and remove it from message_kwargs
            # Safe cast: prompt_kwargs key always contains a dict
            prompt_kwargs = cast(dict, message_kwargs.pop("prompt_kwargs"))

            # Extract paragraphs and remove them from message_kwargs
            # Safe cast: "paragraphs_chunk" key, when present, always contains list[_Paragraph]
            paragraphs_chunk = cast(
                list["_Paragraph"] | None, message_kwargs.pop("paragraphs_chunk", None)
            )
            if paragraphs_chunk is not None:
                paragraphs_enumerated = dict(enumerate(paragraphs_chunk))
            else:
                paragraphs_enumerated = {}

            # Safe cast: references_apply key always contains a bool
            # Use `references_apply` instead of `add_references` since we may have the
            # `add_references` param set to True but for multimodal data extraction
            # such references only apply when text is present.
            references_apply = cast(bool, message_kwargs.pop("references_apply"))

            # Skip instances that have a single occurrence and already have extracted items
            discarded_instances = []
            if extracted_item_type == "concept":
                filtered_instances_to_process = []
                for i in instances_to_process:
                    i_in_source_mapper = sources_mapper.get(i.unique_id)
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
                        f"No {extracted_item_type}s left to process with LLM `{llm.model}`"
                    )
                    continue
                # Do not include instances with a single occurrence that already have extracted items
                # in the prompt kwargs
                prompt_kwargs["concepts"] = filtered_instances_to_process
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
            if extracted_item_type == "aspect":
                rendered_prompt = llm._extract_aspect_items_prompt.render(
                    **prompt_kwargs
                )
            elif extracted_item_type == "concept":
                rendered_prompt = llm._extract_concept_items_prompt.render(
                    **prompt_kwargs
                )
            else:
                raise ValueError(
                    f"Unsupported extracted item type: `{extracted_item_type}`"
                )

            # Build messages to send (system + user, with optional images)
            images: list[_Image] | None = message_kwargs.pop("images", None)
            messages: list[_Message] = []
            system_msg = llm._check_and_build_system_message()
            if system_msg is not None:
                messages.append(system_msg)
            messages.append(
                llm._build_user_message(message_text=rendered_prompt, images=images)
            )

            # Initialize the model call object to pass to the LLM query method
            model_call_obj = _ModelCall(
                prompt_kwargs=prompt_kwargs,
                prompt=rendered_prompt,
            )

            # Update the usage and cost of the LLM with the calls made for the previous
            # chunks before the next call. No usage data is returned if an exception is
            # raised later on, i.e. the usage of these calls would be lost otherwise.
            if all_usage_data is not None:
                await llm._update_usage_and_cost((None, all_usage_data))
                all_usage_data = None

            # Query LLM, process and validate results
            extracted_data, usage_data = await llm._query_llm(
                messages=messages,
                model_call_obj=model_call_obj,
                num_retries_failed_request=num_retries_failed_request,
                max_retries_failed_request=max_retries_failed_request,
                async_limiter=async_limiter,
                raise_exception_on_model_api_error=raise_exception_on_extraction_error,
                has_fallback_pending=has_fallback_pending,
            )
            all_usage_data = merge_usage_data(all_usage_data, usage_data)
            extracted_data = _validate_parsed_llm_output(
                parsed_json=_parse_llm_output_as_json(
                    _remove_thinking_content_from_llm_output(extracted_data)
                ),
                extracted_item_type=extracted_item_type,
                justification_provided=add_justifications,
                references_provided=references_apply,
                reference_depth=reference_depth,
            )
            if extracted_data is None:
                if model_call_obj._output_truncated:
                    # The response was cut off at the max output tokens, i.e. the same
                    # request to the same LLM would be cut off again. As a failed LLM
                    # API request, it is not requested again, and is not final while
                    # a fallback LLM, which may have a higher limit, is yet to be tried.
                    error_msg = (
                        f"{llm._get_output_truncated_msg()} "
                        f"To get less output per LLM call, consider setting "
                        f"`max_items_per_call`, `max_paragraphs_to_analyze_per_call` "
                        f"(for text) or `max_images_to_analyze_per_call` (for images). "
                        f"See the optimization guide for long documents: "
                        f"https://contextgem.dev/optimizations/optimization_long_docs/"
                    )
                    logger.error(error_msg)
                    if has_fallback_pending:
                        logger.info("The call will be retried with a fallback LLM.")
                    elif raise_exception_on_extraction_error:
                        await llm._update_usage_and_cost((None, all_usage_data))
                        raise ModelAPIError(error_msg)
                    else:
                        warnings.warn(error_msg, stacklevel=2)
                else:
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
                                try:
                                    for sent_dict in para_dict["sentences"]:
                                        sent_dict["sentence_id"] = int(
                                            sent_dict["sentence_id"].split("-S")[-1]
                                        )
                                except ValueError:
                                    logger.error(
                                        f"Sentence ID returned by LLM "
                                        f"for paragraph ID {para_id} is invalid"
                                    )
                                    return None, all_usage_data
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
                                        **extracted_item_kwargs  # ty: ignore[invalid-argument-type]
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

                    if add_justifications or references_apply:
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
                            if references_apply:
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
                            try:
                                extracted_item.reference_paragraphs = (
                                    reference_paragraphs
                                )
                                extracted_item.reference_sentences = reference_sentences
                            except ValueError as e:
                                # E.g. the same reference listed twice for the item
                                logger.error(
                                    f"References returned by LLM are invalid: {e}"
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
                source_instance = cast(_Aspect | _Concept, source_data["source"])
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
                    if overwrite_existing:
                        # The references of an aspect that is extracted again are
                        # those of its new items only: the sub-aspects and the concepts
                        # of the aspect are extracted from its reference paragraphs.
                        # Safe cast: source_instance is always an _Aspect instance
                        # because extracted_item_type is "aspect"
                        source_instance = cast(_Aspect, source_instance)
                        source_instance.reference_paragraphs = []
                        source_instance.reference_sentences = []
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
                        # The reference sentences of the aspect are those of its
                        # items, i.e. the extracted sentences, and not all the sentences
                        # of its reference paragraphs. A sentence is added also when
                        # its paragraph is a reference paragraph already, e.g. that of
                        # another sentence extracted from the same paragraph.
                        if source_instance.reference_depth == "sentences":
                            for ref_sent in ei.reference_sentences:
                                if ref_sent not in source_instance.reference_sentences:
                                    source_instance.reference_sentences += [ref_sent]
                source_instance._is_processed = True

        return instances_to_process, all_usage_data or _ModelUsage()

    async def _extract_instances(
        self,
        context: _Document | _Aspect,
        llm: _DocumentModel,
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
        :param llm: The model used for extraction: an LLM or a decision model
        :type llm: _DocumentModel
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
            LLM call (prompt), or whose text is sent to a decision model in one request for concepts.
            Defaults to 0, in which case all the paragraphs are analyzed.
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

        # If concurrency is enabled, get the limiter of the model, as well as of each
        # of its fallback models, for the running event loop. The limiters are keyed
        # by the ID of the model instance, so that the requests to each model are
        # paced by its own limiter. The limiter of a model is the same in all
        # the extraction calls that run in the event loop, so that the calls that
        # run concurrently, e.g. for the concepts of different aspects, share
        # the rate of the model.
        async_limiters: dict[int, AsyncLimiter] = (
            {
                id(llm_instance): llm_instance._get_async_limiter_for_running_loop()
                for llm_instance in [llm, *llm.fallback_models]
            }
            if use_concurrency
            else {}
        )

        async def run_extraction(
            instances_to_process: list[_Aspect] | list[_Concept],
            add_justifications: bool = False,
            justification_depth: JustificationDepth = "brief",
            justification_max_sents: int = 2,
            add_references: bool = False,
            reference_depth: ReferenceDepth = "paragraphs",
            modality: Modality = "text",
        ) -> None:
            """
            Utility function for extracting instances using the provided LLM.

            :param instances_to_process: List of instances to extract
            :type instances_to_process: list[_Aspect] | list[_Concept]
            :param add_justifications: Whether to provide justification for extracted items
            :type add_justifications: bool
            :param add_references: Whether to provide references for extracted items
            :type add_references: bool
            :param justification_depth: The level of detail of justifications. Details to "brief".
            :type justification_depth: JustificationDepth
            :param justification_max_sents: The maximum number of sentences in a justification.
                Defaults to 2.
            :type justification_max_sents: int
            :param reference_depth: The structural depth of the references, i.e. whether to provide
                paragraphs as references or sentences as references. Defaults to "paragraphs".
                ``extracted_items`` will have values based on this parameter.
            :type reference_depth: ReferenceDepth
            :param modality: The document content the concepts are extracted from.
                Defaults to "text".
            :type modality: Modality
            :return: None
            :rtype: None
            """

            if not instances_to_process:
                return None

            if llm is None:
                raise ValueError(
                    f"No LLM with role `{instances_to_process[0].model_role}` is defined in the group, "
                    f"while some {instance_type}s rely on such LLM."
                )

            def get_invalid_output_error_msg(
                model: _DocumentModel,
                instances: list[_Aspect] | list[_Concept],
            ) -> str:
                """
                Returns the error message for the instances that could not be
                processed because the model returned an output that cannot be used.

                :param model: The model that returned such output.
                :type model: _DocumentModel
                :param instances: The instances that could not be processed.
                :type instances: list[_Aspect] | list[_Concept]
                :return: The error message, which names the instances.
                :rtype: str
                """
                instance_names = [instance.name for instance in instances]
                return (
                    f"Some {instance_type}s could not be processed due to "
                    f"{model._invalid_output_desc} returned by {model._kind_name}. "
                    f"Failed {instance_type}s: {instance_names}"
                )

            async def retry_processing_for_result(
                llm_instance: _DocumentModel,
                res: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage],
                instances: list[_Aspect] | list[_Concept],
                n_retries: int = 0,
                retry_is_final: bool = False,
            ) -> bool:
                """
                Checks the processed result for validity and retries it if invalid.

                The retries stop as soon as an LLM API request fails, as such request
                is already retried (if the error is transient) when it is sent, or as
                soon as a response is cut off at the max output tokens, as the same
                request would be cut off again.

                :param llm_instance: The model instance to process the data.
                :type llm_instance: _DocumentModel
                :param res: Result to check and retry if invalid.
                :type res: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]
                :param instances: List of processed instances associated with the result.
                :type instances: list[_Aspect] | list[_Concept]
                :param n_retries: Number of retries to perform.
                :type n_retries: int
                :param retry_is_final: Whether the retry is final and will not be repeated by
                    a (next) fallback LLM.
                :type retry_is_final: bool
                :return: True if retry was successful, False otherwise.
                :rtype: bool
                """

                if n_retries <= 0:
                    return False
                if not _model_call_result_is_valid(res):
                    for i in range(n_retries):
                        logger.info(
                            f"Retrying {instance_type}s with "
                            f"{llm_instance._invalid_output_desc} "
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
                            modality=modality,
                            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                            max_images_to_analyze_per_call=max_images_to_analyze_per_call,
                            num_retries_failed_request=llm_instance.num_retries_failed_request,
                            max_retries_failed_request=llm_instance._provider_retries_failed_request,
                            async_limiter=async_limiters.get(id(llm_instance)),
                            raise_exception_on_extraction_error=raise_exception_on_extraction_error,
                            has_fallback_pending=not retry_is_final,
                        )
                        # Update usage stats and cost
                        await llm_instance._update_usage_and_cost(res)
                        if res[0] is not None:
                            break
                        if _model_call_result_is_api_failure(
                            res
                        ) or _model_call_result_is_output_truncated(res):
                            # The retries of the failed request are already performed,
                            # or the response was cut off at the max output tokens,
                            # i.e. sending it to the same LLM again is pointless
                            break
                    if _model_call_result_is_valid(res):
                        return True
                    else:
                        # A failed LLM API request, as well as a response cut off
                        # at the max output tokens, is already reported at this point
                        if retry_is_final and not (
                            _model_call_result_is_api_failure(res)
                            or _model_call_result_is_output_truncated(res)
                        ):
                            error_msg = get_invalid_output_error_msg(
                                llm_instance, instances
                            )
                            if raise_exception_on_extraction_error:
                                logger.error(error_msg)
                                raise ModelExtractionError(
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

            async def retry_chunk_with_fallbacks(
                chunk: list[_Aspect] | list[_Concept],
                result: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage],
            ) -> bool:
                """
                Retries a single failed chunk on the primary LLM and then on each
                fallback LLM in order (if configured), returning whether any
                retry path produced a valid result.

                :param chunk: The chunk of instances to retry.
                :type chunk: list[_Aspect] | list[_Concept]
                :param result: The original failed extraction result.
                :type result: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]
                :return: True if any retry succeeded, False otherwise.
                :rtype: bool
                """
                retry_successful = False
                # Retry on the primary LLM only if it returned invalid data. If its API
                # request failed, the retries of that request are already performed.
                # If its response was cut off at the max output tokens, the same
                # request would be cut off again.
                if (
                    llm._retries_invalid_data > 0
                    and not _model_call_result_is_api_failure(result)
                    and not _model_call_result_is_output_truncated(result)
                ):
                    retry_successful = await retry_processing_for_result(
                        llm_instance=llm,
                        res=result,
                        instances=chunk,
                        n_retries=llm._retries_invalid_data,
                        retry_is_final=not llm.fallback_models,
                    )
                n_fallbacks = len(llm.fallback_models)
                for i, fallback_llm in enumerate(llm.fallback_models):
                    if retry_successful:
                        break
                    logger.info(
                        f"Trying with fallback {fallback_llm._kind_name} "
                        f"{fallback_llm.model} ({i + 1}/{n_fallbacks})"
                    )
                    retry_successful = await retry_processing_for_result(
                        llm_instance=fallback_llm,
                        res=result,
                        instances=chunk,
                        n_retries=max(
                            fallback_llm._retries_invalid_data,
                            1,
                        ),  # retry with each fallback LLM at least once
                        retry_is_final=i == n_fallbacks - 1,
                    )
                return retry_successful

            def is_reported_failure(
                result: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]
                | None,
            ) -> bool:
                """
                Checks whether the invalid result of a call is already reported
                by the model that made the call, i.e. whether its API request failed
                or its response was cut off at the max output tokens. Such a result
                is not to be reported as an invalid output of the model as well.

                :param result: The invalid result of a call, or None if no result
                    is received.
                :type result: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]
                    | None
                :return: True if the failure is already reported, False otherwise.
                :rtype: bool
                """
                return _model_call_result_is_api_failure(
                    result
                ) or _model_call_result_is_output_truncated(result)

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
                "modality": modality,
                "max_paragraphs_to_analyze_per_call": max_paragraphs_to_analyze_per_call,
                "max_images_to_analyze_per_call": max_images_to_analyze_per_call,
                "num_retries_failed_request": llm.num_retries_failed_request,
                "max_retries_failed_request": llm._provider_retries_failed_request,
                "async_limiter": async_limiters.get(id(llm)),
                "raise_exception_on_extraction_error": raise_exception_on_extraction_error,
                "has_fallback_pending": bool(llm.fallback_models),
            }
            data_list = [
                {**base_params, "instances_to_process": chunk} for chunk in data_chunks
            ]

            if (
                use_concurrency and len(data_list) > 1
            ):  # Disable overhead if a single call

                async def extract_items_or_api_error(**kwargs: Any) -> Any:
                    """
                    Extracts items from a single chunk, returning a model API error
                    instead of raising it, so that the other chunks are processed
                    before the error is raised. Any other error is raised, which
                    stops the processing of the other chunks.

                    :param kwargs: Keyword arguments of `_extract_items_from_instances`.
                    :type kwargs: Any
                    :return: The extraction result, or the model API error raised
                        during the extraction.
                    :rtype: Any
                    """
                    try:
                        result = await self._extract_items_from_instances(**kwargs)
                    except ModelAPIError as e:
                        return e
                    completed_results.append(result)
                    return result

                # Process chunks concurrently
                completed_results: list[Any] = []
                try:
                    results = await _async_multi_executor(
                        func=extract_items_or_api_error,
                        data_list=data_list,
                    )
                except BaseException:
                    # Either a call raised an error that is not reported through
                    # its result, e.g. for a prompt that exceeds the context window
                    # of the model when there is no fallback model to try: the other
                    # calls are stopped by then, and the error is raised unchanged,
                    # as it is without concurrency. Or the extraction is cancelled,
                    # e.g. on a timeout set by the calling code, or as the extraction
                    # with another model of the group failed. In both cases, the calls
                    # completed by then are counted: their results, with the usage,
                    # would be lost otherwise.
                    for result in completed_results:
                        await llm._update_usage_and_cost(result)
                    raise
                if len(results) != len(data_chunks):
                    raise RuntimeError(
                        f"Number of results ({len(results)}) does not match "
                        f"number of data chunks ({len(data_chunks)})."
                    )

                # Update usage stats and cost
                for result in results:
                    if not isinstance(result, ModelAPIError):
                        await llm._update_usage_and_cost(result)

                # A model API error is raised only when it is final, i.e. the retries
                # of the failed request are already performed and there is no fallback
                # LLM to try. Re-raise the first such error by chunk index, instead of
                # sending the failed request again.
                for result in results:
                    if isinstance(result, ModelAPIError):
                        raise result

                # Retry failed chunks concurrently. Original chunk order is preserved
                # in `failed_pairs` so post-gather handling raises the first failure
                # by chunk index, matching prior sequential semantics.
                failed_pairs = [
                    (chunk, result)
                    for chunk, result in zip(data_chunks, results, strict=True)
                    if not _model_call_result_is_valid(result)
                ]
                if failed_pairs:
                    retry_outcomes = await asyncio.gather(
                        *(
                            retry_chunk_with_fallbacks(chunk, result)
                            for chunk, result in failed_pairs
                        ),
                        return_exceptions=True,
                    )
                    for (chunk, result), outcome in zip(
                        failed_pairs, retry_outcomes, strict=True
                    ):
                        if isinstance(outcome, BaseException):
                            raise outcome
                        if not outcome:
                            if llm._retries_invalid_data > 0 or llm.fallback_models:
                                if raise_exception_on_extraction_error:
                                    # Final retry was already performed, therefore an exception should have
                                    # been raised in case of extraction error such as invalid JSON.
                                    raise RuntimeError(
                                        "Extraction failed after all retries with "
                                        "`raise_exception_on_extraction_error` set to True, "
                                        "yet no exception was raised."
                                    )
                            elif not is_reported_failure(result):
                                error_msg = get_invalid_output_error_msg(llm, chunk)
                                if raise_exception_on_extraction_error:
                                    raise ModelExtractionError(error_msg, retry_count=0)
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
                    result = await self._extract_items_from_instances(**data)  # ty: ignore[invalid-argument-type]
                    logger.debug(
                        f"Result for chunk ({i + 1}/{len(data_list)}) processed."
                    )

                    # Update usage stats and cost
                    await llm._update_usage_and_cost(result)

                    # Retry if needed, including with fallback LLMs if they are provided
                    if not _model_call_result_is_valid(result):
                        retry_successful = await retry_chunk_with_fallbacks(
                            chunk=data["instances_to_process"],  # ty: ignore[invalid-argument-type]
                            result=result,
                        )
                        if not retry_successful:
                            if llm._retries_invalid_data > 0 or llm.fallback_models:
                                if raise_exception_on_extraction_error:
                                    # Final retry was already performed, therefore an exception should have
                                    # been raised in case of extraction error such as invalid JSON.
                                    raise RuntimeError(
                                        "Extraction failed after all retries with "
                                        "`raise_exception_on_extraction_error` set to True, "
                                        "yet no exception was raised."
                                    )
                            elif not is_reported_failure(result):
                                error_msg = get_invalid_output_error_msg(
                                    llm,
                                    data["instances_to_process"],  # ty: ignore[invalid-argument-type]
                                )
                                if raise_exception_on_extraction_error:
                                    raise ModelExtractionError(error_msg, retry_count=0)
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
            # An instance that is given more than once is extracted once
            instances_to_process = _get_distinct_instances(from_instances)
        else:
            instances_to_process = all_instances

        # First, select the instances to be extracted by the current LLM. In a model group,
        # an instance is routed to the LLM with the same role. A standalone LLM extracts
        # all the instances, regardless of their model roles.
        if self.is_group:
            filtered_instances = [
                i for i in instances_to_process if i.model_role == llm.role
            ]
        else:
            filtered_instances = list(instances_to_process)

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
        if instance_type == "concept":
            # Concepts with different modalities are extracted from different content
            # (aspects are always extracted from text)
            fields_to_group_by.append("modality")
        # Safe cast: filtered_instances contains only _Aspect or _Concept instances
        # based on instance_type
        typed_instances = (
            cast(list[_Aspect], filtered_instances)
            if instance_type == "aspect"
            else cast(list[_Concept], filtered_instances)
        )
        # A model of a kind whose requests do not depend on some of these params,
        # e.g. a decision model, is not called separately for the instances that
        # differ only in them
        instance_groups = llm._group_instances_for_calls(
            instance_type=instance_type,
            instances=typed_instances,
            fields=fields_to_group_by,
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
                f"with {llm._kind_name} `{llm.model}` with params: {params_combination}"
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

        :param parent_aspect: The parent aspect
        :type parent_aspect: _Aspect
        :return: None
        :rtype: None
        :raises ValueError: If any sub-aspect of the parent aspect has
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
        model_role: str | None = None,
    ) -> list[_ModelUsageOutputContainer | _ModelCostOutputContainer]:
        """
        Retrieves specified information (usage or cost) for either a single LLM or LLMs in a group.
        For groups, optionally filters the results by the specified model role.
        Iterates through primary LLMs and their fallback counterparts, collecting
        details about their usage or cost based on the specified retrieval type.

        :param retrieval_type: Determines the type of information to retrieve. Must be either
            "usage" to collect model usage statistics or "cost" to collect cost details.
        :type retrieval_type: Literal["usage", "cost"]
        :param model_role: The optional role of the LLM to filter the results. If provided,
            only results associated with LLMs matching this role are returned. If no LLM with
            the specified role exists, an exception is raised. If the matching LLM has
            fallback LLMs, their usage or cost details are also collected. The role is
            validated in the same way as the roles assigned to LLMs. Defaults to None.
        :type model_role: str | None
        :return: A list of containers, each representing usage or cost information for
            a primary LLM and its fallback LLMs, if any, in the order they are tried.
            The specific container type depends on the retrieval type.
        :rtype: list[_ModelUsageOutputContainer | _ModelCostOutputContainer]
        """

        info_containers = []
        # For individual LLM, use self and its fallbacks
        # For group, iterate through all LLMs in self.models
        llms_to_process = self.models if self.is_group else [self]  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        # Safe cast: llms_to_process contains only single models, of any kind
        llms_to_process = cast(list["_DocumentModel"], llms_to_process)

        for primary_llm in llms_to_process:
            for llm in [primary_llm, *primary_llm.fallback_models]:
                if retrieval_type == "usage":
                    info_container = _ModelUsageOutputContainer(
                        model=llm.model,
                        role=llm.role,
                        is_fallback=llm.is_fallback,
                        usage=llm._get_raw_usage(),
                    )
                elif retrieval_type == "cost":
                    info_container = _ModelCostOutputContainer(
                        model=llm.model,
                        role=llm.role,
                        is_fallback=llm.is_fallback,
                        cost=llm._get_raw_cost(),
                    )
                else:
                    raise ValueError(f"Invalid retrieval type `{retrieval_type}`")
                info_containers.append(info_container)

        if model_role is not None:
            # Validate the role in the same way as the roles assigned to LLMs,
            # e.g. remove the surrounding whitespace
            model_role = _validate_model_role(model_role)
            info_containers = [
                i for i in info_containers if i.role == model_role
            ]  # for both main and fallback LLMs
            if not info_containers:
                raise ValueError(
                    f"No {_get_common_kind_name(llms_to_process)} with the given role "
                    f"`{model_role}` was found in group."
                )
        return info_containers


@_disable_direct_initialization
class _DocumentModel(_GenericModelProcessor):
    """
    Base class of single models of any kind.

    This class holds what every kind of single model has, whatever requests it sends
    and whatever responses it gets: the model identifier and the API credentials,
    the role used for routing in a model group, the timeout and the retries of failed
    requests, the rate limiter of concurrent requests, the fallback models, and
    the tracking of usage and cost, together with the validation of these params.

    A kind of model, such as an LLM, is defined in a subclass, which adds the params
    and the processing logic of its own.
    """

    # Model config
    model: NonEmptyStr = Field(
        ...,
        description="Model identifier '<provider>/<model>' (e.g., 'openai/gpt-6-luna').",
    )
    api_key: NonEmptyStr | None = Field(
        default=None,
        # The key is a secret. It is not shown when the model is printed (`repr`),
        # e.g. in logs, or in the error of a validation that the model is the input
        # of. It is not serialized by pydantic (`exclude`) either: the `json()`
        # method of such an error, which writes the inputs out, writes the model
        # out without the key, and the serialized data of the model (`to_dict()`)
        # has the key as not set.
        # In the validation of the model itself, the key is an input as it is
        # passed, not as a part of a model: the text of the errors of that
        # validation shows no inputs at all (see `model_config` below).
        repr=False,
        exclude=True,
        description="API key for provider authentication; optional for local models.",
    )
    api_base: NonEmptyStr | None = Field(
        default=None,
        description="Base URL of the provider API endpoint.",
    )
    role: ModelRole | None = Field(
        default=None,
        description=(
            "Free-form role label (up to 64 printable characters) used for routing "
            "in a DocumentModelGroup: the LLM extracts the aspects/concepts with "
            "the same `model_role`. Defaults to None, "
            "i.e. the LLM extracts the aspects/concepts without `model_role`. Ignored "
            "when the LLM is used on its own, as it then extracts all aspects/concepts."
        ),
    )
    num_retries_failed_request: StrictInt = Field(
        default=3,
        ge=0,
        description=(
            "Retry count for requests that fail with a transient error "
            "(timeout, rate limit, server or connection error); "
            "retried with exponential backoff and jitter."
        ),
    )
    timeout: StrictInt = Field(
        default=600,
        ge=0,
        description="Request timeout in seconds.",
    )
    pricing_details: _ModelPricing | None = Field(
        default=None,
        description="Explicit pricing configuration for cost calculation.",
    )
    is_fallback: StrictBool = Field(
        default=False,
        description="Marks this LLM as a fallback model.",
    )
    fallback_models: list[_DocumentModel] = Field(
        default_factory=list,
        description=(
            "Fallback LLMs to try, in the given order, if this one fails; "
            "each must have the same role and output_language."
        ),
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

    # Async
    _async_limiter: AsyncLimiter = PrivateAttr()
    # Limiters that pace the requests, by event loops, at the rate of the limiter above
    _loop_async_limiters: _LoopAsyncLimiters = PrivateAttr(
        default_factory=_LoopAsyncLimiters
    )

    # Token counting
    _usage: _ModelUsage = PrivateAttr(default_factory=_ModelUsage)
    # Processing cost
    _cost: _ModelCost = PrivateAttr(default_factory=_ModelCost)

    # Async lock to guard shared state during async updates
    _async_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    # Fallback LLMs as of the last validation. The config params of the fallback LLMs
    # are checked for redundancy when the fallback LLMs are set, not each time
    # another param of the LLM is assigned.
    _validated_fallback_models: tuple[_DocumentModel, ...] = PrivateAttr(default=())

    # Private attributes must be set only once before model validation.
    # We shouldn't set private attributes in post-init method because
    # some model validators which run before post-init require access to
    # such attributes. We also want to avoid re-setting private attributes
    # in every call of a model validator that validates any new assignment.
    _private_attrs_initialized: bool = PrivateAttr(default=False)

    # The text of a validation error of a model does not show the inputs of
    # the validation (`input_value=..., input_type=...`), as the API key is among
    # them: when the key itself is rejected, e.g. a key of a wrong type, and when
    # a check that involves several params fails, the input of which is all
    # the params that the model is created with. Such an input is cut in the middle
    # in the text of the error, i.e. the beginning or the end of the key would be
    # shown there. The error still names the param and tells what is wrong with it.
    # This applies to the text of the error only: the inputs are still returned by
    # its `errors()` and `json()` methods, unless `include_input=False` is passed.
    # A model that is among the inputs is returned by `errors()` as it is, and is
    # written out by `json()` without its key (see `api_key` above). A key that is
    # among the inputs as it is passed, as in the two cases above, is returned in
    # clear text by both methods.
    model_config = ConfigDict(hide_input_in_errors=True)

    def __init__(self, **data: Any):
        # Pop the async_limiter if provided; otherwise use a default.
        limiter = data.pop("async_limiter", None)

        super().__init__(**data)
        if limiter is not None:
            self.async_limiter = limiter
        else:
            self.async_limiter = AsyncLimiter(10, 10)

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

        # Log the information that applies to the specific kind of the model
        self._log_kind_specific_info()

        # Warn about auto-pricing accuracy limitations
        if self.auto_pricing:
            warnings.warn(
                "Auto-pricing: these prices will not be 100% accurate. "
                "The price data cannot be exactly correct because model providers do not provide "
                "exact price information for their APIs in a format which can be reliably processed. "
                "See Pydantic's genai-prices https://github.com/pydantic/genai-prices for more details.",
                stacklevel=2,
            )

    def _log_kind_specific_info(self) -> None:
        """
        Logs the information on the model that applies to its specific kind only,
        when the model is created. Does nothing by default: to be overridden
        in the subclasses that have such information.

        :return: None
        :rtype: None
        """
        return None

    def _set_private_attrs(self) -> None:
        """
        Initializes and configures private attributes for the LLM instance.
        Runs only once.

        :return: None
        :rtype: None
        """

        if self._private_attrs_initialized:
            return

        # Set the private attributes that only the specific kind of the model has
        self._set_kind_specific_private_attrs()
        if self.auto_pricing:
            self._set_provider_and_model_for_auto_pricing()

        self._private_attrs_initialized = True

    def _set_kind_specific_private_attrs(self) -> None:
        """
        Initializes and configures the private attributes that only the specific
        kind of the model has. Does nothing by default: to be overridden
        in the subclasses that have such attributes.

        :return: None
        :rtype: None
        """
        return None

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
        :return: None
        :rtype: None
        :raises TypeError: If value is not an AsyncLimiter instance.
        """
        if not isinstance(value, AsyncLimiter):
            raise TypeError("async_limiter must be an AsyncLimiter instance")
        self._async_limiter = value

    def _get_async_limiter_for_running_loop(self) -> AsyncLimiter:
        """
        Returns the limiter that paces the requests to the model in the running
        event loop, at the rate of the limiter set for the model
        (see :attr:`async_limiter`).

        The limiter is the same for all the extraction calls that run in the event
        loop, including the calls that run concurrently, e.g. those that extract
        the concepts of different aspects, so that the requests of all of them
        together stay within the rate.

        :return: The limiter of the running event loop.
        :rtype: AsyncLimiter
        :raises RuntimeError: If no event loop is running.
        """
        return self._loop_async_limiters.get(
            max_rate=self._async_limiter.max_rate,
            time_period=self._async_limiter.time_period,
        )

    @property
    def is_group(self) -> bool:
        """
        Returns False indicating this is a single LLM, not a group.

        :return: Always False for DocumentLLM instances.
        :rtype: bool
        """
        return False

    @property
    def list_roles(self) -> list[str | None]:
        """
        Returns a list containing the role of this LLM.

        (For a single LLM, this returns a list with just one element - the LLM's role,
        or None if the LLM has no role. For model groups, the method implementation returns
        roles of all LLMs in the group.)

        :return: A list containing the role of this LLM.
        :rtype: list[str | None]
        """
        return [self.role]

    @property
    def _kind_name(self) -> str:
        """
        Returns the name of the kind of the model, as used in the messages of
        the logic that models of all kinds share. To be overridden in the subclasses.

        :return: The name of the kind of the model, e.g. "LLM".
        :rtype: str
        """
        return _ANY_KIND_NAME

    @property
    def _invalid_output_desc(self) -> str:
        """
        Returns what the model is said to have returned when its output cannot
        be used, as worded in the messages on the extraction that fails for it.
        To be overridden in the subclasses of the kinds of models whose output
        is invalid in a specific way, e.g. is not a valid JSON.

        :return: The description of an output of the model that cannot be used.
        :rtype: str
        """
        return "an invalid response"

    @property
    def _retries_invalid_data(self) -> int:
        """
        Returns the number of times that a call is made again when the model returns
        invalid data for it. Zero by default, i.e. such a call is not made again:
        to be overridden in the subclasses of the kinds of models that may return
        valid data when the same call is repeated, as an LLM may.

        :return: The number of retries of a call with invalid data returned.
        :rtype: int
        """
        return 0

    @property
    def _provider_retries_failed_request(self) -> int:
        """
        Returns the number of retries of a failed request that are left to the SDK
        of the model provider, in addition to the retries performed by the framework
        (``num_retries_failed_request``). Zero by default, i.e. there are no such
        retries: to be overridden in the subclasses of the kinds of models whose
        requests are sent through such an SDK.

        :return: The number of retries of a failed request by the provider's SDK.
        :rtype: int
        """
        return 0

    def _check_instances_servable(
        self,
        instances: Sequence[_Aspect | _Concept],
        routed_in_group: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> None:
        """
        Checks, before any requests are sent, that the model can extract the given
        aspects and concepts, which are to be extracted by it. Does nothing by
        default, i.e. the model extracts any aspect or concept: to be overridden
        in the subclasses of the kinds of models that extract only some of them.

        :param instances: The aspects and concepts to be extracted by the model.
        :type instances: Sequence[_Aspect | _Concept]
        :param routed_in_group: Whether the aspects and concepts are routed to
            the model by its role in a model group, which is what the error is
            to tell how to change in such a case. Defaults to False.
        :type routed_in_group: bool
        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs to
            analyze per call that the extraction is run with, for the kinds of
            models for which it decides what they can extract. Defaults to 0
            (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :return: None
        :rtype: None
        :raises ValueError: If the model cannot extract some of the aspects
            or concepts.
        """
        return None

    def _group_instances_for_calls(
        self,
        instance_type: ExtractedInstanceType,
        instances: list[_Aspect] | list[_Concept],
        fields: list[str],
    ) -> list[list[_Aspect] | list[_Concept]]:
        """
        Groups the aspects or concepts to be extracted by the model into the groups
        that are extracted in separate calls. By default, the instances of a group
        have the same values of all the given extraction params, each of which
        changes the prompt of an LLM and the output expected from it: to be
        overridden in the subclasses of the kinds of models whose requests do not
        depend on some of the params.

        :param instance_type: Type of the instances ("aspect" or "concept").
        :type instance_type: ExtractedInstanceType
        :param instances: The aspects or concepts to be extracted by the model.
        :type instances: list[_Aspect] | list[_Concept]
        :param fields: The extraction params of the instances to group them by.
        :type fields: list[str]
        :return: The groups of the instances.
        :rtype: list[list[_Aspect] | list[_Concept]]
        """
        return _group_instances_by_fields(fields=fields, instances=instances)

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
        # Check what the specific kind of the model requires of the model identifier
        cls._validate_kind_specific_model_id(model)
        return model

    @classmethod
    def _validate_kind_specific_model_id(cls, model: str) -> None:
        """
        Validates the model identifier against what the specific kind of the model
        requires of it, e.g. a provider that serves models of that kind. Does
        nothing by default: to be overridden in the subclasses that have such
        requirements.

        Called when the model identifier is validated, after it is checked to be
        in the form of ``<model_provider>/<model_name>``.

        :param model: The model identifier.
        :type model: str
        :return: None
        :rtype: None
        :raises ValueError: If the model identifier does not meet the requirements.
        """
        return None

    @field_validator("fallback_models")
    @classmethod
    def _validate_fallback_models(
        cls,
        fallback_models: list[_DocumentModel],
    ) -> list[_DocumentModel]:
        """
        Validates the ``fallback_models`` input to ensure each LLM meets the expected
        condition of being a fallback LLM model.

        :param fallback_models: The list of _DocumentModel instances to be validated.
        :type fallback_models: list[_DocumentModel]
        :return: The valid fallback_models that meet the expected criteria.
        :rtype: list[_DocumentModel]
        :raises ValueError: If any of the ``fallback_models`` is not a fallback model, as
            indicated by the ``is_fallback`` attribute set to ``False``.
        """
        for fallback_llm in fallback_models:
            if not fallback_llm.is_fallback:
                raise ValueError(
                    f"Fallback {fallback_llm._kind_name} must be a fallback model. "
                    f"Use `is_fallback=True`."
                )
        return fallback_models

    @model_validator(mode="after")
    def _ensure_private_attrs(self) -> Self:
        """
        Ensures private attributes are available for any code in/after this validator.

        :return: The instance of the current LLM model after setting private attributes.
        :rtype: Self
        """
        self._set_private_attrs()
        return self

    @model_validator(mode="after")
    def _validate_document_model_post(self) -> Self:
        """
        Validate the integrity of the model after initialization: its pricing params
        and its fallback models.

        :return: The instance of the current model after successful validation.
        :rtype: Self
        :raises ValueError: If the model is not properly configured.
        """

        # pricing_details and auto_pricing are mutually exclusive
        if self.pricing_details is not None and self.auto_pricing:
            raise ValueError(
                "`pricing_details` cannot be set together with `auto_pricing=True`. "
                "Provide manual ModelPricing or enable auto_pricing, not both."
            )

        if (
            any(self.model.startswith(provider) for provider in _LOCAL_MODEL_PROVIDERS)
            and self.auto_pricing
        ):
            raise ValueError(
                "`auto_pricing=True` is not supported for local models. "
                "Disable `auto_pricing` or provide explicit `pricing_details`."
            )

        # The models are named in the messages by their kind, e.g. as LLMs.
        # A fallback model may be of another kind than the main one.
        kind = self._kind_name

        # Fallback model validation
        if self.is_fallback and self.fallback_models:
            raise ValueError(
                f"Fallback {kind} cannot have its own fallback "
                f"{_get_common_kind_name([self, *self.fallback_models])}s "
                f"and must be attached to a non-fallback model."
            )

        # A fallback LLM with the same config params as the main LLM or as another
        # fallback LLM is redundant. This is checked when the fallback LLMs are set,
        # i.e. when the LLM is created with them or they are assigned, except when
        # the LLM is deserialized: the API credentials are removed from the serialized
        # data, i.e. the LLMs that differ only in `api_key` or `api_base` (e.g. the same
        # model served from another endpoint) have the same config params in such data.
        # For the same reason, this is not checked when another param is assigned:
        # the API credentials of a deserialized LLM and of its fallback LLMs are set
        # again one by one, i.e. some of these LLMs have the same config params
        # until all the credentials are set.
        fallback_models_are_set = len(self.fallback_models) != len(
            self._validated_fallback_models
        ) or any(
            current is not validated
            for current, validated in zip(
                self.fallback_models, self._validated_fallback_models, strict=True
            )
        )
        check_same_config = (
            fallback_models_are_set and not _is_deserialization_in_progress()
        )

        def get_config_params(llm: _DocumentModel) -> dict[str, Any]:
            """
            Returns the config params of the LLM to compare it with another LLM:
            without the `is_fallback` and `fallback_models` params, and with the tools
            as a set, as the order of the tools makes no difference.

            :param llm: The LLM to get the config params of.
            :type llm: _DocumentModel
            :return: The config params by name.
            :rtype: dict[str, Any]
            """
            config_params = {
                k: v
                for k, v in llm.__dict__.items()
                if k not in ["is_fallback", "fallback_models"]
            }
            if config_params.get("tools") is not None:
                config_params["tools"] = set(config_params["tools"])
            # Models of different classes are different, whatever their config params
            config_params["__class__"] = type(llm)
            return config_params

        # Config params of the main LLM and of the fallback LLMs validated so far
        main_llm_dict = get_config_params(self)
        fallback_llm_dicts: list[dict[str, Any]] = []

        for fallback_llm in self.fallback_models:
            fallback_kind = fallback_llm._kind_name

            # Check for the consistency of the fallback LLM role
            if fallback_llm.role != self.role:
                raise ValueError(
                    f"The fallback {fallback_kind} must have the same role `{self.role}` as the main one."
                )

            # Check the params that the specific kind of the model requires its
            # fallback models to share with it, e.g. the output language of an LLM
            self._validate_kind_specific_fallback(fallback_llm)

            # Check that the fallback LLM is not the replica of the main LLM or of another
            # fallback LLM, just with different `is_fallback` and `fallback_models` params.
            # Only the models of the same kind can have the same config params.
            fallback_llm_dict = get_config_params(fallback_llm)
            if check_same_config:
                if main_llm_dict == fallback_llm_dict:
                    raise ValueError(
                        f"Fallback {fallback_kind} must not have the exact same config params as the main {kind}."
                    )
                if fallback_llm_dict in fallback_llm_dicts:
                    raise ValueError(
                        f"Fallback {fallback_kind}s must not have the exact same config params as each other."
                    )
            fallback_llm_dicts.append(fallback_llm_dict)

        # The fallback LLMs are valid, i.e. are not to be checked for redundancy again
        self._validated_fallback_models = tuple(self.fallback_models)

        # Update the state derived from the params of the specific kind of the model,
        # and emit the warnings on these params, now that all the checks have passed
        self._finish_kind_specific_validation()

        return self

    def _validate_kind_specific_fallback(self, fallback_model: _DocumentModel) -> None:
        """
        Validates a fallback model against the params that the specific kind of
        the current model requires its fallback models to share with it. Does nothing
        by default: to be overridden in the subclasses that have such params.

        Called for each fallback model after its role is validated, and before it is
        checked for being a replica of the current model or of another fallback model.

        :param fallback_model: The fallback model to validate.
        :type fallback_model: _DocumentModel
        :return: None
        :rtype: None
        :raises ValueError: If the fallback model does not share such a param
            with the current model.
        """
        return None

    def _finish_kind_specific_validation(self) -> None:
        """
        Updates the state derived from the params of the specific kind of the model,
        and emits the warnings on these params, after all the params of the model
        are validated. Does nothing by default: to be overridden in the subclasses
        that have such state or warnings.

        :return: None
        :rtype: None
        """
        return None

    def _eq_deserialized_model_config(self, other: _DocumentModel) -> bool:
        """
        Custom config equality method to compare this model with a deserialized instance.

        Compares the __dict__ of both instances and performs specific checks for
        certain attributes that require special handling: first for what models
        of all kinds have, then for what only the specific kind of the model has
        (see ``_eq_deserialized_kind_specific_config``).

        Note that, by default, the reconstructed deserialized model will be only partially
        equal (==) to the original one, as the api credentials are redacted, and the async
        limiter and the async lock are not serialized and point to different objects
        in memory post-initialization. Also, usage and cost are reset by default pre-serialization.

        :param other: Another model instance to compare with
        :type other: _DocumentModel
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """

        kind = self._kind_name

        # A model deserialized as an instance of another class is another model
        if type(self) is not type(other):
            logger.debug(f"Class of deserialized {kind} is different.")
            return False

        # Create a copy of the dictionaries to modify
        self_dict = self.__dict__.copy()
        other_dict = other.__dict__.copy()

        # Fallback models are compared separately, each with this same method
        self_fallback_models = self_dict.pop("fallback_models")
        other_fallback_models = other_dict.pop("fallback_models")
        if len(self_fallback_models) != len(other_fallback_models):
            raise RuntimeError(f"Deserialized fallback {kind}s were not properly set")
        for self_fallback_llm, other_fallback_llm in zip(
            self_fallback_models, other_fallback_models, strict=True
        ):
            if not self_fallback_llm._eq_deserialized_model_config(other_fallback_llm):
                logger.debug(
                    f"Fallback {kind} config of deserialized {kind} is different."
                )
                return False

        # Skip checks for api_key and api_base that were redacted pre-serialization
        self_dict.pop("api_key")
        other_dict.pop("api_key")
        self_dict.pop("api_base")
        other_dict.pop("api_base")
        if not (other.api_key is None and other.api_base is None):
            raise RuntimeError(
                f"Deserialized {kind} has api_key or api_base set, "
                f"while API credentials should have been redacted pre-serialization"
            )

        # Compare the modified dictionaries
        if self_dict != other_dict:
            logger.debug(f"{kind} __dict__ of deserialized {kind} is different.")
            return False

        # Special checks for specific private attributes

        # Check that usage and cost stats were reset pre-serialization
        if not (other._usage == _ModelUsage() and other._cost == _ModelCost()):
            raise RuntimeError(
                "Usage and cost stats were not properly reset during serialization"
            )

        # Check _async_limiter
        if (
            self._async_limiter.time_period != other._async_limiter.time_period
            or self._async_limiter.max_rate != other._async_limiter.max_rate
        ):
            logger.debug(f"Async limiter params of deserialized {kind} are different.")
            return False

        # Check _async_lock
        if not (
            isinstance(self._async_lock, asyncio.Lock)
            and isinstance(other._async_lock, asyncio.Lock)
        ):
            logger.debug(f"Async lock of deserialized {kind} is different.")
            return False

        # Check _private_attrs_initialized
        if self._private_attrs_initialized != other._private_attrs_initialized:
            logger.debug(
                f"`_private_attrs_initialized` of deserialized {kind} is different."
            )
            return False

        # Check what only the specific kind of the model has
        return self._eq_deserialized_kind_specific_config(other)

    def _eq_deserialized_kind_specific_config(self, other: _DocumentModel) -> bool:
        """
        Compares what only the specific kind of the model has with the same
        of a deserialized instance of the same class, e.g. the prompts and
        the capabilities of an LLM. Nothing is compared by default: to be overridden
        in the subclasses whose instances have such attributes.

        Called at the end of ``_eq_deserialized_model_config``, i.e. when what models
        of all kinds have is found to be equal.

        :param other: Another instance of the same class to compare with
        :type other: _DocumentModel
        :return: True if the instances are equal in what is compared, False otherwise
        :rtype: bool
        """
        return True

    def _get_raw_usage(self) -> _ModelUsage:
        """
        Retrieves the raw usage information of the LLM.

        :return: _ModelUsage object containing usage data for the LLM.
        :rtype: _ModelUsage
        """
        return self._usage

    def _get_raw_cost(self) -> _ModelCost:
        """
        Retrieves the cost information of the LLM, quantized for reporting.

        The internal accumulators keep full precision. Quantization is applied
        only on access to ensure stable presentation while avoiding stepwise
        rounding errors during accumulation.

        :return: _ModelCost object containing quantized cost data for the LLM.
        :rtype: _ModelCost
        """

        if self.pricing_details is None and not self.auto_pricing:
            kind = self._kind_name
            logger.info(
                f"No pricing params provided for the {kind} `{self.model}`"
                f"{f' with role `{self.role}`' if self.role else ''}. "
                f"Costs for this {kind} were not calculated."
            )

        # Quantize for presentation without mutating internal state
        input_q = self._cost.input.quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        output_q = self._cost.output.quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        total_q = (input_q + output_q).quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        return _ModelCost(input=input_q, output=output_q, total=total_q)

    def get_usage(self) -> list[_ModelUsageOutputContainer]:
        """
        Retrieves the usage information of the LLM and its fallback LLM if configured.

        This method collects token usage statistics for the current LLM instance and its
        fallback LLM (if configured), providing insights into API consumption.

        :return: A list of usage statistics containers for the LLM and its fallback.
        :rtype: list[_ModelUsageOutputContainer]
        """

        # Safe cast: _get_usage_or_cost with retrieval_type="usage"
        # returns only list of _ModelUsageOutputContainer
        return cast(
            list["_ModelUsageOutputContainer"],
            self._get_usage_or_cost(retrieval_type="usage"),
        )

    def get_cost(self) -> list[_ModelCostOutputContainer]:
        """
        Retrieves the accumulated cost information of the LLM and its fallback LLM if configured.

        This method collects cost statistics for the current LLM instance and its
        fallback LLM (if configured), providing insights into API usage expenses.

        :return: A list of cost statistics containers for the LLM and its fallback.
        :rtype: list[_ModelCostOutputContainer]
        """

        # Safe cast: _get_usage_or_cost with retrieval_type="cost"
        # returns only list of _ModelCostOutputContainer
        return cast(
            list["_ModelCostOutputContainer"],
            self._get_usage_or_cost(retrieval_type="cost"),
        )

    def reset_usage_and_cost(self) -> None:
        """
        Resets the usage and cost statistics for the LLM and its fallback LLMs (if configured).

        This method clears accumulated usage and cost data, which is useful when processing
        multiple documents sequentially and tracking metrics for each document separately.

        :return: None
        :rtype: None
        """

        for llm in [self, *self.fallback_models]:
            llm._usage = _ModelUsage()
            llm._cost = _ModelCost()

    @tethered.scope(
        allow=list(_GENAI_PRICES_REFRESH_HOSTS),
        label="contextgem._calculate_auto_pricing_costs",
        hint=(
            "ContextGem refreshes model pricing data from raw.githubusercontent.com "
            "via genai-prices when DocumentLLM(auto_pricing_refresh=True). "
            "If your app uses tethered.activate(), allow raw.githubusercontent.com "
            "— or set auto_pricing_refresh=False to use bundled pricing data, "
            "or set pricing_details=ModelPricing(...) to bypass genai-prices entirely. "
            "See https://contextgem.dev/llms/llm_config/"
        ),
    )
    def _calculate_auto_pricing_costs(
        self,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> tuple[Decimal, Decimal]:
        """
        Utility method to calculate costs using genai-prices for the given
        input/output token counts. Requires `_auto_pricing_provider_id` and
        `_auto_pricing_model_ref` to be set.

        The input tokens read from and written to the prompt cache are priced
        at the cache prices of the model, where genai-prices has such prices,
        and as regular input tokens otherwise.

        :param input_tokens: Number of input tokens, including the tokens read from
            and written to the prompt cache
        :type input_tokens: int
        :param output_tokens: Number of output tokens
        :type output_tokens: int
        :param cache_read_tokens: Number of input tokens read from the prompt cache.
            Defaults to 0.
        :type cache_read_tokens: int
        :param cache_write_tokens: Number of input tokens written to the prompt cache.
            Defaults to 0.
        :type cache_write_tokens: int
        :return: Tuple of (input_cost, output_cost) as Decimals, where the input cost
            includes the cost of the tokens read from and written to the prompt cache
        :rtype: tuple[Decimal, Decimal]
        :raises LookupError: If auto-pricing provider/model are not set
        :raises Exception: Propagates exceptions from genai-prices lookup/calculation
        """

        if (
            self._auto_pricing_provider_id is None
            or self._auto_pricing_model_ref is None
        ):
            raise LookupError(
                f"Auto-pricing provider/model are not set for this {self._kind_name} instance"
            )

        # Cache tokens are a part of the input tokens, in genai-prices as well
        usage = _GPUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )

        if self.auto_pricing_refresh and not self._auto_pricing_refresh_attempted:
            # Attempt price update only once, as it involves a network call
            self._auto_pricing_refresh_attempted = True
            kind = self._kind_name
            with _GPUpdatePrices() as update_prices:
                logger.info(
                    f"Updating {kind} API prices since `auto_pricing_refresh=True` for model `{self.model}`"
                )
                try:
                    update_prices.wait()
                    logger.info(f"Finished updating {kind} API prices")
                except Exception as e:
                    logger.error(f"Error updating {kind} API prices: {e}.")
                price_data = _calc_auto_price(
                    usage,
                    model_ref=self._auto_pricing_model_ref,
                    provider_id=self._auto_pricing_provider_id,
                )
        else:
            price_data = _calc_auto_price(
                usage,
                model_ref=self._auto_pricing_model_ref,
                provider_id=self._auto_pricing_provider_id,
            )
        input_cost = Decimal(str(price_data.input_price))
        output_cost = Decimal(str(price_data.output_price))
        return input_cost, output_cost

    def _increment_cost(self, usage: _ModelUsage) -> None:
        """
        Calculates and increments the self._cost attribute values based on
        the additional usage details provided. Relevant only if the user has
        provided pricing details for the LLM or if auto-pricing is enabled.

        :param usage: _ModelUsage instance containing usage information on
                      additional number of input and output tokens processed,
                      including the input tokens read from and written to
                      the prompt cache, and on the calls that used these tokens.
        :type usage: _ModelUsage

        :return: None
        :rtype: None
        """

        if self.pricing_details or self.auto_pricing:
            mil_dec = Decimal("1000000")
            cost_input: Decimal = Decimal("0")
            cost_output: Decimal = Decimal("0")

            if self.pricing_details:
                pricing = self.pricing_details
                input_price = Decimal(str(pricing.input_per_1m_tokens))
                # Cache tokens are a part of the input tokens, and are priced
                # as regular input tokens unless they have a price of their own
                cache_read_price = (
                    input_price
                    if pricing.cache_read_per_1m_tokens is None
                    else Decimal(str(pricing.cache_read_per_1m_tokens))
                )
                cache_write_price = (
                    input_price
                    if pricing.cache_write_per_1m_tokens is None
                    else Decimal(str(pricing.cache_write_per_1m_tokens))
                )
                regular_input = usage.input - usage.cache_read - usage.cache_write
                cost_input = (
                    Decimal(str(regular_input)) * input_price
                    + Decimal(str(usage.cache_read)) * cache_read_price
                    + Decimal(str(usage.cache_write)) * cache_write_price
                ) / mil_dec
                cost_output = (Decimal(str(usage.output)) / mil_dec) * Decimal(
                    str(pricing.output_per_1m_tokens)
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
                    # Each call is priced on its own, as the prices of some models
                    # depend on the number of input tokens in a request (price tiers),
                    # which the usage summed up over several calls would misrepresent
                    call_costs = [
                        self._calculate_auto_pricing_costs(
                            call._input_tokens,
                            call._output_tokens,
                            cache_read_tokens=call._cache_read_tokens,
                            cache_write_tokens=call._cache_write_tokens,
                        )
                        for call in usage.calls
                    ]
                    if not all(
                        isinstance(cost, Decimal)
                        for call_cost in call_costs
                        for cost in call_cost
                    ):
                        raise RuntimeError("Auto-pricing returned non-Decimal values")
                    cost_input = sum(
                        (call_cost[0] for call_cost in call_costs), Decimal("0")
                    )
                    cost_output = sum(
                        (call_cost[1] for call_cost in call_costs), Decimal("0")
                    )
                except Exception as e:
                    warning_msg = (
                        f"Unable to fetch pricing data for model `{self.model}`: {e}. "
                        "Auto-pricing is skipped. "
                        f"{self._get_missing_auto_pricing_advice()}"
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
        self, result: tuple[Any, _ModelUsage] | None
    ) -> None:
        """
        Updates the model usage and cost details based on the given processing result.
        This method modifies the LLM instance's usage statistics, whether or not pricing
        params are provided, and increments the associated cost if they are provided.

        :param result: A tuple containing an optional value and usage data. The usage
            data is used to update the instance's input and output usage, as well as
            the total cost. If the result is None, the method does nothing.
        :type result: tuple[Any, _ModelUsage]
        :return: None
        :rtype: None
        """

        async with self._async_lock:
            if result is None:
                return
            new_usage = result[1]
            # Tokens data
            self._usage.input += new_usage.input
            self._usage.output += new_usage.output
            self._usage.cache_read += new_usage.cache_read
            self._usage.cache_write += new_usage.cache_write
            # Pricing data (if pricing is configured)
            self._increment_cost(new_usage)
            # Calls data
            self._usage.calls += new_usage.calls

    def _get_missing_auto_pricing_advice(self) -> str:
        """
        Returns the advice for the warnings issued when the pricing data of the model
        cannot be fetched. Unless the refresh of the pricing data is already enabled,
        it is suggested first, as the pricing data bundled with genai-prices may not
        include a recently released model.

        :return: Advice on getting the costs of the LLM calculated
        :rtype: str
        """

        if self.auto_pricing_refresh:
            return (
                "Consider setting `pricing_details=ModelPricing(...)` "
                "or disabling auto-pricing."
            )
        return (
            "The bundled pricing data may not include recently released models: "
            "consider setting `auto_pricing_refresh=True` to fetch the latest "
            "pricing data, setting `pricing_details=ModelPricing(...)`, "
            "or disabling auto-pricing."
        )

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
                f"Consider setting `pricing_details=ModelPricing(...)` "
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
                f"{self._get_missing_auto_pricing_advice()}"
            )
            warnings.warn(
                warning_msg,
                stacklevel=2,
            )


@_disable_direct_initialization
class _DocumentModelGroup(_GenericModelProcessor):
    """
    Internal implementation of the ``DocumentModelGroup`` class.
    """

    models: list[_DocumentModel] = Field(
        ...,
        min_length=2,
        description=(
            "List of DocumentLLM instances with unique roles, i.e. at most one LLM "
            "without a role; minimum 2. An aspect/concept is routed to the LLM with "
            "the `role` equal to the `model_role` of the aspect/concept."
        ),
    )
    output_language: LanguageRequirement = Field(
        default="en",
        description=(
            "Language for generated textual output (justifications, explanations). "
            "'en' forces English; 'adapt' matches document/image language. "
            "Must be consistent across all LLMs in the group. "
            "Applies only when DocumentLLMs' default system messages are used."
        ),
    )

    # The text of a validation error of a group does not show the inputs of
    # the validation either. The models of a group are shown without their API
    # keys, but an API key that is passed to the group itself by mistake (a group
    # has no such param: the key is set for each model) would be shown in the text
    # of the error that rejects it, as the value of any input that is not a model.
    model_config = ConfigDict(hide_input_in_errors=True)

    def _set_private_attrs(self) -> None:
        """
        Initialize and configure private attributes for the model group.
        The model group has no private attributes to set.

        :return: None
        :rtype: None
        """
        return None

    @property
    def is_group(self) -> bool:
        """
        Returns True indicating this is a group of LLMs.

        :return: Always True for DocumentModelGroup instances.
        :rtype: bool
        """
        return True

    @property
    def list_roles(self) -> list[str | None]:
        """
        Returns a list of all roles assigned to the LLMs in this group.

        :return: A list of model roles, with None standing for the LLM without a role
        :rtype: list[str | None]
        """
        return [i.role for i in self.models]

    @property
    def _llms(self) -> list[_DocumentLLM]:
        """
        Returns the LLMs of the group, i.e. its models other than those of
        the kinds that generate no text, such as decision models. The params
        that only LLMs have, e.g. the output language and the tools, apply
        to these models of the group only.

        :return: The models of the group that are LLMs, in the order of the group.
        :rtype: list[_DocumentLLM]
        """
        # Imported here, as the module that defines LLMs imports this module
        from contextgem.internal.base.llms import _DocumentLLM

        return [model for model in self.models if isinstance(model, _DocumentLLM)]

    def group_update_output_language(
        self, output_language: LanguageRequirement
    ) -> None:
        """
        Updates the output language for all LLMs in the group.

        :param output_language: The new output language to set for all LLMs
        :type output_language: LanguageRequirement
        :return: None
        :rtype: None
        """
        for llm in self._llms:
            llm.output_language = output_language
        self.output_language = output_language
        logger.info(
            f"Updated output language for all LLMs in the group to `{output_language}`"
        )

    def _eq_deserialized_llm_config(
        self,
        other: _DocumentModelGroup,
    ) -> bool:
        """
        Custom config equality method to compare this _DocumentModelGroup with a deserialized instance.

        Uses the `_eq_deserialized_model_config` method of each model in the group, of any
        kind, to compare it with its counterpart, including fallbacks, if any.

        :param other: Another _DocumentModelGroup instance to compare with
        :type other: _DocumentModelGroup
        :return: True if the instances are equal, False otherwise
        :rtype: bool
        """
        for self_llm, other_llm in zip(self.models, other.models, strict=True):
            if not self_llm._eq_deserialized_model_config(other_llm):
                return False
        return True

    def _warn_tools_ignored_if_enabled(self) -> None:
        """
        Warns if any LLM in the group has tools configured, since tools are ignored
        during extraction workflows. Tools are only supported in ``llm.chat(...)``.

        :return: None
        :rtype: None
        """
        models_with_tools = [llm.model for llm in self._llms if llm.tools]
        if models_with_tools:
            models_with_tools_list = ", ".join(models_with_tools)
            warnings.warn(
                "Tool calling is ignored during extraction workflows. "
                f"Models with tools configured: {models_with_tools_list}. "
                "Tools are only supported in `llm.chat(...)`.",
                stacklevel=2,
            )

    @field_validator("models")
    @classmethod
    def _validate_models(cls, models: list[_DocumentModel]) -> list[_DocumentModel]:
        """
        Validates the provided list of _DocumentLLMs ensuring that each
        LLM has a unique role within the group, so that each aspect/concept
        is routed to a single LLM. No role (None) also counts as a role,
        i.e. only one LLM in the group can have no role.

        :param models: A list of `_DocumentModel` instances to be validated.
        :type models: list[_DocumentModel]
        :return: A validated list of `_DocumentModel` instances.
        :rtype: list[_DocumentModel]
        :raises ValueError: If not all model roles are unique.
        """
        seen_roles = set()
        for llm in models:
            if llm.role in seen_roles:
                raise ValueError(
                    _get_non_unique_model_roles_msg(_get_common_kind_name(models))
                )
            seen_roles.add(llm.role)
        return models

    def get_usage(
        self, model_role: str | None = None
    ) -> list[_ModelUsageOutputContainer]:
        """
        Retrieves the usage information of the LLMs in the group, filtered by the specified
        model role if provided.

        :param model_role: Optional; A string representing the role of the LLM to filter
            the usage data, validated in the same way as the roles assigned to LLMs.
            If None, returns usage for all LLMs in the group, including the LLM
            without a role.
        :type model_role: str | None
        :return: A list of usage statistics containers for the specified LLMs and their fallbacks.
        :rtype: list[_ModelUsageOutputContainer]
        :raises ValueError: If the specified role is invalid, or if no LLM with
            the specified role exists in the group.
        """
        # Safe cast: _get_usage_or_cost with retrieval_type="usage"
        # returns only list of _ModelUsageOutputContainer
        return cast(
            list["_ModelUsageOutputContainer"],
            self._get_usage_or_cost(retrieval_type="usage", model_role=model_role),
        )

    def get_cost(
        self, model_role: str | None = None
    ) -> list[_ModelCostOutputContainer]:
        """
        Retrieves the accumulated cost information of the LLMs in the group, filtered by the specified
        model role if provided.

        :param model_role: Optional; A string representing the role of the LLM to filter
            the cost data, validated in the same way as the roles assigned to LLMs.
            If None, returns cost for all LLMs in the group, including the LLM
            without a role.
        :type model_role: str | None
        :return: A list of cost statistics containers for the specified LLMs and their fallbacks.
        :rtype: list[_ModelCostOutputContainer]
        :raises ValueError: If the specified role is invalid, or if no LLM with
            the specified role exists in the group.
        """
        # Safe cast: _get_usage_or_cost with retrieval_type="cost"
        # returns only list of _ModelCostOutputContainer
        return cast(
            list["_ModelCostOutputContainer"],
            self._get_usage_or_cost(retrieval_type="cost", model_role=model_role),
        )

    def reset_usage_and_cost(self, model_role: str | None = None) -> None:
        """
        Resets the usage and cost statistics for LLMs in the group.

        This method clears accumulated usage and cost data, which is useful when processing
        multiple documents sequentially and tracking metrics for each document separately.

        :param model_role: Optional; A string representing the role of the LLM to reset statistics for,
            validated in the same way as the roles assigned to LLMs. If None, resets statistics
            for all LLMs in the group, including the LLM without a role.
        :type model_role: str | None
        :return: None
        :rtype: None
        :raises ValueError: If the specified role is invalid, or if no LLM with
            the specified role exists in the group.
        """

        if model_role is not None:
            # Validate the role in the same way as the roles assigned to LLMs,
            # e.g. remove the surrounding whitespace
            model_role = _validate_model_role(model_role)
            try:
                llm = next(i for i in self.models if i.role == model_role)
                llm.reset_usage_and_cost()
            except StopIteration:
                raise ValueError(
                    f"No {_get_common_kind_name(self.models)} with the given role "
                    f"`{model_role}` was found in group."
                ) from None
        else:
            for llm in self.models:
                llm.reset_usage_and_cost()

    @model_validator(mode="after")
    def _validate_document_model_group_post(self) -> Self:
        """
        Validates the model group to ensure consistency of the `output_language`
        attribute across all LLMs within the group. The models of the group that
        generate no text, such as decision models, have no output language.

        :return: The model group instance after successful validation.
        :rtype: Self
        :raises ValueError: Raised if any LLM's `output_language` differs from the
            group's `output_language`.
        """

        if any(i.output_language != self.output_language for i in self._llms):
            raise ValueError(
                "All LLMs in the group must have the same value of "
                "`output_language` attribute as the group."
            )
        return self
