#
# ContextGem
#
# Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
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
Module defining the internal implementation of decision models.

This module provides the ``_DocumentDecisionModel`` class, which holds the configuration
of a specific decision model and handles the requests to it (questions on a text,
retries, usage and cost tracking), together with the conversion between the questions
and answers of the framework and those of the Decisions API of litellm, through which
the requests are sent. What a decision model shares with the other kinds of models
(``_DocumentModel``), and the extraction logic that it shares with LLMs and model
groups, are defined in the ``models`` module of this package. The questions built for
concepts, and the conversion of the answers to the values of extracted items, are
defined in the ``decisions`` module of the ``internal`` package.
"""

from __future__ import annotations

import asyncio
import json
import warnings
from collections.abc import Sequence
from typing import Any, cast

from aiolimiter import AsyncLimiter

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.concepts import (
    _BooleanConcept,
    _Concept,
    _LabelConcept,
    _RatingConcept,
)
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.base.models import _DocumentModel
from contextgem.internal.data_models import _ModelCall, _ModelUsage
from contextgem.internal.decisions import (
    _build_concept_questions,
    _convert_concept_answers,
    _DecisionAnswer,
    _DecisionQuestion,
)
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.exceptions import ModelAPIError, _InvalidDecisionResponseError
from contextgem.internal.loggers import logger
from contextgem.internal.suppressions import _suppress_litellm_warnings_context
from contextgem.internal.typings.types import ExtractedInstanceType, ItemProbabilities
from contextgem.internal.utils import (
    _MODEL_API_RETRIES_ATTEMPTED_ATTR,
    _chunk_list,
    _get_cache_token_counts,
    _get_model_api_retry_delay,
    _is_retryable_model_api_error,
)


with _suppress_litellm_warnings_context():
    import litellm
    from litellm.types import decisions as _litellm_decision_types


# Concept types that a decision model can extract: the types whose values come
# from a closed set of options
_DECISION_CONCEPT_TYPES = (_BooleanConcept, _LabelConcept, _RatingConcept)

# Model providers of self-hosted decision models, which have no prices
_SELF_HOSTED_DECISION_MODEL_PROVIDERS = [
    "strands_decider/",
]

# Model providers whose decision models cannot be priced automatically: genai-prices
# has only the chat prices of their models, which differ from the prices of
# the decision requests to the same models
_NO_AUTO_PRICING_DECISION_MODEL_PROVIDERS = [
    "openai/",
]


def _get_request_question(question: _DecisionQuestion) -> dict[str, Any]:
    """
    Converts a question to a question of a request to the Decisions API of litellm,
    in the OpenAI format, which litellm accepts for all providers of decision models.

    An option of a "choice" question is sent with its label as the value to pick,
    and a level of a "score" question with its label as the label of the level.
    The description of an option is sent only if the option has it.

    :param question: The question to convert.
    :type question: _DecisionQuestion
    :return: The question as a dictionary of the request.
    :rtype: dict[str, Any]
    """
    request_question: dict[str, Any] = {
        "type": question.kind,
        "name": question.name,
        "instructions": question.instructions,
    }
    if question.kind == "predicate":
        return request_question

    # The key that holds the options, and the key that holds the label of an option
    options_key, label_key = (
        ("choices", "value") if question.kind == "choice" else ("levels", "label")
    )
    request_options: list[dict[str, str]] = []
    for option in question.options:
        request_option = {label_key: option.label}
        if option.description is not None:
            request_option["description"] = option.description
        request_options.append(request_option)
    request_question[options_key] = request_options
    return request_question


def _get_answers_by_question_names(
    questions: Sequence[_DecisionQuestion], response_answers: Any
) -> dict[str, _DecisionAnswer]:
    """
    Converts the answers of a response of the Decisions API of litellm, in the OpenAI
    format, to the answers to the questions of the request.

    The answers of a response are matched with the questions by position: litellm
    returns one answer for each question, in the order of the questions. A refusal,
    which litellm also returns for a question that the model left unanswered, becomes
    a refused answer of the kind of the question. The probabilities of the options
    are keyed by the labels of the options as returned, i.e. by the value of an option
    of a "choice" answer and by the label of a level of a "score" answer (not by
    the position of the level, which is its value in the response).

    Only what cannot be attributed to a question is rejected here. Whether an answer
    is of the kind of its question, and has the probabilities of the options offered,
    is checked when the answer is converted to the value of an extracted item.

    :param questions: The questions of the request, in the order they were sent.
    :type questions: Sequence[_DecisionQuestion]
    :param response_answers: The answers of the response, as returned by litellm.
    :type response_answers: Any
    :return: The answers by the names of the questions.
    :rtype: dict[str, _DecisionAnswer]
    :raises _InvalidDecisionResponseError: If the number of the answers differs from
        the number of the questions, or if an answer has the name of another question,
        is of an unknown type, or repeats an option.
    """
    if not isinstance(response_answers, list | tuple):
        raise _InvalidDecisionResponseError(
            questions[0].name, "the response has no sequence of answers."
        )
    if len(response_answers) != len(questions):
        # Reported for the first question without an answer, or for the last
        # question if there are more answers than questions
        position = min(len(response_answers), len(questions) - 1)
        raise _InvalidDecisionResponseError(
            questions[position].name,
            f"the response has {len(response_answers)} answers, while "
            f"{len(questions)} questions were asked.",
        )

    answers: dict[str, _DecisionAnswer] = {}
    for question, response_answer in zip(questions, response_answers, strict=True):
        answer_name = getattr(response_answer, "name", None)
        if answer_name is not None and answer_name != question.name:
            raise _InvalidDecisionResponseError(
                question.name,
                f"the answer at the position of the question has the name "
                f"`{answer_name}`.",
            )

        answer_type = getattr(response_answer, "type", None)
        if answer_type == "refusal":
            answer = _DecisionAnswer(
                name=question.name, kind=question.kind, refused=True
            )
        elif answer_type == "predicate":
            answer = _DecisionAnswer(
                name=question.name,
                kind="predicate",
                probability=getattr(response_answer, "probability", None),
            )
        elif answer_type == "choice":
            answer = _DecisionAnswer(
                name=question.name,
                kind="choice",
                option_probabilities=_get_option_probabilities_by_labels(
                    question, response_answer, label_attr_name="value"
                ),
            )
        elif answer_type == "score":
            answer = _DecisionAnswer(
                name=question.name,
                kind="score",
                option_probabilities=_get_option_probabilities_by_labels(
                    question, response_answer, label_attr_name="label"
                ),
            )
        else:
            raise _InvalidDecisionResponseError(
                question.name, f"the answer is of an unknown type `{answer_type}`."
            )
        answers[question.name] = answer
    return answers


def _get_option_probabilities_by_labels(
    question: _DecisionQuestion, response_answer: Any, label_attr_name: str
) -> dict[Any, Any]:
    """
    Returns the probabilities of the options of a "choice" or a "score" answer
    of a response of the Decisions API of litellm, by the labels of the options.

    The labels and the probabilities are taken as returned: they are validated
    against the options of the question when the answer is converted.

    :param question: The question that is answered.
    :type question: _DecisionQuestion
    :param response_answer: The answer of the response, as returned by litellm.
    :type response_answer: Any
    :param label_attr_name: The name of the attribute that holds the label of
        an option in the probabilities of the answer.
    :type label_attr_name: str
    :return: The probabilities of the options by their labels, in the order returned.
        Empty if the answer has no probabilities.
    :rtype: dict[Any, Any]
    :raises _InvalidDecisionResponseError: If the answer repeats an option.
    """
    option_probabilities: dict[Any, Any] = {}
    for option_probability in getattr(response_answer, "probabilities", None) or ():
        label = getattr(option_probability, label_attr_name, None)
        if label in option_probabilities:
            raise _InvalidDecisionResponseError(
                question.name, f"the answer repeats the option `{label}`."
            )
        option_probabilities[label] = getattr(option_probability, "probability", None)
    return option_probabilities


@_disable_direct_initialization
class _DocumentDecisionModel(_DocumentModel):
    """
    Internal implementation of the ``DocumentDecisionModel`` class.
    """

    @property
    def _kind_name(self) -> str:
        """
        Returns the name of the kind of the model, as used in the messages of
        the logic that models of all kinds share.

        :return: The name of the kind of the model.
        :rtype: str
        """
        return "decision model"

    def _get_auto_pricing_error_msg(self) -> str | None:
        """
        Returns the error message for the automatic pricing of the decision model,
        if its costs cannot be calculated automatically.

        :return: The error message, or None if the automatic pricing is not ruled
            out for the model.
        :rtype: str | None
        """
        if any(
            self.model.startswith(provider)
            for provider in _SELF_HOSTED_DECISION_MODEL_PROVIDERS
        ):
            return (
                "`auto_pricing=True` is not supported for self-hosted decision models. "
                "Disable `auto_pricing` or provide explicit `pricing_details`."
            )
        if any(
            self.model.startswith(provider)
            for provider in _NO_AUTO_PRICING_DECISION_MODEL_PROVIDERS
        ):
            return (
                f"`auto_pricing=True` is not supported for decision model "
                f"`{self.model}`: the pricing data has only the prices of the chat "
                f"requests to the models of this provider, which differ from "
                f"the prices of the decision requests and would be applied instead "
                f"of them. Disable `auto_pricing` and provide explicit "
                f"`pricing_details` to get the costs calculated."
            )
        return None

    def _set_provider_and_model_for_auto_pricing(self) -> None:
        """
        Sets (provider_id, model_ref) pair for genai-prices from `model` field.
        Does nothing for the decision models that cannot be priced automatically,
        for which enabling the automatic pricing is rejected when the model
        is validated.

        :return: None
        :rtype: None
        """
        if self._get_auto_pricing_error_msg() is not None:
            return
        super()._set_provider_and_model_for_auto_pricing()

    def _finish_kind_specific_validation(self) -> None:
        """
        Validates the params that only a decision model restricts, after the params
        that models of all kinds have are validated: the automatic pricing cannot be
        enabled for a decision model whose costs cannot be calculated automatically.

        :return: None
        :rtype: None
        :raises ValueError: If the automatic pricing is enabled for a self-hosted
            decision model, or for a decision model of a provider whose decision
            requests have no prices in the pricing data.
        """
        if self.auto_pricing:
            error_msg = self._get_auto_pricing_error_msg()
            if error_msg is not None:
                raise ValueError(error_msg)

    def _warn_tools_ignored_if_enabled(self) -> None:
        """
        Does nothing: a decision model has no tools, i.e. there are no tools to warn
        about when it is used in an extraction workflow.

        :return: None
        :rtype: None
        """
        return None

    @staticmethod
    def _get_unservable_reasons(instance: _Aspect | _Concept) -> list[str]:
        """
        Returns the reasons why a decision model cannot extract an aspect or a concept.

        A decision model answers closed questions on a text and generates no text.
        It extracts the concepts whose values come from a closed set of options
        (Boolean, label and rating concepts), from text only, with no justifications
        and no references.

        :param instance: The aspect or concept to be extracted.
        :type instance: _Aspect | _Concept
        :return: The reasons, or an empty list if a decision model can extract
            the aspect or concept.
        :rtype: list[str]
        """
        if isinstance(instance, _Aspect):
            return ["aspects are not supported"]

        reasons: list[str] = []
        if not isinstance(instance, _DECISION_CONCEPT_TYPES):
            reasons.append(f"type `{type(instance).__name__}` is not supported")
        if instance.modality != "text":
            reasons.append(f"modality `{instance.modality}` is not supported")
        if instance.add_justifications:
            reasons.append("justifications are not supported")
        if instance.add_references:
            reasons.append("references are not supported")
        return reasons

    def _check_instances_servable(
        self, instances: Sequence[_Aspect | _Concept]
    ) -> None:
        """
        Checks, before any requests are sent, that the decision model can extract
        the given aspects and concepts, which are to be extracted by it.

        :param instances: The aspects and concepts to be extracted by the model.
        :type instances: Sequence[_Aspect | _Concept]
        :return: None
        :rtype: None
        :raises ValueError: If the decision model cannot extract some of the aspects
            or concepts. The error names each of them with the reasons.
        """
        unservable_instances: list[str] = []
        for instance in instances:
            reasons = self._get_unservable_reasons(instance)
            if reasons:
                instance_type = "aspect" if isinstance(instance, _Aspect) else "concept"
                unservable_instances.append(
                    f"{instance_type} `{instance.name}` ({', '.join(reasons)})"
                )
        if not unservable_instances:
            return

        if self.is_fallback:
            subject = "Fallback decision model"
            condition = " what it is to extract if the model it is a fallback for fails"
        else:
            subject = "Decision model"
            condition = ""
        raise ValueError(
            f"{subject} `{self.model}` cannot extract{condition}: "
            f"{'; '.join(unservable_instances)}. "
            f"A decision model extracts Boolean, label and rating concepts from text, "
            f"without justifications and without references. Route the other aspects "
            f"and concepts to an LLM, e.g. in a model group."
        )

    def _get_source_text(self, source: _Document | _Aspect, document: _Document) -> str:
        """
        Returns the text of the source that the questions on concepts are asked on:
        the text of the document for document-level concepts, or the text of
        the paragraphs extracted for the aspect for aspect-level concepts. The text
        is the one that an LLM is sent for the same concepts when the source is
        processed as a whole.

        :param source: The source that the concepts are extracted from.
        :type source: _Document | _Aspect
        :param document: The document containing the source.
        :type document: _Document
        :return: The text of the source.
        :rtype: str
        :raises ValueError: If the source has no text to extract the concepts from.
        """
        if isinstance(source, _Document):
            if not (source.raw_text or source.paragraphs):
                raise ValueError(
                    "Document lacks text or paragraphs for concept extraction."
                )
            paragraphs = document.paragraphs
        else:
            paragraphs = source.reference_paragraphs
        if not paragraphs:
            raise ValueError("Context lacks paragraphs for text-based extraction.")
        return self._get_text_for_concept_extraction(
            source=source, paragraphs_chunk=paragraphs
        )

    async def _extract_items_from_decisions(
        self,
        extracted_item_type: ExtractedInstanceType,
        source: _Document | _Aspect,
        instances_to_process: list[_Aspect] | list[_Concept],
        document: _Document,
        overwrite_existing: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
        num_retries_failed_request: int = 3,
        async_limiter: AsyncLimiter | None = None,
        raise_exception_on_extraction_error: bool = True,
        has_fallback_pending: bool = False,
    ) -> tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]:
        """
        Extracts the items of concepts with the decision model: asks the model
        the questions of the concepts on the text of the source, and converts
        its answers to the items, each with the probabilities of its options.

        Called by ``_extract_items_from_instances``, the entry point of the extraction
        with a model of any kind, after the source is checked and the cases with
        nothing to extract are handled there. The result is returned under the same
        contract as the result of the extraction with an LLM:

        - if a request fails for good, the instances are None and the last call
          recorded in the usage is unfinished (no response received), unless
          an exception is raised;
        - if the response cannot be used, e.g. the model refused to answer
          a question, the instances are None and the call is recorded as finished,
          so that it is told apart from a failed request.

        All the questions of the concepts are sent in one request, or in several
        requests on the same text if there are more questions than a request can
        hold. The text is always sent as a whole, i.e. it has to fit in a request.

        :param extracted_item_type: Type of the item(s) being extracted. Only
            concepts are extracted by a decision model.
        :type extracted_item_type: ExtractedInstanceType
        :param source: The source to extract from (_Document or _Aspect)
        :type source: _Document | _Aspect
        :param instances_to_process: List of concepts to process
        :type instances_to_process: list[_Aspect] | list[_Concept]
        :param document: The document containing the source.
        :type document: _Document
        :param overwrite_existing: Whether to overwrite the items of the already
            processed concepts. Defaults to False.
        :type overwrite_existing: bool
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs
            to analyze in a single LLM call. Not applied to a decision model, which
            is sent the text of the source as a whole. Defaults to 0.
        :type max_paragraphs_to_analyze_per_call: int
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :param raise_exception_on_extraction_error: Whether to raise an exception
            if a request fails due to an error in the API of the model. If False,
            a warning is issued instead, and no items are extracted. Defaults to True.
        :type raise_exception_on_extraction_error: bool
        :param has_fallback_pending: Whether a fallback model will be tried if
            the call to this model fails. If True, a model API error is logged
            without being raised or warned about, as it is not final.
            Defaults to False.
        :type has_fallback_pending: bool
        :return: A tuple containing:
            (0) List of processed concepts with extracted items, or None if
                the request failed or the response cannot be used
            (1) _ModelUsage instance with the usage of the requests made, which
                the usage and cost of the model are yet to be updated with
        :rtype: tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]
        :raises ValueError: If the decision model cannot extract some of
            the instances, or if the source has no text to extract them from.
        :raises ModelAPIError: If a request fails due to an error in the API of
            the model, no fallback model is yet to be tried, and exceptions are
            to be raised.
        """

        # No request is sent for what a decision model cannot extract, also if
        # the checks made when an extraction starts are bypassed
        self._check_instances_servable(instances_to_process)
        # Safe cast: only concepts pass the check above
        concepts = cast(list[_Concept], instances_to_process)

        if max_paragraphs_to_analyze_per_call:
            logger.info(
                f"`max_paragraphs_to_analyze_per_call` is not applied to decision "
                f"model `{self.model}`: the text is sent to it as a whole."
            )

        text = self._get_source_text(source=source, document=document)
        questions = _build_concept_questions(concepts)

        # Ask the questions. Each request is recorded in the usage: with the tokens
        # used and the answers if it succeeds, or as an unfinished call if it fails.
        usage = _ModelUsage()
        try:
            answers = await self._get_answers(
                text=text,
                questions=questions,
                usage=usage,
                num_retries_failed_request=num_retries_failed_request,
                async_limiter=async_limiter,
            )
        except _InvalidDecisionResponseError as e:
            # The request succeeded, i.e. its call is recorded as finished
            logger.error(
                f"Decision model `{self.model}` returned an invalid response "
                f"for {extracted_item_type}s: {e}"
            )
            return None, usage
        except asyncio.CancelledError:
            # The call is cancelled, e.g. on a timeout set by the calling code
            await self._update_usage_and_cost((None, usage))
            raise
        except Exception as e:
            # e.g. rate limit error, after the retries of the request are exhausted,
            # or a non-transient error such as failed authentication. The failed
            # request is the last call recorded in the usage (call unfinished).
            logger.error(f"Exception occurred while calling decision model API: {e}")
            if has_fallback_pending:
                logger.info("The call will be retried with a fallback model.")
            else:
                n_retries = getattr(e, _MODEL_API_RETRIES_ATTEMPTED_ATTR, 0)
                if raise_exception_on_extraction_error:
                    await self._update_usage_and_cost((None, usage))
                    raise ModelAPIError(
                        "Exception occurred while calling decision model API",
                        retry_count=n_retries,
                        original_error=e,
                    ) from e
                else:
                    warning_msg = (
                        f"Exception occurred while calling decision model API: {e}"
                        + f" ({n_retries} retries)."
                    )
                    logger.warning(warning_msg)
                    warnings.warn(
                        warning_msg,
                        stacklevel=2,
                    )
            return None, usage

        # Convert the answers to the items of each concept. A decision model returns
        # at most one item per concept and source, whatever `singular_occurrence`
        # of the concept is set to.
        concepts_and_items: list[tuple[_Concept, list[_ExtractedItem]]] = []
        try:
            for position, concept in enumerate(concepts):
                extracted_items: list[_ExtractedItem] = []
                value_and_probabilities = _convert_concept_answers(
                    concept, position, answers
                )
                # None if there is no item to extract, i.e. no label applies
                if value_and_probabilities is not None:
                    value = value_and_probabilities[0]
                    probabilities: ItemProbabilities = value_and_probabilities[1]
                    extracted_items.append(
                        concept._item_class(value=value, probabilities=probabilities)
                    )
                concepts_and_items.append((concept, extracted_items))
        except ValueError as e:
            # An answer that cannot be used (missing, refused, of another kind than
            # its question, without the probabilities of the options offered),
            # or a value that the item of the concept does not accept
            logger.error(
                f"Decision model `{self.model}` returned an invalid response "
                f"for {extracted_item_type}s: {e}"
            )
            return None, usage

        # Process all gathered results for all processed concepts
        async with (
            self._async_lock
        ):  # ensure atomicity of the concepts' state check and modification
            for concept, extracted_items in concepts_and_items:
                self._check_instances_already_processed(
                    instance_type=extracted_item_type,
                    instances=[concept],
                    overwrite_existing=overwrite_existing,
                )
                concept.extracted_items = extracted_items
                concept._is_processed = True

        return instances_to_process, usage

    async def _get_answers(
        self,
        *,
        text: str,
        questions: list[_DecisionQuestion],
        usage: _ModelUsage,
        num_retries_failed_request: int = 3,
        async_limiter: AsyncLimiter | None = None,
    ) -> dict[str, _DecisionAnswer]:
        """
        Asks the decision model the questions on the text and returns its answers.

        The questions are sent in one request. If there are more questions than
        litellm accepts in a request, they are split into several requests on
        the same text, sent one after another, and the answers are merged.

        Each request is recorded in the usage object: with the tokens used and
        the answers if it succeeds, or as an unfinished call if it fails, in which
        case no further requests are sent.

        :param text: The text to ask the questions on.
        :type text: str
        :param questions: The questions to ask, with the names unique among them.
        :type questions: list[_DecisionQuestion]
        :param usage: The usage object to record the requests in.
        :type usage: _ModelUsage
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :return: The answers by the names of the questions.
        :rtype: dict[str, _DecisionAnswer]
        :raises _InvalidDecisionResponseError: If the answers of a response cannot be
            attributed to the questions of its request. The call of such a request
            is recorded as finished.
        """
        answers: dict[str, _DecisionAnswer] = {}
        # The cap is the one of litellm, which rejects a request over it
        max_questions_per_request = _litellm_decision_types.MAX_DECISION_QUESTIONS
        questions_chunks: list[list[_DecisionQuestion]] = _chunk_list(
            questions, max_questions_per_request
        )
        if len(questions_chunks) > 1:
            logger.debug(
                f"Sending {len(questions)} questions to decision model "
                f"`{self.model}` in {len(questions_chunks)} requests "
                f"(max {max_questions_per_request} questions per request)."
            )

        for questions_chunk in questions_chunks:
            request_questions = [
                _get_request_question(question) for question in questions_chunk
            ]
            # The request is recorded as sent, without the API credentials
            call_obj = _ModelCall(
                prompt_kwargs={},
                prompt=json.dumps(
                    {
                        "model": self.model,
                        "input": text,
                        "questions": request_questions,
                    },
                    ensure_ascii=False,
                ),
            )
            try:
                if async_limiter:
                    async with async_limiter:
                        response = await self._send_decisions_request(
                            text=text,
                            request_questions=request_questions,
                            num_retries_failed_request=num_retries_failed_request,
                        )
                else:
                    response = await self._send_decisions_request(
                        text=text,
                        request_questions=request_questions,
                        num_retries_failed_request=num_retries_failed_request,
                    )
                self._record_usage_call(
                    usage=usage, call_obj=call_obj, response=response
                )
            except Exception:
                # Record the call details (call unfinished)
                usage.calls.append(call_obj)
                raise
            answers.update(
                _get_answers_by_question_names(
                    questions_chunk, getattr(response, "answers", None)
                )
            )
        return answers

    async def _send_decisions_request(
        self,
        *,
        text: str,
        request_questions: list[dict[str, Any]],
        num_retries_failed_request: int = 3,
    ) -> Any:
        """
        Sends a single request to the Decisions API of litellm, retrying it if
        it fails with a transient error.

        The request is sent in the OpenAI format (the text as ``input``, and a list
        of questions), which litellm accepts for all providers of decision models,
        returning the response in the same format.

        Only transient errors (request timeouts, rate limits, server and connection
        errors) are retried, with capped exponential backoff and jitter between
        the retries. Non-transient errors (e.g. an invalid request or a failed
        authentication) are raised right away. The number of retries attempted is
        recorded on the raised exception.

        :param text: The text to ask the questions on.
        :type text: str
        :param request_questions: The questions, as dictionaries of the request.
        :type request_questions: list[dict[str, Any]]
        :param num_retries_failed_request: Number of retries when the request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :return: Raw response object
        :rtype: Any
        """

        # The request is retried here: litellm does not retry the requests
        # to its Decisions API
        retries_attempted = 0
        while True:
            try:
                return await litellm.adecisions(
                    model=self.model,
                    input=text,
                    questions=request_questions,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    timeout=self.timeout,
                )
            except Exception as e:
                if (
                    retries_attempted >= num_retries_failed_request
                    or not _is_retryable_model_api_error(e)
                ):
                    setattr(e, _MODEL_API_RETRIES_ATTEMPTED_ATTR, retries_attempted)
                    raise
                retries_attempted += 1
                delay = _get_model_api_retry_delay(retries_attempted)
                logger.info(
                    f"Decision model API request failed with a transient error "
                    f"({type(e).__name__}). Retrying in {delay:.1f}s "
                    f"({retries_attempted}/{num_retries_failed_request})"
                )
                await asyncio.sleep(delay)

    def _record_usage_call(
        self,
        *,
        usage: _ModelUsage,
        call_obj: _ModelCall,
        response: Any,
    ) -> None:
        """
        Accumulates token usage and records metadata for a call into the given
        usage object: the input and output tokens, the input tokens read from
        and written to the prompt cache of the model provider, which are a part
        of the input tokens, and the answers of the model as JSON.

        :param usage: The _ModelUsage object to accumulate token usage and metadata.
        :type usage: _ModelUsage
        :param call_obj: The _ModelCall object to record the response timestamp
            and the response.
        :type call_obj: _ModelCall
        :param response: The raw response object from the decision model.
        :type response: Any
        """
        # Everything is read from the response before the call is recorded
        # as finished, so that a response that cannot be read leaves it unfinished
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cache_read_tokens, cache_write_tokens = _get_cache_token_counts(
            input_tokens, getattr(response.usage, "input_tokens_details", None)
        )
        answers_json = json.dumps(
            [answer.model_dump(mode="json") for answer in response.answers],
            ensure_ascii=False,
        )

        call_obj._input_tokens = input_tokens
        call_obj._output_tokens = output_tokens
        call_obj._cache_read_tokens = cache_read_tokens
        call_obj._cache_write_tokens = cache_write_tokens
        usage.input += call_obj._input_tokens
        usage.output += call_obj._output_tokens
        usage.cache_read += call_obj._cache_read_tokens
        usage.cache_write += call_obj._cache_write_tokens
        call_obj._record_response_timestamp()
        call_obj.response = answers_json
        usage.calls.append(call_obj)
