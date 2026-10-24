# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the internal implementation of decision models.

This module provides the ``_DocumentDecisionModel`` class, which holds the configuration
of a specific decision model and handles the requests to it (questions on a text,
retries, usage and cost tracking), together with the conversion between the questions
and answers of the framework and those of the Decisions API of litellm, through which
the requests are sent. What a decision model shares with the other kinds of models
(``_DocumentModel``), and the extraction logic that it shares with LLMs and model
groups, are defined in the ``models`` module of this package. The questions built for
aspects and concepts, and the conversion of the answers to the values of extracted
items, are defined in the ``decisions`` module of the ``internal`` package.
"""

from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, cast

from aiolimiter import AsyncLimiter
from genai_prices import Usage as _GPUsage
from genai_prices import calc_price as _calc_auto_price
from pydantic import Field, StrictFloat, StrictInt, ValidationError

from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.concepts import (
    _BooleanConcept,
    _Concept,
    _LabelConcept,
    _RatingConcept,
)
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.base.models import _LOCAL_MODEL_PROVIDERS, _DocumentModel
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.data_models import _ModelCall, _ModelUsage
from contextgem.internal.decisions import (
    _DEFAULT_DECISION_THRESHOLD,
    _build_aspect_questions,
    _build_concept_questions,
    _convert_aspect_answer,
    _convert_concept_answers,
    _DecisionAnswer,
    _DecisionQuestion,
    _get_aspect_question_name,
    _get_concept_question_name,
    _get_decision_threshold,
    _get_paragraph_text_for_aspects,
    _get_sentence_text_for_aspects,
    _is_refused_answer,
    _is_whole_paragraph_sentence,
    _validate_answers,
)
from contextgem.internal.decorators import _disable_direct_initialization
from contextgem.internal.exceptions import (
    ContextGemError,
    ModelAPIError,
    _InvalidDecisionResponseError,
    _RefusedDecisionAnswerError,
)
from contextgem.internal.items import _StringItem
from contextgem.internal.loggers import logger
from contextgem.internal.registry import _publicize
from contextgem.internal.suppressions import _suppress_litellm_warnings_context
from contextgem.internal.typings.types import ExtractedInstanceType, ItemProbabilities
from contextgem.internal.utils import (
    _MODEL_API_RETRIES_ATTEMPTED_ATTR,
    _chunk_list,
    _describe_error_without_values,
    _get_cache_token_counts,
    _get_model_api_retry_delay,
    _get_model_call_result_refused_call,
    _get_model_refusal_msg,
    _group_instances_by_fields,
    _is_retryable_model_api_error,
)


with _suppress_litellm_warnings_context():
    import litellm
    from litellm.types import decisions as _litellm_decision_types


# Concept types that a decision model can extract: the types whose values come
# from a closed set of options
_DECISION_CONCEPT_TYPES = (_BooleanConcept, _LabelConcept, _RatingConcept)

# Model providers of self-hosted decision models, which have no prices. A local
# server, e.g. Ollama or LM Studio, is reached under the first of them, with
# the ``api_base`` of the server.
_SELF_HOSTED_DECISION_MODEL_PROVIDERS = [
    "strands_decider/",
]

# Providers of decision models whose decision requests genai-prices prices under
# a provider of their own, by the IDs of these providers in genai-prices. Under
# the provider in the identifier of the model, genai-prices has the prices of
# the chat requests to the model of the same name, and it looks up the prices of
# the decision requests only under the ID that is given for them
_DECISION_PRICING_PROVIDER_IDS = {
    "openai": "openai-decisions",
}

# Module of litellm that prepares the requests to its Decisions API, and the name
# of the function in it that returns the providers of decision models that
# litellm supports
_LITELLM_DECISIONS_MODULE_NAME = "litellm.decisions.call"
_LITELLM_DECISION_PROVIDERS_FUNC_NAME = "supported_decisions_providers"

# Names of the types that litellm validates the questions of a request with,
# in the OpenAI format, by the kinds of the questions
_LITELLM_QUESTION_TYPE_NAMES = {
    "predicate": "OpenAIPredicateQuestion",
    "choice": "OpenAIChoiceQuestion",
    "score": "OpenAIScoreQuestion",
}

# Extraction params of aspects and concepts that a decision model is not sent
# and that do not decide whether it can extract an aspect or a concept, i.e.
# the params that do not split the aspects or concepts into separate calls of
# a decision model: those of all instances, and those of concepts only
_NOT_APPLIED_INSTANCE_PARAMS = ("justification_depth", "justification_max_sents")
_NOT_APPLIED_CONCEPT_PARAMS = ("reference_depth",)

# Answers of a decision model on the sentences of one paragraph: each sentence
# that is asked about, in the order of the sentences, with the answers on it
# by the names of the questions
_AnswersOnSentences = list[tuple[_Sentence, dict[str, _DecisionAnswer]]]


def _get_supported_decision_providers() -> tuple[str, ...] | None:
    """
    Returns the providers of decision models that litellm supports, i.e.
    the provider prefixes of the model identifiers that its Decisions API accepts,
    as litellm lists them itself.

    The list is returned by a function of the module of litellm that prepares
    the requests to its Decisions API. The module is imported by its full name:
    ``litellm.decisions`` is the function that sends a request, which shadows
    the package of the same name, i.e. the module cannot be reached as an attribute
    of ``litellm``.

    :return: The providers, or None if they cannot be read from the installed
        version of litellm, in which case the provider of a model cannot be checked
        before a request is sent to it.
    :rtype: tuple[str, ...] | None
    """
    try:
        decisions_module = importlib.import_module(_LITELLM_DECISIONS_MODULE_NAME)
        providers = tuple(
            getattr(decisions_module, _LITELLM_DECISION_PROVIDERS_FUNC_NAME)()
        )
    except Exception:  # noqa: BLE001 - litellm's internals may change
        return None
    if not providers or not all(
        isinstance(provider, str) and provider for provider in providers
    ):
        return None
    return providers


def _get_question_rejection_reasons(question: _DecisionQuestion) -> list[str]:
    """
    Returns the reasons why litellm would reject a question of a request to its
    Decisions API, before sending the request, e.g. for more options than
    a question may have.

    The question is validated with the type that litellm validates it with when
    a request is made, so that the limits of litellm are applied as they are in
    the installed version, without being repeated in the framework.

    :param question: The question to check.
    :type question: _DecisionQuestion
    :return: The reasons, as worded by litellm, each with the part of the question
        that it refers to. Empty if litellm accepts the question, or if the question
        cannot be checked with the installed version of litellm, in which case
        a question that litellm does not accept is rejected when its request is made.
    :rtype: list[str]
    """
    question_type = getattr(
        _litellm_decision_types, _LITELLM_QUESTION_TYPE_NAMES[question.kind], None
    )
    validate_question = getattr(question_type, "model_validate", None)
    if not callable(validate_question):
        return []
    try:
        validate_question(_get_request_question(question))
    except ValidationError as e:
        reasons: list[str] = []
        for error in e.errors():
            location = ".".join(str(part) for part in error["loc"])
            message = error["msg"].removeprefix("Value error, ")
            reasons.append(f"`{location}`: {message}" if location else message)
        return reasons
    except Exception:  # noqa: BLE001 - litellm's internals may change
        return []
    return []


@dataclass
class _QuestionsOnText:
    """
    Questions to ask a decision model on one text, e.g. on one paragraph: what
    the requests to the model are formed from.

    :ivar text: The text to ask the questions on.
    :vartype text: str
    :ivar questions: The questions to ask on the text, with the names unique
        among them. Different texts of the same call may have different questions.
    :vartype questions: list[_DecisionQuestion]
    :ivar location: Where the text is, as a part of the message on a response
        that cannot be used, e.g. " on paragraph 3 of 5 (index 2)". Defaults to
        an empty string, for a text that is not a part of a larger one.
    :vartype location: str
    """

    text: str
    questions: list[_DecisionQuestion]
    location: str = ""


@dataclass
class _RefusedOnAspect:
    """
    What a decision model refused to answer the question of an aspect on, i.e.
    what is not decided for the aspect: paragraphs, and sentences of
    the paragraphs that belong to the aspect, in the order of the source.

    :ivar paragraphs: The paragraphs refused.
    :vartype paragraphs: list[_Paragraph]
    :ivar sentences: The sentences refused, each with its paragraph.
    :vartype sentences: list[tuple[_Paragraph, _Sentence]]
    :ivar n_paragraphs_asked_about: The number of the paragraphs that the model
        was asked about for the aspect.
    :vartype n_paragraphs_asked_about: int
    :ivar n_sentences_asked_about: The number of the sentences that the model
        was asked about for the aspect.
    :vartype n_sentences_asked_about: int
    """

    paragraphs: list[_Paragraph]
    sentences: list[tuple[_Paragraph, _Sentence]]
    n_paragraphs_asked_about: int = 0
    n_sentences_asked_about: int = 0

    @property
    def has_refusals(self) -> bool:
        """
        Returns whether the model refused to answer on any paragraph or sentence.

        :return: True if it did, False otherwise.
        :rtype: bool
        """
        return bool(self.paragraphs or self.sentences)


def _get_sentence_position(paragraph: _Paragraph, sentence: _Sentence) -> int:
    """
    Returns the position of a sentence among the sentences of its paragraph.

    :param paragraph: The paragraph.
    :type paragraph: _Paragraph
    :param sentence: The sentence, an object of the paragraph.
    :type sentence: _Sentence
    :return: The position of the sentence.
    :rtype: int
    """
    return next(
        position
        for position, paragraph_sentence in enumerate(paragraph.sentences)
        if paragraph_sentence is sentence
    )


def _copy_aspect_for_fallback(aspect: _Aspect) -> _Aspect:
    """
    Returns a copy of an aspect for a fallback model to extract from what
    a decision model refused to answer on: with the params of the aspect, and
    without its sub-aspects, its concepts, its items and its references, i.e.
    as an aspect that is not processed.

    :param aspect: The aspect.
    :type aspect: _Aspect
    :return: The copy.
    :rtype: _Aspect
    """
    copy = aspect.clone()
    copy.aspects = []
    copy.concepts = []
    copy.extracted_items = []
    copy.reference_paragraphs = []
    copy.reference_sentences = []
    copy._is_processed = False
    copy._refused_paragraph_ids = set()
    copy._refused_sentence_ids = set()
    return copy


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

    The option that a "choice" answer names as the picked one is passed on as
    returned. litellm returns it as the provider named it, while it returns
    the probabilities of the options offered only, with a zero for each option
    that the provider returned no probability for, e.g. an option that
    the provider named differently.

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
                "the answer at the position of the question has another name.",
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
                chosen_option=getattr(response_answer, "choice", None),
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
                question.name, "the answer is of an unknown type."
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
                question.name, "the answer repeats an option."
            )
        option_probabilities[label] = getattr(option_probability, "probability", None)
    return option_probabilities


@_disable_direct_initialization
class _DocumentDecisionModel(_DocumentModel):
    """
    Internal implementation of the ``DocumentDecisionModel`` class.
    """

    decision_threshold: StrictFloat = Field(
        default=_DEFAULT_DECISION_THRESHOLD,
        gt=0,
        le=1,
        description=(
            "Lowest probability of 'yes' with which a yes/no question is taken as "
            "answered 'yes' (0 < threshold <= 1): a paragraph or a sentence belongs "
            "to an aspect, a Boolean concept is True, a label of a multi-label "
            "concept applies. "
            "Applies unless the aspect or the concept has a `decision_threshold` of "
            "its own. Does not apply to single-label and rating concepts, for which "
            "the most probable option is taken."
        ),
    )
    max_retries_invalid_data: StrictInt = Field(
        default=3,
        ge=0,
        description=(
            "Retries when a response of the model cannot be used, e.g. an answer "
            "without valid probabilities, or when the model or its provider "
            "declines to answer; only the requests whose answers cannot be used "
            "are sent again. A fallback model has its own, after it is asked for "
            "the first time."
        ),
    )

    @property
    def _retries_invalid_data(self) -> int:
        """
        Returns the number of times that a call is made again when a response of
        the decision model cannot be used, or the model or its provider declines to
        answer, as set for the model. The requests of the call whose answers were
        usable are not sent again. A fallback model has the same number of retries,
        after the first call that is made with it.

        :return: The number of retries of a call with a response that cannot be used.
        :rtype: int
        """
        return self.max_retries_invalid_data

    @property
    def _kind_name(self) -> str:
        """
        Returns the name of the kind of the model, as used in the messages of
        the logic that models of all kinds share.

        :return: The name of the kind of the model.
        :rtype: str
        """
        return "decision model"

    @classmethod
    def _validate_kind_specific_model_id(cls, model: str) -> None:
        """
        Validates that the model is served by a provider of decision models that
        litellm supports, when the model identifier is validated. A model of
        another provider, e.g. of a provider of LLMs only, would fail at its first
        request.

        The providers are those listed by the installed version of litellm.
        If they cannot be read from it, the provider is not checked.

        :param model: The model identifier, in the form of
            ``<model_provider>/<model_name>``.
        :type model: str
        :return: None
        :rtype: None
        :raises ValueError: If the provider of the model is not a provider of
            decision models that litellm supports.
        """
        supported_providers = _get_supported_decision_providers()
        if supported_providers is None:
            return
        provider, _, model_name = model.partition("/")
        if provider not in supported_providers:
            msg = (
                f"Provider `{provider}` of model `{model}` is not a provider of "
                f"decision models supported by litellm. Supported providers: "
                f"{', '.join(supported_providers)}."
            )
            self_hosted_prefix = _SELF_HOSTED_DECISION_MODEL_PROVIDERS[0]
            if (
                f"{provider}/" in _LOCAL_MODEL_PROVIDERS
                and self_hosted_prefix.rstrip("/") in supported_providers
            ):
                # The prefix is the one of the LLMs of a local server
                msg += (
                    f" For a decision model served by a local server, e.g. by Ollama "
                    f"or LM Studio, use the `{self_hosted_prefix}` prefix, i.e. "
                    f"`{self_hosted_prefix}{model_name}`, and set `api_base` to "
                    f"the address of the server."
                )
            raise ValueError(msg)

    def _is_self_hosted(self) -> bool:
        """
        Checks whether the decision model is a self-hosted one, i.e. a model of
        a provider that has no prices.

        :return: True if the model is self-hosted, False otherwise.
        :rtype: bool
        """
        return any(
            self.model.startswith(provider)
            for provider in _SELF_HOSTED_DECISION_MODEL_PROVIDERS
        )

    def _has_text_generation_prices(self) -> bool:
        """
        Checks whether the prices that genai-prices has for the model are those
        of a text-generating model, i.e. include a price of output tokens.

        A decision model generates no text and is billed for its input only. Prices
        that include a price of output tokens are therefore the prices of the chat
        requests to a model of the same name, e.g. of a model that is served both
        as an LLM and as a decision model, which differ from the prices of
        the decision requests. The prices are looked up as they are for the costs,
        i.e. under the provider of the decision requests where genai-prices has
        one (see `_get_auto_pricing_provider_id()`).

        :return: True if the prices found for the model include a price of output
            tokens. False if they do not, or if no prices are found for the model.
        :rtype: bool
        """
        provider, _, model_ref = self.model.partition("/")
        provider_id = self._get_auto_pricing_provider_id(provider)
        try:
            price_data = _calc_auto_price(
                _GPUsage(input_tokens=0, output_tokens=1_000_000),
                model_ref=model_ref,
                provider_id=provider_id,
            )
            return price_data.output_price > 0
        except Exception:  # noqa: BLE001 - genai-prices may raise anything
            # No prices are found for the model, which is warned about when
            # the automatic pricing is set up and when it is applied
            return False

    def _get_auto_pricing_error_msg(self) -> str | None:
        """
        Returns the error message for the automatic pricing of the decision model,
        if its costs cannot be calculated automatically: for a self-hosted model,
        which has no prices, and for a model for which the pricing data has
        the prices of a text-generating model.

        :return: The error message, or None if the automatic pricing is not ruled
            out for the model.
        :rtype: str | None
        """
        if self._is_self_hosted():
            return (
                "`auto_pricing=True` is not supported for self-hosted decision models. "
                "Disable `auto_pricing` or provide explicit `pricing_details`."
            )
        if self._has_text_generation_prices():
            return (
                f"`auto_pricing=True` is not supported for decision model "
                f"`{self.model}`: the pricing data found for it has a price of "
                f"output tokens, i.e. it has the prices of the chat requests to "
                f"this model, which differ from the prices of the decision requests "
                f"and would be applied instead of them. Disable `auto_pricing` and "
                f"provide explicit `pricing_details` to get the costs calculated."
            )
        return None

    def _get_auto_pricing_provider_id(self, provider: str) -> str:
        """
        Returns the ID of the provider under which genai-prices has the prices
        of the decision requests to the model.

        For a provider whose decision requests genai-prices prices under
        a provider of their own, e.g. OpenAI, which serves the same models for
        chat and for decisions, this is the ID of that provider: the prices under
        the provider in the identifier of the model are those of the chat requests.

        :param provider: The provider in the identifier of the model, i.e. the part
            of `model` before the first "/".
        :type provider: str
        :return: The ID of the provider for genai-prices.
        :rtype: str
        """
        return _DECISION_PRICING_PROVIDER_IDS.get(provider, provider)

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
            decision model, or for a decision model for which the pricing data has
            the prices of a text-generating model, i.e. a price of output tokens.
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
        return

    def _group_instances_for_calls(
        self,
        instance_type: ExtractedInstanceType,
        instances: list[_Aspect] | list[_Concept],
        fields: list[str],
    ) -> list[list[_Aspect] | list[_Concept]]:
        """
        Groups the aspects or concepts to be extracted by the decision model into
        the groups that are extracted in separate calls.

        The instances are not split by the params that a decision model is not
        sent, i.e. the depth and the length of justifications and, for concepts,
        the depth of references: the same text would otherwise be sent, and billed,
        once for each value of such a param. They are still split by the params
        that decide whether a decision model can extract an instance, which have
        the same values for all the instances that it can extract, and aspects
        also by the depth of references, by which an aspect is extracted with
        paragraphs or with sentences as its items.

        :param instance_type: Type of the instances ("aspect" or "concept").
        :type instance_type: ExtractedInstanceType
        :param instances: The aspects or concepts to be extracted by the model.
        :type instances: list[_Aspect] | list[_Concept]
        :param fields: The extraction params of the instances by which they are
            grouped for the extraction with an LLM.
        :type fields: list[str]
        :return: The groups of the instances.
        :rtype: list[list[_Aspect] | list[_Concept]]
        """
        not_applied_params = _NOT_APPLIED_INSTANCE_PARAMS + (
            _NOT_APPLIED_CONCEPT_PARAMS if instance_type == "concept" else ()
        )
        return _group_instances_by_fields(
            fields=[field for field in fields if field not in not_applied_params],
            instances=instances,
        )

    @staticmethod
    def _get_unservable_reasons(
        instance: _Aspect | _Concept, max_paragraphs_to_analyze_per_call: int = 0
    ) -> list[str]:
        """
        Returns the reasons why a decision model cannot extract an aspect or a concept.

        A decision model answers closed questions on a text and generates no text.
        It extracts an aspect by deciding on each paragraph whether the paragraph
        belongs to the aspect and, for an aspect with the "sentences" reference
        depth, on each sentence of such a paragraph whether the sentence does,
        i.e. with paragraphs or sentences as the extracted items, and
        the concepts whose values come from a closed set of options (Boolean, label
        and rating concepts), from text only and with no references. Neither
        is extracted with justifications.

        A concept with a singular occurrence cannot be extracted if the max number
        of paragraphs to analyze per call is set, i.e. if the text is to be
        analyzed in parts: a decision model answers on each part, i.e. it returns
        an item for each of them, and cannot stop at the first part that has
        an item, as an LLM does for such a concept. Such a concept is rejected
        whatever the number of parts that the text is split into, also if it is
        one: that number is not known before the extraction for the concepts of
        an aspect that is yet to be extracted. With no max number of paragraphs
        set, the text is sent as a whole, and the concept gets at most one item.

        An aspect or a concept of a supported type and with supported settings
        cannot be extracted either if litellm does not accept a question built
        for it, e.g. the question of a rating concept with more levels than
        a question may have: the request with such a question would be rejected
        before it is sent. The questions are checked with the types of litellm
        themselves, i.e. with the limits of the installed version.

        :param instance: The aspect or concept to be extracted.
        :type instance: _Aspect | _Concept
        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs to
            analyze per call that the extraction is run with. Defaults to 0, in
            which case the text is not analyzed in parts.
        :type max_paragraphs_to_analyze_per_call: int
        :return: The reasons, or an empty list if a decision model can extract
            the aspect or concept.
        :rtype: list[str]
        """
        reasons: list[str] = []
        if isinstance(instance, _Aspect):
            if instance.add_justifications:
                reasons.append("justifications are not supported")
            questions = [] if reasons else _build_aspect_questions([instance])
        else:
            if not isinstance(instance, _DECISION_CONCEPT_TYPES):
                reasons.append(f"type `{type(instance).__name__}` is not supported")
            if instance.modality != "text":
                reasons.append(f"modality `{instance.modality}` is not supported")
            if instance.add_justifications:
                reasons.append("justifications are not supported")
            if instance.add_references:
                reasons.append("references are not supported")
            questions = [] if reasons else _build_concept_questions([instance])
            if max_paragraphs_to_analyze_per_call > 0 and instance.singular_occurrence:
                reasons.append(
                    "`singular_occurrence=True` cannot be applied when the text is "
                    "analyzed in parts (`max_paragraphs_to_analyze_per_call` is "
                    "set): a decision model returns an item for each part"
                )

        for question in questions:
            rejection_reasons = _get_question_rejection_reasons(question)
            if rejection_reasons:
                question_desc = f"`{question.kind}` question"
                if question.options:
                    question_desc += f" with {len(question.options)} options"
                reasons.append(
                    f"its {question_desc} is not accepted by litellm: "
                    f"{'; '.join(rejection_reasons)}"
                )
                # The questions of an instance are of one kind, i.e. the other
                # questions are rejected for the same reasons, if they are
                break
        return reasons

    def _check_instances_servable(
        self,
        instances: Sequence[_Aspect | _Concept],
        routed_in_group: bool = False,
        max_paragraphs_to_analyze_per_call: int = 0,
    ) -> None:
        """
        Checks, before any requests are sent, that the decision model can extract
        the given aspects and concepts, which are to be extracted by it.

        :param instances: The aspects and concepts to be extracted by the model.
        :type instances: Sequence[_Aspect | _Concept]
        :param routed_in_group: Whether the aspects and concepts are routed to
            the model by its role in a model group, in which case the error tells
            how to route them to an LLM of the group. Defaults to False.
        :type routed_in_group: bool
        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs to
            analyze per call that the extraction is run with: if it is set,
            a concept with a singular occurrence cannot be extracted. Defaults
            to 0 (all paragraphs).
        :type max_paragraphs_to_analyze_per_call: int
        :return: None
        :rtype: None
        :raises ValueError: If the decision model cannot extract some of the aspects
            or concepts. The error names each of them with the reasons.
        """
        unservable_instances: list[str] = []
        for instance in instances:
            reasons = self._get_unservable_reasons(
                instance,
                max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
            )
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
        if routed_in_group:
            # The model is already in a group: what it cannot extract is to be
            # routed to another model of the group
            hint = (
                "In a model group, an aspect or a concept is extracted by the model "
                "with the `role` equal to its `model_role`: set the `model_role` "
                "of the other aspects and concepts to the role of an LLM in the group."
            )
        else:
            hint = (
                "Route the other aspects and concepts to an LLM, e.g. in a model group."
            )
        raise ValueError(
            f"{subject} `{self.model}` cannot extract{condition}: "
            f"{'; '.join(unservable_instances)}. "
            f"A decision model extracts aspects, and Boolean, label and rating "
            f"concepts from text without references, both without justifications. "
            f"{hint}"
        )

    @staticmethod
    def _get_source_paragraphs(
        extracted_item_type: ExtractedInstanceType,
        source: _Document | _Aspect,
        document: _Document,
    ) -> list[_Paragraph]:
        """
        Returns the paragraphs of the source that the aspects or concepts are
        extracted from. These are the paragraphs that an LLM is given for
        the same extraction:

        - for aspects, the paragraphs of the document. The sub-aspects of an aspect
          are extracted from a document that is built of the paragraphs extracted
          for that aspect;
        - for the concepts of a document, the paragraphs of the document;
        - for the concepts of an aspect, the paragraphs extracted for the aspect.

        :param extracted_item_type: Type of the item(s) being extracted
            ("aspect" or "concept").
        :type extracted_item_type: ExtractedInstanceType
        :param source: The source to extract from.
        :type source: _Document | _Aspect
        :param document: The document containing the source.
        :type document: _Document
        :return: The paragraphs of the source.
        :rtype: list[_Paragraph]
        :raises ValueError: If the source cannot be extracted from, or has no text.
        """
        if extracted_item_type == "aspect":
            if not isinstance(source, _Document):
                raise ValueError(
                    "Aspect extraction is only supported for Document sources."
                )
            if not (source.raw_text or source.paragraphs):
                raise ValueError(
                    "Document lacks text or paragraphs for aspect extraction."
                )
            paragraphs = document.paragraphs
        elif extracted_item_type == "concept":
            if isinstance(source, _Document):
                if not (source.raw_text or source.paragraphs):
                    raise ValueError(
                        "Document lacks text or paragraphs for concept extraction."
                    )
                paragraphs = document.paragraphs
            else:
                paragraphs = source.reference_paragraphs
        else:
            raise ValueError(
                f"Unsupported extracted item type: `{extracted_item_type}`"
            )
        if not paragraphs:
            raise ValueError("Context lacks paragraphs for text-based extraction.")
        return paragraphs

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
        answers_cache: dict[str, dict[str, _DecisionAnswer]] | None = None,
    ) -> tuple[list[_Aspect] | list[_Concept] | None, _ModelUsage]:
        """
        Extracts the items of aspects or concepts with the decision model: asks
        the model the questions of the aspects or concepts on the text of the source,
        and converts its answers to the items, each with the probabilities
        of its options.

        Called by ``_extract_items_from_instances``, the entry point of the extraction
        with a model of any kind, after the source is checked and the cases with
        nothing to extract are handled there. The result is returned under the same
        contract as the result of the extraction with an LLM:

        - if a request fails for good, the instances are None and the last call
          recorded in the usage is unfinished (no response received), unless
          an exception is raised;
        - if a response cannot be used, e.g. an answer without valid probabilities,
          the instances are None and each call is recorded as finished, so that it
          is told apart from a failed request;
        - if the model or its provider declined to answer, the instances are None
          and the refused call is recorded with its refusal, so that the refusal is
          reported if no fallback model answers.

        Concepts: all the questions of the concepts are sent in one request, with
        the text of the source, which is sent as a whole by default, i.e. it has to
        fit in a request. If the max number of paragraphs to analyze per call is set
        and the source has more paragraphs, the paragraphs are split into consecutive
        parts of at most that many paragraphs, as they are for an LLM, and one
        request is sent for each part, with the text of the part and the same
        questions. Each concept then gets an item for each part, in the order of
        the parts, each with the probabilities of its own answers: the answers on
        different parts are not merged. The requests for the parts are sent
        concurrently if a limiter is given, and one after another otherwise.
        A concept with a singular occurrence is rejected if that number is set,
        as it would get several items.

        Aspects: one request is sent for each paragraph of the source, with
        the text of the paragraph and one question for each aspect: whether
        the paragraph belongs to the aspect. A paragraph is judged on its own,
        i.e. without the other paragraphs. Its additional context, if it has any,
        is sent with its text, as an LLM is given it.
        The requests are sent concurrently if a limiter is given, which is the case
        when concurrency is enabled for the extraction, each of them paced by
        the limiter, and one after another otherwise. Each paragraph that belongs
        to an aspect becomes an item of the aspect, in the order of the paragraphs.
        An aspect that no paragraph belongs to is processed and has no items, as
        an aspect for which an LLM returns no paragraphs.

        Aspects with the "sentences" reference depth are extracted in two stages.
        The first stage is the one above: which paragraphs belong to the aspect.
        In the second stage, one request is sent for each sentence of each
        paragraph that belongs to such an aspect, with the sentence and its
        paragraph as the text, and one question for each of such aspects that
        the paragraph belongs to: whether the sentence belongs to the aspect.
        Each sentence that belongs to the aspect becomes an item of the aspect,
        in the order of the document, i.e. a paragraph that belongs to the aspect
        while none of its sentences does gives no items. The requests of
        the second stage are sent as those of the first one are, after them, and
        are recorded in the same usage. They are not sent if the first stage fails.
        It is the reference depth of each aspect that decides how the aspect is
        extracted, i.e. the aspects of a call may have different ones.

        A refused answer to the question of an aspect is not a response that cannot
        be used: the paragraph, or the sentence, that the model refused to answer
        on is not included in the aspect, the other answers are used, and a warning
        is issued for each aspect with such answers, which names the aspect and
        what the model refused to answer on. The request is not sent again.
        If the model has fallback models, they are asked about what it refused to
        answer on first (see ``_ask_fallbacks_about_refused``), and the warning names
        what none of them decided.
        Only if the model refuses to answer the question of an aspect on every
        paragraph, the response cannot be used, as nothing is known about
        the aspect. A refused answer to a question of a concept cannot be used
        either. Both are refusals, as is a request that the provider blocks for its
        content, or whose answers it blocks: the call fails, it is made again up to
        ``max_retries_invalid_data`` times, as it is for a response that cannot be
        used, then the fallback models are tried, and ``ModelRefusalError`` is
        raised (or a warning is logged) if none answers.

        When the call is made again, the requests whose answers were usable are not
        sent again (see ``answers_cache``): only the request whose response cannot
        be used, or that was refused, and those not sent yet.

        If there are more questions on a text than a request can hold, they are
        sent in several requests on the same text.

        An aspect or a concept that is given more than once is asked about, and
        processed, once.

        Whatever the call ends with, the usage of the requests made is not lost:
        it is returned with the result, or the usage and cost of the model are
        updated with it before an exception is raised.

        :param extracted_item_type: Type of the item(s) being extracted
            ("aspect" or "concept").
        :type extracted_item_type: ExtractedInstanceType
        :param source: The source to extract from (_Document or _Aspect)
        :type source: _Document | _Aspect
        :param instances_to_process: List of aspects or concepts to process
        :type instances_to_process: list[_Aspect] | list[_Concept]
        :param document: The document containing the source.
        :type document: _Document
        :param overwrite_existing: Whether to overwrite the items of the already
            processed aspects or concepts. Defaults to False.
        :type overwrite_existing: bool
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs
            of the source whose text is sent in one request for concepts: a source
            with more paragraphs is asked about in parts. Defaults to 0, in which
            case the text of the source is sent as a whole. If it is set, a concept
            with a singular occurrence cannot be extracted. Has no effect on
            aspects, for which each paragraph is sent in a request of its own,
            i.e. any such limit is met.
        :type max_paragraphs_to_analyze_per_call: int
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :param raise_exception_on_extraction_error: Whether to raise an exception
            if a request fails due to an error in the API of the model. If False,
            a warning is logged instead, and no items are extracted. Defaults to True.
        :type raise_exception_on_extraction_error: bool
        :param has_fallback_pending: Whether a fallback model will be tried if
            the call to this model fails. If True, a model API error is logged
            without being raised or warned about, as it is not final.
            Defaults to False.
        :type has_fallback_pending: bool
        :param answers_cache: The usable answers of the requests of the call, which
            the call keeps when it is made again (see ``_PartialExtraction``), and
            which it adds the usable answers of its requests to. Defaults to None,
            i.e. no answers are kept.
        :type answers_cache: dict[str, dict[str, _DecisionAnswer]] | None
        :return: A tuple containing:
            (0) List of processed instances with extracted items, or None if
                a request failed or a response cannot be used
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
        self._check_instances_servable(
            instances_to_process,
            max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
        )

        source_paragraphs = self._get_source_paragraphs(
            extracted_item_type=extracted_item_type, source=source, document=document
        )
        # What the call is for, named in the messages on a refusal
        call_instances_msg = (
            f" {extracted_item_type.capitalize()}s of the call: "
            f"{[instance.name for instance in instances_to_process]}"
        )

        # Ask the questions. Each request is recorded in the usage: with the tokens
        # used and the answers if it succeeds, or as an unfinished call if it fails.
        usage = _ModelUsage()
        # The paragraphs that the questions of aspects are asked on, and the answers
        # on each text: on each of these paragraphs for aspects, or on the text of
        # the source, or of each part of it, for concepts
        paragraphs: list[_Paragraph] = []
        answers_of_texts: list[dict[str, _DecisionAnswer]]
        # The answers on the sentences of each of these paragraphs, for the aspects
        # that are extracted with sentences as their items
        answers_of_sentences: list[_AnswersOnSentences] = []
        try:
            if extracted_item_type == "aspect":
                (
                    paragraphs,
                    answers_of_texts,
                    answers_of_sentences,
                ) = await self._ask_about_aspects(
                    # Safe cast: only aspects are extracted with the "aspect"
                    # item type
                    aspects=cast(list[_Aspect], instances_to_process),
                    source_paragraphs=source_paragraphs,
                    usage=usage,
                    num_retries_failed_request=num_retries_failed_request,
                    async_limiter=async_limiter,
                    answers_cache=answers_cache,
                )
            else:
                answers_of_texts = await self._ask_about_concepts(
                    # Safe cast: only concepts are extracted with the "concept"
                    # item type
                    concepts=cast(list[_Concept], instances_to_process),
                    source=source,
                    source_paragraphs=source_paragraphs,
                    usage=usage,
                    max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                    num_retries_failed_request=num_retries_failed_request,
                    async_limiter=async_limiter,
                    answers_cache=answers_cache,
                )
        except _RefusedDecisionAnswerError as e:
            # The model refused to answer a question of a concept, or the question
            # of an aspect on every paragraph. The calls are recorded as finished,
            # with the refusal.
            return self._handle_refusal(
                refusal_subject=self._get_refused_answer_subject(
                    error=e,
                    extracted_item_type=extracted_item_type,
                    instances=instances_to_process,
                ),
                call_instances_msg=call_instances_msg,
                usage=usage,
            )
        except _InvalidDecisionResponseError as e:
            # The requests succeeded, i.e. their calls are recorded as finished
            logger.error(
                self._get_invalid_response_msg(
                    error=e,
                    extracted_item_type=extracted_item_type,
                    instances=instances_to_process,
                )
            )
            return None, usage
        except asyncio.CancelledError:
            # The call is cancelled, e.g. on a timeout set by the calling code
            await self._update_usage_and_cost((None, usage))
            raise
        except litellm.ContentPolicyViolationError as e:  # type: ignore[attr-defined]
            # The provider blocked a request or its answers for their content:
            # a refusal, as a refused answer to a question of a concept is, i.e.
            # the call is made again, as the verdict of a content filter can differ
            # on another request, and the fallback models are tried next. The calls
            # are recorded as finished, the blocked one with the refusal.
            # The error of litellm, the message of the provider
            logger.debug(
                f"The request or its answers were blocked by the provider: {e}"
            )
            return self._handle_refusal(
                refusal_subject="",
                call_instances_msg=call_instances_msg,
                usage=usage,
            )
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
            return None, usage

        # The requests are made: whatever fails from here on, their usage is counted
        try:
            # Convert the answers to the items of each aspect or concept
            instances_and_items: list[tuple[_Aspect | _Concept, list[_ExtractedItem]]]
            try:
                if extracted_item_type == "aspect":
                    instances_and_items = self._get_aspects_items(
                        # Safe cast: only aspects are extracted with the "aspect"
                        # item type
                        aspects=cast(list[_Aspect], instances_to_process),
                        paragraphs=paragraphs,
                        answers_of_paragraphs=answers_of_texts,
                        decision_threshold=self.decision_threshold,
                        answers_of_sentences=answers_of_sentences,
                    )
                else:
                    instances_and_items = self._get_concepts_items(
                        # Safe cast: only concepts are extracted with the "concept"
                        # item type
                        concepts=cast(list[_Concept], instances_to_process),
                        answers_of_texts=answers_of_texts,
                        decision_threshold=self.decision_threshold,
                    )
            except _RefusedDecisionAnswerError as e:
                # A refused answer to a question of a concept. Such answers are
                # rejected when the responses are received: this is a safeguard.
                # The answers are asked for again if the call is made again.
                if answers_cache is not None:
                    answers_cache.clear()
                return self._handle_refusal(
                    refusal_subject=self._get_refused_answer_subject(
                        error=e,
                        extracted_item_type=extracted_item_type,
                        instances=instances_to_process,
                    ),
                    call_instances_msg=call_instances_msg,
                    usage=usage,
                )
            except ValueError as e:
                # An answer that cannot be used (missing, of another kind than its
                # question, without the probabilities of the options offered), or
                # a value that the item of the aspect or concept does not accept.
                # Which request it is from is not known here: all the answers are
                # asked for again if the call is made again.
                if answers_cache is not None:
                    answers_cache.clear()
                logger.error(
                    self._get_invalid_response_msg(
                        error=e,
                        extracted_item_type=extracted_item_type,
                        instances=instances_to_process,
                    )
                )
                return None, usage

            # What the model refused to answer on for the aspects, and what stays
            # undecided once its fallback models are asked about it, if it has any
            refused_on_aspects: list[_RefusedOnAspect] = []
            undecided_on_aspects: list[_RefusedOnAspect] | None = None
            if extracted_item_type == "aspect":
                # Safe cast: only aspects are extracted with the "aspect" item type
                aspects = cast(list[_Aspect], instances_to_process)
                refused_on_aspects = self._get_refused_on_aspects(
                    aspects=aspects,
                    paragraphs=paragraphs,
                    answers_of_paragraphs=answers_of_texts,
                    answers_of_sentences=answers_of_sentences,
                )
                if self.fallback_models and any(
                    refused.has_refusals for refused in refused_on_aspects
                ):
                    items_of_aspects = [items for _, items in instances_and_items]
                    undecided_on_aspects = await self._ask_fallbacks_about_refused(
                        aspects=aspects,
                        items_of_aspects=items_of_aspects,
                        refused_on_aspects=refused_on_aspects,
                        source_paragraphs=source_paragraphs,
                        max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                        use_concurrency=async_limiter is not None,
                    )
                    instances_and_items = list(
                        zip(aspects, items_of_aspects, strict=True)
                    )

            # Process all gathered results for all processed instances
            async with (
                self._async_lock
            ):  # ensure atomicity of the instances' state check and modification
                for instance, extracted_items in instances_and_items:
                    self._check_instances_already_processed(
                        instance_type=extracted_item_type,
                        # Safe cast: a list with a single aspect or a single concept
                        instances=cast(list[_Aspect] | list[_Concept], [instance]),
                        overwrite_existing=overwrite_existing,
                    )
                    instance.extracted_items = extracted_items
                    if isinstance(instance, _Aspect):
                        if overwrite_existing:
                            # The references of an aspect that is extracted again
                            # are those of its new items only
                            instance.reference_paragraphs = []
                            instance.reference_sentences = []
                        # References are automatically included for aspects, as
                        # paragraphs/sents' texts are the values of extracted items
                        # for the aspects. As when an LLM extracts the aspect:
                        # the reference paragraphs of its items, and the reference
                        # sentences of its items, i.e. the extracted sentences,
                        # and not all the sentences of its reference paragraphs.
                        instance._add_references_of_extracted_items()
                    instance._is_processed = True
                if (
                    extracted_item_type == "aspect"
                    and document._for_fallback_of_refusals
                ):
                    # What the model refused to answer on for each aspect, passed
                    # back to the decision model that it is a fallback model of
                    for instance, undecided in zip(
                        instances_to_process,
                        undecided_on_aspects or refused_on_aspects,
                        strict=True,
                    ):
                        # Safe cast: only aspects are extracted with the "aspect"
                        # item type
                        aspect = cast(_Aspect, instance)
                        aspect._refused_paragraph_ids = {
                            paragraph.unique_id for paragraph in undecided.paragraphs
                        }
                        aspect._refused_sentence_ids = {
                            sentence.unique_id for _, sentence in undecided.sentences
                        }

            # The aspects are extracted without what stays undecided. In a document
            # of what a decision model refused to answer on, this is passed back to
            # that model, which reports it.
            if extracted_item_type == "aspect" and not (
                document._for_fallback_of_refusals
            ):
                for warning_msg in self._get_refused_answers_msgs(
                    # Safe cast: only aspects are extracted with the "aspect"
                    # item type
                    aspects=cast(list[_Aspect], instances_to_process),
                    source_paragraphs=source_paragraphs,
                    refused_on_aspects=refused_on_aspects,
                    undecided_on_aspects=undecided_on_aspects,
                ):
                    logger.warning(warning_msg)
        except (Exception, asyncio.CancelledError):
            # e.g. an aspect or a concept that was processed by another call while
            # the requests of this call were made, or a cancellation while
            # the fallback models are asked about what the model refused to answer
            # on. No usage is returned when an exception is raised, i.e. the usage
            # of the requests would be lost.
            await self._update_usage_and_cost((None, usage))
            raise

        return instances_to_process, usage

    def _get_invalid_response_msg(
        self,
        error: ValueError,
        extracted_item_type: ExtractedInstanceType,
        instances: Sequence[_Aspect | _Concept],
    ) -> str:
        """
        Returns the message on a response of the decision model that cannot be used.

        The message names what the answer that cannot be used is for, so that it
        can be found: the aspect or the concept, with the label if the question is
        on one label of a concept, and where the text that the question was asked
        on is, if the error has its location, e.g. the paragraph, as the questions
        of aspects are asked on each paragraph. The paragraph is given by its
        position among the paragraphs that the aspects are extracted from, i.e.
        those of the document, or of the parent aspect for sub-aspects.
        A sentence that a question of an aspect was asked on is given by its
        position among the sentences of its paragraph, with that paragraph. For
        the concepts whose questions are asked on the text in parts, it is
        the part, given by its position among the parts of the text.

        :param error: The error that tells what is wrong with the response.
        :type error: ValueError
        :param extracted_item_type: Type of the item(s) being extracted
            ("aspect" or "concept").
        :type extracted_item_type: ExtractedInstanceType
        :param instances: The aspects or concepts that the questions of the call
            were built for, in the order of their questions.
        :type instances: Sequence[_Aspect | _Concept]
        :return: The message.
        :rtype: str
        """
        intro = f"Decision model `{self.model}` returned an invalid response for"
        if not isinstance(error, _InvalidDecisionResponseError):
            # Not an answer to a specific question, e.g. a value that the item
            # of an aspect or a concept does not accept
            return (
                f"{intro} {extracted_item_type}s: "
                f"{_describe_error_without_values(error)}"
            )
        question = self._get_question_of_error(
            error=error, extracted_item_type=extracted_item_type, instances=instances
        )
        return f"{intro} {question}: {error.reason}"

    @staticmethod
    def _get_question_of_error(
        *,
        error: _InvalidDecisionResponseError,
        extracted_item_type: ExtractedInstanceType,
        instances: Sequence[_Aspect | _Concept],
    ) -> str:
        """
        Returns what the question with the answer that cannot be used is on, for
        the messages on such answers: the aspect or the concept, with the label if
        the question is on one label of a concept, where the text that the question
        was asked on is, if the error has its location, and the name of the question
        (see ``_get_invalid_response_msg``).

        :param error: The error on the answer that cannot be used.
        :type error: _InvalidDecisionResponseError
        :param extracted_item_type: Type of the item(s) being extracted
            ("aspect" or "concept").
        :type extracted_item_type: ExtractedInstanceType
        :param instances: The aspects or concepts that the questions of the call
            were built for, in the order of their questions.
        :type instances: Sequence[_Aspect | _Concept]
        :return: What the question is on, e.g. "concept `Parties` on part 2 of 3
            (index 1), decision question `C0`".
        :rtype: str
        """
        # What each question of the call is on, by the names of the questions
        subjects: dict[str, str] = {}
        for position, instance in enumerate(instances):
            if extracted_item_type == "aspect":
                subjects[_get_aspect_question_name(position)] = (
                    f"aspect `{instance.name}`"
                )
                continue
            subjects[_get_concept_question_name(position)] = (
                f"concept `{instance.name}`"
            )
            if isinstance(instance, _LabelConcept):
                for label_position, label in enumerate(instance.labels):
                    subjects[_get_concept_question_name(position, label_position)] = (
                        f"concept `{instance.name}` (label `{label}`)"
                    )
        subject = subjects.get(error.question_name, f"{extracted_item_type}s")

        return f"{subject}{error.location}, decision question `{error.question_name}`"

    def _get_refused_answer_subject(
        self,
        *,
        error: _RefusedDecisionAnswerError,
        extracted_item_type: ExtractedInstanceType,
        instances: Sequence[_Aspect | _Concept],
    ) -> str:
        """
        Returns what an answer that the decision model refused to give was for,
        put after the model in the messages on the refusal: the question of
        a concept, or the question of an aspect refused on every paragraph, named
        as the message on an answer that cannot be used names it (see
        ``_get_invalid_response_msg``).

        :param error: The error on the refused answer.
        :type error: _RefusedDecisionAnswerError
        :param extracted_item_type: Type of the item(s) being extracted
            ("aspect" or "concept").
        :type extracted_item_type: ExtractedInstanceType
        :param instances: The aspects or concepts that the questions of the call
            were built for, in the order of their questions.
        :type instances: Sequence[_Aspect | _Concept]
        :return: What the refused answer was for, e.g. " the question of concept
            `Parties`, decision question `C0`".
        :rtype: str
        """
        question = self._get_question_of_error(
            error=error, extracted_item_type=extracted_item_type, instances=instances
        )
        return f" the question of {question}"

    def _handle_refusal(
        self,
        *,
        refusal_subject: str,
        call_instances_msg: str,
        usage: _ModelUsage,
    ) -> tuple[None, _ModelUsage]:
        """
        Reports a request of the extraction that the decision model or its provider
        declined to answer, and returns the result of the failed call, with
        the refusal, and what the refused answer was for, recorded on the refused
        call in its usage. As for a response that cannot be used, the fallback
        models are tried next, and the refusal is raised, or warned about, if none
        of them answers.

        :param refusal_subject: What the refused answer was for, e.g.
            " the question of concept `Parties`, decision question `C0`", or an
            empty string for a request that the provider blocked.
        :type refusal_subject: str
        :param call_instances_msg: The aspects or concepts of the call, named at
            the end of the message.
        :type call_instances_msg: str
        :param usage: The usage of the requests of the call, with the refused call.
        :type usage: _ModelUsage
        :return: The result of the failed call: no instances, and the usage.
        :rtype: tuple[None, _ModelUsage]
        """
        # The refused call is the last one if none is recorded with the refusal
        if usage.calls and not any(call._refusal for call in usage.calls):
            usage.calls[-1]._refusal = "refusal"
        refused_call = _get_model_call_result_refused_call((None, usage))
        if refused_call is None or refused_call._refusal is None:
            # This should never happen: a request of the call is refused
            raise RuntimeError("No refused request is recorded for the refusal.")
        refused_call._refusal_subject = refusal_subject
        logger.error(
            _get_model_refusal_msg(
                kind_name=self._kind_name,
                model=self.model,
                refusal=refused_call._refusal,
                subject=refusal_subject,
            )
            + call_instances_msg
        )
        return None, usage

    @staticmethod
    def _get_refused_on_aspects(
        *,
        aspects: list[_Aspect],
        paragraphs: list[_Paragraph],
        answers_of_paragraphs: list[dict[str, _DecisionAnswer]],
        answers_of_sentences: list[_AnswersOnSentences],
    ) -> list[_RefusedOnAspect]:
        """
        Returns what the decision model refused to answer the questions of
        the aspects on: for each aspect, the paragraphs, and the sentences, with
        the numbers of the paragraphs and of the sentences asked about for it.

        :param aspects: The aspects that the questions of the call were built for,
            in the order of their questions.
        :type aspects: list[_Aspect]
        :param paragraphs: The paragraphs that the questions were asked on.
        :type paragraphs: list[_Paragraph]
        :param answers_of_paragraphs: For each of these paragraphs, the answers of
            the model by the names of the questions.
        :type answers_of_paragraphs: list[dict[str, _DecisionAnswer]]
        :param answers_of_sentences: For each of these paragraphs, the sentences
            that the questions were asked on, each with the answers of the model
            by the names of the questions.
        :type answers_of_sentences:
            list[list[tuple[_Sentence, dict[str, _DecisionAnswer]]]]
        :return: What the model refused to answer on, for each aspect, in
            the order of the aspects.
        :rtype: list[_RefusedOnAspect]
        """
        refused_on_aspects: list[_RefusedOnAspect] = []
        for position in range(len(aspects)):
            question_name = _get_aspect_question_name(position)
            refused = _RefusedOnAspect(
                paragraphs=[], sentences=[], n_paragraphs_asked_about=len(paragraphs)
            )
            for paragraph, answers, sentences_answers in zip(
                paragraphs, answers_of_paragraphs, answers_of_sentences, strict=True
            ):
                if _is_refused_answer(answers, question_name):
                    refused.paragraphs.append(paragraph)
                for sentence, sentence_answers in sentences_answers:
                    if question_name not in sentence_answers:
                        # The sentence is asked about for other aspects only
                        continue
                    refused.n_sentences_asked_about += 1
                    if _is_refused_answer(sentence_answers, question_name):
                        refused.sentences.append((paragraph, sentence))
            refused_on_aspects.append(refused)
        return refused_on_aspects

    def _get_refused_answers_msgs(
        self,
        *,
        aspects: list[_Aspect],
        source_paragraphs: list[_Paragraph],
        refused_on_aspects: list[_RefusedOnAspect],
        undecided_on_aspects: list[_RefusedOnAspect] | None = None,
    ) -> list[str]:
        """
        Returns the messages on the answers that the decision model refused to give
        to the questions of the aspects: one message for each aspect with such
        answers that stay undecided, which names the aspect and what the model
        refused to answer on, i.e. what is not included in the aspect for that
        reason. If the fallback models of the model were asked about it,
        the message tells what none of them answered on either.

        A paragraph is given by its index among the paragraphs that the aspects
        are extracted from, i.e. those of the document, or of the parent aspect
        for sub-aspects, and a sentence by the index of its paragraph and its own
        index among the sentences of that paragraph. The paragraphs and
        the sentences that the model refused to answer on are counted out of those
        that the model was asked about for the aspect.

        :param aspects: The aspects that the questions of the call were built for,
            in the order of their questions.
        :type aspects: list[_Aspect]
        :param source_paragraphs: The paragraphs that the aspects are extracted
            from.
        :type source_paragraphs: list[_Paragraph]
        :param refused_on_aspects: What the model refused to answer on, for each
            aspect.
        :type refused_on_aspects: list[_RefusedOnAspect]
        :param undecided_on_aspects: What none of the fallback models answered on
            either, for each aspect, if they were asked about it. Defaults to None,
            i.e. they were not.
        :type undecided_on_aspects: list[_RefusedOnAspect] | None
        :return: The messages, in the order of the aspects. Empty if nothing stays
            undecided.
        :rtype: list[str]
        """
        # The paragraphs of the source are distinct objects
        source_positions = {
            id(paragraph): position
            for position, paragraph in enumerate(source_paragraphs)
        }

        def get_paragraph_indices(paragraphs: list[_Paragraph]) -> list[int]:
            """
            Returns the indices of paragraphs among those of the source.

            :param paragraphs: The paragraphs.
            :type paragraphs: list[_Paragraph]
            :return: The indices.
            :rtype: list[int]
            """
            return [source_positions[id(paragraph)] for paragraph in paragraphs]

        def get_sentence_indices(
            sentences: list[tuple[_Paragraph, _Sentence]],
        ) -> list[tuple[int, int]]:
            """
            Returns the index of the paragraph of each sentence among those of
            the source, with the index of the sentence among those of its paragraph.

            :param sentences: The sentences, each with its paragraph.
            :type sentences: list[tuple[_Paragraph, _Sentence]]
            :return: The indices.
            :rtype: list[tuple[int, int]]
            """
            return [
                (
                    source_positions[id(paragraph)],
                    _get_sentence_position(paragraph, sentence),
                )
                for paragraph, sentence in sentences
            ]

        msgs: list[str] = []
        for position, (aspect, refused) in enumerate(
            zip(aspects, refused_on_aspects, strict=True)
        ):
            undecided = (
                refused
                if undecided_on_aspects is None
                else undecided_on_aspects[position]
            )
            if not undecided.has_refusals:
                continue
            refused_parts: list[str] = []
            if refused.paragraphs:
                refused_parts.append(
                    f"{len(refused.paragraphs)} of {refused.n_paragraphs_asked_about} "
                    f"paragraphs (indices: {get_paragraph_indices(refused.paragraphs)})"
                )
            if refused.sentences:
                refused_parts.append(
                    f"{len(refused.sentences)} of {refused.n_sentences_asked_about} "
                    f"sentences (paragraph index, sentence index: "
                    f"{get_sentence_indices(refused.sentences)})"
                )
            msg = (
                f"Decision model `{self.model}` refused to answer the question "
                f"of aspect `{aspect.name}` on {' and on '.join(refused_parts)}"
            )
            if undecided_on_aspects is not None:
                undecided_parts: list[str] = []
                if undecided.paragraphs:
                    undecided_parts.append(
                        f"the paragraphs with indices "
                        f"{get_paragraph_indices(undecided.paragraphs)}"
                    )
                if undecided.sentences:
                    undecided_parts.append(
                        f"the sentences with (paragraph index, sentence index) "
                        f"{get_sentence_indices(undecided.sentences)}"
                    )
                msg += (
                    f", and its fallback models did not answer on "
                    f"{' and on '.join(undecided_parts)} either"
                )
            msgs.append(f"{msg}. These are not included in the aspect.")
        return msgs

    async def _ask_fallbacks_about_refused(
        self,
        *,
        aspects: list[_Aspect],
        items_of_aspects: list[list[_ExtractedItem]],
        refused_on_aspects: list[_RefusedOnAspect],
        source_paragraphs: list[_Paragraph],
        max_paragraphs_to_analyze_per_call: int = 0,
        use_concurrency: bool = False,
    ) -> list[_RefusedOnAspect]:
        """
        Asks the fallback models of the decision model about what it refused to
        answer the questions of the aspects of a call on, and adds what they decide
        to the items of the aspects.

        The fallback models are tried in their order. Each extracts copies of
        the aspects with what is still undecided from a document of the paragraphs
        that it is on: the paragraphs refused for any of these aspects, and those
        with refused sentences, as an LLM or a decision model extracts aspects,
        with its own retries. Of what it extracts, only its decisions on what is
        undecided are taken: on a refused paragraph, also on its sentences for
        an aspect with the "sentences" reference depth, and on a refused sentence.
        A fallback decision model can refuse to answer on some of it in turn: that
        is passed on to the next fallback model. A fallback model that fails, e.g.
        on an error of its API, decides nothing, and the next one is tried. What
        none of them decides stays undecided, i.e. is not included in the aspect,
        as without fallback models: the result of the call stands in any case.

        The items that the fallback models give are relinked to the paragraphs and
        the sentences of the source, and added to the items of their aspects, which
        are then in the order of the paragraphs and of the sentences in them.
        An item of a fallback LLM has no probabilities. The usage of the requests
        of each fallback model is recorded for that model.

        :param aspects: The aspects of the call, in the order of their questions.
        :type aspects: list[_Aspect]
        :param items_of_aspects: The items of each aspect, which the items that
            the fallback models give are added to.
        :type items_of_aspects: list[list[_ExtractedItem]]
        :param refused_on_aspects: What the decision model refused to answer on,
            for each aspect.
        :type refused_on_aspects: list[_RefusedOnAspect]
        :param source_paragraphs: The paragraphs that the aspects are extracted from.
        :type source_paragraphs: list[_Paragraph]
        :param max_paragraphs_to_analyze_per_call: The max number of paragraphs
            that a fallback LLM is sent in one request. Defaults to 0, i.e. no limit.
        :type max_paragraphs_to_analyze_per_call: int
        :param use_concurrency: Whether the fallback models extract the aspects with
            concurrency, as the decision model does. Defaults to False.
        :type use_concurrency: bool
        :return: What none of the fallback models decided, for each aspect.
        :rtype: list[_RefusedOnAspect]
        """
        undecided_on_aspects = [
            _RefusedOnAspect(
                paragraphs=list(refused.paragraphs),
                sentences=list(refused.sentences),
            )
            for refused in refused_on_aspects
        ]
        # The paragraphs and the sentences of the source, by their unique IDs, which
        # the references of the items that the fallback models give are relinked to
        paragraphs_by_id = {
            paragraph.unique_id: paragraph for paragraph in source_paragraphs
        }
        sentences_by_id = {
            sentence.unique_id: sentence
            for paragraph in source_paragraphs
            for sentence in paragraph.sentences
        }
        # The positions of the paragraphs of the source, which are distinct objects
        source_positions = {
            id(paragraph): position
            for position, paragraph in enumerate(source_paragraphs)
        }

        def get_sentence_source_position(
            paragraph_and_sentence: tuple[_Paragraph, _Sentence],
        ) -> tuple[int, int]:
            """
            Returns the position of a sentence in the source: of its paragraph, and
            its own among the sentences of the paragraph.

            :param paragraph_and_sentence: The sentence, with its paragraph.
            :type paragraph_and_sentence: tuple[_Paragraph, _Sentence]
            :return: The position.
            :rtype: tuple[int, int]
            """
            paragraph, sentence = paragraph_and_sentence
            return (
                source_positions[id(paragraph)],
                _get_sentence_position(paragraph, sentence),
            )

        for fallback in self.fallback_models:
            pending = [
                position
                for position, undecided in enumerate(undecided_on_aspects)
                if undecided.has_refusals
            ]
            if not pending:
                break
            asked_ids = {
                paragraph.unique_id
                for position in pending
                for paragraph in undecided_on_aspects[position].paragraphs
            } | {
                paragraph.unique_id
                for position in pending
                for paragraph, _ in undecided_on_aspects[position].sentences
            }
            sub_document = _publicize(
                _Document,
                paragraphs=[
                    paragraph
                    for paragraph in source_paragraphs
                    if paragraph.unique_id in asked_ids
                ],
            )
            sub_document._for_fallback_of_refusals = True
            sub_document.add_aspects(
                [_copy_aspect_for_fallback(aspects[position]) for position in pending]
            )
            pending_names = [aspects[position].name for position in pending]
            logger.info(
                f"Asking fallback {fallback._kind_name} `{fallback.model}` about "
                f"what decision model `{self.model}` refused to answer on for "
                f"aspects {pending_names}, on {len(sub_document.paragraphs)} "
                f"paragraph(s)"
            )
            try:
                await fallback.extract_aspects_from_document_async(
                    sub_document,
                    use_concurrency=use_concurrency,
                    max_paragraphs_to_analyze_per_call=max_paragraphs_to_analyze_per_call,
                )
            except ContextGemError:
                # The failure is logged by the fallback model, which decides
                # nothing: the next fallback model is tried
                logger.info(
                    f"Fallback {fallback._kind_name} `{fallback.model}` did not decide "
                    f"what decision model `{self.model}` refused to answer on for "
                    f"aspects {pending_names}."
                )
                continue
            for position, copy in zip(pending, sub_document.aspects, strict=True):
                undecided = undecided_on_aspects[position]
                undecided_paragraph_ids = {
                    paragraph.unique_id for paragraph in undecided.paragraphs
                }
                undecided_sentence_ids = {
                    sentence.unique_id for _, sentence in undecided.sentences
                }
                # The decisions of the fallback model on what is undecided
                for item in copy.extracted_items:
                    if not item.reference_paragraphs:
                        continue
                    paragraph_id = item.reference_paragraphs[0].unique_id
                    sentence_id = (
                        item.reference_sentences[0].unique_id
                        if item.reference_sentences
                        else None
                    )
                    if (
                        paragraph_id not in undecided_paragraph_ids
                        and sentence_id not in undecided_sentence_ids
                    ):
                        continue
                    item.reference_paragraphs = [paragraphs_by_id[paragraph_id]]
                    if sentence_id is not None:
                        item.reference_sentences = [sentences_by_id[sentence_id]]
                    items_of_aspects[position].append(item)
                # What the fallback model refused to answer on in turn: a refused
                # paragraph, a sentence of a paragraph that it decided, or a refused
                # sentence that it did not decide, in the order of the source
                undecided_on_aspects[position] = _RefusedOnAspect(
                    paragraphs=[
                        paragraph
                        for paragraph in undecided.paragraphs
                        if paragraph.unique_id in copy._refused_paragraph_ids
                    ],
                    sentences=sorted(
                        [
                            (paragraph, sentence)
                            for paragraph in undecided.paragraphs
                            if paragraph.unique_id not in copy._refused_paragraph_ids
                            for sentence in paragraph.sentences
                            if sentence.unique_id in copy._refused_sentence_ids
                        ]
                        + [
                            (paragraph, sentence)
                            for paragraph, sentence in undecided.sentences
                            if paragraph.unique_id in copy._refused_paragraph_ids
                            or sentence.unique_id in copy._refused_sentence_ids
                        ],
                        key=get_sentence_source_position,
                    ),
                )
            decided = [
                aspects[position].name
                for position in pending
                if not undecided_on_aspects[position].has_refusals
            ]
            if decided:
                logger.info(
                    f"Fallback {fallback._kind_name} `{fallback.model}` decided what "
                    f"decision model `{self.model}` refused to answer on for "
                    f"aspects {decided}."
                )

        # The items of each aspect in the order of the paragraphs and of
        # the sentences in them
        def get_item_position(item: _ExtractedItem) -> tuple[int, int]:
            """
            Returns the position of an item of an aspect in the source: of its
            paragraph, and of its sentence, or -1 for an item of a paragraph.

            :param item: The item.
            :type item: _ExtractedItem
            :return: The position.
            :rtype: tuple[int, int]
            """
            paragraph = item.reference_paragraphs[0]
            sentence_position = (
                _get_sentence_position(paragraph, item.reference_sentences[0])
                if item.reference_sentences
                else -1
            )
            return source_positions[id(paragraph)], sentence_position

        for items in items_of_aspects:
            items.sort(key=get_item_position)
        return undecided_on_aspects

    async def _ask_about_aspects(
        self,
        *,
        aspects: list[_Aspect],
        source_paragraphs: list[_Paragraph],
        usage: _ModelUsage,
        num_retries_failed_request: int = 3,
        async_limiter: AsyncLimiter | None = None,
        answers_cache: dict[str, dict[str, _DecisionAnswer]] | None = None,
    ) -> tuple[
        list[_Paragraph],
        list[dict[str, _DecisionAnswer]],
        list[_AnswersOnSentences],
    ]:
        """
        Asks the decision model, on each paragraph of the source, whether
        the paragraph belongs to each of the aspects, and then, for the aspects
        with the "sentences" reference depth, on each sentence of the paragraphs
        that belong to them, whether the sentence does (see
        ``_ask_about_sentences``). The sentences are not asked about if
        the paragraphs cannot be, i.e. if a request fails for good or a response
        cannot be used.

        One request is sent for each paragraph, with the text of the paragraph as
        it is shown to an LLM for the extraction of aspects, and one question for
        each aspect. A paragraph that has no such text is not shown to an LLM
        either: it is not asked about, i.e. it belongs to no aspect.

        A paragraph that has additional context is sent with it, as an LLM is given
        it: the text of the request then has the paragraph and its additional
        context, each with a label, and the questions tell that the text has both
        (see ``_get_paragraph_text_for_aspects`` and ``_build_aspect_questions``).
        A paragraph without additional context is sent as its text alone.

        A refused answer on a paragraph is returned with the other answers:
        the paragraph is then not included in the aspect, i.e. its sentences are
        not asked about for that aspect. If the model refuses to answer
        the question of an aspect on every paragraph, the response cannot be used,
        as nothing is known about the aspect.

        The requests are sent concurrently if a limiter is given, which is the case
        when concurrency is enabled for the extraction, and one after another
        otherwise.

        :param aspects: The aspects to ask about.
        :type aspects: list[_Aspect]
        :param source_paragraphs: The paragraphs that the aspects are extracted
            from.
        :type source_paragraphs: list[_Paragraph]
        :param usage: The usage object to record the requests in.
        :type usage: _ModelUsage
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :param answers_cache: The usable answers of the requests of the call, kept
            for the call to be made again (see ``_get_answers``). Defaults to None.
        :type answers_cache: dict[str, dict[str, _DecisionAnswer]] | None
        :return: The paragraphs that are asked about, in the order of
            the paragraphs of the source, for each of them, the answers of
            the model by the names of the questions, and for each of them,
            the sentences that are asked about, each with the answers of the model
            by the names of the questions.
        :rtype: tuple[list[_Paragraph], list[dict[str, _DecisionAnswer]],
            list[list[tuple[_Sentence, dict[str, _DecisionAnswer]]]]]
        :raises _InvalidDecisionResponseError: If a response cannot be used.
        :raises _RefusedDecisionAnswerError: If the model refused to answer
            the question of an aspect on every paragraph. The calls of the requests
            on the paragraphs are then recorded with the refusal.
        :raises Exception: The error of the first request that failed for good.
        """
        # The same questions are asked on each paragraph with additional context,
        # and the same on each paragraph without it: each of the two variants of
        # the questions is built once, for the first paragraph that it is asked on
        questions_variants: dict[bool, list[_DecisionQuestion]] = {}
        paragraphs: list[_Paragraph] = []
        units: list[_QuestionsOnText] = []
        for position, paragraph in enumerate(source_paragraphs):
            text, has_additional_context = _get_paragraph_text_for_aspects(paragraph)
            if not text:
                continue
            if has_additional_context not in questions_variants:
                questions_variants[has_additional_context] = _build_aspect_questions(
                    aspects, has_additional_context=has_additional_context
                )
            paragraphs.append(paragraph)
            units.append(
                _QuestionsOnText(
                    text=text,
                    questions=questions_variants[has_additional_context],
                    # The paragraphs without text are not asked about, i.e.
                    # the position of the paragraph is the one in the source
                    location=(
                        f" on paragraph {position + 1} of "
                        f"{len(source_paragraphs)} (index {position})"
                    ),
                )
            )
        answers_of_paragraphs = await self._get_answers(
            units=units,
            usage=usage,
            num_retries_failed_request=num_retries_failed_request,
            async_limiter=async_limiter,
            send_concurrently=async_limiter is not None,
            accept_refusals=True,
            answers_cache=answers_cache,
        )
        # A refused answer leaves its paragraph out of the aspect. An aspect that
        # the model answered about on no paragraph is not extracted, though:
        # nothing is known about it, which is not the same as no paragraph belonging
        # to it. litellm also returns a refusal for a question that a model left
        # unanswered, i.e. this is also what a model that does not answer gives.
        for position in range(len(aspects)):
            question_name = _get_aspect_question_name(position)
            if answers_of_paragraphs and all(
                _is_refused_answer(answers, question_name)
                for answers in answers_of_paragraphs
            ):
                # A refusal: each request asked about the paragraphs was refused
                # an answer to the question
                for call in usage.calls:
                    call._refusal = "refusal"
                error = _RefusedDecisionAnswerError(
                    question_name,
                    f"the model refused to answer on every paragraph "
                    f"({len(answers_of_paragraphs)} asked about).",
                )
                error.location = (
                    f" on every paragraph ({len(answers_of_paragraphs)} asked about)"
                )
                raise error
        # The second stage, for the aspects extracted with sentences as their items
        answers_of_sentences = await self._ask_about_sentences(
            aspects=aspects,
            source_paragraphs=source_paragraphs,
            paragraphs=paragraphs,
            answers_of_paragraphs=answers_of_paragraphs,
            usage=usage,
            num_retries_failed_request=num_retries_failed_request,
            async_limiter=async_limiter,
            answers_cache=answers_cache,
        )
        return paragraphs, answers_of_paragraphs, answers_of_sentences

    async def _ask_about_sentences(
        self,
        *,
        aspects: list[_Aspect],
        source_paragraphs: list[_Paragraph],
        paragraphs: list[_Paragraph],
        answers_of_paragraphs: list[dict[str, _DecisionAnswer]],
        usage: _ModelUsage,
        num_retries_failed_request: int = 3,
        async_limiter: AsyncLimiter | None = None,
        answers_cache: dict[str, dict[str, _DecisionAnswer]] | None = None,
    ) -> list[_AnswersOnSentences]:
        """
        Asks the decision model, on each sentence of each paragraph that belongs to
        an aspect with the "sentences" reference depth, whether the sentence
        belongs to the aspect. This is the second stage of the extraction of such
        aspects, after the paragraphs are asked about.

        One request is sent for each sentence of such a paragraph, with
        the sentence and its paragraph as the text (see
        ``_get_sentence_text_for_aspects``), and one question for each aspect with
        the "sentences" reference depth that the paragraph belongs to. A paragraph
        belongs to an aspect with a probability of the decision threshold of
        the aspect or more, or of that of the model if the aspect has none of
        its own: the same threshold applies to the sentences. The name of
        a question follows the position of its aspect among the aspects of
        the call, as the name of its question on a paragraph does.

        No request is sent for an aspect with another reference depth, for
        a paragraph that belongs to no aspect with the "sentences" reference depth,
        for a paragraph without sentences, or for a sentence without a text to
        show to a model. A paragraph that the model refused to answer on for
        an aspect does not belong to that aspect.

        Nor is a request sent for a paragraph whose one sentence is the paragraph
        as a whole (see ``_is_whole_paragraph_sentence``), e.g. a table row
        converted from Docling: it would ask about the same text, and could be
        answered otherwise than the paragraph was. The answers on the paragraph
        to the questions of the aspects that it belongs to are taken as those on
        the sentence.

        A refused answer on a sentence is returned with the other answers:
        the sentence is then not included in the aspect.

        The requests are sent concurrently if a limiter is given, which is the case
        when concurrency is enabled for the extraction, and one after another
        otherwise.

        :param aspects: The aspects of the call, as asked about on the paragraphs.
        :type aspects: list[_Aspect]
        :param source_paragraphs: The paragraphs that the aspects are extracted
            from, by the positions of which the paragraphs are named in
            the message on a response that cannot be used.
        :type source_paragraphs: list[_Paragraph]
        :param paragraphs: The paragraphs that are asked about.
        :type paragraphs: list[_Paragraph]
        :param answers_of_paragraphs: For each of these paragraphs, the answers of
            the model by the names of the questions.
        :type answers_of_paragraphs: list[dict[str, _DecisionAnswer]]
        :param usage: The usage object to record the requests in.
        :type usage: _ModelUsage
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :param answers_cache: The usable answers of the requests of the call, kept
            for the call to be made again (see ``_get_answers``). Defaults to None.
        :type answers_cache: dict[str, dict[str, _DecisionAnswer]] | None
        :return: For each of the paragraphs, the sentences that are asked about,
            in the order of the sentences, each with the answers of the model by
            the names of the questions. Empty for a paragraph none of whose
            sentences is asked about.
        :rtype: list[list[tuple[_Sentence, dict[str, _DecisionAnswer]]]]
        :raises _InvalidDecisionResponseError: If a response cannot be used.
        :raises Exception: The error of the first request that failed for good.
        """
        answers_of_sentences: list[_AnswersOnSentences] = [[] for _ in paragraphs]
        # The decision thresholds of the aspects that are extracted with sentences
        # as their items, by the positions of the aspects in the call
        decision_thresholds = {
            position: _get_decision_threshold(aspect, self.decision_threshold)
            for position, aspect in enumerate(aspects)
            if aspect.reference_depth == "sentences"
        }
        if not decision_thresholds:
            return answers_of_sentences

        # The paragraphs of the source are distinct objects
        source_positions = {
            id(paragraph): position
            for position, paragraph in enumerate(source_paragraphs)
        }
        # The questions on a sentence are built once for all the aspects of
        # the call, with and without additional context in the text, when the first
        # sentence needs them: those of the aspects that the paragraph of
        # a sentence belongs to are taken from them
        questions_variants: dict[bool, list[_DecisionQuestion]] = {}
        # The sentences asked about, each with the position of its paragraph
        sentences: list[tuple[int, _Sentence]] = []
        units: list[_QuestionsOnText] = []
        for paragraph_position, (paragraph, answers) in enumerate(
            zip(paragraphs, answers_of_paragraphs, strict=True)
        ):
            aspect_positions: list[int] = []
            for position, decision_threshold in decision_thresholds.items():
                converted_answer = _convert_aspect_answer(
                    position, answers, decision_threshold=decision_threshold
                )
                # None for a refused answer: the paragraph is not included in
                # the aspect, i.e. its sentences are not asked about for it
                if converted_answer is not None and converted_answer[0]:
                    aspect_positions.append(position)
            if not aspect_positions:
                continue
            if _is_whole_paragraph_sentence(paragraph):
                answers_of_sentences[paragraph_position].append(
                    (
                        paragraph.sentences[0],
                        {
                            name: answers[name]
                            for name in map(_get_aspect_question_name, aspect_positions)
                        },
                    )
                )
                continue
            source_position = source_positions[id(paragraph)]
            for sentence_position, sentence in enumerate(paragraph.sentences):
                text, has_additional_context = _get_sentence_text_for_aspects(
                    sentence, paragraph
                )
                if not text:
                    continue
                if has_additional_context not in questions_variants:
                    questions_variants[has_additional_context] = (
                        _build_aspect_questions(
                            aspects,
                            has_additional_context=has_additional_context,
                            text_unit="sentence",
                        )
                    )
                questions = questions_variants[has_additional_context]
                sentences.append((paragraph_position, sentence))
                units.append(
                    _QuestionsOnText(
                        text=text,
                        questions=[
                            questions[position] for position in aspect_positions
                        ],
                        location=(
                            f" on sentence {sentence_position + 1} of "
                            f"{len(paragraph.sentences)} of paragraph "
                            f"{source_position + 1} of {len(source_paragraphs)} "
                            f"(index {source_position})"
                        ),
                    )
                )
        if not units:
            return answers_of_sentences

        answers_of_units = await self._get_answers(
            units=units,
            usage=usage,
            num_retries_failed_request=num_retries_failed_request,
            async_limiter=async_limiter,
            send_concurrently=async_limiter is not None,
            accept_refusals=True,
            answers_cache=answers_cache,
        )
        for (paragraph_position, sentence), answers in zip(
            sentences, answers_of_units, strict=True
        ):
            answers_of_sentences[paragraph_position].append((sentence, answers))
        return answers_of_sentences

    async def _ask_about_concepts(
        self,
        *,
        concepts: list[_Concept],
        source: _Document | _Aspect,
        source_paragraphs: list[_Paragraph],
        usage: _ModelUsage,
        max_paragraphs_to_analyze_per_call: int = 0,
        num_retries_failed_request: int = 3,
        async_limiter: AsyncLimiter | None = None,
        answers_cache: dict[str, dict[str, _DecisionAnswer]] | None = None,
    ) -> list[dict[str, _DecisionAnswer]]:
        """
        Asks the decision model the questions of the concepts on the text of
        the source.

        By default, the text of the source is sent as a whole, in one request, with
        the questions of all the concepts.

        If the max number of paragraphs to analyze per call is set and the source
        has more paragraphs, the paragraphs are split into consecutive parts of
        at most that many paragraphs, as they are split for an LLM, and one request
        is sent for each part, with the text of the part as it is shown to an LLM
        and the questions of all the concepts. A part that has no such text is not
        asked about. The requests for the parts are sent concurrently if a limiter
        is given, which is the case when concurrency is enabled for the extraction,
        and one after another otherwise.

        :param concepts: The concepts to ask about.
        :type concepts: list[_Concept]
        :param source: The source that the concepts are extracted from.
        :type source: _Document | _Aspect
        :param source_paragraphs: The paragraphs that the concepts are extracted
            from.
        :type source_paragraphs: list[_Paragraph]
        :param usage: The usage object to record the requests in.
        :type usage: _ModelUsage
        :param max_paragraphs_to_analyze_per_call: The maximum number of paragraphs
            whose text is sent in one request. Defaults to 0, in which case
            the text of the source is sent as a whole.
        :type max_paragraphs_to_analyze_per_call: int
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :param answers_cache: The usable answers of the requests of the call, kept
            for the call to be made again (see ``_get_answers``). Defaults to None.
        :type answers_cache: dict[str, dict[str, _DecisionAnswer]] | None
        :return: For each text that the questions are asked on, i.e. for the text
            of the source, or for each of its parts that is asked about, in
            the order of the parts, the answers of the model by the names of
            the questions.
        :rtype: list[dict[str, _DecisionAnswer]]
        :raises _InvalidDecisionResponseError: If a response cannot be used.
        :raises Exception: The error of the first request that failed for good.
        """
        # The same questions are asked on each text
        questions = _build_concept_questions(concepts)
        if 0 < max_paragraphs_to_analyze_per_call < len(source_paragraphs):
            # The same parts as the ones that an LLM is sent in separate calls
            parts: list[list[_Paragraph]] = _chunk_list(
                source_paragraphs, max_paragraphs_to_analyze_per_call
            )
            logger.debug(
                f"Sending the text to decision model `{self.model}` in "
                f"{len(parts)} parts (max {max_paragraphs_to_analyze_per_call} "
                f"paragraphs per part)."
            )
            units: list[_QuestionsOnText] = []
            for position, part in enumerate(parts):
                text = self._get_text_for_concept_extraction(
                    source=source, paragraphs_chunk=part
                )
                # The texts of the paragraphs of a part are joined with blank lines
                if not text.strip():
                    continue
                units.append(
                    _QuestionsOnText(
                        text=text,
                        questions=questions,
                        # The parts without text are not asked about, i.e.
                        # the position of the part is the one in the source
                        location=(
                            f" on part {position + 1} of {len(parts)} of the text"
                        ),
                    )
                )
            send_concurrently = async_limiter is not None
        else:
            # The text of the source as a whole, in one request
            units = [
                _QuestionsOnText(
                    text=self._get_text_for_concept_extraction(
                        source=source, paragraphs_chunk=source_paragraphs
                    ),
                    questions=questions,
                )
            ]
            send_concurrently = False
        return await self._get_answers(
            units=units,
            usage=usage,
            num_retries_failed_request=num_retries_failed_request,
            async_limiter=async_limiter,
            send_concurrently=send_concurrently,
            answers_cache=answers_cache,
        )

    @staticmethod
    def _get_concepts_items(
        concepts: list[_Concept],
        answers_of_texts: list[dict[str, _DecisionAnswer]],
        decision_threshold: float = _DEFAULT_DECISION_THRESHOLD,
    ) -> list[tuple[_Aspect | _Concept, list[_ExtractedItem]]]:
        """
        Converts the answers of the decision model to the questions of the concepts,
        asked on each text, to the items of each concept.

        A concept gets at most one item for each text that the questions are asked
        on, and no item for a text if there is nothing to extract from it, i.e.
        no label of the concept applies. The items of a concept are in the order
        of the texts. ``singular_occurrence`` of a concept is not applied here:
        a concept with a singular occurrence is asked about on one text only, i.e.
        it gets at most one item, as it is rejected before any request is sent
        when the text is to be analyzed in parts.

        A yes/no question of a concept is taken as answered "yes" with a probability
        of the decision threshold of the concept or more, or of the given threshold
        if the concept has none of its own.

        :param concepts: The concepts that the questions were built for.
        :type concepts: list[_Concept]
        :param answers_of_texts: For each text that the questions were asked on,
            the answers of the decision model by the names of the questions.
        :type answers_of_texts: list[dict[str, _DecisionAnswer]]
        :param decision_threshold: The decision threshold of the decision model.
            Defaults to 0.5.
        :type decision_threshold: float
        :return: Each concept with its items, in the order of the concepts.
        :rtype: list[tuple[_Aspect | _Concept, list[_ExtractedItem]]]
        :raises ValueError: If an answer cannot be used, or if a value is not
            accepted by the item of the concept.
        """
        concepts_and_items: list[tuple[_Aspect | _Concept, list[_ExtractedItem]]] = []
        for position, concept in enumerate(concepts):
            extracted_items: list[_ExtractedItem] = []
            for answers in answers_of_texts:
                value_and_probabilities = _convert_concept_answers(
                    concept,
                    position,
                    answers,
                    decision_threshold=_get_decision_threshold(
                        concept, decision_threshold
                    ),
                )
                # None if there is no item to extract, i.e. no label applies
                if value_and_probabilities is not None:
                    value = value_and_probabilities[0]
                    probabilities: ItemProbabilities = value_and_probabilities[1]
                    extracted_items.append(
                        concept._item_class(value=value, probabilities=probabilities)
                    )
            concepts_and_items.append((concept, extracted_items))
        return concepts_and_items

    @staticmethod
    def _get_aspects_items(
        aspects: list[_Aspect],
        paragraphs: list[_Paragraph],
        answers_of_paragraphs: list[dict[str, _DecisionAnswer]],
        decision_threshold: float = _DEFAULT_DECISION_THRESHOLD,
        answers_of_sentences: list[_AnswersOnSentences] | None = None,
    ) -> list[tuple[_Aspect | _Concept, list[_ExtractedItem]]]:
        """
        Converts the answers of the decision model to the questions of the aspects,
        asked on each paragraph of the source, to the items of each aspect.

        A paragraph that belongs to an aspect becomes an item of the aspect, as it
        does when an aspect with the "paragraphs" reference depth is extracted by
        an LLM without justifications: the value of the item is the text of
        the paragraph, the item has the custom data of the paragraph, and
        the paragraph is its reference paragraph. In addition, the item has
        the probabilities of the paragraph belonging and not belonging to the aspect.
        The items of an aspect are in the order of the paragraphs. An aspect that
        no paragraph belongs to has no items.

        A paragraph belongs to an aspect with a probability of the decision
        threshold of the aspect or more, or of the given threshold if the aspect
        has none of its own.

        For an aspect with the "sentences" reference depth, a paragraph that
        belongs to the aspect gives no item itself: each of its sentences that
        belongs to the aspect, by the answers on the sentences and with the same
        decision threshold, becomes an item, as it does when such an aspect is
        extracted by an LLM without justifications. The value of the item is
        the text of the sentence, the item has the custom data of the sentence,
        the paragraph is its reference paragraph and the sentence is its reference
        sentence. In addition, the item has the probabilities of the sentence
        belonging and not belonging to the aspect. The items are in the order of
        the paragraphs and of the sentences in them.

        A paragraph, or a sentence, that the model refused to answer on for
        an aspect gives no item of that aspect, as one that does not belong to it.

        :param aspects: The aspects that the questions were built for.
        :type aspects: list[_Aspect]
        :param paragraphs: The paragraphs that the questions were asked on.
        :type paragraphs: list[_Paragraph]
        :param answers_of_paragraphs: For each paragraph, the answers of the decision
            model by the names of the questions.
        :type answers_of_paragraphs: list[dict[str, _DecisionAnswer]]
        :param decision_threshold: The decision threshold of the decision model.
            Defaults to 0.5.
        :type decision_threshold: float
        :param answers_of_sentences: For each paragraph, the sentences of
            the paragraph that the questions were asked on, each with the answers
            of the decision model by the names of the questions: those of
            the aspects with the "sentences" reference depth that the paragraph
            belongs to. Defaults to None, i.e. no sentence was asked about.
        :type answers_of_sentences:
            list[list[tuple[_Sentence, dict[str, _DecisionAnswer]]]] | None
        :return: Each aspect with its items, in the order of the aspects.
        :rtype: list[tuple[_Aspect | _Concept, list[_ExtractedItem]]]
        :raises ValueError: If an answer on any of the paragraphs, or on any of
            the sentences, cannot be used.
        """
        items_of_aspects: list[list[_ExtractedItem]] = [[] for _ in aspects]
        decision_thresholds = [
            _get_decision_threshold(aspect, decision_threshold) for aspect in aspects
        ]
        if answers_of_sentences is None:
            answers_of_sentences = [[] for _ in paragraphs]
        for paragraph, answers, sentences_answers in zip(
            paragraphs, answers_of_paragraphs, answers_of_sentences, strict=True
        ):
            for position in range(len(aspects)):
                # The answer on each paragraph is checked for each aspect, i.e.
                # also the answer that the paragraph does not belong to the aspect
                converted_answer = _convert_aspect_answer(
                    position,
                    answers,
                    decision_threshold=decision_thresholds[position],
                )
                # None for a refused answer: the paragraph is not included
                if converted_answer is None or not converted_answer[0]:
                    continue
                answer_probabilities = converted_answer[1]
                if aspects[position].reference_depth == "sentences":
                    # Reference depth - sentence
                    # Each extracted item has the reference sentence text as value,
                    # the reference paragraph in the list of reference paragraphs,
                    # and the reference sentence in the list of reference sentences
                    for sentence, sentence_answers in sentences_answers:
                        converted_sentence_answer = _convert_aspect_answer(
                            position,
                            sentence_answers,
                            decision_threshold=decision_thresholds[position],
                        )
                        # None for a refused answer: the sentence is not included
                        if (
                            converted_sentence_answer is None
                            or not converted_sentence_answer[0]
                        ):
                            continue
                        item_probabilities: ItemProbabilities = (
                            converted_sentence_answer[1]
                        )
                        extracted_item = _StringItem(
                            value=sentence.raw_text,
                            # A copy of the custom data of the sentence, whose nested
                            # values the item does not share
                            custom_data=deepcopy(sentence.custom_data),
                            probabilities=item_probabilities,
                        )
                        extracted_item.reference_paragraphs = [paragraph]
                        extracted_item.reference_sentences = [sentence]
                        items_of_aspects[position].append(extracted_item)
                    continue
                # Reference depth - paragraph
                probabilities: ItemProbabilities = answer_probabilities
                extracted_item = _StringItem(
                    value=paragraph.raw_text,
                    # A copy of the custom data of the paragraph, whose nested values
                    # the item does not share
                    custom_data=deepcopy(paragraph.custom_data),
                    probabilities=probabilities,
                )
                extracted_item.reference_paragraphs = [paragraph]
                items_of_aspects[position].append(extracted_item)
        return list(zip(aspects, items_of_aspects, strict=True))

    async def _get_answers(
        self,
        *,
        units: list[_QuestionsOnText],
        usage: _ModelUsage,
        num_retries_failed_request: int = 3,
        async_limiter: AsyncLimiter | None = None,
        send_concurrently: bool = False,
        accept_refusals: bool = False,
        answers_cache: dict[str, dict[str, _DecisionAnswer]] | None = None,
    ) -> list[dict[str, _DecisionAnswer]]:
        """
        Asks the decision model the questions on each of the texts and returns
        its answers on each text.

        The questions on a text are sent in one request. If there are more questions
        on a text than litellm accepts in a request, they are split into several
        requests on the same text, and the answers are merged.

        The requests are sent one after another, or concurrently if specified.
        As soon as a request fails for good, or a response cannot be used, no further
        requests are sent: the answers on all the texts are needed. The requests
        that are already sent by then, which is possible when they are sent
        concurrently, are waited for, so that the tokens they use are counted.

        The requests completed, i.e. those that got a response or failed, are
        recorded in the usage object when this method is left, including when it
        is left for an error or cancelled:

        - first the requests that got a response, in the order of the texts,
          each with the tokens used and the answers, and the requests that
          the provider blocked, or whose answers it blocked, each without
          a response, with the refusal;
        - then the requests that failed, as unfinished calls, with the request
          whose error is raised as the last of them, i.e. as the last call recorded.

        A request that is in flight when this method is cancelled is cancelled
        with it and is not recorded, as its outcome is not known. A request whose
        answers are in the cache given is not sent, and not recorded.

        :param units: The texts to ask the questions on, each with its questions.
        :type units: list[_QuestionsOnText]
        :param usage: The usage object to record the requests in.
        :type usage: _ModelUsage
        :param num_retries_failed_request: Number of retries when a request fails
            with a transient error. Defaults to 3.
        :type num_retries_failed_request: int
        :param async_limiter: An optional aiolimiter.AsyncLimiter instance that
            controls the frequency of the requests, when concurrency is enabled.
        :type async_limiter: AsyncLimiter | None
        :param send_concurrently: Whether to send the requests concurrently, each
            paced by the limiter, if it is given. Defaults to False, in which case
            the requests are sent one after another.
        :type send_concurrently: bool
        :param accept_refusals: Whether a refused answer is accepted, as it is for
            the questions of aspects: the answers returned then include the refused
            ones. Defaults to False, in which case a refused answer makes its
            response one that cannot be used: a ``_RefusedDecisionAnswerError`` is
            raised for it, and the call of its request is recorded with the refusal.
        :type accept_refusals: bool
        :param answers_cache: The usable answers of the requests already sent in
            the call, which the call keeps when it is made again (see
            ``_PartialExtraction``), by the request, as recorded in its call:
            a request with answers here is not sent again, and the usable answers
            of each request that is sent are added, unless one of them is refused.
            Defaults to None, i.e. no answers are kept.
        :type answers_cache: dict[str, dict[str, _DecisionAnswer]] | None
        :return: For each text, the answers by the names of the questions.
        :rtype: list[dict[str, _DecisionAnswer]]
        :raises _InvalidDecisionResponseError: If a response cannot be used, i.e.
            its answers cannot be attributed to the questions of its request, or
            an answer is not usable (e.g. it has no valid probability, or the model
            refused to answer while refusals are not accepted), and no
            request failed. The call of each request is then recorded as finished.
            The error has the position and the location of the text that
            the response is for.
        :raises litellm.ContentPolicyViolationError: If the provider blocked
            a request, or its answers, and no request failed. The call of each
            request is then recorded as finished, that of the blocked request with
            the refusal.
        :raises Exception: The error of the first request that failed for good.
        """
        # The cap is the one of litellm, which rejects a request over it
        max_questions_per_request = _litellm_decision_types.MAX_DECISION_QUESTIONS
        # One request for each text and each chunk of the questions on it, with
        # the position of the text among the texts
        requests: list[tuple[int, list[_DecisionQuestion]]] = [
            (unit_position, questions_chunk)
            for unit_position, unit in enumerate(units)
            for questions_chunk in _chunk_list(
                unit.questions, max_questions_per_request
            )
        ]
        if len(requests) > 1:
            logger.debug(
                f"Sending the questions on {len(units)} text(s) to decision model "
                f"`{self.model}` in {len(requests)} requests "
                f"(max {max_questions_per_request} questions per request)."
            )
        # The questions of each request as they are sent, and each request as it
        # is recorded, without the API credentials, which also identifies it among
        # the requests whose answers are kept
        requests_questions: list[list[dict[str, Any]]] = [
            [_get_request_question(question) for question in questions_chunk]
            for _, questions_chunk in requests
        ]
        requests_json: list[str] = [
            json.dumps(
                {
                    "model": self.model,
                    "input": units[unit_position].text,
                    "questions": request_questions,
                },
                ensure_ascii=False,
            )
            for (unit_position, _), request_questions in zip(
                requests, requests_questions, strict=True
            )
        ]

        # Outcomes of the requests, by the positions of the requests
        finished_calls: dict[int, _ModelCall] = {}
        failed_calls: dict[int, _ModelCall] = {}
        answers_of_requests: dict[int, dict[str, _DecisionAnswer]] = {}
        # Errors in the order they occurred: of the requests that failed, with
        # the positions of the requests, of the requests that the provider blocked,
        # and of the responses that cannot be used
        request_errors: list[tuple[int, Exception]] = []
        blocked_errors: list[Exception] = []
        response_errors: list[_InvalidDecisionResponseError] = []
        # Requests sent, and the tasks of the requests when sent concurrently
        sent_positions: set[int] = set()
        tasks: list[asyncio.Task[None]] = []

        def stop_sending() -> None:
            """
            Cancels the requests that are not sent yet, i.e. those waiting for
            the limiter, when the requests are sent concurrently. The requests
            that are already sent are left to complete.

            :return: None
            :rtype: None
            """
            for position, task in enumerate(tasks):
                if position not in sent_positions:
                    task.cancel()

        async def send_request(position: int) -> None:
            """
            Sends the request at the given position, unless a request has already
            failed or a response cannot be used, and keeps its outcome: the call,
            and the answers or the error.

            :param position: The position of the request.
            :type position: int
            :return: None
            :rtype: None
            """
            if request_errors or blocked_errors or response_errors:
                # The answers on all the texts are needed: the request is pointless
                return
            unit_position, questions_chunk = requests[position]
            request_json = requests_json[position]
            sent_positions.add(position)
            call_obj = _ModelCall(prompt_kwargs={}, prompt=request_json)
            try:
                response = await self._send_decisions_request(
                    text=units[unit_position].text,
                    request_questions=requests_questions[position],
                    num_retries_failed_request=num_retries_failed_request,
                )
                self._record_response(call_obj=call_obj, response=response)
            except litellm.ContentPolicyViolationError as e:  # type: ignore[attr-defined]
                # The provider blocked the request or its answers for their
                # content: a refusal without answers (call finished)
                call_obj._record_blocked(e)
                finished_calls[position] = call_obj
                blocked_errors.append(e)
                stop_sending()
                return
            except Exception as e:  # noqa: BLE001 - any request error is collected
                # The call is unfinished
                failed_calls[position] = call_obj
                request_errors.append((position, e))
                stop_sending()
                return
            finished_calls[position] = call_obj
            try:
                answers = _get_answers_by_question_names(
                    questions_chunk, getattr(response, "answers", None)
                )
                # Each answer is checked for being usable right away, e.g. for
                # having a valid probability, so that no further requests are sent
                # if it is not
                _validate_answers(
                    questions_chunk, answers, accept_refusals=accept_refusals
                )
                answers_of_requests[position] = answers
                # Kept for the call to be made again without this request, unless
                # the model refused to answer one of the questions, which it may
                # answer then
                if answers_cache is not None and not any(
                    answer.refused for answer in answers.values()
                ):
                    answers_cache[request_json] = answers
            except _InvalidDecisionResponseError as e:
                # Where the text that the questions of the request were asked on is
                e.location = units[unit_position].location
                if isinstance(e, _RefusedDecisionAnswerError):
                    call_obj._refusal = "refusal"
                response_errors.append(e)
                stop_sending()

        async def send_paced_request(position: int) -> None:
            """
            Sends the request at the given position when the limiter allows for it,
            if a limiter is given, and right away otherwise. A request whose answers
            are kept from a previous call is not sent, and takes no turn of
            the limiter: its answers are used.

            :param position: The position of the request.
            :type position: int
            :return: None
            :rtype: None
            """
            cached_answers = (
                answers_cache.get(requests_json[position])
                if answers_cache is not None
                else None
            )
            if cached_answers is not None:
                # Answered with usable answers when the call was made before
                answers_of_requests[position] = cached_answers
                return
            if async_limiter:
                async with async_limiter:
                    await send_request(position)
            else:
                await send_request(position)

        try:
            if send_concurrently:
                tasks.extend(
                    asyncio.create_task(send_paced_request(position))
                    for position in range(len(requests))
                )
                # All the requests are waited for. If this call is cancelled,
                # the requests that are not completed are cancelled with it.
                outcomes = await asyncio.gather(*tasks, return_exceptions=True)
                for outcome in outcomes:
                    # The errors of the requests are kept by the requests. A request
                    # that was not sent is cancelled, which is not an error.
                    if isinstance(outcome, BaseException) and not isinstance(
                        outcome, asyncio.CancelledError
                    ):
                        raise outcome
            else:
                for position in range(len(requests)):
                    if request_errors or blocked_errors or response_errors:
                        break
                    await send_paced_request(position)
        finally:
            # Record the calls. The failed request whose error is raised is recorded
            # last, so that the call is told apart as a failed one by its last call.
            raised_error_position = request_errors[0][0] if request_errors else None
            calls = [finished_calls[position] for position in sorted(finished_calls)]
            calls += [
                failed_calls[position]
                for position in sorted(failed_calls)
                if position != raised_error_position
            ]
            if raised_error_position is not None:
                calls.append(failed_calls[raised_error_position])
            self._add_calls_to_usage(usage=usage, calls=calls)

        # A failed request comes first: its unfinished call is recorded
        if request_errors:
            raise request_errors[0][1]
        if blocked_errors:
            raise blocked_errors[0]
        if response_errors:
            raise response_errors[0]

        # Merge the answers of the requests on the same text
        answers_of_texts: list[dict[str, _DecisionAnswer]] = [{} for _ in units]
        for position, (unit_position, _) in enumerate(requests):
            answers_of_texts[unit_position].update(answers_of_requests[position])
        return answers_of_texts

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

    @staticmethod
    def _record_response(*, call_obj: _ModelCall, response: Any) -> None:
        """
        Records the response to a request on the call of the request: the input
        and output tokens, the input tokens read from and written to the prompt
        cache of the model provider, which are a part of the input tokens,
        the answers of the model as JSON, and the time the response is received.

        :param call_obj: The _ModelCall object of the request.
        :type call_obj: _ModelCall
        :param response: The raw response object from the decision model.
        :type response: Any
        :return: None
        :rtype: None
        """
        # Everything is read from the response before the call is recorded
        # as finished, so that a response that cannot be read leaves it unfinished
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        for token_count in (input_tokens, output_tokens):
            if not isinstance(token_count, int) or token_count < 0:
                raise ValueError(
                    f"The response of the decision model reports an invalid number "
                    f"of tokens: {token_count!r}"
                )
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
        call_obj._record_response_timestamp()
        call_obj.response = answers_json

    @staticmethod
    def _add_calls_to_usage(*, usage: _ModelUsage, calls: list[_ModelCall]) -> None:
        """
        Accumulates the token usage of the calls into the given usage object, and
        records the calls in it, in the given order. A call of a request that failed
        has no tokens.

        :param usage: The _ModelUsage object to accumulate token usage and calls.
        :type usage: _ModelUsage
        :param calls: The calls to record.
        :type calls: list[_ModelCall]
        :return: None
        :rtype: None
        """
        for call_obj in calls:
            usage.input += call_obj._input_tokens
            usage.output += call_obj._output_tokens
            usage.cache_read += call_obj._cache_read_tokens
            usage.cache_write += call_obj._cache_write_tokens
            usage.calls.append(call_obj)
