# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for decision models.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json
import logging as stdlib_logging
import os
import re
import sys
import types
import warnings
from collections.abc import Callable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter
from docling_core.types.doc import DoclingDocument
from litellm.types.decisions import (
    MAX_DECISION_QUESTIONS,
    OpenAIChoiceAnswer,
    OpenAIChoiceProbability,
    OpenAIDecisionInputTokensDetails,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
    OpenAIRefusalAnswer,
    OpenAIScoreAnswer,
    OpenAIScoreProbability,
)
from pydantic import ValidationError

from contextgem.internal.base import decision_models as decision_models_module
from contextgem.internal.base import documents as documents_module
from contextgem.internal.base import models as models_module
from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.decision_models import _DocumentDecisionModel
from contextgem.internal.base.llms import _DocumentLLM
from contextgem.internal.base.models import _COST_QUANT
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.data_models import _ModelCost, _ModelUsage
from contextgem.internal.decisions import (
    _build_aspect_questions,
    _build_concept_questions,
    _convert_aspect_answer,
    _convert_concept_answers,
    _DecisionAnswer,
    _DecisionOption,
    _DecisionQuestion,
    _get_decision_threshold,
    _get_paragraph_text_for_aspects,
    _get_sentence_text_for_aspects,
    _is_whole_paragraph_sentence,
    _validate_answers,
)
from contextgem.internal.exceptions import (
    ModelAPIError,
    ModelExtractionError,
    ModelRefusalError,
    _InvalidDecisionResponseError,
)
from contextgem.internal.items import (
    _BooleanItem,
    _IntegerItem,
    _LabelItem,
    _StringItem,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    JsonObjectConcept,
    LabelConcept,
    ModelPricing,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    Sentence,
    StringConcept,
    register_tool,
)
from tests.base import BaseTest
from tests.conftest import VCR_DUMMY_ENDPOINT_PREFIX, VCR_REDACTION_MARKER
from tests.utils import REDACTED_VALUE, capture_logs


# The module object that the decision model calls the Decisions API through,
# i.e. the one to patch the API calls on
litellm = decision_models_module.litellm

# Environment variables that litellm reads the API keys of the providers of
# decision models from, when no key is passed
_DECISIONS_API_KEY_ENV_VARS = [
    "TYPESAFE_API_KEY",
    "STRANDS_DECIDER_API_KEY",
    "PERPLEXITYAI_API_KEY",
    "PERPLEXITY_API_KEY",
    "OPENROUTER_API_KEY",
    "CLOUDFLARE_API_KEY",
    "OPENAI_API_KEY",
    "HOSTED_VLLM_API_KEY",
    "AZURE_AI_API_KEY",
]

# Models used in the tests. No requests are sent to them.
_DECISION_MODEL = "typesafe/jev-latest"
_OTHER_DECISION_MODEL = "typesafe/jev-1.13.0"
_LLM_MODEL = "openai/gpt-6-luna"
_OTHER_LLM_MODEL = "openai/gpt-5.6-luna"

# Decision model deployed in Microsoft Foundry that the recorded tests send their
# requests to: a deployment of Microsoft-Decision-1, identified by the name that
# the deployment was given
_FOUNDRY_DECISION_MODEL = "azure_ai/ms-decision-model"

# Prices of 1M input tokens of the decision models whose costs are calculated
# automatically in the recorded tests, as the providers list them for requests
# of the recorded size: the costs are checked against these prices, not against
# the pricing data that they are calculated with
_RECORDED_AUTO_PRICED_INPUT_PRICES = {"openai/gpt-6-luna": "0.10"}

# Text of the document used in the tests, and the texts expected to be sent
# to a decision model for it: the cleaned text of the document for the concepts
# of the document, and the texts of the paragraphs for the concepts of an aspect
_DOCUMENT_RAW_TEXT = (
    "Non-disclosure  agreement\n"
    "The parties agree as follows.\n\n\n\n"
    "Either party may terminate this agreement with a written notice."
)
_DOCUMENT_TEXT_SENT = (
    "Non-disclosure agreement\n"
    "The parties agree as follows.\n\n"
    "Either party may terminate this agreement with a written notice."
)
_ASPECT_PARAGRAPHS_TEXT_SENT = (
    "The parties agree as follows.\n\n"
    "Either party may terminate this agreement with a written notice."
)

# Texts of the paragraphs of the document used in the tests of the extraction of
# aspects, and the texts expected to be sent to a decision model for them: the text
# of each paragraph, in a request of its own, with the additional context of
# the paragraph if it has any, which only the third paragraph does (see
# `_create_aspects_document`). The cleaned texts of the paragraphs are what
# the text sent for the concepts is made of, which has no additional context.
_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS = [
    "Non-disclosure  agreement",
    "The receiving party shall keep the information confidential.",
    "Either party may terminate this agreement with a written notice of 30 days.",
    "The liability of each party is limited to the fees paid.",
    "A notice of termination does not release a party from confidentiality.",
]
_ASPECTS_PARAGRAPH_TEXTS_CLEANED = [
    "Non-disclosure agreement",
    *_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS[1:],
]
_ASPECTS_PARAGRAPH_TEXTS_SENT = [
    *_ASPECTS_PARAGRAPH_TEXTS_CLEANED[:2],
    (
        "Paragraph: Either party may terminate this agreement with a written notice "
        "of 30 days.\n\nAdditional context of the paragraph: Section 3. Termination"
    ),
    *_ASPECTS_PARAGRAPH_TEXTS_CLEANED[3:],
]

# Texts of the documents of the tests whose requests are sent to the APIs of
# decision models and recorded (the tests marked with `vcr`): short, as they are
# paid for when recorded, and unambiguous on what is extracted from them
_RECORDED_CONCEPTS_DOCUMENT_TEXT = (
    "NON-DISCLOSURE AGREEMENT\n\n"
    "This Non-Disclosure Agreement is made between Alpha Ltd and Beta Inc. "
    "Each party shall keep the confidential information of the other party "
    "secret and shall not disclose it to any third party.\n\n"
    "The parties rate the sensitivity of the confidential information as 4 "
    "on a scale from 1 to 5.\n\n"
    "This agreement is governed by the laws of Norway."
)
_RECORDED_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS = [
    "SERVICES AGREEMENT",
    "This agreement is made between Alpha Ltd, the client, and Beta Inc, the provider.",
    "The provider shall deliver consulting services to the client.",
    "The client shall pay the provider a fee of 5,000 euros every month.",
    "Each invoice of the provider is payable within 30 days of the invoice date.",
    "Either party may terminate this agreement by giving a written notice of 60 days.",
    "This agreement is governed by the laws of Norway.",
]
# The sentences of each paragraph of the document from which aspects with
# sentences as their items are extracted: the second paragraph has a sentence on
# a fee and one on another matter, and the third paragraph is about a termination
# by its additional context only
_RECORDED_SENTENCES_DOCUMENT_SENTENCE_TEXTS = [
    [
        "This agreement is made between Alpha Ltd, the client, and Beta Inc, the provider."
    ],
    [
        "The client shall pay the provider a fee of 5,000 euros every month.",
        "The provider shall send the client a report on its work every week.",
    ],
    ["A written notice of 60 days is required."],
    [
        "Either party may terminate this agreement if the other party is in material breach."
    ],
    ["This agreement is governed by the laws of Norway."],
]
_RECORDED_SENTENCES_DOCUMENT_ADDITIONAL_CONTEXT = (
    "Section 5. Termination of the agreement by either party"
)


def _get_foundry_api_key_and_base() -> tuple[str, str]:
    """
    Returns the API key and the endpoint of the Microsoft Foundry resource that
    the recorded tests send their requests to, as set in the environment.

    If either is not set, e.g. in a ``.env`` file that has the keys of other APIs
    only, the dummy one is returned instead, which is enough to replay the requests
    from the cassettes: litellm rejects a request without a key or an endpoint
    before it is sent, and the host of a resource is replaced with the dummy one
    in the cassettes.

    :return: The API key and the endpoint.
    :rtype: tuple[str, str]
    """
    return (
        os.getenv("CONTEXTGEM_AZURE_AI_API_KEY") or VCR_REDACTION_MARKER,
        os.getenv("CONTEXTGEM_AZURE_AI_API_BASE")
        or VCR_DUMMY_ENDPOINT_PREFIX.rstrip("/"),
    )


def _create_decisions_response(
    questions: list[dict[str, Any]],
    answers: dict[str, Any],
    input_tokens: int = 100,
    output_tokens: int = 0,
    cached_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> OpenAIDecisionResponse:
    """
    Creates the response that litellm returns for a request to its Decisions API
    made in the OpenAI format: one answer for each question, in the order of
    the questions.

    :param questions: Questions of the request.
    :type questions: list[dict[str, Any]]
    :param answers: What to answer, by the names of the questions: the probability
        of "yes" for a predicate, the probabilities of the options by their labels
        for a "choice" or a "score" question, "refusal" for a refused answer, or
        an answer object of litellm, which is returned as it is.
    :type answers: dict[str, Any]
    :param input_tokens: Input tokens to report. Defaults to 100.
    :type input_tokens: int
    :param output_tokens: Output tokens to report. Defaults to 0.
    :type output_tokens: int
    :param cached_tokens: Input tokens read from the cache to report. Defaults to 0.
    :type cached_tokens: int
    :param cache_write_tokens: Input tokens written to the cache to report.
        Defaults to 0.
    :type cache_write_tokens: int
    :return: Response of litellm.
    :rtype: OpenAIDecisionResponse
    """
    response_answers: list[Any] = []
    for question in questions:
        name = question["name"]
        answer = answers[name]
        if not isinstance(answer, int | float | dict | str):
            response_answers.append(answer)
        elif answer == "refusal":
            response_answers.append(OpenAIRefusalAnswer(name=name))
        elif question["type"] == "predicate":
            response_answers.append(
                OpenAIPredicateAnswer(name=name, probability=answer)
            )
        elif question["type"] == "choice":
            response_answers.append(
                OpenAIChoiceAnswer(
                    name=name,
                    choice=max(answer, key=answer.__getitem__),
                    confidence=max(answer.values()),
                    probabilities=tuple(
                        OpenAIChoiceProbability(value=label, probability=probability)
                        for label, probability in answer.items()
                    ),
                )
            )
        else:
            response_answers.append(
                OpenAIScoreAnswer(
                    name=name,
                    score=0.0,
                    confidence=max(answer.values()),
                    probabilities=tuple(
                        OpenAIScoreProbability(
                            value=position, label=label, probability=probability
                        )
                        for position, (label, probability) in enumerate(answer.items())
                    ),
                )
            )
    return OpenAIDecisionResponse(
        model="jev-1.13.0",
        answers=tuple(response_answers),
        usage=OpenAIDecisionUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_tokens_details=OpenAIDecisionInputTokensDetails(
                cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens
            ),
        ),
    )


def _answer_all_questions(request: dict[str, Any]) -> OpenAIDecisionResponse:
    """
    Creates the response that answers "yes" to each predicate, with a probability
    of 0.9, and selects the first option of each of the other questions of
    the request, with a probability of 1.

    :param request: Keyword arguments of the request.
    :type request: dict[str, Any]
    :return: Response of litellm.
    :rtype: OpenAIDecisionResponse
    """
    answers: dict[str, Any] = {}
    for question in request["questions"]:
        options = question.get("choices") or question.get("levels")
        if options is None:
            answers[question["name"]] = 0.9
        else:
            labels = [option.get("value", option.get("label")) for option in options]
            answers[question["name"]] = {
                label: 1.0 if position == 0 else 0.0
                for position, label in enumerate(labels)
            }
    return _create_decisions_response(request["questions"], answers)


class _DecisionsAPIStub:
    """
    Simulates the Decisions API of litellm (``litellm.adecisions``), with
    the outcomes of the requests to each model taken from the queue of that model.
    The last outcome on the queue is repeated.

    An outcome is an exception to raise, a function that creates the response for
    the request (the response itself, or an awaitable that gives it), or the answers
    to return by the names of the questions (see :func:`_create_decisions_response`).

    :ivar requests: The requests received, as the keyword arguments of the calls.
    :vartype requests: list[dict[str, Any]]
    :ivar usage_kwargs: Token counts to report in the responses created from answers.
    :vartype usage_kwargs: dict[str, int]
    """

    def __init__(self, outcomes: dict[str, list[Any]], **usage_kwargs: int):
        self.outcomes = outcomes
        self.usage_kwargs = usage_kwargs
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        """
        Records the request and returns the next outcome for its model.

        :param kwargs: Keyword arguments of ``litellm.adecisions``.
        :type kwargs: Any
        :return: Response of litellm.
        :rtype: Any
        """
        self.requests.append(kwargs)
        queue = self.outcomes[kwargs["model"]]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            response = outcome(kwargs)
            return await response if inspect.isawaitable(response) else response
        return _create_decisions_response(
            kwargs["questions"], outcome, **self.usage_kwargs
        )

    def requested_models(self) -> list[str]:
        """
        Returns the models in the order the requests were sent to them.

        :return: Models of the requests.
        :rtype: list[str]
        """
        return [request["model"] for request in self.requests]


class _CompletionsAPIStub:
    """
    Simulates the chat completions API of litellm (``litellm.acompletion``), with
    the outcomes of the requests to each model taken from the queue of that model.
    The last outcome on the queue is repeated.

    An outcome is an exception to raise, the text of the response to return, or
    a completion object to return as it is.

    :ivar requests: The requests received, as the keyword arguments of the calls.
    :vartype requests: list[dict[str, Any]]
    """

    def __init__(self, outcomes: dict[str, list[Any]]):
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> Any:
        """
        Records the request and returns the next outcome for its model.

        :param kwargs: Keyword arguments of ``litellm.acompletion``.
        :type kwargs: Any
        :return: Completion object, with the attributes that the framework reads.
        :rtype: Any
        """
        self.requests.append(kwargs)
        queue = self.outcomes[kwargs["model"]]
        outcome = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(outcome, Exception):
            raise outcome
        if not isinstance(outcome, str):
            return outcome
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content=outcome))
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )

    def requested_models(self) -> list[str]:
        """
        Returns the models in the order the requests were sent to them.

        :return: Models of the requests.
        :rtype: list[str]
        """
        return [request["model"] for request in self.requests]


def _create_api_error(error_class: Any, model: str = _DECISION_MODEL) -> Exception:
    """
    Creates a model API error of the specified litellm exception class.

    :param error_class: Exception class of litellm.
    :type error_class: Any
    :param model: Model that the error is reported for. Defaults to
        the decision model used in the tests.
    :type model: str
    :return: Model API error.
    :rtype: Exception
    """
    provider, _, model_name = model.partition("/")
    return error_class(
        message="Simulated API error", model=model_name, llm_provider=provider
    )


def _create_decision_model(
    model: str = _DECISION_MODEL, **kwargs: Any
) -> DocumentDecisionModel:
    """
    Creates a decision model with a dummy API key.

    :param model: Model identifier. Defaults to the decision model used in the tests.
    :type model: str
    :param kwargs: Other params of the decision model.
    :type kwargs: Any
    :return: Decision model.
    :rtype: DocumentDecisionModel
    """
    return DocumentDecisionModel(model=model, api_key="test_api_key", **kwargs)


def _create_llm(model: str = _LLM_MODEL, **kwargs: Any) -> DocumentLLM:
    """
    Creates an LLM with a dummy API key.

    :param model: Model identifier. Defaults to the LLM used in the tests.
    :type model: str
    :param kwargs: Other params of the LLM.
    :type kwargs: Any
    :return: LLM.
    :rtype: DocumentLLM
    """
    return DocumentLLM(model=model, api_key="test_api_key", **kwargs)


def _create_document(
    concepts: list[Any] | None = None, aspects: list[Aspect] | None = None
) -> Document:
    """
    Creates a document with three paragraphs, with the given concepts and aspects
    assigned to it.

    :param concepts: Concepts of the document. Defaults to None.
    :type concepts: list[Any] | None
    :param aspects: Aspects of the document. Defaults to None.
    :type aspects: list[Aspect] | None
    :return: Document.
    :rtype: Document
    """
    document = Document(raw_text=_DOCUMENT_RAW_TEXT)
    assert [paragraph.raw_text for paragraph in document.paragraphs] == [
        "Non-disclosure  agreement",
        "The parties agree as follows.",
        "Either party may terminate this agreement with a written notice.",
    ]
    if concepts:
        document.concepts = concepts
    if aspects:
        document.aspects = aspects
    return document


def _create_aspects_document(
    aspects: list[Aspect] | None = None, concepts: list[Any] | None = None
) -> Document:
    """
    Creates a document with five paragraphs, with the given aspects and concepts
    assigned to it. The third paragraph has additional context and custom data.

    :param aspects: Aspects of the document. Defaults to None.
    :type aspects: list[Aspect] | None
    :param concepts: Concepts of the document. Defaults to None.
    :type concepts: list[Any] | None
    :return: Document.
    :rtype: Document
    """
    paragraphs: list[_Paragraph] = [
        Paragraph(raw_text=text) for text in _ASPECTS_DOCUMENT_PARAGRAPH_TEXTS
    ]
    paragraphs[2] = Paragraph(
        raw_text=_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS[2],
        additional_context="Section 3. Termination",
        custom_data={"page": 2},
    )
    document = Document(paragraphs=paragraphs)
    if aspects:
        document.aspects = aspects
    if concepts:
        document.concepts = concepts
    return document


def _create_aspects_responder(
    yes_probabilities: dict[str, dict[int, float]],
    default_probability: float = 0.1,
    **usage_kwargs: int,
) -> Callable[[dict[str, Any]], OpenAIDecisionResponse]:
    """
    Creates the function that responds to a request with the questions of aspects,
    asked on one paragraph of the document created by
    :func:`_create_aspects_document`.

    :param yes_probabilities: The probability that a paragraph belongs to an aspect,
        by the name of the aspect and by the position of the paragraph in
        the document.
    :type yes_probabilities: dict[str, dict[int, float]]
    :param default_probability: The probability for the aspects and paragraphs
        that are not given. Defaults to 0.1.
    :type default_probability: float
    :param usage_kwargs: Token counts to report in the responses.
    :type usage_kwargs: int
    :return: Function that creates the response for a request.
    :rtype: Callable[[dict[str, Any]], OpenAIDecisionResponse]
    """

    def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
        """
        Creates the response for the request.

        :param request: Keyword arguments of the request.
        :type request: dict[str, Any]
        :return: Response of litellm.
        :rtype: OpenAIDecisionResponse
        """
        paragraph_position = _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
        answers: dict[str, Any] = {}
        for question in request["questions"]:
            aspect_name_match = re.search(
                r"^Aspect name: (.+)$", question["instructions"], flags=re.MULTILINE
            )
            assert aspect_name_match is not None, question
            answers[question["name"]] = yes_probabilities.get(
                aspect_name_match.group(1), {}
            ).get(paragraph_position, default_probability)
        return _create_decisions_response(request["questions"], answers, **usage_kwargs)

    return respond


def _set_aspect_as_extracted(document: Document, aspect: _Aspect) -> None:
    """
    Sets the aspect as extracted from the document, with the last two paragraphs
    of the document as its extracted items, as if extracted by a model.

    :param document: Document that the aspect is assigned to.
    :type document: Document
    :param aspect: Aspect of the document.
    :type aspect: _Aspect
    """
    paragraphs = document.paragraphs[1:]
    items = []
    for paragraph in paragraphs:
        item = _StringItem(value=paragraph.raw_text)
        item.reference_paragraphs = [paragraph]
        items.append(item)
    aspect.extracted_items = items  # ty: ignore[invalid-assignment]
    aspect.reference_paragraphs = paragraphs
    aspect._is_processed = True


# Sentences of the paragraphs of the document used in the tests of the extraction
# of aspects with sentences as the extracted items. The text of a paragraph is its
# sentences joined with a space.
_SENTENCES_DOCUMENT_SENTENCE_TEXTS = [
    ["Services  agreement"],
    [
        "Either party may terminate this agreement with a notice of 30 days.",
        "The notice shall be given in writing.",
        "The fees paid are not refunded.",
    ],
    ["The client shall pay the fees every month.", "Late payments bear interest."],
    [
        "The provider may suspend the services.",
        "Termination does not affect the accrued rights.",
    ],
    ["This agreement is governed by the laws of Norway."],
]
# The texts expected to be sent to a decision model for the paragraphs of that
# document, each in a request of its own: only the second paragraph has
# additional context
_SENTENCES_PARAGRAPH_TEXTS_SENT = [
    "Services agreement",
    (
        "Paragraph: Either party may terminate this agreement with a notice of 30 days. "
        "The notice shall be given in writing. The fees paid are not refunded.\n\n"
        "Additional context of the paragraph: Section 3. Termination"
    ),
    "The client shall pay the fees every month. Late payments bear interest.",
    (
        "The provider may suspend the services. Termination does not affect the accrued "
        "rights."
    ),
    "This agreement is governed by the laws of Norway.",
]
# ... and for the sentences of the paragraphs, each in a request of its own, by
# the positions of the paragraph and of the sentence in it: the sentence, its
# paragraph, and the additional context of the sentence and of the paragraph,
# where they are set. Additional context that a sentence shares with its paragraph,
# as the third sentence of the second paragraph does, is given once.
_SENTENCES_TEXTS_SENT = {
    (0, 0): "Sentence: Services agreement\n\n"
    "Paragraph that the sentence is from: Services agreement",
    (1, 0): "Sentence: Either party may terminate this agreement with a notice of "
    "30 days.\n\n"
    "Paragraph that the sentence is from: Either party may terminate this agreement "
    "with a notice of 30 days. The notice shall be given in writing. The fees paid "
    "are not refunded.\n\n"
    "Additional context of the paragraph: Section 3. Termination",
    (1, 1): "Sentence: The notice shall be given in writing.\n\n"
    "Paragraph that the sentence is from: Either party may terminate this agreement "
    "with a notice of 30 days. The notice shall be given in writing. The fees paid "
    "are not refunded.\n\n"
    "Additional context of the sentence: List item 2 of 3\n\n"
    "Additional context of the paragraph: Section 3. Termination",
    (1, 2): "Sentence: The fees paid are not refunded.\n\n"
    "Paragraph that the sentence is from: Either party may terminate this agreement "
    "with a notice of 30 days. The notice shall be given in writing. The fees paid "
    "are not refunded.\n\n"
    "Additional context of the paragraph: Section 3. Termination",
    (2, 0): "Sentence: The client shall pay the fees every month.\n\n"
    "Paragraph that the sentence is from: The client shall pay the fees every "
    "month. Late payments bear interest.",
    (2, 1): "Sentence: Late payments bear interest.\n\n"
    "Paragraph that the sentence is from: The client shall pay the fees every "
    "month. Late payments bear interest.",
    (3, 0): "Sentence: The provider may suspend the services.\n\n"
    "Paragraph that the sentence is from: The provider may suspend the services. "
    "Termination does not affect the accrued rights.",
    (3, 1): "Sentence: Termination does not affect the accrued rights.\n\n"
    "Paragraph that the sentence is from: The provider may suspend the services. "
    "Termination does not affect the accrued rights.\n\n"
    "Additional context of the sentence: Footnote 4",
    (4, 0): "Sentence: This agreement is governed by the laws of Norway.\n\n"
    "Paragraph that the sentence is from: This agreement is governed by the laws "
    "of Norway.",
}


def _create_sentences_document(aspects: list[Aspect] | None = None) -> Document:
    """
    Creates a document with five paragraphs that have their sentences, nine in
    total, with the given aspects assigned to it, so that no sentences are to be
    segmented when the aspects are extracted with sentences as their items.

    The second paragraph has additional context and custom data. Its second
    sentence has additional context and custom data of its own, and its third
    sentence has the additional context of the paragraph, as a sentence segmented
    from the paragraph has. The second sentence of the fourth paragraph, which has
    no additional context, has additional context of its own.

    :param aspects: Aspects of the document. Defaults to None.
    :type aspects: list[Aspect] | None
    :return: Document.
    :rtype: Document
    """
    paragraph_kwargs: dict[int, dict[str, Any]] = {
        1: {"additional_context": "Section 3. Termination", "custom_data": {"page": 2}}
    }
    sentence_kwargs: dict[tuple[int, int], dict[str, Any]] = {
        (1, 1): {
            "additional_context": "List item  2 of 3",
            "custom_data": {"line": 7},
        },
        (1, 2): {"additional_context": "Section 3. Termination"},
        (3, 1): {"additional_context": "Footnote 4"},
    }
    paragraphs: list[_Paragraph] = [
        Paragraph(
            raw_text=" ".join(sentence_texts),
            sentences=[
                Sentence(
                    raw_text=sentence_text,
                    **sentence_kwargs.get((paragraph_position, sentence_position), {}),
                )
                for sentence_position, sentence_text in enumerate(sentence_texts)
            ],
            **paragraph_kwargs.get(paragraph_position, {}),
        )
        for paragraph_position, sentence_texts in enumerate(
            _SENTENCES_DOCUMENT_SENTENCE_TEXTS
        )
    ]
    document = Document(paragraphs=paragraphs)
    assert len(document.sentences) == 9
    if aspects:
        document.aspects = aspects
    return document


def _get_sentences_request_position(request: dict[str, Any]) -> int | tuple[int, int]:
    """
    Returns what a request with the questions of aspects is for, among
    the paragraphs and the sentences of the document created by
    :func:`_create_sentences_document`.

    :param request: Keyword arguments of the request.
    :type request: dict[str, Any]
    :return: The position of the paragraph, for a request for a paragraph, or
        the positions of the paragraph and of the sentence in it, for a request
        for a sentence.
    :rtype: int | tuple[int, int]
    """
    text = request["input"]
    if text in _SENTENCES_PARAGRAPH_TEXTS_SENT:
        return _SENTENCES_PARAGRAPH_TEXTS_SENT.index(text)
    (position,) = [
        position
        for position, text_sent in _SENTENCES_TEXTS_SENT.items()
        if text_sent == text
    ]
    return position


def _create_sentences_responder(
    yes_probabilities: dict[str, dict[int | tuple[int, int], float]],
    default_probability: float = 0.1,
    **usage_kwargs: int,
) -> Callable[[dict[str, Any]], OpenAIDecisionResponse]:
    """
    Creates the function that responds to a request with the questions of aspects,
    asked on one paragraph, or on one sentence, of the document created by
    :func:`_create_sentences_document`.

    :param yes_probabilities: The probability that a paragraph or a sentence
        belongs to an aspect, by the name of the aspect, and by the position of
        the paragraph in the document, or by the positions of the paragraph and
        of the sentence in it.
    :type yes_probabilities: dict[str, dict[int | tuple[int, int], float]]
    :param default_probability: The probability for the aspects, paragraphs and
        sentences that are not given. Defaults to 0.1.
    :type default_probability: float
    :param usage_kwargs: Token counts to report in the responses.
    :type usage_kwargs: int
    :return: Function that creates the response for a request.
    :rtype: Callable[[dict[str, Any]], OpenAIDecisionResponse]
    """

    def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
        """
        Creates the response for the request.

        :param request: Keyword arguments of the request.
        :type request: dict[str, Any]
        :return: Response of litellm.
        :rtype: OpenAIDecisionResponse
        """
        position = _get_sentences_request_position(request)
        # The questions on a sentence tell that a sentence is what is judged
        text_unit = "paragraph" if isinstance(position, int) else "sentence"
        answers: dict[str, Any] = {}
        for question in request["questions"]:
            aspect_name_match = re.search(
                r"^Aspect name: (.+)$", question["instructions"], flags=re.MULTILINE
            )
            assert aspect_name_match is not None, question
            assert f"whether the {text_unit} is relevant" in question["instructions"]
            answers[question["name"]] = yes_probabilities.get(
                aspect_name_match.group(1), {}
            ).get(position, default_probability)
        return _create_decisions_response(request["questions"], answers, **usage_kwargs)

    return respond


def _predicate_answer(
    name: str, probability: Any = None, **kwargs: Any
) -> _DecisionAnswer:
    """
    Creates the answer to a predicate.

    :param name: Name of the question.
    :type name: str
    :param probability: Probability of "yes", of any type, so that an invalid
        probability can be given as well. Defaults to None.
    :type probability: Any
    :param kwargs: Other params of the answer.
    :type kwargs: Any
    :return: Answer to the predicate.
    :rtype: _DecisionAnswer
    """
    return _DecisionAnswer(
        name=name, kind="predicate", probability=probability, **kwargs
    )


def _choice_answer(
    name: str, option_probabilities: Any = None, **kwargs: Any
) -> _DecisionAnswer:
    """
    Creates the answer to a "choice" question.

    :param name: Name of the question.
    :type name: str
    :param option_probabilities: Probabilities of the options, of any type, so that
        invalid probabilities can be given as well. Defaults to None.
    :type option_probabilities: Any
    :param kwargs: Other params of the answer.
    :type kwargs: Any
    :return: Answer to the "choice" question.
    :rtype: _DecisionAnswer
    """
    return _DecisionAnswer(
        name=name, kind="choice", option_probabilities=option_probabilities, **kwargs
    )


def _score_answer(
    name: str, option_probabilities: Any = None, **kwargs: Any
) -> _DecisionAnswer:
    """
    Creates the answer to a "score" question.

    :param name: Name of the question.
    :type name: str
    :param option_probabilities: Probabilities of the levels, of any type, so that
        invalid probabilities can be given as well. Defaults to None.
    :type option_probabilities: Any
    :param kwargs: Other params of the answer.
    :type kwargs: Any
    :return: Answer to the "score" question.
    :rtype: _DecisionAnswer
    """
    return _DecisionAnswer(
        name=name, kind="score", option_probabilities=option_probabilities, **kwargs
    )


def _check_invalid_answer(
    convert: Callable[[dict[str, _DecisionAnswer]], Any],
    answers: dict[str, _DecisionAnswer],
    question_name: str,
    reason_match: str,
) -> None:
    """
    Checks that the conversion of the answers is rejected as an invalid response
    of the decision model, with an error that names the question.

    :param convert: Function that converts the answers.
    :type convert: Callable[[dict[str, _DecisionAnswer]], Any]
    :param answers: Answers by the names of the questions.
    :type answers: dict[str, _DecisionAnswer]
    :param question_name: Name of the question whose answer is invalid.
    :type question_name: str
    :param reason_match: Pattern that the reason given in the error is to match.
    :type reason_match: str
    """
    with pytest.raises(_InvalidDecisionResponseError, match=reason_match) as exc_info:
        convert(answers)
    assert isinstance(exc_info.value, ValueError)
    assert exc_info.value.question_name == question_name
    assert str(exc_info.value).startswith(
        f"Invalid answer to the decision question `{question_name}`: "
    )


# Values that are not a probability, i.e. not a number within [0, 1]
_INVALID_PROBABILITIES: list[Any] = [
    -0.01,
    1.01,
    2,
    float("nan"),
    float("inf"),
    "0.9",
    True,
]


class TestDecisionModels(BaseTest):
    """
    Tests for decision models: the questions built for concepts and aspects,
    the conversion of the answers to the values of extracted items with their
    probabilities, and the extraction of aspects and concepts with a decision model,
    on its own, in a model group and in a fallback chain.

    In the tests that are not marked with ``vcr``, no requests are made, i.e. no
    cassettes are required: the API calls are stubbed in each test that makes them,
    and an API call that is not stubbed fails the test (see the fixture below).
    The tests marked with ``vcr`` send their requests to the APIs of the models,
    which are recorded to and replayed from cassettes as in the other test classes.
    """

    @pytest.fixture(autouse=True)
    def fail_on_unstubbed_model_api_calls(
        self, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ):
        """
        Makes any call to a model API that is not stubbed by the test fail the test,
        instead of sending a request.

        Does not apply to the tests marked with ``vcr``, and to them only: their
        requests are to reach the cassette (or, when a cassette is recorded,
        the API), through the calls of litellm as they are set up for such tests
        (see ``vcr_compatible_acompletion`` in ``conftest.py``), with the API keys
        taken from the environment.

        A test stubs the API calls that it expects, on the same module object that
        the framework calls them through. Without a stub, the calls of litellm to
        its Decisions API, in either request format, and to its chat completions
        API are replaced with a failure here. The function of litellm that
        prepares each request to its Decisions API is replaced as well, in each
        module that calls it, i.e. a request cannot be prepared even if
        the framework gets to litellm in another way than through the calls
        replaced.

        The failure is not an ``Exception``, so that it is not handled as an error
        of a model API (retried, passed to a fallback model, or converted into
        a warning), and the calls are also checked after the test, in case
        the failure is suppressed.

        The API keys of the providers of decision models are removed from
        the environment for the time of the test: litellm reads them from there
        when no key is passed.
        """
        if request.node.get_closest_marker("vcr") is not None:
            yield
            return

        unstubbed_calls: list[str] = []

        def create_failure(name: str) -> Callable[..., Any]:
            """
            Creates the replacement of an API call that fails the test.

            :param name: Name of the call replaced, for the failure message.
            :type name: str
            :return: Function that records the call and fails the test.
            :rtype: Callable[..., Any]
            """

            def fail(*args: Any, **kwargs: Any) -> Any:
                """
                Records the call and fails the test.

                :param args: Arguments of the call (not used).
                :type args: Any
                :param kwargs: Keyword arguments of the call (not used).
                :type kwargs: Any
                """
                unstubbed_calls.append(name)
                # The failure of pytest is not an `Exception`
                raise pytest.fail.Exception(
                    f"`{name}` is called without being stubbed by the test."
                )

            return fail

        for env_var_name in _DECISIONS_API_KEY_ENV_VARS:
            monkeypatch.delenv(env_var_name, raising=False)
        for name in [
            "adecisions",
            "decisions",
            "asystemone",
            "systemone",
            "acompletion",
            "completion",
        ]:
            monkeypatch.setattr(litellm, name, create_failure(f"litellm.{name}"))
        # `litellm.decisions` and `litellm.systemone` are the functions, which
        # shadow the packages of the same names, i.e. their modules are reached
        # through `sys.modules`
        for module_name in ["litellm.decisions.main", "litellm.systemone.main"]:
            monkeypatch.setattr(
                sys.modules[module_name],
                "prepare_call",
                create_failure(f"{module_name}.prepare_call"),
            )
        yield
        assert not unstubbed_calls, (
            f"Model API calls not stubbed by the test: {unstubbed_calls}"
        )

    def test_decision_types(self):
        """
        Tests the question and answer types.
        """
        option = _DecisionOption(value="written", label="written")
        assert option.description is None
        question = _DecisionQuestion(
            name="C0", kind="predicate", instructions="Is the document an NDA?"
        )
        assert question.options == ()
        answer = _DecisionAnswer(name="C0", kind="predicate")
        assert answer.probability is None
        assert answer.option_probabilities is None
        assert answer.refused is False
        assert answer.chosen_option is None

        # Instances are immutable
        for instance, attr_name in [
            (option, "label"),
            (question, "instructions"),
            (answer, "probability"),
        ]:
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(instance, attr_name, None)

    def test_build_concept_questions(self):
        """
        Tests the questions built for concepts of each supported type, including
        the names of the questions of several concepts in one sequence.
        """
        concepts = [
            BooleanConcept(
                name="Termination for convenience",
                description="Either party may terminate without cause",
            ),
            LabelConcept(
                name="Notice form",
                description="Required form of the termination notice",
                labels=["written", "email", "any", "not specified"],
            ),
            LabelConcept(
                name="Topics",
                description="Topics covered by the contract",
                labels=["liability", "payment", "termination"],
                classification_type="multi_label",
            ),
            RatingConcept(
                name="Drafting quality",
                description="Overall drafting quality of the contract",
                rating_scale=(0, 4),
            ),
            BooleanConcept(name="Is a contract", description="Document is a contract"),
            RatingConcept(
                name="Risk level",
                description="Level of risk for the receiving party",
                rating_scale=(3, 7),
            ),
        ]
        questions = _build_concept_questions(concepts)
        assert all(isinstance(question, _DecisionQuestion) for question in questions)
        assert [(question.name, question.kind) for question in questions] == [
            ("C0", "predicate"),
            ("C1", "choice"),
            ("C2-L0", "predicate"),
            ("C2-L1", "predicate"),
            ("C2-L2", "predicate"),
            ("C3", "score"),
            ("C4", "predicate"),
            ("C5", "score"),
        ]
        questions_by_name = {question.name: question for question in questions}
        assert len(questions_by_name) == len(questions)
        for question in questions:
            assert question.instructions == question.instructions.strip()
            self.check_rendered_prompt(question.instructions)
            if question.kind == "predicate":
                assert question.options == ()

        # Boolean concept
        assert questions_by_name["C0"].instructions == (
            "Decide whether the statement in the concept description below is true "
            "according to the text.\n"
            "Concept name: Termination for convenience\n"
            "Concept description: Either party may terminate without cause\n"
            'Answer "yes" if the text supports the statement. Answer "no" if '
            "the text contradicts the statement or does not support it."
        )
        assert questions_by_name["C4"].instructions == (
            "Decide whether the statement in the concept description below is true "
            "according to the text.\n"
            "Concept name: Is a contract\n"
            "Concept description: Document is a contract\n"
            'Answer "yes" if the text supports the statement. Answer "no" if '
            "the text contradicts the statement or does not support it."
        )

        # Label concept, a single label to select
        assert questions_by_name["C1"].instructions == (
            "Classify the text by selecting exactly one of the options, based on "
            "the concept below.\n"
            "Concept name: Notice form\n"
            "Concept description: Required form of the termination notice\n"
            "Select the most appropriate option, even if no option fits the text "
            "perfectly."
        )
        assert questions_by_name["C1"].options == (
            _DecisionOption(value="written", label="written", description=None),
            _DecisionOption(value="email", label="email", description=None),
            _DecisionOption(value="any", label="any", description=None),
            _DecisionOption(
                value="not specified", label="not specified", description=None
            ),
        )

        # Label concept, any number of labels to select: a question for each label
        for label_position, label in enumerate(["liability", "payment", "termination"]):
            assert questions_by_name[f"C2-L{label_position}"].instructions == (
                "Decide whether the label below applies to the text, based on "
                "the concept below.\n"
                "Concept name: Topics\n"
                "Concept description: Topics covered by the contract\n"
                f"Label: {label}\n"
                'Answer "yes" if the label applies to the text. Answer "no" if it '
                "does not. The other labels of the concept are decided separately: "
                "several labels, or none, may apply."
            )

        # Rating concept, including the scales that do not start at 1
        assert questions_by_name["C3"].instructions == (
            "Rate the text on a scale from 0 to 4, based on the concept below.\n"
            "Concept name: Drafting quality\n"
            "Concept description: Overall drafting quality of the contract\n"
            "Select the level of the scale that fits the text best."
        )
        assert questions_by_name["C3"].options == (
            _DecisionOption(value=0, label="0", description="0 on a scale from 0 to 4"),
            _DecisionOption(value=1, label="1", description="1 on a scale from 0 to 4"),
            _DecisionOption(value=2, label="2", description="2 on a scale from 0 to 4"),
            _DecisionOption(value=3, label="3", description="3 on a scale from 0 to 4"),
            _DecisionOption(value=4, label="4", description="4 on a scale from 0 to 4"),
        )
        assert questions_by_name["C5"].instructions.startswith(
            "Rate the text on a scale from 3 to 7, based on the concept below.\n"
        )
        assert [
            (option.value, option.label, option.description)
            for option in questions_by_name["C5"].options
        ] == [
            (level, str(level), f"{level} on a scale from 3 to 7")
            for level in [3, 4, 5, 6, 7]
        ]
        assert all(
            type(option.value) is int
            for name in ["C3", "C5"]
            for option in questions_by_name[name].options
        )

        # The name of a question follows the position of the concept in the sequence
        assert [
            question.name for question in _build_concept_questions(concepts[2:4])
        ] == ["C0-L0", "C0-L1", "C0-L2", "C1"]

        # No concepts, no questions
        assert _build_concept_questions([]) == []

    def test_build_aspect_questions(self):
        """
        Tests the questions built for aspects.
        """
        aspects = [
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                aspects=[
                    Aspect(
                        name="Liability cap",
                        description="Clauses that cap the liability of a party",
                    )
                ],
            ),
            Aspect(
                name="Termination",
                description="Clauses on termination of the agreement",
            ),
        ]
        questions = _build_aspect_questions(aspects)
        # A question for each aspect of the sequence, i.e. not for the sub-aspects
        assert questions == [
            _DecisionQuestion(
                name="A0",
                kind="predicate",
                instructions=(
                    "The text is one paragraph of a document. Decide whether "
                    "the paragraph is relevant to the aspect below.\n"
                    "Aspect name: Liability\n"
                    "Aspect description: Clauses describing liability of the parties\n"
                    'Answer "yes" if the paragraph addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
            _DecisionQuestion(
                name="A1",
                kind="predicate",
                instructions=(
                    "The text is one paragraph of a document. Decide whether "
                    "the paragraph is relevant to the aspect below.\n"
                    "Aspect name: Termination\n"
                    "Aspect description: Clauses on termination of the agreement\n"
                    'Answer "yes" if the paragraph addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
        ]
        for question in questions:
            self.check_rendered_prompt(question.instructions)

        # The name of a question follows the position of the aspect in the sequence
        assert [question.name for question in _build_aspect_questions(aspects[1:])] == [
            "A0"
        ]

        # No aspects, no questions
        assert _build_aspect_questions([]) == []

        # Only aspects get the questions of aspects
        not_only_aspects: list[Any] = [
            aspects[0],
            BooleanConcept(name="Is a contract", description="Document is a contract"),
        ]
        with pytest.raises(
            TypeError, match="Expected an aspect .* got `BooleanConcept`"
        ):
            _build_aspect_questions(not_only_aspects)

    def test_unsupported_concept_types(self):
        """
        Tests that the concepts of the types that a decision model cannot extract
        are rejected with an error that names the concept.
        """
        boolean_concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        unsupported_concepts = [
            StringConcept(name="Contract title", description="Title of the contract"),
            NumericalConcept(name="Notice period", description="Notice period in days"),
            DateConcept(name="Effective date", description="Effective date"),
            JsonObjectConcept(
                name="Party details",
                description="Details of a party",
                structure={"name": str},
            ),
        ]
        for concept in unsupported_concepts:
            error_match = (
                f"Concept `{concept.name}` of type `{type(concept).__name__}` "
                f"cannot be extracted by a decision model"
            )
            # Also when the other concepts in the sequence are supported
            for concepts in [[concept], [boolean_concept, concept]]:
                with pytest.raises(TypeError, match=error_match):
                    _build_concept_questions(concepts)
            with pytest.raises(TypeError, match=error_match):
                _convert_concept_answers(
                    concept, 0, {"C0": _predicate_answer("C0", probability=0.9)}
                )

    def test_convert_boolean_concept_answers(self):
        """
        Tests the conversion of the answer to the question of a Boolean concept,
        including the probability of exactly 0.5.
        """
        concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        for probability, expected_value in [
            (0.93, True),
            (0.5, True),
            (0.4999, False),
            (0.07, False),
            (0.0, False),
            (1.0, True),
            (0, False),
            (1, True),
        ]:
            result = _convert_concept_answers(
                concept, 0, {"C0": _predicate_answer("C0", probability=probability)}
            )
            assert result is not None
            value, probabilities = result
            assert value is expected_value
            assert probabilities == pytest.approx(
                {True: probability, False: 1 - probability}
            )
            assert [type(option) for option in probabilities] == [bool, bool]
            assert list(probabilities) == [True, False]
            assert all(type(i) is float for i in probabilities.values())

        # The answer is taken by the position of the concept, and the answers to
        # the other questions of the request are not read
        answers = {
            "C0": _predicate_answer("C0", probability=0.9),
            "C1": _predicate_answer("C1", probability=0.2),
            "C2": _choice_answer("C2", option_probabilities={"a": 0.6, "b": 0.4}),
            "A1": _predicate_answer("A1", probability=0.8),
        }
        assert _convert_concept_answers(concept, 0, answers) == (
            True,
            pytest.approx({True: 0.9, False: 0.1}),
        )
        assert _convert_concept_answers(concept, 1, answers) == (
            False,
            pytest.approx({True: 0.2, False: 0.8}),
        )

    def test_convert_multi_class_label_concept_answers(self):
        """
        Tests the conversion of the answer to the question of a label concept
        with a single label to select, including the equally probable labels and
        the probabilities that do not sum to 1.
        """
        labels = ["written", "email", "any", "not specified"]
        concept = LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=labels,
        )
        assert concept.classification_type == "multi_class"

        for option_probabilities, expected_label in [
            # The most probable label
            (
                {"written": 0.1, "email": 0.7, "any": 0.15, "not specified": 0.05},
                "email",
            ),
            (
                {"written": 0.0, "email": 0.0, "any": 0.0, "not specified": 1.0},
                "not specified",
            ),
            # The probabilities given in another order than the labels of the concept
            (
                {"not specified": 0.05, "any": 0.15, "email": 0.7, "written": 0.1},
                "email",
            ),
            # Equally probable labels: the first of them in the order of the labels
            # of the concept, not in the order of the answer
            (
                {"written": 0.1, "email": 0.4, "any": 0.4, "not specified": 0.1},
                "email",
            ),
            (
                {"not specified": 0.3, "any": 0.3, "email": 0.1, "written": 0.3},
                "written",
            ),
            (
                {"written": 0.25, "email": 0.25, "any": 0.25, "not specified": 0.25},
                "written",
            ),
            # The probabilities that do not sum to 1 are taken as they are
            (
                {"written": 0.1, "email": 0.1, "any": 0.3, "not specified": 0.1},
                "any",
            ),
            (
                {"written": 0.9, "email": 0.8, "any": 1.0, "not specified": 0.7},
                "any",
            ),
            ({"written": 1, "email": 0, "any": 0, "not specified": 0}, "written"),
        ]:
            result = _convert_concept_answers(
                concept,
                3,
                {"C3": _choice_answer("C3", option_probabilities=option_probabilities)},
            )
            assert result is not None
            value, probabilities = result
            assert value == [expected_label]
            # Each label of the concept with its probability as given, in the order
            # of the labels of the concept
            assert probabilities == option_probabilities
            assert list(probabilities) == labels
            assert all(type(i) is float for i in probabilities.values())

        # The label that the answer names as the picked one does not select
        # the label: the most probable label is taken
        result = _convert_concept_answers(
            concept,
            3,
            {
                "C3": _choice_answer(
                    "C3",
                    option_probabilities={
                        "written": 0.1,
                        "email": 0.7,
                        "any": 0.15,
                        "not specified": 0.05,
                    },
                    chosen_option="any",
                )
            },
        )
        assert result is not None
        assert result[0] == ["email"]

    def test_convert_multi_label_label_concept_answers(self):
        """
        Tests the conversion of the answers to the questions of a label concept
        with any number of labels to select, including the probability of exactly
        0.5 and no label selected.
        """
        labels = ["liability", "payment", "termination"]
        concept = LabelConcept(
            name="Topics",
            description="Topics covered by the contract",
            labels=labels,
            classification_type="multi_label",
        )

        for label_probabilities, expected_labels in [
            ([0.9, 0.2, 0.8], ["liability", "termination"]),
            ([0.1, 0.6, 0.3], ["payment"]),
            # All labels: the probabilities of the labels are independent
            ([1.0, 0.95, 0.5], ["liability", "payment", "termination"]),
            # A label with the probability of exactly 0.5 is selected
            ([0.5, 0.4999, 0.0], ["liability"]),
            ([0.4999, 0.5, 0.5], ["payment", "termination"]),
        ]:
            # The answers are given in the reverse order of the labels
            answers = {
                f"C1-L{label_position}": _predicate_answer(
                    f"C1-L{label_position}", probability=probability
                )
                for label_position, probability in reversed(
                    list(enumerate(label_probabilities))
                )
            }
            result = _convert_concept_answers(concept, 1, answers)
            assert result is not None
            value, probabilities = result
            # The selected labels are in the order of the labels of the concept
            assert value == expected_labels
            # Each label of the concept with the probability of its own question
            assert probabilities == dict(zip(labels, label_probabilities, strict=True))
            assert list(probabilities) == labels

        # No label is selected: no item
        for label_probabilities in [[0.4999, 0.1, 0.0], [0.0, 0.0, 0.0]]:
            answers = {
                f"C1-L{label_position}": _predicate_answer(
                    f"C1-L{label_position}", probability=probability
                )
                for label_position, probability in enumerate(label_probabilities)
            }
            assert _convert_concept_answers(concept, 1, answers) is None

    def test_convert_rating_concept_answers(self):
        """
        Tests the conversion of the answer to the question of a rating concept,
        including the scales that do not start at 1 and the equally probable levels.
        """
        for rating_scale, option_probabilities, expected_value in [
            # The most probable level
            ((1, 5), {"1": 0.05, "2": 0.1, "3": 0.15, "4": 0.6, "5": 0.1}, 4),
            ((1, 5), {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0, "5": 1.0}, 5),
            ((0, 4), {"0": 0.05, "1": 0.1, "2": 0.6, "3": 0.2, "4": 0.05}, 2),
            # The lowest level of a scale that starts at 0, i.e. the value 0 is
            # an extracted value
            ((0, 4), {"0": 0.5, "1": 0.3, "2": 0.1, "3": 0.05, "4": 0.05}, 0),
            ((3, 7), {"3": 0.1, "4": 0.1, "5": 0.1, "6": 0.1, "7": 0.6}, 7),
            ((3, 7), {"3": 0.7, "4": 0.1, "5": 0.1, "6": 0.05, "7": 0.05}, 3),
            ((0, 1), {"0": 0.2, "1": 0.8}, 1),
            # The probabilities given in another order than the levels of the scale
            ((3, 7), {"7": 0.1, "5": 0.6, "3": 0.1, "6": 0.1, "4": 0.1}, 5),
            # Equally probable levels: the lowest of them
            ((1, 5), {"1": 0.1, "2": 0.3, "3": 0.3, "4": 0.3, "5": 0.0}, 2),
            ((3, 7), {"7": 0.4, "6": 0.1, "5": 0.1, "4": 0.4, "3": 0.0}, 4),
            ((0, 4), {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2, "4": 0.2}, 0),
            # The probabilities that do not sum to 1 are taken as they are
            ((0, 4), {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.2, "4": 0.1}, 3),
            ((3, 7), {"3": 0.9, "4": 1.0, "5": 0.9, "6": 0.8, "7": 0.7}, 4),
        ]:
            concept = RatingConcept(
                name="Drafting quality",
                description="Overall drafting quality of the contract",
                rating_scale=rating_scale,
            )
            result = _convert_concept_answers(
                concept,
                2,
                {"C2": _score_answer("C2", option_probabilities=option_probabilities)},
            )
            assert result is not None
            value, probabilities = result
            assert type(value) is int
            assert value == expected_value
            # Each level of the scale with its probability as given, with
            # the integers of the scale as keys, in ascending order
            levels = list(range(rating_scale[0], rating_scale[1] + 1))
            assert probabilities == {
                level: option_probabilities[str(level)] for level in levels
            }
            assert list(probabilities) == levels
            assert all(type(level) is int for level in probabilities)

    def test_convert_aspect_answer(self):
        """
        Tests the conversion of the answer to the question of an aspect, asked on
        one paragraph, including the probability of exactly 0.5, and a refused
        answer, with which the paragraph is not included in the aspect.
        """
        for probability, expected_belongs in [
            (0.93, True),
            (0.5, True),
            (0.4999, False),
            (0.0, False),
            (1.0, True),
            (1, True),
        ]:
            converted_answer = _convert_aspect_answer(
                0, {"A0": _predicate_answer("A0", probability=probability)}
            )
            assert converted_answer is not None
            belongs, probabilities = converted_answer
            assert belongs is expected_belongs
            assert probabilities == pytest.approx(
                {True: probability, False: 1 - probability}
            )
            assert [type(option) for option in probabilities] == [bool, bool]
            assert list(probabilities) == [True, False]

        # The answer is taken by the position of the aspect, and the answers to
        # the other questions of the request are not read
        answers = {
            "A0": _predicate_answer("A0", probability=0.1),
            "A1": _predicate_answer("A1", probability=0.7),
            "C1": _predicate_answer("C1", probability=0.0),
        }
        assert _convert_aspect_answer(0, answers) == (
            False,
            pytest.approx({True: 0.1, False: 0.9}),
        )
        assert _convert_aspect_answer(1, answers) == (
            True,
            pytest.approx({True: 0.7, False: 0.3}),
        )

        # A refused answer: it is not known whether the paragraph belongs to
        # the aspect, whatever the decision threshold is
        answers["A1"] = _predicate_answer("A1", refused=True)
        for decision_threshold in [0.0, 0.5, 1.0]:
            assert (
                _convert_aspect_answer(
                    1, answers, decision_threshold=decision_threshold
                )
                is None
            )
        assert _convert_aspect_answer(0, answers) is not None

    def test_convert_answers_with_decision_threshold(self):
        """
        Tests that the answers to yes/no questions are converted with the decision
        threshold given, including the probability of exactly the threshold, that
        the probabilities of the items do not depend on it, and that it does not
        apply to the questions with options, of which the most probable is taken.
        """
        # === Boolean concept ===
        boolean_concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        for decision_threshold, probability, expected_value in [
            (0.7, 0.93, True),
            (0.7, 0.7, True),
            (0.7, 0.6999, False),
            (0.7, 0.5, False),
            (0.3, 0.5, True),
            (0.3, 0.3, True),
            (0.3, 0.2999, False),
            (1.0, 1.0, True),
            (1.0, 0.9999, False),
        ]:
            assert _convert_concept_answers(
                boolean_concept,
                0,
                {"C0": _predicate_answer("C0", probability=probability)},
                decision_threshold=decision_threshold,
            ) == (
                expected_value,
                # The probabilities are those of the answer, whatever the value is
                pytest.approx({True: probability, False: 1 - probability}),
            )

        # === Label concept, any number of labels to select ===
        labels = ["liability", "payment", "termination"]
        multi_label_concept = LabelConcept(
            name="Topics",
            description="Topics covered by the contract",
            labels=labels,
            classification_type="multi_label",
        )
        label_probabilities = [0.9, 0.6, 0.3]
        answers = {
            f"C0-L{label_position}": _predicate_answer(
                f"C0-L{label_position}", probability=probability
            )
            for label_position, probability in enumerate(label_probabilities)
        }
        for decision_threshold, expected_labels in [
            (0.5, ["liability", "payment"]),
            (0.3, ["liability", "payment", "termination"]),
            (0.2, ["liability", "payment", "termination"]),
            (0.6, ["liability", "payment"]),
            (0.61, ["liability"]),
            (0.9, ["liability"]),
        ]:
            assert _convert_concept_answers(
                multi_label_concept, 0, answers, decision_threshold=decision_threshold
            ) == (expected_labels, dict(zip(labels, label_probabilities, strict=True)))
        # No label reaches the threshold: no item
        assert (
            _convert_concept_answers(
                multi_label_concept, 0, answers, decision_threshold=0.91
            )
            is None
        )

        # === Aspect ===
        for decision_threshold, probability, expected_belongs in [
            (0.7, 0.7, True),
            (0.7, 0.6999, False),
            (0.3, 0.3, True),
            (0.3, 0.2999, False),
        ]:
            assert _convert_aspect_answer(
                0,
                {"A0": _predicate_answer("A0", probability=probability)},
                decision_threshold=decision_threshold,
            ) == (
                expected_belongs,
                pytest.approx({True: probability, False: 1 - probability}),
            )

        # === Questions with options: the most probable option, whatever
        # the threshold and the probability of the option are ===
        multi_class_concept = LabelConcept(
            name="Type", description="Type of the contract", labels=["NDA", "lease"]
        )
        rating_concept = RatingConcept(
            name="Clarity", description="Clarity of the contract", rating_scale=(1, 3)
        )
        for decision_threshold in [0.1, 0.5, 0.99]:
            assert _convert_concept_answers(
                multi_class_concept,
                0,
                {
                    "C0": _choice_answer(
                        "C0", option_probabilities={"NDA": 0.45, "lease": 0.55}
                    )
                },
                decision_threshold=decision_threshold,
            ) == (["lease"], {"NDA": 0.45, "lease": 0.55})
            assert _convert_concept_answers(
                rating_concept,
                0,
                {
                    "C0": _score_answer(
                        "C0", option_probabilities={"1": 0.3, "2": 0.4, "3": 0.3}
                    )
                },
                decision_threshold=decision_threshold,
            ) == (2, {1: 0.3, 2: 0.4, 3: 0.3})

        # === The threshold that applies to an aspect or a concept: its own,
        # if set, otherwise the one of the decision model ===
        aspect = Aspect(name="Termination", description="Termination clauses")
        assert _get_decision_threshold(aspect, 0.6) == 0.6
        assert _get_decision_threshold(boolean_concept, 0.6) == 0.6
        assert _get_decision_threshold(multi_label_concept, 0.6) == 0.6
        aspect.decision_threshold = 0.2
        boolean_concept.decision_threshold = 0.8
        multi_label_concept.decision_threshold = 1.0
        assert _get_decision_threshold(aspect, 0.6) == 0.2
        assert _get_decision_threshold(boolean_concept, 0.6) == 0.8
        assert _get_decision_threshold(multi_label_concept, 0.6) == 1.0
        # The concepts that are extracted with a question with options have no
        # threshold of their own
        assert _get_decision_threshold(rating_concept, 0.6) == 0.6
        assert _get_decision_threshold(multi_class_concept, 0.6) == 0.6

    def test_invalid_predicate_answers(self):
        """
        Tests that an answer to a predicate that cannot be used is rejected as
        an invalid response, for a Boolean concept, for a label of a label concept
        and for an aspect, for which a refused answer is not such an answer.
        """
        boolean_concept = BooleanConcept(
            name="Is a contract", description="Document is a contract"
        )
        multi_label_concept = LabelConcept(
            name="Topics",
            description="Topics covered by the contract",
            labels=["liability", "payment", "termination"],
            classification_type="multi_label",
        )
        # Valid answers to the questions of the labels
        multi_label_answers = {
            f"C1-L{label_position}": _predicate_answer(
                f"C1-L{label_position}", probability=0.8
            )
            for label_position in range(3)
        }
        assert _convert_concept_answers(multi_label_concept, 1, multi_label_answers)

        def convert_boolean(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the Boolean concept at position 0.
            """
            return _convert_concept_answers(boolean_concept, 0, answers)

        def convert_multi_label(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the label concept at position 1, with
            the given answers used instead of the valid ones.
            """
            return _convert_concept_answers(
                multi_label_concept, 1, {**multi_label_answers, **answers}
            )

        def convert_aspect(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the aspect at position 2.
            """
            return _convert_aspect_answer(2, answers)

        for convert, name in [
            (convert_boolean, "C0"),
            # One answer that cannot be used makes the response invalid, even if
            # the answers to the questions of the other labels are valid
            (convert_multi_label, "C1-L1"),
            (convert_aspect, "A2"),
        ]:
            # Refused, also if answered as well, or of another kind than
            # the question
            for refused_answer in [
                _predicate_answer(name, refused=True),
                _predicate_answer(name, probability=0.9, refused=True),
                _choice_answer(name, option_probabilities={"yes": 0.9}, refused=True),
            ]:
                if convert is convert_aspect:
                    # It is not known whether the paragraph belongs to the aspect:
                    # the paragraph is not included, and the response is used
                    assert convert({name: refused_answer}) is None
                else:
                    _check_invalid_answer(
                        convert,
                        {name: refused_answer},
                        name,
                        "the model refused to answer",
                    )
            # Of another kind than the question
            _check_invalid_answer(
                convert,
                {name: _choice_answer(name, option_probabilities={"yes": 0.9})},
                name,
                "the answer is of the kind `choice`, while the question is of "
                "the kind `predicate`",
            )
            _check_invalid_answer(
                convert,
                {name: _score_answer(name, option_probabilities={"1": 0.9})},
                name,
                "the answer is of the kind `score`, while the question is of "
                "the kind `predicate`",
            )
            # Without the probability
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name)},
                name,
                "the answer has no probability",
            )
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name, option_probabilities={"yes": 0.9})},
                name,
                "the answer has no probability",
            )
            # With a probability that is not a number within [0, 1]
            for probability in _INVALID_PROBABILITIES:
                _check_invalid_answer(
                    convert,
                    {name: _predicate_answer(name, probability=probability)},
                    name,
                    "the probability of the answer is not a number",
                )

        # Missing
        for answers in [{}, {"C1": _predicate_answer("C1", probability=0.9)}]:
            _check_invalid_answer(
                convert_boolean, answers, "C0", "the answer is missing"
            )
        _check_invalid_answer(
            lambda answers: _convert_concept_answers(multi_label_concept, 1, answers),
            {
                name: answer
                for name, answer in multi_label_answers.items()
                if name != "C1-L2"
            },
            "C1-L2",
            "the answer is missing",
        )
        for answers in [{}, {"A0": _predicate_answer("A0", probability=0.9)}]:
            _check_invalid_answer(
                convert_aspect, answers, "A2", "the answer is missing"
            )

    def test_invalid_choice_and_score_answers(self):
        """
        Tests that an answer to a "choice" or a "score" question that cannot be
        used is rejected as an invalid response, for a label concept with a single
        label to select and for a rating concept.
        """
        multi_class_concept = LabelConcept(
            name="Notice form",
            description="Required form of the termination notice",
            labels=["written", "email", "any"],
        )
        rating_concept = RatingConcept(
            name="Drafting quality",
            description="Overall drafting quality of the contract",
            rating_scale=(3, 5),
        )

        def convert_multi_class(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the label concept at position 0.
            """
            return _convert_concept_answers(multi_class_concept, 0, answers)

        def convert_rating(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Converts the answers for the rating concept at position 4.
            """
            return _convert_concept_answers(rating_concept, 4, answers)

        for convert, name, kind, create_answer, option_labels, other_labels in [
            (
                convert_multi_class,
                "C0",
                "choice",
                _choice_answer,
                ["written", "email", "any"],
                # Labels that are not the labels of the concept, including
                # a label in another case
                ["other", "Written", "0"],
            ),
            (
                convert_rating,
                "C4",
                "score",
                _score_answer,
                ["3", "4", "5"],
                # Levels that are not the levels of the scale, including the level
                # positions and the levels given in another format
                ["2", "6", "0", "3.0", "three"],
            ),
        ]:
            valid_probabilities = dict(zip(option_labels, [0.2, 0.5, 0.3], strict=True))
            assert convert(
                {name: create_answer(name, option_probabilities=valid_probabilities)}
            )

            # Missing
            for answers in [
                {},
                {"C1": create_answer("C1", option_probabilities=valid_probabilities)},
            ]:
                _check_invalid_answer(convert, answers, name, "the answer is missing")
            # Refused
            _check_invalid_answer(
                convert,
                {name: create_answer(name, refused=True)},
                name,
                "the model refused to answer",
            )
            # Refused, even if answered as well
            _check_invalid_answer(
                convert,
                {
                    name: create_answer(
                        name, option_probabilities=valid_probabilities, refused=True
                    )
                },
                name,
                "the model refused to answer",
            )
            # Of another kind than the question
            for other_answer in [
                _predicate_answer(name, probability=0.9),
                _DecisionAnswer(
                    name=name,
                    kind="score" if kind == "choice" else "choice",
                    option_probabilities=valid_probabilities,
                ),
            ]:
                _check_invalid_answer(
                    convert,
                    {name: other_answer},
                    name,
                    f"the answer is of the kind `{other_answer.kind}`, while "
                    f"the question is of the kind `{kind}`",
                )
            # Without the probabilities of the options
            for answer in [
                create_answer(name),
                create_answer(name, option_probabilities={}),
                create_answer(name, probability=0.9),
            ]:
                _check_invalid_answer(
                    convert,
                    {name: answer},
                    name,
                    "the answer has no probabilities of the options",
                )
            # With an option that was not offered, in addition to the options
            # offered or instead of one of them
            for other_label in other_labels:
                for option_probabilities in [
                    {**valid_probabilities, other_label: 0.1},
                    {
                        **dict(zip(option_labels[:2], [0.2, 0.5], strict=True)),
                        other_label: 0.3,
                    },
                    {other_label: 1.0},
                ]:
                    _check_invalid_answer(
                        convert,
                        {
                            name: create_answer(
                                name, option_probabilities=option_probabilities
                            )
                        },
                        name,
                        "the answer has options that were not offered. Options "
                        rf"offered: {re.escape(str(option_labels))}\.$",
                    )
            # Without one of the options offered: an option without a probability
            # is not taken as an option with a zero probability
            for omitted_label in option_labels:
                _check_invalid_answer(
                    convert,
                    {
                        name: create_answer(
                            name,
                            option_probabilities={
                                label: probability
                                for label, probability in valid_probabilities.items()
                                if label != omitted_label
                            },
                        )
                    },
                    name,
                    "the answer has no probabilities of the options "
                    rf"\['{omitted_label}'\]",
                )
            # With a probability that is not a number within [0, 1]
            for probability in [*_INVALID_PROBABILITIES, None]:
                _check_invalid_answer(
                    convert,
                    {
                        name: create_answer(
                            name,
                            option_probabilities={
                                **valid_probabilities,
                                option_labels[1]: probability,
                            },
                        )
                    },
                    name,
                    "the probability of the option "
                    rf"`{option_labels[1]}` is not a number within \[0, 1\]",
                )
            # With a zero probability of each option: the first option is not
            # taken as the most probable one of equally improbable options
            for zero in [0.0, 0]:
                _check_invalid_answer(
                    convert,
                    {
                        name: create_answer(
                            name,
                            option_probabilities=dict.fromkeys(option_labels, zero),
                        )
                    },
                    name,
                    "the probability of each option is zero",
                )
            # ... while one probability above zero is enough
            assert convert(
                {
                    name: create_answer(
                        name,
                        option_probabilities={
                            **dict.fromkeys(option_labels, 0.0),
                            option_labels[2]: 0.01,
                        },
                    )
                }
            )
            # With an option named as the picked one that was not offered,
            # whatever the probabilities of the options offered are
            for chosen_option in [*other_labels, True, ""]:
                for option_probabilities in [
                    valid_probabilities,
                    dict.fromkeys(option_labels, 0.0),
                ]:
                    _check_invalid_answer(
                        convert,
                        {
                            name: create_answer(
                                name,
                                option_probabilities=option_probabilities,
                                chosen_option=chosen_option,
                            )
                        },
                        name,
                        "the answer picks an option that was not offered. Options "
                        rf"offered: {re.escape(str(option_labels))}\.$",
                    )
            # ... while each of the options offered can be the picked one
            for chosen_option in option_labels:
                assert convert(
                    {
                        name: create_answer(
                            name,
                            option_probabilities=valid_probabilities,
                            chosen_option=chosen_option,
                        )
                    }
                )

    def test_validate_answers(self):
        """
        Tests the validation of the answers to a set of questions, which accepts
        what the conversion of the answers accepts, whatever the questions are
        built for.
        """
        concepts = [
            BooleanConcept(name="Is a contract", description="Document is a contract"),
            LabelConcept(
                name="Notice form",
                description="Required form of the termination notice",
                labels=["written", "email"],
            ),
            RatingConcept(
                name="Drafting quality",
                description="Overall drafting quality of the contract",
                rating_scale=(1, 3),
            ),
        ]
        questions = [
            *_build_concept_questions(concepts),
            *_build_aspect_questions(
                [Aspect(name="Liability", description="Clauses on liability")]
            ),
        ]
        assert [question.name for question in questions] == ["C0", "C1", "C2", "A0"]
        valid_answers = {
            "C0": _predicate_answer("C0", probability=0.9),
            "C1": _choice_answer(
                "C1", option_probabilities={"written": 0.7, "email": 0.3}
            ),
            "C2": _score_answer(
                "C2", option_probabilities={"1": 0.2, "2": 0.5, "3": 0.3}
            ),
            "A0": _predicate_answer("A0", probability=0.1),
        }
        assert _validate_answers(questions, valid_answers) is None
        # The answers to other questions are not read
        assert (
            _validate_answers(
                questions,
                {**valid_answers, "C3": _predicate_answer("C3", refused=True)},
            )
            is None
        )
        # No questions: nothing to validate
        assert _validate_answers([], {}) is None
        # A refused answer is accepted only if specified, as for the questions
        # of aspects, while any other answer that cannot be used is rejected
        refused_answers = {
            **valid_answers,
            "A0": _predicate_answer("A0", refused=True),
            "C1": _choice_answer("C1", refused=True),
        }
        assert (
            _validate_answers(questions, refused_answers, accept_refusals=True) is None
        )
        _check_invalid_answer(
            lambda answers: _validate_answers(questions, answers),
            refused_answers,
            "C1",
            "the model refused to answer",
        )
        _check_invalid_answer(
            lambda answers: _validate_answers(questions, answers, accept_refusals=True),
            {**refused_answers, "C0": _predicate_answer("C0", probability=1.5)},
            "C0",
            "the probability of the answer is not a number",
        )

        def validate(answers: dict[str, _DecisionAnswer]) -> Any:
            """
            Validates the given answers together with the valid answers to
            the other questions.
            """
            return _validate_answers(questions, {**valid_answers, **answers})

        for answers, question_name, reason_match in [
            # Refused
            (
                {"C0": _predicate_answer("C0", refused=True)},
                "C0",
                "the model refused to answer",
            ),
            (
                {"A0": _predicate_answer("A0", probability=0.9, refused=True)},
                "A0",
                "the model refused to answer",
            ),
            # Of another kind than the question
            (
                {"C1": _predicate_answer("C1", probability=0.9)},
                "C1",
                (
                    "the answer is of the kind `predicate`, while the question is of "
                    "the kind `choice`"
                ),
            ),
            # Without the probability, or with a probability that is not one
            ({"C0": _predicate_answer("C0")}, "C0", "the answer has no probability"),
            (
                {"A0": _predicate_answer("A0", probability=1.5)},
                "A0",
                "the probability of the answer is not a number",
            ),
            # With an option that was not offered, or without an option offered
            (
                {
                    "C1": _choice_answer(
                        "C1", option_probabilities={"written": 0.7, "fax": 0.3}
                    )
                },
                "C1",
                (
                    r"the answer has options that were not offered\. Options offered: "
                    r"\['written', 'email'\]\.$"
                ),
            ),
            (
                {"C2": _score_answer("C2", option_probabilities={"1": 0.5, "2": 0.5})},
                "C2",
                r"the answer has no probabilities of the options \['3'\]",
            ),
            # With a zero probability of each option, or with an option named
            # as the picked one that was not offered
            (
                {
                    "C2": _score_answer(
                        "C2", option_probabilities={"1": 0.0, "2": 0.0, "3": 0.0}
                    )
                },
                "C2",
                "the probability of each option is zero",
            ),
            (
                {
                    "C1": _choice_answer(
                        "C1",
                        option_probabilities={"written": 0.7, "email": 0.3},
                        chosen_option="Written",
                    )
                },
                "C1",
                (
                    r"the answer picks an option that was not offered\. Options offered: "
                    r"\['written', 'email'\]\.$"
                ),
            ),
        ]:
            _check_invalid_answer(validate, answers, question_name, reason_match)
        # Missing
        _check_invalid_answer(
            lambda answers: _validate_answers(questions, answers),
            {name: answer for name, answer in valid_answers.items() if name != "C2"},
            "C2",
            "the answer is missing",
        )

    def test_decision_model_init(self):
        """
        Tests the creation of a decision model: it has the params that models of
        all kinds have, the decision threshold, which only a decision model has,
        and none of the params and methods that only an LLM has.
        """
        decision_model = DocumentDecisionModel(model=_DECISION_MODEL)
        assert isinstance(decision_model, _DocumentDecisionModel)
        assert not isinstance(decision_model, _DocumentLLM)
        assert set(DocumentDecisionModel.model_fields) == {
            "model",
            "api_key",
            "api_base",
            "role",
            "num_retries_failed_request",
            "max_retries_invalid_data",
            "timeout",
            "pricing_details",
            "auto_pricing",
            "auto_pricing_refresh",
            "is_fallback",
            "fallback_models",
            # Only a decision model has it
            "decision_threshold",
        }

        # Defaults
        assert decision_model.model == _DECISION_MODEL
        assert decision_model.api_key is None
        assert decision_model.api_base is None
        assert decision_model.role is None
        assert decision_model.num_retries_failed_request == 3
        assert decision_model.max_retries_invalid_data == 3
        assert decision_model.timeout == 600
        assert decision_model.pricing_details is None
        assert decision_model.auto_pricing is False
        assert decision_model.auto_pricing_refresh is False
        assert decision_model.is_fallback is False
        assert decision_model.fallback_models == []
        assert decision_model.decision_threshold == 0.5
        assert decision_model.is_group is False
        assert decision_model.list_roles == [None]
        assert isinstance(decision_model.async_limiter, AsyncLimiter)
        assert decision_model.async_limiter.max_rate == 10
        assert decision_model.async_limiter.time_period == 10

        # All the params
        async_limiter = AsyncLimiter(3, 5)
        decision_model = DocumentDecisionModel(
            model=_DECISION_MODEL,
            api_key="test_api_key",
            api_base="https://decisions.example.com",
            role=" decide ",
            num_retries_failed_request=1,
            max_retries_invalid_data=1,
            timeout=30,
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
            is_fallback=True,
            decision_threshold=0.7,
            async_limiter=async_limiter,  # ty: ignore[unknown-argument]
        )
        assert decision_model.max_retries_invalid_data == 1
        assert decision_model.decision_threshold == 0.7
        assert decision_model.role == "decide"  # validated as the role of an LLM
        assert decision_model.list_roles == ["decide"]
        assert decision_model.async_limiter is async_limiter
        with pytest.raises(TypeError, match="AsyncLimiter"):
            decision_model.async_limiter = 10  # ty: ignore[invalid-assignment]

        # Invalid params
        with pytest.raises(ValueError, match="<model_provider>/<model_name>"):
            DocumentDecisionModel(model="jev-latest")
        with pytest.raises(ValueError, match="num_retries_failed_request"):
            DocumentDecisionModel(model=_DECISION_MODEL, num_retries_failed_request=-1)
        with pytest.raises(ValueError, match="max_retries_invalid_data"):
            DocumentDecisionModel(model=_DECISION_MODEL, max_retries_invalid_data=-1)
        with pytest.raises(ValueError, match="timeout"):
            decision_model.timeout = -1
        assert decision_model.timeout == 30

        # The internal class cannot be created directly
        with pytest.raises(TypeError):
            _DocumentDecisionModel(model=_DECISION_MODEL)

        # The params that only an LLM has are not the params of a decision model
        for param_name, value in [
            ("system_message", "You are a helpful assistant."),
            ("output_language", "adapt"),
            ("temperature", 0.5),
            ("top_p", 0.5),
            ("seed", 123),
            ("max_tokens", 100),
            ("reasoning_effort", "low"),
            ("tools", []),
            ("tool_choice", "auto"),
            ("parallel_tool_calls", True),
            ("tool_max_rounds", 5),
            ("max_retries_failed_request", 1),
            ("deployment_id", "deployment"),
            ("api_version", "2026-01-01"),
        ]:
            assert param_name in DocumentLLM.model_fields
            with pytest.raises(ValueError, match=f"{param_name}\n"):
                DocumentDecisionModel(
                    model=_DECISION_MODEL,
                    **{param_name: value},  # ty: ignore[invalid-argument-type]
                )
            assert not hasattr(decision_model, param_name)
        # ... and a decision model cannot chat
        for method_name in ["chat", "chat_async"]:
            assert hasattr(DocumentLLM, method_name)
            assert not hasattr(decision_model, method_name)

        # What the extraction logic shared with LLMs reads from a model: a call
        # for which a decision model returned an invalid response is made again
        # as many times as set for the model, as for an LLM, and a failed request
        # is retried only by the framework
        assert decision_model._retries_invalid_data == 1
        assert decision_model._provider_retries_failed_request == 0
        llm = _create_llm(max_retries_invalid_data=2, max_retries_failed_request=1)
        assert llm._retries_invalid_data == 2
        assert llm._provider_retries_failed_request == 1

        # The extraction methods, and the usage and cost of the model
        for method_name in [
            "extract_all",
            "extract_all_async",
            "extract_aspects_from_document",
            "extract_aspects_from_document_async",
            "extract_concepts_from_aspect",
            "extract_concepts_from_aspect_async",
            "extract_concepts_from_document",
            "extract_concepts_from_document_async",
            "get_usage",
            "get_cost",
            "reset_usage_and_cost",
        ]:
            assert callable(getattr(decision_model, method_name))
        (usage_container,) = decision_model.get_usage()
        assert usage_container.model == _DECISION_MODEL
        assert usage_container.role == "decide"
        assert usage_container.is_fallback is True
        assert usage_container.usage == _ModelUsage()
        (cost_container,) = decision_model.get_cost()
        assert cost_container.model == _DECISION_MODEL
        assert cost_container.cost == _ModelCost()

    def test_decision_model_pricing(self):
        """
        Tests the pricing params of a decision model: the automatic pricing is
        rejected for the models whose decision requests cannot be priced
        automatically, when the model is created and when the params are assigned,
        and the prices of the decision requests are used for an OpenAI model,
        which genai-prices has apart from the prices of its chat requests.
        """
        pricing_details = ModelPricing(
            input_per_1m_tokens=0.10, output_per_1m_tokens=0.0
        )

        # As for an LLM, the pricing details exclude the automatic pricing
        with pytest.raises(ValueError, match="cannot be set together"):
            _create_decision_model(pricing_details=pricing_details, auto_pricing=True)

        for model, error_match in [
            # genai-prices has the chat prices of the model, i.e. a price of
            # output tokens, which a decision model is not billed for: it has
            # the prices of the decision requests to the model for OpenAI only
            (
                "openrouter/openai/gpt-6-luna",
                (
                    "not supported for decision model `openrouter/openai/gpt-6-luna`.*"
                    "has a price of output tokens.*"
                    "prices of the chat requests.*provide explicit `pricing_details`"
                ),
            ),
            # A self-hosted decision model has no prices
            (
                "strands_decider/strands-decider-2B-hobson-v19",
                "not supported for self-hosted decision models.*`pricing_details`",
            ),
        ]:
            # The pricing data is not looked up for such a model, i.e. the error
            # is not preceded by a warning that the pricing data cannot be fetched
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")
                with pytest.raises(ValueError, match=error_match):
                    _create_decision_model(model, auto_pricing=True)
            assert not any(
                "pricing data" in str(caught_warning.message)
                for caught_warning in caught_warnings
            )

            # ... while the pricing details can be set for it
            decision_model = _create_decision_model(
                model, pricing_details=pricing_details
            )
            assert decision_model.pricing_details == pricing_details
            # ... and no pricing params are needed at all
            decision_model = _create_decision_model(model)
            # The automatic pricing is also rejected when it is assigned,
            # and the model is left unchanged
            with pytest.raises(ValueError, match=error_match):
                decision_model.auto_pricing = True
            assert decision_model.auto_pricing is False

        # As for a local LLM, where the provider of the model is not rejected
        # first, which it is while litellm lists no such provider of decision models
        with (
            patch.object(
                decision_models_module,
                "_get_supported_decision_providers",
                new=lambda: None,
            ),
            pytest.raises(ValueError, match="not supported for local models"),
        ):
            _create_decision_model("ollama/nimble", auto_pricing=True)

        # The automatic pricing is accepted where genai-prices has a decision price,
        # i.e. the prices of the model have no price of output tokens
        decision_model = _create_decision_model(auto_pricing=True)
        assert decision_model._auto_pricing_provider_id == "typesafe"
        assert decision_model._auto_pricing_model_ref == "jev-latest"
        assert decision_model._calculate_auto_pricing_costs(1_000_000, 1_000_000) == (
            Decimal("0.042"),
            Decimal(0),
        )
        # ... whichever provider serves the model
        other_decision_model = _create_decision_model(
            "perplexity/pplx-decider-v1-27b", auto_pricing=True
        )
        input_cost, output_cost = other_decision_model._calculate_auto_pricing_costs(
            1_000_000, 1_000_000
        )
        assert input_cost > 0
        assert output_cost == Decimal(0)
        # For an OpenAI model, the prices of the decision requests are looked up
        # under the provider that genai-prices has for them, and not the prices of
        # the chat requests, which it has under the provider of the model
        openai_decision_model = _create_decision_model(
            "openai/gpt-6-luna", auto_pricing=True
        )
        assert openai_decision_model._auto_pricing_provider_id == "openai-decisions"
        assert openai_decision_model._auto_pricing_model_ref == "gpt-6-luna"
        # ... with the input tokens read from the cache priced as the other input
        # tokens, and no price of output tokens
        assert openai_decision_model._calculate_auto_pricing_costs(
            1_000, 100, cache_read_tokens=200
        ) == (Decimal("0.0001"), Decimal(0))
        # ... while an LLM of the same name gets the prices of the chat requests
        llm = _create_llm("openai/gpt-6-luna", auto_pricing=True)
        assert llm._auto_pricing_provider_id == "openai"
        assert llm._calculate_auto_pricing_costs(0, 1_000_000)[1] > 0
        # A model that genai-prices has no prices for is accepted as well, with
        # a warning that its costs are not calculated, as for an LLM: e.g. a model
        # deployed in Microsoft Foundry, which is identified by the name that
        # its deployment was given
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            decision_model_without_prices = _create_decision_model(
                "azure_ai/decision-deployment", auto_pricing=True
            )
        assert decision_model_without_prices.auto_pricing is True
        assert any(
            "Unable to fetch pricing data for model `azure_ai/decision-deployment`"
            in str(caught_warning.message)
            for caught_warning in caught_warnings
        )
        # ... until the model is changed to a model that it is rejected for
        with pytest.raises(ValueError, match="not supported for decision model"):
            decision_model.model = "openrouter/openai/gpt-6-luna"
        assert decision_model.model == _DECISION_MODEL
        assert decision_model.auto_pricing is True
        # ... while it is priced under the provider of the decision requests
        # when it is changed to an OpenAI model
        decision_model.model = "openai/gpt-6-luna"
        assert decision_model._auto_pricing_provider_id == "openai-decisions"
        assert decision_model._auto_pricing_model_ref == "gpt-6-luna"

    def test_extract_concepts_from_document_with_decision_model(self):
        """
        Tests the extraction of the concepts of each supported type from a document
        with a decision model: the request sent, and the items extracted with
        their values and probabilities.
        """
        decision_model = _create_decision_model(
            api_base="https://decisions.example.com", timeout=30
        )
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA"),
                BooleanConcept(
                    name="Lease check",
                    description="The document is a lease agreement",
                    singular_occurrence=True,
                ),
                LabelConcept(
                    name="Notice form",
                    description="Required form of the termination notice",
                    labels=["written", "email", "any"],
                ),
                LabelConcept(
                    name="Topics",
                    description="Topics covered by the document",
                    labels=["confidentiality", "termination", "payment"],
                    classification_type="multi_label",
                ),
                LabelConcept(
                    name="Parties",
                    description="Types of the parties to the agreement",
                    labels=["company", "individual"],
                    classification_type="multi_label",
                ),
                RatingConcept(
                    name="Clarity",
                    description="Clarity of the text of the document",
                    rating_scale=(0, 3),
                ),
            ]
        )
        answers = {
            "C0": 0.93,
            "C1": 0.2,
            "C2": {"written": 0.7, "email": 0.2, "any": 0.1},
            "C3-L0": 0.9,
            "C3-L1": 0.5,
            "C3-L2": 0.1,
            "C4-L0": 0.4,
            "C4-L1": 0.3,
            "C5": {"0": 0.05, "1": 0.15, "2": 0.6, "3": 0.2},
        }
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [answers]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            extracted_concepts = decision_model.extract_concepts_from_document(document)

        # === The request ===
        # One request, with the text of the document and the questions of all
        # the concepts, sent with the params of the model
        (request,) = decisions_stub.requests
        assert set(request) == {
            "model",
            "input",
            "questions",
            "api_key",
            "api_base",
            "timeout",
        }
        assert request["model"] == _DECISION_MODEL
        assert request["api_key"] == "test_api_key"
        assert request["api_base"] == "https://decisions.example.com"
        assert request["timeout"] == 30
        # The text of the document, as it is sent to an LLM
        assert request["input"] == _DOCUMENT_TEXT_SENT
        # The questions, in the OpenAI format of the Decisions API
        questions = _build_concept_questions(document.concepts)
        instructions = {question.name: question.instructions for question in questions}
        assert request["questions"] == [
            {"type": "predicate", "name": "C0", "instructions": instructions["C0"]},
            {"type": "predicate", "name": "C1", "instructions": instructions["C1"]},
            {
                "type": "choice",
                "name": "C2",
                "instructions": instructions["C2"],
                # A label has no description
                "choices": [{"value": "written"}, {"value": "email"}, {"value": "any"}],
            },
            *[
                {"type": "predicate", "name": name, "instructions": instructions[name]}
                for name in ["C3-L0", "C3-L1", "C3-L2", "C4-L0", "C4-L1"]
            ],
            {
                "type": "score",
                "name": "C5",
                "instructions": instructions["C5"],
                "levels": [
                    {
                        "label": str(level),
                        "description": f"{level} on a scale from 0 to 3",
                    }
                    for level in [0, 1, 2, 3]
                ],
            },
        ]
        assert "Concept name: NDA check\n" in instructions["C0"]
        assert "Label: termination\n" in instructions["C3-L1"]

        # === The items ===
        assert [concept.name for concept in extracted_concepts] == [
            concept.name for concept in document.concepts
        ]
        assert all(
            extracted_concept is concept
            for extracted_concept, concept in zip(
                extracted_concepts, document.concepts, strict=True
            )
        )
        assert all(concept._is_processed for concept in document.concepts)
        expected_items: list[tuple[type, Any, dict[bool | int | str, float]] | None] = [
            # A Boolean concept always gets an item, True or False
            (_BooleanItem, True, {True: 0.93, False: 1 - 0.93}),
            (_BooleanItem, False, {True: 0.2, False: 1 - 0.2}),
            # The most probable label, with the probability of each label
            (_LabelItem, ["written"], {"written": 0.7, "email": 0.2, "any": 0.1}),
            # The labels with a probability of 0.5 or more, with the probability
            # of each label, which do not sum to 1
            (
                _LabelItem,
                ["confidentiality", "termination"],
                {"confidentiality": 0.9, "termination": 0.5, "payment": 0.1},
            ),
            # No label has a probability of 0.5 or more: no item
            None,
            # The most probable level of the scale, as an integer of the scale
            (_IntegerItem, 2, {0: 0.05, 1: 0.15, 2: 0.6, 3: 0.2}),
        ]
        for concept, expected_item in zip(
            document.concepts, expected_items, strict=True
        ):
            if expected_item is None:
                assert concept.extracted_items == []
                continue
            # At most one item per concept, whatever `singular_occurrence` is
            (item,) = concept.extracted_items
            item_class, value, probabilities = expected_item
            assert type(item) is item_class
            assert item.value == value
            assert type(item.value) is type(value)
            self.check_item_probabilities(item, probabilities)
            # A decision model generates no text and gives no references
            assert item.justification is None
            assert not item.reference_paragraphs
            assert not item.reference_sentences
        self.check_instance_serialization_and_cloning(document)

        # === Already processed concepts ===
        decisions_stub.requests.clear()
        decisions_stub.outcomes[_DECISION_MODEL] = [
            {**answers, "C0": 0.1, "C4-L1": 0.8, "C5": {**answers["C5"], "3": 0.9}}
        ]
        with patch.object(litellm, "adecisions", new=decisions_stub):
            # ... are not extracted again, before any request is sent
            with pytest.raises(
                ValueError,
                match=(
                    r"^Some concepts have already been processed: \[.+\]\. "
                    r"Set `overwrite_existing=True` to overwrite them\.$"
                ),
            ):
                decision_model.extract_concepts_from_document(document)
            assert not decisions_stub.requests
            # ... unless they are to be overwritten
            decision_model.extract_concepts_from_document(
                document, overwrite_existing=True
            )
        assert len(decisions_stub.requests) == 1
        assert document.concepts[0].extracted_items[0].value is False
        assert document.concepts[4].extracted_items[0].value == ["individual"]
        assert document.concepts[5].extracted_items[0].value == 3

        # === Several calls ===
        expected_values = [
            True,
            True,
            ["written"],
            ["confidentiality", "termination", "payment"],
            ["company", "individual"],
            0,
        ]
        for extraction_kwargs, expected_question_names in [
            # The number of concepts per call, i.e. per request, is limited. The names
            # of the questions follow the position of a concept in its request.
            (
                {"max_items_per_call": 4},
                [
                    ["C0", "C1", "C2", "C3-L0", "C3-L1", "C3-L2"],
                    ["C0-L0", "C0-L1", "C1"],
                ],
            ),
            # With concurrency, each concept is extracted in its own request
            (
                {"use_concurrency": True},
                [
                    ["C0"],
                    ["C0"],
                    ["C0"],
                    ["C0-L0", "C0-L1", "C0-L2"],
                    ["C0-L0", "C0-L1"],
                    ["C0"],
                ],
            ),
            # ... unless more concepts per call are set
            (
                {"use_concurrency": True, "max_items_per_call": 5},
                [
                    ["C0", "C1", "C2", "C3-L0", "C3-L1", "C3-L2", "C4-L0", "C4-L1"],
                    ["C0"],
                ],
            ),
        ]:
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [_answer_all_questions]}
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            # The requests are sent in any order when sent concurrently
            assert sorted(
                [question["name"] for question in request["questions"]]
                for request in decisions_stub.requests
            ) == sorted(expected_question_names)
            # Each request is sent the text of the document as a whole
            assert all(
                request["input"] == _DOCUMENT_TEXT_SENT
                for request in decisions_stub.requests
            )
            assert [
                concept.extracted_items[0].value for concept in document.concepts
            ] == expected_values

    def test_extract_concepts_from_aspect_with_decision_model(self):
        """
        Tests the extraction of concepts from an aspect with a decision model:
        the questions are asked on the text of the paragraphs extracted for
        the aspect, including for the concepts of a sub-aspect.
        """
        decision_model = _create_decision_model()
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                    concepts=[
                        BooleanConcept(
                            name="Termination for convenience",
                            description="Either party may terminate without cause",
                        ),
                        RatingConcept(
                            name="Notice period",
                            description="How long the notice period is",
                            rating_scale=(1, 3),
                        ),
                    ],
                    aspects=[
                        Aspect(
                            name="Notices",
                            description="Clauses on the notices of termination",
                            concepts=[
                                LabelConcept(
                                    name="Notice form",
                                    description="Required form of the notice",
                                    labels=["written", "any"],
                                )
                            ],
                        )
                    ],
                ),
                Aspect(
                    name="Liability",
                    description="Clauses on the liability of the parties",
                    concepts=[
                        BooleanConcept(
                            name="Liability cap",
                            description="The liability of a party is capped",
                        )
                    ],
                ),
            ]
        )
        aspect, empty_aspect = document.aspects
        (sub_aspect,) = aspect.aspects
        _set_aspect_as_extracted(document, aspect)

        answers = {"C0": 0.8, "C1": {"1": 0.1, "2": 0.2, "3": 0.7}}
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [answers]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            # An aspect that is not extracted yet has no text to ask the questions on
            with pytest.raises(ValueError, match="not yet processed"):
                decision_model.extract_concepts_from_aspect(empty_aspect, document)
            # ... and neither has an aspect extracted with no paragraphs
            empty_aspect._is_processed = True
            decision_model.extract_concepts_from_aspect(empty_aspect, document)
            assert not decisions_stub.requests
            assert not empty_aspect.concepts[0]._is_processed
            assert not empty_aspect.concepts[0].extracted_items

            extracted_concepts = decision_model.extract_concepts_from_aspect(
                aspect, document
            )

        # One request, with the text of the paragraphs of the aspect. The sub-aspect
        # has no paragraphs, i.e. no request is sent for its concepts.
        (request,) = decisions_stub.requests
        assert request["input"] == _ASPECT_PARAGRAPHS_TEXT_SENT
        assert [
            (question["type"], question["name"]) for question in request["questions"]
        ] == [("predicate", "C0"), ("score", "C1")]
        assert [concept.name for concept in extracted_concepts] == [
            "Termination for convenience",
            "Notice period",
        ]
        boolean_concept, rating_concept = aspect.concepts
        (item,) = boolean_concept.extracted_items
        assert item.value is True
        self.check_item_probabilities(item, {True: 0.8, False: 1 - 0.8})
        (item,) = rating_concept.extracted_items
        assert item.value == 3
        self.check_item_probabilities(item, {1: 0.1, 2: 0.2, 3: 0.7})
        assert boolean_concept._is_processed and rating_concept._is_processed
        assert not sub_aspect.concepts[0]._is_processed

        # The concepts of a sub-aspect are extracted from the text of its own
        # paragraphs, together with the concepts of the aspect
        sub_aspect_paragraph = document.paragraphs[2]
        sub_aspect_item = _StringItem(value=sub_aspect_paragraph.raw_text)
        sub_aspect_item.reference_paragraphs = [sub_aspect_paragraph]
        sub_aspect.extracted_items = [sub_aspect_item]
        sub_aspect.reference_paragraphs = [sub_aspect_paragraph]
        sub_aspect._is_processed = True
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [answers, {"C0": {"written": 0.6, "any": 0.4}}]}
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_aspect(
                aspect, document, overwrite_existing=True
            )
        assert [request["input"] for request in decisions_stub.requests] == [
            _ASPECT_PARAGRAPHS_TEXT_SENT,
            "Either party may terminate this agreement with a written notice.",
        ]
        (item,) = aspect.aspects[0].concepts[0].extracted_items
        assert item.value == ["written"]
        self.check_item_probabilities(item, {"written": 0.6, "any": 0.4})
        self.check_instance_serialization_and_cloning(document)

    @staticmethod
    def _get_texts_of_parts(
        n_paragraphs_per_part: int, paragraph_texts: list[str] | None = None
    ) -> list[str]:
        """
        Returns the texts expected to be sent to a decision model for concepts
        when the paragraphs are analyzed in parts: for each part, the texts of
        its paragraphs, separated by a blank line.

        :param n_paragraphs_per_part: The max number of paragraphs in a part.
        :type n_paragraphs_per_part: int
        :param paragraph_texts: The texts of the paragraphs, as they are sent.
            Defaults to those of the document created by
            :func:`_create_aspects_document`.
        :type paragraph_texts: list[str] | None
        :return: The text of each part, in the order of the parts.
        :rtype: list[str]
        """
        if paragraph_texts is None:
            paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_CLEANED
        return [
            "\n\n".join(paragraph_texts[start : start + n_paragraphs_per_part])
            for start in range(0, len(paragraph_texts), n_paragraphs_per_part)
        ]

    @staticmethod
    def _create_concepts_responder(
        texts: list[str], answers_of_texts: list[dict[str, Any]], **usage_kwargs: int
    ) -> Callable[[dict[str, Any]], OpenAIDecisionResponse]:
        """
        Creates the function that responds to a request with the questions of
        concepts, asked on one of the given texts, e.g. on a part of a document,
        with the answers given for that text.

        :param texts: The texts that the requests are expected to be sent with.
        :type texts: list[str]
        :param answers_of_texts: For each text, what to answer (see
            :func:`_create_decisions_response`) by the names of the concepts.
            For a concept with a question on each label, what to answer by
            the labels.
        :type answers_of_texts: list[dict[str, Any]]
        :param usage_kwargs: Token counts to report in the responses.
        :type usage_kwargs: int
        :return: Function that creates the response for a request.
        :rtype: Callable[[dict[str, Any]], OpenAIDecisionResponse]
        """

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Creates the response for the request.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            text_answers = answers_of_texts[texts.index(request["input"])]
            answers: dict[str, Any] = {}
            for question in request["questions"]:
                concept_name_match = re.search(
                    r"^Concept name: (.+)$",
                    question["instructions"],
                    flags=re.MULTILINE,
                )
                assert concept_name_match is not None, question
                answer = text_answers[concept_name_match.group(1)]
                label_match = re.search(
                    r"^Label: (.+)$", question["instructions"], flags=re.MULTILINE
                )
                if label_match is not None:
                    answer = answer[label_match.group(1)]
                answers[question["name"]] = answer
            return _create_decisions_response(
                request["questions"], answers, **usage_kwargs
            )

        return respond

    def test_extract_document_concepts_in_parts_with_decision_model(self):
        """
        Tests the extraction of the concepts of a document with a decision model
        when the max number of paragraphs to analyze per call is set:
        the paragraphs are split into parts as they are for an LLM, one request
        is sent for each part, with the questions of all the concepts, and each
        concept gets an item for each part, in the order of the parts.
        The text is sent as a whole if the number is not set, or is not lower
        than the number of the paragraphs.
        """

        def create_document() -> Document:
            """
            Creates a document with five paragraphs and a concept of each kind.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    ),
                    LabelConcept(
                        name="Notice form",
                        description="Required form of the termination notice",
                        labels=["written", "any"],
                    ),
                    LabelConcept(
                        name="Topics",
                        description="Topics covered by the text",
                        labels=["confidentiality", "termination", "payment"],
                        classification_type="multi_label",
                    ),
                    RatingConcept(
                        name="Clarity",
                        description="Clarity of the text",
                        rating_scale=(1, 3),
                    ),
                ]
            )

        def check_items(
            document: Document, expected_items: dict[str, list[tuple[Any, Any]]]
        ) -> None:
            """
            Checks the items of each concept of the document.

            :param document: The document with the extracted concepts.
            :type document: Document
            :param expected_items: The value and the probabilities of each item
                expected, by the names of the concepts.
            :type expected_items: dict[str, list[tuple[Any, Any]]]
            """
            for concept in document.concepts:
                assert concept._is_processed
                expected_concept_items = expected_items[concept.name]
                assert len(concept.extracted_items) == len(expected_concept_items)
                for item, (value, probabilities) in zip(
                    concept.extracted_items, expected_concept_items, strict=True
                ):
                    assert item.value == value
                    assert type(item.value) is type(value)
                    self.check_item_probabilities(item, probabilities)
                    # An item does not tell the part that it is extracted from,
                    # as an item extracted by an LLM without references
                    assert item.justification is None
                    assert not item.reference_paragraphs
                    assert not item.reference_sentences

        def requests_sent(stub: _DecisionsAPIStub) -> list[tuple[str, list[str]]]:
            """
            Returns the text and the names of the questions of each request sent.

            :param stub: The stub with the requests received.
            :type stub: _DecisionsAPIStub
            :return: The requests, in the order they were sent.
            :rtype: list[tuple[str, list[str]]]
            """
            return [
                (
                    request["input"],
                    [question["name"] for question in request["questions"]],
                )
                for request in stub.requests
            ]

        # Five paragraphs, up to two per part: three parts, the last one with
        # the paragraph that is left
        part_texts = self._get_texts_of_parts(2)
        assert part_texts == [
            (
                "Non-disclosure agreement\n\n"
                "The receiving party shall keep the information confidential."
            ),
            (
                "Either party may terminate this agreement with a written notice "
                "of 30 days.\n\n"
                "The liability of each party is limited to the fees paid."
            ),
            "A notice of termination does not release a party from confidentiality.",
        ]
        answers_of_parts: list[dict[str, Any]] = [
            {
                "NDA check": 0.9,
                "Notice form": {"written": 0.2, "any": 0.8},
                "Topics": {"confidentiality": 0.9, "termination": 0.1, "payment": 0.2},
                "Clarity": {"1": 0.1, "2": 0.2, "3": 0.7},
            },
            {
                "NDA check": 0.2,
                "Notice form": {"written": 0.7, "any": 0.3},
                # No label has a probability of 0.5 or more
                "Topics": {"confidentiality": 0.1, "termination": 0.4, "payment": 0.3},
                "Clarity": {"1": 0.6, "2": 0.3, "3": 0.1},
            },
            {
                "NDA check": 0.6,
                "Notice form": {"written": 0.5, "any": 0.5},
                "Topics": {"confidentiality": 0.5, "termination": 0.8, "payment": 0.0},
                "Clarity": {"1": 0.2, "2": 0.5, "3": 0.3},
            },
        ]
        # An item for each part, in the order of the parts, each with
        # the probabilities of the answers on its part: the answers on different
        # parts are not merged
        expected_items: dict[str, list[tuple[Any, Any]]] = {
            # A Boolean concept gets an item for each part, True or False
            "NDA check": [
                (True, {True: 0.9, False: 1 - 0.9}),
                (False, {True: 0.2, False: 1 - 0.2}),
                (True, {True: 0.6, False: 1 - 0.6}),
            ],
            "Notice form": [
                (["any"], {"written": 0.2, "any": 0.8}),
                (["written"], {"written": 0.7, "any": 0.3}),
                (["written"], {"written": 0.5, "any": 0.5}),
            ],
            # No item for the part in which no label applies
            "Topics": [
                (
                    ["confidentiality"],
                    {"confidentiality": 0.9, "termination": 0.1, "payment": 0.2},
                ),
                (
                    ["confidentiality", "termination"],
                    {"confidentiality": 0.5, "termination": 0.8, "payment": 0.0},
                ),
            ],
            "Clarity": [
                (3, {1: 0.1, 2: 0.2, 3: 0.7}),
                (1, {1: 0.6, 2: 0.3, 3: 0.1}),
                (2, {1: 0.2, 2: 0.5, 3: 0.3}),
            ],
        }
        respond = self._create_concepts_responder(
            part_texts, answers_of_parts, input_tokens=50
        )
        all_question_names = ["C0", "C1", "C2-L0", "C2-L1", "C2-L2", "C3"]

        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            capture_logs() as captured_logs,
        ):
            extracted_concepts = decision_model.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )

        # === The requests ===
        # One request for each part, in the order of the parts, with the text of
        # the part and the questions of all the concepts
        assert requests_sent(decisions_stub) == [
            (text, all_question_names) for text in part_texts
        ]
        expected_questions = [
            decision_models_module._get_request_question(question)
            for question in _build_concept_questions(document.concepts)
        ]
        assert all(
            request["questions"] == expected_questions
            for request in decisions_stub.requests
        )
        # The number of paragraphs per call is applied, i.e. there is no message
        # that it is not
        assert not any("is not applied" in log for log in captured_logs)

        # === The items ===
        assert extracted_concepts == document.concepts
        check_items(document, expected_items)
        self.check_instance_serialization_and_cloning(document)

        # === The usage ===
        # Each request is recorded as a call, in the order of the parts
        usage = decision_model.get_usage()[0].usage
        assert [json.loads(call.prompt)["input"] for call in usage.calls] == part_texts
        assert all(call.timestamp_received is not None for call in usage.calls)
        assert usage.input == 3 * 50

        # === Several calls ===
        # Each call sends a request for each part, with the questions of its
        # concepts. The names of the questions follow the position of a concept
        # in its call.
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document,
                overwrite_existing=True,
                max_items_per_call=3,
                max_paragraphs_to_analyze_per_call=2,
            )
        assert requests_sent(decisions_stub) == [
            *[(text, all_question_names[:-1]) for text in part_texts],
            *[(text, ["C0"]) for text in part_texts],
        ]
        check_items(document, expected_items)

        # === One paragraph per part ===
        paragraph_texts = self._get_texts_of_parts(1)
        assert paragraph_texts == _ASPECTS_PARAGRAPH_TEXTS_CLEANED
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, overwrite_existing=True, max_paragraphs_to_analyze_per_call=1
            )
        assert requests_sent(decisions_stub) == [
            (text, all_question_names) for text in paragraph_texts
        ]
        assert all(len(concept.extracted_items) == 5 for concept in document.concepts)

        # === No parts ===
        # The text of the document is sent as a whole, in one request, as it is
        # without the number of paragraphs per call
        whole_text = "\n\n".join(_ASPECTS_PARAGRAPH_TEXTS_CLEANED)
        respond_on_whole_text = self._create_concepts_responder(
            [whole_text], answers_of_parts[:1]
        )
        expected_whole_text_items = {
            name: items[:1] for name, items in expected_items.items()
        }
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_on_whole_text]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, overwrite_existing=True
            )
        (whole_text_request,) = decisions_stub.requests
        assert whole_text_request["input"] == whole_text
        check_items(document, expected_whole_text_items)
        # ... if the number is zero, or is not lower than the number of
        # the paragraphs of the document
        for max_paragraphs in [0, 5, 6, 100]:
            decisions_stub.requests.clear()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs() as captured_logs,
            ):
                decision_model.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=max_paragraphs,
                )
            assert decisions_stub.requests == [whole_text_request]
            check_items(document, expected_whole_text_items)
            assert not any("is not applied" in log for log in captured_logs)

        # === A document created from a raw text ===
        # The text of a part is built of the texts of its paragraphs, while
        # the text sent as a whole is the text of the document, with its line breaks
        for max_paragraphs, expected_texts in [
            (
                1,
                [
                    "Non-disclosure agreement",
                    "The parties agree as follows.",
                    "Either party may terminate this agreement with a written notice.",
                ],
            ),
            (
                2,
                [
                    "Non-disclosure agreement\n\nThe parties agree as follows.",
                    "Either party may terminate this agreement with a written notice.",
                ],
            ),
            (3, [_DOCUMENT_TEXT_SENT]),
        ]:
            document = _create_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    )
                ]
            )
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [_answer_all_questions]}
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_concepts_from_document(
                    document, max_paragraphs_to_analyze_per_call=max_paragraphs
                )
            assert requests_sent(decisions_stub) == [
                (text, ["C0"]) for text in expected_texts
            ]
            assert [item.value for item in document.concepts[0].extracted_items] == [
                True
            ] * len(expected_texts)

        # === A part without text ===
        # Such a part is not asked about, and gives no items. A paragraph cannot
        # be created without text: the text of one is replaced here.
        document = create_document()
        document.paragraphs[1].__dict__["raw_text"] = "\u200b"
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=1
            )
        assert requests_sent(decisions_stub) == [
            (text, all_question_names)
            for text in [paragraph_texts[0], *paragraph_texts[2:]]
        ]
        assert all(len(concept.extracted_items) == 4 for concept in document.concepts)

    def test_extract_aspect_concepts_in_parts_with_decision_model(self):
        """
        Tests the extraction of the concepts of an aspect with a decision model
        when the max number of paragraphs to analyze per call is set:
        the paragraphs extracted for the aspect are split into parts, and so are
        those of a sub-aspect, for the concepts of the sub-aspect.
        """
        decision_model = _create_decision_model()
        document = _create_aspects_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                    concepts=[
                        BooleanConcept(
                            name="Termination for convenience",
                            description="Either party may terminate without cause",
                        ),
                        RatingConcept(
                            name="Notice period",
                            description="How long the notice period is",
                            rating_scale=(1, 3),
                        ),
                    ],
                    aspects=[
                        Aspect(
                            name="Notices",
                            description="Clauses on the notices of termination",
                            concepts=[
                                LabelConcept(
                                    name="Notice form",
                                    description="Required form of the notice",
                                    labels=["written", "any"],
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        (aspect,) = document.aspects
        (sub_aspect,) = aspect.aspects
        # The aspect has the last four paragraphs of the document, and
        # the sub-aspect has the last two of them
        _set_aspect_as_extracted(document, aspect)
        sub_aspect_paragraphs = document.paragraphs[3:]
        sub_aspect_items = []
        for paragraph in sub_aspect_paragraphs:
            sub_aspect_item = _StringItem(value=paragraph.raw_text)
            sub_aspect_item.reference_paragraphs = [paragraph]
            sub_aspect_items.append(sub_aspect_item)
        sub_aspect.extracted_items = sub_aspect_items  # ty: ignore[invalid-assignment]
        sub_aspect.reference_paragraphs = sub_aspect_paragraphs
        sub_aspect._is_processed = True
        aspect_paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_CLEANED[1:]
        sub_aspect_text = "\n\n".join(_ASPECTS_PARAGRAPH_TEXTS_CLEANED[3:])

        # === Up to three paragraphs per part ===
        # The four paragraphs of the aspect are sent in two parts. The two
        # paragraphs of the sub-aspect are within the limit: sent as a whole.
        aspect_part_texts = self._get_texts_of_parts(3, aspect_paragraph_texts)
        assert aspect_part_texts == [
            "\n\n".join(aspect_paragraph_texts[:3]),
            aspect_paragraph_texts[3],
        ]
        respond = self._create_concepts_responder(
            [*aspect_part_texts, sub_aspect_text],
            [
                {
                    "Termination for convenience": 0.8,
                    "Notice period": {"1": 0.1, "2": 0.2, "3": 0.7},
                },
                {
                    "Termination for convenience": 0.3,
                    "Notice period": {"1": 0.5, "2": 0.4, "3": 0.1},
                },
                {"Notice form": {"written": 0.6, "any": 0.4}},
            ],
        )
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_aspect(
                aspect, document, max_paragraphs_to_analyze_per_call=3
            )
        assert [
            (request["input"], [question["name"] for question in request["questions"]])
            for request in decisions_stub.requests
        ] == [
            *[(text, ["C0", "C1"]) for text in aspect_part_texts],
            (sub_aspect_text, ["C0"]),
        ]
        # An item for each part of the text of the aspect, in the order of the parts
        boolean_concept, rating_concept = aspect.concepts
        first_item, second_item = boolean_concept.extracted_items
        assert (first_item.value, second_item.value) == (True, False)
        self.check_item_probabilities(first_item, {True: 0.8, False: 1 - 0.8})
        self.check_item_probabilities(second_item, {True: 0.3, False: 1 - 0.3})
        first_item, second_item = rating_concept.extracted_items
        assert (first_item.value, second_item.value) == (3, 1)
        self.check_item_probabilities(first_item, {1: 0.1, 2: 0.2, 3: 0.7})
        self.check_item_probabilities(second_item, {1: 0.5, 2: 0.4, 3: 0.1})
        # ... and one item for the text of the sub-aspect
        (item,) = sub_aspect.concepts[0].extracted_items
        assert item.value == ["written"]
        self.check_item_probabilities(item, {"written": 0.6, "any": 0.4})
        self.check_instance_serialization_and_cloning(document)

        # === One paragraph per part ===
        # Both the aspect and the sub-aspect are sent in parts, each in the parts
        # of its own paragraphs
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_aspect(
                aspect,
                document,
                overwrite_existing=True,
                max_paragraphs_to_analyze_per_call=1,
            )
        assert [request["input"] for request in decisions_stub.requests] == [
            *aspect_paragraph_texts,
            *_ASPECTS_PARAGRAPH_TEXTS_CLEANED[3:],
        ]
        assert [len(concept.extracted_items) for concept in aspect.concepts] == [4, 4]
        assert len(sub_aspect.concepts[0].extracted_items) == 2

        # === No parts ===
        # The text of the paragraphs of the aspect is sent as a whole if
        # the number is not set, or is not lower than the number of its paragraphs
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_aspect(
                aspect, document, overwrite_existing=True
            )
        whole_text_requests = list(decisions_stub.requests)
        assert [request["input"] for request in whole_text_requests] == [
            "\n\n".join(aspect_paragraph_texts),
            sub_aspect_text,
        ]
        for max_paragraphs in [0, 4, 5]:
            decisions_stub.requests.clear()
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_concepts_from_aspect(
                    aspect,
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=max_paragraphs,
                )
            # The requests are the same as without the number of paragraphs per call
            assert decisions_stub.requests == whole_text_requests
            assert [len(concept.extracted_items) for concept in aspect.concepts] == [
                1,
                1,
            ]
            assert len(sub_aspect.concepts[0].extracted_items) == 1

    def test_decision_model_concept_parts_concurrency(self):
        """
        Tests that the requests for the parts of a text, which concepts are
        extracted with by a decision model when the max number of paragraphs to
        analyze per call is set, are sent one after another by default, and
        concurrently, each paced by the limiter of the model, with concurrency
        enabled, with the same items extracted.
        """
        part_texts = self._get_texts_of_parts(2)
        n_parts = len(part_texts)
        respond = self._create_concepts_responder(
            part_texts,
            [
                {"NDA check": 0.9, "Clarity": {"1": 0.1, "2": 0.2, "3": 0.7}},
                {"NDA check": 0.2, "Clarity": {"1": 0.6, "2": 0.3, "3": 0.1}},
                {"NDA check": 0.6, "Clarity": {"1": 0.2, "2": 0.5, "3": 0.3}},
            ],
        )
        requests_in_progress: list[str] = []
        max_requests_in_progress: list[int] = []
        completed_inputs: list[str] = []
        acquisitions: list[tuple[float, float]] = []

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the config of the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                acquisitions.append((self.max_rate, self.time_period))
                await super().acquire(amount)

        # Set when the expected number of requests are in progress at the same
        # time, and when a request for the part at each position is completed
        events: dict[str | int, asyncio.Event] = {}
        n_requests_at_once: list[int] = [1]

        async def wait_for_event(event: asyncio.Event) -> None:
            """
            Waits for the event, or fails if it is not set in a reasonable time,
            so that requests that are not sent as expected do not hang the test.

            :param event: The event to wait for.
            :type event: asyncio.Event
            """
            await asyncio.wait_for(event.wait(), timeout=10)

        async def respond_in_turn(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to the request, and records how many requests are in progress
            at the same time. If several requests are expected to be in progress at
            the same time, the requests are responded to in the reverse order of
            their parts: the response to a request for the last part comes when
            all the requests are in progress, and the response to a request for
            each of the other parts when a request for the part after it is
            completed.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            requests_in_progress.append(request["input"])
            max_requests_in_progress.append(len(requests_in_progress))
            position = part_texts.index(request["input"])
            if n_requests_at_once[0] == 1:
                # Let any other request be sent, if the requests are sent concurrently
                for _ in range(3):
                    await asyncio.sleep(0)
            else:
                if len(requests_in_progress) == n_requests_at_once[0]:
                    events["all_sent"].set()
                await wait_for_event(
                    events["all_sent"]
                    if position == n_parts - 1
                    else events[position + 1]
                )
            requests_in_progress.remove(request["input"])
            completed_inputs.append(request["input"])
            events[position].set()
            return respond(request)

        def extract(
            n_requests_expected_at_once: int = 1, **extraction_kwargs: Any
        ) -> tuple[Document, _DecisionsAPIStub]:
            """
            Extracts the concepts of a new document with a new decision model,
            with up to two paragraphs per part.

            :param n_requests_expected_at_once: The number of requests expected to
                be in progress at the same time. Defaults to 1.
            :type n_requests_expected_at_once: int
            :param extraction_kwargs: Params of the extraction method.
            :type extraction_kwargs: Any
            :return: The document, and the stub with the requests received.
            :rtype: tuple[Document, _DecisionsAPIStub]
            """
            decision_model = _create_decision_model(async_limiter=AsyncLimiter(20, 5))
            document = _create_aspects_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    ),
                    RatingConcept(
                        name="Clarity",
                        description="Clarity of the text",
                        rating_scale=(1, 3),
                    ),
                ]
            )
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_in_turn]})
            max_requests_in_progress.clear()
            completed_inputs.clear()
            acquisitions.clear()
            # The events are used in the event loop of this extraction only
            n_requests_at_once[0] = n_requests_expected_at_once
            events.clear()
            events["all_sent"] = asyncio.Event()
            for position in range(n_parts):
                events[position] = asyncio.Event()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
            ):
                decision_model.extract_concepts_from_document(
                    document, max_paragraphs_to_analyze_per_call=2, **extraction_kwargs
                )
            # Each request is recorded as a call
            usage = decision_model.get_usage()[0].usage
            assert len(usage.calls) == len(decisions_stub.requests)
            return document, decisions_stub

        def get_items(document: Document) -> list[list[tuple[Any, Any]]]:
            """
            Returns the values and the probabilities of the items of each concept
            of the document.

            :param document: The document with the extracted concepts.
            :type document: Document
            :return: The items of each concept, in the order of the concepts.
            :rtype: list[list[tuple[Any, Any]]]
            """
            assert all(concept._is_processed for concept in document.concepts)
            return [
                [(item.value, item.probabilities) for item in concept.extracted_items]
                for concept in document.concepts
            ]

        # === Without concurrency ===
        # The requests for the parts are sent one after another, with no limiter
        document, decisions_stub = extract()
        assert [request["input"] for request in decisions_stub.requests] == part_texts
        assert max(max_requests_in_progress) == 1
        assert completed_inputs == part_texts
        assert acquisitions == []
        expected_items = get_items(document)
        assert [[value for value, _ in items] for items in expected_items] == [
            [True, False, True],
            [3, 1, 2],
        ]

        # === With concurrency ===
        # The requests for the parts are sent concurrently, each paced by
        # the limiter of the model. The items are in the order of the parts,
        # whatever the order in which the requests are completed.
        document, decisions_stub = extract(
            n_requests_expected_at_once=n_parts,
            use_concurrency=True,
            max_items_per_call=2,
        )
        assert [request["input"] for request in decisions_stub.requests] == part_texts
        assert all(
            len(request["questions"]) == 2 for request in decisions_stub.requests
        )
        assert max(max_requests_in_progress) == n_parts
        assert completed_inputs == part_texts[::-1]
        assert acquisitions == [(20, 5)] * n_parts
        assert get_items(document) == expected_items

        # By default, each concept is extracted in its own call with concurrency,
        # i.e. with its own requests for the parts
        document, decisions_stub = extract(
            n_requests_expected_at_once=2 * n_parts, use_concurrency=True
        )
        assert sorted(request["input"] for request in decisions_stub.requests) == (
            sorted(part_texts * 2)
        )
        assert all(
            [question["name"] for question in request["questions"]] == ["C0"]
            for request in decisions_stub.requests
        )
        assert max(max_requests_in_progress) == 2 * n_parts
        assert acquisitions == [(20, 5)] * 2 * n_parts
        assert get_items(document) == expected_items

    def test_decision_model_concept_part_failures(self):
        """
        Tests the extraction of concepts with a decision model in parts when
        the request for one of the parts gets a response that cannot be used, or
        fails for good, with and without concurrency: no further requests are
        sent, the requests made are counted in the usage, no concept gets its
        items, and a fallback model takes over, also for a refused answer, which
        raises ``ModelRefusalError`` without a fallback model. Before the fallback
        model, a response that cannot be used, or a refused answer, is asked for
        again with the request on the part that failed, and not on the part before
        it.
        """
        part_texts = self._get_texts_of_parts(2)
        n_parts = len(part_texts)
        failed_position = 1
        auth_error = _create_api_error(litellm.AuthenticationError)
        valid_answers: dict[str, Any] = {
            "NDA check": 0.9,
            "Topics": {"confidentiality": 0.8, "termination": 0.6},
        }

        def create_document() -> Document:
            """
            Creates a document with five paragraphs and the concepts to extract.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    ),
                    LabelConcept(
                        name="Topics",
                        description="Topics covered by the text",
                        labels=["confidentiality", "termination"],
                        classification_type="multi_label",
                    ),
                ]
            )

        def create_responder(
            outcomes: dict[int, Any] | None = None,
        ) -> Callable[[dict[str, Any]], OpenAIDecisionResponse]:
            """
            Creates the function that responds to the request for a part.

            :param outcomes: By the position of a part: an error to raise for
                its request, or the answers to respond with instead of the valid
                ones, by the names of the concepts. Defaults to None.
            :type outcomes: dict[int, Any] | None
            :return: Function that responds to a request.
            :rtype: Callable[[dict[str, Any]], OpenAIDecisionResponse]
            """
            outcomes = outcomes or {}
            errors = {
                position: outcome
                for position, outcome in outcomes.items()
                if isinstance(outcome, Exception)
            }
            respond = self._create_concepts_responder(
                part_texts,
                [
                    valid_answers
                    if position in errors
                    else outcomes.get(position, valid_answers)
                    for position in range(n_parts)
                ],
                input_tokens=30,
            )

            def respond_to_request(request: dict[str, Any]) -> OpenAIDecisionResponse:
                """
                Responds to the request for a part, or fails it.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response of litellm.
                :rtype: OpenAIDecisionResponse
                """
                error = errors.get(part_texts.index(request["input"]))
                if error is not None:
                    raise error
                return respond(request)

            return respond_to_request

        def check_usage(
            model: DocumentDecisionModel,
            finished_positions: list[int],
            failed_positions: list[int],
        ) -> None:
            """
            Checks the calls recorded in the usage of the decision model: first
            the requests that got a response, then the requests that failed.

            :param model: The decision model.
            :type model: DocumentDecisionModel
            :param finished_positions: The positions of the parts whose requests
                got a response, in the order of the calls.
            :type finished_positions: list[int]
            :param failed_positions: The positions of the parts whose requests
                failed, in the order of the calls.
            :type failed_positions: list[int]
            """
            usage = model.get_usage()[0].usage
            assert [json.loads(call.prompt)["input"] for call in usage.calls] == [
                part_texts[position]
                for position in [*finished_positions, *failed_positions]
            ]
            assert [call.timestamp_received is not None for call in usage.calls] == (
                [True] * len(finished_positions) + [False] * len(failed_positions)
            )
            assert usage.input == 30 * len(finished_positions)

        def check_not_extracted(document: Document) -> None:
            """
            Checks that no concept of the document is extracted.

            :param document: The document.
            :type document: Document
            """
            assert not any(concept._is_processed for concept in document.concepts)
            assert not any(concept.extracted_items for concept in document.concepts)

        # With immediate responses, the requests sent concurrently are completed in
        # the order they are sent, i.e. the outcome is the same in both modes:
        # the requests for the parts after the failed one are not sent
        for extraction_kwargs in [
            {"max_paragraphs_to_analyze_per_call": 2},
            {
                "max_paragraphs_to_analyze_per_call": 2,
                "use_concurrency": True,
                "max_items_per_call": 2,
            },
        ]:
            # === A response that cannot be used ===
            # The response to the request for one part makes the whole call
            # invalid: no further requests are sent, and each call is finished
            for invalid_answers, log_subject in [
                (
                    {**valid_answers, "NDA check": 1.5},
                    (
                        "concept `NDA check` on part 2 of 3 of the text, decision "
                        "question `C0`"
                    ),
                ),
                (
                    {
                        "NDA check": 0.1,
                        "Topics": {"confidentiality": 0.8, "termination": 1.5},
                    },
                    (
                        "concept `Topics` (label `termination`) on part 2 of 3 of "
                        "the text, decision question `C1-L1`"
                    ),
                ),
            ]:
                # One call, without retries for invalid data
                decision_model = _create_decision_model(max_retries_invalid_data=0)
                document = create_document()
                decisions_stub = _DecisionsAPIStub(
                    {
                        _DECISION_MODEL: [
                            create_responder({failed_position: invalid_answers})
                        ]
                    }
                )
                with (
                    patch.object(litellm, "adecisions", new=decisions_stub),
                    capture_logs(stdlib_logging.ERROR) as captured_logs,
                    pytest.raises(
                        ModelExtractionError,
                        match="Some concepts could not be processed due to "
                        "an invalid response returned by decision model. "
                        r"Failed concepts: \['NDA check', 'Topics'\]",
                    ) as extraction_exc_info,
                ):
                    decision_model.extract_concepts_from_document(
                        document,
                        **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                    )
                assert extraction_exc_info.value.retry_count == 0
                # The log names the concept that the answer is for, and the part
                # of the text that the question was asked on
                assert (
                    f"Decision model `{_DECISION_MODEL}` returned an invalid "
                    f"response for {log_subject}: the probability of the answer is not "
                    f"a number within [0, 1]."
                ) in captured_logs, captured_logs
                assert [request["input"] for request in decisions_stub.requests] == (
                    part_texts[: failed_position + 1]
                )
                check_usage(decision_model, [0, failed_position], [])
                check_not_extracted(document)

            # === A refused answer ===
            # Without a fallback model, the refusal is raised: no further requests
            # are sent, while each call is finished (with a fallback model, below)
            for refused_answers, log_subject in [
                (
                    {**valid_answers, "NDA check": "refusal"},
                    (
                        "concept `NDA check` on part 2 of 3 of the text, decision "
                        "question `C0`"
                    ),
                ),
                (
                    {
                        "NDA check": 0.1,
                        "Topics": {"confidentiality": 0.8, "termination": "refusal"},
                    },
                    (
                        "concept `Topics` (label `termination`) on part 2 of 3 of "
                        "the text, decision question `C1-L1`"
                    ),
                ),
            ]:
                # One call, without retries for invalid data
                decision_model = _create_decision_model(max_retries_invalid_data=0)
                document = create_document()
                decisions_stub = _DecisionsAPIStub(
                    {
                        _DECISION_MODEL: [
                            create_responder({failed_position: refused_answers})
                        ],
                    }
                )
                with (
                    patch.object(litellm, "adecisions", new=decisions_stub),
                    pytest.raises(ModelRefusalError) as refusal_exc_info,
                ):
                    decision_model.extract_concepts_from_document(
                        document,
                        **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                    )
                # The message names the question that was refused, and
                # the concepts of the call
                assert refusal_exc_info.value.message == (
                    f"Decision model `{_DECISION_MODEL}` refused to answer "
                    f"the question of {log_subject}. Failed concepts: "
                    f"['NDA check', 'Topics']"
                )
                assert refusal_exc_info.value.retry_count == 0
                assert refusal_exc_info.value.original_error is None
                assert decisions_stub.requested_models() == [_DECISION_MODEL] * (
                    failed_position + 1
                )
                check_usage(decision_model, [0, failed_position], [])
                # The refused request is recorded with its refusal
                assert [
                    call._refusal for call in decision_model.get_usage()[0].usage.calls
                ] == [None, "refusal"]
                check_not_extracted(document)

            # === A request that fails ===
            # No fallback model: a model API error is raised
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder({failed_position: auth_error})]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(ModelAPIError) as exc_info,
            ):
                decision_model.extract_concepts_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert exc_info.value.original_error is auth_error
            assert exc_info.value.retry_count == 0
            assert [request["input"] for request in decisions_stub.requests] == (
                part_texts[: failed_position + 1]
            )
            # The failed request is the last call, which is unfinished
            check_usage(decision_model, [0], [failed_position])
            check_not_extracted(document)

            # === A fallback decision model ===
            # It is sent the text in the same parts, and its answers are the ones
            # that the items are created from. Before it, the decision model is
            # asked again for a response that cannot be used, up to its
            # `max_retries_invalid_data` times (3 by default), with the request on
            # the part that failed only: the answers on the part before it are
            # kept. A failed request is not sent again.
            n_primary_calls = 1 + 3
            for primary_outcome, finished_positions, failed_positions in [
                (auth_error, [0], [failed_position]),
                (
                    {**valid_answers, "NDA check": 1.5},
                    [0, *[failed_position] * n_primary_calls],
                    [],
                ),
                # A refusal is asked for again, and passed on to the fallback model
                (
                    {**valid_answers, "NDA check": "refusal"},
                    [0, *[failed_position] * n_primary_calls],
                    [],
                ),
            ]:
                decision_model = _create_decision_model(
                    fallback_models=[
                        _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                    ]
                )
                document = create_document()
                decisions_stub = _DecisionsAPIStub(
                    {
                        _DECISION_MODEL: [
                            create_responder({failed_position: primary_outcome})
                        ],
                        _OTHER_DECISION_MODEL: [
                            create_responder({2: {**valid_answers, "NDA check": 0.4}})
                        ],
                    }
                )
                with patch.object(litellm, "adecisions", new=decisions_stub):
                    decision_model.extract_concepts_from_document(
                        document,
                        **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                    )
                primary_positions = [*finished_positions, *failed_positions]
                assert decisions_stub.requested_models() == (
                    [_DECISION_MODEL] * len(primary_positions)
                    + [_OTHER_DECISION_MODEL] * n_parts
                )
                assert [request["input"] for request in decisions_stub.requests] == [
                    *[part_texts[position] for position in primary_positions],
                    *part_texts,
                ]
                check_usage(decision_model, finished_positions, failed_positions)
                fallback_usage = decision_model.get_usage()[1].usage
                assert fallback_usage.input == 30 * n_parts
                assert len(fallback_usage.calls) == n_parts
                boolean_concept, label_concept = document.concepts
                assert [item.value for item in boolean_concept.extracted_items] == [
                    True,
                    True,
                    False,
                ]
                assert [item.value for item in label_concept.extracted_items] == [
                    ["confidentiality", "termination"]
                ] * n_parts

        # === A fallback LLM ===
        # The number of paragraphs per call applies to the LLM as to any LLM: it
        # is sent the same parts, in a call for each of them
        decision_model = _create_decision_model(
            fallback_models=[_create_llm(is_fallback=True)]
        )
        document = _create_aspects_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The text is of an NDA")
            ]
        )
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [create_responder({failed_position: auth_error})]}
        )
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": [true]}]']}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            decision_model.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )
        assert len(decisions_stub.requests) == failed_position + 1
        assert len(completions_stub.requests) == n_parts
        for request, part_text in zip(
            completions_stub.requests, part_texts, strict=True
        ):
            assert part_text in request["messages"][-1]["content"]
        items = document.concepts[0].extracted_items
        assert [item.value for item in items] == [True] * n_parts
        assert all(item.probabilities is None for item in items)

    def test_extract_all_in_parts_with_decision_model(self):
        """
        Tests the extraction of everything that a document has with a decision
        model when the max number of paragraphs to analyze per call is set:
        the aspects are extracted paragraph by paragraph, as without it, and
        the concepts of the document, of an aspect and of a sub-aspect are each
        extracted in the parts of their own text.
        """

        def create_document() -> Document:
            """
            Creates a document with aspects, a sub-aspect and concepts.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        concepts=[
                            BooleanConcept(
                                name="Termination for convenience",
                                description="Either party may terminate without cause",
                            )
                        ],
                        aspects=[
                            Aspect(
                                name="Effects of termination",
                                description="Clauses on what follows the termination",
                                concepts=[
                                    BooleanConcept(
                                        name="Survival",
                                        description="Some obligations survive "
                                        "the termination",
                                    )
                                ],
                            )
                        ],
                    ),
                    Aspect(
                        name="Payment",
                        description="Clauses on the payment of fees",
                        concepts=[
                            BooleanConcept(
                                name="Late fee",
                                description="A fee is charged on late payments",
                            )
                        ],
                    ),
                ],
                concepts=[
                    RatingConcept(
                        name="Clarity",
                        description="Clarity of the text",
                        rating_scale=(1, 3),
                    ),
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    ),
                ],
            )

        respond_on_aspects = _create_aspects_responder(
            {
                "Termination": {1: 0.8, 2: 0.9, 4: 0.7},
                "Effects of termination": {2: 0.6, 4: 0.8},
                "Payment": {},
            }
        )

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request with the questions of aspects, asked on
            a paragraph, or with the questions of concepts.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            if request["questions"][0]["name"].startswith("A"):
                return respond_on_aspects(request)
            return _answer_all_questions(request)

        def get_items(document: Document) -> dict[str, list[tuple[Any, Any]]]:
            """
            Returns the values and the probabilities of the items of each aspect,
            sub-aspect and concept of the document.

            :param document: The document.
            :type document: Document
            :return: The items by the names of the aspects and concepts.
            :rtype: dict[str, list[tuple[Any, Any]]]
            """
            instances: list[Any] = [*document.concepts]
            for aspect in document.aspects:
                instances.extend([aspect, *aspect.concepts])
                for sub_aspect in aspect.aspects:
                    instances.extend([sub_aspect, *sub_aspect.concepts])
            return {
                instance.name: [
                    (item.value, item.probabilities)
                    for item in instance.extracted_items
                ]
                for instance in instances
            }

        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            capture_logs() as captured_logs,
        ):
            decision_model.extract_all(document, max_paragraphs_to_analyze_per_call=2)

        # The paragraphs as they are sent for aspects, i.e. with their additional
        # context, and their texts, which the texts sent for concepts are made of
        paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        cleaned_texts = _ASPECTS_PARAGRAPH_TEXTS_CLEANED
        assert [
            (request["input"], [question["name"] for question in request["questions"]])
            for request in decisions_stub.requests
        ] == [
            # The aspects, on each paragraph of the document
            *[(text, ["A0", "A1"]) for text in paragraph_texts],
            # The sub-aspect, on each of the three paragraphs of its aspect
            *[(paragraph_texts[position], ["A0"]) for position in [1, 2, 4]],
            # The concept of the aspect, on the two parts of its three paragraphs
            (f"{cleaned_texts[1]}\n\n{cleaned_texts[2]}", ["C0"]),
            (cleaned_texts[4], ["C0"]),
            # The concept of the sub-aspect, on the text of its two paragraphs,
            # which are within the limit
            (f"{cleaned_texts[2]}\n\n{cleaned_texts[4]}", ["C0"]),
            # The concepts of the document, on the three parts of its paragraphs
            *[(text, ["C0", "C1"]) for text in self._get_texts_of_parts(2)],
        ]
        assert not any(
            "max_paragraphs_to_analyze_per_call" in log for log in captured_logs
        )
        paragraphs = document.paragraphs
        true_item = (True, {True: 0.9, False: 1 - 0.9})
        expected_items = {
            "Termination": [
                (paragraphs[1].raw_text, {True: 0.8, False: 1 - 0.8}),
                (paragraphs[2].raw_text, {True: 0.9, False: 1 - 0.9}),
                (paragraphs[4].raw_text, {True: 0.7, False: 1 - 0.7}),
            ],
            # An item for each part
            "Termination for convenience": [true_item] * 2,
            "Effects of termination": [
                (paragraphs[2].raw_text, {True: 0.6, False: 1 - 0.6}),
                (paragraphs[4].raw_text, {True: 0.8, False: 1 - 0.8}),
            ],
            "Survival": [true_item],
            # No paragraph belongs to the aspect, i.e. there is no text to extract
            # its concept from
            "Payment": [],
            "Late fee": [],
            "Clarity": [(1, {1: 1.0, 2: 0.0, 3: 0.0})] * 3,
            "NDA check": [true_item] * 3,
        }
        assert get_items(document) == expected_items
        assert len(decision_model.get_usage()[0].usage.calls) == 14
        self.check_instance_serialization_and_cloning(document)

        # The same is extracted with concurrency, with each aspect and each concept
        # extracted in its own call
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_all(
                document, use_concurrency=True, max_paragraphs_to_analyze_per_call=2
            )
        assert get_items(document) == expected_items
        assert all(
            len(request["questions"]) == 1 for request in decisions_stub.requests
        )
        # 2 aspects on 5 paragraphs, the sub-aspect on 3 paragraphs, the concept
        # of the aspect on 2 parts, the concept of the sub-aspect on its text,
        # and 2 concepts of the document on 3 parts
        assert len(decisions_stub.requests) == 2 * 5 + 3 + 2 + 1 + 2 * 3

    def test_decision_model_usage_and_cost(self):
        """
        Tests the usage and cost of a decision model: each request is recorded as
        a call, with its tokens, the request and the answers, and is priced with
        the pricing details of the model or automatically.
        """
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA"),
                LabelConcept(
                    name="Notice form",
                    description="Required form of the termination notice",
                    labels=["written", "any"],
                ),
            ]
        )
        answers = {"C0": 0.93, "C1": {"written": 0.75, "any": 0.25}}

        # === Pricing details ===
        decision_model = _create_decision_model(
            api_base="https://decisions.example.com",
            role="decide",
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.05,
                output_per_1m_tokens=2.0,
                cache_read_per_1m_tokens=0.01,
            ),
        )
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [answers]},
            input_tokens=1_000_000,
            output_tokens=1_000,
            cached_tokens=400_000,
            cache_write_tokens=100_000,
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(document)

        (usage_container,) = decision_model.get_usage()
        assert usage_container.model == _DECISION_MODEL
        assert usage_container.role == "decide"
        assert usage_container.is_fallback is False
        usage = usage_container.usage
        # The cached tokens are a part of the input tokens
        assert usage.input == 1_000_000
        assert usage.output == 1_000
        assert usage.cache_read == 400_000
        assert usage.cache_write == 100_000
        # The call: the request and the answers as JSON, without the credentials
        (call,) = usage.calls
        (request,) = decisions_stub.requests
        assert json.loads(call.prompt) == {
            "model": _DECISION_MODEL,
            "input": _DOCUMENT_TEXT_SENT,
            "questions": request["questions"],
        }
        assert "test_api_key" not in call.prompt
        assert "decisions.example.com" not in call.prompt
        assert call.prompt_kwargs == {}
        assert call.response is not None
        assert json.loads(call.response) == [
            {"type": "predicate", "name": "C0", "probability": 0.93},
            {
                "type": "choice",
                "name": "C1",
                "choice": "written",
                "probabilities": [
                    {"value": "written", "probability": 0.75},
                    {"value": "any", "probability": 0.25},
                ],
                "confidence": 0.75,
            },
        ]
        assert call.timestamp_received is not None
        assert call.timestamp_received >= call.timestamp_sent
        assert call._input_tokens == 1_000_000
        assert call._output_tokens == 1_000
        assert call._cache_read_tokens == 400_000
        assert call._cache_write_tokens == 100_000
        assert call._output_truncated is False
        # The cost: the tokens read from the cache at their own price, the tokens
        # written to the cache at the price of the input tokens
        (cost_container,) = decision_model.get_cost()
        assert cost_container.model == _DECISION_MODEL
        assert cost_container.role == "decide"
        assert cost_container.cost == _ModelCost(
            input=Decimal("0.034"), output=Decimal("0.002"), total=Decimal("0.036")
        )

        # The usage and cost are accumulated over the requests
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, overwrite_existing=True
            )
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 2_000_000
        assert usage.cache_read == 800_000
        assert len(usage.calls) == 2
        assert decision_model.get_cost()[0].cost.total == Decimal("0.072")
        # ... until they are reset
        decision_model.reset_usage_and_cost()
        assert decision_model.get_usage()[0].usage == _ModelUsage()
        assert decision_model.get_cost()[0].cost == _ModelCost()

        # === The counts of the cached tokens are capped at the input tokens ===
        decision_model = _create_decision_model()
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [answers]},
            input_tokens=100,
            cached_tokens=80,
            cache_write_tokens=50,
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, overwrite_existing=True
            )
        usage = decision_model.get_usage()[0].usage
        assert (usage.input, usage.cache_read, usage.cache_write) == (100, 80, 20)
        # No pricing params: no cost
        assert decision_model.get_cost()[0].cost == _ModelCost()

        # === Automatic pricing ===
        decision_model = _create_decision_model(auto_pricing=True)
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [answers]}, input_tokens=1_000_000
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, overwrite_existing=True
            )
        cost = decision_model.get_cost()[0].cost
        assert cost.input == Decimal("0.042")
        assert cost.output == Decimal(0)
        assert cost.total == Decimal("0.042")

    def test_decision_questions_split_over_request_cap(self):
        """
        Tests that the questions on a text that exceed the max number of questions
        that litellm accepts in a request are sent in several requests on
        the same text, and that their answers are merged.
        """
        # The cap is the one of litellm, whatever it is in the installed version
        n_labels = 4
        n_boolean_concepts = MAX_DECISION_QUESTIONS - 2
        # The questions of the label concept are split between two requests:
        # two of them are in the first request, with the questions of the Boolean
        # concepts before it, and the other two in the second request
        concepts: list[Any] = [
            BooleanConcept(name=f"Check {i}", description=f"Statement {i} is true")
            for i in range(n_boolean_concepts)
        ]
        concepts.append(
            LabelConcept(
                name="Topics",
                description="Topics covered by the document",
                labels=[f"topic {i}" for i in range(n_labels)],
                classification_type="multi_label",
            )
        )
        concepts.append(
            BooleanConcept(name="Last check", description="Last statement is true")
        )
        n_questions = n_boolean_concepts + n_labels + 1
        answers: dict[str, Any] = {
            **{
                # The Boolean concepts with even numbers are answered "yes"
                f"C{i}": 0.9 if i % 2 == 0 else 0.1
                for i in range(n_boolean_concepts)
            },
            **{
                # The labels with odd numbers apply
                f"C{n_boolean_concepts}-L{i}": 0.8 if i % 2 else 0.3
                for i in range(n_labels)
            },
            f"C{n_boolean_concepts + 1}": 0.6,
        }
        assert len(answers) == n_questions

        decision_model = _create_decision_model()
        document = _create_document(concepts=concepts)
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [answers]}, input_tokens=500
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(document)

        # Two requests on the same text, none over the cap, with all the questions
        assert len(decisions_stub.requests) == 2
        assert [len(request["questions"]) for request in decisions_stub.requests] == [
            MAX_DECISION_QUESTIONS,
            n_questions - MAX_DECISION_QUESTIONS,
        ]
        assert all(
            request["input"] == _DOCUMENT_TEXT_SENT
            for request in decisions_stub.requests
        )
        assert [
            question["name"]
            for request in decisions_stub.requests
            for question in request["questions"]
        ] == list(answers)
        # The answers are merged: each concept has its item
        for i in range(n_boolean_concepts):
            assert document.concepts[i].extracted_items[0].value is (i % 2 == 0)
        (item,) = document.concepts[n_boolean_concepts].extracted_items
        assert item.value == ["topic 1", "topic 3"]
        self.check_item_probabilities(
            item, {"topic 0": 0.3, "topic 1": 0.8, "topic 2": 0.3, "topic 3": 0.8}
        )
        assert document.concepts[n_boolean_concepts + 1].extracted_items[0].value
        # Each request is recorded as a call
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 1000
        assert len(usage.calls) == 2
        assert all(call.timestamp_received is not None for call in usage.calls)

        # If one of the requests fails, no concept is extracted, and the usage
        # of the request that succeeded is recorded with the failed call
        decision_model = _create_decision_model()
        document = _create_document(concepts=concepts)
        auth_error = _create_api_error(litellm.AuthenticationError)
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [answers, auth_error]}, input_tokens=500
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(ModelAPIError) as exc_info,
        ):
            decision_model.extract_concepts_from_document(document)
        assert exc_info.value.original_error is auth_error
        assert len(decisions_stub.requests) == 2
        assert not any(concept._is_processed for concept in document.concepts)
        assert not any(concept.extracted_items for concept in document.concepts)
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 500
        assert [call.timestamp_received is not None for call in usage.calls] == [
            True,
            False,
        ]

        # If the extraction is cancelled during one of the requests, e.g. on a timeout
        # set by the calling code, the usage of the request that succeeded is recorded
        async def extract_and_cancel(
            decision_model: DocumentDecisionModel, document: Document
        ) -> _DecisionsAPIStub:
            """
            Starts the extraction of the concepts of the document, and cancels it
            when the second request is sent, which gets no response.

            :param decision_model: Decision model to extract with.
            :type decision_model: DocumentDecisionModel
            :param document: Document to extract from.
            :type document: Document
            :return: The stub with the requests received.
            :rtype: _DecisionsAPIStub
            """
            second_request_sent = asyncio.Event()

            async def get_no_response(request: dict[str, Any]) -> None:
                """
                Signals that the request is sent, and waits for a response that
                does not come.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                """
                second_request_sent.set()
                await asyncio.sleep(3600)

            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [answers, get_no_response]}, input_tokens=500
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                extraction = asyncio.ensure_future(
                    decision_model.extract_concepts_from_document_async(document)
                )
                await second_request_sent.wait()
                extraction.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await extraction
            return decisions_stub

        decision_model = _create_decision_model()
        document = _create_document(concepts=concepts)
        decisions_stub = asyncio.run(extract_and_cancel(decision_model, document))
        assert len(decisions_stub.requests) == 2
        assert not any(concept._is_processed for concept in document.concepts)
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 500
        (call,) = usage.calls
        assert call.timestamp_received is not None

    def test_decision_model_api_request_retries(self):
        """
        Tests for retrying failed requests to a decision model: only transient
        errors are retried, with a backoff between the retries, and an error that
        is final is raised as a model API error, or issued as a warning.
        """
        rate_limit_error = _create_api_error(litellm.RateLimitError)
        auth_error = _create_api_error(litellm.AuthenticationError)
        answers = {"C0": 0.93}
        retry_numbers: list[int] = []
        # The stub, the document and the decision model of the last extraction,
        # to be checked when the extraction raises an exception
        last_extraction: list[
            tuple[_DecisionsAPIStub, Document, DocumentDecisionModel]
        ] = []

        def retry_delay_stub(retry_number: int) -> float:
            """
            Records the retry and removes the delay before it.

            :param retry_number: The number of the retry to be performed.
            :type retry_number: int
            :return: No delay.
            :rtype: float
            """
            retry_numbers.append(retry_number)
            return 0.0

        def create_document(n_concepts: int = 1) -> Document:
            """
            Creates a document with the specified number of concepts to extract.

            :param n_concepts: Number of concepts. Defaults to 1.
            :type n_concepts: int
            :return: Document.
            :rtype: Document
            """
            return _create_document(
                concepts=[
                    BooleanConcept(
                        name=f"Check {i}", description=f"Statement {i} is true"
                    )
                    for i in range(n_concepts)
                ]
            )

        def extract(
            outcomes: list[Any],
            document: Document | None = None,
            decision_model: DocumentDecisionModel | None = None,
            **extraction_kwargs: Any,
        ) -> tuple[_DecisionsAPIStub, Document, DocumentDecisionModel]:
            """
            Extracts the concepts of a document with a decision model, with
            the given outcomes of the requests to the model.

            :param outcomes: Outcomes of the requests, in the order of the requests.
            :type outcomes: list[Any]
            :param document: Document to extract from. Defaults to a new document
                with one concept.
            :type document: Document | None
            :param decision_model: Decision model to extract with. Defaults to
                a new decision model with the default params.
            :type decision_model: DocumentDecisionModel | None
            :param extraction_kwargs: Other params of the extraction method.
            :type extraction_kwargs: Any
            :return: The stub with the requests received, the document and
                the decision model.
            :rtype: tuple[_DecisionsAPIStub, Document, DocumentDecisionModel]
            """
            document = document or create_document()
            decision_model = decision_model or _create_decision_model()
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: outcomes})
            retry_numbers.clear()
            last_extraction[:] = [(decisions_stub, document, decision_model)]
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_concepts_from_document(
                    document, **extraction_kwargs
                )
            return decisions_stub, document, decision_model

        with patch.object(
            decision_models_module, "_get_model_api_retry_delay", new=retry_delay_stub
        ):
            # Transient errors are retried until the request succeeds
            for error_class in [
                litellm.RateLimitError,
                litellm.Timeout,
                litellm.InternalServerError,
                litellm.ServiceUnavailableError,
                litellm.APIConnectionError,
            ]:
                error = _create_api_error(error_class)
                decisions_stub, document, decision_model = extract(
                    [error, error, answers]
                )
                assert len(decisions_stub.requests) == 3
                assert retry_numbers == [1, 2]
                assert document.concepts[0].extracted_items[0].value is True
                # Only the request that succeeded is recorded as a call
                usage = decision_model.get_usage()[0].usage
                assert len(usage.calls) == 1
                assert usage.calls[0].timestamp_received is not None
            # ... including a timeout error that litellm sets as its own cause
            timeout_error = _create_api_error(litellm.Timeout)
            timeout_error.__cause__ = timeout_error
            decisions_stub, document, _ = extract([timeout_error, answers])
            assert len(decisions_stub.requests) == 2
            assert document.concepts[0].extracted_items[0].value is True

            # ...or until the retries are exhausted
            for num_retries in [3, 1, 0]:
                decision_model = _create_decision_model(
                    num_retries_failed_request=num_retries
                )
                with pytest.raises(ModelAPIError) as exc_info:
                    extract([rate_limit_error], decision_model=decision_model)
                assert exc_info.value.retry_count == num_retries
                assert exc_info.value.original_error is rate_limit_error
                assert "calling decision model API" in exc_info.value.message
                decisions_stub, document, _ = last_extraction[0]
                assert len(decisions_stub.requests) == 1 + num_retries
                assert retry_numbers == list(range(1, num_retries + 1))
                assert not document.concepts[0]._is_processed
                # The failed request is recorded as one unfinished call
                usage = decision_model.get_usage()[0].usage
                assert usage.input == 0
                (call,) = usage.calls
                assert call.timestamp_received is None
                assert call.response is None
            with pytest.raises(ModelAPIError) as exc_info:
                extract([timeout_error])
            assert exc_info.value.original_error is timeout_error
            assert exc_info.value.retry_count == 3

            # Non-transient errors are not retried
            for error_class in [
                litellm.AuthenticationError,
                litellm.BadRequestError,
                litellm.NotFoundError,
            ]:
                error = _create_api_error(error_class)
                with pytest.raises(ModelAPIError) as exc_info:
                    extract([error])
                assert exc_info.value.retry_count == 0
                assert exc_info.value.original_error is error
                assert len(last_extraction[0][0].requests) == 1
                assert retry_numbers == []
            # ... also when they follow a transient error
            with pytest.raises(ModelAPIError) as exc_info:
                extract([rate_limit_error, auth_error])
            assert exc_info.value.retry_count == 1
            assert exc_info.value.original_error is auth_error
            assert len(last_extraction[0][0].requests) == 2

            # With concurrency, a final API error is raised as it is, without
            # sending the failed requests again
            with pytest.raises(ModelAPIError) as exc_info:
                extract(
                    [auth_error],
                    document=create_document(n_concepts=3),
                    use_concurrency=True,
                )
            assert exc_info.value.original_error is auth_error
            assert len(last_extraction[0][0].requests) == 3  # one per concept

            # A warning is logged if `raise_exception_on_extraction_error` is False,
            # without sending the failed request again: only the warning on
            # the failed request, as the model returned no response to report
            for extraction_kwargs in [{}, {"use_concurrency": True}]:
                with capture_logs(stdlib_logging.WARNING) as captured_logs:
                    decisions_stub, document, decision_model = extract(
                        [rate_limit_error],
                        document=create_document(n_concepts=2),
                        raise_exception_on_extraction_error=False,
                        max_items_per_call=1,
                        **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                    )
                # The warnings tell the retries, unlike the errors logged before them
                warning_messages = [log for log in captured_logs if "retries)" in log]
                assert len(warning_messages) == 2  # one per concept, i.e. per call
                for warning_message in warning_messages:
                    assert "calling decision model API" in warning_message
                    assert "(3 retries)" in warning_message
                assert len(decisions_stub.requests) == 2 * 4
                assert not any(concept._is_processed for concept in document.concepts)
                assert not any(concept.extracted_items for concept in document.concepts)
                usage = decision_model.get_usage()[0].usage
                assert [call.timestamp_received for call in usage.calls] == [None, None]

        # The same applies to an LLM that is not asked again when it returns invalid
        # data: its failed request is reported once, as a failed request
        llm = _create_llm(max_retries_invalid_data=0)
        llm_auth_error = _create_api_error(litellm.AuthenticationError, _LLM_MODEL)
        for extraction_kwargs in [{}, {"use_concurrency": True}]:
            document = create_document(n_concepts=2)
            completions_stub = _CompletionsAPIStub({_LLM_MODEL: [llm_auth_error]})
            with (
                patch.object(litellm, "acompletion", new=completions_stub),
                capture_logs(stdlib_logging.WARNING) as captured_logs,
            ):
                llm.extract_concepts_from_document(
                    document,
                    raise_exception_on_extraction_error=False,
                    max_items_per_call=1,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            # The warnings tell the retries, unlike the errors logged before them
            warning_messages = [log for log in captured_logs if "retries)" in log]
            assert len(warning_messages) == 2  # one per concept, i.e. per call
            for warning_message in warning_messages:
                assert "calling LLM API" in warning_message
            assert completions_stub.requested_models() == [_LLM_MODEL] * 2
            assert not any(concept._is_processed for concept in document.concepts)

    def test_invalid_decision_model_responses(self):
        """
        Tests that a response of a decision model that cannot be used, such as
        an answer of another kind than its question, fails the extraction as
        invalid data returned by the model, and that a refusal to answer
        a question fails it as a refusal, also after the request is sent again
        ``max_retries_invalid_data`` times.
        """
        concepts = [
            BooleanConcept(name="NDA check", description="The document is an NDA"),
            LabelConcept(
                name="Notice form",
                description="Required form of the termination notice",
                labels=["written", "email"],
            ),
            RatingConcept(
                name="Clarity",
                description="Clarity of the text of the document",
                rating_scale=(1, 3),
            ),
        ]
        valid_answers: dict[str, Any] = {
            "C0": 0.9,
            "C1": {"written": 0.7, "email": 0.3},
            "C2": {"1": 0.2, "2": 0.5, "3": 0.3},
        }

        def create_response_with_answers(
            get_answers: Callable[[list[Any]], list[Any]],
        ) -> Callable[[dict[str, Any]], OpenAIDecisionResponse]:
            """
            Creates the function that responds to a request with the valid answers
            changed by the given function.

            :param get_answers: Function that takes the valid answers of
                the response and returns the answers to respond with.
            :type get_answers: Callable[[list[Any]], list[Any]]
            :return: Function that creates the response for a request.
            :rtype: Callable[[dict[str, Any]], OpenAIDecisionResponse]
            """

            def create_response(request: dict[str, Any]) -> OpenAIDecisionResponse:
                """
                Creates the response for the request.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response of litellm.
                :rtype: OpenAIDecisionResponse
                """
                response = _create_decisions_response(
                    request["questions"], valid_answers, input_tokens=300
                )
                return OpenAIDecisionResponse(
                    model=response.model,
                    answers=tuple(get_answers(list(response.answers))),
                    usage=response.usage,
                )

            return create_response

        invalid_outcomes: list[tuple[Any, str]] = [
            # An answer of another kind than its question
            (
                {
                    **valid_answers,
                    "C0": OpenAIChoiceAnswer(
                        name="C0",
                        choice="yes",
                        confidence=0.9,
                        probabilities=(
                            OpenAIChoiceProbability(value="yes", probability=0.9),
                            OpenAIChoiceProbability(value="no", probability=0.1),
                        ),
                    ),
                },
                (
                    "`C0`: the answer is of the kind `choice`, while the question is of "
                    "the kind `predicate`"
                ),
            ),
            (
                {
                    **valid_answers,
                    "C2": OpenAIPredicateAnswer(name="C2", probability=0.5),
                },
                (
                    "`C2`: the answer is of the kind `predicate`, while the question "
                    "is of the kind `score`"
                ),
            ),
            # A probability that is not within [0, 1]
            (
                {**valid_answers, "C0": 1.5},
                r"`C0`: the probability of the answer is not",
            ),
            # An option that was not offered, or an option offered that is missing
            (
                {**valid_answers, "C1": {"written": 0.7, "fax": 0.3}},
                (
                    r"`C1`: the answer has options that were not offered\. Options offered: "
                    r"\['written', 'email'\]\."
                ),
            ),
            (
                {**valid_answers, "C1": {"written": 1.0}},
                r"`C1`: the answer has no probabilities of the options \['email'\]",
            ),
            # The levels of a rating keyed by their labels, not by their positions:
            # a level with the label of another scale is not a level offered
            (
                {**valid_answers, "C2": {"0": 0.2, "1": 0.5, "2": 0.3}},
                (
                    r"`C2`: the answer has options that were not offered\. Options offered: "
                    r"\['1', '2', '3'\]\."
                ),
            ),
            # An option repeated
            (
                {
                    **valid_answers,
                    "C1": OpenAIChoiceAnswer(
                        name="C1",
                        choice="written",
                        confidence=0.6,
                        probabilities=(
                            OpenAIChoiceProbability(value="written", probability=0.6),
                            OpenAIChoiceProbability(value="written", probability=0.4),
                        ),
                    ),
                },
                "`C1`: the answer repeats an option.",
            ),
            # No usable probabilities: a zero for each option, as litellm fills
            # in for the options that the provider returns no probability for
            (
                {**valid_answers, "C1": {"written": 0.0, "email": 0.0}},
                "`C1`: the probability of each option is zero",
            ),
            (
                {**valid_answers, "C2": {"1": 0.0, "2": 0.0, "3": 0.0}},
                "`C2`: the probability of each option is zero",
            ),
            # The option picked is not one of the options offered
            (
                {
                    **valid_answers,
                    "C1": OpenAIChoiceAnswer(
                        name="C1",
                        choice="Written",
                        confidence=0.7,
                        probabilities=(
                            OpenAIChoiceProbability(value="written", probability=0.7),
                            OpenAIChoiceProbability(value="email", probability=0.3),
                        ),
                    ),
                },
                (
                    r"`C1`: the answer picks an option that was not offered\. Options offered: "
                    r"\['written', 'email'\]\."
                ),
            ),
            # Fewer or more answers than questions
            (
                create_response_with_answers(
                    lambda response_answers: response_answers[:2]
                ),
                "`C2`: the response has 2 answers, while 3 questions were asked",
            ),
            (
                create_response_with_answers(
                    lambda response_answers: [*response_answers, response_answers[0]]
                ),
                "`C2`: the response has 4 answers, while 3 questions were asked",
            ),
            # The answers in another order than the questions
            (
                create_response_with_answers(
                    lambda response_answers: [
                        response_answers[1],
                        response_answers[0],
                        response_answers[2],
                    ]
                ),
                "`C0`: the answer at the position of the question has another name.",
            ),
        ]

        for outcome, error_match in invalid_outcomes:
            # One call, without retries for invalid data
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = _create_document(concepts=concepts)
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [outcome]}, input_tokens=300
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.ERROR) as captured_logs,
                pytest.raises(
                    ModelExtractionError,
                    match="Some concepts could not be processed due to "
                    "an invalid response returned by decision model. "
                    r"Failed concepts: \['NDA check', 'Notice form', 'Clarity'\]",
                ) as exc_info,
            ):
                decision_model.extract_concepts_from_document(document)
            assert exc_info.value.retry_count == 0
            # What is wrong with the response is logged, with the concept that
            # the answer is for
            question_name_match = re.match(r"`C(\d)`", error_match)
            assert question_name_match is not None
            concept_name = concepts[int(question_name_match.group(1))].name
            assert any(
                f"Decision model `{_DECISION_MODEL}` returned an invalid response "
                f"for concept `{concept_name}`, decision question "
                in log
                and re.search(error_match, log)
                for log in captured_logs
            ), captured_logs
            # One request, which is not sent again without retries
            assert len(decisions_stub.requests) == 1
            # One answer that cannot be used makes the whole response invalid
            assert not any(concept._is_processed for concept in document.concepts)
            assert not any(concept.extracted_items for concept in document.concepts)
            # The request succeeded, i.e. its call is recorded as finished,
            # with its tokens
            usage = decision_model.get_usage()[0].usage
            assert usage.input == 300
            (call,) = usage.calls
            assert call.timestamp_received is not None
            assert call.response is not None

        # With concurrency, each concept is extracted in its own request: the valid
        # responses are used, and the extraction fails for the first invalid one.
        # One call per concept, without retries for invalid data.
        decision_model = _create_decision_model(max_retries_invalid_data=0)
        document = _create_document(concepts=concepts)
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    lambda request: _create_decisions_response(
                        request["questions"],
                        # The question of each concept is the first in its request
                        {
                            "C0": OpenAIPredicateAnswer(name="C0", probability=0.5)
                            if request["questions"][0]["type"] != "predicate"
                            else 0.9
                        },
                    )
                ]
            }
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(
                ModelExtractionError,
                match="an invalid response returned by decision model. "
                r"Failed concepts: \['Notice form'\]",
            ) as exc_info,
        ):
            decision_model.extract_concepts_from_document(
                document, use_concurrency=True
            )
        assert exc_info.value.retry_count == 0
        assert len(decisions_stub.requests) == 3
        assert document.concepts[0].extracted_items[0].value is True
        assert not document.concepts[1]._is_processed
        assert not document.concepts[2]._is_processed
        usage = decision_model.get_usage()[0].usage
        assert len(usage.calls) == 3
        assert all(call.timestamp_received is not None for call in usage.calls)

        # A warning is logged if `raise_exception_on_extraction_error` is False,
        # after the request is sent again `max_retries_invalid_data` times (3 by
        # default), as the response cannot be used each time
        decision_model = _create_decision_model()
        document = _create_document(concepts=concepts)
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [invalid_outcomes[0][0]]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            capture_logs(stdlib_logging.WARNING) as captured_logs,
        ):
            decision_model.extract_concepts_from_document(
                document, raise_exception_on_extraction_error=False
            )
        assert any(
            re.search(
                r"an invalid response returned by decision model.*\(3 retries\)", log
            )
            for log in captured_logs
        ), captured_logs
        assert len(decisions_stub.requests) == 1 + 3
        assert not any(concept._is_processed for concept in document.concepts)

        # A refused answer fails the extraction as a refusal, without a fallback
        # model, once it is refused again on each retry: the refusal is logged
        # with the concepts of the call, and raised, or warned about, with
        # the concepts that failed and the retries
        n_requests = 1 + 3
        for refused_answers, refused_concept, question_name in [
            ({**valid_answers, "C1": "refusal"}, "Notice form", "C1"),
            ({**valid_answers, "C0": "refusal"}, "NDA check", "C0"),
        ]:
            refusal_sentence = (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question "
                f"of concept `{refused_concept}`, decision question `{question_name}`."
            )
            concept_names = "['NDA check', 'Notice form', 'Clarity']"
            refusal_msg = f"{refusal_sentence} Failed concepts: {concept_names}"
            for raise_exception in [True, False]:
                decision_model = _create_decision_model()
                document = _create_document(concepts=concepts)
                decisions_stub = _DecisionsAPIStub(
                    {_DECISION_MODEL: [refused_answers]}, input_tokens=300
                )
                with (
                    patch.object(litellm, "adecisions", new=decisions_stub),
                    capture_logs(stdlib_logging.WARNING) as captured_logs,
                ):
                    if raise_exception:
                        # Caught as a ModelExtractionError, as before refusals
                        # had an error of their own
                        with pytest.raises(ModelExtractionError) as refusal_exc_info:
                            decision_model.extract_concepts_from_document(document)
                        assert isinstance(refusal_exc_info.value, ModelRefusalError)
                        assert refusal_exc_info.value.message == refusal_msg
                        assert refusal_exc_info.value.retry_count == 3
                    else:
                        decision_model.extract_concepts_from_document(
                            document, raise_exception_on_extraction_error=False
                        )
                        assert (
                            f"{refusal_msg} (3 retries). If you want to raise an "
                            f"exception instead, set "
                            f"`raise_exception_on_extraction_error` to True."
                        ) in captured_logs, captured_logs
                    # Logged for each request refused
                    assert (
                        captured_logs.count(
                            f"{refusal_sentence} Concepts of the call: {concept_names}"
                        )
                        == n_requests
                    ), captured_logs
                assert len(decisions_stub.requests) == n_requests
                assert not any(concept._is_processed for concept in document.concepts)
                # The requests succeeded, i.e. their calls are recorded as finished,
                # with their tokens and their refusals
                usage = decision_model.get_usage()[0].usage
                assert usage.input == 300 * n_requests
                assert len(usage.calls) == n_requests
                assert all(call.timestamp_received is not None for call in usage.calls)
                assert all(call._refusal == "refusal" for call in usage.calls)

        # The valid answers are accepted
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [valid_answers]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(document)
        assert [concept.extracted_items[0].value for concept in document.concepts] == [
            True,
            ["written"],
            2,
        ]

    def test_decision_model_fallback_chains(self):
        """
        Tests the fallback chains with models of different kinds: a decision model
        that falls back to an LLM or to another decision model, and an LLM that
        falls back to a decision model, for a failed request and for a response
        that cannot be used, which each model is asked again for, up to its
        ``max_retries_invalid_data`` times, before the next model is tried.
        A refusal is passed on along the chain as a response that cannot be used
        is, and raised if the last model refuses too.
        """
        llm_answer = '[{"concept_id": "C0", "extracted_items": [true]}]'
        invalid_llm_answer = "<<<not-json>>>"
        answers = {"C0": 0.93}
        # An answer that cannot be used: a probability that is not within [0, 1]
        invalid_answers = {"C0": 1.5}
        auth_error = _create_api_error(litellm.AuthenticationError)
        rate_limit_error = _create_api_error(litellm.RateLimitError)
        llm_auth_error = _create_api_error(litellm.AuthenticationError, _LLM_MODEL)

        def create_document(n_concepts: int = 1) -> Document:
            """
            Creates a document with the specified number of concepts to extract.

            :param n_concepts: Number of concepts. Defaults to 1.
            :type n_concepts: int
            :return: Document.
            :rtype: Document
            """
            return _create_document(
                concepts=[
                    BooleanConcept(
                        name=f"Check {i}", description=f"Statement {i} is true"
                    )
                    for i in range(n_concepts)
                ]
            )

        def extract(
            model: DocumentDecisionModel | DocumentLLM,
            decisions_outcomes: dict[str, list[Any]],
            completions_outcomes: dict[str, list[Any]],
            document: Document | None = None,
            **extraction_kwargs: Any,
        ) -> tuple[Document, list[str]]:
            """
            Extracts the concepts of a document with a model that has fallback
            models, with the given outcomes of the requests to the models.

            :param model: Model to extract with.
            :type model: DocumentDecisionModel | DocumentLLM
            :param decisions_outcomes: Outcomes of the requests to each decision
                model, in the order of the requests.
            :type decisions_outcomes: dict[str, list[Any]]
            :param completions_outcomes: Outcomes of the requests to each LLM,
                in the order of the requests.
            :type completions_outcomes: dict[str, list[Any]]
            :param document: Document to extract from. Defaults to a new document
                with one concept.
            :type document: Document | None
            :param extraction_kwargs: Other params of the extraction method.
            :type extraction_kwargs: Any
            :return: The document, and the models in the order the requests were
                sent to them.
            :rtype: tuple[Document, list[str]]
            """
            document = document or create_document()
            requested_models.clear()

            async def decisions_stub(**kwargs: Any) -> Any:
                """
                Records the model of the request to a decision model and returns
                the next outcome for it.

                :param kwargs: Keyword arguments of ``litellm.adecisions``.
                :type kwargs: Any
                :return: Response of litellm.
                :rtype: Any
                """
                requested_models.append(kwargs["model"])
                return await decisions_api_stub(**kwargs)

            async def completions_stub(**kwargs: Any) -> Any:
                """
                Records the model of the request to an LLM and returns the next
                outcome for it.

                :param kwargs: Keyword arguments of ``litellm.acompletion``.
                :type kwargs: Any
                :return: Completion object.
                :rtype: Any
                """
                requested_models.append(kwargs["model"])
                return await completions_api_stub(**kwargs)

            decisions_api_stub = _DecisionsAPIStub(decisions_outcomes)
            completions_api_stub = _CompletionsAPIStub(completions_outcomes)
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(litellm, "acompletion", new=completions_stub),
                patch.object(
                    decision_models_module,
                    "_get_model_api_retry_delay",
                    new=lambda retry_number: 0.0,
                ),
            ):
                model.extract_concepts_from_document(document, **extraction_kwargs)
            return document, requested_models

        # The models in the order the requests were sent to them in the last
        # extraction, to be checked also when the extraction raises an exception
        requested_models: list[str] = []
        # The requests of a decision model whose responses cannot be used:
        # the first one, and one with each of its 3 retries (the default)
        n_decision_model_calls = 1 + 3

        # === A decision model with an LLM as its fallback model ===
        def create_decision_model_with_llm_fallback(
            **kwargs: Any,
        ) -> DocumentDecisionModel:
            """
            Creates a decision model with a fallback LLM.

            :param kwargs: Params of the decision model.
            :type kwargs: Any
            :return: Decision model.
            :rtype: DocumentDecisionModel
            """
            return _create_decision_model(
                fallback_models=[
                    _create_llm(is_fallback=True, max_retries_invalid_data=2)
                ],
                **kwargs,
            )

        # The request to the decision model fails with an error that is not retried
        decision_model = create_decision_model_with_llm_fallback()
        document, _ = extract(
            decision_model, {_DECISION_MODEL: [auth_error]}, {_LLM_MODEL: [llm_answer]}
        )
        assert requested_models == [_DECISION_MODEL, _LLM_MODEL]
        (item,) = document.concepts[0].extracted_items
        assert item.value is True
        # An item extracted by an LLM has no probabilities
        assert item.probabilities is None
        # The usage of each model: the failed request, and the request to the LLM
        usage_containers = decision_model.get_usage()
        assert [
            (container.model, container.is_fallback) for container in usage_containers
        ] == [(_DECISION_MODEL, False), (_LLM_MODEL, True)]
        (failed_call,) = usage_containers[0].usage.calls
        assert failed_call.timestamp_received is None
        assert usage_containers[1].usage.input == 10
        assert len(usage_containers[1].usage.calls) == 1
        assert [container.model for container in decision_model.get_cost()] == [
            _DECISION_MODEL,
            _LLM_MODEL,
        ]
        decision_model.reset_usage_and_cost()
        assert all(
            container.usage == _ModelUsage() for container in decision_model.get_usage()
        )

        # ... or with a transient error, after the retries of the request
        document, _ = extract(
            create_decision_model_with_llm_fallback(num_retries_failed_request=2),
            {_DECISION_MODEL: [rate_limit_error]},
            {_LLM_MODEL: [llm_answer]},
        )
        assert requested_models == [_DECISION_MODEL] * 3 + [_LLM_MODEL]
        assert document.concepts[0].extracted_items[0].value is True

        # The response of the decision model cannot be used: the request is sent
        # to it again, up to its `max_retries_invalid_data` times, and then
        # the fallback LLM is tried
        decision_model = create_decision_model_with_llm_fallback()
        document, _ = extract(
            decision_model,
            {_DECISION_MODEL: [invalid_answers]},
            {_LLM_MODEL: [llm_answer]},
        )
        assert requested_models == [_DECISION_MODEL] * n_decision_model_calls + [
            _LLM_MODEL
        ]
        assert document.concepts[0].extracted_items[0].value is True
        assert document.concepts[0].extracted_items[0].probabilities is None
        invalid_calls = decision_model.get_usage()[0].usage.calls
        assert len(invalid_calls) == n_decision_model_calls
        assert all(call.timestamp_received is not None for call in invalid_calls)

        # The fallback LLM is asked again when it returns invalid data, as an LLM is
        document, _ = extract(
            create_decision_model_with_llm_fallback(),
            {_DECISION_MODEL: [invalid_answers]},
            {_LLM_MODEL: [invalid_llm_answer, llm_answer]},
        )
        assert requested_models == [_DECISION_MODEL] * n_decision_model_calls + [
            _LLM_MODEL,
            _LLM_MODEL,
        ]
        assert document.concepts[0].extracted_items[0].value is True
        # ... as many times as it has retries for invalid data, i.e. twice after
        # its first call, and the error is the one of the LLM when it is the last
        # model to fail, with the number of the retries of the LLM
        with pytest.raises(
            ModelExtractionError,
            match=r"invalid data returned by LLM.*\(after 2 retries\)",
        ) as extraction_exc_info:
            extract(
                create_decision_model_with_llm_fallback(),
                {_DECISION_MODEL: [invalid_answers]},
                {_LLM_MODEL: [invalid_llm_answer]},
            )
        assert extraction_exc_info.value.retry_count == 2
        assert (
            requested_models
            == [_DECISION_MODEL] * n_decision_model_calls + [_LLM_MODEL] * 3
        )
        with pytest.raises(ModelAPIError, match="calling LLM API") as exc_info:
            extract(
                create_decision_model_with_llm_fallback(),
                {_DECISION_MODEL: [auth_error]},
                {_LLM_MODEL: [llm_auth_error]},
            )
        assert exc_info.value.original_error is llm_auth_error
        assert requested_models == [_DECISION_MODEL, _LLM_MODEL]

        # With concurrency, each call falls back on its own
        document, _ = extract(
            create_decision_model_with_llm_fallback(),
            {_DECISION_MODEL: [auth_error]},
            {_LLM_MODEL: [llm_answer]},
            document=create_document(n_concepts=3),
            use_concurrency=True,
        )
        assert sorted(requested_models) == sorted(
            [_DECISION_MODEL] * 3 + [_LLM_MODEL] * 3
        )
        assert all(
            concept.extracted_items[0].value is True for concept in document.concepts
        )

        # === A decision model with another decision model as its fallback model ===
        def create_decision_model_with_decision_model_fallback() -> (
            DocumentDecisionModel
        ):
            """
            Creates a decision model with a fallback decision model.

            :return: Decision model.
            :rtype: DocumentDecisionModel
            """
            return _create_decision_model(
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ]
            )

        for primary_outcome, n_primary_requests in [
            # A failed request is not sent again
            (auth_error, 1),
            (invalid_answers, n_decision_model_calls),
        ]:
            decision_model = create_decision_model_with_decision_model_fallback()
            document, _ = extract(
                decision_model,
                {_DECISION_MODEL: [primary_outcome], _OTHER_DECISION_MODEL: [answers]},
                {},
            )
            assert requested_models == [_DECISION_MODEL] * n_primary_requests + [
                _OTHER_DECISION_MODEL
            ]
            (item,) = document.concepts[0].extracted_items
            assert item.value is True
            self.check_item_probabilities(item, {True: 0.93, False: 1 - 0.93})
            usage_containers = decision_model.get_usage()
            assert [container.model for container in usage_containers] == [
                _DECISION_MODEL,
                _OTHER_DECISION_MODEL,
            ]
            assert usage_containers[1].usage.input == 100
        # The fallback decision model is asked again too, after its first call,
        # and the error is the one of the last model to fail, with its retries
        with pytest.raises(
            ModelExtractionError,
            match=(
                r"an invalid response returned by decision model"
                r".*\(after 3 retries\)"
            ),
        ) as extraction_exc_info:
            extract(
                create_decision_model_with_decision_model_fallback(),
                {
                    _DECISION_MODEL: [auth_error],
                    _OTHER_DECISION_MODEL: [invalid_answers],
                },
                {},
            )
        assert extraction_exc_info.value.retry_count == 3
        assert (
            requested_models
            == [_DECISION_MODEL] + [_OTHER_DECISION_MODEL] * n_decision_model_calls
        )
        other_auth_error = _create_api_error(
            litellm.AuthenticationError, _OTHER_DECISION_MODEL
        )
        with pytest.raises(ModelAPIError, match="decision model API") as exc_info:
            extract(
                create_decision_model_with_decision_model_fallback(),
                {
                    _DECISION_MODEL: [invalid_answers],
                    _OTHER_DECISION_MODEL: [other_auth_error],
                },
                {},
            )
        assert exc_info.value.original_error is other_auth_error
        assert requested_models == [_DECISION_MODEL] * n_decision_model_calls + [
            _OTHER_DECISION_MODEL
        ]

        # === An LLM with a decision model as its fallback model ===
        def create_llm_with_decision_model_fallback() -> DocumentLLM:
            """
            Creates an LLM with a fallback decision model.

            :return: LLM.
            :rtype: DocumentLLM
            """
            return _create_llm(
                max_retries_invalid_data=1,
                fallback_models=[_create_decision_model(is_fallback=True)],
            )

        llm_outcomes_and_requested_models: list[
            tuple[list[Exception | str], list[str]]
        ] = [
            # The request to the LLM fails
            ([llm_auth_error], [_LLM_MODEL, _DECISION_MODEL]),
            # The LLM returns invalid data, also when asked again
            ([invalid_llm_answer], [_LLM_MODEL, _LLM_MODEL, _DECISION_MODEL]),
        ]
        for (
            llm_outcomes,
            expected_requested_models,
        ) in llm_outcomes_and_requested_models:
            llm = create_llm_with_decision_model_fallback()
            document, _ = extract(
                llm, {_DECISION_MODEL: [answers]}, {_LLM_MODEL: llm_outcomes}
            )
            assert requested_models == expected_requested_models
            (item,) = document.concepts[0].extracted_items
            assert item.value is True
            # The item is extracted by the decision model
            self.check_item_probabilities(item, {True: 0.93, False: 1 - 0.93})
            usage_containers = llm.get_usage()
            assert [
                (container.model, container.is_fallback)
                for container in usage_containers
            ] == [(_LLM_MODEL, False), (_DECISION_MODEL, True)]
            assert usage_containers[1].usage.input == 100
        # The fallback decision model is the last model to fail: the error has its
        # retries, whatever the retries of the LLM
        with pytest.raises(
            ModelExtractionError,
            match=(
                r"an invalid response returned by decision model"
                r".*\(after 3 retries\)"
            ),
        ) as extraction_exc_info:
            extract(
                create_llm_with_decision_model_fallback(),
                {_DECISION_MODEL: [invalid_answers]},
                {_LLM_MODEL: [llm_auth_error]},
            )
        assert extraction_exc_info.value.retry_count == 3
        assert requested_models == [_LLM_MODEL] + [_DECISION_MODEL] * (
            n_decision_model_calls
        )
        with pytest.raises(ModelAPIError, match="decision model API") as exc_info:
            extract(
                create_llm_with_decision_model_fallback(),
                {_DECISION_MODEL: [auth_error]},
                {_LLM_MODEL: [llm_auth_error]},
            )
        assert exc_info.value.original_error is auth_error
        with capture_logs(stdlib_logging.WARNING) as captured_logs:
            document, _ = extract(
                create_llm_with_decision_model_fallback(),
                {_DECISION_MODEL: [auth_error]},
                {_LLM_MODEL: [llm_auth_error]},
                raise_exception_on_extraction_error=False,
            )
        assert any(
            re.search(r"decision model API.*0 retries", log) for log in captured_logs
        ), captured_logs
        assert not document.concepts[0]._is_processed

        # === Several fallback models of both kinds, tried in the given order ===
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True),
                _create_llm(is_fallback=True),
            ]
        )
        document, _ = extract(
            decision_model,
            {_DECISION_MODEL: [invalid_answers], _OTHER_DECISION_MODEL: [auth_error]},
            {_LLM_MODEL: [llm_answer]},
        )
        assert requested_models == [_DECISION_MODEL] * n_decision_model_calls + [
            _OTHER_DECISION_MODEL,
            _LLM_MODEL,
        ]
        assert document.concepts[0].extracted_items[0].value is True
        assert [container.model for container in decision_model.get_usage()] == [
            _DECISION_MODEL,
            _OTHER_DECISION_MODEL,
            _LLM_MODEL,
        ]

        # === A refusal is passed on to the fallback models ===
        # A refused answer of the decision model, which is refused again when
        # the decision model is asked again, is sent to a fallback model of
        # either kind, which answers
        refused_answers = {"C0": "refusal"}
        for fallback_model, fallback_name in [
            (_create_llm(is_fallback=True), _LLM_MODEL),
            (
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True),
                _OTHER_DECISION_MODEL,
            ),
        ]:
            document, _ = extract(
                _create_decision_model(fallback_models=[fallback_model]),
                {_DECISION_MODEL: [refused_answers], _OTHER_DECISION_MODEL: [answers]},
                {_LLM_MODEL: [llm_answer]},
            )
            assert requested_models == [_DECISION_MODEL] * n_decision_model_calls + [
                fallback_name
            ]
            assert document.concepts[0].extracted_items[0].value is True
        # ... and the refusal of the last model of the chain is raised, for that
        # model, after its retries
        document = create_document()
        with pytest.raises(
            ModelRefusalError,
            match=f"Decision model `{_OTHER_DECISION_MODEL}` refused to answer",
        ) as refusal_exc_info:
            extract(
                _create_decision_model(
                    fallback_models=[
                        _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                    ]
                ),
                {
                    _DECISION_MODEL: [auth_error],
                    _OTHER_DECISION_MODEL: [refused_answers],
                },
                {},
                document=document,
            )
        assert refusal_exc_info.value.retry_count == 3
        assert (
            requested_models
            == [_DECISION_MODEL] + [_OTHER_DECISION_MODEL] * n_decision_model_calls
        )
        assert not document.concepts[0]._is_processed
        # A refusal of an LLM is asked again, as invalid data is, and then sent to
        # its fallback decision model
        refused_completion = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=llm_answer),
                    finish_reason="content_filter",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        document, _ = extract(
            create_llm_with_decision_model_fallback(),
            {_DECISION_MODEL: [answers]},
            {_LLM_MODEL: [refused_completion]},
        )
        assert requested_models == [_LLM_MODEL, _LLM_MODEL, _DECISION_MODEL]
        assert document.concepts[0].extracted_items[0].value is True

    def test_decision_model_fallback_validation(self):
        """
        Tests the validation of the fallback models of a decision model, and of
        a decision model as a fallback model of an LLM: what models of all kinds
        require of their fallback models applies, while what an LLM requires of
        its fallback LLMs does not apply to a decision model.
        """

        # A model of either kind is a fallback model of a model of either kind
        decision_model = _create_decision_model(
            role="decide",
            fallback_models=[
                _create_decision_model(
                    _OTHER_DECISION_MODEL, role="decide", is_fallback=True
                ),
                _create_llm(role="decide", is_fallback=True),
            ],
        )
        assert [type(model) for model in decision_model.fallback_models] == [
            DocumentDecisionModel,
            DocumentLLM,
        ]
        # ... also when the fallback models are assigned
        decision_model.fallback_models = decision_model.fallback_models[::-1]
        assert [type(model) for model in decision_model.fallback_models] == [
            DocumentLLM,
            DocumentDecisionModel,
        ]
        # What an LLM requires its fallback LLMs to share with it (the output
        # language, the system message, the tools) is not required of a fallback
        # decision model, which has no such params
        llm = _create_llm(
            output_language="adapt",
            fallback_models=[
                _create_decision_model(is_fallback=True),
                _create_llm(
                    _OTHER_LLM_MODEL, output_language="adapt", is_fallback=True
                ),
            ],
        )
        assert len(llm.fallback_models) == 2
        # ... while it is still required of a fallback LLM
        with pytest.raises(ValueError, match="same output language"):
            _create_llm(
                output_language="adapt",
                fallback_models=[
                    _create_decision_model(is_fallback=True),
                    _create_llm(_OTHER_LLM_MODEL, is_fallback=True),
                ],
            )
        # A decision model requires no params to be shared by a fallback LLM
        _create_decision_model(
            fallback_models=[
                _create_llm(
                    output_language="adapt",
                    system_message="You extract data from contracts.",
                    is_fallback=True,
                )
            ]
        )

        # A fallback model has to be marked as such. The error names its kind.
        with pytest.raises(
            ValueError,
            match="Fallback decision model must be a fallback model. "
            "Use `is_fallback=True`.",
        ):
            _create_decision_model(
                fallback_models=[_create_decision_model(_OTHER_DECISION_MODEL)]
            )
        with pytest.raises(
            ValueError, match="Fallback decision model must be a fallback model"
        ):
            _create_llm(fallback_models=[_create_decision_model()])
        with pytest.raises(ValueError, match="Fallback LLM must be a fallback model"):
            _create_decision_model(fallback_models=[_create_llm()])
        with pytest.raises(ValueError, match="Fallback LLM must be a fallback model"):
            decision_model.fallback_models = [_create_llm(role="decide")]
        assert len(decision_model.fallback_models) == 2  # no change

        # A fallback model has the role of the main model
        with pytest.raises(
            ValueError,
            match="The fallback decision model must have the same role `decide` "
            "as the main one",
        ):
            _create_decision_model(
                role="decide",
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ],
            )
        with pytest.raises(
            ValueError, match="The fallback decision model must have the same role"
        ):
            _create_llm(
                fallback_models=[
                    _create_decision_model(role="decide", is_fallback=True)
                ]
            )
        with pytest.raises(
            ValueError, match="The fallback LLM must have the same role `decide`"
        ):
            _create_decision_model(
                role="decide", fallback_models=[_create_llm(is_fallback=True)]
            )

        # A fallback model has no fallback models of its own
        with pytest.raises(
            ValueError,
            match="Fallback decision model cannot have its own fallback "
            "decision models and must be attached to a non-fallback model",
        ):
            _create_decision_model(
                is_fallback=True,
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ],
            )
        with pytest.raises(
            ValueError,
            match="Fallback decision model cannot have its own fallback models",
        ):
            _create_decision_model(
                is_fallback=True, fallback_models=[_create_llm(is_fallback=True)]
            )
        with pytest.raises(
            ValueError, match="Fallback LLM cannot have its own fallback models"
        ):
            _create_llm(
                is_fallback=True,
                fallback_models=[_create_decision_model(is_fallback=True)],
            )

        # A fallback model is not a replica of the main model, or of another
        # fallback model
        with pytest.raises(
            ValueError,
            match="Fallback decision model must not have the exact same config "
            "params as the main decision model",
        ):
            _create_decision_model(
                fallback_models=[_create_decision_model(is_fallback=True)]
            )
        with pytest.raises(
            ValueError,
            match="Fallback decision models must not have the exact same config "
            "params as each other",
        ):
            _create_decision_model(
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True),
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True),
                ]
            )
        # ... which it is not if it differs in a param
        _create_decision_model(
            fallback_models=[
                _create_decision_model(is_fallback=True, timeout=30),
                _create_decision_model(
                    is_fallback=True, api_base="https://decisions.example.com"
                ),
            ]
        )
        # ... or if it is a model of another kind: the same model can be requested
        # for decisions and, as a fallback, for completions
        model_of_both_kinds = "openai/gpt-6-luna"
        decision_model = _create_decision_model(
            model_of_both_kinds,
            fallback_models=[_create_llm(model_of_both_kinds, is_fallback=True)],
        )
        assert decision_model.model == decision_model.fallback_models[0].model
        llm = _create_llm(
            model_of_both_kinds,
            fallback_models=[
                _create_decision_model(model_of_both_kinds, is_fallback=True)
            ],
        )
        assert llm.model == llm.fallback_models[0].model

    def test_decision_model_in_model_group(self):
        """
        Tests a model group with an LLM and a decision model: each aspect and
        concept is extracted by the model that it is routed to by the role, and
        what a group does for its LLMs only is not applied to the decision model.
        """
        llm = _create_llm()
        decision_model = _create_decision_model(
            role="decide",
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
        )
        model_group = DocumentModelGroup(models=[llm, decision_model])
        assert model_group.is_group is True
        assert model_group.list_roles == [None, "decide"]
        assert model_group.models == [llm, decision_model]

        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                    concepts=[
                        BooleanConcept(
                            name="Termination for convenience",
                            description="Either party may terminate without cause",
                            model_role="decide",
                        ),
                        StringConcept(
                            name="Notice period",
                            description="Length of the notice period",
                        ),
                        LabelConcept(
                            name="Notice form",
                            description="Required form of the termination notice",
                            labels=["written", "any"],
                            model_role="decide",
                        ),
                    ],
                )
            ],
            concepts=[
                StringConcept(name="Title", description="Title of the document"),
                RatingConcept(
                    name="Clarity",
                    description="Clarity of the text of the document",
                    rating_scale=(1, 3),
                    model_role="decide",
                ),
            ],
        )
        llm_prompt_kwargs: list[dict[str, Any]] = []

        async def send_messages_stub(self, *, model_call_obj: Any, **kwargs: Any):
            """
            Simulates a successful LLM API call, which returns the last two
            paragraphs of the document for each aspect in the prompt, or
            an extracted item for each concept in the prompt.

            Same params as `llm._send_messages_with_tools`.
            """
            params = model_call_obj.prompt_kwargs
            llm_prompt_kwargs.append(params)
            if "aspects" in params:
                answer = [
                    {"aspect_id": f"A{i}", "paragraph_ids": ["P1", "P2"]}
                    for i, _ in enumerate(params["aspects"])
                ]
            else:
                answer = [
                    {"concept_id": f"C{i}", "extracted_items": ["Item"]}
                    for i, _ in enumerate(params["concepts"])
                ]
            return json.dumps(answer), False

        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                _DocumentLLM,
                "_send_messages_with_tools",
                autospec=True,
                side_effect=send_messages_stub,
            ),
        ):
            model_group.extract_all(document)

        # The LLM extracts the aspect and the concepts without a role
        assert [
            [
                instance.name
                for instance in params.get("aspects", params.get("concepts"))
            ]
            for params in llm_prompt_kwargs
        ] == [["Termination"], ["Notice period"], ["Title"]]
        # The decision model extracts the concepts with its role: those of
        # the aspect, from the paragraphs that the LLM extracted for the aspect,
        # and those of the document
        assert [
            (
                request["model"],
                request["input"],
                [
                    (question["type"], question["name"])
                    for question in request["questions"]
                ],
            )
            for request in decisions_stub.requests
        ] == [
            (
                _DECISION_MODEL,
                _ASPECT_PARAGRAPHS_TEXT_SENT,
                [("predicate", "C0"), ("choice", "C1")],
            ),
            (_DECISION_MODEL, _DOCUMENT_TEXT_SENT, [("score", "C0")]),
        ]
        (aspect,) = document.aspects
        assert len(aspect.extracted_items) == 2
        boolean_concept, string_concept, label_concept = aspect.concepts
        title_concept, rating_concept = document.concepts
        self.check_item_probabilities(
            boolean_concept.extracted_items[0], {True: 0.9, False: 1 - 0.9}
        )
        assert label_concept.extracted_items[0].value == ["written"]
        self.check_item_probabilities(
            label_concept.extracted_items[0], {"written": 1.0, "any": 0.0}
        )
        assert rating_concept.extracted_items[0].value == 1
        self.check_item_probabilities(
            rating_concept.extracted_items[0], {1: 1.0, 2: 0.0, 3: 0.0}
        )
        for concept in [string_concept, title_concept]:
            assert concept.extracted_items[0].value == "Item"
            assert concept.extracted_items[0].probabilities is None

        # The usage and cost of the group, for all its models or by a role
        assert [
            (container.model, container.role) for container in model_group.get_usage()
        ] == [(_LLM_MODEL, None), (_DECISION_MODEL, "decide")]
        (usage_container,) = model_group.get_usage(model_role="decide")
        assert usage_container.model == _DECISION_MODEL
        assert usage_container.usage.input == 200
        assert len(usage_container.usage.calls) == 2
        (cost_container,) = model_group.get_cost(model_role="decide")
        assert cost_container.model == _DECISION_MODEL
        assert cost_container.cost.total == Decimal("0.00001")
        assert [container.model for container in model_group.get_cost()] == [
            _LLM_MODEL,
            _DECISION_MODEL,
        ]
        model_group.reset_usage_and_cost(model_role="decide")
        assert decision_model.get_usage()[0].usage == _ModelUsage()
        model_group.reset_usage_and_cost()

        # === The output language of a group is the one of its LLMs ===
        assert model_group.output_language == "en"
        model_group.group_update_output_language("adapt")
        assert model_group.output_language == "adapt"
        assert llm.output_language == "adapt"
        assert not hasattr(decision_model, "output_language")
        # A group is created with the output language of its LLMs
        llm_adapt = _create_llm(output_language="adapt")
        model_group = DocumentModelGroup(
            models=[decision_model, llm_adapt], output_language="adapt"
        )
        assert model_group.list_roles == ["decide", None]
        with pytest.raises(ValueError, match="All LLMs in the group must have"):
            DocumentModelGroup(models=[decision_model, llm_adapt])
        # A group of decision models only has no LLM to match its output language
        other_decision_model = _create_decision_model(_OTHER_DECISION_MODEL)
        decision_models_group = DocumentModelGroup(
            models=[decision_model, other_decision_model]
        )
        decision_models_group.group_update_output_language("adapt")
        assert decision_models_group.output_language == "adapt"

        # === The warning on the tools lists the LLMs with tools ===
        @register_tool
        def get_contract_status(contract_id: str) -> str:
            """
            Returns the status of the contract.
            """
            return "active"

        llm_with_tools = _create_llm(tools=[get_contract_status])
        model_group = DocumentModelGroup(models=[llm_with_tools, decision_model])
        with pytest.warns(
            UserWarning, match=f"Models with tools configured: {_LLM_MODEL}\\."
        ):
            model_group._warn_tools_ignored_if_enabled()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            decision_models_group._warn_tools_ignored_if_enabled()
            decision_model._warn_tools_ignored_if_enabled()

        # === The roles of the models of a group are unique ===
        # The models are named by their kind, unless the group has models
        # of different kinds
        with pytest.raises(
            ValueError,
            match="Models in a group must have different roles, and only one "
            "model in a group can have no role",
        ):
            DocumentModelGroup(models=[_create_llm(), other_decision_model])
        with pytest.raises(
            ValueError,
            match="Decision models in a group must have different roles, and only "
            "one decision model in a group can have no role",
        ):
            DocumentModelGroup(
                models=[other_decision_model, _create_decision_model(timeout=30)]
            )
        with pytest.raises(
            ValueError,
            match="LLMs in a group must have different roles, and only one LLM "
            "in a group can have no role",
        ):
            DocumentModelGroup(models=[_create_llm(), _create_llm(_OTHER_LLM_MODEL)])
        # The role of a model can be changed after the model is added to a group,
        # which is checked when an extraction starts
        llm = _create_llm()
        model_group = DocumentModelGroup(models=[llm, decision_model])
        document = _create_document(
            concepts=[
                BooleanConcept(
                    name="NDA check",
                    description="The document is an NDA",
                    model_role="decide",
                )
            ]
        )
        llm.role = "decide"
        with pytest.raises(
            ValueError,
            match="Models in a group must have different roles.*"
            "Roles assigned to several models: `decide`",
        ):
            model_group.extract_concepts_from_document(document)
        llm.role = None

        # === The role of each aspect/concept is the role of a model of the group ===
        document.concepts[0].model_role = "missing"
        with pytest.raises(
            ValueError,
            match="cannot be routed to a model in the group, as no model in "
            r"the group has the role: `missing`\. Roles of the models in the group: "
            r"None \(no role\), `decide`\. An aspect/concept is extracted by "
            "the model with the `role`",
        ):
            model_group.extract_concepts_from_document(document)
        with pytest.raises(
            ValueError,
            match="cannot be routed to a decision model in the group, as no "
            "decision model in the group has the role: `missing`",
        ):
            decision_models_group.extract_all(document)
        for get_or_reset in [
            model_group.get_usage,
            model_group.get_cost,
            model_group.reset_usage_and_cost,
        ]:
            with pytest.raises(
                ValueError, match="No model with the given role `missing` was found"
            ):
                get_or_reset(model_role="missing")
        with pytest.raises(
            ValueError, match="No decision model with the given role `missing`"
        ):
            decision_models_group.get_usage(model_role="missing")

    def test_decision_model_instances_validation(self):
        """
        Tests that what a decision model cannot extract is rejected before any
        request is sent: for a decision model used on its own, in a model group,
        and as a fallback model, by each of the extraction methods.
        """
        unservable_concepts: list[tuple[Any, str]] = [
            # Concepts of the types whose values do not come from a closed set
            (
                StringConcept(name="Title", description="Title of the document"),
                "type `StringConcept` is not supported",
            ),
            (
                NumericalConcept(name="Term", description="Term of the agreement"),
                "type `NumericalConcept` is not supported",
            ),
            (
                DateConcept(name="Effective date", description="Effective date"),
                "type `DateConcept` is not supported",
            ),
            (
                JsonObjectConcept(
                    name="Party",
                    description="Details of a party",
                    structure={"name": str},
                ),
                "type `JsonObjectConcept` is not supported",
            ),
            # Concepts extracted from images
            (
                BooleanConcept(
                    name="Signed", description="Is signed", modality="image"
                ),
                "modality `image` is not supported",
            ),
            (
                RatingConcept(
                    name="Scan quality",
                    description="Quality of the scan",
                    rating_scale=(1, 5),
                    modality="multimodal",
                ),
                "modality `multimodal` is not supported",
            ),
            # Concepts with justifications or references
            (
                BooleanConcept(
                    name="NDA check",
                    description="The document is an NDA",
                    add_justifications=True,
                ),
                "justifications are not supported",
            ),
            (
                LabelConcept(
                    name="Notice form",
                    description="Required form of the termination notice",
                    labels=["written", "any"],
                    add_references=True,
                ),
                "references are not supported",
            ),
            # Each reason is given
            (
                StringConcept(
                    name="Governing law",
                    description="Governing law of the agreement",
                    add_justifications=True,
                    add_references=True,
                ),
                (
                    "type `StringConcept` is not supported, justifications are not "
                    "supported, references are not supported"
                ),
            ),
        ]
        supported_concept = BooleanConcept(
            name="Agreement check", description="The document is an agreement"
        )
        requests: list[Any] = []

        async def record_request(*args: Any, **kwargs: Any) -> Any:
            """
            Records a request to a model API, which is not expected where
            this stub is used, and fails the test.

            :param args: Arguments of the request.
            :type args: Any
            :param kwargs: Keyword arguments of the request.
            :type kwargs: Any
            """
            requests.append(kwargs)
            raise pytest.fail.Exception("No request is expected to be sent.")

        def check_rejected(
            model: DocumentDecisionModel | DocumentLLM | DocumentModelGroup,
            document: Document,
            error_match: str,
            aspect: _Aspect | None = None,
            only_full_extraction: bool = False,
        ) -> None:
            """
            Checks that the extraction methods that are to extract the instances
            reject them with the expected error, before any request is sent and
            with the document left unchanged.

            :param model: Model or model group to extract with.
            :type model: DocumentDecisionModel | DocumentLLM | DocumentModelGroup
            :param document: Document to extract from.
            :type document: Document
            :param error_match: Pattern that the error is to match.
            :type error_match: str
            :param aspect: Aspect of the document to extract the concepts from.
                Defaults to None, in which case the instances of the document
                are extracted.
            :type aspect: _Aspect | None
            :param only_full_extraction: Whether only the extraction of everything
                that the document has is to reject the instances, as the other
                extraction methods do not extract them. Defaults to False.
            :type only_full_extraction: bool
            """
            extraction_methods: list[Callable[..., Any]]
            extraction_kwargs: dict[str, Any] = {"document": document}
            if only_full_extraction:
                extraction_methods = [model.extract_all]
            elif aspect is not None:
                extraction_methods = [model.extract_concepts_from_aspect]
                extraction_kwargs["aspect"] = aspect
            elif document.aspects:
                extraction_methods = [
                    model.extract_aspects_from_document,
                    model.extract_all,
                ]
            else:
                extraction_methods = [
                    model.extract_concepts_from_document,
                    model.extract_all,
                ]
            before = document.to_dict()
            with (
                patch.object(litellm, "adecisions", new=record_request),
                patch.object(litellm, "acompletion", new=record_request),
            ):
                for extraction_method in extraction_methods:
                    for use_concurrency in [False, True]:
                        with pytest.raises(ValueError, match=error_match):
                            extraction_method(
                                **extraction_kwargs, use_concurrency=use_concurrency
                            )
            assert not requests
            assert document.to_dict() == before

        # === A decision model used on its own ===
        decision_model = _create_decision_model()
        for concept, reasons in unservable_concepts:
            error_match = (
                f"^Decision model `{_DECISION_MODEL}` cannot extract: "
                f"concept `{concept.name}` \\({reasons}\\)\\. "
                f"A decision model extracts aspects, and Boolean, label and rating "
                f"concepts from text without references, both without "
                f"justifications\\. Route the other aspects and concepts to an LLM"
            )
            # A concept of the document, next to a concept that can be extracted
            document = _create_document(concepts=[supported_concept, concept])
            check_rejected(decision_model, document, error_match)
            # A concept of an aspect
            if concept.modality == "text":
                document = _create_document(
                    aspects=[
                        Aspect(
                            name="Termination",
                            description="Clauses on termination",
                            concepts=[concept, supported_concept],
                        )
                    ]
                )
                (aspect,) = document.aspects
                _set_aspect_as_extracted(document, aspect)
                check_rejected(decision_model, document, error_match, aspect=aspect)
        # All the instances that cannot be extracted are named
        document = _create_document(
            concepts=[
                unservable_concepts[0][0],
                supported_concept,
                unservable_concepts[6][0],
            ]
        )
        check_rejected(
            decision_model,
            document,
            r"cannot extract: concept `Title` \(type `StringConcept` is not "
            r"supported\); concept `NDA check` \(justifications are not "
            r"supported\)\. A decision model",
        )

        # Aspects that a decision model cannot extract
        unservable_aspects: list[tuple[dict[str, Any], str]] = [
            # Justifications
            ({"add_justifications": True}, "justifications are not supported"),
            # ... whatever the extracted items are: sentences are supported
            (
                {"reference_depth": "sentences", "add_justifications": True},
                "justifications are not supported",
            ),
        ]
        for aspect_kwargs, reasons in unservable_aspects:
            error_match = (
                f"^Decision model `{_DECISION_MODEL}` cannot extract: "
                f"aspect `Termination` \\({reasons}\\)\\. "
                f"A decision model extracts aspects, and Boolean, label and rating "
                f"concepts"
            )
            # An aspect of the document, next to an aspect that can be extracted
            document = _create_document(
                aspects=[
                    Aspect(name="Liability", description="Clauses on liability"),
                    Aspect(
                        name="Termination",
                        description="Clauses on termination",
                        **aspect_kwargs,
                    ),
                ],
                concepts=[supported_concept],
            )
            check_rejected(decision_model, document, error_match)
            # A sub-aspect of an aspect that can be extracted
            document = _create_document(
                aspects=[
                    Aspect(
                        name="Liability",
                        description="Clauses on liability",
                        aspects=[
                            Aspect(
                                name="Termination",
                                description="Clauses on termination",
                                **aspect_kwargs,
                            )
                        ],
                    )
                ]
            )
            check_rejected(decision_model, document, error_match)
        # All the instances that cannot be extracted are named by the extraction
        # of everything that the document has
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    add_justifications=True,
                    concepts=[unservable_concepts[7][0]],
                    aspects=[
                        Aspect(
                            name="Notices",
                            description="Clauses on notices",
                            reference_depth="sentences",
                            add_justifications=True,
                        )
                    ],
                )
            ],
            concepts=[unservable_concepts[0][0]],
        )
        check_rejected(
            decision_model,
            document,
            r"cannot extract: concept `Title` \(type `StringConcept` is not "
            r"supported\); aspect `Termination` \(justifications are not "
            r"supported\); concept `Notice form` \(references are not supported\); "
            r"aspect `Notices` \(justifications are not supported\)\. ",
            only_full_extraction=True,
        )
        # A concept of an aspect is not extracted when the aspects are, i.e. it is
        # rejected only by the extraction of everything that the document has
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    concepts=[unservable_concepts[0][0]],
                )
            ]
        )
        check_rejected(
            decision_model,
            document,
            f"^Decision model `{_DECISION_MODEL}` cannot extract: concept `Title`",
            only_full_extraction=True,
        )
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [{"A0": 0.9}]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert len(document.aspects[0].extracted_items) == 3

        # === A decision model in a model group ===
        llm = _create_llm()
        decision_model = _create_decision_model(role="decide")
        model_group = DocumentModelGroup(models=[llm, decision_model])
        string_concept = StringConcept(
            name="Title", description="Title of the document", model_role="decide"
        )
        # The error tells how to route what the decision model cannot extract
        # to an LLM of the group, which the decision model is already in
        group_hint = (
            "both without justifications\\. In a model group, an aspect or a concept "
            "is extracted by the model with the `role` equal to its `model_role`: "
            "set the `model_role` of the other aspects and concepts to the role "
            "of an LLM in the group\\.$"
        )
        group_error_match = (
            f"^Decision model `{_DECISION_MODEL}` cannot extract: "
            r"concept `Title` \(type `StringConcept` is not supported\)\. "
            f".*{group_hint}"
        )
        # Only what is routed to the decision model is checked: a concept that it
        # cannot extract is rejected if it has the role of the decision model
        document = _create_document(concepts=[supported_concept, string_concept])
        check_rejected(model_group, document, group_error_match)
        # ... also when it is a concept of an aspect that an LLM is to extract:
        # the full extraction rejects it before the request for the aspect is sent
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    concepts=[string_concept],
                    aspects=[
                        Aspect(
                            name="Notices",
                            description="Clauses on notices",
                            concepts=[supported_concept],
                        )
                    ],
                )
            ]
        )
        before = document.to_dict()
        with (
            patch.object(litellm, "adecisions", new=record_request),
            patch.object(litellm, "acompletion", new=record_request),
            pytest.raises(ValueError, match=group_error_match),
        ):
            model_group.extract_all(document)
        assert not requests
        assert document.to_dict() == before
        # ... or a concept of a sub-aspect, when the concepts of the aspect
        # are extracted
        document.aspects[0].concepts = []
        document.aspects[0].aspects[0].concepts = [string_concept]
        (aspect,) = document.aspects
        _set_aspect_as_extracted(document, aspect)
        _set_aspect_as_extracted(document, aspect.aspects[0])
        check_rejected(model_group, document, group_error_match, aspect=aspect)
        # An aspect that the decision model cannot extract is rejected if it has
        # the role of the decision model, also when it is a sub-aspect of an aspect
        # that an LLM is to extract
        for aspects in [
            [
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    add_justifications=True,
                    model_role="decide",
                )
            ],
            [
                Aspect(
                    name="Liability",
                    description="Clauses on liability",
                    aspects=[
                        Aspect(
                            name="Termination",
                            description="Clauses on termination",
                            add_justifications=True,
                            model_role="decide",
                        )
                    ],
                )
            ],
        ]:
            document = _create_document(aspects=aspects)
            check_rejected(
                model_group,
                document,
                f"^Decision model `{_DECISION_MODEL}` cannot extract: "
                r"aspect `Termination` \(justifications are not supported\)\. ",
            )

        # A decision model without a role in a group is routed the aspects and
        # concepts without a model role
        roleless_group = DocumentModelGroup(
            models=[_create_decision_model(), _create_llm(role="generate")]
        )
        document = _create_document(
            concepts=[
                supported_concept,
                StringConcept(name="Title", description="Title of the document"),
            ]
        )
        check_rejected(roleless_group, document, group_error_match)
        # ... which are extracted once they are routed to the LLM of the group
        document.concepts[1].model_role = "generate"
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [{"C0": 0.9}]})
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": ["NDA"]}]']}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            roleless_group.extract_all(document)
        assert [concept.extracted_items[0].value for concept in document.concepts] == [
            True,
            "NDA",
        ]

        # What is routed to an LLM of the group is not checked
        document = _create_document(
            concepts=[
                StringConcept(name="Title", description="Title of the document"),
                BooleanConcept(
                    name="NDA check",
                    description="The document is an NDA",
                    model_role="decide",
                ),
            ]
        )
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [{"C0": 0.9}]})
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": ["NDA"]}]']}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            model_group.extract_all(document)
        assert [concept.extracted_items[0].value for concept in document.concepts] == [
            "NDA",
            True,
        ]

        # === A decision model as a fallback model ===
        # A fallback model is to extract what the model it is a fallback for fails
        # to extract, i.e. all that the LLM is to extract
        fallback_error_match = (
            f"^Fallback decision model `{_DECISION_MODEL}` cannot extract what it "
            f"is to extract if the model it is a fallback for fails: "
            r"concept `Title` \(type `StringConcept` is not supported\)\. "
        )
        llm = _create_llm(fallback_models=[_create_decision_model(is_fallback=True)])
        title_concept = StringConcept(name="Title", description="Title of the document")
        document = _create_document(concepts=[supported_concept, title_concept])
        check_rejected(llm, document, fallback_error_match)
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    reference_depth="sentences",
                    add_justifications=True,
                    concepts=[supported_concept],
                )
            ]
        )
        check_rejected(
            llm,
            document,
            f"^Fallback decision model `{_DECISION_MODEL}` cannot extract what it "
            r"is to extract if the model it is a fallback for fails: aspect "
            r"`Termination` \(justifications are not supported\)\. ",
        )
        # ... while an LLM with such a fallback model extracts what a decision
        # model can extract
        document = _create_document(concepts=[supported_concept])
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": [true]}]']}
        )
        with (
            patch.object(litellm, "adecisions", new=record_request),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            llm.extract_all(document)
        assert document.concepts[0].extracted_items[0].value is True
        assert not requests

        # A fallback decision model of a decision model
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
            ]
        )
        document = _create_document(concepts=[title_concept])
        check_rejected(
            decision_model,
            document,
            f"^Decision model `{_DECISION_MODEL}` cannot extract: concept `Title`",
        )
        # A fallback decision model of an LLM in a model group: what is routed
        # to the LLM by its role is checked
        model_group = DocumentModelGroup(
            models=[
                _create_llm(_OTHER_LLM_MODEL),
                _create_llm(
                    role="decide",
                    fallback_models=[
                        _create_decision_model(role="decide", is_fallback=True)
                    ],
                ),
            ]
        )
        document = _create_document(concepts=[supported_concept, string_concept])
        check_rejected(model_group, document, f"{fallback_error_match}.*{group_hint}")
        document = _create_document(concepts=[title_concept])
        completions_stub = _CompletionsAPIStub(
            {_OTHER_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": ["NDA"]}]']}
        )
        with (
            patch.object(litellm, "adecisions", new=record_request),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            model_group.extract_concepts_from_document(document)
        assert document.concepts[0].extracted_items[0].value == "NDA"
        assert not requests

        # === The checks made by the decision model itself ===
        # Nothing is requested for what a decision model cannot extract, also when
        # the checks made at the start of an extraction are bypassed
        decision_model = _create_decision_model()
        document = _create_document(concepts=[title_concept])
        with (
            patch.object(litellm, "adecisions", new=record_request),
            pytest.raises(ValueError, match="cannot extract: concept `Title`"),
        ):
            asyncio.run(
                decision_model._extract_items_from_instances(
                    extracted_item_type="concept",
                    source=document,
                    llm=decision_model,
                    instances_to_process=list(document.concepts),
                    document=document,
                )
            )
        assert not requests
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    add_justifications=True,
                )
            ]
        )
        with (
            patch.object(litellm, "adecisions", new=record_request),
            pytest.raises(ValueError, match="cannot extract: aspect `Termination`"),
        ):
            asyncio.run(
                decision_model._extract_items_from_instances(
                    extracted_item_type="aspect",
                    source=document,
                    llm=decision_model,
                    instances_to_process=list(document.aspects),
                    document=document,
                )
            )
        assert not requests
        # A document without text has nothing to ask the questions on
        document = Document(images=[self.test_img_png_invoice])
        document.concepts = [supported_concept]
        with (
            patch.object(litellm, "adecisions", new=record_request),
            pytest.raises(ValueError, match="Document lacks text or paragraphs"),
        ):
            decision_model.extract_concepts_from_document(document)
        assert not requests

    def test_decision_model_rejects_singular_occurrence_in_parts(self):
        """
        Tests that a concept with a singular occurrence is rejected, before any
        request is sent, when a decision model is to extract it with the max
        number of paragraphs to analyze per call set, i.e. from a text analyzed
        in parts, for each of which the concept would get an item: for
        a decision model used on its own, with a fallback LLM, as a fallback
        model of an LLM, and in a model group, by each of the extraction methods
        that extract concepts. Without that number, such a concept is extracted.
        """
        singular_reason = (
            r"\(`singular_occurrence=True` cannot be applied when the text is "
            r"analyzed in parts \(`max_paragraphs_to_analyze_per_call` is set\): "
            r"a decision model returns an item for each part\)"
        )
        llm_answer = '[{"concept_id": "C0", "extracted_items": [true]}]'
        # The stubs respond to any request, so that a request that is sent
        # where none is expected is found by its record
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [_answer_all_questions],
                _OTHER_DECISION_MODEL: [_answer_all_questions],
            }
        )
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: [llm_answer]})

        def create_document(
            level: Literal["document", "aspect", "sub_aspect"],
            model_role: str | None = None,
            singular_occurrence: bool = True,
        ) -> Document:
            """
            Creates a document with five paragraphs and a Boolean concept with
            a singular occurrence, next to a Boolean concept without it.

            :param level: What the concept with a singular occurrence is
                attached to: the document, an aspect of it, or a sub-aspect of
                that aspect.
            :type level: Literal["document", "aspect", "sub_aspect"]
            :param model_role: The model role of that concept. Defaults to None.
            :type model_role: str | None
            :param singular_occurrence: Whether that concept has a singular
                occurrence. Defaults to True.
            :type singular_occurrence: bool
            :return: Document.
            :rtype: Document
            """
            concept = BooleanConcept(
                name="Lease check",
                description="The text is of a lease agreement",
                singular_occurrence=singular_occurrence,
                model_role=model_role,
            )
            other_concept = BooleanConcept(
                name="NDA check", description="The text is of an NDA"
            )
            if level == "document":
                return _create_aspects_document(concepts=[other_concept, concept])
            return _create_aspects_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        concepts=[other_concept, concept]
                        if level == "aspect"
                        else [other_concept],
                        aspects=[
                            Aspect(
                                name="Notices",
                                description="Clauses on the notices of termination",
                                concepts=[concept] if level == "sub_aspect" else [],
                            )
                        ],
                    )
                ]
            )

        def check_rejected(
            model: DocumentDecisionModel | DocumentLLM | DocumentModelGroup,
            error_prefix: str,
            model_role: str | None = None,
        ) -> None:
            """
            Checks that each extraction method that is to extract the concept with
            a singular occurrence, attached at each level, rejects it when
            the max number of paragraphs to analyze per call is set, before any
            request is sent and with the document left unchanged.

            :param model: Model or model group to extract with.
            :type model: DocumentDecisionModel | DocumentLLM | DocumentModelGroup
            :param error_prefix: Pattern that the start of the error is to match,
                up to the concept that is rejected.
            :type error_prefix: str
            :param model_role: The model role of the concept. Defaults to None.
            :type model_role: str | None
            """
            error_match = f"{error_prefix}concept `Lease check` {singular_reason}\\. "
            extractions: list[tuple[Document, Callable[..., Any], dict[str, Any]]] = []
            for level in ("document", "aspect", "sub_aspect"):
                # The extraction of everything that the document has, in which
                # the aspects are yet to be extracted
                document = create_document(level, model_role)
                extractions.append((document, model.extract_all, {}))
                # The extraction of the concepts: of the document, or of
                # an aspect that is extracted, as is its sub-aspect
                document = create_document(level, model_role)
                if level == "document":
                    extractions.append(
                        (document, model.extract_concepts_from_document, {})
                    )
                else:
                    (aspect,) = document.aspects
                    _set_aspect_as_extracted(document, aspect)
                    _set_aspect_as_extracted(document, aspect.aspects[0])
                    extractions.append(
                        (
                            document,
                            model.extract_concepts_from_aspect,
                            {"aspect": aspect},
                        )
                    )
            decisions_stub.requests.clear()
            completions_stub.requests.clear()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(litellm, "acompletion", new=completions_stub),
            ):
                for document, extraction_method, method_kwargs in extractions:
                    before = document.to_dict()
                    # The concept is rejected whatever the number of the parts
                    # that the text would be split into, also if it is one
                    for max_paragraphs in [1, 100]:
                        for use_concurrency in [False, True]:
                            with pytest.raises(ValueError, match=error_match):
                                extraction_method(
                                    document=document,
                                    max_paragraphs_to_analyze_per_call=max_paragraphs,
                                    use_concurrency=use_concurrency,
                                    **method_kwargs,
                                )
                    assert document.to_dict() == before
            assert not decisions_stub.requests
            assert not completions_stub.requests

        # === A decision model used on its own ===
        decision_model_error_prefix = (
            f"^Decision model `{_DECISION_MODEL}` cannot extract: "
        )
        decision_model = _create_decision_model()
        check_rejected(decision_model, decision_model_error_prefix)

        # === A decision model with a fallback LLM ===
        check_rejected(
            _create_decision_model(fallback_models=[_create_llm(is_fallback=True)]),
            decision_model_error_prefix,
        )

        # === A decision model that is a fallback model of an LLM ===
        # It is to extract what the LLM fails to extract
        check_rejected(
            _create_llm(fallback_models=[_create_decision_model(is_fallback=True)]),
            f"^Fallback decision model `{_DECISION_MODEL}` cannot extract what it "
            f"is to extract if the model it is a fallback for fails: ",
        )

        # === A decision model in a model group ===
        # The concept is routed to the decision model by its role, also when
        # the aspect that it is attached to is extracted by the LLM of the group
        model_group = DocumentModelGroup(
            models=[_create_llm(), _create_decision_model(role="decide")]
        )
        check_rejected(model_group, decision_model_error_prefix, model_role="decide")

        # === The reasons ===
        get_reasons = _DocumentDecisionModel._get_unservable_reasons
        singular_reason_text = (
            "`singular_occurrence=True` cannot be applied when the text is "
            "analyzed in parts (`max_paragraphs_to_analyze_per_call` is set): "
            "a decision model returns an item for each part"
        )
        (singular_concept,) = create_document("document").concepts[1:]
        assert get_reasons(singular_concept, max_paragraphs_to_analyze_per_call=1) == [
            singular_reason_text
        ]
        # The number is not set: the text is sent as a whole
        for max_paragraphs in [0, -1]:
            assert not get_reasons(
                singular_concept, max_paragraphs_to_analyze_per_call=max_paragraphs
            )
        assert not get_reasons(singular_concept)
        # Each reason is given
        rating_concept = RatingConcept(
            name="Clarity",
            description="Clarity of the text",
            rating_scale=(0, 10),
            singular_occurrence=True,
        )
        singular_rating_reason, question_reason = get_reasons(
            rating_concept, max_paragraphs_to_analyze_per_call=2
        )
        assert singular_rating_reason == singular_reason_text
        assert "`score` question with 11 options is not accepted" in question_reason
        assert get_reasons(rating_concept) == [question_reason]
        assert get_reasons(
            StringConcept(
                name="Title",
                description="Title of the document",
                singular_occurrence=True,
            ),
            max_paragraphs_to_analyze_per_call=2,
        ) == ["type `StringConcept` is not supported", singular_reason_text]
        # An aspect is extracted paragraph by paragraph whatever the number is
        assert not get_reasons(
            Aspect(name="Termination", description="Clauses on termination"),
            max_paragraphs_to_analyze_per_call=2,
        )

        # === The checks made by the decision model itself ===
        # Nothing is requested, also when the checks made at the start of
        # an extraction are bypassed
        document = create_document("document")
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(
                ValueError,
                match=f"{decision_model_error_prefix}concept `Lease check` "
                f"{singular_reason}",
            ),
        ):
            asyncio.run(
                decision_model._extract_items_from_instances(
                    extracted_item_type="concept",
                    source=document,
                    llm=decision_model,
                    instances_to_process=list(document.concepts),
                    document=document,
                    max_paragraphs_to_analyze_per_call=2,
                )
            )
        assert not decisions_stub.requests

        # === What is not rejected ===
        part_texts = self._get_texts_of_parts(2)
        whole_text = "\n\n".join(_ASPECTS_PARAGRAPH_TEXTS_CLEANED)

        def requested_texts() -> list[str]:
            """
            Returns the texts of the requests sent to the decision model since
            the last call of this function.

            :return: The texts, in the order the requests were sent.
            :rtype: list[str]
            """
            texts = [request["input"] for request in decisions_stub.requests]
            decisions_stub.requests.clear()
            return texts

        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            # The number of paragraphs per call is not set: the text is sent as
            # a whole, and the concept gets one item
            document = create_document("document")
            decision_model.extract_concepts_from_document(document)
            assert requested_texts() == [whole_text]
            assert [len(concept.extracted_items) for concept in document.concepts] == [
                1,
                1,
            ]
            document = create_document("sub_aspect")
            decision_model.extract_all(document)
            (aspect,) = document.aspects
            assert len(aspect.aspects[0].concepts[0].extracted_items) == 1
            assert len(requested_texts()) == 5 + 5 + 1 + 1

            # The number of images per call is not the one of paragraphs
            document = create_document("document")
            decision_model.extract_concepts_from_document(
                document, max_images_to_analyze_per_call=1
            )
            assert requested_texts() == [whole_text]
            document = create_document("aspect")
            decision_model.extract_all(document, max_images_to_analyze_per_call=1)
            assert len(document.aspects[0].concepts[1].extracted_items) == 1
            requested_texts()

            # A concept without a singular occurrence gets an item for each part
            document = create_document("document", singular_occurrence=False)
            decision_model.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )
            assert requested_texts() == part_texts
            assert [len(concept.extracted_items) for concept in document.concepts] == [
                3,
                3,
            ]

            # A concept that is not among the concepts to extract
            document = create_document("document")
            other_concept, singular_concept = document.concepts
            decision_model.extract_concepts_from_document(
                document,
                from_concepts=[other_concept],
                max_paragraphs_to_analyze_per_call=2,
            )
            assert requested_texts() == part_texts
            assert len(other_concept.extracted_items) == 3
            assert not singular_concept._is_processed
            # ... or is a concept of an aspect, when the aspects are extracted,
            # for which the number of paragraphs per call changes nothing
            document = create_document("aspect")
            decision_model.extract_aspects_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )
            assert requested_texts() == [
                *_ASPECTS_PARAGRAPH_TEXTS_SENT,
                *_ASPECTS_PARAGRAPH_TEXTS_SENT,
            ]
            assert len(document.aspects[0].extracted_items) == 5
            assert not any(
                concept._is_processed for concept in document.aspects[0].concepts
            )
            assert not completions_stub.requests

            # A concept that is routed to an LLM of a model group: the LLM is
            # sent the first part only, as the concept has its item by then,
            # while the decision model extracts what is routed to it in parts
            document = create_document("document")
            other_concept, singular_concept = document.concepts
            other_concept.model_role = "decide"
            model_group.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )
            assert requested_texts() == part_texts
            assert len(completions_stub.requests) == 1
            assert [item.value for item in singular_concept.extracted_items] == [True]
            assert len(other_concept.extracted_items) == 3

            # An LLM used on its own, with no decision model to fall back to
            completions_stub.requests.clear()
            document = create_document("document")
            _create_llm().extract_concepts_from_document(
                document,
                from_concepts=[document.concepts[1]],
                max_paragraphs_to_analyze_per_call=2,
            )
            assert len(completions_stub.requests) == 1
            assert [item.value for item in document.concepts[1].extracted_items] == [
                True
            ]
            assert not decisions_stub.requests

    def test_chat_skips_decision_model_fallback(self):
        """
        Tests that a decision model that is a fallback model of an LLM is skipped
        in a chat, which only LLMs can take part in, while the fallback LLMs of
        the LLM are still tried.
        """
        auth_error = _create_api_error(litellm.AuthenticationError, _LLM_MODEL)

        def create_llm(*, with_fallback_llm: bool) -> DocumentLLM:
            """
            Creates an LLM with a fallback decision model, optionally followed
            by a fallback LLM.

            :param with_fallback_llm: Whether to add a fallback LLM.
            :type with_fallback_llm: bool
            :return: LLM.
            :rtype: DocumentLLM
            """
            fallback_models: list[Any] = [_create_decision_model(is_fallback=True)]
            if with_fallback_llm:
                fallback_models.append(
                    _create_llm(_OTHER_LLM_MODEL, system_message="", is_fallback=True)
                )
            return _create_llm(system_message="", fallback_models=fallback_models)

        async def decisions_stub(**kwargs: Any) -> Any:
            """
            Fails the test: a decision model is not requested in a chat.

            :param kwargs: Keyword arguments of ``litellm.adecisions``.
            :type kwargs: Any
            """
            raise AssertionError("A decision model is not expected to be requested.")

        skipped_log = (
            f"Fallback models that cannot chat are skipped for chat: "
            f"decision model `{_DECISION_MODEL}`"
        )

        # The LLM answers: its fallback models are not tried
        llm = create_llm(with_fallback_llm=True)
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: ["4"]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
            capture_logs() as captured_logs,
        ):
            assert llm.chat("What's 2+2?") == "4"
        assert completions_stub.requested_models() == [_LLM_MODEL]
        # The skipped fallback model is logged once per call
        assert captured_logs.count(skipped_log) == 1

        # The LLM fails: the decision model is skipped, and the fallback LLM answers
        llm = create_llm(with_fallback_llm=True)
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: [auth_error], _OTHER_LLM_MODEL: ["4"]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
            capture_logs() as captured_logs,
        ):
            assert llm.chat("What's 2+2?") == "4"
        assert completions_stub.requested_models() == [_LLM_MODEL, _OTHER_LLM_MODEL]
        assert captured_logs.count(skipped_log) == 1
        # The usage of the decision model is reported with the usage of the LLMs
        assert [
            (container.model, len(container.usage.calls))
            for container in llm.get_usage()
        ] == [(_LLM_MODEL, 1), (_DECISION_MODEL, 0), (_OTHER_LLM_MODEL, 1)]

        # A decision model is the only fallback model: the error of the LLM is final
        llm = create_llm(with_fallback_llm=False)
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: [auth_error]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
            capture_logs() as captured_logs,
            pytest.raises(ModelAPIError) as exc_info,
        ):
            llm.chat("What's 2+2?")
        assert exc_info.value.original_error is auth_error
        assert completions_stub.requested_models() == [_LLM_MODEL]
        assert captured_logs.count(skipped_log) == 1

        # The checks made on the fallback LLMs before a request is sent do not
        # apply to a decision model: it is not sent the images and has no tools
        @register_tool
        def get_contract_status(contract_id: str) -> str:
            """
            Returns the status of the contract.
            """
            return "active"

        llm = _create_llm(
            system_message="",
            tools=[get_contract_status],
            parallel_tool_calls=True,
            max_tokens=100,
            fallback_models=[_create_decision_model(is_fallback=True)],
        )
        llm.supports_vision = True
        llm.supports_parallel_tool_calls = True
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: ["An invoice."]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            assert (
                llm.chat("What is in the image?", images=[self.test_img_png_invoice])
                == "An invoice."
            )
        assert completions_stub.requested_models() == [_LLM_MODEL]

        # An LLM without a decision model among its fallback models logs nothing
        llm = _create_llm(system_message="")
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: ["4"]})
        with (
            patch.object(litellm, "acompletion", new=completions_stub),
            capture_logs() as captured_logs,
        ):
            assert llm.chat("What's 2+2?") == "4"
        assert not any("cannot chat" in log for log in captured_logs)

    def test_decision_model_serialization(self, tmp_path: Path):
        """
        Tests that a decision model, a model group with models of different kinds,
        and the fallback chains with models of different kinds are restored from
        their serialized data, in which there are no API credentials.
        """
        api_base = "https://decisions.example.com"

        def check_no_credentials(serialized_data: Any) -> int:
            """
            Checks that the serialized data of a model or a model group has no API
            credentials, including in the data of the fallback models.

            :param serialized_data: The serialized data, or a part of it.
            :type serialized_data: Any
            :return: The number of the models in the data.
            :rtype: int
            """
            n_models = 0
            if isinstance(serialized_data, dict):
                if "api_key" in serialized_data:
                    assert serialized_data["api_key"] is None
                    assert serialized_data["api_base"] is None
                    n_models += 1
                for value in serialized_data.values():
                    n_models += check_no_credentials(value)
            elif isinstance(serialized_data, list):
                for value in serialized_data:
                    n_models += check_no_credentials(value)
            return n_models

        def check_restored(
            instance: DocumentDecisionModel | DocumentLLM | DocumentModelGroup,
            n_models: int,
        ) -> None:
            """
            Checks that the instance is restored from a dict, from JSON and from
            disk, with the same config as the original one, except for the API
            credentials, which are not serialized.

            :param instance: Model or model group.
            :type instance: DocumentDecisionModel | DocumentLLM | DocumentModelGroup
            :param n_models: The number of models in the instance, including
                the fallback models.
            :type n_models: int
            """

            def check_config_eq(restored_instance: Any) -> None:
                """
                Checks that the restored instance is of the class and has
                the config of the original one.

                :param restored_instance: The restored instance.
                :type restored_instance: Any
                """
                assert type(restored_instance) is type(instance)
                if isinstance(instance, DocumentModelGroup):
                    assert instance._eq_deserialized_llm_config(restored_instance)
                else:
                    assert instance._eq_deserialized_model_config(restored_instance)

            instance_dict = instance.to_dict()
            assert check_no_credentials(instance_dict) == n_models
            check_config_eq(type(instance).from_dict(instance_dict))

            instance_json = instance.to_json()
            assert "test_api_key" not in instance_json
            assert "example.com" not in instance_json
            assert check_no_credentials(json.loads(instance_json)) == n_models
            check_config_eq(type(instance).from_json(instance_json))

            disk_path = tmp_path / f"instance_{len(list(tmp_path.iterdir()))}.json"
            instance.to_disk(disk_path)
            assert "test_api_key" not in disk_path.read_text(encoding="utf-8")
            check_config_eq(type(instance).from_disk(disk_path))

            # Pydantic-specific methods are disabled
            with pytest.raises(NotImplementedError):
                instance.model_dump()
            with pytest.raises(NotImplementedError):
                instance.model_dump_json()

        # === A decision model ===
        decision_model = _create_decision_model(
            api_base=api_base,
            role="decide",
            num_retries_failed_request=1,
            timeout=30,
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
            async_limiter=AsyncLimiter(3, 5),
        )
        check_restored(decision_model, n_models=1)
        decision_model_dict = decision_model.to_dict()
        assert decision_model_dict["__class__"] == "DocumentDecisionModel"
        # Only the params that a decision model has are serialized
        assert {key for key in decision_model_dict if not key.startswith("_")} == set(
            DocumentDecisionModel.model_fields
        )
        restored_decision_model = DocumentDecisionModel.from_dict(decision_model_dict)
        assert isinstance(restored_decision_model, DocumentDecisionModel)
        assert restored_decision_model.model == _DECISION_MODEL
        assert restored_decision_model.role == "decide"
        assert restored_decision_model.num_retries_failed_request == 1
        assert restored_decision_model.timeout == 30
        assert restored_decision_model.pricing_details == decision_model.pricing_details
        assert restored_decision_model.async_limiter.max_rate == 3
        assert restored_decision_model.async_limiter.time_period == 5
        # The API credentials are to be set again
        assert restored_decision_model.api_key is None
        assert restored_decision_model.api_base is None
        restored_decision_model.api_key = "test_api_key"
        restored_decision_model.api_base = api_base
        # A decision model is not restored as a model of another class
        with pytest.raises(TypeError, match="does not match"):
            DocumentLLM.from_dict(decision_model_dict)
        with pytest.raises(TypeError, match="does not match"):
            DocumentDecisionModel.from_dict(_create_llm().to_dict())

        # The usage and cost are not serialized
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA")
            ]
        )
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [{"C0": 0.9}]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(document)
        assert decision_model.get_usage()[0].usage.input == 100
        check_restored(decision_model, n_models=1)
        restored_decision_model = DocumentDecisionModel.from_json(
            decision_model.to_json()
        )
        assert restored_decision_model.get_usage()[0].usage == _ModelUsage()
        assert restored_decision_model.get_cost()[0].cost == _ModelCost()
        # ... while the items that the decision model extracted are serialized
        # with the document, with their probabilities
        restored_document = Document.from_json(document.to_json())
        self.check_item_probabilities(
            restored_document.concepts[0].extracted_items[0],
            {True: 0.9, False: 1 - 0.9},
        )

        # === Fallback chains with models of different kinds ===
        decision_model_with_fallbacks = _create_decision_model(
            api_base=api_base,
            role="decide",
            fallback_models=[
                _create_decision_model(
                    _OTHER_DECISION_MODEL,
                    api_base=api_base,
                    role="decide",
                    is_fallback=True,
                ),
                _create_llm(role="decide", is_fallback=True),
            ],
        )
        check_restored(decision_model_with_fallbacks, n_models=3)
        restored_decision_model = DocumentDecisionModel.from_json(
            decision_model_with_fallbacks.to_json()
        )
        assert [
            (type(model), model.model, model.role, model.is_fallback)
            for model in restored_decision_model.fallback_models
        ] == [
            (DocumentDecisionModel, _OTHER_DECISION_MODEL, "decide", True),
            (DocumentLLM, _LLM_MODEL, "decide", True),
        ]
        llm_with_fallbacks = _create_llm(
            fallback_models=[
                _create_decision_model(is_fallback=True),
                _create_llm(_OTHER_LLM_MODEL, is_fallback=True),
            ]
        )
        check_restored(llm_with_fallbacks, n_models=3)
        self._check_deserialized_llm_config_eq(llm_with_fallbacks)
        restored_llm = DocumentLLM.from_json(llm_with_fallbacks.to_json())
        assert [type(model) for model in restored_llm.fallback_models] == [
            DocumentDecisionModel,
            DocumentLLM,
        ]
        # A fallback model restored as a model of another kind is another config
        llm_dict = llm_with_fallbacks.to_dict()
        llm_dict["fallback_models"] = llm_dict["fallback_models"][::-1]
        assert not llm_with_fallbacks._eq_deserialized_llm_config(
            DocumentLLM.from_dict(llm_dict)
        )

        # === A model group with models of different kinds ===
        model_group = DocumentModelGroup(
            models=[llm_with_fallbacks, decision_model_with_fallbacks]
        )
        check_restored(model_group, n_models=6)
        self._check_deserialized_llm_config_eq(model_group)
        restored_model_group = DocumentModelGroup.from_json(model_group.to_json())
        assert [type(model) for model in restored_model_group.models] == [
            DocumentLLM,
            DocumentDecisionModel,
        ]
        assert restored_model_group.list_roles == [None, "decide"]
        # A model of a restored group with another param is another config
        model_group_dict = model_group.to_dict()
        model_group_dict["models"][1]["timeout"] = 30
        assert not model_group._eq_deserialized_llm_config(
            DocumentModelGroup.from_dict(model_group_dict)
        )

        # === Fallback models that differ only in the API credentials ===
        # The API credentials are removed from the serialized data, so such models
        # have the same config params in that data, which is accepted when
        # the models are deserialized
        decision_model_fallbacks_other_credentials = _create_decision_model(
            api_base=api_base,
            fallback_models=[
                DocumentDecisionModel(
                    model=_DECISION_MODEL, api_key="api_key_2", is_fallback=True
                ),
                DocumentDecisionModel(
                    model=_DECISION_MODEL, api_key="api_key_3", is_fallback=True
                ),
            ],
        )
        check_restored(decision_model_fallbacks_other_credentials, n_models=3)
        restored_decision_model = DocumentDecisionModel.from_json(
            decision_model_fallbacks_other_credentials.to_json()
        )
        assert len(restored_decision_model.fallback_models) == 2
        # ... and the credentials can be set again in any order
        fallback_model_1, fallback_model_2 = restored_decision_model.fallback_models
        restored_decision_model.api_key = "test_api_key"
        fallback_model_1.api_key = "api_key_2"
        restored_decision_model.timeout = 60  # an unrelated param
        restored_decision_model.api_base = api_base
        fallback_model_2.api_key = "api_key_3"
        # ... while the fallback models with the same config params are still
        # rejected when they are assigned, including to a deserialized model
        with pytest.raises(
            ValueError, match="same config params as the main decision model"
        ):
            restored_decision_model.fallback_models = [
                _create_decision_model(api_base=api_base, timeout=60, is_fallback=True)
            ]
        assert restored_decision_model.fallback_models == [
            fallback_model_1,
            fallback_model_2,
        ]
        # The other requirements for the fallback models apply to the serialized
        # data as well, e.g. the automatic pricing of a decision model
        decision_model_dict = decision_model_with_fallbacks.to_dict()
        decision_model_dict["fallback_models"][0]["role"] = "another_role"
        with pytest.raises(ValueError, match="must have the same role"):
            DocumentDecisionModel.from_dict(decision_model_dict)
        decision_model_dict = _create_decision_model(
            "openrouter/openai/gpt-6-luna"
        ).to_dict()
        decision_model_dict["auto_pricing"] = True
        with pytest.raises(ValueError, match="not supported for decision model"):
            DocumentDecisionModel.from_dict(decision_model_dict)

    def test_extract_aspects_with_decision_model(self):
        """
        Tests the extraction of aspects from a document with a decision model:
        the requests sent, one for each paragraph, and the items extracted, with
        their values, probabilities and reference paragraphs.
        """
        decision_model = _create_decision_model(
            api_base="https://decisions.example.com", timeout=30
        )
        document = _create_aspects_document(
            aspects=[
                Aspect(
                    name="Confidentiality",
                    description="Clauses on the confidentiality of the information",
                ),
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                ),
                Aspect(name="Payment", description="Clauses on the payment of fees"),
            ]
        )
        paragraphs = document.paragraphs
        # The probability that a paragraph belongs to an aspect, by the position of
        # the paragraph: 0.1 for the paragraphs that are not listed
        yes_probabilities = {
            # A paragraph with a probability of exactly 0.5 belongs to the aspect
            "Confidentiality": {0: 0.6, 1: 0.95, 4: 0.5},
            "Termination": {2: 0.9, 4: 0.7},
            # No paragraph has a probability of 0.5 or more
            "Payment": {3: 0.4999},
        }
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    _create_aspects_responder(yes_probabilities, input_tokens=20)
                ]
            }
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            extracted_aspects = decision_model.extract_aspects_from_document(document)

        # === The requests ===
        # One request for each paragraph, in the order of the paragraphs, with
        # the text of the paragraph and one question for each aspect
        questions = _build_aspect_questions(document.aspects)
        assert [question.name for question in questions] == ["A0", "A1", "A2"]
        assert "Aspect name: Termination\n" in questions[1].instructions
        assert len(decisions_stub.requests) == len(paragraphs) == 5
        for request, text_sent, paragraph in zip(
            decisions_stub.requests,
            _ASPECTS_PARAGRAPH_TEXTS_SENT,
            paragraphs,
            strict=True,
        ):
            # The questions on a paragraph that has additional context tell that
            # the text has it
            expected_questions = [
                {
                    "type": "predicate",
                    "name": question.name,
                    "instructions": question.instructions,
                }
                for question in _build_aspect_questions(
                    document.aspects,
                    has_additional_context=paragraph.additional_context is not None,
                )
            ]
            assert set(request) == {
                "model",
                "input",
                "questions",
                "api_key",
                "api_base",
                "timeout",
            }
            assert request["model"] == _DECISION_MODEL
            assert request["api_key"] == "test_api_key"
            assert request["api_base"] == "https://decisions.example.com"
            assert request["timeout"] == 30
            # The text of the paragraph, as it is shown to an LLM
            assert request["input"] == text_sent
            assert request["questions"] == expected_questions
        # The additional context of a paragraph is sent with the paragraph that
        # has it, and only with it
        assert paragraphs[2].additional_context == "Section 3. Termination"
        assert [
            "Section 3" in request["input"] for request in decisions_stub.requests
        ] == [False, False, True, False, False]

        # === The items ===
        assert len(extracted_aspects) == 3
        assert all(
            extracted_aspect is aspect
            for extracted_aspect, aspect in zip(
                extracted_aspects, document.aspects, strict=True
            )
        )
        # The paragraphs that belong to each aspect, with their probabilities
        expected_paragraphs: dict[str, list[tuple[int, float]]] = {
            "Confidentiality": [(0, 0.6), (1, 0.95), (4, 0.5)],
            "Termination": [(2, 0.9), (4, 0.7)],
            # An aspect that no paragraph belongs to is processed and has no items,
            # as an aspect for which an LLM returns no paragraphs
            "Payment": [],
        }
        for aspect in document.aspects:
            assert aspect._is_processed
            aspect_paragraphs = expected_paragraphs[aspect.name]
            # One item for each paragraph, in the order of the paragraphs
            assert len(aspect.extracted_items) == len(aspect_paragraphs)
            for item, (paragraph_position, probability) in zip(
                aspect.extracted_items, aspect_paragraphs, strict=True
            ):
                paragraph = paragraphs[paragraph_position]
                assert type(item) is _StringItem
                # The text of the paragraph as it is in the document
                assert item.value == paragraph.raw_text
                self.check_item_probabilities(
                    item, {True: probability, False: 1 - probability}
                )
                assert item.reference_paragraphs == [paragraph]
                assert item.reference_paragraphs[0].unique_id == paragraph.unique_id
                assert not item.reference_sentences
                # A decision model generates no text
                assert item.justification is None
                # The custom data of the paragraph
                assert item.custom_data == paragraph.custom_data
            assert aspect.reference_paragraphs == [
                paragraphs[paragraph_position]
                for paragraph_position, _ in aspect_paragraphs
            ]
            assert not aspect.reference_sentences
        assert document.aspects[0].extracted_items[0].value == (
            "Non-disclosure  agreement"
        )
        assert document.aspects[1].extracted_items[0].custom_data == {"page": 2}
        assert document.aspects[1].extracted_items[1].custom_data == {}

        # === The usage ===
        # Each request is recorded as a call, in the order of the paragraphs
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 5 * 20
        assert len(usage.calls) == 5
        for call, request in zip(usage.calls, decisions_stub.requests, strict=True):
            assert json.loads(call.prompt) == {
                "model": _DECISION_MODEL,
                "input": request["input"],
                "questions": request["questions"],
            }
            assert call.timestamp_received is not None
            assert call._input_tokens == 20
        assert usage.calls[0].response is not None
        assert json.loads(usage.calls[0].response) == [
            {"type": "predicate", "name": "A0", "probability": 0.6},
            {"type": "predicate", "name": "A1", "probability": 0.1},
            {"type": "predicate", "name": "A2", "probability": 0.1},
        ]

        # === Serialization ===
        self.check_instance_serialization_and_cloning(document)
        for restored_document in [
            Document.from_dict(document.to_dict()),
            Document.from_json(document.to_json()),
            document.clone(),
        ]:
            assert restored_document == document
            for restored_aspect, aspect in zip(
                restored_document.aspects, document.aspects, strict=True
            ):
                assert restored_aspect._is_processed
                assert restored_aspect.extracted_items == aspect.extracted_items
                assert restored_aspect.reference_paragraphs == (
                    aspect.reference_paragraphs
                )
                for restored_item, item in zip(
                    restored_aspect.extracted_items, aspect.extracted_items, strict=True
                ):
                    self.check_item_probabilities(restored_item, item.probabilities)
                    assert restored_item.reference_paragraphs == (
                        item.reference_paragraphs
                    )
                    assert restored_item.custom_data == item.custom_data

        # === Already processed aspects ===
        decisions_stub.requests.clear()
        decisions_stub.outcomes[_DECISION_MODEL] = [
            _create_aspects_responder({"Termination": {2: 0.9}, "Payment": {3: 0.8}})
        ]
        with patch.object(litellm, "adecisions", new=decisions_stub):
            # ... are not extracted again, before any request is sent
            with pytest.raises(ValueError, match="already been processed"):
                decision_model.extract_aspects_from_document(document)
            assert not decisions_stub.requests
            # ... unless they are to be overwritten
            decision_model.extract_aspects_from_document(
                document, overwrite_existing=True
            )
        assert len(decisions_stub.requests) == 5
        confidentiality, termination, payment = document.aspects
        assert confidentiality._is_processed and not confidentiality.extracted_items
        assert [item.value for item in termination.extracted_items] == [
            paragraphs[2].raw_text
        ]
        assert [item.value for item in payment.extracted_items] == [
            paragraphs[3].raw_text
        ]
        assert payment.reference_paragraphs == [paragraphs[3]]

        # === The params of the extraction ===
        def requests_sent() -> list[tuple[str, list[str]]]:
            """
            Returns the text and the names of the questions of each request sent.

            :return: The requests, in the order they were sent.
            :rtype: list[tuple[str, list[str]]]
            """
            return [
                (
                    request["input"],
                    [question["name"] for question in request["questions"]],
                )
                for request in decisions_stub.requests
            ]

        with patch.object(litellm, "adecisions", new=decisions_stub):
            # The number of aspects per call is limited: each call sends a request
            # for each paragraph, with the questions of its aspects. The names of
            # the questions follow the position of an aspect in its call.
            decisions_stub.requests.clear()
            decision_model.extract_aspects_from_document(
                document, overwrite_existing=True, max_items_per_call=2
            )
            assert requests_sent() == [
                *[(text, ["A0", "A1"]) for text in _ASPECTS_PARAGRAPH_TEXTS_SENT],
                *[(text, ["A0"]) for text in _ASPECTS_PARAGRAPH_TEXTS_SENT],
            ]
            assert [item.value for item in payment.extracted_items] == [
                paragraphs[3].raw_text
            ]

            # Only the aspects selected are extracted
            decisions_stub.requests.clear()
            decision_model.extract_aspects_from_document(
                document, from_aspects=[termination], overwrite_existing=True
            )
            assert requests_sent() == [
                (text, ["A0"]) for text in _ASPECTS_PARAGRAPH_TEXTS_SENT
            ]
            assert all(
                "Aspect name: Termination\n" in request["questions"][0]["instructions"]
                for request in decisions_stub.requests
            )

            # The number of paragraphs per call changes nothing for aspects: each
            # paragraph is always sent in a request of its own, i.e. any such limit
            # is met, which there is no message about
            decisions_stub.requests.clear()
            with capture_logs() as captured_logs:
                decision_model.extract_aspects_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=2,
                )
            assert requests_sent() == [
                (text, ["A0", "A1", "A2"]) for text in _ASPECTS_PARAGRAPH_TEXTS_SENT
            ]
            assert not any(
                "max_paragraphs_to_analyze_per_call" in log for log in captured_logs
            )

        # === More aspects than the questions that a request can hold ===
        # The questions on a paragraph are sent in several requests on that paragraph
        n_aspects = MAX_DECISION_QUESTIONS + 2
        document = _create_aspects_document(
            aspects=[
                Aspect(name=f"Topic {i}", description=f"Clauses on the topic {i}")
                for i in range(n_aspects)
            ]
        )
        last_topic = f"Topic {n_aspects - 1}"
        decision_model = _create_decision_model()
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    _create_aspects_responder(
                        {"Topic 0": {0: 0.8, 4: 0.7}, last_topic: {4: 0.9}},
                        input_tokens=20,
                    )
                ]
            }
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        question_names = [f"A{i}" for i in range(n_aspects)]
        assert requests_sent() == [
            (text, names)
            for text in _ASPECTS_PARAGRAPH_TEXTS_SENT
            for names in [
                question_names[:MAX_DECISION_QUESTIONS],
                question_names[MAX_DECISION_QUESTIONS:],
            ]
        ]
        assert all(aspect._is_processed for aspect in document.aspects)
        assert {
            aspect.name: [item.value for item in aspect.extracted_items]
            for aspect in document.aspects
            if aspect.extracted_items
        } == {
            "Topic 0": [
                document.paragraphs[0].raw_text,
                document.paragraphs[4].raw_text,
            ],
            last_topic: [document.paragraphs[4].raw_text],
        }
        self.check_item_probabilities(
            document.aspects[-1].extracted_items[0], {True: 0.9, False: 1 - 0.9}
        )
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 10 * 20
        assert len(usage.calls) == 10

        # === A document that has no text ===
        document = Document(images=[self.test_img_png_invoice])
        document.aspects = [
            Aspect(name="Payment", description="Clauses on the payment of fees")
        ]
        decisions_stub.requests.clear()
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(ValueError, match="Document lacks text or paragraphs"),
        ):
            decision_model.extract_aspects_from_document(document)
        assert not decisions_stub.requests

    def test_extract_sub_aspects_with_decision_model(self):
        """
        Tests the extraction of sub-aspects with a decision model: the sub-aspects
        of an aspect are extracted from the paragraphs extracted for that aspect,
        with one request for each of these paragraphs.
        """
        decision_model = _create_decision_model()
        document = _create_aspects_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                    aspects=[
                        Aspect(
                            name="Notice of termination",
                            description="Clauses on the notice that terminates "
                            "the agreement",
                        ),
                        Aspect(
                            name="Effects of termination",
                            description="Clauses on what follows the termination",
                        ),
                        Aspect(
                            name="Termination fee",
                            description="Clauses on the fee paid on the termination",
                        ),
                    ],
                ),
                Aspect(
                    name="Payment",
                    description="Clauses on the payment of fees",
                    aspects=[
                        Aspect(
                            name="Late payment",
                            description="Clauses on the payments that are late",
                        )
                    ],
                ),
            ]
        )
        paragraphs = document.paragraphs
        yes_probabilities = {
            "Termination": {2: 0.9, 4: 0.7},
            "Notice of termination": {2: 0.8, 4: 0.55},
            "Effects of termination": {4: 0.6},
            # A paragraph that does not belong to the aspect is not asked about
            # for its sub-aspects
            "Termination fee": {3: 0.9},
            "Late payment": {3: 0.9},
        }
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [_create_aspects_responder(yes_probabilities)]}
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)

        # The aspects are asked about on each paragraph of the document, and
        # the sub-aspects of an aspect on each paragraph extracted for the aspect.
        # No paragraph is extracted for the other aspect, i.e. there is nothing
        # to ask about its sub-aspect on.
        assert [
            (request["input"], [question["name"] for question in request["questions"]])
            for request in decisions_stub.requests
        ] == [
            *[(text, ["A0", "A1"]) for text in _ASPECTS_PARAGRAPH_TEXTS_SENT],
            (_ASPECTS_PARAGRAPH_TEXTS_SENT[2], ["A0", "A1", "A2"]),
            (_ASPECTS_PARAGRAPH_TEXTS_SENT[4], ["A0", "A1", "A2"]),
        ]
        assert all(
            "Aspect name: Notice of termination\n"
            in request["questions"][0]["instructions"]
            for request in decisions_stub.requests[5:]
        )

        termination, payment = document.aspects
        assert [item.value for item in termination.extracted_items] == [
            paragraphs[2].raw_text,
            paragraphs[4].raw_text,
        ]
        assert payment._is_processed and not payment.extracted_items
        expected_paragraphs: dict[str, list[tuple[int, float]]] = {
            "Notice of termination": [(2, 0.8), (4, 0.55)],
            "Effects of termination": [(4, 0.6)],
            "Termination fee": [],
        }
        for sub_aspect in termination.aspects:
            assert sub_aspect._is_processed
            sub_aspect_paragraphs = expected_paragraphs[sub_aspect.name]
            assert len(sub_aspect.extracted_items) == len(sub_aspect_paragraphs)
            for item, (paragraph_position, probability) in zip(
                sub_aspect.extracted_items, sub_aspect_paragraphs, strict=True
            ):
                paragraph = paragraphs[paragraph_position]
                assert item.value == paragraph.raw_text
                self.check_item_probabilities(
                    item, {True: probability, False: 1 - probability}
                )
                # The paragraph of the document
                assert item.reference_paragraphs == [paragraph]
                assert item.reference_paragraphs[0].unique_id == paragraph.unique_id
                assert item.custom_data == paragraph.custom_data
            assert sub_aspect.reference_paragraphs == [
                paragraphs[paragraph_position]
                for paragraph_position, _ in sub_aspect_paragraphs
            ]
        # The sub-aspect of the aspect with no paragraphs is not extracted
        (late_payment,) = payment.aspects
        assert not late_payment._is_processed
        assert not late_payment.extracted_items
        self.check_instance_serialization_and_cloning(document)

    def test_decision_model_aspect_paragraph_additional_context(self):
        """
        Tests that the additional context of a paragraph is sent to a decision
        model when aspects are extracted: the questions built for a text with
        additional context, the text of a paragraph with and without it, and
        the requests sent for the aspects and sub-aspects of a document in which
        only some of the paragraphs have additional context.
        """
        # === The questions ===
        aspects = [
            Aspect(
                name="Termination",
                description="Clauses on the termination of the agreement",
                aspects=[
                    Aspect(
                        name="Notice of termination",
                        description="Clauses on the notice that terminates "
                        "the agreement",
                    )
                ],
            ),
            Aspect(name="Payment", description="Clauses on the payment of fees"),
        ]
        questions_with_context = _build_aspect_questions(
            aspects, has_additional_context=True
        )
        assert questions_with_context == [
            _DecisionQuestion(
                name="A0",
                kind="predicate",
                instructions=(
                    "The text has one paragraph of a document and additional context "
                    "of the paragraph, such as its formatting, list context or table "
                    "position. Use the additional context to understand "
                    "the structural role and position of the paragraph in "
                    "the document. Decide whether the paragraph is relevant to "
                    "the aspect below.\n"
                    "Aspect name: Termination\n"
                    "Aspect description: Clauses on the termination of the agreement\n"
                    'Answer "yes" if the paragraph addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
            _DecisionQuestion(
                name="A1",
                kind="predicate",
                instructions=(
                    "The text has one paragraph of a document and additional context "
                    "of the paragraph, such as its formatting, list context or table "
                    "position. Use the additional context to understand "
                    "the structural role and position of the paragraph in "
                    "the document. Decide whether the paragraph is relevant to "
                    "the aspect below.\n"
                    "Aspect name: Payment\n"
                    "Aspect description: Clauses on the payment of fees\n"
                    'Answer "yes" if the paragraph addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
        ]
        for question in questions_with_context:
            self.check_rendered_prompt(question.instructions)
        # Without additional context, the questions are those asked on a paragraph
        # alone, which differ only in what the text is said to be
        questions_without_context = _build_aspect_questions(
            aspects, has_additional_context=False
        )
        assert questions_without_context == _build_aspect_questions(aspects)
        for question_with_context, question_without_context in zip(
            questions_with_context, questions_without_context, strict=True
        ):
            assert question_with_context.name == question_without_context.name
            lines_with_context = question_with_context.instructions.split("\n")
            lines_without_context = question_without_context.instructions.split("\n")
            assert lines_without_context[0] == (
                "The text is one paragraph of a document. Decide whether "
                "the paragraph is relevant to the aspect below."
            )
            assert lines_with_context[0] != lines_without_context[0]
            assert lines_with_context[1:] == lines_without_context[1:]

        # === The text of a paragraph ===
        # A paragraph without additional context: its cleaned text alone
        assert _get_paragraph_text_for_aspects(
            Paragraph(raw_text="Either  party may terminate this agreement.")
        ) == ("Either party may terminate this agreement.", False)
        # A paragraph with additional context: both, each with a label.
        # The additional context is cleaned as the text of the paragraph is.
        assert _get_paragraph_text_for_aspects(
            Paragraph(
                raw_text="Either  party may terminate this agreement.",
                additional_context=" List item  2 of 5\u200b ",
            )
        ) == (
            (
                "Paragraph: Either party may terminate this agreement.\n\n"
                "Additional context of the paragraph: List item 2 of 5"
            ),
            True,
        )
        # Additional context that has nothing to show to a model is not sent
        assert _get_paragraph_text_for_aspects(
            Paragraph(
                raw_text="Either party may terminate this agreement.",
                additional_context="\u200b",
            )
        ) == ("Either party may terminate this agreement.", False)

        # === The requests ===
        paragraph_kwargs: list[dict[str, Any]] = [
            {"raw_text": "Services  agreement"},
            {
                "raw_text": "Either party may terminate this agreement.",
                "additional_context": "Section  3. Termination\u200b",
                "custom_data": {"page": 2},
            },
            {"raw_text": "The fees are paid every month."},
            {
                "raw_text": "The same applies to the provider.",
                "additional_context": "List item 2 of 2",
            },
        ]
        texts_sent = [
            "Services agreement",
            (
                "Paragraph: Either party may terminate this agreement.\n\n"
                "Additional context of the paragraph: Section 3. Termination"
            ),
            "The fees are paid every month.",
            (
                "Paragraph: The same applies to the provider.\n\n"
                "Additional context of the paragraph: List item 2 of 2"
            ),
        ]
        has_context = [False, True, False, True]
        yes_probabilities = {
            "Termination": {1: 0.9, 3: 0.7},
            "Notice of termination": {1: 0.8},
            "Payment": {2: 0.6},
        }

        def create_document() -> Document:
            """
            Creates a document in which two of the four paragraphs have
            additional context, with the aspects to extract.

            :return: Document.
            :rtype: Document
            """
            document = Document(
                paragraphs=[Paragraph(**kwargs) for kwargs in paragraph_kwargs]
            )
            document.aspects = aspects
            return document

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request with the questions of aspects, asked on one
            paragraph of the document.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            paragraph_position = texts_sent.index(request["input"])
            answers: dict[str, Any] = {}
            for question in request["questions"]:
                aspect_name_match = re.search(
                    r"^Aspect name: (.+)$", question["instructions"], flags=re.MULTILINE
                )
                assert aspect_name_match is not None, question
                answers[question["name"]] = yes_probabilities[
                    aspect_name_match.group(1)
                ].get(paragraph_position, 0.1)
            return _create_decisions_response(request["questions"], answers)

        def get_request_questions(
            aspects: Sequence[_Aspect], has_additional_context: bool
        ) -> list[dict[str, Any]]:
            """
            Returns the questions expected in a request for a paragraph.

            :param aspects: The aspects that the paragraph is asked about.
            :type aspects: Sequence[_Aspect]
            :param has_additional_context: Whether the paragraph has additional
                context.
            :type has_additional_context: bool
            :return: The questions of the request.
            :rtype: list[dict[str, Any]]
            """
            return [
                {
                    "type": "predicate",
                    "name": question.name,
                    "instructions": question.instructions,
                }
                for question in _build_aspect_questions(
                    aspects, has_additional_context=has_additional_context
                )
            ]

        # The questions built for the requests: those built to check what
        # a decision model can extract are built without the param
        questions_built: list[tuple[list[str], bool]] = []

        def build_aspect_questions(*args: Any, **kwargs: Any) -> Any:
            """
            Builds the questions of aspects, and records for which aspects and
            for which kind of text they are built.

            :param args: Arguments of the function that builds the questions.
            :type args: Any
            :param kwargs: Keyword arguments of the function.
            :type kwargs: Any
            :return: The questions.
            :rtype: Any
            """
            if "has_additional_context" in kwargs:
                questions_built.append(
                    (
                        [aspect.name for aspect in args[0]],
                        kwargs["has_additional_context"],
                    )
                )
            return _build_aspect_questions(*args, **kwargs)

        decision_model = _create_decision_model()
        document = create_document()
        paragraphs = document.paragraphs
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                decision_models_module,
                "_build_aspect_questions",
                new=build_aspect_questions,
            ),
        ):
            decision_model.extract_aspects_from_document(document)

        termination, payment = document.aspects
        (notice,) = termination.aspects
        assert [
            (request["input"], request["questions"])
            for request in decisions_stub.requests
        ] == [
            # The aspects, on each paragraph of the document: a paragraph that has
            # additional context is sent with it, and with the questions that tell
            # that the text has it, and the other paragraphs as their texts alone
            *[
                (text, get_request_questions(document.aspects, with_context))
                for text, with_context in zip(texts_sent, has_context, strict=True)
            ],
            # The sub-aspect, on each paragraph extracted for its aspect, which
            # are the paragraphs of the document, i.e. with their additional context
            (texts_sent[1], get_request_questions([notice], True)),
            (texts_sent[3], get_request_questions([notice], True)),
        ]
        # The questions differ between the paragraphs with and without additional
        # context only
        assert (
            decisions_stub.requests[1]["questions"]
            == decisions_stub.requests[3]["questions"]
            != decisions_stub.requests[0]["questions"]
            == decisions_stub.requests[2]["questions"]
        )
        assert all(
            ("additional context" in question["instructions"]) is with_context
            for request, with_context in zip(
                decisions_stub.requests, [*has_context, True, True], strict=True
            )
            for question in request["questions"]
        )
        # Each variant of the questions is built once per call, and only if
        # a paragraph of the call is asked about with it
        assert questions_built == [
            (["Termination", "Payment"], False),
            (["Termination", "Payment"], True),
            (["Notice of termination"], True),
        ]
        # The request is recorded in the usage as sent
        usage = decision_model.get_usage()[0].usage
        assert json.loads(usage.calls[1].prompt) == {
            "model": _DECISION_MODEL,
            "input": texts_sent[1],
            "questions": get_request_questions(document.aspects, True),
        }

        # The items are the paragraphs as they are in the document, i.e. without
        # the labels and the additional context
        assert [
            [(item.value, item.reference_paragraphs) for item in aspect.extracted_items]
            for aspect in [termination, notice, payment]
        ] == [
            [
                (paragraphs[1].raw_text, [paragraphs[1]]),
                (paragraphs[3].raw_text, [paragraphs[3]]),
            ],
            [(paragraphs[1].raw_text, [paragraphs[1]])],
            [(paragraphs[2].raw_text, [paragraphs[2]])],
        ]
        assert termination.extracted_items[0].value == (
            "Either party may terminate this agreement."
        )
        assert termination.extracted_items[0].custom_data == {"page": 2}
        self.check_item_probabilities(
            notice.extracted_items[0], {True: 0.8, False: 1 - 0.8}
        )
        self.check_instance_serialization_and_cloning(document)

        # === A response that cannot be used ===
        # The log names the paragraph by its position, as for a paragraph
        # without additional context
        for failed_position in [1, 2]:
            # One call, without retries for invalid data
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = create_document()

            def respond_or_fail(
                request: dict[str, Any], failed_position: int = failed_position
            ) -> OpenAIDecisionResponse:
                """
                Responds to the request for a paragraph, with an answer that
                cannot be used to the second question on one of the paragraphs.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :param failed_position: The position of the paragraph with
                    the answer that cannot be used.
                :type failed_position: int
                :return: Response of litellm.
                :rtype: OpenAIDecisionResponse
                """
                if request["input"] == texts_sent[failed_position]:
                    return _create_decisions_response(
                        request["questions"], {"A0": 0.9, "A1": 1.5}
                    )
                return respond(request)

            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_or_fail]})
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.ERROR) as captured_logs,
                pytest.raises(ModelExtractionError),
            ):
                decision_model.extract_aspects_from_document(document)
            assert (
                f"Decision model `{_DECISION_MODEL}` returned an invalid response "
                f"for aspect `Payment` on paragraph {failed_position + 1} of 4 "
                f"(index {failed_position}), decision question `A1`: "
                "the probability of the answer is not a number within [0, 1]."
            ) in captured_logs, captured_logs
            assert len(decisions_stub.requests) == failed_position + 1
            assert not any(aspect._is_processed for aspect in document.aspects)

    def test_decision_model_whole_paragraph_sentence(self):
        """
        Tests which paragraphs a decision model is not asked about the sentence
        of, which takes the answers on the paragraph instead: those with one
        sentence that is the paragraph as a whole, i.e. with the text of
        the paragraph as the model is shown it, and with no additional context of
        its own, as a table row converted from Docling has.
        """
        text = "Invoices are payable within 30 days."
        assert _is_whole_paragraph_sentence(
            Paragraph(raw_text=text, sentences=[Sentence(raw_text=text)])
        )
        # The texts are compared as the model is shown them
        assert _is_whole_paragraph_sentence(
            Paragraph(
                raw_text=" Invoices are  payable\nwithin 30 days. ",
                sentences=[Sentence(raw_text="Invoices are  payable\nwithin 30 days.")],
            )
        )
        # The additional context of the paragraph, which a sentence segmented from
        # the paragraph has
        assert _is_whole_paragraph_sentence(
            Paragraph(
                raw_text=text,
                sentences=[Sentence(raw_text=text, additional_context="Table row")],
                additional_context="Table row",
            )
        )
        # Additional context of its own, which the model is shown with
        # the sentence
        assert not _is_whole_paragraph_sentence(
            Paragraph(
                raw_text=text,
                sentences=[Sentence(raw_text=text, additional_context="Footnote 2")],
                additional_context="Table row",
            )
        )
        assert not _is_whole_paragraph_sentence(
            Paragraph(
                raw_text=text,
                sentences=[Sentence(raw_text=text, additional_context="Footnote 2")],
            )
        )
        # A part of the paragraph
        assert not _is_whole_paragraph_sentence(
            Paragraph(
                raw_text=f"{text} Late payments bear interest.",
                sentences=[Sentence(raw_text=text)],
            )
        )
        # Several sentences, or none
        assert not _is_whole_paragraph_sentence(
            Paragraph(
                raw_text=f"{text} {text}",
                sentences=[Sentence(raw_text=text), Sentence(raw_text=text)],
            )
        )
        assert not _is_whole_paragraph_sentence(Paragraph(raw_text=text))

    def test_decision_model_aspect_sentence_questions_and_texts(self):
        """
        Tests the questions built for aspects to be asked on one sentence, for
        the aspects that are extracted with sentences as their items, and the text
        of a sentence that such questions are asked on.
        """
        aspects = [
            Aspect(
                name="Termination",
                description="Clauses on the termination of the agreement",
                reference_depth="sentences",
            ),
            Aspect(name="Payment", description="Clauses on the payment of fees"),
        ]

        # === The questions ===
        questions = _build_aspect_questions(aspects, text_unit="sentence")
        assert questions == [
            _DecisionQuestion(
                name="A0",
                kind="predicate",
                instructions=(
                    "The text has one sentence of a document and the paragraph "
                    "that the sentence is from, given as its context. Decide "
                    "whether the sentence is relevant to the aspect below.\n"
                    "Aspect name: Termination\n"
                    "Aspect description: Clauses on the termination of the agreement\n"
                    'Answer "yes" if the sentence addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
            _DecisionQuestion(
                name="A1",
                kind="predicate",
                instructions=(
                    "The text has one sentence of a document and the paragraph "
                    "that the sentence is from, given as its context. Decide "
                    "whether the sentence is relevant to the aspect below.\n"
                    "Aspect name: Payment\n"
                    "Aspect description: Clauses on the payment of fees\n"
                    'Answer "yes" if the sentence addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
        ]
        # The text of a sentence may also have additional context
        questions_with_context = _build_aspect_questions(
            aspects, has_additional_context=True, text_unit="sentence"
        )
        assert questions_with_context == [
            _DecisionQuestion(
                name="A0",
                kind="predicate",
                instructions=(
                    "The text has one sentence of a document, the paragraph that "
                    "the sentence is from, given as its context, and additional "
                    "context of the sentence, of the paragraph, or of both, such as "
                    "formatting, list context or table position. Use the additional "
                    "context to understand the structural role and position of "
                    "the sentence in the document. Decide whether the sentence is "
                    "relevant to the aspect below.\n"
                    "Aspect name: Termination\n"
                    "Aspect description: Clauses on the termination of the agreement\n"
                    'Answer "yes" if the sentence addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
            _DecisionQuestion(
                name="A1",
                kind="predicate",
                instructions=(
                    "The text has one sentence of a document, the paragraph that "
                    "the sentence is from, given as its context, and additional "
                    "context of the sentence, of the paragraph, or of both, such as "
                    "formatting, list context or table position. Use the additional "
                    "context to understand the structural role and position of "
                    "the sentence in the document. Decide whether the sentence is "
                    "relevant to the aspect below.\n"
                    "Aspect name: Payment\n"
                    "Aspect description: Clauses on the payment of fees\n"
                    'Answer "yes" if the sentence addresses or clearly relates to '
                    'the aspect. Answer "no" if it does not.'
                ),
                options=(),
            ),
        ]
        for question in [*questions, *questions_with_context]:
            self.check_rendered_prompt(question.instructions)
        # The questions on a paragraph are those built by default: the reference
        # depth of an aspect does not change its question on a paragraph
        for has_additional_context in [False, True]:
            questions_on_paragraph = _build_aspect_questions(
                aspects, has_additional_context=has_additional_context
            )
            assert questions_on_paragraph == _build_aspect_questions(
                aspects,
                has_additional_context=has_additional_context,
                text_unit="paragraph",
            )
            assert all(
                "sentence" not in question.instructions
                for question in questions_on_paragraph
            )
        assert _build_aspect_questions(aspects)[0].instructions == (
            "The text is one paragraph of a document. Decide whether "
            "the paragraph is relevant to the aspect below.\n"
            "Aspect name: Termination\n"
            "Aspect description: Clauses on the termination of the agreement\n"
            'Answer "yes" if the paragraph addresses or clearly relates to '
            'the aspect. Answer "no" if it does not.'
        )

        # === The text of a sentence ===
        # The sentence, which is what is judged, and its paragraph, each with
        # a label, with the additional context of the sentence and of the paragraph
        # where they are set
        document = _create_sentences_document()
        sentences_with_context = {(1, 0), (1, 1), (1, 2), (3, 1)}
        for paragraph_position, paragraph in enumerate(document.paragraphs):
            for sentence_position, sentence in enumerate(paragraph.sentences):
                position = (paragraph_position, sentence_position)
                assert _get_sentence_text_for_aspects(sentence, paragraph) == (
                    _SENTENCES_TEXTS_SENT[position],
                    position in sentences_with_context,
                ), position
        # The texts and the additional context are cleaned as the text of
        # a paragraph is
        assert "Services  agreement" in document.paragraphs[0].raw_text
        assert document.paragraphs[1].sentences[1].additional_context == (
            "List item  2 of 3"
        )
        # Additional context that a sentence shares with its paragraph, as
        # a sentence segmented from the paragraph does, is given once, as that
        # of the paragraph, also when it differs in what is cleaned off
        paragraph = Paragraph(
            raw_text="The notice shall be given in writing. It is sent by post.",
            additional_context="Section  3. Termination",
            sentences=[
                Sentence(
                    raw_text="The notice shall be given in writing.",
                    additional_context="Section 3. Termination\u200b",
                ),
                Sentence(raw_text="It is sent by post."),
            ],
        )
        assert _get_sentence_text_for_aspects(paragraph.sentences[0], paragraph) == (
            (
                "Sentence: The notice shall be given in writing.\n\n"
                "Paragraph that the sentence is from: The notice shall be given in "
                "writing. It is sent by post.\n\n"
                "Additional context of the paragraph: Section 3. Termination"
            ),
            True,
        )
        # The additional context of the paragraph is given for each sentence
        assert _get_sentence_text_for_aspects(paragraph.sentences[1], paragraph) == (
            (
                "Sentence: It is sent by post.\n\n"
                "Paragraph that the sentence is from: The notice shall be given in "
                "writing. It is sent by post.\n\n"
                "Additional context of the paragraph: Section 3. Termination"
            ),
            True,
        )

    def test_extract_aspects_with_sentence_reference_depth(self):
        """
        Tests the extraction of aspects with sentences as the extracted items
        (``reference_depth="sentences"``) with a decision model, in two stages:
        the requests sent, one for each paragraph and then one for each sentence
        of the paragraphs that belong to such an aspect, and the items extracted,
        with their values, probabilities, custom data and references.
        """
        # The probability that a paragraph, or a sentence, belongs to an aspect:
        # 0.1 for those that are not listed
        yes_probabilities: dict[str, dict[int | tuple[int, int], float]] = {
            "Termination": {
                1: 0.9,
                # A paragraph with a probability of exactly 0.5 belongs to
                # the aspect, while none of its sentences does
                2: 0.5,
                3: 0.6,
                (1, 0): 0.95,
                # A sentence with a probability of exactly 0.5 belongs to the aspect
                (1, 1): 0.5,
                (1, 2): 0.2,
                (2, 0): 0.4,
                (2, 1): 0.2,
                (3, 0): 0.3,
                (3, 1): 0.85,
            },
            # The aspect has a decision threshold of 0.7 of its own
            "Payment": {
                1: 0.75,
                2: 0.8,
                # A paragraph below the threshold: its sentences are not asked
                # about for the aspect
                3: 0.65,
                (1, 2): 0.72,
                (2, 0): 0.9,
                # A sentence below the threshold
                (2, 1): 0.69,
            },
            # No paragraph belongs to the aspect
            "Governing law": {4: 0.4},
        }

        def create_document() -> Document:
            """
            Creates a document with three aspects that are extracted with
            sentences as their items.

            :return: Document.
            :rtype: Document
            """
            return _create_sentences_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        reference_depth="sentences",
                    ),
                    Aspect(
                        name="Payment",
                        description="Clauses on the payment of fees",
                        reference_depth="sentences",
                        decision_threshold=0.7,
                    ),
                    Aspect(
                        name="Governing law",
                        description="Clauses on the governing law",
                        reference_depth="sentences",
                    ),
                ]
            )

        def get_request_questions(
            aspects: Sequence[_Aspect],
            aspect_positions: list[int],
            has_additional_context: bool,
            text_unit: Literal["paragraph", "sentence"],
        ) -> list[dict[str, Any]]:
            """
            Returns the questions expected in a request for a paragraph or
            for a sentence.

            :param aspects: The aspects of the call.
            :type aspects: Sequence[_Aspect]
            :param aspect_positions: The positions of the aspects that
                the request has the questions of.
            :type aspect_positions: list[int]
            :param has_additional_context: Whether the text of the request has
                additional context.
            :type has_additional_context: bool
            :param text_unit: What the request is for.
            :type text_unit: Literal["paragraph", "sentence"]
            :return: The questions of the request.
            :rtype: list[dict[str, Any]]
            """
            questions = _build_aspect_questions(
                aspects,
                has_additional_context=has_additional_context,
                text_unit=text_unit,
            )
            return [
                {
                    "type": "predicate",
                    "name": questions[position].name,
                    "instructions": questions[position].instructions,
                }
                for position in aspect_positions
            ]

        # The questions built for the requests: those built to check what
        # a decision model can extract are built without the params
        questions_built: list[tuple[int, bool, str]] = []

        def build_aspect_questions(*args: Any, **kwargs: Any) -> Any:
            """
            Builds the questions of aspects, and records for how many aspects and
            for which kind of text they are built.

            :param args: Arguments of the function that builds the questions.
            :type args: Any
            :param kwargs: Keyword arguments of the function.
            :type kwargs: Any
            :return: The questions.
            :rtype: Any
            """
            if "has_additional_context" in kwargs:
                questions_built.append(
                    (
                        len(args[0]),
                        kwargs["has_additional_context"],
                        kwargs.get("text_unit", "paragraph"),
                    )
                )
            return _build_aspect_questions(*args, **kwargs)

        decision_model = _create_decision_model()
        document = create_document()
        paragraphs = document.paragraphs
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    _create_sentences_responder(yes_probabilities, input_tokens=20)
                ]
            }
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                decision_models_module,
                "_build_aspect_questions",
                new=build_aspect_questions,
            ),
        ):
            extracted_aspects = decision_model.extract_aspects_from_document(document)
        assert [aspect.name for aspect in extracted_aspects] == [
            "Termination",
            "Payment",
            "Governing law",
        ]
        termination, payment, governing_law = document.aspects

        # === The requests ===
        # First, one request for each paragraph, with one question for each
        # aspect, as for the aspects extracted with paragraphs as their items.
        # Then, one request for each sentence of each paragraph that belongs to
        # an aspect, with one question for each of the aspects that the paragraph
        # belongs to, named by the position of the aspect in the call.
        assert [
            (request["input"], request["questions"])
            for request in decisions_stub.requests
        ] == [
            *[
                (
                    text,
                    get_request_questions(
                        document.aspects,
                        [0, 1, 2],
                        # Only the second paragraph has additional context
                        has_additional_context=position == 1,
                        text_unit="paragraph",
                    ),
                )
                for position, text in enumerate(_SENTENCES_PARAGRAPH_TEXTS_SENT)
            ],
            # The second paragraph belongs to the first two aspects. The text of
            # each of its sentences has the additional context of the paragraph.
            *[
                (
                    _SENTENCES_TEXTS_SENT[(1, sentence_position)],
                    get_request_questions(document.aspects, [0, 1], True, "sentence"),
                )
                for sentence_position in range(3)
            ],
            # The third paragraph belongs to the first two aspects as well
            *[
                (
                    _SENTENCES_TEXTS_SENT[(2, sentence_position)],
                    get_request_questions(document.aspects, [0, 1], False, "sentence"),
                )
                for sentence_position in range(2)
            ],
            # The fourth paragraph belongs to the first aspect only. Its second
            # sentence has additional context of its own.
            (
                _SENTENCES_TEXTS_SENT[(3, 0)],
                get_request_questions(document.aspects, [0], False, "sentence"),
            ),
            (
                _SENTENCES_TEXTS_SENT[(3, 1)],
                get_request_questions(document.aspects, [0], True, "sentence"),
            ),
        ]
        # The number of requests: the paragraphs, and the sentences of
        # the paragraphs that belong to an aspect
        assert len(decisions_stub.requests) == 5 + (3 + 2 + 2)
        assert [
            [question["name"] for question in request["questions"]]
            for request in decisions_stub.requests
        ] == [["A0", "A1", "A2"]] * 5 + [["A0", "A1"]] * 5 + [["A0"]] * 2
        # The request for a paragraph of an aspect that is extracted with
        # sentences is the one for an aspect that is extracted with paragraphs
        assert decisions_stub.requests[0]["questions"] == [
            {
                "type": "predicate",
                "name": question.name,
                "instructions": question.instructions,
            }
            for question in _build_aspect_questions(
                [
                    Aspect(name=aspect.name, description=aspect.description)
                    for aspect in document.aspects
                ]
            )
        ]
        # Each variant of the questions is built once per call, for all
        # the aspects of the call, and only if a text is asked about with it
        assert questions_built == [
            (3, False, "paragraph"),
            (3, True, "paragraph"),
            (3, True, "sentence"),
            (3, False, "sentence"),
        ]

        # === The items ===
        # The sentences that belong to each aspect, by the positions of
        # the paragraph and of the sentence in it, with their probabilities
        expected_sentences: dict[str, list[tuple[tuple[int, int], float]]] = {
            # No sentence of the third paragraph belongs to the aspect, i.e.
            # the paragraph gives no items, while it belongs to the aspect
            "Termination": [((1, 0), 0.95), ((1, 1), 0.5), ((3, 1), 0.85)],
            "Payment": [((1, 2), 0.72), ((2, 0), 0.9)],
            # An aspect that no paragraph belongs to is processed and has no items
            "Governing law": [],
        }
        for aspect in document.aspects:
            assert aspect._is_processed
            aspect_sentences = expected_sentences[aspect.name]
            # One item for each sentence, in the order of the document
            assert len(aspect.extracted_items) == len(aspect_sentences)
            for item, ((paragraph_position, sentence_position), probability) in zip(
                aspect.extracted_items, aspect_sentences, strict=True
            ):
                paragraph = paragraphs[paragraph_position]
                sentence = paragraph.sentences[sentence_position]
                assert type(item) is _StringItem
                # The text of the sentence as it is in the document
                assert item.value == sentence.raw_text
                # The probabilities of the sentence, not those of its paragraph
                self.check_item_probabilities(
                    item, {True: probability, False: 1 - probability}
                )
                assert item.reference_paragraphs == [paragraph]
                assert item.reference_paragraphs[0] is paragraph
                assert item.reference_sentences == [sentence]
                assert item.reference_sentences[0] is sentence
                # A decision model generates no text
                assert item.justification is None
                # The custom data of the sentence, not that of its paragraph
                assert item.custom_data == sentence.custom_data
            # The references of the aspect, as when an LLM extracts it: those of
            # its items, i.e. the paragraphs of the extracted sentences, each
            # once, and the extracted sentences, in the order of the items. The
            # sentences of these paragraphs that are not extracted, e.g. the third
            # sentence of the second paragraph for the first aspect, are not
            # among the reference sentences of the aspect.
            reference_paragraphs = [
                paragraphs[paragraph_position]
                for paragraph_position in dict.fromkeys(
                    paragraph_position
                    for (paragraph_position, _), _ in aspect_sentences
                )
            ]
            assert aspect.reference_paragraphs == reference_paragraphs
            assert aspect.reference_sentences == [
                paragraphs[paragraph_position].sentences[sentence_position]
                for (paragraph_position, sentence_position), _ in aspect_sentences
            ]
        assert [item.value for item in termination.extracted_items] == [
            "Either party may terminate this agreement with a notice of 30 days.",
            "The notice shall be given in writing.",
            "Termination does not affect the accrued rights.",
        ]
        assert [item.custom_data for item in termination.extracted_items] == [
            {},
            {"line": 7},
            {},
        ]
        assert paragraphs[1].custom_data == {"page": 2}
        assert len(termination.reference_paragraphs) == 2
        # Three of the five sentences of the two reference paragraphs are
        # extracted, and only these are the reference sentences of the aspect
        assert len(paragraphs[1].sentences) + len(paragraphs[3].sentences) == 5
        assert len(termination.reference_sentences) == 3
        assert paragraphs[2] not in termination.reference_paragraphs
        assert not governing_law.reference_paragraphs
        assert not governing_law.reference_sentences
        assert [item.value for item in payment.extracted_items] == [
            "The fees paid are not refunded.",
            "The client shall pay the fees every month.",
        ]

        # === The usage ===
        # Each request is recorded as a call: those for the paragraphs, and then
        # those for the sentences
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 12 * 20
        assert len(usage.calls) == 12
        for call, request in zip(usage.calls, decisions_stub.requests, strict=True):
            assert json.loads(call.prompt) == {
                "model": _DECISION_MODEL,
                "input": request["input"],
                "questions": request["questions"],
            }
            assert call.timestamp_received is not None
            assert call._input_tokens == 20
        assert usage.calls[6].response is not None
        assert json.loads(usage.calls[6].response) == [
            {"type": "predicate", "name": "A0", "probability": 0.5},
            {"type": "predicate", "name": "A1", "probability": 0.1},
        ]

        # === Serialization ===
        self.check_instance_serialization_and_cloning(document)
        for restored_document in [
            Document.from_dict(document.to_dict()),
            Document.from_json(document.to_json()),
            document.clone(),
        ]:
            assert restored_document == document
            for restored_aspect, aspect in zip(
                restored_document.aspects, document.aspects, strict=True
            ):
                assert restored_aspect._is_processed
                assert restored_aspect.extracted_items == aspect.extracted_items
                assert restored_aspect.reference_paragraphs == (
                    aspect.reference_paragraphs
                )
                assert restored_aspect.reference_sentences == (
                    aspect.reference_sentences
                )
                for restored_item, item in zip(
                    restored_aspect.extracted_items, aspect.extracted_items, strict=True
                ):
                    self.check_item_probabilities(restored_item, item.probabilities)
                    assert restored_item.value == item.value
                    assert restored_item.reference_paragraphs == (
                        item.reference_paragraphs
                    )
                    assert restored_item.reference_sentences == (
                        item.reference_sentences
                    )
                    assert restored_item.custom_data == item.custom_data

        # === Already processed aspects ===
        decisions_stub.requests.clear()
        decisions_stub.outcomes[_DECISION_MODEL] = [
            _create_sentences_responder(
                {"Termination": {3: 0.9, (3, 0): 0.8}, "Governing law": {4: 0.9}}
            )
        ]
        with patch.object(litellm, "adecisions", new=decisions_stub):
            # ... are not extracted again, before any request is sent
            with pytest.raises(ValueError, match="already been processed"):
                decision_model.extract_aspects_from_document(document)
            assert not decisions_stub.requests
            # ... unless they are to be overwritten: the references of an aspect
            # are then those of its new items only
            decision_model.extract_aspects_from_document(
                document, overwrite_existing=True
            )
        # The only sentence of the last paragraph is the paragraph as a whole: it
        # is not asked about, and has the answer on the paragraph
        assert [request["input"] for request in decisions_stub.requests] == [
            *_SENTENCES_PARAGRAPH_TEXTS_SENT,
            _SENTENCES_TEXTS_SENT[(3, 0)],
            _SENTENCES_TEXTS_SENT[(3, 1)],
        ]
        assert [item.value for item in termination.extracted_items] == [
            paragraphs[3].sentences[0].raw_text
        ]
        assert termination.reference_paragraphs == [paragraphs[3]]
        # The sentence of the new item: the second sentence of its paragraph is
        # not extracted, i.e. it is not a reference sentence of the aspect
        assert len(paragraphs[3].sentences) == 2
        assert termination.reference_sentences == [paragraphs[3].sentences[0]]
        assert payment._is_processed and not payment.extracted_items
        assert not payment.reference_paragraphs
        assert not payment.reference_sentences
        assert [item.value for item in governing_law.extracted_items] == [
            paragraphs[4].sentences[0].raw_text
        ]
        self.check_item_probabilities(
            governing_law.extracted_items[0], {True: 0.9, False: 1 - 0.9}
        )
        assert governing_law.reference_paragraphs == [paragraphs[4]]
        assert governing_law.reference_sentences == [paragraphs[4].sentences[0]]
        # Without overwriting, the references that an aspect has are kept, and
        # those of its items are added to them, as when an LLM extracts it
        document = create_document()
        paragraphs = document.paragraphs
        termination = document.aspects[0]
        termination.reference_paragraphs = [paragraphs[0]]
        termination.reference_sentences = [paragraphs[0].sentences[0]]
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert [id(paragraph) for paragraph in termination.reference_paragraphs] == [
            id(paragraphs[0]),
            id(paragraphs[3]),
        ]
        # ... i.e. the sentence of the new item, without the other sentence of
        # its paragraph
        assert [id(sentence) for sentence in termination.reference_sentences] == [
            id(paragraphs[0].sentences[0]),
            id(paragraphs[3].sentences[0]),
        ]

    def test_extract_sentence_depth_aspect_from_docling_document(self):
        """
        Tests the extraction of an aspect with sentences as the extracted items by
        a decision model from a document converted from Docling, whose table rows
        are one sentence each, their whole text. The SaT model segments only
        the other paragraphs, when the aspect needs their sentences. A table row
        that belongs to the aspect is an item as a whole, with its pages and
        the probabilities of the paragraph, without a request for its sentence,
        as is any paragraph whose one sentence is the paragraph as a whole. Each
        sentence of the other paragraphs that belong to the aspect is asked about.
        """
        docling_document = DoclingDocument.load_from_json(
            Path(__file__).parent
            / "other_files"
            / "docling"
            / "service_agreement.pdf.json"
        )
        document = Document.from_docling(docling_document)
        document.aspects = [
            Aspect(
                name="Fees",
                description="The fees and their payment",
                reference_depth="sentences",
            )
        ]
        rows = [
            paragraph
            for paragraph in document.paragraphs
            if (paragraph.additional_context or "").startswith("Table row")
        ]
        assert len(rows) == 3
        assert all(len(row.sentences) == 1 for row in rows)

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Answers that a paragraph or a sentence on fees or invoices belongs to
            the aspect, by the text that the model judges, i.e. without
            the paragraph that a sentence is from.
            """
            judged = request["input"].split("\n\n", 1)[0].lower()
            probability = 0.9 if "fee" in judged or "invoice" in judged else 0.1
            return _create_decisions_response(
                request["questions"],
                {question["name"]: probability for question in request["questions"]},
            )

        texts_segmented: list[str] = []
        split_into_sentences = documents_module._split_into_sentences

        def record_split(model: Any, texts: list[str]) -> list[list[str]]:
            """
            Records the texts that the SaT model segments into sentences.
            """
            texts_segmented.extend(texts)
            return split_into_sentences(model, texts)

        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(documents_module, "_split_into_sentences", new=record_split),
        ):
            _create_decision_model().extract_aspects_from_document(document)

        # The SaT model segments the paragraphs that have no sentences, not
        # the table rows
        assert texts_segmented
        assert not {row.raw_text for row in rows} & set(texts_segmented)
        # One request for each paragraph, and one for each sentence of
        # the paragraphs that belong to the aspect, but those whose one sentence
        # is the paragraph as a whole
        belonging = [
            paragraph
            for paragraph in document.paragraphs
            if "fee" in paragraph.raw_text.lower()
            or "invoice" in paragraph.raw_text.lower()
        ]
        asked = [
            paragraph
            for paragraph in belonging
            if not (
                len(paragraph.sentences) == 1
                and paragraph.sentences[0].raw_text == paragraph.raw_text
            )
        ]
        assert set(map(id, rows)) <= set(map(id, belonging)) - set(map(id, asked))
        sentence_requests = [
            request["input"]
            for request in decisions_stub.requests
            if request["input"].startswith("Sentence: ")
        ]
        assert len(decisions_stub.requests) == len(document.paragraphs) + len(
            sentence_requests
        )
        assert len(sentence_requests) == sum(
            len(paragraph.sentences) for paragraph in asked
        )
        assert not [
            text
            for text in sentence_requests
            for row in rows
            if text.startswith(f"Sentence: {row.raw_text}")
        ]
        # Each table row is an item as a whole, with the probabilities and
        # the pages of its paragraph
        (fees,) = document.aspects
        items_of_rows = [
            item
            for item in fees.extracted_items
            if item.reference_paragraphs[0] in rows
        ]
        assert [item.value for item in items_of_rows] == [row.raw_text for row in rows]
        for item, row in zip(items_of_rows, rows, strict=True):
            assert item.reference_sentences == row.sentences
            assert item.reference_sentences[0] is row.sentences[0]
            assert item.reference_page_numbers == row.page_numbers == [1]
            self.check_item_probabilities(item, {True: 0.9, False: 1 - 0.9})

    def test_sentence_depth_aspects_decision_threshold(self):
        """
        Tests that the decision threshold applies at both stages of the extraction
        of aspects with sentences as the extracted items: to the paragraphs, whose
        sentences are asked about only if the paragraph belongs to the aspect, and
        to the sentences. The threshold is that of the aspect, or that of
        the decision model if the aspect has none of its own.
        """
        # The same probabilities for each aspect
        probabilities: dict[int | tuple[int, int], float] = {
            1: 0.9,
            2: 0.6,
            3: 0.4,
            (1, 0): 0.9,
            (1, 1): 0.6,
            (1, 2): 0.4,
            (2, 0): 0.8,
            (2, 1): 0.35,
            (3, 0): 0.75,
            (3, 1): 0.3,
        }
        respond = _create_sentences_responder(
            {
                "Default": probabilities,
                "Lenient": probabilities,
                "Strict": probabilities,
            }
        )

        def extract(
            **model_kwargs: Any,
        ) -> tuple[dict[str, list[tuple[int, int]]], dict[int, list[str]]]:
            """
            Extracts the aspects of a new document with a new decision model.

            :param model_kwargs: Params of the decision model.
            :type model_kwargs: Any
            :return: The sentences extracted for each aspect, by the name of
                the aspect, as the positions of the paragraph and of the sentence
                in it, and the names of the questions asked on the sentences of
                each paragraph, by the position of the paragraph.
            :rtype: tuple[dict[str, list[tuple[int, int]]], dict[int, list[str]]]
            """
            decision_model = _create_decision_model(**model_kwargs)
            document = _create_sentences_document(
                aspects=[
                    Aspect(
                        name="Default",
                        description="Clauses of the default kind",
                        reference_depth="sentences",
                    ),
                    Aspect(
                        name="Lenient",
                        description="Clauses of the lenient kind",
                        reference_depth="sentences",
                        decision_threshold=0.3,
                    ),
                    Aspect(
                        name="Strict",
                        description="Clauses of the strict kind",
                        reference_depth="sentences",
                        decision_threshold=0.7,
                    ),
                ]
            )
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_aspects_from_document(document)
            sentence_positions = {
                id(sentence): (paragraph_position, sentence_position)
                for paragraph_position, paragraph in enumerate(document.paragraphs)
                for sentence_position, sentence in enumerate(paragraph.sentences)
            }
            extracted: dict[str, list[tuple[int, int]]] = {}
            for aspect in document.aspects:
                assert aspect._is_processed
                extracted[aspect.name] = []
                for item in aspect.extracted_items:
                    (sentence,) = item.reference_sentences
                    position = sentence_positions[id(sentence)]
                    extracted[aspect.name].append(position)
                    # The probabilities are those of the answer on the sentence,
                    # whatever threshold the sentence is extracted with
                    self.check_item_probabilities(
                        item,
                        {
                            True: probabilities[position],
                            False: 1 - probabilities[position],
                        },
                    )
            questions_on_sentences: dict[int, list[str]] = {}
            for request in decisions_stub.requests[5:]:
                position = _get_sentences_request_position(request)
                assert isinstance(position, tuple)
                question_names = [question["name"] for question in request["questions"]]
                # The same questions on each sentence of a paragraph
                assert (
                    questions_on_sentences.setdefault(position[0], question_names)
                    == question_names
                )
            # Each sentence of the paragraphs that are asked about
            assert len(decisions_stub.requests) == 5 + sum(
                len(_SENTENCES_DOCUMENT_SENTENCE_TEXTS[paragraph_position])
                for paragraph_position in questions_on_sentences
            )
            return extracted, questions_on_sentences

        # The sentences that the aspects with a threshold of their own get,
        # whatever the threshold of the decision model is
        lenient_sentences = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1)]
        # The sentences with a probability above the threshold in the paragraphs
        # that are below it are not asked about, i.e. are not extracted
        strict_sentences = [(1, 0)]
        assert probabilities[(2, 0)] > 0.7 > probabilities[2]

        # === The default threshold of the decision model ===
        extracted, questions_on_sentences = extract()
        assert extracted == {
            "Default": [(1, 0), (1, 1), (2, 0)],
            "Lenient": lenient_sentences,
            "Strict": strict_sentences,
        }
        # The sentences of a paragraph are asked about for the aspects that
        # the paragraph belongs to, by the threshold of each aspect
        assert questions_on_sentences == {
            1: ["A0", "A1", "A2"],
            2: ["A0", "A1"],
            3: ["A1"],
        }

        # === A higher threshold of the decision model ===
        # It applies to the aspect without a threshold of its own, at both stages
        extracted, questions_on_sentences = extract(decision_threshold=0.8)
        assert extracted == {
            "Default": [(1, 0)],
            "Lenient": lenient_sentences,
            "Strict": strict_sentences,
        }
        assert questions_on_sentences == {1: ["A0", "A1", "A2"], 2: ["A1"], 3: ["A1"]}

        # === A lower threshold of the decision model ===
        extracted, questions_on_sentences = extract(decision_threshold=0.35)
        assert extracted == {
            "Default": [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0)],
            "Lenient": lenient_sentences,
            "Strict": strict_sentences,
        }
        assert questions_on_sentences == {
            1: ["A0", "A1", "A2"],
            2: ["A0", "A1"],
            3: ["A0", "A1"],
        }

    def test_sentence_depth_sub_aspects_and_concepts_with_decision_model(self):
        """
        Tests that the sub-aspects and the concepts of an aspect that a decision
        model extracts with sentences as the extracted items are extracted from
        the paragraphs of the aspect, and that a sub-aspect is extracted with
        sentences as its items, from the paragraphs of an aspect of any reference
        depth.
        """
        yes_probabilities: dict[str, dict[int | tuple[int, int], float]] = {
            "Termination": {
                1: 0.9,
                3: 0.6,
                (1, 0): 0.9,
                (1, 1): 0.8,
                (3, 1): 0.7,
            },
            "Notice of termination": {1: 0.8, (1, 0): 0.3, (1, 1): 0.9},
            "Effects of termination": {3: 0.7},
            "Payment": {2: 0.8},
            "Late payment": {2: 0.9, (2, 1): 0.85},
        }

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request with the questions of aspects, asked on one
            paragraph or on one sentence, or with the question of a concept.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            if request["questions"][0]["name"].startswith("C"):
                return _answer_all_questions(request)
            return _create_sentences_responder(yes_probabilities)(request)

        def create_document() -> Document:
            """
            Creates a document with an aspect with sentences as its items, which
            has a sub-aspect of each reference depth and a concept, and an aspect
            with paragraphs as its items, which has a sub-aspect with sentences
            as its items.

            :return: Document.
            :rtype: Document
            """
            return _create_sentences_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        reference_depth="sentences",
                        aspects=[
                            Aspect(
                                name="Notice of termination",
                                description="Clauses on the notice that terminates "
                                "the agreement",
                                reference_depth="sentences",
                            ),
                            Aspect(
                                name="Effects of termination",
                                description="Clauses on what follows the termination",
                            ),
                        ],
                        concepts=[
                            BooleanConcept(
                                name="Written notice",
                                description="A written notice is required",
                            )
                        ],
                    ),
                    Aspect(
                        name="Payment",
                        description="Clauses on the payment of fees",
                        aspects=[
                            Aspect(
                                name="Late payment",
                                description="Clauses on the payments that are late",
                                reference_depth="sentences",
                            )
                        ],
                    ),
                ]
            )

        decision_model = _create_decision_model()
        document = create_document()
        paragraphs = document.paragraphs
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)

        def get_instance_name(question: dict[str, Any]) -> str:
            """
            Returns the name of the aspect that the question is about.

            :param question: The question of a request.
            :type question: dict[str, Any]
            :return: The name of the aspect.
            :rtype: str
            """
            name_match = re.search(
                r"^Aspect name: (.+)$", question["instructions"], flags=re.MULTILINE
            )
            assert name_match is not None, question
            return name_match.group(1)

        paragraph_texts = _SENTENCES_PARAGRAPH_TEXTS_SENT
        sentence_texts = _SENTENCES_TEXTS_SENT
        assert [
            (
                request["input"],
                [get_instance_name(question) for question in request["questions"]],
            )
            for request in decisions_stub.requests
        ] == [
            # The aspect with sentences as its items: each paragraph of
            # the document, and each sentence of the paragraphs that belong to it
            *[(text, ["Termination"]) for text in paragraph_texts],
            *[(sentence_texts[(1, i)], ["Termination"]) for i in range(3)],
            *[(sentence_texts[(3, i)], ["Termination"]) for i in range(2)],
            # The aspect with paragraphs as its items, in a call of its own
            *[(text, ["Payment"]) for text in paragraph_texts],
            # The sub-aspect with sentences as its items: each paragraph extracted
            # for its aspect, and each sentence of the paragraph that belongs to it
            (paragraph_texts[1], ["Notice of termination"]),
            (paragraph_texts[3], ["Notice of termination"]),
            *[(sentence_texts[(1, i)], ["Notice of termination"]) for i in range(3)],
            # The sub-aspect with paragraphs as its items
            (paragraph_texts[1], ["Effects of termination"]),
            (paragraph_texts[3], ["Effects of termination"]),
            # The sub-aspect of the aspect with paragraphs as its items
            (paragraph_texts[2], ["Late payment"]),
            *[(sentence_texts[(2, i)], ["Late payment"]) for i in range(2)],
        ]

        termination, payment = document.aspects
        notice, effects = termination.aspects
        (late_payment,) = payment.aspects
        # What each aspect and sub-aspect gets: the sentences, by the positions
        # of the paragraph and of the sentence in it, or the paragraphs, with
        # the probabilities
        expected_items: list[tuple[_Aspect, list[tuple[Any, float]]]] = [
            (termination, [((1, 0), 0.9), ((1, 1), 0.8), ((3, 1), 0.7)]),
            (notice, [((1, 1), 0.9)]),
            (effects, [(3, 0.7)]),
            (payment, [(2, 0.8)]),
            (late_payment, [((2, 1), 0.85)]),
        ]
        for aspect, aspect_items in expected_items:
            assert aspect._is_processed
            assert len(aspect.extracted_items) == len(aspect_items), aspect.name
            reference_paragraphs: list[_Paragraph] = []
            reference_sentences: list[_Sentence] = []
            for item, (position, probability) in zip(
                aspect.extracted_items, aspect_items, strict=True
            ):
                self.check_item_probabilities(
                    item, {True: probability, False: 1 - probability}
                )
                # The references are the paragraph and the sentence of the document
                # itself, i.e. the same objects, for an aspect of the document and
                # for a sub-aspect, as when an LLM extracts it
                if isinstance(position, tuple):
                    paragraph = paragraphs[position[0]]
                    sentence = paragraph.sentences[position[1]]
                    assert item.value == sentence.raw_text
                    assert item.reference_sentences == [sentence]
                    assert item.reference_sentences[0] is sentence
                    reference_sentences.append(sentence)
                else:
                    paragraph = paragraphs[position]
                    assert item.value == paragraph.raw_text
                    assert not item.reference_sentences
                assert item.reference_paragraphs == [paragraph]
                assert item.reference_paragraphs[0] is paragraph
                if paragraph not in reference_paragraphs:
                    reference_paragraphs.append(paragraph)
            assert aspect.reference_paragraphs == reference_paragraphs
            # The reference sentences of the items, i.e. the extracted sentences
            # and not the other sentences of the reference paragraphs: an aspect
            # with paragraphs as its items has none
            assert aspect.reference_sentences == reference_sentences
            assert bool(reference_sentences) is (aspect.reference_depth == "sentences")
        self.check_instance_serialization_and_cloning(document)

        # === The concept of the aspect ===
        # It is decided on the text of the paragraphs of the aspect, as for
        # an aspect with paragraphs as its items
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_aspect(termination, document)
        (request,) = decisions_stub.requests
        assert request["input"] == (
            f"{' '.join(_SENTENCES_DOCUMENT_SENTENCE_TEXTS[1])}\n\n"
            f"{' '.join(_SENTENCES_DOCUMENT_SENTENCE_TEXTS[3])}"
        )
        assert [question["name"] for question in request["questions"]] == ["C0"]
        (concept,) = termination.concepts
        assert concept.extracted_items[0].value is True
        self.check_item_probabilities(
            concept.extracted_items[0], {True: 0.9, False: 1 - 0.9}
        )

        # === Everything that the document has ===
        # The aspects, sub-aspects and concepts are extracted in the same way
        other_document = create_document()
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            _create_decision_model().extract_all(other_document)
        assert len(decisions_stub.requests) == 10 + 5 + 5 + 2 + 3 + 1

        def get_values(document: Document) -> dict[str, list[Any]]:
            """
            Returns the values of the items of each aspect, sub-aspect and
            concept of the document.

            :param document: The document.
            :type document: Document
            :return: The values of the items, by the names of the instances.
            :rtype: dict[str, list[Any]]
            """
            values: dict[str, list[Any]] = {}
            for aspect in document.aspects:
                for instance in [aspect, *aspect.aspects, *aspect.concepts]:
                    assert instance._is_processed
                    values[instance.name] = [
                        item.value for item in instance.extracted_items
                    ]
            return values

        assert get_values(other_document) == get_values(document)
        assert len(get_values(document)) == 6

    def test_sentence_depth_aspects_in_mixed_calls_and_segmentation(self):
        """
        Tests that it is the reference depth of each aspect that decides how
        a decision model extracts the aspect, also when the aspects of one call
        have different ones, that a paragraph without sentences gives no items of
        an aspect with sentences as its items, and that the sentences are
        segmented before such an aspect is extracted.
        """
        yes_probabilities: dict[str, dict[int | tuple[int, int], float]] = {
            "Payment": {1: 0.6, 2: 0.8},
            "Termination": {1: 0.9, 3: 0.6, (1, 0): 0.9, (3, 1): 0.7},
        }
        respond = _create_sentences_responder(yes_probabilities)

        def create_document() -> Document:
            """
            Creates a document with an aspect with paragraphs as its items,
            followed by an aspect with sentences as its items.

            :return: Document.
            :rtype: Document
            """
            return _create_sentences_document(
                aspects=[
                    Aspect(
                        name="Payment", description="Clauses on the payment of fees"
                    ),
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        reference_depth="sentences",
                    ),
                ]
            )

        def requests_sent(stub: _DecisionsAPIStub) -> list[tuple[str, list[str]]]:
            """
            Returns the text and the names of the questions of each request sent.

            :param stub: The stub with the requests received.
            :type stub: _DecisionsAPIStub
            :return: The requests, in the order they were sent.
            :rtype: list[tuple[str, list[str]]]
            """
            return [
                (
                    request["input"],
                    [question["name"] for question in request["questions"]],
                )
                for request in stub.requests
            ]

        def check_items(document: Document) -> None:
            """
            Checks the items of the two aspects of the document: the paragraphs
            of one, and the sentences of the other.

            :param document: The document with the extracted aspects.
            :type document: Document
            """
            paragraphs = document.paragraphs
            payment, termination = document.aspects
            assert payment._is_processed and termination._is_processed
            assert [
                (item.value, item.reference_paragraphs, item.reference_sentences)
                for item in payment.extracted_items
            ] == [
                (paragraphs[1].raw_text, [paragraphs[1]], []),
                (paragraphs[2].raw_text, [paragraphs[2]], []),
            ]
            self.check_item_probabilities(
                payment.extracted_items[0], {True: 0.6, False: 1 - 0.6}
            )
            assert payment.reference_paragraphs == [paragraphs[1], paragraphs[2]]
            assert not payment.reference_sentences
            assert [
                (item.value, item.reference_paragraphs, item.reference_sentences)
                for item in termination.extracted_items
            ] == [
                (
                    paragraphs[1].sentences[0].raw_text,
                    [paragraphs[1]],
                    [paragraphs[1].sentences[0]],
                ),
                (
                    paragraphs[3].sentences[1].raw_text,
                    [paragraphs[3]],
                    [paragraphs[3].sentences[1]],
                ),
            ]
            self.check_item_probabilities(
                termination.extracted_items[1], {True: 0.7, False: 1 - 0.7}
            )
            assert termination.reference_paragraphs == [paragraphs[1], paragraphs[3]]
            # The two extracted sentences, and not the five sentences of
            # the two reference paragraphs
            assert termination.reference_sentences == [
                paragraphs[1].sentences[0],
                paragraphs[3].sentences[1],
            ]

        # === Aspects of both reference depths in one call ===
        # Each paragraph is asked about once, for both aspects, and the sentences
        # for the aspect with sentences as its items only, whose question keeps
        # the name that tells the position of the aspect in the call
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            extracted_aspects, usage = asyncio.run(
                decision_model._extract_items_from_instances(
                    extracted_item_type="aspect",
                    source=document,
                    llm=decision_model,
                    instances_to_process=list(document.aspects),
                    document=document,
                )
            )
        assert extracted_aspects == list(document.aspects)
        assert requests_sent(decisions_stub) == [
            *[(text, ["A0", "A1"]) for text in _SENTENCES_PARAGRAPH_TEXTS_SENT],
            *[(_SENTENCES_TEXTS_SENT[(1, i)], ["A1"]) for i in range(3)],
            *[(_SENTENCES_TEXTS_SENT[(3, i)], ["A1"]) for i in range(2)],
        ]
        assert all(
            "Aspect name: Termination\n" in request["questions"][0]["instructions"]
            for request in decisions_stub.requests[5:]
        )
        assert len(usage.calls) == 10
        check_items(document)

        # === The same aspects, through an extraction method ===
        # The aspects are split into the calls by the reference depth, as for
        # an LLM: each call has its own requests for the paragraphs
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert requests_sent(decisions_stub) == [
            *[(text, ["A0"]) for text in _SENTENCES_PARAGRAPH_TEXTS_SENT],
            *[(text, ["A0"]) for text in _SENTENCES_PARAGRAPH_TEXTS_SENT],
            *[(_SENTENCES_TEXTS_SENT[(1, i)], ["A0"]) for i in range(3)],
            *[(_SENTENCES_TEXTS_SENT[(3, i)], ["A0"]) for i in range(2)],
        ]
        check_items(document)

        # === Paragraphs without sentences ===
        # A paragraph that belongs to the aspect gives no items if it has no
        # sentences, which is possible only if the sentences are not segmented
        # before the extraction, as the extraction methods do
        decision_model = _create_decision_model()
        document = _create_aspects_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                    reference_depth="sentences",
                )
            ]
        )
        assert not document.sentences
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [_create_aspects_responder({"Termination": {2: 0.9}})]}
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            asyncio.run(
                decision_model._extract_items_from_instances(
                    extracted_item_type="aspect",
                    source=document,
                    llm=decision_model,
                    instances_to_process=list(document.aspects),
                    document=document,
                )
            )
        assert [request["input"] for request in decisions_stub.requests] == (
            _ASPECTS_PARAGRAPH_TEXTS_SENT
        )
        (aspect,) = document.aspects
        assert aspect._is_processed
        assert not aspect.extracted_items
        assert not aspect.reference_paragraphs
        assert not aspect.reference_sentences

        # === Sentences segmented before the extraction ===
        # An extraction method segments the sentences of a document that has
        # an aspect with sentences as its items, as it does for an LLM. A sentence
        # segmented from a paragraph has the additional context and the custom
        # data of the paragraph.
        segmented_texts: list[list[str]] = []

        class SentenceSplitter:
            """
            Splits the paragraphs into sentences at the full stops, in place of
            the model that segments the sentences.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, e.g. the batch
                    size, which the stub does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                segmented_texts.append(sorted(texts))
                return [re.split(r"(?<=\.) ", text) for text in texts]

        paragraph_texts = [
            "Services agreement",
            "Either party may terminate this agreement. The notice shall be written.",
        ]
        document = Document(
            paragraphs=[
                Paragraph(raw_text=paragraph_texts[0]),
                Paragraph(
                    raw_text=paragraph_texts[1],
                    additional_context="Section 3. Termination",
                    custom_data={"page": 2},
                ),
            ]
        )
        document.aspects = [
            Aspect(
                name="Termination",
                description="Clauses on the termination of the agreement",
                reference_depth="sentences",
            )
        ]
        assert not document.sentences

        def respond_on_segmented(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request for a paragraph, or for a sentence, of
            the document without sentences: the second paragraph and its second
            sentence belong to the aspect.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            # The sentences are segmented before the first request is sent
            assert segmented_texts == [sorted(paragraph_texts)]
            text = request["input"].split("\n\n")[0]
            belongs = text in [
                f"Paragraph: {paragraph_texts[1]}",
                "Sentence: The notice shall be written.",
            ]
            return _create_decisions_response(
                request["questions"], {"A0": 0.8 if belongs else 0.1}
            )

        decision_model = _create_decision_model()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_on_segmented]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                documents_module,
                "_get_sat_model",
                new=lambda model_id: SentenceSplitter(),
            ),
        ):
            decision_model.extract_aspects_from_document(document)
        assert segmented_texts == [sorted(paragraph_texts)]
        paragraph = document.paragraphs[1]
        assert [
            (sentence.raw_text, sentence.additional_context, sentence.custom_data)
            for sentence in paragraph.sentences
        ] == [
            (
                "Either party may terminate this agreement.",
                "Section 3. Termination",
                {"page": 2},
            ),
            ("The notice shall be written.", "Section 3. Termination", {"page": 2}),
        ]
        # The additional context that the sentences have from their paragraph is
        # sent once with each of them, as that of the paragraph
        context_sent = "Additional context of the paragraph: Section 3. Termination"
        assert [request["input"] for request in decisions_stub.requests] == [
            paragraph_texts[0],
            f"Paragraph: {paragraph_texts[1]}\n\n{context_sent}",
            (
                "Sentence: Either party may terminate this agreement.\n\n"
                f"Paragraph that the sentence is from: {paragraph_texts[1]}\n\n"
                f"{context_sent}"
            ),
            (
                "Sentence: The notice shall be written.\n\n"
                f"Paragraph that the sentence is from: {paragraph_texts[1]}\n\n"
                f"{context_sent}"
            ),
        ]
        (aspect,) = document.aspects
        (item,) = aspect.extracted_items
        assert item.value == "The notice shall be written."
        assert item.custom_data == {"page": 2}
        assert item.reference_paragraphs == [paragraph]
        assert item.reference_sentences == [paragraph.sentences[1]]
        # The reference sentence of the aspect is that of its item: the first
        # sentence of the paragraph is not extracted
        assert aspect.reference_sentences == [paragraph.sentences[1]]

    def test_decision_model_sentence_requests_concurrency(self):
        """
        Tests that the requests for the sentences, which the aspects with sentences
        as the extracted items are extracted with by a decision model, are sent
        after the requests for the paragraphs: one after another by default, and
        concurrently, each paced by the limiter of the model, with concurrency
        enabled, with the same items extracted.
        """
        yes_probabilities: dict[str, dict[int | tuple[int, int], float]] = {
            "Termination": {1: 0.9, 3: 0.6, (1, 0): 0.9, (1, 1): 0.8, (3, 1): 0.7},
            "Payment": {1: 0.7, 2: 0.8, (1, 2): 0.6, (2, 0): 0.9},
        }
        respond = _create_sentences_responder(yes_probabilities)
        # The requests for the paragraphs, and those for the sentences of
        # the paragraphs that belong to an aspect
        n_requests = {"paragraph": 5, "sentence": 3 + 2 + 2}
        requests_in_progress: dict[str, int] = {}
        max_requests_in_progress: dict[str, int] = {}
        completed: list[int | tuple[int, int]] = []
        # The number of the requests for the paragraphs that are completed when
        # each request for a sentence is sent
        paragraphs_completed_before_sentences: list[int] = []
        acquisitions: list[tuple[float, float]] = []
        # Set when all the requests of a stage are in progress at the same time
        all_sent: dict[str, asyncio.Event] = {}
        concurrency_expected: list[bool] = [False]

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the config of the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                acquisitions.append((self.max_rate, self.time_period))
                await super().acquire(amount)

        async def respond_in_turn(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to the request, and records how many requests of its stage
            are in progress at the same time. If the requests are expected to be
            sent concurrently, the response comes when all the requests of
            the stage are in progress.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            position = _get_sentences_request_position(request)
            stage = "paragraph" if isinstance(position, int) else "sentence"
            if stage == "sentence":
                paragraphs_completed_before_sentences.append(
                    len([i for i in completed if isinstance(i, int)])
                )
            requests_in_progress[stage] = requests_in_progress.get(stage, 0) + 1
            max_requests_in_progress[stage] = max(
                max_requests_in_progress.get(stage, 0), requests_in_progress[stage]
            )
            if concurrency_expected[0]:
                if requests_in_progress[stage] == n_requests[stage]:
                    all_sent[stage].set()
                # Fails if the requests are not sent concurrently, so that
                # the test does not hang
                await asyncio.wait_for(all_sent[stage].wait(), timeout=10)
            else:
                # Let any other request be sent, if the requests are sent
                # concurrently
                for _ in range(3):
                    await asyncio.sleep(0)
            requests_in_progress[stage] -= 1
            completed.append(position)
            return respond(request)

        def extract(
            **extraction_kwargs: Any,
        ) -> tuple[list[list[tuple[str, Any]]], _DecisionsAPIStub]:
            """
            Extracts the aspects of a new document with a new decision model.

            :param extraction_kwargs: Params of the extraction method.
            :type extraction_kwargs: Any
            :return: The values and the probabilities of the items of each
                aspect, and the stub with the requests received.
            :rtype: tuple[list[list[tuple[str, Any]]], _DecisionsAPIStub]
            """
            decision_model = _create_decision_model(async_limiter=AsyncLimiter(20, 5))
            document = _create_sentences_document(
                aspects=[
                    Aspect(
                        name=name,
                        description=f"Clauses on {name.lower()}",
                        reference_depth="sentences",
                    )
                    for name in yes_probabilities
                ]
            )
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_in_turn]})
            requests_in_progress.clear()
            max_requests_in_progress.clear()
            completed.clear()
            paragraphs_completed_before_sentences.clear()
            acquisitions.clear()
            concurrency_expected[0] = bool(extraction_kwargs.get("use_concurrency"))
            # The events are used in the event loop of this extraction only
            all_sent.clear()
            for stage in n_requests:
                all_sent[stage] = asyncio.Event()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
            ):
                decision_model.extract_aspects_from_document(
                    document, **extraction_kwargs
                )
            assert all(aspect._is_processed for aspect in document.aspects)
            # The requests for the sentences are sent when the requests for all
            # the paragraphs are completed, as they depend on the answers on them
            assert paragraphs_completed_before_sentences == (
                [n_requests["paragraph"]] * n_requests["sentence"]
            )
            # Each request is recorded as a call: those for the paragraphs, and
            # then those for the sentences, in the order of the document, whatever
            # the order in which the requests are completed
            usage = decision_model.get_usage()[0].usage
            assert [json.loads(call.prompt)["input"] for call in usage.calls] == [
                *_SENTENCES_PARAGRAPH_TEXTS_SENT,
                *[_SENTENCES_TEXTS_SENT[(1, i)] for i in range(3)],
                *[_SENTENCES_TEXTS_SENT[(2, i)] for i in range(2)],
                *[_SENTENCES_TEXTS_SENT[(3, i)] for i in range(2)],
            ]
            return (
                [
                    [
                        (item.value, item.probabilities)
                        for item in aspect.extracted_items
                    ]
                    for aspect in document.aspects
                ],
                decisions_stub,
            )

        # === Without concurrency ===
        # The requests are sent one after another, with no limiter
        expected_items, decisions_stub = extract()
        assert len(decisions_stub.requests) == 12
        assert max_requests_in_progress == {"paragraph": 1, "sentence": 1}
        assert acquisitions == []
        sentence_texts = _SENTENCES_DOCUMENT_SENTENCE_TEXTS
        assert [[value for value, _ in items] for items in expected_items] == [
            [sentence_texts[1][0], sentence_texts[1][1], sentence_texts[3][1]],
            [sentence_texts[1][2], sentence_texts[2][0]],
        ]

        # === With concurrency ===
        # The requests for the sentences are sent concurrently, as those for
        # the paragraphs are, each paced by the limiter of the model. The items
        # are in the order of the document, whatever the order in which
        # the requests are completed.
        items, decisions_stub = extract(use_concurrency=True, max_items_per_call=2)
        assert len(decisions_stub.requests) == 12
        assert max_requests_in_progress == n_requests
        assert acquisitions == [(20, 5)] * 12
        assert items == expected_items

    def test_decision_model_sentence_request_failures(self):
        """
        Tests the extraction of aspects with sentences as the extracted items with
        a decision model when the request for one of the sentences fails for good,
        or gets a response that cannot be used, with and without concurrency:
        no further requests are sent, the requests completed are counted in
        the usage, a failed request is recorded as the last call, which is
        unfinished, and a fallback model takes over, after a response that cannot
        be used is asked for again with the request that got it only. If a request
        for a paragraph fails, no request for a sentence is sent.
        """
        auth_error = _create_api_error(litellm.AuthenticationError)
        rate_limit_error = _create_api_error(litellm.RateLimitError)
        yes_probabilities: dict[str, dict[int | tuple[int, int], float]] = {
            "Termination": {1: 0.9, (1, 0): 0.9},
            "Payment": {1: 0.7, 3: 0.8, (1, 2): 0.8, (3, 1): 0.6},
        }
        respond = _create_sentences_responder(yes_probabilities, input_tokens=20)
        # The requests of an extraction in which each request gets a response:
        # for the paragraphs, for the sentences of the second paragraph, which
        # belongs to both aspects, and for those of the fourth one, which belongs
        # to the second aspect only
        all_positions: list[int | tuple[int, int]] = [
            0,
            1,
            2,
            3,
            4,
            (1, 0),
            (1, 1),
            (1, 2),
            (3, 0),
            (3, 1),
        ]
        failed_position = (1, 1)
        n_requests_until_failed = all_positions.index(failed_position) + 1

        def get_text_sent(position: int | tuple[int, int]) -> str:
            """
            Returns the text of the request for a paragraph or for a sentence.

            :param position: The position of the paragraph, or the positions of
                the paragraph and of the sentence in it.
            :type position: int | tuple[int, int]
            :return: The text of the request.
            :rtype: str
            """
            if isinstance(position, int):
                return _SENTENCES_PARAGRAPH_TEXTS_SENT[position]
            return _SENTENCES_TEXTS_SENT[position]

        def create_document() -> Document:
            """
            Creates a document with two aspects that are extracted with
            sentences as their items.

            :return: Document.
            :rtype: Document
            """
            return _create_sentences_document(
                aspects=[
                    Aspect(
                        name=name,
                        description=f"Clauses on {name.lower()}",
                        reference_depth="sentences",
                    )
                    for name in yes_probabilities
                ]
            )

        def create_responder(
            outcomes: dict[int | tuple[int, int], Any],
        ) -> Callable[[dict[str, Any]], Any]:
            """
            Creates the function that responds to the request for a paragraph or
            for a sentence.

            :param outcomes: By the position of a paragraph, or by the positions
                of a paragraph and of a sentence in it: an error to raise for its
                request, or the answers to respond with instead of the valid ones.
            :type outcomes: dict[int | tuple[int, int], Any]
            :return: Function that responds to a request.
            :rtype: Callable[[dict[str, Any]], Any]
            """

            def respond_to_request(request: dict[str, Any]) -> Any:
                """
                Responds to the request for a paragraph or for a sentence, or
                fails it.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response of litellm.
                :rtype: Any
                """
                outcome = outcomes.get(_get_sentences_request_position(request))
                if isinstance(outcome, Exception):
                    raise outcome
                if outcome is not None:
                    return _create_decisions_response(
                        request["questions"], outcome, input_tokens=20
                    )
                return respond(request)

            return respond_to_request

        def check_usage(
            model: DocumentDecisionModel,
            finished_positions: list[int | tuple[int, int]],
            failed_positions: list[int | tuple[int, int]],
            usage_position: int = 0,
        ) -> None:
            """
            Checks the calls recorded in the usage of the decision model, or of
            its fallback model: first the requests that got a response, then
            the requests that failed.

            :param model: The decision model.
            :type model: DocumentDecisionModel
            :param finished_positions: What the requests that got a response
                were for, in the order of the calls.
            :type finished_positions: list[int | tuple[int, int]]
            :param failed_positions: What the requests that failed were for, in
                the order of the calls.
            :type failed_positions: list[int | tuple[int, int]]
            :param usage_position: The position of the model among the model and
                its fallback models. Defaults to 0, i.e. the model itself.
            :type usage_position: int
            """
            usage = model.get_usage()[usage_position].usage
            assert [json.loads(call.prompt)["input"] for call in usage.calls] == [
                get_text_sent(position)
                for position in [*finished_positions, *failed_positions]
            ]
            assert [call.timestamp_received is not None for call in usage.calls] == (
                [True] * len(finished_positions) + [False] * len(failed_positions)
            )
            assert usage.input == 20 * len(finished_positions)

        def check_not_extracted(document: Document) -> None:
            """
            Checks that no aspect of the document is extracted.

            :param document: The document.
            :type document: Document
            """
            assert not any(aspect._is_processed for aspect in document.aspects)
            assert not any(aspect.extracted_items for aspect in document.aspects)
            assert not any(aspect.reference_paragraphs for aspect in document.aspects)
            assert not any(aspect.reference_sentences for aspect in document.aspects)

        def check_extracted(document: Document) -> None:
            """
            Checks that the aspects of the document are extracted, with
            the sentences that belong to them as their items.

            :param document: The document.
            :type document: Document
            """
            sentence_texts = _SENTENCES_DOCUMENT_SENTENCE_TEXTS
            assert [
                [(item.value, item.probabilities) for item in aspect.extracted_items]
                for aspect in document.aspects
            ] == [
                [(sentence_texts[1][0], {True: 0.9, False: 1 - 0.9})],
                [
                    (sentence_texts[1][2], {True: 0.8, False: 1 - 0.8}),
                    (sentence_texts[3][1], {True: 0.6, False: 1 - 0.6}),
                ],
            ]

        finished_before_failed = all_positions[: n_requests_until_failed - 1]
        # With immediate responses, the requests sent concurrently are completed
        # in the order they are sent, i.e. the outcome is the same in both modes:
        # the requests for the sentences after the failed one are not sent
        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            # === Each request gets a response ===
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert [
                (
                    request["input"],
                    [question["name"] for question in request["questions"]],
                )
                for request in decisions_stub.requests
            ] == [
                *[(get_text_sent(position), ["A0", "A1"]) for position in range(5)],
                *[(get_text_sent((1, i)), ["A0", "A1"]) for i in range(3)],
                *[(get_text_sent((3, i)), ["A1"]) for i in range(2)],
            ]
            check_usage(decision_model, all_positions, [])
            check_extracted(document)

            # === The request for a sentence fails ===
            # No fallback model: a model API error is raised
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder({failed_position: auth_error})]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(ModelAPIError) as exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert exc_info.value.original_error is auth_error
            assert exc_info.value.retry_count == 0
            assert [request["input"] for request in decisions_stub.requests] == [
                get_text_sent(position)
                for position in all_positions[:n_requests_until_failed]
            ]
            # The requests for the paragraphs and for the sentences before
            # the failed one are counted, and the failed one is the last call
            check_usage(decision_model, finished_before_failed, [failed_position])
            check_not_extracted(document)

            # ... or a warning is logged, if no exception is to be raised
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub.requests.clear()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.WARNING) as captured_logs,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    raise_exception_on_extraction_error=False,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            # The warning tells the retries, unlike the errors logged before it
            (warning_message,) = [log for log in captured_logs if "retries)" in log]
            assert "calling decision model API" in warning_message
            assert "(0 retries)" in warning_message
            assert len(decisions_stub.requests) == n_requests_until_failed
            check_usage(decision_model, finished_before_failed, [failed_position])
            check_not_extracted(document)

            # With a fallback model, the aspects are extracted by it: it asks
            # about the paragraphs and the sentences itself
            decision_model = _create_decision_model(
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ]
            )
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [create_responder({failed_position: auth_error})],
                    _OTHER_DECISION_MODEL: [respond],
                }
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert decisions_stub.requested_models() == (
                [_DECISION_MODEL] * n_requests_until_failed
                + [_OTHER_DECISION_MODEL] * len(all_positions)
            )
            check_usage(decision_model, finished_before_failed, [failed_position])
            check_usage(decision_model, all_positions, [], usage_position=1)
            check_extracted(document)

            # === A response that cannot be used ===
            # The response to the request for one sentence makes the whole call
            # invalid: no further requests are sent, and each call is finished.
            # One call, without retries for invalid data.
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        create_responder({failed_position: {"A0": 0.9, "A1": 1.5}})
                    ]
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.ERROR) as captured_logs,
                pytest.raises(
                    ModelExtractionError,
                    match="Some aspects could not be processed due to "
                    "an invalid response returned by decision model. "
                    r"Failed aspects: \['Termination', 'Payment'\]",
                ) as extraction_exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert extraction_exc_info.value.retry_count == 0
            # The log names the aspect that the answer is for, and the sentence
            # that the question was asked on, by its position in its paragraph,
            # with the paragraph, by its position in the document
            assert (
                f"Decision model `{_DECISION_MODEL}` returned an invalid response "
                "for aspect `Payment` on sentence 2 of 3 of paragraph 2 of 5 "
                "(index 1), decision question `A1`: the probability of the answer is not a number within [0, 1]."
            ) in captured_logs, captured_logs
            assert len(decisions_stub.requests) == n_requests_until_failed
            check_usage(decision_model, all_positions[:n_requests_until_failed], [])
            check_not_extracted(document)

        # The question on a sentence is named by the position of its aspect in
        # the call, also when the sentence is not asked about for the aspects
        # before it: the log names the aspect that the answer is for. One call,
        # without retries for invalid data.
        decision_model = _create_decision_model(max_retries_invalid_data=0)
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [create_responder({(3, 0): {"A1": 1.5}})]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            capture_logs(stdlib_logging.ERROR) as captured_logs,
            pytest.raises(ModelExtractionError),
        ):
            decision_model.extract_aspects_from_document(document)
        assert (
            f"Decision model `{_DECISION_MODEL}` returned an invalid response "
            "for aspect `Payment` on sentence 1 of 2 of paragraph 4 of 5 "
            "(index 3), decision question `A1`: the probability of the answer is not a number within [0, 1]."
        ) in captured_logs, captured_logs
        assert len(decisions_stub.requests) == 9
        check_not_extracted(document)

        # A fallback model takes over after a response that cannot be used as well,
        # once the decision model is asked again up to its
        # `max_retries_invalid_data` times (3 by default): with the request for
        # that sentence only, as the answers on the paragraphs and on
        # the sentences before it are kept
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
            ]
        )
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [create_responder({(3, 0): {"A1": 1.5}})],
                _OTHER_DECISION_MODEL: [respond],
            }
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert decisions_stub.requested_models() == (
            [_DECISION_MODEL] * (9 + 3) + [_OTHER_DECISION_MODEL] * len(all_positions)
        )
        assert [request["input"] for request in decisions_stub.requests] == [
            *[get_text_sent(position) for position in all_positions[:9]],
            *[get_text_sent((3, 0))] * 3,
            *[get_text_sent(position) for position in all_positions],
        ]
        check_usage(decision_model, [*all_positions[:9], *[(3, 0)] * 3], [])
        check_extracted(document)

        # === A transient error ===
        # The request for a sentence is retried before it fails for good
        decision_model = _create_decision_model(num_retries_failed_request=2)
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [create_responder({failed_position: rate_limit_error})]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                decision_models_module,
                "_get_model_api_retry_delay",
                new=lambda retry_number: 0.0,
            ),
            pytest.raises(ModelAPIError) as exc_info,
        ):
            decision_model.extract_aspects_from_document(document)
        assert exc_info.value.original_error is rate_limit_error
        assert exc_info.value.retry_count == 2
        assert [request["input"] for request in decisions_stub.requests] == [
            *[get_text_sent(position) for position in finished_before_failed],
            *[get_text_sent(failed_position)] * 3,
        ]
        check_usage(decision_model, finished_before_failed, [failed_position])
        check_not_extracted(document)

        # === The request for a paragraph fails ===
        # No request for a sentence is sent: which paragraphs belong to
        # the aspects is not known
        for outcome, error_class in [
            (auth_error, ModelAPIError),
            ({"A0": 1.5, "A1": 0.9}, ModelExtractionError),
        ]:
            # One call, without retries for invalid data
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder({1: outcome})]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(error_class),
            ):
                decision_model.extract_aspects_from_document(document)
            assert [request["input"] for request in decisions_stub.requests] == [
                get_text_sent(0),
                get_text_sent(1),
            ]
            check_not_extracted(document)

    def test_decision_model_accepts_sentence_depth_aspects(self):
        """
        Tests that an aspect with sentences as the extracted items is accepted by
        the checks made before any request is sent, and is extracted, wherever
        a decision model is used: on its own, in a fallback chain with a model of
        either kind, and in a model group, while references for concepts are
        still rejected, whatever their depth is.
        """
        yes_probabilities: dict[str, dict[int | tuple[int, int], float]] = {
            "Termination": {1: 0.9, (1, 0): 0.8},
            "Late payment": {2: 0.9, (2, 1): 0.7},
        }
        respond = _create_sentences_responder(yes_probabilities)
        sentence_texts = _SENTENCES_DOCUMENT_SENTENCE_TEXTS
        auth_error = _create_api_error(litellm.AuthenticationError)
        llm_auth_error = _create_api_error(litellm.AuthenticationError, _LLM_MODEL)
        # The answer of an LLM for an aspect with sentences as its items
        llm_answer = json.dumps(
            [
                {
                    "aspect_id": "A0",
                    "paragraphs": [
                        {
                            "paragraph_id": "P1",
                            "sentences": [{"sentence_id": "P1-S0"}],
                        }
                    ],
                }
            ]
        )

        def create_aspect(name: str = "Termination", **kwargs: Any) -> Aspect:
            """
            Creates an aspect that is extracted with sentences as its items.

            :param name: Name of the aspect. Defaults to "Termination".
            :type name: str
            :param kwargs: Other params of the aspect.
            :type kwargs: Any
            :return: Aspect.
            :rtype: Aspect
            """
            return Aspect(
                name=name,
                description=f"Clauses on the {name.lower()} of the agreement",
                reference_depth="sentences",
                **kwargs,
            )

        def check_extracted(document: Document, with_probabilities: bool) -> None:
            """
            Checks that the first aspect of the document is extracted, with
            the first sentence of the second paragraph as its item.

            :param document: The document.
            :type document: Document
            :param with_probabilities: Whether the item is to have
                the probabilities, as an item extracted by a decision model does.
            :type with_probabilities: bool
            """
            aspect = document.aspects[0]
            paragraph = document.paragraphs[1]
            assert aspect._is_processed
            (item,) = aspect.extracted_items
            assert item.value == sentence_texts[1][0]
            assert item.reference_paragraphs == [paragraph]
            assert item.reference_sentences == [paragraph.sentences[0]]
            self.check_item_probabilities(
                item, {True: 0.8, False: 1 - 0.8} if with_probabilities else None
            )
            assert aspect.reference_paragraphs == [paragraph]
            # The extracted sentence only, of the three sentences of the paragraph,
            # whichever kind of model extracts the aspect
            assert len(paragraph.sentences) == 3
            assert aspect.reference_sentences == [paragraph.sentences[0]]

        # === The check of what a decision model can extract ===
        decision_model = _create_decision_model()
        assert _DocumentDecisionModel._get_unservable_reasons(create_aspect()) == []
        decision_model._check_instances_servable(
            [create_aspect(), create_aspect(aspects=[create_aspect(name="Notices")])]
        )
        # ... which still rejects what a decision model cannot extract, with
        # a summary that does not limit the aspects to a reference depth
        with pytest.raises(
            ValueError,
            match=f"^Decision model `{_DECISION_MODEL}` cannot extract: aspect "
            r"`Termination` \(justifications are not supported\)\. A decision "
            "model extracts aspects, and Boolean, label and rating concepts from "
            r"text without references, both without justifications\. Route the "
            r"other aspects and concepts to an LLM, e\.g\. in a model group\.$",
        ):
            decision_model._check_instances_servable(
                [create_aspect(add_justifications=True)]
            )

        # === A decision model used on its own ===
        for extraction_method_name in ["extract_aspects_from_document", "extract_all"]:
            for use_concurrency in [False, True]:
                decision_model = _create_decision_model()
                document = _create_sentences_document(aspects=[create_aspect()])
                decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
                with patch.object(litellm, "adecisions", new=decisions_stub):
                    getattr(decision_model, extraction_method_name)(
                        document, use_concurrency=use_concurrency
                    )
                assert len(decisions_stub.requests) == 5 + 3
                check_extracted(document, with_probabilities=True)

        # === A decision model with a fallback decision model ===
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
            ]
        )
        document = _create_sentences_document(aspects=[create_aspect()])
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [auth_error], _OTHER_DECISION_MODEL: [respond]}
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert decisions_stub.requested_models() == (
            [_DECISION_MODEL] + [_OTHER_DECISION_MODEL] * 8
        )
        check_extracted(document, with_probabilities=True)

        # === A decision model with a fallback LLM ===
        # The LLM extracts the aspect with sentences as its items, as it does on
        # its own: its items have no probabilities
        decision_model = _create_decision_model(
            fallback_models=[_create_llm(is_fallback=True)]
        )
        document = _create_sentences_document(aspects=[create_aspect()])
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [auth_error]})
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: [llm_answer]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            decision_model.extract_aspects_from_document(document)
        assert len(decisions_stub.requests) == 1
        assert len(completions_stub.requests) == 1
        check_extracted(document, with_probabilities=False)

        # === An LLM with a fallback decision model ===
        # The check made before the request to the LLM accepts the aspect, and
        # the decision model extracts it when the LLM fails
        llm = _create_llm(fallback_models=[_create_decision_model(is_fallback=True)])
        document = _create_sentences_document(aspects=[create_aspect()])
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: [llm_auth_error]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            llm.extract_aspects_from_document(document)
        assert len(completions_stub.requests) == 1
        assert len(decisions_stub.requests) == 5 + 3
        check_extracted(document, with_probabilities=True)

        # === A decision model in a model group ===
        # The aspect is routed to the decision model by its role, as is
        # a sub-aspect, with sentences as its items, of an aspect that the LLM of
        # the group extracts
        model_group = DocumentModelGroup(
            models=[_create_llm(), _create_decision_model(role="decide")]
        )
        document = _create_sentences_document(
            aspects=[
                create_aspect(model_role="decide"),
                Aspect(
                    name="Payment",
                    description="Clauses on the payment of fees",
                    aspects=[
                        Aspect(
                            name="Late payment",
                            description="Clauses on the payments that are late",
                            reference_depth="sentences",
                            model_role="decide",
                        )
                    ],
                ),
            ]
        )
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: ['[{"aspect_id": "A0", "paragraph_ids": ["P2"]}]']}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            model_group.extract_aspects_from_document(document)
        # The LLM is asked about its aspect only
        assert len(completions_stub.requests) == 1
        assert [request["input"] for request in decisions_stub.requests] == [
            # The aspect of the decision model: the paragraphs of the document,
            # and the sentences of the paragraph that belongs to it
            *_SENTENCES_PARAGRAPH_TEXTS_SENT,
            *[_SENTENCES_TEXTS_SENT[(1, i)] for i in range(3)],
            # Its sub-aspect: the paragraph that the LLM extracted for the aspect
            # of the sub-aspect, and the sentences of that paragraph
            _SENTENCES_PARAGRAPH_TEXTS_SENT[2],
            *[_SENTENCES_TEXTS_SENT[(2, i)] for i in range(2)],
        ]
        check_extracted(document, with_probabilities=True)
        payment = document.aspects[1]
        assert [item.value for item in payment.extracted_items] == [
            document.paragraphs[2].raw_text
        ]
        self.check_item_probabilities(payment.extracted_items[0], None)
        (late_payment,) = payment.aspects
        (item,) = late_payment.extracted_items
        assert item.value == sentence_texts[2][1]
        self.check_item_probabilities(item, {True: 0.7, False: 1 - 0.7})
        assert item.reference_sentences == [document.paragraphs[2].sentences[1]]
        # ... which is the only reference sentence of the sub-aspect: the first
        # sentence of the paragraph is not extracted
        assert late_payment.reference_sentences == item.reference_sentences

        # === References for concepts ===
        # ... are still rejected, whatever their depth is
        decision_model = _create_decision_model()
        reference_depths: list[Literal["paragraphs", "sentences"]] = [
            "paragraphs",
            "sentences",
        ]
        for reference_depth in reference_depths:
            document = _create_sentences_document()
            document.concepts = [
                BooleanConcept(
                    name="Written notice",
                    description="A written notice is required",
                    add_references=True,
                    reference_depth=reference_depth,
                )
            ]
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(
                    ValueError,
                    match=r"concept `Written notice` \(references are not "
                    r"supported\)\. A decision model extracts aspects, and Boolean, "
                    "label and rating concepts from text without references",
                ),
            ):
                decision_model.extract_concepts_from_document(document)
            assert not decisions_stub.requests

    def test_decision_model_aspect_requests_concurrency(self):
        """
        Tests that the requests for the paragraphs, which aspects are extracted
        with by a decision model, are sent one after another by default, and
        concurrently, each paced by the limiter of the model, with concurrency
        enabled, with the same items extracted.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)
        yes_probabilities = {
            "Confidentiality": {0: 0.6, 1: 0.95, 4: 0.5},
            "Termination": {2: 0.9, 4: 0.7},
            "Payment": {},
        }
        respond = _create_aspects_responder(yes_probabilities)
        requests_in_progress: list[str] = []
        max_requests_in_progress: list[int] = []
        completed_inputs: list[str] = []
        acquisitions: list[tuple[float, float]] = []

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the config of the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                acquisitions.append((self.max_rate, self.time_period))
                await super().acquire(amount)

        # Set when the expected number of requests are in progress at the same
        # time, and when the request for the paragraph at each position is completed
        events: dict[str | int, asyncio.Event] = {}
        n_requests_at_once: list[int] = [1]

        async def wait_for_event(event: asyncio.Event) -> None:
            """
            Waits for the event, or fails if it is not set in a reasonable time,
            so that requests that are not sent as expected do not hang the test.

            :param event: The event to wait for.
            :type event: asyncio.Event
            """
            await asyncio.wait_for(event.wait(), timeout=10)

        async def respond_in_turn(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to the request, and records how many requests are in progress
            at the same time. If several requests are expected to be in progress at
            the same time, the requests are responded to in the reverse order of
            their paragraphs: the response to the request for the last paragraph
            comes when all the requests are in progress, and the response to
            the request for each of the other paragraphs when the request for
            the paragraph after it is completed.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            requests_in_progress.append(request["input"])
            max_requests_in_progress.append(len(requests_in_progress))
            position = _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
            if n_requests_at_once[0] == 1:
                # Let any other request be sent, if the requests are sent concurrently
                for _ in range(3):
                    await asyncio.sleep(0)
            else:
                if len(requests_in_progress) == n_requests_at_once[0]:
                    events["all_sent"].set()
                await wait_for_event(
                    events["all_sent"]
                    if position == n_paragraphs - 1
                    else events[position + 1]
                )
            requests_in_progress.remove(request["input"])
            completed_inputs.append(request["input"])
            events[position].set()
            return respond(request)

        def extract(
            n_requests_expected_at_once: int = 1, **extraction_kwargs: Any
        ) -> tuple[Document, _DecisionsAPIStub]:
            """
            Extracts the aspects of a new document with a new decision model.

            :param n_requests_expected_at_once: The number of requests expected to
                be in progress at the same time. Defaults to 1.
            :type n_requests_expected_at_once: int
            :param extraction_kwargs: Params of the extraction method.
            :type extraction_kwargs: Any
            :return: The document, and the stub with the requests received.
            :rtype: tuple[Document, _DecisionsAPIStub]
            """
            decision_model = _create_decision_model(async_limiter=AsyncLimiter(20, 5))
            document = _create_aspects_document(
                aspects=[
                    Aspect(name=name, description=f"Clauses on {name.lower()}")
                    for name in yes_probabilities
                ]
            )
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_in_turn]})
            max_requests_in_progress.clear()
            completed_inputs.clear()
            acquisitions.clear()
            # The events are used in the event loop of this extraction only
            n_requests_at_once[0] = n_requests_expected_at_once
            events.clear()
            events["all_sent"] = asyncio.Event()
            for position in range(n_paragraphs):
                events[position] = asyncio.Event()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
            ):
                decision_model.extract_aspects_from_document(
                    document, **extraction_kwargs
                )
            # Each request is recorded as a call, in the order of the paragraphs,
            # whatever the order in which the requests are completed
            usage = decision_model.get_usage()[0].usage
            assert len(usage.calls) == len(decisions_stub.requests)
            assert [
                json.loads(call.prompt)["input"] for call in usage.calls[:n_paragraphs]
            ] == _ASPECTS_PARAGRAPH_TEXTS_SENT
            return document, decisions_stub

        def get_items(document: Document) -> list[list[tuple[str, Any]]]:
            """
            Returns the values and the probabilities of the items of each aspect
            of the document.

            :param document: The document with the extracted aspects.
            :type document: Document
            :return: The items of each aspect, in the order of the aspects.
            :rtype: list[list[tuple[str, Any]]]
            """
            assert all(aspect._is_processed for aspect in document.aspects)
            return [
                [(item.value, item.probabilities) for item in aspect.extracted_items]
                for aspect in document.aspects
            ]

        # === Without concurrency ===
        # The requests are sent one after another, with no limiter
        document, decisions_stub = extract()
        assert [request["input"] for request in decisions_stub.requests] == (
            _ASPECTS_PARAGRAPH_TEXTS_SENT
        )
        assert max(max_requests_in_progress) == 1
        assert completed_inputs == _ASPECTS_PARAGRAPH_TEXTS_SENT
        assert acquisitions == []
        expected_items = get_items(document)
        assert [[value for value, _ in items] for items in expected_items] == [
            [document.paragraphs[position].raw_text for position in positions]
            for positions in yes_probabilities.values()
        ]

        # === With concurrency ===
        # The requests for the paragraphs are sent concurrently, each paced by
        # the limiter of the model. The items are in the order of the paragraphs,
        # whatever the order in which the requests are completed.
        document, decisions_stub = extract(
            n_requests_expected_at_once=n_paragraphs,
            use_concurrency=True,
            max_items_per_call=3,
        )
        assert [request["input"] for request in decisions_stub.requests] == (
            _ASPECTS_PARAGRAPH_TEXTS_SENT
        )
        assert all(
            len(request["questions"]) == 3 for request in decisions_stub.requests
        )
        assert max(max_requests_in_progress) == n_paragraphs
        assert completed_inputs == _ASPECTS_PARAGRAPH_TEXTS_SENT[::-1]
        assert acquisitions == [(20, 5)] * n_paragraphs
        assert get_items(document) == expected_items

        # By default, each aspect is extracted in its own call with concurrency,
        # i.e. with its own requests for the paragraphs
        document, decisions_stub = extract(
            n_requests_expected_at_once=3 * n_paragraphs, use_concurrency=True
        )
        assert sorted(request["input"] for request in decisions_stub.requests) == (
            sorted(_ASPECTS_PARAGRAPH_TEXTS_SENT * 3)
        )
        assert all(
            [question["name"] for question in request["questions"]] == ["A0"]
            for request in decisions_stub.requests
        )
        assert max(max_requests_in_progress) == 3 * n_paragraphs
        assert acquisitions == [(20, 5)] * 3 * n_paragraphs
        assert get_items(document) == expected_items

    def test_decision_model_limiter_shared_by_concurrent_calls(self):
        """
        Tests that, with concurrency enabled, all the requests to a decision model
        that are sent in the same event loop are paced by the same limiter, i.e.
        share the rate set for the model: the requests for the paragraphs, sent for
        each aspect, and the requests of the extraction calls that run concurrently
        for the concepts of different aspects.
        """
        paragraph_texts = ["Services agreement", "The parties agree as follows."]
        document = Document(raw_text="\n\n".join(paragraph_texts))
        document.aspects = [
            Aspect(
                name=f"Aspect {i}",
                description=f"Clauses on topic {i}",
                concepts=[
                    BooleanConcept(
                        name=f"Concept {i}", description=f"Topic {i} is covered"
                    )
                ],
            )
            for i in range(3)
        ]
        document.concepts = [
            BooleanConcept(name="Agreement", description="It is an agreement")
        ]
        decision_model = _create_decision_model(async_limiter=AsyncLimiter(1000, 1))
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})

        # The limiter of each request, in the order the requests are paced
        limiters: list[AsyncLimiter] = []

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                limiters.append(self)
                await super().acquire(amount)

        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
        ):
            decision_model.extract_all(document, use_concurrency=True)

        for aspect in document.aspects:
            assert [item.value for item in aspect.extracted_items] == paragraph_texts
            assert aspect.concepts[0].extracted_items[0].value is True
        assert document.concepts[0].extracted_items[0].value is True
        # With concurrency, each aspect and each concept is processed in a call
        # of its own: a request for each paragraph and each aspect, a request for
        # the concept of each aspect, and one for the concept of the document
        n_requests = 3 * len(paragraph_texts) + 3 + 1
        assert len(decisions_stub.requests) == n_requests
        assert len(limiters) == n_requests
        (limiter,) = {id(i): i for i in limiters}.values()
        assert (limiter.max_rate, limiter.time_period) == (1000, 1)
        # The limiter set for the model gives the rate, and is not used itself
        assert limiter is not decision_model.async_limiter

    def test_decision_threshold(self):
        """
        Tests the decision threshold: the one of a decision model applies to
        the aspects and the concepts that have none of their own, and the one of
        an aspect or a concept applies whichever decision model extracts it,
        including a fallback one, while an LLM does not use it. Also tests
        the values that are accepted, and that the threshold is serialized.
        """
        # === The threshold of a decision model ===
        assert _create_decision_model().decision_threshold == 0.5
        for value in [0.01, 0.5, 1.0]:
            decision_model = _create_decision_model(decision_threshold=value)
            assert decision_model.decision_threshold == value
        for value in [0.0, -0.1, 1.01, "0.5", None]:
            with pytest.raises(ValueError):
                _create_decision_model(decision_threshold=value)
        decision_model = _create_decision_model()
        with pytest.raises(ValueError):
            decision_model.decision_threshold = 0.0
        assert decision_model.decision_threshold == 0.5
        # An LLM has no decision threshold
        with pytest.raises(ValueError):
            _create_llm(decision_threshold=0.5)

        # === The threshold of an aspect or a concept ===
        aspect = Aspect(name="Termination", description="Termination clauses")
        boolean_concept = BooleanConcept(name="NDA", description="It is an NDA")
        label_concept = LabelConcept(
            name="Topics",
            description="Topics covered",
            labels=["liability", "payment"],
            classification_type="multi_label",
        )
        for instance in [aspect, boolean_concept, label_concept]:
            # Not set by default, i.e. the threshold of the decision model applies
            assert instance.decision_threshold is None
            for value in [0.01, 1.0]:
                instance.decision_threshold = value
                assert instance.decision_threshold == value
            for value in [0.0, -0.1, 1.01, "0.5"]:
                with pytest.raises(ValueError):
                    instance.decision_threshold = value  # ty: ignore[invalid-assignment]
            assert instance.decision_threshold == 1.0
            instance.decision_threshold = None
            assert instance.decision_threshold is None

        # A label concept with a single label to select has no yes/no question:
        # the threshold cannot be set for it
        match = "can be set only for the `multi_label` classification type"
        with pytest.raises(ValueError, match=match):
            LabelConcept(
                name="Type",
                description="Type of the document",
                labels=["NDA", "lease"],
                decision_threshold=0.5,
            )
        label_concept.decision_threshold = 0.7
        with pytest.raises(ValueError, match=match):
            label_concept.classification_type = "multi_class"
        assert label_concept.classification_type == "multi_label"
        assert label_concept.decision_threshold == 0.7
        label_concept.decision_threshold = None
        label_concept.classification_type = "multi_class"
        with pytest.raises(ValueError, match=match):
            label_concept.decision_threshold = 0.7
        assert label_concept.decision_threshold is None
        # The other concepts have no decision threshold
        for concept_class, kwargs in [
            (RatingConcept, {"rating_scale": (1, 5)}),
            (StringConcept, {}),
        ]:
            with pytest.raises(ValueError):
                concept_class(
                    name="Concept",
                    description="Concept description",
                    decision_threshold=0.5,  # ty: ignore[unknown-argument]
                    **kwargs,  # ty: ignore[invalid-argument-type]
                )

        # === Extraction ===
        # The probability that each paragraph belongs to each aspect, by
        # the positions of the paragraphs
        aspect_probabilities = {1: 0.9, 2: 0.6, 3: 0.4}
        respond_on_aspects = _create_aspects_responder(
            {
                "Default": aspect_probabilities,
                "Lenient": aspect_probabilities,
                "Strict": aspect_probabilities,
                "Part of lenient": {1: 0.9, 2: 0.4, 3: 0.4},
            }
        )
        # The probability of "yes" for a Boolean concept, and for each label of
        # a label concept, by the positions of the labels
        boolean_probability = 0.6
        label_probabilities = [0.8, 0.6, 0.4]
        labels = ["liability", "payment", "termination"]

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Creates the response for a request with the questions of aspects, asked
            on one paragraph, or with the questions of concepts.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            if request["questions"][0]["name"].startswith("A"):
                return respond_on_aspects(request)
            answers: dict[str, Any] = {}
            for question in request["questions"]:
                _, _, label_position = question["name"].partition("-L")
                answers[question["name"]] = (
                    label_probabilities[int(label_position)]
                    if label_position
                    else boolean_probability
                )
            return _create_decisions_response(request["questions"], answers)

        def create_document() -> Document:
            """
            Creates a document with aspects and concepts without a decision
            threshold, and with a lower and a higher one.

            :return: The document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(name="Default", description="Clauses of the default kind"),
                    Aspect(
                        name="Lenient",
                        description="Clauses of the lenient kind",
                        decision_threshold=0.3,
                        # A sub-aspect does not get the threshold of its aspect
                        aspects=[
                            Aspect(
                                name="Part of lenient",
                                description="Clauses of a part of the lenient kind",
                            )
                        ],
                    ),
                    Aspect(
                        name="Strict",
                        description="Clauses of the strict kind",
                        decision_threshold=0.7,
                    ),
                ],
                concepts=[
                    BooleanConcept(name="Flag default", description="Default flag"),
                    BooleanConcept(
                        name="Flag lenient",
                        description="Lenient flag",
                        decision_threshold=0.3,
                    ),
                    BooleanConcept(
                        name="Flag strict",
                        description="Strict flag",
                        decision_threshold=0.7,
                    ),
                    LabelConcept(
                        name="Topics default",
                        description="Default topics",
                        labels=labels,
                        classification_type="multi_label",
                    ),
                    LabelConcept(
                        name="Topics lenient",
                        description="Lenient topics",
                        labels=labels,
                        classification_type="multi_label",
                        decision_threshold=0.3,
                    ),
                    LabelConcept(
                        name="Topics strict",
                        description="Strict topics",
                        labels=labels,
                        classification_type="multi_label",
                        decision_threshold=0.9,
                    ),
                ],
            )

        def get_extracted(document: Document) -> dict[str, Any]:
            """
            Returns what is extracted from the document: the positions of
            the paragraphs of each aspect and sub-aspect, and the values of
            the items of each concept.

            :param document: The document.
            :type document: Document
            :return: What is extracted, by the names of the aspects and concepts.
            :rtype: dict[str, Any]
            """
            paragraph_texts = [paragraph.raw_text for paragraph in document.paragraphs]
            extracted: dict[str, Any] = {}
            for aspect in [
                *document.aspects,
                *[sub_aspect for i in document.aspects for sub_aspect in i.aspects],
            ]:
                extracted[aspect.name] = [
                    paragraph_texts.index(item.value) for item in aspect.extracted_items
                ]
            for concept in document.concepts:
                extracted[concept.name] = [
                    item.value for item in concept.extracted_items
                ]
            return extracted

        def check_probabilities(document: Document) -> None:
            """
            Checks that the items have the probabilities of the answers, whatever
            threshold their values are set with.

            :param document: The document.
            :type document: Document
            """
            for aspect in document.aspects:
                for item in aspect.extracted_items:
                    assert item.probabilities is not None
                    assert item.probabilities[True] in aspect_probabilities.values()
            for concept in document.concepts:
                for item in concept.extracted_items:
                    assert item.probabilities == (
                        dict(zip(labels, label_probabilities, strict=True))
                        if isinstance(concept, LabelConcept)
                        else pytest.approx(
                            {True: boolean_probability, False: 1 - boolean_probability}
                        )
                    )

        # What is extracted with the threshold of the aspect or the concept itself,
        # whatever the threshold of the decision model is
        extracted_with_own_thresholds = {
            "Lenient": [1, 2, 3],
            "Strict": [1],
            "Flag lenient": [True],
            "Flag strict": [False],
            "Topics lenient": [labels],
            # No label reaches the threshold: no item
            "Topics strict": [],
        }

        for decision_threshold, extracted_with_model_threshold in [
            (
                None,
                {
                    "Default": [1, 2],
                    # Extracted from the paragraphs of its aspect
                    "Part of lenient": [1],
                    "Flag default": [True],
                    "Topics default": [labels[:2]],
                },
            ),
            (
                0.65,
                {
                    "Default": [1],
                    "Part of lenient": [1],
                    "Flag default": [False],
                    "Topics default": [labels[:1]],
                },
            ),
            (
                0.35,
                {
                    "Default": [1, 2, 3],
                    "Part of lenient": [1, 2, 3],
                    "Flag default": [True],
                    "Topics default": [labels],
                },
            ),
        ]:
            decision_model = _create_decision_model(
                **(  # ty: ignore[invalid-argument-type]
                    {}
                    if decision_threshold is None
                    else {"decision_threshold": decision_threshold}
                )
            )
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
            document = create_document()
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_all(document)
            assert get_extracted(document) == {
                **extracted_with_model_threshold,
                **extracted_with_own_thresholds,
            }
            check_probabilities(document)
            self.check_instance_serialization_and_cloning(document)

        # === A fallback decision model: with its own threshold ===
        decision_model = _create_decision_model(
            decision_threshold=0.35,
            fallback_models=[
                _create_decision_model(
                    _OTHER_DECISION_MODEL, is_fallback=True, decision_threshold=0.65
                )
            ],
        )
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [_create_api_error(litellm.AuthenticationError)],
                _OTHER_DECISION_MODEL: [respond],
            }
        )
        document = create_document()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_all(document)
        assert get_extracted(document) == {
            "Default": [1],
            "Part of lenient": [1],
            "Flag default": [False],
            "Topics default": [labels[:1]],
            **extracted_with_own_thresholds,
        }

        # === An LLM: without the threshold ===
        document = _create_document(
            concepts=[
                BooleanConcept(
                    name="Flag strict",
                    description="Strict flag",
                    decision_threshold=1.0,
                )
            ]
        )
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": [true]}]']}
        )
        with patch.object(litellm, "acompletion", new=completions_stub):
            _create_llm().extract_concepts_from_document(document)
        (item,) = document.concepts[0].extracted_items
        assert item.value is True
        assert item.probabilities is None

        # === Serialization ===
        for decision_model in [
            _create_decision_model(),
            _create_decision_model(decision_threshold=0.65),
        ]:
            restored_decision_model = DocumentDecisionModel.from_dict(
                decision_model.to_dict()
            )
            assert (
                restored_decision_model.decision_threshold
                == decision_model.decision_threshold
            )
            assert decision_model._eq_deserialized_model_config(restored_decision_model)
        # Data serialized without the threshold, i.e. by an earlier version
        aspect = Aspect(name="Termination", description="Termination clauses")
        aspect.decision_threshold = 0.3
        aspect_dict = aspect.to_dict()
        assert aspect_dict["decision_threshold"] == 0.3
        assert Aspect.from_dict(aspect_dict).decision_threshold == 0.3
        del aspect_dict["decision_threshold"]
        assert Aspect.from_dict(aspect_dict).decision_threshold is None

    def test_decision_model_aspect_request_failures(self):
        """
        Tests the extraction of aspects with a decision model when the request for
        one of the paragraphs fails for good, or gets a response that cannot be
        used, with and without concurrency: no further requests are sent,
        the requests completed are counted in the usage, and a failed request is
        recorded as the last call, which is unfinished.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)
        failed_position = 2
        failed_text = _ASPECTS_PARAGRAPH_TEXTS_SENT[failed_position]
        auth_error = _create_api_error(litellm.AuthenticationError)
        rate_limit_error = _create_api_error(litellm.RateLimitError)
        yes_probabilities = {"Termination": {2: 0.9, 4: 0.7}, "Payment": {}}
        respond = _create_aspects_responder(yes_probabilities, input_tokens=20)

        def create_document() -> Document:
            """
            Creates a document with the aspects to extract.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(name=name, description=f"Clauses on {name.lower()}")
                    for name in yes_probabilities
                ]
            )

        def create_responder(
            outcomes: dict[int, Any] | None = None,
            delays: dict[int, float] | None = None,
        ) -> Callable[[dict[str, Any]], Any]:
            """
            Creates the function that responds to the request for a paragraph.

            :param outcomes: By the position of a paragraph: an error to raise for
                its request, or the answers to respond with instead of the valid
                ones. Defaults to None.
            :type outcomes: dict[int, Any] | None
            :param delays: By the position of a paragraph, the delay of
                the response to its request, in seconds. Defaults to None, in which
                case each response is immediate, i.e. the requests sent
                concurrently are completed in the order they are sent.
            :type delays: dict[int, float] | None
            :return: Function that responds to a request.
            :rtype: Callable[[dict[str, Any]], Any]
            """

            async def respond_to_request(request: dict[str, Any]) -> Any:
                """
                Responds to the request for a paragraph, or fails it.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response of litellm.
                :rtype: Any
                """
                position = _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
                if delays:
                    await asyncio.sleep(delays.get(position, 0))
                outcome = (outcomes or {}).get(position)
                if isinstance(outcome, Exception):
                    raise outcome
                if outcome is not None:
                    return _create_decisions_response(
                        request["questions"], outcome, input_tokens=20
                    )
                return respond(request)

            return respond_to_request

        def check_usage(
            model: DocumentDecisionModel,
            finished_positions: list[int],
            failed_positions: list[int],
        ) -> None:
            """
            Checks the calls recorded in the usage of the decision model: first
            the requests that got a response, then the requests that failed.

            :param model: The decision model.
            :type model: DocumentDecisionModel
            :param finished_positions: The positions of the paragraphs whose
                requests got a response, in the order of the calls.
            :type finished_positions: list[int]
            :param failed_positions: The positions of the paragraphs whose requests
                failed, in the order of the calls.
            :type failed_positions: list[int]
            """
            usage = model.get_usage()[0].usage
            assert [json.loads(call.prompt)["input"] for call in usage.calls] == [
                _ASPECTS_PARAGRAPH_TEXTS_SENT[position]
                for position in [*finished_positions, *failed_positions]
            ]
            assert [call.timestamp_received is not None for call in usage.calls] == (
                [True] * len(finished_positions) + [False] * len(failed_positions)
            )
            assert usage.input == 20 * len(finished_positions)

        def check_not_extracted(document: Document) -> None:
            """
            Checks that no aspect of the document is extracted.

            :param document: The document.
            :type document: Document
            """
            assert not any(aspect._is_processed for aspect in document.aspects)
            assert not any(aspect.extracted_items for aspect in document.aspects)
            assert not any(aspect.reference_paragraphs for aspect in document.aspects)

        # === The request for a paragraph in the middle of the document fails ===
        # With immediate responses, the requests sent concurrently are completed in
        # the order they are sent, i.e. the outcome is the same in both modes:
        # the requests for the paragraphs after the failed one are not sent
        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            # No fallback model: a model API error is raised
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder({failed_position: auth_error})]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(ModelAPIError) as exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert exc_info.value.original_error is auth_error
            assert exc_info.value.retry_count == 0
            assert [request["input"] for request in decisions_stub.requests] == (
                _ASPECTS_PARAGRAPH_TEXTS_SENT[: failed_position + 1]
            )
            check_usage(decision_model, [0, 1], [failed_position])
            check_not_extracted(document)

            # ... or a warning is logged, if no exception is to be raised: only
            # the one on the failed request, as no response is there to report
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub.requests.clear()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.WARNING) as captured_logs,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    raise_exception_on_extraction_error=False,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            # The warning tells the retries, unlike the errors logged before it
            (warning_message,) = [log for log in captured_logs if "retries)" in log]
            assert "calling decision model API" in warning_message
            assert "(0 retries)" in warning_message
            assert len(decisions_stub.requests) == failed_position + 1
            check_usage(decision_model, [0, 1], [failed_position])
            check_not_extracted(document)

            # A transient error is retried before the request fails for good
            decision_model = _create_decision_model(num_retries_failed_request=2)
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        create_responder({failed_position: rate_limit_error})
                    ]
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(
                    decision_models_module,
                    "_get_model_api_retry_delay",
                    new=lambda retry_number: 0.0,
                ),
                pytest.raises(ModelAPIError) as exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert exc_info.value.original_error is rate_limit_error
            assert exc_info.value.retry_count == 2
            if extraction_kwargs:
                # With concurrency, the requests for the other paragraphs are sent
                # while the retry of the failed request is waited for
                expected_inputs = [*_ASPECTS_PARAGRAPH_TEXTS_SENT, *[failed_text] * 2]
                finished_positions = [0, 1, 3, 4]
            else:
                expected_inputs = [
                    *_ASPECTS_PARAGRAPH_TEXTS_SENT[:failed_position],
                    *[failed_text] * 3,
                ]
                finished_positions = [0, 1]
            assert [request["input"] for request in decisions_stub.requests] == (
                expected_inputs
            )
            check_usage(decision_model, finished_positions, [failed_position])
            check_not_extracted(document)

            # With a fallback model, the aspects are extracted by it
            decision_model = _create_decision_model(
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ]
            )
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [create_responder({failed_position: auth_error})],
                    _OTHER_DECISION_MODEL: [respond],
                }
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert decisions_stub.requested_models() == (
                [_DECISION_MODEL] * (failed_position + 1)
                + [_OTHER_DECISION_MODEL] * n_paragraphs
            )
            check_usage(decision_model, [0, 1], [failed_position])
            fallback_usage = decision_model.get_usage()[1].usage
            assert fallback_usage.input == 20 * n_paragraphs
            assert len(fallback_usage.calls) == n_paragraphs
            termination, payment = document.aspects
            assert [item.value for item in termination.extracted_items] == [
                document.paragraphs[2].raw_text,
                document.paragraphs[4].raw_text,
            ]
            assert payment._is_processed and not payment.extracted_items

            # === A response that cannot be used ===
            # The response to the request for one paragraph makes the whole call
            # invalid: no further requests are sent, and each call is finished.
            # One call, without retries for invalid data.
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        create_responder({failed_position: {"A0": 0.9, "A1": 1.5}})
                    ]
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.ERROR) as captured_logs,
                pytest.raises(
                    ModelExtractionError,
                    match="Some aspects could not be processed due to "
                    "an invalid response returned by decision model. "
                    r"Failed aspects: \['Termination', 'Payment'\]",
                ) as extraction_exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert extraction_exc_info.value.retry_count == 0
            # The log names the aspect that the answer is for, and the paragraph
            # that the question was asked on, by its position in the document
            assert (
                f"Decision model `{_DECISION_MODEL}` returned an invalid response "
                f"for aspect `Payment` on paragraph {failed_position + 1} of "
                f"{n_paragraphs} (index {failed_position}), decision question `A1`: "
                "the probability of the answer is not a number within [0, 1]."
            ) in captured_logs, captured_logs
            assert len(decisions_stub.requests) == failed_position + 1
            check_usage(decision_model, [0, 1, failed_position], [])
            check_not_extracted(document)

        # === With concurrency, the requests already sent are waited for ===
        # All the requests are sent before the request in the middle fails:
        # the other ones are completed, and the failed one is the last call
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    create_responder(
                        {failed_position: auth_error},
                        delays={0: 0.05, 1: 0.05, 2: 0.01, 3: 0.05, 4: 0.05},
                    )
                ]
            }
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(ModelAPIError) as exc_info,
        ):
            decision_model.extract_aspects_from_document(
                document, use_concurrency=True, max_items_per_call=2
            )
        assert exc_info.value.original_error is auth_error
        assert len(decisions_stub.requests) == n_paragraphs
        check_usage(decision_model, [0, 1, 3, 4], [failed_position])
        check_not_extracted(document)

        # A failed request comes before a response that cannot be used: the call
        # is a failed one, with the failed request as the last call
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    create_responder(
                        {1: {"A0": 1.5, "A1": 0.1}, 3: auth_error},
                        delays={0: 0.05, 1: 0.01, 2: 0.05, 3: 0.03, 4: 0.05},
                    )
                ]
            }
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(ModelAPIError) as exc_info,
        ):
            decision_model.extract_aspects_from_document(
                document, use_concurrency=True, max_items_per_call=2
            )
        assert exc_info.value.original_error is auth_error
        check_usage(decision_model, [0, 1, 2, 4], [3])
        check_not_extracted(document)

        # Two requests fail in the same call: the error of the request that failed
        # first is raised, and that request is recorded as the last call, whatever
        # the positions of the two requests are
        other_error = _create_api_error(litellm.BadRequestError)
        for first_failed_position, second_failed_position in [(3, 1), (1, 3)]:
            requests_in_progress: list[int] = []
            sent_positions: list[int] = []

            async def fail_two_requests_in_turn(
                request: dict[str, Any],
                first_failed_position: int = first_failed_position,
                second_failed_position: int = second_failed_position,
                requests_in_progress: list[int] = requests_in_progress,
                sent_positions: list[int] = sent_positions,
            ) -> Any:
                """
                Responds to the request when all the requests are sent, or fails
                it: one of the two requests to fail when all the requests are
                sent, and the other one when the first one has failed.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :param first_failed_position: The position of the paragraph whose
                    request fails first.
                :type first_failed_position: int
                :param second_failed_position: The position of the paragraph whose
                    request fails after it.
                :type second_failed_position: int
                :param requests_in_progress: The positions of the paragraphs
                    whose requests are sent and not completed.
                :type requests_in_progress: list[int]
                :param sent_positions: The positions of the paragraphs whose
                    requests are sent.
                :type sent_positions: list[int]
                :return: Response of litellm.
                :rtype: Any
                """
                position = _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
                requests_in_progress.append(position)
                sent_positions.append(position)
                for _ in range(1000):
                    all_sent = len(sent_positions) == n_paragraphs
                    if all_sent and (
                        position != second_failed_position
                        or first_failed_position not in requests_in_progress
                    ):
                        break
                    await asyncio.sleep(0)
                requests_in_progress.remove(position)
                if position == first_failed_position:
                    raise auth_error
                if position == second_failed_position:
                    raise other_error
                return respond(request)

            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [fail_two_requests_in_turn]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(ModelAPIError) as exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document, use_concurrency=True, max_items_per_call=2
                )
            assert exc_info.value.original_error is auth_error
            assert len(decisions_stub.requests) == n_paragraphs
            check_usage(
                decision_model,
                [0, 2, 4],
                [second_failed_position, first_failed_position],
            )
            check_not_extracted(document)

        # The requests that wait for the limiter when a request fails are not sent
        held_acquisitions: list[int] = []
        cancelled_acquisitions: list[int] = []

        class HoldingAsyncLimiter(AsyncLimiter):
            """
            Lets the first three requests through at once, and holds each of
            the other ones for much longer than the requests take.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Acquires the capacity at once for the first three requests,
                and after a long delay for the other ones.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                held_acquisitions.append(len(held_acquisitions))
                if len(held_acquisitions) > 3:
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        cancelled_acquisitions.append(len(held_acquisitions))
                        raise

        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    create_responder(
                        {failed_position: auth_error},
                        delays={0: 0.05, 1: 0.05, 2: 0.01},
                    )
                ]
            }
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(models_module, "AsyncLimiter", new=HoldingAsyncLimiter),
            pytest.raises(ModelAPIError) as exc_info,
        ):
            decision_model.extract_aspects_from_document(
                document, use_concurrency=True, max_items_per_call=2
            )
        assert exc_info.value.original_error is auth_error
        # Each request waited for the limiter, and those held by it were not sent:
        # they were cancelled when the request failed, without the limiter being
        # waited for
        assert len(held_acquisitions) == n_paragraphs
        assert len(cancelled_acquisitions) == n_paragraphs - 3
        assert [request["input"] for request in decisions_stub.requests] == (
            _ASPECTS_PARAGRAPH_TEXTS_SENT[:3]
        )
        check_usage(decision_model, [0, 1], [failed_position])
        check_not_extracted(document)

        # === A cancelled extraction ===
        # The usage of the requests completed by then is recorded, and no request
        # is left running
        async def extract_and_cancel(
            decision_model: DocumentDecisionModel,
            document: Document,
            **extraction_kwargs: Any,
        ) -> _DecisionsAPIStub:
            """
            Starts the extraction of the aspects of the document, and cancels it
            when the request for the paragraph in the middle is sent, which gets
            no response, as the requests after it.

            :param decision_model: Decision model to extract with.
            :type decision_model: DocumentDecisionModel
            :param document: Document to extract from.
            :type document: Document
            :param extraction_kwargs: Other params of the extraction method.
            :type extraction_kwargs: Any
            :return: The stub with the requests received.
            :rtype: _DecisionsAPIStub
            """
            request_in_the_middle_sent = asyncio.Event()

            async def respond_or_hang(request: dict[str, Any]) -> Any:
                """
                Responds to the requests for the first paragraphs, and leaves
                the other requests without a response.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response of litellm.
                :rtype: Any
                """
                position = _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
                if position < failed_position:
                    return respond(request)
                if position == failed_position:
                    request_in_the_middle_sent.set()
                await asyncio.sleep(3600)

            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_or_hang]})
            with patch.object(litellm, "adecisions", new=decisions_stub):
                extraction = asyncio.ensure_future(
                    decision_model.extract_aspects_from_document_async(
                        document, **extraction_kwargs
                    )
                )
                await request_in_the_middle_sent.wait()
                # Let the other requests be sent, if they are sent concurrently
                await asyncio.sleep(0.01)
                extraction.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await extraction
            # No request is left running
            current_task = asyncio.current_task()
            assert not [
                task
                for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            ]
            return decisions_stub

        for extraction_kwargs, expected_n_requests in [
            ({}, failed_position + 1),
            ({"use_concurrency": True, "max_items_per_call": 2}, n_paragraphs),
        ]:
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = asyncio.run(
                extract_and_cancel(decision_model, document, **extraction_kwargs)
            )
            assert len(decisions_stub.requests) == expected_n_requests
            check_usage(decision_model, [0, 1], [])
            check_not_extracted(document)

    def test_decision_model_refused_aspect_answers(self):
        """
        Tests the extraction of aspects with a decision model that refuses to
        answer the question of an aspect on some of the paragraphs, or of
        the sentences: these are not included in the aspect, the other answers are
        used, a warning names the aspect and what the model refused to answer on,
        and no request is sent again. An aspect that the model refuses to answer
        about on every paragraph is not extracted: its call fails as a refusal,
        which is asked for again on every paragraph, up to
        ``max_retries_invalid_data`` times, and then passed on to the fallback
        model.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)

        def create_document(
            yes_probabilities: dict[str, Any], sub_aspect_name: str | None = None
        ) -> Document:
            """
            Creates a document with the aspects to extract.

            :param yes_probabilities: The answers by the names of the aspects.
            :type yes_probabilities: dict[str, Any]
            :param sub_aspect_name: The name of the aspect to assign as
                a sub-aspect of the first aspect. Defaults to None.
            :type sub_aspect_name: str | None
            :return: Document.
            :rtype: Document
            """
            aspects = [
                Aspect(name=name, description=f"Clauses on {name.lower()}")
                for name in yes_probabilities
                if name != sub_aspect_name
            ]
            if sub_aspect_name:
                aspects[0].aspects = [
                    Aspect(
                        name=sub_aspect_name,
                        description=f"Clauses on {sub_aspect_name.lower()}",
                    )
                ]
            return _create_aspects_document(aspects=aspects)

        def extract_aspects(
            model: DocumentDecisionModel,
            document: Document,
            decisions_stub: _DecisionsAPIStub,
            **extraction_kwargs: Any,
        ) -> tuple[list[str], list[str]]:
            """
            Extracts the aspects of the document with the decision model.

            :param model: The decision model.
            :type model: DocumentDecisionModel
            :param document: The document.
            :type document: Document
            :param decisions_stub: The stub of the Decisions API.
            :type decisions_stub: _DecisionsAPIStub
            :param extraction_kwargs: Params of the extraction method.
            :type extraction_kwargs: Any
            :return: The warnings issued on what is not included in an aspect for
                a refused answer, and the messages logged on that.
            :rtype: tuple[list[str], list[str]]
            """
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.WARNING) as captured_logs,
                warnings.catch_warnings(record=True) as caught_warnings,
            ):
                warnings.simplefilter("always")
                model.extract_aspects_from_document(document, **extraction_kwargs)
            return (
                [
                    str(caught_warning.message)
                    for caught_warning in caught_warnings
                    if "not included in the aspect" in str(caught_warning.message)
                ],
                [log for log in captured_logs if "not included in the aspect" in log],
            )

        def get_refusal_warning(aspect_name: str, refused: str) -> str:
            """
            Returns the warning expected for an aspect with refused answers.

            :param aspect_name: The name of the aspect.
            :type aspect_name: str
            :param refused: What the model refused to answer on.
            :type refused: str
            :return: The warning message.
            :rtype: str
            """
            return (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question "
                f"of aspect `{aspect_name}` on {refused}. These are not included in "
                "the aspect."
            )

        # === Refused answers on some of the paragraphs ===
        yes_probabilities: dict[str, Any] = {
            "Termination": {0: "refusal", 2: 0.9, 4: 0.7},
            "Payment": {1: "refusal", 3: 0.8, 4: "refusal"},
        }
        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            decision_model = _create_decision_model()
            document = create_document(yes_probabilities)
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [_create_aspects_responder(yes_probabilities)]}
            )
            refusal_warnings, refusal_logs = extract_aspects(
                decision_model, document, decisions_stub, **extraction_kwargs
            )
            # Each paragraph is asked about once: a refused answer neither stops
            # the other requests, nor is requested again
            assert [request["input"] for request in decisions_stub.requests] == (
                _ASPECTS_PARAGRAPH_TEXTS_SENT
            )
            # The paragraphs with a refused answer are not included in the aspect
            # that the answer is for, while they can belong to another aspect
            termination, payment = document.aspects
            paragraphs = document.paragraphs
            assert termination._is_processed and payment._is_processed
            assert [item.value for item in termination.extracted_items] == [
                paragraphs[2].raw_text,
                paragraphs[4].raw_text,
            ]
            assert termination.reference_paragraphs == [paragraphs[2], paragraphs[4]]
            assert [item.value for item in payment.extracted_items] == [
                paragraphs[3].raw_text
            ]
            assert payment.reference_paragraphs == [paragraphs[3]]
            self.check_item_probabilities(
                termination.extracted_items[1], {True: 0.7, False: 1 - 0.7}
            )
            # One warning for each aspect with refused answers, which names
            # the aspect and the paragraphs, by their indices in the document
            # Logged only, i.e. not issued as Python warnings
            assert refusal_warnings == []
            assert refusal_logs == [
                get_refusal_warning("Termination", "1 of 5 paragraphs (indices: [0])"),
                get_refusal_warning("Payment", "2 of 5 paragraphs (indices: [1, 4])"),
            ]
            # The refused answers are recorded in the usage as returned
            usage = decision_model.get_usage()[0].usage
            assert len(usage.calls) == n_paragraphs
            assert all(call.timestamp_received is not None for call in usage.calls)
            assert usage.calls[0].response is not None
            assert json.loads(usage.calls[0].response) == [
                {"type": "refusal", "name": "A0"},
                {"type": "predicate", "name": "A1", "probability": 0.1},
            ]
            self.check_instance_serialization_and_cloning(document)

        # No warning if the model answers each question
        answered_probabilities = {"Termination": {2: 0.9}, "Payment": {3: 0.8}}
        document = create_document(answered_probabilities)
        assert extract_aspects(
            _create_decision_model(),
            document,
            _DecisionsAPIStub(
                {_DECISION_MODEL: [_create_aspects_responder(answered_probabilities)]}
            ),
        ) == ([], [])

        # === A sub-aspect: the paragraphs are those of its parent aspect ===
        yes_probabilities = {
            "Termination": {2: 0.9, 4: 0.7},
            "Notice of termination": {2: 0.8, 4: "refusal"},
        }
        decision_model = _create_decision_model()
        document = create_document(
            yes_probabilities, sub_aspect_name="Notice of termination"
        )
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [_create_aspects_responder(yes_probabilities)]}
        )
        refusal_warnings, refusal_logs = extract_aspects(
            decision_model, document, decisions_stub
        )
        (termination,) = document.aspects
        (notice,) = termination.aspects
        assert [item.value for item in notice.extracted_items] == [
            document.paragraphs[2].raw_text
        ]
        # The index is the one among the two paragraphs extracted for the parent
        # Logged only, i.e. not issued as Python warnings
        assert refusal_warnings == []
        assert refusal_logs == [
            get_refusal_warning(
                "Notice of termination", "1 of 2 paragraphs (indices: [1])"
            )
        ]

        # === Refused answers on every paragraph, for one of the aspects ===
        # Nothing is known about the aspect, i.e. the response cannot be used:
        # the call is made again, and, as the model refuses again, no aspect of
        # the call is extracted, and the refusal is raised, as there is no
        # fallback model
        yes_probabilities = {
            "Termination": {2: 0.9, 4: 0.7},
            "Payment": dict.fromkeys(range(n_paragraphs), "refusal"),
        }
        respond_refusing = _create_aspects_responder(yes_probabilities)
        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            # One retry, which keeps the requests within the rate of the limiter
            # of the model when they are sent concurrently
            decision_model = _create_decision_model(max_retries_invalid_data=1)
            document = create_document(yes_probabilities)
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_refusing]})
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.WARNING) as captured_logs,
                pytest.raises(ModelRefusalError) as extraction_exc_info,
            ):
                decision_model.extract_aspects_from_document(
                    document,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            assert extraction_exc_info.value.retry_count == 1
            refusal_sentence = (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question "
                "of aspect `Payment` on every paragraph (5 asked about), decision "
                "question `A1`."
            )
            aspect_names = "['Termination', 'Payment']"
            assert extraction_exc_info.value.message == (
                f"{refusal_sentence} Failed aspects: {aspect_names}"
            )
            # Logged for each call
            assert (
                captured_logs.count(
                    f"{refusal_sentence} Aspects of the call: {aspect_names}"
                )
                == 2
            ), captured_logs
            # No warning on the paragraphs left out, as no aspect is extracted
            assert not any("not included" in log for log in captured_logs)
            # Each paragraph is asked about, and again with the retry, as
            # the answers of a request with a refused answer are not kept
            assert [request["input"] for request in decisions_stub.requests] == (
                _ASPECTS_PARAGRAPH_TEXTS_SENT * 2
            )
            usage = decision_model.get_usage()[0].usage
            assert all(call.timestamp_received is not None for call in usage.calls)
            assert len(usage.calls) == n_paragraphs * 2
            # Each request is recorded with the refusal
            assert all(call._refusal == "refusal" for call in usage.calls)
            assert not any(aspect._is_processed for aspect in document.aspects)
            assert not any(aspect.extracted_items for aspect in document.aspects)

        # The fallback model is tried, after the retries of the decision model (3 by
        # default), and its answers are used
        answered_probabilities = {"Termination": {2: 0.9, 4: 0.7}, "Payment": {3: 0.8}}
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
            ]
        )
        document = create_document(yes_probabilities)
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [respond_refusing],
                _OTHER_DECISION_MODEL: [
                    _create_aspects_responder(answered_probabilities)
                ],
            }
        )
        extract_aspects(decision_model, document, decisions_stub)
        assert decisions_stub.requested_models() == (
            [_DECISION_MODEL] * n_paragraphs * (1 + 3)
            + [_OTHER_DECISION_MODEL] * n_paragraphs
        )
        assert all(aspect._is_processed for aspect in document.aspects)
        assert document.get_aspect_by_name("Payment").extracted_items

        # A warning is logged instead, if no exception is to be raised
        decision_model = _create_decision_model()
        document = create_document(yes_probabilities)
        with (
            patch.object(
                litellm,
                "adecisions",
                new=_DecisionsAPIStub({_DECISION_MODEL: [respond_refusing]}),
            ),
            capture_logs(stdlib_logging.WARNING) as captured_logs,
        ):
            decision_model.extract_aspects_from_document(
                document, raise_exception_on_extraction_error=False
            )
        assert any(
            "refused to answer the question of aspect `Payment` on every paragraph"
            in log
            and "(3 retries). If you want to raise an exception instead" in log
            for log in captured_logs
        ), captured_logs
        assert not any(aspect._is_processed for aspect in document.aspects)

        # === Aspects with sentences as their items ===
        # A paragraph with a refused answer is not included, i.e. its sentences
        # are not asked about, and a sentence with a refused answer is not included
        sentence_probabilities: dict[str, Any] = {
            "Termination": {
                0: "refusal",
                1: 0.9,
                (1, 0): 0.8,
                (1, 1): "refusal",
                3: 0.9,
                (3, 0): "refusal",
                (3, 1): 0.7,
            },
            "Payment": {2: 0.9, (2, 0): 0.8, (2, 1): 0.6, 4: "refusal"},
        }

        def create_sentences_document(names: list[str]) -> Document:
            """
            Creates a document with the aspects to extract with sentences as
            their items.

            :param names: The names of the aspects.
            :type names: list[str]
            :return: Document.
            :rtype: Document
            """
            return _create_sentences_document(
                aspects=[
                    Aspect(
                        name=name,
                        description=f"Clauses on {name.lower()}",
                        reference_depth="sentences",
                    )
                    for name in names
                ]
            )

        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            decision_model = _create_decision_model()
            document = create_sentences_document(["Termination", "Payment"])
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [_create_sentences_responder(sentence_probabilities)]}
            )
            refusal_warnings, refusal_logs = extract_aspects(
                decision_model, document, decisions_stub, **extraction_kwargs
            )
            # The paragraphs, then the sentences of the paragraphs that belong to
            # an aspect, with the question of that aspect
            assert [
                (
                    _get_sentences_request_position(request),
                    [question["name"] for question in request["questions"]],
                )
                for request in decisions_stub.requests
            ] == [
                *[(position, ["A0", "A1"]) for position in range(5)],
                *[((1, position), ["A0"]) for position in range(3)],
                *[((2, position), ["A1"]) for position in range(2)],
                *[((3, position), ["A0"]) for position in range(2)],
            ]
            termination, payment = document.aspects
            paragraphs = document.paragraphs
            assert [
                [item.reference_sentences for item in aspect.extracted_items]
                for aspect in [termination, payment]
            ] == [
                [[paragraphs[1].sentences[0]], [paragraphs[3].sentences[1]]],
                [[paragraphs[2].sentences[0]], [paragraphs[2].sentences[1]]],
            ]
            assert termination.reference_paragraphs == [paragraphs[1], paragraphs[3]]
            # The sentences are counted out of those asked about for the aspect,
            # and each is given by the indices of its paragraph and of itself
            # Logged only, i.e. not issued as Python warnings
            assert refusal_warnings == []
            assert refusal_logs == [
                get_refusal_warning(
                    "Termination",
                    "1 of 5 paragraphs (indices: [0]) and on 2 of 5 sentences "
                    "(paragraph index, sentence index: [(1, 1), (3, 0)])",
                ),
                get_refusal_warning("Payment", "1 of 5 paragraphs (indices: [4])"),
            ]
            self.check_instance_serialization_and_cloning(document)

        # Refused answers on every sentence asked about: the aspect is extracted,
        # as the model answered about it on the paragraphs, and has no items
        sentence_probabilities = {
            "Termination": {
                1: 0.9,
                **dict.fromkeys([(1, 0), (1, 1), (1, 2)], "refusal"),
            }
        }
        decision_model = _create_decision_model()
        document = create_sentences_document(["Termination"])
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [_create_sentences_responder(sentence_probabilities)]}
        )
        refusal_warnings, refusal_logs = extract_aspects(
            decision_model, document, decisions_stub
        )
        (termination,) = document.aspects
        assert termination._is_processed and not termination.extracted_items
        assert len(decisions_stub.requests) == 5 + 3
        # Logged only, i.e. not issued as a Python warning
        assert refusal_warnings == []
        assert refusal_logs == [
            get_refusal_warning(
                "Termination",
                "3 of 3 sentences (paragraph index, sentence index: "
                "[(1, 0), (1, 1), (1, 2)])",
            )
        ]

    def test_extract_all_with_decision_model(self):
        """
        Tests the extraction of everything that a document has with a decision
        model used on its own: the aspects with their sub-aspects, the concepts
        of the aspects and of the sub-aspects, and the concepts of the document.
        """

        def create_document() -> Document:
            """
            Creates a document with aspects, sub-aspects and concepts.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        concepts=[
                            BooleanConcept(
                                name="Termination for convenience",
                                description="Either party may terminate without cause",
                            ),
                            LabelConcept(
                                name="Notice form",
                                description="Required form of the termination notice",
                                labels=["written", "any"],
                            ),
                        ],
                        aspects=[
                            Aspect(
                                name="Effects of termination",
                                description="Clauses on what follows the termination",
                                concepts=[
                                    BooleanConcept(
                                        name="Survival",
                                        description="Some obligations survive "
                                        "the termination",
                                    )
                                ],
                            )
                        ],
                    ),
                    Aspect(
                        name="Payment",
                        description="Clauses on the payment of fees",
                        concepts=[
                            BooleanConcept(
                                name="Late fee",
                                description="A fee is charged on late payments",
                            )
                        ],
                    ),
                ],
                concepts=[
                    RatingConcept(
                        name="Clarity",
                        description="Clarity of the text of the document",
                        rating_scale=(1, 3),
                    ),
                    BooleanConcept(
                        name="NDA check", description="The document is an NDA"
                    ),
                ],
            )

        respond_on_aspects = _create_aspects_responder(
            {
                "Termination": {2: 0.9, 4: 0.7},
                "Effects of termination": {4: 0.8},
                "Payment": {},
            },
            input_tokens=20,
        )

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request with the questions of aspects, asked on
            a paragraph, or with the questions of concepts.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            if request["questions"][0]["name"].startswith("A"):
                return respond_on_aspects(request)
            return _answer_all_questions(request)

        def get_items(document: Document) -> dict[str, list[tuple[Any, Any]]]:
            """
            Returns the values and the probabilities of the items of each aspect,
            sub-aspect and concept of the document.

            :param document: The document.
            :type document: Document
            :return: The items by the names of the aspects and concepts.
            :rtype: dict[str, list[tuple[Any, Any]]]
            """
            instances: list[Any] = [*document.concepts]
            for aspect in document.aspects:
                instances.extend([aspect, *aspect.concepts])
                for sub_aspect in aspect.aspects:
                    instances.extend([sub_aspect, *sub_aspect.concepts])
            return {
                instance.name: [
                    (item.value, item.probabilities)
                    for item in instance.extracted_items
                ]
                for instance in instances
            }

        decision_model = _create_decision_model(
            pricing_details=ModelPricing(
                input_per_1m_tokens=10.0, output_per_1m_tokens=0.0
            )
        )
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_all(document)

        paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        # The text sent for the concepts has no additional context of the paragraphs
        cleaned_texts = _ASPECTS_PARAGRAPH_TEXTS_CLEANED
        termination_text = f"{cleaned_texts[2]}\n\n{cleaned_texts[4]}"
        assert [
            (request["input"], [question["name"] for question in request["questions"]])
            for request in decisions_stub.requests
        ] == [
            # The aspects, on each paragraph of the document
            *[(text, ["A0", "A1"]) for text in paragraph_texts],
            # The sub-aspect, on each paragraph of its aspect
            (paragraph_texts[2], ["A0"]),
            (paragraph_texts[4], ["A0"]),
            # The concepts of the aspect, on the text of its paragraphs
            (termination_text, ["C0", "C1"]),
            # The concept of the sub-aspect, on the text of its paragraph
            (cleaned_texts[4], ["C0"]),
            # The concepts of the document, on its text
            ("\n\n".join(cleaned_texts), ["C0", "C1"]),
        ]
        paragraphs = document.paragraphs
        termination_items = [
            (paragraphs[2].raw_text, {True: 0.9, False: 1 - 0.9}),
            (paragraphs[4].raw_text, {True: 0.7, False: 1 - 0.7}),
        ]
        expected_items = {
            "Termination": termination_items,
            "Termination for convenience": [(True, {True: 0.9, False: 1 - 0.9})],
            "Notice form": [(["written"], {"written": 1.0, "any": 0.0})],
            "Effects of termination": [
                (paragraphs[4].raw_text, {True: 0.8, False: 1 - 0.8})
            ],
            "Survival": [(True, {True: 0.9, False: 1 - 0.9})],
            # No paragraph belongs to the aspect, i.e. there is no text to extract
            # its concept from
            "Payment": [],
            "Late fee": [],
            "Clarity": [(1, {1: 1.0, 2: 0.0, 3: 0.0})],
            "NDA check": [(True, {True: 0.9, False: 1 - 0.9})],
        }
        assert get_items(document) == expected_items
        termination, payment = document.aspects
        assert payment._is_processed
        assert not payment.concepts[0]._is_processed
        assert all(concept._is_processed for concept in termination.concepts)
        assert termination.aspects[0].concepts[0]._is_processed
        assert all(concept._is_processed for concept in document.concepts)
        # Each request is recorded in the usage and priced
        (usage_container,) = decision_model.get_usage()
        assert len(usage_container.usage.calls) == len(decisions_stub.requests) == 10
        assert usage_container.usage.input == 7 * 20 + 3 * 100
        assert decision_model.get_cost()[0].cost.total == Decimal("0.0044")
        self.check_instance_serialization_and_cloning(document)

        # The same is extracted with concurrency, with each aspect and each concept
        # extracted in its own call
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_all(document, use_concurrency=True)
        assert get_items(document) == expected_items
        assert all(
            len(request["questions"]) == 1 for request in decisions_stub.requests
        )
        # 2 aspects and 1 sub-aspect on their paragraphs, and 5 concepts
        assert len(decisions_stub.requests) == 2 * 5 + 2 + 5

    def test_decision_model_aspect_fallback_chains(self):
        """
        Tests the fallback chains with models of different kinds for the extraction
        of aspects: an LLM that falls back to a decision model, and a decision
        model that falls back to an LLM.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)
        auth_error = _create_api_error(litellm.AuthenticationError)
        llm_auth_error = _create_api_error(litellm.AuthenticationError, _LLM_MODEL)
        yes_probabilities = {
            "Termination": {2: 0.9, 4: 0.7},
            "Effects of termination": {4: 0.8},
            "Payment": {},
        }
        respond = _create_aspects_responder(yes_probabilities)
        # The answers of an LLM for the aspects, and for the sub-aspect, which
        # is extracted from the two paragraphs of its aspect
        llm_aspects_answer = json.dumps(
            [
                {"aspect_id": "A0", "paragraph_ids": ["P2", "P4"]},
                {"aspect_id": "A1", "paragraph_ids": []},
            ]
        )
        llm_sub_aspect_answer = json.dumps(
            [{"aspect_id": "A0", "paragraph_ids": ["P1"]}]
        )
        invalid_llm_answer = "<<<not-json>>>"

        def create_document() -> Document:
            """
            Creates a document with two aspects, one of which has a sub-aspect.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on the termination of the agreement",
                        aspects=[
                            Aspect(
                                name="Effects of termination",
                                description="Clauses on what follows the termination",
                            )
                        ],
                    ),
                    Aspect(
                        name="Payment", description="Clauses on the payment of fees"
                    ),
                ]
            )

        def get_items(document: Document) -> dict[str, list[tuple[Any, Any]]]:
            """
            Returns the values and the probabilities of the items of each aspect
            and sub-aspect of the document.

            :param document: The document.
            :type document: Document
            :return: The items by the names of the aspects.
            :rtype: dict[str, list[tuple[Any, Any]]]
            """
            aspects: list[Any] = []
            for aspect in document.aspects:
                assert aspect._is_processed
                aspects.extend([aspect, *aspect.aspects])
            return {
                aspect.name: [
                    (item.value, item.probabilities) for item in aspect.extracted_items
                ]
                for aspect in aspects
            }

        paragraph_texts = [
            paragraph.raw_text for paragraph in create_document().paragraphs
        ]
        # The items as extracted by a decision model, with their probabilities,
        # and as extracted by an LLM, without them
        decision_model_items = {
            "Termination": [
                (paragraph_texts[2], {True: 0.9, False: 1 - 0.9}),
                (paragraph_texts[4], {True: 0.7, False: 1 - 0.7}),
            ],
            "Effects of termination": [
                (paragraph_texts[4], {True: 0.8, False: 1 - 0.8})
            ],
            "Payment": [],
        }
        llm_items = {
            name: [(value, None) for value, _ in items]
            for name, items in decision_model_items.items()
        }

        # === An LLM with a decision model as its fallback model ===
        for llm_outcome, n_llm_requests_per_call in [
            # The request to the LLM fails
            (llm_auth_error, 1),
            # The LLM returns invalid data, also when asked again
            (invalid_llm_answer, 2),
        ]:
            for extraction_kwargs in [
                {},
                {"use_concurrency": True, "max_items_per_call": 2},
            ]:
                llm = _create_llm(
                    max_retries_invalid_data=1,
                    fallback_models=[_create_decision_model(is_fallback=True)],
                )
                document = create_document()
                decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
                completions_stub = _CompletionsAPIStub({_LLM_MODEL: [llm_outcome]})
                with (
                    patch.object(litellm, "adecisions", new=decisions_stub),
                    patch.object(litellm, "acompletion", new=completions_stub),
                ):
                    llm.extract_aspects_from_document(
                        document,
                        **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                    )
                # The LLM is asked for the aspects and for the sub-aspect, and
                # the decision model about each paragraph for both
                assert len(completions_stub.requests) == 2 * n_llm_requests_per_call
                assert sorted(
                    request["input"] for request in decisions_stub.requests
                ) == sorted(
                    [
                        *_ASPECTS_PARAGRAPH_TEXTS_SENT,
                        _ASPECTS_PARAGRAPH_TEXTS_SENT[2],
                        _ASPECTS_PARAGRAPH_TEXTS_SENT[4],
                    ]
                )
                assert get_items(document) == decision_model_items
                usage_containers = llm.get_usage()
                assert [
                    (container.model, container.is_fallback)
                    for container in usage_containers
                ] == [(_LLM_MODEL, False), (_DECISION_MODEL, True)]
                assert len(usage_containers[1].usage.calls) == n_paragraphs + 2
                assert usage_containers[1].usage.input == (n_paragraphs + 2) * 100

        # The fallback decision model is the last model to fail
        llm = _create_llm(fallback_models=[_create_decision_model(is_fallback=True)])
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [auth_error]})
        completions_stub = _CompletionsAPIStub({_LLM_MODEL: [llm_auth_error]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
            pytest.raises(ModelAPIError, match="decision model API") as exc_info,
        ):
            llm.extract_aspects_from_document(document)
        assert exc_info.value.original_error is auth_error
        assert len(decisions_stub.requests) == 1
        assert not any(aspect._is_processed for aspect in document.aspects)

        # === A decision model with an LLM as its fallback model ===
        for decision_model_outcomes, n_requests_on_failed_paragraph in [
            # The request for the paragraph in the middle fails: it is not sent
            # again
            ({2: auth_error}, 1),
            # The response to it cannot be used: it is sent again, up to
            # the `max_retries_invalid_data` of the decision model (3 by default)
            ({2: {"A0": 0.9, "A1": 1.5}}, 1 + 3),
        ]:
            decision_model = _create_decision_model(
                fallback_models=[_create_llm(is_fallback=True)]
            )
            document = create_document()

            def respond_or_fail(
                request: dict[str, Any],
                outcomes: dict[int, Any] = decision_model_outcomes,
            ) -> OpenAIDecisionResponse:
                """
                Responds to the request for a paragraph, unless there is another
                outcome for the paragraph.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :param outcomes: By the position of a paragraph: an error to raise
                    for its request, or the answers to respond with.
                :type outcomes: dict[int, Any]
                :return: Response of litellm.
                :rtype: OpenAIDecisionResponse
                """
                outcome = None
                # Only the requests for the two aspects, not for the sub-aspect
                if len(request["questions"]) == 2:
                    outcome = outcomes.get(
                        _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
                    )
                if isinstance(outcome, Exception):
                    raise outcome
                if outcome is not None:
                    return _create_decisions_response(request["questions"], outcome)
                return respond(request)

            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_or_fail]})
            completions_stub = _CompletionsAPIStub(
                {_LLM_MODEL: [llm_aspects_answer, llm_sub_aspect_answer]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(litellm, "acompletion", new=completions_stub),
            ):
                decision_model.extract_aspects_from_document(document)
            # The aspects are extracted by the LLM, after the decision model fails
            # for them on the third paragraph, which is the only one that it is
            # asked about again, as its answers on the paragraphs before it are
            # kept. The sub-aspect is extracted by the decision model, which is
            # asked about each paragraph of its aspect.
            assert [request["input"] for request in decisions_stub.requests] == [
                *_ASPECTS_PARAGRAPH_TEXTS_SENT[:2],
                *[_ASPECTS_PARAGRAPH_TEXTS_SENT[2]] * n_requests_on_failed_paragraph,
                _ASPECTS_PARAGRAPH_TEXTS_SENT[2],
                _ASPECTS_PARAGRAPH_TEXTS_SENT[4],
            ]
            assert len(completions_stub.requests) == 1
            assert get_items(document) == {
                **llm_items,
                "Effects of termination": decision_model_items[
                    "Effects of termination"
                ],
            }
            # The paragraphs that the LLM extracted are the reference paragraphs
            assert document.aspects[0].reference_paragraphs == [
                document.paragraphs[2],
                document.paragraphs[4],
            ]

        # The fallback LLM extracts what the decision model fails to extract at
        # each level: the aspects and the sub-aspect
        decision_model = _create_decision_model(
            fallback_models=[_create_llm(is_fallback=True)]
        )
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [auth_error]})
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: [llm_aspects_answer, llm_sub_aspect_answer]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            decision_model.extract_aspects_from_document(document)
        assert len(decisions_stub.requests) == 2
        assert len(completions_stub.requests) == 2
        assert get_items(document) == llm_items

    def test_decision_model_aspects_in_model_group(self):
        """
        Tests a model group with an LLM and a decision model for the extraction
        of aspects: each aspect, sub-aspect and concept is extracted by the model
        that it is routed to by the role, whatever model extracts its parent.
        """
        llm = _create_llm()
        decision_model = _create_decision_model(role="decide")
        model_group = DocumentModelGroup(models=[llm, decision_model])
        document = _create_aspects_document(
            aspects=[
                # An aspect of the decision model, with a sub-aspect and a concept
                # of the LLM, and a concept of the decision model
                Aspect(
                    name="Termination",
                    description="Clauses on the termination of the agreement",
                    model_role="decide",
                    aspects=[
                        Aspect(
                            name="Effects of termination",
                            description="Clauses on what follows the termination",
                        )
                    ],
                    concepts=[
                        StringConcept(
                            name="Notice period",
                            description="Length of the notice period",
                        ),
                        BooleanConcept(
                            name="Termination for convenience",
                            description="Either party may terminate without cause",
                            model_role="decide",
                        ),
                    ],
                ),
                # An aspect of the LLM, with a sub-aspect of the decision model
                Aspect(
                    name="Confidentiality",
                    description="Clauses on the confidentiality of the information",
                    aspects=[
                        Aspect(
                            name="Survival of confidentiality",
                            description="Clauses on the confidentiality after "
                            "the termination",
                            model_role="decide",
                        )
                    ],
                ),
            ],
            concepts=[StringConcept(name="Title", description="Title of the document")],
        )
        paragraphs = document.paragraphs
        # The paragraphs that the LLM returns for an aspect, by their positions
        # among the paragraphs that the aspect is extracted from
        llm_paragraph_ids = {
            # From the paragraphs of the document
            "Confidentiality": ["P1", "P4"],
            # From the two paragraphs that the decision model extracted for
            # the aspect of this sub-aspect
            "Effects of termination": ["P1"],
        }
        llm_prompt_kwargs: list[dict[str, Any]] = []

        async def send_messages_stub(self, *, model_call_obj: Any, **kwargs: Any):
            """
            Simulates a successful LLM API call, which returns the paragraphs of
            each aspect in the prompt, or an extracted item for each concept in
            the prompt.

            Same params as `llm._send_messages_with_tools`.
            """
            params = model_call_obj.prompt_kwargs
            llm_prompt_kwargs.append(params)
            if "aspects" in params:
                answer = [
                    {
                        "aspect_id": f"A{i}",
                        "paragraph_ids": llm_paragraph_ids[aspect.name],
                    }
                    for i, aspect in enumerate(params["aspects"])
                ]
            else:
                answer = [
                    {"concept_id": f"C{i}", "extracted_items": ["Item"]}
                    for i, _ in enumerate(params["concepts"])
                ]
            return json.dumps(answer), False

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request with the questions of aspects, asked on
            a paragraph, or with the questions of concepts.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            if request["questions"][0]["name"].startswith("A"):
                return _create_aspects_responder(
                    {
                        "Termination": {2: 0.9, 4: 0.7},
                        "Survival of confidentiality": {4: 0.8},
                    },
                    input_tokens=20,
                )(request)
            return _answer_all_questions(request)

        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                _DocumentLLM,
                "_send_messages_with_tools",
                autospec=True,
                side_effect=send_messages_stub,
            ),
        ):
            model_group.extract_all(document)

        # === The requests to the decision model ===
        def get_instance_name(question: dict[str, Any]) -> str:
            """
            Returns the name of the aspect or concept that the question is about.

            :param question: The question of a request.
            :type question: dict[str, Any]
            :return: The name of the aspect or concept.
            :rtype: str
            """
            name_match = re.search(
                r"^(?:Aspect|Concept) name: (.+)$",
                question["instructions"],
                flags=re.MULTILINE,
            )
            assert name_match is not None, question
            return name_match.group(1)

        paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        assert sorted(
            (
                request["input"],
                [get_instance_name(question) for question in request["questions"]],
            )
            for request in decisions_stub.requests
        ) == sorted(
            [
                # Its aspect, on each paragraph of the document
                *[(text, ["Termination"]) for text in paragraph_texts],
                # Its sub-aspect, on each paragraph that the LLM extracted for
                # the aspect of the sub-aspect
                (paragraph_texts[1], ["Survival of confidentiality"]),
                (paragraph_texts[4], ["Survival of confidentiality"]),
                # Its concept, on the text of the paragraphs of its aspect, which
                # has no additional context of the paragraphs
                (
                    (
                        f"{_ASPECTS_PARAGRAPH_TEXTS_CLEANED[2]}\n\n"
                        f"{_ASPECTS_PARAGRAPH_TEXTS_CLEANED[4]}"
                    ),
                    ["Termination for convenience"],
                ),
            ]
        )

        # === The calls of the LLM ===
        assert sorted(
            (
                # What is extracted
                [
                    instance.name
                    for instance in params.get("aspects", params.get("concepts"))
                ],
                # The paragraphs given to extract aspects from
                [paragraph.raw_text for paragraph in params.get("paragraphs", [])],
            )
            for params in llm_prompt_kwargs
        ) == sorted(
            [
                # Its aspect, from the paragraphs of the document
                (["Confidentiality"], [paragraph.raw_text for paragraph in paragraphs]),
                # Its sub-aspect, from the paragraphs that the decision model
                # extracted for the aspect of the sub-aspect
                (
                    ["Effects of termination"],
                    [paragraphs[2].raw_text, paragraphs[4].raw_text],
                ),
                # Its concepts
                (["Notice period"], []),
                (["Title"], []),
            ]
        )

        # === The items ===
        termination, confidentiality = document.aspects
        assert [
            (item.value, item.probabilities) for item in termination.extracted_items
        ] == [
            (paragraphs[2].raw_text, {True: 0.9, False: 1 - 0.9}),
            (paragraphs[4].raw_text, {True: 0.7, False: 1 - 0.7}),
        ]
        assert termination.reference_paragraphs == [paragraphs[2], paragraphs[4]]
        # The sub-aspect extracted by the LLM from the paragraphs of the aspect
        # extracted by the decision model
        (effects_of_termination,) = termination.aspects
        assert [
            (item.value, item.probabilities)
            for item in effects_of_termination.extracted_items
        ] == [(paragraphs[4].raw_text, None)]
        notice_period, termination_for_convenience = termination.concepts
        assert notice_period.extracted_items[0].value == "Item"
        assert notice_period.extracted_items[0].probabilities is None
        self.check_item_probabilities(
            termination_for_convenience.extracted_items[0], {True: 0.9, False: 1 - 0.9}
        )
        # The aspect extracted by the LLM
        assert [
            (item.value, item.probabilities) for item in confidentiality.extracted_items
        ] == [(paragraphs[1].raw_text, None), (paragraphs[4].raw_text, None)]
        # ... and its sub-aspect extracted by the decision model
        (survival,) = confidentiality.aspects
        assert [
            (item.value, item.probabilities) for item in survival.extracted_items
        ] == [(paragraphs[4].raw_text, {True: 0.8, False: 1 - 0.8})]
        assert survival.reference_paragraphs == [paragraphs[4]]
        assert document.concepts[0].extracted_items[0].value == "Item"

        # === The usage of the group, by the role ===
        (usage_container,) = model_group.get_usage(model_role="decide")
        assert usage_container.model == _DECISION_MODEL
        assert len(usage_container.usage.calls) == len(decisions_stub.requests) == 8
        assert usage_container.usage.input == 7 * 20 + 100
        self.check_instance_serialization_and_cloning(document)

        # === With concurrency ===
        # The same is extracted
        items_extracted = [
            [(item.value, item.probabilities) for item in aspect.extracted_items]
            for aspect in [
                termination,
                effects_of_termination,
                confidentiality,
                survival,
            ]
        ]
        decisions_stub.requests.clear()
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                _DocumentLLM,
                "_send_messages_with_tools",
                autospec=True,
                side_effect=send_messages_stub,
            ),
        ):
            model_group.extract_aspects_from_document(
                document, overwrite_existing=True, use_concurrency=True
            )
        termination, confidentiality = document.aspects
        assert [
            [(item.value, item.probabilities) for item in aspect.extracted_items]
            for aspect in [
                termination,
                termination.aspects[0],
                confidentiality,
                confidentiality.aspects[0],
            ]
        ] == items_extracted
        assert len(decisions_stub.requests) == 7

    def test_answers_without_usable_probabilities(self):
        """
        Tests that an answer of a provider without usable probabilities of
        the options offered ends as an invalid response, and not as an item with
        the first option and zero probabilities. The responses are created by
        the conversion functions of litellm themselves, from the responses of
        a provider in the System One format: litellm returns the probabilities of
        the options offered only, with a zero for each option that the provider
        returned no probability for, e.g. an option that it named differently.
        """
        # The functions that litellm converts the response of a System One
        # provider with, for a request made in the OpenAI format. They are
        # internals of litellm, i.e. they are imported here: if they are moved in
        # another version of litellm, this test fails, and no other one.
        from litellm.llms.base_llm.decisions.transformation import (
            parse_systemone_response,
        )
        from litellm.llms.openai.decisions.transformation import (
            ir_to_openai_response,
            openai_request_to_ir,
        )
        from litellm.types.decisions import OpenAIDecisionRequestBody

        def create_responder(
            provider_answer: dict[str, Any],
        ) -> Callable[[dict[str, Any]], Any]:
            """
            Creates the function that responds to a request with a single question
            as litellm does when the provider returns the given answer to it.

            :param provider_answer: The answer of the provider to the question,
                in the System One format.
            :type provider_answer: dict[str, Any]
            :return: Function that creates the response for a request.
            :rtype: Callable[[dict[str, Any]], Any]
            """

            def respond(request: dict[str, Any]) -> Any:
                """
                Converts the response of the provider with the functions of litellm.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response of litellm.
                :rtype: Any
                """
                (question,) = request["questions"]
                ir_request = openai_request_to_ir(
                    OpenAIDecisionRequestBody.model_validate(
                        {"input": request["input"], "questions": request["questions"]}
                    )
                )
                ir_response = parse_systemone_response(
                    {
                        "model": "jev-1.13.0",
                        "answers": {question["name"]: provider_answer},
                        "usage": {"input_tokens": 50, "output_tokens": 0},
                    },
                    ir_request,
                )
                return ir_to_openai_response(ir_response, ir_request, request["model"])

            return respond

        label_concept = LabelConcept(
            name="Document type",
            description="Type of the document",
            labels=["Contract", "Invoice", "Letter"],
        )
        rating_concept = RatingConcept(
            name="Clarity",
            description="Clarity of the text of the document",
            rating_scale=(1, 5),
        )

        # === The answers with the probabilities of the options offered ===
        for concept, provider_answer, expected_value, expected_probabilities in [
            (
                label_concept,
                {
                    "type": "choice",
                    "choice": "Letter",
                    "confidence": 0.9,
                    "probabilities": {"Contract": 0.04, "Invoice": 0.06, "Letter": 0.9},
                },
                ["Letter"],
                {"Contract": 0.04, "Invoice": 0.06, "Letter": 0.9},
            ),
            # The levels of a rating are keyed by their positions on the scale
            (
                rating_concept,
                {
                    "type": "score",
                    "score": 3.9,
                    "confidence": 0.7,
                    "legend": {},
                    "probabilities": {"0": 0.0, "1": 0.0, "2": 0.2, "3": 0.7, "4": 0.1},
                },
                4,
                {1: 0.0, 2: 0.0, 3: 0.2, 4: 0.7, 5: 0.1},
            ),
        ]:
            decision_model = _create_decision_model()
            document = _create_document(concepts=[concept])
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder(provider_answer)]}
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_concepts_from_document(document)
            (item,) = document.concepts[0].extracted_items
            assert item.value == expected_value
            assert item.probabilities == expected_probabilities

        # === The answers without usable probabilities ===
        for concept, provider_answer, reason in [
            # No probabilities at all: only the picked option is returned
            (
                label_concept,
                {
                    "type": "choice",
                    "choice": "Letter",
                    "confidence": 0.9,
                    "probabilities": {},
                },
                "the probability of each option is zero",
            ),
            (
                rating_concept,
                {
                    "type": "score",
                    "score": 4.0,
                    "confidence": 0.7,
                    "legend": {},
                    "probabilities": {},
                },
                "the probability of each option is zero",
            ),
            # The options named differently than they were offered, here in
            # another case: none of the probabilities is of an option offered
            (
                label_concept,
                {
                    "type": "choice",
                    "choice": "letter",
                    "confidence": 0.9,
                    "probabilities": {"contract": 0.04, "invoice": 0.06, "letter": 0.9},
                },
                (
                    "the answer picks an option that was not offered. Options offered: "
                    "['Contract', 'Invoice', 'Letter']."
                ),
            ),
            # ... also when only the picked option is named differently
            (
                label_concept,
                {
                    "type": "choice",
                    "choice": "LETTER",
                    "confidence": 0.9,
                    "probabilities": {"Contract": 0.04, "Invoice": 0.06, "Letter": 0.9},
                },
                (
                    "the answer picks an option that was not offered. Options offered: "
                    "['Contract', 'Invoice', 'Letter']."
                ),
            ),
        ]:
            # One call, without retries for invalid data
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = _create_document(concepts=[concept])
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder(provider_answer)]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs(stdlib_logging.ERROR) as captured_logs,
                pytest.raises(
                    ModelExtractionError,
                    match="an invalid response returned by decision model",
                ),
            ):
                decision_model.extract_concepts_from_document(document)
            assert any(
                f"returned an invalid response for concept `{concept.name}`, "
                f"decision question `C0`: {reason}" in log
                for log in captured_logs
            ), captured_logs
            assert not document.concepts[0]._is_processed
            assert not document.concepts[0].extracted_items
            # The request succeeded: it is counted, and it is not sent again
            # without retries
            assert len(decisions_stub.requests) == 1
            usage = decision_model.get_usage()[0].usage
            assert usage.input == 50
            assert [call.timestamp_received is not None for call in usage.calls] == [
                True
            ]

    def test_overwrite_existing_resets_aspect_references(self):
        """
        Tests that an aspect extracted again by a decision model, with
        ``overwrite_existing=True``, has the reference paragraphs of its new items
        only, so that its sub-aspects and its concepts are extracted from them,
        and that the references of an aspect are still added to without it.
        """
        yes_positions: dict[str, list[int]] = {}

        def respond(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to a request with the questions of aspects, asked on one
            paragraph, or with the questions of concepts.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            answers: dict[str, Any] = {}
            for question in request["questions"]:
                aspect_name_match = re.search(
                    r"^Aspect name: (.+)$", question["instructions"], flags=re.MULTILINE
                )
                if aspect_name_match is None:
                    answers[question["name"]] = 0.9
                    continue
                position = _ASPECTS_PARAGRAPH_TEXTS_SENT.index(request["input"])
                answers[question["name"]] = (
                    0.9
                    if position in yes_positions[aspect_name_match.group(1)]
                    else 0.1
                )
            return _create_decisions_response(request["questions"], answers)

        decision_model = _create_decision_model()
        document = _create_aspects_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    aspects=[Aspect(name="Notices", description="Clauses on notices")],
                    concepts=[
                        BooleanConcept(
                            name="Written notice",
                            description="A written notice is required",
                        )
                    ],
                )
            ]
        )
        (aspect,) = document.aspects
        paragraphs = document.paragraphs

        # === The first extraction ===
        yes_positions.update({"Termination": [2, 4], "Notices": [2]})
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraphs[2]),
            id(paragraphs[4]),
        ]
        assert not aspect.reference_sentences
        assert [request["input"] for request in decisions_stub.requests[5:]] == [
            _ASPECTS_PARAGRAPH_TEXTS_SENT[2],
            _ASPECTS_PARAGRAPH_TEXTS_SENT[4],
        ]

        # === The aspect is extracted again, from other paragraphs ===
        # A paragraph of the first extraction is not extracted any more, and
        # a paragraph before the other one is
        yes_positions.update({"Termination": [1, 4], "Notices": [4]})
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(
                document, overwrite_existing=True
            )
        assert [item.value for item in aspect.extracted_items] == [
            paragraphs[1].raw_text,
            paragraphs[4].raw_text,
        ]
        # The reference paragraphs are those of the new items only
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraphs[1]),
            id(paragraphs[4]),
        ]
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraph)
            for item in aspect.extracted_items
            for paragraph in item.reference_paragraphs
        ]
        assert not aspect.reference_sentences
        # The sub-aspect is asked about on the new paragraphs only
        assert [request["input"] for request in decisions_stub.requests[5:]] == [
            _ASPECTS_PARAGRAPH_TEXTS_SENT[1],
            _ASPECTS_PARAGRAPH_TEXTS_SENT[4],
        ]
        (sub_aspect,) = aspect.aspects
        assert [item.value for item in sub_aspect.extracted_items] == [
            paragraphs[4].raw_text
        ]
        # ... and so is the concept of the aspect
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_aspect(aspect, document)
        (request,) = decisions_stub.requests
        assert request["input"] == "\n\n".join(
            [_ASPECTS_PARAGRAPH_TEXTS_SENT[1], _ASPECTS_PARAGRAPH_TEXTS_SENT[4]]
        )
        assert aspect.concepts[0].extracted_items[0].value is True

        # An aspect that no paragraph belongs to any more has no references
        yes_positions.update({"Termination": [], "Notices": []})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(
                document, overwrite_existing=True
            )
        assert aspect._is_processed
        assert not aspect.extracted_items
        assert not aspect.reference_paragraphs

        # === Without overwriting ===
        # The reference paragraphs that an aspect has are kept, and those of
        # its items are added to them, as before
        document = _create_aspects_document(
            aspects=[Aspect(name="Termination", description="Clauses on termination")]
        )
        (aspect,) = document.aspects
        paragraphs = document.paragraphs
        aspect.reference_paragraphs = [paragraphs[0], paragraphs[4]]
        yes_positions.update({"Termination": [2, 4]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert [id(paragraph) for paragraph in aspect.reference_paragraphs] == [
            id(paragraphs[0]),
            id(paragraphs[4]),
            id(paragraphs[2]),
        ]

    def test_model_group_stops_other_models_when_one_fails(self):
        """
        Tests that, with concurrency, the failure of one model of a model group
        stops the extraction with the other models of the group before the error
        is raised: no task is left running, no request is sent after the error,
        and the requests completed by then are counted in the usage.
        """

        def create_document() -> Document:
            """
            Creates a document with an aspect of the LLM and two aspects of
            the decision model.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(name="Liability", description="Clauses on liability"),
                    Aspect(
                        name="Termination",
                        description="Clauses on termination",
                        model_role="decide",
                    ),
                    Aspect(
                        name="Payment",
                        description="Clauses on payment",
                        model_role="decide",
                    ),
                ]
            )

        def get_running_tasks() -> list[asyncio.Task[Any]]:
            """
            Returns the tasks of the current event loop that are not done,
            other than the current one.

            :return: The tasks still running.
            :rtype: list[asyncio.Task[Any]]
            """
            current_task = asyncio.current_task()
            return [
                task
                for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            ]

        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)
        # The requests of the decision model that its limiter lets through:
        # those of one of its aspects, and two of those of the other one
        n_requests_let_through = n_paragraphs + 2

        # === The LLM fails while the decision model is extracting ===
        async def fail_llm() -> None:
            """
            Extracts the aspects with the group, with the request of the LLM
            failing when the decision model has extracted one of its aspects and
            has requests for the other one waiting for its limiter.
            """
            llm = _create_llm()
            decision_model = _create_decision_model(
                role="decide", async_limiter=AsyncLimiter(7, 3)
            )
            model_group = DocumentModelGroup(models=[llm, decision_model])
            document = create_document()
            liability, termination, payment = document.aspects
            llm_error = _create_api_error(litellm.AuthenticationError, model=_LLM_MODEL)
            release_held_requests = asyncio.Event()
            cancelled_acquisitions: list[int] = []
            n_acquisitions: list[int] = [0]

            class HoldingAsyncLimiter(AsyncLimiter):
                """
                Lets the requests of the LLM through, as well as the first requests
                of the decision model, and holds the other requests of the decision
                model until they are released.
                """

                async def acquire(self, amount: float = 1) -> None:
                    """
                    Acquires the capacity at once, or when the requests held
                    are released.

                    :param amount: The capacity to acquire.
                    :type amount: float
                    """
                    if (self.max_rate, self.time_period) != (7, 3):
                        # The limiter of the LLM
                        return
                    n_acquisitions[0] += 1
                    if n_acquisitions[0] > n_requests_let_through:
                        try:
                            await release_held_requests.wait()
                        except asyncio.CancelledError:
                            cancelled_acquisitions.append(n_acquisitions[0])
                            raise

            async def fail_when_decision_model_is_halfway(**kwargs: Any) -> Any:
                """
                Fails the request to the LLM when the decision model has extracted
                the first of its aspects.

                :param kwargs: Keyword arguments of ``litellm.acompletion``.
                :type kwargs: Any
                """
                for _ in range(1000):
                    if termination._is_processed:
                        break
                    await asyncio.sleep(0)
                assert termination._is_processed
                assert len(decisions_stub.requests) == n_requests_let_through
                raise llm_error

            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        _create_aspects_responder(
                            {"Termination": {2: 0.9}}, input_tokens=20
                        )
                    ]
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(
                    litellm, "acompletion", new=fail_when_decision_model_is_halfway
                ),
                patch.object(models_module, "AsyncLimiter", new=HoldingAsyncLimiter),
            ):
                with pytest.raises(ModelAPIError) as exc_info:
                    await model_group.extract_aspects_from_document_async(
                        document, use_concurrency=True
                    )
                assert exc_info.value.original_error is llm_error
                # The extraction with the decision model is stopped: no task is
                # left running, and the requests held by the limiter are cancelled
                assert not get_running_tasks()
                assert len(cancelled_acquisitions) == n_paragraphs - 2
                # ... i.e. they are not sent when the limiter would let them through
                release_held_requests.set()
                await asyncio.sleep(0.05)
                assert len(decisions_stub.requests) == n_requests_let_through

            # The aspect extracted by then is kept, and nothing else is extracted
            assert termination._is_processed
            assert [item.value for item in termination.extracted_items] == [
                document.paragraphs[2].raw_text
            ]
            assert not payment._is_processed and not payment.extracted_items
            assert not liability._is_processed and not liability.extracted_items
            # The requests completed by then are counted in the usage
            usage = decision_model.get_usage()[0].usage
            assert len(usage.calls) == n_requests_let_through
            assert all(call.timestamp_received is not None for call in usage.calls)
            assert usage.input == 20 * n_requests_let_through

        asyncio.run(fail_llm())

        # === The decision model fails while the LLM is extracting ===
        async def fail_decision_model() -> None:
            """
            Extracts the aspects with the group, with a request of the decision
            model failing when the request of the LLM is sent and not answered.
            """
            llm = _create_llm()
            decision_model = _create_decision_model(role="decide")
            model_group = DocumentModelGroup(models=[llm, decision_model])
            document = create_document()
            decision_model_error = _create_api_error(litellm.AuthenticationError)
            llm_request_sent = asyncio.Event()
            llm_requests: list[dict[str, Any]] = []
            cancelled_llm_requests: list[dict[str, Any]] = []

            async def never_respond(**kwargs: Any) -> Any:
                """
                Leaves the request to the LLM without a response.

                :param kwargs: Keyword arguments of ``litellm.acompletion``.
                :type kwargs: Any
                """
                llm_requests.append(kwargs)
                llm_request_sent.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    cancelled_llm_requests.append(kwargs)
                    raise

            async def fail_when_llm_request_is_sent(request: dict[str, Any]) -> Any:
                """
                Fails the request to the decision model when the request to
                the LLM is sent.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                """
                await asyncio.wait_for(llm_request_sent.wait(), timeout=10)
                raise decision_model_error

            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [fail_when_llm_request_is_sent]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(litellm, "acompletion", new=never_respond),
            ):
                with pytest.raises(ModelAPIError) as exc_info:
                    await model_group.extract_aspects_from_document_async(
                        document, use_concurrency=True
                    )
                assert exc_info.value.original_error is decision_model_error
                # The request of the LLM is cancelled, and no task is left running
                assert not get_running_tasks()
                assert len(llm_requests) == len(cancelled_llm_requests) == 1
                n_decision_requests = len(decisions_stub.requests)
                await asyncio.sleep(0.05)
                assert len(llm_requests) == 1
                assert len(decisions_stub.requests) == n_decision_requests
            assert not any(aspect._is_processed for aspect in document.aspects)
            assert not any(aspect.extracted_items for aspect in document.aspects)
            # A request that is in flight when it is cancelled is not recorded
            assert not llm.get_usage()[0].usage.calls

        asyncio.run(fail_decision_model())

        # === Without concurrency ===
        # The models extract one after another: the models after the one that
        # fails are not called
        llm = _create_llm()
        decision_model = _create_decision_model(role="decide")
        model_group = DocumentModelGroup(models=[llm, decision_model])
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: [_create_api_error(litellm.AuthenticationError, _LLM_MODEL)]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
            pytest.raises(ModelAPIError),
        ):
            model_group.extract_aspects_from_document(document)
        assert len(completions_stub.requests) == 1
        assert not decisions_stub.requests

    def test_decision_model_extracts_repeated_instances_once(self):
        """
        Tests that an aspect or a concept that is given to a decision model more
        than once is asked about, and processed, once, and that the usage of
        the requests made is not lost if an error is raised after them.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)

        # === A concept given twice ===
        for extraction_kwargs in [{}, {"use_concurrency": True}]:
            decision_model = _create_decision_model()
            document = _create_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The document is an NDA"
                    ),
                    BooleanConcept(
                        name="Lease check", description="The document is a lease"
                    ),
                ]
            )
            nda_concept, lease_concept = document.concepts
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [_answer_all_questions]}
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                extracted_concepts = decision_model.extract_concepts_from_document(
                    document,
                    from_concepts=[nda_concept, lease_concept, nda_concept],
                    **extraction_kwargs,
                )
            # One question for each distinct concept
            assert sorted(
                len(request["questions"]) for request in decisions_stub.requests
            ) == ([1, 1] if extraction_kwargs else [2])
            # The concepts returned are the distinct ones, in the order given
            assert [concept.name for concept in extracted_concepts] == [
                "NDA check",
                "Lease check",
            ]
            for concept in document.concepts:
                assert concept._is_processed
                assert [item.value for item in concept.extracted_items] == [True]
            usage = decision_model.get_usage()[0].usage
            assert len(usage.calls) == len(decisions_stub.requests)
            assert usage.input == 100 * len(decisions_stub.requests)

        # === An aspect given twice ===
        for extraction_kwargs in [{}, {"use_concurrency": True}]:
            decision_model = _create_decision_model()
            document = _create_aspects_document(
                aspects=[
                    Aspect(name="Termination", description="Clauses on termination")
                ]
            )
            (aspect,) = document.aspects
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        _create_aspects_responder({"Termination": {2: 0.9}})
                    ]
                }
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_aspects_from_document(
                    document,
                    from_aspects=[aspect, aspect],
                    **extraction_kwargs,
                )
            assert len(decisions_stub.requests) == n_paragraphs
            assert all(
                len(request["questions"]) == 1 for request in decisions_stub.requests
            )
            assert [item.value for item in aspect.extracted_items] == [
                document.paragraphs[2].raw_text
            ]

        # === A fallback decision model given a concept twice ===
        # The LLM is asked about the concept once, and so is the decision model
        # that takes over
        llm = _create_llm(fallback_models=[_create_decision_model(is_fallback=True)])
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA")
            ]
        )
        (concept,) = document.concepts
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        completions_stub = _CompletionsAPIStub(
            {_LLM_MODEL: [_create_api_error(litellm.AuthenticationError, _LLM_MODEL)]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            llm.extract_concepts_from_document(
                document, from_concepts=[concept, concept]
            )
        (llm_request,) = completions_stub.requests
        assert llm_request["messages"][-1]["content"].count("<concept_id>C") == 1
        (request,) = decisions_stub.requests
        assert [question["name"] for question in request["questions"]] == ["C0"]
        assert [item.value for item in concept.extracted_items] == [True]

        # === An error after the requests are made ===
        # The concept is processed by another call while the request is made:
        # the error is raised, with the request counted in the usage
        decision_model = _create_decision_model()
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA")
            ]
        )
        (concept,) = document.concepts

        def respond_and_process(request: dict[str, Any]) -> OpenAIDecisionResponse:
            """
            Responds to the request, with the concept set as processed
            in the meantime.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: OpenAIDecisionResponse
            """
            concept._is_processed = True
            return _answer_all_questions(request)

        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_and_process]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.raises(ValueError, match="have already been processed"),
        ):
            decision_model.extract_concepts_from_document(document)
        assert not concept.extracted_items
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 100
        (call,) = usage.calls
        assert call.timestamp_received is not None

    def test_decision_model_rejects_questions_over_litellm_limits(self):
        """
        Tests that a concept whose question litellm does not accept, e.g. for
        more options than a question may have, is rejected before any request is
        sent, as the other concepts that a decision model cannot extract, with
        the reason given by litellm, and that the check is skipped if the types
        of litellm cannot be used for it.
        """
        # A rating scale and labels over the limits of the pinned litellm version
        rating_concept = RatingConcept(
            name="Clarity",
            description="Clarity of the text of the document",
            rating_scale=(0, 10),
        )
        label_concept = LabelConcept(
            name="Clause type",
            description="Type of the clause",
            labels=[f"type {number}" for number in range(256)],
        )
        supported_concepts = [
            BooleanConcept(name="NDA check", description="The document is an NDA"),
            # The widest scale and the most labels that litellm accepts
            RatingConcept(
                name="Quality",
                description="Quality of the drafting",
                rating_scale=(1, 10),
            ),
            LabelConcept(
                name="Contract type",
                description="Type of the contract",
                labels=[f"type {number}" for number in range(255)],
            ),
            # Each label of such a concept is a question of its own
            LabelConcept(
                name="Topics",
                description="Topics of the document",
                labels=[f"topic {number}" for number in range(130)],
                classification_type="multi_label",
            ),
        ]
        rating_reason = (
            r"concept `Clarity` \(its `score` question with 11 options is not "
            r"accepted by litellm: `levels`: [^;)]*10[^;)]*\)"
        )
        label_reason = (
            r"concept `Clause type` \(its `choice` question with 256 options is "
            r"not accepted by litellm: `choices`: [^;)]*255[^;)]*\)"
        )
        requests: list[Any] = []

        async def record_request(*args: Any, **kwargs: Any) -> Any:
            """
            Records a request to a model API, which is not expected where
            this stub is used, and fails the test.

            :param args: Arguments of the request.
            :type args: Any
            :param kwargs: Keyword arguments of the request.
            :type kwargs: Any
            """
            requests.append(kwargs)
            raise pytest.fail.Exception("No request is expected to be sent.")

        # === A decision model used on its own, with and without a fallback LLM ===
        for decision_model in [
            _create_decision_model(),
            # The concepts are not passed on to the fallback model silently
            _create_decision_model(fallback_models=[_create_llm(is_fallback=True)]),
        ]:
            document = _create_document(
                concepts=[*supported_concepts, rating_concept, label_concept]
            )
            for extraction_method in [
                decision_model.extract_concepts_from_document,
                decision_model.extract_all,
            ]:
                for use_concurrency in [False, True]:
                    with (
                        patch.object(litellm, "adecisions", new=record_request),
                        patch.object(litellm, "acompletion", new=record_request),
                        pytest.raises(
                            ValueError,
                            match=f"^Decision model `{_DECISION_MODEL}` cannot "
                            f"extract: {rating_reason}; {label_reason}"
                            r"\. A decision model extracts",
                        ),
                    ):
                        extraction_method(document, use_concurrency=use_concurrency)
            assert not requests
            assert not any(concept._is_processed for concept in document.concepts)
            assert not decision_model.get_usage()[0].usage.calls

        # A concept of an aspect, and a decision model that is a fallback model
        document = _create_document(
            aspects=[
                Aspect(
                    name="Termination",
                    description="Clauses on termination",
                    concepts=[rating_concept],
                )
            ]
        )
        with (
            patch.object(litellm, "adecisions", new=record_request),
            pytest.raises(ValueError, match=f"cannot extract: {rating_reason}"),
        ):
            _create_decision_model().extract_all(document)
        llm = _create_llm(fallback_models=[_create_decision_model(is_fallback=True)])
        with (
            patch.object(litellm, "adecisions", new=record_request),
            patch.object(litellm, "acompletion", new=record_request),
            pytest.raises(
                ValueError,
                match=f"^Fallback decision model `{_DECISION_MODEL}` cannot extract "
                "what it is to extract if the model it is a fallback for fails: "
                f"{rating_reason}",
            ),
        ):
            llm.extract_all(document)
        assert not requests

        # === The concepts within the limits ===
        decision_model = _create_decision_model()
        document = _create_document(concepts=supported_concepts)
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(document)
        assert all(concept._is_processed for concept in document.concepts)
        # The questions over the cap of a request are sent in another request
        assert [len(request["questions"]) for request in decisions_stub.requests] == [
            MAX_DECISION_QUESTIONS,
            3 + 130 - MAX_DECISION_QUESTIONS,
        ]

        # === The types of litellm cannot be used for the check ===
        # The questions are not checked before the request, as before: a question
        # that litellm does not accept is rejected by it when the request is made
        class UnusableQuestionType:
            """
            Type of a question that cannot validate a question.
            """

            @staticmethod
            def model_validate(question: Any) -> Any:
                """
                Fails with an error that is not a validation error.

                :param question: The question to validate.
                :type question: Any
                """
                raise RuntimeError("Simulated change of litellm")

        for question_types in [
            # The types are not there
            types.SimpleNamespace(MAX_DECISION_QUESTIONS=MAX_DECISION_QUESTIONS),
            # ... or they do not validate a question as expected
            types.SimpleNamespace(
                MAX_DECISION_QUESTIONS=MAX_DECISION_QUESTIONS,
                OpenAIPredicateQuestion=UnusableQuestionType,
                OpenAIChoiceQuestion=UnusableQuestionType,
                OpenAIScoreQuestion=object(),
            ),
        ]:
            decision_model = _create_decision_model()
            document = _create_document(concepts=[rating_concept, label_concept])
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [_answer_all_questions]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(
                    decision_models_module,
                    "_litellm_decision_types",
                    new=question_types,
                ),
            ):
                decision_model.extract_concepts_from_document(document)
            (request,) = decisions_stub.requests
            assert len(request["questions"][0]["levels"]) == 11
            assert len(request["questions"][1]["choices"]) == 256

    def test_decision_model_provider_validation(self):
        """
        Tests that a decision model of a provider that litellm serves no decision
        models of is rejected when the model is created or its identifier is
        assigned, with the providers that litellm supports, and that the provider
        is not checked if the providers cannot be read from litellm.
        """
        # The providers, as litellm lists them itself
        supported_providers = list(
            sys.modules["litellm.decisions.call"].supported_decisions_providers()
        )
        assert {"typesafe", "openai", "azure_ai"} <= set(supported_providers)
        assert decision_models_module._get_supported_decision_providers() == tuple(
            supported_providers
        )

        # Each of the providers is accepted
        for provider in supported_providers:
            assert DocumentDecisionModel(model=f"{provider}/decider").model == (
                f"{provider}/decider"
            )

        # A provider of LLMs only, including those of local LLMs
        for model in [
            "anthropic/claude-x",
            "ollama/nimble",
            "ollama_chat/nimble",
            "lm_studio/nimble",
            # The provider is the prefix of the model identifier
            "jev/typesafe",
        ]:
            provider = model.partition("/")[0]
            error_match = (
                f"Provider `{provider}` of model `{model}` is not a provider of "
                f"decision models supported by litellm. Supported providers: "
                f"{', '.join(supported_providers)}."
            )
            with pytest.raises(ValueError, match=re.escape(error_match)) as exc_info:
                DocumentDecisionModel(model=model, api_key="test_api_key")
            # The error is the one of the model identifier, i.e. it does not
            # show the other params that the model is created with
            assert "test_api_key" not in str(exc_info.value)
            # A prefix of the LLMs of a local server: the error names the prefix
            # that a decision model of a local server is reached under
            local_server_hint = (
                "For a decision model served by a local server, e.g. by Ollama or "
                "LM Studio, use the `strands_decider/` prefix, i.e. "
                "`strands_decider/nimble`, and set `api_base` to the address of "
                "the server."
            )
            is_local_llm_prefix = provider in ["ollama", "ollama_chat", "lm_studio"]
            assert (local_server_hint in str(exc_info.value)) is is_local_llm_prefix
            # ... also when the model identifier is assigned, with the model
            # left unchanged
            decision_model = _create_decision_model()
            with pytest.raises(ValueError, match=re.escape(error_match)):
                decision_model.model = model
            assert decision_model.model == _DECISION_MODEL
        # The format of the identifier is checked first
        with pytest.raises(ValueError, match="<model_provider>/<model_name>"):
            DocumentDecisionModel(model="nimble")

        # === The providers cannot be read from litellm ===
        def fail_to_list() -> tuple[str, ...]:
            """
            Fails as a function of litellm that cannot list the providers.

            :return: Nothing: an error is raised.
            :rtype: tuple[str, ...]
            """
            raise RuntimeError("The providers cannot be listed.")

        # The provider is not checked: a model that litellm cannot serve fails
        # at its first request
        for decisions_module in [
            # The module is not there, or cannot be imported
            None,
            # The function that returns the providers is not where it is expected,
            # is not a function, fails, or returns what are not providers
            types.SimpleNamespace(),
            types.SimpleNamespace(supported_decisions_providers=None),
            types.SimpleNamespace(supported_decisions_providers=("typesafe",)),
            types.SimpleNamespace(supported_decisions_providers=fail_to_list),
            types.SimpleNamespace(supported_decisions_providers=lambda: None),
            types.SimpleNamespace(supported_decisions_providers=lambda: ()),
            types.SimpleNamespace(
                supported_decisions_providers=lambda: (1, "typesafe")
            ),
            types.SimpleNamespace(
                supported_decisions_providers=lambda: ("typesafe", "")
            ),
        ]:
            with patch.dict(sys.modules, {"litellm.decisions.call": decisions_module}):
                assert decision_models_module._get_supported_decision_providers() is (
                    None
                )
                decision_model = DocumentDecisionModel(model="ollama/nimble")
                decision_model.model = "anthropic/claude-x"
            assert decision_model.model == "anthropic/claude-x"

    def test_decision_model_call_grouping(self):
        """
        Tests that the aspects and concepts that differ only in the extraction
        params that a decision model is not sent are extracted in the same calls
        of a decision model, while an LLM is still called for each group of them.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)

        def create_document() -> Document:
            """
            Creates a document with aspects and concepts that differ in the depth
            and the length of justifications and in the depth of references,
            none of which is requested.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                aspects=[
                    Aspect(name="Termination", description="Clauses on termination"),
                    Aspect(
                        name="Liability",
                        description="Clauses on liability",
                        justification_depth="balanced",
                    ),
                    Aspect(
                        name="Payment",
                        description="Clauses on payment",
                        justification_max_sents=5,
                    ),
                ],
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The document is an NDA"
                    ),
                    BooleanConcept(
                        name="Lease check",
                        description="The document is a lease",
                        reference_depth="sentences",
                    ),
                    RatingConcept(
                        name="Clarity",
                        description="Clarity of the text of the document",
                        rating_scale=(1, 3),
                        justification_depth="comprehensive",
                    ),
                    LabelConcept(
                        name="Notice form",
                        description="Required form of the termination notice",
                        labels=["written", "any"],
                        justification_max_sents=4,
                    ),
                ],
            )

        # === A decision model ===
        # Each paragraph is sent once, with the questions of all the aspects,
        # and the text of the document once, with the questions of all the concepts
        decision_model = _create_decision_model()
        document = create_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [_answer_all_questions]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_all(document)
        assert [request["input"] for request in decisions_stub.requests[:-1]] == (
            _ASPECTS_PARAGRAPH_TEXTS_SENT
        )
        assert [
            [question["name"] for question in request["questions"]]
            for request in decisions_stub.requests
        ] == [["A0", "A1", "A2"]] * n_paragraphs + [["C0", "C1", "C2", "C3"]]
        assert all(aspect._is_processed for aspect in document.aspects)
        assert all(concept._is_processed for concept in document.concepts)
        # ... also with the number of the items of a call set
        document = create_document()
        decisions_stub.requests.clear()
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_all(
                document, use_concurrency=True, max_items_per_call=4
            )
        assert sorted(
            len(request["questions"]) for request in decisions_stub.requests
        ) == [3] * n_paragraphs + [4]

        # === An LLM ===
        # The aspects and concepts are extracted in the same calls as before:
        # each group of them, by all the extraction params, in a call of its own
        llm = _create_llm()
        document = create_document()
        prompts: list[str] = []

        async def respond_to_prompt(**kwargs: Any) -> Any:
            """
            Responds to the request to the LLM with no items extracted.

            :param kwargs: Keyword arguments of ``litellm.acompletion``.
            :type kwargs: Any
            :return: Completion object, with the attributes that the framework reads.
            :rtype: Any
            """
            prompt = kwargs["messages"][-1]["content"]
            prompts.append(prompt)
            if "aspect_id" in prompt:
                content = '[{"aspect_id": "A0", "paragraph_ids": []}]'
            else:
                content = '[{"concept_id": "C0", "extracted_items": []}]'
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        with patch.object(litellm, "acompletion", new=respond_to_prompt):
            llm.extract_all(document)
        # One call for each of the three aspects and of the four concepts
        assert len(prompts) == 7

        # === A decision model with a fallback LLM ===
        # The fallback LLM takes over the calls of the decision model as they
        # are, i.e. it is given all the concepts in one call: none of the params
        # that they differ in changes its prompt, as neither justifications nor
        # references are requested
        decision_model = _create_decision_model(
            fallback_models=[_create_llm(is_fallback=True)]
        )
        document = create_document()
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [_create_api_error(litellm.AuthenticationError)]}
        )
        completions_stub = _CompletionsAPIStub(
            {
                _LLM_MODEL: [
                    json.dumps(
                        [
                            {"concept_id": "C0", "extracted_items": [True]},
                            {"concept_id": "C1", "extracted_items": [False]},
                            {"concept_id": "C2", "extracted_items": [3]},
                            {
                                "concept_id": "C3",
                                "extracted_items": [{"labels": ["any"]}],
                            },
                        ]
                    )
                ]
            }
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
        ):
            decision_model.extract_concepts_from_document(document)
        assert len(decisions_stub.requests) == len(completions_stub.requests) == 1
        assert [concept.extracted_items[0].value for concept in document.concepts] == [
            True,
            False,
            3,
            ["any"],
        ]

    def test_decision_model_api_key_not_shown(self, tmp_path: Path):
        """
        Tests that the API key of a decision model is not shown when the model
        is printed, while its API base is, and that no part of the key is shown
        in the text of a validation error of a decision model, or of an LLM or
        a model group that the decision model is passed to, which does not show
        the inputs of the validation, nor given by the ``errors()`` and ``json()``
        methods of the error: neither for a decision model that is among
        the inputs, nor for a key that is among them as it is passed, which is
        reported as redacted.
        """
        api_key = "sk-test-SECRET-0123456789"
        api_base = "https://decisions.example.com"
        decision_model = DocumentDecisionModel(
            model=_DECISION_MODEL,
            api_key=api_key,
            api_base=api_base,
            fallback_models=[
                DocumentLLM(model=_LLM_MODEL, api_key=api_key, is_fallback=True)
            ],
        )
        # The key is set, and is sent with the requests
        assert decision_model.api_key == api_key
        for text in [repr(decision_model), str(decision_model), f"{decision_model}"]:
            assert "SECRET" not in text
            assert "api_key" not in text
            assert f"model='{_DECISION_MODEL}'" in text
            assert f"api_base='{api_base}'" in text

        def check_error(error: Exception, expected_texts: list[str]) -> None:
            """
            Checks that the validation error does not show the key, neither in
            its text, which shows no inputs of the validation, nor in the inputs
            that it reports, which the decision model is among: when what its
            ``errors()`` method returns is printed, and in what its ``json()``
            method writes out.

            :param error: The validation error.
            :type error: Exception
            :param expected_texts: The texts that the error is to have.
            :type expected_texts: list[str]
            """
            assert isinstance(error, ValidationError)
            self.check_validation_error_hides_inputs(error, api_key, expected_texts)
            assert "DocumentDecisionModel(" in repr(error.errors())
            self.check_validation_error_omits_model_keys(error, api_key)

        not_fallback_error = ["fallback_models", "must be a fallback model"]
        different_roles_error = ["models", "must have different roles"]

        # The model as an invalid item of the fallback models of another model,
        # an LLM or a decision model: it is not a fallback model
        with pytest.raises(ValidationError) as exc_info:
            DocumentLLM(
                model=_LLM_MODEL, api_key=api_key, fallback_models=[decision_model]
            )
        check_error(exc_info.value, not_fallback_error)
        # The `errors()` method of the error returns the model as it is, i.e. with
        # the key as its attribute, while its `json()` method writes the model out
        # with all its params but the key
        (error_details,) = exc_info.value.errors()
        assert error_details["input"][0] is decision_model
        (written_error_details,) = json.loads(exc_info.value.json())
        (written_model,) = written_error_details["input"]
        assert list(written_model) == [
            param for param in DocumentDecisionModel.model_fields if param != "api_key"
        ]
        assert written_model["model"] == _DECISION_MODEL
        assert written_model["api_base"] == api_base
        # ... as is its fallback LLM
        (written_fallback_llm,) = written_model["fallback_models"]
        assert written_fallback_llm["model"] == _LLM_MODEL
        assert "api_key" not in written_fallback_llm
        # ... unless the inputs are left out
        (written_error_details,) = json.loads(exc_info.value.json(include_input=False))
        assert "input" not in written_error_details
        with pytest.raises(ValidationError) as exc_info:
            DocumentDecisionModel(
                model=_DECISION_MODEL,
                api_key=api_key,
                fallback_models=[decision_model],
            )
        check_error(exc_info.value, not_fallback_error)
        # ... and as an invalid model of a group: with the role of another model,
        # an LLM or a decision model
        other_decision_model = DocumentDecisionModel(
            model=_DECISION_MODEL, api_key=api_key, role="other"
        )
        model_group = DocumentModelGroup(models=[decision_model, other_decision_model])
        with pytest.raises(ValidationError) as exc_info:
            DocumentModelGroup(
                models=[decision_model, DocumentLLM(model=_LLM_MODEL, api_key=api_key)]
            )
        check_error(exc_info.value, different_roles_error)
        with pytest.raises(ValidationError) as exc_info:
            DocumentModelGroup(models=[other_decision_model, other_decision_model])
        check_error(exc_info.value, different_roles_error)
        with pytest.raises(ValidationError) as exc_info:
            model_group.models = [other_decision_model, other_decision_model]
        check_error(exc_info.value, different_roles_error)
        assert model_group.models == [decision_model, other_decision_model]

        # The model as the input of its own check that involves several params,
        # which it is when a param is assigned: a fallback model cannot have
        # fallback models of its own
        with pytest.raises(ValidationError) as exc_info:
            decision_model.is_fallback = True
        check_error(exc_info.value, ["cannot have its own fallback"])
        assert not decision_model.is_fallback

        # A fallback LLM that is not valid for the decision model: with another
        # role. The input of this check is all the params that the decision model
        # is created with, i.e. the key, which is reported as redacted, whether
        # it is the first or the last of them.
        fallback_llm = DocumentLLM(
            model=_LLM_MODEL, api_key=api_key, is_fallback=True, role="other"
        )
        invalid_model_params: list[dict[str, Any]] = [
            {
                "api_key": api_key,
                "model": _DECISION_MODEL,
                "fallback_models": [fallback_llm],
            },
            {
                "model": _DECISION_MODEL,
                "fallback_models": [fallback_llm],
                "api_key": api_key,
            },
        ]
        for model_params in invalid_model_params:
            with pytest.raises(ValidationError) as exc_info:
                DocumentDecisionModel(**model_params)
            self.check_validation_error_hides_inputs(
                exc_info.value,
                api_key,
                ["The fallback LLM must have the same role `None` as the main one"],
            )
            self.check_validation_error_omits_model_keys(exc_info.value, api_key)
            # The other params are reported as they are passed, in their order:
            # the fallback LLM is reported as it is, i.e. holds its key
            (error_details,) = exc_info.value.errors()
            assert error_details["input"] == {**model_params, "api_key": REDACTED_VALUE}
            assert list(error_details["input"]) == list(model_params)
            assert error_details["input"]["fallback_models"][0] is fallback_llm
            assert fallback_llm.api_key == api_key
            # The rest of the error is as it is raised by the validation
            assert error_details["type"] == "value_error"
            assert error_details["loc"] == ()
            assert error_details["msg"] == (
                "Value error, The fallback LLM must have the same role `None` "
                "as the main one."
            )
            assert isinstance(error_details["ctx"]["error"], ValueError)
            assert error_details["url"].endswith("/v/value_error")
            assert exc_info.value.title == "DocumentDecisionModel"
            assert exc_info.value.error_count() == 1

        # The decision model that is invalid itself, when it is created, changed or
        # deserialized: the key is among the inputs of its validation. The input
        # of a check that involves several params is all the params that
        # the decision model is created with, and the model when a param is assigned.
        self.check_model_validation_errors_hide_inputs(
            DocumentDecisionModel,
            {"model": _DECISION_MODEL},
            api_key,
            tmp_path=tmp_path,
        )

    def test_decision_model_image_concepts_rejected(self):
        """
        Tests that a concept extracted from images, which a decision model cannot
        extract, is rejected for a document that has images as it is for
        a document without them, for a decision model used on its own and as
        a fallback model of an LLM that supports vision: vision is a capability
        of LLMs, and is not checked for a decision model.
        """
        requests: list[Any] = []

        async def record_request(*args: Any, **kwargs: Any) -> Any:
            """
            Records a request to a model API, which is not expected where
            this stub is used, and fails the test.

            :param args: Arguments of the request.
            :type args: Any
            :param kwargs: Keyword arguments of the request.
            :type kwargs: Any
            """
            requests.append(kwargs)
            raise pytest.fail.Exception("No request is expected to be sent.")

        llm = _create_llm(fallback_models=[_create_decision_model(is_fallback=True)])
        # The LLM can extract from images, whatever litellm tells of its model
        llm.supports_vision = True
        modalities: list[Literal["image", "multimodal"]] = ["image", "multimodal"]
        for modality in modalities:
            document = Document(
                raw_text=_DOCUMENT_RAW_TEXT, images=[self.test_img_png_invoice]
            )
            document.concepts = [
                BooleanConcept(
                    name="Signed", description="Is signed", modality=modality
                )
            ]
            for model, error_match in [
                (
                    _create_decision_model(),
                    (
                        f"^Decision model `{_DECISION_MODEL}` cannot extract: concept "
                        rf"`Signed` \(modality `{modality}` is not supported\)\. "
                    ),
                ),
                (
                    llm,
                    (
                        f"^Fallback decision model `{_DECISION_MODEL}` cannot extract "
                        "what it is to extract if the model it is a fallback for "
                        rf"fails: concept `Signed` \(modality `{modality}` is not "
                        r"supported\)\. "
                    ),
                ),
            ]:
                for extraction_method in [
                    model.extract_concepts_from_document,
                    model.extract_all,
                ]:
                    with (
                        patch.object(litellm, "adecisions", new=record_request),
                        patch.object(litellm, "acompletion", new=record_request),
                        pytest.raises(ValueError, match=error_match),
                    ):
                        extraction_method(document)
            assert not requests
            assert not document.concepts[0]._is_processed

    def test_fallback_decision_model_paced_by_own_limiter(self):
        """
        Tests that, with concurrency, the requests of a fallback decision model
        for the paragraphs are sent concurrently and paced by the limiter of
        that fallback model, as the requests of the primary model are by its own.
        """
        n_paragraphs = len(_ASPECTS_PARAGRAPH_TEXTS_SENT)
        acquisitions: list[tuple[float, float]] = []
        requests_in_progress: list[str] = []
        max_requests_in_progress: list[int] = [0]

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the config of the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                acquisitions.append((self.max_rate, self.time_period))
                await super().acquire(amount)

        respond = _create_aspects_responder({"Termination": {2: 0.9, 4: 0.7}})

        async def respond_after_other_requests(request: dict[str, Any]) -> Any:
            """
            Responds to the request to the fallback model, after the other
            requests sent concurrently are in progress.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: Any
            """
            requests_in_progress.append(request["input"])
            max_requests_in_progress[0] = max(
                max_requests_in_progress[0], len(requests_in_progress)
            )
            for _ in range(3):
                await asyncio.sleep(0)
            requests_in_progress.remove(request["input"])
            return respond(request)

        for primary_model in [
            _create_decision_model(async_limiter=AsyncLimiter(20, 5)),
            _create_llm(async_limiter=AsyncLimiter(20, 5)),
        ]:
            primary_model.fallback_models = [
                _create_decision_model(
                    _OTHER_DECISION_MODEL,
                    is_fallback=True,
                    async_limiter=AsyncLimiter(7, 3),
                )
            ]
            document = _create_aspects_document(
                aspects=[
                    Aspect(name="Termination", description="Clauses on termination")
                ]
            )
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [_create_api_error(litellm.AuthenticationError)],
                    _OTHER_DECISION_MODEL: [respond_after_other_requests],
                }
            )
            completions_stub = _CompletionsAPIStub(
                {
                    _LLM_MODEL: [
                        _create_api_error(litellm.AuthenticationError, _LLM_MODEL)
                    ]
                }
            )
            acquisitions.clear()
            max_requests_in_progress[0] = 0
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(litellm, "acompletion", new=completions_stub),
                patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
            ):
                primary_model.extract_aspects_from_document(
                    document, use_concurrency=True
                )
            # Each request to the fallback model is paced by the limiter of
            # the fallback model, not by the one of the primary model
            assert acquisitions.count((7, 3)) == n_paragraphs
            assert set(acquisitions) == {(20, 5), (7, 3)}
            assert acquisitions[-n_paragraphs:] == [(7, 3)] * n_paragraphs
            # ... and the requests are sent concurrently
            assert max_requests_in_progress[0] == n_paragraphs
            (aspect,) = document.aspects
            assert [item.value for item in aspect.extracted_items] == [
                document.paragraphs[2].raw_text,
                document.paragraphs[4].raw_text,
            ]

    def test_decision_model_response_with_invalid_token_counts(self):
        """
        Tests that a response that reports a number of tokens that cannot be
        counted, such as a negative one, is not recorded as a response: its request
        fails, with an unfinished call, and nothing is added to the usage.
        """
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA")
            ]
        )

        def create_responder(
            input_tokens: Any, output_tokens: Any
        ) -> Callable[[dict[str, Any]], Any]:
            """
            Creates the function that responds to a request with a response
            that reports the given numbers of tokens.

            :param input_tokens: The number of input tokens to report.
            :type input_tokens: Any
            :param output_tokens: The number of output tokens to report.
            :type output_tokens: Any
            :return: Function that creates the response for a request.
            :rtype: Callable[[dict[str, Any]], Any]
            """

            def respond(request: dict[str, Any]) -> Any:
                """
                Creates the response for the request.

                :param request: Keyword arguments of the request.
                :type request: dict[str, Any]
                :return: Response, with the attributes that the framework reads.
                :rtype: Any
                """
                response = _answer_all_questions(request)
                return types.SimpleNamespace(
                    model=response.model,
                    answers=response.answers,
                    usage=types.SimpleNamespace(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        input_tokens_details=None,
                    ),
                )

            return respond

        for input_tokens, output_tokens in [
            (-1, 0),
            (10, -5),
            (None, 0),
            (10, None),
            ("12", 0),
            (10.5, 0),
        ]:
            decision_model = _create_decision_model(
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.05, output_per_1m_tokens=0.0
                )
            )
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [create_responder(input_tokens, output_tokens)]}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                pytest.raises(ModelAPIError) as exc_info,
            ):
                decision_model.extract_concepts_from_document(document)
            assert isinstance(exc_info.value.original_error, ValueError)
            assert "reports an invalid number of tokens" in str(
                exc_info.value.original_error
            )
            assert not document.concepts[0]._is_processed
            # The request is recorded as a failed one, with no tokens
            usage = decision_model.get_usage()[0].usage
            (call,) = usage.calls
            assert call.timestamp_received is None
            assert call.response is None
            assert (usage.input, usage.output) == (0, 0)
            assert decision_model.get_cost()[0].cost.total == 0

        # The numbers of tokens that can be counted, including zero
        decision_model = _create_decision_model()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [create_responder(0, 0)]})
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(document)
        usage = decision_model.get_usage()[0].usage
        assert usage.calls[0].timestamp_received is not None
        assert (usage.input, usage.output) == (0, 0)
        assert document.concepts[0].extracted_items[0].value is True

    def _create_concepts_for_recorded_tests(self) -> list[Any]:
        """
        Creates the concepts extracted in the tests that send their requests to
        the API of a decision model: one concept of each kind of question, with
        the answers that are unambiguous for the text they are extracted from
        (see ``_RECORDED_CONCEPTS_DOCUMENT_TEXT``).

        :return: The concepts.
        :rtype: list[Any]
        """
        return [
            BooleanConcept(
                name="NDA check",
                description="The document is a non-disclosure agreement",
            ),
            LabelConcept(
                name="Document type",
                description="Type of the document",
                labels=[
                    "non-disclosure agreement",
                    "employment agreement",
                    "lease agreement",
                    "invoice",
                ],
            ),
            LabelConcept(
                name="Topics",
                description="Topics that the document covers",
                labels=["confidentiality", "governing law", "salary", "real estate"],
                classification_type="multi_label",
            ),
            RatingConcept(
                name="Sensitivity",
                description="Sensitivity of the confidential information, as rated "
                "by the parties in the document",
                rating_scale=(1, 5),
            ),
        ]

    def _check_concepts_of_recorded_tests(
        self, document: Document, decision_model: DocumentDecisionModel
    ) -> None:
        """
        Checks the items of the concepts created by
        ``_create_concepts_for_recorded_tests()``, as extracted by a decision model
        through its API, and the usage of the model.

        The values are checked, as the answers are unambiguous. The probabilities
        are checked for having the options of the concept as their keys, for being
        within [0, 1], and for being consistent with the value: their exact values
        are up to the model.

        :param document: The document with the extracted concepts.
        :type document: Document
        :param decision_model: The decision model that extracted the concepts.
        :type decision_model: DocumentDecisionModel
        """
        boolean_concept, label_concept, multi_label_concept, rating_concept = (
            document.concepts
        )
        for concept in document.concepts:
            assert concept._is_processed
            assert len(concept.extracted_items) == 1
            probabilities = concept.extracted_items[0].probabilities
            assert probabilities is not None
            assert all(
                type(probability) is float and 0 <= probability <= 1
                for probability in probabilities.values()
            )

        # Boolean concept
        (item,) = boolean_concept.extracted_items
        assert item.value is True
        assert item.probabilities is not None
        assert list(item.probabilities) == [True, False]
        assert item.probabilities[True] >= 0.5
        assert item.probabilities[False] == pytest.approx(1 - item.probabilities[True])

        # Label concept, a single label to select: the most probable label
        (item,) = label_concept.extracted_items
        assert item.value == ["non-disclosure agreement"]
        assert item.probabilities is not None
        assert list(item.probabilities) == [
            "non-disclosure agreement",
            "employment agreement",
            "lease agreement",
            "invoice",
        ]
        assert item.probabilities["non-disclosure agreement"] == max(
            item.probabilities.values()
        )

        # Label concept, any number of labels to select: the labels with
        # a probability of 0.5 or more
        (item,) = multi_label_concept.extracted_items
        assert item.probabilities is not None
        assert list(item.probabilities) == [
            "confidentiality",
            "governing law",
            "salary",
            "real estate",
        ]
        assert item.value == [
            label
            for label, probability in item.probabilities.items()
            if probability >= 0.5
        ]
        assert "confidentiality" in item.value
        assert "salary" not in item.value
        assert "real estate" not in item.value

        # Rating concept: the most probable level of the scale
        (item,) = rating_concept.extracted_items
        assert item.value == 4
        assert item.probabilities is not None
        assert list(item.probabilities) == [1, 2, 3, 4, 5]
        assert item.probabilities[4] == max(item.probabilities.values())

        # One request for all the concepts
        (usage_container,) = decision_model.get_usage()
        assert usage_container.usage.input > 0
        (call,) = usage_container.usage.calls
        assert call.timestamp_received is not None
        assert call.response is not None
        assert [answer["name"] for answer in json.loads(call.response)] == [
            "C0",
            "C1",
            "C2-L0",
            "C2-L1",
            "C2-L2",
            "C2-L3",
            "C3",
        ]
        self._check_cost_of_recorded_tests(decision_model)
        self.check_instance_serialization_and_cloning(document)

    @staticmethod
    def _check_cost_of_recorded_tests(decision_model: DocumentDecisionModel) -> None:
        """
        Checks that the cost of the decision model is the cost of its input tokens,
        as reported by its API, at the price set for the model, or at the price
        that the provider lists for it if its costs are calculated automatically,
        or zero if no price is set for the model.

        :param decision_model: The decision model.
        :type decision_model: DocumentDecisionModel
        """
        usage = decision_model.get_usage()[0].usage
        cost = decision_model.get_cost()[0].cost
        if decision_model.auto_pricing:
            input_price = Decimal(
                _RECORDED_AUTO_PRICED_INPUT_PRICES[decision_model.model]
            )
        elif decision_model.pricing_details is None:
            assert (cost.input, cost.output, cost.total) == (0, 0, 0)
            return
        else:
            input_price = Decimal(
                str(decision_model.pricing_details.input_per_1m_tokens)
            )
        expected_cost = (
            Decimal(usage.input) * input_price / Decimal(1_000_000)
        ).quantize(_COST_QUANT, rounding=ROUND_HALF_UP)
        assert cost.input == expected_cost
        assert cost.output == 0
        assert cost.total == expected_cost

    @pytest.mark.vcr
    def test_extract_concepts_with_typesafe_api(self):
        """
        Tests the extraction of concepts of each supported type from a document
        with a decision model of TypeSafe, with the requests sent to its API.
        """
        decision_model = DocumentDecisionModel(
            model="typesafe/jev-latest",
            api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = self._create_concepts_for_recorded_tests()

        decision_model.extract_concepts_from_document(document)

        self._check_concepts_of_recorded_tests(document, decision_model)

    @pytest.mark.vcr
    def test_extract_all_with_typesafe_api(self):
        """
        Tests the extraction of aspects, and of the concepts of the aspects, from
        a document with a decision model of TypeSafe, with the requests sent to
        its API, one after another.
        """
        decision_model = DocumentDecisionModel(
            model="typesafe/jev-latest",
            api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
        )
        paragraph_texts = _RECORDED_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS
        document = Document(raw_text="\n\n".join(paragraph_texts))
        assert [paragraph.raw_text for paragraph in document.paragraphs] == (
            paragraph_texts
        )
        document.aspects = [
            Aspect(
                name="Payment",
                description="Clauses on the fees and on how the fees are paid",
                concepts=[
                    BooleanConcept(
                        name="Monthly fee",
                        description="The fee is paid every month",
                    )
                ],
            ),
            Aspect(
                name="Termination",
                description="Clauses on how the agreement is terminated",
                concepts=[
                    LabelConcept(
                        name="Notice form",
                        description="Form in which the notice of termination "
                        "is to be given",
                        labels=["written", "oral", "not specified"],
                    )
                ],
            ),
        ]

        # Without concurrency, i.e. with the requests in the order of the paragraphs
        decision_model.extract_all(document)

        # === The aspects ===
        payment, termination = document.aspects
        fee_text, invoices_text, termination_text = paragraph_texts[3:6]
        assert [item.value for item in payment.extracted_items] == [
            fee_text,
            invoices_text,
        ]
        assert [item.value for item in termination.extracted_items] == [
            termination_text
        ]
        for aspect in document.aspects:
            assert aspect._is_processed
            for item in aspect.extracted_items:
                # The paragraph that the item is extracted for
                (paragraph,) = item.reference_paragraphs
                assert paragraph in document.paragraphs
                assert item.value == paragraph.raw_text
                # Whether the paragraph belongs to the aspect: it does
                assert item.probabilities is not None
                assert list(item.probabilities) == [True, False]
                assert 0.5 <= item.probabilities[True] <= 1
                assert item.probabilities[False] == pytest.approx(
                    1 - item.probabilities[True]
                )
                assert item.justification is None
            assert aspect.reference_paragraphs == [
                item.reference_paragraphs[0] for item in aspect.extracted_items
            ]

        # === The concepts of the aspects ===
        (item,) = payment.concepts[0].extracted_items
        assert item.value is True
        assert item.probabilities is not None
        assert list(item.probabilities) == [True, False]
        assert item.probabilities[True] >= 0.5
        (item,) = termination.concepts[0].extracted_items
        assert item.value == ["written"]
        assert item.probabilities is not None
        assert list(item.probabilities) == ["written", "oral", "not specified"]
        assert item.probabilities["written"] == max(item.probabilities.values())
        assert all(0 <= probability <= 1 for probability in item.probabilities.values())

        # === The usage ===
        # One request for each paragraph, with the questions of both aspects, and
        # one request for the concepts of each aspect
        (usage_container,) = decision_model.get_usage()
        usage = usage_container.usage
        assert usage.input > 0
        assert len(usage.calls) == len(paragraph_texts) + 2
        assert [
            json.loads(call.prompt)["input"]
            for call in usage.calls[: len(paragraph_texts)]
        ] == paragraph_texts
        assert all(call.timestamp_received is not None for call in usage.calls)
        self._check_cost_of_recorded_tests(decision_model)
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.vcr
    def test_model_group_with_typesafe_api(self):
        """
        Tests a model group with an LLM and a decision model of TypeSafe, with
        the requests sent to the APIs of both: each concept is extracted by
        the model that it is routed to by the role.
        """
        llm = DocumentLLM(**self._llm_extractor_kwargs_openai)
        decision_model = DocumentDecisionModel(
            model="typesafe/jev-latest",
            api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
            role="decide",
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
        )
        model_group = DocumentModelGroup(models=[llm, decision_model])
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = [
            # Extracted by the LLM, which has no role
            StringConcept(
                name="Governing law",
                description="Country whose laws govern the agreement",
            ),
            # Extracted by the decision model
            BooleanConcept(
                name="NDA check",
                description="The document is a non-disclosure agreement",
                model_role="decide",
            ),
        ]

        model_group.extract_concepts_from_document(document)

        string_concept, boolean_concept = document.concepts
        # An item extracted by an LLM has no probabilities
        assert string_concept.extracted_items
        assert any(
            "Norway" in item.value and item.probabilities is None
            for item in string_concept.extracted_items
        )
        (item,) = boolean_concept.extracted_items
        assert item.value is True
        assert item.probabilities is not None
        assert list(item.probabilities) == [True, False]
        assert 0.5 <= item.probabilities[True] <= 1

        # The usage and cost of each model of the group
        llm_usage_container, usage_container = model_group.get_usage()
        assert llm_usage_container.model == llm.model
        assert llm_usage_container.usage.input > 0
        assert llm_usage_container.usage.output > 0
        assert usage_container.model == "typesafe/jev-latest"
        assert usage_container.role == "decide"
        assert usage_container.usage.input > 0
        assert len(usage_container.usage.calls) == 1
        assert model_group.get_usage(model_role="decide") == [usage_container]
        self._check_cost_of_recorded_tests(decision_model)
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.vcr
    def test_extract_concepts_with_openai_decisions_api(self):
        """
        Tests the extraction of concepts of each supported type from a document
        with a decision model of OpenAI, with the requests sent to its Decisions
        API: the only provider whose API takes the requests in the format that
        they are sent to litellm in, without a conversion. The costs are calculated
        automatically, from the usage that the API reports, with the prices of
        the decision requests of OpenAI.
        """
        decision_model = DocumentDecisionModel(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            auto_pricing=True,
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = self._create_concepts_for_recorded_tests()

        decision_model.extract_concepts_from_document(document)

        self._check_concepts_of_recorded_tests(document, decision_model)

    @pytest.mark.vcr
    def test_extract_concepts_with_ollama_api(self):
        """
        Tests the extraction of concepts of each supported type from a document
        with a decision model served by a local Ollama server, with the requests
        sent to the server. A local server is reached under the prefix of
        self-hosted decision models, without an API key, and has no prices.
        """
        decision_model = DocumentDecisionModel(
            model="strands_decider/nimble",
            api_base="http://localhost:11434",
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = self._create_concepts_for_recorded_tests()

        decision_model.extract_concepts_from_document(document)

        self._check_concepts_of_recorded_tests(document, decision_model)

    @pytest.mark.vcr
    def test_extract_concepts_with_lm_studio_api(self):
        """
        Tests the extraction of concepts of each supported type from a document
        with a decision model served by a local LM Studio server, with the requests
        sent to the server, under the same prefix as for an Ollama server.
        """
        decision_model = DocumentDecisionModel(
            model="strands_decider/kev-4b",
            api_base="http://localhost:1234",
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = self._create_concepts_for_recorded_tests()

        decision_model.extract_concepts_from_document(document)

        self._check_concepts_of_recorded_tests(document, decision_model)

    @pytest.mark.vcr
    def test_extract_concepts_with_microsoft_foundry_api(self):
        """
        Tests the extraction of concepts of each supported type from a document
        with a decision model deployed in Microsoft Foundry, with the requests
        sent to the endpoint of the Foundry resource. The model is identified by
        the name of its deployment, and is created with the endpoint of
        the resource and with its prices, which cannot be looked up by the name
        of a deployment.
        """
        decision_model = self._create_decision_model_for_recorded_tests(
            "microsoft_foundry"
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = self._create_concepts_for_recorded_tests()

        decision_model.extract_concepts_from_document(document)

        self._check_concepts_of_recorded_tests(document, decision_model)

    @pytest.mark.vcr
    def test_unknown_deployment_with_microsoft_foundry_api(self):
        """
        Tests that a request for a deployment that the Microsoft Foundry resource
        does not have fails with the error of the API, without being retried.
        The name of the model is such a deployment: a model deployed in Microsoft
        Foundry is requested by the name of its deployment.
        """
        api_key, api_base = _get_foundry_api_key_and_base()
        decision_model = DocumentDecisionModel(
            model="azure_ai/Microsoft-Decision-1", api_key=api_key, api_base=api_base
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = [
            BooleanConcept(
                name="NDA check",
                description="The document is a non-disclosure agreement",
            )
        ]

        with pytest.raises(ModelAPIError) as exc_info:
            decision_model.extract_concepts_from_document(document)

        original_error = exc_info.value.original_error
        assert isinstance(original_error, litellm.NotFoundError)
        assert "DeploymentNotFound" in str(original_error)
        assert not document.concepts[0]._is_processed
        assert not document.concepts[0].extracted_items
        # The error is not a transient one: the request is not sent again, and
        # is recorded as a failed one
        assert exc_info.value.retry_count == 0
        (call,) = decision_model.get_usage()[0].usage.calls
        assert call.timestamp_received is None
        assert call.response is None

    @staticmethod
    def _create_decision_model_for_recorded_tests(
        provider: Literal[
            "typesafe", "openai", "ollama", "lm_studio", "microsoft_foundry"
        ],
    ) -> DocumentDecisionModel:
        """
        Creates the decision model of a provider for a test that sends its
        requests to the API of the provider: with the API key and the prices of
        a hosted model, or with the address of a local server, which needs no
        API key and has no prices. A model deployed in Microsoft Foundry also
        takes the endpoint of the Foundry resource.

        :param provider: The provider of the decision model.
        :type provider: Literal["typesafe", "openai", "ollama", "lm_studio",
            "microsoft_foundry"]
        :return: The decision model.
        :rtype: DocumentDecisionModel
        """
        if provider == "microsoft_foundry":
            api_key, api_base = _get_foundry_api_key_and_base()
            return DocumentDecisionModel(
                model=_FOUNDRY_DECISION_MODEL,
                api_key=api_key,
                api_base=api_base,
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
                ),
            )
        if provider == "typesafe":
            return DocumentDecisionModel(
                model="typesafe/jev-latest",
                api_key=os.getenv("CONTEXTGEM_TYPESAFE_API_KEY"),
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
                ),
            )
        if provider == "openai":
            return DocumentDecisionModel(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.10, output_per_1m_tokens=0.0
                ),
            )
        if provider == "ollama":
            return DocumentDecisionModel(
                model="strands_decider/nimble",
                api_base="http://localhost:11434",
            )
        return DocumentDecisionModel(
            model="strands_decider/kev-4b",
            api_base="http://localhost:1234",
        )

    def _check_sentence_level_aspects_of_recorded_tests(
        self,
        provider: Literal[
            "typesafe", "openai", "ollama", "lm_studio", "microsoft_foundry"
        ],
    ) -> None:
        """
        Extracts, with the decision model of a provider through its API, an aspect
        with sentences as its items and an aspect with paragraphs as its items
        from a document in which a paragraph has additional context, and checks
        the items, the requests sent, and the usage of the model (see
        ``_RECORDED_SENTENCES_DOCUMENT_SENTENCE_TEXTS``).

        The items are checked, as the answers are unambiguous. The probabilities
        are checked for being consistent with the items: their exact values are
        those of the model.

        :param provider: The provider of the decision model.
        :type provider: Literal["typesafe", "openai", "ollama", "lm_studio",
            "microsoft_foundry"]
        :return: None
        :rtype: None
        """
        decision_model = self._create_decision_model_for_recorded_tests(provider)
        sentence_texts = _RECORDED_SENTENCES_DOCUMENT_SENTENCE_TEXTS
        paragraph_texts = [" ".join(texts) for texts in sentence_texts]
        context = _RECORDED_SENTENCES_DOCUMENT_ADDITIONAL_CONTEXT
        # The sentences are given, so that they do not depend on a segmentation
        document = Document(
            paragraphs=[
                Paragraph(
                    raw_text=paragraph_text,
                    sentences=[Sentence(raw_text=text) for text in texts],
                    additional_context=context if position == 2 else None,
                )
                for position, (paragraph_text, texts) in enumerate(
                    zip(paragraph_texts, sentence_texts, strict=True)
                )
            ]
        )
        document.aspects = [
            Aspect(
                name="Payment",
                description="Clauses on the fees and on how the fees are paid",
                reference_depth="sentences",
            ),
            Aspect(
                name="Termination",
                description="Clauses on how the agreement is terminated",
            ),
        ]

        # Without concurrency, i.e. with the requests in the order of the paragraphs
        decision_model.extract_aspects_from_document(document)

        payment, termination = document.aspects
        fee_paragraph, notice_paragraph, breach_paragraph = document.paragraphs[1:4]
        fee_sentence, report_sentence = fee_paragraph.sentences

        # === The aspect with sentences as its items ===
        # Of the two sentences of the paragraph that belongs to the aspect,
        # the one on the fee is extracted, and the one on the reports is not
        (item,) = payment.extracted_items
        assert item.value == fee_sentence.raw_text
        assert item.reference_paragraphs == [fee_paragraph]
        assert item.reference_sentences == [fee_sentence]
        assert payment.reference_paragraphs == [fee_paragraph]
        # As when an LLM extracts the aspect, its reference sentences are those
        # of its items, i.e. the extracted sentences: the sentence on the reports
        # is in the reference paragraph, and is not a reference sentence
        assert payment.reference_sentences == [fee_sentence]

        # === The aspect with paragraphs as its items ===
        # The paragraph on the notice is about the termination by its additional
        # context only
        assert [item.value for item in termination.extracted_items] == [
            notice_paragraph.raw_text,
            breach_paragraph.raw_text,
        ]
        assert termination.reference_paragraphs == [notice_paragraph, breach_paragraph]
        assert not termination.reference_sentences
        for item, paragraph in zip(
            termination.extracted_items,
            termination.reference_paragraphs,
            strict=True,
        ):
            assert item.reference_paragraphs == [paragraph]
            assert not item.reference_sentences

        for aspect in document.aspects:
            assert aspect._is_processed
            for item in aspect.extracted_items:
                # Whether the sentence or the paragraph belongs to the aspect:
                # it does
                assert item.probabilities is not None
                assert list(item.probabilities) == [True, False]
                assert 0.5 <= item.probabilities[True] <= 1
                assert item.probabilities[False] == pytest.approx(
                    1 - item.probabilities[True]
                )
                assert item.justification is None

        # === The requests ===
        # The aspects differ in what their items are, so each is extracted in
        # a call of its own: a request for each paragraph and, for the aspect
        # with sentences as its items, a request for each sentence of
        # the paragraph that belongs to it. Each request has one question.
        (usage_container,) = decision_model.get_usage()
        usage = usage_container.usage
        requests = [json.loads(call.prompt) for call in usage.calls]
        paragraph_inputs = list(paragraph_texts)
        # A paragraph with additional context is sent with it, each with a label
        paragraph_inputs[2] = (
            f"Paragraph: {paragraph_texts[2]}\n\n"
            f"Additional context of the paragraph: {context}"
        )
        # A sentence is sent with its paragraph as its context
        sentence_inputs = [
            f"Sentence: {sentence.raw_text}\n\n"
            f"Paragraph that the sentence is from: {fee_paragraph.raw_text}"
            for sentence in (fee_sentence, report_sentence)
        ]
        assert [request["input"] for request in requests] == [
            *paragraph_inputs,
            *sentence_inputs,
            *paragraph_inputs,
        ]
        assert all(len(request["questions"]) == 1 for request in requests)
        assert usage.input > 0
        assert all(call.timestamp_received is not None for call in usage.calls)
        self._check_cost_of_recorded_tests(decision_model)
        self.check_instance_serialization_and_cloning(document)

        # === The paragraph on the notice, without its additional context ===
        # The model is not more sure that the paragraph is about the termination
        # than it is with the additional context
        other_decision_model = self._create_decision_model_for_recorded_tests(provider)
        other_document = Document(
            paragraphs=[Paragraph(raw_text=notice_paragraph.raw_text)]
        )
        other_document.aspects = [
            Aspect(
                name="Termination",
                description="Clauses on how the agreement is terminated",
            )
        ]
        other_decision_model.extract_aspects_from_document(other_document)
        (call,) = other_decision_model.get_usage()[0].usage.calls
        assert json.loads(call.prompt)["input"] == notice_paragraph.raw_text
        assert call.response is not None
        (answer,) = json.loads(call.response)
        probabilities_with_context = termination.extracted_items[0].probabilities
        assert probabilities_with_context is not None
        assert 0 <= answer["probability"] <= probabilities_with_context[True]

    def _check_concepts_in_parts_of_recorded_tests(
        self,
        provider: Literal[
            "typesafe", "openai", "ollama", "lm_studio", "microsoft_foundry"
        ],
    ) -> None:
        """
        Extracts, with the decision model of a provider through its API, concepts
        from a document in parts of two paragraphs, and checks the items of
        the concepts, one for each part, the requests sent, and the usage of
        the model (see ``_RECORDED_CONCEPTS_DOCUMENT_TEXT``).

        The values are checked where the answers are unambiguous for the part.

        :param provider: The provider of the decision model.
        :type provider: Literal["typesafe", "openai", "ollama", "lm_studio",
            "microsoft_foundry"]
        :return: None
        :rtype: None
        """
        decision_model = self._create_decision_model_for_recorded_tests(provider)
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        paragraph_texts = [paragraph.raw_text for paragraph in document.paragraphs]
        assert len(paragraph_texts) == 4
        document.concepts = [
            # The company is named in the first part only
            BooleanConcept(
                name="Mentions Alpha Ltd",
                description="The text mentions the company Alpha Ltd",
            ),
            # The country is named in the second part only
            BooleanConcept(
                name="Mentions Norway",
                description="The text mentions Norway",
            ),
            LabelConcept(
                name="Topics",
                description="Topics that the text covers",
                labels=["confidentiality", "governing law", "salary", "real estate"],
                classification_type="multi_label",
            ),
            # The rating is given in the second part only
            RatingConcept(
                name="Sensitivity",
                description="Sensitivity of the confidential information, as rated "
                "by the parties in the text",
                rating_scale=(1, 5),
            ),
        ]

        # Without concurrency, i.e. with the requests in the order of the parts
        decision_model.extract_concepts_from_document(
            document, max_paragraphs_to_analyze_per_call=2
        )

        # === The items: one for each part, in the order of the parts ===
        alpha, norway, topics, sensitivity = document.concepts
        assert [item.value for item in alpha.extracted_items] == [True, False]
        assert [item.value for item in norway.extracted_items] == [False, True]
        assert [item.value for item in topics.extracted_items] == [
            ["confidentiality"],
            ["confidentiality", "governing law"],
        ]
        # The first part gives no rating: the model picks a level for it anyway
        rating_of_first_part, rating_of_second_part = sensitivity.extracted_items
        assert rating_of_first_part.value in range(1, 6)
        assert rating_of_second_part.value == 4
        for concept in document.concepts:
            assert concept._is_processed
            for item in concept.extracted_items:
                assert item.probabilities is not None
                assert all(
                    0 <= probability <= 1 for probability in item.probabilities.values()
                )
                assert item.justification is None
        for item in [*alpha.extracted_items, *norway.extracted_items]:
            assert item.probabilities is not None
            assert list(item.probabilities) == [True, False]
            assert item.value is (item.probabilities[True] >= 0.5)
        for item in topics.extracted_items:
            assert item.probabilities is not None
            assert item.value == [
                label
                for label, probability in item.probabilities.items()
                if probability >= 0.5
            ]
        for item in sensitivity.extracted_items:
            assert item.probabilities is not None
            assert list(item.probabilities) == [1, 2, 3, 4, 5]
            assert item.probabilities[item.value] == max(item.probabilities.values())

        # === The requests ===
        # A request for each part, with the text of its two paragraphs and
        # the questions of all the concepts: one of each concept, and one for
        # each label of the concept with any number of labels to select
        (usage_container,) = decision_model.get_usage()
        usage = usage_container.usage
        requests = [json.loads(call.prompt) for call in usage.calls]
        assert [request["input"] for request in requests] == [
            "\n\n".join(paragraph_texts[:2]),
            "\n\n".join(paragraph_texts[2:]),
        ]
        assert [len(request["questions"]) for request in requests] == [7, 7]
        assert usage.input > 0
        assert all(call.timestamp_received is not None for call in usage.calls)
        self._check_cost_of_recorded_tests(decision_model)
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.vcr
    def test_extract_sentence_level_aspects_with_typesafe_api(self):
        """
        Tests the extraction of an aspect with sentences as its items, and of
        a paragraph that is sent with its additional context, with a decision
        model of TypeSafe, with the requests sent to its API.
        """
        self._check_sentence_level_aspects_of_recorded_tests("typesafe")

    @pytest.mark.vcr
    def test_extract_sentence_level_aspects_with_openai_decisions_api(self):
        """
        Tests the extraction of an aspect with sentences as its items, and of
        a paragraph that is sent with its additional context, with a decision
        model of OpenAI, with the requests sent to its Decisions API.
        """
        self._check_sentence_level_aspects_of_recorded_tests("openai")

    @pytest.mark.vcr
    def test_extract_sentence_level_aspects_with_ollama_api(self):
        """
        Tests the extraction of an aspect with sentences as its items, and of
        a paragraph that is sent with its additional context, with a decision
        model served by a local Ollama server, with the requests sent to
        the server.
        """
        self._check_sentence_level_aspects_of_recorded_tests("ollama")

    @pytest.mark.vcr
    def test_extract_sentence_level_aspects_with_lm_studio_api(self):
        """
        Tests the extraction of an aspect with sentences as its items, and of
        a paragraph that is sent with its additional context, with a decision
        model served by a local LM Studio server, with the requests sent to
        the server.
        """
        self._check_sentence_level_aspects_of_recorded_tests("lm_studio")

    @pytest.mark.vcr
    def test_extract_sentence_level_aspects_with_microsoft_foundry_api(self):
        """
        Tests the extraction of an aspect with sentences as its items, and of
        a paragraph that is sent with its additional context, with a decision
        model deployed in Microsoft Foundry, with the requests sent to
        the endpoint of the Foundry resource.
        """
        self._check_sentence_level_aspects_of_recorded_tests("microsoft_foundry")

    @pytest.mark.vcr
    def test_extract_concepts_in_parts_with_typesafe_api(self):
        """
        Tests the extraction of concepts from a document in parts, with one
        item of a concept for each part, with a decision model of TypeSafe,
        with the requests sent to its API.
        """
        self._check_concepts_in_parts_of_recorded_tests("typesafe")

    @pytest.mark.vcr
    def test_extract_concepts_in_parts_with_openai_decisions_api(self):
        """
        Tests the extraction of concepts from a document in parts, with one
        item of a concept for each part, with a decision model of OpenAI, with
        the requests sent to its Decisions API.
        """
        self._check_concepts_in_parts_of_recorded_tests("openai")

    @pytest.mark.vcr
    def test_extract_concepts_in_parts_with_ollama_api(self):
        """
        Tests the extraction of concepts from a document in parts, with one
        item of a concept for each part, with a decision model served by
        a local Ollama server, with the requests sent to the server.
        """
        self._check_concepts_in_parts_of_recorded_tests("ollama")

    @pytest.mark.vcr
    def test_extract_concepts_in_parts_with_lm_studio_api(self):
        """
        Tests the extraction of concepts from a document in parts, with one
        item of a concept for each part, with a decision model served by
        a local LM Studio server, with the requests sent to the server.
        """
        self._check_concepts_in_parts_of_recorded_tests("lm_studio")

    @pytest.mark.vcr
    def test_extract_concepts_in_parts_with_microsoft_foundry_api(self):
        """
        Tests the extraction of concepts from a document in parts, with one
        item of a concept for each part, with a decision model deployed in
        Microsoft Foundry, with the requests sent to the endpoint of
        the Foundry resource.
        """
        self._check_concepts_in_parts_of_recorded_tests("microsoft_foundry")

    def test_decision_model_blocked_request_falls_back(self):
        """
        Tests that a request to a decision model that the provider blocks, or whose
        answers it blocks, for their content (``litellm.ContentPolicyViolationError``)
        is handled as a refusal: the request is sent again, up to
        ``max_retries_invalid_data`` times, before it is passed on to the fallback
        model, which answers, and without a fallback model, ``ModelRefusalError``
        is raised, with the error of litellm for the last request blocked, or
        a warning is logged instead. Each blocked request is recorded in the usage
        as a finished call, without a response, with its refusal and the error of
        litellm.
        """
        # The first request, and one with each of the 3 retries (the default)
        n_requests = 1 + 3
        concept_names = "['NDA check']"
        refusal_sentence = (
            f"The provider of decision model `{_DECISION_MODEL}` blocked the request, "
            f"or the answer to it, for its content."
        )
        refusal_msg = f"{refusal_sentence} Failed concepts: {concept_names}"
        for with_fallback, raise_exception in [
            (True, True),
            (False, True),
            (False, False),
        ]:
            decision_model = _create_decision_model(
                fallback_models=[_create_llm(is_fallback=True)] if with_fallback else []
            )
            document = _create_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The document is an NDA"
                    )
                ]
            )
            # An error for each request, to tell the requests apart
            blocked_errors = [
                _create_api_error(litellm.ContentPolicyViolationError)
                for _ in range(n_requests)
            ]
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: list(blocked_errors)})
            completions_stub = _CompletionsAPIStub(
                {_LLM_MODEL: ['[{"concept_id": "C0", "extracted_items": [true]}]']}
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                patch.object(litellm, "acompletion", new=completions_stub),
                capture_logs() as captured_logs,
            ):
                if with_fallback:
                    decision_model.extract_concepts_from_document(document)
                elif raise_exception:
                    with pytest.raises(ModelRefusalError) as exc_info:
                        decision_model.extract_concepts_from_document(document)
                    assert exc_info.value.message == refusal_msg
                    # The error of litellm for the last request blocked
                    assert exc_info.value.original_error is blocked_errors[-1]
                    assert exc_info.value.retry_count == 3
                else:
                    decision_model.extract_concepts_from_document(
                        document, raise_exception_on_extraction_error=False
                    )
                    assert (
                        f"{refusal_msg} (3 retries). If you want to raise an "
                        f"exception instead, set `raise_exception_on_extraction_error` "
                        f"to True."
                    ) in captured_logs, captured_logs
            # The refusal is logged for each request blocked, with the concepts of
            # the call, which is made again after each refusal but the last one
            assert (
                captured_logs.count(
                    f"{refusal_sentence} Concepts of the call: {concept_names}"
                )
                == n_requests
            ), captured_logs
            assert [log for log in captured_logs if log.startswith("Retrying")] == [
                f"Retrying concepts after a refusal ({i}/3)" for i in range(1, 4)
            ]
            # The request is sent again to the decision model, and then the fallback
            # LLM answers it
            assert len(decisions_stub.requests) == n_requests
            assert len(completions_stub.requests) == (1 if with_fallback else 0)
            assert document.concepts[0]._is_processed is with_fallback
            if with_fallback:
                assert document.concepts[0].extracted_items[0].value is True
            usage = decision_model.get_usage()[0].usage
            assert usage.input == 0
            assert len(usage.calls) == n_requests
            for call, blocked_error in zip(usage.calls, blocked_errors, strict=True):
                assert call.timestamp_received is not None
                assert call.response is None
                assert call._refusal == "blocked"
                assert call._original_error_of_refusal is blocked_error

    def test_decision_model_retries_invalid_response(self):
        """
        Tests that a call for which a decision model returns a response that cannot
        be used is made again, by default (``max_retries_invalid_data=3``): with
        a usable response to the retry, the items are extracted from its answers,
        the retry is logged, and the requests of both calls are counted in
        the usage.
        """
        decision_model = _create_decision_model()
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA")
            ]
        )
        # A probability that is not within [0, 1], and then a valid one
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [{"C0": 1.5}, {"C0": 0.8}]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            capture_logs() as captured_logs,
        ):
            decision_model.extract_concepts_from_document(document)
        # The same request is sent again, once
        first_request, second_request = decisions_stub.requests
        assert first_request["input"] == second_request["input"] == _DOCUMENT_TEXT_SENT
        assert first_request["questions"] == second_request["questions"]
        assert (
            f"Decision model `{_DECISION_MODEL}` returned an invalid response for "
            "concept `NDA check`, decision question `C0`: the probability of "
            "the answer is not a number within [0, 1]."
        ) in captured_logs, captured_logs
        assert [log for log in captured_logs if log.startswith("Retrying")] == [
            "Retrying concepts with an invalid response (1/3)"
        ]
        # The item is created from the answer to the retry
        (item,) = document.concepts[0].extracted_items
        assert item.value is True
        self.check_item_probabilities(item, {True: 0.8, False: 1 - 0.8})
        # Both requests are counted, as finished calls with their answers
        usage = decision_model.get_usage()[0].usage
        assert usage.input == 2 * 100
        assert all(call.timestamp_received is not None for call in usage.calls)
        assert [json.loads(call.response or "null") for call in usage.calls] == [
            [{"type": "predicate", "name": "C0", "probability": 1.5}],
            [{"type": "predicate", "name": "C0", "probability": 0.8}],
        ]

    def test_decision_model_retry_sends_failed_requests_only(self):
        """
        Tests that, when a call of a decision model is made again for a response
        that cannot be used, only the request that got it is sent again, with
        the requests that were not sent after it, and not the requests whose
        answers were usable, which are kept: for aspects, asked about on each
        paragraph, and for concepts, asked about on each part of the text.
        If the answers cannot be converted into items, which request they are
        from is not known: all the requests are sent again.
        """
        yes_probabilities = {"Termination": {2: 0.9, 4: 0.7}, "Payment": {3: 0.8}}
        respond = _create_aspects_responder(yes_probabilities)
        paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        failed_position = 2

        # === Aspects, asked about on each paragraph ===
        # The response on the third paragraph cannot be used, once or twice
        for n_failures in [1, 2]:
            decision_model = _create_decision_model()
            document = _create_aspects_document(
                aspects=[
                    Aspect(name=name, description=f"Clauses on {name.lower()}")
                    for name in yes_probabilities
                ]
            )
            # The outcomes in the order of the requests
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        respond,
                        respond,
                        *[{"A0": 0.9, "A1": 1.5}] * n_failures,
                        respond,
                    ]
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs() as captured_logs,
            ):
                decision_model.extract_aspects_from_document(document)
            # The paragraphs before the third one are asked about once, as their
            # answers are kept, the third one with each call, and those after it
            # once it is answered
            assert [request["input"] for request in decisions_stub.requests] == [
                *paragraph_texts[:failed_position],
                *[paragraph_texts[failed_position]] * (1 + n_failures),
                *paragraph_texts[failed_position + 1 :],
            ]
            assert [log for log in captured_logs if log.startswith("Retrying")] == [
                f"Retrying aspects with an invalid response ({i}/3)"
                for i in range(1, n_failures + 1)
            ]
            termination, payment = document.aspects
            paragraphs = document.paragraphs
            assert [item.value for item in termination.extracted_items] == [
                paragraphs[2].raw_text,
                paragraphs[4].raw_text,
            ]
            assert [item.value for item in payment.extracted_items] == [
                paragraphs[3].raw_text
            ]
            # Each request sent is counted, as a finished call
            usage = decision_model.get_usage()[0].usage
            assert len(usage.calls) == len(paragraph_texts) + n_failures
            assert all(call.timestamp_received is not None for call in usage.calls)
            assert usage.input == 100 * len(usage.calls)

        # === Aspects, with concurrency ===
        # A request whose answers are kept is not sent, and takes no turn of
        # the limiter: each turn is a request sent
        n_acquisitions: list[int] = [0]

        class CountingAsyncLimiter(AsyncLimiter):
            """
            Counts the times that its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Counts the acquisition and acquires the capacity.

                :param amount: The capacity to acquire.
                :type amount: float
                """
                n_acquisitions[0] += 1
                await super().acquire(amount)

        failed_inputs: list[str] = []

        def respond_failing_once(request: dict[str, Any]) -> Any:
            """
            Responds to the first request on the third paragraph with a response
            that cannot be used, and to the other requests with their answers.

            :param request: Keyword arguments of ``litellm.adecisions``.
            :type request: dict[str, Any]
            :return: Response of litellm.
            :rtype: Any
            """
            if request["input"] == paragraph_texts[failed_position] and not (
                failed_inputs
            ):
                failed_inputs.append(request["input"])
                return _create_decisions_response(
                    request["questions"], {"A0": 0.9, "A1": 1.5}
                )
            return respond(request)

        decision_model = _create_decision_model(async_limiter=AsyncLimiter(1000, 1))
        document = _create_aspects_document(
            aspects=[
                Aspect(name=name, description=f"Clauses on {name.lower()}")
                for name in yes_probabilities
            ]
        )
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [respond_failing_once]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            # The limiter that the requests are paced by, created from the one
            # of the model for the event loop of the extraction
            patch.object(models_module, "AsyncLimiter", new=CountingAsyncLimiter),
        ):
            # Both aspects in one call, as without concurrency
            decision_model.extract_aspects_from_document(
                document, use_concurrency=True, max_items_per_call=2
            )
        sent_inputs = [request["input"] for request in decisions_stub.requests]
        assert failed_inputs == [paragraph_texts[failed_position]]
        assert sent_inputs.count(paragraph_texts[failed_position]) == 2, sent_inputs
        # The paragraphs before the third one were answered before it failed
        assert all(
            sent_inputs.count(text) == 1 for text in paragraph_texts[:failed_position]
        ), sent_inputs
        assert n_acquisitions[0] == len(sent_inputs), sent_inputs
        assert all(aspect.extracted_items for aspect in document.aspects)

        # === Concepts, asked about on each part of the text ===
        part_texts = self._get_texts_of_parts(2)

        def create_concepts_document() -> Document:
            """
            Creates a document with five paragraphs and a concept to extract.

            :return: Document.
            :rtype: Document
            """
            return _create_aspects_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    )
                ]
            )

        # The answer on the second part cannot be used: the retry asks about it,
        # and about the part after it, and not about the first part
        decision_model = _create_decision_model()
        document = create_concepts_document()
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [{"C0": 0.9}, {"C0": 1.5}, {"C0": 0.4}]}
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )
        assert [request["input"] for request in decisions_stub.requests] == [
            part_texts[0],
            part_texts[1],
            part_texts[1],
            part_texts[2],
        ]
        assert [item.value for item in document.concepts[0].extracted_items] == [
            True,
            False,
            False,
        ]

        # === Answers that cannot be converted into items ===
        # The answers on every part are asked for again
        original_get_concepts_items = _DocumentDecisionModel._get_concepts_items
        conversion_errors = [ValueError("Simulated conversion error")]

        def get_concepts_items(*args: Any, **kwargs: Any) -> Any:
            """
            Fails the first conversion of the answers into the items of
            the concepts, and converts them afterwards.

            :param args: Arguments of the conversion.
            :type args: Any
            :param kwargs: Keyword arguments of the conversion.
            :type kwargs: Any
            :return: The concepts with their items.
            :rtype: Any
            """
            if conversion_errors:
                raise conversion_errors.pop()
            return original_get_concepts_items(*args, **kwargs)

        decision_model = _create_decision_model()
        document = create_concepts_document()
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [{"C0": 0.9}]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(
                _DocumentDecisionModel,
                "_get_concepts_items",
                new=staticmethod(get_concepts_items),
            ),
        ):
            decision_model.extract_concepts_from_document(
                document, max_paragraphs_to_analyze_per_call=2
            )
        assert [request["input"] for request in decisions_stub.requests] == (
            part_texts * 2
        )
        assert [item.value for item in document.concepts[0].extracted_items] == [
            True
        ] * len(part_texts)

    def test_decision_model_retry_sends_refused_requests_again(self):
        """
        Tests that, when a call of a decision model for aspects is made again,
        a request with a refused answer, which is accepted for an aspect on some
        of the paragraphs, is sent again with the call, as the model may answer it
        then, while the requests with usable answers are not. An aspect that
        the model refuses to answer about on every paragraph is asked about again
        on every paragraph.
        """
        yes_probabilities = {"Termination": {0: 0.9, 2: 0.9}, "Payment": {4: 0.8}}
        respond = _create_aspects_responder(yes_probabilities)
        paragraph_texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        n_paragraphs = len(paragraph_texts)

        def extract_aspects(
            decisions_stub: _DecisionsAPIStub,
        ) -> tuple[Document, DocumentDecisionModel, list[str]]:
            """
            Extracts the aspects of a new document with a new decision model.

            :param decisions_stub: The stub of the Decisions API.
            :type decisions_stub: _DecisionsAPIStub
            :return: The document, the decision model, and the messages logged.
            :rtype: tuple[Document, DocumentDecisionModel, list[str]]
            """
            decision_model = _create_decision_model()
            document = _create_aspects_document(
                aspects=[
                    Aspect(name=name, description=f"Clauses on {name.lower()}")
                    for name in yes_probabilities
                ]
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs() as captured_logs,
            ):
                decision_model.extract_aspects_from_document(document)
            return document, decision_model, captured_logs

        def check_extracted(document: Document) -> None:
            """
            Checks that the aspects are extracted with the answers of ``respond``.

            :param document: The document.
            :type document: Document
            """
            termination, payment = document.aspects
            paragraphs = document.paragraphs
            assert [item.value for item in termination.extracted_items] == [
                paragraphs[0].raw_text,
                paragraphs[2].raw_text,
            ]
            assert [item.value for item in payment.extracted_items] == [
                paragraphs[4].raw_text
            ]

        # === A refused answer on a paragraph, then a response that cannot be used ===
        # The model refuses to answer about the first aspect on the first
        # paragraph, which is accepted, and its response on the fourth paragraph
        # cannot be used: the call is made again with the request on the first
        # paragraph, and with those on the fourth paragraph and after it
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    {"A0": "refusal", "A1": 0.1},
                    respond,
                    respond,
                    {"A0": 0.1, "A1": 1.5},
                    respond,
                ]
            }
        )
        document, _, captured_logs = extract_aspects(decisions_stub)
        assert [request["input"] for request in decisions_stub.requests] == [
            *paragraph_texts[:4],
            paragraph_texts[0],
            *paragraph_texts[3:],
        ]
        assert [log for log in captured_logs if log.startswith("Retrying")] == [
            "Retrying aspects with an invalid response (1/3)"
        ]
        # The model answers about the first paragraph when asked again: it is
        # included in the aspect, and no answer is left refused
        check_extracted(document)
        assert not any("not included in the aspect" in log for log in captured_logs)

        # === An aspect refused on every paragraph ===
        # The model refuses to answer about the second aspect on every paragraph,
        # and answers when it is asked again: every paragraph is asked about again,
        # as no request has answers to keep
        refusing_probabilities: dict[str, Any] = {
            "Termination": {0: 0.9, 2: 0.9},
            "Payment": dict.fromkeys(range(n_paragraphs), "refusal"),
        }
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    *[_create_aspects_responder(refusing_probabilities)] * n_paragraphs,
                    respond,
                ]
            }
        )
        document, decision_model, captured_logs = extract_aspects(decisions_stub)
        assert [request["input"] for request in decisions_stub.requests] == (
            paragraph_texts * 2
        )
        assert [log for log in captured_logs if log.startswith("Retrying")] == [
            "Retrying aspects after a refusal (1/3)"
        ]
        check_extracted(document)
        # The requests of the first call are recorded with the refusal
        usage = decision_model.get_usage()[0].usage
        assert [call._refusal for call in usage.calls] == (
            ["refusal"] * n_paragraphs + [None] * n_paragraphs
        )

    def test_decision_model_without_retries_for_invalid_data(self):
        """
        Tests that a decision model with ``max_retries_invalid_data=0`` is called
        once for a response that cannot be used, a refused answer and a blocked
        request: the error is raised with no retries, or the fallback model is
        tried right away.
        """
        blocked_error = _create_api_error(litellm.ContentPolicyViolationError)

        def create_document() -> Document:
            """
            Creates a document with a concept to extract.

            :return: Document.
            :rtype: Document
            """
            return _create_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The document is an NDA"
                    )
                ]
            )

        for outcome, error_class in [
            # A probability that is not within [0, 1]
            ({"C0": 1.5}, ModelExtractionError),
            ({"C0": "refusal"}, ModelRefusalError),
            (blocked_error, ModelRefusalError),
        ]:
            # Without a fallback model, the error is raised
            decision_model = _create_decision_model(max_retries_invalid_data=0)
            document = create_document()
            decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [outcome]})
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs() as captured_logs,
                pytest.raises(error_class, match=r"\(after 0 retries\)") as exc_info,
            ):
                decision_model.extract_concepts_from_document(document)
            assert exc_info.value.retry_count == 0
            assert len(decisions_stub.requests) == 1
            assert not any(log.startswith("Retrying") for log in captured_logs)
            assert not document.concepts[0]._is_processed

            # With a fallback model, it is tried right away
            decision_model = _create_decision_model(
                max_retries_invalid_data=0,
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ],
            )
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [outcome], _OTHER_DECISION_MODEL: [{"C0": 0.9}]}
            )
            with patch.object(litellm, "adecisions", new=decisions_stub):
                decision_model.extract_concepts_from_document(document)
            assert decisions_stub.requested_models() == [
                _DECISION_MODEL,
                _OTHER_DECISION_MODEL,
            ]
            assert document.concepts[0].extracted_items[0].value is True

    def test_fallback_decision_model_retries_invalid_data(self):
        """
        Tests that a fallback decision model is asked once, and then again up to
        its own ``max_retries_invalid_data`` times, once the model that it falls
        back from has used up its own, with the answers that it keeps of its own:
        it is asked about every part of the text first, as the answers of
        the model before it are not given to it, and then about the part that
        failed.
        """
        part_texts = self._get_texts_of_parts(2)
        invalid_answers = {"C0": 1.5}
        # The requests of the last extraction, as their models and texts, and
        # the messages logged in it on the retries and on the fallback model, to be
        # checked also when the extraction raises an exception
        requests_sent: list[tuple[str, str]] = []
        retry_logs: list[str] = []

        def extract(
            fallback_outcomes: list[Any],
        ) -> tuple[Document, DocumentDecisionModel]:
            """
            Extracts a concept from a document in parts with a decision model that
            fails on the second part, also when it is asked again, and that has
            a fallback decision model with the given outcomes.

            :param fallback_outcomes: The outcomes of the requests to the fallback
                decision model, in the order of the requests.
            :type fallback_outcomes: list[Any]
            :return: The document and the decision model.
            :rtype: tuple[Document, DocumentDecisionModel]
            """
            # Retries of their own: 1 for the model, 2 for its fallback model
            decision_model = _create_decision_model(
                max_retries_invalid_data=1,
                fallback_models=[
                    _create_decision_model(
                        _OTHER_DECISION_MODEL,
                        is_fallback=True,
                        max_retries_invalid_data=2,
                    )
                ],
            )
            document = _create_aspects_document(
                concepts=[
                    BooleanConcept(
                        name="NDA check", description="The text is of an NDA"
                    )
                ]
            )
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [{"C0": 0.9}, invalid_answers],
                    _OTHER_DECISION_MODEL: fallback_outcomes,
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                capture_logs() as captured_logs,
            ):
                try:
                    decision_model.extract_concepts_from_document(
                        document, max_paragraphs_to_analyze_per_call=2
                    )
                finally:
                    requests_sent[:] = [
                        (request["model"], request["input"])
                        for request in decisions_stub.requests
                    ]
                    retry_logs[:] = [
                        log
                        for log in captured_logs
                        if log.startswith(("Retrying", "Trying with fallback"))
                    ]
            return document, decision_model

        expected_retry_logs = [
            "Retrying concepts with an invalid response (1/1)",
            f"Trying with fallback decision model {_OTHER_DECISION_MODEL} (1/1)",
            "Retrying concepts with an invalid response (1/2)",
            "Retrying concepts with an invalid response (2/2)",
        ]
        # The requests of the model, and then of its fallback model, which fails
        # on the third part on its first call and when asked again the first time
        expected_requests = [
            *[(_DECISION_MODEL, part_texts[position]) for position in [0, 1, 1]],
            *[
                (_OTHER_DECISION_MODEL, part_texts[position])
                for position in [0, 1, 2, 2, 2]
            ],
        ]

        # === The fallback model answers when asked again the second time ===
        document, decision_model = extract(
            [{"C0": 0.9}, {"C0": 0.8}, invalid_answers, invalid_answers, {"C0": 0.3}]
        )
        assert requests_sent == expected_requests
        assert retry_logs == expected_retry_logs
        # The items are created from the answers of the fallback model
        assert [item.value for item in document.concepts[0].extracted_items] == [
            True,
            True,
            False,
        ]
        fallback_usage = decision_model.get_usage()[1].usage
        assert [json.loads(call.prompt)["input"] for call in fallback_usage.calls] == [
            part_texts[position] for position in [0, 1, 2, 2, 2]
        ]

        # === The fallback model fails each time ===
        # The error is the one of the fallback model, with its retries
        with pytest.raises(
            ModelExtractionError,
            match=r"an invalid response returned by decision model.*\(after 2 retries\)",
        ) as exc_info:
            extract([{"C0": 0.9}, {"C0": 0.8}, invalid_answers])
        assert exc_info.value.retry_count == 2
        assert requests_sent == expected_requests
        assert retry_logs == expected_retry_logs

    def test_decision_model_blocked_request_retried(self):
        """
        Tests that a request that the provider of a decision model blocks for its
        content (``litellm.ContentPolicyViolationError``) is sent again when
        the call is made again, as the verdict of a content filter can differ on
        another request: the items are extracted from the answers to it, and
        the requests whose answers were usable are not sent again.
        """
        blocked_error = _create_api_error(litellm.ContentPolicyViolationError)

        # === Concepts ===
        decision_model = _create_decision_model()
        document = _create_document(
            concepts=[
                BooleanConcept(name="NDA check", description="The document is an NDA")
            ]
        )
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [blocked_error, {"C0": 0.8}]}
        )
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            capture_logs() as captured_logs,
        ):
            decision_model.extract_concepts_from_document(document)
        assert len(decisions_stub.requests) == 2
        assert (
            f"The provider of decision model `{_DECISION_MODEL}` blocked the request, "
            "or the answer to it, for its content. Concepts of the call: "
            "['NDA check']"
        ) in captured_logs, captured_logs
        assert [log for log in captured_logs if log.startswith("Retrying")] == [
            "Retrying concepts after a refusal (1/3)"
        ]
        (item,) = document.concepts[0].extracted_items
        assert item.value is True
        self.check_item_probabilities(item, {True: 0.8, False: 1 - 0.8})
        # The blocked request is a finished call without a response, with
        # the refusal and the error of litellm
        usage = decision_model.get_usage()[0].usage
        blocked_call, answered_call = usage.calls
        assert blocked_call.timestamp_received is not None
        assert blocked_call.response is None
        assert blocked_call._refusal == "blocked"
        assert blocked_call._original_error_of_refusal is blocked_error
        assert answered_call.response is not None
        assert answered_call._refusal is None
        assert usage.input == 100

        # === Aspects ===
        # The request on the third paragraph is blocked once: the call is made
        # again with it and with the requests after it, while the answers on
        # the paragraphs before it are kept
        yes_probabilities = {"Termination": {2: 0.9, 4: 0.7}, "Payment": {3: 0.8}}
        respond = _create_aspects_responder(yes_probabilities)
        decision_model = _create_decision_model()
        document = _create_aspects_document(
            aspects=[
                Aspect(name=name, description=f"Clauses on {name.lower()}")
                for name in yes_probabilities
            ]
        )
        decisions_stub = _DecisionsAPIStub(
            {_DECISION_MODEL: [respond, respond, blocked_error, respond]}
        )
        with patch.object(litellm, "adecisions", new=decisions_stub):
            decision_model.extract_aspects_from_document(document)
        assert [request["input"] for request in decisions_stub.requests] == [
            *_ASPECTS_PARAGRAPH_TEXTS_SENT[:3],
            *_ASPECTS_PARAGRAPH_TEXTS_SENT[2:],
        ]
        termination, payment = document.aspects
        paragraphs = document.paragraphs
        assert [item.value for item in termination.extracted_items] == [
            paragraphs[2].raw_text,
            paragraphs[4].raw_text,
        ]
        assert [item.value for item in payment.extracted_items] == [
            paragraphs[3].raw_text
        ]
        usage = decision_model.get_usage()[0].usage
        assert [call._refusal for call in usage.calls] == [
            None,
            None,
            "blocked",
            None,
            None,
            None,
        ]

    def test_decision_model_max_retries_invalid_data_param(self):
        """
        Tests the validation of ``max_retries_invalid_data`` of a decision model,
        a non-negative integer, also when it is assigned, and that it is serialized
        with the model and with its fallback decision model.
        """
        # Valid values
        for value in [0, 1, 10]:
            decision_model = _create_decision_model(max_retries_invalid_data=value)
            assert decision_model.max_retries_invalid_data == value
            assert decision_model._retries_invalid_data == value
        # Invalid values: a negative one, and a value that is not an integer, also
        # a bool or a float with an integral value
        invalid_values: list[Any] = [-1, True, 1.0, "1", None]
        for value in invalid_values:
            with pytest.raises(ValueError, match="max_retries_invalid_data"):
                _create_decision_model(max_retries_invalid_data=value)
        # ... also when assigned
        decision_model = _create_decision_model(max_retries_invalid_data=2)
        for value in invalid_values:
            with pytest.raises(ValueError, match="max_retries_invalid_data"):
                decision_model.max_retries_invalid_data = value
        assert decision_model.max_retries_invalid_data == 2

        # Serialized with the model and with its fallback model
        decision_model = _create_decision_model(
            max_retries_invalid_data=5,
            fallback_models=[
                _create_decision_model(
                    _OTHER_DECISION_MODEL, is_fallback=True, max_retries_invalid_data=0
                )
            ],
        )
        decision_model_dict = decision_model.to_dict()
        assert decision_model_dict["max_retries_invalid_data"] == 5
        assert (
            decision_model_dict["fallback_models"][0]["max_retries_invalid_data"] == 0
        )
        for restored_decision_model in [
            DocumentDecisionModel.from_dict(decision_model_dict),
            DocumentDecisionModel.from_json(decision_model.to_json()),
        ]:
            assert decision_model._eq_deserialized_model_config(restored_decision_model)
            assert restored_decision_model.max_retries_invalid_data == 5
            (restored_fallback_model,) = restored_decision_model.fallback_models
            assert isinstance(restored_fallback_model, DocumentDecisionModel)
            assert restored_fallback_model.max_retries_invalid_data == 0
        # A model restored with another value is another config
        decision_model_dict["max_retries_invalid_data"] = 4
        assert not decision_model._eq_deserialized_model_config(
            DocumentDecisionModel.from_dict(decision_model_dict)
        )
        # ... and an invalid value is rejected in the serialized data as well
        decision_model_dict["max_retries_invalid_data"] = -1
        with pytest.raises(ValueError, match="max_retries_invalid_data"):
            DocumentDecisionModel.from_dict(decision_model_dict)

    @staticmethod
    def _extract_with_refusal_fallbacks(
        decision_model: DocumentDecisionModel,
        document: Document,
        decisions_outcomes: dict[str, list[Any]],
        completions_outcomes: dict[str, list[Any]] | None = None,
        **extraction_kwargs: Any,
    ) -> tuple[
        _DecisionsAPIStub, _CompletionsAPIStub, list[str], list[tuple[str, bool]]
    ]:
        """
        Extracts the aspects of a document with a decision model that has fallback
        models, with the given outcomes of the requests to the models, and records
        the extractions that the fallback models are asked to make.

        :param decision_model: The decision model.
        :type decision_model: DocumentDecisionModel
        :param document: The document.
        :type document: Document
        :param decisions_outcomes: The outcomes of the requests to each decision
            model (see :class:`_DecisionsAPIStub`).
        :type decisions_outcomes: dict[str, list[Any]]
        :param completions_outcomes: The outcomes of the requests to each LLM (see
            :class:`_CompletionsAPIStub`). Defaults to None, i.e. no LLM is asked.
        :type completions_outcomes: dict[str, list[Any]] | None
        :param extraction_kwargs: Other params of the extraction method.
        :type extraction_kwargs: Any
        :return: The stubs with the requests received, the messages logged at
            the INFO level and above, and the extractions made by models other
            than the decision model, each as the model and whether it was made
            with concurrency, in the order they were made.
        :rtype: tuple[_DecisionsAPIStub, _CompletionsAPIStub, list[str],
            list[tuple[str, bool]]]
        """
        decisions_stub = _DecisionsAPIStub(decisions_outcomes)
        completions_stub = _CompletionsAPIStub(completions_outcomes or {})
        original_extract = (
            models_module._DocumentModel.extract_aspects_from_document_async
        )
        fallback_extractions: list[tuple[str, bool]] = []

        async def record_extraction(model: Any, *args: Any, **kwargs: Any) -> Any:
            """
            Records an extraction of aspects made by a model other than
            the decision model, and makes it.

            :param model: The model that extracts the aspects.
            :type model: Any
            :param args: Arguments of the extraction method.
            :type args: Any
            :param kwargs: Keyword arguments of the extraction method.
            :type kwargs: Any
            :return: The aspects extracted.
            :rtype: Any
            """
            if model is not decision_model:
                fallback_extractions.append(
                    (model.model, kwargs.get("use_concurrency", False))
                )
            return await original_extract(model, *args, **kwargs)

        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            patch.object(litellm, "acompletion", new=completions_stub),
            patch.object(
                models_module._DocumentModel,
                "extract_aspects_from_document_async",
                new=record_extraction,
            ),
            capture_logs() as captured_logs,
        ):
            decision_model.extract_aspects_from_document(document, **extraction_kwargs)
        return decisions_stub, completions_stub, captured_logs, fallback_extractions

    @staticmethod
    def _create_refusals_document() -> Document:
        """
        Creates the document created by :func:`_create_aspects_document`, with
        the aspects `Termination` and `Payment` to extract.

        :return: Document.
        :rtype: Document
        """
        return _create_aspects_document(
            aspects=[
                Aspect(name=name, description=f"Clauses on {name.lower()}")
                for name in ["Termination", "Payment"]
            ]
        )

    @staticmethod
    def _get_aspect_request_questions(aspect_names: list[str]) -> list[dict[str, Any]]:
        """
        Returns the questions expected in a request on a paragraph without
        additional context for the aspects with the given names, created as by
        :meth:`_create_refusals_document`, in the given order.

        :param aspect_names: The names of the aspects.
        :type aspect_names: list[str]
        :return: The questions of the request.
        :rtype: list[dict[str, Any]]
        """
        return [
            {
                "type": "predicate",
                "name": question.name,
                "instructions": question.instructions,
            }
            for question in _build_aspect_questions(
                [
                    Aspect(name=name, description=f"Clauses on {name.lower()}")
                    for name in aspect_names
                ]
            )
        ]

    @staticmethod
    def _get_refusal_fallback_logs(captured_logs: list[str]) -> list[str]:
        """
        Returns the messages logged on the fallback models asked about what
        a decision model refused to answer on, and on what they decided.

        :param captured_logs: The messages logged.
        :type captured_logs: list[str]
        :return: The messages on the fallback models, in the order logged.
        :rtype: list[str]
        """
        return [
            log
            for log in captured_logs
            if log.startswith(("Asking fallback ", "Fallback "))
        ]

    @staticmethod
    def _get_asking_fallback_log(
        fallback_kind: str,
        fallback_model: str,
        aspect_names: list[str],
        n_paragraphs: int,
    ) -> str:
        """
        Returns the message expected before a fallback model is asked about what
        the decision model used in the tests refused to answer on.

        :param fallback_kind: The kind of the fallback model, e.g. "LLM".
        :type fallback_kind: str
        :param fallback_model: The fallback model.
        :type fallback_model: str
        :param aspect_names: The names of the aspects asked about.
        :type aspect_names: list[str]
        :param n_paragraphs: The number of the paragraphs asked about.
        :type n_paragraphs: int
        :return: The message.
        :rtype: str
        """
        return (
            f"Asking fallback {fallback_kind} `{fallback_model}` about what decision "
            f"model `{_DECISION_MODEL}` refused to answer on for aspects "
            f"{aspect_names}, on {n_paragraphs} paragraph(s)"
        )

    @staticmethod
    def _get_fallback_outcome_log(
        fallback_kind: str,
        fallback_model: str,
        aspect_names: list[str],
        decided: bool = True,
    ) -> str:
        """
        Returns the message expected after a fallback model is asked about what
        the decision model used in the tests refused to answer on.

        :param fallback_kind: The kind of the fallback model, e.g. "LLM".
        :type fallback_kind: str
        :param fallback_model: The fallback model.
        :type fallback_model: str
        :param aspect_names: The names of the aspects that the message names.
        :type aspect_names: list[str]
        :param decided: Whether the fallback model decided what it was asked about
            for these aspects, or failed. Defaults to True.
        :type decided: bool
        :return: The message.
        :rtype: str
        """
        outcome = "decided" if decided else "did not decide"
        return (
            f"Fallback {fallback_kind} `{fallback_model}` {outcome} what decision "
            f"model `{_DECISION_MODEL}` refused to answer on for aspects "
            f"{aspect_names}."
        )

    @staticmethod
    def _get_refusal_warnings(captured_logs: list[str]) -> list[str]:
        """
        Returns the warnings logged on what is not included in an aspect, as
        a decision model, or its fallback model, refused to answer on it.

        :param captured_logs: The messages logged.
        :type captured_logs: list[str]
        :return: The warnings, in the order logged.
        :rtype: list[str]
        """
        return [
            log
            for log in captured_logs
            if "These are not included in the aspect" in log
        ]

    def _check_aspect_items(
        self,
        document: Document,
        aspect: _Aspect,
        expected_items: list[tuple[int | tuple[int, int], float | None]],
    ) -> None:
        """
        Checks the items of an aspect of the document, in their order: the text
        of each, its paragraph, or its sentence and the paragraph of it, as
        objects of the document, and its probabilities.

        :param document: The document.
        :type document: Document
        :param aspect: The aspect of the document.
        :type aspect: _Aspect
        :param expected_items: For each item, the position of its paragraph in
            the document, or the positions of its paragraph and of its sentence in
            it, and the probability of "yes", or None for an item without
            probabilities, i.e. extracted by an LLM.
        :type expected_items: list[tuple[int | tuple[int, int], float | None]]
        """
        assert len(aspect.extracted_items) == len(expected_items)
        for item, (position, probability) in zip(
            aspect.extracted_items, expected_items, strict=True
        ):
            if isinstance(position, int):
                paragraph = document.paragraphs[position]
                assert item.value == paragraph.raw_text
                assert item.reference_sentences == []
            else:
                paragraph = document.paragraphs[position[0]]
                sentence = paragraph.sentences[position[1]]
                assert item.value == sentence.raw_text
                (reference_sentence,) = item.reference_sentences
                assert reference_sentence is sentence
            (reference_paragraph,) = item.reference_paragraphs
            assert reference_paragraph is paragraph
            self.check_item_probabilities(
                item,
                None
                if probability is None
                else {True: probability, False: 1 - probability},
            )

    def test_decision_model_fallback_decides_refused_paragraphs(self):
        """
        Tests that the fallback decision model of a decision model that refused to
        answer the question of an aspect on some paragraphs, in a call that
        succeeds, is asked about these paragraphs, with and without concurrency:
        it is sent these paragraphs only, with the questions of the aspects
        refused on them, its decisions on them are added to the items of
        the aspect, in the order of the paragraphs, with its probabilities,
        the other items are those of the decision model, its requests are in its
        usage, and no warning on refused answers is logged.
        """
        texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        # The decision model refuses to answer about the second aspect on the first
        # and the last paragraphs, which the fallback model decides
        primary_probabilities: dict[str, Any] = {
            "Termination": {2: 0.9},
            "Payment": {0: "refusal", 3: 0.8, 4: "refusal"},
        }
        fallback_probabilities = {"Payment": {0: 0.2, 4: 0.7}}
        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            decision_model = _create_decision_model(
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ]
            )
            document = self._create_refusals_document()
            decisions_stub, _, captured_logs, fallback_extractions = (
                self._extract_with_refusal_fallbacks(
                    decision_model,
                    document,
                    {
                        _DECISION_MODEL: [
                            _create_aspects_responder(primary_probabilities)
                        ],
                        _OTHER_DECISION_MODEL: [
                            _create_aspects_responder(fallback_probabilities)
                        ],
                    },
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            )
            # The decision model is asked about each paragraph, and then its
            # fallback model about the paragraphs refused, with the question of
            # the aspect refused only, which is the first one of its requests
            assert [
                (request["model"], request["input"])
                for request in decisions_stub.requests
            ] == [
                *[(_DECISION_MODEL, text) for text in texts],
                (_OTHER_DECISION_MODEL, texts[0]),
                (_OTHER_DECISION_MODEL, texts[4]),
            ]
            assert decisions_stub.requests[0]["questions"] == (
                self._get_aspect_request_questions(["Termination", "Payment"])
            )
            for request in decisions_stub.requests[len(texts) :]:
                assert request["questions"] == self._get_aspect_request_questions(
                    ["Payment"]
                )
            # ... with concurrency if the call has it
            assert fallback_extractions == [
                (_OTHER_DECISION_MODEL, bool(extraction_kwargs))
            ]
            # The decisions of the fallback model are added to those of
            # the decision model, in the order of the paragraphs
            termination, payment = document.aspects
            self._check_aspect_items(document, termination, [(2, 0.9)])
            self._check_aspect_items(document, payment, [(3, 0.8), (4, 0.7)])
            assert payment.reference_paragraphs == [
                document.paragraphs[3],
                document.paragraphs[4],
            ]
            # The usage of each model has its own requests
            primary_usage, fallback_usage = (
                container.usage for container in decision_model.get_usage()
            )
            assert len(primary_usage.calls) == len(texts)
            assert [
                json.loads(call.prompt)["input"] for call in fallback_usage.calls
            ] == [
                texts[0],
                texts[4],
            ]
            assert fallback_usage.input == 2 * 100
            assert self._get_refusal_fallback_logs(captured_logs) == [
                self._get_asking_fallback_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Payment"], 2
                ),
                self._get_fallback_outcome_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Payment"]
                ),
            ]
            assert self._get_refusal_warnings(captured_logs) == []

    def test_llm_fallback_decides_refused_paragraphs(self):
        """
        Tests that the fallback LLM of a decision model that refused to answer
        the questions of aspects on some paragraphs, in a call that succeeds,
        extracts the aspects from these paragraphs only: its items on what was
        refused for an aspect are added to the items of the aspect, in the order
        of the paragraphs, without probabilities, and with the paragraphs of
        the document as their references, while its items on what the decision
        model decided are ignored.
        """
        # The decision model refuses to answer about each aspect on another
        # paragraph
        primary_probabilities: dict[str, Any] = {
            "Termination": {1: "refusal", 2: 0.9},
            "Payment": {3: 0.8, 4: "refusal"},
        }
        # The LLM answers that both paragraphs that it is asked about belong to
        # both aspects
        llm_answer = json.dumps(
            [
                {"aspect_id": "A0", "paragraph_ids": ["P0", "P1"]},
                {"aspect_id": "A1", "paragraph_ids": ["P0", "P1"]},
            ]
        )
        decision_model = _create_decision_model(
            fallback_models=[_create_llm(is_fallback=True)]
        )
        document = self._create_refusals_document()
        decisions_stub, completions_stub, captured_logs, fallback_extractions = (
            self._extract_with_refusal_fallbacks(
                decision_model,
                document,
                {_DECISION_MODEL: [_create_aspects_responder(primary_probabilities)]},
                {_LLM_MODEL: [llm_answer]},
            )
        )
        assert len(decisions_stub.requests) == len(_ASPECTS_PARAGRAPH_TEXTS_SENT)
        assert fallback_extractions == [(_LLM_MODEL, False)]
        # The prompt lists the two paragraphs refused, for the two aspects
        (llm_request,) = completions_stub.requests
        prompt = "\n".join(
            str(message["content"]) for message in llm_request["messages"]
        )
        assert re.findall(r'<paragraph id="(P\d+)">(.*?)</paragraph>', prompt) == [
            ("P0", _ASPECTS_PARAGRAPH_TEXTS_CLEANED[1]),
            ("P1", _ASPECTS_PARAGRAPH_TEXTS_CLEANED[4]),
        ]
        assert re.findall(
            r"<aspect_name>(.*?)</aspect_name>", prompt.split("DOCUMENT ASPECTS:")[1]
        ) == ["Termination", "Payment"]
        # Only the items on the paragraph refused for each aspect are taken
        termination, payment = document.aspects
        self._check_aspect_items(document, termination, [(1, None), (2, 0.9)])
        self._check_aspect_items(document, payment, [(3, 0.8), (4, None)])
        (_, llm_usage_container) = decision_model.get_usage()
        assert llm_usage_container.model == _LLM_MODEL
        assert len(llm_usage_container.usage.calls) == 1
        assert self._get_refusal_fallback_logs(captured_logs) == [
            self._get_asking_fallback_log(
                "LLM", _LLM_MODEL, ["Termination", "Payment"], 2
            ),
            self._get_fallback_outcome_log(
                "LLM", _LLM_MODEL, ["Termination", "Payment"]
            ),
        ]
        assert self._get_refusal_warnings(captured_logs) == []

    def test_decision_model_refusal_fallback_chain(self):
        """
        Tests a chain of fallback models asked about what a decision model refused
        to answer on, in a call that succeeds: what a fallback decision model
        refuses to answer on in turn is passed on to the next fallback model, and
        what none of them decides is not included in the aspect, with a warning
        that tells it. The call stands, i.e. nothing is raised, also if every
        fallback model fails or refuses.
        """
        texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        # The decision model refuses to answer about the second aspect on three of
        # the paragraphs
        primary_probabilities: dict[str, Any] = {
            "Termination": {2: 0.9},
            "Payment": {0: "refusal", 1: 0.8, 3: "refusal", 4: "refusal"},
        }
        refused_positions = [0, 3, 4]
        # The fallback decision model refuses to answer on the first of them in turn
        partly_refusing_probabilities: dict[str, Any] = {
            "Payment": {0: "refusal", 3: 0.6, 4: 0.2}
        }
        llm_answer = json.dumps([{"aspect_id": "A0", "paragraph_ids": ["P0"]}])
        invalid_llm_answer = "<<<not-json>>>"

        def extract(
            fallback_decision_outcome: Any, llm_outcome: str
        ) -> tuple[Document, _DecisionsAPIStub, _CompletionsAPIStub, list[str]]:
            """
            Extracts the aspects with the decision model, which has a fallback
            decision model and then a fallback LLM, each asked again once for
            a response that cannot be used.

            :param fallback_decision_outcome: The outcome of each request to
                the fallback decision model.
            :type fallback_decision_outcome: Any
            :param llm_outcome: The outcome of each request to the fallback LLM.
            :type llm_outcome: str
            :return: The document, the stubs with the requests received, and
                the messages logged.
            :rtype: tuple[Document, _DecisionsAPIStub, _CompletionsAPIStub,
                list[str]]
            """
            decision_model = _create_decision_model(
                fallback_models=[
                    _create_decision_model(
                        _OTHER_DECISION_MODEL,
                        is_fallback=True,
                        max_retries_invalid_data=1,
                    ),
                    _create_llm(is_fallback=True, max_retries_invalid_data=1),
                ]
            )
            document = self._create_refusals_document()
            decisions_stub, completions_stub, captured_logs, _ = (
                self._extract_with_refusal_fallbacks(
                    decision_model,
                    document,
                    {
                        _DECISION_MODEL: [
                            _create_aspects_responder(primary_probabilities)
                        ],
                        _OTHER_DECISION_MODEL: [fallback_decision_outcome],
                    },
                    {_LLM_MODEL: [llm_outcome]},
                    raise_exception_on_extraction_error=True,
                )
            )
            # The items of the decision model stand in any case
            termination, payment = document.aspects
            assert termination._is_processed and payment._is_processed
            self._check_aspect_items(document, termination, [(2, 0.9)])
            return document, decisions_stub, completions_stub, captured_logs

        def get_fallback_inputs(decisions_stub: _DecisionsAPIStub) -> list[str]:
            """
            Returns the texts of the requests to the fallback decision model.

            :param decisions_stub: The stub with the requests received.
            :type decisions_stub: _DecisionsAPIStub
            :return: The texts, in the order of the requests.
            :rtype: list[str]
            """
            return [
                request["input"]
                for request in decisions_stub.requests
                if request["model"] == _OTHER_DECISION_MODEL
            ]

        def get_prompt_paragraphs(request: dict[str, Any]) -> list[tuple[str, str]]:
            """
            Returns the paragraphs listed in the prompt of a request to the LLM,
            each by its ID in the prompt, with its text.

            :param request: Keyword arguments of the request.
            :type request: dict[str, Any]
            :return: The IDs and the texts of the paragraphs.
            :rtype: list[tuple[str, str]]
            """
            prompt = "\n".join(
                str(message["content"]) for message in request["messages"]
            )
            return re.findall(r'<paragraph id="(P\d+)">(.*?)</paragraph>', prompt)

        asking_fallback_decision_model_log = self._get_asking_fallback_log(
            "decision model", _OTHER_DECISION_MODEL, ["Payment"], 3
        )

        # === The second fallback model decides what the first one refused ===
        document, decisions_stub, completions_stub, captured_logs = extract(
            _create_aspects_responder(partly_refusing_probabilities), llm_answer
        )
        assert get_fallback_inputs(decisions_stub) == [
            texts[position] for position in refused_positions
        ]
        (llm_request,) = completions_stub.requests
        assert get_prompt_paragraphs(llm_request) == [
            ("P0", _ASPECTS_PARAGRAPH_TEXTS_CLEANED[0])
        ]
        # The items of the three models, in the order of the paragraphs
        self._check_aspect_items(
            document, document.aspects[1], [(0, None), (1, 0.8), (3, 0.6)]
        )
        assert self._get_refusal_fallback_logs(captured_logs) == [
            asking_fallback_decision_model_log,
            self._get_asking_fallback_log("LLM", _LLM_MODEL, ["Payment"], 1),
            self._get_fallback_outcome_log("LLM", _LLM_MODEL, ["Payment"]),
        ]
        # No warning, also of the fallback decision model on what it refused
        assert self._get_refusal_warnings(captured_logs) == []

        # === The second fallback model fails ===
        # What the first one refused stays undecided
        document, decisions_stub, completions_stub, captured_logs = extract(
            _create_aspects_responder(partly_refusing_probabilities),
            invalid_llm_answer,
        )
        # The LLM is asked again once, for its invalid data
        assert len(completions_stub.requests) == 2
        self._check_aspect_items(document, document.aspects[1], [(1, 0.8), (3, 0.6)])
        assert self._get_refusal_fallback_logs(captured_logs) == [
            asking_fallback_decision_model_log,
            self._get_asking_fallback_log("LLM", _LLM_MODEL, ["Payment"], 1),
            self._get_fallback_outcome_log(
                "LLM", _LLM_MODEL, ["Payment"], decided=False
            ),
        ]
        assert self._get_refusal_warnings(captured_logs) == [
            (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question of "
                "aspect `Payment` on 3 of 5 paragraphs (indices: [0, 3, 4]), and its "
                "fallback models did not answer on the paragraphs with indices [0] "
                "either. These are not included in the aspect."
            )
        ]

        # === Each fallback model fails or refuses ===
        # The fallback decision model refuses to answer on every paragraph, also
        # when asked again, and the fallback LLM returns invalid data
        refusing_probabilities: dict[str, Any] = {
            "Payment": dict.fromkeys(refused_positions, "refusal")
        }
        document, decisions_stub, completions_stub, captured_logs = extract(
            _create_aspects_responder(refusing_probabilities), invalid_llm_answer
        )
        assert (
            get_fallback_inputs(decisions_stub)
            == [texts[position] for position in refused_positions] * 2
        )
        assert len(completions_stub.requests) == 2
        # The LLM is asked about each paragraph refused
        assert get_prompt_paragraphs(completions_stub.requests[0]) == [
            ("P0", _ASPECTS_PARAGRAPH_TEXTS_CLEANED[0]),
            ("P1", _ASPECTS_PARAGRAPH_TEXTS_CLEANED[3]),
            ("P2", _ASPECTS_PARAGRAPH_TEXTS_CLEANED[4]),
        ]
        self._check_aspect_items(document, document.aspects[1], [(1, 0.8)])
        assert self._get_refusal_fallback_logs(captured_logs) == [
            asking_fallback_decision_model_log,
            self._get_fallback_outcome_log(
                "decision model", _OTHER_DECISION_MODEL, ["Payment"], decided=False
            ),
            self._get_asking_fallback_log("LLM", _LLM_MODEL, ["Payment"], 3),
            self._get_fallback_outcome_log(
                "LLM", _LLM_MODEL, ["Payment"], decided=False
            ),
        ]
        assert self._get_refusal_warnings(captured_logs) == [
            (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question of "
                "aspect `Payment` on 3 of 5 paragraphs (indices: [0, 3, 4]), and its "
                "fallback models did not answer on the paragraphs with indices "
                "[0, 3, 4] either. These are not included in the aspect."
            )
        ]

    def test_failed_fallback_decides_nothing_refused(self):
        """
        Tests that a fallback model whose request fails, asked about what
        a decision model refused to answer on in a call that succeeds, decides
        nothing: the next fallback model is asked, and without one, the warning
        tells that everything refused stays undecided. The call stands, i.e.
        nothing is raised.
        """
        texts = _ASPECTS_PARAGRAPH_TEXTS_SENT
        primary_probabilities: dict[str, Any] = {
            "Termination": {2: 0.9},
            "Payment": {0: "refusal", 3: 0.8, 4: "refusal"},
        }
        other_auth_error = _create_api_error(
            litellm.AuthenticationError, _OTHER_DECISION_MODEL
        )
        failed_fallback_logs = [
            self._get_asking_fallback_log(
                "decision model", _OTHER_DECISION_MODEL, ["Payment"], 2
            ),
            self._get_fallback_outcome_log(
                "decision model", _OTHER_DECISION_MODEL, ["Payment"], decided=False
            ),
        ]

        # === The next fallback model decides ===
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True),
                _create_llm(is_fallback=True),
            ]
        )
        document = self._create_refusals_document()
        decisions_stub, completions_stub, captured_logs, fallback_extractions = (
            self._extract_with_refusal_fallbacks(
                decision_model,
                document,
                {
                    _DECISION_MODEL: [_create_aspects_responder(primary_probabilities)],
                    _OTHER_DECISION_MODEL: [other_auth_error],
                },
                {
                    _LLM_MODEL: [
                        json.dumps([{"aspect_id": "A0", "paragraph_ids": ["P1"]}])
                    ]
                },
            )
        )
        # The failed request is not sent again, and the requests after it are not
        # sent
        assert decisions_stub.requested_models() == [_DECISION_MODEL] * len(texts) + [
            _OTHER_DECISION_MODEL
        ]
        assert decisions_stub.requests[-1]["input"] == texts[0]
        assert len(completions_stub.requests) == 1
        assert fallback_extractions == [
            (_OTHER_DECISION_MODEL, False),
            (_LLM_MODEL, False),
        ]
        termination, payment = document.aspects
        self._check_aspect_items(document, termination, [(2, 0.9)])
        self._check_aspect_items(document, payment, [(3, 0.8), (4, None)])
        # The failed request is in the usage of the fallback decision model
        (failed_call,) = decision_model.get_usage()[1].usage.calls
        assert failed_call.timestamp_received is None
        assert self._get_refusal_fallback_logs(captured_logs) == [
            *failed_fallback_logs,
            self._get_asking_fallback_log("LLM", _LLM_MODEL, ["Payment"], 2),
            self._get_fallback_outcome_log("LLM", _LLM_MODEL, ["Payment"]),
        ]
        assert self._get_refusal_warnings(captured_logs) == []

        # === No next fallback model ===
        decision_model = _create_decision_model(
            fallback_models=[
                _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
            ]
        )
        document = self._create_refusals_document()
        _, _, captured_logs, _ = self._extract_with_refusal_fallbacks(
            decision_model,
            document,
            {
                _DECISION_MODEL: [_create_aspects_responder(primary_probabilities)],
                _OTHER_DECISION_MODEL: [other_auth_error],
            },
            raise_exception_on_extraction_error=True,
        )
        termination, payment = document.aspects
        self._check_aspect_items(document, termination, [(2, 0.9)])
        self._check_aspect_items(document, payment, [(3, 0.8)])
        assert self._get_refusal_fallback_logs(captured_logs) == failed_fallback_logs
        assert self._get_refusal_warnings(captured_logs) == [
            (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question of "
                "aspect `Payment` on 2 of 5 paragraphs (indices: [0, 4]), and its "
                "fallback models did not answer on the paragraphs with indices [0, 4] "
                "either. These are not included in the aspect."
            )
        ]

    def test_fallback_decides_refused_sentences(self):
        """
        Tests that the fallback decision model of a decision model that extracts
        aspects with sentences as their items is asked about what the decision
        model refused to answer on, with and without concurrency: about
        a paragraph refused, with its sentences if the paragraph belongs to
        the aspect, and about a sentence refused of a paragraph that belongs to
        the aspect, which is not included if the fallback model answers that its
        paragraph does not belong to the aspect. What the fallback model refuses
        to answer on in turn is not included, with a warning that tells it, in
        the order of the document.
        """

        def get_sentence_texts(
            positions: list[int | tuple[int, int]],
        ) -> list[str]:
            """
            Returns the texts of the requests for paragraphs or for sentences.

            :param positions: The position of each paragraph, or the positions of
                each paragraph and of the sentence in it.
            :type positions: list[int | tuple[int, int]]
            :return: The texts of the requests.
            :rtype: list[str]
            """
            return [
                _SENTENCES_PARAGRAPH_TEXTS_SENT[position]
                if isinstance(position, int)
                else _SENTENCES_TEXTS_SENT[position]
                for position in positions
            ]

        def extract(
            primary_probabilities: dict[str, Any],
            fallback_probabilities: dict[str, Any],
            extraction_kwargs: dict[str, Any],
        ) -> tuple[Document, _DecisionsAPIStub, list[str]]:
            """
            Extracts an aspect with sentences as its items with a decision model
            that has a fallback decision model.

            :param primary_probabilities: The answers of the decision model (see
                :func:`_create_sentences_responder`).
            :type primary_probabilities: dict[str, Any]
            :param fallback_probabilities: The answers of the fallback model.
            :type fallback_probabilities: dict[str, Any]
            :param extraction_kwargs: Params of the extraction method.
            :type extraction_kwargs: dict[str, Any]
            :return: The document, the stub with the requests received, and
                the messages logged.
            :rtype: tuple[Document, _DecisionsAPIStub, list[str]]
            """
            decision_model = _create_decision_model(
                fallback_models=[
                    _create_decision_model(_OTHER_DECISION_MODEL, is_fallback=True)
                ]
            )
            document = _create_sentences_document(
                aspects=[
                    Aspect(
                        name="Termination",
                        description="Clauses on termination",
                        reference_depth="sentences",
                    )
                ]
            )
            decisions_stub, _, captured_logs, fallback_extractions = (
                self._extract_with_refusal_fallbacks(
                    decision_model,
                    document,
                    {
                        _DECISION_MODEL: [
                            _create_sentences_responder(primary_probabilities)
                        ],
                        _OTHER_DECISION_MODEL: [
                            _create_sentences_responder(fallback_probabilities)
                        ],
                    },
                    **extraction_kwargs,
                )
            )
            assert fallback_extractions == [
                (_OTHER_DECISION_MODEL, bool(extraction_kwargs))
            ]
            # The questions of the fallback model are those of the aspect
            for request in decisions_stub.requests:
                assert [question["name"] for question in request["questions"]] == ["A0"]
            return document, decisions_stub, captured_logs

        def get_inputs(decisions_stub: _DecisionsAPIStub, model: str) -> list[str]:
            """
            Returns the texts of the requests to a decision model.

            :param decisions_stub: The stub with the requests received.
            :type decisions_stub: _DecisionsAPIStub
            :param model: The decision model.
            :type model: str
            :return: The texts, in the order of the requests.
            :rtype: list[str]
            """
            return [
                request["input"]
                for request in decisions_stub.requests
                if request["model"] == model
            ]

        for extraction_kwargs in [
            {},
            {"use_concurrency": True, "max_items_per_call": 2},
        ]:
            # === A paragraph refused ===
            # The fallback model answers that it belongs to the aspect, and is
            # asked about its sentences
            document, decisions_stub, captured_logs = extract(
                {"Termination": {1: "refusal", 3: 0.9, (3, 0): 0.8}},
                {"Termination": {1: 0.9, (1, 0): 0.9, (1, 1): 0.2, (1, 2): 0.7}},
                extraction_kwargs,
            )
            assert get_inputs(decisions_stub, _DECISION_MODEL) == get_sentence_texts(
                [0, 1, 2, 3, 4, (3, 0), (3, 1)]
            )
            assert get_inputs(
                decisions_stub, _OTHER_DECISION_MODEL
            ) == get_sentence_texts([1, (1, 0), (1, 1), (1, 2)])
            (termination,) = document.aspects
            self._check_aspect_items(
                document, termination, [((1, 0), 0.9), ((1, 2), 0.7), ((3, 0), 0.8)]
            )
            assert termination.reference_paragraphs == [
                document.paragraphs[1],
                document.paragraphs[3],
            ]
            assert self._get_refusal_fallback_logs(captured_logs) == [
                self._get_asking_fallback_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Termination"], 1
                ),
                self._get_fallback_outcome_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Termination"]
                ),
            ]
            assert self._get_refusal_warnings(captured_logs) == []

            # === Sentences refused of paragraphs that belong to the aspect ===
            # The fallback model answers about the sentence refused of the second
            # paragraph, of whose other sentences its answers are ignored, and
            # answers that the fourth paragraph does not belong to the aspect,
            # i.e. its sentence refused does not either
            document, decisions_stub, captured_logs = extract(
                {
                    "Termination": {
                        1: 0.9,
                        (1, 0): 0.8,
                        (1, 1): "refusal",
                        3: 0.9,
                        (3, 1): "refusal",
                    }
                },
                {"Termination": {1: 0.9, (1, 1): 0.85, (1, 2): 0.9, 3: 0.2}},
                extraction_kwargs,
            )
            assert get_inputs(decisions_stub, _DECISION_MODEL) == get_sentence_texts(
                [0, 1, 2, 3, 4, (1, 0), (1, 1), (1, 2), (3, 0), (3, 1)]
            )
            assert get_inputs(
                decisions_stub, _OTHER_DECISION_MODEL
            ) == get_sentence_texts([1, 3, (1, 0), (1, 1), (1, 2)])
            (termination,) = document.aspects
            self._check_aspect_items(
                document, termination, [((1, 0), 0.8), ((1, 1), 0.85)]
            )
            assert self._get_refusal_fallback_logs(captured_logs) == [
                self._get_asking_fallback_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Termination"], 2
                ),
                self._get_fallback_outcome_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Termination"]
                ),
            ]
            assert self._get_refusal_warnings(captured_logs) == []

            # === What the fallback model refuses in turn ===
            # It refuses to answer on the paragraph that the decision model refused,
            # on a sentence that the decision model refused, and on the paragraph
            # of another one, which are not included in the aspect
            document, decisions_stub, captured_logs = extract(
                {
                    "Termination": {
                        0: "refusal",
                        1: 0.9,
                        (1, 0): 0.8,
                        (1, 1): "refusal",
                        3: 0.9,
                        (3, 1): "refusal",
                    }
                },
                {
                    "Termination": {
                        0: "refusal",
                        1: 0.9,
                        (1, 1): "refusal",
                        3: "refusal",
                    }
                },
                extraction_kwargs,
            )
            assert get_inputs(
                decisions_stub, _OTHER_DECISION_MODEL
            ) == get_sentence_texts([0, 1, 3, (1, 0), (1, 1), (1, 2)])
            (termination,) = document.aspects
            self._check_aspect_items(document, termination, [((1, 0), 0.8)])
            assert self._get_refusal_fallback_logs(captured_logs) == [
                self._get_asking_fallback_log(
                    "decision model", _OTHER_DECISION_MODEL, ["Termination"], 3
                ),
            ]
            assert self._get_refusal_warnings(captured_logs) == [
                (
                    f"Decision model `{_DECISION_MODEL}` refused to answer the question "
                    "of aspect `Termination` on 1 of 5 paragraphs (indices: [0]) and on "
                    "2 of 5 sentences (paragraph index, sentence index: [(1, 1), "
                    "(3, 1)]), and its fallback models did not answer on the paragraphs "
                    "with indices [0] and on the sentences with (paragraph index, "
                    "sentence index) [(1, 1), (3, 1)] either. These are not included in "
                    "the aspect."
                )
            ]

        # === The order of what stays undecided ===
        # A sentence that the fallback model refuses of a paragraph that
        # the decision model refused comes, among what stays undecided, in
        # the order of the sentences in the document
        document, decisions_stub, captured_logs = extract(
            {
                "Termination": {
                    1: 0.9,
                    (1, 0): 0.8,
                    (1, 1): "refusal",
                    3: "refusal",
                }
            },
            {
                "Termination": {
                    1: 0.9,
                    (1, 1): "refusal",
                    3: 0.9,
                    (3, 0): "refusal",
                }
            },
            {},
        )
        assert get_inputs(decisions_stub, _OTHER_DECISION_MODEL) == get_sentence_texts(
            [1, 3, (1, 0), (1, 1), (1, 2), (3, 0), (3, 1)]
        )
        (termination,) = document.aspects
        self._check_aspect_items(document, termination, [((1, 0), 0.8)])
        assert self._get_refusal_warnings(captured_logs) == [
            (
                f"Decision model `{_DECISION_MODEL}` refused to answer the question "
                "of aspect `Termination` on 1 of 5 paragraphs (indices: [3]) and on "
                "1 of 3 sentences (paragraph index, sentence index: [(1, 1)]), and "
                "its fallback models did not answer on the sentences with "
                "(paragraph index, sentence index) [(1, 1), (3, 0)] either. These "
                "are not included in the aspect."
            )
        ]
