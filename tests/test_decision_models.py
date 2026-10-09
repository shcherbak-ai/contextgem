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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter
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

from contextgem.internal.base import decision_models as decision_models_module
from contextgem.internal.base import models as models_module
from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.decision_models import _DocumentDecisionModel
from contextgem.internal.base.llms import _DocumentLLM
from contextgem.internal.base.models import _COST_QUANT
from contextgem.internal.base.paras_and_sents import _Paragraph
from contextgem.internal.data_models import _ModelCost, _ModelUsage
from contextgem.internal.decisions import (
    _build_aspect_questions,
    _build_concept_questions,
    _convert_aspect_answer,
    _convert_concept_answers,
    _DecisionAnswer,
    _DecisionOption,
    _DecisionQuestion,
    _validate_answers,
)
from contextgem.internal.exceptions import (
    ModelAPIError,
    ModelExtractionError,
    _InvalidDecisionResponseError,
)
from contextgem.internal.items import (
    _BooleanItem,
    _IntegerItem,
    _LabelItem,
    _StringItem,
)
from contextgem.internal.loggers import logger
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
    StringConcept,
    register_tool,
)
from tests.base import BaseTest


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
]

# Models used in the tests. No requests are sent to them.
_DECISION_MODEL = "typesafe/jev-latest"
_OTHER_DECISION_MODEL = "typesafe/jev-1.13.0"
_LLM_MODEL = "openai/gpt-6-luna"
_OTHER_LLM_MODEL = "openai/gpt-5.6-luna"

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
# of each paragraph, in a request of its own
_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS = [
    "Non-disclosure  agreement",
    "The receiving party shall keep the information confidential.",
    "Either party may terminate this agreement with a written notice of 30 days.",
    "The liability of each party is limited to the fees paid.",
    "A notice of termination does not release a party from confidentiality.",
]
_ASPECTS_PARAGRAPH_TEXTS_SENT = [
    "Non-disclosure agreement",
    *_ASPECTS_DOCUMENT_PARAGRAPH_TEXTS[1:],
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

    An outcome is an exception to raise, or the text of the response to return.

    :ivar requests: The requests received, as the keyword arguments of the calls.
    :vartype requests: list[dict[str, Any]]
    """

    def __init__(self, outcomes: dict[str, list[Exception | str]]):
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
    aspect.extracted_items = items
    aspect.reference_paragraphs = paragraphs
    aspect._is_processed = True


@contextmanager
def _capture_logs(level: int = stdlib_logging.INFO) -> Iterator[list[str]]:
    """
    Captures the messages logged by the framework at the given level and above.

    :param level: The lowest level of the messages to capture. Defaults to INFO.
    :type level: int
    :return: Context manager that gives the list the messages are added to.
    :rtype: Iterator[list[str]]
    """
    captured_logs: list[str] = []

    class CaptureHandler(stdlib_logging.Handler):
        """
        Handler to capture log messages for testing.
        """

        def emit(self, record: stdlib_logging.LogRecord) -> None:
            """
            Adds the log message to the captured ones.

            :param record: The log record.
            :type record: stdlib_logging.LogRecord
            """
            captured_logs.append(record.getMessage())

    capture_handler = CaptureHandler()
    capture_handler.setLevel(level)
    logger.addHandler(capture_handler)
    try:
        yield captured_logs
    finally:
        logger.removeHandler(capture_handler)


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
        its Decisions and chat completions APIs are replaced with a failure here.
        The function of litellm that prepares each request to its Decisions API is
        replaced as well, i.e. a request cannot be prepared even if the framework
        gets to litellm in another way than through the calls replaced.

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
        for name in ["adecisions", "decisions", "acompletion", "completion"]:
            monkeypatch.setattr(litellm, name, create_failure(f"litellm.{name}"))
        # `litellm.decisions` is the function, which shadows the package of
        # the same name, i.e. its module is reached through `sys.modules`
        monkeypatch.setattr(
            sys.modules["litellm.decisions.main"],
            "_prepare_request",
            create_failure("litellm.decisions.main._prepare_request"),
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
            (
                {"written": 0.0, "email": 0.0, "any": 0.0, "not specified": 0.0},
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
        one paragraph, including the probability of exactly 0.5.
        """
        for probability, expected_belongs in [
            (0.93, True),
            (0.5, True),
            (0.4999, False),
            (0.0, False),
            (1.0, True),
            (1, True),
        ]:
            belongs, probabilities = _convert_aspect_answer(
                0, {"A0": _predicate_answer("A0", probability=probability)}
            )
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

    def test_invalid_predicate_answers(self):
        """
        Tests that an answer to a predicate that cannot be used is rejected as
        an invalid response, for a Boolean concept, for a label of a label concept
        and for an aspect.
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
            # Refused
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name, refused=True)},
                name,
                "the model refused to answer",
            )
            # Refused, even if answered as well
            _check_invalid_answer(
                convert,
                {name: _predicate_answer(name, probability=0.9, refused=True)},
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
                    r"is not a number within \[0, 1\]",
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
                        "the answer has options that were not offered: "
                        rf"\['{other_label}'\]",
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
                    rf"of the option `{option_labels[1]}` is not a number within "
                    r"\[0, 1\]",
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
                "the answer is of the kind `predicate`, while the question is of "
                "the kind `choice`",
            ),
            # Without the probability, or with a probability that is not one
            ({"C0": _predicate_answer("C0")}, "C0", "the answer has no probability"),
            (
                {"A0": _predicate_answer("A0", probability=1.5)},
                "A0",
                r"is not a number within \[0, 1\]",
            ),
            # With an option that was not offered, or without an option offered
            (
                {
                    "C1": _choice_answer(
                        "C1", option_probabilities={"written": 0.7, "fax": 0.3}
                    )
                },
                "C1",
                r"the answer has options that were not offered: \['fax'\]",
            ),
            (
                {"C2": _score_answer("C2", option_probabilities={"1": 0.5, "2": 0.5})},
                "C2",
                r"the answer has no probabilities of the options \['3'\]",
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
        all kinds have, and none of the params and methods that only an LLM has.
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
            "timeout",
            "pricing_details",
            "auto_pricing",
            "auto_pricing_refresh",
            "is_fallback",
            "fallback_models",
        }

        # Defaults
        assert decision_model.model == _DECISION_MODEL
        assert decision_model.api_key is None
        assert decision_model.api_base is None
        assert decision_model.role is None
        assert decision_model.num_retries_failed_request == 3
        assert decision_model.timeout == 600
        assert decision_model.pricing_details is None
        assert decision_model.auto_pricing is False
        assert decision_model.auto_pricing_refresh is False
        assert decision_model.is_fallback is False
        assert decision_model.fallback_models == []
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
            timeout=30,
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.042, output_per_1m_tokens=0.0
            ),
            is_fallback=True,
            async_limiter=async_limiter,  # ty: ignore[unknown-argument]
        )
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
            ("max_retries_invalid_data", 1),
            ("max_retries_failed_request", 1),
            ("deployment_id", "deployment"),
            ("api_version", "2026-01-01"),
        ]:
            assert param_name in DocumentLLM.model_fields
            with pytest.raises(ValueError, match=f"{param_name}\n"):
                DocumentDecisionModel(
                    model=_DECISION_MODEL,
                    **{param_name: value},
                )
            assert not hasattr(decision_model, param_name)
        # ... and a decision model cannot chat
        for method_name in ["chat", "chat_async"]:
            assert hasattr(DocumentLLM, method_name)
            assert not hasattr(decision_model, method_name)

        # What the extraction logic shared with LLMs reads from a model: a call
        # for which a decision model returned an invalid response is not made
        # again, and a failed request is retried only by the framework
        assert decision_model._retries_invalid_data == 0
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
        automatically, when the model is created and when the params are assigned.
        """
        pricing_details = ModelPricing(
            input_per_1m_tokens=0.10, output_per_1m_tokens=0.0
        )

        # As for an LLM, the pricing details exclude the automatic pricing
        with pytest.raises(ValueError, match="cannot be set together"):
            _create_decision_model(pricing_details=pricing_details, auto_pricing=True)

        for model, error_match in [
            # genai-prices has only the chat prices of the models of OpenAI
            (
                "openai/gpt-6-luna",
                "not supported for decision model `openai/gpt-6-luna`.*"
                "prices of the chat requests.*provide explicit `pricing_details`",
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

        # As for a local LLM
        with pytest.raises(ValueError, match="not supported for local models"):
            _create_decision_model("ollama/nimble", auto_pricing=True)

        # The automatic pricing is accepted where genai-prices has a decision price
        with pytest.warns(UserWarning, match="Auto-pricing"):
            decision_model = _create_decision_model(auto_pricing=True)
        assert decision_model._auto_pricing_provider_id == "typesafe"
        assert decision_model._auto_pricing_model_ref == "jev-latest"
        # ... until the model is changed to a model that it is rejected for
        with pytest.raises(ValueError, match="not supported for decision model"):
            decision_model.model = "openai/gpt-6-luna"
        assert decision_model.model == _DECISION_MODEL
        assert decision_model.auto_pricing is True

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
            with pytest.raises(ValueError, match="already been processed"):
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
            # The text is always sent as a whole: the number of paragraphs per call
            # is not limited for a decision model
            (
                {"max_paragraphs_to_analyze_per_call": 1},
                [
                    [
                        "C0",
                        "C1",
                        "C2",
                        "C3-L0",
                        "C3-L1",
                        "C3-L2",
                        "C4-L0",
                        "C4-L1",
                        "C5",
                    ]
                ],
            ),
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
        sub_aspect.extracted_items = [sub_aspect_item]  # ty: ignore[invalid-assignment]
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
        with pytest.warns(UserWarning, match="Auto-pricing"):
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
        assert cost.output == Decimal("0")
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

            # A warning is issued if `raise_exception_on_extraction_error` is False,
            # without sending the failed request again: only the warning on
            # the failed request, as the model returned no response to report
            for extraction_kwargs in [{}, {"use_concurrency": True}]:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    warnings.simplefilter("always")
                    decisions_stub, document, decision_model = extract(
                        [rate_limit_error],
                        document=create_document(n_concepts=2),
                        raise_exception_on_extraction_error=False,
                        max_items_per_call=1,
                        **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                    )
                warning_messages = [
                    str(caught_warning.message)
                    for caught_warning in caught_warnings
                    if "concepts" in str(caught_warning.message)
                    or "API" in str(caught_warning.message)
                ]
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
                warnings.catch_warnings(record=True) as caught_warnings,
            ):
                warnings.simplefilter("always")
                llm.extract_concepts_from_document(
                    document,
                    raise_exception_on_extraction_error=False,
                    max_items_per_call=1,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            warning_messages = [
                str(caught_warning.message)
                for caught_warning in caught_warnings
                if "concepts" in str(caught_warning.message)
                or "API" in str(caught_warning.message)
            ]
            assert len(warning_messages) == 2  # one per concept, i.e. per call
            for warning_message in warning_messages:
                assert "calling LLM API" in warning_message
            assert completions_stub.requested_models() == [_LLM_MODEL] * 2
            assert not any(concept._is_processed for concept in document.concepts)

    def test_invalid_decision_model_responses(self):
        """
        Tests that a response of a decision model that cannot be used, such as
        a refusal to answer a question, fails the extraction as invalid data
        returned by the model, without the same request being sent again.
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
            # The model refused to answer a question
            ({**valid_answers, "C1": "refusal"}, "`C1`: the model refused to answer"),
            ({**valid_answers, "C0": "refusal"}, "`C0`: the model refused to answer"),
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
                "`C0`: the answer is of the kind `choice`, while the question is of "
                "the kind `predicate`",
            ),
            (
                {
                    **valid_answers,
                    "C2": OpenAIPredicateAnswer(name="C2", probability=0.5),
                },
                "`C2`: the answer is of the kind `predicate`, while the question "
                "is of the kind `score`",
            ),
            # A probability that is not within [0, 1]
            ({**valid_answers, "C0": 1.5}, r"`C0`: the probability 1.5 is not"),
            # An option that was not offered, or an option offered that is missing
            (
                {**valid_answers, "C1": {"written": 0.7, "fax": 0.3}},
                r"`C1`: the answer has options that were not offered: \['fax'\]",
            ),
            (
                {**valid_answers, "C1": {"written": 1.0}},
                r"`C1`: the answer has no probabilities of the options \['email'\]",
            ),
            # The levels of a rating keyed by their labels, not by their positions:
            # a level with the label of another scale is not a level offered
            (
                {**valid_answers, "C2": {"0": 0.2, "1": 0.5, "2": 0.3}},
                r"`C2`: the answer has options that were not offered: \['0'\]",
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
                "`C1`: the answer repeats the option `written`",
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
                "`C0`: the answer at the position of the question has the name `C1`",
            ),
        ]

        for outcome, error_match in invalid_outcomes:
            decision_model = _create_decision_model()
            document = _create_document(concepts=concepts)
            decisions_stub = _DecisionsAPIStub(
                {_DECISION_MODEL: [outcome]}, input_tokens=300
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                _capture_logs(stdlib_logging.ERROR) as captured_logs,
                pytest.raises(
                    ModelExtractionError,
                    match="Some concepts could not be processed due to "
                    "an invalid response returned by decision model. "
                    r"Failed concepts: \['NDA check', 'Notice form', 'Clarity'\]",
                ) as exc_info,
            ):
                decision_model.extract_concepts_from_document(document)
            assert exc_info.value.retry_count == 0
            # What is wrong with the response is logged
            assert any(
                f"Decision model `{_DECISION_MODEL}` returned an invalid response"
                in log
                and re.search(error_match, log)
                for log in captured_logs
            ), captured_logs
            # The request is not sent again: the same response is expected
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
        # responses are used, and the extraction fails for the first invalid one
        decision_model = _create_decision_model()
        document = _create_document(concepts=concepts)
        decisions_stub = _DecisionsAPIStub(
            {
                _DECISION_MODEL: [
                    lambda request: _create_decisions_response(
                        request["questions"],
                        # The question of each concept is the first in its request
                        {
                            "C0": "refusal"
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

        # A warning is issued if `raise_exception_on_extraction_error` is False
        decision_model = _create_decision_model()
        document = _create_document(concepts=concepts)
        decisions_stub = _DecisionsAPIStub({_DECISION_MODEL: [invalid_outcomes[0][0]]})
        with (
            patch.object(litellm, "adecisions", new=decisions_stub),
            pytest.warns(
                UserWarning,
                match=r"an invalid response returned by decision model.*\(0 retries\)",
            ),
        ):
            decision_model.extract_concepts_from_document(
                document, raise_exception_on_extraction_error=False
            )
        assert len(decisions_stub.requests) == 1
        assert not any(concept._is_processed for concept in document.concepts)

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
        that cannot be used.
        """
        llm_answer = '[{"concept_id": "C0", "extracted_items": [true]}]'
        invalid_llm_answer = "<<<not-json>>>"
        answers = {"C0": 0.93}
        refusal = {"C0": "refusal"}
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
            completions_outcomes: dict[str, list[Exception | str]],
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
            :type completions_outcomes: dict[str, list[Exception | str]]
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

        # The response of the decision model cannot be used: the request is not sent
        # to it again, and the fallback LLM is tried
        decision_model = create_decision_model_with_llm_fallback()
        document, _ = extract(
            decision_model, {_DECISION_MODEL: [refusal]}, {_LLM_MODEL: [llm_answer]}
        )
        assert requested_models == [_DECISION_MODEL, _LLM_MODEL]
        assert document.concepts[0].extracted_items[0].value is True
        assert document.concepts[0].extracted_items[0].probabilities is None
        (invalid_call,) = decision_model.get_usage()[0].usage.calls
        assert invalid_call.timestamp_received is not None

        # The fallback LLM is asked again when it returns invalid data, as an LLM is
        document, _ = extract(
            create_decision_model_with_llm_fallback(),
            {_DECISION_MODEL: [refusal]},
            {_LLM_MODEL: [invalid_llm_answer, llm_answer]},
        )
        assert requested_models == [_DECISION_MODEL, _LLM_MODEL, _LLM_MODEL]
        assert document.concepts[0].extracted_items[0].value is True
        # ... and the error is the one of the LLM when it is the last model to fail
        with pytest.raises(
            ModelExtractionError, match=r"invalid JSON returned by LLM.*2 retries"
        ):
            extract(
                create_decision_model_with_llm_fallback(),
                {_DECISION_MODEL: [refusal]},
                {_LLM_MODEL: [invalid_llm_answer]},
            )
        assert requested_models == [_DECISION_MODEL, _LLM_MODEL, _LLM_MODEL]
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

        for primary_outcome in [auth_error, refusal]:
            decision_model = create_decision_model_with_decision_model_fallback()
            document, _ = extract(
                decision_model,
                {_DECISION_MODEL: [primary_outcome], _OTHER_DECISION_MODEL: [answers]},
                {},
            )
            assert requested_models == [_DECISION_MODEL, _OTHER_DECISION_MODEL]
            (item,) = document.concepts[0].extracted_items
            assert item.value is True
            self.check_item_probabilities(item, {True: 0.93, False: 1 - 0.93})
            usage_containers = decision_model.get_usage()
            assert [container.model for container in usage_containers] == [
                _DECISION_MODEL,
                _OTHER_DECISION_MODEL,
            ]
            assert usage_containers[1].usage.input == 100
        # The fallback decision model is not asked again either, and the error is
        # the one of the last model to fail
        with pytest.raises(
            ModelExtractionError,
            match="an invalid response returned by decision model",
        ):
            extract(
                create_decision_model_with_decision_model_fallback(),
                {_DECISION_MODEL: [auth_error], _OTHER_DECISION_MODEL: [refusal]},
                {},
            )
        assert requested_models == [_DECISION_MODEL, _OTHER_DECISION_MODEL]
        other_auth_error = _create_api_error(
            litellm.AuthenticationError, _OTHER_DECISION_MODEL
        )
        with pytest.raises(ModelAPIError, match="decision model API") as exc_info:
            extract(
                create_decision_model_with_decision_model_fallback(),
                {_DECISION_MODEL: [refusal], _OTHER_DECISION_MODEL: [other_auth_error]},
                {},
            )
        assert exc_info.value.original_error is other_auth_error
        assert requested_models == [_DECISION_MODEL, _OTHER_DECISION_MODEL]

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
        # The fallback decision model is the last model to fail
        with pytest.raises(
            ModelExtractionError,
            match=r"an invalid response returned by decision model.*1 retries",
        ):
            extract(
                create_llm_with_decision_model_fallback(),
                {_DECISION_MODEL: [refusal]},
                {_LLM_MODEL: [llm_auth_error]},
            )
        assert requested_models == [_LLM_MODEL, _DECISION_MODEL]
        with pytest.raises(ModelAPIError, match="decision model API") as exc_info:
            extract(
                create_llm_with_decision_model_fallback(),
                {_DECISION_MODEL: [auth_error]},
                {_LLM_MODEL: [llm_auth_error]},
            )
        assert exc_info.value.original_error is auth_error
        with pytest.warns(UserWarning, match=r"decision model API.*0 retries"):
            document, _ = extract(
                create_llm_with_decision_model_fallback(),
                {_DECISION_MODEL: [auth_error]},
                {_LLM_MODEL: [llm_auth_error]},
                raise_exception_on_extraction_error=False,
            )
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
            {_DECISION_MODEL: [refusal], _OTHER_DECISION_MODEL: [auth_error]},
            {_LLM_MODEL: [llm_answer]},
        )
        assert requested_models == [_DECISION_MODEL, _OTHER_DECISION_MODEL, _LLM_MODEL]
        assert document.concepts[0].extracted_items[0].value is True
        assert [container.model for container in decision_model.get_usage()] == [
            _DECISION_MODEL,
            _OTHER_DECISION_MODEL,
            _LLM_MODEL,
        ]

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
                "type `StringConcept` is not supported, justifications are not "
                "supported, references are not supported",
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
                f"A decision model extracts aspects with the `paragraphs` reference "
                f"depth, and Boolean, label and rating concepts from text without "
                f"references, both without justifications\\. Route the other aspects "
                f"and concepts to an LLM"
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
            # Sentences as the extracted items
            (
                {"reference_depth": "sentences"},
                "reference depth `sentences` is not supported",
            ),
            # Justifications
            ({"add_justifications": True}, "justifications are not supported"),
            # Each reason is given
            (
                {"reference_depth": "sentences", "add_justifications": True},
                "reference depth `sentences` is not supported, justifications "
                "are not supported",
            ),
        ]
        for aspect_kwargs, reasons in unservable_aspects:
            error_match = (
                f"^Decision model `{_DECISION_MODEL}` cannot extract: "
                f"aspect `Termination` \\({reasons}\\)\\. "
                f"A decision model extracts aspects with the `paragraphs` reference "
                f"depth"
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
            r"aspect `Notices` \(reference depth `sentences` is not supported\)\. ",
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
        group_error_match = (
            f"^Decision model `{_DECISION_MODEL}` cannot extract: "
            r"concept `Title` \(type `StringConcept` is not supported\)\. "
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
                    concepts=[supported_concept],
                )
            ]
        )
        check_rejected(
            llm,
            document,
            f"^Fallback decision model `{_DECISION_MODEL}` cannot extract what it "
            r"is to extract if the model it is a fallback for fails: aspect "
            r"`Termination` \(reference depth `sentences` is not supported\)\. ",
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
        check_rejected(model_group, document, fallback_error_match)
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
            _capture_logs() as captured_logs,
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
            _capture_logs() as captured_logs,
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
            _capture_logs() as captured_logs,
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
        llm._supports_vision = True
        llm._supports_parallel_tool_calls = True
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
            _capture_logs() as captured_logs,
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
        decision_model_dict = _create_decision_model("openai/gpt-6-luna").to_dict()
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
        expected_questions = [
            {
                "type": "predicate",
                "name": question.name,
                "instructions": question.instructions,
            }
            for question in questions
        ]
        assert "Aspect name: Termination\n" in questions[1].instructions
        assert len(decisions_stub.requests) == len(paragraphs) == 5
        for request, text_sent in zip(
            decisions_stub.requests, _ASPECTS_PARAGRAPH_TEXTS_SENT, strict=True
        ):
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
        # A paragraph is judged on its own text: its additional context is not sent
        assert paragraphs[2].additional_context == "Section 3. Termination"
        assert "Section 3" not in json.dumps(decisions_stub.requests)

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

            # The number of paragraphs per call is not applied to a decision model:
            # each paragraph is always sent in a request of its own
            decisions_stub.requests.clear()
            with _capture_logs() as captured_logs:
                decision_model.extract_aspects_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=2,
                )
            assert requests_sent() == [
                (text, ["A0", "A1", "A2"]) for text in _ASPECTS_PARAGRAPH_TEXTS_SENT
            ]
            assert (
                f"`max_paragraphs_to_analyze_per_call` is not applied to decision "
                f"model `{_DECISION_MODEL}`: each paragraph is sent to it in "
                f"a request of its own." in captured_logs
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

            # ... or a warning is issued, if no exception is to be raised: only
            # the one on the failed request, as no response is there to report
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub.requests.clear()
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                warnings.catch_warnings(record=True) as caught_warnings,
            ):
                warnings.simplefilter("always")
                decision_model.extract_aspects_from_document(
                    document,
                    raise_exception_on_extraction_error=False,
                    **extraction_kwargs,  # ty: ignore[invalid-argument-type]
                )
            warning_messages = [
                str(caught_warning.message)
                for caught_warning in caught_warnings
                if "aspects" in str(caught_warning.message)
                or "API" in str(caught_warning.message)
            ]
            (warning_message,) = warning_messages
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
            # invalid: no further requests are sent, and each call is finished
            decision_model = _create_decision_model()
            document = create_document()
            decisions_stub = _DecisionsAPIStub(
                {
                    _DECISION_MODEL: [
                        create_responder(
                            {failed_position: {"A0": 0.9, "A1": "refusal"}}
                        )
                    ]
                }
            )
            with (
                patch.object(litellm, "adecisions", new=decisions_stub),
                _capture_logs(stdlib_logging.ERROR) as captured_logs,
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
            assert any(
                f"Decision model `{_DECISION_MODEL}` returned an invalid response "
                "for aspects"
                in log
                and "`A1`: the model refused to answer" in log
                for log in captured_logs
            ), captured_logs
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
                        {1: {"A0": "refusal", "A1": 0.1}, 3: auth_error},
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

        # The requests that wait for the limiter when a request fails are not sent
        held_acquisitions: list[int] = []

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
                    await asyncio.sleep(5)

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
        # Each request waited for the limiter, and those held by it were not sent
        assert len(held_acquisitions) == n_paragraphs
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
        termination_text = f"{paragraph_texts[2]}\n\n{paragraph_texts[4]}"
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
            (paragraph_texts[4], ["C0"]),
            # The concepts of the document, on its text
            ("\n\n".join(paragraph_texts), ["C0", "C1"]),
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
        for decision_model_outcomes in [
            # The request for the paragraph in the middle fails
            {2: auth_error},
            # The response to it cannot be used
            {2: {"A0": 0.9, "A1": "refusal"}},
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
            # for them on the third paragraph. The sub-aspect is extracted by
            # the decision model, which is asked about each paragraph of its aspect.
            assert [request["input"] for request in decisions_stub.requests] == [
                *_ASPECTS_PARAGRAPH_TEXTS_SENT[:3],
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
                # Its concept, on the text of the paragraphs of its aspect
                (
                    f"{paragraph_texts[2]}\n\n{paragraph_texts[4]}",
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
        as reported by its API, at the price set for the model.

        :param decision_model: The decision model, with its pricing details set.
        :type decision_model: DocumentDecisionModel
        """
        assert decision_model.pricing_details is not None
        usage = decision_model.get_usage()[0].usage
        cost = decision_model.get_cost()[0].cost
        expected_cost = (
            Decimal(usage.input)
            * Decimal(str(decision_model.pricing_details.input_per_1m_tokens))
            / Decimal(1_000_000)
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
            api_key=os.getenv("TYPESAFE_API_KEY"),
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
            api_key=os.getenv("TYPESAFE_API_KEY"),
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
        llm = DocumentLLM(**self._llm_extractor_kwargs_openai)  # ty: ignore[invalid-argument-type]
        decision_model = DocumentDecisionModel(
            model="typesafe/jev-latest",
            api_key=os.getenv("TYPESAFE_API_KEY"),
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
        they are sent to litellm in, without a conversion.
        """
        decision_model = DocumentDecisionModel(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            pricing_details=ModelPricing(
                input_per_1m_tokens=0.10, output_per_1m_tokens=0.0
            ),
        )
        document = Document(raw_text=_RECORDED_CONCEPTS_DOCUMENT_TEXT)
        document.concepts = self._create_concepts_for_recorded_tests()

        decision_model.extract_concepts_from_document(document)

        self._check_concepts_of_recorded_tests(document, decision_model)
