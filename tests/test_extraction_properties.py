# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the property-based tests of the extraction with LLMs and with
decision models.

Hypothesis generates the documents, the aspects and concepts to extract, and what
the APIs of the models return, which are stubbed: no requests are sent, and no
cassettes are needed. What the APIs return includes what a model is not supposed
to return, e.g. an ID that is not in the prompt, a value of another type than
the concept has, or a probability that is not a number within [0, 1]. Each test
checks the outcome of the extraction against what the responses entail:
the items extracted, or the error that the extraction fails with, and the usage
recorded.

Only hosted models are used: where a model is served does not change how its
responses are handled, and the local models are tested by tests of their own.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import functools
import json
import keyword
import logging as stdlib_logging
import math
import operator
import re
import sys
import types
import unicodedata
import warnings
from collections import Counter
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from types import UnionType
from typing import (  # noqa: UP035 - legacy forms of type hints are tested
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    cast,
    get_args,
    get_origin,
)
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter
from hypothesis import HealthCheck, assume, event, given, note, settings
from hypothesis import strategies as st
from litellm.types.decisions import (
    OpenAIChoiceAnswer,
    OpenAIChoiceProbability,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
    OpenAIRefusalAnswer,
    OpenAIScoreAnswer,
    OpenAIScoreProbability,
)

from contextgem.internal.base import decision_models as decision_models_module
from contextgem.internal.base import llms as llms_module
from contextgem.internal.base.paras_and_sents import _Paragraph
from contextgem.internal.exceptions import (
    ModelAPIError,
    ModelExtractionError,
    ModelRefusalError,
)
from contextgem.internal.loggers import logger
from contextgem.internal.typings.types import (
    ClassificationType,
    JustificationDepth,
    ReferenceDepth,
)
from contextgem.internal.utils import _clean_text_for_llm_prompt
from contextgem.public import (
    Aspect,
    BooleanConcept,
    DateConcept,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    ExtractionPipeline,
    JsonObjectClassStruct,
    JsonObjectConcept,
    JsonObjectExample,
    LabelConcept,
    ModelPricing,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    Sentence,
    StringConcept,
)
from tests.base import BaseTest


# The module object that the models call the APIs through, i.e. the one to patch
# the API calls on
litellm = llms_module.litellm

# The examples of a test share the function-scoped fixtures of the test, e.g.
# the network guard, which is intended: each example creates what it uses.
# An example runs an extraction, which can take longer than a deadline allows
# for on a busy machine.
_PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# Hosted models of the tests, all different, so that the stubs and the usage
# tell them apart. No requests are sent to them.
_LLM_MODELS = ("openai/gpt-6-luna", "openai/gpt-5.6-luna", "openai/gpt-6.1-sol")
_DECISION_MODELS = ("typesafe/jev-latest", "typesafe/jev-1.13.0")
_LLM_MODEL = _LLM_MODELS[0]
_DECISION_MODEL = _DECISION_MODELS[0]

# Values of the literal params of aspects and concepts
_REFERENCE_DEPTHS: tuple[ReferenceDepth, ...] = ("paragraphs", "sentences")
_JUSTIFICATION_DEPTHS: tuple[JustificationDepth, ...] = (
    "brief",
    "balanced",
    "comprehensive",
)
_NUMERIC_TYPES: tuple[Literal["int", "float", "any"], ...] = ("int", "float", "any")

# Tokens that the stubbed APIs report for each response, and the prices of
# the models per 1M tokens, with which the costs of the tokens are exact
# in the units that costs are recorded in
_LLM_INPUT_TOKENS = 1000
_LLM_OUTPUT_TOKENS = 50
_DECISION_INPUT_TOKENS = 100
_INPUT_PRICE = Decimal(2)
_OUTPUT_PRICE = Decimal(8)

# Value that an LLM output entails for an item, or for the whole output, which
# the extraction is to reject: the LLM is then asked again, or the extraction fails
_INVALID = object()


# Texts


# Characters of the generated texts: any but the digits, which the markers of
# the paragraphs and the sentences are made of (see `_documents`)
_TEXT_CHARACTERS = st.characters(exclude_categories=("Cs", "Nd", "Nl", "No"))
_TEXTS = st.text(_TEXT_CHARACTERS, max_size=20)
_NON_BLANK_TEXTS = st.text(_TEXT_CHARACTERS, min_size=1, max_size=20).filter(str.strip)

# Marker of a sentence, which starts the text of the sentence: the position of
# its paragraph in the document, and its position in the paragraph. The stubs of
# the decision models tell the paragraph or the sentence that a request is on by
# the first marker of the text of the request.
_RE_MARKER = re.compile(r"Clause (\d+)-(\d+)\.")

# Custom data, also with nested values, as e.g. the Docling data of a paragraph
_CUSTOM_DATA = st.dictionaries(
    st.text(max_size=5),
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(-1000, 1000),
        _TEXTS,
        st.dictionaries(st.text(max_size=5), st.integers(-1000, 1000), max_size=2),
        st.lists(st.integers(-1000, 1000), max_size=2),
    ),
    max_size=2,
)


@st.composite
def _documents(
    draw: st.DrawFn, max_paragraphs: int = 5, max_sentences: int = 3
) -> Document:
    """
    Generates a document of paragraphs with their sentences given, so that
    the sentences are not segmented with a SaT model. Each sentence starts with
    its marker, followed by text of any characters but digits. The paragraphs and
    the sentences may have additional context, of any characters but digits as
    well, e.g. line breaks or other control characters, and custom data.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param max_paragraphs: Maximum number of paragraphs. Defaults to 5.
    :type max_paragraphs: int
    :param max_sentences: Maximum number of sentences of a paragraph. Defaults to 3.
    :type max_sentences: int
    :return: Document.
    :rtype: Document
    """
    paragraphs = []
    for paragraph_position in range(draw(st.integers(1, max_paragraphs))):
        sentence_texts = [
            f"Clause {paragraph_position}-{sentence_position}. {draw(_TEXTS)}"
            for sentence_position in range(draw(st.integers(1, max_sentences)))
        ]
        paragraphs.append(
            Paragraph(
                raw_text=" ".join(sentence_texts),
                sentences=[
                    Sentence(
                        raw_text=text,
                        additional_context=draw(st.none() | _NON_BLANK_TEXTS),
                        custom_data=draw(_CUSTOM_DATA),
                    )
                    for text in sentence_texts
                ],
                additional_context=draw(st.none() | _NON_BLANK_TEXTS),
                custom_data=draw(_CUSTOM_DATA),
            )
        )
    return Document(paragraphs=paragraphs)


# What an extraction prompt shows in the tags of additional context, and the start
# of its instructions on additional context, which it has when it shows some
_RE_SHOWN_ADDITIONAL_CONTEXT = re.compile(
    r"<additional_context>\s*(.*?)\s*</additional_context>", re.DOTALL
)
_ADDITIONAL_CONTEXT_INSTRUCTIONS = "When available, additional context information"


def _check_prompt_texts(
    prompt: str, document: Document, reference_depth: str, lists_paragraphs: bool
) -> None:
    """
    Checks the texts of a prompt of an LLM for extraction from a document:
    the prompt has no control characters but line breaks, it shows the additional
    context of each paragraph that it lists, and of each sentence with
    the "sentences" reference depth unless it is that of the paragraph, which is
    shown once, cleaned as the texts are, if anything is left of it, and it has
    the instructions on additional context exactly if it shows some.

    :param prompt: The prompt.
    :type prompt: str
    :param document: Document that the prompt is for.
    :type document: Document
    :param reference_depth: Reference depth of the aspects or concepts.
    :type reference_depth: str
    :param lists_paragraphs: Whether the prompt lists the paragraphs, as it does
        for aspects, and for concepts with references.
    :type lists_paragraphs: bool
    :return: None
    :rtype: None
    """
    assert not [
        character
        for character in prompt
        if unicodedata.category(character) == "Cc" and character != "\n"
    ]

    def clean(text: str | None) -> str:
        """
        Cleans a text as the prompt does.
        """
        return _clean_text_for_llm_prompt(
            text or "", preserve_linebreaks=False, strip_text=True
        )

    expected: list[str] = []
    for paragraph in document.paragraphs if lists_paragraphs else []:
        if not clean(paragraph.raw_text):
            continue
        if clean(paragraph.additional_context):
            expected.append(clean(paragraph.additional_context))
        if reference_depth == "sentences":
            expected.extend(
                clean(sentence.additional_context)
                for sentence in paragraph.sentences
                if clean(sentence.raw_text)
                and clean(sentence.additional_context)
                and clean(sentence.additional_context)
                != clean(paragraph.additional_context)
            )
    shown = [
        context for context in _RE_SHOWN_ADDITIONAL_CONTEXT.findall(prompt) if context
    ]
    assert shown == expected
    assert (_ADDITIONAL_CONTEXT_INSTRUCTIONS in prompt) is bool(expected)


def _get_marker(text: str) -> tuple[int, int]:
    """
    Returns the positions of the paragraph and of the sentence in the first marker
    of a text.

    :param text: Text that has a marker.
    :type text: str
    :return: Position of the paragraph in the document, and of the sentence in
        the paragraph.
    :rtype: tuple[int, int]
    """
    match = _RE_MARKER.search(text)
    assert match is not None, f"No marker in the text: {text!r}"
    return int(match.group(1)), int(match.group(2))


# Stubs of the model APIs


def _no_retry_delay(retry_number: int) -> float:
    """
    Returns no delay before a retry of a failed model API request.

    :param retry_number: Number of the retry.
    :type retry_number: int
    :return: Zero.
    :rtype: float
    """
    return 0.0


class _ModelAPIsStub:
    """
    Simulates the API of LLMs (``litellm.acompletion``) and the Decisions API of
    litellm that decision models are requested through (``litellm.adecisions``):
    each request is recorded and answered with what the given function returns
    for it, or fails with it if it is an exception.

    :ivar completion_requests: The requests to LLMs, as the keyword arguments
        of the calls.
    :vartype completion_requests: list[dict[str, Any]]
    :ivar decision_requests: The requests to decision models, as the keyword
        arguments of the calls.
    :vartype decision_requests: list[dict[str, Any]]
    """

    def __init__(
        self,
        complete: Callable[[dict[str, Any]], Any] | None = None,
        decide: Callable[[dict[str, Any]], Any] | None = None,
    ):
        self.complete = complete
        self.decide = decide
        self.completion_requests: list[dict[str, Any]] = []
        self.decision_requests: list[dict[str, Any]] = []

    async def acompletion(self, **kwargs: Any) -> Any:
        """
        Answers a request to an LLM.

        :param kwargs: Keyword arguments of ``litellm.acompletion``.
        :type kwargs: Any
        :return: Completion object, with the attributes that the framework reads.
        :rtype: Any
        """
        self.completion_requests.append(kwargs)
        assert self.complete is not None, "Unexpected request to an LLM"
        outcome = self.complete(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def adecisions(self, **kwargs: Any) -> Any:
        """
        Answers a request to a decision model.

        :param kwargs: Keyword arguments of ``litellm.adecisions``.
        :type kwargs: Any
        :return: Response of litellm.
        :rtype: Any
        """
        self.decision_requests.append(kwargs)
        assert self.decide is not None, "Unexpected request to a decision model"
        outcome = self.decide(kwargs)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @contextmanager
    def installed(self) -> Generator[_ModelAPIsStub]:
        """
        Puts the stub in place of the APIs, and removes the delays between
        the retries of failed requests, while the context is entered.

        :return: Context in which the stub is installed.
        :rtype: Generator[_ModelAPIsStub]
        """
        with (
            patch.object(litellm, "acompletion", new=self.acompletion),
            patch.object(litellm, "adecisions", new=self.adecisions),
            patch.object(
                llms_module, "_get_model_api_retry_delay", new=_no_retry_delay
            ),
            patch.object(
                decision_models_module,
                "_get_model_api_retry_delay",
                new=_no_retry_delay,
            ),
        ):
            yield self


def _completion(text: str | None, finish_reason: str = "stop") -> Any:
    """
    Creates the completion object of a response of an LLM.

    :param text: Text of the response.
    :type text: str | None
    :param finish_reason: Why the LLM stopped: "length" for a response cut off at
        the max output tokens. Defaults to "stop".
    :type finish_reason: str
    :return: Completion object, with the attributes that the framework reads.
    :rtype: Any
    """
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=_LLM_INPUT_TOKENS, completion_tokens=_LLM_OUTPUT_TOKENS
        ),
    )


def _decisions_response(answers: Sequence[Any]) -> OpenAIDecisionResponse:
    """
    Creates a response of the Decisions API of litellm.

    :param answers: Answers of the response, as litellm returns them.
    :type answers: Sequence[Any]
    :return: Response of litellm.
    :rtype: OpenAIDecisionResponse
    """
    return OpenAIDecisionResponse(
        model="decision-model",
        answers=tuple(answers),
        usage=OpenAIDecisionUsage(
            input_tokens=_DECISION_INPUT_TOKENS,
            output_tokens=0,
            total_tokens=_DECISION_INPUT_TOKENS,
        ),
    )


def _create_unpaced_limiter() -> AsyncLimiter:
    """
    Creates a limiter of the requests to a model that does not pace them in
    practice, for a model whose API is stubbed: the default limiter of a model
    lets 10 requests through at once, and then one a second.

    :return: The limiter.
    :rtype: AsyncLimiter
    """
    return AsyncLimiter(10_000, 1)


def _api_error(transient: bool, model: str) -> Exception:
    """
    Creates the error of a failed request to a model API: a rate limit error,
    which is transient, or an authentication error, which is not.

    :param transient: Whether the error is transient.
    :type transient: bool
    :param model: Model that the error is reported for.
    :type model: str
    :return: The error.
    :rtype: Exception
    """
    provider, _, model_name = model.partition("/")
    error_class = litellm.RateLimitError if transient else litellm.AuthenticationError
    return error_class(
        message="Simulated API error", model=model_name, llm_provider=provider
    )


@contextmanager
def _recorded_warnings() -> Generator[list[str]]:
    """
    Records the texts of the warnings reported while the context is entered:
    the messages that the framework logs at the WARNING level, e.g. for a call
    of an extraction that failed, and the Python warnings, e.g. for
    a configuration that does not have the expected effect. The Python warnings
    are added when the context is exited.

    :return: Context with the list of the texts of the warnings recorded.
    :rtype: Generator[list[str]]
    """
    recorded: list[str] = []

    class _WarningLogHandler(stdlib_logging.Handler):
        """
        Records the messages logged at the WARNING level.
        """

        def emit(self, record: stdlib_logging.LogRecord) -> None:
            """
            Adds the message of a record at the WARNING level to the recorded ones.

            :param record: The log record.
            :type record: stdlib_logging.LogRecord
            """
            if record.levelno == stdlib_logging.WARNING:
                recorded.append(record.getMessage())

    handler = _WarningLogHandler()
    logger.addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield recorded
        recorded.extend(str(warning.message) for warning in caught)
    finally:
        logger.removeHandler(handler)


def _get_messages(recorded: list[str]) -> list[str]:
    """
    Returns the texts of recorded warnings.

    :param recorded: Texts of the recorded warnings.
    :type recorded: list[str]
    :return: Texts of the warnings.
    :rtype: list[str]
    """
    return list(recorded)


# The warnings that an extraction logs for a call that failed, or for what
# a decision model refused to answer on
_RE_EXTRACTION_WARNING = re.compile(
    r"could not be processed|Exception occurred while calling"
    r"|was cut off at the max output tokens|refused to answer"
)


def _get_extraction_warnings(caught: list[str]) -> list[str]:
    """
    Returns the texts of the recorded warnings that an extraction logs for
    a call that failed, or for refused answers, leaving out any other warning,
    e.g. a deprecation warning of a dependency.

    :param caught: Texts of the recorded warnings.
    :type caught: list[str]
    :return: Texts of the warnings.
    :rtype: list[str]
    """
    return [
        message
        for message in _get_messages(caught)
        if _RE_EXTRACTION_WARNING.search(message)
    ]


# IDs in the output of an LLM


def _get_position_of_id(prompt_id: Any, prefix: str, count: int) -> int | None:
    """
    Returns the position that an ID returned by an LLM refers to, if it is one of
    the IDs that the prompt shows, e.g. "P3" for the fourth paragraph of the prompt:
    the prefix and the position, in ASCII digits without leading zeros, with or
    without surrounding whitespace.

    :param prompt_id: ID returned by the LLM.
    :type prompt_id: Any
    :param prefix: Prefix of the IDs: "A" for aspects, "C" for concepts, "P" for
        paragraphs.
    :type prefix: str
    :param count: Number of the IDs of that prefix in the prompt.
    :type count: int
    :return: The position, or None if the ID is not one of the prompt.
    :rtype: int | None
    """
    if not isinstance(prompt_id, str):
        return None
    match = re.fullmatch(prefix + r"(0|[1-9][0-9]*)", prompt_id.strip())
    if match is None or int(match.group(1)) >= count:
        return None
    return int(match.group(1))


def _get_position_of_sentence_id(
    sentence_id: Any, paragraph_position: int, sentence_count: int
) -> int | None:
    """
    Returns the position of the sentence that an ID returned by an LLM refers to,
    if it is the ID of a sentence of the given paragraph that the prompt shows,
    e.g. "P3-S1" for the second sentence of the fourth paragraph.

    :param sentence_id: ID returned by the LLM.
    :type sentence_id: Any
    :param paragraph_position: Position of the paragraph that the LLM listed
        the sentence for.
    :type paragraph_position: int
    :param sentence_count: Number of the sentences of the paragraph.
    :type sentence_count: int
    :return: The position of the sentence in the paragraph, or None if the ID is
        not that of a sentence of the paragraph.
    :rtype: int | None
    """
    if not isinstance(sentence_id, str):
        return None
    match = re.fullmatch(r"P(0|[1-9][0-9]*)-S(0|[1-9][0-9]*)", sentence_id.strip())
    if (
        match is None
        or int(match.group(1)) != paragraph_position
        or int(match.group(2)) >= sentence_count
    ):
        return None
    return int(match.group(2))


def _invalid_ids(prefix: str, count: int) -> st.SearchStrategy[str]:
    """
    Returns a strategy of the IDs that an LLM may return for the IDs of a prefix
    that the prompt shows `count` of, and that are not IDs of the prompt: an ID
    out of the range of the prompt, or a malformed one, e.g. with a leading zero,
    a sign, an underscore or non-ASCII digits, which ``int()`` accepts.

    :param prefix: Prefix of the IDs.
    :type prefix: str
    :param count: Number of the IDs of the prefix in the prompt, at least one.
    :type count: int
    :return: Strategy of IDs.
    :rtype: st.SearchStrategy[str]
    """
    malformed = st.integers(0, count - 1).flatmap(
        lambda position: st.sampled_from(
            [
                f"{prefix}0{position}",
                f"{prefix}{prefix}{position}",
                f"{prefix}-{position}",
                f"{prefix}+{position}",
                f"{prefix}{position}_0",
                f"{prefix}{position}.0",
                f"{prefix}{position}x",
                f"{prefix.lower()}{position}",
                f"{prefix} {position}",
                str(position),
                prefix,
                "",
                # Arabic-Indic digits, which int() reads as ASCII ones
                prefix + "".join(chr(0x0660 + int(digit)) for digit in str(position)),
            ]
        )
    )
    not_in_prompt = st.integers(count, count + 3).map(
        lambda position: f"{prefix}{position}"
    )
    return st.one_of(malformed, not_in_prompt)


def _ids(prefix: str, count: int, faulty: bool = True) -> st.SearchStrategy[str]:
    """
    Returns a strategy of the IDs that an LLM may return for the IDs of a prefix
    that the prompt shows `count` of: an ID of the prompt, with or without
    surrounding whitespace, and if the IDs may be faulty, sometimes an ID that is
    not in the prompt or that is malformed, or any text.

    :param prefix: Prefix of the IDs.
    :type prefix: str
    :param count: Number of the IDs of the prefix in the prompt, at least one.
    :type count: int
    :param faulty: Whether the IDs may be faulty. Defaults to True.
    :type faulty: bool
    :return: Strategy of IDs.
    :rtype: st.SearchStrategy[str]
    """
    positions = st.integers(0, count - 1)
    in_prompt = positions.map(lambda position: f"{prefix}{position}")
    padded = st.tuples(positions, st.sampled_from([" ", "\t", "\n"])).map(
        lambda args: f"{args[1]}{prefix}{args[0]}{args[1]}"
    )
    if not faulty:
        return st.one_of(in_prompt, in_prompt, in_prompt, padded)
    return st.one_of(
        in_prompt,
        in_prompt,
        padded,
        _invalid_ids(prefix, count),
        _invalid_ids(prefix, count),
        st.text(max_size=5),
    )


@st.composite
def _entry_ids(
    draw: st.DrawFn, prefix: str, count: int, faulty: bool = True
) -> list[str]:
    """
    Generates the IDs of the entries of an output of an LLM for aspects or for
    concepts, in their order: an entry for each aspect or concept of the prompt,
    in any order, and sometimes a second entry for one of them, which an LLM
    may return, with or without whitespace around the IDs. In a faulty output,
    any IDs of `_ids`, of which some aspects or concepts may have none, i.e. be
    left out.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param prefix: Prefix of the IDs: "A" for aspects, "C" for concepts.
    :type prefix: str
    :param count: Number of the aspects or concepts in the prompt.
    :type count: int
    :param faulty: Whether the output may be faulty. Defaults to True.
    :type faulty: bool
    :return: The IDs.
    :rtype: list[str]
    """
    if faulty:
        return draw(st.lists(_ids(prefix, count, faulty=True), max_size=count + 1))
    positions = [
        *draw(st.permutations(range(count))),
        *draw(st.lists(st.integers(0, count - 1), max_size=1)),
    ]
    return [
        draw(
            st.sampled_from(
                [f"{prefix}{position}"] * 3
                + [f" {prefix}{position}", f"{prefix}{position}\n"]
            )
        )
        for position in positions
    ]


def _sentence_ids_of_other_paragraphs(
    paragraph_position: int, sentence_count: int, paragraph_count: int
) -> st.SearchStrategy[str]:
    """
    Returns a strategy of the IDs of sentences of other paragraphs than the given
    one, which an LLM may list for the given one by mistake: of any paragraph
    of the prompt but the given one, or of one past the last paragraph.

    :param paragraph_position: Position of the paragraph.
    :type paragraph_position: int
    :param sentence_count: Number of the sentences of the paragraph.
    :type sentence_count: int
    :param paragraph_count: Number of the paragraphs of the prompt.
    :type paragraph_count: int
    :return: Strategy of sentence IDs.
    :rtype: st.SearchStrategy[str]
    """
    return st.tuples(
        st.integers(0, paragraph_count - 1).map(
            lambda q: q if q < paragraph_position else q + 1
        ),
        st.integers(0, sentence_count - 1),
    ).map(lambda args: f"P{args[0]}-S{args[1]}")


def _malformed_sentence_ids(
    paragraph_position: int, sentence_count: int
) -> st.SearchStrategy[str]:
    """
    Returns a strategy of the IDs that an LLM may return for the sentences of
    a paragraph, and that are not IDs of its sentences: of a sentence out of
    the range of the paragraph, or a malformed one.

    :param paragraph_position: Position of the paragraph.
    :type paragraph_position: int
    :param sentence_count: Number of the sentences of the paragraph.
    :type sentence_count: int
    :return: Strategy of sentence IDs.
    :rtype: st.SearchStrategy[str]
    """
    p = paragraph_position
    malformed = st.integers(0, sentence_count - 1).flatmap(
        lambda s: st.sampled_from(
            [
                f"P{p}-S0{s}",
                f"P{p}-S{s}-S{s}",
                f"P{p}-s{s}",
                f"P{p}S{s}",
                f"P{p}-S-{s}",
                f"P{p}-S{s}_0",
                f"S{s}",
                str(s),
            ]
        )
    )
    out_of_range = st.integers(sentence_count, sentence_count + 2).map(
        lambda s: f"P{p}-S{s}"
    )
    return st.one_of(malformed, out_of_range)


def _sentence_ids(
    paragraph_position: int,
    sentence_count: int,
    paragraph_count: int,
    faulty: bool = True,
) -> st.SearchStrategy[str]:
    """
    Returns a strategy of the IDs that an LLM may return for the sentences of
    a paragraph: the ID of a sentence of the paragraph, and if the IDs may be
    faulty, sometimes the ID of a sentence of another paragraph, or of no
    sentence, or a malformed ID, or any text.

    :param paragraph_position: Position of the paragraph.
    :type paragraph_position: int
    :param sentence_count: Number of the sentences of the paragraph.
    :type sentence_count: int
    :param paragraph_count: Number of the paragraphs of the prompt.
    :type paragraph_count: int
    :param faulty: Whether the IDs may be faulty. Defaults to True.
    :type faulty: bool
    :return: Strategy of sentence IDs.
    :rtype: st.SearchStrategy[str]
    """
    in_paragraph = st.integers(0, sentence_count - 1).map(
        lambda s: f"P{paragraph_position}-S{s}"
    )
    if not faulty:
        return in_paragraph
    return st.one_of(
        in_paragraph,
        in_paragraph,
        in_paragraph,
        _sentence_ids_of_other_paragraphs(
            paragraph_position, sentence_count, paragraph_count
        ),
        _malformed_sentence_ids(paragraph_position, sentence_count),
        st.text(max_size=6),
    )


def _lists_with_repeats(
    elements: st.SearchStrategy[Any], max_size: int, min_size: int = 0
) -> st.SearchStrategy[list[Any]]:
    """
    Returns a strategy of lists of the given elements, of which a third repeat
    an element of theirs at the end, as an LLM may list an ID more than once.

    :param elements: Strategy of the elements.
    :type elements: st.SearchStrategy[Any]
    :param max_size: Maximum number of the elements drawn for a list.
    :type max_size: int
    :param min_size: Minimum number of the elements drawn for a list. Defaults
        to 0.
    :type min_size: int
    :return: Strategy of the lists.
    :rtype: st.SearchStrategy[list[Any]]
    """
    return st.lists(elements, min_size=min_size, max_size=max_size).flatmap(
        lambda items: (
            st.sampled_from([items, items, [*items, items[0]]])
            if items
            else st.just(items)
        )
    )


# Justifications that are not valid: empty or blank, or not a text
_INVALID_JUSTIFICATIONS = st.sampled_from(["", "  ", "\n", None, 0, []])


def _justifications(faulty: bool = True) -> st.SearchStrategy[Any]:
    """
    Returns a strategy of the justifications that an LLM may return: a text, and
    if the justifications may be faulty, sometimes an empty or blank text, or
    no text.

    :param faulty: Whether the justifications may be faulty. Defaults to True.
    :type faulty: bool
    :return: Strategy of the justifications.
    :rtype: st.SearchStrategy[Any]
    """
    if not faulty:
        return _NON_BLANK_TEXTS
    return st.one_of(
        _NON_BLANK_TEXTS,
        _NON_BLANK_TEXTS,
        _NON_BLANK_TEXTS,
        _INVALID_JUSTIFICATIONS,
    )


# The whitespace that a non-empty string field of an item is stripped of:
# the characters with the White_Space property of Unicode, which pydantic strips.
# `str.strip()` also strips the information separators U+001C to U+001F.
_FIELD_WHITESPACE = "".join(
    chr(code_point)
    for code_point in range(sys.maxunicode + 1)
    if chr(code_point).isspace() and not 0x1C <= code_point <= 0x1F
)


def _get_non_empty_text(text: Any) -> Any:
    """
    Returns the text that a non-empty string field of an item gets from a text
    returned by an LLM: the text without surrounding whitespace, or `_INVALID` if
    it is not a text or if nothing is left of it.

    :param text: The text returned by the LLM.
    :type text: Any
    :return: The text of the field, or `_INVALID`.
    :rtype: Any
    """
    if isinstance(text, str) and text.strip(_FIELD_WHITESPACE):
        return text.strip(_FIELD_WHITESPACE)
    return _INVALID


def _get_justification(justification: Any) -> Any:
    """
    Returns the justification that an item gets from the justification returned
    by an LLM, or `_INVALID` if it is not valid.

    :param justification: Justification returned by the LLM.
    :type justification: Any
    :return: Justification of the item, without surrounding whitespace, or `_INVALID`.
    :rtype: Any
    """
    return _get_non_empty_text(justification)


# Output of an LLM


@st.composite
def _llm_response_texts(
    draw: st.DrawFn, output: Any, faulty: bool = True
) -> tuple[str, bool]:
    """
    Generates the text of a response of an LLM with the given output: the output
    as JSON, on its own or wrapped as LLMs may wrap it, and if the response may
    be faulty, sometimes a text that is not JSON, or JSON of another structure
    than the output.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param output: Output of the LLM, as a JSON-compatible object.
    :type output: Any
    :param faulty: Whether the response may be faulty. Defaults to True.
    :type faulty: bool
    :return: Text of the response, and whether it has the output.
    :rtype: tuple[str, bool]
    """
    text = json.dumps(output, ensure_ascii=draw(st.booleans()))
    forms = ["as_is", "as_is", "as_is", "fenced", "thinking"]
    form = draw(st.sampled_from([*forms, "altered"] if faulty else forms))
    if form == "as_is":
        return text, True
    if form == "fenced":
        return f"```json\n{text}\n```", True
    if form == "thinking":
        return f"<think>{draw(_TEXTS)}</think>\n{text}", True
    altered = [
        "",
        "Sorry, I cannot help with that.",
        text[:-1],
        json.dumps({"result": output}),
        json.dumps("[]"),
        "42",
        "null",
    ]
    if output and isinstance(output[0], dict):
        first_entry = dict(output[0])
        # A key that the output does not have
        altered.append(json.dumps([{**first_entry, "extra": 1}, *output[1:]]))
        # A key missing
        altered.append(json.dumps([dict(list(first_entry.items())[1:]), *output[1:]]))
    return draw(st.sampled_from(altered)), False


def _has_non_finite_number(value: Any) -> bool:
    """
    Checks whether a JSON-compatible object has a number that is not finite, i.e.
    NaN or an infinity, which JSON does not have, and which `json.dumps` writes as
    ``NaN``, ``Infinity`` or ``-Infinity``.

    :param value: JSON-compatible object.
    :type value: Any
    :return: True if the object has such a number, False otherwise.
    :rtype: bool
    """
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_non_finite_number(item) for item in value)
    return False


@dataclass(frozen=True)
class _ExpectedItem:
    """
    Item that the output of an LLM entails for an aspect or a concept.

    :ivar value: Value of the item, which is the text of a paragraph or of
        a sentence for an aspect.
    :vartype value: Any
    :ivar justification: Justification of the item, if justifications are
        extracted.
    :vartype justification: str | None
    :ivar paragraphs: Positions of the reference paragraphs of the item.
    :vartype paragraphs: tuple[int, ...]
    :ivar sentences: Positions of the reference sentences of the item, each with
        the position of its paragraph.
    :vartype sentences: tuple[tuple[int, int], ...]
    """

    value: Any
    justification: str | None = None
    paragraphs: tuple[int, ...] = ()
    sentences: tuple[tuple[int, int], ...] = ()


def _check_items(
    instance: Aspect | Any,
    expected_items: list[_ExpectedItem],
    document: Document,
) -> None:
    """
    Checks the extracted items of an aspect or a concept against the expected ones.
    The reference paragraphs and sentences of an item must be the paragraphs and
    sentences of the document themselves.

    :param instance: Aspect or concept.
    :type instance: Aspect | Any
    :param expected_items: Items expected, in their order.
    :type expected_items: list[_ExpectedItem]
    :param document: Document that the items are extracted from.
    :type document: Document
    :return: None
    :rtype: None
    """
    items = instance.extracted_items
    assert len(items) == len(expected_items), (items, expected_items)
    for item, expected in zip(items, expected_items, strict=True):
        assert item.value == expected.value
        assert type(item.value) is type(expected.value)
        assert item.justification == expected.justification
        reference_paragraphs = [document.paragraphs[p] for p in expected.paragraphs]
        assert len(item.reference_paragraphs) == len(reference_paragraphs)
        assert all(
            actual is reference
            for actual, reference in zip(
                item.reference_paragraphs, reference_paragraphs, strict=True
            )
        )
        reference_sentences = [
            document.paragraphs[p].sentences[s] for p, s in expected.sentences
        ]
        assert len(item.reference_sentences) == len(reference_sentences)
        assert all(
            actual is reference
            for actual, reference in zip(
                item.reference_sentences, reference_sentences, strict=True
            )
        )


def _check_custom_data_copy(item: Any, reference: Any) -> None:
    """
    Checks that an extracted item has the custom data of its reference paragraph or
    sentence, in a copy of its own: a change to a value nested in it, e.g. to
    the Docling data of a paragraph, leaves the reference as it is.

    :param item: Extracted item of an aspect.
    :type item: Any
    :param reference: The reference paragraph or sentence of the item, whose text
        is its value.
    :type reference: Any
    :return: None
    :rtype: None
    """
    assert item.custom_data == reference.custom_data
    for key, value in item.custom_data.items():
        if isinstance(value, dict | list):
            assert value is not reference.custom_data[key], key


# Output of an LLM for aspects


@st.composite
def _llm_aspect_outputs(
    draw: st.DrawFn,
    document: Document,
    aspect_count: int,
    reference_depth: str,
    add_justifications: bool,
    faulty: bool = True,
    nonempty: bool = False,
) -> list[dict[str, Any]]:
    """
    Generates the output of an LLM for aspects, in the structure that the prompt
    asks for. Each aspect is listed, some more than once, and a paragraph or
    a sentence may be listed more than once for an aspect. A faulty output may
    leave aspects out, and have IDs that are not those of the prompt, and
    justifications that are not valid.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param document: Document that the aspects are extracted from.
    :type document: Document
    :param aspect_count: Number of the aspects in the prompt.
    :type aspect_count: int
    :param reference_depth: Reference depth of the aspects.
    :type reference_depth: str
    :param add_justifications: Whether the aspects are extracted with
        justifications.
    :type add_justifications: bool
    :param faulty: Whether the output may be faulty. Defaults to True.
    :type faulty: bool
    :param nonempty: Whether each listing of an aspect has at least one
        paragraph, and each paragraph at least one sentence. Defaults to False.
    :type nonempty: bool
    :return: Output of the LLM.
    :rtype: list[dict[str, Any]]
    """
    paragraph_count = len(document.paragraphs)
    paragraph_ids = _ids("P", paragraph_count, faulty)
    justifications = _justifications(faulty)
    min_size = 1 if nonempty else 0
    output = []
    for aspect_id in draw(_entry_ids("A", aspect_count, faulty)):
        entry: dict[str, Any] = {"aspect_id": aspect_id}
        if reference_depth == "paragraphs" and not add_justifications:
            entry["paragraph_ids"] = draw(
                _lists_with_repeats(
                    paragraph_ids, max_size=paragraph_count + 1, min_size=min_size
                )
            )
            output.append(entry)
            continue
        paragraphs = []
        for paragraph_id in draw(
            _lists_with_repeats(
                paragraph_ids, max_size=paragraph_count, min_size=min_size
            )
        ):
            paragraph: dict[str, Any] = {"paragraph_id": paragraph_id}
            if reference_depth == "sentences":
                position = _get_position_of_id(paragraph_id, "P", paragraph_count)
                position = 0 if position is None else position
                sentence_ids = _sentence_ids(
                    position,
                    len(document.paragraphs[position].sentences),
                    paragraph_count,
                    faulty,
                )
                sentences = []
                for sentence_id in draw(
                    _lists_with_repeats(sentence_ids, max_size=3, min_size=min_size)
                ):
                    sentence: dict[str, Any] = {"sentence_id": sentence_id}
                    if add_justifications:
                        sentence["justification"] = draw(justifications)
                    sentences.append(sentence)
                paragraph["sentences"] = sentences
            else:
                paragraph["justification"] = draw(justifications)
            paragraphs.append(paragraph)
        entry["paragraphs"] = paragraphs
        output.append(entry)
    return output


def _get_expected_aspect_items(
    output: list[dict[str, Any]],
    document: Document,
    aspect_count: int,
    reference_depth: str,
    add_justifications: bool,
) -> dict[int, list[_ExpectedItem]] | None:
    """
    Returns the items that the output of an LLM entails for each aspect, or None
    if the output is to be rejected, e.g. as it leaves an aspect out: it must list
    each aspect, also one without items.

    The paragraphs of an aspect, or the sentences of a paragraph, are taken in
    the order of their positions, each paragraph or sentence once, i.e. the first
    time that it is listed for the aspect. An aspect listed more than once gets
    the items of each listing, in the order of the listings.

    :param output: Output of the LLM, which has the structure that the prompt
        asks for.
    :type output: list[dict[str, Any]]
    :param document: Document that the aspects are extracted from.
    :type document: Document
    :param aspect_count: Number of the aspects in the prompt.
    :type aspect_count: int
    :param reference_depth: Reference depth of the aspects.
    :type reference_depth: str
    :param add_justifications: Whether the aspects are extracted with
        justifications.
    :type add_justifications: bool
    :return: Items of each aspect listed, by the position of the aspect, or None.
    :rtype: dict[int, list[_ExpectedItem]] | None
    """
    paragraph_count = len(document.paragraphs)
    expected: dict[int, list[_ExpectedItem]] = {}
    seen: dict[int, set[tuple[int, int | None]]] = {}
    for entry in output:
        aspect_position = _get_position_of_id(entry["aspect_id"], "A", aspect_count)
        if aspect_position is None:
            return None
        items = expected.setdefault(aspect_position, [])
        listed = seen.setdefault(aspect_position, set())
        if reference_depth == "paragraphs" and not add_justifications:
            listings = [
                (_get_position_of_id(paragraph_id, "P", paragraph_count), None)
                for paragraph_id in entry["paragraph_ids"]
            ]
        else:
            listings = [
                (
                    _get_position_of_id(
                        paragraph["paragraph_id"], "P", paragraph_count
                    ),
                    paragraph,
                )
                for paragraph in entry["paragraphs"]
            ]
        if any(position is None for position, _ in listings):
            return None
        # Sorted by the positions of the paragraphs, keeping the order of
        # the listings of the same paragraph
        for paragraph_position, listing in sorted(listings, key=lambda x: x[0]):
            assert paragraph_position is not None
            paragraph = document.paragraphs[paragraph_position]
            if reference_depth == "paragraphs":
                justification = None
                if add_justifications:
                    assert listing is not None
                    justification = _get_justification(listing["justification"])
                    if justification is _INVALID:
                        return None
                if (paragraph_position, None) in listed:
                    continue
                listed.add((paragraph_position, None))
                items.append(
                    _ExpectedItem(
                        value=paragraph.raw_text,
                        justification=justification,
                        paragraphs=(paragraph_position,),
                    )
                )
                continue
            assert listing is not None
            sentence_listings = []
            for sentence in listing["sentences"]:
                sentence_position = _get_position_of_sentence_id(
                    sentence["sentence_id"],
                    paragraph_position,
                    len(paragraph.sentences),
                )
                if sentence_position is None:
                    return None
                justification = None
                if add_justifications:
                    justification = _get_justification(sentence["justification"])
                    if justification is _INVALID:
                        return None
                sentence_listings.append((sentence_position, justification))
            for sentence_position, justification in sorted(
                sentence_listings, key=lambda x: x[0]
            ):
                if (paragraph_position, sentence_position) in listed:
                    continue
                listed.add((paragraph_position, sentence_position))
                items.append(
                    _ExpectedItem(
                        value=paragraph.sentences[sentence_position].raw_text,
                        justification=justification,
                        paragraphs=(paragraph_position,),
                        sentences=((paragraph_position, sentence_position),),
                    )
                )
    if len(expected) < aspect_count:
        # An aspect is left out
        return None
    return expected


# Concepts, and the values that an LLM may return for them


# Any JSON value, without numbers that are not finite
_JSON_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**6), 10**6),
    st.floats(allow_nan=False, allow_infinity=False),
    _TEXTS,
)
_ANY_JSON = st.recursive(
    _JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=3)
        | st.dictionaries(st.text(max_size=5), children, max_size=3)
    ),
    max_leaves=6,
)
_NON_FINITE_NUMBERS = st.sampled_from([math.nan, math.inf, -math.inf])
# Any JSON value but a string
_NON_STRING_JSON = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(-(10**6), 10**6),
    st.floats(allow_nan=False, allow_infinity=False),
    st.lists(_TEXTS, max_size=2),
    st.dictionaries(st.text(max_size=3), _TEXTS, max_size=2),
)

# Values written into the text of an output in place of a placeholder, which
# `json.dumps` cannot write: a number too large for a float, an integer with
# more digits than Python converts, and arrays nested deeper than the parser
# can recurse. None of them is valid JSON for the output.
_RAW_JSON_VALUES = {
    "<too large float>": "1e400",
    "<too long integer>": "9" * 5000,
    "<too deep nesting>": "[" * 100_000 + "]" * 100_000,
}

_LABELS = ["Low", "medium", "HIGH", "Not applicable", "Ünïcode", "ネット", "متوسط", "x"]
# Names of the fields of JSON object structures: plain ones, ones that pydantic
# gives a meaning to or that are attributes of its BaseModel, and ones that are not
# Python identifiers
_FIELD_NAMES = [
    "name",
    "count",
    "price",
    "active",
    "tags",
    "kind",
    "address",
    "model_config",
    "model_fields",
    "json",
    "copy",
    "schema",
    "validate",
    "dict",
    "first name",
    "имя",
    "123",
    "class",
    "None",
]
# Values of the Literals of JSON object structures, of each JSON primitive type,
# among which some are equal while of different types: 1, 1.0 (1.5 is not) and True
_LITERAL_VALUES: list[Any] = ["x", "y", "Ünï", 0, 1, 2, 1.5, True, False, None]
_PRIMITIVE_TYPES: list[Any] = [str, int, float, bool]


@dataclass(frozen=True)
class _ConceptSpec:
    """
    Concept to extract, with the values that an LLM may return for it, and
    the value of the item that each of these values gives.

    :ivar concept: The concept.
    :vartype concept: Any
    :ivar values: Strategy of the values that an LLM may return for the concept,
        of which some are not valid for it.
    :vartype values: st.SearchStrategy[Any]
    :ivar valid_values: Strategy of the values that are valid for the concept.
    :vartype valid_values: st.SearchStrategy[Any]
    :ivar invalid_values: Strategy of the values that are not valid for
        the concept, e.g. of another type.
    :vartype invalid_values: st.SearchStrategy[Any]
    :ivar get_item_value: Function that returns the value of the item that
        a returned value gives, or `_INVALID` if the value is to be rejected.
    :vartype get_item_value: Callable[[Any], Any]
    """

    concept: Any
    values: st.SearchStrategy[Any]
    valid_values: st.SearchStrategy[Any]
    invalid_values: st.SearchStrategy[Any]
    get_item_value: Callable[[Any], Any]


def _is_optional(field_type: Any) -> bool:
    """
    Checks whether a type of a field of a JSON object structure is a union with
    None, i.e. whether the field may be left out of a value.

    :param field_type: Type of the field.
    :type field_type: Any
    :return: True if the type is a union with None, False otherwise.
    :rtype: bool
    """
    is_union = get_origin(field_type) is Union or isinstance(field_type, UnionType)
    return is_union and type(None) in get_args(field_type)


@st.composite
def _literal_types(draw: st.DrawFn) -> Any:
    """
    Generates a Literal of one to three values of any JSON primitive type.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :return: The Literal.
    :rtype: Any
    """
    values = draw(
        st.lists(
            st.sampled_from(_LITERAL_VALUES),
            min_size=1,
            max_size=3,
            unique_by=lambda value: (type(value), value),
        )
    )
    return Literal[tuple(values)]  # ty: ignore[invalid-type-form]


@st.composite
def _field_types(draw: st.DrawFn, depth: int = 0, of_field: bool = True) -> Any:
    """
    Generates a type of a field of a JSON object structure, in any of the forms
    that a structure accepts: a primitive type, ``NoneType``, a Literal, an optional
    primitive type or Literal (``X | None``, ``Optional[X]``, ``Union[X, None]``),
    a union of other types (``|`` or ``Union``), a list (``list[X]``, ``List[X]``
    or, as the type of a field itself, ``[X]`` for an X that is not a list or
    a mapping), a mapping with string keys (``dict``
    or ``Dict``), and as the type of a field itself, unless the depth is reached,
    a nested structure or a list of nested structures.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param depth: Depth of the field: the top-level fields of a structure are at
        depth 0, and the types in a list, a mapping or a nested structure one
        level deeper. Defaults to 0.
    :type depth: int
    :param of_field: Whether the type is the type of a field itself, and not of
        the items of a list or of the values of a mapping, which a nested structure
        or the ``[X]`` form of a list cannot be. Defaults to True.
    :type of_field: bool
    :return: The type.
    :rtype: Any
    """
    kinds = ["primitive", "primitive", "none", "literal", "optional", "union"]
    if depth < 3:
        kinds += ["list", "dict"]
    if of_field and depth < 2:
        kinds += ["structure", "structures"]
    kind = draw(st.sampled_from(kinds))
    if kind == "primitive":
        return draw(st.sampled_from(_PRIMITIVE_TYPES))
    if kind == "none":
        return type(None)
    if kind == "literal":
        return draw(_literal_types())
    if kind == "optional":
        inner = draw(st.sampled_from(_PRIMITIVE_TYPES) | _literal_types())
        return draw(
            st.sampled_from(
                [inner | None, Optional[inner], Union[inner, None]]  # noqa: UP007, UP045
            )
        )
    if kind == "union":
        members = draw(
            st.lists(
                st.sampled_from([*_PRIMITIVE_TYPES, list[str], dict[str, int]]),
                min_size=2,
                max_size=3,
                unique=True,
            )
        )
        if draw(st.booleans()):
            return Union[tuple(members)]  # noqa: UP007
        return functools.reduce(operator.or_, members)
    if kind == "list":
        item = draw(_field_types(depth + 1, of_field=False))
        # The [X] form of a list is accepted only as the type of a field itself,
        # and only for the items that are not lists or mappings
        forms = ["list", "List"]
        if of_field and get_origin(item) not in (list, dict):
            forms.append("shorthand")
        form = draw(st.sampled_from(forms))
        if form == "shorthand":
            return [item]
        return list[item] if form == "list" else List[item]  # noqa: UP006
    if kind == "dict":
        key = draw(st.sampled_from([str, str, Literal["a", "b"]]))
        value = draw(_field_types(depth + 1, of_field=False))
        return dict[key, value] if draw(st.booleans()) else Dict[key, value]  # noqa: UP006
    nested = draw(_structures(depth + 1))
    return nested if kind == "structure" else [nested]


@st.composite
def _structures(draw: st.DrawFn, depth: int = 0) -> dict[str, Any]:
    """
    Generates a structure of a JSON object concept, as a dictionary: one to four
    fields of any of the names of `_FIELD_NAMES` and of any of the types of
    `_field_types`, with structures nested up to three levels.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param depth: Depth of the structure: 0 for the structure of a concept.
        Defaults to 0.
    :type depth: int
    :return: The structure.
    :rtype: dict[str, Any]
    """
    names = draw(
        st.lists(st.sampled_from(_FIELD_NAMES), min_size=1, max_size=4, unique=True)
    )
    return {name: draw(_field_types(depth)) for name in names}


def _values_of_type(field_type: Any) -> st.SearchStrategy[Any]:
    """
    Returns a strategy of the values of a type of a field of a JSON object
    structure.

    :param field_type: Type of the field, of those of `_field_types`.
    :type field_type: Any
    :return: Strategy of the values of the type.
    :rtype: st.SearchStrategy[Any]
    """
    if isinstance(field_type, dict):
        return _values_of_structure(field_type)
    if isinstance(field_type, list):
        return st.lists(_values_of_type(field_type[0]), max_size=2)
    if field_type is type(None):
        return st.none()
    origin = get_origin(field_type)
    if origin is Literal:
        return st.sampled_from(get_args(field_type))
    if origin is Union or isinstance(field_type, UnionType):
        return st.one_of(*[_values_of_type(arg) for arg in get_args(field_type)])
    if origin is list:
        return st.lists(_values_of_type(get_args(field_type)[0]), max_size=2)
    if origin is dict:
        key_type, value_type = get_args(field_type)
        return st.dictionaries(
            _values_of_type(key_type), _values_of_type(value_type), max_size=2
        )
    return {
        str: _TEXTS,
        int: st.integers(-1000, 1000),
        float: st.floats(allow_nan=False, allow_infinity=False)
        | st.integers(-1000, 1000),
        bool: st.booleans(),
    }[field_type]


@st.composite
def _values_of_structure(draw: st.DrawFn, structure: dict[str, Any]) -> Any:
    """
    Generates a value of a JSON object structure, which an optional field may be
    left out of.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param structure: The structure.
    :type structure: dict[str, Any]
    :return: The value.
    :rtype: Any
    """
    value = {}
    for name, field_type in structure.items():
        if _is_optional(field_type) and draw(st.booleans()):
            continue
        value[name] = draw(_values_of_type(field_type))
    return value


# Values that a value of a field may be replaced with: of other JSON types, and
# values that are equal to others while of other types, or that look like them
_REPLACEMENT_VALUES = st.one_of(
    st.sampled_from([True, False, 0, 1, 1.0, 0.0, 1.5, None, "", "1", "true", [], {}]),
    _ANY_JSON,
)


@st.composite
def _altered_values_of_structure(draw: st.DrawFn, structure: dict[str, Any]) -> Any:
    """
    Generates a value of a JSON object structure that is altered at one place:
    the value of a field, at any depth, is replaced with another value, a field is
    left out, or a field that the structure does not have is added. The altered
    value may still be valid, e.g. when the field left out is optional, or when
    the value of a field is replaced with another value of its type.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param structure: The structure.
    :type structure: dict[str, Any]
    :return: The value.
    :rtype: Any
    """
    value = draw(_values_of_structure(structure))
    target = value
    # Down to an object nested in the value, at any depth
    while True:
        nested = [item for item in target.values() if isinstance(item, dict) and item]
        if not nested or draw(st.booleans()):
            break
        target = draw(st.sampled_from(nested))
    alteration = draw(st.sampled_from(["replace", "replace", "remove", "add"]))
    if alteration == "add" or not target:
        target["unexpected"] = draw(_JSON_SCALARS)
    elif alteration == "remove":
        del target[draw(st.sampled_from(sorted(target)))]
    else:
        target[draw(st.sampled_from(sorted(target)))] = draw(_REPLACEMENT_VALUES)
    return value


def _conforms_to_type(value: Any, field_type: Any) -> bool:
    """
    Checks whether a value conforms to a type of a field of a JSON object
    structure, which is validated strictly: a value is not converted to the type,
    except an integer, which a float accepts, and a value of a Literal must be of
    the type of the value of the Literal that it equals.

    :param value: The value.
    :type value: Any
    :param field_type: Type of the field, of those of `_field_types`.
    :type field_type: Any
    :return: True if the value conforms to the type, False otherwise.
    :rtype: bool
    """
    if isinstance(field_type, dict):
        return _conforms_to_structure(value, field_type)
    if isinstance(field_type, list):
        return isinstance(value, list) and all(
            _conforms_to_type(item, field_type[0]) for item in value
        )
    if field_type is type(None):
        return value is None
    origin = get_origin(field_type)
    if origin is Literal:
        return any(
            type(value) is type(allowed) and value == allowed
            for allowed in get_args(field_type)
        )
    if origin is Union or isinstance(field_type, UnionType):
        return any(_conforms_to_type(value, arg) for arg in get_args(field_type))
    if origin is list:
        return isinstance(value, list) and all(
            _conforms_to_type(item, get_args(field_type)[0]) for item in value
        )
    if origin is dict:
        key_type, value_type = get_args(field_type)
        return isinstance(value, dict) and all(
            _conforms_to_type(key, key_type) and _conforms_to_type(item, value_type)
            for key, item in value.items()
        )
    if field_type is float:
        return type(value) in (int, float)
    return type(value) is field_type


def _conforms_to_structure(value: Any, structure: dict[str, Any]) -> bool:
    """
    Checks whether a value conforms to a JSON object structure: a dict that has
    the fields of the structure and no other keys, of which only an optional
    field may be left out.

    :param value: The value.
    :type value: Any
    :param structure: The structure.
    :type structure: dict[str, Any]
    :return: True if the value conforms to the structure, False otherwise.
    :rtype: bool
    """
    if not isinstance(value, dict) or set(value) - set(structure):
        return False
    return all(
        _conforms_to_type(value[name], field_type)
        if name in value
        else _is_optional(field_type)
        for name, field_type in structure.items()
    )


def _get_json_object_value(value: Any, structure: dict[str, Any]) -> Any:
    """
    Returns the value of the item that a value returned by an LLM for a JSON
    object concept gives, i.e. the value itself if it conforms to the structure
    and is not empty, or `_INVALID`.

    :param value: The value.
    :type value: Any
    :param structure: The structure of the concept.
    :type structure: dict[str, Any]
    :return: The value of the item, or `_INVALID`.
    :rtype: Any
    """
    if _conforms_to_structure(value, structure) and value:
        return value
    return _INVALID


def _get_field_names(structure: dict[str, Any]) -> list[str]:
    """
    Returns the names of the fields of a JSON object structure and of the structures
    nested in it, at any depth.

    :param structure: The structure.
    :type structure: dict[str, Any]
    :return: The names.
    :rtype: list[str]
    """
    names = []
    for name, field_type in structure.items():
        names.append(name)
        nested = field_type[0] if isinstance(field_type, list) else field_type
        if isinstance(nested, dict):
            names.extend(_get_field_names(nested))
    return names


def _to_class_structure(
    structure: dict[str, Any], class_name: str = "Record"
) -> type | None:
    """
    Returns a structure of dataclasses equivalent to a structure of dictionaries,
    as JSON object concepts accept: each nested structure becomes a dataclass of
    its own, and all the dataclasses inherit from ``JsonObjectClassStruct``.

    :param structure: The structure of dictionaries.
    :type structure: dict[str, Any]
    :param class_name: Name of the dataclass. Defaults to "Record".
    :type class_name: str
    :return: The dataclass, or None if a field name is not a valid name of
        a dataclass field, e.g. "first name" or "class".
    :rtype: type | None
    """
    fields = []
    for position, (name, field_type) in enumerate(structure.items()):
        if not name.isidentifier() or keyword.iskeyword(name):
            return None
        nested = field_type[0] if isinstance(field_type, list) else field_type
        if isinstance(nested, dict):
            nested_class = _to_class_structure(nested, f"{class_name}{position}")
            if nested_class is None:
                return None
            field_type = (
                list[nested_class]  # ty: ignore[invalid-type-form]
                if isinstance(field_type, list)
                else nested_class
            )
        fields.append((name, field_type))
    return dataclasses.make_dataclass(
        class_name, fields, bases=(JsonObjectClassStruct,)
    )


def _accepts_json_object(concept: JsonObjectConcept, value: Any) -> bool:
    """
    Checks whether a JSON object concept accepts a value returned by an LLM, as
    the extraction does: the value must pass the concept, and make an item.

    :param concept: The concept.
    :type concept: JsonObjectConcept
    :param value: The value.
    :type value: Any
    :return: True if the value is accepted, False otherwise.
    :rtype: bool
    """
    try:
        concept._item_class(value=concept._process_item_value(value))
    except ValueError:
        return False
    return True


def _get_date(value: Any) -> Any:
    """
    Returns the date that a value returned by an LLM for a date concept gives,
    i.e. a text of a date in the DD-MM-YYYY format, or `_INVALID`.

    :param value: The value.
    :type value: Any
    :return: The date, or `_INVALID`.
    :rtype: Any
    """
    if not isinstance(value, str):
        return _INVALID
    try:
        return datetime.strptime(value.strip(), "%d-%m-%Y").date()
    except ValueError:
        return _INVALID


def _format_date(value: date) -> str:
    """
    Formats a date in the DD-MM-YYYY format.

    :param value: The date.
    :type value: date
    :return: The formatted date.
    :rtype: str
    """
    return f"{value.day:02d}-{value.month:02d}-{value.year:04d}"


def _get_labels(value: Any, labels: list[str], multi_class: bool) -> Any:
    """
    Returns the labels that a value returned by an LLM for a label concept gives,
    i.e. an object with the key "labels" only, whose value is a list of at least
    one label of the concept (one, for the "multi_class" classification type),
    without labels that differ in case only, or `_INVALID`.

    :param value: The value.
    :type value: Any
    :param labels: Labels of the concept.
    :type labels: list[str]
    :param multi_class: Whether the concept has the "multi_class" classification
        type.
    :type multi_class: bool
    :return: The labels, or `_INVALID`.
    :rtype: Any
    """
    if not isinstance(value, dict) or set(value) != {"labels"}:
        return _INVALID
    selected = value["labels"]
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(label, str) and label in labels for label in selected)
        or (multi_class and len(selected) > 1)
        or len({label.lower() for label in selected}) < len(selected)
    ):
        return _INVALID
    return selected


# Types of fields that a JSON object structure does not support: types that are
# not JSON, mappings with keys of other types than strings, and optional types
# that are not primitive types or Literals, and the [X] form of a list of lists or
# of mappings
_UNSUPPORTED_FIELD_TYPES: list[Any] = [
    tuple[int, int],
    set[str],
    frozenset[int],
    bytes,
    complex,
    object,
    None,
    dict[int, str],
    Dict[bool, int],  # noqa: UP006
    dict[float, str],
    dict[str | int, str],
    dict[Literal[1, 2], str],
    dict[str, dict[int, str]],
    list[dict[int, str]],
    [dict[int, int]],
    # A list of lists or of mappings in the [X] form
    [list[str]],
    [dict[str, int]],
    str | dict[int, int],
    Optional[list[int]],  # noqa: UP045
    Optional[dict[str, int]],  # noqa: UP045
    int | list[str] | None,
]

# Types of the concepts that an LLM extracts
_CONCEPT_TYPES = ["string", "boolean", "numerical", "rating", "date", "label", "json"]


@st.composite
def _concept_specs(
    draw: st.DrawFn,
    position: int,
    extraction_params: dict[str, Any],
    concept_type: str | None = None,
    singular_occurrence: bool | None = None,
) -> _ConceptSpec:
    """
    Generates a concept that an LLM extracts, of the given type or of any type,
    with the values that the LLM may return for it: mostly values of the type of
    the concept, and otherwise values of other types, or values of the type that
    the concept does not accept.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param position: Position of the concept, which gives its name.
    :type position: int
    :param extraction_params: Extraction params of the concept.
    :type extraction_params: dict[str, Any]
    :param concept_type: Type of the concept, one of `_CONCEPT_TYPES`. Defaults to
        None, i.e. any type.
    :type concept_type: str | None
    :param singular_occurrence: Whether the concept has a singular occurrence.
        Defaults to None, i.e. either.
    :type singular_occurrence: bool | None
    :return: The concept, with the values and the oracle of the item values.
    :rtype: _ConceptSpec
    """
    params: dict[str, Any] = {
        "name": f"Concept {position}",
        "description": f"Description of concept {position}",
        "singular_occurrence": (
            draw(st.booleans()) if singular_occurrence is None else singular_occurrence
        ),
        **extraction_params,
    }
    if concept_type is None:
        concept_type = draw(st.sampled_from(_CONCEPT_TYPES))
    if concept_type == "string":
        return _ConceptSpec(
            concept=StringConcept(**params),
            values=st.one_of(_TEXTS, _TEXTS, _ANY_JSON),
            valid_values=_NON_BLANK_TEXTS,
            invalid_values=st.one_of(
                st.sampled_from(["", "  ", "\n", "　"]), _NON_STRING_JSON
            ),
            get_item_value=_get_non_empty_text,
        )
    if concept_type == "boolean":
        return _ConceptSpec(
            concept=BooleanConcept(**params),
            values=st.one_of(st.booleans(), st.booleans(), _ANY_JSON),
            valid_values=st.booleans(),
            invalid_values=st.one_of(
                st.integers(0, 1),
                st.sampled_from(["true", "false", "yes"]),
                st.none(),
                st.floats(0, 1),
            ),
            get_item_value=lambda value: value if type(value) is bool else _INVALID,
        )
    if concept_type == "numerical":
        numeric_type = draw(st.sampled_from(_NUMERIC_TYPES))

        def get_number(value: Any) -> Any:
            """
            Returns the number that a value gives for the numeric type of
            the concept, or `_INVALID`.
            """
            if type(value) is float and not math.isfinite(value):
                return _INVALID
            if numeric_type == "int":
                return value if type(value) is int else _INVALID
            if numeric_type == "float" and type(value) is int:
                return float(value)
            return value if type(value) in (int, float) else _INVALID

        integers = st.integers(-(10**9), 10**9)
        floats = st.floats(allow_nan=False, allow_infinity=False)
        return _ConceptSpec(
            concept=NumericalConcept(numeric_type=numeric_type, **params),
            values=st.one_of(integers, floats, _NON_FINITE_NUMBERS, _ANY_JSON),
            valid_values=integers if numeric_type == "int" else integers | floats,
            # A float is not an integer, even with no fraction
            invalid_values=st.one_of(
                floats if numeric_type == "int" else st.nothing(),
                st.sampled_from(["5", "1,000", "five"]),
                st.booleans(),
                st.none(),
                st.lists(integers, max_size=2),
            ),
            get_item_value=get_number,
        )
    if concept_type == "rating":
        start = draw(st.integers(0, 3))
        end = draw(st.integers(start + 1, start + 5))
        return _ConceptSpec(
            concept=RatingConcept(rating_scale=(start, end), **params),
            values=st.one_of(
                st.integers(start - 2, end + 2),
                st.integers(start - 2, end + 2),
                st.floats(start, end),
                st.integers(start, end).map(str),
                _ANY_JSON,
            ),
            valid_values=st.integers(start, end),
            invalid_values=st.one_of(
                st.integers(end + 1, end + 5),
                st.integers(start - 5, start - 1),
                st.integers(start, end).map(str),
                st.floats(start, end),
                st.booleans(),
                st.none(),
            ),
            get_item_value=lambda value: (
                value if type(value) is int and start <= value <= end else _INVALID
            ),
        )
    if concept_type == "date":
        dates = st.dates(date(1900, 1, 1), date(2100, 12, 31))
        formatted_dates = st.one_of(
            dates.map(_format_date),
            dates.map(lambda value: f" {_format_date(value)}\n"),
        )
        return _ConceptSpec(
            concept=DateConcept(**params),
            values=st.one_of(
                formatted_dates,
                dates.map(lambda value: value.isoformat()),
                st.sampled_from(["31-02-2026", "00-01-2026", "1-1-26", "", "today"]),
                st.integers(19000101, 21001231),
                _ANY_JSON,
            ),
            valid_values=formatted_dates,
            invalid_values=st.one_of(
                dates.map(lambda value: value.isoformat()),
                st.sampled_from(["31-02-2026", "00-01-2026", "1-1-26", "", "today"]),
                st.integers(19000101, 21001231),
                st.none(),
                st.booleans(),
            ),
            get_item_value=_get_date,
        )
    if concept_type == "label":
        labels = draw(
            st.lists(st.sampled_from(_LABELS), min_size=2, max_size=4, unique=True)
        )
        multi_class = draw(st.booleans())
        concept = LabelConcept(
            labels=labels,
            classification_type="multi_class" if multi_class else "multi_label",
            **params,
        )
        selections = st.lists(
            st.sampled_from(labels),
            min_size=1,
            max_size=1 if multi_class else len(labels),
            unique=True,
        )
        return _ConceptSpec(
            concept=concept,
            values=st.one_of(
                selections.map(lambda selected: {"labels": selected}),
                selections.map(lambda selected: {"labels": selected}),
                selections.map(lambda selected: {"labels": [*selected, "Unknown"]}),
                selections.map(lambda selected: {"labels": [*selected, *selected]}),
                selections.map(lambda selected: {"labels": [selected[0].swapcase()]}),
                # Two labels, which are one too many for the "multi_class" type
                st.lists(
                    st.sampled_from(labels), min_size=2, max_size=2, unique=True
                ).map(lambda selected: {"labels": selected}),
                selections.map(lambda selected: {"labels": [f" {selected[0]}"]}),
                selections.map(lambda selected: {"labels": selected, "extra": 1}),
                selections,
                st.just({"labels": []}),
                _ANY_JSON,
            ),
            valid_values=selections.map(lambda selected: {"labels": selected}),
            invalid_values=st.one_of(
                selections.map(lambda selected: {"labels": [*selected, "Unknown"]}),
                selections.map(lambda selected: {"labels": [*selected, *selected]}),
                selections.map(lambda selected: {"labels": [f" {selected[0]}"]}),
                selections.map(lambda selected: {"labels": selected, "extra": 1}),
                selections,
                st.just({"labels": []}),
                st.none(),
                # Two labels, which are one too many for the "multi_class" type
                st.lists(
                    st.sampled_from(labels), min_size=2, max_size=2, unique=True
                ).map(lambda selected: {"labels": selected})
                if multi_class
                else st.nothing(),
            ),
            get_item_value=lambda value: _get_labels(value, labels, multi_class),
        )
    structure = draw(_structures())
    valid_values = _values_of_structure(structure)
    altered_values = _altered_values_of_structure(structure)
    return _ConceptSpec(
        concept=JsonObjectConcept(structure=structure, **params),
        values=st.one_of(valid_values, valid_values, altered_values, _ANY_JSON),
        valid_values=valid_values,
        invalid_values=st.one_of(
            altered_values,
            st.none(),
            st.lists(st.integers(), max_size=2),
            st.integers(),
            _TEXTS,
        ),
        get_item_value=lambda value: _get_json_object_value(value, structure),
    )


# Output of an LLM for concepts


@st.composite
def _llm_concept_outputs(
    draw: st.DrawFn,
    specs: list[_ConceptSpec],
    document: Document,
    add_justifications: bool,
    add_references: bool,
    reference_depth: str,
    faulty: bool = True,
    nonempty: bool = False,
) -> list[dict[str, Any]]:
    """
    Generates the output of an LLM for concepts, in the structure that the prompt
    asks for. Each concept is listed, some more than once, and a paragraph or
    a sentence may be listed more than once among the references of an item.
    A concept with a singular occurrence gets one item at most, which is what
    the prompt asks for. A faulty output may leave concepts out, and have IDs
    that are not those of the prompt, values that are not valid for
    the concepts, and justifications that are not valid.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param specs: The concepts, in the order of the prompt.
    :type specs: list[_ConceptSpec]
    :param document: Document that the concepts are extracted from.
    :type document: Document
    :param add_justifications: Whether the concepts are extracted with
        justifications.
    :type add_justifications: bool
    :param add_references: Whether the concepts are extracted with references.
    :type add_references: bool
    :param reference_depth: Reference depth of the concepts.
    :type reference_depth: str
    :param faulty: Whether the output may be faulty. Defaults to True.
    :type faulty: bool
    :param nonempty: Whether the first listing of each concept has an item, and
        each list of the references of an item at least one element. Defaults
        to False.
    :type nonempty: bool
    :return: Output of the LLM.
    :rtype: list[dict[str, Any]]
    """
    paragraph_count = len(document.paragraphs)
    paragraph_ids = _ids("P", paragraph_count, faulty)
    justifications = _justifications(faulty)
    min_size = 1 if nonempty else 0
    # The number of the items of each concept, by the IDs of the concepts
    item_counts: Counter[int] = Counter()
    listed_positions: set[int | None] = set()
    output = []
    for concept_id in draw(_entry_ids("C", len(specs), faulty)):
        position = _get_position_of_id(concept_id, "C", len(specs))
        spec = specs[position] if position is not None else specs[0]
        max_items = 2
        if position is not None and spec.concept.singular_occurrence:
            max_items = 1 - item_counts[position]
        # The first listing of a concept has an item, if the output is not to be
        # empty
        first_listing = position not in listed_positions
        listed_positions.add(position)
        min_items = min(max_items, 1 if nonempty and first_listing else 0)
        items: list[Any] = []
        for _ in range(draw(st.integers(min_items, max_items))):
            value = draw(spec.values if faulty else spec.valid_values)
            if not (add_justifications or add_references):
                items.append(value)
                continue
            item: dict[str, Any] = {"value": value}
            if add_justifications:
                item["justification"] = draw(justifications)
            if add_references and reference_depth == "paragraphs":
                item["reference_paragraph_ids"] = draw(
                    _lists_with_repeats(paragraph_ids, max_size=3, min_size=min_size)
                )
            elif add_references:
                reference_paragraphs = []
                for paragraph_id in draw(
                    _lists_with_repeats(paragraph_ids, max_size=3, min_size=min_size)
                ):
                    paragraph_position = _get_position_of_id(
                        paragraph_id, "P", paragraph_count
                    )
                    paragraph_position = (
                        0 if paragraph_position is None else paragraph_position
                    )
                    sentence_ids = _sentence_ids(
                        paragraph_position,
                        len(document.paragraphs[paragraph_position].sentences),
                        paragraph_count,
                        faulty,
                    )
                    reference_paragraphs.append(
                        {
                            "reference_paragraph_id": paragraph_id,
                            "reference_sentence_ids": draw(
                                _lists_with_repeats(
                                    sentence_ids, max_size=3, min_size=min_size
                                )
                            ),
                        }
                    )
                item["reference_paragraphs"] = reference_paragraphs
            items.append(item)
        if position is not None:
            item_counts[position] += len(items)
        output.append({"concept_id": concept_id, "extracted_items": items})
    return output


def _get_expected_references(
    item: dict[str, Any], document: Document, reference_depth: str
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None:
    """
    Returns the references that an item returned by an LLM for a concept entails,
    or None if the references are to be rejected.

    The reference paragraphs are in the order of their positions, each once.
    The reference sentences are in the order of the positions of their paragraphs,
    keeping the order of the listings of the same paragraph, and in the order of
    their positions within a listing, each once.

    :param item: The item returned by the LLM, with its references.
    :type item: dict[str, Any]
    :param document: Document that the concept is extracted from.
    :type document: Document
    :param reference_depth: Reference depth of the concept.
    :type reference_depth: str
    :return: Positions of the reference paragraphs, and of the reference
        sentences, each with the position of its paragraph, or None.
    :rtype: tuple[tuple[int, ...], tuple[tuple[int, int], ...]] | None
    """
    paragraph_count = len(document.paragraphs)
    if reference_depth == "paragraphs":
        positions: list[int] = []
        for paragraph_id in item["reference_paragraph_ids"]:
            position = _get_position_of_id(paragraph_id, "P", paragraph_count)
            if position is None:
                return None
            positions.append(position)
        return tuple(sorted(set(positions))), ()
    listings: list[tuple[int, list[int]]] = []
    for reference_paragraph in item["reference_paragraphs"]:
        paragraph_position = _get_position_of_id(
            reference_paragraph["reference_paragraph_id"], "P", paragraph_count
        )
        if paragraph_position is None:
            return None
        sentence_positions: list[int] = []
        for sentence_id in reference_paragraph["reference_sentence_ids"]:
            sentence_position = _get_position_of_sentence_id(
                sentence_id,
                paragraph_position,
                len(document.paragraphs[paragraph_position].sentences),
            )
            if sentence_position is None:
                return None
            sentence_positions.append(sentence_position)
        listings.append((paragraph_position, sentence_positions))
    paragraphs: list[int] = []
    sentences: list[tuple[int, int]] = []
    for paragraph_position, sentence_positions in sorted(listings, key=lambda x: x[0]):
        if paragraph_position not in paragraphs:
            paragraphs.append(paragraph_position)
        for sentence_position in sorted(sentence_positions):
            if (paragraph_position, sentence_position) not in sentences:
                sentences.append((paragraph_position, sentence_position))
    return tuple(paragraphs), tuple(sentences)


def _get_expected_concept_items(
    output: list[dict[str, Any]],
    specs: list[_ConceptSpec],
    document: Document,
    add_justifications: bool,
    add_references: bool,
    reference_depth: str,
) -> dict[int, list[_ExpectedItem]] | None:
    """
    Returns the items that the output of an LLM entails for each concept, or None
    if the output is to be rejected: if any of its IDs, values, justifications or
    references is to be rejected, if it has a number that is not finite, which
    is not valid JSON, if it leaves a concept out (it must list each concept,
    also one without items), or if it has more than one item for a concept with
    a singular occurrence. A concept listed more than once gets the items of each
    listing, in the order of the listings.

    :param output: Output of the LLM, which has the structure that the prompt
        asks for.
    :type output: list[dict[str, Any]]
    :param specs: The concepts, in the order of the prompt.
    :type specs: list[_ConceptSpec]
    :param document: Document that the concepts are extracted from.
    :type document: Document
    :param add_justifications: Whether the concepts are extracted with
        justifications.
    :type add_justifications: bool
    :param add_references: Whether the concepts are extracted with references.
    :type add_references: bool
    :param reference_depth: Reference depth of the concepts.
    :type reference_depth: str
    :return: Items of each concept listed, by the position of the concept, or None.
    :rtype: dict[int, list[_ExpectedItem]] | None
    """
    if _has_non_finite_number(output):
        return None
    expected: dict[int, list[_ExpectedItem]] = {}
    for entry in output:
        position = _get_position_of_id(entry["concept_id"], "C", len(specs))
        if position is None:
            return None
        items = expected.setdefault(position, [])
        for item in entry["extracted_items"]:
            value = item["value"] if add_justifications or add_references else item
            item_value = specs[position].get_item_value(value)
            if item_value is _INVALID:
                return None
            justification = None
            if add_justifications:
                justification = _get_justification(item["justification"])
                if justification is _INVALID:
                    return None
            paragraphs: tuple[int, ...] = ()
            sentences: tuple[tuple[int, int], ...] = ()
            if add_references:
                references = _get_expected_references(item, document, reference_depth)
                if references is None:
                    return None
                paragraphs, sentences = references
            items.append(
                _ExpectedItem(
                    value=item_value,
                    justification=justification,
                    paragraphs=paragraphs,
                    sentences=sentences,
                )
            )
    if len(expected) < len(specs):
        # A concept is left out
        return None
    if any(
        spec.concept.singular_occurrence and len(expected[position]) > 1
        for position, spec in enumerate(specs)
    ):
        # A concept with a singular occurrence has more than one item
        return None
    return expected


# Answers of a decision model


@dataclass(frozen=True)
class _AnswerSpec:
    """
    Answer of a decision model to a question, as returned by litellm.

    :ivar kind: Kind of the answer: "predicate", "choice", "score" or "refusal".
    :vartype kind: str
    :ivar probability: Probability of "yes", for a predicate.
    :vartype probability: float | None
    :ivar option_probabilities: Labels of the options with their probabilities,
        as returned, for a "choice" or a "score" answer.
    :vartype option_probabilities: tuple[tuple[str, float], ...]
    :ivar chosen: Label of the option that a "choice" answer names as picked.
    :vartype chosen: str | None
    """

    kind: str
    probability: float | None = None
    option_probabilities: tuple[tuple[str, float], ...] = ()
    chosen: str | None = None

    def to_answer(self, name: str | None) -> Any:
        """
        Returns the answer as litellm returns it.

        :param name: Name of the answer, which litellm returns as the name of
            the question, or None.
        :type name: str | None
        :return: Answer of litellm.
        :rtype: Any
        """
        if self.kind == "refusal":
            return OpenAIRefusalAnswer(name=name)
        if self.kind == "predicate":
            return OpenAIPredicateAnswer(
                name=name, probability=cast(float, self.probability)
            )
        confidence = max((p for _, p in self.option_probabilities), default=0.0)
        if self.kind == "choice":
            return OpenAIChoiceAnswer(
                name=name,
                choice=self.chosen or "",
                confidence=confidence,
                probabilities=tuple(
                    OpenAIChoiceProbability(value=label, probability=p)
                    for label, p in self.option_probabilities
                ),
            )
        return OpenAIScoreAnswer(
            name=name,
            score=0.0,
            confidence=confidence,
            probabilities=tuple(
                OpenAIScoreProbability(value=position, label=label, probability=p)
                for position, (label, p) in enumerate(self.option_probabilities)
            ),
        )


# Probabilities that a decision model may return: a number within [0, 1], or
# a number that is not one, or NaN, which litellm passes on as they are
# A few values, so that probabilities are often equal
_VALID_PROBABILITIES = st.one_of(
    st.floats(0, 1),
    st.sampled_from([0.0, 0.5, 1.0]),
    st.sampled_from([0.0, 0.5, 1.0]),
)
_INVALID_PROBABILITIES = st.one_of(
    st.floats(min_value=1, exclude_min=True),
    st.floats(max_value=0, exclude_max=True),
    st.just(math.nan),
)
_PROBABILITIES = st.one_of(
    _VALID_PROBABILITIES,
    _VALID_PROBABILITIES,
    _VALID_PROBABILITIES,
    _INVALID_PROBABILITIES,
)


def _is_probability(value: Any) -> bool:
    """
    Checks whether a probability returned by a decision model is a number within
    [0, 1].

    :param value: The probability.
    :type value: Any
    :return: True if it is such a number, False otherwise, also for NaN.
    :rtype: bool
    """
    return isinstance(value, float) and 0 <= value <= 1


@st.composite
def _option_answers(
    draw: st.DrawFn,
    kind: Literal["choice", "score"],
    labels: list[str],
    faulty: bool = True,
) -> _AnswerSpec:
    """
    Generates an answer of a decision model to a "choice" or a "score" question
    with the options of the given labels: the probability of each of these
    options, and a "choice" answer picks the most probable one. A faulty answer
    may have a probability that is not a number within [0, 1], the probabilities
    of other options than these, of some of them, of an option twice, or of none
    of them above zero, or may pick an option not offered.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param kind: Kind of the answer.
    :type kind: Literal["choice", "score"]
    :param labels: Labels of the options of the question.
    :type labels: list[str]
    :param faulty: Whether the answer may be faulty. Defaults to True.
    :type faulty: bool
    :return: The answer.
    :rtype: _AnswerSpec
    """
    probabilities = draw(
        st.lists(
            _PROBABILITIES if faulty else _VALID_PROBABILITIES,
            min_size=len(labels),
            max_size=len(labels),
        )
    )
    pairs = tuple(zip(labels, probabilities, strict=True))
    alterations = [
        "none",
        "none",
        "none",
        "missing",
        "other",
        "repeated",
        "zeros",
        "empty",
    ]
    alteration = draw(st.sampled_from(alterations)) if faulty else "none"
    if alteration == "missing":
        pairs = pairs[1:]
    elif alteration == "other":
        pairs = (*pairs, ("Not offered", 0.5))
    elif alteration == "repeated":
        pairs = (*pairs, pairs[0])
    elif alteration == "zeros":
        pairs = tuple((label, 0.0) for label, _ in pairs)
    elif alteration == "empty":
        pairs = ()
    chosen = None
    if kind == "choice":
        valid_pairs = [(label, p) for label, p in pairs if _is_probability(p)]
        chosen = (
            max(valid_pairs, key=lambda pair: pair[1])[0] if valid_pairs else labels[0]
        )
        if faulty:
            chosen = draw(st.sampled_from([chosen, chosen, chosen, "Not offered"]))
    return _AnswerSpec(kind=kind, option_probabilities=pairs, chosen=chosen)


def _answers(
    kind: Literal["predicate", "choice", "score"],
    labels: list[str] | None = None,
    faulty: bool = True,
    refusals: bool = True,
) -> st.SearchStrategy[_AnswerSpec]:
    """
    Returns a strategy of the answers of a decision model to a question of
    the given kind: mostly an answer of that kind, and otherwise a refusal.
    A faulty answer may be of another kind, or faulty in itself (see
    `_option_answers`), e.g. have a probability that is not a number within [0, 1].

    :param kind: Kind of the question.
    :type kind: Literal["predicate", "choice", "score"]
    :param labels: Labels of the options of a "choice" or a "score" question.
    :type labels: list[str] | None
    :param faulty: Whether the answers may be faulty. Defaults to True.
    :type faulty: bool
    :param refusals: Whether a refusal is among the answers. Defaults to True.
    :type refusals: bool
    :return: Strategy of the answers.
    :rtype: st.SearchStrategy[_AnswerSpec]
    """
    predicates = (_PROBABILITIES if faulty else _VALID_PROBABILITIES).map(
        lambda p: _AnswerSpec(kind="predicate", probability=p)
    )
    if kind == "predicate":
        answers = predicates
        others = _option_answers("choice", ["yes", "no"])
    else:
        assert labels is not None
        answers = _option_answers(kind, labels, faulty=faulty)
        others = predicates
    strategies = [answers] * 6
    if refusals:
        strategies.append(st.just(_AnswerSpec(kind="refusal")))
    if faulty:
        strategies.append(others)
    return st.one_of(*strategies)


def _get_predicate_probability(answer: _AnswerSpec) -> Any:
    """
    Returns the probability of "yes" of an answer to a predicate, or `_INVALID` if
    the answer cannot be used: it is refused, of another kind, or its probability
    is not a number within [0, 1].

    :param answer: The answer.
    :type answer: _AnswerSpec
    :return: The probability, or `_INVALID`.
    :rtype: Any
    """
    if answer.kind != "predicate" or not _is_probability(answer.probability):
        return _INVALID
    return answer.probability


def _get_option_probabilities(answer: _AnswerSpec, kind: str, labels: list[str]) -> Any:
    """
    Returns the probabilities of the options of an answer to a "choice" or
    a "score" question, by the labels of the options in their order, or `_INVALID`
    if the answer cannot be used: it is refused or of another kind, it does not
    have the probability of each option offered, once, and of no other option,
    a probability is not a number within [0, 1], none of them is above zero, or
    it picks an option not offered.

    :param answer: The answer.
    :type answer: _AnswerSpec
    :param kind: Kind of the question.
    :type kind: str
    :param labels: Labels of the options offered.
    :type labels: list[str]
    :return: The probabilities, or `_INVALID`.
    :rtype: Any
    """
    returned_labels = [label for label, _ in answer.option_probabilities]
    if (
        answer.kind != kind
        or sorted(returned_labels) != sorted(labels)
        or not all(_is_probability(p) for _, p in answer.option_probabilities)
        or not any(p for _, p in answer.option_probabilities)
        or (kind == "choice" and answer.chosen not in labels)
    ):
        return _INVALID
    probabilities = dict(answer.option_probabilities)
    return {label: probabilities[label] for label in labels}


def _has_repeated_option(answer: _AnswerSpec) -> bool:
    """
    Checks whether an answer has the probability of an option more than once,
    which makes the whole response unusable, whatever question it answers.

    :param answer: The answer.
    :type answer: _AnswerSpec
    :return: True if an option is repeated, False otherwise.
    :rtype: bool
    """
    labels = [label for label, _ in answer.option_probabilities]
    return len(set(labels)) < len(labels)


@st.composite
def _responses_of_answers(
    draw: st.DrawFn,
    names: list[str],
    answers: dict[str, _AnswerSpec],
    faulty: bool = True,
) -> tuple[list[Any], bool]:
    """
    Generates the answers of a response of a decision model to the questions of
    the given names, as litellm returns them: one answer for each question, in
    the order of the questions. An answer may have no name, which litellm allows.
    In a faulty response, an answer may be missing, there may be an extra one,
    or an answer may have the name of another question.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :param names: Names of the questions of the request, in their order.
    :type names: list[str]
    :param answers: Answers to the questions, by their names.
    :type answers: dict[str, _AnswerSpec]
    :param faulty: Whether the response may be faulty. Defaults to True.
    :type faulty: bool
    :return: The answers of the response, and whether the response can be
        attributed to the questions.
    :rtype: tuple[list[Any], bool]
    """
    response_answers = [
        answers[name].to_answer(None if draw(st.integers(0, 4)) == 0 else name)
        for name in names
    ]
    alterations = ["none"] * 4 + ["missing", "extra", "renamed"]
    alteration = draw(st.sampled_from(alterations)) if faulty else "none"
    if alteration == "missing":
        return response_answers[:-1], False
    if alteration == "extra":
        return [*response_answers, OpenAIRefusalAnswer(name="Extra")], False
    if alteration == "renamed":
        renamed = list(response_answers)
        renamed[0] = answers[names[0]].to_answer(f"{names[0]}-renamed")
        return renamed, False
    return response_answers, True


# Faults that an output of an LLM may have, each of which makes it invalid:
# an ID that is not one of the prompt, a sentence ID of another paragraph than
# the one that it is listed for, a justification that is not valid, a value that
# the concept does not accept, a value that is not valid JSON, an aspect or
# a concept left out, and a second item of a concept with a singular occurrence
_ASPECT_FAULTS = [
    "aspect_id",
    "missing_entry",
    "paragraph_id",
    "sentence_id_of_other_paragraph",
    "malformed_sentence_id",
    "justification",
]
_CONCEPT_FAULTS = [
    "concept_id",
    "reference_paragraph_id",
    "sentence_id_of_other_paragraph",
    "malformed_sentence_id",
    "justification",
    "value",
    "invalid_json_value",
    "missing_entry",
    "second_item_of_singular_concept",
]


@dataclass
class _FaultSlot:
    """
    Place of an output of an LLM that a fault can be injected in.

    :ivar container: The dict or the list that holds the value.
    :vartype container: Any
    :ivar key: The key or the index of the value in the container.
    :vartype key: Any
    :ivar strategy: Strategy of the faulty values for the place.
    :vartype strategy: st.SearchStrategy[Any]
    """

    container: Any
    key: Any
    strategy: st.SearchStrategy[Any]


def _get_fault_slots(
    output: list[dict[str, Any]],
    fault: str,
    document: Document,
    instance_count: int,
    specs: list[_ConceptSpec] | None = None,
) -> list[_FaultSlot]:
    """
    Returns the places of a valid output of an LLM, for aspects or for concepts,
    that a fault of the given kind can be injected in, each with the faulty values
    for it.

    :param output: The output, valid, for aspects or for concepts.
    :type output: list[dict[str, Any]]
    :param fault: Kind of the fault, one of `_ASPECT_FAULTS` or `_CONCEPT_FAULTS`
        that replace a value of the output: no places are returned for the others.
    :type fault: str
    :param document: Document that the output is for.
    :type document: Document
    :param instance_count: Number of the aspects or concepts in the prompt.
    :type instance_count: int
    :param specs: The concepts, for an output for concepts. Defaults to None,
        for an output for aspects.
    :type specs: list[_ConceptSpec] | None
    :return: The places.
    :rtype: list[_FaultSlot]
    """
    paragraph_count = len(document.paragraphs)
    invalid_paragraph_ids = _invalid_ids("P", paragraph_count)
    slots: list[_FaultSlot] = []

    def get_sentence_id_faults(paragraph_id: str) -> st.SearchStrategy[str] | None:
        """
        Returns the faulty sentence IDs for the sentences listed for a paragraph,
        if the fault is one of a sentence ID.
        """
        position = _get_position_of_id(paragraph_id, "P", paragraph_count)
        assert position is not None
        sentence_count = len(document.paragraphs[position].sentences)
        if fault == "sentence_id_of_other_paragraph":
            return _sentence_ids_of_other_paragraphs(
                position, sentence_count, paragraph_count
            )
        if fault == "malformed_sentence_id":
            return _malformed_sentence_ids(position, sentence_count)
        return None

    for entry in output:
        if specs is None:
            # Output for aspects
            if fault == "aspect_id":
                slots.append(
                    _FaultSlot(entry, "aspect_id", _invalid_ids("A", instance_count))
                )
            if fault == "paragraph_id":
                slots.extend(
                    _FaultSlot(entry["paragraph_ids"], index, invalid_paragraph_ids)
                    for index in range(len(entry.get("paragraph_ids", [])))
                )
            for paragraph in entry.get("paragraphs", []):
                if fault == "paragraph_id":
                    slots.append(
                        _FaultSlot(paragraph, "paragraph_id", invalid_paragraph_ids)
                    )
                if fault == "justification" and "justification" in paragraph:
                    slots.append(
                        _FaultSlot(paragraph, "justification", _INVALID_JUSTIFICATIONS)
                    )
                sentence_id_faults = get_sentence_id_faults(paragraph["paragraph_id"])
                for sentence in paragraph.get("sentences", []):
                    if fault == "justification" and "justification" in sentence:
                        slots.append(
                            _FaultSlot(
                                sentence, "justification", _INVALID_JUSTIFICATIONS
                            )
                        )
                    if sentence_id_faults is not None:
                        slots.append(
                            _FaultSlot(sentence, "sentence_id", sentence_id_faults)
                        )
            continue
        # Output for concepts
        if fault == "concept_id":
            slots.append(
                _FaultSlot(entry, "concept_id", _invalid_ids("C", instance_count))
            )
        position = _get_position_of_id(entry["concept_id"], "C", instance_count)
        assert position is not None
        spec = specs[position]
        items = entry["extracted_items"]
        for index, item in enumerate(items):
            # The value of an item without extra data is the item itself
            holder, key = (item, "value") if isinstance(item, dict) else (items, index)
            if fault == "value":
                slots.append(
                    _FaultSlot(
                        holder,
                        key,
                        spec.invalid_values.filter(
                            lambda value, spec=spec: (
                                spec.get_item_value(value) is _INVALID
                            )
                        ),
                    )
                )
            if fault == "invalid_json_value":
                slots.append(
                    _FaultSlot(
                        holder,
                        key,
                        _NON_FINITE_NUMBERS | st.sampled_from(list(_RAW_JSON_VALUES)),
                    )
                )
            if not isinstance(item, dict):
                continue
            if fault == "justification" and "justification" in item:
                slots.append(_FaultSlot(item, "justification", _INVALID_JUSTIFICATIONS))
            if fault == "reference_paragraph_id":
                slots.extend(
                    _FaultSlot(
                        item["reference_paragraph_ids"], index, invalid_paragraph_ids
                    )
                    for index in range(len(item.get("reference_paragraph_ids", [])))
                )
            for reference_paragraph in item.get("reference_paragraphs", []):
                if fault == "reference_paragraph_id":
                    slots.append(
                        _FaultSlot(
                            reference_paragraph,
                            "reference_paragraph_id",
                            invalid_paragraph_ids,
                        )
                    )
                sentence_id_faults = get_sentence_id_faults(
                    reference_paragraph["reference_paragraph_id"]
                )
                if sentence_id_faults is not None:
                    sentence_ids = reference_paragraph["reference_sentence_ids"]
                    slots.extend(
                        _FaultSlot(sentence_ids, index, sentence_id_faults)
                        for index in range(len(sentence_ids))
                    )
    return slots


def _write_output(output: Any) -> str:
    """
    Writes an output of an LLM as JSON, with the raw JSON values of
    `_RAW_JSON_VALUES` in place of their placeholders.

    :param output: The output.
    :type output: Any
    :return: The text of the output.
    :rtype: str
    """
    text = json.dumps(output)
    for placeholder, raw_value in _RAW_JSON_VALUES.items():
        text = text.replace(json.dumps(placeholder), raw_value)
    return text


# Answers of a decision model to the questions of the aspects on each text: by
# the position of the paragraph and of the sentence, which is None for the text of
# a paragraph, and by the position of the aspect in the document
_AnswersOnTexts = dict[tuple[int, int | None], dict[int, _AnswerSpec]]


def _get_answer_probability(answer: _AnswerSpec) -> Any:
    """
    Returns the probability of "yes" of an answer to the question of an aspect,
    None if the answer is refused, or `_INVALID` if it cannot be used otherwise.

    :param answer: The answer.
    :type answer: _AnswerSpec
    :return: The probability, None, or `_INVALID`.
    :rtype: Any
    """
    if answer.kind == "refusal":
        return None
    return _get_predicate_probability(answer)


def _is_paragraph_one_whole_sentence(paragraph: _Paragraph) -> bool:
    """
    Returns whether a paragraph has one sentence that is the paragraph as a whole,
    in what a decision model is shown: the same cleaned text, and no additional
    context of its own, i.e. none or that of the paragraph once cleaned.

    :param paragraph: Paragraph.
    :type paragraph: _Paragraph
    :return: Whether a decision model is not asked about the sentence of
        the paragraph, which takes the answers on the paragraph instead.
    :rtype: bool
    """

    def clean(text: str | None) -> str:
        """
        Cleans a text as the text of a request to a decision model is cleaned.
        """
        return _clean_text_for_llm_prompt(
            text or "", preserve_linebreaks=False, strip_text=True
        )

    if len(paragraph.sentences) != 1:
        return False
    (sentence,) = paragraph.sentences
    return clean(sentence.raw_text) == clean(paragraph.raw_text) and clean(
        sentence.additional_context
    ) in ("", clean(paragraph.additional_context))


def _get_expected_decision_aspect_items(
    positions: list[int],
    reference_depth: str,
    document: Document,
    thresholds: list[float],
    answers: _AnswersOnTexts,
    attributable: dict[tuple[int, int | None], bool],
) -> tuple[dict[int, list[tuple[int, int | None, float]]], set[int]] | None:
    """
    Returns what the answers of a decision model entail for the aspects of a call,
    which have the same reference depth, or None if the call is to fail.

    Each paragraph is asked about for all the aspects of the call. A response that
    cannot be attributed to its questions, or an answer that is neither a refusal
    nor a probability within [0, 1], fails the call, as does the refusal to answer
    the question of an aspect on every paragraph. For the "sentences" reference
    depth, each sentence of a paragraph that belongs to some aspects is then asked
    about for these aspects, where the same applies, except for the refusals,
    unless the paragraph is one sentence as a whole, which takes the answers on
    the paragraph.

    :param positions: Positions of the aspects of the call in the document.
    :type positions: list[int]
    :param reference_depth: Reference depth of the aspects of the call.
    :type reference_depth: str
    :param document: Document that the aspects are extracted from.
    :type document: Document
    :param thresholds: Decision threshold of each aspect of the document.
    :type thresholds: list[float]
    :param answers: Answers on each text.
    :type answers: _AnswersOnTexts
    :param attributable: Whether the response on each text can be attributed to
        its questions.
    :type attributable: dict[tuple[int, int | None], bool]
    :return: The items of each aspect of the call (the position of the paragraph,
        of the sentence or None, and the probability of "yes"), and the positions of
        the aspects with refused answers, or None.
    :rtype: tuple[dict[int, list[tuple[int, int | None, float]]], set[int]] | None
    """
    paragraph_positions = range(len(document.paragraphs))
    probabilities = {
        paragraph_position: {
            position: _get_answer_probability(
                answers[(paragraph_position, None)][position]
            )
            for position in positions
        }
        for paragraph_position in paragraph_positions
    }
    if (
        not all(attributable[(p, None)] for p in paragraph_positions)
        or any(_INVALID in by_aspect.values() for by_aspect in probabilities.values())
        or any(
            all(probabilities[p][position] is None for p in paragraph_positions)
            for position in positions
        )
    ):
        return None
    refused = {
        position
        for position in positions
        if any(probabilities[p][position] is None for p in paragraph_positions)
    }
    items: dict[int, list[tuple[int, int | None, float]]] = {
        position: [] for position in positions
    }
    for paragraph_position in paragraph_positions:
        belonging = [
            position
            for position in positions
            if probabilities[paragraph_position][position] is not None
            and probabilities[paragraph_position][position] >= thresholds[position]
        ]
        if reference_depth == "paragraphs":
            for position in belonging:
                items[position].append(
                    (
                        paragraph_position,
                        None,
                        probabilities[paragraph_position][position],
                    )
                )
            continue
        if not belonging:
            continue
        if _is_paragraph_one_whole_sentence(document.paragraphs[paragraph_position]):
            for position in belonging:
                items[position].append(
                    (paragraph_position, 0, probabilities[paragraph_position][position])
                )
            continue
        sentence_count = len(document.paragraphs[paragraph_position].sentences)
        for sentence_position in range(sentence_count):
            key = (paragraph_position, sentence_position)
            sentence_probabilities = {
                position: _get_answer_probability(answers[key][position])
                for position in belonging
            }
            if not attributable[key] or _INVALID in sentence_probabilities.values():
                return None
            for position, p in sentence_probabilities.items():
                if p is None:
                    refused.add(position)
                elif p >= thresholds[position]:
                    items[position].append((paragraph_position, sentence_position, p))
    return items, refused


# Models in a fallback chain


# What a model API does with a request: answers it with data that can be used
# ("valid") or not ("invalid"), with a response cut off at the max output tokens
# (LLMs only), or fails with a transient error, which is retried, or with an error
# that is not
_LLM_OUTCOMES = ["valid", "invalid", "truncated", "transient", "permanent"]
_DECISION_OUTCOMES = ["valid", "invalid", "transient", "permanent"]
# Longest script that a model of a fallback chain can need: an LLM is called up to
# three times (once, and again twice for invalid data), and a decision model sends
# a request for each of up to three paragraphs, each sent up to twice
_SCRIPT_LENGTH = 6


@dataclass(frozen=True)
class _ScriptedModel:
    """
    Model of a fallback chain, with what its API does with each request sent
    to it, in order.

    :ivar kind: Kind of the model: "llm" or "decision".
    :vartype kind: str
    :ivar model: Model identifier.
    :vartype model: str
    :ivar retries_invalid_data: Number of times that an LLM is called again for
        invalid data. Zero for a decision model.
    :vartype retries_invalid_data: int
    :ivar retries_failed_request: Number of times that a request that fails with
        a transient error is sent again.
    :vartype retries_failed_request: int
    :ivar outcomes: What the API does with each request, in order.
    :vartype outcomes: tuple[str, ...]
    """

    kind: str
    model: str
    retries_invalid_data: int
    retries_failed_request: int
    outcomes: tuple[str, ...]

    def create(self, is_fallback: bool) -> DocumentLLM | DocumentDecisionModel:
        """
        Creates the model, with prices of its tokens.

        :param is_fallback: Whether the model is a fallback model.
        :type is_fallback: bool
        :return: The model.
        :rtype: DocumentLLM | DocumentDecisionModel
        """
        pricing = ModelPricing(
            input_per_1m_tokens=float(_INPUT_PRICE),
            output_per_1m_tokens=float(_OUTPUT_PRICE),
        )
        if self.kind == "llm":
            return DocumentLLM(
                model=self.model,
                api_key="test_api_key",
                max_retries_invalid_data=self.retries_invalid_data,
                num_retries_failed_request=self.retries_failed_request,
                pricing_details=pricing,
                is_fallback=is_fallback,
            )
        return DocumentDecisionModel(
            model=self.model,
            api_key="test_api_key",
            num_retries_failed_request=self.retries_failed_request,
            pricing_details=pricing,
            is_fallback=is_fallback,
        )


@st.composite
def _fallback_chains(draw: st.DrawFn) -> list[_ScriptedModel]:
    """
    Generates a primary model and up to two fallback models, each an LLM or
    a decision model, with what their APIs do with the requests sent to them.

    :param draw: Function to draw values from strategies.
    :type draw: st.DrawFn
    :return: The models, the primary one first.
    :rtype: list[_ScriptedModel]
    """
    kinds = draw(
        st.lists(st.sampled_from(["llm", "decision"]), min_size=1, max_size=3).filter(
            lambda kinds: kinds.count("decision") <= len(_DECISION_MODELS)
        )
    )
    llm_models = iter(_LLM_MODELS)
    decision_models = iter(_DECISION_MODELS)
    chain = []
    for kind in kinds:
        outcomes = _LLM_OUTCOMES if kind == "llm" else _DECISION_OUTCOMES
        chain.append(
            _ScriptedModel(
                kind=kind,
                model=next(llm_models if kind == "llm" else decision_models),
                retries_invalid_data=draw(st.integers(0, 2)) if kind == "llm" else 0,
                retries_failed_request=draw(st.integers(0, 1)),
                outcomes=tuple(
                    draw(
                        st.lists(
                            st.sampled_from(outcomes),
                            min_size=_SCRIPT_LENGTH,
                            max_size=_SCRIPT_LENGTH,
                        )
                    )
                ),
            )
        )
    return chain


@dataclass
class _SimulatedUsage:
    """
    Usage of a model that the extraction entails.

    :ivar requests: Number of the requests sent, including those sent again.
    :vartype requests: int
    :ivar calls: Number of the calls recorded: one for each request, together
        with the times that it is sent again for transient errors.
    :vartype calls: int
    :ivar responses: Number of the responses received, each with its tokens.
    :vartype responses: int
    """

    requests: int = 0
    calls: int = 0
    responses: int = 0


@dataclass
class _SimulatedExtraction:
    """
    Outcome of an extraction with a fallback chain that the scripts of the models
    entail.

    :ivar outcome: Outcome of the last call made: "valid", "invalid",
        "truncated", or "api_failure" for a request that failed for good.
    :vartype outcome: str
    :ivar last_model: The last model called.
    :vartype last_model: _ScriptedModel | None
    :ivar retries_attempted: Number of the times that the failed request of
        the last call was sent again, for transient errors.
    :vartype retries_attempted: int
    :ivar usage: Usage of each model, by its identifier.
    :vartype usage: dict[str, _SimulatedUsage]
    """

    outcome: str = ""
    last_model: _ScriptedModel | None = None
    retries_attempted: int = 0
    usage: dict[str, _SimulatedUsage] = field(default_factory=dict)


def _simulate_extraction(
    chain: list[_ScriptedModel], paragraph_count: int
) -> _SimulatedExtraction:
    """
    Simulates the extraction of aspects, in a single call, with a primary model
    and its fallback models, as documented:

    - an LLM is sent one request for the call, a decision model one request for
      each paragraph, which it stops sending once a request fails or a response
      cannot be used;
    - a request that fails with a transient error is sent again, up to
      ``num_retries_failed_request`` times;
    - the primary model is called again for invalid data, up to
      ``max_retries_invalid_data`` times, and a fallback model is called once,
      and again for invalid data in the same way;
    - a model is not called again once its request failed for good, or its
      response was cut off at the max output tokens: the next fallback model is
      tried.

    :param chain: The models, the primary one first.
    :type chain: list[_ScriptedModel]
    :param paragraph_count: Number of the paragraphs of the document.
    :type paragraph_count: int
    :return: The outcome of the extraction.
    :rtype: _SimulatedExtraction
    """
    simulated = _SimulatedExtraction(
        usage={model.model: _SimulatedUsage() for model in chain}
    )
    scripts = {model.model: iter(model.outcomes) for model in chain}

    def send(model: _ScriptedModel) -> tuple[str, int]:
        """
        Sends a request, again for transient errors, and returns what the API
        does with it, and how many times it was sent again.
        """
        usage = simulated.usage[model.model]
        usage.calls += 1
        for attempt in range(model.retries_failed_request + 1):
            outcome = next(scripts[model.model])
            usage.requests += 1
            if outcome == "transient" and attempt < model.retries_failed_request:
                continue
            if outcome in ("transient", "permanent"):
                return "api_failure", attempt
            usage.responses += 1
            return outcome, attempt
        raise AssertionError("Unreachable")

    def call(model: _ScriptedModel) -> tuple[str, int]:
        """
        Makes a call with the model, and returns its outcome, and how many times
        the last request of the call was sent again.
        """
        if model.kind == "llm":
            return send(model)
        for _ in range(paragraph_count):
            outcome, retries = send(model)
            if outcome != "valid":
                return outcome, retries
        return "valid", 0

    primary, *fallback_models = chain
    simulated.last_model = primary
    outcome, retries = call(primary)
    if outcome == "invalid":
        for _ in range(primary.retries_invalid_data):
            outcome, retries = call(primary)
            if outcome != "invalid":
                break
    if outcome != "valid":
        for fallback_model in fallback_models:
            simulated.last_model = fallback_model
            for _ in range(fallback_model.retries_invalid_data + 1):
                outcome, retries = call(fallback_model)
                if outcome != "invalid":
                    break
            if outcome == "valid":
                break
    simulated.outcome = outcome
    simulated.retries_attempted = retries
    return simulated


def _get_prompt(request: dict[str, Any]) -> str:
    """
    Returns the text of the prompt of a request to an LLM, i.e. of its last
    message.

    :param request: The request, as the keyword arguments of the call.
    :type request: dict[str, Any]
    :return: The text of the prompt.
    :rtype: str
    """
    content = request["messages"][-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content)


def _chunk(positions: list[int], size: int) -> list[list[int]]:
    """
    Splits positions into consecutive parts of the given size, or returns them
    as one part if the size is zero.

    :param positions: The positions.
    :type positions: list[int]
    :param size: Size of the parts, or zero.
    :type size: int
    :return: The parts.
    :rtype: list[list[int]]
    """
    if not size:
        return [positions]
    return [positions[start : start + size] for start in range(0, len(positions), size)]


def _create_group_concept(name: str, role: str) -> StringConcept | BooleanConcept:
    """
    Creates a concept for the model of the given role in a group: a string
    concept with references for the LLM ("extractor"), or a Boolean concept for
    the decision model ("decider").

    :param name: Name of the concept.
    :type name: str
    :param role: Role of the model that extracts the concept.
    :type role: str
    :return: The concept.
    :rtype: StringConcept | BooleanConcept
    """
    if role == "extractor":
        return StringConcept(
            name=name,
            description=f"Description of {name}",
            model_role=role,
            add_references=True,
        )
    return BooleanConcept(
        name=name, description=f"Description of {name}", model_role=role
    )


def _get_failed_instance_names(messages: list[str]) -> set[str]:
    """
    Returns the names of the aspects or concepts that the messages of extraction
    errors or warnings name as failed.

    :param messages: The messages.
    :type messages: list[str]
    :return: The names.
    :rtype: set[str]
    """
    names: set[str] = set()
    for message in messages:
        match = re.search(r"Failed (?:aspect|concept)s: (\[.*?\])", message)
        assert match is not None, message
        names.update(ast.literal_eval(match.group(1)))
    return names


class TestExtractionProperties(BaseTest):
    """
    Property-based tests of the extraction with LLMs and with decision models, whose
    APIs are stubbed.
    """

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_llm_aspects_from_any_output(self, data: st.DataObject):
        """
        Tests that the aspects extracted by an LLM have the items that its output
        entails, whatever the output is: each paragraph or sentence listed for
        an aspect, once, with its justification, and as the reference of
        the item. An output that has an ID that is not one of the prompt, or
        a sentence ID of another paragraph than the one it is listed under, or
        an invalid justification, or that does not have the structure asked
        for, is rejected as a whole: the extraction fails with
        ``ModelExtractionError``, or issues a warning, and no aspect gets items.
        So is an output that leaves an aspect out: it must list each aspect,
        also one without items.

        The API of the LLM is stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        aspect_count = data.draw(st.integers(1, 3))
        reference_depth = data.draw(st.sampled_from(_REFERENCE_DEPTHS))
        add_justifications = data.draw(st.booleans())
        document.aspects = [
            Aspect(
                name=f"Aspect {position}",
                description=f"Description of aspect {position}",
                reference_depth=reference_depth,
                add_justifications=add_justifications,
            )
            for position in range(aspect_count)
        ]
        # A faulty output is mostly rejected, while the others test the items in
        # more detail
        faulty = data.draw(st.booleans())
        output = data.draw(
            _llm_aspect_outputs(
                document, aspect_count, reference_depth, add_justifications, faulty
            )
        )
        text, has_output = data.draw(_llm_response_texts(output, faulty))
        expected = (
            _get_expected_aspect_items(
                output, document, aspect_count, reference_depth, add_justifications
            )
            if has_output
            else None
        )
        note(f"Response: {text!r}")
        note(f"Expected: {expected!r}")
        event("rejected" if expected is None else "accepted")
        raise_exception = data.draw(st.booleans())

        llm = DocumentLLM(
            model=_LLM_MODEL, api_key="test_api_key", max_retries_invalid_data=0
        )
        stub = _ModelAPIsStub(complete=lambda request: _completion(text))
        error: Exception | None = None
        with stub.installed(), _recorded_warnings() as caught:
            try:
                llm.extract_aspects_from_document(
                    document, raise_exception_on_extraction_error=raise_exception
                )
            except ModelExtractionError as e:
                error = e
        assert len(stub.completion_requests) == 1
        _check_prompt_texts(
            _get_prompt(stub.completion_requests[0]),
            document,
            reference_depth,
            lists_paragraphs=True,
        )
        (usage,) = llm.get_usage()
        assert len(usage.usage.calls) == 1
        assert usage.usage.input == _LLM_INPUT_TOKENS
        assert usage.usage.output == _LLM_OUTPUT_TOKENS

        if expected is None:
            if raise_exception:
                assert error is not None
            else:
                assert error is None
                assert any(
                    "could not be processed" in message
                    for message in _get_messages(caught)
                )
            for aspect in document.aspects:
                assert not aspect._is_processed
                assert not aspect.extracted_items
                assert not aspect.reference_paragraphs
            return

        assert error is None
        assert not any(
            "could not be processed" in message for message in _get_messages(caught)
        )
        for position, aspect in enumerate(document.aspects):
            assert aspect._is_processed is (position in expected)
            _check_items(aspect, expected.get(position, []), document)
            for item in aspect.extracted_items:
                reference = (
                    item.reference_sentences[0]
                    if reference_depth == "sentences"
                    else item.reference_paragraphs[0]
                )
                _check_custom_data_copy(item, reference)
            # The references of the aspect are those of its items, in their order
            expected_paragraphs = list(
                dict.fromkeys(
                    item.reference_paragraphs[0].unique_id
                    for item in aspect.extracted_items
                )
            )
            assert [
                paragraph.unique_id for paragraph in aspect.reference_paragraphs
            ] == expected_paragraphs
            if reference_depth == "sentences":
                assert [
                    sentence.unique_id for sentence in aspect.reference_sentences
                ] == [
                    item.reference_sentences[0].unique_id
                    for item in aspect.extracted_items
                ]
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.parametrize("concept_type", _CONCEPT_TYPES)
    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_llm_concepts_from_any_output(self, concept_type: str, data: st.DataObject):
        """
        Tests that the concepts extracted by an LLM have the items that its output
        entails, whatever the output is: each value that the concept accepts,
        as the item value of the concept type, with its justification and its
        references. An output that has an ID that is not one of the prompt,
        a value that the concept does not accept, e.g. a value of another type,
        a number that is not finite, an invalid justification or reference, or
        that does not have the structure asked for, is rejected as a whole:
        the extraction fails with ``ModelExtractionError``, or issues a warning,
        and no concept gets items. So is an output that leaves a concept out (it
        must list each concept, also one without items), or that has more than
        one item for a concept with a singular occurrence.

        The first concept is of the given type, and the others, if any, of any
        type.

        The API of the LLM is stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        add_justifications = data.draw(st.booleans())
        add_references = data.draw(st.booleans())
        reference_depth = data.draw(st.sampled_from(_REFERENCE_DEPTHS))
        extraction_params = {
            "add_justifications": add_justifications,
            "add_references": add_references,
            "reference_depth": reference_depth,
        }
        specs = [
            data.draw(
                _concept_specs(
                    position,
                    extraction_params,
                    concept_type if position == 0 else None,
                )
            )
            for position in range(data.draw(st.integers(1, 3)))
        ]
        document.concepts = [spec.concept for spec in specs]
        # A faulty output is mostly rejected, while the others test the items in
        # more detail
        faulty = data.draw(st.booleans())
        output = data.draw(
            _llm_concept_outputs(
                specs,
                document,
                add_justifications,
                add_references,
                reference_depth,
                faulty,
            )
        )
        text, has_output = data.draw(_llm_response_texts(output, faulty))
        expected = (
            _get_expected_concept_items(
                output,
                specs,
                document,
                add_justifications,
                add_references,
                reference_depth,
            )
            if has_output
            else None
        )
        note(f"Response: {text!r}")
        note(f"Expected: {expected!r}")
        event("rejected" if expected is None else "accepted")
        raise_exception = data.draw(st.booleans())

        llm = DocumentLLM(
            model=_LLM_MODEL, api_key="test_api_key", max_retries_invalid_data=0
        )
        stub = _ModelAPIsStub(complete=lambda request: _completion(text))
        error: Exception | None = None
        with stub.installed(), _recorded_warnings() as caught:
            try:
                llm.extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=raise_exception
                )
            except ModelExtractionError as e:
                error = e
        assert len(stub.completion_requests) == 1
        _check_prompt_texts(
            _get_prompt(stub.completion_requests[0]),
            document,
            reference_depth,
            lists_paragraphs=add_references,
        )
        (usage,) = llm.get_usage()
        assert len(usage.usage.calls) == 1

        if expected is None:
            if raise_exception:
                assert error is not None
            else:
                assert error is None
                assert any(
                    "could not be processed" in message
                    for message in _get_messages(caught)
                )
            for concept in document.concepts:
                assert not concept._is_processed
                assert not concept.extracted_items
            return

        assert error is None
        for position, concept in enumerate(document.concepts):
            assert concept._is_processed is (position in expected)
            _check_items(concept, expected.get(position, []), document)
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.parametrize(
        ("instance_type", "fault", "concept_type"),
        [("aspect", fault, None) for fault in _ASPECT_FAULTS]
        + [
            ("concept", fault, None)
            for fault in _CONCEPT_FAULTS
            if fault not in ("value", "invalid_json_value")
        ]
        # A faulty value is rejected by each type of concept in its own way
        + [("concept", "value", concept_type) for concept_type in _CONCEPT_TYPES]
        + [
            ("concept", "invalid_json_value", concept_type)
            for concept_type in _CONCEPT_TYPES
        ],
    )
    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_llm_output_with_one_fault_is_requested_again(
        self,
        instance_type: str,
        fault: str,
        concept_type: str | None,
        data: st.DataObject,
    ):
        """
        Tests that an output of an LLM that is valid but for a single fault of
        the given kind is rejected as a whole, and requested again, and that
        the valid output that the LLM then returns gives exactly its items. A fault
        is an ID that is not one of the prompt, e.g. ``P1_0`` or ``P03`` for
        a paragraph, a sentence ID of another paragraph than the one it is listed
        for, a justification that is not valid, a value that the concept does
        not accept, e.g. a value of another type, or a value that is not valid
        JSON: NaN, an infinity, a number too large for a float, an integer with
        more digits than Python converts, or arrays nested deeper than
        the parser can recurse. It is also an aspect or a concept left out of
        the output, and a second item of a concept with a singular occurrence.

        A faulty value is injected in an output for a single concept of the given
        type.

        The API of the LLM is stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        count = 1 if concept_type is not None else data.draw(st.integers(1, 3))
        sentence_fault = "sentence_id" in fault
        reference_depth: ReferenceDepth = (
            "sentences"
            if sentence_fault
            else data.draw(st.sampled_from(_REFERENCE_DEPTHS))
        )
        add_justifications = fault == "justification" or data.draw(st.booleans())
        add_references = False
        specs: list[_ConceptSpec] | None = None
        if instance_type == "aspect":
            document.aspects = [
                Aspect(
                    name=f"Aspect {position}",
                    description=f"Description of aspect {position}",
                    reference_depth=reference_depth,
                    add_justifications=add_justifications,
                )
                for position in range(count)
            ]
            output = data.draw(
                _llm_aspect_outputs(
                    document,
                    count,
                    reference_depth,
                    add_justifications,
                    faulty=False,
                    nonempty=True,
                )
            )

            def get_expected(output: Any) -> dict[int, list[_ExpectedItem]] | None:
                """
                Returns what an output for the aspects entails.
                """
                return _get_expected_aspect_items(
                    output, document, count, reference_depth, add_justifications
                )

        else:
            add_references = (
                fault == "reference_paragraph_id"
                or sentence_fault
                or data.draw(st.booleans())
            )
            extraction_params = {
                "add_justifications": add_justifications,
                "add_references": add_references,
                "reference_depth": reference_depth,
            }
            concept_specs = [
                data.draw(
                    _concept_specs(
                        position,
                        extraction_params,
                        concept_type,
                        singular_occurrence=(
                            True
                            if fault == "second_item_of_singular_concept"
                            and position == 0
                            else None
                        ),
                    )
                )
                for position in range(count)
            ]
            specs = concept_specs
            document.concepts = [spec.concept for spec in concept_specs]
            output = data.draw(
                _llm_concept_outputs(
                    concept_specs,
                    document,
                    add_justifications,
                    add_references,
                    reference_depth,
                    faulty=False,
                    nonempty=True,
                )
            )

            def get_expected(output: Any) -> dict[int, list[_ExpectedItem]] | None:
                """
                Returns what an output for the concepts entails.
                """
                return _get_expected_concept_items(
                    output,
                    concept_specs,
                    document,
                    add_justifications,
                    add_references,
                    reference_depth,
                )

        expected = get_expected(output)
        # A value of a JSON object concept may have an empty nested object
        assume(expected is not None)
        assert expected is not None

        # The output with the fault, at one of the places that it can be at
        faulty_output = copy.deepcopy(output)
        id_key, prefix = (
            ("aspect_id", "A")
            if instance_type == "aspect"
            else (
                "concept_id",
                "C",
            )
        )
        if fault == "missing_entry":
            # Every entry of an aspect or a concept is left out
            left_out = data.draw(st.integers(0, count - 1))
            faulty_output = [
                entry
                for entry in faulty_output
                if _get_position_of_id(entry[id_key], prefix, count) != left_out
            ]
        elif fault == "second_item_of_singular_concept":
            # The first listing of the first concept has its single item
            entry = next(
                entry
                for entry in faulty_output
                if _get_position_of_id(entry[id_key], prefix, count) == 0
                and entry["extracted_items"]
            )
            entry["extracted_items"].append(copy.deepcopy(entry["extracted_items"][0]))
        else:
            slots = _get_fault_slots(faulty_output, fault, document, count, specs)
            assert slots, faulty_output
            slot = data.draw(st.sampled_from(slots))
            slot.container[slot.key] = data.draw(slot.strategy)
        if fault != "invalid_json_value":
            # The output is not valid JSON otherwise, which no oracle reads
            assert get_expected(faulty_output) is None, faulty_output
        faulty_text = _write_output(faulty_output)
        note(f"Faulty output: {faulty_text[:2000]!r}")
        note(f"Expected after the retry: {expected!r}")

        responses = [_completion(faulty_text), _completion(_write_output(output))]
        llm = DocumentLLM(
            model=_LLM_MODEL, api_key="test_api_key", max_retries_invalid_data=1
        )
        stub = _ModelAPIsStub(complete=lambda request: responses.pop(0))
        with stub.installed(), _recorded_warnings() as caught:
            if instance_type == "aspect":
                llm.extract_aspects_from_document(document)
            else:
                llm.extract_concepts_from_document(document)
        assert not responses
        for request in stub.completion_requests:
            _check_prompt_texts(
                _get_prompt(request),
                document,
                reference_depth,
                lists_paragraphs=instance_type == "aspect" or add_references,
            )
        assert not _get_extraction_warnings(caught)
        (usage,) = llm.get_usage()
        assert len(usage.usage.calls) == 2
        instances = document.aspects if instance_type == "aspect" else document.concepts
        for position, instance in enumerate(instances):
            assert instance._is_processed is (position in expected)
            _check_items(instance, expected.get(position, []), document)

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_decision_model_concepts_from_any_answers(self, data: st.DataObject):
        """
        Tests that the concepts extracted by a decision model have the items that
        its answers entail, whatever the answers are:

        - Boolean concept: True if the probability of "yes" is the decision
          threshold or more;
        - label concept with the "multi_class" classification type: the most
          probable label, the first of them if several are;
        - label concept with the "multi_label" classification type: the labels
          with a probability of the decision threshold or more, and no item if
          there are none;
        - rating concept: the most probable integer of the scale, the lowest of
          them if several are;

        each item with the probabilities of its options. The decision threshold
        is the one of the concept, if it has one, or else the one of the model.
        A response with an answer that cannot be used (refused, of another kind
        than its question, without the probabilities of the options offered,
        with a probability that is not a number within [0, 1], or not attributable
        to its question) is rejected as a whole: the extraction fails with
        ``ModelExtractionError``, or issues a warning, and no concept gets items.
        A refused answer that the rejection is for raises ``ModelRefusalError``,
        its subclass.

        The Decisions API is stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        model_threshold = data.draw(st.floats(0.01, 1))
        thresholds = st.none() | st.floats(0.01, 1)
        concepts: list[Any] = []
        for position in range(data.draw(st.integers(1, 4))):
            params: dict[str, Any] = {
                "name": f"Concept {position}",
                "description": f"Description of concept {position}",
            }
            concept_type = data.draw(
                st.sampled_from(["boolean", "multi_class", "multi_label", "rating"])
            )
            if concept_type == "boolean":
                concept = BooleanConcept(
                    decision_threshold=data.draw(thresholds), **params
                )
            elif concept_type == "rating":
                start = data.draw(st.integers(0, 3))
                concept = RatingConcept(
                    rating_scale=(start, data.draw(st.integers(start + 1, start + 4))),
                    **params,
                )
            else:
                labels = data.draw(
                    st.lists(
                        st.sampled_from(_LABELS), min_size=2, max_size=3, unique=True
                    )
                )
                concept = LabelConcept(
                    labels=labels,
                    classification_type=cast(ClassificationType, concept_type),
                    decision_threshold=(
                        data.draw(thresholds) if concept_type == "multi_label" else None
                    ),
                    **params,
                )
            concepts.append(concept)
        document.concepts = concepts

        # The questions of the concepts, by their documented names, with their
        # kinds and the labels of their options, and the answers to them
        questions: list[
            tuple[str, Literal["predicate", "choice", "score"], list[str] | None]
        ] = []
        for position, concept in enumerate(concepts):
            if isinstance(concept, BooleanConcept):
                questions.append((f"C{position}", "predicate", None))
            elif isinstance(concept, RatingConcept):
                start, end = concept.rating_scale
                levels = [str(level) for level in range(start, end + 1)]
                questions.append((f"C{position}", "score", levels))
            elif concept.classification_type == "multi_class":
                questions.append((f"C{position}", "choice", list(concept.labels)))
            else:
                questions.extend(
                    (f"C{position}-L{label_position}", "predicate", None)
                    for label_position in range(len(concept.labels))
                )
        # The answers of a faulty response may not be usable, while those of
        # another one are, except that the probabilities of the options of
        # a question may all be zero
        faulty = data.draw(st.booleans())
        answers = {
            name: data.draw(_answers(kind, labels, faulty=faulty, refusals=faulty))
            for name, kind, labels in questions
        }
        response_answers, attributable = data.draw(
            _responses_of_answers(
                [name for name, _, _ in questions], answers, faulty=faulty
            )
        )
        note(f"Answers: {answers!r}")

        # What the answers entail for each concept
        expected: list[Any] | None = []
        if not attributable or any(
            _has_repeated_option(answer) for answer in answers.values()
        ):
            expected = None
        else:
            for position, concept in enumerate(concepts):
                threshold = getattr(concept, "decision_threshold", None)
                threshold = model_threshold if threshold is None else threshold
                if isinstance(concept, BooleanConcept):
                    p = _get_predicate_probability(answers[f"C{position}"])
                    if p is _INVALID:
                        expected = None
                        break
                    expected.append((p >= threshold, {True: p, False: 1 - p}))
                elif isinstance(concept, RatingConcept):
                    start, end = concept.rating_scale
                    probabilities = _get_option_probabilities(
                        answers[f"C{position}"],
                        "score",
                        [str(level) for level in range(start, end + 1)],
                    )
                    if probabilities is _INVALID:
                        expected = None
                        break
                    by_level = {int(label): p for label, p in probabilities.items()}
                    expected.append((max(by_level, key=by_level.__getitem__), by_level))
                elif concept.classification_type == "multi_class":
                    probabilities = _get_option_probabilities(
                        answers[f"C{position}"], "choice", list(concept.labels)
                    )
                    if probabilities is _INVALID:
                        expected = None
                        break
                    expected.append(
                        (
                            [max(probabilities, key=probabilities.__getitem__)],
                            probabilities,
                        )
                    )
                else:
                    probabilities = {
                        label: _get_predicate_probability(
                            answers[f"C{position}-L{label_position}"]
                        )
                        for label_position, label in enumerate(concept.labels)
                    }
                    if _INVALID in probabilities.values():
                        expected = None
                        break
                    selected = [
                        label
                        for label in concept.labels
                        if probabilities[label] >= threshold
                    ]
                    expected.append((selected, probabilities) if selected else None)
        note(f"Expected: {expected!r}")
        event("rejected" if expected is None else "accepted")
        raise_exception = data.draw(st.booleans())

        def decide(request: dict[str, Any]) -> Any:
            """
            Answers the request for the concepts with the answers drawn.
            """
            assert [question["name"] for question in request["questions"]] == [
                name for name, _, _ in questions
            ]
            return _decisions_response(response_answers)

        model = DocumentDecisionModel(
            model=_DECISION_MODEL,
            api_key="test_api_key",
            decision_threshold=model_threshold,
        )
        stub = _ModelAPIsStub(decide=decide)
        error: Exception | None = None
        with stub.installed(), _recorded_warnings() as caught:
            try:
                model.extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=raise_exception
                )
            except ModelExtractionError as e:
                error = e
        assert len(stub.decision_requests) == 1
        (usage,) = model.get_usage()
        assert len(usage.usage.calls) == 1
        assert usage.usage.input == _DECISION_INPUT_TOKENS

        if expected is None:
            if raise_exception:
                assert error is not None
                if isinstance(error, ModelRefusalError):
                    # Only a refused answer makes the call fail as a refusal
                    assert any(answer.kind == "refusal" for answer in answers.values())
            else:
                assert error is None
                # The warning on the failed call, for invalid data or a refusal
                assert any(
                    "Failed concepts:" in message
                    and "If you want to raise an exception instead" in message
                    for message in _get_messages(caught)
                )
            for concept in document.concepts:
                assert not concept._is_processed
                assert not concept.extracted_items
            return

        assert error is None
        for concept, expected_item in zip(document.concepts, expected, strict=True):
            assert concept._is_processed
            if expected_item is None:
                assert not concept.extracted_items
                continue
            (item,) = concept.extracted_items
            value, probabilities = expected_item
            assert item.value == value
            assert type(item.value) is type(value)
            assert item.probabilities == probabilities
            assert [type(option) for option in item.probabilities] == [
                type(option) for option in probabilities
            ]
        self.check_instance_serialization_and_cloning(document)

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_decision_model_aspects_from_any_answers(self, data: st.DataObject):
        """
        Tests that the aspects extracted by a decision model have the items that
        its answers entail, whatever the answers are: each paragraph with
        a probability of "yes" of the decision threshold or more, and for
        an aspect with the "sentences" reference depth, each sentence of such
        a paragraph with such a probability, in the order of the document, with
        the probabilities of the item. The one sentence of a paragraph that is
        the paragraph as a whole is not asked about: it takes the answer on
        the paragraph. The decision threshold is the one of
        the aspect, if it has one, or else the one of the model. A paragraph or
        a sentence that the model refused to answer on is left out, with
        a warning that names the aspect, unless the model refused to answer on
        every paragraph for the aspect, which fails the call as a refusal
        (``ModelRefusalError``, a subclass of ``ModelExtractionError``).

        The aspects with each reference depth are extracted in calls of their
        own, of up to ``max_items_per_call`` aspects. A call with an answer that
        cannot be used otherwise fails as a whole: the extraction fails with
        ``ModelExtractionError``, or issues a warning that names the aspects of
        the call, which get no items, while the aspects of the other calls are
        extracted unless the extraction failed before. Apart from that,
        the outcome does not depend on whether the calls and the requests are
        made concurrently.

        The Decisions API is stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        model_threshold = data.draw(st.floats(0.01, 1))
        aspects = [
            Aspect(
                name=f"Aspect {position}",
                description=f"Description of aspect {position}",
                reference_depth=data.draw(st.sampled_from(_REFERENCE_DEPTHS)),
                decision_threshold=data.draw(st.none() | st.floats(0.01, 1)),
            )
            for position in range(data.draw(st.integers(1, 3)))
        ]
        document.aspects = aspects
        thresholds = [
            model_threshold
            if aspect.decision_threshold is None
            else aspect.decision_threshold
            for aspect in aspects
        ]
        # The answers on each paragraph, and on each sentence, by the positions of
        # the aspects, and whether the response on each of these texts can be
        # attributed to its questions. Faulty answers and responses may not be
        # usable, while the others are, refusals included.
        faulty = data.draw(st.booleans())
        answer_strategy = _answers("predicate", faulty=faulty)
        answers: _AnswersOnTexts = {
            (paragraph_position, None): {
                position: data.draw(answer_strategy) for position in range(len(aspects))
            }
            for paragraph_position in range(len(document.paragraphs))
        }
        answers.update(
            {
                (paragraph_position, sentence_position): {
                    position: data.draw(answer_strategy)
                    for position in range(len(aspects))
                }
                for paragraph_position, paragraph in enumerate(document.paragraphs)
                for sentence_position in range(len(paragraph.sentences))
            }
        )
        attributable = {
            key: data.draw(st.sampled_from([True] * 9 + [False])) if faulty else True
            for key in answers
        }
        note(f"Answers: {answers!r}")
        note(f"Attributable: {attributable!r}")

        raise_exception = data.draw(st.booleans())
        use_concurrency = data.draw(st.booleans())
        max_items_per_call = data.draw(st.integers(0, 2))
        # With concurrency, each aspect is extracted in a call of its own by
        # default, and otherwise all the aspects of a reference depth together
        items_per_call = (
            max(1, max_items_per_call) if use_concurrency else max_items_per_call
        )

        # What the answers entail for the aspects of each call
        expected: dict[int, list[tuple[int, int | None, float]]] = {}
        refused: set[int] = set()
        failed: set[int] = set()
        for reference_depth in ("paragraphs", "sentences"):
            positions = [
                position
                for position, aspect in enumerate(aspects)
                if aspect.reference_depth == reference_depth
            ]
            if not positions:
                continue
            call_size = items_per_call or len(positions)
            for call_start in range(0, len(positions), call_size):
                call_positions = positions[call_start : call_start + call_size]
                outcome = _get_expected_decision_aspect_items(
                    call_positions,
                    reference_depth,
                    document,
                    thresholds,
                    answers,
                    attributable,
                )
                if outcome is None:
                    failed.update(call_positions)
                else:
                    expected.update(outcome[0])
                    refused.update(outcome[1])
        note(f"Expected: {expected!r}, refused: {refused!r}, failed: {failed!r}")
        event(f"failed calls: {bool(failed)}, refusals: {bool(refused)}")

        def decide(request: dict[str, Any]) -> Any:
            """
            Answers the request on a paragraph or on a sentence with the answers
            drawn for the text of the request, each to the question of the aspect
            that the instructions of the question name.
            """
            text = request["input"]
            paragraph_position, sentence_position = _get_marker(text)
            key = (
                paragraph_position,
                sentence_position if text.startswith("Sentence: ") else None,
            )
            response_answers = []
            for question in request["questions"]:
                match = re.search(
                    r"Aspect name: Aspect (\d+)", question["instructions"]
                )
                assert match is not None
                response_answers.append(
                    answers[key][int(match.group(1))].to_answer(question["name"])
                )
            if not attributable[key]:
                response_answers = response_answers[:-1]
            return _decisions_response(response_answers)

        model = DocumentDecisionModel(
            model=_DECISION_MODEL,
            api_key="test_api_key",
            decision_threshold=model_threshold,
        )
        model.async_limiter = _create_unpaced_limiter()
        stub = _ModelAPIsStub(decide=decide)
        error: Exception | None = None
        with stub.installed(), _recorded_warnings() as caught:
            try:
                model.extract_aspects_from_document(
                    document,
                    use_concurrency=use_concurrency,
                    max_items_per_call=max_items_per_call,
                    raise_exception_on_extraction_error=raise_exception,
                )
            except ModelExtractionError as e:
                error = e
        (usage,) = model.get_usage()
        assert len(usage.usage.calls) == len(stub.decision_requests)
        assert usage.usage.input == _DECISION_INPUT_TOKENS * len(stub.decision_requests)
        messages = _get_messages(caught)
        # The warnings on the paragraphs and the sentences left out of an aspect
        # for refused answers. A refusal on every paragraph fails the call instead,
        # with the messages on a failed call.
        refusal_warnings = {
            match.group(1)
            for message in messages
            if "These are not included in the aspect." in message
            and (
                match := re.search(
                    r"refused to answer the question of aspect `(.+?)`", message
                )
            )
        }
        # The warnings on the failed calls, for invalid data or a refusal
        failure_warnings = [
            message
            for message in messages
            if "Failed aspects:" in message
            and "If you want to raise an exception instead" in message
        ]

        failed_names = {aspects[position].name for position in failed}
        refused_names = {aspects[position].name for position in refused}
        if not failed:
            assert error is None
            assert not failure_warnings
        elif raise_exception:
            assert error is not None
            assert _get_failed_instance_names([str(error)]) <= failed_names
        else:
            assert error is None
            assert _get_failed_instance_names(failure_warnings) == failed_names
        if failed and raise_exception:
            # The extraction failed when one of the calls failed, i.e. the other
            # call may not have been made, or completed
            assert refusal_warnings <= refused_names
        else:
            assert refusal_warnings == refused_names

        for position, aspect in enumerate(document.aspects):
            if position in failed or (
                failed and raise_exception and not aspect._is_processed
            ):
                assert not aspect._is_processed
                assert not aspect.extracted_items
                continue
            assert aspect._is_processed
            expected_items = expected[position]
            assert len(aspect.extracted_items) == len(expected_items)
            for item, (paragraph_position, sentence_position, p) in zip(
                aspect.extracted_items, expected_items, strict=True
            ):
                paragraph = document.paragraphs[paragraph_position]
                assert item.reference_paragraphs[0] is paragraph
                if sentence_position is None:
                    assert item.value == paragraph.raw_text
                    assert not item.reference_sentences
                    _check_custom_data_copy(item, paragraph)
                else:
                    sentence = paragraph.sentences[sentence_position]
                    assert item.value == sentence.raw_text
                    assert item.reference_sentences[0] is sentence
                    _check_custom_data_copy(item, sentence)
                assert item.probabilities == {True: p, False: 1 - p}
        self.check_instance_serialization_and_cloning(document)

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_decision_threshold_monotonicity(self, data: st.DataObject):
        """
        Tests that a higher decision threshold of a decision model never adds to
        what it extracts with the same answers: the paragraphs of an aspect, and
        the labels of a label concept with the "multi_label" classification
        type, are a subset of those extracted with a lower threshold, and
        a Boolean concept that is False with a lower threshold stays False.

        The Decisions API is stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        low_threshold, high_threshold = sorted(
            data.draw(st.lists(st.floats(0.01, 1), min_size=2, max_size=2))
        )
        labels = data.draw(
            st.lists(st.sampled_from(_LABELS), min_size=2, max_size=3, unique=True)
        )
        document.aspects = [
            Aspect(name=f"Aspect {position}", description=f"Description {position}")
            for position in range(data.draw(st.integers(1, 2)))
        ]
        document.concepts = [
            BooleanConcept(name="Boolean", description="Boolean concept"),
            LabelConcept(
                name="Labels",
                description="Label concept",
                labels=labels,
                classification_type="multi_label",
            ),
        ]
        probability = st.floats(0, 1)
        aspect_probabilities = {
            (paragraph_position, f"A{position}"): data.draw(probability)
            for paragraph_position in range(len(document.paragraphs))
            for position in range(len(document.aspects))
        }
        concept_probabilities = {
            name: data.draw(probability)
            for name in ["C0", *[f"C1-L{j}" for j in range(len(labels))]]
        }

        def decide(request: dict[str, Any]) -> Any:
            """
            Answers a request with the probabilities drawn for its questions.
            """
            names = [question["name"] for question in request["questions"]]
            if names[0].startswith("A"):
                paragraph_position, _ = _get_marker(request["input"])
                return _decisions_response(
                    [
                        OpenAIPredicateAnswer(
                            name=name,
                            probability=aspect_probabilities[
                                (paragraph_position, name)
                            ],
                        )
                        for name in names
                    ]
                )
            return _decisions_response(
                [
                    OpenAIPredicateAnswer(
                        name=name, probability=concept_probabilities[name]
                    )
                    for name in names
                ]
            )

        extracted = []
        for threshold in (low_threshold, high_threshold):
            model = DocumentDecisionModel(
                model=_DECISION_MODEL,
                api_key="test_api_key",
                decision_threshold=threshold,
            )
            processed_document = document.clone()
            with _ModelAPIsStub(decide=decide).installed():
                model.extract_all(processed_document)
            extracted.append(processed_document)
        low, high = extracted
        for low_aspect, high_aspect in zip(low.aspects, high.aspects, strict=True):
            assert {item.value for item in high_aspect.extracted_items} <= {
                item.value for item in low_aspect.extracted_items
            }
        low_boolean, low_labels = low.concepts
        high_boolean, high_labels = high.concepts
        if not low_boolean.extracted_items[0].value:
            assert not high_boolean.extracted_items[0].value
        high_selected = (
            set(high_labels.extracted_items[0].value)
            if high_labels.extracted_items
            else set()
        )
        low_selected = (
            set(low_labels.extracted_items[0].value)
            if low_labels.extracted_items
            else set()
        )
        assert high_selected <= low_selected

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_fallback_chain_outcome_and_usage(self, data: st.DataObject):
        """
        Tests the outcome of an extraction with a primary model and up to two
        fallback models, each an LLM or a decision model, whatever their APIs do
        with the requests: answer with data that can be used or not, with
        a response cut off at the max output tokens, or fail with a transient
        error or with one that is not. The outcome must be the one that the retry
        and fallback rules entail (see ``_simulate_extraction``): the items of
        the model whose call succeeded, or the error of the last call, i.e.
        ``ModelAPIError`` for a failed request or a response cut off,
        ``ModelExtractionError`` for invalid data, each with the number of
        retries made, or a single warning instead. The usage of each model must
        have a call for each request, and the tokens of each response, and its
        cost the price of these tokens, also when the extraction fails.

        The APIs are stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents(max_paragraphs=3))
        paragraph_count = len(document.paragraphs)
        aspect_count = data.draw(st.integers(1, 2))
        document.aspects = [
            Aspect(name=f"Aspect {position}", description=f"Description {position}")
            for position in range(aspect_count)
        ]
        chain = data.draw(_fallback_chains())
        raise_exception = data.draw(st.booleans())
        simulated = _simulate_extraction(chain, paragraph_count)
        note(f"Simulated: {simulated!r}")
        assert simulated.last_model is not None
        event(f"outcome: {simulated.outcome}, last model: {simulated.last_model.kind}")

        scripts = {model.model: iter(model.outcomes) for model in chain}
        valid_output = json.dumps(
            [
                {
                    "aspect_id": f"A{position}",
                    "paragraph_ids": [f"P{p}" for p in range(paragraph_count)],
                }
                for position in range(aspect_count)
            ]
        )

        def complete(request: dict[str, Any]) -> Any:
            """
            Does with a request to an LLM what the script of the LLM says.
            """
            outcome = next(scripts[request["model"]])
            if outcome in ("transient", "permanent"):
                return _api_error(outcome == "transient", request["model"])
            if outcome == "truncated":
                return _completion(
                    valid_output[: len(valid_output) // 2], finish_reason="length"
                )
            if outcome == "invalid":
                # A paragraph that is not in the prompt
                return _completion(
                    json.dumps(
                        [{"aspect_id": "A0", "paragraph_ids": [f"P{paragraph_count}"]}]
                    )
                )
            return _completion(valid_output)

        def decide(request: dict[str, Any]) -> Any:
            """
            Does with a request to a decision model what the script of the model
            says: a valid response has the paragraph belong to each aspect.
            """
            outcome = next(scripts[request["model"]])
            if outcome in ("transient", "permanent"):
                return _api_error(outcome == "transient", request["model"])
            answers = [
                OpenAIPredicateAnswer(name=question["name"], probability=0.9)
                for question in request["questions"]
            ]
            # An answer missing makes the response one that cannot be used
            return _decisions_response(
                answers[:-1] if outcome == "invalid" else answers
            )

        primary_model = chain[0].create(is_fallback=False)
        primary_model.fallback_models = [
            model.create(is_fallback=True) for model in chain[1:]
        ]
        stub = _ModelAPIsStub(complete=complete, decide=decide)
        error: Exception | None = None
        with stub.installed(), _recorded_warnings() as caught:
            try:
                primary_model.extract_aspects_from_document(
                    document, raise_exception_on_extraction_error=raise_exception
                )
            except (ModelAPIError, ModelExtractionError) as e:
                error = e
        messages = _get_extraction_warnings(caught)
        note(f"Error: {error!r}, warnings: {messages!r}")

        # The requests, the usage and the cost of each model
        requests = Counter(
            request["model"]
            for request in [*stub.completion_requests, *stub.decision_requests]
        )
        assert requests == Counter(
            {model: usage.requests for model, usage in simulated.usage.items()}
        )
        usages = {entry.model: entry.usage for entry in primary_model.get_usage()}
        costs = {entry.model: entry.cost for entry in primary_model.get_cost()}
        for model in chain:
            usage = usages[model.model]
            simulated_usage = simulated.usage[model.model]
            is_llm = model.kind == "llm"
            assert len(usage.calls) == simulated_usage.calls
            assert usage.input == simulated_usage.responses * (
                _LLM_INPUT_TOKENS if is_llm else _DECISION_INPUT_TOKENS
            )
            assert usage.output == simulated_usage.responses * (
                _LLM_OUTPUT_TOKENS if is_llm else 0
            )
            assert costs[model.model].input == usage.input * _INPUT_PRICE / 10**6
            assert costs[model.model].output == usage.output * _OUTPUT_PRICE / 10**6

        assert simulated.last_model is not None
        if simulated.outcome == "valid":
            assert error is None
            assert not messages
            decided = simulated.last_model.kind == "decision"
            for aspect in document.aspects:
                assert aspect._is_processed
                assert [item.value for item in aspect.extracted_items] == [
                    paragraph.raw_text for paragraph in document.paragraphs
                ]
                assert all(
                    (item.probabilities is not None) is decided
                    for item in aspect.extracted_items
                )
            return

        for aspect in document.aspects:
            assert not aspect._is_processed
            assert not aspect.extracted_items
        if not raise_exception:
            assert error is None
            (message,) = messages
            if simulated.outcome == "invalid":
                assert "could not be processed" in message
            elif simulated.outcome == "truncated":
                assert "was cut off at the max output tokens" in message
            else:
                assert "Exception occurred while calling" in message
            return
        if simulated.outcome == "invalid":
            assert isinstance(error, ModelExtractionError)
            assert error.retry_count == simulated.last_model.retries_invalid_data
        else:
            assert isinstance(error, ModelAPIError)
            if simulated.outcome == "api_failure":
                assert error.retry_count == simulated.retries_attempted

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_model_group_extract_all(self, data: st.DataObject):
        """
        Tests that ``extract_all()`` with a group of an LLM and a decision model
        gives each aspect and each concept the items that the answers of
        the models entail, with and without concurrency, in calls of up to
        ``max_items_per_call`` aspects or concepts, and over parts of up to
        ``max_paragraphs_to_analyze_per_call`` paragraphs:

        - an aspect, extracted by either model: the paragraphs that belong to it,
          in the order of the document;
        - a concept of the document or of an aspect, extracted by either model:
          an item for each part of the paragraphs of the document or of
          the aspect, with the paragraphs of the part as its references when
          the LLM extracts it, and no items for an aspect without paragraphs.

        The usage of each model must have a call and the tokens of each request
        sent to it.

        The APIs are stubbed, i.e. no cassette is required.
        """
        document = data.draw(_documents())
        paragraph_count = len(document.paragraphs)
        roles = st.sampled_from(["extractor", "decider"])
        aspects = []
        for aspect_position in range(data.draw(st.integers(1, 3))):
            aspects.append(
                Aspect(
                    name=f"Aspect {aspect_position}",
                    description=f"Description of aspect {aspect_position}",
                    model_role=data.draw(roles),
                    concepts=[
                        _create_group_concept(
                            f"Aspect {aspect_position} concept {position}", role
                        )
                        for position, role in enumerate(
                            data.draw(st.lists(roles, max_size=2))
                        )
                    ],
                )
            )
        document.aspects = aspects
        document.concepts = [
            _create_group_concept(f"Concept {position}", role)
            for position, role in enumerate(data.draw(st.lists(roles, max_size=2)))
        ]
        # Whether each paragraph belongs to each aspect
        belongs = {
            (aspect_position, paragraph_position): data.draw(st.booleans())
            for aspect_position in range(len(aspects))
            for paragraph_position in range(paragraph_count)
        }
        use_concurrency = data.draw(st.booleans())
        max_items_per_call = data.draw(st.integers(0, 2))
        max_paragraphs = data.draw(st.integers(0, 3))

        def complete(request: dict[str, Any]) -> Any:
            """
            Answers a request to the LLM from its prompt: an aspect gets
            the paragraphs of the prompt that belong to it, and a concept an item
            that names the paragraphs of the prompt, which are its references.
            """
            prompt = _get_prompt(request)
            paragraphs = [
                (paragraph_id, int(position))
                for paragraph_id, position in re.findall(
                    r"<paragraph_id>(P\d+)</paragraph_id>\s*<paragraph_text>\s*Clause (\d+)-",
                    prompt,
                )
            ]
            concepts = re.findall(
                r"<concept_id>(C\d+)</concept_id>\s*<concept_name>(.+?)</concept_name>",
                prompt,
            )
            if concepts:
                positions = [position for _, position in paragraphs]
                output = [
                    {
                        "concept_id": concept_id,
                        "extracted_items": [
                            {
                                "value": f"{name} from {positions}",
                                "reference_paragraph_ids": [
                                    paragraph_id for paragraph_id, _ in paragraphs
                                ],
                            }
                        ],
                    }
                    for concept_id, name in concepts
                ]
            else:
                output = [
                    {
                        "aspect_id": aspect_id,
                        "paragraph_ids": [
                            paragraph_id
                            for paragraph_id, position in paragraphs
                            if belongs[(int(aspect_position), position)]
                        ],
                    }
                    for aspect_id, aspect_position in re.findall(
                        r"<aspect_id>(A\d+)</aspect_id>\s*<aspect_name>Aspect (\d+)</aspect_name>",
                        prompt,
                    )
                ]
            assert output, prompt
            return _completion(json.dumps(output))

        def decide(request: dict[str, Any]) -> Any:
            """
            Answers a request to the decision model: "yes" for an aspect that
            the paragraph of the request belongs to, and for a Boolean concept,
            if the text of the request has the first paragraph of the document.
            """
            text = request["input"]
            answers = []
            for question in request["questions"]:
                match = re.search(
                    r"Aspect name: Aspect (\d+)", question["instructions"]
                )
                if match:
                    paragraph_position, _ = _get_marker(text)
                    yes = belongs[(int(match.group(1)), paragraph_position)]
                else:
                    yes = re.search(r"\bClause 0-", text) is not None
                answers.append(
                    OpenAIPredicateAnswer(
                        name=question["name"], probability=0.9 if yes else 0.1
                    )
                )
            return _decisions_response(answers)

        llm = DocumentLLM(model=_LLM_MODEL, api_key="test_api_key", role="extractor")
        decision_model = DocumentDecisionModel(
            model=_DECISION_MODEL, api_key="test_api_key", role="decider"
        )
        llm.async_limiter = _create_unpaced_limiter()
        decision_model.async_limiter = _create_unpaced_limiter()
        group = DocumentModelGroup(models=[llm, decision_model])
        stub = _ModelAPIsStub(complete=complete, decide=decide)
        with stub.installed(), _recorded_warnings() as caught:
            group.extract_all(
                document,
                use_concurrency=use_concurrency,
                max_items_per_call=max_items_per_call,
                max_paragraphs_to_analyze_per_call=max_paragraphs,
            )
        assert not _get_extraction_warnings(caught)

        def check_concept(concept: Any, positions: list[int]) -> None:
            """
            Checks the items of a concept extracted from the given paragraphs.
            """
            if not positions:
                assert not concept.extracted_items
                return
            parts = _chunk(positions, max_paragraphs)
            if isinstance(concept, StringConcept):
                assert [item.value for item in concept.extracted_items] == [
                    f"{concept.name} from {part}" for part in parts
                ]
                for item, part in zip(concept.extracted_items, parts, strict=True):
                    assert len(item.reference_paragraphs) == len(part)
                    assert all(
                        reference is document.paragraphs[p]
                        for reference, p in zip(
                            item.reference_paragraphs, part, strict=True
                        )
                    )
            else:
                assert [item.value for item in concept.extracted_items] == [
                    0 in part for part in parts
                ]

        for aspect_position, aspect in enumerate(document.aspects):
            positions = [
                p for p in range(paragraph_count) if belongs[(aspect_position, p)]
            ]
            assert aspect._is_processed
            assert len(aspect.extracted_items) == len(positions)
            assert all(
                item.reference_paragraphs[0] is document.paragraphs[p]
                for item, p in zip(aspect.extracted_items, positions, strict=True)
            )
            for concept in aspect.concepts:
                check_concept(concept, positions)
        for concept in document.concepts:
            check_concept(concept, list(range(paragraph_count)))

        usages = {entry.model: entry.usage for entry in group.get_usage()}
        assert len(usages[_LLM_MODEL].calls) == len(stub.completion_requests)
        assert usages[_LLM_MODEL].input == _LLM_INPUT_TOKENS * len(
            stub.completion_requests
        )
        assert len(usages[_DECISION_MODEL].calls) == len(stub.decision_requests)
        assert usages[_DECISION_MODEL].input == _DECISION_INPUT_TOKENS * len(
            stub.decision_requests
        )
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.parametrize("use_concurrency", [False, True])
    def test_extract_all_after_failed_aspect(self, use_concurrency: bool):
        """
        Tests that ``extract_all()`` with ``raise_exception_on_extraction_error=False``
        extracts the concepts of the other aspects and of the document when
        the extraction of an aspect fails, and leaves the concepts of that aspect
        unextracted, with a warning that names them.

        The API of the LLM is stubbed, i.e. no cassette is required.
        """
        document = Document(
            paragraphs=[
                Paragraph(raw_text="The monthly fee is 100 EUR."),
                Paragraph(raw_text="The term of the agreement is two years."),
            ],
            aspects=[
                Aspect(
                    name="Fees",
                    description="Fees of the agreement",
                    concepts=[StringConcept(name="Fee", description="The fee")],
                ),
                Aspect(
                    name="Term",
                    description="Term of the agreement",
                    concepts=[StringConcept(name="Duration", description="The term")],
                ),
            ],
            concepts=[StringConcept(name="Currency", description="The currency")],
        )

        def complete(request: dict[str, Any]) -> Any:
            """
            Answers a request to the LLM: an output that is not JSON for the aspect
            "Fees", the second paragraph for the aspect "Term", and an item for
            each concept.
            """
            prompt = _get_prompt(request)
            if "<aspect_name>Fees</aspect_name>" in prompt:
                return _completion("not JSON")
            aspects = re.findall(r"<aspect_id>(A\d+)</aspect_id>", prompt)
            if aspects:
                output = [
                    {"aspect_id": aspect_id, "paragraph_ids": ["P1"]}
                    for aspect_id in aspects
                ]
            else:
                output = [
                    {"concept_id": concept_id, "extracted_items": [f"{name} value"]}
                    for concept_id, name in re.findall(
                        r"<concept_id>(C\d+)</concept_id>\s*"
                        r"<concept_name>(.+?)</concept_name>",
                        prompt,
                    )
                ]
            assert output, prompt
            return _completion(json.dumps(output))

        llm = DocumentLLM(
            model=_LLM_MODEL, api_key="test_api_key", max_retries_invalid_data=0
        )
        stub = _ModelAPIsStub(complete=complete)
        with stub.installed(), _recorded_warnings() as caught:
            llm.extract_all(
                document,
                max_items_per_call=1,
                use_concurrency=use_concurrency,
                raise_exception_on_extraction_error=False,
            )

        fees, term = document.aspects
        assert not fees._is_processed
        assert not fees.concepts[0]._is_processed
        assert not fees.concepts[0].extracted_items
        assert term._is_processed
        assert [item.value for item in term.extracted_items] == [
            document.paragraphs[1].raw_text
        ]
        assert [item.value for item in term.concepts[0].extracted_items] == [
            "Duration value"
        ]
        assert [item.value for item in document.concepts[0].extracted_items] == [
            "Currency value"
        ]
        messages = _get_messages(caught)
        assert any(
            "could not be processed" in message and "'Fees'" in message
            for message in messages
        ), messages
        assert any(
            "Concepts of aspect `Fees`" in message and "'Fee'" in message
            for message in messages
        ), messages

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_serialization_of_any_configuration(self, data: st.DataObject):
        """
        Tests that a document with any aspects, sub-aspects and concepts, with any
        of their params, and the extraction pipeline of these aspects and concepts,
        are the same after being serialized to a dict, to JSON and to a file and
        restored, and after being cloned.

        No requests are sent.
        """
        roles = st.none() | st.sampled_from(["extractor", "decider"])
        thresholds = st.none() | st.floats(0.01, 1)

        def draw_extraction_params() -> dict[str, Any]:
            """
            Draws the extraction params that aspects and concepts have.
            """
            return {
                "add_justifications": data.draw(st.booleans()),
                "justification_depth": data.draw(
                    st.sampled_from(_JUSTIFICATION_DEPTHS)
                ),
                "justification_max_sents": data.draw(st.integers(1, 5)),
                "reference_depth": data.draw(st.sampled_from(_REFERENCE_DEPTHS)),
                "model_role": data.draw(roles),
            }

        def draw_concepts() -> list[Any]:
            """
            Draws concepts of any types and params.
            """
            return [
                data.draw(
                    _concept_specs(
                        position,
                        {
                            **draw_extraction_params(),
                            "add_references": data.draw(st.booleans()),
                        },
                    )
                ).concept
                for position in range(data.draw(st.integers(0, 3)))
            ]

        def draw_aspects(prefix: str, nested: bool) -> list[Aspect]:
            """
            Draws aspects of any params, with sub-aspects if they are not nested.
            """
            return [
                Aspect(
                    name=f"{prefix}Aspect {position}",
                    description=f"Description of {prefix}aspect {position}",
                    decision_threshold=data.draw(thresholds),
                    aspects=(
                        []
                        if nested
                        else draw_aspects(f"{prefix}Aspect {position} / ", nested=True)
                    ),
                    concepts=draw_concepts(),
                    **draw_extraction_params(),
                )
                for position in range(data.draw(st.integers(0, 2)))
            ]

        aspects = draw_aspects("", nested=False)
        concepts = draw_concepts()
        document = data.draw(_documents())
        document.aspects = aspects
        document.concepts = concepts
        self.check_instance_serialization_and_cloning(document)
        pipeline = ExtractionPipeline(aspects=aspects, concepts=concepts)
        self.check_instance_serialization_and_cloning(pipeline)

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_json_object_concept_any_structure(self, data: st.DataObject):
        """
        Tests that a JSON object concept with any structure of the forms that it
        accepts, at any depth and with any field names, e.g. ``model_config``,
        ``json`` or ``first name``, accepts exactly the values that conform to
        the structure: a value of each field of the type of the field, strictly
        (a value of a Literal of the type of the value that it equals), each field
        that is not optional, and no other field. The concept is created and
        validates the values without warnings, shows each field in the prompt,
        and accepts the same values after serialization, when its structure is
        given as dataclasses, and in its examples.

        No requests are sent.
        """
        structure = data.draw(_structures())
        values = data.draw(
            st.lists(
                st.one_of(
                    _values_of_structure(structure),
                    _altered_values_of_structure(structure),
                    _altered_values_of_structure(structure),
                    _ANY_JSON,
                ),
                min_size=1,
                max_size=4,
            )
        )
        expected = [
            _get_json_object_value(value, structure) is not _INVALID for value in values
        ]
        note(f"Structure: {structure!r}")
        note(f"Values: {values!r}, accepted: {expected!r}")
        event(f"accepted values: {sum(expected)} of {len(expected)}")

        params: dict[str, Any] = {
            "name": "Record",
            "description": "Details of a record",
        }
        with _recorded_warnings() as caught:
            concept = JsonObjectConcept(structure=structure, **params)
            assert [_accepts_json_object(concept, value) for value in values] == (
                expected
            )
        assert not _get_messages(caught)
        prompt_structure = concept._format_structure_in_prompt()
        for name in _get_field_names(structure):
            assert f"{json.dumps(name, ensure_ascii=False)}:" in prompt_structure

        restored = JsonObjectConcept.from_json(concept.to_json())
        assert restored.structure == concept.structure
        assert [_accepts_json_object(restored, value) for value in values] == expected

        class_structure = _to_class_structure(structure)
        if class_structure is not None:
            class_concept = JsonObjectConcept(structure=class_structure, **params)
            assert class_concept.structure == concept.structure
            assert [
                _accepts_json_object(class_concept, value) for value in values
            ] == expected

        for value, accepted in zip(values, expected, strict=True):
            if accepted:
                # Safe cast: an accepted value is an object
                JsonObjectConcept(
                    structure=structure,
                    examples=[JsonObjectExample(content=cast(dict[str, Any], value))],
                    **params,
                )
            elif (
                isinstance(value, dict)
                and value
                and not _conforms_to_structure(value, structure)
            ):
                with pytest.raises(ValueError):
                    JsonObjectConcept(
                        structure=structure,
                        examples=[JsonObjectExample(content=value)],
                        **params,
                    )

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_json_object_concept_unsupported_structure(self, data: st.DataObject):
        """
        Tests that a JSON object concept is not created with a structure that has
        a type that it does not support, at any depth: a type that is not JSON,
        e.g. a tuple or a set, a mapping with keys of another type than strings,
        which a JSON object cannot have, or an optional type that is not
        a primitive type or a Literal. The structure is rejected with
        ``ValueError``, and not with another error.

        No requests are sent.
        """
        structure = data.draw(_structures())
        # A field of the structure, or of a structure nested in it, at any depth
        target = structure
        while True:
            nested = [
                field_type[0] if isinstance(field_type, list) else field_type
                for field_type in target.values()
                if isinstance(
                    field_type[0] if isinstance(field_type, list) else field_type,
                    dict,
                )
            ]
            if not nested or data.draw(st.booleans()):
                break
            target = data.draw(st.sampled_from(nested))
        unsupported_type = data.draw(st.sampled_from(_UNSUPPORTED_FIELD_TYPES))
        target[data.draw(st.sampled_from(sorted(target)))] = unsupported_type
        note(f"Structure: {structure!r}")
        with pytest.raises(ValueError):
            JsonObjectConcept(
                name="Record", description="Details of a record", structure=structure
            )
