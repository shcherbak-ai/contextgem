# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining custom exceptions for the ContextGem framework.
"""

from __future__ import annotations


# Base exception


class ContextGemError(Exception):
    """
    Base class of the exceptions that the framework raises when a run fails, i.e.
    of :class:`ModelAPIError`, :class:`ModelExtractionError` (with its subclass
    :class:`ModelRefusalError`) and :class:`LLMToolLoopLimitError`. Catch it to
    handle any of them in one clause, e.g. ``except ContextGemError``.

    An error in the calling code, e.g. an invalid argument or an object that
    cannot be used where it is passed, is raised as ``ValueError`` or ``TypeError``
    instead, which includes the ``ValidationError`` of pydantic, a subclass
    of ``ValueError``. A file that cannot be read or written, e.g. by ``from_disk()``
    or ``to_disk()``, raises ``OSError``.

    Can be imported from ``contextgem``.

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of retries attempted before failure
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure
    :vartype original_error: Exception or None
    """

    def __init__(
        self,
        message: str,
        retry_count: int | None = None,
        original_error: Exception | None = None,
    ):
        """
        Initialize the exception.

        :param message: Human-readable error description
        :type message: str
        :param retry_count: Number of retries attempted before failure
        :type retry_count: int or None
        :param original_error: The underlying exception that caused the failure
        :type original_error: Exception or None
        """
        super().__init__(message)
        self.message = message
        self.retry_count = retry_count
        self.original_error = original_error

    def __str__(self) -> str:
        """
        Return a string representation of the exception.

        :return: Formatted error message with retry and original error information
        :rtype: str
        """
        base_msg = self.message

        if self.retry_count is not None:
            base_msg += f" (after {self.retry_count} retries)"

        if self.original_error is not None:
            base_msg += f" - Original error: {self.original_error}"

        return base_msg


# Model exceptions


class ModelExtractionError(ContextGemError):
    """
    Exception raised by the extraction methods when a model returns data that cannot
    be used, and there is no fallback model left to try.

    This is the case when:

    - the response of an LLM cannot be parsed or validated, also after the LLM
      was asked again ``max_retries_invalid_data`` times;
    - the response of a decision model cannot be used, e.g. an answer without
      valid probabilities, also after the requests whose answers cannot be used
      were sent again ``max_retries_invalid_data`` times;
    - the model or its provider declined to answer, also after the retries and
      the fallback models: then the subclass :class:`ModelRefusalError` is raised.

    If ``raise_exception_on_extraction_error=False`` is passed to the extraction
    method, a warning is logged instead.

    Can be imported from ``contextgem``.

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of times that the model the error is for, i.e. the last
        model that was tried, was called again after it returned invalid data.
        The first call of the model, also of a fallback model, is not counted.
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure
    :vartype original_error: Exception or None
    """


class ModelRefusalError(ModelExtractionError):
    """
    Exception raised when a model or its provider declined to answer, in extraction
    and in chat, and there is no retry and no fallback model left to try:

    - the model refused to answer, e.g. an LLM of OpenAI with its ``refusal``
      field, or a decision model with a refused answer to a question of a concept,
      or to the question of an aspect on every paragraph;
    - the content filter of the provider stopped the answer, which litellm reports
      with the ``content_filter`` finish reason for all the providers, e.g. for
      a refusal of a Claude model, the safety filters of Gemini and of Azure
      OpenAI, and the guardrails of Amazon Bedrock;
    - the provider blocked the request or its answer for their content, which
      litellm raises as ``litellm.ContentPolicyViolationError`` for both, e.g.
      the content filter of Azure OpenAI on a prompt, or a blocked answer of
      a Gemini or a Claude model on Vertex AI.

    litellm reports a model's refusal and a content filter with the same finish
    reason, i.e. the message cannot always tell them apart. It also reports
    a question that a decision model left unanswered as a refused answer.

    The answer of a refused request is not used, also if it is valid JSON. A refusal
    is asked for again as invalid data is, up to ``max_retries_invalid_data`` times,
    as the verdict of a model or of a content filter can differ on another request,
    and then from the fallback models, each with its own retries. In a chat,
    a refused answer that has a text is returned as it is, with a warning logged,
    while a refused answer without a text, or a blocked request, is asked for again
    as an empty answer is, and then from the next fallback LLM.

    As a subclass of :class:`ModelExtractionError`, it is caught by
    ``except ModelExtractionError`` and ``except ContextGemError``. If
    ``raise_exception_on_extraction_error=False`` is passed to the extraction
    method, a warning is logged instead, and the aspects or concepts of the call
    get no items.

    The message names the model and what declined the answer. It never has
    the answer of the model or the text of the document.

    Can be imported from ``contextgem``.

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of times that the model the error is for, i.e. the last
        model that was tried, was called again, as for
        :class:`ModelExtractionError`. In a chat, the number of times that
        the request was sent again.
    :vartype retry_count: int or None
    :ivar original_error: The exception of litellm for a request or an answer that
        the provider blocked (``litellm.ContentPolicyViolationError``), of the last
        request that was sent, None otherwise
    :vartype original_error: Exception or None
    """


class ModelAPIError(ContextGemError):
    """
    Exception raised when a request to a model fails for good, and there is
    no fallback model left to try.

    This is the case when:

    - the request fails with a non-transient error, e.g. an invalid request or
      a failed authentication;
    - the request fails with a transient error (request timeout, rate limit, server
      or connection error), also after ``num_retries_failed_request`` retries;
    - the response of an LLM is cut off at the max output tokens during extraction,
      which is not requested again, or is empty in a chat, also after the request
      was sent again ``max_retries_invalid_data`` times.

    A request that the provider blocks for its content, or whose answer it blocks,
    is a refusal, which raises :class:`ModelRefusalError`.

    During extraction, if ``raise_exception_on_extraction_error=False`` is passed
    to the extraction method, a warning is logged instead.

    Can be imported from ``contextgem``.

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of retries attempted for the failed request, or, for
        an empty response in a chat, the number of times that the request was sent
        again. None for a response cut off at the max output tokens during
        extraction, which is not requested again.
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure, of
        ``litellm`` for a failed request. None for a response that is cut off or
        empty, which the message describes.
    :vartype original_error: Exception or None
    """


# Tool calling / tooling exceptions


class LLMToolLoopLimitError(ContextGemError):
    """
    Exception raised from a chat when the tool-calling loop reaches the configured
    ``tool_max_rounds`` limit and the LLM continues requesting tools.

    Typically indicates that the LLM is stuck in a loop or needs more rounds to
    complete the workflow. Consider increasing ``tool_max_rounds`` or adjusting
    the tool instructions or the tool choice.

    Can be imported from ``contextgem``.
    """


# Warnings


class ContextGemWarning(UserWarning):
    """
    Category of the warnings that the framework issues for something in the calling
    code that will not have the expected effect, e.g. tools set on an LLM that is
    used for extraction, or an output language set together with a custom system
    message. Such a warning is attributed to the line of the calling code that
    caused it, where possible.

    The warnings are Python warnings, not log messages, i.e. the logging level does
    not affect them. To filter them, use this category, e.g.
    ``warnings.filterwarnings("ignore", category=ContextGemWarning)``, or make them
    errors, e.g. in the tests of the calling code: a filter by the module of
    the framework does not match a warning attributed to the calling code.
    What happens while the framework runs, e.g. an extraction that fails when
    ``raise_exception_on_extraction_error=False`` is passed to the extraction
    method, is logged instead.

    Can be imported from ``contextgem``.
    """


class UnknownPricesWarning(ContextGemWarning):
    """
    Category of the warning that the framework issues for a model whose costs it
    cannot calculate: a model with ``auto_pricing=True`` that ``genai-prices`` has
    no prices for. The warning is issued when such a model is created, or when its
    ``model`` or ``auto_pricing`` is assigned, and again when the usage of its calls
    is recorded. The costs of the model stay at zero.

    To keep such a model from running, e.g. where its costs count against a budget
    of your own, make the warning an error:
    ``warnings.filterwarnings("error", category=UnknownPricesWarning)``. The model
    then raises where it is created, and a model created before the filter is set
    stops the extraction method after the usage of the call is recorded. A subclass
    of ``ContextGemWarning``, which a filter on that category matches as well.

    Can be imported from ``contextgem``.
    """


class _ToolReturnValueError(Exception):
    """
    Internal exception that carries the error for an invalid value returned by
    a tool function (a value other than a non-empty string) out of the tool-calling
    loop, so that it is not handled as an error of the LLM API: it is an error
    in the calling code. It is never raised to the calling code: the error
    it carries is raised instead.

    :ivar error: The error for the invalid value returned by the tool function
    :vartype error: Exception
    """

    def __init__(self, error: Exception):
        """
        Initialize the exception.

        :param error: The error for the invalid value returned by the tool function
        :type error: Exception
        """
        super().__init__(str(error))
        self.error = error


# Decision model exceptions


class _InvalidDecisionResponseError(ValueError):
    """
    Internal exception raised when the answer of a decision model to a question
    cannot be used, which makes the whole response of the model invalid: the answer
    is missing, refused, of another kind than the question, without the needed
    probabilities, with a probability outside [0, 1], with options other than
    the options offered, or with a zero probability of each option. A refused
    answer to the question of an aspect can be used, as "not included in
    the aspect", unless the model refused to answer the question on every paragraph.

    :ivar question_name: Name of the question whose answer cannot be used
    :vartype question_name: str
    :ivar reason: What is wrong with the answer
    :vartype reason: str
    :ivar location: Where the text that the question was asked on is, as a part of
        a message, e.g. " on paragraph 3 of 5 (index 2)". Empty if it is not known,
        or if the text is not a part of a larger one.
    :vartype location: str
    """

    def __init__(self, question_name: str, reason: str):
        """
        Initialize the exception.

        :param question_name: Name of the question whose answer cannot be used
        :type question_name: str
        :param reason: What is wrong with the answer
        :type reason: str
        """
        super().__init__(
            f"Invalid answer to the decision question `{question_name}`: {reason}"
        )
        self.question_name = question_name
        self.reason = reason
        self.location: str = ""


class _RefusedDecisionAnswerError(_InvalidDecisionResponseError):
    """
    Internal exception raised when the answer of a decision model to a question
    cannot be used because the model refused to answer it: a refused answer to
    a question of a concept, or to the question of an aspect on every paragraph.
    The response is handled as other responses that cannot be used, i.e. passed
    on to the fallback models, and the extraction raises
    :class:`ModelRefusalError` for it if none of them answers.
    """
