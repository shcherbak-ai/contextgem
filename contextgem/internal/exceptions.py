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
Module defining custom exceptions for the ContextGem framework.
"""

from __future__ import annotations


# Model exceptions


class _BaseModelError(Exception):
    """
    Base exception class for model-related errors.

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


class ModelExtractionError(_BaseModelError):
    """
    Exception raised when model extraction operations fail after all configured retries.

    This exception is typically raised when:
    - Model extraction attempts have been exhausted without success
    - The extracted data doesn't meet validation requirements
    - Parsing of extracted content fails consistently

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of times that the model the error is for, i.e. the last
        model that was tried, was called again after it returned invalid data.
        The first call of the model, also of a fallback model, is not counted.
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure
    :vartype original_error: Exception or None
    """

    pass


class ModelAPIError(_BaseModelError):
    """
    Exception raised when model API calls fail.

    This exception is typically raised when:
    - Network errors occur during API communication
    - API authentication or authorization fails
    - Rate limits are exceeded
    - API service is unavailable or returns unexpected responses
    - Request timeouts occur
    - The response of the model is cut off at the max output tokens during extraction,
      or is empty in a chat

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of retries attempted before failure
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure
    :vartype original_error: Exception or None
    """

    pass


# Tool calling / tooling exceptions


class LLMToolLoopLimitError(_BaseModelError):
    """
    Exception raised when the assistant's tool-calling loop reaches the
    configured `tool_max_rounds` limit and the model continues requesting tools.

    Typically indicates the model is stuck in a loop or needs more rounds to
    complete the workflow. Consider increasing `tool_max_rounds` or adjusting
    tool instructions/choice.
    """

    pass


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
    :ivar text_position: Position of the text that the question was asked on, among
        the texts that the questions of the same call were asked on, if it is known
    :vartype text_position: int | None
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
        self.text_position: int | None = None
        self.location: str = ""
