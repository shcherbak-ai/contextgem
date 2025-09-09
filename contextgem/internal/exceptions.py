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


# LLM exceptions


class _BaseLLMError(Exception):
    """
    Base exception class for LLM-related errors.

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


class LLMExtractionError(_BaseLLMError):
    """
    Exception raised when LLM extraction operations fail after all configured retries.

    This exception is typically raised when:
    - LLM extraction attempts have been exhausted without success
    - The extracted data doesn't meet validation requirements
    - Parsing of extracted content fails consistently

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of retries attempted before failure
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure
    :vartype original_error: Exception or None
    """

    pass


class LLMAPIError(_BaseLLMError):
    """
    Exception raised when LLM API calls fail.

    This exception is typically raised when:
    - Network errors occur during API communication
    - API authentication or authorization fails
    - Rate limits are exceeded
    - API service is unavailable or returns unexpected responses
    - Request timeouts occur

    :ivar message: Human-readable error description
    :vartype message: str
    :ivar retry_count: Number of retries attempted before failure
    :vartype retry_count: int or None
    :ivar original_error: The underlying exception that caused the failure
    :vartype original_error: Exception or None
    """

    pass


# Tool calling / tooling exceptions


class LLMToolLoopLimitError(_BaseLLMError):
    """
    Exception raised when the assistant's tool-calling loop reaches the
    configured `tool_max_rounds` limit and the model continues requesting tools.

    Typically indicates the model is stuck in a loop or needs more rounds to
    complete the workflow. Consider increasing `tool_max_rounds` or adjusting
    tool instructions/choice.
    """

    pass


# DocxConverter exceptions


class DocxConverterError(Exception):
    """
    Base exception class for DOCX converter errors.
    """

    pass


class DocxFormatError(DocxConverterError):
    """
    Exception raised when the DOCX file format is invalid or corrupted.
    """

    pass


class DocxXmlError(DocxConverterError):
    """
    Exception raised when there's an error parsing XML in the DOCX file.
    """

    pass


class DocxContentError(DocxConverterError):
    """
    Exception raised when required content is missing from the DOCX file.
    """

    pass
