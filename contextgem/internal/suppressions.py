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
Centralized suppression utilities for third-party dependency noise (such as LiteLLM).

- Logging: filters to hide non-actionable error logs emitted by third-party dependencies
  that are irrelevant to ContextGem users.

- Warnings: context managers and decorators to suppress non-actionable warnings emitted
  by third-party dependencies that are irrelevant to ContextGem users.

TODO: Remove this module (in whole or in part) once the relevant third-party dependencies
stop emitting the corresponding noisy logs/warnings that are irrelevant to ContextGem users.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import os
import traceback
import warnings
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, TypeVar, cast


F = TypeVar("F", bound=Callable[..., Any])


# Singleton used to avoid adding duplicate LiteLLM suppression filters on reloads
_CONTEXTGEM_SUPPRESS_LITELLM_NOISE_FILTER: _SuppressLiteLLMNoiseFilter | None = None


class _SuppressLiteLLMNoiseFilter(logging.Filter):
    """
    Suppress noisy LiteLLM logs that are non-actionable for ContextGem users, such as:
    - Proxy optional dependency errors (e.g., missing 'backoff', suggests `litellm[proxy]`).
    - Logging worker noise (e.g., "LoggingWorker cancelled", asyncio.CancelledError traces).

    Rationale:
    - ContextGem does not require LiteLLM proxy features. Recent LiteLLM versions
      attempt to set up proxy logging on import and emit ERROR logs if proxy extras are not
      installed. This is confusing for users since ContextGem works without those extras.
    - LiteLLM's internal async logging worker can generate cancellation/loop-mismatch messages
      during test teardown or when event loops change. These are benign in ContextGem's usage
      and create distracting noise.

    Scope:
    - The filter is attached only to `LiteLLM`, `litellm`, and `asyncio` loggers to avoid global
      side effects. For `asyncio`, suppression is limited to entries clearly tied to LiteLLM
      (requires "litellm" in the message or traceback frames from litellm).
    - Suppression can be disabled by env var `CONTEXTGEM_SUPPRESS_LITELLM_NOISE=0`.
    """

    PROXY_DEP_MISSING_KEYWORDS = (
        "missing dependency",
        "backoff",
        "litellm[proxy]",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter LiteLLM-related noisy records.

        Suppresses:
        - Proxy optional dependency errors (e.g., missing 'backoff', suggests `litellm[proxy]`).
        - Async logging worker noise tied to LiteLLM (e.g., cancellations, "Task exception was
          never retrieved", or event-loop mismatch), only when the message or traceback references
          LiteLLM.

        :param record: The log record to evaluate
        :type record: logging.LogRecord
        :return: True to log the record, False to suppress it
        :rtype: bool
        """
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg)

        name_lower = record.name.lower()
        msg_lower = message.lower()

        is_litellm_logger = name_lower.startswith("litellm")
        is_asyncio_logger = name_lower.startswith("asyncio")

        mentions_litellm = "litellm" in msg_lower
        mentions_worker = ("loggingworker" in msg_lower) or (
            "logging_worker" in msg_lower
        )

        # Check traceback frames for litellm involvement when available
        litellm_in_traceback = False
        if (
            record.exc_info
            and isinstance(record.exc_info, tuple)
            and record.exc_info[2] is not None
        ):
            try:
                for frame in traceback.extract_tb(record.exc_info[2]):
                    if "litellm" in str(frame.filename).lower():
                        litellm_in_traceback = True
                        break
            except Exception:
                litellm_in_traceback = False

        # Only handle records from litellm, or asyncio records clearly tied to litellm
        if not (
            is_litellm_logger
            or (is_asyncio_logger and (mentions_litellm or litellm_in_traceback))
        ):
            return True

        # Case 1: Proxy optional dependency error noise
        is_proxy_dep_error = all(
            kw in msg_lower for kw in self.PROXY_DEP_MISSING_KEYWORDS
        )
        if is_proxy_dep_error:
            return False

        # Case 2: Async logging worker cancellation/noise
        # Examples: "LoggingWorker cancelled", stack traces with asyncio.CancelledError,
        # "coroutine ... was never retrieved" related to logging worker
        has_cancelled = "cancelled" in msg_lower
        has_never_retrieved = "never retrieved" in msg_lower
        has_different_loop = "different event loop" in msg_lower

        exc_is_cancelled = False
        if (
            record.exc_info
            and isinstance(record.exc_info, tuple)
            and len(record.exc_info) >= 1
        ):
            try:
                exc_type = record.exc_info[0]
                exc_is_cancelled = isinstance(exc_type, type) and issubclass(
                    exc_type, asyncio.CancelledError
                )
            except Exception:
                exc_is_cancelled = False

        is_litellm_related = (
            mentions_litellm or litellm_in_traceback or is_litellm_logger
        )

        is_logging_worker_noise = is_litellm_related and (
            has_cancelled
            or has_never_retrieved
            or has_different_loop
            or mentions_worker
            or exc_is_cancelled
        )

        # Allow everything else unless it's recognized as logging worker noise
        return not is_logging_worker_noise


def _install_litellm_noise_filters() -> None:
    """
    Install standard-logging filters to hide noisy, non-actionable dependency logs.

    Controlled by env var `CONTEXTGEM_SUPPRESS_LITELLM_NOISE` (default: "1").
    Set to "0" to disable suppression.

    :return: None
    :rtype: None
    """

    suppress = os.getenv("CONTEXTGEM_SUPPRESS_LITELLM_NOISE", "1") != "0"
    if not suppress:
        return

    global _CONTEXTGEM_SUPPRESS_LITELLM_NOISE_FILTER
    if _CONTEXTGEM_SUPPRESS_LITELLM_NOISE_FILTER is None:
        _CONTEXTGEM_SUPPRESS_LITELLM_NOISE_FILTER = _SuppressLiteLLMNoiseFilter()

    # Apply only to LiteLLM-related loggers and asyncio (for logging worker noise)
    # to avoid global side effects
    for name in ("LiteLLM", "litellm", "asyncio"):
        lg = logging.getLogger(name)
        existing_filters = getattr(lg, "filters", [])
        if not any(
            isinstance(f, _SuppressLiteLLMNoiseFilter) for f in existing_filters
        ):
            lg.addFilter(_CONTEXTGEM_SUPPRESS_LITELLM_NOISE_FILTER)


@contextmanager
def _suppress_litellm_warnings_context() -> Generator[None, None, None]:
    """
    Context manager that suppresses Pydantic and httpx deprecation and serialization warnings.

    This context manager temporarily suppresses DeprecationWarnings and specific UserWarnings
    from the Pydantic and httpx modules. These warnings are typically generated by litellm's
    internal usage of these libraries and are not actionable by users.

    Specifically suppresses:
    - Pydantic DeprecationWarnings
    - Pydantic serialization UserWarnings
    - httpx DeprecationWarnings about content parameter usage (typically when using local LLMs)

    The context manager uses Python's warnings.catch_warnings() to create a temporary
    warning filter that ignores the specified warnings only within its scope.

    :return: A generator that yields None and expects no values to be sent back.
    :rtype: Generator[None, None, None]
    """
    with warnings.catch_warnings():
        # Pydantic
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="pydantic"
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="pydantic",
            message=r"^Pydantic serializer warnings",
        )
        # httpx (typically when using local LLMs)
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="httpx",
            message="Use 'content=<...>'",
        )
        yield


def _suppress_litellm_warnings(func: F) -> F:
    """
    A decorator that suppresses warnings related to Pydantic and httpx deprecation
    and serialization in litellm>1.71.1 (latest available version as of 2025-07-10).

    This decorator wraps both synchronous and asynchronous functions to suppress
    Pydantic and httpx warnings that originate from litellm's internal usage.
    These warnings are not actionable by users and should not be displayed.

    Suppression rationale:
    - The warnings originate from litellm's internal pydantic and httpx usage, not user code
    - These are dependency warnings that users cannot fix and should not see

    :param func: The function to wrap with warning suppression. Can be either
        synchronous or asynchronous.
    :type func: F
    :return: The wrapped function with the same signature as the input function,
        but with Pydantic and httpx deprecation and serialization warnings suppressed
        during execution.
    :rtype: F

    TODO: Remove deprecation-related suppression when deprecation warnings are fixed
    in a future litellm release.

    TODO: Remove serialization-related suppression when the related issue is fixed
    in a future litellm release: https://github.com/BerriAI/litellm/issues/11759
    """

    @functools.wraps(func)
    async def _async_wrapper(*args, **kwargs) -> Any:
        """
        Async wrapper that suppresses Pydantic and httpx deprecation and serialization
        warnings during async function execution.

        :param args: Positional arguments to pass to the wrapped function.
        :param kwargs: Keyword arguments to pass to the wrapped function.
        :return: The result of the wrapped async function.
        :rtype: Any
        """
        with _suppress_litellm_warnings_context():
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def _sync_wrapper(*args, **kwargs) -> Any:
        """
        Sync wrapper that suppresses Pydantic and httpx deprecation and serialization
        warnings during sync function execution.

        :param args: Positional arguments to pass to the wrapped function.
        :param kwargs: Keyword arguments to pass to the wrapped function.
        :return: The result of the wrapped sync function.
        :rtype: Any
        """
        with _suppress_litellm_warnings_context():
            return func(*args, **kwargs)

    # Safe cast: tell type checker that functools.wraps preserves the original function's type.
    # The wrapper functions maintain the same signature as the original function.
    return cast(
        F, _async_wrapper if inspect.iscoroutinefunction(func) else _sync_wrapper
    )
