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
Module providing a customized logging configuration for the ContextGem framework.

This module configures a standard library logging system with environment variable controls
for log level and enabling/disabling logging. It uses a namespaced logger ('contextgem').
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Protocol

import colorlog

from contextgem.internal.suppressions import _install_litellm_noise_filters


class _LoggerProtocol(Protocol):
    """
    Protocol defining the logger interface with custom methods.

    This Protocol is used purely for type checking to inform type checkers
    (e.g. Pyright) about the custom .trace() and .success() methods that
    are dynamically added to logging.Logger at runtime.
    """

    propagate: bool
    handlers: list[logging.Handler]

    def trace(self, message: str, *args, **kwargs) -> None:
        """
        Log with TRACE level (below DEBUG).
        """
        ...

    def debug(self, message: str, *args, **kwargs) -> None:
        """
        Log with DEBUG level.
        """
        ...

    def info(self, message: str, *args, **kwargs) -> None:
        """
        Log with INFO level.
        """
        ...

    def success(self, message: str, *args, **kwargs) -> None:
        """
        Log with SUCCESS level (between INFO and WARNING).
        """
        ...

    def warning(self, message: str, *args, **kwargs) -> None:
        """
        Log with WARNING level.
        """
        ...

    def error(self, message: str, *args, **kwargs) -> None:
        """
        Log with ERROR level.
        """
        ...

    def critical(self, message: str, *args, **kwargs) -> None:
        """
        Log with CRITICAL level.
        """
        ...

    def addHandler(self, handler: logging.Handler) -> None:  # noqa: N802
        """
        Adds a handler to the logger.
        """
        ...

    def removeHandler(self, handler: logging.Handler) -> None:  # noqa: N802
        """
        Removes a handler from the logger.
        """
        ...

    def setLevel(self, level: int) -> None:  # noqa: N802
        """
        Sets the logging level.
        """
        ...


DEFAULT_LOGGER_LEVEL = "INFO"

# Dynamically control logging state with env vars
LOGGER_LEVEL_ENV_VAR_NAME = "CONTEXTGEM_LOGGER_LEVEL"

# Add custom levels
TRACE_LEVEL_NUM = 5  # Below DEBUG (10)
SUCCESS_LEVEL_NUM = 25  # Between INFO (20) and WARNING (30)
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.addLevelName(SUCCESS_LEVEL_NUM, "SUCCESS")


def _trace(self, message, *args, **kwargs):
    """
    Logs a message with severity 'TRACE' on this logger.

    This is a custom level below DEBUG.
    """
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)


def _success(self, message, *args, **kwargs):
    """
    Logs a message with severity 'SUCCESS' on this logger.

    This is a custom level between INFO and WARNING.
    """
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)


# Add custom methods to Logger class
logging.Logger.trace = _trace  # type: ignore[attr-defined]
logging.Logger.success = _success  # type: ignore[attr-defined]

# Create a namespaced logger for ContextGem
logger: _LoggerProtocol = logging.getLogger("contextgem")  # type: ignore[assignment]

# Add NullHandler by default
logger.addHandler(logging.NullHandler())

# Track our handler to avoid duplicates
_contextgem_handler: logging.Handler | None = None
_handler_lock = threading.Lock()


def _read_env_vars() -> tuple[bool, str]:
    """
    Returns the (disabled_status, level) read from environment variables.

    :return: Tuple of (should_disable, level_string)
    :rtype: tuple[bool, str]
    """

    # Default to DEFAULT_LOGGER_LEVEL if no variable is set or invalid
    level_str = os.getenv(LOGGER_LEVEL_ENV_VAR_NAME, DEFAULT_LOGGER_LEVEL).upper()
    valid_levels = [
        "TRACE",
        "DEBUG",
        "INFO",
        "SUCCESS",
        "WARNING",
        "ERROR",
        "CRITICAL",
        "OFF",
    ]
    if level_str not in valid_levels:
        level_str = DEFAULT_LOGGER_LEVEL

    # Check if logging should be disabled
    disable_logger = level_str == "OFF"
    return disable_logger, level_str


def _get_colored_formatter() -> logging.Formatter:
    """
    Creates a colored formatter using colorlog with millisecond precision.

    :return: A logging formatter with color support and milliseconds
    :rtype: logging.Formatter
    """

    # Use colorlog for colored output with milliseconds
    return colorlog.ColoredFormatter(
        "[contextgem] %(log_color)s%(asctime)s.%(msecs)03d%(reset)s | "
        "%(log_color)s%(levelname)-7s%(reset)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "TRACE": "dim",
            "DEBUG": "cyan",
            "INFO": "blue",
            "SUCCESS": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bold",
        },
        style="%",
    )


def _configure_logger_from_env():
    """
    Configures the contextgem logger based on environment variables.
    This can be called at import time (once) or re-called any time.

    This function only affects the 'contextgem' logger and is thread-safe.

    :return: None
    :rtype: None
    """
    global _contextgem_handler

    with _handler_lock:
        disable_logger, level_str = _read_env_vars()

        # Remove our previous handler if it exists
        if _contextgem_handler is not None:
            logger.removeHandler(_contextgem_handler)
            _contextgem_handler = None

        # If logging is disabled (OFF level), remove all handlers except NullHandler
        if disable_logger:
            # Remove all non-NullHandler handlers
            for handler in logger.handlers[:]:
                if not isinstance(handler, logging.NullHandler):
                    logger.removeHandler(handler)
            # Set level high to ensure nothing gets through
            logger.setLevel(logging.CRITICAL + 1)
            # Don't propagate to avoid any output
            logger.propagate = False
            return

        # Enable logging and add handler
        # Handle custom levels specially
        if level_str == "TRACE":
            logger.setLevel(TRACE_LEVEL_NUM)
        elif level_str == "SUCCESS":
            logger.setLevel(SUCCESS_LEVEL_NUM)
        else:
            logger.setLevel(getattr(logging, level_str))

        # Don't propagate - we manage our own output
        # This prevents duplicate logs if the root logger also has handlers
        logger.propagate = False

        # Create and configure handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_get_colored_formatter())
        logger.addHandler(handler)
        _contextgem_handler = handler

        # Install filters to suppress noisy third-party dependency logs
        _install_litellm_noise_filters()


# Configure on import
_configure_logger_from_env()
