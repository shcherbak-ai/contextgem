# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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
from typing import Protocol, TextIO

import colorlog

from contextgem.internal.suppressions import _configure_litellm_noise_filters


class _LoggerProtocol(Protocol):
    """
    Protocol defining the logger interface with custom methods.

    This Protocol is used purely for type checking to inform type checkers
    (e.g. ty) about the custom .success() method that is dynamically added
    to logging.Logger at runtime.
    """

    propagate: bool
    handlers: list[logging.Handler]

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


def _success(self, message, *args, **kwargs):
    """
    Logs a message with severity 'SUCCESS' on this logger.

    This is a custom level between INFO and WARNING.
    """
    if self.isEnabledFor(SUCCESS_LEVEL_NUM):
        self._log(SUCCESS_LEVEL_NUM, message, args, **kwargs)


# Add the custom method to the Logger class
logging.Logger.success = _success  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]

# Create a namespaced logger for ContextGem
logger: _LoggerProtocol = logging.getLogger("contextgem")  # type: ignore[assignment]  # ty: ignore[invalid-assignment]

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


def _get_colored_formatter(stream: TextIO) -> logging.Formatter:
    """
    Creates a colored formatter using colorlog with millisecond precision.

    The colors are written only where the stream is a terminal, e.g. not to a file
    or to the log of a CI job, which would show their escape codes as text, unless
    the ``FORCE_COLOR`` environment variable is set. With ``NO_COLOR`` set, they are
    not written at all (see https://no-color.org/).

    :param stream: The stream that the log is written to.
    :type stream: TextIO
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
        stream=stream,
    )


def _configure_logger_from_env():
    """
    Configures the contextgem logger based on environment variables.
    This can be called at import time (once) or re-called any time.

    This function is thread-safe. Besides the 'contextgem' logger, it only installs
    the filters that suppress the noisy logs of third-party dependencies, which
    are installed whatever the level of the logger is, including OFF, or removes
    them if the suppression is disabled.

    :return: None
    :rtype: None
    """
    global _contextgem_handler

    with _handler_lock:
        disable_logger, level_str = _read_env_vars()

        # Install filters to suppress noisy third-party dependency logs, or
        # remove them if the suppression is disabled.
        # They do not depend on the level, which is that of our logger only:
        # with our logger disabled, such logs would otherwise still be shown
        _configure_litellm_noise_filters()

        # Remove our previous handler if it exists
        if _contextgem_handler is not None:
            logger.removeHandler(_contextgem_handler)
            _contextgem_handler = None

        # If logging is disabled (OFF level), no handler of ours is added. The handlers
        # that the application added to the logger are kept, to get the messages again
        # when logging is enabled
        if disable_logger:
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
        handler.setFormatter(_get_colored_formatter(sys.stdout))
        logger.addHandler(handler)
        _contextgem_handler = handler


# Configure on import
_configure_logger_from_env()
