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

This module configures a Loguru-based logging system with environment variable controls
for log level and enabling/disabling logging. It includes a dedicated stream wrapper
for consistent log formatting.
"""

from __future__ import annotations

import os
import sys

from loguru import logger

from contextgem.internal.suppressions import _install_litellm_noise_filters


DEFAULT_LOGGER_LEVEL = "INFO"

# Dynamically control logging state with env vars
LOGGER_LEVEL_ENV_VAR_NAME = "CONTEXTGEM_LOGGER_LEVEL"


class _DedicatedStream:
    """
    A dedicated stream wrapper for formatting and directing messages to
    a base stream.
    """

    def __init__(self, base):
        self.base = base

    def write(self, message):
        """
        Writes a message to the base stream with contextgem prefix.

        :param message: The message to write to the stream.
        :type message: str
        """
        # You can add a prefix or other formatting if you wish
        self.base.write(f"[contextgem] {message}")

    def flush(self):
        """
        Flushes the base stream to ensure all output is written.
        """
        self.base.flush()


dedicated_stream = _DedicatedStream(sys.stdout)


# Helper to read environment config at import time
def _read_env_vars() -> tuple[bool, str]:
    """
    Returns the (disabled_status, level) read from environment variables.
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


def _apply_color_scheme():
    """
    Defines custom colors for each log level (mimicking colorlog style)
    """
    logger.level("DEBUG", color="<cyan>")
    logger.level("INFO", color="<blue>")
    logger.level("SUCCESS", color="<green>")
    logger.level("WARNING", color="<yellow>")
    logger.level("ERROR", color="<red>")
    logger.level("CRITICAL", color="<red><bold>")


# Main configuration function
def _configure_logger_from_env():
    """
    Configures the Loguru logger based on environment variables.
    This can be called at import time (once) or re-called any time.

    (Loguru does not require `getLogger(name)`; we just import `logger` and use it.)
    """
    disable_logger, level_str = _read_env_vars()

    # Remove default handlers
    logger.remove()

    # If logging is disabled (OFF level), just disable and don't add any handlers
    if disable_logger:
        logger.disable("")
        return

    # Enable logging and add handler
    logger.enable("")

    # Apply custom level color scheme
    _apply_color_scheme()

    logger.add(
        dedicated_stream,
        level=level_str,
        colorize=True,
        format=(
            "<white>{time:YYYY-MM-DD HH:mm:ss.SSS}</white> | "
            "<level>{level: <7}</level> | "
            "{message}"
        ),
    )

    # Install filters to suppress noisy third-party dependency logs
    _install_litellm_noise_filters()


_configure_logger_from_env()
