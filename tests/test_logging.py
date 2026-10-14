# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the logger.
"""

from __future__ import annotations

import logging
import sys
import warnings

import pytest
from pydantic import TypeAdapter

from contextgem.internal.loggers import LOGGER_LEVEL_ENV_VAR_NAME, logger
from contextgem.internal.suppressions import (
    _suppress_litellm_warnings_context,
    _SuppressLiteLLMNoiseFilter,
)
from contextgem.public import reload_logger_settings
from tests.base import BaseTest


# Loggers that the filter of the noisy logs of litellm is installed on
_LITELLM_NOISE_FILTER_LOGGER_NAMES = ("LiteLLM", "litellm", "asyncio")

# Environment variable that the filter of the noisy logs of litellm is disabled with
_SUPPRESS_LITELLM_NOISE_ENV_VAR_NAME = "CONTEXTGEM_SUPPRESS_LITELLM_NOISE"

if sys.version_info >= (3, 13):
    from typing import ReadOnly, TypedDict

    class _ReadOnlyItemInCall(TypedDict):
        """
        A TypedDict with a read-only item, as some of the response types of litellm,
        for which Pydantic warns when it builds its schema: the schema built within
        a call of litellm.
        """

        summary: ReadOnly[str]

    class _ReadOnlyItemOutsideCall(TypedDict):
        """
        The same TypedDict as ``_ReadOnlyItemInCall``, for the schema built outside
        a call of litellm: a schema is built for each of them once.
        """

        summary: ReadOnly[str]


@pytest.mark.usefixtures("restore_logger_settings")
class TestLogging(BaseTest):
    """
    Tests for the logger settings and log levels.

    Each test changes the logger settings, which are restored after the test.
    """

    @pytest.mark.skipif(
        sys.version_info < (3, 13), reason="`typing.ReadOnly` requires Python 3.13+"
    )
    def test_litellm_warnings_suppressed_in_calls(self):
        """
        Tests that the warning that Pydantic issues for a read-only item of
        a TypedDict, as when it builds the schemas of the response types of litellm,
        which users cannot act on, is suppressed within a call of litellm, while
        it is not suppressed outside such a call, and other warnings are not
        suppressed within it.
        """

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with _suppress_litellm_warnings_context():
                TypeAdapter(_ReadOnlyItemInCall)
                warnings.warn("Another warning", UserWarning, stacklevel=1)
        assert [str(w.message) for w in caught_warnings] == ["Another warning"]

        with pytest.warns(UserWarning, match="is using the `ReadOnly` qualifier"):
            TypeAdapter(_ReadOnlyItemOutsideCall)

    def test_logger_disabled(self, monkeypatch, capsys):
        """
        Tests for disabling the logger.
        """

        # 1) Set environment variable to disable logger
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "OFF")
        reload_logger_settings()

        # 2) Attempt to log a message
        logger.debug("This message should NOT appear.")

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Assert that the message is indeed missing
        assert "This message should NOT appear." not in captured.out

    def test_logger_enabled(self, monkeypatch, capsys):
        """
        Tests for enabling the logger.
        """

        # 1) Set environment variable to enable logger
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "DEBUG")
        reload_logger_settings()

        # 2) Log a message
        logger.debug("This message SHOULD appear.")

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Assert that the message is in the output
        assert "This message SHOULD appear." in captured.out

    @pytest.mark.parametrize(
        "env_level,should_debug,should_info,should_success,should_warning,should_error,should_critical",
        [
            ("DEBUG", True, True, True, True, True, True),
            ("INFO", False, True, True, True, True, True),
            ("SUCCESS", False, False, True, True, True, True),
            ("WARNING", False, False, False, True, True, True),
            ("ERROR", False, False, False, False, True, True),
            ("CRITICAL", False, False, False, False, False, True),
        ],
    )
    def test_log_levels(
        self,
        monkeypatch,
        capsys,
        env_level,
        should_debug,
        should_info,
        should_success,
        should_warning,
        should_error,
        should_critical,
    ):
        """
        Tests for outputting log messages of different log levels.

        For each env_level, verifies which messages appear in stdout.
        We test debug, info, success, warning, error, critical logging calls.

        Also handy for debugging format changes in the logger's output,
        e.g. overall design, icon width, etc.
        """

        # 1) Set the log level.
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, env_level)
        reload_logger_settings()

        # 2) Emit one message per level:
        def log_messages():
            """
            Helper function to emit log messages at all levels for testing.
            """
            logger.debug("DEBUG message")
            logger.info("INFO message")
            logger.success("SUCCESS message")
            logger.warning("WARNING message")
            logger.error("ERROR message")
            logger.critical("CRITICAL message")

        log_messages()

        # 3) Capture output
        captured = capsys.readouterr()

        # 4) Check presence/absence of each message
        debug_present = "DEBUG message" in captured.out
        info_present = "INFO message" in captured.out
        success_present = "SUCCESS message" in captured.out
        warning_present = "WARNING message" in captured.out
        error_present = "ERROR message" in captured.out
        critical_present = "CRITICAL message" in captured.out

        # 5) Compare actual results with expected booleans
        assert debug_present == should_debug, f"DEBUG unexpected for level {env_level}"
        assert info_present == should_info, f"INFO unexpected for level {env_level}"
        assert success_present == should_success, (
            f"SUCCESS unexpected for level {env_level}"
        )
        assert warning_present == should_warning, (
            f"WARNING unexpected for level {env_level}"
        )
        assert error_present == should_error, f"ERROR unexpected for level {env_level}"
        assert critical_present == should_critical, (
            f"CRITICAL unexpected for level {env_level}"
        )

        # 6) After the last check, output all messages for visual formatting check
        if env_level == "CRITICAL":
            monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "DEBUG")
            reload_logger_settings()
            print()
            log_messages()

    @pytest.mark.parametrize(
        ("first_level", "second_level"),
        [
            # The logger is disabled first, i.e. when the package is imported
            ("OFF", "INFO"),
            # ... or after it was enabled
            ("INFO", "OFF"),
            ("OFF", "OFF"),
        ],
    )
    def test_litellm_noise_filter_installed_at_any_level(
        self, monkeypatch, first_level, second_level
    ):
        """
        Tests that the filter of the noisy logs of litellm is installed whatever
        the level of the logger is, including when the logger is disabled, and
        in whichever order the levels are set: with the logger disabled, the logs
        of litellm would otherwise be shown, while they are hidden at other levels.
        """
        monkeypatch.delenv(_SUPPRESS_LITELLM_NOISE_ENV_VAR_NAME, raising=False)
        loggers = [
            logging.getLogger(name) for name in _LITELLM_NOISE_FILTER_LOGGER_NAMES
        ]
        filters_before_test = [list(lg.filters) for lg in loggers]

        def get_noise_filters(lg: logging.Logger) -> list[logging.Filter]:
            """
            Returns the filters of the noisy logs of litellm that a logger has.

            :param lg: The logger.
            :type lg: logging.Logger
            :return: The filters.
            :rtype: list[logging.Filter]
            """
            return [f for f in lg.filters if isinstance(f, _SuppressLiteLLMNoiseFilter)]

        def create_record(name: str, message: str) -> logging.LogRecord:
            """
            Creates a warning of a logger.

            :param name: The name of the logger.
            :type name: str
            :param message: The message of the warning.
            :type message: str
            :return: The log record.
            :rtype: logging.LogRecord
            """
            return logging.LogRecord(
                name, logging.WARNING, __file__, 0, message, None, None
            )

        try:
            # As when the package is imported: no filter is installed yet
            for lg in loggers:
                for noise_filter in get_noise_filters(lg):
                    lg.removeFilter(noise_filter)
                assert not get_noise_filters(lg)

            for level in (first_level, second_level):
                monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, level)
                reload_logger_settings()
                # Installed once on each logger, also when the settings are
                # reloaded
                for lg in loggers:
                    assert len(get_noise_filters(lg)) == 1
                # The warning that litellm logs when its async calls are made from
                # another event loop, as each call of a sync method does, is hidden
                litellm_logger = logging.getLogger("LiteLLM")
                assert not litellm_logger.filter(
                    create_record(
                        "LiteLLM",
                        "LoggingWorker: event loop changed; carried 1 pending and "
                        "revived 0 dequeued logging task(s) onto the new loop",
                    )
                )
                # ... while the other logs of litellm are not
                assert litellm_logger.filter(
                    create_record("LiteLLM", "Request failed with status 500")
                )
        finally:
            for lg, filters in zip(loggers, filters_before_test, strict=True):
                lg.filters[:] = filters

    @pytest.mark.parametrize("level", ["OFF", "INFO"])
    def test_litellm_noise_filter_not_installed_when_disabled(self, monkeypatch, level):
        """
        Tests that the filter of the noisy logs of litellm is not installed if it
        is disabled with its environment variable, whatever the level of
        the logger is: the variable, and not the level, decides on the filter.
        """
        monkeypatch.setenv(_SUPPRESS_LITELLM_NOISE_ENV_VAR_NAME, "0")
        loggers = [
            logging.getLogger(name) for name in _LITELLM_NOISE_FILTER_LOGGER_NAMES
        ]
        filters_before_test = [list(lg.filters) for lg in loggers]
        try:
            # As when the package is imported: no filter is installed yet
            for lg in loggers:
                lg.filters[:] = [
                    f
                    for f in lg.filters
                    if not isinstance(f, _SuppressLiteLLMNoiseFilter)
                ]

            monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, level)
            reload_logger_settings()

            for lg in loggers:
                assert not any(
                    isinstance(f, _SuppressLiteLLMNoiseFilter) for f in lg.filters
                )
        finally:
            for lg, filters in zip(loggers, filters_before_test, strict=True):
                lg.filters[:] = filters

    @pytest.mark.parametrize("level", ["OFF", "INFO"])
    def test_litellm_noise_filter_follows_its_variable_on_reload(
        self, monkeypatch, level
    ):
        """
        Tests that the filter of the noisy logs of litellm follows its environment
        variable when the logger settings are reloaded, whatever the level of
        the logger is: a filter that is installed is removed once the variable
        disables it, and is installed again once the variable no longer does.
        The other filters of the same loggers are left as they are.
        """
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, level)
        loggers = [
            logging.getLogger(name) for name in _LITELLM_NOISE_FILTER_LOGGER_NAMES
        ]
        filters_before_test = [list(lg.filters) for lg in loggers]
        # A filter of another kind, e.g. one that the application has added
        other_filter = logging.Filter()
        try:
            for lg in loggers:
                lg.addFilter(other_filter)

            # The value of the variable (None if it is not set), and whether
            # the filter is installed after the settings are reloaded with it
            for value, is_installed in [
                (None, True),
                # Disabled after it was installed: removed
                ("0", False),
                # ... and not installed when the settings are reloaded again
                ("0", False),
                # No longer disabled: installed again
                ("1", True),
                ("0", False),
                (None, True),
                # Reloading the settings does not install it twice
                (None, True),
            ]:
                if value is None:
                    monkeypatch.delenv(
                        _SUPPRESS_LITELLM_NOISE_ENV_VAR_NAME, raising=False
                    )
                else:
                    monkeypatch.setenv(_SUPPRESS_LITELLM_NOISE_ENV_VAR_NAME, value)
                reload_logger_settings()
                for lg in loggers:
                    n_noise_filters = sum(
                        isinstance(f, _SuppressLiteLLMNoiseFilter) for f in lg.filters
                    )
                    assert n_noise_filters == (1 if is_installed else 0)
                    assert other_filter in lg.filters
        finally:
            for lg, filters in zip(loggers, filters_before_test, strict=True):
                lg.filters[:] = filters
