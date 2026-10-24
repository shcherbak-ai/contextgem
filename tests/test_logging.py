# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the logger.
"""

from __future__ import annotations

import io
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

    @pytest.mark.parametrize(
        "env_level,should_debug,should_info,should_success,should_warning,should_error,should_critical",
        [
            ("DEBUG", True, True, True, True, True, True),
            ("INFO", False, True, True, True, True, True),
            ("SUCCESS", False, False, True, True, True, True),
            ("WARNING", False, False, False, True, True, True),
            ("ERROR", False, False, False, False, True, True),
            ("CRITICAL", False, False, False, False, False, True),
            # The logger is disabled
            ("OFF", False, False, False, False, False, False),
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

    def test_colors_only_in_a_terminal(self, monkeypatch):
        """
        Tests that the log has the escape codes of its colors only where it is
        written to a terminal, e.g. not to a file or to the log of a CI job, unless
        the colors are forced with ``FORCE_COLOR``, and never with ``NO_COLOR``.
        """

        class _TerminalStream(io.StringIO):
            """
            A stream that is taken for a terminal.
            """

            def isatty(self) -> bool:
                return True

        escape_code = "\x1b["
        monkeypatch.delenv("FORCE_COLOR", raising=False)
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, "INFO")
        for stream_class, env_var, is_colored in [
            (io.StringIO, None, False),
            (io.StringIO, "FORCE_COLOR", True),
            (_TerminalStream, None, True),
            (_TerminalStream, "NO_COLOR", False),
        ]:
            stream = stream_class()
            monkeypatch.setattr(sys, "stdout", stream)
            if env_var is not None:
                monkeypatch.setenv(env_var, "1")
            reload_logger_settings()
            logger.warning("Colored message")
            assert "Colored message" in stream.getvalue()
            assert (escape_code in stream.getvalue()) == is_colored
            if env_var is not None:
                monkeypatch.delenv(env_var)

    def test_application_handler_kept_at_any_level(self, monkeypatch, capsys):
        """
        Tests that a handler that the application added to the logger is kept when
        the settings are reloaded, also with the logger disabled: it gets no messages
        while the logger is disabled, and gets them again once it is enabled. Only
        the handler of ContextGem is replaced, i.e. it is not added twice.
        """

        class _ListHandler(logging.Handler):
            """
            A handler that keeps the messages of the records it gets.
            """

            def __init__(self) -> None:
                super().__init__()
                self.messages: list[str] = []

            def emit(self, record: logging.LogRecord) -> None:
                self.messages.append(record.getMessage())

        app_handler = _ListHandler()
        logger.addHandler(app_handler)
        try:
            for level, is_received in [
                ("INFO", True),
                ("OFF", False),
                ("INFO", True),
                ("WARNING", True),
            ]:
                monkeypatch.setenv(LOGGER_LEVEL_ENV_VAR_NAME, level)
                reload_logger_settings()
                assert app_handler in logger.handlers
                n_stream_handlers = sum(
                    type(handler) is logging.StreamHandler
                    for handler in logger.handlers
                )
                assert n_stream_handlers == (0 if level == "OFF" else 1)

                app_handler.messages.clear()
                logger.warning(f"Message at {level}")
                assert app_handler.messages == (
                    [f"Message at {level}"] if is_received else []
                )
                assert (f"Message at {level}" in capsys.readouterr().out) == (
                    is_received
                )
        finally:
            logger.removeHandler(app_handler)

    @pytest.mark.parametrize(
        ("first_level", "second_level"),
        [
            # The logger is disabled first, i.e. when the package is imported
            ("OFF", "INFO"),
            # ... or after it was enabled
            ("INFO", "OFF"),
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
    def test_litellm_noise_filter_follows_its_variable_on_reload(
        self, monkeypatch, level
    ):
        """
        Tests that the filter of the noisy logs of litellm follows its environment
        variable when the logger settings are reloaded, whatever the level of
        the logger is: the variable, and not the level, decides on the filter.
        A filter that is installed is removed once the variable disables it, is
        not installed while it does, and is installed again once it no longer does.
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
