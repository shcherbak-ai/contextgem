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
Pytest configuration and fixtures for ContextGem tests.

This module provides test configuration, command-line options, fixtures,
and utilities for running ContextGem tests, including VCR cassette handling
and memory profiling support.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import nest_asyncio
import pytest
from vcr.stubs import httpcore_stubs

from contextgem.internal.loggers import logger
from contextgem.internal.suppressions import _suppress_litellm_warnings_context
from contextgem.public.utils import reload_logger_settings


# Apply nest_asyncio to allow nested event loops
# This fixes VCR's async handling issues where asyncio.run() is called
# from sync code while an event loop is already running
nest_asyncio.apply()

with _suppress_litellm_warnings_context():
    import litellm


# Patch VCR's _deserialize_response to handle string bodies correctly
# This fixes a bug where cassette bodies stored as strings cause TypeErrors
_original_deserialize_response = httpcore_stubs._deserialize_response


def _patched_deserialize_response(vcr_response):
    """
    Patched version of VCR's _deserialize_response that ensures body is bytes.

    VCR stores response bodies as strings in YAML cassettes for readability,
    but httpcore's Response expects bytes. This patch converts string bodies
    to bytes before creating the Response object.
    """
    # Ensure body is bytes, not string
    body = vcr_response.get("body", {}).get("string", b"")
    if isinstance(body, str):
        vcr_response["body"]["string"] = body.encode("utf-8")

    return _original_deserialize_response(vcr_response)


# Apply the patch
httpcore_stubs._deserialize_response = _patched_deserialize_response


# Patch VCR's _run_async_function to fix a bug in httpcore stubs.
# When called from inside a running event loop, VCR's original implementation
# returns asyncio.ensure_future() (a Future object) instead of the actual result.
# With nest_asyncio applied, we can safely use asyncio.run() in all cases.
def _patched_run_async_function(async_func, *args, **kwargs):
    """
    Run async function using asyncio.run(), which works with nest_asyncio.
    """
    return asyncio.run(async_func(*args, **kwargs))


httpcore_stubs._run_async_function = _patched_run_async_function  # type: ignore[attr-defined]


# Memory profiling behavior
_ENABLE_MEMORY_PROFILING_FLAG = "--mem-profile"
_MEMORY_PROFILING_ENABLED = False

# VCR redaction of API keys, private endpoints, etc.
VCR_REDACTION_MARKER = "DUMMY"
VCR_DUMMY_ENDPOINT_PREFIX = "https://dummy-endpoint.local/"


def pytest_addoption(parser):
    """
    Add custom command line options to pytest.
    """
    parser.addoption(
        _ENABLE_MEMORY_PROFILING_FLAG,
        action="store_true",
        default=False,
        help="Enable memory profiling during test execution",
    )


def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    global _MEMORY_PROFILING_ENABLED

    # Set contextgem logger level to DEBUG
    os.environ["CONTEXTGEM_LOGGER_LEVEL"] = "DEBUG"
    reload_logger_settings()

    # Set the global memory profiling flag
    _MEMORY_PROFILING_ENABLED = config.getoption(_ENABLE_MEMORY_PROFILING_FLAG)


def is_memory_profiling_enabled() -> bool:
    """
    Check if memory profiling has been enabled via command line flag.

    :return: True if memory profiling is enabled, False otherwise
    :rtype: bool
    """
    return _MEMORY_PROFILING_ENABLED


def _get_cassette_path(request) -> str:
    """
    Generates the expected cassette file path for a test.

    :param request: pytest request object
    :return: Expected cassette file path
    :rtype: str
    """
    # Get test class and method name
    test_class = request.cls.__name__ if request.cls else ""
    test_method = request.node.name

    # Build cassette name following pytest-vcr convention
    if test_class:
        cassette_name = f"{test_class}.{test_method}.yaml"
    else:
        cassette_name = f"{test_method}.yaml"

    # Return full path to cassette
    return os.path.join("tests", "cassettes", cassette_name)


def _cassette_exists(cassette_path: str) -> bool:
    """
    Checks if a cassette file exists.

    :param cassette_path: Path to the cassette file
    :return: True if cassette exists, False otherwise
    :rtype: bool
    """
    return os.path.exists(cassette_path)


@pytest.fixture(autouse=True)
def vcr_compatible_acompletion(request):
    """
    Automatically monkey-patches acompletion for VCR-marked tests.

    This works around the issue where acompletion() is not supported in
    VCR recording/replay after transport change in litellm>1.71.1.

    The patch is applied for both recording AND replay modes because
    VCR.py cannot intercept httpx async requests made by litellm's acompletion.

    With nest_asyncio applied at module level, we can safely call sync
    litellm.completion() from async context. VCR's _run_async_function is
    also patched to use asyncio.run() which works with nested loops.

    TODO: Remove this when vcr supports acompletion() with litellm's httpx transport
    """
    # Check if the test is marked with @pytest.mark.vcr
    if request.node.get_closest_marker("vcr"):
        cassette_path = _get_cassette_path(request)

        # Determine mode for logging
        if not _cassette_exists(cassette_path):
            logger.debug(
                f"Recording mode detected for {cassette_path}, applying acompletion patch"
            )
        else:
            logger.debug(
                f"Replay mode detected for {cassette_path}, applying acompletion patch"
            )

        async def vcr_compatible_acompletion_fn(*args, **kwargs):
            """
            VCR-compatible wrapper for litellm.acompletion().

            Uses sync litellm.completion() which VCR can properly intercept.
            With nest_asyncio applied, VCR's internal async handling works correctly
            even when called from nested event loops.

            :param args: Arguments to pass to litellm.completion
            :param kwargs: Keyword arguments to pass to litellm.completion
            :return: Completion response from litellm
            """
            logger.debug("Using monkey-patched acompletion()")
            # Call sync completion directly - nest_asyncio handles nested loops
            # and VCR's patched _run_async_function uses asyncio.run() safely
            return litellm.completion(*args, **kwargs)

        with patch.object(litellm, "acompletion", vcr_compatible_acompletion_fn):
            yield
    else:
        yield
