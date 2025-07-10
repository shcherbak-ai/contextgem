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

import asyncio
import os
from unittest.mock import patch

import pytest

from contextgem.internal.utils import _suppress_litellm_pydantic_warnings_context

with _suppress_litellm_pydantic_warnings_context():
    import litellm

from contextgem.internal.loggers import logger
from contextgem.public.utils import reload_logger_settings

# Memory profiling behavior
_ENABLE_MEMORY_PROFILING_FLAG = "--mem-profile"
_MEMORY_PROFILING_ENABLED = False

# VCR redaction of API keys, private endpoints, etc.
VCR_REDACTION_MARKER = "DUMMY"
VCR_DUMMY_ENDPOINT_PREFIX = "https://<DUMMY-ENDPOINT>/"


def pytest_addoption(parser):
    """
    Add custom command line options to pytest.
    """
    parser.addoption(
        _ENABLE_MEMORY_PROFILING_FLAG,
        action="store_true",
        default=False,
        help="Disable memory profiling during test execution",
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
    Automatically monkey-patches acompletion for VCR-marked tests during recording.

    This works around the issue where acompletion() is not supported in
    VCR recording after transport change in litellm>1.71.1

    The patch is only applied when recording (no cassette exists).
    During replay (cassette exists), the original acompletion() is used.

    TODO: Remove this when vcr recording supports acompletion()
    """
    # Check if the test is marked with @pytest.mark.vcr
    if request.node.get_closest_marker("vcr"):
        cassette_path = _get_cassette_path(request)

        # Only patch if cassette doesn't exist (recording mode)
        if not _cassette_exists(cassette_path):
            logger.debug(
                f"Recording mode detected for {cassette_path}, applying acompletion patch"
            )

            async def vcr_compatible_acompletion(*args, **kwargs):
                logger.debug("Using monkey-patched acompletion()")
                return await asyncio.to_thread(litellm.completion, *args, **kwargs)

            with patch.object(litellm, "acompletion", vcr_compatible_acompletion):
                yield
        else:
            logger.debug(
                f"Replay mode detected for {cassette_path}, using original acompletion"
            )
            yield
    else:
        yield
