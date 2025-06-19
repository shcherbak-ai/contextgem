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

import os

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
