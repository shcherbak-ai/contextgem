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

This module provides test configuration, fixtures, and utilities
for running ContextGem tests, including VCR cassette handling.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch
from urllib.parse import urlsplit

import nest_asyncio
import pytest
import tethered
from vcr.stubs import httpx_stubs

from contextgem.internal.base.models import _GENAI_PRICES_REFRESH_HOSTS
from contextgem.internal.loggers import LOGGER_LEVEL_ENV_VAR_NAME, logger
from contextgem.internal.suppressions import _suppress_litellm_warnings_context
from contextgem.internal.utils import _SAT_MODEL_DOWNLOAD_HOSTS
from contextgem.public.utils import reload_logger_settings


# Apply nest_asyncio to allow nested event loops
# This allows the sync litellm.completion() call in the acompletion patch below
# to run its internal event loop while already inside a running loop
nest_asyncio.apply()

with _suppress_litellm_warnings_context():
    import litellm


# Patch VCR's _deserialize_response to handle string bodies correctly
# This fixes a bug where cassette bodies stored as strings cause TypeErrors
_original_deserialize_response = httpx_stubs._deserialize_response


def _patched_deserialize_response(vcr_response, httpx):
    """
    Patched version of VCR's _deserialize_response that ensures body is bytes.

    VCR stores response bodies as strings in YAML cassettes for readability,
    but ``httpx.ByteStream`` expects bytes. This patch converts string bodies
    to bytes before creating the Response object.
    """
    # Ensure body is bytes, not string
    body = vcr_response.get("body", {}).get("string", b"")
    if isinstance(body, str):
        vcr_response["body"]["string"] = body.encode("utf-8")

    return _original_deserialize_response(vcr_response, httpx)


# Apply the patch
httpx_stubs._deserialize_response = _patched_deserialize_response  # ty: ignore[invalid-assignment]


# VCR redaction of API keys, private endpoints, etc.
VCR_REDACTION_MARKER = "DUMMY"
VCR_DUMMY_ENDPOINT_PREFIX = "https://dummy-endpoint.local/"


@pytest.fixture(scope="module")
def vcr_config():
    """
    Pytest fixture providing VCR configuration for recording and replaying HTTP interactions.

    This fixture configures VCR to record HTTP requests and responses during test execution,
    allowing tests to replay recorded interactions without making actual network calls.

    :return: Dictionary containing VCR configuration options including header filtering,
             request/response processing hooks, matching criteria, and cassette settings.
    :rtype: dict
    """
    # Imported here, as `tests.utils` imports from this module
    from tests.utils import (
        VCR_FILTER_HEADERS,
        vcr_before_record_request,
        vcr_before_record_response,
    )

    return {
        "filter_headers": [(i, VCR_REDACTION_MARKER) for i in VCR_FILTER_HEADERS],
        "before_record_request": vcr_before_record_request,
        "before_record_response": vcr_before_record_response,
        "match_on": ["method", "host", "path", "body"],
        "record_mode": "once",
        "cassette_library_dir": "tests/cassettes",
        "ignore_localhost": False,
        "ignore_hosts": ["huggingface.co", "hf.co"],
    }


def pytest_configure(config):
    """
    Configure pytest with custom settings.
    """
    # Set contextgem logger level to DEBUG
    os.environ[LOGGER_LEVEL_ENV_VAR_NAME] = "DEBUG"
    reload_logger_settings()


@pytest.fixture
def restore_logger_settings():
    """
    Restores the logger settings after a test that changes them.

    The logger level is restored to its value before the test, and the logger
    is reconfigured, which also binds it to the current output stream.

    Apply the fixture with ``@pytest.mark.usefixtures``: it is then set up before
    the fixtures requested as the arguments of the test, and finalized after them.
    A test that uses ``capsys`` leaves the logger bound to a stream that is closed
    when ``capsys`` is finalized, so the logger has to be reconfigured after that.
    """
    previous_level = os.environ.get(LOGGER_LEVEL_ENV_VAR_NAME)
    yield
    if previous_level is None:
        os.environ.pop(LOGGER_LEVEL_ENV_VAR_NAME, None)
    else:
        os.environ[LOGGER_LEVEL_ENV_VAR_NAME] = previous_level
    reload_logger_settings()


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


# Approved network destinations for tethered egress control.
# During replay: only HuggingFace (SaT model downloads not captured by VCR).
# During recording: LLM API endpoints + HuggingFace + genai-prices refresh.
# Host constants are sourced from production code so the test allowlist
# cannot drift from the per-call `tethered.scope` allowlists.
_TETHERED_REPLAY_ALLOW = list(_SAT_MODEL_DOWNLOAD_HOSTS)

_TETHERED_RECORDING_ALLOW = [
    "api.openai.com",  # OpenAI
    "api.typesafe.ai",  # TypeSafe (decision models)
    *_GENAI_PRICES_REFRESH_HOSTS,  # genai-prices auto-refresh
    *_TETHERED_REPLAY_ALLOW,
]

# Environment variable with the endpoint of the Microsoft Foundry resource that
# the decision model of the tests is deployed in, and the domain of such endpoints
_FOUNDRY_API_BASE_ENV_VAR_NAME = "CONTEXTGEM_AZURE_AI_API_BASE"
_FOUNDRY_RESOURCE_DOMAIN = ".services.ai.azure.com"


def _get_tethered_recording_allow() -> list[str]:
    """
    Returns the approved network destinations for recording, including the host
    of the Microsoft Foundry resource that the decision model of the tests is
    deployed in, if its endpoint is set in the environment.

    The host of a resource has the name of the resource, i.e. it differs for
    each contributor, so it is read from the environment and not listed above.
    It is read when a test starts: the ``.env`` file is loaded after this module.
    It is approved only if it is the host of a Microsoft Foundry resource, in
    an HTTPS endpoint, i.e. the variable cannot open the network to a host of
    another kind, and the key of the resource is not sent unencrypted.

    :return: The approved network destinations.
    :rtype: list[str]
    """
    allow = list(_TETHERED_RECORDING_ALLOW)
    try:
        endpoint = urlsplit(os.getenv(_FOUNDRY_API_BASE_ENV_VAR_NAME, ""))
        foundry_host = endpoint.hostname if endpoint.scheme == "https" else None
    except ValueError:
        # Not a URL
        foundry_host = None
    if foundry_host and foundry_host.lower().endswith(_FOUNDRY_RESOURCE_DOMAIN):
        allow.append(foundry_host.lower())
    return allow


# Sentinel for tethered's locked mode — required to deactivate a locked policy.
# Held privately in this module so test code cannot trip the policy off.
_TETHERED_LOCK = object()

# Host that tiktoken downloads the token encodings from. Deliberately not a part
# of the allowlists above: it is reachable only while the token encodings are
# prepared before the tests start (see `_warm_token_encodings`).
_TOKEN_ENCODINGS_DOWNLOAD_HOST = "openaipublic.blob.core.windows.net"  # nosec B105 - not a password


@pytest.fixture(scope="session", autouse=True)
def _warm_token_encodings():
    """
    Counts tokens once before the tests start, so that the token encodings
    used by litellm are available during the tests.

    The Windows build of litellm bundles the token encoding files with Windows
    line endings. tiktoken rejects such files (hash mismatch) and downloads them
    again on the first token count. During a test, this download is blocked (by VCR
    in replay mode, by tethered otherwise), in which case tokens cannot be counted
    and the tests that rely on token counting fail.

    Here, the download happens before VCR and tethered are activated for a test,
    with the egress limited to the single host it needs, and the encodings are
    cached in the virtual environment. tiktoken verifies each download against
    the hash pinned in its package. No network call is made if the encodings are
    valid, i.e. on Linux and macOS, or once they are cached.
    """
    try:
        with tethered.scope(
            allow=[_TOKEN_ENCODINGS_DOWNLOAD_HOST],
            label="tests._warm_token_encodings",
        ):
            litellm.token_counter(model="gpt-6-luna", text="warm-up")
    except Exception as e:
        logger.warning(
            f"Could not prepare the token encodings before the tests: {e}. "
            "The tests that rely on token counting may fail. "
            "See the Windows note in CONTRIBUTING.md (Running Tests)."
        )


@pytest.fixture(autouse=True)
def _tethered_network_guard(request):
    """
    Enforces network egress control for all tests using tethered.

    - **VCR replay** (cassette exists): allows only HuggingFace
      (SaT model downloads not captured by VCR).
    - **VCR recording** (no cassette): allows approved endpoints
      (LLM APIs, HuggingFace for model downloads, genai-prices for cost data),
      the Microsoft Foundry resource whose endpoint is set in the environment,
      and localhost for local LLMs.
    - **Non-VCR tests**: blocks all network calls entirely.

    Uses tethered's hardened config: ``locked=True`` makes the policy
    tamper-resistant within the test, and ``external_subprocess_policy="block"``
    refuses non-Python subprocess launches (closing shell-based egress bypass).
    """
    is_vcr = request.node.get_closest_marker("vcr")
    if is_vcr:
        is_replay = _cassette_exists(_get_cassette_path(request))
        tethered.activate(
            allow=(
                _TETHERED_REPLAY_ALLOW if is_replay else _get_tethered_recording_allow()
            ),
            allow_localhost=not is_replay,  # localhost for local LLMs during recording
            locked=True,
            lock_token=_TETHERED_LOCK,
            external_subprocess_policy="block",
        )
    else:
        tethered.activate(
            allow=[],
            allow_localhost=False,
            locked=True,
            lock_token=_TETHERED_LOCK,
            external_subprocess_policy="block",
        )
    yield
    tethered.deactivate(lock_token=_TETHERED_LOCK)


def _run_tasks_left_on_event_loop() -> None:
    """
    Runs the tasks that are left pending on the event loop of the thread, if any.

    With nest_asyncio applied, ``asyncio.run()`` neither cancels the tasks that
    a coroutine leaves pending nor closes the event loop, which the next call
    uses again. The sync ``litellm.completion()``, which the VCR-marked tests
    route ``acompletion()`` to, reports the timing of the call in a task of
    the running loop. The task of the last call of a test stays pending until
    the next test runs the loop, or, if no test does, e.g. when the test is run
    on its own, until the interpreter exits, which then prints
    "Task was destroyed but it is pending!".
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in the thread, i.e. nothing was run
        return
    if loop.is_closed() or loop.is_running():
        return
    if asyncio.all_tasks(loop):
        loop.run_until_complete(asyncio.sleep(0))


@pytest.fixture(autouse=True)
def vcr_compatible_acompletion(request):
    """
    Automatically monkey-patches acompletion for VCR-marked tests.

    This works around the issue where acompletion() is not supported in
    VCR recording/replay after transport change in litellm>1.71.1.

    The patch is applied for both recording AND replay modes because
    VCR.py cannot intercept httpx async requests made by litellm's acompletion.

    With nest_asyncio applied at module level, we can safely call sync
    litellm.completion() from async context.

    The same applies to adecisions(), the async call of the Decisions API that
    decision models are requested through: it is routed to the sync
    litellm.decisions(), which has the same signature.

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

        async def vcr_compatible_adecisions_fn(*args, **kwargs):
            """
            VCR-compatible wrapper for litellm.adecisions().

            Uses sync litellm.decisions() which VCR can properly intercept:
            the async call is sent through a transport that VCR cannot record.

            :param args: Arguments to pass to litellm.decisions
            :param kwargs: Keyword arguments to pass to litellm.decisions
            :return: Decisions response from litellm
            """
            logger.debug("Using monkey-patched adecisions()")
            return litellm.decisions(*args, **kwargs)

        with (
            patch.object(litellm, "acompletion", vcr_compatible_acompletion_fn),
            patch.object(litellm, "adecisions", vcr_compatible_adecisions_fn),
        ):
            yield
        # The sync calls leave tasks of litellm on the event loop
        _run_tasks_left_on_event_loop()
    else:
        yield
