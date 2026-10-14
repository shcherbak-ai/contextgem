# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Pytest configuration and fixtures for ContextGem tests.

This module provides test configuration, fixtures, and utilities
for running ContextGem tests, including VCR cassette handling.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import patch
from urllib.parse import urlsplit

import pytest
import tethered
from vcr.stubs import httpx_stubs

from contextgem.internal.loggers import LOGGER_LEVEL_ENV_VAR_NAME, logger
from contextgem.internal.suppressions import _suppress_litellm_warnings_context
from contextgem.public.utils import reload_logger_settings


with _suppress_litellm_warnings_context():
    import litellm

# The thread that the VCR-marked tests send the sync calls of litellm from (see
# `vcr_compatible_acompletion`). VCR runs such a call with `asyncio.run()`,
# which cannot be done in a thread with a running event loop. A single thread
# sends the calls one at a time, in the order they are made.
_VCR_SYNC_CALLS = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vcr-sync")


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


# Hosts that SaT models are downloaded from (HuggingFace)
_SAT_MODEL_DOWNLOAD_HOSTS = ("huggingface.co", "*.huggingface.co", "hf.co", "*.hf.co")

# Host that genai-prices refreshes its pricing data from
_GENAI_PRICES_REFRESH_HOSTS = ("raw.githubusercontent.com",)

# Approved network destinations for tethered egress control.
# During replay: only HuggingFace (SaT model downloads not captured by VCR).
# During recording: LLM API endpoints + HuggingFace + genai-prices refresh.
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


async def _call_in_vcr_thread(func: Callable[..., Any], *args, **kwargs) -> Any:
    """
    Runs a sync call of litellm in the thread for such calls, and returns its
    result. The event loop that awaits the call keeps running meanwhile.

    :param func: The sync function of litellm to call
    :param args: Arguments to pass to the function
    :param kwargs: Keyword arguments to pass to the function
    :return: The result of the call
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    return await loop.run_in_executor(
        _VCR_SYNC_CALLS, functools.partial(context.run, func, *args, **kwargs)
    )


@pytest.fixture(autouse=True)
def vcr_compatible_acompletion(request):
    """
    Automatically monkey-patches acompletion for VCR-marked tests.

    This works around the issue where acompletion() is not supported in
    VCR recording/replay after transport change in litellm>1.71.1.

    The patch is applied for both recording AND replay modes because
    VCR.py cannot intercept httpx async requests made by litellm's acompletion.

    The sync litellm.completion() is called in a thread of its own (see
    `_call_in_vcr_thread`), as VCR runs it with `asyncio.run()`, which cannot
    be done in the thread of the running event loop.

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

            :param args: Arguments to pass to litellm.completion
            :param kwargs: Keyword arguments to pass to litellm.completion
            :return: Completion response from litellm
            """
            logger.debug("Using monkey-patched acompletion()")
            return await _call_in_vcr_thread(litellm.completion, *args, **kwargs)

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
            return await _call_in_vcr_thread(litellm.decisions, *args, **kwargs)

        with (
            patch.object(litellm, "acompletion", vcr_compatible_acompletion_fn),
            patch.object(litellm, "adecisions", vcr_compatible_adecisions_fn),
        ):
            yield
    else:
        yield
