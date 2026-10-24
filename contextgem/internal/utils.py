# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining internal utility functions of the framework.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import json
import math
import os
import random
import re
import threading
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable, Coroutine, Sequence
from functools import cache
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, get_args

import pydantic
from jinja2 import Environment, Template, nodes

from contextgem.internal.data_models import _ModelCall, _ModelUsage
from contextgem.internal.exceptions import ContextGemWarning
from contextgem.internal.llm_output_structs.aspect_structs import (
    _get_aspect_extraction_output_struct,
)
from contextgem.internal.llm_output_structs.concept_structs import (
    _get_concept_extraction_output_struct,
)
from contextgem.internal.loggers import logger
from contextgem.internal.suppressions import _suppress_litellm_warnings
from contextgem.internal.typings.types import (
    AsyncCalsAndKwargs,
    ExtractedInstanceType,
    ModelRefusal,
    ReferenceDepth,
    SaTModelId,
    StandardSaTModelId,
)


if TYPE_CHECKING:
    from wtpsplit_lite import SaT

    from contextgem.internal.base.aspects import _Aspect
    from contextgem.internal.base.concepts import _Concept
    from contextgem.internal.base.paras_and_sents import _Paragraph


T = TypeVar("T")
# An aspect or a concept, i.e. an instance that items are extracted for
InstanceT = TypeVar("InstanceT", bound="_Aspect | _Concept")


# Directories of the code that the checks of an instance run in when it is validated:
# the framework, and Pydantic, which calls the validators of the framework.
# A warning issued by such a check is attributed to the first frame outside them.
_FRAMEWORK_AND_PYDANTIC_DIRS: tuple[str, ...] = tuple(
    os.path.normcase(os.path.abspath(package_dir)) + os.sep
    for package_dir in (
        os.path.dirname(os.path.dirname(__file__)),  # the package of the framework
        os.path.dirname(pydantic.__file__),
    )
)
# Paths of the code of the standard library that the coroutines of the framework
# run under: the event loop, and the thread that a sync method runs its event loop
# in when the calling code runs one already. A warning issued while a model is
# called, e.g. on its prices, is attributed to the first frame outside them and
# the directories above, i.e. to the line of the calling code that called
# the method of the framework, where the thread has such a line.
_EVENT_LOOP_PATHS: tuple[str, ...] = (
    *(
        os.path.normcase(os.path.abspath(os.path.dirname(module_file))) + os.sep
        for module_file in (asyncio.__file__, concurrent.futures.__file__)
    ),
    os.path.normcase(os.path.abspath(threading.__file__)),
)

# SaT models in use, by their keys (see `_get_sat_model()`): a model is here only
# while something else holds it, e.g. a segmentation in a thread, or the entry
# below
_SAT_MODELS_IN_USE: weakref.WeakValueDictionary[str, SaT] = (
    weakref.WeakValueDictionary()
)
# The SaT model used last, by its key: kept loaded for the documents that follow,
# until another model is used or `_release_sat_model()` is called. One model at
# most, as a loaded model holds about a gigabyte of memory
_KEPT_SAT_MODEL: dict[str, SaT] = {}
# Lets one thread at a time look up, keep or release the SaT models
_SAT_MODELS_LOCK = threading.Lock()
# Lets one thread at a time load a SaT model, by the keys of the models
_SAT_MODEL_LOADING_LOCKS: dict[str, threading.Lock] = {}
# Number of texts that a SaT model splits into sentences in one batch (see
# `_split_into_sentences()`): in batches of 16 rather than of the 32 of
# wtpsplit-lite, the paragraphs of the long documents measured took about a tenth
# less time, with the same sentences
_SAT_BATCH_SIZE = 16

# HTTP status codes of model API errors that are worth retrying, in addition to
# server errors (5xx): request timeout, conflict (lock timeout), rate limit.
_MODEL_API_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 409, 429})

# Bounds (in seconds) of the exponential backoff between the retries
# of a failed model API request.
_MODEL_API_RETRY_BACKOFF_BASE = 1.0
_MODEL_API_RETRY_BACKOFF_MAX = 10.0

# Name of the attribute set on an exception raised by a failed model API request,
# holding the number of retries attempted for that request.
_MODEL_API_RETRIES_ATTEMPTED_ATTR = "_contextgem_retries_attempted"


# Pre-compiled patterns for hot-path text cleaning and LLM-output parsing.
_RE_CONTROL_CHARS_PRESERVE_NEWLINES = re.compile(
    r"[\x00-\x09\x0B-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202F\uFEFF]"
)
_RE_CONTROL_CHARS_ALL = re.compile(
    r"[\x00-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202F\uFEFF]"
)
# The control characters above that are not whitespace, e.g. zero-width ones.
_RE_NON_WHITESPACE_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0E-\x1B\x7F-\x84\x86-\x9F\u200B-\u200F\u202A-\u202E\uFEFF]"
)
# The other control characters above, i.e. whitespace such as the tab, the line
# feed and the form feed, where they are all that separates two words. In a text
# cleaned for a prompt, they are replaced with a space before the control
# characters are removed, so that the words are not joined: all of them, or all
# but the line feed where the line breaks are preserved.
_RE_WHITESPACE_CONTROL_CHARS_BETWEEN_WORDS = re.compile(
    r"(?<=\S)[\x09-\x0D\x1C-\x1F\x85\u2028\u2029\u202F]+(?=\S)"
)
_RE_WHITESPACE_CONTROL_CHARS_BETWEEN_WORDS_EXCEPT_NEWLINES = re.compile(
    r"(?<=\S)[\x09\x0B-\x0D\x1C-\x1F\x85\u2028\u2029\u202F]+(?=\S)"
)
_RE_NORMALIZE_NEWLINES = re.compile(r"\r\n|\r")
_RE_HORIZONTAL_WHITESPACE = re.compile(r"[ \t]+")
_RE_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")
_RE_ANY_WHITESPACE = re.compile(r"\s+")
_RE_JSON_CODE_BLOCK = re.compile(r"^```json\s*([\s\S]*?)\s*```$")

# The IDs that the extraction prompts show: of an aspect ("A3"), a concept ("C3")
# or a paragraph ("P3"), and of a sentence of a paragraph ("P3-S1"). A position
# in an ID is a number in ASCII digits, without a sign or leading zeros.
_ID_POSITION_PATTERN = r"(0|[1-9][0-9]*)"
_RE_PROMPT_IDS = {
    prefix: re.compile(prefix + _ID_POSITION_PATTERN) for prefix in ("A", "C", "P")
}
_RE_PROMPT_SENTENCE_ID = re.compile(f"P{_ID_POSITION_PATTERN}-S{_ID_POSITION_PATTERN}")


@cache
def _get_template(
    template_name: str,
    template_type: Literal["prompt", "system"] = "prompt",
) -> Template:
    """
    Retrieves a Jinja2 template based on the given parameters. The template is read from a specific file path
    determined by the `template_type`.

    The template is read, validated and compiled once: the calls that follow with
    the same params return the same object. Compiling the templates of the built-in
    prompts takes tens of milliseconds, which each LLM would otherwise spend when it
    is created. The object returned is thus shared, e.g. by all the LLMs that use
    the built-in prompts, and must not be modified. Rendering a compiled template
    does not change it: each rendering has a context of its own, so that a template
    can be rendered for several LLMs, and from several threads, at the same time.
    A template that replaces a built-in one for an LLM is compiled for that LLM
    (see ``_DocumentLLM._update_default_prompt``).

    Threads that request a template for the first time at the same moment may each
    compile it. The templates they get are then separate objects with the same
    content, one of which is returned by the calls that follow.

    :param template_name: The name of the template to be retrieved. This is the base name of the file without any
        extensions.
    :type template_name: str
    :param template_type: The type of the template, which determines the folder it is retrieved from. It can
        be either "prompt" or "system". Defaults to "prompt".
    :type template_type: Literal["prompt", "system"]
    :return: The loaded template, as a compiled Jinja2 `Template` object.
    :rtype: Template
    :raises NotImplementedError: If the `template_type` provided is unsupported.
    :raises RuntimeError: If the template file is not found in the package resources, is empty, or contains
        only whitespace, or if the template validation fails.
    """

    if template_type == "prompt":
        template_package = resources.files("contextgem.internal.prompts")
        template_file = f"{template_name}.j2"
    elif template_type == "system":
        template_package = resources.files("contextgem.internal.system")
        template_file = f"{template_name}.j2"
    else:
        raise NotImplementedError(f"Unknown template type: {template_type}")

    try:
        template_text = (
            (template_package / template_file).read_text(encoding="utf-8").strip()
        )
        if not template_text:
            raise RuntimeError(
                f"Template file '{template_file}' is empty or contains only whitespace"
            )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Template file '{template_file}' not found in {template_type} templates"
        ) from e
    # Validate template text
    if not _are_prompt_template_brackets_balanced(template_text):
        raise RuntimeError("Prompt template brackets are not balanced.")
    if not _are_prompt_template_xml_tags_balanced(template_text):
        raise RuntimeError("Prompt template XML tags are not balanced.")
    if bool(re.search(r"(\r\n|\r|\n){3,}", template_text)):
        raise RuntimeError("Too many newlines in template.")
    return _setup_jinja2_template(template_text)


def _setup_jinja2_template(template_text: str) -> Template:
    """
    Creates and configures a Jinja2 template from the provided text.

    This function handles the complete process of:
    1. Validating that the template contains Jinja2 tags (dynamic content)
    2. Creating the Template object with appropriate options
    3. Setting up global functions available to the template

    :param template_text: The raw text content of the template.
    :type template_text: str
    :return: A fully configured Jinja2 Template object.
    :rtype: Template
    :raises ValueError: If the template does not contain any Jinja2 tags.
    """

    if not _contains_jinja2_tags(template_text):
        raise ValueError("Template contains no Jinja2 tags.")

    # Create the Template object with appropriate options
    template = Template(template_text, trim_blocks=True, lstrip_blocks=True)

    # Set up global functions
    template.globals["enumerate"] = enumerate
    template.globals["_clean_text_for_llm_prompt"] = _clean_text_for_llm_prompt

    return template


def _contains_jinja2_tags(text: str) -> bool:
    """
    Determines if a string contains Jinja2 template tags or expressions.

    This function parses the input text using Jinja2's parser and examines
    the resulting abstract syntax tree (AST) to check for the presence of
    dynamic content such as variables, expressions, or control structures.

    :param text: The text to check for Jinja2 template tags
    :type text: str
    :return: True if the text contains Jinja2 tags, False otherwise
    :rtype: bool
    """
    env = Environment()  # nosec B701 - templates used for internal LLM prompts
    parsed = env.parse(text)
    # If any node in the top-level body is not TemplateData (and isn't an Output
    # wrapping only TemplateData), it indicates the presence of Jinja2 tags,
    # i.e. dynamic content.
    for node in parsed.body:
        if isinstance(node, nodes.Output):
            if not all(isinstance(child, nodes.TemplateData) for child in node.nodes):
                return True
        elif not isinstance(node, nodes.TemplateData):
            return True
    return False


def _clean_control_characters(
    text: str, preserve_newlines: bool = True, strip_text: bool = True
) -> str:
    """
    Removes control characters from text.

    :param text: The input string to be cleaned of control characters.
    :type text: str
    :param preserve_newlines: Whether to preserve newline characters (\n).
        If True, removes control characters except newlines.
        If False, removes all control characters including newlines.
    :type preserve_newlines: bool
    :param strip_text: Whether to strip the text of leading and trailing whitespace.
        If False, the text is not stripped. Defaults to True.
    :type strip_text: bool
    :return: The text with control characters removed and optionally stripped.
    :rtype: str
    """
    if preserve_newlines:
        # Remove control characters EXCEPT newlines (\n = ASCII 10)
        # This includes:
        # - ASCII control characters except LF (0x00-0x09, 0x0B-0x1F and 0x7F)
        # - Zero-width characters
        # - Bidirectional text markers
        # - Other invisible unicode characters
        cleaned = _RE_CONTROL_CHARS_PRESERVE_NEWLINES.sub("", text)
    else:
        # Remove all control characters including newlines
        cleaned = _RE_CONTROL_CHARS_ALL.sub("", text)

    return cleaned.strip() if strip_text else cleaned


def _clean_text_for_llm_prompt(
    text: str, preserve_linebreaks: bool = True, strip_text: bool = True
) -> str:
    """
    Removes control characters and other problematic elements from text
    to make it suitable for model input.

    The words that a tab, a line break or another whitespace character separates
    stay separated: where such a character is all that separates two words, it is
    replaced with a space, or kept as a line break, and is not removed with
    the control characters.

    :param text: The input string to be cleaned.
    :type text: str
    :param preserve_linebreaks: Whether to preserve linebreaks in the text.
        If False, all whitespace is collapsed to a single space. Defaults to True.
    :type preserve_linebreaks: bool
    :param strip_text: Whether to strip the text of leading and trailing whitespace.
        If False, the text is not stripped. Defaults to True.
    :type strip_text: bool
    :return: A cleaned and formatted version of the input text.
    :rtype: str
    """

    if preserve_linebreaks:
        # Normalize newlines to \n
        cleaned = _RE_NORMALIZE_NEWLINES.sub("\n", text)

        # Remove the control characters that are not whitespace
        cleaned = _RE_NON_WHITESPACE_CONTROL_CHARS.sub("", cleaned)

        # Replace the whitespace that would be removed with the control characters,
        # e.g. a tab, with a space where it is all that separates two words, so that
        # the words are not joined
        cleaned = _RE_WHITESPACE_CONTROL_CHARS_BETWEEN_WORDS_EXCEPT_NEWLINES.sub(
            " ", cleaned
        )

        # Remove the remaining control characters except newlines
        cleaned = _clean_control_characters(
            cleaned, preserve_newlines=True, strip_text=strip_text
        )

        # Replace horizontal whitespace sequences (spaces and tabs) with a single space
        # while preserving linebreaks
        cleaned = _RE_HORIZONTAL_WHITESPACE.sub(" ", cleaned)

        # Remove extra blank lines (more than one consecutive newline)
        cleaned = _RE_EXTRA_BLANK_LINES.sub("\n\n", cleaned)

    else:
        # Remove the control characters that are not whitespace
        cleaned = _RE_NON_WHITESPACE_CONTROL_CHARS.sub("", text)

        # Replace the whitespace that would be removed with the control characters,
        # e.g. a tab or a line break, with a space where it is all that separates
        # two words, so that the words are not joined
        cleaned = _RE_WHITESPACE_CONTROL_CHARS_BETWEEN_WORDS.sub(" ", cleaned)

        # Remove the remaining control characters including newlines
        cleaned = _clean_control_characters(
            cleaned, preserve_newlines=False, strip_text=strip_text
        )

        # Remove all whitespace sequences with a single space
        cleaned = _RE_ANY_WHITESPACE.sub(" ", cleaned)

    return cleaned.strip() if strip_text else cleaned


def _is_text_content_empty(text: str) -> bool:
    """
    Checks if text content is empty or whitespace-only after removing control characters.

    This function first removes all control characters (including newlines) and then
    checks if the remaining text is empty or contains only whitespace. This catches
    cases where text contains only invisible unicode characters, control characters,
    tabs, spaces, or any other whitespace characters.

    :param text: Text to check
    :return: True if text is empty or contains only whitespace after control character removal
    """
    if not text:
        return True

    # Remove all control characters including newlines
    cleaned_text = _clean_control_characters(
        text, preserve_newlines=False, strip_text=True
    )

    # Check if remaining text is only whitespace (covers all Unicode whitespace categories)
    return not cleaned_text or cleaned_text.isspace()


def _contains_linebreaks(text: str) -> bool:
    """
    Checks if the given string contains line breaks, considering both Unix (\n) and
    Windows (\r\n) style line breaks.

    :param text: The string to be checked.
    :type text: str
    :return: True if the string contains one or more line breaks, False otherwise.
    :rtype: bool
    """
    # Check for both Unix (\n) and Windows (\r\n) style line breaks
    return "\n" in text or "\r" in text


def _are_prompt_template_brackets_balanced(prompt: str) -> bool:
    """
    Checks whether each opening bracket in prompt template has a matching closing bracket.
    Relevant for JSON object / array structures in the prompt.

    To be used only on a prompt template, not on a rendered prompt, which may contain arbitrary text
    submitted by users, i.e. may contain any combinations of brackets.

    :param prompt: The text prompt to be validated.
    :type prompt: str
    :return: bool
    :rtype: bool
    """
    stack = []
    brackets = {
        "]": "[",
        "}": "{",
    }  # Closing brackets mapped to their counterparts

    for char in prompt:
        if char in "[{":  # If opening bracket, push onto stack
            stack.append(char)
        elif char in "]}":  # If closing bracket
            if (
                not stack or stack[-1] != brackets[char]
            ):  # Check for matching opening bracket
                return False
            stack.pop()  # Pop the matching opening bracket off the stack

    return not stack  # If stack is empty, all brackets were matched


def _are_prompt_template_xml_tags_balanced(prompt: str) -> bool:
    """
    Checks whether each opening XML tag in prompt template has a matching closing tag.
    Relevant for XML structure validation in prompts with semantic tags.

    To be used only on a prompt template, not on a rendered prompt, which may contain arbitrary text
    submitted by users, i.e. may contain any combinations of XML-like content.

    :param prompt: The text prompt to be validated.
    :type prompt: str
    :return: bool
    :rtype: bool
    """

    stack = []

    # Find all XML tags (both opening and closing)
    # Pattern matches: <tag>, <tag id="value">, </tag>
    tag_pattern = r"<(/?)([a-zA-Z_][a-zA-Z0-9_-]*)\s*[^>]*>"

    for match in re.finditer(tag_pattern, prompt):
        is_closing = match.group(1) == "/"  # True if it's a closing tag
        tag_name = match.group(2)

        if is_closing:
            # Closing tag - check if it matches the most recent opening tag
            if not stack or stack[-1] != tag_name:
                return False
            stack.pop()  # Remove the matching opening tag
        else:
            # Opening tag - add to stack
            stack.append(tag_name)

    if stack:
        logger.error(f"Unmatched XML tags in prompt: {stack}")

    return not stack  # If stack is empty, all tags were matched


def _split_text_into_paragraphs(raw_text: str) -> list[str]:
    """
    Splits a given raw text into a list of paragraphs, filtering out empty ones.

    :param raw_text: The input raw text to be split.
    :type raw_text: str
    :return: A list of paragraphs.
    :rtype: list[str]
    """
    paragraphs = re.split(r"[\r\n]+", raw_text)
    paragraphs = [i.strip() for i in paragraphs]
    paragraphs = [i for i in paragraphs if not _is_text_content_empty(i)]
    return paragraphs


def _chunk_list(lst: list, n: int) -> list[list]:
    """
    Divides a given list into smaller lists (chunks) of a specified size.

    This function takes a list and an integer `n` as input and returns a new list
    of lists where each sublist contains up to `n` elements from the original
    list. The last sublist may contain fewer elements if the list's length
    is not evenly divisible by `n`.

    :param lst: The original list to be divided into chunks.
    :type lst: list
    :param n: The size of each chunk.
    :type n: int
    :return: A list of lists where each sublist has up to `n` elements from
        the original list.
    :rtype: list[list]
    """
    return [lst[i : i + n] for i in range(0, len(lst), n)]


async def _async_multi_executor(
    func: Callable[..., Any], data_list: list[dict[str, Any]]
) -> list[Any]:
    """
    Executes an async function concurrently over a list of keyword-arg dictionaries.

    An error raised by a call is raised as it is, i.e. it is not turned into
    a result of the call, and only when no call is left running: on the first error
    of a call, or when the caller is cancelled, the calls that are still running
    are cancelled and waited for, as in `_run_async_calls`.

    :param func: An async function (i.e., defined with "async def").
                 It must accept keyword arguments that match each dict in data_list.
    :param data_list: A list of dictionaries to pass as keyword arguments to `func`.
    :return: A list of results corresponding to each input dictionary in `data_list`.
    """

    if not data_list:
        logger.warning("No data to process in tasks.")
        return []

    # Run one call per item in data_list, and wait for all the calls to finish
    logger.debug(f"Starting tasks ({len(data_list)})")
    results = await _run_async_calls(
        cals_and_kwargs=[(func, kwargs) for kwargs in data_list],
        use_concurrency=True,
    )
    logger.debug(f"Completed all tasks ({len(data_list)})")
    return results


@_suppress_litellm_warnings
async def _run_async_calls(
    cals_and_kwargs: AsyncCalsAndKwargs,
    use_concurrency: bool,
) -> list[Any]:
    """
    Runs a series of asynchronous callables with the provided arguments, optionally using
    concurrency for executing them. If concurrency is enabled, all callables will be scheduled
    and awaited simultaneously; otherwise, they will be awaited sequentially.

    "Sequential" mode is "stop on first error," while "concurrent" mode is
    "cancel other tasks on first error." Returned results maintain the same index order as
    cals_and_kwargs.

    In "concurrent" mode, the error is raised only when no task is left running: on the first
    error of a task, or when the caller is cancelled, the tasks that are still running are
    cancelled and waited for. A task therefore does nothing in the background, e.g. sends
    no requests and changes no objects, after the caller gets the error.

    :param cals_and_kwargs: A list of tuples where each tuple contains an asynchronous
        callable and a dictionary of keyword arguments to be passed to the callable.
    :param use_concurrency: A boolean flag indicating whether all tasks should be executed
        concurrently or sequentially.
    :return: A list containing the results of each asynchronous task. The results are
        ordered according to the order of the input list.
    """

    if use_concurrency:
        tasks = [asyncio.create_task(i[0](**i[1])) for i in cals_and_kwargs]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # `gather` raises the first error without stopping the other tasks.
            # Cancelling a task that is already done has no effect.
            for task in tasks:
                task.cancel()
            # The errors of the other tasks, including their cancellation, are
            # not raised: the first error is
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
    else:
        results = []
        for cal, kwargs in cals_and_kwargs:
            result = await cal(**kwargs)
            results.append(result)
        return results


@_suppress_litellm_warnings
def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """
    Synchronously runs an async function, whether or not an event
    loop is already running (e.g. in a Jupyter notebook).

    :param coro: A coroutine object to run.
    :return: Whatever the coroutine returns.
    """

    try:
        # If there's a running loop, get it.
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, so just create a new one
        loop = None

    if loop and loop.is_running():
        # We are in an environment (like Jupyter) that already has a running loop.
        # Start a new loop in a fresh thread to avoid conflicts.
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()  # ty: ignore[invalid-return-type]
    else:
        # No running loop, so just run it directly.
        return asyncio.run(coro)


def _warn_from_caller_code(message: str) -> None:
    """
    Issues a ``ContextGemWarning`` attributed to the code that called the framework,
    i.e. to the first frame of the call stack outside the framework, Pydantic and
    the event loop that the framework runs its coroutines in, e.g. to the line that
    creates a model, assigns a value to its param or calls an extraction method.

    A warning issued by a check that runs when an instance is validated is issued
    under frames of the framework and of Pydantic, whose number differs between
    the creation of the instance, its deserialization and an assignment to its
    param, i.e. no fixed ``stacklevel`` attributes it to the calling code in all
    these cases. A warning issued in a coroutine of the framework is issued under
    the frames of the event loop as well, and of the thread that runs it, if any.
    If there is no frame outside them, e.g. in a thread that the framework started,
    the warning is attributed to the outermost frame.

    :param message: The message of the warning.
    :type message: str
    :return: None
    :rtype: None
    """
    current_frame = inspect.currentframe()
    frame = current_frame.f_back if current_frame is not None else None
    # The level of the frame of the caller of this function
    stacklevel = 2
    while (
        frame is not None
        and frame.f_back is not None
        and os.path.normcase(frame.f_code.co_filename).startswith(
            _FRAMEWORK_AND_PYDANTIC_DIRS + _EVENT_LOOP_PATHS
        )
    ):
        frame = frame.f_back
        stacklevel += 1
    # The frames are released before the warning is issued (reference cycles)
    del current_frame, frame
    warnings.warn(message, ContextGemWarning, stacklevel=stacklevel)


def _model_call_result_is_valid(res: tuple[Any, _ModelUsage] | None) -> bool:
    """
    Determines whether the result from a model call is valid.

    Determines if a result exists and contains validated data as a result of a model processing function.
    Result may be one of the following:
    - None - if no result is received, in which case token and cost calculation is not possible.
        An exception raised during model processing does not give such a result: it is raised
        as it is, with and without concurrency.
    - None, _ModelUsage - if model processing failed due to a caught exception or failed data validation,
        e.g. invalid data returned by the model. In this case, token and cost calculation is still possible.
    - Any, _ModelUsage - if model processing was successful and result is validated. In this case,
        token and cost calculation are performed as normal.

    :param res: The result to check, represented as a tuple where the first element is the relevant
        result value, and the second element is model usage stats dict that follows the _ModelUsage pattern,
        or None if the result is not received or failed validation.
    :type res: tuple[Any, _ModelUsage] | None
    :return: A boolean indicating whether the result is successfully received and validated.
    :rtype: bool
    """
    return not (res is None or res[0] is None)


def _model_call_result_is_api_failure(res: tuple[Any, _ModelUsage] | None) -> bool:
    """
    Determines whether the result from a model call is invalid because the model API
    request failed, as opposed to the model returning invalid data.

    A failed model API request is recorded as an unfinished call, i.e. as the last call
    in the usage stats with no response received. By the time such a result is returned,
    the retries of the failed request are already exhausted (or the error is not
    retryable), so that sending the same request to the same model again is pointless.
    A request that the provider blocked for its content, or whose answer it blocked,
    is not a failed request: it is recorded as a finished call with its refusal (see
    ``_model_call_result_is_refused``).

    :param res: The result to check, represented as a tuple where the first element is the relevant
        result value, and the second element is model usage stats dict that follows the _ModelUsage pattern,
        or None if the result is not received (in which case the reason of the failure is unknown).
    :type res: tuple[Any, _ModelUsage] | None
    :return: A boolean indicating whether the result is invalid due to a failed model API request.
    :rtype: bool
    """
    if res is None or res[0] is not None:
        return False
    calls = res[1].calls
    return bool(calls) and calls[-1].timestamp_received is None


def _model_call_result_is_output_truncated(
    res: tuple[Any, _ModelUsage] | None,
) -> bool:
    """
    Determines whether the result from an LLM call is invalid because the response
    of the LLM was cut off at the max output tokens, as opposed to the LLM returning
    invalid data in a complete response.

    Such a response is recorded as the last call in the usage stats, with the tokens
    it used. The same request to the same LLM would be cut off again, so that sending
    it to that LLM again is pointless.

    :param res: The result to check, represented as a tuple where the first element is the relevant
        result value, and the second element is model usage stats dict that follows the _ModelUsage pattern,
        or None if the result is not received (in which case the reason of the failure is unknown).
    :type res: tuple[Any, _ModelUsage] | None
    :return: A boolean indicating whether the result is invalid due to a response
        cut off at the max output tokens.
    :rtype: bool
    """
    if res is None or res[0] is not None:
        return False
    calls = res[1].calls
    return bool(calls) and calls[-1]._output_truncated


def _model_call_result_is_refused(res: tuple[Any, _ModelUsage] | None) -> bool:
    """
    Determines whether the result from a model call is invalid because the model or
    its provider declined to answer (see ``ModelRefusal``), as opposed to the model
    returning invalid data, or a request failing for another reason.

    The answer of a refused request is not used. The call is made again as for
    invalid data, as the verdict of a model or of a content filter can differ on
    another request, and then with the fallback models, and the refusal is
    reported if none of them answers. So is a request that the provider blocked
    for its content, or whose answer it blocked, which litellm raises as
    ``ContentPolicyViolationError`` for both: a finished call without an answer.
    The refused request is recorded in the usage stats with its refusal, as the last
    call of an LLM, or as one of the calls of a decision model, whose requests on
    the texts of the call are recorded together.

    :param res: The result to check, represented as a tuple where the first element is the relevant
        result value, and the second element is model usage stats dict that follows the _ModelUsage pattern,
        or None if the result is not received (in which case the reason of the failure is unknown).
    :type res: tuple[Any, _ModelUsage] | None
    :return: A boolean indicating whether the result is invalid due to a refusal.
    :rtype: bool
    """
    return _get_model_call_result_refused_call(res) is not None


def _get_model_call_result_refused_call(
    res: tuple[Any, _ModelUsage] | None,
) -> _ModelCall | None:
    """
    Returns the request that the model or its provider declined to answer in
    the call that gave the result, i.e. the last refused request of the call,
    with its refusal and what the refused answer was for, or None if the result
    is valid, or invalid for another reason.

    :param res: The result to check, represented as a tuple where the first element is the relevant
        result value, and the second element is model usage stats dict that follows the _ModelUsage pattern,
        or None if the result is not received.
    :type res: tuple[Any, _ModelUsage] | None
    :return: The refused request, or None.
    :rtype: _ModelCall | None
    """
    if res is None or res[0] is not None:
        return None
    for call in reversed(res[1].calls):
        if call._refusal is not None:
            return call
    return None


def _get_model_refusal_msg(
    *, kind_name: str, model: str, refusal: ModelRefusal, subject: str = ""
) -> str:
    """
    Returns the message on a request that a model or its provider declined to
    answer. It names the model and what declined the answer, and never has
    the answer of the model or the text that the request was sent with.

    :param kind_name: The name of the kind of the model, e.g. "LLM" or "decision
        model", capitalized at the start of the message.
    :type kind_name: str
    :param model: The model, as set on the model object.
    :type model: str
    :param refusal: How the request was declined.
    :type refusal: ModelRefusal
    :param subject: What the request was for, put after the model in the message,
        e.g. " for concepts ['Parties']". Defaults to an empty string.
    :type subject: str
    :return: The message.
    :rtype: str
    """
    if refusal == "refusal":
        kind = kind_name[:1].upper() + kind_name[1:]
        what = f"{kind} `{model}` refused to answer{subject}."
    elif refusal == "content_filter":
        what = (
            f"The answer of {kind_name} `{model}`{subject} was stopped with "
            f"the `content_filter` finish reason: the model refused to answer, "
            f"or the content filter of its provider stopped the answer."
        )
    else:
        what = (
            f"The provider of {kind_name} `{model}` blocked the request{subject}, "
            f"or the answer to it, for its content."
        )
    return what


def _is_retryable_model_api_error(error: Exception) -> bool:
    """
    Determines whether a failed model API request is worth retrying, i.e. whether
    the error is transient.

    Relies on the HTTP status code that litellm assigns to the errors of all model providers.
    Request timeouts, conflicts, rate limits, and server errors (including connection errors,
    which litellm reports as server errors) are retryable. Other client errors, such as
    an invalid request, a failed authentication, or an exceeded context window, will fail
    again in the same way and are therefore not retryable.

    An error without a status code is not an error of litellm, which gives one to each
    error of a request that it raises, also to an error that it cannot map
    (``APIConnectionError``, 500). It is e.g. an error of a stand-in for the API in
    a test, which would fail again in the same way: it is not retryable, unless it is
    a timeout or a connection error of Python (``TimeoutError``, ``ConnectionError``),
    e.g. one that a stand-in raises to simulate such a failure.

    :param error: The exception raised by the model API request.
    :type error: Exception
    :return: True if the request is worth retrying, False otherwise.
    :rtype: bool
    """
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        return isinstance(error, TimeoutError | ConnectionError)
    return status_code in _MODEL_API_RETRYABLE_STATUS_CODES or status_code >= 500


def _get_model_api_retry_delay(retry_number: int) -> float:
    """
    Calculates the delay before a retry of a failed model API request.

    Uses capped exponential backoff with jitter: the backoff doubles with each retry,
    and a random part of it (up to a half) is cut off, so that the requests that failed
    together, e.g. when hitting a rate limit with concurrency enabled, are not retried
    all at the same moment.

    :param retry_number: The number of the retry to be performed, starting from 1.
    :type retry_number: int
    :return: The delay in seconds.
    :rtype: float
    """
    backoff = min(
        _MODEL_API_RETRY_BACKOFF_BASE * 2 ** (retry_number - 1),
        _MODEL_API_RETRY_BACKOFF_MAX,
    )
    return backoff * random.uniform(0.5, 1.0)  # nosec B311 - not a security context


def _get_cache_token_counts(
    input_tokens: int, input_tokens_details: Any
) -> tuple[int, int]:
    """
    Retrieves the number of input tokens read from, and written to, the prompt cache
    of the model provider, from the details of the input tokens of a request, as
    reported in the usage data of the response to it.

    Relies on the token details that litellm populates in the same way for all
    model providers that report such tokens, where they are a part of the input tokens.
    Missing or invalid counts are treated as zero, and the counts are capped so that
    together they do not exceed the number of input tokens, e.g. if a custom API
    endpoint reports them separately from the input tokens.

    :param input_tokens: The number of input tokens of the request.
    :type input_tokens: int
    :param input_tokens_details: The details of the input tokens of the request,
        as returned by litellm, or None if there are no such details.
    :type input_tokens_details: Any
    :return: A tuple of the number of tokens read from the cache and the number
        of tokens written to the cache.
    :rtype: tuple[int, int]
    """

    def get_count(attr_name: str) -> int:
        """
        Retrieves a token count from the details of the input tokens.

        :param attr_name: The name of the attribute holding the token count.
        :type attr_name: str
        :return: The token count, or zero if it is missing or invalid.
        :rtype: int
        """
        count = getattr(input_tokens_details, attr_name, None)
        return count if isinstance(count, int) and count > 0 else 0

    cache_read = min(get_count("cached_tokens"), input_tokens)
    cache_write = min(get_count("cache_write_tokens"), input_tokens - cache_read)
    return cache_read, cache_write


def _get_llm_cache_token_counts(completion_usage: Any) -> tuple[int, int]:
    """
    Retrieves the number of input tokens read from, and written to, the prompt cache
    of the LLM provider, as reported in the usage data of a completion.

    Relies on the prompt token details that litellm populates in the same way for all
    LLM providers that report such tokens, where they are a part of the prompt tokens.
    Missing or invalid counts are treated as zero, and the counts are capped so that
    together they do not exceed the number of prompt tokens, e.g. if a custom API
    endpoint reports them separately from the prompt tokens.

    :param completion_usage: The usage data of a completion returned by litellm.
    :type completion_usage: Any
    :return: A tuple of the number of tokens read from the cache and the number
        of tokens written to the cache.
    :rtype: tuple[int, int]
    """
    return _get_cache_token_counts(
        completion_usage.prompt_tokens,
        getattr(completion_usage, "prompt_tokens_details", None),
    )


def _remove_thinking_content_from_llm_output(output_str: str | None) -> str | None:
    """
    Removes thinking content enclosed in <think></think> tags from the beginning of LLM outputs.

    When using local reasoning LLMs (e.g. DeepSeek R1 in Ollama), the output may include
    thinking steps enclosed in <think></think> tags at the beginning. This function removes those tags
    and their content only if they appear at the start of the string, then strips any remaining whitespace.

    This preserves any <think></think> tags that might appear later in the content as part of the
    actual response.

    :param output_str: The output string from an LLM that may contain thinking content, can be None
                      if LLM outputs invalid content
    :type output_str: str | None

    :return: The cleaned string without initial thinking content and extra whitespace, or None if
             the input was None or an error occurred during processing
    :rtype: str | None
    """
    if output_str is None:
        return None

    try:
        # Check if the string starts with <think> tag
        if output_str.strip().startswith("<think>"):
            # Find the first closing </think> tag
            end_tag_pos = output_str.find("</think>")
            if end_tag_pos != -1:
                # Remove everything from start to the end of </think> tag
                cleaned_str = output_str[end_tag_pos + len("</think>") :]
                # Strip any remaining whitespace
                cleaned_str = cleaned_str.strip()
                if not cleaned_str:
                    raise ValueError("Cleaned string is empty")
                return cleaned_str

        return output_str.strip()
    except (ValueError, AttributeError):
        return None


def _get_finite_float(number_text: str) -> float:
    """
    Converts a number of a JSON text with a fraction or an exponent to a float,
    which must be finite: a number too large for a float, e.g. ``1e400``, is not
    read as an infinity, which JSON does not have.

    :param number_text: The number, as written in the JSON text.
    :type number_text: str
    :return: The number.
    :rtype: float
    :raises ValueError: If the number is too large for a float.
    """
    number = float(number_text)
    if not math.isfinite(number):
        raise ValueError(f"The number `{number_text}` is too large for a float.")
    return number


def _reject_json_constant(constant: str) -> float:
    """
    Rejects ``NaN``, ``Infinity`` and ``-Infinity`` in a JSON text, which JSON does
    not have, while ``json.loads()`` reads them as floats by default.

    :param constant: The constant, as written in the JSON text.
    :type constant: str
    :return: Never returns.
    :rtype: float
    :raises ValueError: Always.
    """
    raise ValueError(f"`{constant}` is not a JSON value.")


def _load_json(text: str) -> Any:
    """
    Parses a JSON text, which must be valid JSON: a number that is not finite
    (``NaN``, ``Infinity``, ``-Infinity``), or that is too large for a float, is
    rejected rather than read as a float that is not finite.

    :param text: The JSON text.
    :type text: str
    :return: The parsed object.
    :rtype: Any
    :raises ValueError: If the text is not valid JSON, has a number that is not
        finite, or has an integer with more digits than Python converts.
    :raises RecursionError: If arrays or objects are nested deeper than
        the parser can recurse.
    """
    return json.loads(
        text, parse_float=_get_finite_float, parse_constant=_reject_json_constant
    )


def _parse_llm_output_as_json(
    output_str: str | dict | list | None,
) -> dict | list | None:
    """
    Parses the provided LLM-generated output into a JSON-compatible Python object.

    This function attempts to parse the `output_str` string into a dictionary or a list.
    If the string cannot be parsed (either due to a JSON decoding error or an incompatible type),
    it returns `None`. The function also includes handling for strings that include a surrounding
    ` ```json` code block, removing them before parsing.

    The output must be valid JSON: a number that is not finite (``NaN``, ``Infinity``,
    ``-Infinity``), or that is too large for a float (e.g. ``1e400``), which
    ``json.loads()`` reads by default as a float that is not finite, makes the output
    unparseable, as do an integer with more digits than Python converts, and
    arrays or objects nested deeper than the parser can recurse.

    :param output_str: The output string to parse. It may already be a JSON-parsed Python object,
        a JSON string, or a string containing a JSON code block marked with ` ```json`. Can be None
        if LLM outputs invalid content.
    :type output_str: str | dict | list | None

    :return: A dictionary, a list, or `None` if parsing fails or the `output_str` type is invalid.
    :rtype: dict | list | None
    """

    # Handle None case
    if output_str is None:
        return None

    # Handle already parsed JSON
    if isinstance(output_str, dict | list):
        return output_str

    # Only proceed if it's a string
    if not isinstance(output_str, str):
        return None

    # A JSON text that cannot be parsed raises json.JSONDecodeError, which is
    # a ValueError, as is the error for a number that is not finite, or for
    # an integer with more digits than Python converts. Arrays or objects nested
    # deeper than the parser can recurse raise RecursionError.
    try:
        parsed = _load_json(output_str)

    except (ValueError, RecursionError):
        try:
            # Handle markdown code blocks using regex
            answer = output_str.strip()
            match = _RE_JSON_CODE_BLOCK.match(answer)

            if match:
                # Get the content between the markers
                answer = match.group(1).strip()

            parsed = _load_json(answer)
        except (ValueError, RecursionError):
            return None

    # JSON primitives (numbers, booleans, strings, null) are not extraction
    # outputs — only dict/list root values are valid. Reject anything else
    # to honor the declared return type.
    return parsed if isinstance(parsed, dict | list) else None


def _get_position_from_prompt_id(prompt_id: str, prefix: Literal["A", "C", "P"]) -> int:
    """
    Returns the position that an ID returned by an LLM refers to, e.g. 3 for "P3",
    the ID of the fourth paragraph shown in the prompt.

    The ID must have the form of the IDs that the prompt shows: the prefix and
    the position in ASCII digits, without a sign or leading zeros. Surrounding
    whitespace is ignored. Any other ID, e.g. "P03", "P-3", "PP3", "3" or "P3_0",
    is not an ID of the prompt, and is not read as one.

    :param prompt_id: The ID returned by the LLM.
    :type prompt_id: str
    :param prefix: Prefix of the IDs of the kind: "A" for aspects, "C" for
        concepts, "P" for paragraphs.
    :type prefix: Literal["A", "C", "P"]
    :return: The position in the ID.
    :rtype: int
    :raises ValueError: If the ID does not have the form of the IDs of the kind.
    """
    match = _RE_PROMPT_IDS[prefix].fullmatch(prompt_id.strip())
    if match is None:
        raise ValueError(
            f"Invalid ID `{prompt_id}`: an ID of the form `{prefix}<position>` "
            f"is expected."
        )
    return int(match.group(1))


def _get_sentence_position_from_prompt_id(
    sentence_id: str, paragraph_position: int
) -> int:
    """
    Returns the position of the sentence that a sentence ID returned by an LLM
    refers to in its paragraph, e.g. 1 for "P3-S1", the ID of the second sentence
    of the paragraph "P3" shown in the prompt.

    The ID must have the form of the sentence IDs that the prompt shows, and be
    the ID of a sentence of the paragraph that the LLM lists it for: a sentence of
    another paragraph is an error of the LLM, and is not read as a sentence of
    either paragraph. Surrounding whitespace is ignored.

    :param sentence_id: The sentence ID returned by the LLM.
    :type sentence_id: str
    :param paragraph_position: Position of the paragraph that the LLM lists
        the sentence for.
    :type paragraph_position: int
    :return: The position of the sentence in its paragraph.
    :rtype: int
    :raises ValueError: If the ID does not have the form of a sentence ID, or is
        the ID of a sentence of another paragraph.
    """
    match = _RE_PROMPT_SENTENCE_ID.fullmatch(sentence_id.strip())
    if match is None or int(match.group(1)) != paragraph_position:
        raise ValueError(
            f"Invalid sentence ID `{sentence_id}`: an ID of the form "
            f"`P{paragraph_position}-S<position>` is expected."
        )
    return int(match.group(2))


def _describe_error_without_values(error: Exception) -> str:
    """
    Describes an error raised while an output of a model is validated or processed,
    without the values of the output that the error was raised for, so that it can
    be logged: the output can hold text of the document, e.g. in an extracted value,
    which does not belong in the logs of the calling code.

    A ``ValidationError`` of pydantic is described by the location and the message
    of each of its errors, i.e. without the input of each error that its text
    gives. The messages of the validators of the framework that check the values
    of a model's output do not give the values either. Any other error is described
    by its text.

    :param error: The error to describe.
    :type error: Exception
    :return: The description of the error.
    :rtype: str
    """
    if not isinstance(error, pydantic.ValidationError):
        return str(error)
    descriptions: list[str] = []
    for details in error.errors(
        include_url=False, include_context=False, include_input=False
    ):
        location = ".".join(str(part) for part in details["loc"])
        message = details["msg"]
        descriptions.append(f"`{location}`: {message}" if location else message)
    return "; ".join(descriptions)


def _validate_parsed_llm_output(
    parsed_json: dict | list | None,
    extracted_item_type: ExtractedInstanceType,
    justification_provided: bool,
    references_provided: bool,
    reference_depth: ReferenceDepth,
) -> dict | list | None:
    """
    Validates the parsed LLM output against a specific validation model.

    :param parsed_json: The JSON content output from the LLM to be validated.
        If None, the previous parsing step failed.
    :type parsed_json: dict | list | None
    :param extracted_item_type: Specifies the type of extracted item(s) to validate
        ("aspect" or "concept").
    :type extracted_item_type: ExtractedInstanceType
    :param justification_provided: Indicates whether the extracted data includes
        justifications.
    :type justification_provided: bool
    :param references_provided: Indicates whether the extracted data includes references.
    :type references_provided: bool
    :param reference_depth: The structural depth of the references, i.e. whether to provide
        paragraphs as references or sentences as references. Defaults to "paragraphs".
        ``extracted_items`` will have values based on this parameter.
    :type reference_depth: ReferenceDepth
    :return: The validated JSON content if it conforms to the schema, otherwise ``None``.
    :rtype: dict | list | None
    """

    if parsed_json is None:
        logger.error("Error when validating parsed JSON: parsed_json is None")
        return None

    validation_context = {}
    if extracted_item_type == "aspect":
        with_extra_data = justification_provided
        validation_model = _get_aspect_extraction_output_struct(
            with_extra_data=with_extra_data, reference_depth=reference_depth
        )
    elif extracted_item_type == "concept":
        if justification_provided or references_provided:
            with_extra_data = True
        else:
            with_extra_data = False
        validation_model = _get_concept_extraction_output_struct(
            with_extra_data=with_extra_data,
            with_justification=justification_provided,
            with_references=references_provided,
            reference_depth=reference_depth,
        )
    else:
        raise ValueError(f"Invalid extracted item type: `{extracted_item_type}`")

    try:
        validation_model.model_validate(parsed_json, context=validation_context)
    except ValueError as e:
        logger.error(
            f"LLM output does not match the answer format: "
            f"{_describe_error_without_values(e)}"
        )
        return None

    return parsed_json


def _load_sat_model(model_id: SaTModelId = "sat-3l-sm") -> SaT:
    """
    Loads a SaT model to be used for paragraphs and sentence segmentation.
    Performs validation of the model ID or path before attempting to load the model.

    :param model_id:
        The identifier of the SaT model. Defaults to "sat-3l-sm".
        Can be:
        - A standard SaT model ID (e.g., "sat-3l-sm")
        - A local path to a SaT model directory (as a string or Path object).
          If the directory also has the tokenizer of the model (tokenizer.json),
          the model is loaded without network access. Otherwise, the tokenizer
          is downloaded from Hugging Face, or read from its cache.

    :return:
        An instance of the SaT model associated with the given `model_id`.

    :raises ValueError:
        If the provided path doesn't exist or is not a directory.
    :raises RuntimeError:
        If the provided path exists but the SaT model cannot be loaded from it.
    """
    # Convert Path object to string if needed
    if isinstance(model_id, Path):
        model_id = str(model_id)

    # Check if it's a standard model ID
    is_standard_model = False
    if isinstance(model_id, str):
        # Get standard models directly from the type definition
        standard_models = get_args(StandardSaTModelId)
        is_standard_model = model_id in standard_models

    # Determine if it's a local path (but not a standard model ID)
    is_local_path = False
    if isinstance(model_id, str) and not is_standard_model:
        path = Path(model_id)

        # Validate that the path exists and is a directory
        if not path.exists() or not path.is_dir():
            raise ValueError(
                f"The provided SaT model path '{model_id}' does not exist or is not a directory."
            )

        is_local_path = True

    # Log appropriate message
    if is_local_path:
        logger.info(f"Loading SaT model from local path {model_id}...")
    else:
        logger.info(f"Loading SaT model {model_id}...")

    # A local model is loaded with the tokenizer in its directory, if it has one.
    # Otherwise, wtpsplit-lite takes the tokenizer from Hugging Face, also for
    # a local model, i.e. a local model alone would not load without network access
    local_tokenizer_path = None
    if is_local_path:
        tokenizer_path = Path(model_id) / "tokenizer.json"
        if tokenizer_path.is_file():
            local_tokenizer_path = tokenizer_path

    # wtpsplit-lite is imported when a SaT model is first needed: with ONNX Runtime
    # and NumPy, its import takes about as long as that of the rest of the framework
    # without litellm, and a document that is segmented into paragraphs by newlines,
    # with no sentence-level references, does not need it
    from wtpsplit_lite import SaT

    # Attempt to load the model
    try:
        if local_tokenizer_path is not None:
            model = SaT(model_id, tokenizer_name_or_path=local_tokenizer_path)
        else:
            model = SaT(model_id)
        logger.info("SaT model loaded successfully.")
        return model
    except Exception as e:
        if is_local_path:
            # If it's a local path that exists but isn't a valid SaT model
            logger.error(f"Failed to load SaT model from path '{model_id}': {e}")
            tokenizer_note = (
                ""
                if local_tokenizer_path is not None
                else ", or the tokenizer of the model, which is taken from Hugging Face "
                "as the directory has no tokenizer.json, could not be loaded"
            )
            raise RuntimeError(
                f"The directory at '{model_id}' exists but does not contain a valid SaT model"
                f"{tokenizer_note}. Error: {e}"
            ) from e
        else:
            # For standard model IDs or other errors
            logger.error(f"Failed to load SaT model '{model_id}': {e}")
            raise


def _get_sat_model(model_id: SaTModelId = "sat-3l-sm") -> SaT:
    """
    Returns the SaT model with the given ID or path: the one that is loaded, i.e.
    kept for the documents that follow the one it was used for last, or in use,
    e.g. by a segmentation in another thread, or a new one, which it loads.

    The model is then kept as the one used last, in place of the one kept before,
    which is released unless it is in use: documents processed one after another
    load their model once, while the memory of at most one model is kept between
    them. The documents segmented at the same time share one model, rather than
    each loading a model of its own. Threads that need the same model at the same
    time load it once: they wait for the one that loads it, while the threads
    that need other models do not.

    :param model_id: The ID of the SaT model or the path of a folder with one,
        as for ``_load_sat_model()``. Defaults to "sat-3l-sm".
    :type model_id: SaTModelId
    :return: The SaT model.
    :rtype: SaT
    """
    key = _get_sat_model_key(model_id)
    with _SAT_MODELS_LOCK:
        model = _SAT_MODELS_IN_USE.get(key)
        if model is not None:
            _keep_sat_model(key, model)
            return model
        loading_lock = _SAT_MODEL_LOADING_LOCKS.setdefault(key, threading.Lock())
    # The model is loaded under a lock of its own, so that the threads that need
    # other models, e.g. one that is in use, do not wait for it, which can take
    # minutes when the model is downloaded
    with loading_lock:
        with _SAT_MODELS_LOCK:
            # Loaded by another thread while this one waited
            model = _SAT_MODELS_IN_USE.get(key)
        if model is None:
            model = _load_sat_model(model_id)
        with _SAT_MODELS_LOCK:
            _SAT_MODELS_IN_USE[key] = model
            _keep_sat_model(key, model)
    return model


def _keep_sat_model(key: str, model: SaT) -> None:
    """
    Keeps the SaT model loaded as the one used last, in place of the one kept
    before, if another, which is released unless it is in use. To be called with
    ``_SAT_MODELS_LOCK`` held.

    :param key: The key of the model (see ``_get_sat_model_key()``).
    :type key: str
    :param model: The model.
    :type model: SaT
    :return: None
    :rtype: None
    """
    if _KEPT_SAT_MODEL.get(key) is not model:
        _KEPT_SAT_MODEL.clear()
        _KEPT_SAT_MODEL[key] = model


def _release_sat_model() -> None:
    """
    Releases the SaT model that is kept loaded for the documents that follow
    the one it was used for last (see ``_get_sat_model()``). A model that is in
    use, e.g. by a segmentation in another thread, is released when it is no
    longer used. A document that needs a model then loads it again.

    :return: None
    :rtype: None
    """
    with _SAT_MODELS_LOCK:
        _KEPT_SAT_MODEL.clear()


def _get_sat_model_key(model_id: SaTModelId) -> str:
    """
    Returns what tells a SaT model from the others in ``_get_sat_model()``: the ID
    of a standard model, or the absolute path of a folder with a model, with
    symbolic links resolved and, on Windows, the case of the letters folded.

    The same folder thus gets the same key in whatever form its path is given,
    e.g. relative or absolute, with a trailing separator or a symbolic link,
    while the same relative path gets another key when the current directory has
    changed, as it is then the path of another folder.

    :param model_id: The ID of the SaT model or the path of a folder with one.
    :type model_id: SaTModelId
    :return: The key of the model.
    :rtype: str
    """
    model_id = str(model_id)
    if model_id in get_args(StandardSaTModelId):
        # As in `_load_sat_model()`, such an ID is that of a standard model
        return model_id
    return os.path.normcase(os.path.realpath(model_id))


def _split_into_sentences(model: SaT, texts: list[str]) -> list[list[str]]:
    """
    Splits each of the texts, e.g. those of the paragraphs of a document, into
    sentences with a SaT model.

    The model pads each text of a batch to the longest text of the batch, up to
    its block size of 512 tokens, and runs on the padding as well. The texts are
    therefore passed to it sorted by length, in batches of their own
    (``_SAT_BATCH_SIZE``), so that the texts of a batch are of about the same
    length. Passed in their order, the texts were padded to the longest text of
    each 1,000, which for the paragraphs of a long document, most of them short,
    took about ten times as long. The sentences are the same: each text is split
    into the same chunks of the input of the model either way, and the padding
    is masked.

    :param model: The SaT model.
    :type model: SaT
    :param texts: The texts.
    :type texts: list[str]
    :return: The sentences of each text, in the order of the texts.
    :rtype: list[list[str]]
    """
    order = sorted(range(len(texts)), key=lambda position: len(texts[position]))
    sentences_of_texts: list[list[str]] = [[] for _ in texts]
    for position, sentences in zip(
        order,
        model.split(
            [texts[position] for position in order],
            batch_size=_SAT_BATCH_SIZE,
            outer_batch_size=_SAT_BATCH_SIZE,
        ),
        strict=True,
    ):
        sentences_of_texts[position] = sentences
    return sentences_of_texts


def _group_instances_by_fields(
    fields: list[str], instances: list[_Aspect] | list[_Concept]
) -> list[list[_Aspect] | list[_Concept]]:
    """
    Group instances by a list of fields.

    :param fields: A list of field names by which to group the instances.
    :param instances: A list of instances to be grouped.
    :return: A list of lists, where each inner list contains instances
        that share the same values for the given fields.
    """
    grouped_dict = defaultdict(list)
    for instance in instances:
        # Build the key based on the specified fields
        key = tuple(getattr(instance, field, False) for field in fields)
        grouped_dict[key].append(instance)
    # Return the grouped instances as a list of lists
    return list(grouped_dict.values())


def _get_distinct_instances(instances: Sequence[InstanceT]) -> list[InstanceT]:
    """
    Returns the given aspects or concepts without the repeated ones, i.e. each of
    them once, in the order in which they are first given.

    An aspect or a concept that is passed to an extraction method more than once
    is extracted once: it would otherwise be listed in a prompt several times and
    get the items of each listing, or be processed in several calls at once.

    :param instances: The aspects or concepts, possibly with repeated ones.
    :type instances: Sequence[_Aspect] | Sequence[_Concept]
    :return: The distinct aspects or concepts.
    :rtype: list[_Aspect] | list[_Concept]
    """
    distinct_instances: dict[str, InstanceT] = {}
    for instance in instances:
        distinct_instances.setdefault(instance.unique_id, instance)
    return list(distinct_instances.values())


def _is_json_serializable(data: Any) -> bool:
    """
    Determines if the provided data is JSON serializable.

    This function attempts to serialize the given input data to JSON format
    using the `json.dumps` method. If the data cannot be serialized due to a
    TypeError or ValueError, the function returns False. Otherwise, it returns
    True, indicating that the data is JSON serializable.

    :param data: The input data to check for JSON serializability.
    :type data: Any
    :return: A boolean value indicating whether the data is JSON
             serializable (True) or not (False).
    :rtype: bool
    """
    try:
        json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError, OverflowError, RecursionError) as e:
        logger.error(f"Data is not JSON serializable. Error: {e}")
        return False
    return True


def _are_paragraphs_in_text_in_order(
    paragraphs: list[_Paragraph], document_text: str
) -> bool:
    """
    Returns whether the texts of the paragraphs are found in the document text one
    after another, in the order of the paragraphs and without overlapping.

    Paragraphs that are found so pass ``_check_paragraphs_match_in_text()`` and
    ``_check_paragraphs_ordering_in_text()``, as well as the matching of
    the segmented paragraphs in the text of a document. Each of those looks for
    a paragraph from the start of the text, i.e. takes time proportional to
    the number of the paragraphs times the length of the text, which adds up to
    seconds for a long document. This check looks for a paragraph from the end of
    the previous one, i.e. reads the text once, so the others are needed only to
    report the paragraphs that fail it.

    :param paragraphs: The paragraphs, in the order of the document.
    :type paragraphs: list[_Paragraph]
    :param document_text: The text of the document.
    :type document_text: str
    :return: Whether the paragraphs are found in the text in their order.
    :rtype: bool
    """
    position = 0
    for paragraph in paragraphs:
        found_at = document_text.find(paragraph.raw_text, position)
        if found_at == -1:
            return False
        position = found_at + len(paragraph.raw_text)
    return True


def _check_paragraphs_match_in_text(
    paragraphs: list[_Paragraph], document_text: str
) -> list[_Paragraph]:
    """
    Check that all paragraph texts exist in the given document text.

    :param paragraphs: List of paragraph objects to check
    :param document_text: The document text to search in
    :return: List of unmatched paragraphs
    """
    return [p for p in paragraphs if p.raw_text not in document_text]


def _check_paragraphs_ordering_in_text(
    paragraphs: list[_Paragraph], document_text: str
) -> None:
    """
    Check that paragraphs are ordered according to their appearance in document text.
    Handles cases where paragraphs may have duplicate text content.

    :param paragraphs: List of paragraph objects to check ordering for
    :param document_text: The document text to check ordering in
    :raises ValueError: If paragraphs are not ordered correctly
    """
    if len(paragraphs) <= 1:
        return

    current_search_pos = 0
    for i in range(len(paragraphs) - 1):
        current_para_text = paragraphs[i].raw_text
        next_para_text = paragraphs[i + 1].raw_text

        # Find current paragraph starting from the current search position
        current_pos = document_text.find(current_para_text, current_search_pos)
        if current_pos == -1:  # This shouldn't happen due to earlier check
            current_pos = document_text.find(current_para_text)

        # Update search position for next paragraph to start after current paragraph
        current_search_pos = current_pos + len(current_para_text)

        # Find next paragraph starting from the current search position
        next_pos = document_text.find(next_para_text, current_search_pos)
        if (
            next_pos == -1
        ):  # If not found from current position, check if it exists earlier
            next_pos = document_text.find(next_para_text)
            if next_pos < current_search_pos:
                raise ValueError(
                    "Paragraphs are not ordered according to their appearance "
                    "in the document's raw text."
                )
