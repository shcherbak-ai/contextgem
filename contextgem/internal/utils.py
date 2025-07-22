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
Module defining internal utility functions of the framework.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import inspect
import json
import re
import warnings
from collections import defaultdict
from collections.abc import Callable, Coroutine, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast, get_args

from jinja2 import Environment, Template, nodes
from wtpsplit_lite import SaT


if TYPE_CHECKING:
    from contextgem.public.aspects import Aspect
    from contextgem.public.concepts import _Concept
    from contextgem.public.paragraphs import Paragraph

from contextgem.internal.data_models import _LLMUsage
from contextgem.internal.llm_output_structs.aspect_structs import (
    _get_aspect_extraction_output_struct,
)
from contextgem.internal.llm_output_structs.concept_structs import (
    _get_concept_extraction_output_struct,
)
from contextgem.internal.loggers import logger
from contextgem.internal.typings.aliases import (
    AsyncCalsAndKwargs,
    ExtractedInstanceType,
    ReferenceDepth,
    SaTModelId,
    StandardSaTModelId,
    TextMode,
)


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


def _get_template(
    template_name: str,
    template_type: Literal["prompt", "system"] = "prompt",
    template_extension: Literal["j2", "txt"] = "j2",
) -> Template | str:
    """
    Retrieves a template based on the given parameters. The template is read from a specific file path determined
    by the `template_type` and its content is either processed as a Jinja2 template or returned as plain text based
    on the `template_extension`.

    :param template_name: The name of the template to be retrieved. This is the base name of the file without any
        extensions.
    :type template_name: str
    :param template_type: The type of the template, which determines the folder it is retrieved from. It can
        be either "prompt" or "system". Defaults to "prompt".
    :type template_type: Literal["prompt", "system"]
    :param template_extension: The file extension of the template, determining its format. Supported values
        are "j2" for Jinja2 templates and "txt" for plain text. Defaults to "j2".
    :type template_extension: Literal["j2", "txt"]
    :return: The loaded template, either as a compiled Jinja2 `Template` object for "j2" extensions or a plain
        string for "txt" extensions.
    :rtype: Template | str
    :raises NotImplementedError: If the `template_type` provided is unsupported.
    :raises NotImplementedError: If the `template_extension` provided is unsupported.
    """

    current_file = Path(__file__).resolve()
    project_root = current_file.parents[1]
    if template_type == "prompt":
        template_path = (
            project_root
            / "internal"
            / "prompts"
            / f"{template_name}.{template_extension}"
        )
    elif template_type == "system":
        template_path = (
            project_root
            / "internal"
            / "system"
            / f"{template_name}.{template_extension}"
        )
    else:
        raise NotImplementedError(f"Unknown template type: {template_type}")
    with open(template_path, encoding="utf-8") as file:
        template_text = file.read().strip()
        if not template_text:
            raise RuntimeError(
                f"Template file '{template_path}' is empty or contains only whitespace"
            )
    if template_extension == "j2":
        # Validate template text
        if not _are_prompt_template_brackets_balanced(template_text):
            raise RuntimeError("Prompt template brackets are not balanced.")
        if bool(re.search(r"(\r\n|\r|\n){3,}", template_text)):
            raise RuntimeError("Too many newlines in template.")
        template = _setup_jinja2_template(template_text)
    elif template_extension == "txt":
        template = template_text
    else:
        raise NotImplementedError(
            f"Unsupported template extension: {template_extension}"
        )
    return template


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
        cleaned = re.sub(
            r"[\x00-\x09\x0B-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202F\uFEFF]",
            "",
            text,
        )
    else:
        # Remove all control characters including newlines
        cleaned = re.sub(
            r"[\x00-\x1F\x7F-\x9F\u200B-\u200F\u2028-\u202F\uFEFF]", "", text
        )

    return cleaned.strip() if strip_text else cleaned


def _clean_text_for_llm_prompt(
    text: str, preserve_linebreaks: bool = True, strip_text: bool = True
) -> str:
    """
    Removes control characters and other problematic elements from text
    to make it suitable for LLM input.

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
        cleaned = re.sub(r"\r\n|\r", "\n", text)

        # Remove control characters except newlines
        cleaned = _clean_control_characters(
            cleaned, preserve_newlines=True, strip_text=strip_text
        )

        # Replace horizontal whitespace sequences (spaces and tabs) with a single space
        # while preserving linebreaks
        cleaned = re.sub(r"[ \t]+", " ", cleaned)

        # Remove extra blank lines (more than one consecutive newline)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    else:
        # Remove all control characters including newlines
        cleaned = _clean_control_characters(
            text, preserve_newlines=False, strip_text=strip_text
        )

        # Remove all whitespace sequences with a single space
        cleaned = re.sub(r"\s+", " ", cleaned)

    # We may want to preserve leading/trailing whitespace for markdown representation
    # of paragraphs, e.g. for indentation in lists.
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

    :param func: An async function (i.e., defined with "async def").
                 It must accept keyword arguments that match each dict in data_list.
    :param data_list: A list of dictionaries to pass as keyword arguments to `func`.
    :return: A list of results corresponding to each input dictionary in `data_list`.
             If a call raises an exception, that slot in the list is None.
    """

    if not data_list:
        logger.warning("No data to process in tasks.")
        return []

    # Prepare a list to hold results (same size as data_list)
    results = [None] * len(data_list)

    async def worker(index: int, kwargs: dict[str, Any]) -> None:
        try:
            results[index] = await func(**kwargs)
        except Exception as e:
            logger.error(f"Error in worker {index}: {e}")
            results[index] = None

    # Create one worker task per item in data_list
    tasks = [asyncio.create_task(worker(i, data)) for i, data in enumerate(data_list)]

    # Wait for all tasks to finish
    logger.debug(f"Starting tasks ({len(tasks)})")
    await asyncio.gather(*tasks)
    logger.debug(f"Completed all tasks ({len(tasks)})")
    return results


@contextmanager
def _suppress_litellm_pydantic_warnings_context() -> Generator[None, None, None]:
    """
    Context manager that suppresses Pydantic deprecation and serialization warnings.

    This context manager temporarily suppresses DeprecationWarnings and UserWarnings
    from the Pydantic module. These warnings are typically generated by litellm's
    internal Pydantic usage and are not actionable by users.

    The context manager uses Python's warnings.catch_warnings() to create a temporary
    warning filter that ignores the specified warnings only within its scope.

    :return: A generator that yields None and expects no values to be sent back.
    :rtype: Generator[None, None, None]
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="pydantic"
        )
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            module="pydantic",
            message=r"^Pydantic serializer warnings",
        )
        yield


def _suppress_litellm_pydantic_warnings(func: F) -> F:
    """
    Suppresses warnings related to Pydantic deprecation and serialization
    in litellm>1.71.1 (latest available version as of 2025-07-10)

    This decorator wraps both synchronous and asynchronous functions to suppress
    Pydantic deprecation and serialization warnings that originate from litellm's
    internal usage. These warnings are not actionable by users and should not
    be displayed.

    Suppression rationale:
    - The warnings originate from litellm's internal pydantic usage, not user code
    - These are dependency warnings that users cannot fix and should not see

    :param func: The function to wrap with warning suppression. Can be either
        synchronous or asynchronous.
    :type func: F
    :return: The wrapped function with the same signature as the input function,
        but with Pydantic deprecation and serialization warnings suppressed
        during execution.
    :rtype: F

    TODO: Remove deprecation-related suppression when deprecation warnings are fixed
    in a future litellm release.

    TODO: Remove serialization-related suppression when the related issue is fixed
    in a future litellm release: https://github.com/BerriAI/litellm/issues/11759
    """

    @functools.wraps(func)
    async def _async_wrapper(*args, **kwargs):
        with _suppress_litellm_pydantic_warnings_context():
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def _sync_wrapper(*args, **kwargs):
        with _suppress_litellm_pydantic_warnings_context():
            return func(*args, **kwargs)

    # Safe cast: tell type checker that functools.wraps preserves the original function's type.
    # The wrapper functions maintain the same signature as the original function.
    return cast(
        F, _async_wrapper if inspect.iscoroutinefunction(func) else _sync_wrapper
    )


@_suppress_litellm_pydantic_warnings
async def _run_async_calls(
    cals_and_kwargs: AsyncCalsAndKwargs,
    use_concurrency: bool,
) -> list[Any]:
    """
    Runs a series of asynchronous callables with the provided arguments, optionally using
    concurrency for executing them. If concurrency is enabled, all callables will be scheduled
    and awaited simultaneously; otherwise, they will be awaited sequentially.

    "Sequential" mode is "stop on first error," while "concurrent" mode is "cancel other tasks on first error."
    Returned results maintain the same index order as cals_and_kwargs.

    :param cals_and_kwargs: A list of tuples where each tuple contains an asynchronous
        callable and a dictionary of keyword arguments to be passed to the callable.
    :param use_concurrency: A boolean flag indicating whether all tasks should be executed
        concurrently or sequentially.
    :return: A list containing the results of each asynchronous task. The results are
        ordered according to the order of the input list.
    """

    if use_concurrency:
        tasks = [asyncio.create_task(i[0](**i[1])) for i in cals_and_kwargs]
        return await asyncio.gather(*tasks)
    else:
        results = []
        for cal, kwargs in cals_and_kwargs:
            result = await cal(**kwargs)
            results.append(result)
        return results


@_suppress_litellm_pydantic_warnings
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
            return future.result()
    else:
        # No running loop, so just run it directly.
        return asyncio.run(coro)


def _llm_call_result_is_valid(res: tuple[Any, _LLMUsage] | None) -> bool:
    """
    Determines whether the result from a LLM call is valid.

    Determines if a result exists and contains validated data as a result of an LLM processing function.
    Result may be one of the following:
    - None - if LLM processing failed (e.g. due to RateLimitError, or another uncaught exception).
        In this case, token and cost calculation is not possible.
    - None, _LLMUsage - if LLM processing failed due to a caught exception or failed data validation,
        e.g. invalid data returned by the LLM. In this case, token and cost calculation is still possible.
    - Any, _LLMUsage - if LLM processing was successful and result is validated. In this case,
        token and cost calculation are performed as normal.

    :param res: The result to check, represented as a tuple where the first element is the relevant
        result value, and the second element is LLM usage stats dict that follows the _LLMUsage pattern,
        or None if the result is not received or failed validation.
    :type res: tuple[Any, _LLMUsage] | None
    :return: A boolean indicating whether the result is successfully received and validated.
    :rtype: bool
    """
    return not (res is None or res[0] is None)


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


def _parse_llm_output_as_json(
    output_str: str | dict | list | None,
) -> dict | list | None:
    """
    Parses the provided LLM-generated output into a JSON-compatible Python object.

    This function attempts to parse the `output_str` string into a dictionary or a list.
    If the string cannot be parsed (either due to a JSON decoding error or an incompatible type),
    it returns `None`. The function also includes handling for strings that include a surrounding
    ` ```json` code block, removing them before parsing.

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

    try:
        return json.loads(output_str)

    except json.JSONDecodeError:
        try:
            # Handle markdown code blocks using regex

            answer = output_str.strip()

            # Pattern to match content between ```json
            # (at string start) and ``` (at string end) markers
            json_block_pattern = r"^```json\s*([\s\S]*?)\s*```$"
            match = re.match(json_block_pattern, answer)

            if match:
                # Get the content between the markers
                answer = match.group(1).strip()

            return json.loads(answer)
        except json.JSONDecodeError:
            return None


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
        logger.error(f"Error when using validation model {validation_model}: {e}")
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
        - A local path to a SaT model directory (as a string or Path object)

    :return:
        An instance of the SaT model associated with the given `model_id`.

    :raises ValueError:
        If the provided path doesn't exist or is not a directory.
    :raises RuntimeError:
        If the provided path exists but does not contain a valid SaT model.
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

    # Attempt to load the model
    try:
        model = SaT(model_id)
        logger.info("SaT model loaded successfully.")
        return model
    except Exception as e:
        if is_local_path:
            # If it's a local path that exists but isn't a valid SaT model
            logger.error(f"Failed to load SaT model from path '{model_id}': {e}")
            raise RuntimeError(
                f"The directory at '{model_id}' exists but does not contain a valid SaT model. "
                f"Error: {e}"
            ) from e
        else:
            # For standard model IDs or other errors
            logger.error(f"Failed to load SaT model '{model_id}': {e}")
            raise


def _group_instances_by_fields(
    fields: list[str], instances: list[Aspect] | list[_Concept]
) -> list[list[Aspect] | list[_Concept]]:
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


def _check_paragraphs_match_in_text(
    paragraphs: list[Paragraph], document_text: str, text_mode: TextMode
) -> list[Paragraph]:
    """
    Check that all relevant paragraph texts exist in the given document text.

    :param paragraphs: List of paragraph objects to check
    :param document_text: The document text to search in
    :param text_mode: Type of text being checked ('raw' or 'markdown')
    :return: List of unmatched paragraphs
    """
    if text_mode == "raw":
        return [p for p in paragraphs if p.raw_text not in document_text]
    elif text_mode == "markdown":
        return [p for p in paragraphs if p._md_text and p._md_text not in document_text]
    else:
        raise ValueError(f"Unknown text_mode: {text_mode}")


def _check_paragraphs_ordering_in_text(
    paragraphs: list[Paragraph], document_text: str, text_mode: TextMode
) -> None:
    """
    Check that paragraphs are ordered according to their appearance in document text.
    Handles cases where paragraphs may have duplicate text content.

    :param paragraphs: List of paragraph objects to check ordering for
    :param document_text: The document text to check ordering in
    :param text_mode: Type of text being checked ('raw' or 'markdown')
    :raises ValueError: If paragraphs are not ordered correctly
    """
    if len(paragraphs) <= 1:
        return

    current_search_pos = 0
    for i in range(len(paragraphs) - 1):
        current_para_text = (
            paragraphs[i].raw_text if text_mode == "raw" else paragraphs[i]._md_text
        )
        next_para_text = (
            paragraphs[i + 1].raw_text
            if text_mode == "raw"
            else paragraphs[i + 1]._md_text
        )

        if not current_para_text or not next_para_text:
            continue

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
                    f"Paragraphs are not ordered according to their appearance "
                    f"in the document's {text_mode} text."
                )
