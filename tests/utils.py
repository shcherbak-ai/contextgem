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
Module defining utility functions and classes for the tests.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import time
import warnings
from collections.abc import Callable
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pytest
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv
from pydantic import ValidationError

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.documents import _POSITION_CACHE_KEYS
from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.data_models import (
    _ModelCost,
    _ModelCostOutputContainer,
    _ModelUsage,
    _ModelUsageOutputContainer,
)
from contextgem.internal.loggers import logger
from contextgem.internal.utils import (
    _are_prompt_template_brackets_balanced,
    _are_prompt_template_xml_tags_balanced,
    _group_instances_by_fields,
    _is_json_serializable,
)
from contextgem.public import (
    Aspect,
    Document,
    DocumentDecisionModel,
    DocumentLLM,
    DocumentModelGroup,
    Image,
    ModelPricing,
    StringConcept,
    create_image,
)
from tests.conftest import VCR_DUMMY_ENDPOINT_PREFIX, VCR_REDACTION_MARKER


# A global VCR recordings counter
vcr_new_recording_count = 0


VCR_FILTER_HEADERS = {
    "api-key",
    "apim-request-id",
    "authorization",
    "azureml-destination-deployment",
    "azureml-destination-endpoint",
    "azureml-destination-model-group",
    "azureml-destination-region",
    "azureml-model-deployment",
    "azureml-model-group",
    "azureml-model-session",
    "azureml-served-by-cluster",
    "CF-RAY",
    "cookie",
    "Date",
    "host",
    "openai-organization",
    "openai-project",
    "set-cookie",
    "Set-Cookie",
    "uri",
    "x-aml-cluster",
    "x-ms-client-request-id",
    "x-ms-deployment-name",
    "x-ms-inference-retry-history",
    "x-ms-region",
    "x-ratelimit-key",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ms-rai-invoked",
    "x-request-id",
    "x-typesafe-request-id",
    "x-stainless-os",
    "x-stainless-runtime-version",
}
assert len(set(VCR_FILTER_HEADERS)) == len(VCR_FILTER_HEADERS)

# What a validation error of a model or of a model group reports in place of
# an input that is, or may be, an API key
REDACTED_VALUE = "**********"

# Start of the URI of a request to a Microsoft Foundry resource: the host has
# the name of the resource
_FOUNDRY_RESOURCE_URI_START_PATTERN = re.compile(
    r"^https://[^/?#]+\.services\.ai\.azure\.com(?::\d+)?/", flags=re.IGNORECASE
)


def vcr_count_recording(response):
    """
    Counts VCR recordings by incrementing the global counter.

    :param response: The VCR response object.
    :return: The unmodified response object.
    """
    global vcr_new_recording_count
    vcr_new_recording_count += 1
    return response


def vcr_before_record_request(request):
    """
    Processes VCR requests before recording to redact sensitive information.

    Redacts Azure OpenAI domain and deployment names from request URIs
    to protect sensitive configuration details in VCR cassettes, and the host
    of a Microsoft Foundry resource, which has the name of the resource.

    VCR also applies the function to a request before the request is matched
    with the recorded ones, i.e. a request to a resource is replayed whatever
    the resource is, including the dummy one that is set when no ``.env`` file
    is found.

    :param request: The VCR request object to process.
    :return: The modified request object with redacted URI.
    """
    # Redact the host of the Microsoft Foundry resource
    request.uri = _FOUNDRY_RESOURCE_URI_START_PATTERN.sub(
        VCR_DUMMY_ENDPOINT_PREFIX, request.uri
    )

    # Redact the Azure OpenAI domain & deployment name
    path_parts = request.uri.split("/openai/deployments/")
    if len(path_parts) > 1:
        # Split deployment name and rest of path
        deployment_and_rest = path_parts[1]
        deployment_parts = deployment_and_rest.split("/", 1)

        deployment_name = f"{VCR_REDACTION_MARKER}-DEPLOYMENT"

        if len(deployment_parts) > 1:
            # We have a deployment name and a path after it
            rest_of_path = deployment_parts[1]
            request.uri = f"{VCR_DUMMY_ENDPOINT_PREFIX}openai/deployments/{deployment_name}/{rest_of_path}"
        else:
            # Just a deployment name
            request.uri = (
                f"{VCR_DUMMY_ENDPOINT_PREFIX}openai/deployments/{deployment_name}"
            )
    return request


def redact_home_dir(text: str) -> str:
    """
    Redacts the name of the home directory of the current user in the file paths
    found in a text, e.g. in the paths of the model files that a local LLM server
    returns with the details of a model.

    A path is matched with slashes or backslashes as separators, the latter
    escaped any number of times, as in a JSON string.

    :param text: The text to redact.
    :type text: str
    :return: The text with the name of the home directory replaced
        by the redaction marker.
    :rtype: str
    """
    home_dir = Path.home()
    dir_names = [part.strip("\\/") for part in home_dir.parts if part.strip("\\/")]
    if not dir_names:
        return text
    *parent_dir_names, home_dir_name = dir_names
    separator = r"[\\/]+"
    # A Windows path starts with a drive, any other path with a separator
    parents = ("" if home_dir.drive else separator) + "".join(
        re.escape(name) + separator for name in parent_dir_names
    )
    # The name of the home directory is followed by a separator, or ends the path
    pattern = rf"(?P<parents>{parents}){re.escape(home_dir_name)}(?=[\\/\"'\s]|$)"
    return re.sub(
        pattern, rf"\g<parents>{VCR_REDACTION_MARKER}", text, flags=re.IGNORECASE
    )


def vcr_before_record_response(response):
    """
    Processes VCR responses before recording to redact sensitive information.

    Redacts headers and response body content that might contain sensitive
    information like API keys, request IDs, and other identifiable data, including
    the name of the home directory of the current user in local file paths.

    :param response: The VCR response object to process.
    :return: The modified response object with redacted sensitive data.
    """
    # Redact headers (use actual header key from response to handle case differences)
    headers = response.get("headers", {})
    filter_lower = {h.lower() for h in VCR_FILTER_HEADERS}
    for key in list(headers):
        if key.lower() in filter_lower:
            headers[key] = [VCR_REDACTION_MARKER]
    # Redact body
    if response["body"]["string"]:
        try:
            body = json.loads(response["body"]["string"])
            if "id" in body:
                body["id"] = f"chatcmpl-{VCR_REDACTION_MARKER}"
            response["body"]["string"] = redact_home_dir(
                json.dumps(body, ensure_ascii=False)
            )
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            # Skip redaction if response body is not valid JSON or lacks expected structure
            logger.debug(
                "Could not redact response body - not valid JSON or unexpected structure"
            )
    vcr_count_recording(response)
    return response


@lru_cache(maxsize=1)
def get_test_document_text(lang: Literal["en", "ua", "zh"] = "en") -> str:
    """
    Returns the raw text of a test NDA document.

    :param lang: The language of the test document. Defaults to "en".
    :type lang: Literal["en", "ua", "zh"]
    :return: The raw text content of the test document.
    :rtype: str
    """
    project_root = get_project_root_path()
    test_doc_fname = lang + "_nda_with_anomalies.txt"
    test_doc_fpath = project_root / "tests" / "ndas" / test_doc_fname
    test_doc_text = read_text_file(str(test_doc_fpath))
    return test_doc_text


def read_text_file(filepath: str) -> str:
    """
    Reads the content of a text file and returns it as a string.

    :param filepath: The path to the text file to be read.
    :type filepath: str
    :return: The content of the text file as a string.
    :rtype: str
    """
    with open(filepath, encoding="utf-8") as file:
        return file.read().strip()


def get_project_root_path() -> Path:
    """
    Retrieves the project root path.

    :return: The project root path.
    :rtype: Path
    """
    current_file = Path(__file__).resolve()
    return current_file.parents[1]


def get_test_img(
    img_filename: str,
    img_folder: str,
) -> Image:
    """
    Retrieves a test image for a given image filename and constructs an `Image`
    object using the create_image utility function.

    This function uses the create_image utility which automatically determines the
    MIME type using PIL and handles base64 encoding.

    :param img_filename: The filename of the image to be retrieved.
    :type img_filename: str
    :param img_folder: The folder under `tests/images/` containing the image.
    :type img_folder: str
    :return: An `Image` object containing the base64-encoded image data and
        the automatically detected MIME type.
    :rtype: Image

    :raises FileNotFoundError: If the image file does not exist.
    :raises ValueError: If the image format is not supported.
    :raises OSError: If the image cannot be opened or processed.
    """
    project_root = get_project_root_path()
    test_img_path = project_root / "tests" / "images" / img_folder / img_filename
    return create_image(test_img_path)


def model_dict_without_position_caches(instance: _InstanceBase) -> dict:
    """
    Returns the instance's raw ``__dict__`` with document position lookup cache
    entries excluded.

    The position caches are derived state stored as plain instance-dict entries
    (not model fields or private attributes): pydantic model equality ignores
    them, ``to_dict()`` never serializes them, and they may or may not be
    present depending on whether position lookups have been performed. They are
    therefore excluded from raw ``__dict__`` comparisons, which assert that all
    *model* state survives serialization round-trips and cloning.

    :param instance: The instance whose ``__dict__`` to filter.
    :type instance: _InstanceBase
    :return: The instance's ``__dict__`` without position cache entries.
    :rtype: dict
    """
    return {k: v for k, v in instance.__dict__.items() if k not in _POSITION_CACHE_KEYS}


def remove_file(filepath):
    """
    Removes a file from the filesystem if it exists.

    Attempts to delete the specified file and logs success. If the file
    doesn't exist, the operation is silently ignored.

    :param filepath: Path to the file to remove.
    :type filepath: str
    """
    try:
        os.remove(filepath)
        logger.debug(f"File '{filepath}' has been removed.")
    except FileNotFoundError:
        pass


class TestUtils:
    """
    Class with tests-related utility functions.
    """

    def check_instance_serialization_and_cloning(self, instance: _InstanceBase) -> None:
        """
        Check the serialization and deserialization functionality of an instance.

        :param instance: The instance of `InstanceBase` to be tested.
        :type instance: _InstanceBase
        :return: None
        """
        attrs_to_recheck = [
            "aspects",
            "concepts",
            "paragraphs",
            "reference_paragraphs",
            "sentences",
            "images",
            "examples",
            "structure",
            "rating_scale",
            "labels",
            "_unique_id",
            "_extracted_items",
            "_reference_paragraphs",
            "_reference_sentences",
            "_is_processed",
            "_messages",
        ]

        # To / from dict
        instance_dict = instance.to_dict()
        new_instance = instance.__class__.from_dict(instance_dict)
        assert model_dict_without_position_caches(
            new_instance
        ) == model_dict_without_position_caches(instance)
        assert new_instance == instance
        for attr_name in attrs_to_recheck:
            if (
                hasattr(instance, attr_name)
                or attr_name in instance.__private_attributes__
            ):
                assert getattr(instance, attr_name) == getattr(new_instance, attr_name)

        # To / from json
        instance_json = instance.to_json()
        new_instance = instance.__class__.from_json(instance_json)
        assert model_dict_without_position_caches(
            new_instance
        ) == model_dict_without_position_caches(instance)
        assert new_instance == instance
        for attr_name in attrs_to_recheck:
            if (
                hasattr(instance, attr_name)
                or attr_name in instance.__private_attributes__
            ):
                assert getattr(instance, attr_name) == getattr(new_instance, attr_name)

        # Saving to disk / loading from disk
        disk_path = os.path.join(
            get_project_root_path(), "tests", f"instance_{str(uuid4())}.json"
        )
        instance.to_disk(disk_path)
        new_instance = instance.__class__.from_disk(disk_path)
        assert model_dict_without_position_caches(
            new_instance
        ) == model_dict_without_position_caches(instance)
        assert new_instance == instance
        for attr_name in attrs_to_recheck:
            if (
                hasattr(instance, attr_name)
                or attr_name in instance.__private_attributes__
            ):
                assert getattr(instance, attr_name) == getattr(new_instance, attr_name)
        remove_file(disk_path)

        # Cloning
        instance_clone = instance.clone()
        assert model_dict_without_position_caches(
            instance_clone
        ) == model_dict_without_position_caches(instance)
        assert instance_clone == instance
        for attr_name in attrs_to_recheck:
            if (
                hasattr(instance, attr_name)
                or attr_name in instance.__private_attributes__
            ):
                assert getattr(instance, attr_name) == getattr(new_instance, attr_name)
        with pytest.raises(NotImplementedError):
            instance.model_copy(deep=True)

        # Custom data serialization
        if hasattr(instance, "custom_data"):
            self.check_custom_data_json_serializable(instance)

    def _check_deserialized_llm_config_eq(
        self, instance: DocumentLLM | DocumentModelGroup
    ) -> None:
        """
        Checks if the LLM config of a deserialized DocumentLLM/DocumentModelGroup is equal to
        the LLM config of the original instance.

        :param instance: The original DocumentLLM or DocumentModelGroup instance to be checked.
        :type instance: DocumentLLM | DocumentModelGroup
        :return: None
        """
        # From json
        instance_json = instance.to_json()
        if isinstance(instance, DocumentLLM):
            new_instance = DocumentLLM.from_json(instance_json)
            assert instance._eq_deserialized_llm_config(new_instance)
        else:  # DocumentModelGroup
            new_instance = DocumentModelGroup.from_json(instance_json)
            assert instance._eq_deserialized_llm_config(new_instance)

        # From dict
        instance_dict = instance.to_dict()
        if isinstance(instance, DocumentLLM):
            new_instance = DocumentLLM.from_dict(instance_dict)
            assert instance._eq_deserialized_llm_config(new_instance)
        else:  # DocumentModelGroup
            new_instance = DocumentModelGroup.from_dict(instance_dict)
            assert instance._eq_deserialized_llm_config(new_instance)

        # From disk
        disk_path = os.path.join(
            get_project_root_path(), "tests", f"instance_{str(uuid4())}.json"
        )
        instance.to_disk(disk_path)
        if isinstance(instance, DocumentLLM):
            new_instance = DocumentLLM.from_disk(disk_path)
            assert instance._eq_deserialized_llm_config(new_instance)
        else:  # DocumentModelGroup
            new_instance = DocumentModelGroup.from_disk(disk_path)
            assert instance._eq_deserialized_llm_config(new_instance)
        remove_file(disk_path)

        # Check that pydantic-specific methods are disabled
        with pytest.raises(NotImplementedError):
            instance.model_dump()
        with pytest.raises(NotImplementedError):
            instance.model_dump_json()

    def check_usage(self, llm: DocumentModelGroup | DocumentLLM) -> None:
        """
        Evaluates the usage stats for an LLM or a group of LLMs
        (when using DocumentModelGroup). Ensures that the usage information is recorded
        properly and validates the structure and presence of related data.

        :param llm: A DocumentLLM or DocumentModelGroup instance to assess usage
            data. Must be an instance of either DocumentLLM or DocumentModelGroup.
        :return: None
        """

        usages = llm.get_usage()
        assert isinstance(usages, list)

        # For model group: At least one LLM in group must have recorded tokens usage
        # For individual LLM: Either the primary model or its fallback model must have recorded usage
        usage_input_used = False
        usage_output_used = False
        for usage_item in usages:
            assert isinstance(usage_item, _ModelUsageOutputContainer)
            assert hasattr(usage_item, "model") and usage_item.model
            assert hasattr(usage_item, "role")  # None for an LLM without a role
            assert hasattr(usage_item, "is_fallback")
            assert hasattr(usage_item, "usage") and usage_item.usage
            assert isinstance(usage_item.usage, _ModelUsage)
            if usage_item.usage.input > 0:
                assert usage_item.usage.calls
                for call in usage_item.usage.calls:
                    self.check_rendered_prompt(call.prompt)
                usage_input_used = True
            if usage_item.usage.output > 0:
                assert usage_item.usage.calls
                for call in usage_item.usage.calls:
                    self.check_rendered_prompt(call.prompt)
                usage_output_used = True
        assert usage_input_used and usage_output_used

    @staticmethod
    def check_cost(llm: DocumentModelGroup | DocumentLLM) -> None:
        """
        Checks the cost details of a LLM or a group of LLMs, ensuring that the
        cost calculation and structure meet the required conditions.

        :param llm: The LLM or group of LLMs whose processing cost is to be checked.
        :type llm: DocumentModelGroup | DocumentLLM
        :return: None
        :rtype: None
        """

        costs = llm.get_cost()
        assert isinstance(costs, list)

        # For model group: At least one LLM in group must have recorded costs
        # For individual LLM: Either the primary model or its fallback model must have recorded costs
        cost_input_updated = False
        cost_output_updated = False
        for cost_item in costs:
            assert isinstance(cost_item, _ModelCostOutputContainer)
            assert hasattr(cost_item, "model") and cost_item.model
            assert hasattr(cost_item, "role")  # None for an LLM without a role
            assert hasattr(cost_item, "is_fallback")
            assert hasattr(cost_item, "cost") and cost_item.cost
            assert isinstance(cost_item.cost, _ModelCost)
            if cost_item.cost.input > Decimal("0"):
                cost_input_updated = True
            if cost_item.cost.output > Decimal("0"):
                cost_output_updated = True
        assert cost_input_updated and cost_output_updated

    @staticmethod
    def check_rendered_prompt(prompt: str) -> None:
        """
        Checks for formatting and consistency of a given rendered prompt.

        :param prompt: The text of the rendered prompt to be validated.
        :type prompt: str
        :return: None
        :rtype: None
        :raises AssertionError: If the rendered prompt does not pass validation.
        """

        assert not bool(re.search(r"(\r\n|\r|\n){3,}", prompt))
        assert _are_prompt_template_brackets_balanced(prompt)
        assert _are_prompt_template_xml_tags_balanced(prompt)

    @staticmethod
    def check_extra_data_in_extracted_items(
        obj: Document | Aspect | _Concept,
    ) -> None:
        """
        Checks and validates extra data presence in the extracted items of the given object.

        :param obj: The target object which is either a `Document`, `Aspect`, or `Concept`
                    instance. It may contain extracted items or concepts to be validated.
        :type obj: Document | Aspect

        :return: None
        """
        items_to_check_lists = []
        if isinstance(obj, Aspect | _Concept):
            items_to_check_lists.append([obj])
        if isinstance(obj, Document | Aspect):
            items_to_check_lists.append(obj.aspects)
            items_to_check_lists.append(obj.concepts)
        for i in itertools.chain.from_iterable(items_to_check_lists):
            logger.debug(
                f"Checking extracted items for {i.__class__.__name__} ({len(i.extracted_items)})"
            )
            if i.add_justifications:
                assert all(x.justification for x in i.extracted_items)
            else:
                assert not any(x.justification for x in i.extracted_items)
            if hasattr(i, "reference_depth"):
                if (
                    hasattr(i, "add_references")
                    and i.add_references
                    or isinstance(i, Aspect)
                ):
                    assert all(
                        x.reference_paragraphs and not x.reference_sentences
                        for x in i.extracted_items
                        if i.reference_depth == "paragraphs"
                    )
                    assert all(
                        x.reference_paragraphs and x.reference_sentences
                        for x in i.extracted_items
                        if i.reference_depth == "sentences"
                    )
                else:
                    assert not any(
                        x.reference_paragraphs or x.reference_sentences
                        for x in i.extracted_items
                    )

    @staticmethod
    def check_item_probabilities(
        item: _ExtractedItem,
        expected_probabilities: dict[bool | int | str, float] | None,
    ) -> None:
        """
        Checks that the probabilities of the extracted item are exactly the expected
        ones, including the types and the order of the options, and that each
        probability is a float.

        The comparison of dicts alone does not cover that: it ignores the order
        of the keys and does not tell ``True`` from ``1``, or ``1`` from ``1.0``.

        :param item: The extracted item to check.
        :type item: _ExtractedItem
        :param expected_probabilities: The expected options mapped to their
            probabilities, or None if the item is expected to have no probabilities.
        :type expected_probabilities: dict[bool | int | str, float] | None
        :return: None
        """
        if expected_probabilities is None:
            assert item.probabilities is None
            return
        assert isinstance(item.probabilities, dict)
        assert [
            (type(option), option, type(probability), probability)
            for option, probability in item.probabilities.items()
        ] == [
            (type(option), option, float, probability)
            for option, probability in expected_probabilities.items()
        ]

    @staticmethod
    def get_api_key_fragments(api_key: str) -> set[str]:
        """
        Returns the fragments of the API key that the key is looked for by,
        so that a part of the key is found as well as the whole key.

        :param api_key: The API key.
        :type api_key: str
        :return: The fragments of the key.
        :rtype: set[str]
        """
        fragment_length = 6
        key_fragments = {
            api_key[start : start + fragment_length]
            for start in range(len(api_key) - fragment_length + 1)
        }
        assert key_fragments, "The key is too short to be looked for by its fragments"
        return key_fragments

    @staticmethod
    def check_validation_error_hides_inputs(
        error: ValidationError, api_key: str, expected_texts: list[str]
    ) -> None:
        """
        Checks that the text of a validation error of a model or of a model group
        does not show the inputs of the validation, i.e. no part of the API key
        that is among them, while it still tells what is invalid and why.

        The key is looked for by its fragments: an input is cut in the middle in
        the text of an error, so that only the beginning or the end of the key
        may be shown there.

        :param error: The validation error to check.
        :type error: ValidationError
        :param api_key: The API key that is among the inputs of the validation.
        :type api_key: str
        :param expected_texts: The texts that the error is to have, e.g. the name
            of the invalid param and the reason why it is invalid.
        :type expected_texts: list[str]
        :return: None
        """
        key_fragments = TestUtils.get_api_key_fragments(api_key)
        for text in [str(error), repr(error)]:
            for input_marker in ["input_value", "input_type"]:
                assert input_marker not in text, f"The error shows an input: {text}"
            shown_key_fragments = sorted(i for i in key_fragments if i in text)
            assert not shown_key_fragments, (
                f"The error shows parts of the key ({shown_key_fragments}): {text}"
            )
            for expected_text in expected_texts:
                assert expected_text in text, (
                    f"The error does not have the text `{expected_text}`: {text}"
                )

    @staticmethod
    def check_validation_error_omits_model_keys(
        error: ValidationError, api_key: str
    ) -> None:
        """
        Checks that a validation error does not give the API key that is among
        the inputs of the validation: neither in what its ``errors()`` method
        returns, when printed, nor in what its ``json()`` method writes out.
        Both methods are called with their defaults, i.e. the inputs are included.

        This holds for the key of a model that is among the inputs, which is
        printed and written out without its key, and for a key that is among
        the inputs as it is passed, e.g. a key of a wrong type or one of the params
        that a model is created with, in place of which the error reports
        ``REDACTED_VALUE``. No other error, which would report the key, is to be
        attached to the error.

        :param error: The validation error to check.
        :type error: ValidationError
        :param api_key: The API key that is among the inputs of the validation.
        :type api_key: str
        :return: None
        """
        key_fragments = TestUtils.get_api_key_fragments(api_key)
        returned_by_methods = {
            "errors()": repr(error.errors()),
            "json()": error.json(),
            "json(indent=2)": error.json(indent=2),
        }
        for method, text in returned_by_methods.items():
            shown_key_fragments = sorted(i for i in key_fragments if i in text)
            assert not shown_key_fragments, (
                f"`{method}` of the error gives parts of the key "
                f"({shown_key_fragments}): {text}"
            )
        # The inputs are included, i.e. the key is not left out together with them
        assert all("input" in details for details in error.errors())
        assert all("input" in details for details in json.loads(error.json()))
        # The error is not raised from the error that reports the key as it is
        # passed, nor while that error is handled
        assert error.__cause__ is None
        assert error.__context__ is None

    def check_model_validation_errors_hide_inputs(
        self,
        model_class: type[DocumentLLM] | type[DocumentDecisionModel],
        model_params: dict[str, Any],
        api_key: str,
        created_model_is_input: bool,
        tmp_path: Path,
    ) -> None:
        """
        Checks that the errors of the validation of a model of the given class do
        not give the API key of the model, which is among the inputs of
        the validation: when a model is created, when a value is assigned to its
        param, and when it is deserialized, for a fallback model as well.

        The text of such an error does not show the inputs at all, and its
        ``errors()`` and ``json()`` methods give ``REDACTED_VALUE`` in place of
        the key that is among the inputs as it is passed: a key that is rejected
        itself, a value passed in a param that the model does not have, and the key
        among all the params that the model is created with. The other inputs are
        given as they are, including a model, which holds its key as an attribute
        and is printed and written out without it.

        :param model_class: The class of the model.
        :type model_class: type[DocumentLLM] | type[DocumentDecisionModel]
        :param model_params: The valid params to create a model of the class with,
            without the API key.
        :type model_params: dict[str, Any]
        :param api_key: The API key of the model.
        :type api_key: str
        :param created_model_is_input: Whether the input of a check that involves
            several params is the model itself, when such a check fails for a model
            that is being created with these ``model_params``. If not, the input is
            all the params that the model is created with, i.e. the key as it is
            passed. For a model that a value is assigned to, it is always the model.
        :type created_model_is_input: bool
        :param tmp_path: The directory for the files that a model is loaded from.
        :type tmp_path: Path
        :return: None
        """

        def create_model(
            key_first: bool = True, **params: Any
        ) -> DocumentLLM | DocumentDecisionModel:
            """
            Creates a model with the API key as its first or its last param.

            :param key_first: Whether the key is the first param, or the last one.
            :type key_first: bool
            :param params: The params of the model that are added to the valid ones.
            :type params: Any
            :return: The model.
            :rtype: DocumentLLM | DocumentDecisionModel
            """
            params = {**model_params, **params}
            key = params.pop("api_key", api_key)
            if key_first:
                return model_class(api_key=key, **params)
            return model_class(**params, api_key=key)

        def check_error(error: ValidationError, expected_texts: list[str]) -> Any:
            """
            Checks that the validation error, which is to have a single invalid
            input, does not give the key: neither in its text, which shows no
            inputs, nor in what its ``errors()`` and ``json()`` methods return.

            :param error: The validation error.
            :type error: ValidationError
            :param expected_texts: The texts that the error is to have.
            :type expected_texts: list[str]
            :return: The invalid input that the error reports.
            :rtype: Any
            """
            self.check_validation_error_hides_inputs(error, api_key, expected_texts)
            self.check_validation_error_omits_model_keys(error, api_key)
            (error_details,) = error.errors()
            return error_details["input"]

        def check_model_is_input(reported_input: Any) -> None:
            """
            Checks that the input that a validation error reports is the model,
            which is reported as it is, i.e. holds the key as its attribute.

            :param reported_input: The input that the error reports.
            :type reported_input: Any
            """
            assert isinstance(reported_input, model_class)
            assert reported_input.api_key == api_key

        pricing = ModelPricing(input_per_1m_tokens=1.0, output_per_1m_tokens=2.0)
        # A key of a wrong type, e.g. a tuple that a trailing comma makes of the key
        invalid_key = (api_key,)
        fallback_model = create_model(is_fallback=True)
        fallback_of_other_role = create_model(is_fallback=True, role="other")

        # What the errors tell of the invalid params
        timeout_error = ["timeout", "Input should be a valid integer"]
        key_error = ["api_key", "Input should be a valid string"]
        pricing_error = ["`pricing_details` cannot be set together with `auto_pricing"]
        own_fallbacks_error = ["cannot have its own fallback"]
        fallback_role_error = ["must have the same role `None` as the main one"]
        same_config_error = ["must not have the exact same config params as the main"]

        # A model that is created with invalid params. The last item of a case is
        # the input that the error reports: the invalid value as it is, or
        # the redacted one if the value is passed as the key or in a param that
        # the model does not have. The input of a check that involves several
        # params (`all_params`) is all the params that the model is created with,
        # which are reported with the key redacted, unless it is the model (see
        # `created_model_is_input`).
        all_params = object()
        invalid_params: list[tuple[dict[str, Any], list[str], Any]] = [
            ({"timeout": "x"}, timeout_error, "x"),
            ({"api_key": invalid_key}, key_error, REDACTED_VALUE),
            ({"api_key": [api_key]}, key_error, REDACTED_VALUE),
            # A value of a wrong type passed as the key, which has no key in it
            ({"api_key": 12345}, key_error, REDACTED_VALUE),
            # The key passed in a param that the model does not have. Any value of
            # such a param is redacted, as it is not known to be something else,
            # unless there is no value.
            (
                {"apikey": api_key},
                ["apikey", "Extra inputs are not permitted"],
                REDACTED_VALUE,
            ),
            ({"no_such_param": 0.5}, ["no_such_param", "Extra inputs"], REDACTED_VALUE),
            ({"no_such_param": None}, ["no_such_param", "Extra inputs"], None),
            (
                {"pricing_details": pricing, "auto_pricing": True},
                pricing_error,
                all_params,
            ),
            # A fallback model that is invalid itself, one that has a fallback model
            # of its own, and the ones that are not valid for the model that they
            # are passed to
            ({"is_fallback": True, "timeout": "x"}, timeout_error, "x"),
            (
                {"is_fallback": True, "fallback_models": [fallback_model]},
                own_fallbacks_error,
                all_params,
            ),
            (
                {"fallback_models": [fallback_of_other_role]},
                fallback_role_error,
                all_params,
            ),
            ({"fallback_models": [fallback_model]}, same_config_error, all_params),
        ]
        for params, expected_texts, expected_input in invalid_params:
            for key_first in [True, False]:
                with pytest.raises(ValidationError) as exc_info:
                    create_model(key_first=key_first, **params)
                reported_input = check_error(exc_info.value, expected_texts)
                if expected_input is not all_params:
                    assert reported_input == expected_input
                elif created_model_is_input:
                    check_model_is_input(reported_input)
                else:
                    # The other params are reported as they are passed
                    assert reported_input == {
                        **model_params,
                        **params,
                        "api_key": REDACTED_VALUE,
                    }
                    for param, value in params.items():
                        assert reported_input[param] is value

        # A model that is created without a required param: the input is all
        # the params that are passed, i.e. the key, also when it is passed in
        # a param that the model does not have, which is rejected as well
        params_without_model = {
            param: value for param, value in model_params.items() if param != "model"
        }
        with pytest.raises(ValidationError) as exc_info:
            model_class(api_key=api_key, **params_without_model)
        assert check_error(exc_info.value, ["model", "Field required"]) == {
            **params_without_model,
            "api_key": REDACTED_VALUE,
        }
        misspelled_key_params: dict[str, Any] = {
            **params_without_model,
            "apikey": api_key,
        }
        with pytest.raises(ValidationError) as exc_info:
            model_class(**misspelled_key_params)
        self.check_validation_error_hides_inputs(
            exc_info.value, api_key, ["model", "Field required", "apikey"]
        )
        self.check_validation_error_omits_model_keys(exc_info.value, api_key)
        # The rest of what the errors report is not changed
        assert exc_info.value.title == model_class.__name__
        assert exc_info.value.errors(include_url=False) == [
            {
                "type": "missing",
                "loc": ("model",),
                "msg": "Field required",
                "input": {**params_without_model, "apikey": REDACTED_VALUE},
            },
            {
                "type": "extra_forbidden",
                "loc": ("apikey",),
                "msg": "Extra inputs are not permitted",
                "input": REDACTED_VALUE,
            },
        ]
        for details in exc_info.value.errors():
            assert details["url"].endswith(f"/v/{details['type']}")
        # A key that is not passed is not reported as a redacted one
        with pytest.raises(ValidationError) as exc_info:
            model_class(api_key=None, **params_without_model)
        assert check_error(exc_info.value, ["model", "Field required"]) == {
            **params_without_model,
            "api_key": None,
        }

        # A model, and its fallback model, that an invalid value is assigned to.
        # The input of a check that involves several params is the model, which
        # holds the key.
        model = create_model(pricing_details=pricing, fallback_models=[fallback_model])
        model_is_input = object()
        invalid_assignments: list[tuple[Any, str, Any, list[str], Any]] = [
            (model, "timeout", "x", timeout_error, "x"),
            (model, "api_key", invalid_key, key_error, REDACTED_VALUE),
            (model, "auto_pricing", True, pricing_error, model_is_input),
            (model, "is_fallback", True, own_fallbacks_error, model_is_input),
            (
                model,
                "fallback_models",
                [fallback_of_other_role],
                fallback_role_error,
                model_is_input,
            ),
            (fallback_model, "timeout", "x", timeout_error, "x"),
            (fallback_model, "api_key", invalid_key, key_error, REDACTED_VALUE),
        ]
        for (
            instance,
            param,
            value,
            expected_texts,
            expected_input,
        ) in invalid_assignments:
            with pytest.raises(ValidationError) as exc_info:
                setattr(instance, param, value)
            reported_input = check_error(exc_info.value, expected_texts)
            if expected_input is model_is_input:
                check_model_is_input(reported_input)
                assert reported_input is instance
            else:
                assert reported_input == expected_input
        # The key that is assigned to a param that the model does not have
        misspelled_param = "apikey"
        with pytest.raises(ValidationError) as exc_info:
            setattr(model, misspelled_param, api_key)
        assert check_error(exc_info.value, ["apikey", "Object has no attribute"]) == (
            REDACTED_VALUE
        )
        assert exc_info.value.errors(include_url=False, include_input=False) == [
            {
                "type": "no_such_attribute",
                "loc": ("apikey",),
                "msg": "Object has no attribute 'apikey'",
                "ctx": {"attribute": "apikey"},
            }
        ]
        assert not hasattr(model, misspelled_param)
        # The models are left as they were
        assert model.api_key == fallback_model.api_key == api_key
        assert model.fallback_models == [fallback_model]
        assert not model.is_fallback

        def get_serialized_model() -> dict[str, Any]:
            """
            Returns the serialized data of the model with the API keys of the model
            and of its fallback model added: the keys are not serialized, i.e.
            the data has them only if they are added to it.

            :return: The serialized data of the model, with the keys.
            :rtype: dict[str, Any]
            """
            serialized_model = model.to_dict()
            # The data has all the params of the model, in their order, with
            # the keys not set
            assert [
                param for param in serialized_model if not param.startswith("_")
            ] == list(model_class.model_fields)
            assert serialized_model["api_key"] is None
            assert serialized_model["fallback_models"][0]["api_key"] is None
            serialized_model["api_key"] = api_key
            serialized_model["fallback_models"][0]["api_key"] = api_key
            return serialized_model

        # A model that is deserialized from invalid data. The data is valid, and
        # has the keys, before it is changed. The input of a check that involves
        # several params is the data, i.e. the key that is added to it.
        deserialized_model = model_class.from_dict(get_serialized_model())
        assert deserialized_model.api_key == api_key
        assert deserialized_model.fallback_models[0].api_key == api_key
        invalid_data: list[tuple[dict[str, Any], list[str], Any]] = []
        for param, value, expected_texts, expected_input in [
            ("timeout", "x", timeout_error, "x"),
            ("api_key", [api_key], key_error, REDACTED_VALUE),
            ("apikey", api_key, ["apikey", "Extra inputs"], REDACTED_VALUE),
            ("auto_pricing", True, pricing_error, all_params),
        ]:
            serialized_model = get_serialized_model()
            serialized_model[param] = value
            invalid_data.append((serialized_model, expected_texts, expected_input))
        # ... and from the data in which its fallback model is invalid
        serialized_model = get_serialized_model()
        serialized_model["fallback_models"][0]["timeout"] = "x"
        invalid_data.append((serialized_model, timeout_error, "x"))
        serialized_model = get_serialized_model()
        serialized_model["fallback_models"][0]["api_key"] = [api_key]
        invalid_data.append((serialized_model, key_error, REDACTED_VALUE))

        def check_data_error(error: BaseException | None, case_index: int) -> None:
            """
            Checks the validation error of the model deserialized from the invalid
            data of the given case.

            :param error: The validation error.
            :type error: BaseException | None
            :param case_index: The index of the case of the invalid data.
            :type case_index: int
            """
            _, expected_texts, expected_input = invalid_data[case_index]
            assert isinstance(error, ValidationError)
            reported_input = check_error(error, expected_texts)
            if expected_input is not all_params:
                assert reported_input == expected_input
                return
            # The data of the model, with its key redacted and with its fallback
            # model, which is a model by now, i.e. holds its key
            assert list(reported_input) == list(model_class.model_fields)
            assert reported_input["api_key"] == REDACTED_VALUE
            assert reported_input["model"] == model.model
            assert reported_input["auto_pricing"] is True
            (restored_fallback_model,) = reported_input["fallback_models"]
            check_model_is_input(restored_fallback_model)

        for case_index, (serialized_model, _, _) in enumerate(invalid_data):
            with pytest.raises(ValidationError) as exc_info:
                model_class.from_dict(serialized_model)
            check_data_error(exc_info.value, case_index)
            with pytest.raises(ValidationError) as exc_info:
                model_class.from_json(json.dumps(serialized_model))
            check_data_error(exc_info.value, case_index)
            # The error of loading from a file has the text of the validation
            # error, which it is raised from
            disk_path = tmp_path / f"{model_class.__name__}_{uuid4()}.json"
            disk_path.write_text(json.dumps(serialized_model), encoding="utf-8")
            with pytest.raises(RuntimeError) as runtime_exc_info:
                model_class.from_disk(disk_path)
            assert not any(
                fragment in str(runtime_exc_info.value)
                for fragment in self.get_api_key_fragments(api_key)
            )
            check_data_error(runtime_exc_info.value.__cause__, case_index)
        # The data without the keys is reported as it is, i.e. with the key not set
        serialized_model = model.to_dict()
        serialized_model["auto_pricing"] = True
        with pytest.raises(ValidationError) as exc_info:
            model_class.from_dict(serialized_model)
        assert check_error(exc_info.value, pricing_error)["api_key"] is None

    def check_instance_container_states(
        self,
        original_container: list[Aspect] | list[_Concept],
        assigned_container: list[Aspect] | list[_Concept],
        assigned_instance_class: type[Aspect | _Concept],
    ) -> None:
        """
        Checks the states of an original container and an assigned container of instances,
        ensuring that the containers have the expected state and type of its instances.

        :param original_container: The original container of instances that will be compared to
            the assigned container.
        :type original_container: list[Aspect] | list[_Concept]
        :param assigned_container: The newly assigned container that will be checked for
            correct state and type of instances.
        :type assigned_container: list[Aspect] | list[_Concept]
        :param assigned_instance_class: The expected class type of the instances in the
            assigned container.
        :type assigned_instance_class: type[Aspect, _Concept]
        :return: None
        """
        if len(original_container) or len(assigned_container):
            assert len(original_container) == len(assigned_container)
            assert original_container is not assigned_container
            assert original_container != assigned_container
            assert not any(i._is_processed for i in original_container)
            assert not any(i.extracted_items for i in original_container)

            def check_instances(instances: list[Aspect] | list[_Concept]) -> None:
                """
                Helper function to check instances for proper processing state and extracted items.

                :param instances: List of Aspect or Concept instances to validate.
                :type instances: list[Aspect] | list[_Concept]
                :return: None
                :rtype: None
                """
                assert all(isinstance(i, assigned_instance_class) for i in instances)
                # All instances are processed, regardless of their model roles: a standalone
                # LLM extracts all instances, and a model group routes each instance to an LLM
                assert all(i._is_processed for i in instances)
                fields_to_group_by = [
                    "add_justifications",
                    "justification_depth",
                    "justification_max_sents",
                    "add_references",  # references are always added to aspects automatically
                    "reference_depth",
                ]
                instance_groups = _group_instances_by_fields(
                    fields=fields_to_group_by,
                    instances=instances,  # ty: ignore[invalid-argument-type]
                )
                for idx, group in enumerate(instance_groups, start=1):
                    logger.debug(
                        f"Checking group {idx} ({len(group)} instances "
                        f"with model role(s) {[i.model_role for i in group]})"
                    )
                    if not group:
                        continue
                    try:
                        assert any(i.extracted_items for i in group)
                        for i in group:
                            self.check_extra_data_in_extracted_items(i)  # ty: ignore[invalid-argument-type]
                            # Check for singular occurrence enforcement
                            if getattr(i, "singular_occurrence", False):
                                assert len(i.extracted_items) <= 1
                        logger.debug("Check has passed for group")
                    except AssertionError as e:
                        warnings.warn(
                            f"Check has failed for instance group ({len(group)}) "
                            f"with model role(s) {[i.model_role for i in group]}: {e}. "
                            f"Instance group: {[i.name for i in group]}",
                            stacklevel=2,
                        )

            check_instances(assigned_container)
            for i in assigned_container:
                if isinstance(i, Aspect):
                    # Check sub-aspects
                    check_instances(i.aspects)  # ty: ignore[invalid-argument-type]

    def check_custom_data_json_serializable(
        self,
        instance: _InstanceBase,
    ) -> None:
        """
        Check if custom data assigned to an object is serializable to JSON.

        :param instance: The _InstanceBase instance whose `custom_data` attribute is being tested.
        :type instance: _InstanceBase
        :raises ValueError: If `custom_data` contains non-serializable key-value
                            pairs or unsupported data types.
        :return: None
        """
        if not hasattr(instance, "custom_data"):
            return
        with pytest.raises(ValueError):
            instance.custom_data = {int: bool}  # type: ignore
        with pytest.raises(ValueError):
            instance.custom_data = {str: [object]}  # type: ignore
        with pytest.raises(TypeError):
            instance.custom_data = {self.document: self.extraction_pipeline}  # type: ignore
        with pytest.raises(TypeError):
            instance.custom_data = {
                self.llm_extractor: self.invalid_llm_with_valid_fallback  # type: ignore
            }
        instance.custom_data = {"test": True}
        instance.custom_data = {}
        assert _is_json_serializable(instance.custom_data)

    def compare_with_concurrent_execution(
        self,
        llm: DocumentModelGroup | DocumentLLM,
        expected_n_calls_no_concurrency: int,
        expected_n_calls_1_item_per_call: int,
        expected_n_calls_with_concurrency: int,
        func: Callable,
        func_kwargs: dict,
        original_container: list[Aspect] | list[_Concept],
        assigned_container: list[Aspect] | list[_Concept],
        assigned_instance_class: type[Aspect | _Concept],
        compare_sequential_1_item_in_call: bool = False,
    ) -> None:
        """
        Compares the efficacy of a provided function's execution under different concurrency and
        execution settings. This method evaluates the time and state of the execution process
        with and without concurrency, as well as in specific configurations, such as sequential
        processing with only one item processed per call. It validates the expected number of
        function calls and final states after execution.

        :param llm: The DocumentModelGroup or DocumentLLM instance involved in the testing process.
        :param expected_n_calls_no_concurrency: The expected number of LLM calls when
            concurrency is not utilized in execution.
        :param expected_n_calls_1_item_per_call: The expected number of LLM calls during
            sequential execution when the function processes one item per call.
        :param expected_n_calls_with_concurrency: The expected number of LLM calls when
            concurrency is enabled during execution.
        :param func: The callable function to be tested for different execution settings.
        :param func_kwargs: A dictionary of arguments to be passed to the function during
            execution.
        :param original_container: The initial state of the container whose state
            will be compared before and after execution.
        :param assigned_container: The modified or updated container to compare against the
            `original_container` following function execution.
        :param assigned_instance_class: The expected type of instances that should be found
            within the `assigned_container`.
        :param compare_sequential_1_item_in_call: A boolean flag indicating whether the
            comparison should include processing with one item per call in the sequential
            (non-concurrent) setup.
        :return: None
        """

        def get_current_n_calls() -> int:
            """
            Gets the current number of executed successful (finished) calls to the LLM.
            :return: int
            """
            return sum(
                len([c for c in i.usage.calls if c.timestamp_received])
                for i in llm.get_usage()
            )

        # Deep copy original kwargs to avoid state modification in comparison calls
        func_kwargs_copy_1 = deepcopy(func_kwargs)
        func_kwargs_copy_2 = deepcopy(func_kwargs)

        time_no_concurrency_start = time.time()
        initial_n_calls = get_current_n_calls()
        func(
            **func_kwargs,
            use_concurrency=False,
        )
        if get_current_n_calls() - initial_n_calls != expected_n_calls_no_concurrency:
            raise AssertionError(
                f"Expected {expected_n_calls_no_concurrency} calls, "
                f"but got {get_current_n_calls()} - {initial_n_calls} = "
                f"{get_current_n_calls() - initial_n_calls}"
            )
        self.check_instance_container_states(
            original_container=original_container,
            assigned_container=assigned_container,
            assigned_instance_class=assigned_instance_class,
        )
        time_no_concurrency_end = time.time()
        time_no_concurrency = time_no_concurrency_end - time_no_concurrency_start

        if compare_sequential_1_item_in_call:
            time_no_concurrency_1_item_per_call_start = time.time()
            initial_n_calls = get_current_n_calls()
            func_kwargs_copy_1["max_items_per_call"] = 1
            func(
                **func_kwargs_copy_1,
                use_concurrency=False,
            )
            if (
                get_current_n_calls() - initial_n_calls
                != expected_n_calls_1_item_per_call
            ):
                raise AssertionError(
                    f"Expected {expected_n_calls_1_item_per_call} calls, "
                    f"but got {get_current_n_calls()} - {initial_n_calls} = "
                    f"{get_current_n_calls() - initial_n_calls}"
                )
            time_no_concurrency_1_item_per_call_end = time.time()
            time_no_concurrency_1_item_per_call = (
                time_no_concurrency_1_item_per_call_end
                - time_no_concurrency_1_item_per_call_start
            )
            # Normally, the below comparison should work, but it depends on the amount
            # of the items to be processed. If e.g. the N of items is 1, then the time
            # will be about equal.
            try:
                assert time_no_concurrency_1_item_per_call > time_no_concurrency
            except AssertionError:
                warnings.warn(
                    "No concurrency time is same or slower than 1 item per call time",
                    stacklevel=2,
                )
            logger.debug(
                f"no concurrency time {time_no_concurrency}; N calls {expected_n_calls_no_concurrency}\n"
                f"1 item per call time (no concurrency) {time_no_concurrency_1_item_per_call}; "
                f"N calls {expected_n_calls_1_item_per_call}\n"
            )

        time_concurrency_start = time.time()
        initial_n_calls = get_current_n_calls()
        func(
            **func_kwargs_copy_2,
            use_concurrency=True,
        )
        if get_current_n_calls() - initial_n_calls != expected_n_calls_with_concurrency:
            raise AssertionError(
                f"Expected {expected_n_calls_with_concurrency} calls, "
                f"but got {get_current_n_calls()} - {initial_n_calls} = "
                f"{get_current_n_calls() - initial_n_calls}"
            )
        time_concurrency_end = time.time()
        time_concurrency = time_concurrency_end - time_concurrency_start

        # Normally, the below comparison should work, but it depends on the complexity of the extracted items.
        # Sometimes, for simple items the processing is faster without concurrency, in one go.
        try:
            assert time_no_concurrency > time_concurrency
        except AssertionError:
            warnings.warn(
                "No concurrency is faster than default concurrency",
                stacklevel=2,
            )
        logger.debug(
            f"\nno concurrency time {time_no_concurrency}; N calls {expected_n_calls_no_concurrency}\n"
            f"with concurrency time {time_concurrency}; N calls {expected_n_calls_with_concurrency}\n"
        )

    def config_llms_for_output_lang(
        self, document: Document, llm: DocumentModelGroup | DocumentLLM
    ) -> None:
        """
        Configures the output language adaptation for the LLMs
        based on the language of the test document.

        :param document: The document for which the output language of the LLM is being configured.
        :type document: Document
        :param llm: The model group or single LLM instance to be configured.
        :type llm: DocumentModelGroup | DocumentLLM

        :return: None
        """

        # Configure the output language adaptation for the LLMs based on the document language
        if document in [self.document_ua, self.document_zh]:  # type: ignore
            output_language = "adapt"
        else:
            output_language = "en"
        if isinstance(llm, DocumentModelGroup):
            llm.group_update_output_language(output_language)
        else:  # DocumentLLM
            llm.output_language = output_language
        logger.debug(f"LLM output language set to `{output_language}`")

    @staticmethod
    def config_llm_async_limiter_for_mock_responses(
        llm: DocumentModelGroup | DocumentLLM,
    ) -> None:
        """
        Configures an asynchronous rate limiter for mock responses within a specified
        DocumentLLM instance or DocumentModelGroup. The limiter is applied to the primary model
        and its fallback models, if applicable.

        :param llm: A DocumentLLM or DocumentModelGroup instance, representing the LLM
            models to configure.
        :type llm: DocumentModelGroup | DocumentLLM
        :return: None
        :rtype: None
        """
        if vcr_new_recording_count:
            limiter = AsyncLimiter(
                100, 60
            )  # considerably increase the rate limit comparing to default limiter, as all responses are mock

            def set_new_limiter(model):
                """
                Helper function to set the async limiter on a model and its fallbacks.

                :param model: The DocumentLLM model to configure.
                :type model: DocumentLLM
                """
                model.async_limiter = limiter
                for fallback_llm in model.fallback_models:
                    fallback_llm.async_limiter = limiter

            if isinstance(llm, DocumentModelGroup):
                for model in llm.models:
                    set_new_limiter(model)
            else:
                set_new_limiter(llm)

    @staticmethod
    def log_extracted_items_for_instance(
        instance: Aspect | _Concept, full_repr: bool = True
    ) -> None:
        """
        Logs the extracted items associated with the given instance.

        :param instance: An instance of type `Aspect` or `_Concept` that contains the extracted
            items to be logged.
        :type instance: Aspect | _Concept
        :param full_repr: Whether to log the full representation of the extracted items.
            If True, logs the full dictionary representation. If False, logs a simplified
            representation with just the value and justification.
        :type full_repr: bool
        :return: None
        """
        logger.debug(f"========{instance.name}========")
        for idx, item in enumerate(instance.extracted_items, start=1):
            if full_repr:
                item_repr = item.to_dict()
            else:
                item_repr = f"{item.value} (justification: {item.justification})"
            logger.debug(f"Extracted item {idx}: {item_repr}")
        logger.debug("===============================")

    def extract_with_local_llm(self, llm: DocumentLLM):
        """
        Helper method to test extraction with a local LLM model.

        :param llm: The DocumentLLM instance to test extraction with.
        :type llm: DocumentLLM
        """
        # Configure document
        document = Document(
            raw_text=get_test_document_text()[:1000],
        )
        concept = StringConcept(
            name="Contract title",
            description="The title of the contract.",
        )
        document.add_concepts([concept])

        # Run extraction
        self.config_llm_async_limiter_for_mock_responses(llm)
        extracted_concepts = llm.extract_concepts_from_document(document)
        assert llm.get_usage()[0].usage.calls[-1].prompt
        assert llm.get_usage()[0].usage.calls[-1].response
        extracted_items = extracted_concepts[0].extracted_items
        assert len(extracted_items), (
            f"No extracted items returned with local LLM {llm.model}"
        )
        self.log_extracted_items_for_instance(extracted_concepts[0])

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm)


def set_dummy_env_variables_for_testing_from_cassettes() -> None:
    """
    Sets dummy environment variables for testing from VCR cassettes.
    This function is typically called when load_dotenv() doesn't find a .env file,
    to ensure tests have the necessary environment variables set with dummy values.
    It's used in CI as well as locally by new contributors whose changes
    do not require re-recording VCR cassettes. See CONTRIBUTING.md for more details.

    :return: None
    """

    if load_dotenv():
        raise RuntimeError(
            "A .env file was found, but dummy environment variables "
            "for testing from VCR cassettes were expected. "
            "Please remove the .env file and run the tests again."
        )

    logger.debug(
        "No .env file found, setting dummy environment variables "
        "for testing from VCR cassettes"
    )

    default_env_vars = {
        "CONTEXTGEM_OPENAI_API_KEY": "DUMMY",
        # The key of the decision models of TypeSafe: a request for which no key
        # is passed is rejected by litellm before it is sent, i.e. before it can
        # be replayed
        "CONTEXTGEM_TYPESAFE_API_KEY": "DUMMY",
        # The key and the endpoint of the Microsoft Foundry resource of the decision
        # model deployed in it: litellm rejects a request without either of them
        # before it is sent. The endpoint is the one that the host of a resource
        # is replaced with in the cassettes.
        "CONTEXTGEM_AZURE_AI_API_KEY": "DUMMY",
        "CONTEXTGEM_AZURE_AI_API_BASE": VCR_DUMMY_ENDPOINT_PREFIX.rstrip("/"),
        "CONTEXTGEM_LOGGER_LEVEL": "DEBUG",
    }

    for key, value in default_env_vars.items():
        # Force dummy credentials to be used in tests if no .env file is found
        os.environ[key] = value
