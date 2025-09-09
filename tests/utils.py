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
from typing import Literal, cast
from uuid import uuid4

import pytest
from aiolimiter import AsyncLimiter
from dotenv import load_dotenv

from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.data_models import (
    _LLMCost,
    _LLMCostOutputContainer,
    _LLMUsage,
    _LLMUsageOutputContainer,
)
from contextgem.internal.loggers import logger
from contextgem.internal.typings.types import LLMRoleAny
from contextgem.internal.utils import (
    _are_prompt_template_brackets_balanced,
    _are_prompt_template_xml_tags_balanced,
    _group_instances_by_fields,
    _is_json_serializable,
)
from contextgem.public import (
    Aspect,
    Document,
    DocumentLLM,
    DocumentLLMGroup,
    Image,
    create_image,
)
from tests.conftest import VCR_DUMMY_ENDPOINT_PREFIX, VCR_REDACTION_MARKER


# A global VCR recordings counter
vcr_new_recording_count = 0


VCR_FILTER_HEADERS = {
    "api-key",
    "apim-request-id",
    "authorization",
    "azureml-model-session",
    "CF-RAY",
    "cookie",
    "Date",
    "host",
    "openai-organization",
    "Set-Cookie",
    "uri",
    "x-aml-cluster",
    "x-ms-client-request-id",
    "x-ms-deployment-name",
    "x-ms-region",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ms-rai-invoked",
    "x-request-id",
    "x-stainless-os",
    "x-stainless-runtime-version",
}
assert len(set(VCR_FILTER_HEADERS)) == len(VCR_FILTER_HEADERS)


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
    to protect sensitive configuration details in VCR cassettes.

    :param request: The VCR request object to process.
    :return: The modified request object with redacted URI.
    """
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


def vcr_before_record_response(response):
    """
    Processes VCR responses before recording to redact sensitive information.

    Redacts headers and response body content that might contain sensitive
    information like API keys, request IDs, and other identifiable data.

    :param response: The VCR response object to process.
    :return: The modified response object with redacted sensitive data.
    """
    # Redact headers
    headers = response.get("headers", {})
    for header in VCR_FILTER_HEADERS:
        if header.lower() in [h.lower() for h in headers]:
            headers[header] = [VCR_REDACTION_MARKER]
    # Redact body
    if response["body"]["string"]:
        try:
            body = json.loads(response["body"]["string"])
            if "id" in body:
                body["id"] = f"chatcmpl-{VCR_REDACTION_MARKER}"
            response["body"]["string"] = json.dumps(body, ensure_ascii=False)
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
            "_md_text",
            "_messages",
        ]

        # To / from dict
        instance_dict = instance.to_dict()
        new_instance = instance.__class__.from_dict(instance_dict)
        assert new_instance.__dict__ == instance.__dict__
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
        assert new_instance.__dict__ == instance.__dict__
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
        assert new_instance.__dict__ == instance.__dict__
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
        assert instance_clone.__dict__ == instance.__dict__
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
        self, instance: DocumentLLM | DocumentLLMGroup
    ) -> None:
        """
        Checks if the LLM config of a deserialized DocumentLLM/DocumentLLMGroup is equal to
        the LLM config of the original instance.

        :param instance: The original DocumentLLM or DocumentLLMGroup instance to be checked.
        :type instance: DocumentLLM | DocumentLLMGroup
        :return: None
        """
        # From json
        instance_json = instance.to_json()
        if isinstance(instance, DocumentLLM):
            new_instance = DocumentLLM.from_json(instance_json)
            assert instance._eq_deserialized_llm_config(new_instance)
        else:  # DocumentLLMGroup
            new_instance = DocumentLLMGroup.from_json(instance_json)
            assert instance._eq_deserialized_llm_config(new_instance)

        # From dict
        instance_dict = instance.to_dict()
        if isinstance(instance, DocumentLLM):
            new_instance = DocumentLLM.from_dict(instance_dict)
            assert instance._eq_deserialized_llm_config(new_instance)
        else:  # DocumentLLMGroup
            new_instance = DocumentLLMGroup.from_dict(instance_dict)
            assert instance._eq_deserialized_llm_config(new_instance)

        # From disk
        disk_path = os.path.join(
            get_project_root_path(), "tests", f"instance_{str(uuid4())}.json"
        )
        instance.to_disk(disk_path)
        if isinstance(instance, DocumentLLM):
            new_instance = DocumentLLM.from_disk(disk_path)
            assert instance._eq_deserialized_llm_config(new_instance)
        else:  # DocumentLLMGroup
            new_instance = DocumentLLMGroup.from_disk(disk_path)
            assert instance._eq_deserialized_llm_config(new_instance)
        remove_file(disk_path)

        # Check that pydantic-specific methods are disabled
        with pytest.raises(NotImplementedError):
            instance.model_dump()
        with pytest.raises(NotImplementedError):
            instance.model_dump_json()

    def check_usage(self, llm: DocumentLLMGroup | DocumentLLM) -> None:
        """
        Evaluates the usage stats for an LLM or a group of LLMs
        (when using DocumentLLMGroup). Ensures that the usage information is recorded
        properly and validates the structure and presence of related data.

        :param llm: A DocumentLLM or DocumentLLMGroup instance to assess usage
            data. Must be an instance of either DocumentLLM or DocumentLLMGroup.
        :return: None
        """

        usages = llm.get_usage()
        assert isinstance(usages, list)

        # For LLM group: At least one LLM in group must have recorded tokens usage
        # For individual LLM: Either the primary model or its fallback model must have recorded usage
        usage_input_used = False
        usage_output_used = False
        for usage_item in usages:
            assert isinstance(usage_item, _LLMUsageOutputContainer)
            assert hasattr(usage_item, "model") and usage_item.model
            assert hasattr(usage_item, "role") and usage_item.role
            assert hasattr(usage_item, "is_fallback")
            assert hasattr(usage_item, "usage") and usage_item.usage
            assert isinstance(usage_item.usage, _LLMUsage)
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
    def check_cost(llm: DocumentLLMGroup | DocumentLLM) -> None:
        """
        Checks the cost details of a LLM or a group of LLMs, ensuring that the
        cost calculation and structure meet the required conditions.

        :param llm: The LLM or group of LLMs whose processing cost is to be checked.
        :type llm: DocumentLLMGroup | DocumentLLM
        :return: None
        :rtype: None
        """

        costs = llm.get_cost()
        assert isinstance(costs, list)

        # For LLM group: At least one LLM in group must have recorded costs
        # For individual LLM: Either the primary model or its fallback model must have recorded costs
        cost_input_updated = False
        cost_output_updated = False
        for cost_item in costs:
            assert isinstance(cost_item, _LLMCostOutputContainer)
            assert hasattr(cost_item, "model") and cost_item.model
            assert hasattr(cost_item, "role") and cost_item.role
            assert hasattr(cost_item, "is_fallback")
            assert hasattr(cost_item, "cost") and cost_item.cost
            assert isinstance(cost_item.cost, _LLMCost)
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

    def check_instance_container_states(
        self,
        original_container: list[Aspect] | list[_Concept],
        assigned_container: list[Aspect] | list[_Concept],
        assigned_instance_class: type[Aspect | _Concept],
        llm_roles: list[LLMRoleAny],
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
        :param llm_roles: List of strings representing the roles of the LLM(s) involved in processing.
        :type llm_roles: list[LLMRoleAny]
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
                # instances may have different LLM roles
                filtered_instances = [i for i in instances if i.llm_role in llm_roles]
                assert all(i._is_processed for i in filtered_instances)
                fields_to_group_by = [
                    "add_justifications",
                    "justification_depth",
                    "justification_max_sents",
                    "add_references",  # references are always added to aspects automatically
                    "reference_depth",
                ]
                instance_groups = _group_instances_by_fields(
                    fields=fields_to_group_by,
                    instances=cast(list[Aspect] | list[_Concept], filtered_instances),
                )
                for idx, group in enumerate(instance_groups, start=1):
                    logger.debug(
                        f"Checking group {idx} ({len(group)} instances "
                        f"with LLM role(s) {[i.llm_role for i in group]})"
                    )
                    if not group:
                        continue
                    try:
                        assert any(i.extracted_items for i in group)
                        for i in group:
                            self.check_extra_data_in_extracted_items(i)
                            # Check for singular occurrence enforcement
                            if getattr(i, "singular_occurrence", False):
                                assert len(i.extracted_items) <= 1
                        logger.debug("Check has passed for group")
                    except AssertionError as e:
                        warnings.warn(
                            f"Check has failed for instance group ({len(group)}) "
                            f"with LLM role(s) {[i.llm_role for i in group]}: {e}. "
                            f"Instance group: {[i.name for i in group]}",
                            stacklevel=2,
                        )

            check_instances(assigned_container)
            for i in assigned_container:
                if isinstance(i, Aspect):
                    # Check sub-aspects
                    check_instances(i.aspects)

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
                self.llm_extractor_text: self.invalid_llm_with_valid_fallback  # type: ignore
            }
        instance.custom_data = {"test": True}
        instance.custom_data = {}
        assert _is_json_serializable(instance.custom_data)

    def compare_with_concurrent_execution(
        self,
        llm: DocumentLLMGroup | DocumentLLM,
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

        :param llm: The DocumentLLMGroup or DocumentLLM instance involved in the testing process.
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
            llm_roles=llm.list_roles,
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
        self, document: Document, llm: DocumentLLMGroup | DocumentLLM
    ) -> None:
        """
        Configures the output language adaptation for the LLMs
        based on the language of the test document.

        :param document: The document for which the output language of the LLM is being configured.
        :type document: Document
        :param llm: The LLM group or single LLM instance to be configured.
        :type llm: DocumentLLMGroup | DocumentLLM

        :return: None
        """

        # Configure the output language adaptation for the LLMs based on the document language
        if document in [self.document_ua, self.document_zh]:  # type: ignore
            output_language = "adapt"
        else:
            output_language = "en"
        if isinstance(llm, DocumentLLMGroup):
            llm.group_update_output_language(output_language)
        else:  # DocumentLLM
            llm.output_language = output_language
        logger.debug(f"LLM output language set to `{output_language}`")

    @staticmethod
    def config_llm_async_limiter_for_mock_responses(
        llm: DocumentLLMGroup | DocumentLLM,
    ) -> None:
        """
        Configures an asynchronous rate limiter for mock responses within a specified
        DocumentLLM instance or DocumentLLMGroup. The limiter is applied to the primary model
        and its fallback model, if applicable.

        :param llm: A DocumentLLM or DocumentLLMGroup instance, representing the LLM
            models to configure.
        :type llm: DocumentLLMGroup | DocumentLLM
        :return: None
        :rtype: None
        """
        if vcr_new_recording_count:
            limiter = AsyncLimiter(
                100, 60
            )  # considerably increase the rate limit comparing to default limiter, as all responses are mock

            def set_new_limiter(model):
                """
                Helper function to set the async limiter on a model and its fallback.

                :param model: The DocumentLLM model to configure.
                :type model: DocumentLLM
                """
                model.async_limiter = limiter
                if model.fallback_llm:
                    model.fallback_llm.async_limiter = limiter

            if isinstance(llm, DocumentLLMGroup):
                for model in llm.llms:
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
        "CONTEXTGEM_AZURE_OPENAI_API_KEY": "DUMMY",
        "CONTEXTGEM_AZURE_OPENAI_API_VERSION": "2025-03-01-preview",
        "CONTEXTGEM_AZURE_OPENAI_API_BASE": "https://<DUMMY-ENDPOINT>/openai/deployments/DUMMY-DEPLOYMENT",
        "CONTEXTGEM_OPENAI_API_KEY": "DUMMY",
        "CONTEXTGEM_LOGGER_LEVEL": "DEBUG",
    }

    for key, value in default_env_vars.items():
        # Force dummy credentials to be used in tests if no .env file is found
        os.environ[key] = value
