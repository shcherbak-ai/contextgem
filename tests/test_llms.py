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
Module defining tests for LLMs.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import types
import warnings
from typing import Any
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter
from pydantic import ValidationError

from contextgem.internal.base import llms as llms_module
from contextgem.internal.base import models as models_module
from contextgem.internal.exceptions import ModelAPIError, ModelExtractionError
from contextgem.internal.loggers import logger
from contextgem.internal.utils import (
    _get_model_api_retry_delay,
    _is_retryable_model_api_error,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    ModelPricing,
    StringConcept,
    register_tool,
)
from tests.base import BaseTest
from tests.utils import get_project_root_path


class TestLLMs(BaseTest):
    """
    Tests for the configuration of LLMs, local LLMs, prompts, token limits,
    retries of API requests and rate limiting.
    """

    @pytest.mark.vcr
    def test_local_llms_text(self):
        """
        Tests for initialization of and getting a response from local LLMs,
        e.g. models run on Ollama local server. Limited to text processing.

        Note that using `ollama_chat/` (preferred in live mode) instead of
        `ollama/` prefix for Ollama models does not work with VCR.
        """

        # Ollama
        # Non-reasoning LLM
        llm_non_reasoning_ollama = DocumentLLM(
            model="ollama/mistral-small:24b",
            api_base="http://localhost:11434",
            seed=123,
        )
        self.extract_with_local_llm(llm_non_reasoning_ollama)
        # Reasoning (CoT-capable) LLM
        llm_reasoning_ollama = DocumentLLM(
            model="ollama/deepseek-r1:32b",
            api_base="http://localhost:11434",
            seed=123,
        )
        llm_reasoning_ollama._supports_reasoning = True
        self.extract_with_local_llm(llm_reasoning_ollama)

        # LM Studio
        # Non-reasoning LLM
        llm_non_reasoning_lm_studio = DocumentLLM(
            model="lm_studio/mistralai/mistral-small-3.2",
            api_base="http://localhost:1234/v1",
            api_key="random-key",  # required for LM Studio API
            seed=123,
        )
        self.extract_with_local_llm(llm_non_reasoning_lm_studio)

    @pytest.mark.vcr
    def test_local_llms_vision(self):
        """
        Tests for initialization of and getting a response from local LLMs
        with vision capabilities.
        """
        document_concepts = [
            StringConcept(
                name="Invoice number",
                description="Number of the invoice",
                modality="image",
                add_justifications=True,
            ),
        ]
        document = Document(
            images=[self.test_img_png_invoice], concepts=document_concepts
        )
        # Warn about using ollama/ prefix when images are sent to a local model
        llm_ollama_chat = DocumentLLM(
            model="ollama_chat/gemma3:27b",
            api_base="http://localhost:11434",
        )
        llm_ollama_chat._supports_vision = True
        with pytest.warns(UserWarning, match="use `ollama/` prefix instead"):
            llm_ollama_chat._ensure_vision_support_for_messages(
                [
                    llm_ollama_chat._build_user_message(
                        message_text="Test", images=[self.test_img_png_invoice]
                    )
                ]
            )

        # Ollama
        llm_ollama = DocumentLLM(
            model="ollama/gemma3:27b",
            api_base="http://localhost:11434",
        )
        # The model is not detected as vision-capable, i.e. the extraction from images
        # fails until the capability is declared explicitly
        # TODO: remove this check once litellm supports vision for this model
        with pytest.raises(ValueError, match="does not support vision"):
            llm_ollama.extract_concepts_from_document(document)
        llm_ollama._supports_vision = True
        extracted_concepts = llm_ollama.extract_concepts_from_document(document)
        assert extracted_concepts[0].extracted_items
        # Log the extracted item for debugging
        self.log_extracted_items_for_instance(extracted_concepts[0])

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm_ollama)

    @pytest.mark.vcr
    def test_local_llms_tools(self):
        """
        Tests for chat with tools with a local LLM, e.g. a model run on LM Studio
        local server. The LLM has to call the tools one after another, using
        the results of the previous tool calls in the next one.
        """

        tool_runs: list[str] = []

        @register_tool
        def get_order_total(order_id: str) -> str:
            """
            Returns the total amount of the order.
            """
            tool_runs.append("total")
            return {"ORD-123": "9"}.get(order_id, "0")

        @register_tool
        def get_discount_amount(order_id: str) -> str:
            """
            Returns the discount amount for the order.
            """
            tool_runs.append("discount")
            return {"ORD-123": "4"}.get(order_id, "0")

        @register_tool
        def compute_final_amount(order_total: str, discount: str) -> str:
            """
            Computes the final amount as the order total minus the discount.
            """
            tool_runs.append("final")
            return str(int(float(order_total)) - int(float(discount)))

        # LM Studio
        # The model is not detected as tools-capable, i.e. the capability
        # has to be declared explicitly
        with pytest.warns(UserWarning, match="does not detect it as tools-capable"):
            llm_lm_studio = DocumentLLM(
                model="lm_studio/openai/gpt-oss-20b",
                api_base="http://localhost:1234/v1",
                api_key="random-key",  # required for LM Studio API
                system_message="",
                tools=[get_order_total, get_discount_amount, compute_final_amount],
                seed=123,
            )
        llm_lm_studio._supports_tools = True

        answer = llm_lm_studio.chat(
            "Use the available tools to compute a final amount. "
            "First call `get_order_total` with order_id='ORD-123'. "
            "Then call `get_discount_amount` with order_id='ORD-123'. "
            "Then call `compute_final_amount` using the exact outputs returned by "
            "the previous tools as order_total and discount. "
            "Reply only with the number returned by `compute_final_amount`, nothing else. "
            "Do not guess any values; you must call the tools."
        )
        logger.debug(f"Answer of the local LLM with tools: {answer}")

        # Each tool is called once, the last one after the tools it depends on
        assert sorted(tool_runs[:2]) == ["discount", "total"]
        assert tool_runs[2:] == ["final"]
        # Expected: 9 - 4 = 5
        assert answer and "5" in answer

        # The LLM is called at least for the two tool rounds and for the final answer
        assert len(llm_lm_studio.get_usage()[0].usage.calls) >= 3

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm_lm_studio)

    def test_init_api_llm(self):
        """
        Tests the behaviour of the `DocumentLLM` class initialization.
        """
        DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )

        # Optional params
        DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            top_p=None,
            temperature=None,
        )

        # Default: the sampling params and the seed are not set, and the output
        # tokens are not limited, i.e. the defaults of the model apply
        llm_defaults = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert llm_defaults.temperature is None
        assert llm_defaults.top_p is None
        assert llm_defaults.seed is None
        assert llm_defaults.max_tokens is None
        assert llm_defaults.timeout == 600
        assert (
            llm_defaults.async_limiter.max_rate,
            llm_defaults.async_limiter.time_period,
        ) == (10, 10)

        # Reasoning (CoT-capable) model: the sampling params, the seed and the output
        # token limit are not sent unless set
        optional_params = [
            "temperature",
            "top_p",
            "seed",
            "max_tokens",
            "max_completion_tokens",
        ]
        assert llm_defaults._supports_reasoning
        default_request_config = llm_defaults._build_request_config()
        for param in optional_params:
            assert param not in default_request_config

        # ...in which case the sampling params and the seed are sent as they are,
        # as for any other model, and the limit is sent as `max_completion_tokens`
        llm_with_max_tokens = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            temperature=0.2,
            top_p=0.9,
            seed=123,
            max_tokens=8000,
        )
        request_config_with_max_tokens = llm_with_max_tokens._build_request_config()
        assert request_config_with_max_tokens["temperature"] == 0.2
        assert request_config_with_max_tokens["top_p"] == 0.9
        assert request_config_with_max_tokens["seed"] == 123
        assert request_config_with_max_tokens["max_completion_tokens"] == 8000
        assert "max_tokens" not in request_config_with_max_tokens
        # ...or as `max_tokens`, if `max_completion_tokens` is not supported for the model
        for supported_params in [["max_tokens", "reasoning_effort"], None]:
            with patch.object(
                llms_module.litellm,
                "get_supported_openai_params",
                return_value=supported_params,
            ):
                request_config_with_max_tokens = (
                    llm_with_max_tokens._build_request_config()
                )
            assert request_config_with_max_tokens["max_tokens"] == 8000
            assert "max_completion_tokens" not in request_config_with_max_tokens

        # Model without reasoning support: same, with the limit sent as `max_tokens`
        non_reasoning_model = "cerebras/llama-3.3-70b"
        llm_non_reasoning = DocumentLLM(
            model=non_reasoning_model, api_key="test_api_key"
        )
        assert not llm_non_reasoning._supports_reasoning
        non_reasoning_request_config = llm_non_reasoning._build_request_config()
        for param in optional_params:
            assert param not in non_reasoning_request_config

        # A zero value is a value that is set, i.e. it is sent
        llm_with_top_p = DocumentLLM(
            model=non_reasoning_model,
            api_key="test_api_key",
            temperature=0.0,
            top_p=0.3,
            seed=0,
            max_tokens=2000,
        )
        request_config_with_top_p = llm_with_top_p._build_request_config()
        assert request_config_with_top_p["temperature"] == 0.0
        assert request_config_with_top_p["top_p"] == 0.3
        assert request_config_with_top_p["seed"] == 0
        assert request_config_with_top_p["max_tokens"] == 2000
        assert "max_completion_tokens" not in request_config_with_top_p

        # Unsupported methods
        with pytest.raises(NotImplementedError):
            self.model_group.model_dump()
        with pytest.raises(NotImplementedError):
            self.model_group.model_dump_json()
        with pytest.raises(NotImplementedError):
            self.llm_extractor.model_dump()
        with pytest.raises(NotImplementedError):
            self.llm_extractor.model_dump_json()

        # Invalid params
        with pytest.raises(ValueError):
            DocumentLLM()
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
            )

        # Fallback LLM validation
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                fallback_models=[
                    self.llm_reasoner
                ],  # inconsistent role of the fallback LLM
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                fallback_models=[
                    DocumentLLM(
                        model="openai/gpt-6-luna",
                        api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                        is_fallback=True,
                    )
                ],  # same params as the main LLM, except for `is_fallback` flag
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                is_fallback=True,
                fallback_models=[
                    self.llm_extractor
                ],  # a fallback LLM is passed to fallback LLM
            )
        # Validate fallback LLMs assignment
        document_llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert document_llm.is_group is False
        assert document_llm.fallback_models == []
        document_llm.is_fallback = True
        with pytest.raises(ValueError):
            document_llm.fallback_models = [
                self.llm_extractor
            ]  # a fallback LLM is passed to fallback LLM
        assert document_llm.fallback_models == []  # No change due to invalid assignment
        document_llm.is_fallback = False
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                is_fallback=False,
                fallback_models=[document_llm],  # non-fallback LLM passed as a fallback
            )

        with pytest.raises(ValueError):
            DocumentLLM(
                model="gpt-6-luna",  # no provider prefix in model name
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                role="",  # invalid role (empty label)
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                temperature=-1.0,  # negative temperature
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                max_tokens=-4000,  # negative value
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                # removed param: `max_tokens` applies to all models
                max_completion_tokens=8000,  # ty: ignore[unknown-argument]
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    **{  # invalid pricing details keys
                        "in": 0.1,
                        "out": 0.2,
                    }
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    **{  # invalid pricing details values
                        "input_per_1m_tokens": -0.00015,
                        "output_per_1m_tokens": 0.0006,
                    }
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key="",  # empty api key
            )

        # Warnings
        with pytest.warns(
            UserWarning, match="`output_language` parameter will take no effect"
        ):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="",
            )
        with pytest.warns(
            UserWarning, match="`output_language` parameter will take no effect"
        ):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="You are a helpful assistant.",
            )

    def test_update_default_prompt(self):
        """
        Tests for updating the default prompt for the LLM.
        """
        # Test custom prompts assignment
        llm_with_updated_prompts = DocumentLLM(
            **self._llm_extractor_kwargs_openai,  # ty: ignore[invalid-argument-type]
        )
        project_root = get_project_root_path()
        prompt_aspects_without_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_aspects_no_tags.j2"
        )
        prompt_concepts_without_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_concepts_no_tags.j2"
        )
        prompt_aspects_with_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_aspects_with_tags.j2"
        )
        prompt_concepts_with_tags_path = (
            project_root
            / "tests"
            / "custom_prompts"
            / "custom_prompt_concepts_with_tags.j2"
        )
        with pytest.raises(ValueError, match="no Jinja2 tags"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_aspects_without_tags_path,
                prompt_type="aspect",  # type: ignore
            )
        with pytest.raises(ValueError, match="no Jinja2 tags"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_concepts_without_tags_path,
                prompt_type="concept",  # type: ignore
            )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_aspects_with_tags_path,
            prompt_type="aspect",  # type: ignore
        )
        assert (
            "test custom prompt template for aspects"
            in llm_with_updated_prompts._extract_aspect_items_prompt.render()
        )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_concepts_with_tags_path,
            prompt_type="concept",  # type: ignore
        )
        assert (
            "test custom prompt template for concepts"
            in llm_with_updated_prompts._extract_concept_items_prompt.render()
        )

    def test_input_output_token_validation(self):
        """
        Tests for max input and max output token validation.
        """

        # Test 1: Max input tokens validation
        llm = DocumentLLM(
            model="anthropic/claude-haiku-4-5",  # 200k context window
            api_key="...",  # dummy key (no LLM calls are made)
        )

        # Create a very long message that should exceed input limits
        long_message = "This is a test message. " * 50000
        messages = [
            {"role": "user", "content": long_message},
        ]

        with pytest.raises(
            ValueError, match="exceeds the model's maximum input tokens"
        ):
            llm._validate_input_tokens(messages)

        # Test 2: Max output tokens validation
        llm_excessive_output = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            max_tokens=999999,  # excessive output tokens
        )

        with pytest.raises(
            ValueError,
            match=r"max_tokens \(999999\) exceeds the model's maximum output tokens",
        ):
            llm_excessive_output._validate_output_tokens()

        # The same applies to a model without reasoning support
        with pytest.raises(
            ValueError, match="exceeds the model's maximum output tokens"
        ):
            DocumentLLM(
                model="cerebras/llama-3.3-70b",
                api_key="...",
                max_tokens=999999,
            )._validate_output_tokens()

        # Max output tokens within the limit of the model
        llm_excessive_output.max_tokens = 8000
        llm_excessive_output._validate_output_tokens()

        # Max output tokens not set (default): there is nothing to validate,
        # as no limit is sent, and the default of the model applies
        llm_excessive_output.max_tokens = None
        with patch.object(llms_module.litellm, "get_model_info") as get_model_info:
            llm_excessive_output._validate_output_tokens()
        get_model_info.assert_not_called()

    @pytest.mark.vcr
    def test_system_messages(self):
        """
        Tests the system messages functionality of LLMs.
        """
        system_message = "When asked, introduce yourself as ContextGem."

        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message=system_message,
        )
        model.chat("What's your name?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "ContextGem" in response
        logger.debug(response)

        # Test empty system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )
        assert model.system_message == ""

        # Test None system message (set to default system message) with default output language (en)
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert (
            model.system_message is not None
            and model.system_message == self.default_system_message_en
        )

        # Test None system message (set to default system message) with non-en output language (adapt)
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            output_language="adapt",
        )
        assert (
            model.system_message is not None
            and model.system_message == self.default_system_message_non_en
        )

        # Test with custom system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="You are a helpful assistant that can answer questions and help with tasks.",
        )
        assert (
            model.system_message
            == "You are a helpful assistant that can answer questions and help with tasks."
        )

    def test_model_api_request_retries(self):
        """
        Tests for retrying failed LLM API requests: only transient errors are retried,
        with a backoff between the retries, and each LLM, including each fallback LLM,
        applies its own retry params.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm

        def create_api_error(error_class: Any) -> Exception:
            """
            Creates a model API error of the specified litellm exception class.
            """
            return error_class(
                message="Simulated API error",
                model="openai/gpt-6-luna",
                llm_provider="openai",
            )

        rate_limit_error = create_api_error(litellm.RateLimitError)
        auth_error = create_api_error(litellm.AuthenticationError)

        # === Transient vs non-transient errors ===
        for error in [
            create_api_error(litellm.Timeout),
            rate_limit_error,
            create_api_error(litellm.InternalServerError),
            create_api_error(litellm.ServiceUnavailableError),
            create_api_error(litellm.APIConnectionError),
            RuntimeError("Error not classified by litellm"),
        ]:
            assert _is_retryable_model_api_error(error)
        for error in [
            create_api_error(litellm.BadRequestError),
            auth_error,
            create_api_error(litellm.NotFoundError),
            create_api_error(litellm.ContextWindowExceededError),
            create_api_error(litellm.ContentPolicyViolationError),
        ]:
            assert not _is_retryable_model_api_error(error)

        # === Backoff with jitter ===
        for retry_number, backoff in [
            (1, 1.0),
            (2, 2.0),
            (3, 4.0),
            (4, 8.0),
            (9, 10.0),
        ]:
            delays = {_get_model_api_retry_delay(retry_number) for _ in range(50)}
            assert all(backoff / 2 <= delay <= backoff for delay in delays)
            assert len(delays) > 1  # delays are randomized

        # === Retries of the requests ===
        primary_model = "openai/gpt-6-luna"
        fallback_model_1 = "openai/gpt-5.6-luna"
        fallback_model_2 = "openai/gpt-6.1-sol"
        invalid_answer = "<<<not-json>>>"
        concepts_answer = '[{"concept_id": "C0", "extracted_items": ["Item"]}]'

        requests: list[tuple[str, int]] = []  # (model, provider-specific retries)
        retry_numbers: list[int] = []
        outcomes: dict[str, list[Exception | str]] = {}

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, with the outcomes of the requests to each model
            taken from the queue of that model. The last outcome on the queue is repeated.
            """
            assert kwargs["num_retries"] == 0  # retries are not delegated to litellm
            model = kwargs["model"]
            requests.append((model, kwargs["max_retries"]))
            queue = outcomes[model]
            outcome = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(outcome, Exception):
                raise outcome
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=outcome)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def retry_delay_stub(retry_number: int) -> float:
            """
            Records the retry and removes the delay before it.
            """
            retry_numbers.append(retry_number)
            return 0.0

        def set_outcomes(model_outcomes: dict[str, list[Exception | str]]) -> None:
            """
            Sets the outcomes of the requests to each model and resets the records.
            """
            outcomes.clear()
            outcomes.update(model_outcomes)
            requests.clear()
            retry_numbers.clear()

        def requested_models() -> list[str]:
            """
            Returns the models in the order the requests were sent to them.
            """
            return [model for model, _ in requests]

        def create_llm(model: str = primary_model, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM, with the default retry params unless specified.
            """
            return DocumentLLM(
                model=model, api_key="test_api_key", system_message="", **kwargs
            )

        def create_llm_with_fallbacks() -> DocumentLLM:
            """
            Creates an LLM with two fallback LLMs, each with different retry params.
            """
            return create_llm(
                num_retries_failed_request=1,
                fallback_models=[
                    create_llm(
                        fallback_model_1,
                        num_retries_failed_request=2,
                        is_fallback=True,
                    ),
                    create_llm(
                        fallback_model_2,
                        num_retries_failed_request=3,
                        is_fallback=True,
                    ),
                ],
            )

        def create_document(n_concepts: int = 1) -> Document:
            """
            Creates a document with the specified number of concepts to extract.
            """
            document = Document(
                raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
            )
            document.concepts = [
                StringConcept(name=f"Concept {i}", description=f"Description {i}")
                for i in range(n_concepts)
            ]
            return document

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(
                llms_module, "_get_model_api_retry_delay", new=retry_delay_stub
            ),
        ):
            # Transient errors are retried until the request succeeds
            set_outcomes({primary_model: [rate_limit_error, rate_limit_error, "4"]})
            assert create_llm().chat("What's 2+2?") == "4"
            assert requested_models() == [primary_model] * 3
            assert retry_numbers == [1, 2]

            # ...or until the retries are exhausted
            for num_retries in [3, 1, 0]:
                set_outcomes({primary_model: [rate_limit_error]})
                with pytest.raises(ModelAPIError) as exc_info:
                    create_llm(num_retries_failed_request=num_retries).chat("Test")
                assert exc_info.value.retry_count == num_retries
                assert exc_info.value.original_error is rate_limit_error
                assert requested_models() == [primary_model] * (1 + num_retries)
                assert retry_numbers == list(range(1, num_retries + 1))

            # Non-transient errors are not retried
            set_outcomes({primary_model: [auth_error]})
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm().chat("Test")
            assert exc_info.value.retry_count == 0
            assert exc_info.value.original_error is auth_error
            assert requested_models() == [primary_model]
            assert retry_numbers == []
            # ...also when they follow a transient error
            set_outcomes({primary_model: [rate_limit_error, auth_error]})
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm().chat("Test")
            assert exc_info.value.retry_count == 1
            assert requested_models() == [primary_model] * 2

            # Provider-specific retries apply to the first attempt only
            set_outcomes({primary_model: [rate_limit_error, "4"]})
            assert create_llm(max_retries_failed_request=2).chat("Test") == "4"
            assert requests == [(primary_model, 2), (primary_model, 0)]

            # Chat: each LLM applies its own retry params before the next one is tried
            expected_requested_models = (
                [primary_model] * 2 + [fallback_model_1] * 3 + [fallback_model_2] * 3
            )
            set_outcomes(
                {
                    primary_model: [rate_limit_error],
                    fallback_model_1: [rate_limit_error],
                    fallback_model_2: [rate_limit_error, rate_limit_error, "4"],
                }
            )
            assert create_llm_with_fallbacks().chat("What's 2+2?") == "4"
            assert requested_models() == expected_requested_models

            # Extraction: same, and a failed request is not sent to the same LLM again
            set_outcomes(
                {
                    primary_model: [rate_limit_error],
                    fallback_model_1: [rate_limit_error],
                    fallback_model_2: [
                        rate_limit_error,
                        rate_limit_error,
                        concepts_answer,
                    ],
                }
            )
            document = create_document()
            create_llm_with_fallbacks().extract_concepts_from_document(document)
            assert requested_models() == expected_requested_models
            assert document.concepts[0].extracted_items[0].value == "Item"

            # Extraction: the last fallback LLM also retries its request before failing
            set_outcomes(
                {
                    primary_model: [auth_error],
                    fallback_model_1: [auth_error],
                    fallback_model_2: [rate_limit_error],
                }
            )
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm_with_fallbacks().extract_concepts_from_document(
                    create_document()
                )
            assert exc_info.value.retry_count == 3
            assert requested_models() == (
                [primary_model, fallback_model_1] + [fallback_model_2] * 4
            )

            # Extraction: invalid data is requested again without a delay
            set_outcomes({primary_model: [invalid_answer]})
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*2 retries"):
                create_llm(max_retries_invalid_data=2).extract_concepts_from_document(
                    create_document()
                )
            assert requested_models() == [primary_model] * 3
            assert retry_numbers == []
            # ...while a request that fails when repeated is passed to a fallback LLM
            set_outcomes(
                {
                    primary_model: [invalid_answer, rate_limit_error],
                    fallback_model_1: [concepts_answer],
                }
            )
            document = create_document()
            create_llm_with_fallbacks().extract_concepts_from_document(document)
            assert requested_models() == [primary_model] * 3 + [fallback_model_1]
            assert document.concepts[0].extracted_items[0].value == "Item"

            # Extraction with concurrency: a final API error is raised as is,
            # without sending the failed requests again. Such an error stops
            # the other calls (see `test_failed_concurrent_call_stops_other_calls`
            # in the extraction tests). Here, the request for the other concept is
            # sent nevertheless: each call sends its request as soon as it starts,
            # and the stubbed API fails the first request without waiting, i.e.
            # before the call of the other concept can be stopped.
            set_outcomes({primary_model: [auth_error]})
            with pytest.raises(ModelAPIError) as exc_info:
                create_llm().extract_concepts_from_document(
                    create_document(n_concepts=2), use_concurrency=True
                )
            assert exc_info.value.original_error is auth_error
            assert requested_models() == [primary_model] * 2  # each sent once

            # Extraction: a warning is issued if `raise_exception_on_extraction_error`
            # is False, without sending the failed request again
            set_outcomes({primary_model: [rate_limit_error]})
            document = create_document()
            with pytest.warns(UserWarning, match=r"LLM API.*3 retries"):
                create_llm().extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=False
                )
            assert requested_models() == [primary_model] * 4
            assert not document.concepts[0].extracted_items

    def test_extraction_response_cut_off_at_max_output_tokens(self):
        """
        Tests that a response cut off at the max output tokens fails the extraction
        with a dedicated error, without being requested from the same LLM again,
        and that a fallback LLM is tried for it.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"
        concepts_answer = '[{"concept_id": "C0", "extracted_items": ["Item"]}]'
        cut_off_answer = '[{"concept_id": "C0", "extracted_items": ["It'
        invalid_answer = "<<<not-json>>>"

        requests: list[str] = []
        outcomes: dict[str, list[tuple[str | None, str]]] = {}

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, with the outcomes of the requests to each model,
            as (text, finish reason), taken from the queue of that model. The last
            outcome on the queue is repeated.
            """
            model = kwargs["model"]
            requests.append(model)
            queue = outcomes[model]
            content, finish_reason = queue.pop(0) if len(queue) > 1 else queue[0]
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        finish_reason=finish_reason,
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        def set_outcomes(
            model_outcomes: dict[str, list[tuple[str | None, str]]],
        ) -> None:
            """
            Sets the outcomes of the requests to each model and resets the records.
            """
            outcomes.clear()
            outcomes.update(model_outcomes)
            requests.clear()

        def create_llm(*, with_fallback: bool = False, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with the default retry params, optionally with a fallback LLM.
            """
            llm = DocumentLLM(model=primary_model, api_key="test_api_key", **kwargs)
            if with_fallback:
                llm.fallback_models = [
                    DocumentLLM(
                        model=fallback_model, api_key="test_api_key", is_fallback=True
                    )
                ]
            return llm

        def create_document(n_concepts: int = 1) -> Document:
            """
            Creates a document with the specified number of concepts to extract.
            """
            document = Document(
                raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
            )
            document.concepts = [
                StringConcept(name=f"Concept {i}", description=f"Description {i}")
                for i in range(n_concepts)
            ]
            return document

        # The message names the limit that applies, and the reasoning tokens
        # only for a model that supports reasoning
        msg = create_llm()._get_output_truncated_msg()
        assert "max output tokens of the model" in msg
        assert "reasoning_effort" in msg
        msg = create_llm(max_tokens=100)._get_output_truncated_msg()
        assert "set by `max_tokens` (100)" in msg
        msg = DocumentLLM(
            model="cerebras/llama-3.3-70b", api_key="test_api_key"
        )._get_output_truncated_msg()
        assert "reasoning_effort" not in msg

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # The cut-off response is not requested again, although the retries
            # for invalid data are enabled, and its tokens are counted
            set_outcomes({primary_model: [(cut_off_answer, "length")]})
            llm = create_llm(max_tokens=100)
            assert llm.max_retries_invalid_data == 3
            with pytest.raises(
                ModelAPIError,
                match=r"cut off at the max output tokens set by `max_tokens` \(100\)"
                r".*max_items_per_call",
            ):
                llm.extract_concepts_from_document(create_document())
            assert requests == [primary_model]
            usage = llm.get_usage()[0].usage
            assert (usage.input, usage.output) == (10, 5)
            assert [call.response for call in usage.calls] == [cut_off_answer]

            # The same applies when all the tokens are used up before any text is returned
            set_outcomes({primary_model: [(None, "length")]})
            with pytest.raises(ModelAPIError, match="cut off at the max output tokens"):
                create_llm().extract_concepts_from_document(create_document())
            assert requests == [primary_model]

            # A fallback LLM is tried for the cut-off response
            set_outcomes(
                {
                    primary_model: [(cut_off_answer, "length")],
                    fallback_model: [(concepts_answer, "stop")],
                }
            )
            document = create_document()
            create_llm(with_fallback=True).extract_concepts_from_document(document)
            assert requests == [primary_model, fallback_model]
            assert document.concepts[0].extracted_items[0].value == "Item"

            # ...and the error is raised if the response of the last fallback LLM
            # is cut off as well
            set_outcomes(
                {
                    primary_model: [(cut_off_answer, "length")],
                    fallback_model: [(cut_off_answer, "length")],
                }
            )
            with pytest.raises(
                ModelAPIError, match=f"response of LLM `{fallback_model}` was cut off"
            ):
                create_llm(with_fallback=True).extract_concepts_from_document(
                    create_document()
                )
            assert requests == [primary_model, fallback_model]

            # A response cut off when invalid data is requested again stops the retries
            set_outcomes(
                {primary_model: [(invalid_answer, "stop"), (cut_off_answer, "length")]}
            )
            with pytest.raises(ModelAPIError, match="cut off at the max output tokens"):
                create_llm().extract_concepts_from_document(create_document())
            assert requests == [primary_model] * 2

            # Invalid data in a complete response is still requested again
            set_outcomes({primary_model: [(invalid_answer, "stop")]})
            with pytest.raises(ModelExtractionError, match=r"invalid JSON.*3 retries"):
                create_llm().extract_concepts_from_document(create_document())
            assert requests == [primary_model] * 4

            # With concurrency, the error is raised as is, without sending
            # the requests again. It stops the other calls, as a failed API request
            # does; the request for the other concept is sent nevertheless, as each
            # call sends its request as soon as it starts, and the stubbed API
            # answers the first request without waiting.
            set_outcomes({primary_model: [(cut_off_answer, "length")]})
            with pytest.raises(ModelAPIError, match="cut off at the max output tokens"):
                create_llm().extract_concepts_from_document(
                    create_document(n_concepts=2), use_concurrency=True
                )
            assert requests == [primary_model] * 2  # each sent once

            # A warning is issued if `raise_exception_on_extraction_error` is False,
            # without sending the request again
            set_outcomes({primary_model: [(cut_off_answer, "length")]})
            document = create_document()
            with pytest.warns(UserWarning, match="cut off at the max output tokens"):
                create_llm().extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=False
                )
            assert requests == [primary_model]
            assert not document.concepts[0].extracted_items

            # A response that is valid is used, whatever its finish reason
            set_outcomes({primary_model: [(concepts_answer, "length")]})
            document = create_document()
            create_llm().extract_concepts_from_document(document)
            assert requests == [primary_model]
            assert document.concepts[0].extracted_items[0].value == "Item"

    def test_unsupported_request_params_dropped(self):
        """
        Tests that a request with a param that litellm reports as not supported
        for the model, e.g. a temperature set for a model that accepts only its
        default one, is sent again without such params.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm
        model = "openai/gpt-6-luna"

        # litellm reports the temperature as not supported for the model
        # before any request is sent
        with pytest.raises(litellm.UnsupportedParamsError):
            litellm.utils.get_optional_params(
                model="gpt-6-luna", custom_llm_provider="openai", temperature=0.2
            )

        requests: list[dict[str, Any]] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates litellm, which raises an error for the unsupported temperature,
            unless it is told to drop the unsupported params.
            """
            requests.append(kwargs)
            if "temperature" in kwargs and not kwargs["drop_params"]:
                raise litellm.UnsupportedParamsError(
                    message="Simulated error: temperature is not supported",
                    model=model,
                    llm_provider="openai",
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content="4"))
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        def sent_params() -> list[tuple[float | None, bool]]:
            """
            Returns the temperature and the `drop_params` flag of each request.
            """
            return [(r.get("temperature"), r["drop_params"]) for r in requests]

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # The request is sent again, with the unsupported params to be dropped
            llm = DocumentLLM(
                model=model, api_key="test_api_key", system_message="", temperature=0.2
            )
            assert llm.chat("What's 2+2?") == "4"
            assert sent_params() == [(0.2, False), (0.2, True)]
            # Only the request that got a response is recorded
            usage = llm.get_usage()[0].usage
            assert (usage.input, usage.output) == (10, 5)
            assert len(usage.calls) == 1

            # No temperature is sent unless set
            requests.clear()
            llm = DocumentLLM(model=model, api_key="test_api_key", system_message="")
            assert llm.chat("What's 2+2?") == "4"
            assert sent_params() == [(None, False)]

    def test_async_limiter_per_llm(self):
        """
        Tests that, with concurrency enabled, the requests to each LLM, including
        each fallback LLM, are paced by the async limiter of that LLM.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"

        llm = DocumentLLM(
            model=primary_model,
            api_key="test_api_key",
            fallback_models=[
                DocumentLLM(
                    model=fallback_model, api_key="test_api_key", is_fallback=True
                )
            ],
        )
        llm.async_limiter = AsyncLimiter(10, 5)
        llm.fallback_models[0].async_limiter = AsyncLimiter(20, 5)

        acquisitions: list[tuple[float, float]] = []

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the config of the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.
                """
                acquisitions.append((self.max_rate, self.time_period))
                await super().acquire(amount)

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: requests to the primary LLM fail,
            requests to the fallback LLM succeed.
            """
            if kwargs["model"] == primary_model:
                raise litellm.AuthenticationError(
                    message="Simulated API error",
                    llm_provider="openai",
                    model=primary_model,
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"concept_id": "C0", "extracted_items": ["Item"]}]'
                        )
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        document = Document(
            raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
        )
        document.concepts = [
            StringConcept(name="Title", description="Contract title"),
            StringConcept(name="Parties", description="Contract parties"),
        ]

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
        ):
            # With concurrency, each concept is processed in a separate request:
            # first to the primary LLM, then to the fallback LLM
            llm.extract_concepts_from_document(document, use_concurrency=True)
            assert sorted(acquisitions) == [(10, 5)] * 2 + [(20, 5)] * 2
            assert all(i.extracted_items for i in document.concepts)

            # Without concurrency, requests are sent one by one, with no limiter
            acquisitions.clear()
            llm.extract_concepts_from_document(document, overwrite_existing=True)
            assert acquisitions == []
            assert all(i.extracted_items for i in document.concepts)

    def test_async_limiter_shared_by_concurrent_calls(self):
        """
        Tests that, with concurrency enabled, all the requests to an LLM that are
        sent in the same event loop are paced by the same limiter, i.e. share
        the rate set for the LLM: the requests of the extraction calls that run
        concurrently for the aspects, the sub-aspects and the concepts of different
        aspects, and those of the extractions that the calling code runs concurrently.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        llm.async_limiter = AsyncLimiter(1000, 1)

        # The limiter of each request, in the order the requests are paced
        limiters: list[AsyncLimiter] = []

        class RecordingAsyncLimiter(AsyncLimiter):
            """
            Records the limiter each time its capacity is acquired.
            """

            async def acquire(self, amount: float = 1) -> None:
                """
                Records the acquisition and acquires the capacity.
                """
                limiters.append(self)
                await super().acquire(amount)

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: each aspect is found in the first paragraph,
            and each concept is extracted. With concurrency, an aspect or a concept
            is processed in a request of its own, i.e. is the first one in it.
            """
            prompt = kwargs["messages"][-1]["content"]
            content = (
                '[{"concept_id": "C0", "extracted_items": [true]}]'
                if '"concept_id"' in prompt
                else '[{"aspect_id": "A0", "paragraph_ids": ["P0"]}]'
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        def create_document() -> Document:
            """
            Creates a document with three aspects, each with a sub-aspect and
            a concept, and with a document-level concept.
            """
            document = Document(
                raw_text="Services agreement\n\nThe parties agree as follows."
            )
            document.aspects = [
                Aspect(
                    name=f"Aspect {i}",
                    description=f"Clauses on topic {i}",
                    aspects=[
                        Aspect(
                            name=f"Sub-aspect {i}",
                            description=f"Clauses on a part of topic {i}",
                        )
                    ],
                    concepts=[
                        BooleanConcept(
                            name=f"Concept {i}", description=f"Topic {i} is covered"
                        )
                    ],
                )
                for i in range(3)
            ]
            document.concepts = [
                BooleanConcept(name="Agreement", description="It is an agreement")
            ]
            return document

        def check_extracted(document: Document) -> None:
            """
            Checks that each aspect, sub-aspect and concept of the document is
            extracted.
            """
            for aspect in document.aspects:
                assert aspect.extracted_items
                assert aspect.aspects[0].extracted_items
                assert aspect.concepts[0].extracted_items
            assert document.concepts[0].extracted_items

        # 3 aspects, 3 sub-aspects, 3 concepts of the aspects, 1 document concept
        n_requests = 10

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(models_module, "AsyncLimiter", new=RecordingAsyncLimiter),
        ):
            # === The extraction calls of one extraction ===
            document = create_document()
            llm.extract_all(document, use_concurrency=True)
            check_extracted(document)
            assert len(limiters) == n_requests
            (limiter,) = {id(i): i for i in limiters}.values()
            assert (limiter.max_rate, limiter.time_period) == (1000, 1)
            # The limiter set for the LLM gives the rate, and is not used itself
            assert limiter is not llm.async_limiter

            # === Extractions that the calling code runs concurrently ===
            async def extract_from_two_documents() -> list[Document]:
                """
                Runs the extraction from two documents concurrently, in the same
                event loop.
                """
                documents = [create_document(), create_document()]
                await asyncio.gather(
                    *[
                        llm.extract_all_async(document, use_concurrency=True)
                        for document in documents
                    ]
                )
                return documents

            limiters.clear()
            for document in asyncio.run(extract_from_two_documents()):
                check_extracted(document)
            assert len(limiters) == n_requests * 2
            assert len({id(i) for i in limiters}) == 1

            # === Another rate is set for the LLM ===
            limiters.clear()
            # The class of the limiter is the one that is patched in
            llm.async_limiter = RecordingAsyncLimiter(2000, 1)
            document = create_document()
            llm.extract_all(document, use_concurrency=True)
            check_extracted(document)
            (limiter,) = {id(i): i for i in limiters}.values()
            assert (limiter.max_rate, limiter.time_period) == (2000, 1)

    def test_request_cancelled_at_async_limiter_leaves_no_coroutine(self):
        """
        Tests that a request to an LLM that is cancelled while it waits for
        the limiter leaves no coroutine that was never awaited: the coroutine of
        the request is created only when the limiter lets the request through.
        Otherwise, Python issues a ``RuntimeWarning`` for each such request, e.g.
        when the extraction is cancelled on a timeout set by the calling code, or
        as another call of the extraction failed.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        # One request is let through, and the other requests wait for the limiter
        llm.async_limiter = AsyncLimiter(1, 60)
        n_requests_sent = 0

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API with a request that takes long to complete.
            """
            nonlocal n_requests_sent
            n_requests_sent += 1
            await asyncio.sleep(60)

        document = Document(
            raw_text="Services agreement\n\nThe parties agree as follows."
        )
        document.concepts = [
            StringConcept(name=f"Concept {i}", description=f"Detail {i}")
            for i in range(3)
        ]

        async def extract_with_timeout() -> None:
            """
            Runs the extraction, a concept per call, with a timeout that cancels
            it while one request is in flight and the others wait for the limiter.
            """
            await asyncio.wait_for(
                llm.extract_concepts_from_document_async(
                    document, use_concurrency=True
                ),
                timeout=0.5,
            )

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            warnings.catch_warnings(record=True) as caught_warnings,
        ):
            warnings.simplefilter("always")
            with pytest.raises(asyncio.TimeoutError):
                asyncio.run(extract_with_timeout())
            # The warning is issued when a coroutine that was never awaited
            # is destroyed
            gc.collect()
        assert n_requests_sent == 1
        never_awaited = [
            str(warning.message)
            for warning in caught_warnings
            if "was never awaited" in str(warning.message)
        ]
        assert not never_awaited, never_awaited

    def test_api_key_not_shown(self):
        """
        Tests that the API key of an LLM is not shown when the LLM is printed,
        on its own, as a fallback LLM or in a model group, while its API base is,
        and that no part of the key is shown in the text of a validation error
        of an LLM or of a model group, which does not show the inputs of
        the validation, nor given by the ``errors()`` and ``json()`` methods of
        the error for an LLM that is among the inputs.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        api_key = "sk-test-SECRET-0123456789"
        api_base = "https://llm.example.com"
        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=api_key,
            api_base=api_base,
            fallback_models=[
                DocumentLLM(
                    model="openai/gpt-5.6-luna", api_key=api_key, is_fallback=True
                )
            ],
        )
        other_llm = DocumentLLM(
            model="openai/gpt-5.6-luna", api_key=api_key, role="other"
        )
        model_group = DocumentModelGroup(models=[llm, other_llm])
        # The key is set, and is sent with the requests
        assert llm.api_key == api_key
        assert llm.fallback_models[0].api_key == api_key
        requests: list[dict[str, Any]] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Records the request and returns a response to it.
            """
            requests.append(kwargs)
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content="4"))
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            assert llm.chat("What's 2+2?") == "4"
        assert [request["api_key"] for request in requests] == [api_key]
        for instance in [llm, other_llm, model_group]:
            for text in [repr(instance), str(instance), f"{instance}"]:
                assert "SECRET" not in text
                assert "api_key" not in text
                assert "model='openai/gpt-" in text
        assert f"api_base='{api_base}'" in repr(llm)
        assert f"api_base='{api_base}'" in repr(model_group)

        def check_error(error: ValidationError, expected_texts: list[str]) -> None:
            """
            Checks that the validation error does not show the key, neither in
            its text, which shows no inputs of the validation, nor in the inputs
            that it reports, which the LLM is among: when what its ``errors()``
            method returns is printed, and in what its ``json()`` method writes out.

            :param error: The validation error.
            :type error: ValidationError
            :param expected_texts: The texts that the error is to have.
            :type expected_texts: list[str]
            """
            self.check_validation_error_hides_inputs(error, api_key, expected_texts)
            assert "DocumentLLM(" in repr(error.errors())
            self.check_validation_error_omits_model_keys(error, api_key)

        different_roles_error = ["models", "must have different roles"]
        output_language_error = ["must have the same value of `output_language`"]

        # The LLM as an invalid item of the fallback LLMs of another LLM:
        # it is not a fallback LLM
        with pytest.raises(ValidationError, match="must be a fallback model") as e:
            DocumentLLM(
                model="openai/gpt-6-luna", api_key=api_key, fallback_models=[other_llm]
            )
        check_error(e.value, ["fallback_models", "must be a fallback model"])
        # The `errors()` method of the error returns the LLM as it is, i.e. with
        # the key as its attribute, while its `json()` method writes the LLM out
        # with all its params but the key
        (error_details,) = e.value.errors()
        assert error_details["input"][0] is other_llm
        (written_error_details,) = json.loads(e.value.json())
        (written_llm,) = written_error_details["input"]
        assert list(written_llm) == [
            param for param in DocumentLLM.model_fields if param != "api_key"
        ]
        assert written_llm["model"] == other_llm.model
        assert written_llm["role"] == "other"
        # ... unless the inputs are left out
        (written_error_details,) = json.loads(e.value.json(include_input=False))
        assert "input" not in written_error_details
        # ... and as an invalid LLM of a group: with the role of another LLM
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup(models=[other_llm, other_llm])
        check_error(e.value, different_roles_error)
        with pytest.raises(ValidationError) as e:
            model_group.models = [other_llm, other_llm]
        check_error(e.value, different_roles_error)
        # ... or with another output language than the group has
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup(models=[llm, other_llm], output_language="adapt")
        check_error(e.value, output_language_error)
        with pytest.raises(ValidationError) as e:
            model_group.output_language = "adapt"
        check_error(e.value, output_language_error)
        assert model_group.models == [llm, other_llm]
        assert model_group.output_language == "en"

        # The key that is passed to a group by mistake: a group has no such param,
        # as the key is set for each of its LLMs
        invalid_group_params: list[dict[str, Any]] = [
            {"api_key": api_key, "models": [llm, other_llm]},
            {"models": [llm, other_llm], "api_key": api_key},
        ]
        for group_params in invalid_group_params:
            with pytest.raises(ValidationError) as e:
                DocumentModelGroup(**group_params)
            self.check_validation_error_hides_inputs(
                e.value, api_key, ["api_key", "Extra inputs are not permitted"]
            )

        # The LLM that is invalid itself, when it is created, changed or deserialized:
        # the key is among the inputs of its validation. For an LLM created with
        # the default system message, the input of a check that involves several
        # params is the LLM, which is shown, and written out, without the key.
        # For an LLM created with a system message of its own, it is all the params
        # that the LLM is created with: the key was shown if it was the first or
        # the last of them.
        model_params_cases: list[tuple[dict[str, Any], bool]] = [
            ({"model": "openai/gpt-6-luna"}, True),
            ({"model": "openai/gpt-6-luna", "system_message": "Be concise."}, False),
        ]
        for model_params, created_model_is_input in model_params_cases:
            self.check_model_validation_errors_hide_inputs(
                DocumentLLM,
                model_params,
                api_key,
                created_model_is_input=created_model_is_input,
            )
