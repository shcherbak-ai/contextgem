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
Module defining tests for usage, cost and pricing.
"""

from __future__ import annotations

import os
import types
from decimal import Decimal
from typing import Any
from unittest.mock import patch

import pytest

from contextgem.internal.base import llms as llms_module
from contextgem.internal.data_models import _ModelCost, _ModelUsage
from contextgem.internal.exceptions import ModelAPIError
from contextgem.internal.loggers import logger
from contextgem.internal.utils import _get_llm_cache_token_counts
from contextgem.public import (
    Document,
    DocumentLLM,
    DocumentModelGroup,
    ModelPricing,
    RatingConcept,
    StringConcept,
)
from tests.base import BaseTest
from tests.utils import get_test_document_text


class TestUsageAndCost(BaseTest):
    """
    Tests for tracking the usage and cost of models, and for model pricing.
    """

    def test_attr_models(self):
        """
        Tests for attribute validation models.
        """
        # Usage data model
        _ModelUsage(input=30, output=10)
        _ModelUsage(input=30, output=10, cache_read=20, cache_write=5)
        with pytest.raises(ValueError):
            _ModelUsage(input=-1, output=2)
        with pytest.raises(ValueError):
            _ModelUsage(input=30, output=10, cache_read=-1)
        with pytest.raises(ValueError):
            _ModelUsage(input=30, output=10, cache_write=-1)
        # Pricing data model
        ModelPricing(input_per_1m_tokens=0.00015, output_per_1m_tokens=0.0006)
        ModelPricing(
            input_per_1m_tokens=0.00015,
            output_per_1m_tokens=0.0006,
            cache_read_per_1m_tokens=0.000075,
            cache_write_per_1m_tokens=0.0001875,
        )
        with pytest.raises(ValueError):
            ModelPricing(input_per_1m_tokens=-1, output_per_1m_tokens=0)
        with pytest.raises(ValueError):
            ModelPricing(input_per_1m_tokens=0.1)
        # Cost data model
        _ModelCost(input=Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03"))
        with pytest.raises(ValueError):
            _ModelCost(
                input=-Decimal("0.01"), output=Decimal("0.02"), total=Decimal("0.03")
            )
        with pytest.raises(ValueError):
            _ModelCost(input=-Decimal("0.001"))

    def test_model_group_usage_and_cost_by_role(self):
        """
        Tests for retrieving and resetting the usage and cost of the LLMs in a group
        by the model role, which also applies to the fallback LLMs of the LLM with such role.
        """

        fallback = DocumentLLM(
            model="openai/gpt-5.6-luna",
            api_key="test_api_key",
            role="named",
            is_fallback=True,
        )
        primary = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key="test_api_key",
            role="named",
            fallback_models=[fallback],
        )
        llm_no_role = DocumentLLM(model="openai/gpt-6.1-sol", api_key="test_api_key")
        group = DocumentModelGroup(models=[primary, llm_no_role])
        for llm in [primary, fallback, llm_no_role]:
            llm._usage = _ModelUsage(input=10, output=5)
            llm._cost = _ModelCost(input=Decimal("1"), total=Decimal("1"))

        invalid_roles: list[Any] = ["", "  ", 1, "line\nbreak", "r" * 65]
        for getter in [group.get_usage, group.get_cost]:
            # The LLM with the role is selected together with its fallback LLMs
            selected = getter(model_role="named")
            assert [i.model for i in selected] == [primary.model, fallback.model]
            assert [i.is_fallback for i in selected] == [False, True]
            # All LLMs are selected if no role is specified
            assert [i.role for i in getter()] == ["named", "named", None]
            assert len(getter(model_role=None)) == 3
            with pytest.raises(ValueError, match="`missing`"):
                getter(model_role="missing")
            # The role is validated in the same way as the roles assigned to LLMs
            assert len(getter(model_role=" named ")) == 2
            for invalid_role in invalid_roles:
                with pytest.raises(ValueError, match="Model role must"):
                    getter(model_role=invalid_role)

        for invalid_role in invalid_roles:
            with pytest.raises(ValueError, match="Model role must"):
                group.reset_usage_and_cost(model_role=invalid_role)
        assert all(i.usage.input == 10 for i in group.get_usage())  # no change
        group.reset_usage_and_cost(model_role=" named ")
        assert [i.usage.input for i in group.get_usage()] == [0, 0, 10]
        assert [i.cost.total for i in group.get_cost()] == [0, 0, 1]
        with pytest.raises(ValueError, match="`missing`"):
            group.reset_usage_and_cost(model_role="missing")
        group.reset_usage_and_cost()
        assert all(i.usage.input == 0 for i in group.get_usage())
        assert all(i.cost.total == 0 for i in group.get_cost())

    @pytest.mark.parametrize("raise_exception_on_extraction_error", [True, False])
    def test_usage_recorded_when_extraction_fails(
        self, raise_exception_on_extraction_error: bool
    ):
        """
        Tests that the usage of the requests sent in an extraction is recorded when
        the extraction fails after some of them, whether or not the error is raised.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        class StubbedLLMAPIError(Exception):
            """
            Non-transient error of the stubbed LLM API, i.e. an error that is not retried.
            """

            status_code = 401

        requests: list[str] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, with the second request failing.
            """
            requests.append(kwargs["model"])
            if len(requests) == 2:
                raise StubbedLLMAPIError("Authentication failed")
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content='[{"concept_id": "C0", "extracted_items": ["Item"]}]'
                        )
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        # Three paragraphs processed one per call, i.e. in three requests
        document = Document(
            raw_text="First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        )
        document.concepts = [
            StringConcept(name="Topic", description="Topic of the text")
        ]

        with patch.object(llms_module.litellm, "acompletion", new=acompletion_stub):
            if raise_exception_on_extraction_error:
                with pytest.raises(ModelAPIError, match="Authentication failed"):
                    llm.extract_concepts_from_document(
                        document, max_paragraphs_to_analyze_per_call=1
                    )
            else:
                with pytest.warns(UserWarning, match="Authentication failed"):
                    llm.extract_concepts_from_document(
                        document,
                        max_paragraphs_to_analyze_per_call=1,
                        raise_exception_on_extraction_error=False,
                    )
        assert len(requests) == 2

        # The usage has the request that succeeded, and the request that failed
        # as an unfinished call
        usage = llm.get_usage()[0].usage
        assert (usage.input, usage.output) == (10, 5)
        assert [bool(call.timestamp_received) for call in usage.calls] == [True, False]

    def test_cached_tokens_usage_and_cost(self):
        """
        Tests for tracking the input tokens read from and written to the prompt cache
        of the LLM provider, and for pricing such tokens when calculating the costs.
        """

        litellm = llms_module.litellm
        answer = '[{"concept_id": "C0", "extracted_items": ["Item"]}]'

        # === Cache token counts reported by litellm ===
        usage_cases: list[tuple[dict[str, Any], tuple[int, int]]] = [
            # No cache tokens reported
            ({}, (0, 0)),
            ({"prompt_tokens_details": {"cached_tokens": None}}, (0, 0)),
            # As reported e.g. for OpenAI, Gemini
            ({"prompt_tokens_details": {"cached_tokens": 1500}}, (1500, 0)),
            # As reported e.g. for Anthropic, Bedrock
            ({"cache_read_input_tokens": 1200}, (1200, 0)),
            ({"cache_creation_input_tokens": 500}, (0, 500)),
            (
                {"cache_read_input_tokens": 1200, "cache_creation_input_tokens": 500},
                (1200, 500),
            ),
            # As reported e.g. for DeepSeek
            ({"prompt_cache_hit_tokens": 800}, (800, 0)),
            # Cache tokens are a part of the prompt tokens, i.e. cannot exceed them
            ({"prompt_tokens_details": {"cached_tokens": 3000}}, (2000, 0)),
            (
                {"cache_read_input_tokens": 1500, "cache_creation_input_tokens": 900},
                (1500, 500),
            ),
        ]
        for usage_kwargs, expected_counts in usage_cases:
            completion_usage = litellm.Usage(
                prompt_tokens=2000, completion_tokens=100, **usage_kwargs
            )
            assert _get_llm_cache_token_counts(completion_usage) == expected_counts
        # Usage data without the prompt token details
        assert _get_llm_cache_token_counts(
            types.SimpleNamespace(prompt_tokens=2000, completion_tokens=100)
        ) == (0, 0)

        # === Usage and cost of an LLM ===
        completion_usages: list[Any] = []  # usage data of the upcoming completions

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API, with the usage data of each completion
            taken from the queue.
            """
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(message=types.SimpleNamespace(content=answer))
                ],
                usage=completion_usages.pop(0),
            )

        def create_llm(model: str = "openai/gpt-6-luna", **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with the specified pricing params.
            """
            return DocumentLLM(
                model=model, api_key="test_api_key", system_message="", **kwargs
            )

        def queue_completion_usages() -> None:
            """
            Queues the usage data of two completions: the first one writes
            a part of the prompt to the cache, the second one reads it from the cache.
            """
            completion_usages[:] = [
                litellm.Usage(
                    prompt_tokens=200_000,
                    completion_tokens=10_000,
                    cache_creation_input_tokens=150_000,
                ),
                litellm.Usage(
                    prompt_tokens=200_000,
                    completion_tokens=10_000,
                    prompt_tokens_details={"cached_tokens": 150_000},
                ),
            ]

        def calc_auto_prices(
            model: str, **usage_kwargs: int
        ) -> tuple[Decimal, Decimal]:
            """
            Calculates the input and output prices of the usage with genai-prices.
            """
            provider_id, model_ref = model.split("/")
            price = llms_module._calc_auto_price(
                llms_module._GPUsage(**usage_kwargs),
                model_ref=model_ref,
                provider_id=provider_id,
            )
            return price.input_price, price.output_price

        def check_usage(llm: DocumentLLM) -> None:
            """
            Checks the usage of the LLM after the two queued completions.
            """
            assert not completion_usages  # both completions are received
            usage = llm.get_usage()[0].usage
            assert (usage.input, usage.output) == (400_000, 20_000)
            assert (usage.cache_read, usage.cache_write) == (150_000, 150_000)
            assert len(usage.calls) == 2

        document = Document(
            raw_text="Non-disclosure agreement\n\nThe parties agree as follows."
        )
        document.concepts = [StringConcept(name="Title", description="Contract title")]

        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # Cache tokens are priced at the cache prices, if set
            for pricing_kwargs, expected_input_cost in [
                # 100K regular * 1.0 + 150K read * 0.1 + 150K written * 1.25
                (
                    {
                        "cache_read_per_1m_tokens": 0.1,
                        "cache_write_per_1m_tokens": 1.25,
                    },
                    Decimal("0.3025"),
                ),
                # 250K regular and written * 1.0 + 150K read * 0.1
                ({"cache_read_per_1m_tokens": 0.1}, Decimal("0.265")),
                # 250K regular and read * 1.0 + 150K written * 1.25
                ({"cache_write_per_1m_tokens": 1.25}, Decimal("0.4375")),
                # 150K read * 0.0 (cache tokens can be free of charge)
                ({"cache_read_per_1m_tokens": 0.0}, Decimal("0.25")),
                # ...and as regular input tokens otherwise: 400K * 1.0
                ({}, Decimal("0.4")),
            ]:
                llm = create_llm(
                    pricing_details=ModelPricing(
                        input_per_1m_tokens=1.0,
                        output_per_1m_tokens=4.0,
                        **pricing_kwargs,
                    )
                )
                # Each paragraph is processed in a separate request,
                # with the usage of the requests merged
                queue_completion_usages()
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=1,
                )
                check_usage(llm)
                cost = llm.get_cost()[0].cost
                assert cost.input == expected_input_cost
                assert cost.output == Decimal("0.08")  # 20K * 4.0
                assert cost.total == cost.input + cost.output
                self._check_deserialized_llm_config_eq(llm)

            # The same applies to chat
            llm = create_llm(
                pricing_details=ModelPricing(
                    input_per_1m_tokens=1.0,
                    output_per_1m_tokens=4.0,
                    cache_read_per_1m_tokens=0.1,
                    cache_write_per_1m_tokens=1.25,
                )
            )
            queue_completion_usages()
            llm.chat("Test")
            llm.chat("Test")
            check_usage(llm)
            assert llm.get_cost()[0].cost.input == Decimal("0.3025")

            llm.reset_usage_and_cost()
            usage = llm.get_usage()[0].usage
            assert (usage.cache_read, usage.cache_write) == (0, 0)

            # Tokens are counted whether or not pricing is configured
            llm = create_llm()
            queue_completion_usages()
            llm.extract_concepts_from_document(
                document, overwrite_existing=True, max_paragraphs_to_analyze_per_call=1
            )
            check_usage(llm)
            cost = llm.get_cost()[0].cost
            assert (cost.input, cost.output, cost.total) == (0, 0, 0)

            # Auto-pricing: cache tokens are priced at the cache prices of the model.
            # Each request is priced on its own, also when the usage of the requests
            # is merged, as the price tier of a model with tiered prices depends
            # on the number of input tokens in a request.
            for model, has_price_tiers in [
                ("anthropic/claude-haiku-4-5", False),
                ("openai/gpt-6-luna", True),  # higher prices above 272K
            ]:
                with pytest.warns(
                    UserWarning, match="prices will not be 100% accurate."
                ):
                    llm = create_llm(model, auto_pricing=True)
                queue_completion_usages()
                llm.extract_concepts_from_document(
                    document,
                    overwrite_existing=True,
                    max_paragraphs_to_analyze_per_call=1,
                )
                check_usage(llm)
                request_prices = [
                    calc_auto_prices(
                        model,
                        input_tokens=200_000,
                        output_tokens=10_000,
                        **cache_tokens,
                    )
                    for cache_tokens in [
                        {"cache_write_tokens": 150_000},
                        {"cache_read_tokens": 150_000},
                    ]
                ]
                cost = llm.get_cost()[0].cost
                assert cost.input == sum(i[0] for i in request_prices)
                assert cost.output == sum(i[1] for i in request_prices)
                # ...which differs from the price of the same tokens without the cache
                assert (
                    cost.input
                    != 2
                    * calc_auto_prices(
                        model, input_tokens=200_000, output_tokens=10_000
                    )[0]
                )
                # ...and from the price of the merged usage, if it gets to another tier
                merged_usage_prices = calc_auto_prices(
                    model,
                    input_tokens=400_000,
                    output_tokens=20_000,
                    cache_read_tokens=150_000,
                    cache_write_tokens=150_000,
                )
                assert (
                    merged_usage_prices != (cost.input, cost.output)
                ) == has_price_tiers

        # === Missing pricing data ===
        # The warning suggests refreshing the pricing data, as the bundled data
        # may not include a recently released model
        with pytest.warns(UserWarning, match="auto_pricing_refresh=True"):
            llm = create_llm("openai/gptXX", auto_pricing=True)
        # ...unless the refresh is already enabled
        llm.auto_pricing_refresh = True
        assert "auto_pricing_refresh" not in llm._get_missing_auto_pricing_advice()

        # === Pricing model ===
        invalid_pricing_cases: list[dict[str, Any]] = [
            {"cache_read_per_1m_tokens": -0.1},
            {"cache_write_per_1m_tokens": -0.1},
            {"cache_read_per_1m_tokens": "0.1"},
        ]
        for invalid_pricing_kwargs in invalid_pricing_cases:
            with pytest.raises(ValueError):
                ModelPricing(
                    input_per_1m_tokens=1.0,
                    output_per_1m_tokens=4.0,
                    **invalid_pricing_kwargs,
                )
        # Pricing serialized without the cache prices can still be deserialized
        pricing = ModelPricing(input_per_1m_tokens=1.0, output_per_1m_tokens=4.0)
        pricing_dict = pricing.to_dict()
        for key in ["cache_read_per_1m_tokens", "cache_write_per_1m_tokens"]:
            assert pricing_dict.pop(key) is None
        assert ModelPricing.from_dict(pricing_dict) == pricing

    @pytest.mark.vcr
    def test_auto_pricing(self):
        """
        Tests for optional auto-pricing using genai-prices when no manual pricing is provided.
        """

        # Use an LLM with auto pricing enabled and no manual pricing details
        with pytest.warns(UserWarning, match="prices will not be 100% accurate."):
            llm = DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                auto_pricing=True,
                auto_pricing_refresh=True,  # test with auto-refresh
            )
        assert (
            llm._auto_pricing_refresh_attempted
        )  # since we use dummy call to check auto-pricing support for provider/model
        document = Document(raw_text=get_test_document_text())
        document.concepts = [
            RatingConcept(
                name="NDA compliance rating",
                description="Rating of the NDA's compliance with best practices",
                rating_scale=(1, 10),
                add_justifications=True,
                justification_max_sents=5,
            ),
        ]
        llm.extract_concepts_from_document(document)
        cost_info = llm.get_cost()
        assert len(cost_info) == 1
        cost = cost_info[0].cost
        assert cost.input > Decimal("0")
        assert cost.output > Decimal("0")
        assert cost.total == cost.input + cost.output
        logger.debug(
            f"Costs for model {llm.model} (auto-pricing): "
            f"input {cost.input}, output {cost.output}, total {cost.total}"
        )

        # Check serialization of LLM
        self._check_deserialized_llm_config_eq(llm)

        # Test errors

        # Local LLMs
        with pytest.raises(ValueError, match="local models"):
            llm = DocumentLLM(
                model="lm_studio/mistralai/mistral-small-3.2",
                api_base="http://localhost:1234/v1",
                api_key="random-key",  # required for LM Studio API
                auto_pricing=True,
            )
        with pytest.raises(ValueError, match="local models"):
            DocumentLLM(
                model="ollama_chat/mistral-small:24b",
                api_base="http://localhost:11434",
                auto_pricing=True,
            )
        with pytest.raises(ValueError, match="local models"):
            DocumentLLM(
                model="ollama/mistral-small:24b",
                api_base="http://localhost:11434",
                auto_pricing=True,
            )

        # Setting auto-pricing and ModelPricing together
        with pytest.raises(ValueError, match="auto_pricing=True"):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.00015, output_per_1m_tokens=0.0006
                ),
                auto_pricing=True,
            )

        # Test warnings

        # Unknown model initialization
        with pytest.warns(UserWarning, match="Unable to fetch pricing data for model"):
            llm = DocumentLLM(
                model="openai/gptXX",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                auto_pricing=True,
            )
