# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for usage, cost and pricing.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import types
import warnings
from decimal import Decimal
from typing import Any, cast
from unittest.mock import patch

import pytest
from litellm.types.decisions import (
    OpenAIDecisionInputTokensDetails,
    OpenAIDecisionResponse,
    OpenAIDecisionUsage,
    OpenAIPredicateAnswer,
)

from contextgem.internal.base import llms as llms_module
from contextgem.internal.base import models as models_module
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.data_models import _ModelCall, _ModelCost, _ModelUsage
from contextgem.internal.exceptions import ModelAPIError
from contextgem.internal.loggers import logger
from contextgem.internal.utils import _get_llm_cache_token_counts
from contextgem.public import (
    BooleanConcept,
    Document,
    DocumentDecisionModel,
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
            ModelPricing(input_per_1m_tokens=0.1)  # ty: ignore[missing-argument]
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
            llm._cost = _ModelCost(input=Decimal(1), total=Decimal(1))

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

    def test_usage_recorded_when_error_is_raised_after_response(self):
        """
        Tests that the usage and cost of a request that got a response are recorded,
        once, when an error is raised afterwards: when the response is processed,
        e.g. as a concept is found processed by another extraction that ran
        meanwhile, or before the next request is sent. This holds with and without
        concurrency, for an LLM, a fallback LLM, a decision model and the models of
        a group. A request that failed is recorded as a call without a response,
        and a request cancelled in flight is not recorded.

        The usage of a request to an LLM was lost in such case: no usage is returned
        when an error is raised.

        The model APIs are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        llm_model = "openai/gpt-6-luna"
        fallback_llm_model = "openai/gpt-5.6-luna"
        decision_model = "typesafe/jev-latest"
        cut_off_answer = '[{"concept_id": "C0", "extracted_items": [tr'
        # Each input token costs 1, and each output token costs 2
        pricing = ModelPricing(
            input_per_1m_tokens=1_000_000.0, output_per_1m_tokens=2_000_000.0
        )
        # What the request with a given number does before it is answered
        actions: dict[int, Any] = {}
        # The models of the requests sent, answered, failed and cancelled in flight
        requests: list[str] = []
        responses: list[str] = []
        failed_requests: list[str] = []
        cancelled_requests: list[str] = []

        async def send_request(model: str) -> Any:
            """
            Records the request, and does what is set for the request with its
            number before it is answered: raises the error, or calls the function,
            waiting for what the function returns if it is to be waited for.
            Returns what the function returns.
            """
            requests.append(model)
            action = actions.get(len(requests))
            if isinstance(action, Exception):
                failed_requests.append(model)
                raise action
            result = action() if action else None
            if asyncio.iscoroutine(result):
                try:
                    result = await result
                except asyncio.CancelledError:
                    cancelled_requests.append(model)
                    raise
            responses.append(model)
            return result

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates the LLM API: responds with the text that the request is set
            to be answered with, which is cut off if it is the cut-off answer, or
            with an item for each concept of the prompt. Each response uses 10 input
            tokens and 5 output tokens.
            """
            content = await send_request(kwargs["model"])
            if content is None:
                content = json.dumps(
                    [
                        {"concept_id": concept_id, "extracted_items": [True]}
                        for concept_id in re.findall(
                            r"<concept_id>(C\d+)</concept_id>",
                            kwargs["messages"][-1]["content"],
                        )
                    ]
                )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content),
                        finish_reason="length" if content == cut_off_answer else "stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        async def adecisions_stub(**kwargs: Any):
            """
            Simulates the API of the decision model: answers "yes" to each question.
            Each response uses 10 input tokens and no output tokens.
            """
            await send_request(kwargs["model"])
            return OpenAIDecisionResponse(
                model="jev-1.13.0",
                answers=tuple(
                    OpenAIPredicateAnswer(name=question["name"], probability=0.9)
                    for question in kwargs["questions"]
                ),
                usage=OpenAIDecisionUsage(
                    input_tokens=10,
                    output_tokens=0,
                    total_tokens=10,
                    input_tokens_details=OpenAIDecisionInputTokensDetails(
                        cached_tokens=0, cache_write_tokens=0
                    ),
                ),
            )

        def create_llm(model: str = llm_model, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with the pricing set.
            """
            return DocumentLLM(
                model=model, api_key="test_api_key", pricing_details=pricing, **kwargs
            )

        def create_decision_model(**kwargs: Any) -> DocumentDecisionModel:
            """
            Creates a decision model with the pricing set.
            """
            return DocumentDecisionModel(
                model=decision_model,
                api_key="test_api_key",
                pricing_details=pricing,
                **kwargs,
            )

        def create_document(n_concepts: int = 1, n_paragraphs: int = 2) -> Document:
            """
            Creates a document with concepts that both an LLM and a decision model
            can extract. In a model group, each second concept is routed to
            the model with the role "other".
            """
            document = Document(
                raw_text="\n\n".join(
                    f"Paragraph {i} of the agreement." for i in range(n_paragraphs)
                )
            )
            document.concepts = [
                BooleanConcept(
                    name=f"Concept {i}",
                    description=f"Topic {i} is covered",
                    model_role="other" if i % 2 else None,
                )
                for i in range(n_concepts)
            ]
            return document

        def set_actions(request_actions: dict[int, Any]) -> None:
            """
            Sets what the requests do, by their numbers, and resets the records.
            """
            actions.clear()
            actions.update(request_actions)
            for record in (requests, responses, failed_requests, cancelled_requests):
                record.clear()

        def process_meanwhile(concept: _Concept) -> Any:
            """
            Returns the action of a request during which the concept is processed,
            as by another extraction, i.e. the concept is found processed when
            the response to the request is processed.
            """

            def mark_processed() -> None:
                """
                Marks the concept as processed.
                """
                concept._is_processed = True

            return mark_processed

        def process_when_extracted(concept: _Concept, other_concept: _Concept) -> Any:
            """
            Returns the action of a request that is answered when the other concept
            is extracted by another call, the concept being processed meanwhile.
            """

            async def wait_and_mark_processed() -> None:
                """
                Waits for the other concept to be extracted, and marks the concept
                as processed.
                """
                for _ in range(1000):
                    if other_concept._is_processed:
                        break
                    await asyncio.sleep(0)
                concept._is_processed = True

            return wait_and_mark_processed

        async def cancel_extraction() -> None:
            """
            The action of a request during which the extraction is cancelled, as
            on a timeout set by the calling code.
            """
            # Safe cast: the request is sent in a task
            cast(asyncio.Task, asyncio.current_task()).cancel()
            await asyncio.sleep(0)

        async def extract_twice(
            model: DocumentLLM | DocumentDecisionModel, document: Document
        ) -> tuple[Any, Any]:
            """
            Runs two extractions of the concepts of the document at the same time,
            and returns what each extraction returns, or the error that it raises.
            """
            return await asyncio.gather(
                model.extract_concepts_from_document_async(document),
                model.extract_concepts_from_document_async(document),
                return_exceptions=True,
            )

        def check_usage(
            model: DocumentLLM | DocumentDecisionModel | DocumentModelGroup,
            expected_responses: list[str],
        ) -> None:
            """
            Checks that the requests of the given models, in this order, got
            a response, and that the usage and cost recorded for the model, with
            its fallback models, or for the models of the group, are those of
            the requests made: each request that got a response is recorded once,
            with its tokens and their cost, each request that failed is recorded
            as a call without a response, and a request cancelled in flight is
            not recorded.
            """
            assert responses == expected_responses
            for usage_info, cost_info in zip(
                model.get_usage(), model.get_cost(), strict=True
            ):
                usage = usage_info.usage
                n_responses = responses.count(usage_info.model)
                n_output_tokens = 0 if usage_info.model == decision_model else 5
                assert usage.input == 10 * n_responses
                assert usage.output == n_output_tokens * n_responses
                assert cost_info.cost.total == usage.input + 2 * usage.output
                assert [
                    call.timestamp_received is not None for call in usage.calls
                ].count(True) == n_responses
                assert len(usage.calls) - n_responses == failed_requests.count(
                    usage_info.model
                )
            assert len(requests) == (
                len(responses) + len(failed_requests) + len(cancelled_requests)
            )

        api_error = litellm.AuthenticationError(
            message="Simulated API error", model=llm_model, llm_provider="openai"
        )
        already_processed_error = "have already been processed"

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(litellm, "adecisions", new=adecisions_stub),
        ):
            # === Two extractions of the same concept that overlap ===
            # The request of the first extraction is answered when the second
            # extraction is completed, i.e. when the concept is processed
            for create_model in (create_llm, create_decision_model):
                model = create_model()
                set_actions({1: lambda: asyncio.sleep(0)})
                first_outcome, second_outcome = asyncio.run(
                    extract_twice(model, create_document())
                )
                assert isinstance(first_outcome, ValueError)
                assert already_processed_error in str(first_outcome)
                assert not isinstance(second_outcome, BaseException)
                # Both requests got a response
                check_usage(model, [model.model] * 2)

            # === A concept that is found processed when the response is processed ===
            for create_model in (create_llm, create_decision_model):
                # Without concurrency, and with it, in which case a single call
                # is made as well
                for use_concurrency in (False, True):
                    model = create_model()
                    document = create_document()
                    set_actions({1: process_meanwhile(document.concepts[0])})
                    with pytest.raises(ValueError, match=already_processed_error):
                        model.extract_concepts_from_document(
                            document, use_concurrency=use_concurrency
                        )
                    check_usage(model, [model.model])

                # With concurrency and several calls: the request of the first
                # call is answered when the second call is completed, and
                # the request of the third call is in flight
                model = create_model()
                document = create_document(n_concepts=3)
                set_actions(
                    {
                        1: process_when_extracted(*document.concepts[:2]),
                        3: lambda: asyncio.sleep(5),
                    }
                )
                with pytest.raises(ValueError, match=already_processed_error):
                    model.extract_concepts_from_document(document, use_concurrency=True)
                assert cancelled_requests == [model.model]
                # The requests of the second and the first call got a response
                check_usage(model, [model.model] * 2)

            # === The request that is sent again, as the LLM returned invalid data ===
            llm = create_llm()
            document = create_document()
            set_actions(
                {
                    1: lambda: "<<<not-json>>>",
                    2: process_meanwhile(document.concepts[0]),
                }
            )
            with pytest.raises(ValueError, match=already_processed_error):
                llm.extract_concepts_from_document(document)
            check_usage(llm, [llm_model] * 2)

            # === The request to a fallback model, after the request to the LLM failed ===
            for fallback_model in (
                create_llm(fallback_llm_model, is_fallback=True),
                create_decision_model(is_fallback=True),
            ):
                llm = create_llm(fallback_models=[fallback_model])
                document = create_document()
                set_actions({1: api_error, 2: process_meanwhile(document.concepts[0])})
                with pytest.raises(ValueError, match=already_processed_error):
                    llm.extract_concepts_from_document(document)
                assert failed_requests == [llm_model]
                check_usage(llm, [fallback_model.model])

            # === A call that takes several requests, one for each paragraph ===
            # The concept is found processed when the second response is processed
            llm = create_llm()
            document = create_document(n_paragraphs=3)
            set_actions({2: process_meanwhile(document.concepts[0])})
            with pytest.raises(ValueError, match=already_processed_error):
                llm.extract_concepts_from_document(
                    document, max_paragraphs_to_analyze_per_call=1
                )
            check_usage(llm, [llm_model] * 2)

            # An error is raised after the first response is processed, when
            # the second request is prepared
            llm = create_llm()
            build_user_message = DocumentLLM._build_user_message
            error = RuntimeError("Simulated error")

            def build_first_user_message(*args: Any, **kwargs: Any) -> Any:
                """
                Builds the user message of the first request, and fails to build
                the message of the next one.
                """
                if requests:
                    raise error
                return build_user_message(*args, **kwargs)

            set_actions({})
            with (
                patch.object(
                    DocumentLLM, "_build_user_message", new=build_first_user_message
                ),
                pytest.raises(RuntimeError) as exc_info,
            ):
                llm.extract_concepts_from_document(
                    create_document(n_paragraphs=3),
                    max_paragraphs_to_analyze_per_call=1,
                )
            assert exc_info.value is error
            check_usage(llm, [llm_model])

            # The extraction is cancelled with the second request in flight:
            # the first request, which the usage and cost of the LLM are already
            # updated with, is not recorded again
            llm = create_llm()
            set_actions({2: cancel_extraction})
            with pytest.raises(asyncio.CancelledError):
                llm.extract_concepts_from_document(
                    create_document(n_paragraphs=3),
                    max_paragraphs_to_analyze_per_call=1,
                )
            assert cancelled_requests == [llm_model]
            check_usage(llm, [llm_model])

            # === A response cut off at the max output tokens ===
            # It fails the call with a model API error, which is raised when
            # the usage and cost of the LLM are already updated
            llm = create_llm()
            set_actions({1: lambda: cut_off_answer})
            with pytest.raises(ModelAPIError, match="cut off at the max output tokens"):
                llm.extract_concepts_from_document(create_document())
            check_usage(llm, [llm_model])

            # === An error of another type, raised when the response is processed ===
            llm = create_llm()
            set_actions({})
            with (
                patch.object(
                    models_module, "_validate_parsed_llm_output", side_effect=error
                ),
                pytest.raises(RuntimeError) as exc_info,
            ):
                llm.extract_concepts_from_document(create_document())
            assert exc_info.value is error
            check_usage(llm, [llm_model])

            # === A model group ===
            # The concept of the decision model is extracted, and the concept of
            # the LLM is found processed when the response of the LLM is processed
            for use_concurrency in (False, True):
                group = DocumentModelGroup(
                    models=[create_decision_model(), create_llm(role="other")]
                )
                document = create_document(n_concepts=2)
                set_actions({2: process_meanwhile(document.concepts[1])})
                with pytest.raises(ValueError, match=already_processed_error):
                    group.extract_concepts_from_document(
                        document, use_concurrency=use_concurrency
                    )
                check_usage(group, [decision_model, llm_model])

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
            price = models_module._calc_auto_price(
                models_module._GPUsage(**usage_kwargs),
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

    def test_auto_pricing_follows_assigned_params(self):
        """
        Tests that the automatic pricing of a model follows the values assigned to
        its `model` and `auto_pricing` params after the model is created: the costs
        are calculated with the prices of the current model, as for a model created
        with the same params, for LLMs and decision models.
        """

        def get_cost_of_call(model: DocumentLLM | DocumentDecisionModel) -> Decimal:
            """
            Records the usage of one call of 1M input and 1M output tokens for
            the model, and returns the total cost of the model.
            """
            model.reset_usage_and_cost()
            call = _ModelCall(
                prompt_kwargs={}, prompt="Prompt", response="Response", timestamp_sent=0
            )
            call._input_tokens = 1_000_000
            call._output_tokens = 1_000_000
            with warnings.catch_warnings():
                # The costs of a model without prices are skipped with a warning
                warnings.simplefilter("ignore")
                model._increment_cost(
                    _ModelUsage(input=1_000_000, output=1_000_000, calls=[call])
                )
            return model.get_cost()[0].cost.total

        def get_warning_messages(
            caught_warnings: list[warnings.WarningMessage],
        ) -> list[str]:
            """
            Returns the messages of the caught warnings.
            """
            return [str(caught_warning.message) for caught_warning in caught_warnings]

        def create_llm(model: str, **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with a dummy API key.
            """
            return DocumentLLM(model=model, api_key="test_api_key", **kwargs)

        def check_warning_attributed_to_this_file(
            caught_warnings: pytest.WarningsRecorder, match: str
        ) -> None:
            """
            Checks that the warning that has the given text is attributed to
            this file, i.e. to the code that creates the model or assigns a value
            to its param, and not to the framework or to Pydantic.
            """
            (caught_warning,) = [
                caught_warning
                for caught_warning in caught_warnings
                if match in str(caught_warning.message)
            ]
            assert os.path.normcase(caught_warning.filename) == os.path.normcase(
                __file__
            )

        with pytest.warns(
            UserWarning, match="not be 100% accurate"
        ) as caught_pricing_warnings:
            llm = create_llm("openai/gpt-6-luna", auto_pricing=True)
        check_warning_attributed_to_this_file(
            caught_pricing_warnings, "not be 100% accurate"
        )
        cost_of_model = get_cost_of_call(llm)
        with pytest.warns(UserWarning, match="not be 100% accurate"):
            other_llm = create_llm("openai/gpt-5.6-luna", auto_pricing=True)
        cost_of_other_model = get_cost_of_call(other_llm)
        assert Decimal(0) < cost_of_model != cost_of_other_model

        # === The model is changed ===
        # The costs are calculated with the prices of the new model, which are
        # looked up without the warning on the accuracy of the prices again
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            llm.model = "openai/gpt-5.6-luna"
        assert not any(
            "100% accurate" in message
            for message in get_warning_messages(caught_warnings)
        )
        assert llm._auto_pricing_model_ref == "gpt-5.6-luna"
        assert get_cost_of_call(llm) == cost_of_other_model
        # ... and are not calculated with the prices of the previous model, if
        # there are no prices for the new one
        with pytest.warns(
            UserWarning, match="Unable to fetch pricing data"
        ) as caught_pricing_warnings:
            llm.model = "openai/model-without-prices"
        check_warning_attributed_to_this_file(
            caught_pricing_warnings, "Unable to fetch pricing data"
        )
        assert get_cost_of_call(llm) == Decimal(0)
        # The prices are not looked up again when another param is assigned
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            llm.timeout = 60
        assert not any(
            "pricing data" in message
            for message in get_warning_messages(caught_warnings)
        )

        # === The automatic pricing is enabled and disabled ===
        llm = create_llm("openai/gpt-6-luna")
        assert llm._auto_pricing_provider_id is None
        assert get_cost_of_call(llm) == Decimal(0)
        with pytest.warns(
            UserWarning, match="not be 100% accurate"
        ) as caught_pricing_warnings:
            llm.auto_pricing = True
        check_warning_attributed_to_this_file(
            caught_pricing_warnings, "not be 100% accurate"
        )
        assert get_cost_of_call(llm) == cost_of_model
        llm.auto_pricing = False
        assert llm._auto_pricing_provider_id is None
        assert llm._auto_pricing_model_ref is None
        assert get_cost_of_call(llm) == Decimal(0)
        # ... also when the model is changed while it is disabled
        llm.model = "openai/gpt-5.6-luna"
        with pytest.warns(UserWarning, match="not be 100% accurate"):
            llm.auto_pricing = True
        assert get_cost_of_call(llm) == cost_of_other_model

        # === A decision model ===
        with pytest.warns(UserWarning, match="not be 100% accurate"):
            decision_model = DocumentDecisionModel(
                model="typesafe/jev-latest", api_key="test_api_key", auto_pricing=True
            )
        cost_of_decision_model = get_cost_of_call(decision_model)
        decision_model.model = "perplexity/pplx-decider-v1-27b"
        with pytest.warns(UserWarning, match="not be 100% accurate"):
            other_decision_model = DocumentDecisionModel(
                model="perplexity/pplx-decider-v1-27b",
                api_key="test_api_key",
                auto_pricing=True,
            )
        cost_of_other_decision_model = get_cost_of_call(other_decision_model)
        assert cost_of_other_decision_model != cost_of_decision_model
        assert get_cost_of_call(decision_model) == cost_of_other_decision_model

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
        assert cost.input > Decimal(0)
        assert cost.output > Decimal(0)
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
