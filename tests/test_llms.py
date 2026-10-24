# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for LLMs.
"""

from __future__ import annotations

import asyncio
import gc
import json
import os
import re
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import openai
import pytest
from aiolimiter import AsyncLimiter
from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from contextgem.internal import utils as utils_module
from contextgem.internal.base import llms as llms_module
from contextgem.internal.base import models as models_module
from contextgem.internal.base.mixins import _redact_api_key_in_validation_error
from contextgem.internal.exceptions import (
    ContextGemWarning,
    ModelAPIError,
    ModelExtractionError,
)
from contextgem.internal.loggers import logger
from contextgem.internal.utils import (
    _get_model_api_retry_delay,
    _is_retryable_model_api_error,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    ChatSession,
    DateConcept,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    JsonObjectConcept,
    LabelConcept,
    ModelPricing,
    NumericalConcept,
    Paragraph,
    RatingConcept,
    StringConcept,
    register_tool,
)
from tests.base import BaseTest
from tests.conftest import VCR_REDACTION_MARKER
from tests.utils import REDACTED_VALUE, capture_logs, get_project_root_path


class TestLLMs(BaseTest):
    """
    Tests for the configuration of LLMs, local LLMs, LLMs of Anthropic, prompts,
    token limits, retries of API requests and rate limiting.
    """

    # LLMs of Anthropic: Claude Haiku as the extractor (no role) and Claude Sonnet
    # as the reasoner, as the LLMs of OpenAI in the other tests. The tests replay
    # their cassettes with a dummy key if the environment has no key of Anthropic.
    _anthropic_api_key = (
        os.getenv("CONTEXTGEM_ANTHROPIC_API_KEY") or VCR_REDACTION_MARKER
    )
    _llm_extractor_kwargs_anthropic: ClassVar[dict[str, Any]] = {
        "model": "anthropic/claude-haiku-5-5",
        "api_key": _anthropic_api_key,
        "auto_pricing": True,
        "auto_pricing_refresh": False,  # avoid network auto-refresh during tests
    }
    _llm_reasoner_kwargs_anthropic: ClassVar[dict[str, Any]] = {
        "model": "anthropic/claude-sonnet-5-5",
        "api_key": _anthropic_api_key,
        "role": "reasoner",
        "reasoning_effort": "low",
        "auto_pricing": True,
        "auto_pricing_refresh": False,  # avoid network auto-refresh during tests
    }

    def create_anthropic_model_group(self) -> DocumentModelGroup:
        """
        Creates a model group of the LLMs of Anthropic: Claude Haiku as the extractor
        and Claude Sonnet as the reasoner.

        :return: The model group, set up for replayed responses if the test replays
            its cassette.
        :rtype: DocumentModelGroup
        """
        model_group = DocumentModelGroup(
            models=[
                DocumentLLM(**self._llm_extractor_kwargs_anthropic),
                DocumentLLM(**self._llm_reasoner_kwargs_anthropic),
            ]
        )
        self.config_llm_async_limiter_for_mock_responses(model_group)
        return model_group

    @staticmethod
    def check_usage_and_cost_of_each_model(model_group: DocumentModelGroup) -> None:
        """
        Checks that each model of a group sent requests, and that their tokens
        are priced.

        :param model_group: The model group.
        :type model_group: DocumentModelGroup
        :return: None
        :rtype: None
        """
        usages = model_group.get_usage()
        assert len(usages) == len(model_group.models)
        for usage_item in usages:
            assert usage_item.usage.calls
            assert usage_item.usage.input > 0 and usage_item.usage.output > 0
        for cost_item in model_group.get_cost():
            assert cost_item.cost.input > 0 and cost_item.cost.output > 0

    @pytest.mark.vcr
    def test_anthropic_llms_text(self):
        """
        Tests for extracting aspects and concepts of every type from a text with
        a model group of LLMs of Anthropic, with references and justifications.
        """
        model_group = self.create_anthropic_model_group()

        document = self.document.clone()  # clone for method-scoped state modification
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Clauses on the liability of the parties",
                    add_justifications=True,
                    concepts=[
                        NumericalConcept(
                            name="Liability cap amount",
                            description="Fixed amount of the total liability cap, in USD",
                            numeric_type="int",
                            add_references=True,
                            reference_depth="sentences",
                        ),
                        NumericalConcept(
                            name="Liquidated damages",
                            description="Liquidated damages for each instance of "
                            "unauthorized disclosure, in USD",
                            numeric_type="int",
                            model_role="reasoner",
                            add_justifications=True,
                        ),
                    ],
                ),
            ]
        )
        document.add_concepts(
            [
                BooleanConcept(
                    name="Is a contract",
                    description="Document is a contract",
                    model_role="reasoner",
                ),
                DateConcept(
                    name="Effective date",
                    description="Date as of which the agreement is made and entered into",
                    add_references=True,
                ),
                StringConcept(
                    name="Governing law",
                    description="Law that governs the agreement",
                    add_references=True,
                    reference_depth="sentences",
                    add_justifications=True,
                ),
                LabelConcept(
                    name="Contract type",
                    description="Type of the contract",
                    labels=["NDA", "Supply agreement", "Employment agreement"],
                    classification_type="multi_class",
                ),
                RatingConcept(
                    name="Strength of confidentiality protection",
                    description="How well the agreement protects confidential information",
                    rating_scale=(1, 10),
                    model_role="reasoner",
                    add_justifications=True,
                ),
                JsonObjectConcept(
                    name="Party B",
                    description="Details of Party B",
                    structure={"name": str, "city": str, "contact_person": str},
                ),
                NumericalConcept(
                    name="Agreement term",
                    description="Term of the agreement, in years",
                    numeric_type="int",
                ),
            ]
        )

        model_group.extract_all(document)

        # Aspect and its concepts
        liability = document.get_aspect_by_name("Liability")
        assert liability.extracted_items
        assert all(i.justification for i in liability.extracted_items)
        liability_cap = liability.get_concept_by_name("Liability cap amount")
        self.log_extracted_items_for_instance(liability_cap)
        assert 100_000 in [i.value for i in liability_cap.extracted_items]
        assert all(i.reference_sentences for i in liability_cap.extracted_items)
        liquidated_damages = liability.get_concept_by_name("Liquidated damages")
        self.log_extracted_items_for_instance(liquidated_damages)
        assert 25_000 in [i.value for i in liquidated_damages.extracted_items]
        assert all(i.justification for i in liquidated_damages.extracted_items)

        # Document concepts
        for concept in document.concepts:
            self.log_extracted_items_for_instance(concept)
            assert concept.extracted_items, f"No items extracted for `{concept.name}`"
        assert [
            i.value
            for i in document.get_concept_by_name("Is a contract").extracted_items
        ] == [True]
        effective_date = document.get_concept_by_name("Effective date")
        assert date(2024, 7, 16) in [i.value for i in effective_date.extracted_items]
        assert all(i.reference_paragraphs for i in effective_date.extracted_items)
        governing_law = document.get_concept_by_name("Governing law")
        assert any("California" in i.value for i in governing_law.extracted_items)
        assert all(
            i.reference_sentences and i.justification
            for i in governing_law.extracted_items
        )
        contract_type = document.get_concept_by_name("Contract type")
        assert contract_type.extracted_items[0].value == ["NDA"]
        rating = document.get_concept_by_name("Strength of confidentiality protection")
        assert all(
            1 <= i.value <= 10 and i.justification for i in rating.extracted_items
        )
        party_b = document.get_concept_by_name("Party B")
        assert any("Example.Com" in i.value["name"] for i in party_b.extracted_items)
        agreement_term = document.get_concept_by_name("Agreement term")
        assert 5 in [i.value for i in agreement_term.extracted_items]

        # Both models sent requests, priced by auto-pricing
        self.check_usage_and_cost_of_each_model(model_group)
        self.check_usage(model_group)
        self.check_cost(model_group)

        # Check serialization of the model group
        self._check_deserialized_llm_config_eq(model_group)

    @pytest.mark.vcr
    def test_anthropic_llms_vision(self):
        """
        Tests for extracting concepts from an image, and for chat with an image,
        with LLMs of Anthropic.
        """
        model_group = self.create_anthropic_model_group()

        document = Document(
            images=[self.test_img_png_invoice],
            concepts=[
                StringConcept(
                    name="Invoice number",
                    description="Number of the invoice",
                    modality="image",
                    add_justifications=True,
                ),
                JsonObjectConcept(
                    name="Service items",
                    description="Service items, their quantity and price",
                    structure={"service_name": str, "quantity": int, "price": str},
                    modality="image",
                ),
                NumericalConcept(
                    name="Total invoice amount",
                    description="Total amount of the invoice",
                    numeric_type="int",
                    model_role="reasoner",
                    modality="image",
                    add_justifications=True,
                ),
            ],
        )

        model_group.extract_concepts_from_document(document)
        for concept in document.concepts:
            self.log_extracted_items_for_instance(concept)
        invoice_number = document.get_concept_by_name("Invoice number")
        assert any("1234" in i.value for i in invoice_number.extracted_items)
        assert all(i.justification for i in invoice_number.extracted_items)
        service_items = document.get_concept_by_name("Service items")
        assert {
            i.value["service_name"].lower() for i in service_items.extracted_items
        } == {"branding design", "web design", "brochure"}
        total_amount = document.get_concept_by_name("Total invoice amount")
        assert [i.value for i in total_amount.extracted_items] == [4800]

        # Both models sent requests, priced by auto-pricing
        self.check_usage_and_cost_of_each_model(model_group)

        # Chat with an image, in a chat session
        llm_reasoner = next(m for m in model_group.models if m.role == "reasoner")
        assert isinstance(llm_reasoner, DocumentLLM)
        chat_session = ChatSession()
        with pytest.warns(UserWarning, match="default system message"):
            answer = llm_reasoner.chat(
                "What's the type of this document? Reply in one word.",
                images=[self.test_img_png_invoice],
                chat_session=chat_session,
            )
        assert answer and "invoice" in answer.lower()
        with pytest.warns(UserWarning, match="default system message"):
            answer = llm_reasoner.chat(
                "What is the name of the company that issued it? "
                "Reply only with the name.",
                chat_session=chat_session,
            )
        assert answer and "platypus" in answer.lower()
        assert [m.role for m in chat_session.messages] == [
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
        ]

    @pytest.mark.vcr
    def test_anthropic_llms_tools(self):
        """
        Tests for chat with tools with LLMs of Anthropic, in a chat session. The LLM
        has to call the tools one after another, using the results of the previous
        tool calls in the next one, and then to answer a second question from
        the history of the session, which has the tool calls of the first turn.

        Claude Haiku thinks by default, and Claude Sonnet thinks as its reasoning
        effort is set, i.e. the thinking blocks of the LLM are returned with
        the tool calls.
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

        for llm_kwargs in (
            self._llm_extractor_kwargs_anthropic,
            self._llm_reasoner_kwargs_anthropic,
        ):
            tool_runs.clear()
            llm = DocumentLLM(
                **llm_kwargs,
                system_message="",
                tools=[get_order_total, get_discount_amount, compute_final_amount],
            )
            self.config_llm_async_limiter_for_mock_responses(llm)
            chat_session = ChatSession()

            answer = llm.chat(
                "Use the available tools to compute a final amount. "
                "Call `get_order_total` and `get_discount_amount` with "
                "order_id='ORD-123'. Then call `compute_final_amount` using the exact "
                "outputs returned by these tools as order_total and discount. "
                "Reply only with the number returned by `compute_final_amount`, "
                "nothing else. Do not guess any values; you must call the tools.",
                chat_session=chat_session,
            )
            logger.debug(f"Answer of {llm.model} with tools: {answer}")

            # Each tool is called once, the last one after the tools it depends on
            assert sorted(tool_runs[:2]) == ["discount", "total"]
            assert tool_runs[2:] == ["final"]
            # Expected: 9 - 4 = 5
            assert answer and "5" in answer

            # The LLM is called at least for the two tool rounds and for the final answer
            usage = llm.get_usage()[0].usage
            assert len(usage.calls) >= 3
            assert usage.input > 0 and usage.output > 0
            self.check_cost(llm)

            # The tool calls and their results are kept in the history of the session,
            # between the prompt and the answer, but are not listed in its messages
            history_roles = [m.role for m in chat_session._messages]
            assert history_roles[0] == "user" and history_roles[-1] == "assistant"
            assert "tool" in history_roles
            assert [m.role for m in chat_session.messages] == ["user", "assistant"]

            # Second turn, in the same session: the history sent with it has the tool
            # calls of the first turn, i.e. the LLM answers without calling the tools
            # again
            answer = llm.chat(
                "What was the final amount? Reply only with the number.",
                chat_session=chat_session,
            )
            assert answer and "5" in answer
            assert len(tool_runs) == 3
            assert [m.role for m in chat_session.messages] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]

            # Check serialization of LLM with registered tools
            self._check_deserialized_llm_config_eq(llm)

    @pytest.mark.vcr
    def test_anthropic_llms_thinking_blocks(self):
        """
        Tests for chat with tools with Claude Sonnet at a high reasoning effort, which
        thinks before it calls the tools: the thinking blocks that it returns with
        a tool call are sent back with the tool call in the next tool rounds, and with
        the history of the session in the next turn.
        """

        tool_runs: list[str] = []

        @register_tool
        def get_order_total(order_id: str) -> str:
            """
            Returns the total amount of the order.
            """
            tool_runs.append("total")
            return {"ORD-15": "9"}.get(order_id, "0")

        @register_tool
        def get_discount_amount(order_id: str) -> str:
            """
            Returns the discount amount for the order.
            """
            tool_runs.append("discount")
            return {"ORD-15": "4"}.get(order_id, "0")

        llm_kwargs: dict[str, Any] = {
            **self._llm_reasoner_kwargs_anthropic,
            "reasoning_effort": "high",
        }
        llm = DocumentLLM(
            **llm_kwargs,
            system_message="",
            tools=[get_order_total, get_discount_amount],
        )
        self.config_llm_async_limiter_for_mock_responses(llm)

        # The messages of the requests, as they are passed to litellm
        sent_messages: list[list[dict[str, Any]]] = []
        send_request = llms_module.litellm.acompletion

        async def record_and_send(*args: Any, **kwargs: Any) -> Any:
            """
            Records the messages of a request, and sends it.
            """
            sent_messages.append(list(kwargs["messages"]))
            return await send_request(*args, **kwargs)

        chat_session = ChatSession()
        with patch.object(llms_module.litellm, "acompletion", new=record_and_send):
            answer = llm.chat(
                "The order ID is ORD- followed by the number of prime numbers below "
                "50. Work out the order ID first. Then get the total and the discount "
                "of the order with the tools, one tool per step, and reply only with "
                "the total minus the discount.",
                chat_session=chat_session,
            )
            assert answer and "5" in answer
            assert sorted(tool_runs) == ["discount", "total"]

            # The thinking blocks returned with a tool call, with the text that
            # the model returns when a reasoning effort is set
            tool_call_messages = [
                m
                for m in chat_session._messages
                if m.role == "assistant" and m.tool_calls
            ]
            thinking_messages = [m for m in tool_call_messages if m.thinking_blocks]
            assert thinking_messages, "The model returned no thinking blocks"
            thinking_blocks = thinking_messages[0].thinking_blocks
            assert thinking_blocks is not None
            assert any(
                block.get("type") == "thinking" and block.get("thinking", "").strip()
                for block in thinking_blocks
            )

            def count_requests_with(blocks: list[dict[str, Any]]) -> int:
                """
                Returns the number of requests that sent the thinking blocks back.
                """
                return sum(
                    any(m.get("thinking_blocks") == blocks for m in messages)
                    for messages in sent_messages
                )

            # They are sent back with the tool call in the next tool rounds
            n_requests_in_turn = len(sent_messages)
            assert count_requests_with(thinking_blocks) >= 1

            # ...and in the next turn, with the history of the session
            answer = llm.chat(
                "What was the discount? Reply only with the number.",
                chat_session=chat_session,
            )
            assert answer and "4" in answer
            assert len(tool_runs) == 2  # no tool is called again
            assert len(sent_messages) == n_requests_in_turn + 1
            assert any(
                m.get("thinking_blocks") == thinking_blocks for m in sent_messages[-1]
            )

    @pytest.mark.vcr
    def test_anthropic_llms_fallback(self):
        """
        Tests for extraction with Claude Haiku whose requests are rejected by the API
        of Anthropic (invalid API key), with Claude Sonnet as its fallback LLM.
        """
        main_kwargs: dict[str, Any] = {
            **self._llm_extractor_kwargs_anthropic,
            "api_key": "invalid_api_key",
        }
        # A fallback LLM has the role of its main LLM
        fallback_kwargs = {
            k: v for k, v in self._llm_reasoner_kwargs_anthropic.items() if k != "role"
        }
        llm = DocumentLLM(
            **main_kwargs,
            fallback_models=[DocumentLLM(**fallback_kwargs, is_fallback=True)],
        )
        self.config_llm_async_limiter_for_mock_responses(llm)

        document = self.document.clone()  # clone for method-scoped state modification
        document.add_concepts(
            [StringConcept(name="Title", description="Title of the contract")]
        )
        llm.extract_concepts_from_document(document)

        title = document.get_concept_by_name("Title")
        self.log_extracted_items_for_instance(title)
        assert any(
            "non-disclosure agreement" in i.value.lower() for i in title.extracted_items
        )

        # The request of the main LLM failed, the fallback LLM answered
        main_usage, fallback_usage = llm.get_usage()
        assert not main_usage.is_fallback and fallback_usage.is_fallback
        assert main_usage.usage.calls and main_usage.usage.input == 0
        assert fallback_usage.usage.input > 0 and fallback_usage.usage.output > 0
        self.check_cost(llm)

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
        llm_reasoning_ollama.supports_reasoning = True
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
        llm_ollama_chat.supports_vision = True
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
        llm_ollama.supports_vision = True
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
        with pytest.warns(
            UserWarning, match="does not support tool calls according to `litellm"
        ):
            llm_lm_studio = DocumentLLM(
                model="lm_studio/openai/gpt-oss-20b",
                api_base="http://localhost:1234/v1",
                api_key="random-key",  # required for LM Studio API
                system_message="",
                tools=[get_order_total, get_discount_amount, compute_final_amount],
                seed=123,
            )
        llm_lm_studio.supports_tools = True

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

    def test_capability_params(self):
        """
        Tests the params that declare the capabilities of the model of an LLM: not set,
        a capability is detected with litellm, once for each model, also when `model`
        is changed, and a value that is set applies instead. The messages on what
        the LLM cannot do say where its capability comes from, a warning is issued
        when `reasoning_effort` is set but not sent, and the params are serialized.
        """
        model = "cerebras/llama-3.3-70b"  # detected with tool calls only
        params = [
            "supports_vision",
            "supports_reasoning",
            "supports_tools",
            "supports_parallel_tool_calls",
        ]
        detected = {
            "supports_vision": False,
            "supports_reasoning": False,
            "supports_tools": True,
            "supports_parallel_tool_calls": False,
        }

        def get_capabilities(llm: DocumentLLM) -> dict[str, bool]:
            """
            Returns the capabilities that apply to the LLM, by their params.
            """
            return {param: getattr(llm, f"_{param}") for param in params}

        # Not set: detected
        llm = DocumentLLM(model=model, api_key="test_api_key")
        assert [getattr(llm, param) for param in params] == [None] * 4
        assert get_capabilities(llm) == detected

        # Set: applies instead of the detection, which is not made for it
        with patch.object(
            llms_module.litellm, "supports_vision", side_effect=AssertionError
        ):
            llm = DocumentLLM(
                model=model,
                api_key="test_api_key",
                supports_vision=True,
                supports_tools=False,
            )
            assert get_capabilities(llm) == {
                **detected,
                "supports_vision": True,
                "supports_tools": False,
            }
        with pytest.raises(ValueError):
            DocumentLLM(
                model=model,
                api_key="test_api_key",
                supports_vision="yes",
            )

        # Detected once for a model, and again for a new model
        llm = DocumentLLM(model=model, api_key="test_api_key")
        detected_for: list[str] = []

        def supports_vision_stub(detected_model: str) -> bool:
            """
            Records the model that vision is detected for.
            """
            detected_for.append(detected_model)
            return detected_model == "openai/gpt-6-luna"

        with patch.object(
            llms_module.litellm, "supports_vision", new=supports_vision_stub
        ):
            assert not llm._supports_vision
            llm.temperature = 0.5
            assert not llm._supports_vision
            assert detected_for == []  # detected when the LLM was created
            llm.model = "openai/gpt-6-luna"
            assert llm._supports_vision
            assert llm._supports_vision
            assert detected_for == ["openai/gpt-6-luna"]
            # Assigning None detects the capability again, if needed
            llm.supports_vision = False
            assert not llm._supports_vision
            llm.supports_vision = None
            assert llm._supports_vision
            assert detected_for == ["openai/gpt-6-luna"]

        # The capabilities are params: the private attrs that held them before
        # cannot be assigned
        with pytest.raises(AttributeError):
            llm._supports_vision = True

        # The messages say where a missing capability comes from
        image_message = llm._build_user_message(
            message_text="Test", images=[self.test_img_png_invoice]
        )
        llm = DocumentLLM(model=model, api_key="test_api_key")
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Model `{model}` does not support vision according to "
                "`litellm.supports_vision()`. If the model supports vision, "
                "set `supports_vision=True` on the LLM."
            ),
        ):
            llm._ensure_vision_support_for_messages([image_message])
        llm.supports_vision = False
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Model `{model}` does not support vision as set with "
                "`supports_vision=False`."
            )
            + "$",
        ):
            llm._ensure_vision_support_for_messages([image_message])
        llm.supports_vision = True
        llm._ensure_vision_support_for_messages([image_message])

        # A warning on a param of the LLM is issued once when the LLM is created
        @register_tool
        def get_status() -> str:
            """
            Returns the status.
            """
            return "ok"

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            DocumentLLM(
                model="some_provider/some-model",
                api_key="test_api_key",
                tools=[get_status],
            )
        assert [str(w.message) for w in caught_warnings] == [
            (
                "Model `some_provider/some-model` is assigned tools, but does not "
                "support tool calls according to `litellm.supports_function_calling()`. "
                "The tools are sent with the requests of a chat, which the model or "
                "its provider may reject or answer without calling the tools. If the "
                "model supports tool calls, set `supports_tools=True` on the LLM."
            )
        ]

        # `reasoning_effort` that is not sent is warned about: for a model without
        # reasoning support...
        with pytest.warns(
            ContextGemWarning,
            match="`reasoning_effort` is set for model .*, but is not sent with its "
            "requests: the model does not support reasoning according to "
            r"`litellm.supports_reasoning\(\)`",
        ):
            llm = DocumentLLM(
                model=model, api_key="test_api_key", reasoning_effort="low"
            )
        assert "reasoning_effort" not in llm._build_request_config()
        # ... and for a model whose params litellm does not list it among
        with patch.object(
            llms_module.litellm,
            "get_supported_openai_params",
            return_value=["max_tokens"],
        ):
            with pytest.warns(ContextGemWarning, match="litellm does not list it"):
                llm = DocumentLLM(
                    model=model,
                    api_key="test_api_key",
                    reasoning_effort="low",
                    supports_reasoning=True,
                )
            assert "reasoning_effort" not in llm._build_request_config()
        # Sent, without a warning, where both apply
        with (
            patch.object(
                llms_module.litellm,
                "get_supported_openai_params",
                return_value=["max_tokens", "reasoning_effort"],
            ),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("error", category=ContextGemWarning)
            llm = DocumentLLM(
                model=model,
                api_key="test_api_key",
                reasoning_effort="low",
                supports_reasoning=True,
            )
            assert llm._build_request_config()["reasoning_effort"] == "low"

        # Serialization: the params, and not the private attrs
        llm = DocumentLLM(
            model=model,
            api_key="test_api_key",
            supports_vision=True,
            supports_parallel_tool_calls=False,
        )
        llm_dict = llm.to_dict()
        assert {param: llm_dict[param] for param in params} == {
            "supports_vision": True,
            "supports_reasoning": None,
            "supports_tools": None,
            "supports_parallel_tool_calls": False,
        }
        assert not any(f"_{param}" in llm_dict for param in params)
        self._check_deserialized_llm_config_eq(llm)

    def test_skipped_token_checks(self):
        """
        Tests the checks of the token limits of the requests that cannot be made:
        for a model that litellm does not know, the warning that the limits are not
        checked is logged once for the LLM and its model, not for each request,
        and for a model whose token encodings cannot be downloaded, e.g. without
        network access, the input tokens are not counted again in the process, as
        each count would try the download again.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        litellm = llms_module.litellm

        async def acompletion_stub(**kwargs: Any) -> Any:
            """
            Simulates the LLM API: answers each request with a text.
            """
            return litellm.mock_completion(**kwargs, mock_response="Answer.")

        def count_logged(logs: list[str], text: str) -> int:
            """
            Returns the number of the logged messages that have the text.
            """
            return sum(text in message for message in logs)

        # A model that litellm does not know: logged once for the LLM and its model
        unknown_model = "openai/model-unknown-to-litellm"
        other_unknown_model = "openai/other-model-unknown-to-litellm"
        llm = DocumentLLM(
            model=unknown_model,
            api_key="test_api_key",
            system_message="",
            max_tokens=100,
        )
        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            capture_logs() as logs,
        ):
            llm.chat("Hi")
            llm.chat("Hi again")
            for limit in ["input", "output"]:
                assert (
                    count_logged(
                        logs,
                        f"Could not validate max {limit} tokens for model "
                        f"`{unknown_model}`",
                    )
                    == 1
                )
            assert count_logged(logs, "`litellm.register_model()`") == 2
            assert count_logged(logs, "Not logged again for this LLM and model") == 2
            # ... and again for another model of the LLM
            llm.model = other_unknown_model
            llm.chat("Hi")
            llm.chat("Hi again")
            assert (
                count_logged(
                    logs,
                    f"Could not validate max input tokens for model "
                    f"`{other_unknown_model}`",
                )
                == 1
            )

        # Token encodings that cannot be downloaded: not counted again in the process
        counted_models: list[str] = []

        def token_counter_stub(model: str, **kwargs: Any) -> int:
            """
            Simulates a count of tokens whose encodings cannot be downloaded,
            e.g. behind a proxy that refuses the download, for one model, and
            a count that fails otherwise for the other one.
            """
            counted_models.append(model)
            if model == "openai/gpt-6-luna":
                # As the download error of requests, an OSError
                raise ConnectionError("Unable to connect to proxy")
            raise ValueError("Unexpected content")

        with (
            patch.object(litellm, "acompletion", new=acompletion_stub),
            patch.object(litellm, "token_counter", new=token_counter_stub),
            patch.object(llms_module, "_MODELS_WITHOUT_TOKEN_COUNT", new=set()),
            capture_logs() as logs,
        ):
            for _ in range(2):
                DocumentLLM(
                    model="openai/gpt-6-luna", api_key="test_api_key", system_message=""
                ).chat("Hi")
            assert counted_models == ["openai/gpt-6-luna"]
            assert (
                count_logged(
                    logs,
                    "Skipping input token validation for the requests of this model "
                    "until the process is restarted",
                )
                == 1
            )
            assert count_logged(logs, "https://contextgem.shcherbak.ai/testing/") == 1
            # Another failure of the count is not remembered
            other_llm = DocumentLLM(
                model="openai/gpt-5.6-luna", api_key="test_api_key", system_message=""
            )
            other_llm.chat("Hi")
            other_llm.chat("Hi again")
            assert counted_models == ["openai/gpt-6-luna"] + ["openai/gpt-5.6-luna"] * 2

    def test_missing_api_key_not_retried(self, monkeypatch):
        """
        Tests that a request without an API key, which the client of the OpenAI SDK
        rejects before sending it, and which litellm raises as a server error, fails
        at once, without the retries of a transient error.

        No request is sent: the client raises before it is built, and the API base
        is a local port where nothing listens, in case a key were found.
        """
        for name in ["OPENAI_API_KEY", "OPENAI_ADMIN_KEY"]:
            monkeypatch.delenv(name, raising=False)
        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_base="http://127.0.0.1:9/v1",
            system_message="",
        )
        with capture_logs() as logs, pytest.raises(ModelAPIError) as exc_info:
            llm.chat("Hi")
        assert "Missing credentials" in str(exc_info.value.original_error)
        assert not any("Retrying" in message for message in logs), logs

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
            DocumentLLM()  # ty: ignore[missing-argument]
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
                pricing_details=ModelPricing(  # ty: ignore[missing-argument]
                    **{  # ty: ignore[unknown-argument]  # invalid pricing details keys
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
                    # invalid pricing details values
                    input_per_1m_tokens=-0.00015,
                    output_per_1m_tokens=0.0006,
                ),
            )
        with pytest.raises(ValueError):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key="",  # empty api key
            )

        # Warnings
        with pytest.warns(
            UserWarning,
            match="`output_language` applies to the extraction prompts only",
        ):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="",
            )
        with pytest.warns(
            UserWarning,
            match="`output_language` applies to the extraction prompts only",
        ):
            DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="You are a helpful assistant.",
            )
        # ... while there is no such warning if `output_language` has its default
        # value, e.g. when the system message is disabled for chat, as suggested
        # by the warning on the default system message
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            llm_without_system_message = DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                system_message="",
            )
            llm_without_system_message.system_message = "Be concise."
        assert not any(
            "`output_language`" in str(caught_warning.message)
            for caught_warning in caught_warnings
        )

        # The warnings are attributed to the code that creates the LLM, assigns
        # a value to its param or deserializes it, i.e. not to the framework or
        # to Pydantic, which run the checks
        def check_warning_attributed_to_this_file(
            caught_warnings: pytest.WarningsRecorder,
        ) -> None:
            """
            Checks that the warning on the output language is attributed to this file.
            """
            (warning_on_output_language,) = [
                caught_warning
                for caught_warning in caught_warnings
                if "`output_language`" in str(caught_warning.message)
            ]
            assert os.path.normcase(
                warning_on_output_language.filename
            ) == os.path.normcase(__file__)

        with pytest.warns(UserWarning, match="`output_language`") as caught_warnings:
            llm_with_ignored_language = DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                output_language="adapt",
                system_message="",
            )
        check_warning_attributed_to_this_file(caught_warnings)
        with pytest.warns(UserWarning, match="`output_language`") as caught_warnings:
            llm_without_system_message.output_language = "adapt"
        check_warning_attributed_to_this_file(caught_warnings)
        with pytest.warns(UserWarning, match="`output_language`") as caught_warnings:
            DocumentLLM.from_dict(llm_with_ignored_language.to_dict())
        check_warning_attributed_to_this_file(caught_warnings)

    def test_update_default_prompt(self):
        """
        Tests for updating the default prompt for the LLM.
        """
        # Test custom prompts assignment
        llm_with_updated_prompts = DocumentLLM(
            **self._llm_extractor_kwargs_openai,
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
                prompt_type="aspect",
            )
        with pytest.raises(ValueError, match="no Jinja2 tags"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_concepts_without_tags_path,
                prompt_type="concept",
            )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_aspects_with_tags_path,
            prompt_type="aspect",
        )
        assert (
            "test custom prompt template for aspects"
            in llm_with_updated_prompts._extract_aspect_items_prompt.render()
        )
        llm_with_updated_prompts._update_default_prompt(
            prompt_path=prompt_concepts_with_tags_path,
            prompt_type="concept",
        )
        assert (
            "test custom prompt template for concepts"
            in llm_with_updated_prompts._extract_concept_items_prompt.render()
        )
        # A prompt type is the singular name of what the prompt extracts
        with pytest.raises(ValueError, match="Invalid prompt type: aspects"):
            llm_with_updated_prompts._update_default_prompt(
                prompt_path=prompt_aspects_with_tags_path,
                prompt_type="aspects",  # ty: ignore[invalid-argument-type]
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

    def test_fallback_request_params_checked_before_extraction(self):
        """
        Tests that the request params of the fallback LLMs are checked before any
        request of an extraction is sent, as before a chat: a fallback LLM with max
        output tokens above the limit of its model raises ``ValueError`` before
        the primary LLM is sent a request.

        LLM API calls are stubbed, i.e. no cassette is required.
        """
        fallback_llm = DocumentLLM(
            model="openai/gpt-5.6-luna",
            api_key="...",  # dummy key (no LLM calls are made)
            max_tokens=999999,
            is_fallback=True,
        )
        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key="...",
            fallback_models=[fallback_llm],
        )
        document = Document(raw_text="This agreement is governed by Norwegian law.")
        document.add_concepts(
            [StringConcept(name="Governing law", description="The governing law")]
        )
        requests = []

        async def acompletion_stub(**kwargs: Any) -> Any:
            """
            Records a request, which none of the calls below is to send.
            """
            requests.append(kwargs)
            raise AssertionError("No request is to be sent")

        with (
            patch.object(llms_module.litellm, "acompletion", new=acompletion_stub),
            pytest.raises(
                ValueError,
                match="exceeds the model's maximum output tokens .* `openai/gpt-5.6-luna`",
            ),
        ):
            llm.extract_concepts_from_document(document)
        assert requests == []
        assert not document.concepts[0].extracted_items

        # With max output tokens within the limit of the model, the check passes
        fallback_llm.max_tokens = 8000
        fallback_llm._check_request_params()

    def test_default_system_message_follows_output_language(self):
        """
        Tests that the default system message of an LLM is rendered again when its
        output language is changed after the LLM is created, also in a model group,
        that ``None`` assigned to the system message restores the default one, and
        that a custom or an empty system message is kept as it is.
        """
        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        # No warning that the output language takes no effect, as it does
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            llm.output_language = "adapt"
        assert not [
            caught_warning
            for caught_warning in caught_warnings
            if "`output_language`" in str(caught_warning.message)
        ]
        assert llm.system_message == self.default_system_message_non_en
        llm.output_language = "en"
        assert llm.system_message == self.default_system_message_en

        # None stands for the default system message, as when the LLM is created
        llm.output_language = "adapt"
        llm.system_message = ""
        assert llm.system_message == ""
        llm.system_message = None
        assert llm.system_message == self.default_system_message_non_en

        # A custom or an empty system message is kept
        for custom_system_message in ("Be concise.", ""):
            llm_with_custom_message = DocumentLLM(
                model="openai/gpt-6-luna",
                api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                system_message=custom_system_message,
            )
            with pytest.warns(ContextGemWarning, match="`output_language`"):
                llm_with_custom_message.output_language = "adapt"
            assert llm_with_custom_message.system_message == custom_system_message

        # A fallback LLM must have the same output language and system message as
        # the primary LLM: the output language of the fallback LLM is changed first.
        # An output language that is rejected leaves both params as they were.
        fallback_llm = DocumentLLM(
            model="openai/gpt-5.6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            is_fallback=True,
        )
        primary_llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            fallback_models=[fallback_llm],
        )
        with pytest.raises(ValueError, match="same output language"):
            primary_llm.output_language = "adapt"
        assert primary_llm.output_language == "en"
        assert primary_llm.system_message == self.default_system_message_en
        fallback_llm.output_language = "adapt"
        primary_llm.output_language = "adapt"
        assert (
            primary_llm.system_message
            == fallback_llm.system_message
            == self.default_system_message_non_en
        )

        # The LLMs of a model group
        model_group = DocumentModelGroup(
            models=[
                DocumentLLM(
                    model="openai/gpt-6-luna",
                    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                ),
                DocumentLLM(
                    model="openai/gpt-5.6-luna",
                    api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
                    role="reasoner",
                ),
            ]
        )
        model_group.group_update_output_language("adapt")
        assert all(
            model.system_message == self.default_system_message_non_en
            for model in model_group.models
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
            # A timeout or a connection error of Python, without a status code
            TimeoutError("Simulated timeout"),
            ConnectionError("Simulated connection error"),
            ConnectionResetError("Simulated connection reset"),
        ]:
            assert _is_retryable_model_api_error(error)
        for error in [
            create_api_error(litellm.BadRequestError),
            auth_error,
            create_api_error(litellm.NotFoundError),
            create_api_error(litellm.ContextWindowExceededError),
            create_api_error(litellm.ContentPolicyViolationError),
            # Another error without a status code, i.e. not one of litellm, e.g.
            # of a stand-in for the API in a test, would fail again in the same way
            RuntimeError("Error not classified by litellm"),
            ValueError("Simulated error of a stand-in"),
        ]:
            assert not _is_retryable_model_api_error(error)

        # A request that the OpenAI SDK rejects before sending it, e.g. for a missing
        # API key, which litellm raises as a server error, is not retried, unlike
        # a server error of an HTTP exchange
        def raise_from(error: Exception, cause: Exception) -> Exception:
            """
            Returns the error, raised from the cause, as litellm raises its errors.
            """
            try:
                raise error from cause
            except Exception as raised:  # noqa: BLE001 - returned for the checks
                return raised

        for cause, is_retryable in [
            # The client of the SDK raises its base class itself
            (
                openai.OpenAIError(
                    "Missing credentials. Please pass an `api_key`, or set "
                    "the `OPENAI_API_KEY` environment variable."
                ),
                False,
            ),
            # ... and its subclasses for the other errors, e.g. after a response
            (openai.ContentFilterFinishReasonError(), True),
            # An error that is not of the SDK
            (RuntimeError("Simulated error"), True),
        ]:
            for error in [
                # As for an OpenAI model and for an Azure OpenAI model
                create_api_error(litellm.InternalServerError),
                litellm.APIError(
                    status_code=500,
                    message="Simulated API error",
                    llm_provider="azure",
                    model="azure/my-deployment",
                ),
            ]:
                # litellm wraps the error of the SDK in an error of its own
                # first, as for the requests of OpenAI
                wrapped = raise_from(RuntimeError("OpenAIException"), cause)
                raised = raise_from(error, wrapped)
                assert _is_retryable_model_api_error(raised) is is_retryable

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
            with pytest.raises(ModelExtractionError, match=r"invalid data.*2 retries"):
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

            # Extraction: a warning is logged if `raise_exception_on_extraction_error`
            # is False, without sending the failed request again
            set_outcomes({primary_model: [rate_limit_error]})
            document = create_document()
            with patch.object(logger, "warning", wraps=logger.warning) as log_warning:
                create_llm().extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=False
                )
            assert any(
                re.search(r"LLM API.*3 retries", str(call.args[0]))
                for call in log_warning.call_args_list
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
            with pytest.raises(ModelExtractionError, match=r"invalid data.*3 retries"):
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

            # The error is logged, and not raised, if
            # `raise_exception_on_extraction_error` is False, without sending
            # the request again
            set_outcomes({primary_model: [(cut_off_answer, "length")]})
            document = create_document()
            with patch.object(logger, "error", wraps=logger.error) as log_error:
                create_llm().extract_concepts_from_document(
                    document, raise_exception_on_extraction_error=False
                )
            assert any(
                "cut off at the max output tokens" in str(call.args[0])
                for call in log_error.call_args_list
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

    def test_unsupported_request_params_failure_warned_of_once(self):
        """
        Tests that a request that still fails for its params when it is sent
        again with the unsupported params to be dropped is warned of as any
        other failed request is: only when the warning is issued instead of
        the exception, i.e. with ``raise_exception_on_extraction_error=False``
        and no fallback LLM left to try, and once. A warning was also issued
        when a fallback LLM took over and when the exception was raised, and
        two warnings when they were issued instead of the exception.

        LLM API calls are stubbed, i.e. no cassette is required.
        """

        litellm = llms_module.litellm
        primary_model = "openai/gpt-6-luna"
        fallback_model = "openai/gpt-5.6-luna"
        warning_start = "Exception occurred while calling LLM API with drop_params=True"
        requests: list[tuple[str, bool]] = []

        async def acompletion_stub(**kwargs: Any):
            """
            Simulates litellm, which raises the error for the params of
            a request to the primary LLM also when it is told to drop
            the unsupported ones. The fallback LLM extracts the concept.
            """
            requests.append((kwargs["model"], kwargs["drop_params"]))
            if kwargs["model"] == primary_model:
                raise litellm.UnsupportedParamsError(
                    message="Simulated error: a param is not supported",
                    model=primary_model,
                    llm_provider="openai",
                )
            content = json.dumps([{"concept_id": "C0", "extracted_items": ["Item"]}])
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(content=content)
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

        def extract(
            with_fallback: bool, raise_exception: bool
        ) -> tuple[Exception | None, list[Any], list[str]]:
            """
            Extracts a concept. Returns the error raised, if any, the items
            extracted, and the warnings issued for the request that failed with
            the unsupported params dropped.
            """
            llm = DocumentLLM(model=primary_model, api_key="test_api_key")
            if with_fallback:
                llm.fallback_models = [
                    DocumentLLM(
                        model=fallback_model, api_key="test_api_key", is_fallback=True
                    )
                ]
            document = Document(raw_text="Services agreement")
            document.concepts = [
                StringConcept(name="Title", description="Title of the document")
            ]
            requests.clear()
            error = None
            with patch.object(logger, "warning", wraps=logger.warning) as log_warning:
                try:
                    llm.extract_concepts_from_document(
                        document, raise_exception_on_extraction_error=raise_exception
                    )
                except ModelAPIError as e:
                    error = e
            return (
                error,
                [item.value for item in document.concepts[0].extracted_items],
                [
                    str(call.args[0])
                    for call in log_warning.call_args_list
                    if warning_start in str(call.args[0])
                ],
            )

        failed_requests = [(primary_model, False), (primary_model, True)]
        with patch.object(litellm, "acompletion", new=acompletion_stub):
            # A fallback LLM takes over: nothing is warned of
            for raise_exception in (True, False):
                error, values, warnings_issued = extract(
                    with_fallback=True, raise_exception=raise_exception
                )
                assert error is None
                assert values == ["Item"]
                assert requests == [*failed_requests, (fallback_model, False)]
                assert not warnings_issued

            # No fallback LLM: the exception is raised, without a warning
            error, values, warnings_issued = extract(
                with_fallback=False, raise_exception=True
            )
            assert isinstance(error, ModelAPIError)
            assert isinstance(error.original_error, litellm.UnsupportedParamsError)
            assert requests == failed_requests
            assert not values
            assert not warnings_issued

            # ... or one warning is logged instead of the exception
            error, values, warnings_issued = extract(
                with_fallback=False, raise_exception=False
            )
            assert error is None
            assert requests == failed_requests
            assert not values
            (warning,) = warnings_issued
            assert warning.startswith(f"{warning_start}: ")
            assert warning.endswith(" (0 retries).")

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
            requests to the fallback LLM succeed, with an item for each concept
            of the prompt.
            """
            if kwargs["model"] == primary_model:
                raise litellm.AuthenticationError(
                    message="Simulated API error",
                    llm_provider="openai",
                    model=primary_model,
                )
            concept_ids = re.findall(
                r"<concept_id>(C\d+)</concept_id>", kwargs["messages"][-1]["content"]
            )
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content=json.dumps(
                                [
                                    {
                                        "concept_id": concept_id,
                                        "extracted_items": ["Item"],
                                    }
                                    for concept_id in concept_ids
                                ]
                            )
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

    def test_api_key_not_shown(self, tmp_path: Path):
        """
        Tests that the API key of an LLM is not shown when the LLM is printed,
        on its own, as a fallback LLM or in a model group, while its API base is,
        and that no part of the key is shown in the text of a validation error
        of an LLM or of a model group, which does not show the inputs of
        the validation, nor given by the ``errors()`` and ``json()`` methods of
        the error: neither for an LLM that is among the inputs, nor for a key
        that is among them as it is passed, which is reported as redacted.

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
        # as the key is set for each of its LLMs. The error that rejects the key
        # reports the redacted value in its place.
        key_rejected_texts = ["api_key", "Extra inputs are not permitted"]
        key_rejected_details = {
            "type": "extra_forbidden",
            "loc": ("api_key",),
            "msg": "Extra inputs are not permitted",
            "input": REDACTED_VALUE,
        }

        def check_group_error(error: BaseException | None) -> None:
            """
            Checks that the validation error of a group that the key is passed
            to does not give the key, and reports nothing else as invalid.

            :param error: The validation error.
            :type error: BaseException | None
            """
            assert isinstance(error, ValidationError)
            self.check_validation_error_hides_inputs(error, api_key, key_rejected_texts)
            self.check_validation_error_omits_model_keys(error, api_key)
            assert error.title == "DocumentModelGroup"
            assert error.errors(include_url=False) == [key_rejected_details]

        invalid_group_params: list[dict[str, Any]] = [
            {"api_key": api_key, "models": [llm, other_llm]},
            {"models": [llm, other_llm], "api_key": api_key},
        ]
        for group_params in invalid_group_params:
            with pytest.raises(ValidationError) as e:
                DocumentModelGroup(**group_params)
            check_group_error(e.value)
        # ... also when the group is invalid otherwise. Without its LLMs, the input
        # is all the params that are passed, i.e. the key.
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup(api_key=api_key)  # ty: ignore[missing-argument, unknown-argument]
        self.check_validation_error_hides_inputs(
            e.value, api_key, ["models", "Field required", *key_rejected_texts]
        )
        self.check_validation_error_omits_model_keys(e.value, api_key)
        assert e.value.errors(include_url=False) == [
            {
                "type": "missing",
                "loc": ("models",),
                "msg": "Field required",
                "input": {"api_key": REDACTED_VALUE},
            },
            key_rejected_details,
        ]
        # With a single LLM, the LLM is reported as it is
        single_llm = [llm]
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup(models=single_llm, api_key=api_key)  # ty: ignore[unknown-argument]
        self.check_validation_error_hides_inputs(
            e.value, api_key, ["models", "at least 2 items", *key_rejected_texts]
        )
        self.check_validation_error_omits_model_keys(e.value, api_key)
        assert e.value.errors(include_url=False) == [
            {
                "type": "too_short",
                "loc": ("models",),
                "msg": "List should have at least 2 items after validation, not 1",
                "input": single_llm,
                "ctx": {"field_type": "List", "min_length": 2, "actual_length": 1},
            },
            key_rejected_details,
        ]
        assert e.value.errors()[0]["input"] is single_llm
        # ... and when the group is deserialized from the data that the key is
        # added to
        serialized_group = {**model_group.to_dict(), "api_key": api_key}
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup.from_dict(serialized_group)
        check_group_error(e.value)
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup.from_json(json.dumps(serialized_group))
        check_group_error(e.value)
        disk_path = tmp_path / "model_group.json"
        disk_path.write_text(json.dumps(serialized_group), encoding="utf-8")
        with pytest.raises(ValidationError) as e:
            DocumentModelGroup.from_disk(disk_path)
        assert "SECRET" not in str(e.value)
        check_group_error(e.value)
        # ... and when the key is assigned to the group
        key_param = "api_key"
        with pytest.raises(ValidationError) as e:
            setattr(model_group, key_param, api_key)
        self.check_validation_error_hides_inputs(
            e.value, api_key, ["api_key", "Object has no attribute"]
        )
        self.check_validation_error_omits_model_keys(e.value, api_key)
        assert e.value.errors(include_url=False) == [
            {
                "type": "no_such_attribute",
                "loc": ("api_key",),
                "msg": "Object has no attribute 'api_key'",
                "input": REDACTED_VALUE,
                "ctx": {"attribute": "api_key"},
            }
        ]
        assert not hasattr(model_group, key_param)

        # The LLM that is invalid itself, when it is created, changed or deserialized:
        # the key is among the inputs of its validation. When the LLM is created,
        # with the default system message or with one of its own, the input of
        # a check that involves several params is all the params that the LLM is
        # created with: the key is reported as redacted.
        for model_params in [
            {"model": "openai/gpt-6-luna"},
            {"model": "openai/gpt-6-luna", "system_message": "Be concise."},
        ]:
            self.check_model_validation_errors_hide_inputs(
                DocumentLLM,
                model_params,
                api_key,
                tmp_path=tmp_path,
            )

    def test_api_key_redacted_in_validation_error_inputs(self):
        """
        Tests that the API key is redacted in the inputs that a validation error
        of an LLM reports wherever the key is in them: in the values passed as
        dicts and lists, at any depth, which are not changed themselves, and that
        the error is rebuilt with nothing else changed, whatever errors it has.
        """
        api_key = "sk-test-SECRET-0123456789"

        # The key in the params that are passed as a dict, under a name that
        # does not exist. The input of the error for a required param that is not
        # passed is all the params: the key is redacted in the dict as well.
        pricing_params = {
            "input_per_1m_tokens": 1.0,
            "output_per_1m_tokens": 2.0,
            "apikey": api_key,
        }
        llm_params: dict[str, Any] = {
            "api_key": api_key,
            "pricing_details": pricing_params,
        }
        with pytest.raises(ValidationError) as e:
            DocumentLLM(**llm_params)
        self.check_validation_error_omits_model_keys(e.value, api_key)
        assert e.value.errors(include_url=False) == [
            {
                "type": "missing",
                "loc": ("model",),
                "msg": "Field required",
                "input": {
                    "api_key": REDACTED_VALUE,
                    "pricing_details": {**pricing_params, "apikey": REDACTED_VALUE},
                },
            },
            {
                "type": "extra_forbidden",
                "loc": ("pricing_details", "apikey"),
                "msg": "Extra inputs are not permitted",
                "input": REDACTED_VALUE,
            },
        ]
        # ... and in the dicts of a list
        tools: list[Any] = [
            {"type": "function", "api_key": api_key},
            {"type": "function"},
        ]
        with pytest.raises(ValidationError) as e:
            DocumentLLM(model="openai/gpt-6-luna", api_key=api_key, tools=tools)
        self.check_validation_error_omits_model_keys(e.value, api_key)
        (error_details,) = e.value.errors()
        assert error_details["loc"] == ("tools",)
        assert error_details["input"] == [
            {"type": "function", "api_key": REDACTED_VALUE},
            {"type": "function"},
        ]
        # What has no key is reported as it is passed, i.e. as the same object
        assert error_details["input"][1] is tools[1]
        # The values that are passed are not changed
        assert pricing_params["apikey"] == api_key
        assert tools[0] == {"type": "function", "api_key": api_key}

        # An error with errors of different kinds: of the types of Pydantic, with
        # and without a context, with an exception in the context, and of a custom
        # type. The key is in the params that are the input of the first error,
        # which also hold themselves.
        other_input = {"timeout": -1, "names": ["api_key"]}
        nested_params = {"api_key": api_key, "apikey": api_key, "n": 1}
        params: dict[str, Any] = {
            "api_key": api_key,
            "items": [1, (nested_params,)],
            "read_only": types.MappingProxyType({"api_key": api_key, "n": 2}),
            "other": other_input,
            "not_set": {"api_key": None},
        }
        params["itself"] = params
        value_error = ValueError("A value is invalid.")
        line_errors: list[Any] = [
            {"type": "missing", "loc": ("model",), "input": params},
            {"type": "extra_forbidden", "loc": ("items", 1, 0, "apikey"), "input": 1},
            {
                "type": "string_type",
                "loc": ("fallback_models", 0, "api_key"),
                "input": 2,
            },
            {
                "type": "greater_than_equal",
                "loc": ("other", "timeout"),
                "input": other_input,
                "ctx": {"ge": 0},
            },
            {
                "type": "value_error",
                "loc": (),
                "input": other_input,
                "ctx": {"error": value_error},
            },
            {
                "type": PydanticCustomError(
                    "custom_type", "A custom error of {what}", {"what": "a kind"}
                ),
                "loc": ("other",),
                "input": other_input,
            },
            {
                "type": PydanticCustomError("no_context_type", "No {context} here"),
                "loc": ("other", 0),
                "input": None,
            },
            # A custom error that is given a type of Pydantic
            {
                "type": PydanticCustomError("value_error", "A custom value error"),
                "loc": ("other", 1),
                "input": other_input,
            },
        ]
        for hide_input in [True, False]:
            error = ValidationError.from_exception_data(
                "Title", line_errors, hide_input=hide_input
            )
            assert api_key in repr(error.errors())
            redacted_error = _redact_api_key_in_validation_error(
                error, hide_input=hide_input
            )
            assert type(redacted_error) is ValidationError
            assert redacted_error is not error
            self.check_validation_error_omits_model_keys(redacted_error, api_key)
            # The error is the same, except for its inputs
            assert redacted_error.title == error.title == "Title"
            assert redacted_error.error_count() == error.error_count() == 8
            assert redacted_error.errors(include_input=False) == error.errors(
                include_input=False
            )
            assert redacted_error.json(include_input=False) == error.json(
                include_input=False
            )
            assert [
                (details["type"], details["msg"], details.get("ctx"))
                for details in redacted_error.errors()
            ] == [
                ("missing", "Field required", None),
                ("extra_forbidden", "Extra inputs are not permitted", None),
                ("string_type", "Input should be a valid string", None),
                (
                    "greater_than_equal",
                    "Input should be greater than or equal to 0",
                    {"ge": 0},
                ),
                (
                    "value_error",
                    "Value error, A value is invalid.",
                    {"error": value_error},
                ),
                ("custom_type", "A custom error of a kind", {"what": "a kind"}),
                ("no_context_type", "No {context} here", None),
                ("value_error", "A custom value error", None),
            ]
            # The text of the error shows the inputs, or does not, as the original
            assert ("input_value" in str(redacted_error)) is not hide_input
            if hide_input:
                assert str(redacted_error) == str(error)
            reported_inputs = [details["input"] for details in redacted_error.errors()]
            # The key is redacted at any depth, in a dict and in a mapping of
            # another type, under the name of its param and under the name that
            # is rejected as not existing. A key that is not set is not redacted,
            # and the params that hold themselves are not looked through again.
            assert reported_inputs[0] == {
                "api_key": REDACTED_VALUE,
                "items": [
                    1,
                    ({"api_key": REDACTED_VALUE, "apikey": REDACTED_VALUE, "n": 1},),
                ],
                "read_only": {"api_key": REDACTED_VALUE, "n": 2},
                "other": other_input,
                "not_set": {"api_key": None},
                "itself": REDACTED_VALUE,
            }
            # The inputs of the errors for these params are redacted, whatever
            # they are
            assert reported_inputs[1:3] == [REDACTED_VALUE, REDACTED_VALUE]
            # The other inputs are reported as they are, i.e. as the same objects
            assert reported_inputs[0]["other"] is other_input
            assert reported_inputs[0]["not_set"] is params["not_set"]
            assert all(
                reported_input is other_input
                for reported_input in [*reported_inputs[3:6], reported_inputs[7]]
            )
            assert reported_inputs[6] is None
            # The same error is given when it is redacted once more
            redacted_again_error = _redact_api_key_in_validation_error(
                redacted_error, hide_input=hide_input
            )
            assert redacted_again_error.errors() == redacted_error.errors()
        # The params are not changed
        assert params["api_key"] == api_key
        assert params["items"] == [1, (nested_params,)]
        assert nested_params == {"api_key": api_key, "apikey": api_key, "n": 1}
        assert params["read_only"]["api_key"] == api_key
        assert params["itself"] is params

    def test_built_in_prompt_templates_compiled_once(self):
        """
        Tests that the templates of the built-in prompts are compiled once and
        shared by the LLMs, instead of being compiled for each LLM that is created,
        and that a shared template carries nothing from one LLM, or from one
        rendering, to another: a custom prompt set for an LLM does not reach
        the other LLMs, and the LLMs created in several threads each render
        the prompt with the data of their own.

        No LLM API is called, i.e. no cassette is required.
        """
        llm_kwargs: dict[str, Any] = {
            "model": "openai/gpt-6-luna",
            "api_key": "test_api_key",
        }
        # The built-in templates are compiled when this LLM is created, at the latest
        llm = DocumentLLM(**llm_kwargs)
        default_prompts = (
            llm._extract_aspect_items_prompt.render(),
            llm._extract_concept_items_prompt.render(),
        )

        # No template is compiled for the LLMs created afterwards, nor when an LLM
        # is validated again as one of its params is changed
        with patch.object(
            utils_module,
            "_setup_jinja2_template",
            wraps=utils_module._setup_jinja2_template,
        ) as setup_template_mock:
            other_llms = [
                DocumentLLM(**llm_kwargs),
                DocumentLLM(**llm_kwargs, output_language="adapt"),
                DocumentLLM(
                    **llm_kwargs,
                    fallback_models=[
                        DocumentLLM(
                            model="openai/gpt-5.6-luna",
                            api_key="test_api_key",
                            is_fallback=True,
                        )
                    ],
                ),
            ]
            other_llms.append(DocumentLLM(**llm_kwargs, system_message="Be concise."))
            other_llms[0].temperature = 0.5
        assert setup_template_mock.call_count == 0
        for other_llm in [*other_llms, other_llms[2].fallback_models[0]]:
            assert isinstance(other_llm, DocumentLLM)
            assert (
                other_llm._extract_aspect_items_prompt
                is llm._extract_aspect_items_prompt
            )
            assert (
                other_llm._extract_concept_items_prompt
                is llm._extract_concept_items_prompt
            )

        # A custom prompt replaces the template of the LLM that it is set for,
        # and of no other LLM, whether created before or after
        custom_prompts_path = get_project_root_path() / "tests" / "custom_prompts"
        llm_with_custom_prompts = other_llms[0]
        llm_with_custom_prompts._update_default_prompt(
            prompt_path=custom_prompts_path / "custom_prompt_aspects_with_tags.j2",
            prompt_type="aspect",
        )
        llm_with_custom_prompts._update_default_prompt(
            prompt_path=custom_prompts_path / "custom_prompt_concepts_with_tags.j2",
            prompt_type="concept",
        )
        assert (
            "test custom prompt template for aspects"
            in llm_with_custom_prompts._extract_aspect_items_prompt.render()
        )
        assert (
            "test custom prompt template for concepts"
            in llm_with_custom_prompts._extract_concept_items_prompt.render()
        )
        for unaffected_llm in [llm, *other_llms[1:], DocumentLLM(**llm_kwargs)]:
            assert (
                unaffected_llm._extract_aspect_items_prompt.render(),
                unaffected_llm._extract_concept_items_prompt.render(),
            ) == default_prompts

        # The LLMs created in several threads render the shared template at
        # the same time, each with the data of its own only
        n_threads = 8

        def render_aspects_prompt(index: int) -> str:
            """
            Creates an LLM and renders its prompt for an aspect and a paragraph
            that are named after the index.

            :param index: The index that the aspect and the paragraph are named after.
            :type index: int
            :return: The rendered prompt.
            :rtype: str
            """
            thread_llm = DocumentLLM(**llm_kwargs)
            return thread_llm._extract_aspect_items_prompt.render(
                aspects=[
                    Aspect(
                        name=f"Topic of thread {index}",
                        description=f"Clauses on the topic of thread {index}",
                    )
                ],
                paragraphs=[Paragraph(raw_text=f"Text of thread {index}.")],
                add_justifications=False,
                justification_depth="brief",
                justification_max_sents=2,
                reference_depth="paragraphs",
                output_language="en",
                supports_reasoning=False,
            )

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            prompts = list(executor.map(render_aspects_prompt, range(n_threads)))
        for index, prompt in enumerate(prompts):
            self.check_rendered_prompt(prompt)
            for other_index in range(n_threads):
                is_own_data = other_index == index
                assert (f"Topic of thread {other_index}" in prompt) is is_own_data
                assert (f"Text of thread {other_index}." in prompt) is is_own_data
            # ... i.e. the same prompt as the one rendered on its own
            assert prompt == render_aspects_prompt(index)

    def test_refusal_read_from_litellm_responses(self):
        """
        Tests that a refusal is read from the responses as litellm builds them
        from the responses of the providers: OpenAI's ``refusal`` field of
        the message, which litellm keeps in the provider-specific fields of
        the message, and the ``content_filter`` finish reason, to which litellm
        maps e.g. the ``refusal`` stop reason of Anthropic. A response without
        either is not a refusal, also one cut off at the max output tokens.

        No LLM API calls are made, i.e. no cassette is required.
        """
        litellm = llms_module.litellm
        # Imported here, after litellm is imported by the framework, which
        # suppresses its warnings on import
        from litellm.litellm_core_utils.core_helpers import map_finish_reason
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_model_response_object,
        )

        def create_response(
            content: str | None, finish_reason: str, refusal: str | None = None
        ) -> Any:
            """
            Creates the response of litellm from a response of the OpenAI API.
            """
            return convert_to_model_response_object(
                response_object={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-6-luna",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": finish_reason,
                            "message": {
                                "role": "assistant",
                                "content": content,
                                "refusal": refusal,
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
                model_response_object=litellm.ModelResponse(),
                response_type="completion",
            )

        get_refusal = llms_module._DocumentLLM._get_refusal_of_completion
        assert map_finish_reason("refusal") == "content_filter"
        assert (
            get_refusal(create_response(None, "stop", refusal="I can't help."))
            == "refusal"
        )
        assert get_refusal(create_response("[]", "content_filter")) == "content_filter"
        assert (
            get_refusal(create_response("[]", map_finish_reason("refusal")))
            == "content_filter"
        )
        for response in [
            create_response("[]", "stop"),
            create_response("[]", "stop", refusal=""),
            create_response("[", "length"),
        ]:
            assert get_refusal(response) is None
