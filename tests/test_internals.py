# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for the internals of the framework.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from _pytest.nodes import Item as PytestItem
from aiolimiter import AsyncLimiter

from contextgem.internal.base import llms as llms_module
from contextgem.internal.base.aspects import _Aspect
from contextgem.internal.base.attrs import (
    _AssignedAspectsProcessor,
    _AssignedConceptsProcessor,
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.base.concepts import (
    _BooleanConcept,
    _Concept,
    _DateConcept,
    _JsonObjectConcept,
    _LabelConcept,
    _NumericalConcept,
    _RatingConcept,
    _StringConcept,
)
from contextgem.internal.base.data_models import _ModelPricing
from contextgem.internal.base.documents import _Document
from contextgem.internal.base.examples import _JsonObjectExample, _StringExample
from contextgem.internal.base.images import _Image
from contextgem.internal.base.llms import _DocumentLLM
from contextgem.internal.base.models import _DocumentModelGroup, _LoopAsyncLimiters
from contextgem.internal.base.paras_and_sents import _Paragraph, _Sentence
from contextgem.internal.base.pipelines import _ExtractionPipeline
from contextgem.internal.base.utils import _JsonObjectClassStruct
from contextgem.internal.registry import _publicize, _resolve_public_type
from contextgem.internal.utils import _run_async_calls, _run_sync
from contextgem.public import (
    Aspect,
    Document,
    DocumentLLM,
    DocumentModelGroup,
    ExtractionPipeline,
    JsonObjectConcept,
    JsonObjectExample,
    ModelPricing,
    Paragraph,
    Sentence,
    StringConcept,
    register_tool,
)
from tests.base import BaseTest


class TestInternals(BaseTest):
    """
    Tests for prompt templates, the registry of public types and base classes,
    and for the rejection of invalid names and assignments.
    """

    def test_prompt_templates(self):
        """
        Tests for content validity and consistency in the prompt templates.
        """
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[1]
        prompts_folder_path = project_root / "contextgem" / "internal" / "prompts"
        j2_files_found = False
        for root, _, files in os.walk(prompts_folder_path):
            for file in files:
                if file.endswith(".j2"):
                    j2_files_found = True
                    file_path = os.path.join(root, file)
                    with open(file_path, encoding="utf-8") as f:
                        raw_content = f.read()
                    # For testing, apply the same check as for the rendered prompt
                    self.check_rendered_prompt(raw_content)
        assert j2_files_found, prompts_folder_path

    def test_registry(self):
        """
        Sanity checks for internal -> public mappings via the registry.
        """

        pub_aspect = _publicize(
            _Aspect,
            name="Test Aspect",
            description="Test Description",
            model_role="reasoner",
            reference_depth="sentences",
        )
        assert isinstance(pub_aspect, Aspect)
        assert pub_aspect.model_role == "reasoner"
        assert pub_aspect.reference_depth == "sentences"
        assert _resolve_public_type(_Aspect) is Aspect

        pub_concept = _publicize(
            _StringConcept,
            name="Test Concept",
            description="Test Description",
            add_references=True,
        )
        assert isinstance(pub_concept, StringConcept)
        assert pub_concept.add_references
        assert _resolve_public_type(_StringConcept) is StringConcept

        pub_doc = _publicize(_Document, raw_text="Hello")
        assert isinstance(pub_doc, Document)
        assert _resolve_public_type(_Document) is Document
        pub_doc.add_aspects([pub_aspect])
        assert all(isinstance(a, Aspect) for a in pub_doc.aspects)
        pub_doc.add_concepts([pub_concept])
        assert all(isinstance(c, StringConcept) for c in pub_doc.concepts)

    def test_init_instance_bases(self):
        """
        Tests for direct initialization of the base classes (internal classes).
        """

        # Base instance classes
        base_instance_classes = [
            _Aspect,
            _StringConcept,
            _BooleanConcept,
            _DateConcept,
            _NumericalConcept,
            _RatingConcept,
            _JsonObjectConcept,
            _LabelConcept,
            _ModelPricing,
            _StringExample,
            _JsonObjectExample,
            _Image,
            _Document,
            _Paragraph,
            _Sentence,
            _ExtractionPipeline,
            _DocumentLLM,
            _DocumentModelGroup,
            _JsonObjectClassStruct,
        ]
        for base_class in base_instance_classes:
            with pytest.raises(
                TypeError, match="internal and cannot be instantiated directly"
            ):
                base_class()  # ty: ignore[missing-argument]

        # Shared attribute classes
        base_attr_classes = [
            _AssignedAspectsProcessor,
            _AssignedConceptsProcessor,
            _AssignedInstancesProcessor,
            _ExtractedItemsAttributeProcessor,
            _RefParasAndSentsAttrituteProcessor,
        ]
        for base_class in base_attr_classes:

            class TestNoRequiredAttrs(base_class):  # ty: ignore[unsupported-base]
                value: str = "Test"  # no required attributes

                @property
                def _item_class(self) -> type:
                    return PytestItem

            with pytest.raises(AttributeError):
                TestNoRequiredAttrs()  # initialized with no required attributes

    def test_public_exceptions(self):
        """
        Tests that the exceptions raised to the calling code can be imported from
        the package, and are the classes that the framework raises.
        """

        import contextgem
        from contextgem import public
        from contextgem.internal import exceptions

        for name in [
            "ContextGemError",
            "ModelAPIError",
            "ModelExtractionError",
            "ModelRefusalError",
            "LLMToolLoopLimitError",
        ]:
            assert name in contextgem.__all__
            assert name in public.__all__
            # The same class, so that an `except` clause with the public name
            # catches what the framework raises
            assert getattr(contextgem, name) is getattr(exceptions, name)
            assert getattr(public, name) is getattr(exceptions, name)
            # ... and an `except ContextGemError` clause catches each of them
            assert issubclass(getattr(contextgem, name), contextgem.ContextGemError)
        assert contextgem.ContextGemError.__bases__ == (Exception,)

        # The exception that a tool raises for the LLM is the class that the framework
        # catches, and not an error of the framework, i.e. not a ContextGemError
        assert "ToolCallError" in contextgem.__all__
        assert "ToolCallError" in public.__all__
        assert contextgem.ToolCallError is exceptions.ToolCallError
        assert public.ToolCallError is exceptions.ToolCallError
        assert contextgem.ToolCallError.__bases__ == (Exception,)

        # Internal exceptions are not exported
        for name in ["_ToolFunctionError", "_InvalidDecisionResponseError"]:
            assert not hasattr(contextgem, name)

    def test_pre_v1_names_rejected(self):
        """
        Tests that the names of the classes and parameters that were renamed in v1.0
        are rejected, i.e. not silently ignored, both in the code and in the
        serialized data.
        """

        import contextgem
        from contextgem.internal import exceptions

        # Renamed classes are not available under the old names
        for module, old_name in [
            (contextgem, "DocumentLLMGroup"),
            (contextgem, "LLMPricing"),
            (exceptions, "LLMAPIError"),
            (exceptions, "LLMExtractionError"),
            (contextgem, "LLMAPIError"),
            (contextgem, "LLMExtractionError"),
        ]:
            assert not hasattr(module, old_name)

        # Renamed params
        with pytest.raises(ValueError, match="llm_role\n"):
            Aspect(
                name="Liability",
                description="Liability clauses",
                llm_role="reasoner",  # ty: ignore[unknown-argument]
            )
        with pytest.raises(ValueError, match="llm_role\n"):
            StringConcept(
                name="Title",
                description="Contract title",
                llm_role="reasoner",  # ty: ignore[unknown-argument]
            )
        with pytest.raises(ValueError, match="llms\n"):
            DocumentModelGroup(  # ty: ignore[missing-argument]
                llms=[self.llm_extractor, self.llm_reasoner],  # ty: ignore[unknown-argument]
            )
        model_group = DocumentModelGroup(models=[self.llm_extractor, self.llm_reasoner])
        for method in [
            model_group.get_usage,
            model_group.get_cost,
            model_group.reset_usage_and_cost,
        ]:
            with pytest.raises(TypeError, match="llm_role"):
                method(llm_role=self.llm_reasoner.role)  # ty: ignore[unknown-argument]

        # Renamed keys and class names in the serialized data
        concept = StringConcept(
            name="Title", description="Contract title", model_role="reasoner"
        )
        concept_dict = concept.to_dict()
        concept_dict["llm_role"] = concept_dict.pop("model_role")
        with pytest.raises(ValueError, match="llm_role\n"):
            StringConcept.from_dict(concept_dict)
        model_group_dict = model_group.to_dict()
        model_group_dict["llms"] = model_group_dict.pop("models")
        with pytest.raises(ValueError, match="llms\n"):
            DocumentModelGroup.from_dict(model_group_dict)
        model_group_dict = model_group.to_dict()
        model_group_dict["__class__"] = "DocumentLLMGroup"
        with pytest.raises(TypeError, match="DocumentLLMGroup"):
            DocumentModelGroup.from_dict(model_group_dict)
        pricing_dict = ModelPricing(
            input_per_1m_tokens=0.1, output_per_1m_tokens=0.5
        ).to_dict()
        pricing_dict["__class__"] = "LLMPricing"
        with pytest.raises(TypeError, match="LLMPricing"):
            ModelPricing.from_dict(pricing_dict)

    def test_rejected_assignment_keeps_instance_unchanged(self):
        """
        Tests that an instance is left unchanged when a value assigned to its param
        is rejected by the validation that involves the other params of the instance,
        i.e. the rejected value is not kept.

        No LLM API calls are made, i.e. no cassette is required.
        """

        def check_rejected_change(
            instance: Any, change: Callable[[], Any], error_match: str
        ) -> None:
            """
            Checks that the change of the instance is rejected, and that the params
            of the instance are the same as before.
            """
            params_before = {
                name: getattr(instance, name) for name in type(instance).model_fields
            }
            params_set_before = set(instance.model_fields_set)
            with pytest.raises(ValueError, match=error_match):
                change()
            for name, value_before in params_before.items():
                assert getattr(instance, name) is value_before
            assert instance.model_fields_set == params_set_before

        def check_rejected(
            instance: Any, param: str, value: Any, error_match: str
        ) -> None:
            """
            Checks that the value is rejected for the param of the instance,
            and that the params of the instance are the same as before.
            """
            check_rejected_change(
                instance, partial(setattr, instance, param, value), error_match
            )

        def create_llm(model: str = "openai/gpt-6-luna", **kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with the given params.
            """
            return DocumentLLM(model=model, api_key="test_api_key", **kwargs)

        def create_llm_with_fallback(**kwargs: Any) -> DocumentLLM:
            """
            Creates an LLM with a fallback LLM, both with the given params.
            """
            return create_llm(
                fallback_models=[
                    create_llm("openai/gpt-6.1-sol", is_fallback=True, **kwargs)
                ],
                **kwargs,
            )

        @register_tool
        def rollback_test_tool_1(order_id: str) -> str:
            """
            Returns the status of the order.
            """
            return f"Status of {order_id}"

        @register_tool
        def rollback_test_tool_2(order_id: str) -> str:
            """
            Returns the total of the order.
            """
            return f"Total of {order_id}"

        # === LLM ===
        check_rejected(
            create_llm_with_fallback(role="extractor"),
            "role",
            "reasoner",
            "must have the same role",
        )
        check_rejected(
            create_llm_with_fallback(),
            "output_language",
            "adapt",
            "must have the same output language",
        )
        check_rejected(
            create_llm_with_fallback(),
            "is_fallback",
            True,
            "cannot have its own fallback LLMs",
        )
        check_rejected(
            create_llm(
                pricing_details=ModelPricing(
                    input_per_1m_tokens=0.1, output_per_1m_tokens=0.5
                )
            ),
            "auto_pricing",
            True,
            "cannot be set together with `auto_pricing=True`",
        )
        llm = create_llm_with_fallback()
        fallback_llm = llm.fallback_models[0]
        check_rejected(
            llm,
            "fallback_models",
            [fallback_llm, create_llm(is_fallback=True)],
            "same config params as the main LLM",
        )
        assert llm.fallback_models == [fallback_llm]
        # The rejected fallback LLMs are not taken for the validated ones,
        # i.e. are rejected again
        check_rejected(
            llm,
            "fallback_models",
            [fallback_llm, create_llm(is_fallback=True)],
            "same config params as the main LLM",
        )
        # The LLM remains valid, i.e. its other params can be assigned
        llm.temperature = 0.5
        llm.fallback_models = []
        assert llm.fallback_models == []

        # The state derived from the params is not changed either
        llm_with_tools = create_llm_with_fallback(tools=[rollback_test_tool_1])
        other_tools = create_llm(tools=[rollback_test_tool_2]).tools
        assert list(llm_with_tools._tool_registry) == ["rollback_test_tool_1"]
        check_rejected(
            llm_with_tools,
            "tools",
            other_tools,
            "must have the same `tools` configuration",
        )
        assert list(llm_with_tools._tool_registry) == ["rollback_test_tool_1"]

        # === Model group ===
        check_rejected(
            DocumentModelGroup(
                models=[create_llm(), create_llm("openai/gpt-6.1-sol", role="reasoner")]
            ),
            "output_language",
            "adapt",
            "must have the same value of `output_language`",
        )

        # === Concept ===
        json_object_concept = JsonObjectConcept(
            name="Party",
            description="Party details",
            structure={"name": str},
            examples=[JsonObjectExample(content={"name": "Acme Inc."})],
        )
        check_rejected(
            json_object_concept,
            "examples",
            [JsonObjectExample(content={"name": 1})],
            "Invalid JSON object structure",
        )
        check_rejected(
            json_object_concept,
            "structure",
            {"name": int},
            "Invalid JSON object structure",
        )
        # The concept validates the extracted values against the structure it has
        assert json_object_concept._process_item_value({"name": "Other Inc."}) == {
            "name": "Other Inc."
        }

        # === Paragraph ===
        check_rejected(
            Paragraph(raw_text="The liability of the parties is capped."),
            "sentences",
            [Sentence(raw_text="This sentence is not in the paragraph.")],
            "Not all sentences were matched",
        )

        # === Aspect ===
        sub_aspect = Aspect(name="Liability caps", description="Liability caps")
        aspect = Aspect(
            name="Liability", description="Liability clauses", aspects=[sub_aspect]
        )
        check_rejected(
            aspect,
            "aspects",
            [sub_aspect, Aspect(name="Liability", description="Liability clauses")],
            "cannot have the same name as parent aspect",
        )
        assert [i.name for i in aspect.aspects] == ["Liability caps"]

        # === Added aspects and concepts ===
        # They are assigned in one list with the aspects or concepts that
        # the instance has
        aspect = Aspect(name="Liability", description="Liability clauses")
        for instance in [
            Document(raw_text="The liability of the parties is capped."),
            ExtractionPipeline(),
            aspect,
        ]:
            instance.add_aspects(
                [Aspect(name="Liability caps", description="Liability caps")]
            )
            instance.add_concepts(
                [StringConcept(name="Cap", description="Amount of the cap")]
            )
            repeated_aspects: list[_Aspect] = [
                Aspect(name="Indemnity", description="Indemnity clauses"),
                Aspect(name="liability caps", description="Other caps"),
            ]
            check_rejected_change(
                instance,
                partial(instance.add_aspects, repeated_aspects),
                "unique names",
            )
            repeated_concepts: list[_Concept] = [
                StringConcept(name="cap", description="Other amount")
            ]
            check_rejected_change(
                instance,
                partial(instance.add_concepts, repeated_concepts),
                "unique names",
            )
            assert [i.name for i in instance.aspects] == ["Liability caps"]
            assert [i.name for i in instance.concepts] == ["Cap"]
        # Rejected after the list is assigned, by the validation that involves
        # the other params of the aspect
        sub_aspects_named_as_parent: list[_Aspect] = [
            Aspect(name="Liability", description="Other liability clauses")
        ]
        check_rejected_change(
            aspect,
            partial(aspect.add_aspects, sub_aspects_named_as_parent),
            "cannot have the same name as parent aspect",
        )
        assert [i.name for i in aspect.aspects] == ["Liability caps"]

    def test_loop_async_limiters(self):
        """
        Tests the keeping of the limiters that pace the requests to a model: one
        limiter for each event loop, at the rate requested, for as long as
        the event loop is open.
        """

        class Loop:
            """
            Stands in for an event loop.
            """

            def __init__(self) -> None:
                self.closed = False

            def is_closed(self) -> bool:
                """
                Tells whether the event loop is closed.
                """
                return self.closed

        limiters = _LoopAsyncLimiters()
        loop, other_loop = Loop(), Loop()

        def get(
            running_loop: Loop, max_rate: float = 10, time_period: float = 5
        ) -> AsyncLimiter:
            """
            Returns the limiter for the given rate, with the given event loop
            as the running one.
            """
            with patch.object(asyncio, "get_running_loop", return_value=running_loop):
                return limiters.get(max_rate=max_rate, time_period=time_period)

        # No event loop is running
        with pytest.raises(RuntimeError):
            limiters.get(max_rate=10, time_period=5)

        limiter = get(loop)
        assert type(limiter) is AsyncLimiter
        assert (limiter.max_rate, limiter.time_period) == (10, 5)
        # The same limiter in the same event loop
        assert get(loop) is limiter
        # A limiter of its own in another event loop
        other_limiter = get(other_loop)
        assert other_limiter is not limiter
        assert (other_limiter.max_rate, other_limiter.time_period) == (10, 5)
        assert get(other_loop) is other_limiter
        assert get(loop) is limiter

        # Another rate: a new limiter, in this event loop only
        for max_rate, time_period in [(20, 5), (20, 1)]:
            new_limiter = get(loop, max_rate, time_period)
            assert new_limiter is not limiter
            assert (new_limiter.max_rate, new_limiter.time_period) == (
                max_rate,
                time_period,
            )
            assert get(loop, max_rate, time_period) is new_limiter
            limiter = new_limiter
        assert get(other_loop) is other_limiter

        # The limiter of an event loop that is closed is not kept
        assert list(limiters._limiters) == [loop, other_loop]
        loop.closed = True
        assert get(other_loop) is other_limiter
        assert list(limiters._limiters) == [other_loop]

        # A copy has no limiters, and gets its own
        limiters_copy = deepcopy(limiters)
        assert type(limiters_copy) is _LoopAsyncLimiters
        assert limiters_copy._limiters == {}
        assert list(limiters._limiters) == [other_loop]
        with patch.object(asyncio, "get_running_loop", return_value=other_loop):
            assert limiters_copy.get(max_rate=10, time_period=5) is not other_limiter

        # The limiter of a model has the rate of the limiter set for the model,
        # which is not used itself, and follows the rate when it is changed
        llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        other_llm = DocumentLLM(model="openai/gpt-6-luna", api_key="test_api_key")
        with patch.object(asyncio, "get_running_loop", return_value=other_loop):
            limiter = llm._get_async_limiter_for_running_loop()
            assert (limiter.max_rate, limiter.time_period) == (10, 10)
            assert limiter is not llm.async_limiter
            assert llm._get_async_limiter_for_running_loop() is limiter
            # Each model has its own limiters
            assert other_llm._get_async_limiter_for_running_loop() is not limiter
            llm.async_limiter = AsyncLimiter(3, 7)
            new_limiter = llm._get_async_limiter_for_running_loop()
            assert new_limiter is not limiter
            assert (new_limiter.max_rate, new_limiter.time_period) == (3, 7)

    def test_run_async_calls(self):
        """
        Tests the running of asynchronous calls one after another and concurrently:
        the order of the results, and what is left running when a call fails or
        the caller is cancelled, which is nothing.
        """
        started: list[str] = []
        cancelled: list[str] = []
        finished: list[str] = []

        class Interruption(BaseException):
            """
            Error that is not an ``Exception``, as an interruption is.
            """

        async def run_call(
            name: str,
            wait_for: asyncio.Event | None = None,
            error: BaseException | None = None,
            steps: int = 0,
        ) -> str:
            """
            Runs a call that takes the given number of steps of the event loop,
            or waits for the event, and then fails with the error, if given.
            """
            started.append(name)
            try:
                for _ in range(steps):
                    await asyncio.sleep(0)
                if wait_for is not None:
                    await wait_for.wait()
            except asyncio.CancelledError:
                cancelled.append(name)
                raise
            if error is not None:
                raise error
            finished.append(name)
            return name

        def get_running_tasks() -> list[asyncio.Task[Any]]:
            """
            Returns the tasks of the current event loop that are not done,
            other than the current one.
            """
            current_task = asyncio.current_task()
            return [
                task
                for task in asyncio.all_tasks()
                if task is not current_task and not task.done()
            ]

        async def check() -> None:
            """
            Runs the checks in one event loop.
            """
            # === The results are in the order of the calls ===
            for use_concurrency in [False, True]:
                started.clear()
                finished.clear()
                results = await _run_async_calls(
                    cals_and_kwargs=[
                        (run_call, {"name": "slow", "steps": 5}),
                        (run_call, {"name": "fast"}),
                    ],
                    use_concurrency=use_concurrency,
                )
                assert results == ["slow", "fast"]
                assert started == ["slow", "fast"]
                assert finished == (
                    ["fast", "slow"] if use_concurrency else ["slow", "fast"]
                )
            assert await _run_async_calls([], use_concurrency=True) == []

            # === A call fails ===
            # One after another: the calls after the failed one are not started
            started.clear()
            finished.clear()
            error = RuntimeError("Simulated error")
            with pytest.raises(RuntimeError) as exc_info:
                await _run_async_calls(
                    cals_and_kwargs=[
                        (run_call, {"name": "first"}),
                        (run_call, {"name": "failed", "error": error}),
                        (run_call, {"name": "not started"}),
                    ],
                    use_concurrency=False,
                )
            assert exc_info.value is error
            assert started == ["first", "failed"]

            # Concurrently: the calls still running are cancelled and waited
            # for before the error is raised, i.e. nothing is left running,
            # whatever the error is
            other_error = ValueError("Another simulated error")
            first_error: BaseException
            for first_error in [error, Interruption("Simulated interruption")]:
                started.clear()
                cancelled.clear()
                finished.clear()
                never_set = asyncio.Event()
                with pytest.raises(type(first_error)) as concurrent_exc_info:
                    await _run_async_calls(
                        cals_and_kwargs=[
                            (run_call, {"name": "waiting", "wait_for": never_set}),
                            (run_call, {"name": "done"}),
                            (
                                run_call,
                                {"name": "failed", "error": first_error, "steps": 3},
                            ),
                            (run_call, {"name": "also waiting", "wait_for": never_set}),
                            # A call that fails later is cancelled as well:
                            # the first error is the one raised
                            (
                                run_call,
                                {
                                    "name": "failed later",
                                    "error": other_error,
                                    "steps": 9,
                                },
                            ),
                        ],
                        use_concurrency=True,
                    )
                assert concurrent_exc_info.value is first_error
                assert not get_running_tasks()
                assert finished == ["done"]
                assert sorted(cancelled) == ["also waiting", "failed later", "waiting"]
                # ... and nothing happens afterwards
                never_set.set()
                for _ in range(20):
                    await asyncio.sleep(0)
                assert finished == ["done"]

            # === The caller is cancelled ===
            started.clear()
            cancelled.clear()
            finished.clear()
            never_set = asyncio.Event()
            caller = asyncio.ensure_future(
                _run_async_calls(
                    cals_and_kwargs=[
                        (run_call, {"name": "waiting", "wait_for": never_set}),
                        (run_call, {"name": "done"}),
                        (run_call, {"name": "also waiting", "wait_for": never_set}),
                    ],
                    use_concurrency=True,
                )
            )
            for _ in range(5):
                await asyncio.sleep(0)
            assert started == ["waiting", "done", "also waiting"]
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            assert not get_running_tasks()
            assert finished == ["done"]
            assert sorted(cancelled) == ["also waiting", "waiting"]

        asyncio.run(check())

    def test_run_sync_closes_litellm_clients(self):
        """
        Tests that a sync method closes the HTTP clients that litellm created for
        the event loop of the method before the event loop is closed, whether its
        coroutine returns or fails, and whether or not an event loop runs in
        the calling thread, and leaves open the clients that litellm was given and
        those of other event loops.
        """
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        cache = llms_module.litellm.in_memory_llm_clients_cache

        class Client:
            """
            Stands in for an HTTP client, whose close fails if asked.
            """

            def __init__(self, fails: bool = False) -> None:
                self.fails = fails
                self.closed = False

            def _close(self) -> None:
                """
                Closes the client, or fails.
                """
                if self.fails:
                    raise RuntimeError("Simulated close error")
                self.closed = True

        class AsyncClient(Client):
            """
            Stands in for an HTTP client whose close is a coroutine function.
            """

            async def aclose(self) -> None:
                """
                Closes the client, or fails.
                """
                self._close()

        class SyncClient(Client):
            """
            Stands in for an HTTP client whose close is a function.
            """

            def close(self) -> None:
                """
                Closes the client, or fails.
                """
                self._close()

        def remove_from_cache(key: str) -> None:
            """
            Removes a client from the cache of litellm, as litellm removes an expired
            client.
            """
            cache.cache_dict.pop(key, None)
            cache.ttl_dict.pop(key, None)

        # A client of no event loop, which litellm created
        client_without_loop = AsyncClient()
        cache.set_cache(
            "test_client_without_loop", client_without_loop, litellm_owned_client=True
        )
        assert "test_client_without_loop" in cache.cache_dict

        # The clients of the event loops of the sync calls, by their keys
        created_clients: dict[str, Any] = {}
        given_clients: dict[str, AsyncClient] = {}

        async def use_clients(error: Exception | None = None) -> str:
            """
            Creates HTTP clients of the running event loop as litellm does, and then
            fails with the error, if given.
            """
            created_clients.clear()
            given_clients.clear()
            handler = get_async_httpx_client(llm_provider="anthropic")
            for key, client in cache.cache_dict.items():
                if client is handler:
                    created_clients[key] = handler
            for name, client in [
                ("test_async_client", AsyncClient()),
                ("test_sync_client", SyncClient()),
                ("test_failing_client", AsyncClient(fails=True)),
            ]:
                cache.set_cache(name, client, litellm_owned_client=True)
                created_clients[cache.update_cache_key_with_event_loop(name)] = client
            given_client = AsyncClient()
            cache.set_cache("test_given_client", given_client)
            given_clients[
                cache.update_cache_key_with_event_loop("test_given_client")
            ] = given_client
            assert len(created_clients) == 4
            assert all(key in cache.cache_dict for key in created_clients)
            if error is not None:
                raise error
            return "Done"

        async def run_from_event_loop() -> str:
            """
            Runs the sync call from a thread in which an event loop runs.
            """
            return _run_sync(use_clients())

        error = ValueError("Simulated error")
        for run_call, raised_error in [
            (lambda: _run_sync(use_clients()), None),
            (lambda: _run_sync(use_clients(error)), error),
            (lambda: asyncio.run(run_from_event_loop()), None),
        ]:
            if raised_error is None:
                assert run_call() == "Done"
            else:
                with pytest.raises(ValueError) as exc_info:
                    run_call()
                assert exc_info.value is raised_error
            try:
                # The clients that litellm created are closed and removed from
                # its cache, also when one of them fails to close
                for key, client in created_clients.items():
                    assert key not in cache.cache_dict
                    if isinstance(client, Client):
                        assert client.closed is not client.fails
                    else:
                        assert client._client.is_closed
                # The client that litellm was given is left open
                for key, client in given_clients.items():
                    assert cache.cache_dict.get(key) is client
                    assert not client.closed
            finally:
                for key in given_clients:
                    remove_from_cache(key)
        # The client of no event loop is left open
        try:
            assert cache.cache_dict.get("test_client_without_loop") is (
                client_without_loop
            )
            assert not client_without_loop.closed
        finally:
            remove_from_cache("test_client_without_loop")
