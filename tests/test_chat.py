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
Module defining tests for chat and tools.
"""

from __future__ import annotations

import os
import warnings
from copy import deepcopy
from datetime import datetime

import pytest

from contextgem.internal.data_models import _Message
from contextgem.internal.exceptions import LLMToolLoopLimitError, ModelAPIError
from contextgem.internal.loggers import logger
from contextgem.public import ChatSession, Document, DocumentLLM, register_tool
from tests.base import BaseTest
from tests.memory_profiling import check_locals_memory_usage, memory_profile_and_capture


class TestChat(BaseTest):
    """
    Tests for messages, chat sessions, chat with LLMs and tool calling.
    """

    @memory_profile_and_capture
    def test_init_message(self):
        """
        Tests for constructing a _Message instance and validating its content schema.
        """
        # Simple text message
        msg = _Message(role="user", content="Hello")
        msg._response_succeeded = False
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert isinstance(msg.unique_id, str) and msg.unique_id
        assert isinstance(msg.time_created, datetime) and msg.time_created
        assert msg._to_message_dict() == {"role": "user", "content": "Hello"}
        assert msg._response_succeeded is False

        # Validation errors for frozen fields
        with pytest.raises(ValueError):
            msg.content = "New content"
        with pytest.raises(ValueError):
            msg.role = "assistant"
        # Check for no overrides
        assert msg.role == "user"
        assert msg.content == "Hello"

        # Multimodal content (text + image_url)
        multimodal = [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
        ]
        msg_mm = _Message(role="user", content=multimodal)
        assert isinstance(msg_mm.content, list) and len(msg_mm.content) == 2
        assert msg_mm._response_succeeded is None

        # Validation errors for malformed parts
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"text": "missing type"}],
            )
        with pytest.raises(ValueError):
            _Message(role="user", content=[{"type": "text"}])
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "text", "text": ""}],
            )
        with pytest.raises(ValueError):
            _Message(role="user", content=[{"type": "image_url"}])
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "image_url", "image_url": {}}],
            )
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "image_url", "image_url": {"url": ""}}],
            )
        with pytest.raises(ValueError):
            _Message(
                role="user",
                content=[{"type": "audio", "url": "x"}],
            )

        # Assignment validation (validate_assignment=True)
        msg2 = _Message(role="assistant", content="ok")
        with pytest.raises(ValueError):
            msg2.content = [{"type": "text", "text": ""}]  # invalid after init
        # `_response_succeeded` defaults and assignment rules
        assert msg2._response_succeeded is None
        with pytest.raises(AttributeError):
            msg2._response_succeeded = True

        # System role should also forbid assignment
        msg_sys = _Message(role="system", content="hi")
        assert msg_sys._response_succeeded is None
        with pytest.raises(AttributeError):
            msg_sys._response_succeeded = False

        # User role allows assignment
        msg._response_succeeded = True
        assert msg._response_succeeded is True

        # Serialization roundtrip (dict/json) preserves fields and private unique_id
        d = msg.to_dict()
        msg_from_d = _Message.from_dict(d)
        assert msg_from_d == msg
        assert msg_from_d.role == msg.role
        assert msg_from_d.content == msg.content
        assert msg_from_d.unique_id == msg.unique_id
        assert msg_from_d.time_created == msg.time_created
        assert msg_from_d._response_succeeded == msg._response_succeeded

        j = msg.to_json()
        msg_from_j = _Message.from_json(j)
        assert msg_from_j == msg
        assert msg_from_j.role == msg.role
        assert msg_from_j.content == msg.content
        assert msg_from_j.unique_id == msg.unique_id
        assert msg_from_j.time_created == msg.time_created
        assert msg_from_j._response_succeeded == msg._response_succeeded

        msg._response_succeeded = None  # allow for re-setting to None

        # Disabled pydantic dump methods
        with pytest.raises(NotImplementedError):
            msg.model_dump()
        with pytest.raises(NotImplementedError):
            msg.model_dump_json()

        # Tool-related messages
        tool_msg = _Message(
            role="tool",
            content="result payload",
            tool_call_id="call_123",
            name="function_name",
        )
        tool_dict = tool_msg._to_message_dict()
        assert tool_dict == {
            "role": "tool",
            "content": "result payload",
            "tool_call_id": "call_123",
            "name": "function_name",
        }
        assert "tool_calls" not in tool_dict

        with pytest.raises(
            ValueError, match="must be set for messages with role 'tool'"
        ):
            _Message(role="tool", content="ok")

        with pytest.raises(
            ValueError, match="`name` must be set for messages with role 'tool'"
        ):
            _Message(role="tool", content="ok", tool_call_id="id_missing_name")

        with pytest.raises(
            ValueError, match="Tool message content must be a non-empty string"
        ):
            _Message(
                role="tool",
                content=[{"type": "text", "text": "x"}],
                tool_call_id="id1",
            )

        with pytest.raises(
            ValueError,
            match="`tool_calls` is only allowed for messages with role 'assistant'",
        ):
            _Message(
                role="tool",
                content="ok",
                name="fn",
                tool_call_id="id2",
                tool_calls=[
                    {"type": "function", "function": {"name": "fn", "arguments": "{}"}}
                ],
            )

        with pytest.raises(
            ValueError,
            match="`tool_call_id` is only allowed for messages with role 'tool'",
        ):
            _Message(role="user", content="hello", tool_call_id="some_id")

        tool_calls = [
            {
                "type": "function",
                "function": {"name": "search", "arguments": '{"q": "weather"}'},
            }
        ]
        asst_msg = _Message(
            role="assistant", content="thinking...", tool_calls=tool_calls
        )
        asst_dict = asst_msg._to_message_dict()
        assert asst_dict["role"] == "assistant"
        assert asst_dict["content"] == "thinking..."
        assert asst_dict["tool_calls"] == tool_calls
        assert "tool_call_id" not in asst_dict

        with pytest.raises(
            ValueError,
            match="`tool_calls` is only allowed for messages with role 'assistant'",
        ):
            _Message(role="user", content="hi", tool_calls=tool_calls)

        with pytest.raises(
            ValueError,
            match="`tool_call_id` is only allowed for messages with role 'tool'",
        ):
            _Message(role="assistant", content="ok", tool_call_id="id3")

        # Serialization roundtrip tests for tool and assistant-with-tools messages
        # Tool message roundtrip (dict/json)
        tool_d = tool_msg.to_dict()
        tool_from_d = _Message.from_dict(tool_d)
        assert tool_from_d == tool_msg
        assert tool_from_d.role == "tool"
        assert tool_from_d.content == "result payload"
        assert tool_from_d.tool_call_id == "call_123"
        assert tool_from_d.name == "function_name"
        assert tool_from_d.unique_id == tool_msg.unique_id
        assert tool_from_d.time_created == tool_msg.time_created
        assert "_response_succeeded" not in tool_d

        tool_j = tool_msg.to_json()
        tool_from_j = _Message.from_json(tool_j)
        assert tool_from_j == tool_msg
        assert tool_from_j.role == "tool"
        assert tool_from_j.content == "result payload"
        assert tool_from_j.tool_call_id == "call_123"
        assert tool_from_j.name == "function_name"
        assert tool_from_j.unique_id == tool_msg.unique_id
        assert tool_from_j.time_created == tool_msg.time_created

        # Assistant message with tool_calls roundtrip (dict/json)
        asst_d = asst_msg.to_dict()
        asst_from_d = _Message.from_dict(asst_d)
        assert asst_from_d == asst_msg
        assert asst_from_d.role == "assistant"
        assert asst_from_d.content == "thinking..."
        assert asst_from_d.tool_calls == tool_calls
        assert asst_from_d.tool_call_id is None
        assert asst_from_d.unique_id == asst_msg.unique_id
        assert asst_from_d.time_created == asst_msg.time_created
        assert "_response_succeeded" not in asst_d

        asst_j = asst_msg.to_json()
        asst_from_j = _Message.from_json(asst_j)
        assert asst_from_j == asst_msg
        assert asst_from_j.role == "assistant"
        assert asst_from_j.content == "thinking..."
        assert asst_from_j.tool_calls == tool_calls
        assert asst_from_j.tool_call_id is None
        assert asst_from_j.unique_id == asst_msg.unique_id
        assert asst_from_j.time_created == asst_msg.time_created

        check_locals_memory_usage(locals(), test_name="test_init_message")

    @memory_profile_and_capture
    def test_init_chat_session(self):
        """
        Tests for initializing `ChatSession` and enforcing message history rules.
        """

        # 1) System + user message order is preserved, and messages property returns a copy
        cs = ChatSession()
        system_msg = _Message(role="system", content="You are ContextGem.")
        user_msg = _Message(role="user", content="Hello")
        cs._set_messages([system_msg, user_msg])
        msgs = cs.messages
        assert len(msgs) == 2
        assert msgs[0].role == "system" and msgs[0].content == "You are ContextGem."
        assert msgs[1] == user_msg

        # messages property should return a copy
        msgs.append(_Message(role="assistant", content="Hi"))
        assert len(cs.messages) == 2

        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs)

        # reset clears all messages
        cs.reset()
        assert cs.messages == []

        # 2) Duplicate messages are not allowed
        dup = _Message(role="user", content="dup")
        # LLM instance is not required for chat session tests now
        cs2 = ChatSession()
        with pytest.raises(ValueError, match="contain duplicates"):
            cs2._extend_messages([dup, dup])

        # 3) A single system message must be the first message
        cs3 = ChatSession()
        with pytest.raises(ValueError, match="must be the first"):
            cs3._extend_messages(
                [
                    _Message(role="user", content="first"),
                    _Message(role="system", content="sys"),
                ]
            )

        # 4) Multiple system messages are not allowed
        cs4 = ChatSession()
        with pytest.raises(ValueError, match="Multiple 'system' messages"):
            cs4._extend_messages(
                [
                    _Message(role="system", content="one"),
                    _Message(role="system", content="another"),
                ]
            )

        check_locals_memory_usage(locals(), test_name="test_init_chat_session")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_chat_no_session(self):
        """
        Tests for the chat method.
        """
        for model, with_image in [
            (self.llm_extractor, False),
            (self.llm_extractor, True),
            (self.invalid_llm_with_valid_fallback, False),
        ]:
            with pytest.raises(ValueError):
                model.chat("")
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=1)  # type: ignore
            with pytest.raises(ValueError):
                model.chat(prompt="Test", images=[1])  # type: ignore
            with pytest.raises(TypeError):
                model.chat(images=self.test_img_png_invoice)  # type: ignore
            if with_image:
                # Check with text + image
                with pytest.warns(UserWarning, match="default system message"):
                    model.chat(
                        "What's the type of this document?",
                        images=[self.test_img_png_invoice],
                    )
                response = model.get_usage()[0].usage.calls[-1].response
                assert response is not None
                assert "invoice" in response.lower()
            else:
                # Check with text
                with pytest.warns(UserWarning, match="default system message"):
                    model.chat("What's the result of 2+2?")
                if model == self.invalid_llm_with_valid_fallback:
                    assert model.fallback_models
                    response = (
                        model.fallback_models[0].get_usage()[0].usage.calls[-1].response
                    )
                else:
                    response = model.get_usage()[0].usage.calls[-1].response
                assert response is not None
                assert "4" in response
            logger.debug(response)

        # Test for non-vision model
        text_only_model = DocumentLLM(model="cerebras/llama-3.3-70b")
        with pytest.raises(ValueError, match="vision"):
            text_only_model.chat(
                "What's the type of this document?", images=[self.test_img_png_invoice]
            )

        # Test for empty system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )
        assert model.system_message == ""
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")  # Capture all warnings
            model.chat("What's the result of 10+10?")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "Expected no warning, but got: " + str(
            user_warnings
        )
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "20" in response

        # Test for None system message (output language "en", which is the default)
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
        )
        assert model.system_message == self.default_system_message_en
        with pytest.warns(UserWarning, match="default system message"):
            model.chat("What's the result of 10+10?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "20" in response

        # Test for None system message (output language "adapt")
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            output_language="adapt",
        )
        assert model.system_message == self.default_system_message_non_en
        with pytest.warns(UserWarning, match="default system message"):
            model.chat("Hva er resultatet av 30+10?")
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "40" in response

        # Test with custom system message
        model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="Your name is John Doe. Introduce yourself as such.",
        )
        assert (
            model.system_message == "Your name is John Doe. Introduce yourself as such."
        )
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")  # Capture all warnings
            model.chat("What's your name?")
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0, "Expected no warning, but got: " + str(
            user_warnings
        )
        response = model.get_usage()[0].usage.calls[-1].response
        assert response is not None
        assert "John Doe" in response

        check_locals_memory_usage(locals(), test_name="test_chat_no_session")

    @pytest.mark.vcr
    @memory_profile_and_capture
    def test_chat_with_session(self):
        """
        Tests for the chat method with a chat session that preserves message history.
        """
        # ChatSession: text-only flow
        cs = ChatSession()
        # With default system message, a warning should be emitted on first call
        with pytest.warns(UserWarning, match="default system message"):
            reply1 = self.llm_extractor.chat(
                "What's the result of 3+5?",
                chat_session=cs,
            )
        assert reply1 and "8" in reply1
        # System + user + assistant should be in history
        msgs = cs.messages
        assert msgs[0].role == "system"
        assert (
            msgs[1].role == "user"
            and "3+5" in msgs[1].content
            and msgs[1]._response_succeeded is True
        )
        assert msgs[2].role == "assistant" and "8" in msgs[2].content
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs)

        # Second turn within same session
        with pytest.warns(UserWarning, match="default system message"):
            reply2 = self.llm_extractor.chat(
                "And what's 10-3?",
                chat_session=cs,
            )
        assert reply2 and "7" in reply2
        msgs2 = cs.messages
        # Expect two more messages appended (user, assistant)
        assert len(msgs2) == len(msgs) + 2
        assert (
            msgs2[-2].role == "user"
            and "10-3" in msgs2[-2].content
            and msgs2[-2]._response_succeeded is True
        )
        assert msgs2[-1].role == "assistant" and "7" in msgs2[-1].content
        # Follow-up: ask about previous answer to ensure history is preserved
        with pytest.warns(UserWarning, match="default system message"):
            reply3 = self.llm_extractor.chat(
                "What was your answer to my last question? Reply with only the number.",
                chat_session=cs,
            )
        assert reply3 and "7" in reply3
        msgs3 = cs.messages
        assert len(msgs3) == len(msgs2) + 2
        assert msgs3[-2].role == "user" and msgs3[-2]._response_succeeded is True
        assert msgs3[-1].role == "assistant"
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs)

        # ChatSession: vision model with images
        cs_v = ChatSession()
        with pytest.warns(UserWarning, match="default system message"):
            r_img = self.llm_extractor.chat(
                "What's the type of this document?",
                images=[self.test_img_png_invoice],
                chat_session=cs_v,
            )
        assert r_img and "invoice" in r_img.lower()
        msgs_v = cs_v.messages
        assert len(msgs_v) == 3
        assert msgs_v[0].role == "system"
        assert (
            msgs_v[1].role == "user"
            and "type of this document" in msgs_v[1].content[0]["text"]  # type: ignore
            and msgs_v[1]._response_succeeded is True
        )
        # user message content is multimodal list when images are supplied
        assert isinstance(msgs_v[1].content, list) and len(msgs_v[1].content) == 2
        assert any(
            isinstance(i, dict) and i.get("type") == "image_url"
            for i in msgs_v[1].content
        )
        assert msgs_v[2].role == "assistant" and "invoice" in msgs_v[2].content.lower()  # type: ignore
        # Follow-up in the same vision session without re-sending images: ensure history is preserved
        with pytest.warns(UserWarning, match="default system message"):
            r_img2 = self.llm_extractor.chat(
                "What type of document did you say it was earlier?",
                chat_session=cs_v,
            )
        assert r_img2 and "invoice" in r_img2.lower()
        msgs_v2 = cs_v.messages
        assert len(msgs_v2) == len(msgs_v) + 2
        assert msgs_v2[-2].role == "user" and msgs_v2[-2]._response_succeeded is True
        assert (
            msgs_v2[-1].role == "assistant" and "invoice" in msgs_v2[-1].content.lower()  # type: ignore
        )
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_v)

        # Non-vision model must still fail with images even with a ChatSession
        text_only_model = DocumentLLM(model="cerebras/llama-3.3-70b")
        cs_empty = ChatSession()
        with pytest.raises(ValueError, match="vision"):
            text_only_model.chat(
                "What's the type of this document?",
                images=[self.test_img_png_invoice],
                chat_session=ChatSession(),
            )
        assert not cs_empty.messages
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_empty)

        # ChatSession: empty system message should not warn
        no_sys_model = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )
        cs_no_sys = ChatSession()
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")
            resp = no_sys_model.chat("What's 9+1?", chat_session=cs_no_sys)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
        assert resp and "10" in resp
        # history: no system auto-injected when system_message is ""
        assert (
            len(cs_no_sys.messages) == 2
            and cs_no_sys.messages[0].role == "user"
            and "9+1" in cs_no_sys.messages[0].content
            and cs_no_sys.messages[0]._response_succeeded is True
            and cs_no_sys.messages[1].role == "assistant"
            and "10" in cs_no_sys.messages[1].content
        )
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_no_sys)

        # ChatSession: custom system message should not warn and be first
        custom_sys = "Your name is John Doe. Introduce yourself as such."
        m_custom = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message=custom_sys,
        )
        cs_custom = ChatSession()
        with warnings.catch_warnings(record=True) as w:  # expect no warning
            warnings.simplefilter("always")
            m_custom.chat("What's your name?", chat_session=cs_custom)
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(user_warnings) == 0
        assert cs_custom.messages and cs_custom.messages[0].role == "system"
        assert cs_custom.messages[0].content == custom_sys
        assert (
            cs_custom.messages[1].role == "user"
            and "what's your name?" in cs_custom.messages[1].content.lower()  # type: ignore
            and cs_custom.messages[1]._response_succeeded is True
        )
        assert (
            cs_custom.messages[2].role == "assistant"
            and "John Doe" in cs_custom.messages[2].content
        )
        # test serialization and cloning
        self.check_instance_serialization_and_cloning(cs_custom)

        # ChatSession: fallback path
        cs_fb = ChatSession()
        with pytest.warns(UserWarning, match="default system message"):
            self.invalid_llm_with_valid_fallback.chat("What's 2+2?", chat_session=cs_fb)
        # Ensure no duplicate messages in chat despite retries
        assert len(cs_fb.messages) == 3
        assert cs_fb.messages[0].role == "system"
        assert (
            cs_fb.messages[1].role == "user"
            and "what's 2+2?" in cs_fb.messages[1].content.lower()  # type: ignore
        )
        assert cs_fb.messages[1]._response_succeeded is True
        assert (
            cs_fb.messages[2].role == "assistant" and "4" in cs_fb.messages[2].content
        )

        # If the LLM call fails with an API error, the pending user message in the
        # provided ChatSession should remain with _response_succeeded == False and
        # no assistant message should be appended.
        cs_err = ChatSession()
        with pytest.raises(ModelAPIError):
            self.llm_invalid.chat("Will this fail?", chat_session=cs_err)

        msgs = cs_err.messages
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert (
            msgs[1].role == "user"
            and "will this fail" in msgs[1].content.lower()  # type: ignore
            and msgs[1]._response_succeeded is False
        )

        # Second attempt with the same prompt should not duplicate the user message
        with pytest.raises(ModelAPIError):
            self.llm_invalid.chat("Will this fail?", chat_session=cs_err)
        msgs2 = cs_err.messages
        assert len(msgs2) == 2
        assert (
            msgs2[1].role == "user"
            and "will this fail" in msgs2[1].content.lower()  # type: ignore
            and msgs2[1]._response_succeeded is False
        )

        # If both primary and fallback LLMs fail, the pending user message should
        # remain with _response_succeeded == False and no assistant message should
        # be appended.
        cs_err_fb = ChatSession()
        bad_primary = DocumentLLM(**self._invalid_llm_kwargs)  # ty: ignore[invalid-argument-type]
        invalid_llm_kwargs_2 = deepcopy(self._invalid_llm_kwargs)
        invalid_llm_kwargs_2["model"] = "newmodel/123"
        bad_primary.fallback_models = [
            DocumentLLM(
                **invalid_llm_kwargs_2,  # ty: ignore[invalid-argument-type]
                is_fallback=True,
            )
        ]

        with pytest.raises(ModelAPIError):
            bad_primary.chat("Another failing call", chat_session=cs_err_fb)

        msgs = cs_err_fb.messages
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[1].role == "user" and msgs[1]._response_succeeded is False

        check_locals_memory_usage(locals(), test_name="test_chat_with_session")

    @pytest.mark.vcr
    @pytest.mark.parametrize(
        "parallel_tool_calls, use_session",
        [(False, False), (False, True), (True, False), (True, True)],
    )
    @memory_profile_and_capture
    def test_chat_with_tools(self, parallel_tool_calls: bool, use_session: bool):
        """
        Test chat with tools.
        """

        # Track invocations per tool
        calls = {"invoice": 0, "discount": 0, "surcharge": 0, "final": 0}

        # Tool 1: compute invoice total from structured list of items
        @register_tool
        def compute_invoice_total(items: list[dict]) -> str:
            """
            Computes invoice total as sum(qty*price) over items.
            """
            calls["invoice"] += 1
            total = 0
            for it in items:
                qty = int(it.get("qty", 0))
                price = int(it.get("price", 0))
                total += qty * price
            return str(total)

        # Tool 2: discount amount lookup by opaque order id (LLM cannot infer value)
        @register_tool
        def get_discount_amount(order_id: str) -> str:
            """
            Returns discount amount for a given order id.
            """
            calls["discount"] += 1
            mapping = {"ORD-123": "4"}
            return mapping.get(str(order_id), "0")

        # Tool 3: surcharge lookup by opaque order id
        @register_tool
        def get_surcharge(order_id: str) -> str:
            """
            Returns surcharge amount for a given order id.
            """
            calls["surcharge"] += 1
            mapping = {"ORD-123": "2"}
            return mapping.get(str(order_id), "0")

        # Tool 4: compute final amount from previous tool outputs
        @register_tool
        def compute_final_amount(
            invoice_total: str, discount: str, surcharge: str
        ) -> str:
            """
            Computes final amount as (invoice_total - discount) + surcharge.
            """
            calls["final"] += 1
            try:
                res = (int(invoice_total) - int(discount)) + int(surcharge)
            except Exception:
                # Be robust if model passes non-ints
                res = (int(float(invoice_total)) - int(float(discount))) + int(
                    float(surcharge)
                )
            return str(res)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "compute_invoice_total",
                    "description": "Compute invoice total as sum(qty*price) over items",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "qty": {"type": "integer"},
                                        "price": {"type": "integer"},
                                    },
                                    "required": ["qty", "price"],
                                },
                                "minItems": 1,
                            }
                        },
                        "required": ["items"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_discount_amount",
                    "description": "Return discount amount for a given order id",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_surcharge",
                    "description": "Return surcharge amount for a given order id",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "compute_final_amount",
                    "description": "Compute final amount as (invoice_total - discount) + surcharge",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "invoice_total": {"type": "string"},
                            "discount": {"type": "string"},
                            "surcharge": {"type": "string"},
                        },
                        "required": ["invoice_total", "discount", "surcharge"],
                    },
                },
            },
        ]

        # Configure LLM for tools; leave tool_choice to auto so it can call both
        llm_fallback = DocumentLLM(
            model="openai/gpt-5.6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
            is_fallback=True,
        )
        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
            tools=tools,
            parallel_tool_calls=parallel_tool_calls,
        )

        # Test for `supports_tools` mismatch between main and fallback LLMs
        llm._supports_tools = False
        with pytest.raises(ValueError, match="supports_tools"):
            llm.fallback_models = [llm_fallback]
        llm._supports_tools = True
        llm.fallback_models = [llm_fallback]

        # Mark support exolicitly due to litellm bug where litellm.supports_parallel_function_calling()
        # returns False for all models: https://github.com/BerriAI/litellm/issues/13980
        # TODO: remove this once litellm bug is fixed
        llm._supports_parallel_tool_calls = True

        # Optional chat session to preserve history
        cs = ChatSession() if use_session else None
        prompt = (
            "Use the available tools to compute a final amount. "
            'First call `compute_invoice_total` with exactly items=[{"qty":2,"price":3},{"qty":1,"price":3}]. '
            "Then call `get_discount_amount` with order_id='ORD-123'. "
            "Then call `get_surcharge` with order_id='ORD-123'. "
            "Then call `compute_final_amount` using the exact outputs returned by the previous tools as "
            "invoice_total, discount, surcharge. "
            "Reply only with the number returned by `compute_final_amount`, nothing else. "
            "Do not guess any values; you must call the tools."
        )
        if parallel_tool_calls:
            llm._supports_parallel_tool_calls = False  # set to False to test error
            with pytest.raises(ValueError, match="parallel tool calls"):
                llm.chat(prompt, chat_session=cs)
            llm._supports_parallel_tool_calls = True
        else:
            llm.tool_max_rounds = 1
            with pytest.raises(LLMToolLoopLimitError):
                # compute_invoice_total will be called first, then error
                llm.chat(prompt, chat_session=cs)
            llm.tool_max_rounds = (
                4  # now this should pass, as we have 4 tools, 1 round per tool
            )
        answer = llm.chat(prompt, chat_session=cs)
        logger.debug(
            f"Answer with tools (parallel tool calls: {parallel_tool_calls}):\n"
            + answer
        )

        # Expected: (2*3 + 1*3) - 4 + 2 = 7
        assert answer and "7" in answer

        # Only user + assistant in primary history (when session is used)
        if use_session:
            assert cs is not None
            assert (
                len(cs.messages) == 2
                and cs.messages[0].role == "user"
                and cs.messages[1].role == "assistant"
            )

        # Each tool must be invoked exactly once
        assert (
            calls["invoice"] == 1 if parallel_tool_calls else 2
        )  # twice when parallel_tool_calls=False because of the caught LLMToolLoopLimitError above
        assert calls["discount"] == 1
        assert calls["surcharge"] == 1
        assert calls["final"] == 1

        # Usage must include at least two provider calls (tool round-trip + follow-up)
        usage_calls = llm.get_usage()[0].usage.calls
        logger.info(f"Usage calls: {len(usage_calls)}")
        assert len(usage_calls) == 3 if parallel_tool_calls else 5

        # Tool messages are stored separately when session is used
        if use_session:
            assert cs is not None
            tool_msgs = cs._tool_messages
            assert tool_msgs and any(
                m.role == "assistant" and m.tool_calls for m in tool_msgs
            )
            assert any(m.role == "tool" for m in tool_msgs)

        # Check serialization of LLM with registered tools
        self._check_deserialized_llm_config_eq(llm)

        # Warning should be emitted that tools are ignored during extraction workflows
        with pytest.warns(UserWarning, match="ignored during extraction workflows"):
            llm.extract_all(document=Document(raw_text="test"))

        check_locals_memory_usage(locals(), test_name="test_chat_with_tools")

    @memory_profile_and_capture
    def test_invalid_tool_registrations(self):
        """
        Validate errors for invalid LLM tool registrations and tool configs.
        """

        llm = DocumentLLM(
            model="openai/gpt-6-luna",
            api_key=os.getenv("CONTEXTGEM_OPENAI_API_KEY"),
            system_message="",
        )

        valid_schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"],
        }

        # 1) register_tool with non-callable
        with pytest.raises(TypeError, match="Tool handler must be callable"):
            register_tool(123)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        # 2) register_tool with positional-only parameter
        def define_bad_handler():
            """
            Defines invalid handler with positional-only parameter.
            """

            @register_tool
            def bad(a, /, b: int) -> str:  # positional-only arg is invalid
                """
                Invalid handler with positional-only parameter.
                """
                return str(b)

        with pytest.raises(ValueError, match="keyword arguments"):
            define_bad_handler()

        # 3) LLM.tools missing `function` dict
        with pytest.raises(
            ValueError, match="Tool function must be a non-empty dictionary"
        ):
            llm.tools = [{"type": "function"}]
        assert llm.tools is None

        # 4) Empty tool name
        with pytest.raises(ValueError, match="Tool name must be a non-empty string"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {"name": " ", "parameters": valid_schema},
                }
            ]
        assert llm.tools is None

        # 5) Parameters not a dict
        with pytest.raises(ValueError, match="Expected a JSON object"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {"name": "foo", "parameters": "not a dict"},
                }
            ]
        assert llm.tools is None

        # 6) Parameters with wrong `type`
        with pytest.raises(ValueError, match="type == 'object'"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "array",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 7) Parameters with non-dict properties
        with pytest.raises(ValueError, match="must include a 'properties' object"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "object",
                            "properties": [],
                            "required": [],
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 8) Parameters with invalid `required`
        with pytest.raises(ValueError, match="'required' must be a list of strings"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": "x",
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 9) Parameters with missing required keys in properties
        with pytest.raises(ValueError, match="missing in properties: x"):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "foo",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": ["x"],
                        },
                    },
                }
            ]
        assert llm.tools is None

        # 10) Name does not match any registered handler
        with pytest.raises(
            ValueError, match="No registered handler found for tool 'unknown_tool'"
        ):
            llm.tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "unknown_tool",
                        "parameters": valid_schema,
                    },
                }
            ]
        assert llm.tools is None

        check_locals_memory_usage(locals(), test_name="test_invalid_tool_registrations")
