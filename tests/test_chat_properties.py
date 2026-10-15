# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the property-based tests of chat sessions.

Hypothesis generates a sequence of steps: chat turns, which are answered after any
number of tool rounds or fail (on an error of the LLM API, or at the limit of tool
rounds) and are then sent again or left, taken over by a fallback LLM, or sent by
another LLM that continues the session; changes of the tools of an LLM; and
the serialization, the cloning and the reset of the session. What the LLM API
returns is scripted, with or without thinking blocks, and stubbed: no requests are
sent, and no cassettes are needed.

After each step, the session and the requests sent are checked against a model of
the conversation: the history keeps the tool calls and the thinking blocks of
the answered turns only, ``messages`` lists the conversation without the tool calls,
each request is sent with the history, and the thinking blocks are sent back only
to the model with the tools that they were returned for.
"""

from __future__ import annotations

import itertools
import json
import types
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

import pytest
from aiolimiter import AsyncLimiter
from hypothesis import HealthCheck, event, given, note, settings
from hypothesis import strategies as st

from contextgem.internal.base import llms as llms_module
from contextgem.internal.exceptions import LLMToolLoopLimitError, ModelAPIError
from contextgem.public import ChatSession, DocumentLLM, register_tool
from tests.base import BaseTest


# The module object that the LLMs call the API through, i.e. the one to patch
# the API calls on
litellm = llms_module.litellm

# The examples of a test share the function-scoped fixtures of the test, e.g.
# the network guard, which is intended: each example creates what it uses.
_PROPERTY_SETTINGS = settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# Models of the LLMs: the main LLM, its fallback LLM, and another LLM that
# continues the session. No requests are sent to them.
_MAIN_MODEL = "openai/gpt-6-luna"
_FALLBACK_MODEL = "openai/gpt-5.6-luna"
_OTHER_MODEL = "openai/gpt-6.1-sol"

# The rounds of tool calls that an LLM may make in a turn
_TOOL_MAX_ROUNDS = 2


@register_tool
def get_a(x: str) -> str:
    """
    Returns the value A of x.
    """
    return f"get_a:{x}"


@register_tool
def get_b(x: str) -> str:
    """
    Returns the value B of x.
    """
    return f"get_b:{x}"


_TOOLS = {"get_a": get_a, "get_b": get_b}

# The tools of the main LLM and of its fallback LLM, and those that the other LLM
# can be given, in the order in which they are given
_MAIN_TOOLS = ("get_a", "get_b")
_OTHER_TOOL_SETS = (("get_a",), ("get_a", "get_b"), ("get_b", "get_a"))

# The thinking blocks of a response, as litellm returns them, if any
_THINKING_BLOCKS = st.one_of(
    st.none(),
    st.lists(
        st.one_of(
            st.builds(
                lambda text, signature: {
                    "type": "thinking",
                    "thinking": text,
                    "signature": signature,
                },
                st.text(alphabet="abc ", max_size=5),
                st.text(alphabet="xyz", min_size=1, max_size=4),
            ),
            st.builds(
                lambda data: {"type": "redacted_thinking", "data": data},
                st.text(alphabet="xyz", min_size=1, max_size=4),
            ),
        ),
        min_size=1,
        max_size=2,
    ),
)

# What identifies a request for its thinking blocks: the model and the names
# of the tools (the system message is the same for all the requests)
_Identity = tuple[str, frozenset[str]]


class _StubbedAPIError(Exception):
    """
    Non-transient error of the stubbed LLM API, i.e. an error that is not retried.
    """

    status_code = 401


@dataclass
class _Response:
    """
    A scripted response of the LLM API: tool calls, an answer, or an error.

    :ivar kind: "tool_calls", "answer" or "error".
    :vartype kind: str
    :ivar text: The text of the response, if any.
    :vartype text: str | None
    :ivar tool_calls: The tool calls, as (call ID, tool name, argument).
    :vartype tool_calls: list[tuple[str, str, str]]
    :ivar thinking_blocks: The thinking blocks of the response, if any.
    :vartype thinking_blocks: list[dict[str, Any]] | None
    """

    kind: str
    text: str | None = None
    tool_calls: list[tuple[str, str, str]] = field(default_factory=list)
    thinking_blocks: list[dict[str, Any]] | None = None

    def to_completion(self) -> Any:
        """
        Returns the completion, as litellm returns it, of the response.

        :return: The completion.
        :rtype: Any
        """
        tool_calls = [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps({"x": x})},
            }
            for call_id, name, x in self.tool_calls
        ]
        message = types.SimpleNamespace(
            content=self.text,
            tool_calls=tool_calls or None,
            thinking_blocks=self.thinking_blocks,
        )
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=message,
                    finish_reason="tool_calls" if tool_calls else "stop",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


class _ChatAPIStub:
    """
    Stub of the LLM API that returns the responses scripted for each model,
    one by one, and records the requests.
    """

    def __init__(self) -> None:
        self.responses: dict[str, deque[_Response]] = {
            model: deque() for model in (_MAIN_MODEL, _FALLBACK_MODEL, _OTHER_MODEL)
        }
        self.requests: list[dict[str, Any]] = []

    async def acompletion(self, **kwargs: Any) -> Any:
        """
        Simulates the LLM API.
        """
        response = self.responses[kwargs["model"]].popleft()
        # A copy, as the messages are added to after the request is sent
        self.requests.append(
            {
                "model": kwargs["model"],
                "messages": json.loads(json.dumps(kwargs["messages"])),
            }
        )
        if response.kind == "error":
            raise _StubbedAPIError("Authentication failed")
        return response.to_completion()


@dataclass
class _Entry:
    """
    A message of the modelled conversation.
    """

    role: str
    content: str | None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    thinking_blocks: list[dict[str, Any]] | None = None
    # What identifies the request that the thinking blocks were returned for
    identity: _Identity | None = None
    # Whether the turn of a user message is answered
    answered: bool | None = None

    def render(self, identity: _Identity) -> dict[str, Any]:
        """
        Returns the message as it is sent in a request with the given identity.

        :param identity: What identifies the request.
        :type identity: _Identity
        :return: The message.
        :rtype: dict[str, Any]
        """
        message: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.role == "tool":
            message["tool_call_id"] = self.tool_call_id
            message["name"] = self.name
        if self.tool_calls is not None:
            message["tool_calls"] = self.tool_calls
        if self.thinking_blocks is not None and self.identity == identity:
            message["thinking_blocks"] = self.thinking_blocks
        return message

    def as_tuple(self) -> tuple[Any, ...]:
        """
        Returns what is compared with a message of the session.

        :return: The role, the content, the tool calls, the ID of the tool call,
            the name of the tool and the thinking blocks.
        :rtype: tuple[Any, ...]
        """
        return (
            self.role,
            self.content,
            self.tool_calls,
            self.tool_call_id,
            self.name,
            self.thinking_blocks,
        )


@dataclass
class _Attempt:
    """
    An attempt of an LLM to answer a turn, and what it is expected to do.
    """

    model: str
    identity: _Identity
    responses: list[_Response]
    # The messages of the requests that the attempt sends
    requests: list[list[dict[str, Any]]] = field(default_factory=list)
    # The tool calls and their results of the rounds that the attempt makes
    round_entries: list[_Entry] = field(default_factory=list)
    # "answered", "error" or "limit"
    outcome: str = ""
    answer: _Entry | None = None

    def simulate(self, history: list[_Entry]) -> None:
        """
        Works out what the attempt sends and what it ends with.

        :param history: The history of the session, with the user message of the turn.
        :type history: list[_Entry]
        :return: None
        :rtype: None
        """
        messages = [entry.render(self.identity) for entry in history]
        # Each response before the current one is a response with tool calls,
        # i.e. a round of tool calls made
        for rounds, response in enumerate(self.responses):
            self.requests.append(list(messages))
            if response.kind == "error":
                self.outcome = "error"
                return
            if response.kind == "answer":
                self.outcome = "answered"
                self.answer = _Entry(
                    "assistant",
                    response.text,
                    thinking_blocks=response.thinking_blocks,
                    identity=self.identity if response.thinking_blocks else None,
                )
                return
            if rounds == _TOOL_MAX_ROUNDS:
                self.outcome = "limit"
                return
            assistant = _Entry(
                "assistant",
                response.text,
                tool_calls=[
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps({"x": x})},
                    }
                    for call_id, name, x in response.tool_calls
                ],
                thinking_blocks=response.thinking_blocks,
                identity=self.identity if response.thinking_blocks else None,
            )
            results = [
                _Entry("tool", f"{name}:{x}", tool_call_id=call_id, name=name)
                for call_id, name, x in response.tool_calls
            ]
            self.round_entries += [assistant, *results]
            messages += [entry.render(self.identity) for entry in (assistant, *results)]
        raise AssertionError("The scripted responses end without an outcome")


def _draw_responses(
    data: st.DataObject, tool_names: tuple[str, ...], call_ids: Iterator[int]
) -> list[_Response]:
    """
    Draws the responses of an attempt: rounds of tool calls, then an answer, unless
    the rounds exceed the limit, and possibly an error of the API at one of them.

    :param data: The Hypothesis data object.
    :type data: st.DataObject
    :param tool_names: The names of the tools of the LLM.
    :type tool_names: tuple[str, ...]
    :param call_ids: The source of the IDs of the tool calls, unique in the session.
    :type call_ids: Iterator[int]
    :return: The responses.
    :rtype: list[_Response]
    """
    # Mostly within the limit of tool rounds, sometimes over it
    n_tool_responses = data.draw(
        st.sampled_from([*range(_TOOL_MAX_ROUNDS + 1)] * 2 + [_TOOL_MAX_ROUNDS + 1])
    )
    responses = [
        _Response(
            "tool_calls",
            text=data.draw(st.sampled_from([None, "Checking."])),
            tool_calls=[
                (
                    f"call_{next(call_ids)}",
                    data.draw(st.sampled_from(tool_names)),
                    data.draw(st.sampled_from("123")),
                )
                for _ in range(data.draw(st.integers(1, 2)))
            ],
            thinking_blocks=data.draw(_THINKING_BLOCKS),
        )
        for _ in range(n_tool_responses)
    ]
    if n_tool_responses <= _TOOL_MAX_ROUNDS:
        responses.append(
            _Response(
                "answer",
                text=data.draw(st.sampled_from(["Done.", "It is get_a:1."])),
                thinking_blocks=data.draw(_THINKING_BLOCKS),
            )
        )
    # An error of the API at one of the requests, in one attempt of four
    if data.draw(st.integers(0, 3)) == 0:
        fail_at = data.draw(st.integers(0, len(responses) - 1))
        responses = [*responses[:fail_at], _Response("error")]
    return responses


def _create_llm(
    model: str, tool_names: tuple[str, ...], is_fallback: bool = False
) -> DocumentLLM:
    """
    Creates an LLM for a chat with the stubbed LLM API. Failed requests are
    not retried, and the requests are not paced.

    :param model: The model.
    :type model: str
    :param tool_names: The names of the tools of the LLM.
    :type tool_names: tuple[str, ...]
    :param is_fallback: Whether the LLM is a fallback LLM. Defaults to False.
    :type is_fallback: bool
    :return: The LLM.
    :rtype: DocumentLLM
    """
    llm = DocumentLLM(
        model=model,
        api_key="test_api_key",
        system_message="",
        tools=[_TOOLS[name] for name in tool_names],
        num_retries_failed_request=0,
        tool_max_rounds=_TOOL_MAX_ROUNDS,
        is_fallback=is_fallback,
    )
    llm.async_limiter = AsyncLimiter(10_000, 1)
    return llm


def _check_session(session: ChatSession, history: list[_Entry]) -> None:
    """
    Checks the session against the modelled conversation.

    :param session: The chat session.
    :type session: ChatSession
    :param history: The modelled history of the conversation.
    :type history: list[_Entry]
    :return: None
    :rtype: None
    """
    # The history keeps the tool calls, their results and the thinking blocks
    # of the answered turns
    assert [
        (m.role, m.content, m.tool_calls, m.tool_call_id, m.name, m.thinking_blocks)
        for m in session._messages
    ] == [entry.as_tuple() for entry in history]
    # The key of the thinking blocks tells their identities apart
    keys_by_identity: dict[_Identity, set[str | None]] = {}
    for message, entry in zip(session._messages, history, strict=True):
        if entry.identity is not None:
            keys_by_identity.setdefault(entry.identity, set()).add(
                message.thinking_blocks_key
            )
    assert all(len(keys) == 1 for keys in keys_by_identity.values())
    keys = [next(iter(keys)) for keys in keys_by_identity.values()]
    assert None not in keys and len(set(keys)) == len(keys)
    # `messages` lists the conversation without the tool calls
    conversation = [
        entry
        for entry in history
        if entry.role == "user" or entry.tool_calls is None and entry.role != "tool"
    ]
    assert [(m.role, m.content) for m in session.messages] == [
        (entry.role, entry.content) for entry in conversation
    ]
    assert [m._response_succeeded for m in session.messages if m.role == "user"] == [
        entry.answered for entry in conversation if entry.role == "user"
    ]


class TestChatProperties(BaseTest):
    """
    Property-based tests of chat sessions with tools and thinking blocks.
    """

    @_PROPERTY_SETTINGS
    @given(data=st.data())
    def test_chat_session_over_any_steps(self, data: st.DataObject):
        """
        Tests a chat session over any sequence of turns, changes of the tools,
        serializations, clonings and resets, against a model of the conversation
        (see the module docstring).
        """
        stub = _ChatAPIStub()
        main_llm = _create_llm(_MAIN_MODEL, _MAIN_TOOLS)
        main_llm.fallback_models = [
            _create_llm(_FALLBACK_MODEL, _MAIN_TOOLS, is_fallback=True)
        ]
        other_tools = _OTHER_TOOL_SETS[0]
        other_llm = _create_llm(_OTHER_MODEL, other_tools)
        session = ChatSession()
        history: list[_Entry] = []
        call_ids = itertools.count()
        prompt_ids = itertools.count()

        with patch.object(litellm, "acompletion", new=stub.acompletion):
            for _ in range(data.draw(st.integers(1, 12), label="number of steps")):
                step = data.draw(
                    st.sampled_from(
                        ["main_turn"] * 2
                        + ["other_turn"] * 3
                        + ["change_tools"] * 2
                        + ["restore", "reset"]
                    ),
                    label="step",
                )

                if step == "change_tools":
                    # Other tools, or the same tools in another order
                    other_tools = data.draw(
                        st.sampled_from(
                            [t for t in _OTHER_TOOL_SETS if t != other_tools]
                        ),
                        label="tools",
                    )
                    other_llm.tools = [_TOOLS[name] for name in other_tools]

                elif step == "restore":
                    how = data.draw(
                        st.sampled_from(["json", "dict", "clone"]), label="restore"
                    )
                    if how == "json":
                        session = ChatSession.from_json(session.to_json())
                    elif how == "dict":
                        session = ChatSession.from_dict(session.to_dict())
                    else:
                        session = session.clone()

                elif step == "reset":
                    session.reset()
                    history.clear()

                else:
                    # The turn: the user message is sent again if the last turn
                    # failed, or a new one is sent
                    last = history[-1] if history else None
                    if (
                        last is not None
                        and last.role == "user"
                        and not last.answered
                        and data.draw(st.booleans(), label="send again")
                    ):
                        user_entry = last
                    else:
                        user_entry = _Entry(
                            "user", f"Question {next(prompt_ids)}", answered=False
                        )
                        history.append(user_entry)

                    # The attempts of the LLMs to answer the turn: the main LLM is
                    # followed by its fallback LLM if it fails on an error of the API
                    if step == "main_turn":
                        llm = main_llm
                        attempts = [
                            _Attempt(
                                _MAIN_MODEL,
                                (_MAIN_MODEL, frozenset(_MAIN_TOOLS)),
                                _draw_responses(data, _MAIN_TOOLS, call_ids),
                            )
                        ]
                    else:
                        llm = other_llm
                        attempts = [
                            _Attempt(
                                _OTHER_MODEL,
                                (_OTHER_MODEL, frozenset(other_tools)),
                                _draw_responses(data, other_tools, call_ids),
                            )
                        ]
                    attempts[0].simulate(history)
                    if step == "main_turn" and attempts[0].outcome == "error":
                        attempts.append(
                            _Attempt(
                                _FALLBACK_MODEL,
                                (_FALLBACK_MODEL, frozenset(_MAIN_TOOLS)),
                                _draw_responses(data, _MAIN_TOOLS, call_ids),
                            )
                        )
                        attempts[1].simulate(history)
                    for attempt in attempts:
                        stub.responses[attempt.model].extend(attempt.responses)
                    note(
                        f"{step}: "
                        + "; ".join(
                            f"{a.model} -> {[r.kind for r in a.responses]} ({a.outcome})"
                            for a in attempts
                        )
                    )

                    n_requests_before = len(stub.requests)
                    outcome = attempts[-1].outcome
                    event(f"turn: {step}, {len(attempts)} attempt(s), {outcome}")
                    if any(
                        "thinking_blocks" in message
                        for message in attempts[-1].requests[0][: len(history) - 1]
                    ):
                        event("thinking blocks of an earlier turn sent back")
                    if any(
                        entry.thinking_blocks is not None
                        and entry.identity != attempts[-1].identity
                        for entry in history
                    ):
                        event("thinking blocks of another identity left out")
                    if any(
                        entry.thinking_blocks is not None
                        and entry.identity is not None
                        and entry.identity[0] == attempts[-1].model
                        and entry.identity != attempts[-1].identity
                        for entry in history
                    ):
                        event("thinking blocks of the same model with other tools")
                    if outcome == "answered":
                        answer_entry = attempts[-1].answer
                        assert answer_entry is not None
                        answer = llm.chat(
                            user_entry.content or "", chat_session=session
                        )
                        assert answer == answer_entry.content
                        user_entry.answered = True
                        history += [*attempts[-1].round_entries, answer_entry]
                    else:
                        error = (
                            LLMToolLoopLimitError
                            if outcome == "limit"
                            else ModelAPIError
                        )
                        with pytest.raises(error):
                            llm.chat(user_entry.content or "", chat_session=session)

                    # Each request is sent with the history, and every scripted
                    # response is used
                    sent = stub.requests[n_requests_before:]
                    assert [request["model"] for request in sent] == [
                        attempt.model for attempt in attempts for _ in attempt.requests
                    ]
                    assert [request["messages"] for request in sent] == [
                        messages
                        for attempt in attempts
                        for messages in attempt.requests
                    ]
                    assert not any(stub.responses.values())

                _check_session(session, history)
