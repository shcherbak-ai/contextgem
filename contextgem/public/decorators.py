# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Public decorators for extending or integrating with the framework.

This module contains decorators that are part of the public API and intended
for end users to apply to their own functions or classes.
"""

from __future__ import annotations

import functools
import inspect

from contextgem.internal.typings.types import ToolHandler


def register_tool(func: ToolHandler, /) -> ToolHandler:
    """
    Registers a function as a tool for LLM chat with tools.

    The function defines the tool: its name is the name of the tool, its docstring
    describes the tool to the LLM, and its parameters, with their type hints and
    default values, define the parameters of the tool. The descriptions of the
    parameters are taken from the docstring, or from ``Annotated`` type hints, which
    can also set constraints, e.g. ``Annotated[int, Field(description="...", ge=1)]``.

    Pass the decorated function to ``tools`` of the LLM. To use an object of your
    application in a tool, e.g. a database connection, define the tool as a function
    that uses it, e.g. a closure: a bound method or a ``functools.partial`` cannot be
    registered.

    An async function runs in the event loop of the chat, and a sync function in
    a worker thread, without an event loop, so that it does not block the loop and
    can use libraries that refuse to run where one is running, e.g. the ORM of Django.
    Such a function does not see what the calling code keeps for its own thread,
    e.g. an open database transaction.

    To tell the LLM why a call failed, e.g. that a record does not exist, the function
    raises ``ToolCallError``, whose message is sent to the LLM, which can call
    the tool again. Any other exception that it raises stops the chat, and is raised
    from ``chat()`` as it is.

    :param func: A function to be used as a tool. Can be sync or async, and must
        return a non-empty string.
    :type func: ToolHandler
    :return: The same function, marked as a registered tool.
    :rtype: ToolHandler
    :raises TypeError: If the provided object is not callable, if it is a bound method
        or a ``functools.partial``, or if a parameter of the function has no type hint.
    :raises ValueError: If the signature cannot be inspected, if the function has
        positional-only parameters or accepts ``*args`` or ``**kwargs``, if it has
        no docstring, or if the function name is empty.
    """

    if not callable(func):
        raise TypeError("Tool handler must be callable")
    if inspect.ismethod(func) or isinstance(func, functools.partial):
        # Neither can be marked as a tool, nor has the name and signature of a tool
        raise TypeError(
            "Tool handler must be a function: a bound method or a functools.partial "
            "cannot be registered as a tool. To use an object of your application in "
            "a tool, e.g. a database connection, define a function that uses it, "
            "e.g. a closure."
        )
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Tool handler must be a function with an inspectable signature"
        ) from e
    for param in sig.parameters.values():
        if param.kind == param.POSITIONAL_ONLY:
            raise ValueError(
                "Tool handler must accept keyword arguments (no positional-only params)"
            )
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise ValueError(
                "Tool handler must not accept `*args` or `**kwargs`: each parameter "
                "of a tool has to be declared, as the parameters are described to the LLM."
            )
        if param.annotation is param.empty:
            raise TypeError(
                f"Tool parameter '{param.name}' has no type annotation. "
                "All tool parameters must have type hints."
            )

    tool_name = getattr(func, "__name__", None) or ""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("Tool name must be a non-empty string")

    if not (inspect.getdoc(func) or "").strip():
        raise ValueError(
            f"Tool handler `{tool_name}` must have a docstring: "
            "the docstring describes the tool to the LLM."
        )

    func.__contextgem_tool__ = True  # ty: ignore[invalid-assignment]
    func.__contextgem_tool_name__ = tool_name  # ty: ignore[invalid-assignment]

    return func
