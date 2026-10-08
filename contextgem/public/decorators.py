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
Public decorators for extending or integrating with the framework.

This module contains decorators that are part of the public API and intended
for end users to apply to their own functions or classes.
"""

from __future__ import annotations

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

    Pass the decorated function to ``tools`` of the LLM.

    :param func: A function to be used as a tool. Can be sync or async, and must
        return a non-empty string.
    :type func: ToolHandler
    :return: The same function, marked as a registered tool.
    :rtype: ToolHandler
    :raises TypeError: If the provided object is not callable, or if a parameter
        of the function has no type hint.
    :raises ValueError: If the signature cannot be inspected, if the function has
        positional-only parameters or accepts ``*args`` or ``**kwargs``, if it has
        no docstring, or if the function name is empty.
    """

    if not callable(func):
        raise TypeError("Tool handler must be callable")
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
