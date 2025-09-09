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
    Registers a function as a tool handler for LLM chat with tools.

    Validates that the function has an inspectable signature and accepts keyword
    arguments (no positional-only parameters). Marks the function so the runtime
    can recognize and call it by name.

    :param func: A callable to be used as a tool handler.
    :type func: ToolHandler
    :return: The same function, marked as a registered tool.
    :rtype: ToolHandler
    :raises TypeError: If the provided object is not callable.
    :raises ValueError: If the signature cannot be inspected or has
        positional-only parameters, or if the function name is empty.
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

    tool_name = getattr(func, "__name__", None) or ""
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise ValueError("Tool name must be a non-empty string")

    func.__contextgem_tool__ = True
    func.__contextgem_tool_name__ = tool_name

    return func
