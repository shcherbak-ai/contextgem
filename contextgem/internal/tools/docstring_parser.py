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
Module for parsing docstrings in multiple formats (Google, NumPy, Sphinx/reST).

Provides a thin wrapper around the ``docstring_parser`` package to extract
summary, description, and parameter descriptions from function docstrings
for use in auto-generating tool schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import docstring_parser


@dataclass
class _ParsedDocstring:
    """
    Parsed docstring data extracted from a function's docstring.

    :ivar summary: The first line of the docstring (short description).
    :vartype summary: str
    :ivar description: Extended description after the first line (may be empty).
    :vartype description: str
    :ivar params: Mapping of parameter names to their descriptions.
    :vartype params: dict[str, str]
    """

    summary: str = ""
    description: str = ""
    params: dict[str, str] = field(default_factory=dict)


def _parse_docstring(docstring: str | None) -> _ParsedDocstring:
    """
    Parse a docstring and extract summary, description, and parameter descriptions.

    Uses the ``docstring_parser`` package which supports multiple formats:
    - Sphinx/reST: ``:param name: description``
    - Google style: ``Args:`` section
    - NumPy style: ``Parameters`` section with underlines
    - Epydoc style

    :param docstring: The raw docstring to parse.
    :type docstring: str | None
    :returns: Parsed docstring data with summary, description, and params.
    :rtype: _ParsedDocstring
    """
    if not docstring:
        return _ParsedDocstring()

    parsed = docstring_parser.parse(docstring)

    # Extract parameter descriptions
    params: dict[str, str] = {}
    for param in parsed.params:
        if param.arg_name and param.description:
            params[param.arg_name] = param.description

    return _ParsedDocstring(
        summary=parsed.short_description or "",
        description=parsed.long_description or "",
        params=params,
    )
