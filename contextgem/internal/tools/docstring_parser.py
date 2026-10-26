# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

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

    :ivar summary: The first paragraph of the docstring (short description), which
        can span several lines.
    :vartype summary: str
    :ivar description: Extended description after the first paragraph (may be empty).
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

    summary = parsed.short_description or ""
    description = parsed.long_description or ""
    if description and not parsed.blank_after_short_description:
        # `docstring_parser` takes the first line as the short description, also
        # when the first sentence goes on in the next line: the rest of the first
        # paragraph is kept with it, so that the sentence is not split by the blank
        # line that the tool description has between the two
        continuation, _, description = description.partition("\n\n")
        summary = f"{summary}\n{continuation}"
        description = description.strip()

    return _ParsedDocstring(summary=summary, description=description, params=params)
