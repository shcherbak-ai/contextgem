# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Internal tools module for auto-generating tool schemas from Python functions.
"""

from contextgem.internal.tools.docstring_parser import (
    _parse_docstring,
)
from contextgem.internal.tools.schema_generator import (
    _generate_tool_schema,
)


__all__ = (
    "_parse_docstring",
    "_generate_tool_schema",
)
