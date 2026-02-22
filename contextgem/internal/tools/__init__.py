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
