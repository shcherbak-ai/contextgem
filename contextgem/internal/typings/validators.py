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
Module defining custom validators.
"""

from __future__ import annotations

from collections.abc import Sequence


def _validate_sequence_is_list(
    v: Sequence,
) -> list:
    """
    Validator function to ensure that the sequence value is a list.
    Necessary for type compliance of Sequence fields.

    Can be used either as a field validator (e.g. in a pydantic field -
    Annotated[Sequence[_Concept], BeforeValidator(_validate_sequence_is_list)])
    or as a standalone function.

    :param v: The sequence value to validate.
    :return: The validated list.
    :raises ValueError: If the sequence is not a list.
    """
    if not isinstance(v, list):
        raise ValueError("Sequence must be a list")
    return v
