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
Module defining the base classes for example subclasses.

This module provides the foundational class structure for examples that can be used
in the ContextGem framework. Examples serve as user-provided samples for extraction tasks,
helping to guide and improve the extraction process by providing reference patterns
or expected outputs.
"""

from __future__ import annotations

from typing import Any

from contextgem.internal.base.instances import _InstanceBase


class _Example(_InstanceBase):
    """
    Base class that represents an example for extraction tasks in the ContextGem framework.

    Examples serve as user-provided samples that guide the extraction process by
    demonstrating expected patterns or outputs for specific extraction tasks.

    :ivar content: Arbitrary content associated with the example.
    :vartype content: Any
    """

    content: Any
