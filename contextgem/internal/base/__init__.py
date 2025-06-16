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

from contextgem.internal.base.attrs import (
    _AssignedAspectsProcessor,
    _AssignedConceptsProcessor,
    _AssignedInstancesProcessor,
    _ExtractedItemsAttributeProcessor,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.base.concepts import _Concept
from contextgem.internal.base.instances import _InstanceBase
from contextgem.internal.base.items import _ExtractedItem
from contextgem.internal.base.md_text import _MarkdownTextAttributesProcessor
from contextgem.internal.base.mixins import _PostInitCollectorMixin
from contextgem.internal.base.paras_and_sents import _ParasAndSentsBase

__all__ = [
    # Instances
    "_InstanceBase",
    # Attrs processors
    "_AssignedAspectsProcessor",
    "_AssignedConceptsProcessor",
    "_AssignedInstancesProcessor",
    "_ExtractedItemsAttributeProcessor",
    "_RefParasAndSentsAttrituteProcessor",
    # Mixins
    "_PostInitCollectorMixin",
    # Concepts
    "_Concept",
    # Extracted items
    "_ExtractedItem",
    # Paragraphs and sentences
    "_ParasAndSentsBase",
    # Markdown text
    "_MarkdownTextAttributesProcessor",
]
