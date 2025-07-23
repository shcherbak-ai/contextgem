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

from contextgem.internal.typings.aliases import (
    AssignedInstancesAttrName,
    AsyncCalsAndKwargs,
    DefaultDecimalField,
    DefaultPromptType,
    ExtractedInstanceType,
    JustificationDepth,
    LanguageRequirement,
    LLMRoleAny,
    LLMRoleAspect,
    NonEmptyStr,
    ReasoningEffort,
    ReferenceDepth,
    SaTModelId,
    StandardSaTModelId,
    TextMode,
)
from contextgem.internal.typings.strings_to_types import _deserialize_type_hint
from contextgem.internal.typings.typed_class_utils import (
    _get_model_fields,
    _is_typed_class,
    _raise_dict_class_type_error,
)
from contextgem.internal.typings.types_normalization import _normalize_type_annotation
from contextgem.internal.typings.types_to_strings import (
    _format_dict_structure,
    _format_type,
    _is_json_serializable_type,
    _JsonObjectItemStructure,
    _raise_json_serializable_type_error,
    _serialize_type_hint,
)
from contextgem.internal.typings.user_type_hints_validation import (
    _dynamic_pydantic_model,
)
from contextgem.internal.typings.validators import _validate_sequence_is_list


__all__ = [
    # Aliases
    "NonEmptyStr",
    "LLMRoleAny",
    "LLMRoleAspect",
    "AssignedInstancesAttrName",
    "ExtractedInstanceType",
    "DefaultPromptType",
    "ReferenceDepth",
    "SaTModelId",
    "StandardSaTModelId",
    "LanguageRequirement",
    "JustificationDepth",
    "AsyncCalsAndKwargs",
    "DefaultDecimalField",
    "ReasoningEffort",
    "TextMode",
    # Strings to types
    "_deserialize_type_hint",
    # Types to strings
    "_is_json_serializable_type",
    "_format_type",
    "_JsonObjectItemStructure",
    "_serialize_type_hint",
    "_format_dict_structure",
    "_raise_json_serializable_type_error",
    # User type hints validation
    "_dynamic_pydantic_model",
    # Typed class utils
    "_is_typed_class",
    "_get_model_fields",
    "_raise_dict_class_type_error",
    # Types normalization
    "_normalize_type_annotation",
    # Validators
    "_validate_sequence_is_list",
]
