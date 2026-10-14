# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

from contextgem.internal.typings.strings_to_types import _deserialize_type_hint
from contextgem.internal.typings.typed_class_utils import (
    _get_model_fields,
    _is_typed_class,
    _raise_dict_class_type_error,
)
from contextgem.internal.typings.types import (
    AssignedInstancesAttrName,
    AsyncCalsAndKwargs,
    DefaultDecimalField,
    DefaultPromptType,
    ExtractedInstanceType,
    ItemProbabilities,
    JSONDict,
    JSONDictField,
    JustificationDepth,
    LanguageRequirement,
    MessageRole,
    Modality,
    ModelRole,
    NonEmptyStr,
    Probability,
    ReasoningEffort,
    ReferenceDepth,
    SaTModelId,
    StandardSaTModelId,
    ToolHandler,
    ToolRegistration,
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
from contextgem.internal.typings.validators import (
    _validate_is_json_dict,
    _validate_model_role,
    _validate_sequence_is_list,
)


__all__ = (
    # Types
    "NonEmptyStr",
    "ModelRole",
    "Modality",
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
    "MessageRole",
    "Probability",
    "ItemProbabilities",
    "JSONDict",
    "JSONDictField",
    "ToolHandler",
    "ToolRegistration",
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
    "_validate_is_json_dict",
    "_validate_model_role",
)
