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

from contextgem.internal.base import (
    _AssignedAspectsProcessor,
    _AssignedConceptsProcessor,
    _AssignedInstancesProcessor,
    _Concept,
    _ExtractedItem,
    _ExtractedItemsAttributeProcessor,
    _InstanceBase,
    _MarkdownTextAttributesProcessor,
    _ParasAndSentsBase,
    _PostInitCollectorMixin,
    _RefParasAndSentsAttrituteProcessor,
)
from contextgem.internal.converters import (
    DocxConverterError,
    _DocxConverterBase,
    _DocxPackage,
)
from contextgem.internal.data_models import (
    _LLMCall,
    _LLMCost,
    _LLMCostOutputContainer,
    _LLMUsage,
    _LLMUsageOutputContainer,
)
from contextgem.internal.decorators import _post_init_method, _timer_decorator
from contextgem.internal.items import (
    _BooleanItem,
    _DateItem,
    _FloatItem,
    _IntegerItem,
    _IntegerOrFloatItem,
    _JsonObjectItem,
    _LabelItem,
    _StringItem,
)
from contextgem.internal.llm_output_structs import (
    _get_aspect_extraction_output_struct,
    _get_concept_extraction_output_struct,
    _LabelConceptItemValueModel,
)
from contextgem.internal.loggers import logger
from contextgem.internal.typings import (
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
    _deserialize_type_hint,
    _dynamic_pydantic_model,
    _format_dict_structure,
    _format_type,
    _get_model_fields,
    _is_json_serializable_type,
    _is_typed_class,
    _JsonObjectItemStructure,
    _normalize_type_annotation,
    _raise_dict_class_type_error,
    _raise_json_serializable_type_error,
    _serialize_type_hint,
)
from contextgem.internal.utils import (
    _async_multi_executor,
    _chunk_list,
    _clean_control_characters,
    _clean_text_for_llm_prompt,
    _contains_linebreaks,
    _get_template,
    _group_instances_by_fields,
    _is_json_serializable,
    _is_text_content_empty,
    _llm_call_result_is_valid,
    _load_sat_model,
    _parse_llm_output_as_json,
    _remove_thinking_content_from_llm_output,
    _run_async_calls,
    _run_sync,
    _setup_jinja2_template,
    _split_text_into_paragraphs,
    _validate_parsed_llm_output,
)

__all__ = [
    # Base
    "_InstanceBase",
    "_AssignedAspectsProcessor",
    "_AssignedConceptsProcessor",
    "_AssignedInstancesProcessor",
    "_ExtractedItemsAttributeProcessor",
    "_RefParasAndSentsAttrituteProcessor",
    "_PostInitCollectorMixin",
    "_Concept",
    "_ExtractedItem",
    "_ParasAndSentsBase",
    "_MarkdownTextAttributesProcessor",
    # LLM output structs
    "_get_aspect_extraction_output_struct",
    "_get_concept_extraction_output_struct",
    "_LabelConceptItemValueModel",
    # Typings
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
    "_deserialize_type_hint",
    "_is_json_serializable_type",
    "_format_type",
    "_JsonObjectItemStructure",
    "_serialize_type_hint",
    "_raise_json_serializable_type_error",
    "_dynamic_pydantic_model",
    "_format_dict_structure",
    "_is_typed_class",
    "_get_model_fields",
    "_raise_dict_class_type_error",
    "_normalize_type_annotation",
    # Data models
    "_LLMCall",
    "_LLMUsage",
    "_LLMUsageOutputContainer",
    "_LLMCost",
    "_LLMCostOutputContainer",
    # Decorators
    "_post_init_method",
    "_timer_decorator",
    # Extracted items
    "_StringItem",
    "_IntegerItem",
    "_FloatItem",
    "_IntegerOrFloatItem",
    "_BooleanItem",
    "_JsonObjectItem",
    "_DateItem",
    "_LabelItem",
    # Logging
    "logger",
    # Utils
    "_get_template",
    "_contains_linebreaks",
    "_split_text_into_paragraphs",
    "_chunk_list",
    "_async_multi_executor",
    "_run_async_calls",
    "_run_sync",
    "_llm_call_result_is_valid",
    "_parse_llm_output_as_json",
    "_validate_parsed_llm_output",
    "_group_instances_by_fields",
    "_is_json_serializable",
    "_load_sat_model",
    "_setup_jinja2_template",
    "_remove_thinking_content_from_llm_output",
    "_clean_control_characters",
    "_is_text_content_empty",
    "_clean_text_for_llm_prompt",
    # Converters
    # DOCX
    "DocxConverterError",
    "_DocxConverterBase",
    "_DocxPackage",
]
