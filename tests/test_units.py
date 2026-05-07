#
# ContextGem
#
# Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
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
Example-based unit tests for pure internal helpers.

Targets utilities that transform Python types and structured data without
touching LLMs, the network, or any I/O. These tests pin specific
input/output contracts that property-based tests in
:mod:`tests.test_properties` are not well suited to express, including
error branches (which must raise the right exception with the right
message), JSON schema shape for tool calling, and edge cases of
type-hint serialization/normalization.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import (  # noqa: UP035
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    get_args,
)

import pytest

from contextgem.internal.tools.docstring_parser import _ParsedDocstring
from contextgem.internal.tools.schema_generator import (
    _class_to_json_schema,
    _contains_collection_type,
    _generate_tool_schema,
    _python_type_to_json_schema,
    _typeddict_to_json_schema,
)
from contextgem.internal.typings.strings_to_types import _deserialize_type_hint
from contextgem.internal.typings.types_normalization import _normalize_type_annotation
from contextgem.internal.typings.types_to_strings import (
    _is_json_serializable_type,
    _raise_json_serializable_type_error,
    _serialize_type_hint,
)


# ---------------------------------------------------------------------------
# schema_generator: _contains_collection_type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "type_hint,target,expected",
    [
        (set[int], (set,), True),
        (frozenset[str], (set, frozenset), True),
        (list[set[int]], (set,), True),  # nested
        (dict[str, tuple[int, ...]], (tuple,), True),  # nested in dict value
        (list[int], (set, frozenset), False),
        (str, (tuple,), False),
        (Optional[set[int]], (set,), True),  # noqa: UP045
    ],
)
def test_contains_collection_type(
    type_hint: Any, target: tuple[type, ...], expected: bool
) -> None:
    """Recursive detection of forbidden collection types in nested generics."""
    assert _contains_collection_type(type_hint, target) is expected


# ---------------------------------------------------------------------------
# schema_generator: _python_type_to_json_schema
# ---------------------------------------------------------------------------


def test_json_schema_primitive_with_description() -> None:
    """Primitives produce the mapped schema with description appended."""
    assert _python_type_to_json_schema(str, "a name") == {
        "type": "string",
        "description": "a name",
    }


def test_json_schema_any_type_returns_empty_schema() -> None:
    """``Any`` produces an empty schema (with description if provided)."""
    assert _python_type_to_json_schema(Any) == {}
    assert _python_type_to_json_schema(Any, "anything") == {"description": "anything"}


def test_json_schema_list_of_int() -> None:
    """``list[int]`` becomes an array schema with int items."""
    assert _python_type_to_json_schema(list[int]) == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_json_schema_bare_list_falls_through_to_class_fallback() -> None:
    """Bare ``list`` (no subscript) has no ``get_origin`` and lands in the class
    fallback branch, producing ``{"type": "object"}``. Pinned to detect future
    behavioural changes — use ``list[Any]`` to get an array schema instead."""
    assert _python_type_to_json_schema(list) == {"type": "object"}


def test_json_schema_variadic_tuple() -> None:
    """``tuple[int, ...]`` becomes an array of int."""
    assert _python_type_to_json_schema(tuple[int, ...]) == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_json_schema_fixed_length_tuple_uses_prefix_items() -> None:
    """``tuple[int, str, bool]`` becomes a prefix-typed fixed-length array."""
    assert _python_type_to_json_schema(tuple[int, str, bool]) == {
        "type": "array",
        "prefixItems": [
            {"type": "integer"},
            {"type": "string"},
            {"type": "boolean"},
        ],
        "minItems": 3,
        "maxItems": 3,
    }


def test_json_schema_bare_tuple_falls_through_to_class_fallback() -> None:
    """Bare ``tuple`` (no subscript) lands in the class fallback branch.

    Same caveat as bare ``list``: use ``tuple[X, ...]`` for an array schema.
    """
    assert _python_type_to_json_schema(tuple) == {"type": "object"}


def test_json_schema_set_has_unique_items_flag() -> None:
    """``set[int]`` becomes an array of int with ``uniqueItems``."""
    assert _python_type_to_json_schema(set[int]) == {
        "type": "array",
        "items": {"type": "integer"},
        "uniqueItems": True,
    }


def test_json_schema_frozenset_treated_like_set() -> None:
    """``frozenset[str]`` mirrors ``set[str]``."""
    assert _python_type_to_json_schema(frozenset[str]) == {
        "type": "array",
        "items": {"type": "string"},
        "uniqueItems": True,
    }


def test_json_schema_dict_with_str_int_args() -> None:
    """``dict[str, int]`` becomes object with int additionalProperties."""
    assert _python_type_to_json_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }


def test_json_schema_bare_dict_is_plain_object() -> None:
    """``dict`` without args becomes a plain object."""
    assert _python_type_to_json_schema(dict) == {"type": "object"}


def test_json_schema_optional_unwraps_to_inner_type() -> None:
    """``Optional[int]`` returns the schema for ``int`` (None handled at param level)."""
    assert _python_type_to_json_schema(Optional[int]) == {  # noqa: UP045
        "type": "integer"
    }


def test_json_schema_union_uses_anyof() -> None:
    """A union with multiple non-None types uses ``anyOf``."""
    assert _python_type_to_json_schema(Union[int, str]) == {  # noqa: UP007
        "anyOf": [{"type": "integer"}, {"type": "string"}]
    }


def test_json_schema_literal_strings_become_string_enum() -> None:
    """``Literal["a", "b"]`` becomes a string enum."""
    assert _python_type_to_json_schema(Literal["a", "b"]) == {
        "type": "string",
        "enum": ["a", "b"],
    }


def test_json_schema_literal_ints_become_integer_enum() -> None:
    """``Literal[1, 2, 3]`` becomes an integer enum."""
    assert _python_type_to_json_schema(Literal[1, 2, 3]) == {
        "type": "integer",
        "enum": [1, 2, 3],
    }


def test_json_schema_literal_booleans_become_boolean_enum() -> None:
    """``Literal[True, False]`` becomes a boolean enum.

    Note: ``Literal`` deduplicates ``True``/``False``-equivalent ints, so this
    pins the boolean-detection branch via two distinct booleans.
    """
    schema = _python_type_to_json_schema(Literal[True, False])
    assert schema["type"] == "boolean"
    assert set(schema["enum"]) == {True, False}


def test_json_schema_unsupported_type_raises_typeerror() -> None:
    """A non-type, non-typing-construct value raises ``TypeError``.

    Note: built-in classes like ``complex`` are accepted by the class-fallback
    branch (returning ``{"type": "object"}``). To trigger the unsupported-type
    error we pass a value that is not a type, not a typing construct, and not
    in the primitive map — here, a plain string sentinel.
    """
    with pytest.raises(TypeError, match="Unsupported type hint"):
        _python_type_to_json_schema("not a type")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# schema_generator: _typeddict_to_json_schema
# ---------------------------------------------------------------------------


class _AllRequiredTD(TypedDict):
    """All keys required (default TypedDict behavior)."""

    name: str
    age: int


class _AllOptionalTD(TypedDict, total=False):
    """All keys optional via ``total=False``."""

    name: str
    nickname: str


def test_typeddict_all_required_keys_listed() -> None:
    """All annotated keys are listed as required in alphabetic order."""
    schema = _typeddict_to_json_schema(_AllRequiredTD)
    assert schema["type"] == "object"
    assert schema["required"] == ["age", "name"]
    assert set(schema["properties"]) == {"age", "name"}


def test_typeddict_total_false_omits_required_field() -> None:
    """A ``total=False`` TypedDict has no required keys, so 'required' is omitted."""
    schema = _typeddict_to_json_schema(_AllOptionalTD)
    assert schema["type"] == "object"
    assert "required" not in schema
    assert set(schema["properties"]) == {"name", "nickname"}


# ---------------------------------------------------------------------------
# schema_generator: _class_to_json_schema
# ---------------------------------------------------------------------------


@dataclass
class _PointDC:
    """A point with x and y."""

    x: int
    y: int


def test_class_to_json_schema_dataclass() -> None:
    """A dataclass is converted to an object schema with all fields required."""
    schema = _class_to_json_schema(_PointDC)
    assert schema == {
        "type": "object",
        "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        "required": ["x", "y"],
    }


# ---------------------------------------------------------------------------
# schema_generator: _generate_tool_schema (end-to-end)
# ---------------------------------------------------------------------------


def test_generate_tool_schema_basic_function() -> None:
    """End-to-end shape for a simple function with two parameters and a default."""

    def my_tool(name: str, count: int = 0) -> str:
        """Do a thing.

        :param name: the name
        :param count: how many times
        """
        return ""

    docstring = _ParsedDocstring(
        summary="Do a thing.",
        params={"name": "the name", "count": "how many times"},
    )
    schema = _generate_tool_schema(my_tool, docstring)

    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "my_tool"
    assert fn["description"] == "Do a thing."
    params = fn["parameters"]
    assert params["type"] == "object"
    assert params["properties"]["name"] == {
        "type": "string",
        "description": "the name",
    }
    assert params["properties"]["count"] == {
        "type": "integer",
        "description": "how many times",
    }
    assert params["required"] == ["name"]  # count has a default → optional


def test_generate_tool_schema_concatenates_summary_and_description() -> None:
    """Summary + description are joined with a blank line in the tool description."""

    def f(x: int) -> int:
        """Sample tool."""
        return x

    docstring = _ParsedDocstring(summary="Short.", description="Longer text.")
    schema = _generate_tool_schema(f, docstring)
    assert schema["function"]["description"] == "Short.\n\nLonger text."


def test_generate_tool_schema_set_param_emits_warning() -> None:
    """A ``set`` parameter triggers a UserWarning about JSON list runtime behaviour."""

    def f(items: set[int]) -> int:
        """Sample tool with a set parameter."""
        return len(items)

    docstring = _ParsedDocstring(summary="…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _generate_tool_schema(f, docstring)
    messages = [str(w.message) for w in caught]
    assert any("'set' type hint" in m for m in messages)


def test_generate_tool_schema_tuple_param_emits_warning() -> None:
    """A ``tuple`` parameter triggers a distinct UserWarning."""

    def f(values: tuple[int, ...]) -> int:
        """Sample tool with a tuple parameter."""
        return sum(values)

    docstring = _ParsedDocstring(summary="…")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _generate_tool_schema(f, docstring)
    messages = [str(w.message) for w in caught]
    assert any("'tuple' type hint" in m for m in messages)


def test_generate_tool_schema_missing_annotation_raises_typeerror() -> None:
    """A parameter without a type annotation raises ``TypeError``."""

    def f(x):  # type: ignore[no-untyped-def]
        """Sample tool missing an annotation."""
        return x

    with pytest.raises(TypeError, match="no type annotation"):
        _generate_tool_schema(f, _ParsedDocstring(summary="…"))


def test_generate_tool_schema_optional_via_union_with_none_marks_param_optional() -> (
    None
):
    """``Optional[T]`` parameter is excluded from the required list even without default."""

    def f(maybe: Optional[int]) -> int:  # noqa: UP045
        """Sample tool with an optional parameter."""
        return maybe or 0

    schema = _generate_tool_schema(f, _ParsedDocstring(summary="…"))
    # When the only param is Optional, ``required`` is omitted entirely.
    assert "maybe" not in schema["function"]["parameters"].get("required", [])


def test_generate_tool_schema_skips_var_args_and_kwargs() -> None:
    """``*args`` and ``**kwargs`` are skipped silently and never appear in properties."""

    def f(name: str, *args: int, **kwargs: str) -> str:
        """Sample tool with var-positional and var-keyword args."""
        return name

    schema = _generate_tool_schema(f, _ParsedDocstring(summary="…"))
    properties = schema["function"]["parameters"]["properties"]
    assert "name" in properties
    assert "args" not in properties and "kwargs" not in properties


# ---------------------------------------------------------------------------
# types_normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("primitive", [str, int, float, bool, type(None)])
def test_normalize_primitives_pass_through(primitive: type) -> None:
    """Built-in primitives normalize to themselves."""
    assert _normalize_type_annotation(primitive) is primitive


def test_normalize_bare_typing_list_becomes_builtin_list() -> None:
    """``typing.List`` (no args) maps to built-in ``list``."""
    assert _normalize_type_annotation(List) is list  # noqa: UP006


def test_normalize_bare_typing_dict_becomes_builtin_dict() -> None:
    """``typing.Dict`` (no args) maps to built-in ``dict``."""
    assert _normalize_type_annotation(Dict) is dict  # noqa: UP006


def test_normalize_typing_list_str_becomes_builtin_list_str() -> None:
    """``typing.List[str]`` normalizes to ``list[str]``."""
    assert _normalize_type_annotation(List[str]) == list[str]  # noqa: UP006


def test_normalize_typing_dict_str_int_becomes_builtin_dict() -> None:
    """``typing.Dict[str, int]`` normalizes to ``dict[str, int]``."""
    assert _normalize_type_annotation(Dict[str, int]) == dict[str, int]  # noqa: UP006


def test_normalize_optional_int_becomes_int_or_none() -> None:
    """``Optional[int]`` normalizes to a union with ``None``."""
    normalized = _normalize_type_annotation(Optional[int])  # noqa: UP045
    # Union[int, None] and int | None compare equal in Python 3.10+.
    assert normalized == (int | None)


def test_normalize_union_pass_through() -> None:
    """``Union[int, str]`` normalizes to an equivalent union."""
    normalized = _normalize_type_annotation(Union[int, str])  # noqa: UP007
    assert normalized == (int | str)


def test_normalize_rejects_tuple_type() -> None:
    """Tuple types are explicitly rejected as non-JSON-serializable."""
    with pytest.raises(ValueError, match="not JSON-serializable"):
        _normalize_type_annotation(tuple[int, ...])


def test_normalize_rejects_set_type() -> None:
    """Set types are explicitly rejected as non-JSON-serializable."""
    with pytest.raises(ValueError, match="not JSON-serializable"):
        _normalize_type_annotation(set[int])


def test_normalize_idempotent_for_complex_nested_type() -> None:
    """Running normalization twice yields the same result as once."""
    original = Dict[str, List[Optional[int]]]  # noqa: UP006, UP045
    once = _normalize_type_annotation(original)
    twice = _normalize_type_annotation(once)
    assert once == twice


# ---------------------------------------------------------------------------
# types_to_strings: error branches and edge cases
# ---------------------------------------------------------------------------


def test_serialize_type_hint_unsupported_type_raises() -> None:
    """An unsupported root type raises ``ValueError`` with a guidance message."""
    with pytest.raises(ValueError, match="Unsupported type hint"):
        _serialize_type_hint(complex)


def test_serialize_type_hint_list_with_zero_args_raises() -> None:
    """``list[X]`` with zero type args (constructed via origin manipulation) raises.

    ``get_origin(list)`` is ``None``, so we can only trigger the ``len(args) != 1``
    path through generic forms. Construct via __class_getitem__ with no args.
    """
    # Constructing list[()] gives a tuple of zero type args, which the origin/args
    # machinery treats as "list has zero args".
    bad = list[()]  # type: ignore[misc]  # ty: ignore[invalid-type-arguments]
    with pytest.raises(ValueError, match="List must have one type argument"):
        _serialize_type_hint(bad)


def test_serialize_type_hint_dict_with_one_arg_raises() -> None:
    """``dict[K]`` with only one type arg raises ``ValueError``."""
    bad = dict[(str,)]  # type: ignore[misc]  # ty: ignore[invalid-type-arguments]
    with pytest.raises(ValueError, match="Dict must have two type arguments"):
        _serialize_type_hint(bad)


def test_serialize_optional_with_one_non_none_arg() -> None:
    """``Optional[T]`` serializes to ``optional[T]``."""
    assert _serialize_type_hint(Optional[int]) == "optional[int]"  # noqa: UP045


def test_serialize_optional_union_serializes_with_inner_union() -> None:
    """``Optional[int | str]`` serializes to ``optional[union[...]]``."""
    serialized = _serialize_type_hint(Optional[Union[int, str]])  # noqa: UP007, UP045
    # Union members are sorted alphabetically inside the optional.
    assert serialized == "optional[union[int, str]]"


def test_serialize_literal_with_quoted_strings() -> None:
    """String literals are double-quoted with internal quotes escaped."""
    serialized = _serialize_type_hint(Literal["a", 'has"quote'])
    assert serialized == 'literal["a", "has\\"quote"]'


def test_is_json_serializable_type_for_primitives() -> None:
    """Each JSON primitive type is recognised."""
    for primitive in (str, int, float, bool, type(None)):
        assert _is_json_serializable_type(primitive) is True


def test_is_json_serializable_type_for_unsupported() -> None:
    """``complex`` (and similar non-JSON types) are rejected."""
    assert _is_json_serializable_type(complex) is False


def test_is_json_serializable_type_for_dict_of_primitives() -> None:
    """A dict-of-primitives schema definition is JSON-serializable."""
    assert _is_json_serializable_type({"name": str, "age": int}) is True


def test_is_json_serializable_type_for_list_of_dict_structure() -> None:
    """A ``[{"k": int}]`` list-of-dict structure is JSON-serializable."""
    assert _is_json_serializable_type([{"k": int}]) is True


def test_is_json_serializable_type_for_union() -> None:
    """A union of JSON-serializable types is JSON-serializable."""
    assert _is_json_serializable_type(Union[int, str]) is True  # noqa: UP007


def test_is_json_serializable_type_literal_with_non_primitive_rejected() -> None:
    """``Literal`` containing non-JSON values is rejected."""
    # Literal of bytes (not a JSON primitive in the project's strict sense)
    assert _is_json_serializable_type(Literal[b"x"]) is False


def test_raise_json_serializable_type_error_with_field_and_context() -> None:
    """Both ``field_name`` and ``context`` produce a combined error message."""
    with pytest.raises(TypeError, match="Invalid item type for field 'tags':"):
        _raise_json_serializable_type_error(
            complex, field_name="tags", context="item type"
        )


def test_raise_json_serializable_type_error_with_only_context() -> None:
    """Only ``context`` produces a context-prefixed message."""
    with pytest.raises(TypeError, match="Invalid root type:"):
        _raise_json_serializable_type_error(complex, context="root type")


def test_raise_json_serializable_type_error_default_message() -> None:
    """No field/context produces the bare 'Invalid type:' message."""
    with pytest.raises(TypeError, match=r"Invalid type:"):
        _raise_json_serializable_type_error(complex)


def test_raise_json_serializable_type_error_list_instance_message() -> None:
    """A 'list instance' context appends the list-specific guidance."""
    with pytest.raises(TypeError, match="List instances can only contain"):
        _raise_json_serializable_type_error(
            complex, context="item type in list instance"
        )


def test_raise_json_serializable_type_error_custom_exception_type() -> None:
    """``exception_type`` overrides the raised exception class."""
    with pytest.raises(ValueError, match="Invalid type:"):
        _raise_json_serializable_type_error(complex, exception_type=ValueError)


# ---------------------------------------------------------------------------
# strings_to_types: error branches
# ---------------------------------------------------------------------------


def test_deserialize_type_hint_extra_chars_after_valid_prefix_raises() -> None:
    """Trailing garbage after a valid type hint raises ``ValueError``."""
    with pytest.raises(ValueError, match="Extra characters"):
        _deserialize_type_hint("int garbage")


def test_deserialize_type_hint_unknown_identifier_raises() -> None:
    """An unknown generic identifier (with brackets) raises ``ValueError``.

    The "Unknown type identifier" branch fires only after the parser has
    consumed an opening bracket — bare unknown names trip the
    "Expected '['" path instead.
    """
    with pytest.raises(ValueError, match="Unknown type identifier"):
        _deserialize_type_hint("unknown_name[int]")


def test_deserialize_type_hint_missing_open_bracket_for_generic_raises() -> None:
    """A generic name without ``[`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Expected '\\['"):
        _deserialize_type_hint("list")


def test_deserialize_type_hint_missing_close_bracket_for_list_raises() -> None:
    """A list type missing ``]`` raises ``ValueError``."""
    with pytest.raises(ValueError, match="Expected '\\]'"):
        _deserialize_type_hint("list[int")


def test_deserialize_type_hint_missing_comma_in_dict_raises() -> None:
    """A dict type missing the comma between key/value raises ``ValueError``."""
    with pytest.raises(ValueError, match="Expected ','"):
        _deserialize_type_hint("dict[int int]")


def test_deserialize_type_hint_empty_literal_raises() -> None:
    """``literal[]`` (no values) raises ``ValueError``."""
    with pytest.raises(ValueError, match="Expected literal value"):
        _deserialize_type_hint("literal[]")


def test_deserialize_type_hint_unterminated_string_literal_raises() -> None:
    """A string literal missing its closing quote raises ``ValueError``."""
    with pytest.raises(ValueError, match="Unterminated string literal"):
        _deserialize_type_hint('literal["unterminated]')


def test_deserialize_type_hint_handles_new_union_syntax() -> None:
    """Pipe-style ``int | str`` is accepted via preprocessing."""
    restored = _deserialize_type_hint("int | str")
    # Equivalent to typing.Union[int, str] for equality.
    assert restored == (int | str)


def test_deserialize_type_hint_handles_optional_via_pipe_with_none() -> None:
    """``int | None`` is preprocessed into ``optional[int]``."""
    restored = _deserialize_type_hint("int | None")
    assert restored == (int | None)


def test_deserialize_type_hint_handles_three_arg_union() -> None:
    """A three-arg union builds via ``|`` reduction."""
    restored = _deserialize_type_hint("union[int, str, bool]")
    assert restored == (int | str | bool)


def test_deserialize_type_hint_literal_with_booleans_and_null() -> None:
    """Literal parsing handles ``true``, ``false``, ``null``, ints and floats."""
    restored = _deserialize_type_hint("literal[true, false, null, 42]")
    assert restored == Literal[True, False, None, 42]
    # Float literal values are accepted at runtime (parser converts via float()),
    # but typing.Literal does not statically accept floats. Compare via get_args.
    restored_with_float = _deserialize_type_hint("literal[1.5]")
    assert get_args(restored_with_float) == (1.5,)
