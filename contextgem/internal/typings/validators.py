# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining custom validators.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import langcodes


if TYPE_CHECKING:
    from contextgem.internal.typings.types import JSONDict


# Maximum number of characters in a model role
_MAX_MODEL_ROLE_LENGTH = 64

# The output language that stands for the language of the document
_ADAPT_OUTPUT_LANGUAGE = "adapt"

# The special codes of ISO 639, which name no language of their own: languages
# that have no code, several languages, an undetermined one, and no linguistic content
_SPECIAL_LANGUAGE_CODES = frozenset({"mis", "mul", "und", "zxx"})

# The script codes of ISO 15924 that name no script: unwritten and uncoded
_SPECIAL_SCRIPT_CODES = frozenset({"Zxxx", "Zzzz"})

# The region subtags of BCP 47 that name no region: unknown, and private use
# outside the ranges below
_SPECIAL_REGION_CODES = frozenset({"AA", "ZZ"})


def _validate_model_role(v: str) -> str:
    """
    Validator function for a model role, i.e. a free-form label that routes
    aspects/concepts to a model in a model group. Ensures that the role is a non-empty
    string of a limited length that consists of printable characters only, so that
    it remains readable in error messages, logs, and usage data.

    Can be used either as a field validator (e.g. in a pydantic field -
    Annotated[StrictStr, AfterValidator(_validate_model_role)]) or as a standalone
    function, e.g. to validate a model role passed as an argument.

    :param v: The model role to validate.
    :type v: str
    :return: The validated model role, with the surrounding whitespace removed.
    :rtype: str
    :raises ValueError: If the model role is not a string, is empty, exceeds the maximum
        length, or contains non-printable characters.
    """
    if not isinstance(v, str):
        raise ValueError("Model role must be a string")  # noqa: TRY004 - pydantic and the callers handle ValueError
    v = v.strip()
    if not v:
        raise ValueError("Model role must be a non-empty string")
    if len(v) > _MAX_MODEL_ROLE_LENGTH:
        raise ValueError(
            f"Model role must not be longer than {_MAX_MODEL_ROLE_LENGTH} characters"
        )
    if not v.isprintable():
        raise ValueError(
            "Model role must consist of printable characters only, i.e. it must not "
            "contain line breaks, tabs, or other control or invisible characters"
        )
    return v


def _is_output_language_tag(tag: str) -> bool:
    """
    Returns True if the tag is a valid BCP 47 language tag that names a language,
    i.e. each of its subtags is assigned in the IANA Language Subtag Registry, and
    none of them is for private use, undetermined or special (e.g. "qaa", "und",
    "mul", "en-x-legal" or the region "ZZ"). A tag with an extension (e.g.
    "de-u-co-phonebk") is not accepted either, as it adds nothing that the output
    is written in.

    :param tag: The language tag, in its standard form.
    :type tag: str
    :return: True if the tag names a language, False otherwise.
    :rtype: bool
    """
    if not langcodes.tag_is_valid(tag):
        return False
    language = langcodes.Language.get(tag)
    if language.extensions or language.private:
        return False
    code = language.language
    if (
        code is None
        or code.startswith("x-")
        or code in _SPECIAL_LANGUAGE_CODES
        or (len(code) == 3 and "qaa" <= code <= "qtz")  # private use
    ):
        return False
    script = language.script
    if script is not None and (
        script in _SPECIAL_SCRIPT_CODES or "Qaaa" <= script <= "Qabx"  # private use
    ):
        return False
    region = language.territory
    return region is None or not (
        region in _SPECIAL_REGION_CODES
        or "QM" <= region <= "QZ"  # private use
        or "XA" <= region <= "XZ"  # private use
    )


def _output_language_hint(value: str) -> str:
    """
    Returns a hint at the tag that an output language that is not a valid tag
    likely stands for: the tag of the language that it names (e.g. "ja" for
    "Japanese"), or the tag of the main language of the region whose code it is
    (e.g. "ja" for "jp", the code of Japan).

    :param value: The output language that is not a valid tag.
    :type value: str
    :return: The hint, starting with a space, or an empty string if there is none.
    :rtype: str
    """
    with contextlib.suppress(LookupError, ValueError):
        tag = langcodes.find(value).to_tag()
        if _is_output_language_tag(tag):
            return f" {value!r} is the name of a language, whose tag is {tag!r}."
    if len(value) == 2 and value.isascii() and value.isalpha():
        region = f"und-{value.upper()}"
        if _is_output_language_tag(f"en-{value.upper()}"):
            main_language = langcodes.Language.get(region).maximize().language
            if main_language and main_language != value.lower():
                return (
                    f" {value!r} is the code of a region, "
                    f"{langcodes.Language.get(region).territory_name()}, whose main "
                    f"language is {langcodes.Language.get(main_language).display_name()}, "
                    f"with the tag {main_language!r}."
                )
    return ""


def _validate_output_language(v: str) -> str:
    """
    Validator function for the output language of an LLM: "adapt", or a BCP 47
    language tag that names a language, e.g. "en", "ja", "zh-Hant" or "pt-BR"
    (see ``_is_output_language_tag``). The tag is returned in its standard form,
    in which a tag of a language is always written the same way, so that output
    languages compare by their tags: e.g. "en-GB" for "en_gb" or the deprecated
    "en-uk", "he" for "iw", and "en" for "eng".

    Can be used either as a field validator (e.g. in a pydantic field -
    Annotated[StrictStr, AfterValidator(_validate_output_language)]) or as a
    standalone function.

    :param v: The output language to validate.
    :type v: str
    :return: "adapt", or the tag in its standard form.
    :rtype: str
    :raises ValueError: If the output language is not a string, or is neither
        "adapt" nor a valid tag of a language. The message hints at the tag that
        a language name or a region code likely stands for.
    """
    if not isinstance(v, str):
        raise ValueError("Output language must be a string")  # noqa: TRY004 - pydantic and the callers handle ValueError
    v = v.strip()
    if v.lower() == _ADAPT_OUTPUT_LANGUAGE:
        return _ADAPT_OUTPUT_LANGUAGE
    try:
        tag = langcodes.standardize_tag(v)
    except ValueError:
        tag = None
    if tag is None or not _is_output_language_tag(tag):
        raise ValueError(
            f'Output language must be "{_ADAPT_OUTPUT_LANGUAGE}" or a BCP 47 '
            f'language tag, e.g. "en", "ja", "zh-Hant" or "pt-BR", not {v!r}.'
            f"{_output_language_hint(v)}"
        )
    return tag


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
        raise ValueError("Sequence must be a list")  # noqa: TRY004 - pydantic and the callers handle ValueError
    return v


def _is_json_value(v: Any) -> bool:
    """
    Returns True if the value conforms to JSONValue
    (str|int|float|bool|None|list|dict[str,JSONValue]).

    :param v: The value to check.
    :type v: Any
    :return: True if the value conforms to JSONValue, False otherwise.
    :rtype: bool
    """
    if v is None or isinstance(v, str | int | float | bool):
        return True
    if isinstance(v, list):
        return all(_is_json_value(item) for item in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json_value(val) for k, val in v.items())
    return False


def _validate_is_json_dict(v: Any) -> JSONDict:
    """
    Validates that v is a dict[str, JSONValue]. Raises ValueError otherwise.

    :param v: The value to validate.
    :type v: Any
    :return: The validated value.
    :rtype: JSONDict
    :raises ValueError: If the value is not a dict or does not conform to JSONValue.
    """
    if not isinstance(v, dict):
        raise ValueError("Expected a JSON object (dict)")  # noqa: TRY004 - pydantic and the callers handle ValueError
    if not _is_json_value(v):
        raise ValueError("Schema must be JSON-serializable with string keys")
    # v matches JSONDict by construction
    return v
