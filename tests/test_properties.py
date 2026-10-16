# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Property-based tests for pure utility functions.

These tests use Hypothesis to throw varied inputs at deterministic helpers
in :mod:`contextgem.internal.utils` and assert invariants that must hold
for every input. They are independent of LLM cassettes, run quickly, and
target functions where regressions are most likely to slip past
example-based tests (unicode handling, edge-case whitespace, malformed
JSON, paragraph splitting).
"""

from __future__ import annotations

import json
import math
import sys
from types import SimpleNamespace
from typing import Any, Literal, Union

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from contextgem.internal.typings.strings_to_types import _deserialize_type_hint
from contextgem.internal.typings.types_to_strings import _serialize_type_hint
from contextgem.internal.utils import (
    _RE_CONTROL_CHARS_ALL,
    _are_paragraphs_in_text_in_order,
    _are_prompt_template_brackets_balanced,
    _are_prompt_template_xml_tags_balanced,
    _check_paragraphs_match_in_text,
    _check_paragraphs_ordering_in_text,
    _chunk_list,
    _clean_control_characters,
    _clean_text_for_llm_prompt,
    _get_position_from_prompt_id,
    _get_sentence_position_from_prompt_id,
    _group_instances_by_fields,
    _is_text_content_empty,
    _parse_llm_output_as_json,
    _remove_thinking_content_from_llm_output,
    _split_text_into_paragraphs,
)


# Suppress the "function-scoped fixture across multiple Hypothesis examples"
# health check: the autouse tethered network guard re-runs per Hypothesis
# example, which is intentional and harmless for these pure-function tests.
_PROPERTY_SETTINGS = settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

# All the characters that are whitespace, i.e. that separate words
_WHITESPACE_CHARS = [
    chr(code_point)
    for code_point in range(sys.maxunicode + 1)
    if chr(code_point).isspace()
]


# ---------------------------------------------------------------------------
# _clean_control_characters
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_control_characters_never_raises(text: str) -> None:
    """It must accept any string input and return a string."""
    assert isinstance(_clean_control_characters(text), str)
    assert isinstance(_clean_control_characters(text, preserve_newlines=False), str)


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_control_characters_idempotent(text: str) -> None:
    """Running the function twice produces the same output as once."""
    once = _clean_control_characters(text)
    twice = _clean_control_characters(once)
    assert once == twice

    once_no_nl = _clean_control_characters(text, preserve_newlines=False)
    twice_no_nl = _clean_control_characters(once_no_nl, preserve_newlines=False)
    assert once_no_nl == twice_no_nl


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_control_characters_no_newlines_when_disabled(text: str) -> None:
    """With preserve_newlines=False, output must contain no \\n or \\r."""
    cleaned = _clean_control_characters(text, preserve_newlines=False)
    assert "\n" not in cleaned
    assert "\r" not in cleaned


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_control_characters_output_no_longer_than_input(text: str) -> None:
    """Cleaning only removes characters; output cannot grow."""
    assert len(_clean_control_characters(text, strip_text=False)) <= len(text)


# ---------------------------------------------------------------------------
# _clean_text_for_llm_prompt
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(text=st.text())
@example(text="")
@example(text="   ")
@example(text="a\r\nb\rc\nd")
@example(text="a\n\n\n\n\nb")
@example(text="hello")
def test_clean_text_for_llm_prompt_idempotent(text: str) -> None:
    """f(f(x)) == f(x) under each combination of flags."""
    for preserve_linebreaks in (True, False):
        for strip_text in (True, False):
            once = _clean_text_for_llm_prompt(
                text,
                preserve_linebreaks=preserve_linebreaks,
                strip_text=strip_text,
            )
            twice = _clean_text_for_llm_prompt(
                once,
                preserve_linebreaks=preserve_linebreaks,
                strip_text=strip_text,
            )
            assert once == twice, (
                f"Not idempotent for preserve_linebreaks={preserve_linebreaks}, "
                f"strip_text={strip_text}: input={text!r}, once={once!r}, "
                f"twice={twice!r}"
            )


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_text_for_llm_prompt_no_carriage_return_when_preserving(
    text: str,
) -> None:
    """preserve_linebreaks=True normalizes \\r and \\r\\n to \\n."""
    cleaned = _clean_text_for_llm_prompt(text, preserve_linebreaks=True)
    assert "\r" not in cleaned


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_text_for_llm_prompt_no_triple_newline_when_preserving(
    text: str,
) -> None:
    """preserve_linebreaks=True collapses 3+ consecutive newlines to exactly 2."""
    cleaned = _clean_text_for_llm_prompt(text, preserve_linebreaks=True)
    assert "\n\n\n" not in cleaned


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_text_for_llm_prompt_no_newlines_when_disabled(text: str) -> None:
    """preserve_linebreaks=False removes all \\n and \\r."""
    cleaned = _clean_text_for_llm_prompt(text, preserve_linebreaks=False)
    assert "\n" not in cleaned
    assert "\r" not in cleaned


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_text_for_llm_prompt_no_double_whitespace_when_collapsed(
    text: str,
) -> None:
    """preserve_linebreaks=False collapses any \\s+ run to a single space."""
    cleaned = _clean_text_for_llm_prompt(text, preserve_linebreaks=False)
    assert "  " not in cleaned
    assert "\t" not in cleaned


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_clean_text_for_llm_prompt_strip_removes_edge_whitespace(text: str) -> None:
    """strip_text=True yields no leading/trailing whitespace, given non-empty output."""
    cleaned = _clean_text_for_llm_prompt(text, strip_text=True)
    if cleaned:
        assert cleaned == cleaned.strip()


@_PROPERTY_SETTINGS
@given(
    words_and_separators=st.lists(
        st.tuples(
            st.text(alphabet=st.characters(categories=["L", "N"]), min_size=1),
            st.text(alphabet=st.sampled_from(_WHITESPACE_CHARS), min_size=1),
        ),
        min_size=1,
        max_size=10,
    )
)
@example(words_and_separators=[("Fee", "\t"), ("5000", " "), ("EUR", "\n")])
@example(words_and_separators=[("line", "\n"), ("break", "\r\n"), ("page", "\x0c")])
def test_clean_text_for_llm_prompt_keeps_words_separated(
    words_and_separators: list[tuple[str, str]],
) -> None:
    """
    Words that whitespace separates stay separate words: whitespace that is
    a control character, e.g. a tab or a line break, is not removed where it is
    all that separates two words.
    """
    words = [word for word, _ in words_and_separators]
    text = "".join(word + separator for word, separator in words_and_separators)
    for preserve_linebreaks in (True, False):
        cleaned = _clean_text_for_llm_prompt(
            text, preserve_linebreaks=preserve_linebreaks
        )
        assert cleaned.split() == words, (
            f"Words joined or changed for preserve_linebreaks={preserve_linebreaks}: "
            f"input={text!r}, cleaned={cleaned!r}"
        )


@pytest.mark.parametrize(
    ("text", "with_linebreaks", "without_linebreaks"),
    [
        # A tab is replaced with a space
        ("Fee:\t5,000 EUR", "Fee: 5,000 EUR", "Fee: 5,000 EUR"),
        ("Fee:\t\t 5,000 EUR", "Fee: 5,000 EUR", "Fee: 5,000 EUR"),
        # A line break is kept, or replaced with a space
        ("line one\nline two", "line one\nline two", "line one line two"),
        ("line one\r\nline two", "line one\nline two", "line one line two"),
        ("line one \n line two", "line one \n line two", "line one line two"),
        # Other whitespace among the control characters is replaced with a space
        ("page one\x0cpage two", "page one page two", "page one page two"),
        ("line one\u2028line two", "line one line two", "line one line two"),
        ("5\u202f000 EUR", "5 000 EUR", "5 000 EUR"),
        # Whitespace that does not separate two words on its own is removed with
        # the control characters, as before
        ("Fee: \t5,000 EUR", "Fee: 5,000 EUR", "Fee: 5,000 EUR"),
        ("\tIndented\n\tline\t", "Indented\nline", "Indented line"),
        ("a\n\t\n\t\nb", "a\n\nb", "a b"),
        # ... also where a control character that is not whitespace is next to it
        ("a\t\u200b\nb", "a\nb", "a b"),
        ("a\x00\tb", "a b", "a b"),
        # Control characters that are not whitespace are removed
        ("zero\u200bwidth", "zerowidth", "zerowidth"),
        ("a \u200b b\x00c", "a bc", "a bc"),
    ],
)
def test_clean_text_for_llm_prompt_whitespace(
    text: str, with_linebreaks: str, without_linebreaks: str
) -> None:
    """
    Whitespace between two words is replaced with a space, or kept as a line
    break, and is not removed.
    """
    assert _clean_text_for_llm_prompt(text, preserve_linebreaks=True) == with_linebreaks
    assert (
        _clean_text_for_llm_prompt(text, preserve_linebreaks=False)
        == without_linebreaks
    )


def test_clean_text_for_llm_prompt_handles_each_removed_char() -> None:
    """
    Each whitespace character among the control characters that are removed from
    a text is replaced with a space between two words, or kept as a line break,
    and each of the other control characters is removed.
    """
    removed_chars = [
        chr(code_point)
        for code_point in range(sys.maxunicode + 1)
        if _RE_CONTROL_CHARS_ALL.fullmatch(chr(code_point))
    ]
    assert {"\t", "\n", "\r"} < {char for char in removed_chars if char.isspace()}
    for char in removed_chars:
        text = f"a{char}b"
        if not char.isspace():
            with_linebreaks = without_linebreaks = "ab"
        else:
            with_linebreaks = "a\nb" if char in "\r\n" else "a b"
            without_linebreaks = "a b"
        assert (
            _clean_text_for_llm_prompt(text, preserve_linebreaks=True)
            == with_linebreaks
        ), f"U+{ord(char):04X}"
        assert (
            _clean_text_for_llm_prompt(text, preserve_linebreaks=False)
            == without_linebreaks
        ), f"U+{ord(char):04X}"


# ---------------------------------------------------------------------------
# _is_text_content_empty
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_is_text_content_empty_never_raises(text: str) -> None:
    """The function returns a bool for any string input."""
    assert isinstance(_is_text_content_empty(text), bool)


@_PROPERTY_SETTINGS
@given(
    whitespace=st.text(
        alphabet=st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c"]),
        min_size=1,
        max_size=50,
    )
)
def test_is_text_content_empty_for_pure_whitespace(whitespace: str) -> None:
    """Whitespace-only strings are reported as empty."""
    assert _is_text_content_empty(whitespace) is True


@_PROPERTY_SETTINGS
@given(
    # At least one alphanumeric so the string definitely has content.
    prefix=st.text(),
    letter=st.sampled_from("abcdefghijABCDEF0123456789"),
    suffix=st.text(),
)
def test_is_text_content_empty_false_for_strings_with_alphanumeric(
    prefix: str, letter: str, suffix: str
) -> None:
    """Any string containing at least one alphanumeric is not empty."""
    text = prefix + letter + suffix
    assert _is_text_content_empty(text) is False


def test_is_text_content_empty_for_empty_string() -> None:
    """The empty string is reported as empty."""
    assert _is_text_content_empty("") is True


# ---------------------------------------------------------------------------
# _parse_llm_output_as_json
# ---------------------------------------------------------------------------


# JSON-serializable atomic values.
_JSON_ATOMS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000, max_value=1_000_000),
    st.floats(allow_nan=False, allow_infinity=False, width=32),
    st.text(max_size=20),
)

# Bounded JSON values (dicts, lists, atoms).
_JSON_VALUES = st.recursive(
    _JSON_ATOMS,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=10), children, max_size=4),
    ),
    max_leaves=15,
)


@_PROPERTY_SETTINGS
@given(garbage=st.text())
@example(garbage="[" * 100_000 + "]" * 100_000)
@example(garbage="```json\n" + "[" * 100_000 + "]" * 100_000 + "\n```")
@example(garbage="[" + "9" * 5000 + "]")
def test_parse_llm_output_as_json_never_raises_on_strings(garbage: str) -> None:
    """
    Arbitrary string input must never raise — only return None or dict/list.
    That includes arrays nested deeper than the parser can recurse, and integers
    with more digits than Python converts.
    """
    result = _parse_llm_output_as_json(garbage)
    assert result is None or isinstance(result, dict | list)


@pytest.mark.parametrize(
    "primitive_json",
    ['"hello"', "0", "1.5", "true", "false", "null"],
)
def test_parse_llm_output_as_json_rejects_primitive_top_level_values(
    primitive_json: str,
) -> None:
    """Valid JSON primitives at the top level are rejected (only dict/list are valid roots)."""
    assert _parse_llm_output_as_json(primitive_json) is None


@_PROPERTY_SETTINGS
@given(value=st.one_of(st.dictionaries(st.text(), _JSON_ATOMS), st.lists(_JSON_ATOMS)))
def test_parse_llm_output_as_json_roundtrip(value: dict | list) -> None:
    """For any JSON-serializable dict/list, parse(json.dumps(x)) == x."""
    encoded = json.dumps(value)
    assert _parse_llm_output_as_json(encoded) == value


@_PROPERTY_SETTINGS
@given(value=st.one_of(st.dictionaries(st.text(), _JSON_ATOMS), st.lists(_JSON_ATOMS)))
def test_parse_llm_output_as_json_dict_or_list_passthrough(value: dict | list) -> None:
    """When input is already a dict/list, it is returned unchanged."""
    assert _parse_llm_output_as_json(value) is value


@_PROPERTY_SETTINGS
@given(value=st.one_of(st.dictionaries(st.text(), _JSON_ATOMS), st.lists(_JSON_ATOMS)))
def test_parse_llm_output_as_json_unwraps_markdown_fence(value: dict | list) -> None:
    """JSON wrapped in ```json ... ``` markdown is extracted and parsed."""
    fenced = f"```json\n{json.dumps(value)}\n```"
    assert _parse_llm_output_as_json(fenced) == value


@pytest.mark.parametrize(
    "non_string_input",
    [42, 3.14, True, False, b"bytes", object()],
)
def test_parse_llm_output_as_json_returns_none_for_non_string_non_collection(
    non_string_input: object,
) -> None:
    """Inputs that are neither None, dict, list, nor str must return None."""
    assert _parse_llm_output_as_json(non_string_input) is None  # ty: ignore[invalid-argument-type]


def test_parse_llm_output_as_json_returns_none_for_none() -> None:
    """``None`` input maps to ``None`` output."""
    assert _parse_llm_output_as_json(None) is None


def _has_non_finite_float(value: object) -> bool:
    """Whether a parsed JSON value holds NaN or an infinity at any depth."""
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_non_finite_float(item) for item in value)
    return False


@_PROPERTY_SETTINGS
@given(text=st.text())
@example(text="[NaN]")
@example(text='{"value": Infinity}')
@example(text="[-Infinity]")
@example(text="[1e400]")
@example(text="[" + "9" * 5000 + "]")
@example(text="```json\n[NaN]\n```")
def test_parse_llm_output_as_json_never_returns_non_finite_numbers(
    text: str,
) -> None:
    """
    The output must be valid JSON, which has no NaN or infinities: ``NaN``,
    ``Infinity``, a number too large for a float, and an integer with more
    digits than Python converts make the output unparseable, rather than
    raising or giving a float that is not finite.
    """
    result = _parse_llm_output_as_json(text)
    assert result is None or not _has_non_finite_float(result)


@_PROPERTY_SETTINGS
@given(
    values=st.lists(
        st.one_of(_JSON_ATOMS, st.sampled_from([math.nan, math.inf, -math.inf])),
        min_size=1,
        max_size=5,
    ).filter(_has_non_finite_float),
    fenced=st.booleans(),
)
def test_parse_llm_output_as_json_rejects_non_finite_numbers(
    values: list, fenced: bool
) -> None:
    """JSON with NaN or an infinity, as ``json.dumps`` writes them, is rejected."""
    text = json.dumps(values)
    if fenced:
        text = f"```json\n{text}\n```"
    assert _parse_llm_output_as_json(text) is None


# ---------------------------------------------------------------------------
# _get_position_from_prompt_id / _get_sentence_position_from_prompt_id
# ---------------------------------------------------------------------------


# Strings that look like the IDs of a prompt, or almost do: with other prefixes,
# signs, leading zeros, non-ASCII digits, suffixes and surrounding whitespace
_ID_LIKE_STRINGS = st.one_of(
    st.from_regex(
        r"\s?[ACPacp]{0,2}[-+]?[0-9٠-٩]{0,3}[_.x]?[0-9]?\s?",
        fullmatch=True,
    ),
    st.from_regex(
        r"\s?[Pp]?[0-9]{0,2}-?[Ss]?-?[0-9٠-٩]{0,2}(-S[0-9])?\s?",
        fullmatch=True,
    ),
    st.text(max_size=8),
)


def _get_ascii_position(text: str) -> int | None:
    """The number written in ASCII digits without a sign or leading zeros, if any."""
    if text and text.isascii() and text.isdigit() and str(int(text)) == text:
        return int(text)
    return None


@_PROPERTY_SETTINGS
@given(
    prefix=st.sampled_from(["A", "C", "P"]),
    position=st.integers(min_value=0, max_value=10**6),
    padding=st.sampled_from(["", " ", "\t", "\n", " \n"]),
)
def test_position_from_prompt_id_roundtrip(
    prefix: Literal["A", "C", "P"], position: int, padding: str
) -> None:
    """The ID of a position, as a prompt shows it, gives the position back."""
    assert (
        _get_position_from_prompt_id(f"{padding}{prefix}{position}{padding}", prefix)
        == position
    )


@_PROPERTY_SETTINGS
@given(prompt_id=_ID_LIKE_STRINGS, prefix=st.sampled_from(["A", "C", "P"]))
@example(prompt_id="P1_0", prefix="P")
@example(prompt_id="P٣", prefix="P")
@example(prompt_id="PP3", prefix="P")
@example(prompt_id="P-0", prefix="P")
@example(prompt_id="3", prefix="P")
@example(prompt_id=" A0 ", prefix="A")
def test_position_from_prompt_id_only_for_ids_of_the_prompt(
    prompt_id: str, prefix: Literal["A", "C", "P"]
) -> None:
    """
    An ID gives a position exactly if it has the form of the IDs that a prompt
    shows: the prefix and the position, in ASCII digits without a sign or
    leading zeros, with or without surrounding whitespace. Any other ID is
    rejected with ``ValueError``, rather than read as another position.
    """
    stripped = prompt_id.strip()
    expected = (
        _get_ascii_position(stripped[1:]) if stripped.startswith(prefix) else None
    )
    if expected is None:
        with pytest.raises(ValueError):
            _get_position_from_prompt_id(prompt_id, prefix)
    else:
        assert _get_position_from_prompt_id(prompt_id, prefix) == expected


@_PROPERTY_SETTINGS
@given(
    paragraph_position=st.integers(min_value=0, max_value=1000),
    sentence_position=st.integers(min_value=0, max_value=1000),
    padding=st.sampled_from(["", " ", "\t", "\n"]),
)
def test_sentence_position_from_prompt_id_roundtrip(
    paragraph_position: int, sentence_position: int, padding: str
) -> None:
    """The ID of a sentence, as a prompt shows it, gives the position back."""
    sentence_id = f"{padding}P{paragraph_position}-S{sentence_position}{padding}"
    assert (
        _get_sentence_position_from_prompt_id(sentence_id, paragraph_position)
        == sentence_position
    )


@_PROPERTY_SETTINGS
@given(sentence_id=_ID_LIKE_STRINGS, paragraph_position=st.integers(0, 12))
@example(sentence_id="P4-S1", paragraph_position=3)
@example(sentence_id="P3-S01", paragraph_position=3)
@example(sentence_id="P3-S1-S2", paragraph_position=3)
@example(sentence_id="S1", paragraph_position=3)
@example(sentence_id="1", paragraph_position=3)
def test_sentence_position_from_prompt_id_only_for_sentences_of_the_paragraph(
    sentence_id: str, paragraph_position: int
) -> None:
    """
    A sentence ID gives a position exactly if it has the form of the sentence IDs
    that a prompt shows and is the ID of a sentence of the given paragraph.
    The ID of a sentence of another paragraph, or any other ID, is rejected with
    ``ValueError``, rather than read as a sentence of the given paragraph.
    """
    stripped = sentence_id.strip()
    paragraph_prefix = f"P{paragraph_position}-S"
    expected = (
        _get_ascii_position(stripped.removeprefix(paragraph_prefix))
        if stripped.startswith(paragraph_prefix)
        else None
    )
    if expected is None:
        with pytest.raises(ValueError):
            _get_sentence_position_from_prompt_id(sentence_id, paragraph_position)
    else:
        assert (
            _get_sentence_position_from_prompt_id(sentence_id, paragraph_position)
            == expected
        )


# ---------------------------------------------------------------------------
# _split_text_into_paragraphs
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_split_text_into_paragraphs_never_raises(text: str) -> None:
    """The function returns a list of strings for any string input."""
    result = _split_text_into_paragraphs(text)
    assert isinstance(result, list)
    assert all(isinstance(p, str) for p in result)


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_split_text_into_paragraphs_outputs_are_stripped(text: str) -> None:
    """Each paragraph in the output has no leading/trailing whitespace."""
    for para in _split_text_into_paragraphs(text):
        assert para == para.strip()


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_split_text_into_paragraphs_no_linebreaks_in_output(text: str) -> None:
    """No paragraph in the output contains a line-break character."""
    for para in _split_text_into_paragraphs(text):
        assert "\n" not in para
        assert "\r" not in para


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_split_text_into_paragraphs_no_empty_outputs(text: str) -> None:
    """No paragraph in the output is whitespace-only."""
    for para in _split_text_into_paragraphs(text):
        assert not _is_text_content_empty(para)


@_PROPERTY_SETTINGS
@given(
    paragraphs=st.lists(
        # Paragraphs without their own newlines or carriage returns,
        # and with at least one alphanumeric to ensure non-emptiness.
        st.from_regex(r"[A-Za-z0-9][A-Za-z0-9 ]{0,40}", fullmatch=True),
        min_size=1,
        max_size=8,
    ),
    separator=st.sampled_from(["\n", "\r\n", "\n\n", "\r\n\r\n"]),
)
def test_split_text_into_paragraphs_roundtrip(
    paragraphs: list[str], separator: str
) -> None:
    """Joining well-formed paragraphs with a newline separator and splitting recovers them."""
    stripped = [p.strip() for p in paragraphs]
    joined = separator.join(stripped)
    assert _split_text_into_paragraphs(joined) == stripped


# ---------------------------------------------------------------------------
# _are_paragraphs_in_text_in_order
# ---------------------------------------------------------------------------

# Texts of few characters, so that a paragraph is often found in a text several
# times, overlapping other paragraphs, or out of order
_FEW_CHARS_PARAGRAPHS = st.lists(
    st.text(alphabet="ab", min_size=1, max_size=4), max_size=6
)


def _are_segmented_paragraphs_matched(paragraphs: list[str], text: str) -> bool:
    """
    Returns whether the segmented paragraphs are matched in the text of
    a document as ``_Document._segment_document_text()`` matches them when they
    are not found in the text in their order: each one in what is left of
    the text when the paragraphs before it are removed from it.
    """
    remaining_text = text
    for paragraph in paragraphs:
        if paragraph not in remaining_text:
            return False
        remaining_text = remaining_text.replace(paragraph, "", 1)
    return True


@_PROPERTY_SETTINGS
@given(
    paragraphs=_FEW_CHARS_PARAGRAPHS,
    text=st.one_of(
        st.text(alphabet="ab ", max_size=30),
        # Texts made of the paragraphs, in their order or not
        _FEW_CHARS_PARAGRAPHS.map(" ".join),
    ),
)
def test_are_paragraphs_in_text_in_order_passes_the_slower_checks(
    paragraphs: list[str], text: str
) -> None:
    """
    Paragraphs that are found in the text in their order pass the checks that
    a document runs only for the paragraphs that are not.
    """
    paragraph_objs = [SimpleNamespace(raw_text=p) for p in paragraphs]
    if _are_paragraphs_in_text_in_order(paragraph_objs, text):  # ty: ignore[invalid-argument-type]
        assert _check_paragraphs_match_in_text(paragraph_objs, text) == []  # ty: ignore[invalid-argument-type]
        _check_paragraphs_ordering_in_text(paragraph_objs, text)  # ty: ignore[invalid-argument-type]
        assert _are_segmented_paragraphs_matched(paragraphs, text)


@_PROPERTY_SETTINGS
@given(
    paragraphs=_FEW_CHARS_PARAGRAPHS,
    separators=st.lists(st.text(alphabet="ab \n", max_size=3), min_size=7, max_size=7),
)
def test_are_paragraphs_in_text_in_order_finds_joined_paragraphs(
    paragraphs: list[str], separators: list[str]
) -> None:
    """
    Paragraphs that the text is made of, in their order, are found in it, also
    when they are found in it elsewhere as well, i.e. the slower checks are not
    run for them.
    """
    text = separators[-1] + "".join(
        p + separator for p, separator in zip(paragraphs, separators, strict=False)
    )
    paragraph_objs = [SimpleNamespace(raw_text=p) for p in paragraphs]
    assert _are_paragraphs_in_text_in_order(paragraph_objs, text)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# _chunk_list
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(
    lst=st.lists(st.integers(), max_size=50),
    n=st.integers(min_value=1, max_value=20),
)
def test_chunk_list_concatenation_preserves_input(lst: list[int], n: int) -> None:
    """Joining all chunks back together yields the original list."""
    chunks = _chunk_list(lst, n)
    flattened = [item for chunk in chunks for item in chunk]
    assert flattened == lst


@_PROPERTY_SETTINGS
@given(
    lst=st.lists(st.integers(), max_size=50),
    n=st.integers(min_value=1, max_value=20),
)
def test_chunk_list_chunk_sizes_within_bound(lst: list[int], n: int) -> None:
    """Every chunk has at most ``n`` elements."""
    for chunk in _chunk_list(lst, n):
        assert 1 <= len(chunk) <= n


@_PROPERTY_SETTINGS
@given(
    lst=st.lists(st.integers(), max_size=50),
    n=st.integers(min_value=1, max_value=20),
)
def test_chunk_list_only_last_chunk_may_be_short(lst: list[int], n: int) -> None:
    """All chunks except possibly the last are exactly ``n`` long."""
    chunks = _chunk_list(lst, n)
    for chunk in chunks[:-1]:
        assert len(chunk) == n


def test_chunk_list_empty_input_returns_empty_list() -> None:
    """An empty input list yields no chunks."""
    assert _chunk_list([], 5) == []


# ---------------------------------------------------------------------------
# _are_prompt_template_brackets_balanced
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(text=st.text(alphabet=st.characters(blacklist_characters="[]{}")))
def test_brackets_balanced_for_text_without_brackets(text: str) -> None:
    """Strings containing no bracket characters are trivially balanced."""
    assert _are_prompt_template_brackets_balanced(text) is True


@_PROPERTY_SETTINGS
@given(
    inner=st.text(alphabet=st.characters(blacklist_characters="[]{}"), max_size=30),
    wrapper=st.sampled_from(["[]", "{}"]),
)
def test_brackets_balanced_for_simple_wrapped_text(inner: str, wrapper: str) -> None:
    """Wrapping bracket-free text with a matched pair stays balanced."""
    composed = wrapper[0] + inner + wrapper[1]
    assert _are_prompt_template_brackets_balanced(composed) is True


@pytest.mark.parametrize(
    "unbalanced",
    ["[", "]", "{", "}", "[}", "{]", "[[}]", "{[}]"],
)
def test_brackets_balanced_detects_clearly_unbalanced(unbalanced: str) -> None:
    """Hand-picked unbalanced strings are correctly rejected."""
    assert _are_prompt_template_brackets_balanced(unbalanced) is False


def test_brackets_balanced_for_empty_string() -> None:
    """The empty string is balanced by definition."""
    assert _are_prompt_template_brackets_balanced("") is True


# ---------------------------------------------------------------------------
# _are_prompt_template_xml_tags_balanced
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(
    # Plain text that cannot accidentally form a tag-like substring.
    text=st.text(alphabet=st.characters(blacklist_characters="<>")),
)
def test_xml_balanced_for_text_without_tag_chars(text: str) -> None:
    """Text containing neither '<' nor '>' has no tags and is balanced."""
    assert _are_prompt_template_xml_tags_balanced(text) is True


@_PROPERTY_SETTINGS
@given(
    tag=st.from_regex(r"[a-zA-Z][a-zA-Z0-9_-]{0,15}", fullmatch=True),
    inner=st.text(alphabet=st.characters(blacklist_characters="<>"), max_size=30),
)
def test_xml_balanced_for_simple_paired_tags(tag: str, inner: str) -> None:
    """A well-formed `<tag>...</tag>` wrapping non-tag text is balanced."""
    composed = f"<{tag}>{inner}</{tag}>"
    assert _are_prompt_template_xml_tags_balanced(composed) is True


def test_xml_balanced_for_empty_string() -> None:
    """The empty string is balanced by definition."""
    assert _are_prompt_template_xml_tags_balanced("") is True


@pytest.mark.parametrize(
    "unbalanced",
    [
        "<a>",  # unclosed
        "</a>",  # closing without opening
        "<a></b>",  # name mismatch
        "<a><b></a></b>",  # interleaved
    ],
)
def test_xml_balanced_detects_clearly_unbalanced(unbalanced: str) -> None:
    """Hand-picked unbalanced XML tag strings are correctly rejected."""
    assert _are_prompt_template_xml_tags_balanced(unbalanced) is False


# ---------------------------------------------------------------------------
# _remove_thinking_content_from_llm_output
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_remove_thinking_returns_str_or_none(text: str) -> None:
    """Output is always either a string or None — never raises."""
    result = _remove_thinking_content_from_llm_output(text)
    assert result is None or isinstance(result, str)


def test_remove_thinking_passes_none_through() -> None:
    """``None`` input maps to ``None`` output."""
    assert _remove_thinking_content_from_llm_output(None) is None


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_remove_thinking_output_is_stripped(text: str) -> None:
    """When the output is a string, it has no leading/trailing whitespace."""
    result = _remove_thinking_content_from_llm_output(text)
    if isinstance(result, str):
        assert result == result.strip()


@_PROPERTY_SETTINGS
@given(
    # Text that cannot contain a `<think>` opening tag at any position.
    text=st.text(alphabet=st.characters(blacklist_characters="<")),
)
def test_remove_thinking_passthrough_when_no_think_prefix(text: str) -> None:
    """Inputs that do not strip-start with `<think>` come back as ``input.strip()``."""
    result = _remove_thinking_content_from_llm_output(text)
    assert result == text.strip()


@_PROPERTY_SETTINGS
@given(text=st.text())
def test_remove_thinking_idempotent(text: str) -> None:
    """Running the function twice yields the same result as once."""
    once = _remove_thinking_content_from_llm_output(text)
    twice = _remove_thinking_content_from_llm_output(once)
    assert once == twice


@_PROPERTY_SETTINGS
@given(
    inside=st.text(alphabet=st.characters(blacklist_characters="<"), max_size=30),
    after=st.from_regex(
        r"[A-Za-z0-9][A-Za-z0-9 ]{0,30}",
        fullmatch=True,
    ),
)
def test_remove_thinking_extracts_content_after_close_tag(
    inside: str, after: str
) -> None:
    """`<think>X</think>Y` returns ``Y.strip()`` for non-empty stripped Y."""
    composed = f"<think>{inside}</think>{after}"
    assert _remove_thinking_content_from_llm_output(composed) == after.strip()


# ---------------------------------------------------------------------------
# _serialize_type_hint <-> _deserialize_type_hint
# ---------------------------------------------------------------------------


# Primitives recognised by both the serializer and deserializer.
_PRIMITIVE_TYPE_HINTS = st.sampled_from([str, int, float, bool, type(None)])


def _build_type_hint_strategy() -> st.SearchStrategy[Any]:
    """Recursive Hypothesis strategy for type hints supported by the round-trip."""
    return st.recursive(
        _PRIMITIVE_TYPE_HINTS,
        lambda children: st.one_of(
            children.map(lambda t: list[t]),
            st.tuples(_PRIMITIVE_TYPE_HINTS, children).map(
                lambda kv: dict[kv[0], kv[1]]  # ty: ignore[invalid-type-form]
            ),
            # Optional[T] — typing.Union with None — exercises the "optional"
            # branch in the (de)serializer.
            children.map(lambda t: Union[t, None]),  # noqa: UP007
        ),
        max_leaves=6,
    )


_TYPE_HINTS = _build_type_hint_strategy()


@_PROPERTY_SETTINGS
@given(type_hint=_TYPE_HINTS)
def test_type_hint_roundtrip(type_hint: Any) -> None:
    """``deserialize(serialize(t)) == t`` for any supported type hint."""
    serialized = _serialize_type_hint(type_hint)
    assert isinstance(serialized, str)
    restored = _deserialize_type_hint(serialized)
    assert restored == type_hint


@_PROPERTY_SETTINGS
@given(type_hint=_TYPE_HINTS)
def test_type_hint_serialize_is_deterministic(type_hint: Any) -> None:
    """Serialization is a pure function — the same input always yields the same output."""
    assert _serialize_type_hint(type_hint) == _serialize_type_hint(type_hint)


# ---------------------------------------------------------------------------
# _group_instances_by_fields
# ---------------------------------------------------------------------------


@_PROPERTY_SETTINGS
@given(
    instances=st.lists(
        st.tuples(
            st.sampled_from(["A", "B", "C"]),  # role
            st.integers(min_value=0, max_value=3),  # priority
            st.booleans(),  # active
        ),
        max_size=20,
    ),
    fields=st.lists(
        st.sampled_from(["role", "priority", "active"]),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
def test_group_instances_preserves_total_count(
    instances: list[tuple[str, int, bool]], fields: list[str]
) -> None:
    """Grouping never adds or drops items — total count is preserved."""
    objs = [
        SimpleNamespace(role=role, priority=prio, active=active)
        for role, prio, active in instances
    ]
    groups = _group_instances_by_fields(fields, objs)  # ty: ignore[invalid-argument-type]
    assert sum(len(g) for g in groups) == len(objs)


@_PROPERTY_SETTINGS
@given(
    instances=st.lists(
        st.tuples(
            st.sampled_from(["A", "B", "C"]),
            st.integers(min_value=0, max_value=3),
            st.booleans(),
        ),
        max_size=20,
    ),
    fields=st.lists(
        st.sampled_from(["role", "priority", "active"]),
        min_size=1,
        max_size=3,
        unique=True,
    ),
)
def test_group_instances_share_field_values_within_group(
    instances: list[tuple[str, int, bool]], fields: list[str]
) -> None:
    """All instances inside a group have identical values for each grouping field."""
    objs = [
        SimpleNamespace(role=role, priority=prio, active=active)
        for role, prio, active in instances
    ]
    groups = _group_instances_by_fields(fields, objs)  # ty: ignore[invalid-argument-type]
    for group in groups:
        for field in fields:
            values = {getattr(obj, field) for obj in group}
            assert len(values) == 1


def test_group_instances_empty_input_returns_empty_list() -> None:
    """An empty instance list yields no groups."""
    assert _group_instances_by_fields(["role"], []) == []
