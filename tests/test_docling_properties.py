# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the property-based tests of the creation of documents from
Docling documents (``Document.from_docling()``).

Hypothesis generates Docling documents with the structures that Docling gives
the files it converts: the title and headings, with their own text, formatted
text or a text under them, texts of several kinds, with items under them,
formatted text with inline code, formulas and pictures in it, nested lists,
bulleted, numbered with their markers, without them or after a bullet, with
texts, checkboxes and lists that are not in an item, list items outside lists,
tables with merged cells, header rows, row headers, row sections and long
captions, and with rich cells, whose items are under the table, as in DOCX, HTML
or OpenDocument files: texts, checkboxes, formatted text, headings, lists, code
blocks, nested tables and pictures, pictures,
code blocks with items under them, key-value regions and forms with links of
every kind, each with captions and footnotes, in the body or not, with items
under them, form fields, sheets and slides, and the furniture and the notes of
the pages. Each item is on no page or in one or two boxes of the pages, and
a text can be from the track of a subtitle file or a transcript, with its
speaker. Each text is a word of its own, with whitespace around it, so that
the tests can tell where it went.

One test works out from the generated structure what the document must have:
its paragraphs, in order, with their text, additional context and page numbers,
and the Docling data of their custom data (the reference and the label of the
item, the provenance, the tracks and the index of a table row), and its images.
Another checks what holds for any Docling document, without working out
the paragraphs: each text of the body is kept, once unless it is repeated by
the structure, e.g. as the header of a table column, the rest is left out, and
the document round-trips through serialization.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pprint import pformat
from typing import Any

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItem,
    DocItemLabel,
    DoclingDocument,
    GraphCell,
    GraphCellLabel,
    GraphData,
    GraphLink,
    GraphLinkLabel,
    GroupLabel,
    ImageRef,
    ListGroup,
    ListItem,
    NodeItem,
    ProvenanceItem,
    RichTableCell,
    TableCell,
    TableData,
    TableItem,
    TrackSource,
)
from hypothesis import HealthCheck, event, given, note, settings
from hypothesis import strategies as st
from PIL import Image as PILImage

from contextgem import Document
from tests.base import BaseTest
from tests.utils import capture_logs


# An example creates a document, which can take longer than a deadline allows
_PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)

# The colors of the generated pictures: pictures of the same color have the same
# image, which a document holds once
_COLORS = ("red", "green", "blue")

_BODY = ContentLayer.BODY
_FURNITURE = ContentLayer.FURNITURE


def _image_ref(color: str) -> ImageRef:
    """
    Returns a Docling image reference that embeds a square image of one color.

    :param color: The color of the image, e.g. "red".
    :type color: str
    :return: The image reference, with the image as a data URI.
    :rtype: ImageRef
    """
    return ImageRef.from_pil(PILImage.new("RGB", (4, 4), color), dpi=72)


# The base64 data of the image of each color, as a document holds it
_IMAGE_DATA = {color: str(_image_ref(color).uri).split(",", 1)[1] for color in _COLORS}

# The labels of body text, which the additional context of a paragraph does not
# name, and what it calls a text with another label
_BODY_TEXT_LABELS = (DocItemLabel.TEXT, DocItemLabel.PARAGRAPH)
_TEXT_KINDS = {
    DocItemLabel.FOOTNOTE: "Footnote",
    DocItemLabel.CODE: "Code",
    DocItemLabel.FORMULA: "Formula",
    DocItemLabel.CAPTION: "Caption",
    DocItemLabel.CHECKBOX_SELECTED: "Checkbox, checked",
    DocItemLabel.CHECKBOX_UNSELECTED: "Checkbox, unchecked",
}
# The labels of the items whose paragraphs are not running text, each of which is
# one sentence, its whole text: headings, table headers and rows, key-value pairs
# and form fields, code, formulas, checkboxes, references and form headings
_ONE_SENTENCE_LABELS = frozenset(
    label.value
    for label in (
        DocItemLabel.TITLE,
        DocItemLabel.SECTION_HEADER,
        DocItemLabel.TABLE,
        DocItemLabel.KEY_VALUE_REGION,
        DocItemLabel.FORM,
        DocItemLabel.FIELD_ITEM,
        DocItemLabel.CODE,
        DocItemLabel.FORMULA,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        DocItemLabel.REFERENCE,
        DocItemLabel.FIELD_HEADING,
    )
)
# The states of checkboxes, as the text of a table cell gives them
_CHECKBOX_STATES = {
    DocItemLabel.CHECKBOX_SELECTED: "checked",
    DocItemLabel.CHECKBOX_UNSELECTED: "unchecked",
}

# The start and the end time of the track of a text from a track, in seconds
_TRACK_TIMES = (1.5, 4.0)


def _bbox(box: int) -> BoundingBox:
    """
    Returns a bounding box on a page, one of three side by side.

    :param box: The index of the box, from 0 to 2.
    :type box: int
    :return: The bounding box.
    :rtype: BoundingBox
    """
    return BoundingBox(
        l=72 + 150 * box,
        t=700,
        r=192 + 150 * box,
        b=680,
        coord_origin=CoordOrigin.BOTTOMLEFT,
    )


# ---------------------------------------------------------------------------
# The generated structure
# ---------------------------------------------------------------------------

# Where an item is: the number of the page and the index of the box (see `_bbox()`)
# of each of its parts, e.g. of a paragraph that continues in another column
_Prov = list[tuple[int, int]]


@dataclass
class _Text:
    """
    A text: a text item, or a part of formatted text, a cell or a caption. Its word
    is given when the Docling document is built. A text item can be from a track,
    e.g. the cue of a subtitle file, with or without the voice of a speaker, and
    can have items under it.
    """

    label: DocItemLabel
    prov: _Prov
    layer: ContentLayer
    before: str
    after: str
    tracked: bool = False
    voice: str | None = None
    # The words after its word, each after a space, e.g. those of a long caption
    tail: str = ""
    children: list[_Block] = field(default_factory=list)
    word: str = ""
    self_ref: str = ""

    @property
    def content(self) -> str:
        """
        The text of the item, without the whitespace around it.
        """
        return self.word + self.tail

    @property
    def speaker(self) -> str | None:
        """
        The speaker of the text, as the additional context gives it, if it is from
        a track with a voice.
        """
        voice = " ".join((self.voice or "").split())
        return voice if self.tracked and voice else None


@dataclass
class _Picture:
    """
    A picture, with the image of a color or without an image, its caption and its
    footnote, in its layer or not, and the text found in it, which is not
    converted.
    """

    color: str | None
    caption: _Text | None
    footnote: _Text | None
    inner_text: _Text | None
    prov: _Prov
    layer: ContentLayer
    self_ref: str = ""


@dataclass
class _Inline:
    """
    Formatted text: an inline group of texts, with pictures and inline groups in it.
    """

    parts: list[_Text | _Picture | _Inline]
    self_ref: str = ""


@dataclass
class _Heading:
    """
    The title (level 0) or a heading, with its own text, with formatted text or
    a text under it, or without text, and the items under it, as in HTML.
    """

    level: int
    prov: _Prov
    own: _Text | None
    formatted: _Inline | None
    child: _Text | None
    children: list[_Block]
    self_ref: str = ""


@dataclass
class _ListItem:
    """
    A list item, bulleted or numbered, with its marker, e.g. "-", "3." or "a)", or
    none, with its own text, formatted text or a text under it, or no text, and
    the items under it, e.g. a nested list.
    """

    enumerated: bool
    marker: str
    prov: _Prov
    own: _Text | None
    formatted: _Inline | None
    child: _Text | None
    children: list[_Block]
    self_ref: str = ""


@dataclass
class _List:
    """
    A list: its items, and a text that is in the list but not in an item, e.g.
    a checkbox, and a list in the list but not in an item, as Docling gives some
    lists, e.g. those of DOCX.
    """

    items: list[_ListItem | _Text | _List]


@dataclass
class _Cell:
    """
    A cell of a table, with a text or empty, or a rich cell, whose items are under
    the table, as Docling gives the cells of e.g. DOCX, HTML or OpenDocument
    files, at its row and column, over one or more rows and columns.
    """

    text: _Text | None
    row: int
    column: int
    rows: int
    columns: int
    column_header: bool
    row_header: bool
    row_section: bool
    # The items of a rich cell, or None for a cell with a text or empty
    content: list[_Block] | None = None


@dataclass
class _Table:
    """
    A table: its cells, which cover its rows and columns, with its header rows at
    the top, its caption (a text, or a picture, which is not a text) and its
    footnote, and the pictures in its cells.
    """

    rows: int
    columns: int
    cells: list[_Cell]
    caption: _Text | _Picture | None
    footnote: _Text | None
    prov: _Prov
    pictures: list[_Picture]
    layer: ContentLayer
    self_ref: str = ""


@dataclass
class _Code:
    """
    A code block, with its caption and footnote, and the items under it. Its
    caption is before or after it in the body, as Docling gives it in different
    formats.
    """

    text: _Text
    caption: _Text | None
    caption_after: bool
    footnote: _Text | None


@dataclass
class _Graph:
    """
    A key-value region or a form: its cells, which are labelled keys or values,
    the links between them, in order, as (key, value, kind) for a link from a key
    to a value ("to value"), from a value to a key ("to key"), or both ("both"),
    and for the links that are not between a key and a value ("to child",
    "to parent") and the links to a missing cell ("dangling value", "dangling
    key"), the region's place, and its caption and footnote.
    """

    form: bool
    cells: list[_Text]
    keys: list[bool]
    links: list[tuple[int, int, str]]
    prov: _Prov
    caption: _Text | None
    footnote: _Text | None
    self_ref: str = ""


@dataclass
class _Checkbox:
    """
    The checkbox of a form field, with or without a text.
    """

    checked: bool
    text: _Text | None


@dataclass
class _Field:
    """
    A form field in a field region: its key, its value and its checkbox, and its
    number, which is left out.
    """

    key: _Text
    value: _Text | None
    checkbox: _Checkbox | None
    number: bool
    prov: _Prov
    self_ref: str = ""


@dataclass
class _Group:
    """
    A group of items: a sheet, with or without a name, a slide, the number of
    which is given if its name is that of a slide, or a plain group, e.g.
    a section of HTML.
    """

    kind: str
    label: GroupLabel
    name: str
    number: int | None
    children: list[_Block]


_Block = (
    _Text
    | _Inline
    | _Heading
    | _List
    | _ListItem
    | _Table
    | _Picture
    | _Code
    | _Graph
    | _Field
    | _Group
)

# The items that have a place on the pages
_Located = _Text | _Picture | _Heading | _ListItem | _Table | _Graph | _Field


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Unicode whitespace, which the converter strips from the ends of a text
_NOISE = st.sampled_from(["", " ", "\n", " \t", "\u00a0", "\u2003"])
# The voices of the tracks: a speaker, once with whitespace around the name, or none
_VOICES = st.sampled_from([None, "", "Ann", " Ann\n", "Bob Lee"])
# The layer of most items, and that of a caption or a footnote, which can differ
# from that of its item
_BODY_LAYER = st.just(_BODY)
_ATTACHED_LAYERS = st.sampled_from([_BODY, _BODY, _BODY, _FURNITURE])
# The words after the word of a long caption, some ending with punctuation
_TAILS = st.one_of(
    st.just(""),
    st.lists(
        st.tuples(
            st.sampled_from(["fees", "of", "the", "services", "in", "2026", "per"]),
            st.sampled_from(["", "", ",", ";", ":", "."]),
        ),
        min_size=18,
        max_size=30,
    ).map(lambda words: "".join(f" {word}{mark}" for word, mark in words)),
)
# The items under an item of the leaves: none, as a new list for each item
_NO_CHILDREN = st.builds(list)


def _provs(max_boxes: int = 2) -> st.SearchStrategy[_Prov]:
    """
    Returns a strategy for the places of items: none, or boxes on the pages.

    :param max_boxes: The maximum number of boxes.
    :type max_boxes: int
    :return: The strategy.
    :rtype: st.SearchStrategy[_Prov]
    """
    return st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=3), st.integers(min_value=0, max_value=2)
        ),
        max_size=max_boxes,
    )


def _texts(
    *labels: DocItemLabel,
    layers: st.SearchStrategy[ContentLayer] = _BODY_LAYER,
    tracks: bool = False,
    max_boxes: int = 2,
) -> st.SearchStrategy[_Text]:
    """
    Returns a strategy for texts with one of the labels.

    :param labels: The labels, by default that of body text.
    :type labels: DocItemLabel
    :param layers: The strategy for the content layer of a text.
    :type layers: st.SearchStrategy[ContentLayer]
    :param tracks: Whether a text can be from a track, as a text item can.
    :type tracks: bool
    :param max_boxes: The maximum number of boxes of a text.
    :type max_boxes: int
    :return: The strategy.
    :rtype: st.SearchStrategy[_Text]
    """
    return st.builds(
        _Text,
        label=st.sampled_from(labels or (DocItemLabel.TEXT,)),
        prov=_provs(max_boxes),
        layer=layers,
        before=_NOISE,
        after=_NOISE,
        tracked=st.booleans() if tracks else st.just(False),
        voice=_VOICES if tracks else st.none(),
    )


@st.composite
def _attached_texts(
    draw: st.DrawFn,
    label: DocItemLabel,
    children: st.SearchStrategy[list[_Block]] | None = None,
    layers: st.SearchStrategy[ContentLayer] = _ATTACHED_LAYERS,
    tails: bool = False,
) -> _Text:
    """
    Generates the caption or the footnote of an item, in the body or not, with
    items under it if it is in the body.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param label: The label of the text, e.g. that of a caption.
    :type label: DocItemLabel
    :param children: The strategy for the items under the text, if it can have any.
    :type children: st.SearchStrategy[list[_Block]] | None
    :param layers: The strategy for the content layer of the text.
    :type layers: st.SearchStrategy[ContentLayer]
    :param tails: Whether the text can be long, with words after its word.
    :type tails: bool
    :return: The caption or the footnote.
    :rtype: _Text
    """
    text = draw(_texts(label, layers=layers))
    if tails:
        text.tail = draw(_TAILS)
    if children is not None and text.layer == _BODY:
        text.children = draw(children)
    return text


def _pictures(
    layers: st.SearchStrategy[ContentLayer],
    children: st.SearchStrategy[list[_Block]] | None = None,
    attached_layers: st.SearchStrategy[ContentLayer] = _ATTACHED_LAYERS,
) -> st.SearchStrategy[_Picture]:
    """
    Returns a strategy for pictures in one of the content layers.

    :param layers: The strategy for the content layer of a picture.
    :type layers: st.SearchStrategy[ContentLayer]
    :param children: The strategy for the items under its caption and footnote, if
        they can have any.
    :type children: st.SearchStrategy[list[_Block]] | None
    :param attached_layers: The strategy for the content layer of its caption and
        footnote.
    :type attached_layers: st.SearchStrategy[ContentLayer]
    :return: The strategy.
    :rtype: st.SearchStrategy[_Picture]
    """
    return st.builds(
        _Picture,
        color=st.one_of(st.none(), st.sampled_from(_COLORS)),
        caption=st.one_of(
            st.none(),
            _attached_texts(DocItemLabel.CAPTION, children, layers=attached_layers),
        ),
        footnote=st.one_of(
            st.none(),
            _attached_texts(DocItemLabel.FOOTNOTE, children, layers=attached_layers),
        ),
        inner_text=st.one_of(st.none(), _texts(max_boxes=0)),
        prov=_provs(),
        layer=layers,
    )


# A picture given as the caption of a table, which is not a text
_NON_TEXT_CAPTIONS = st.builds(
    _Picture,
    color=st.sampled_from(_COLORS),
    caption=st.none(),
    footnote=st.none(),
    inner_text=st.none(),
    prov=_provs(),
    layer=_BODY_LAYER,
)


@st.composite
def _tables(
    draw: st.DrawFn,
    children: st.SearchStrategy[list[_Block]] | None = None,
    cell_items: st.SearchStrategy[list[_Block]] | None = None,
) -> _Table:
    """
    Generates a table, in the body or not, whose cells can span rows and columns,
    with header rows at the top, whose cells are column headers, but for some, and
    with row headers and row sections in its other rows. A table of the body can
    have rich cells.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under its caption and footnote, if
        they can have any.
    :type children: st.SearchStrategy[list[_Block]] | None
    :param cell_items: The strategy for the items of a rich cell, if the table can
        have rich cells.
    :type cell_items: st.SearchStrategy[list[_Block]] | None
    :return: The table.
    :rtype: _Table
    """
    columns = draw(st.integers(min_value=1, max_value=3))
    header_rows = draw(st.integers(min_value=0, max_value=2))
    rows = header_rows + draw(st.integers(min_value=1, max_value=3))
    layer = draw(st.sampled_from([_BODY, _BODY, _BODY, _BODY, _FURNITURE]))
    # Most cells have a text, and some of a table of the body are rich cells
    cell_kinds = ["text", "text", "text", "empty"]
    if cell_items is not None and layer == _BODY:
        cell_kinds.append("rich")
    taken: set[tuple[int, int]] = set()
    cells = []
    for row in range(rows):
        in_header = row < header_rows
        # A cell spans the rows of its part of the table: the header or the rest
        end_row = header_rows if in_header else rows
        for column in range(columns):
            if (row, column) in taken:
                continue
            free = 1
            while column + free < columns and (row, column + free) not in taken:
                free += 1
            # A cell of the body often spans the rest of the row, e.g. a section
            # of the table
            width = draw(
                st.sampled_from([*([] if in_header else [free]), *range(1, free + 1)])
            )
            height = min(draw(st.sampled_from([1, 1, 1, 2])), end_row - row)
            taken.update(
                (cell_row, cell_column)
                for cell_row in range(row, row + height)
                for cell_column in range(column, column + width)
            )
            kind = draw(st.sampled_from(cell_kinds))
            cells.append(
                _Cell(
                    text=draw(_texts(max_boxes=0)) if kind == "text" else None,
                    content=draw(cell_items)
                    if kind == "rich" and cell_items is not None
                    else None,
                    row=row,
                    column=column,
                    rows=height,
                    columns=width,
                    # A row with a value that is not a column header is not
                    # a header row
                    column_header=in_header
                    and draw(st.sampled_from([True, True, True, True, False])),
                    row_header=column == 0 and draw(st.booleans()),
                    row_section=not in_header
                    and draw(st.sampled_from([False, False, False, True])),
                )
            )
    caption_kind = draw(st.sampled_from(["none", "text", "text", "picture"]))
    caption: _Text | _Picture | None = None
    if caption_kind == "text":
        caption = draw(
            _attached_texts(
                DocItemLabel.CAPTION,
                children,
                layers=_ATTACHED_LAYERS if layer == _BODY else _BODY_LAYER,
                tails=True,
            )
        )
    elif caption_kind == "picture":
        caption = draw(_NON_TEXT_CAPTIONS)
    return _Table(
        rows=rows,
        columns=columns,
        cells=cells,
        caption=caption,
        footnote=draw(
            st.one_of(st.none(), _attached_texts(DocItemLabel.FOOTNOTE, children))
        ),
        prov=draw(_provs()),
        # The pictures in the cells, in the layer of the table, with their captions
        pictures=draw(
            st.lists(
                _pictures(st.just(layer), attached_layers=st.just(layer)), max_size=1
            )
        ),
        layer=layer,
    )


def _caption_child() -> st.SearchStrategy[_Block]:
    """
    Returns a strategy for an item under a caption or a footnote.

    :return: The strategy, for texts, pictures and tables.
    :rtype: st.SearchStrategy[_Block]
    """
    return st.one_of(
        _texts(*_BODY_TEXT_LABELS, DocItemLabel.FOOTNOTE),
        _pictures(_BODY_LAYER),
        _tables(),
    )


# The items under a caption or a footnote: often none
_CAPTION_CHILDREN = st.one_of(
    _NO_CHILDREN, _NO_CHILDREN, st.lists(_caption_child(), min_size=1, max_size=2)
)

_BODY_PICTURES = _pictures(_BODY_LAYER, _CAPTION_CHILDREN)
# A picture of the furniture is e.g. a logo in the header of each page
_ANY_PICTURES = _pictures(
    st.sampled_from([_BODY, _BODY, _BODY, _FURNITURE]), _CAPTION_CHILDREN
)

# The texts of formatted text: text, inline code and formulas, as in HTML or
# Markdown
_INLINE_TEXTS = _texts(
    DocItemLabel.TEXT,
    DocItemLabel.TEXT,
    DocItemLabel.CODE,
    DocItemLabel.FORMULA,
    tracks=True,
)

_INLINES = st.recursive(
    st.builds(
        _Inline,
        parts=st.lists(
            st.one_of(_INLINE_TEXTS, _BODY_PICTURES), min_size=1, max_size=4
        ),
    ),
    lambda inner: st.builds(
        _Inline,
        parts=st.lists(
            st.one_of(_INLINE_TEXTS, _BODY_PICTURES, inner),
            min_size=1,
            max_size=4,
        ),
    ),
    max_leaves=6,
)

_TEXT_BLOCKS = st.one_of(
    _texts(
        DocItemLabel.TEXT,
        DocItemLabel.PARAGRAPH,
        DocItemLabel.FOOTNOTE,
        DocItemLabel.CODE,
        DocItemLabel.FORMULA,
        DocItemLabel.CAPTION,
        DocItemLabel.CHECKBOX_SELECTED,
        DocItemLabel.CHECKBOX_UNSELECTED,
        tracks=True,
    ),
    _texts(
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        layers=st.just(_FURNITURE),
    ),
    _texts(DocItemLabel.TEXT, layers=st.just(ContentLayer.NOTES)),
)


@st.composite
def _text_parents(draw: st.DrawFn, children: st.SearchStrategy[list[_Block]]) -> _Text:
    """
    Generates a text item with items under it, e.g. a paragraph of HTML with
    a picture or a list in it.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the text.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The text.
    :rtype: _Text
    """
    text = draw(
        _texts(
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.CAPTION,
            tracks=True,
        )
    )
    text.children = draw(children)
    return text


@st.composite
def _codes(
    draw: st.DrawFn,
    children: st.SearchStrategy[list[_Block]],
    attached_children: st.SearchStrategy[list[_Block]] = _CAPTION_CHILDREN,
) -> _Code:
    """
    Generates a code block, with its caption and its footnote, and items under it.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the code block.
    :type children: st.SearchStrategy[list[_Block]]
    :param attached_children: The strategy for the items under its caption and
        footnote.
    :type attached_children: st.SearchStrategy[list[_Block]]
    :return: The code block.
    :rtype: _Code
    """
    text = draw(_texts(DocItemLabel.CODE, tracks=True))
    text.children = draw(children)
    return _Code(
        text=text,
        caption=draw(
            st.one_of(
                st.none(), _attached_texts(DocItemLabel.CAPTION, attached_children)
            )
        ),
        caption_after=draw(st.booleans()),
        footnote=draw(
            st.one_of(
                st.none(), _attached_texts(DocItemLabel.FOOTNOTE, attached_children)
            )
        ),
    )


# The kinds of the links of a key-value region or a form, those between a key and
# a value weighted up
_LINK_KINDS = (
    *(("to value", "to key", "both") * 3),
    "to child",
    "to parent",
    "dangling value",
    "dangling key",
)


@st.composite
def _graphs(draw: st.DrawFn) -> _Graph:
    """
    Generates a key-value region or a form, whose cells are linked in any way:
    a value can be linked to several keys, or be the key of other values.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :return: The key-value region or the form.
    :rtype: _Graph
    """
    cells = draw(st.lists(_texts(max_boxes=1), min_size=2, max_size=4))
    links = []
    for _ in range(draw(st.integers(min_value=0, max_value=len(cells) + 1))):
        key = draw(st.integers(min_value=0, max_value=len(cells) - 1))
        # Another cell than the key
        value = (key + draw(st.integers(min_value=1, max_value=len(cells) - 1))) % len(
            cells
        )
        links.append((key, value, draw(st.sampled_from(_LINK_KINDS))))
    return _Graph(
        form=draw(st.booleans()),
        cells=cells,
        keys=draw(st.lists(st.booleans(), min_size=len(cells), max_size=len(cells))),
        links=links,
        prov=draw(_provs()),
        caption=draw(
            st.one_of(
                st.none(), _attached_texts(DocItemLabel.CAPTION, _CAPTION_CHILDREN)
            )
        ),
        footnote=draw(
            st.one_of(
                st.none(), _attached_texts(DocItemLabel.FOOTNOTE, _CAPTION_CHILDREN)
            )
        ),
    )


_FIELDS = st.builds(
    _Field,
    key=_texts(),
    value=st.one_of(st.none(), _texts()),
    checkbox=st.one_of(
        st.none(),
        st.builds(_Checkbox, checked=st.booleans(), text=st.none() | _texts()),
    ),
    number=st.booleans(),
    prov=_provs(),
)

# Where the text of a heading or a list item is: its own, in formatted text under
# it, in a text under it, or nowhere, unless its first item gives it
_TEXT_FORMS = ("own", "own", "formatted", "child", "none")


@st.composite
def _first_texts(draw: st.DrawFn, children: st.SearchStrategy[list[_Block]]) -> _Text:
    """
    Generates the body text under a heading or a list item that gives it its text,
    sometimes with items under it.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the text.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The text.
    :rtype: _Text
    """
    text = draw(_texts(*_BODY_TEXT_LABELS))
    if draw(st.booleans()):
        text.children = draw(children)
    return text


@st.composite
def _headings(draw: st.DrawFn, children: st.SearchStrategy[list[_Block]]) -> _Heading:
    """
    Generates the title or a heading.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the heading.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The title or the heading.
    :rtype: _Heading
    """
    form = draw(st.sampled_from(_TEXT_FORMS))
    return _Heading(
        level=draw(st.integers(min_value=0, max_value=3)),
        prov=draw(_provs()),
        own=draw(_texts(max_boxes=0)) if form == "own" else None,
        formatted=draw(_INLINES) if form == "formatted" else None,
        child=draw(_first_texts(children)) if form == "child" else None,
        children=draw(children),
    )


@st.composite
def _list_items(
    draw: st.DrawFn,
    children: st.SearchStrategy[list[_Block]],
    enumerated: bool,
    marker: str,
) -> _ListItem:
    """
    Generates a list item.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the list item.
    :type children: st.SearchStrategy[list[_Block]]
    :param enumerated: Whether the item is numbered.
    :type enumerated: bool
    :param marker: The marker of the item, e.g. "-" or "a)", or an empty string.
    :type marker: str
    :return: The list item.
    :rtype: _ListItem
    """
    form = draw(st.sampled_from(_TEXT_FORMS))
    return _ListItem(
        enumerated=enumerated,
        marker=marker,
        prov=draw(_provs()),
        own=draw(_texts(max_boxes=0)) if form == "own" else None,
        formatted=draw(_INLINES) if form == "formatted" else None,
        child=draw(_first_texts(children)) if form == "child" else None,
        children=draw(children),
    )


# The markers of numbered list items, unlike their positions, and whitespace or
# nothing, which is no marker
_MARKERS = ("a)", "iv.", "7.", "(2)", " 3. ", " ", "")


@st.composite
def _lists(
    draw: st.DrawFn, children: st.SearchStrategy[list[_Block]], nested: bool = True
) -> _List:
    """
    Generates a list: bulleted, numbered with markers, numbered without markers
    as Docling gives the lists of some formats, e.g. HTML, numbered after a bullet,
    or mixed, sometimes with a text or a checkbox that is not in an item, and
    sometimes with a list that is not in an item, as Docling gives the nested lists
    of DOCX.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under each list item.
    :type children: st.SearchStrategy[list[_Block]]
    :param nested: Whether the list can have a list in it that is not in an item.
    :type nested: bool
    :return: The list.
    :rtype: _List
    """
    style = draw(
        st.sampled_from(
            ["bulleted", "numbered", "unnumbered", "after a bullet", "mixed"]
        )
    )
    count = draw(
        st.integers(min_value=2 if style == "after a bullet" else 1, max_value=3)
    )
    items: list[_ListItem | _Text | _List] = []
    for index in range(count):
        if style == "bulleted":
            enumerated, marker = False, draw(st.sampled_from(["-", "\u2022", ""]))
        elif style == "numbered":
            enumerated, marker = True, draw(st.sampled_from(_MARKERS))
        elif style == "unnumbered":
            enumerated, marker = True, draw(st.sampled_from(["", " "]))
        elif style == "after a bullet":
            enumerated, marker = index > 0, "" if index else "-"
        else:
            enumerated = draw(st.booleans())
            marker = draw(st.sampled_from(["-", "b)", ""]))
        items.append(draw(_list_items(children, enumerated, marker)))
    if not draw(st.integers(min_value=0, max_value=4)):
        items.insert(
            draw(st.integers(min_value=0, max_value=len(items))),
            draw(
                _texts(
                    *_BODY_TEXT_LABELS,
                    DocItemLabel.CHECKBOX_SELECTED,
                    DocItemLabel.CHECKBOX_UNSELECTED,
                )
            ),
        )
    if nested and not draw(st.integers(min_value=0, max_value=4)):
        items.insert(
            draw(st.integers(min_value=0, max_value=len(items))),
            draw(_lists(_NO_CHILDREN, nested=False)),
        )
    return _List(items=items)


@st.composite
def _lone_list_items(draw: st.DrawFn) -> _ListItem:
    """
    Generates a list item that is not in a list.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :return: The list item.
    :rtype: _ListItem
    """
    return draw(
        _list_items(
            _NO_CHILDREN, draw(st.booleans()), draw(st.sampled_from(["", "a)", "-"]))
        )
    )


@st.composite
def _groups(draw: st.DrawFn, children: st.SearchStrategy[list[_Block]]) -> _Group:
    """
    Generates a sheet, with a name, a name of whitespace or none, a slide, a group
    named like a slide that is not one, or a plain group.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items in the group.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The group.
    :rtype: _Group
    """
    kind = draw(st.sampled_from(["plain", "sheet", "slide", "slide"]))
    number = None
    if kind == "sheet":
        label = GroupLabel.SHEET
        name = draw(st.sampled_from(["Q3", " Fees \n2026", "", " ", "\n\t"]))
    elif kind == "slide":
        # A slide is a chapter in the Docling documents of some versions
        label = draw(
            st.sampled_from([GroupLabel.SLIDE, GroupLabel.CHAPTER, GroupLabel.SECTION])
        )
        slide = draw(st.integers(min_value=0, max_value=30))
        name = draw(
            st.sampled_from(
                [
                    f"slide-{slide}",
                    f"slide-{slide}",
                    f"slide-{slide}-extra",
                    f"x-slide-{slide}",
                ]
            )
        )
        if name == f"slide-{slide}" and label != GroupLabel.SECTION:
            number = slide
    else:
        label, name = GroupLabel.SECTION, "group"
    return _Group(
        kind=kind, label=label, name=name, number=number, children=draw(children)
    )


# The items of a rich cell: texts and checkboxes, formatted text, headings, lists,
# code blocks, nested tables and pictures, or none
_RICH_CELL_ITEMS = st.lists(
    st.one_of(
        _texts(
            *_BODY_TEXT_LABELS,
            DocItemLabel.CHECKBOX_SELECTED,
            DocItemLabel.CHECKBOX_UNSELECTED,
        ),
        _INLINES,
        _headings(_NO_CHILDREN),
        _lists(_NO_CHILDREN),
        _codes(_NO_CHILDREN, _NO_CHILDREN),
        _tables(_CAPTION_CHILDREN),
        _ANY_PICTURES,
    ),
    max_size=3,
)

_BLOCKS = st.recursive(
    st.one_of(
        _TEXT_BLOCKS,
        _INLINES,
        _tables(_CAPTION_CHILDREN, _RICH_CELL_ITEMS),
        _ANY_PICTURES,
        _codes(_NO_CHILDREN),
        _graphs(),
        _FIELDS,
        _headings(_NO_CHILDREN),
        _lists(_NO_CHILDREN),
        _lone_list_items(),
    ),
    lambda block: st.one_of(
        _headings(st.lists(block, max_size=3)),
        _lists(st.lists(block, max_size=2)),
        _groups(st.lists(block, max_size=3)),
        _text_parents(st.lists(block, min_size=1, max_size=2)),
        _codes(st.lists(block, min_size=1, max_size=2)),
    ),
    max_leaves=12,
)
_DOCUMENTS = st.lists(_BLOCKS, max_size=6)


# ---------------------------------------------------------------------------
# The Docling document of a generated structure
# ---------------------------------------------------------------------------

# The Docling links of each kind of link of a key-value region or a form, as
# (label, whether the link is from the key, whether it is to a missing cell)
_GRAPH_LINKS = {
    "to value": [(GraphLinkLabel.TO_VALUE, True, False)],
    "to key": [(GraphLinkLabel.TO_KEY, False, False)],
    "both": [
        (GraphLinkLabel.TO_VALUE, True, False),
        (GraphLinkLabel.TO_KEY, False, False),
    ],
    "to child": [(GraphLinkLabel.TO_CHILD, True, False)],
    "to parent": [(GraphLinkLabel.TO_PARENT, False, False)],
    "dangling value": [(GraphLinkLabel.TO_VALUE, True, True)],
    "dangling key": [(GraphLinkLabel.TO_KEY, False, True)],
}


class _Builder:
    """
    Builds the Docling document of a generated structure, and gives each text its
    word and each item its Docling reference.
    """

    def __init__(self) -> None:
        """
        Starts an empty Docling document.
        """
        self.document = DoclingDocument(name="generated")
        self.words: list[str] = []
        # The words that are not converted: those of the furniture, of the notes,
        # of the texts in pictures and of the items of tables outside the body
        self.excluded: set[str] = set()
        # The words that the structure can repeat: those of the headers of table
        # columns, which each row repeats, of the cells that span several rows,
        # of the values of key-value regions and forms, which can be linked to
        # several keys, and of the captions of the tables and the pictures in rich
        # cells, which are in the text of the cell and a paragraph of their own
        self.repeated: set[str] = set()
        # The pages that the items are on, and the images of the pictures of
        # the body
        self.pages: set[int] = set()
        self.body_images: set[str] = set()
        # Whether the items being added are in a rich cell
        self._in_rich_cell = False

    def build(self, blocks: list[_Block]) -> DoclingDocument:
        """
        Builds the Docling document.

        :param blocks: The items of the body.
        :type blocks: list[_Block]
        :return: The Docling document.
        :rtype: DoclingDocument
        """
        self._blocks(blocks, None)
        return self.document

    def _text(self, text: _Text, layer: ContentLayer | None = None) -> str:
        """
        Gives a text its word.

        :param text: The text.
        :type text: _Text
        :param layer: The content layer of the text, if not its own, e.g. that of
            the table of a cell.
        :type layer: ContentLayer | None
        :return: The text of its Docling item: the word and the words after it,
            with whitespace around them.
        :rtype: str
        """
        text.word = f"w{len(self.words) + 1}z"
        self.words.append(text.word)
        if (layer or text.layer) != _BODY:
            self.excluded.add(text.word)
        return f"{text.before}{text.word}{text.tail}{text.after}"

    def _provenance_item(self, page: int, box: int) -> ProvenanceItem:
        """
        Returns the provenance of a part of an item, in a box on a page.

        :param page: The number of the page.
        :type page: int
        :param box: The index of the box.
        :type box: int
        :return: The provenance.
        :rtype: ProvenanceItem
        """
        self.pages.add(page)
        return ProvenanceItem(page_no=page, bbox=_bbox(box), charspan=(0, 0))

    def _locate(self, item: DocItem, prov: _Prov) -> None:
        """
        Gives an item its place on the pages.

        :param item: The Docling item.
        :type item: DocItem
        :param prov: The page and the box of each part of the item.
        :type prov: _Prov
        :return: None
        :rtype: None
        """
        item.prov.extend(self._provenance_item(page, box) for page, box in prov)

    @staticmethod
    def _source(text: _Text) -> TrackSource | None:
        """
        Returns the track that a text is from.

        :param text: The text.
        :type text: _Text
        :return: The track, with the voice of the text, or None for a text that is
            not from a track.
        :rtype: TrackSource | None
        """
        if not text.tracked:
            return None
        start, end = _TRACK_TIMES
        return TrackSource(start_time=start, end_time=end, voice=text.voice)

    def _blocks(self, blocks: list[_Block], parent: NodeItem | None) -> None:
        """
        Adds items under a parent.

        :param blocks: The items.
        :type blocks: list[_Block]
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        for block in blocks:
            self._block(block, parent)

    def _block(self, block: _Block, parent: NodeItem | None) -> None:
        """
        Adds an item and the items under it.

        :param block: The item.
        :type block: _Block
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        if isinstance(block, _Text):
            self._add_text(block, parent)
        elif isinstance(block, _Inline):
            self._inline(block, parent)
        elif isinstance(block, _Picture):
            self._picture(block, parent)
        elif isinstance(block, _Code):
            self._code(block, parent)
        elif isinstance(block, _Heading):
            self._heading(block, parent)
        elif isinstance(block, _List):
            group = self.document.add_list_group(parent=parent)
            for entry in block.items:
                if isinstance(entry, _Text):
                    self._add_text(entry, group)
                elif isinstance(entry, _List):
                    self._block(entry, group)
                else:
                    self._list_item(entry, group)
        elif isinstance(block, _ListItem):
            self._list_item(block, parent)
        elif isinstance(block, _Table):
            self._table(block, parent)
        elif isinstance(block, _Graph):
            self._graph(block, parent)
        elif isinstance(block, _Field):
            self._field(block, parent)
        else:
            group = self.document.add_group(
                label=block.label, name=block.name, parent=parent
            )
            self._blocks(block.children, group)

    def _add_text(
        self, text: _Text, parent: NodeItem | None, layer: ContentLayer | None = None
    ) -> DocItem:
        """
        Adds a text item and the items under it.

        :param text: The text.
        :type text: _Text
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :param layer: The content layer of the text, if not its own.
        :type layer: ContentLayer | None
        :return: The text item.
        :rtype: DocItem
        """
        item = self.document.add_text(
            text.label,
            self._text(text, layer),
            parent=parent,
            content_layer=layer or text.layer,
            source=self._source(text),
        )
        self._locate(item, text.prov)
        text.self_ref = item.self_ref
        self._blocks(text.children, item)
        return item

    def _heading(self, heading: _Heading, parent: NodeItem | None) -> None:
        """
        Adds the title or a heading, and the items under it.

        :param heading: The title or the heading.
        :type heading: _Heading
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        text = self._text(heading.own) if heading.own else ""
        if heading.level == 0:
            item = self.document.add_title(text, parent=parent)
        else:
            item = self.document.add_heading(text, level=heading.level, parent=parent)
        self._locate(item, heading.prov)
        heading.self_ref = item.self_ref
        if heading.formatted:
            self._inline(heading.formatted, item)
        if heading.child:
            self._add_text(heading.child, item)
        self._blocks(heading.children, item)

    def _list_item(self, list_item: _ListItem, parent: NodeItem | None) -> None:
        """
        Adds a list item, in a list or not, and the items under it.

        :param list_item: The list item.
        :type list_item: _ListItem
        :param parent: The parent: a list, or another item, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        document = self.document
        text = self._text(list_item.own) if list_item.own else ""
        if isinstance(parent, ListGroup):
            item = document.add_list_item(
                text,
                enumerated=list_item.enumerated,
                marker=list_item.marker,
                parent=parent,
            )
        else:
            # Added by hand, as add_list_item() puts an item outside a list in
            # a new list
            owner = parent or document.body
            item = ListItem(
                text=text,
                orig=text,
                self_ref=f"#/texts/{len(document.texts)}",
                parent=owner.get_ref(),
                enumerated=list_item.enumerated,
                marker=list_item.marker,
            )
            document.texts.append(item)
            owner.children.append(item.get_ref())
        self._locate(item, list_item.prov)
        list_item.self_ref = item.self_ref
        if list_item.formatted:
            self._inline(list_item.formatted, item)
        if list_item.child:
            self._add_text(list_item.child, item)
        self._blocks(list_item.children, item)

    def _inline(self, inline: _Inline, parent: NodeItem | None) -> None:
        """
        Adds an inline group and its parts.

        :param inline: The inline group.
        :type inline: _Inline
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        group = self.document.add_inline_group(parent=parent)
        inline.self_ref = group.self_ref
        for part in inline.parts:
            if isinstance(part, _Text):
                self._add_text(part, group)
            elif isinstance(part, _Picture):
                self._picture(part, group)
            else:
                self._inline(part, group)

    def _picture(self, picture: _Picture, parent: NodeItem | None) -> DocItem:
        """
        Adds a picture, with its caption before it and its footnote after it, and
        the text in it.

        :param picture: The picture.
        :type picture: _Picture
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: The picture item.
        :rtype: DocItem
        """
        document = self.document
        caption = self._add_text(picture.caption, parent) if picture.caption else None
        item = document.add_picture(
            image=_image_ref(picture.color) if picture.color else None,
            parent=parent,
            content_layer=picture.layer,
        )
        self._locate(item, picture.prov)
        picture.self_ref = item.self_ref
        if caption:
            item.captions.append(caption.get_ref())
        if picture.caption:
            self._caption_in_rich_cell(picture.caption, picture.layer)
        if picture.color and picture.layer == _BODY:
            self.body_images.add(_IMAGE_DATA[picture.color])
        if picture.inner_text:
            self._add_text(picture.inner_text, item, picture.layer)
            self.excluded.add(picture.inner_text.word)
        if picture.footnote:
            item.footnotes.append(self._add_text(picture.footnote, parent).get_ref())
        return item

    def _code(self, code: _Code, parent: NodeItem | None) -> None:
        """
        Adds a code block, with its caption before it or after it, and its footnote
        after it, and the items under it.

        :param code: The code block.
        :type code: _Code
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        caption = None
        if code.caption and not code.caption_after:
            caption = self._add_text(code.caption, parent)
        item = self.document.add_code(
            self._text(code.text), parent=parent, source=self._source(code.text)
        )
        self._locate(item, code.text.prov)
        code.text.self_ref = item.self_ref
        self._blocks(code.text.children, item)
        if code.caption and code.caption_after:
            caption = self._add_text(code.caption, parent)
        if caption:
            item.captions.append(caption.get_ref())
        if code.footnote:
            item.footnotes.append(self._add_text(code.footnote, parent).get_ref())

    def _table(self, table: _Table, parent: NodeItem | None) -> None:
        """
        Adds a table, with its caption before it and its footnote after it, and
        the pictures in its cells.

        :param table: The table.
        :type table: _Table
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        caption = None
        if isinstance(table.caption, _Text):
            caption = self._add_text(table.caption, parent)
        elif isinstance(table.caption, _Picture):
            # Converted where the body has it, as it is not a text
            caption = self._picture(table.caption, parent)
        cells = []
        for cell in table.cells:
            if cell.content is not None:
                continue
            text = self._text(cell.text, table.layer) if cell.text else ""
            if cell.text and (cell.column_header or cell.rows > 1):
                self.repeated.add(cell.text.word)
            cells.append(TableCell(text=text, **self._cell_place(cell)))
        item = self.document.add_table(
            data=TableData(
                num_rows=table.rows, num_cols=table.columns, table_cells=cells
            ),
            parent=parent,
            content_layer=table.layer,
        )
        self._locate(item, table.prov)
        table.self_ref = item.self_ref
        if caption:
            item.captions.append(caption.get_ref())
        if isinstance(table.caption, _Text):
            self._caption_in_rich_cell(table.caption, table.layer)
        if table.footnote:
            item.footnotes.append(self._add_text(table.footnote, parent).get_ref())
        for cell in table.cells:
            if cell.content is not None:
                self._rich_cell(cell, item)
        for picture in table.pictures:
            self._picture(picture, item)

    @staticmethod
    def _cell_place(cell: _Cell) -> dict[str, Any]:
        """
        Returns the place of a cell in its table and what kind of cell it is, as
        Docling gives them.

        :param cell: The cell.
        :type cell: _Cell
        :return: The fields of a Docling table cell other than its text.
        :rtype: dict[str, Any]
        """
        return {
            "row_span": cell.rows,
            "col_span": cell.columns,
            "start_row_offset_idx": cell.row,
            "end_row_offset_idx": cell.row + cell.rows,
            "start_col_offset_idx": cell.column,
            "end_col_offset_idx": cell.column + cell.columns,
            "column_header": cell.column_header,
            "row_header": cell.row_header,
            "row_section": cell.row_section,
        }

    def _rich_cell(self, cell: _Cell, table: TableItem) -> None:
        """
        Adds a rich cell to a table: its items, in a group under the table, and
        the cell, which refers to the group. The text of the cell is that of all
        its items, those of a nested table and of the furniture included, without
        the states of its checkboxes or the numbers of its list items, as Docling
        gives it, which the converter does not use.

        :param cell: The rich cell.
        :type cell: _Cell
        :param table: The Docling table.
        :type table: TableItem
        :return: None
        :rtype: None
        """
        assert cell.content is not None
        group = self.document.add_group(name="rich_cell_group", parent=table)
        first_word = len(self.words)
        self._in_rich_cell = True
        self._blocks(cell.content, group)
        self._in_rich_cell = False
        words = self.words[first_word:]
        if cell.column_header or cell.rows > 1:
            self.repeated.update(words)
        self.document.add_table_cell(
            table,
            RichTableCell(
                text="\n".join(words), ref=group.get_ref(), **self._cell_place(cell)
            ),
        )

    def _caption_in_rich_cell(self, caption: _Text, layer: ContentLayer) -> None:
        """
        Notes the caption of a table or a picture in a rich cell, if it is in
        the text of the cell and a paragraph of its own: if it and its item are
        in the body.

        :param caption: The caption.
        :type caption: _Text
        :param layer: The content layer of its table or picture.
        :type layer: ContentLayer
        :return: None
        :rtype: None
        """
        if self._in_rich_cell and caption.layer == _BODY and layer == _BODY:
            self.repeated.add(caption.word)

    def _graph(self, graph: _Graph, parent: NodeItem | None) -> None:
        """
        Adds a key-value region or a form, with its caption before it and its
        footnote after it.

        :param graph: The key-value region or the form.
        :type graph: _Graph
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        document = self.document
        caption = self._add_text(graph.caption, parent) if graph.caption else None
        cells = []
        for index, (cell, is_key) in enumerate(
            zip(graph.cells, graph.keys, strict=True)
        ):
            text = self._text(cell)
            cells.append(
                GraphCell(
                    label=GraphCellLabel.KEY if is_key else GraphCellLabel.VALUE,
                    cell_id=index,
                    text=text,
                    orig=text,
                    prov=self._provenance_item(*cell.prov[0]) if cell.prov else None,
                )
            )
        links = []
        for key, value, kind in graph.links:
            for label, from_key, dangling in _GRAPH_LINKS[kind]:
                source, target = (key, value) if from_key else (value, key)
                links.append(
                    GraphLink(
                        label=label,
                        source_cell_id=source,
                        target_cell_id=len(cells) + 10 if dangling else target,
                    )
                )
            if kind in ("to value", "to key", "both"):
                self.repeated.add(graph.cells[value].word)
        add = document.add_form if graph.form else document.add_key_values
        item = add(graph=GraphData(cells=cells), parent=parent)
        # Added once the graph is validated, which refuses links to missing cells
        item.graph.links.extend(links)
        self._locate(item, graph.prov)
        graph.self_ref = item.self_ref
        if caption:
            item.captions.append(caption.get_ref())
        if graph.footnote:
            item.footnotes.append(self._add_text(graph.footnote, parent).get_ref())

    def _field(self, field_spec: _Field, parent: NodeItem | None) -> None:
        """
        Adds a form field in a field region.

        :param field_spec: The form field.
        :type field_spec: _Field
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        document = self.document
        region = document.add_field_region(parent=parent)
        item = document.add_field_item(parent=region)
        self._locate(item, field_spec.prov)
        field_spec.self_ref = item.self_ref
        if field_spec.number:
            document.add_text(DocItemLabel.MARKER, "1", parent=item)
        key = document.add_field_key(self._text(field_spec.key), parent=item)
        self._locate(key, field_spec.key.prov)
        if field_spec.value or field_spec.checkbox:
            value = document.add_field_value(
                self._text(field_spec.value) if field_spec.value else "", parent=item
            )
            if field_spec.value:
                self._locate(value, field_spec.value.prov)
            checkbox = field_spec.checkbox
            if checkbox:
                box = document.add_text(
                    DocItemLabel.CHECKBOX_SELECTED
                    if checkbox.checked
                    else DocItemLabel.CHECKBOX_UNSELECTED,
                    self._text(checkbox.text) if checkbox.text else "",
                    parent=value,
                )
                if checkbox.text:
                    self._locate(box, checkbox.text.prov)


# ---------------------------------------------------------------------------
# What the document of a generated structure must have
# ---------------------------------------------------------------------------


def _inline_words(inline: _Inline) -> list[_Text]:
    """
    Returns the texts of formatted text, in order, without those of its pictures.

    :param inline: The formatted text.
    :type inline: _Inline
    :return: The texts.
    :rtype: list[_Text]
    """
    texts = []
    for part in inline.parts:
        if isinstance(part, _Text):
            texts.append(part)
        elif isinstance(part, _Inline):
            texts.extend(_inline_words(part))
    return texts


def _inline_pictures(inline: _Inline) -> list[_Picture]:
    """
    Returns the pictures in formatted text, in order.

    :param inline: The formatted text.
    :type inline: _Inline
    :return: The pictures.
    :rtype: list[_Picture]
    """
    pictures = []
    for part in inline.parts:
        if isinstance(part, _Picture):
            pictures.append(part)
        elif isinstance(part, _Inline):
            pictures.extend(_inline_pictures(part))
    return pictures


def _first_in_body(blocks: list[_Block]) -> tuple[int, _Block | None]:
    """
    Returns the first of the items under an item that the body of the document
    has. The caption and the footnote of a table or a picture of the body are not
    where the body has them, but with it, and a table or a picture outside the body
    is not in the body, but its caption and footnote can be.

    :param blocks: The items under the item.
    :type blocks: list[_Block]
    :return: The index of the item among the items, and the item, e.g. a text, or
        the caption of a table outside the body, or None if the body has none of
        them.
    :rtype: tuple[int, _Block | None]
    """
    for index, block in enumerate(blocks):
        if isinstance(block, _Text):
            if block.layer == _BODY:
                return index, block
        elif isinstance(block, _Table | _Picture) and block.layer != _BODY:
            for attached in (block.caption, block.footnote):
                if attached is not None and attached.layer == _BODY:
                    return index, attached
        else:
            return index, block
    return len(blocks), None


def _provenance(parts: Sequence[_Located]) -> list[dict[str, Any]]:
    """
    Returns the provenance that the custom data of a paragraph or an image has:
    the page and the bounding box of each part of the items that it is made from,
    in order, without repetitions.

    :param parts: The items.
    :type parts: Sequence[_Located]
    :return: The page and the bounding box of each part.
    :rtype: list[dict[str, Any]]
    """
    entries: list[dict[str, Any]] = []
    for part in parts:
        for page, box in part.prov:
            entry = {
                "page_no": page,
                "bbox": {
                    "l": 72.0 + 150 * box,
                    "t": 700.0,
                    "r": 192.0 + 150 * box,
                    "b": 680.0,
                    "coord_origin": "BOTTOMLEFT",
                },
            }
            if entry not in entries:
                entries.append(entry)
    return entries


class _Expected:
    """
    Works out the paragraphs and the images of the document of a generated
    structure, in the order that the converter documents: in reading order, with
    the captions of a table, a picture, a code block or a key-value region before
    it and its footnotes after it, those in the body of an item outside it where
    the body has them, the pictures in formatted text after it, and the tables
    and the pictures in the cells of a table after it, those of its rich cells
    with their captions in the text of their cell.
    The section of a paragraph is given by the headings before it, and by
    the titles before it in a document with several titles; the headings in
    a sheet or a slide end with it.
    """

    def __init__(self, blocks: list[_Block]) -> None:
        """
        Works out the paragraphs and the images.

        :param blocks: The items of the body, after the Docling document is built.
        :type blocks: list[_Block]
        """
        # The text, the additional context, the page numbers, the Docling data and
        # the texts of the sentences of each paragraph: its whole text for one that
        # is not running text, and none for the others, which the SaT model
        # segments when sentences are needed
        self.paragraphs: list[
            tuple[str, str | None, list[int], dict[str, Any], list[str]]
        ] = []
        # The base64 data and the Docling data of each image
        self.images: list[tuple[str, dict[str, Any]]] = []
        self.pictures_without_images = 0
        self.structures: set[str] = set()
        # The headings of the current section, a title at level 0
        self._headings: list[tuple[int, str]] = []
        self._containers: list[str] = []
        # The paragraphs, with the headings of their sections, until it is known
        # whether the titles are among them, and the number of titles
        self._pending: list[
            tuple[str, list[str], list[tuple[int, str]], dict[str, Any]]
        ] = []
        self._titles = 0
        self._items(blocks, list_level=0)
        if self._titles > 1:
            self.structures.add("several titles")
        for text, context, headings, docling in self._pending:
            path = [heading for level, heading in headings if level or self._titles > 1]
            if path:
                context = [*context, "Section: " + " > ".join(path)]
            pages = sorted({entry["page_no"] for entry in docling["prov"]})
            sentences = [text] if docling["label"] in _ONE_SENTENCE_LABELS else []
            self.paragraphs.append(
                (text, "; ".join(context) or None, pages, docling, sentences)
            )

    def _paragraph(
        self,
        text: str,
        kind: str | None,
        source: str,
        label: str,
        parts: Sequence[_Located],
        texts: Sequence[_Text] = (),
        row_index: int | None = None,
    ) -> bool:
        """
        Adds a paragraph, unless its text is empty, with its additional context:
        what it is, the speakers of its tracks, the sheets or slides it is on, and
        the headings of its section, and its Docling data.

        :param text: The text of the paragraph.
        :type text: str
        :param kind: What the paragraph is, or None for body text.
        :type kind: str | None
        :param source: The reference of the Docling item it is made from.
        :type source: str
        :param label: The label of that item.
        :type label: str
        :param parts: The items that its provenance is that of, in order.
        :type parts: Sequence[_Located]
        :param texts: The texts it is made from that can be from a track.
        :type texts: Sequence[_Text]
        :param row_index: The index of its row in its table, for a table row.
        :type row_index: int | None
        :return: Whether the paragraph is added.
        :rtype: bool
        """
        if not text:
            return False
        context = [kind] if kind else []
        speakers = list(dict.fromkeys(part.speaker for part in texts if part.speaker))
        if speakers:
            self.structures.add("speakers" if len(speakers) > 1 else "speaker")
            label_of_speakers = "Speaker" if len(speakers) == 1 else "Speakers"
            context.append(f"{label_of_speakers}: {', '.join(speakers)}")
        context += self._containers
        boxes = [place for part in parts for place in part.prov]
        if len(set(boxes)) < len(boxes):
            self.structures.add("box repeated in a paragraph")
        if len({box[0] for box in boxes}) < len(set(boxes)):
            self.structures.add("two boxes on a page in a paragraph")
        docling: dict[str, Any] = {
            "self_ref": source,
            "label": label,
            "prov": _provenance(parts),
        }
        tracks = list(
            dict.fromkeys((*_TRACK_TIMES, part.voice) for part in texts if part.tracked)
        )
        if tracks:
            docling["source"] = tracks
        if row_index is not None:
            docling["row_index"] = row_index
        self._pending.append((text, context, list(self._headings), docling))
        return True

    def _items(self, blocks: list[_Block], list_level: int) -> None:
        """
        Converts items.

        :param blocks: The items.
        :type blocks: list[_Block]
        :param list_level: The nesting level of the list that the items are in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        for block in blocks:
            self._item(block, list_level)

    def _item(self, block: _Block, list_level: int) -> None:
        """
        Converts an item and the items under it, if it is in the body.

        :param block: The item.
        :type block: _Block
        :param list_level: The nesting level of the list that the item is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        if isinstance(block, _Text):
            if block.layer == _BODY:
                self._text(block, list_level)
        elif isinstance(block, _Table | _Picture) and block.layer != _BODY:
            self._outside_body(block, list_level)
        elif isinstance(block, _Inline):
            texts = _inline_words(block)
            if any(text.label != DocItemLabel.TEXT for text in texts):
                self.structures.add("code or formula in formatted text")
            self._paragraph(
                " ".join(text.content for text in texts),
                None,
                block.self_ref,
                GroupLabel.INLINE.value,
                texts,
                texts,
            )
            for picture in _inline_pictures(block):
                self.structures.add("picture in formatted text")
                self._picture(picture)
        elif isinstance(block, _Picture):
            self._picture(block)
        elif isinstance(block, _Code):
            self._code(block, list_level)
        elif isinstance(block, _Heading):
            self._heading(block, list_level)
        elif isinstance(block, _List):
            self._list(block, list_level)
        elif isinstance(block, _ListItem):
            self.structures.add("list item outside a list")
            self._list_item(block, list_level, "")
        elif isinstance(block, _Table):
            self._table(block)
        elif isinstance(block, _Graph):
            self._graph(block)
        elif isinstance(block, _Field):
            self._field(block)
        else:
            self._group(block, list_level)

    def _text(self, text: _Text, list_level: int) -> None:
        """
        Converts a text item of the body, and the items under it.

        :param text: The text.
        :type text: _Text
        :param list_level: The nesting level of the list that the text is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        self._paragraph(
            text.content,
            _TEXT_KINDS.get(text.label),
            text.self_ref,
            text.label.value,
            [text],
            [text],
        )
        if text.children:
            self.structures.add("items under a text")
        self._items(text.children, list_level)

    def _outside_body(self, block: _Table | _Picture, list_level: int) -> None:
        """
        Converts the caption and the footnote of a table or a picture outside
        the body, which are converted as other items where the body has them, if
        they are in it: a text, or a picture given as the caption of a table.

        :param block: The table or the picture.
        :type block: _Table | _Picture
        :param list_level: The nesting level of the list that the item is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        for attached in (block.caption, block.footnote):
            if isinstance(attached, _Text) and attached.layer == _BODY:
                self.structures.add("caption or footnote of an item outside the body")
                self._text(attached, list_level)
            elif isinstance(attached, _Picture):
                self.structures.add("picture caption of a table outside the body")
                self._picture(attached)

    def _attached(self, text: _Text | None, kind: str) -> None:
        """
        Converts the caption or the footnote of an item, if it is a text in
        the body, and the items under it.

        :param text: The caption or the footnote, if the item has one.
        :type text: _Text | None
        :param kind: What the paragraph is, e.g. "Table caption".
        :type kind: str
        :return: None
        :rtype: None
        """
        if text is None:
            return
        if text.layer != _BODY:
            self.structures.add("caption or footnote outside the body")
            return
        self._paragraph(
            text.content, kind, text.self_ref, text.label.value, [text], [text]
        )
        if text.children:
            self.structures.add("items under a caption or a footnote")
        self._items(text.children, list_level=0)

    def _text_of(
        self, own: _Text | None, under: list[_Block]
    ) -> tuple[str, list[_Text], list[_Block]]:
        """
        Returns the text of a heading or a list item: its own, else that of
        the formatted text or the body text first under it in the body.

        :param own: Its own text, if it has one.
        :type own: _Text | None
        :param under: The items under it.
        :type under: list[_Block]
        :return: The text, the texts under it that the text is made from, and the
            items to convert after it: the pictures in that formatted text or
            the items under that text, and the other items under it.
        :rtype: tuple[str, list[_Text], list[_Block]]
        """
        if own is not None:
            return own.content, [], under
        index, first = _first_in_body(under)
        if isinstance(first, _Inline):
            texts = _inline_words(first)
            if _inline_pictures(first):
                self.structures.add("picture in formatted text")
            return (
                " ".join(text.content for text in texts),
                texts,
                [*_inline_pictures(first), *under[index + 1 :]],
            )
        if isinstance(first, _Text) and first.label in _BODY_TEXT_LABELS:
            if first.children:
                self.structures.add("items under the text of a heading or list item")
            return first.content, [first], [*first.children, *under[index + 1 :]]
        return "", [], under

    def _heading(self, heading: _Heading, list_level: int) -> None:
        """
        Converts the title or a heading, which ends the sections of its level and
        below, and the items under it.

        :param heading: The title or the heading.
        :type heading: _Heading
        :param list_level: The nesting level of the list that the heading is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        level = heading.level
        self._headings = [item for item in self._headings if item[0] < level]
        if heading.formatted:
            self.structures.add("formatted heading")
        under: list[_Block] = [
            *([heading.formatted] if heading.formatted else []),
            *([heading.child] if heading.child else []),
            *heading.children,
        ]
        text, texts, after = self._text_of(heading.own, under)
        added = self._paragraph(
            text,
            f"Heading, level {level}" if level else "Title",
            heading.self_ref,
            (DocItemLabel.SECTION_HEADER if level else DocItemLabel.TITLE).value,
            [heading, *texts],
            texts,
        )
        if added:
            self._headings.append((level, text))
            if not level:
                self._titles += 1
        else:
            self.structures.add("heading without text")
        self._items(after, list_level)

    def _list(self, block: _List, list_level: int) -> None:
        """
        Converts a list. A numbered item without a marker is given its position
        among the items of the list, if the first item of the list is numbered.

        :param block: The list.
        :type block: _List
        :param list_level: The nesting level of the list that the list is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        if list_level:
            self.structures.add("nested list")
        for entry, number in self._list_entries(block):
            if isinstance(entry, _Text):
                self.structures.add(
                    "checkbox in a list"
                    if entry.label in _CHECKBOX_STATES
                    else "text in a list"
                )
                self._item(entry, list_level + 1)
            elif isinstance(entry, _List):
                self.structures.add("list in a list")
                self._list(entry, list_level + 1)
            else:
                self._list_item(entry, list_level + 1, number)

    def _list_entries(
        self, block: _List
    ) -> list[tuple[_ListItem | _Text | _List, str]]:
        """
        Returns the entries of a list, each with the number that it is given if it
        is a numbered item without a marker: its position among the items of the
        list, if the first item of the list is numbered.

        :param block: The list.
        :type block: _List
        :return: Each entry, with its number, or an empty string.
        :rtype: list[tuple[_ListItem | _Text | _List, str]]
        """
        items = [item for item in block.items if isinstance(item, _ListItem)]
        numbered = items[0].enumerated
        if not numbered and any(item.enumerated for item in items):
            self.structures.add("numbered items after a bullet")
        entries: list[tuple[_ListItem | _Text | _List, str]] = []
        position = 0
        for entry in block.items:
            number = ""
            if isinstance(entry, _ListItem):
                position += 1
                number = f"{position}." if numbered else ""
            entries.append((entry, number))
        return entries

    def _list_item(self, item: _ListItem, list_level: int, number: str) -> None:
        """
        Converts a list item, and the items under it.

        :param item: The list item.
        :type item: _ListItem
        :param list_level: The nesting level of its list, 0 outside lists.
        :type list_level: int
        :param number: The number of the item if it is numbered without a marker,
            else an empty string.
        :type number: str
        :return: None
        :rtype: None
        """
        text, texts, after = self._list_item_text(item, number)
        level = max(list_level, 1)
        self._paragraph(
            text,
            "List item" if level == 1 else f"List item, level {level}",
            item.self_ref,
            DocItemLabel.LIST_ITEM.value,
            [item, *texts],
            texts,
        )
        self._items(after, list_level)

    def _list_item_text(
        self, item: _ListItem, number: str
    ) -> tuple[str, list[_Text], list[_Block]]:
        """
        Returns the text of a list item (see ``_text_of()``), which starts with its
        marker if it is numbered, or with its number if it has no marker.

        :param item: The list item.
        :type item: _ListItem
        :param number: The number of the item if it is numbered without a marker,
            else an empty string.
        :type number: str
        :return: The text, the texts under the item that it is made from, and
            the items to convert after it.
        :rtype: tuple[str, list[_Text], list[_Block]]
        """
        under: list[_Block] = [
            *([item.formatted] if item.formatted else []),
            *([item.child] if item.child else []),
            *item.children,
        ]
        text, texts, after = self._text_of(item.own, under)
        if texts:
            self.structures.add("list item with the text of an item under it")
        marker = item.marker.strip()
        if item.enumerated and marker:
            self.structures.add("numbered item with its marker")
        elif item.enumerated:
            self.structures.add(
                "numbered item given its position" if number else "unnumbered item"
            )
            marker = number
        if text and item.enumerated and marker:
            text = f"{marker} {text}"
        return text, texts, after

    def _group(self, group: _Group, list_level: int) -> None:
        """
        Converts the items in a group, with the sheet or the slide in their
        context.

        :param group: The group.
        :type group: _Group
        :param list_level: The nesting level of the list that the group is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        container = None
        if group.kind == "sheet":
            if group.name.split():
                container = "Sheet: " + " ".join(group.name.split())
            else:
                self.structures.add("sheet without a name")
        elif group.number is not None:
            container = f"Slide {group.number + 1}"
        elif group.kind == "slide":
            self.structures.add("group that is not a slide")
        # The headings in a sheet or a slide, e.g. its title, end with it
        headings = list(self._headings)
        if container:
            self.structures.add("sheet or slide")
            self._containers.append(container)
        self._items(group.children, list_level)
        if container:
            self._containers.pop()
            if self._headings != headings:
                self.structures.add("heading that ends with its sheet or slide")
            self._headings = headings

    def _picture(self, picture: _Picture) -> None:
        """
        Converts a picture of the body: its caption, its image, unless an earlier
        picture has the same image, and its footnote.

        :param picture: The picture.
        :type picture: _Picture
        :return: None
        :rtype: None
        """
        self._attached(picture.caption, "Picture caption")
        if picture.color is None:
            self.pictures_without_images += 1
        else:
            data = _IMAGE_DATA[picture.color]
            if all(data != image for image, _ in self.images):
                self.images.append(
                    (
                        data,
                        {
                            "self_ref": picture.self_ref,
                            "label": DocItemLabel.PICTURE.value,
                            "prov": _provenance([picture]),
                        },
                    )
                )
            else:
                self.structures.add("repeated image")
        self._attached(picture.footnote, "Picture footnote")

    def _code(self, code: _Code, list_level: int) -> None:
        """
        Converts a code block: its caption, wherever the body has it, its text, its
        footnote, and the items under it.

        :param code: The code block.
        :type code: _Code
        :param list_level: The nesting level of the list that the code is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        if code.caption and code.caption_after:
            self.structures.add("caption after its code block")
        self._attached(code.caption, "Code caption")
        self._paragraph(
            code.text.content,
            "Code",
            code.text.self_ref,
            DocItemLabel.CODE.value,
            [code.text],
            [code.text],
        )
        self._attached(code.footnote, "Code footnote")
        if code.text.children:
            self.structures.add("items under a code block")
        self._items(code.text.children, list_level)

    def _caption_in_rows(self, caption: str) -> str:
        """
        Returns the caption of a table as the context of its rows gives it: cut
        at the last space within its first 97 characters if it is longer than 100,
        without the punctuation before the cut, and followed by "...".

        :param caption: The caption, with single spaces.
        :type caption: str
        :return: The caption, of 100 characters at most.
        :rtype: str
        """
        if len(caption) <= 100:
            return caption
        self.structures.add("long table caption")
        kept: list[str] = []
        length = 0
        for word in caption.split(" "):
            # The word and the space after it are within the 97 characters
            length += len(word) + 1
            if length > 97:
                break
            kept.append(word)
        if kept[-1][-1] in ",;:":
            self.structures.add("long table caption cut after punctuation")
        return " ".join(kept).rstrip(",;:") + "..."

    def _table(self, table: _Table) -> None:
        """
        Converts a table of the body: its caption, its header, a paragraph for
        each other row with the label of each value, its footnote, and the tables
        and the pictures in it.

        The header rows are the rows at the top whose cells with text are all
        column headers. The label of a column is the text of its cells in
        the header rows, from the top, once each. A cell is labelled with the
        labels of its columns, unless it is a row section, or spans the whole
        row under several labels. A label ending with a colon is given without it
        in the rows.

        :param table: The table.
        :type table: _Table
        :return: None
        :rtype: None
        """
        self.structures.add("table")
        caption = ""
        if isinstance(table.caption, _Text):
            if table.caption.layer == _BODY:
                caption = table.caption.content
            self._attached(table.caption, "Table caption")
        elif isinstance(table.caption, _Picture):
            # Not a text: converted where the body has it, before the table
            self.structures.add("picture caption of a table")
            self._picture(table.caption)
        in_caption = (
            f"; Table caption: {self._caption_in_rows(caption)}" if caption else ""
        )
        # The text of each cell, and the tables and the pictures in its rich
        # cells, which are converted after it
        floating: list[_Block] = []
        values = {id(cell): self._cell_text(cell, floating) for cell in table.cells}
        # The cell at each place of the table: a merged cell at each of its places
        grid: list[list[_Cell | None]] = [
            [None] * table.columns for _ in range(table.rows)
        ]
        for cell in table.cells:
            if cell.rows > 1:
                self.structures.add("cell spanning rows")
            if cell.columns > 1:
                self.structures.add("cell spanning columns")
            for row in range(cell.row, cell.row + cell.rows):
                for column in range(cell.column, cell.column + cell.columns):
                    grid[row][column] = cell
        header_rows = 0
        for row_cells in grid:
            with_text = [cell for cell in row_cells if cell and values[id(cell)]]
            if not with_text:
                break
            if not all(cell.column_header for cell in with_text):
                if any(cell.column_header for cell in with_text):
                    self.structures.add("header row with a value")
                break
            header_rows += 1
        labels = []
        for column in range(table.columns if header_rows else 0):
            headers: list[str] = []
            for row_cells in grid[:header_rows]:
                cell = row_cells[column]
                value = values[id(cell)] if cell else ""
                if value and value not in headers:
                    headers.append(value)
            labels.append(" > ".join(headers))
        header = []
        for column, label in enumerate(labels):
            # A label repeated by the next column, under a merged header, is
            # listed once
            if column and label == labels[column - 1]:
                if label:
                    self.structures.add("column label of a merged header")
            elif label:
                header.append(label)
        self._paragraph(
            "; ".join(header),
            "Table header" + in_caption,
            table.self_ref,
            DocItemLabel.TABLE.value,
            [table],
        )
        for index in range(header_rows, table.rows):
            row_values = []
            cells: list[_Cell] = []
            for cell in grid[index]:
                if cell is None or any(cell is other for other in cells):
                    continue
                cells.append(cell)
                value = values[id(cell)]
                if value:
                    label = self._cell_label(cell, labels, table.columns)
                    label = label.rstrip(":").rstrip()
                    row_values.append(
                        f"{label}: {value}" if label and label != value else value
                    )
            self._paragraph(
                "; ".join(row_values),
                "Table row" + in_caption,
                table.self_ref,
                DocItemLabel.TABLE.value,
                [table],
                row_index=index,
            )
        self._attached(table.footnote, "Table footnote")
        for block in floating:
            self._item(block, list_level=0)
        for picture in table.pictures:
            self._picture(picture)

    def _cell_text(self, cell: _Cell, floating: list[_Block]) -> str:
        """
        Returns the text of a cell of a table: its text, or for a rich cell,
        the text of its items.

        :param cell: The cell.
        :type cell: _Cell
        :param floating: The tables and the pictures in the rich cells of
            the table, which the tables and the pictures of the cell are added to.
        :type floating: list[_Block]
        :return: The text, or an empty string.
        :rtype: str
        """
        if cell.content is None:
            return cell.text.content if cell.text else ""
        self.structures.add("rich cell")
        if cell.column_header:
            self.structures.add("rich column header")
        parts = self._cell_items(cell.content, floating)
        return " ".join(part for part in parts if part)

    def _cell_items(
        self, blocks: Sequence[_Block], floating: list[_Block]
    ) -> list[str]:
        """
        Returns the texts of items in a rich cell, in order.

        :param blocks: The items.
        :type blocks: Sequence[_Block]
        :param floating: The tables and the pictures in the rich cells of
            the table, which those among the items are added to.
        :type floating: list[_Block]
        :return: The texts, some of which can be empty.
        :rtype: list[str]
        """
        return [part for block in blocks for part in self._cell_parts(block, floating)]

    def _cell_parts(self, block: _Block, floating: list[_Block]) -> list[str]:
        """
        Returns the texts of an item in a rich cell and of the items under it: of
        a table or a picture of the body, which is converted after the table, its
        caption only, and of one outside the body, which is not converted, its
        caption and footnote in the body, which are items of the cell. A checkbox
        is followed by its state, a numbered list item starts with its number,
        and the caption of a code block is before its text and its footnote after.

        :param block: The item.
        :type block: _Block
        :param floating: The tables and the pictures in the rich cells of
            the table, which the item is added to if it is one.
        :type floating: list[_Block]
        :return: The texts, some of which can be empty.
        :rtype: list[str]
        """
        if isinstance(block, _Table | _Picture):
            if block.layer != _BODY:
                self.structures.add("table or picture outside the body in a rich cell")
                attached = [
                    item for item in (block.caption, block.footnote) if item is not None
                ]
                return self._cell_items(attached, floating)
            self.structures.add(
                "table in a rich cell"
                if isinstance(block, _Table)
                else "picture in a rich cell"
            )
            floating.append(block)
            caption = block.caption
            if isinstance(caption, _Text) and caption.layer == _BODY:
                return [caption.content]
            return []
        if isinstance(block, _Text):
            if block.layer != _BODY:
                return []
            state = _CHECKBOX_STATES.get(block.label)
            if state:
                self.structures.add("checkbox in a rich cell")
            text = f"{block.content} ({state})" if state else block.content
            return [text, *self._cell_items(block.children, floating)]
        if isinstance(block, _Inline):
            text = " ".join(part.content for part in _inline_words(block))
            return [text, *self._cell_items(_inline_pictures(block), floating)]
        if isinstance(block, _Heading):
            self.structures.add("heading in a rich cell")
            under: list[_Block] = [
                *([block.formatted] if block.formatted else []),
                *([block.child] if block.child else []),
                *block.children,
            ]
            text, _, after = self._text_of(block.own, under)
            return [text, *self._cell_items(after, floating)]
        if isinstance(block, _List):
            self.structures.add("list in a rich cell")
            parts = []
            for entry, number in self._list_entries(block):
                if isinstance(entry, _ListItem):
                    text, _, after = self._list_item_text(entry, number)
                    parts += [text, *self._cell_items(after, floating)]
                else:
                    parts += self._cell_parts(entry, floating)
            return parts
        if isinstance(block, _Code):
            self.structures.add("code block in a rich cell")
            attached = [block.caption] if block.caption else []
            return [
                *self._cell_items(attached, floating),
                block.text.content,
                *self._cell_items([block.footnote] if block.footnote else [], floating),
                *self._cell_items(block.text.children, floating),
            ]
        raise AssertionError(f"Not generated in a rich cell: {block}")

    def _cell_label(self, cell: _Cell, labels: list[str], columns: int) -> str:
        """
        Returns the label of a cell of a table: the labels of its columns, unless
        it is a row section, or spans the whole row under several labels.

        :param cell: The cell.
        :type cell: _Cell
        :param labels: The label of each column, if the table has header rows.
        :type labels: list[str]
        :param columns: The number of columns of the table.
        :type columns: int
        :return: The label, or an empty string.
        :rtype: str
        """
        names = list(
            dict.fromkeys(
                labels[column]
                for column in range(cell.column, cell.column + cell.columns)
                if column < len(labels) and labels[column]
            )
        )
        if cell.row_header:
            self.structures.add("row header")
        across = len(names) > 1 and cell.columns == columns
        if cell.row_section:
            if names and not across:
                self.structures.add(
                    "row section under the labels of its columns"
                    if cell.columns == columns
                    else "row section in a part of the row"
                )
            return ""
        if across:
            self.structures.add("cell across the row under several labels")
            return ""
        return ", ".join(names)

    def _graph(self, graph: _Graph) -> None:
        """
        Converts a key-value region or a form: its caption, a paragraph for each
        cell that is not only a value, with its values, on the pages of its cells,
        or of the region where its cells are on none, and its footnote. A key and
        a value are linked from the key to the value, from the value to the key, or
        both; other links, and links to missing cells, are left out.

        :param graph: The key-value region or the form.
        :type graph: _Graph
        :return: None
        :rtype: None
        """
        self.structures.add("key-value region or form")
        name = "Form" if graph.form else "Key-value"
        self._attached(graph.caption, f"{name} caption")
        values_of: dict[int, list[int]] = {}
        values: set[int] = set()
        for key, value, kind in graph.links:
            if kind in ("to value", "to key", "both"):
                self.structures.add(f"value linked {kind}")
                if value not in values_of.setdefault(key, []):
                    values_of[key].append(value)
                values.add(value)
            else:
                self.structures.add(f"link {kind}, left out")
        for index, cell in enumerate(graph.cells):
            linked = [graph.cells[value] for value in values_of.get(index, [])]
            if index in values:
                if not linked:
                    continue
                self.structures.add("value that is the key of other values")
            parts: list[_Located] = [cell, *linked]
            if not any(part.prov for part in parts):
                if graph.prov:
                    self.structures.add("cells on the page of their region")
                parts = [graph]
            self._paragraph(
                f"{cell.content}: {'; '.join(value.content for value in linked)}"
                if linked
                else cell.content,
                "Form field" if graph.form else "Key-value pair",
                graph.self_ref,
                (
                    DocItemLabel.FORM if graph.form else DocItemLabel.KEY_VALUE_REGION
                ).value,
                parts,
            )
        self._attached(graph.footnote, f"{name} footnote")

    def _field(self, field_spec: _Field) -> None:
        """
        Converts a form field: its key, and its value and its checkbox.

        :param field_spec: The form field.
        :type field_spec: _Field
        :return: None
        :rtype: None
        """
        self.structures.add("form field")
        values = []
        parts: list[_Located] = [field_spec, field_spec.key]
        if field_spec.value:
            values.append(field_spec.value.content)
            parts.append(field_spec.value)
        checkbox = field_spec.checkbox
        if checkbox:
            state = "checked" if checkbox.checked else "unchecked"
            if checkbox.text:
                values.append(f"{checkbox.text.content} ({state})")
                parts.append(checkbox.text)
            else:
                values.append(state)
        value = " ".join(values)
        self._paragraph(
            f"{field_spec.key.content}: {value}" if value else field_spec.key.content,
            "Form field",
            field_spec.self_ref,
            DocItemLabel.FIELD_ITEM.value,
            parts,
        )


def _docling_data(custom_data: dict[str, Any]) -> dict[str, Any]:
    """
    Returns the Docling data of the custom data of a paragraph, with each of its
    tracks as its start time, end time and voice.

    :param custom_data: The custom data.
    :type custom_data: dict[str, Any]
    :return: The Docling data.
    :rtype: dict[str, Any]
    """
    data = dict(custom_data["docling"])
    if "source" in data:
        data["source"] = [
            (track["start_time"], track["end_time"], track.get("voice"))
            for track in data["source"]
        ]
    return data


def _count_word(word: str, texts: list[str]) -> int:
    """
    Counts the occurrences of a word in texts.

    :param word: The word.
    :type word: str
    :param texts: The texts.
    :type texts: list[str]
    :return: The number of occurrences.
    :rtype: int
    """
    return sum(len(re.findall(rf"\b{word}\b", text)) for text in texts)


class TestDoclingProperties(BaseTest):
    """
    Property-based tests of the creation of documents from Docling documents.
    """

    @_PROPERTY_SETTINGS
    @given(blocks=_DOCUMENTS)
    def test_from_docling_paragraphs_and_images(self, blocks: list[_Block]):
        """
        Tests that a document created from a Docling document has the paragraphs
        and the images that the structure of the Docling document gives: each
        paragraph with its text, its additional context, its page numbers, its
        Docling data and its sentences, in order, and the image of each picture of
        the body that has one, once, with its Docling data. A paragraph that is not
        running text, e.g. a table row, is one sentence, which has the additional
        context and the pages of the paragraph and a copy of its custom data.
        """
        docling_document = _Builder().build(blocks)
        expected = _Expected(blocks)
        note(pformat(blocks))
        for structure in sorted(expected.structures):
            event(structure)
        if not expected.paragraphs and not expected.images:
            event("nothing to convert")
            with pytest.raises(ValueError, match="has no text or images"):
                Document.from_docling(docling_document)
            return

        with capture_logs() as logs:
            document = Document.from_docling(docling_document)
        assert [
            (
                paragraph.raw_text,
                paragraph.additional_context,
                paragraph.page_numbers,
                _docling_data(paragraph.custom_data),
                [sentence.raw_text for sentence in paragraph.sentences],
            )
            for paragraph in document.paragraphs
        ] == expected.paragraphs
        for paragraph in document.paragraphs:
            for sentence in paragraph.sentences:
                assert sentence.additional_context == paragraph.additional_context
                assert sentence.page_numbers == paragraph.page_numbers
                assert sentence.custom_data == paragraph.custom_data
                assert sentence.custom_data is not paragraph.custom_data
        assert [
            (image.base64_data, image.custom_data["docling"])
            for image in document.images
        ] == expected.images
        # The pictures without images are counted in a message
        not_attached = [message for message in logs if "are not attached" in message]
        if expected.pictures_without_images:
            assert len(not_attached) == 1
            assert not_attached[0].startswith(
                f"{expected.pictures_without_images} picture(s) "
            )
        else:
            assert not not_attached

    @_PROPERTY_SETTINGS
    @given(blocks=_DOCUMENTS)
    def test_from_docling_invariants(self, blocks: list[_Block]):
        """
        Tests what holds for a document created from any Docling document, without
        working out its paragraphs: each text of the body is in the text of one
        paragraph (or of several where the structure repeats it, e.g. the header of
        a table column in each row with a value in the column), no text of
        the furniture, the notes or a picture is in the document, the paragraphs
        are not empty and have contexts on one line and the pages of their items,
        each box once, the speakers in their contexts are those of the tracks in
        their custom data, the images are those of the pictures of the body, once
        each, the Docling document is left as it was, and the document round-trips
        through serialization.
        """
        builder = _Builder()
        docling_document = builder.build(blocks)
        note(pformat(blocks))
        body_words = [word for word in builder.words if word not in builder.excluded]
        if not body_words and not builder.body_images:
            with pytest.raises(ValueError, match="has no text or images"):
                Document.from_docling(docling_document)
            return
        before = docling_document.export_to_dict()

        document = Document.from_docling(docling_document)
        assert docling_document.export_to_dict() == before
        texts = [paragraph.raw_text for paragraph in document.paragraphs]
        contexts = [
            paragraph.additional_context or "" for paragraph in document.paragraphs
        ]
        for word in body_words:
            count = _count_word(word, texts)
            if word in builder.repeated:
                assert count >= 1, word
            else:
                assert count == 1, word
        for word in builder.excluded:
            assert not _count_word(word, texts + contexts), word
        for paragraph in document.paragraphs:
            assert paragraph.raw_text
            assert paragraph.raw_text == paragraph.raw_text.strip()
            context = paragraph.additional_context
            assert context is None or (context and context == " ".join(context.split()))
            assert paragraph.page_numbers == sorted(set(paragraph.page_numbers))
            assert set(paragraph.page_numbers) <= builder.pages
            entries = paragraph.custom_data["docling"]["prov"]
            assert not [
                entry for i, entry in enumerate(entries) if entry in entries[:i]
            ]
            assert paragraph.page_numbers == sorted(
                {entry["page_no"] for entry in entries}
            )
            tracks = paragraph.custom_data["docling"].get("source", [])
            assert all(track["kind"] == "track" for track in tracks)
            voices = [" ".join((track.get("voice") or "").split()) for track in tracks]
            speakers = list(dict.fromkeys(voice for voice in voices if voice))
            label = "Speaker" if len(speakers) == 1 else "Speakers"
            if speakers:
                assert f"{label}: {', '.join(speakers)}" in (context or "")
            else:
                assert "Speaker" not in (context or "")
        images = [image.base64_data for image in document.images]
        assert len(images) == len(set(images))
        assert set(images) == builder.body_images

        restored = Document.from_json(document.to_json())
        assert [
            (p.raw_text, p.additional_context, p.page_numbers, p.custom_data)
            for p in restored.paragraphs
        ] == [
            (p.raw_text, p.additional_context, p.page_numbers, p.custom_data)
            for p in document.paragraphs
        ]
        assert [
            (image.base64_data, image.custom_data) for image in restored.images
        ] == [(image.base64_data, image.custom_data) for image in document.images]
