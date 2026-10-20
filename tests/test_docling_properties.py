# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining the property-based tests of the creation of documents from
Docling documents (``Document.from_docling()``).

Hypothesis generates Docling documents with the structures that Docling gives
the files it converts: the title and headings, also with formatted text, texts
of several kinds, formatted text with pictures in it, nested lists, tables with
captions, footnotes and pictures, pictures, key-value regions and forms, form
fields, sheets and slides, and the furniture and the notes of the pages, each
item on a page or on none. Each text is a word of its own, with whitespace
around it, so that the tests can tell where it went.

One test works out from the generated structure what the document must have:
its paragraphs, in order, with their text, additional context and page numbers,
and its images. Another checks what holds for any Docling document, without
working out the paragraphs: each text of the body is kept, once unless it is
the header of a table column, the rest is left out, and the document round-trips
through serialization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pprint import pformat

import pytest
from docling_core.types.doc import (
    BoundingBox,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    GraphCell,
    GraphCellLabel,
    GraphData,
    GraphLink,
    GraphLinkLabel,
    GroupLabel,
    ImageRef,
    NodeItem,
    ProvenanceItem,
    TableCell,
    TableData,
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


# ---------------------------------------------------------------------------
# The generated structure
# ---------------------------------------------------------------------------


@dataclass
class _Text:
    """
    A text: a text item, or a part of formatted text, a cell or a caption. Its word
    is given when the Docling document is built.
    """

    label: DocItemLabel
    page: int | None
    layer: ContentLayer
    before: str
    after: str
    word: str = ""


@dataclass
class _Picture:
    """
    A picture, with the image of a color or without an image, and the text found
    in it, which is not converted.
    """

    color: str | None
    caption: _Text | None
    footnote: _Text | None
    inner_text: _Text | None
    page: int | None
    layer: ContentLayer
    self_ref: str = ""


@dataclass
class _Inline:
    """
    Formatted text: an inline group of texts, with pictures and inline groups in it.
    """

    parts: list[_Text | _Picture | _Inline]


@dataclass
class _Heading:
    """
    The title (level 0) or a heading, with its own text or with formatted text
    under it, and the items under it, as in HTML. The page is that of the heading
    item.
    """

    level: int
    page: int | None
    own: _Text | None
    formatted: _Inline | None
    children: list[_Block]


@dataclass
class _ListItem:
    """
    A list item, with its own text, formatted text under it, a text item under it,
    or no text, and the items under it, e.g. a nested list. The page is that of
    the list item.
    """

    page: int | None
    own: _Text | None
    formatted: _Inline | None
    child: _Text | None
    children: list[_Block]


@dataclass
class _List:
    """
    A list, bulleted or numbered.
    """

    enumerated: bool
    items: list[_ListItem]


@dataclass
class _Table:
    """
    A table: its header rows and its other rows, with a text or no text in each
    cell, its caption and footnote, and the pictures in its cells.
    """

    columns: int
    header: list[list[_Text | None]]
    rows: list[list[_Text | None]]
    caption: _Text | None
    footnote: _Text | None
    page: int | None
    pictures: list[_Picture]


@dataclass
class _Graph:
    """
    A key-value region or a form: its cells, the key cell that each value cell is
    linked to, if any, and the page of the region.
    """

    form: bool
    cells: list[_Text]
    keys: list[bool]
    value_of: list[int | None]
    page: int | None


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


@dataclass
class _Group:
    """
    A group of items: a sheet, a slide or a plain group, e.g. a section of HTML.
    """

    kind: str
    slide_number: int
    slide_label: GroupLabel
    children: list[_Block]
    name: str = field(default="")


_Block = (
    _Text | _Inline | _Heading | _List | _Table | _Picture | _Graph | _Field | _Group
)


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_PAGES = st.one_of(st.none(), st.integers(min_value=1, max_value=3))
# Unicode whitespace, which the converter strips from the ends of a text
_NOISE = st.sampled_from(["", " ", "\n", " \t", "\u00a0", "\u2003"])


def _texts(
    *labels: DocItemLabel, layer: ContentLayer = ContentLayer.BODY
) -> st.SearchStrategy[_Text]:
    """
    Returns a strategy for texts with one of the labels.

    :param labels: The labels, by default that of body text.
    :type labels: DocItemLabel
    :param layer: The content layer of the texts.
    :type layer: ContentLayer
    :return: The strategy.
    :rtype: st.SearchStrategy[_Text]
    """
    return st.builds(
        _Text,
        label=st.sampled_from(labels or (DocItemLabel.TEXT,)),
        page=_PAGES,
        layer=st.just(layer),
        before=_NOISE,
        after=_NOISE,
    )


def _pictures(layers: st.SearchStrategy[ContentLayer]) -> st.SearchStrategy[_Picture]:
    """
    Returns a strategy for pictures in one of the content layers.

    :param layers: The strategy for the content layer of a picture.
    :type layers: st.SearchStrategy[ContentLayer]
    :return: The strategy.
    :rtype: st.SearchStrategy[_Picture]
    """
    return st.builds(
        _Picture,
        color=st.one_of(st.none(), st.sampled_from(_COLORS)),
        caption=st.one_of(st.none(), _texts(DocItemLabel.CAPTION)),
        footnote=st.one_of(st.none(), _texts(DocItemLabel.FOOTNOTE)),
        inner_text=st.one_of(st.none(), _texts()),
        page=_PAGES,
        layer=layers,
    )


_BODY_PICTURES = _pictures(st.just(ContentLayer.BODY))
# A picture of the furniture is e.g. a logo in the header of each page
_ANY_PICTURES = _pictures(
    st.sampled_from([ContentLayer.BODY] * 3 + [ContentLayer.FURNITURE])
)

_INLINES = st.recursive(
    st.builds(
        _Inline,
        parts=st.lists(st.one_of(_texts(), _BODY_PICTURES), min_size=1, max_size=4),
    ),
    lambda inner: st.builds(
        _Inline,
        parts=st.lists(
            st.one_of(_texts(), _BODY_PICTURES, inner), min_size=1, max_size=4
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
    ),
    _texts(
        DocItemLabel.PAGE_HEADER,
        DocItemLabel.PAGE_FOOTER,
        layer=ContentLayer.FURNITURE,
    ),
    _texts(DocItemLabel.TEXT, layer=ContentLayer.NOTES),
)


@st.composite
def _tables(draw: st.DrawFn) -> _Table:
    """
    Generates a table, whose header rows have a text in one cell at least.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :return: The table.
    :rtype: _Table
    """
    columns = draw(st.integers(min_value=1, max_value=3))
    cells = st.lists(st.one_of(st.none(), _texts()), min_size=columns, max_size=columns)
    header = []
    for _ in range(draw(st.integers(min_value=0, max_value=2))):
        row = draw(cells)
        if all(cell is None for cell in row):
            row[draw(st.integers(min_value=0, max_value=columns - 1))] = draw(_texts())
        header.append(row)
    return _Table(
        columns=columns,
        header=header,
        rows=draw(st.lists(cells, min_size=1, max_size=3)),
        caption=draw(st.one_of(st.none(), _texts(DocItemLabel.CAPTION))),
        footnote=draw(st.one_of(st.none(), _texts(DocItemLabel.FOOTNOTE))),
        page=draw(_PAGES),
        pictures=draw(st.lists(_BODY_PICTURES, max_size=1)),
    )


@st.composite
def _graphs(draw: st.DrawFn) -> _Graph:
    """
    Generates a key-value region or a form, whose value cells are each linked to
    one key cell at most.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :return: The key-value region or the form.
    :rtype: _Graph
    """
    cells = draw(st.lists(_texts(), min_size=1, max_size=4))
    keys = draw(st.lists(st.booleans(), min_size=len(cells), max_size=len(cells)))
    key_indexes = [index for index, is_key in enumerate(keys) if is_key]
    value_of = [
        None
        if is_key or not key_indexes
        else draw(st.one_of(st.none(), st.sampled_from(key_indexes)))
        for is_key in keys
    ]
    return _Graph(
        form=draw(st.booleans()),
        cells=cells,
        keys=keys,
        value_of=value_of,
        page=draw(_PAGES),
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
)


@st.composite
def _headings(draw: st.DrawFn, children: st.SearchStrategy[list[_Block]]) -> _Heading:
    """
    Generates the title or a heading, with its own text or with formatted text.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the heading.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The title or the heading.
    :rtype: _Heading
    """
    formatted = draw(st.booleans())
    return _Heading(
        level=draw(st.integers(min_value=0, max_value=3)),
        page=draw(_PAGES),
        own=None if formatted else draw(_texts()),
        formatted=draw(_INLINES) if formatted else None,
        children=draw(children),
    )


@st.composite
def _list_items(
    draw: st.DrawFn, children: st.SearchStrategy[list[_Block]]
) -> _ListItem:
    """
    Generates a list item, with its own text, formatted text, a text item under it,
    or no text.

    :param draw: Draws a value from a strategy.
    :type draw: st.DrawFn
    :param children: The strategy for the items under the list item.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The list item.
    :rtype: _ListItem
    """
    form = draw(st.sampled_from(["own", "formatted", "child", "none"]))
    return _ListItem(
        page=draw(_PAGES),
        own=draw(_texts()) if form == "own" else None,
        formatted=draw(_INLINES) if form == "formatted" else None,
        child=draw(_texts(*_BODY_TEXT_LABELS)) if form == "child" else None,
        children=draw(children),
    )


def _lists(children: st.SearchStrategy[list[_Block]]) -> st.SearchStrategy[_List]:
    """
    Returns a strategy for lists.

    :param children: The strategy for the items under each list item.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The strategy.
    :rtype: st.SearchStrategy[_List]
    """
    return st.builds(
        _List,
        enumerated=st.booleans(),
        items=st.lists(_list_items(children), min_size=1, max_size=3),
    )


def _groups(children: st.SearchStrategy[list[_Block]]) -> st.SearchStrategy[_Group]:
    """
    Returns a strategy for sheets, slides and plain groups.

    :param children: The strategy for the items in a group.
    :type children: st.SearchStrategy[list[_Block]]
    :return: The strategy.
    :rtype: st.SearchStrategy[_Group]
    """
    return st.builds(
        _Group,
        kind=st.sampled_from(["plain", "sheet", "slide"]),
        slide_number=st.integers(min_value=0, max_value=30),
        # A slide is a chapter in the Docling documents of some versions
        slide_label=st.sampled_from([GroupLabel.SLIDE, GroupLabel.CHAPTER]),
        children=children,
    )


# The items under an item of the leaves: none, as a new list for each item
_NO_CHILDREN = st.builds(list)

_BLOCKS = st.recursive(
    st.one_of(
        _TEXT_BLOCKS,
        _INLINES,
        _tables(),
        _ANY_PICTURES,
        _graphs(),
        _FIELDS,
        _headings(_NO_CHILDREN),
        _lists(_NO_CHILDREN),
    ),
    lambda block: st.one_of(
        _headings(st.lists(block, max_size=3)),
        _lists(st.lists(block, max_size=2)),
        _groups(st.lists(block, max_size=3)),
    ),
    max_leaves=12,
)
_DOCUMENTS = st.lists(_BLOCKS, max_size=6)


# ---------------------------------------------------------------------------
# The Docling document of a generated structure
# ---------------------------------------------------------------------------


class _Builder:
    """
    Builds the Docling document of a generated structure, and gives each text its
    word.
    """

    def __init__(self) -> None:
        """
        Starts an empty Docling document.
        """
        self.document = DoclingDocument(name="generated")
        self.words: list[str] = []
        # The words of the furniture, of the notes and of the texts in pictures
        self.excluded: set[str] = set()
        # The words of the headers of table columns, which each row repeats
        self.header_words: set[str] = set()
        # The pages that the items are on, and the images of the pictures of the body
        self.pages: set[int] = set()
        self.body_images: set[str] = set()
        self._sheets = 0

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
            the picture of a caption.
        :type layer: ContentLayer | None
        :return: The text of its Docling item: the word, with whitespace around it.
        :rtype: str
        """
        text.word = f"w{len(self.words) + 1}z"
        self.words.append(text.word)
        if (layer or text.layer) != ContentLayer.BODY:
            self.excluded.add(text.word)
        return f"{text.before}{text.word}{text.after}"

    def _prov(self, page: int | None) -> ProvenanceItem | None:
        """
        Returns the provenance of an item on a page.

        :param page: The number of the page, or None for an item on no page.
        :type page: int | None
        :return: The provenance, or None.
        :rtype: ProvenanceItem | None
        """
        if page is None:
            return None
        self.pages.add(page)
        return ProvenanceItem(
            page_no=page,
            bbox=BoundingBox(
                l=72, t=700, r=520, b=680, coord_origin=CoordOrigin.BOTTOMLEFT
            ),
            charspan=(0, 0),
        )

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
        document = self.document
        if isinstance(block, _Text):
            document.add_text(
                block.label,
                self._text(block),
                prov=self._prov(block.page),
                parent=parent,
                content_layer=block.layer,
            )
        elif isinstance(block, _Inline):
            self._inline(block, parent)
        elif isinstance(block, _Picture):
            self._picture(block, parent)
        elif isinstance(block, _Heading):
            text = self._text(block.own) if block.own else ""
            if block.level == 0:
                heading = document.add_title(
                    text, prov=self._prov(block.page), parent=parent
                )
            else:
                heading = document.add_heading(
                    text, level=block.level, prov=self._prov(block.page), parent=parent
                )
            if block.formatted:
                self._inline(block.formatted, heading)
            self._blocks(block.children, heading)
        elif isinstance(block, _List):
            group = document.add_list_group(parent=parent)
            for index, item in enumerate(block.items):
                list_item = document.add_list_item(
                    self._text(item.own) if item.own else "",
                    enumerated=block.enumerated,
                    marker=f"{index + 1}." if block.enumerated else "-",
                    prov=self._prov(item.page),
                    parent=group,
                )
                if item.formatted:
                    self._inline(item.formatted, list_item)
                if item.child:
                    document.add_text(
                        item.child.label,
                        self._text(item.child),
                        prov=self._prov(item.child.page),
                        parent=list_item,
                    )
                self._blocks(item.children, list_item)
        elif isinstance(block, _Table):
            self._table(block, parent)
        elif isinstance(block, _Graph):
            cells = []
            for index, (cell, is_key) in enumerate(
                zip(block.cells, block.keys, strict=True)
            ):
                text = self._text(cell)
                cells.append(
                    GraphCell(
                        label=GraphCellLabel.KEY if is_key else GraphCellLabel.VALUE,
                        cell_id=index,
                        text=text,
                        orig=text,
                        prov=self._prov(cell.page),
                    )
                )
            graph = GraphData(
                cells=cells,
                links=[
                    GraphLink(
                        label=GraphLinkLabel.TO_VALUE,
                        source_cell_id=key,
                        target_cell_id=index,
                    )
                    for index, key in enumerate(block.value_of)
                    if key is not None
                ],
            )
            add = document.add_form if block.form else document.add_key_values
            add(graph=graph, prov=self._prov(block.page), parent=parent)
        elif isinstance(block, _Field):
            self._field(block, parent)
        else:
            if block.kind == "sheet":
                self._sheets += 1
                block.name = f"Q{self._sheets}"
                group = document.add_group(
                    label=GroupLabel.SHEET, name=block.name, parent=parent
                )
            elif block.kind == "slide":
                group = document.add_group(
                    label=block.slide_label,
                    name=f"slide-{block.slide_number}",
                    parent=parent,
                )
            else:
                group = document.add_group(label=GroupLabel.SECTION, parent=parent)
            self._blocks(block.children, group)

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
        for part in inline.parts:
            if isinstance(part, _Text):
                self.document.add_text(
                    part.label,
                    self._text(part),
                    prov=self._prov(part.page),
                    parent=group,
                )
            elif isinstance(part, _Picture):
                self._picture(part, group)
            else:
                self._inline(part, group)

    def _picture(self, picture: _Picture, parent: NodeItem | None) -> None:
        """
        Adds a picture, with its caption before it and its footnote after it, and
        the text in it.

        :param picture: The picture.
        :type picture: _Picture
        :param parent: The parent, or None for the body.
        :type parent: NodeItem | None
        :return: None
        :rtype: None
        """
        document = self.document
        layer = picture.layer
        caption = None
        if picture.caption:
            caption = document.add_text(
                DocItemLabel.CAPTION,
                self._text(picture.caption, layer),
                prov=self._prov(picture.caption.page),
                parent=parent,
                content_layer=layer,
            )
        item = document.add_picture(
            image=_image_ref(picture.color) if picture.color else None,
            caption=caption,
            prov=self._prov(picture.page),
            parent=parent,
            content_layer=layer,
        )
        picture.self_ref = item.self_ref
        if picture.color and layer == ContentLayer.BODY:
            self.body_images.add(_IMAGE_DATA[picture.color])
        if picture.inner_text:
            document.add_text(
                DocItemLabel.TEXT,
                self._text(picture.inner_text, layer),
                parent=item,
                content_layer=layer,
            )
            self.excluded.add(picture.inner_text.word)
        if picture.footnote:
            footnote = document.add_text(
                DocItemLabel.FOOTNOTE,
                self._text(picture.footnote, layer),
                prov=self._prov(picture.footnote.page),
                parent=parent,
                content_layer=layer,
            )
            item.footnotes.append(footnote.get_ref())

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
        document = self.document
        caption = None
        if table.caption:
            caption = document.add_text(
                DocItemLabel.CAPTION,
                self._text(table.caption),
                prov=self._prov(table.caption.page),
                parent=parent,
            )
        cells = []
        for row_index, row in enumerate([*table.header, *table.rows]):
            is_header = row_index < len(table.header)
            for column, cell in enumerate(row):
                text = self._text(cell) if cell is not None else ""
                if cell is not None and is_header:
                    self.header_words.add(cell.word)
                cells.append(
                    TableCell(
                        text=text,
                        start_row_offset_idx=row_index,
                        end_row_offset_idx=row_index + 1,
                        start_col_offset_idx=column,
                        end_col_offset_idx=column + 1,
                        column_header=is_header,
                    )
                )
        item = document.add_table(
            data=TableData(
                num_rows=len(table.header) + len(table.rows),
                num_cols=table.columns,
                table_cells=cells,
            ),
            caption=caption,
            prov=self._prov(table.page),
            parent=parent,
        )
        if table.footnote:
            footnote = document.add_text(
                DocItemLabel.FOOTNOTE,
                self._text(table.footnote),
                prov=self._prov(table.footnote.page),
                parent=parent,
            )
            item.footnotes.append(footnote.get_ref())
        for picture in table.pictures:
            self._picture(picture, item)

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
        if field_spec.number:
            document.add_text(DocItemLabel.MARKER, "1", parent=item)
        document.add_field_key(
            self._text(field_spec.key),
            prov=self._prov(field_spec.key.page),
            parent=item,
        )
        if field_spec.value or field_spec.checkbox:
            value = document.add_field_value(
                self._text(field_spec.value) if field_spec.value else "",
                prov=self._prov(field_spec.value.page) if field_spec.value else None,
                parent=item,
            )
            checkbox = field_spec.checkbox
            if checkbox:
                document.add_text(
                    DocItemLabel.CHECKBOX_SELECTED
                    if checkbox.checked
                    else DocItemLabel.CHECKBOX_UNSELECTED,
                    self._text(checkbox.text) if checkbox.text else "",
                    prov=self._prov(checkbox.text.page) if checkbox.text else None,
                    parent=value,
                )


# ---------------------------------------------------------------------------
# What the document of a generated structure must have
# ---------------------------------------------------------------------------


def _in_body(block: _Block) -> bool:
    """
    Tells whether an item is in the body layer, and thus converted.

    :param block: The item.
    :type block: _Block
    :return: Whether the item is converted.
    :rtype: bool
    """
    return not isinstance(block, _Text | _Picture) or (block.layer == ContentLayer.BODY)


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


class _Expected:
    """
    Works out the paragraphs and the images of the document of a generated
    structure, in the order that the converter documents: in reading order, with
    the captions of a table or a picture before it and its footnotes after it,
    and the pictures in formatted text or in a table after it.
    """

    def __init__(self, blocks: list[_Block]) -> None:
        """
        Works out the paragraphs and the images.

        :param blocks: The items of the body, after the Docling document is built.
        :type blocks: list[_Block]
        """
        # The text, the additional context and the page numbers of each paragraph
        self.paragraphs: list[tuple[str, str | None, list[int]]] = []
        # The base64 data and the Docling reference of each image
        self.images: list[tuple[str, str]] = []
        self.pictures_without_images = 0
        self.structures: set[str] = set()
        self._headings: list[tuple[int, str]] = []
        self._containers: list[str] = []
        self._items(blocks, list_level=0)

    def _paragraph(self, text: str, kind: str | None, pages: list[int | None]) -> None:
        """
        Adds a paragraph, unless its text is empty, with its additional context:
        what it is, the sheets or slides it is on, and the headings of its section.

        :param text: The text of the paragraph.
        :type text: str
        :param kind: What the paragraph is, or None for body text.
        :type kind: str | None
        :param pages: The pages of the items it is made from, None for an item on
            no page.
        :type pages: list[int | None]
        :return: None
        :rtype: None
        """
        if not text:
            return
        context = ([kind] if kind else []) + self._containers
        if self._headings:
            context.append("Section: " + " > ".join(text for _, text in self._headings))
        self.paragraphs.append(
            (
                text,
                "; ".join(context) or None,
                sorted({page for page in pages if page is not None}),
            )
        )

    def _items(self, blocks: list[_Block], list_level: int) -> None:
        """
        Converts the items of the body layer.

        :param blocks: The items.
        :type blocks: list[_Block]
        :param list_level: The nesting level of the list that the items are in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        for block in blocks:
            if _in_body(block):
                self._item(block, list_level)

    def _item(self, block: _Block, list_level: int) -> None:
        """
        Converts an item and the items under it.

        :param block: The item.
        :type block: _Block
        :param list_level: The nesting level of the list that the item is in.
        :type list_level: int
        :return: None
        :rtype: None
        """
        if isinstance(block, _Text):
            self._paragraph(block.word, _TEXT_KINDS.get(block.label), [block.page])
        elif isinstance(block, _Inline):
            texts = _inline_words(block)
            self._paragraph(
                " ".join(text.word for text in texts),
                None,
                [text.page for text in texts],
            )
            self._floating_pictures(block)
        elif isinstance(block, _Picture):
            self._picture(block)
        elif isinstance(block, _Heading):
            self._heading(block, list_level)
        elif isinstance(block, _List):
            if list_level:
                self.structures.add("nested list")
            for index, item in enumerate(block.items):
                marker = f"{index + 1}." if block.enumerated else ""
                self._list_item(item, list_level + 1, marker)
        elif isinstance(block, _Table):
            self._table(block)
        elif isinstance(block, _Graph):
            self._graph(block)
        elif isinstance(block, _Field):
            self._field(block)
        else:
            container = None
            if block.kind == "sheet":
                container = f"Sheet: {block.name}"
            elif block.kind == "slide":
                container = f"Slide {block.slide_number + 1}"
            if container:
                self.structures.add("sheet or slide")
                self._containers.append(container)
            self._items(block.children, list_level)
            if container:
                self._containers.pop()

    def _floating_pictures(self, inline: _Inline) -> None:
        """
        Converts the pictures in formatted text, after its text.

        :param inline: The formatted text.
        :type inline: _Inline
        :return: None
        :rtype: None
        """
        for picture in _inline_pictures(inline):
            self.structures.add("picture in formatted text")
            self._picture(picture)

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
        pages: list[int | None] = [heading.page]
        if heading.formatted:
            self.structures.add("formatted heading")
            texts = _inline_words(heading.formatted)
            text = " ".join(item.word for item in texts)
            pages += [item.page for item in texts]
        else:
            assert heading.own is not None
            text = heading.own.word
        self._paragraph(text, f"Heading, level {level}" if level else "Title", pages)
        if level and text:
            self._headings.append((level, text))
        if heading.formatted:
            self._floating_pictures(heading.formatted)
        self._items(heading.children, list_level)

    def _list_item(self, item: _ListItem, list_level: int, marker: str) -> None:
        """
        Converts a list item, whose text is its own, or that of the formatted text
        or the body text first under it, and the items under it.

        :param item: The list item.
        :type item: _ListItem
        :param list_level: The nesting level of its list.
        :type list_level: int
        :param marker: The number of a numbered list item, else an empty string.
        :type marker: str
        :return: None
        :rtype: None
        """
        children = [child for child in item.children if _in_body(child)]
        pages: list[int | None] = [item.page]
        text = ""
        pictures: list[_Picture] = []
        if item.own:
            text = item.own.word
        elif item.child:
            text = item.child.word
            pages.append(item.child.page)
        else:
            inline = item.formatted
            first = children[0] if children else None
            if inline is None and isinstance(first, _Inline):
                self.structures.add("list item with the text of its first item")
                inline = first
                children = children[1:]
            elif (
                inline is None
                and isinstance(first, _Text)
                and first.label in _BODY_TEXT_LABELS
            ):
                self.structures.add("list item with the text of its first item")
                text = first.word
                pages.append(first.page)
                children = children[1:]
            if inline is not None:
                texts = _inline_words(inline)
                text = " ".join(part.word for part in texts)
                pages += [part.page for part in texts]
                pictures = _inline_pictures(inline)
        if text and marker:
            text = f"{marker} {text}"
        level = max(list_level, 1)
        self._paragraph(
            text, "List item" if level == 1 else f"List item, level {level}", pages
        )
        for picture in pictures:
            self.structures.add("picture in formatted text")
            self._picture(picture)
        for child in children:
            self._item(child, list_level)

    def _picture(self, picture: _Picture) -> None:
        """
        Converts a picture: its caption, its image, unless an earlier picture has
        the same image, and its footnote.

        :param picture: The picture.
        :type picture: _Picture
        :return: None
        :rtype: None
        """
        if picture.caption:
            self._paragraph(
                picture.caption.word, "Picture caption", [picture.caption.page]
            )
        if picture.color is None:
            self.pictures_without_images += 1
        else:
            data = _IMAGE_DATA[picture.color]
            if all(data != image for image, _ in self.images):
                self.images.append((data, picture.self_ref))
            else:
                self.structures.add("repeated image")
        if picture.footnote:
            self._paragraph(
                picture.footnote.word, "Picture footnote", [picture.footnote.page]
            )

    def _table(self, table: _Table) -> None:
        """
        Converts a table: its caption, its header, a paragraph for each row with
        the label of each value, its footnote, and the pictures in it.

        :param table: The table.
        :type table: _Table
        :return: None
        :rtype: None
        """
        self.structures.add("table")
        caption = table.caption.word if table.caption else ""
        if table.caption:
            self._paragraph(caption, "Table caption", [table.caption.page])
        in_caption = f"; Table caption: {caption}" if caption else ""
        labels = [
            " > ".join(
                cell.word
                for cell in (row[column] for row in table.header)
                if cell is not None
            )
            for column in range(table.columns)
        ]
        self._paragraph(
            "; ".join(label for label in labels if label),
            "Table header" + in_caption,
            [table.page],
        )
        for row in table.rows:
            self._paragraph(
                "; ".join(
                    f"{label}: {cell.word}" if label else cell.word
                    for label, cell in zip(labels, row, strict=True)
                    if cell
                ),
                "Table row" + in_caption,
                [table.page],
            )
        if table.footnote:
            self._paragraph(
                table.footnote.word, "Table footnote", [table.footnote.page]
            )
        for picture in table.pictures:
            self._picture(picture)

    def _graph(self, graph: _Graph) -> None:
        """
        Converts a key-value region or a form: a paragraph for each cell that is
        not the value of a key, with its values, on the pages of its cells, or of
        the region where its cells are on none.

        :param graph: The key-value region or the form.
        :type graph: _Graph
        :return: None
        :rtype: None
        """
        self.structures.add("key-value region or form")
        for index, cell in enumerate(graph.cells):
            if graph.value_of[index] is not None:
                continue
            values = [
                graph.cells[value]
                for value, key in enumerate(graph.value_of)
                if key == index
            ]
            pages: list[int | None] = [
                part.page for part in (cell, *values) if part.page is not None
            ]
            if not pages and graph.page is not None:
                self.structures.add("cells on the page of their region")
            self._paragraph(
                f"{cell.word}: {'; '.join(value.word for value in values)}"
                if values
                else cell.word,
                "Form field" if graph.form else "Key-value pair",
                pages or [graph.page],
            )

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
        pages = [field_spec.key.page]
        if field_spec.value:
            values.append(field_spec.value.word)
            pages.append(field_spec.value.page)
        checkbox = field_spec.checkbox
        if checkbox:
            state = "checked" if checkbox.checked else "unchecked"
            if checkbox.text:
                values.append(f"{checkbox.text.word} ({state})")
                pages.append(checkbox.text.page)
            else:
                values.append(state)
        value = " ".join(values)
        self._paragraph(
            f"{field_spec.key.word}: {value}" if value else field_spec.key.word,
            "Form field",
            pages,
        )


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
        paragraph with its text, its additional context and its page numbers, in
        order, and the image of each picture of the body that has one, once.
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
            )
            for paragraph in document.paragraphs
        ] == expected.paragraphs
        assert [
            (image.base64_data, image.custom_data["docling"]["self_ref"])
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
        paragraph (that of the header of a table column also in each row with
        a value in the column), no text of the furniture, the notes or a picture is
        in the document, the paragraphs are not empty and have contexts on one line
        and the pages of their items, the images are those of the pictures of the
        body, once each, the Docling document is left as it was, and the document
        round-trips through serialization.
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
            if word in builder.header_words:
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
            assert paragraph.page_numbers == sorted(
                {entry["page_no"] for entry in paragraph.custom_data["docling"]["prov"]}
            )
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
