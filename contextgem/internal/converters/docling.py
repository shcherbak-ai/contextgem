# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module for converting Docling documents into the paragraphs and images of
a ContextGem document.

A Docling document (``DoclingDocument`` of ``docling-core``) holds what Docling found
in a file: its headings, text, lists, tables, pictures and form fields, in reading
order, each with the page and the bounding box that it was found at. This module
walks the body of such a document and makes a paragraph of each piece of text. The
additional context of the paragraph describes its place in the structure of the
document, and its custom data keeps its Docling reference and provenance.

``docling-core`` is an optional dependency, installed with the ``docling`` extra.
This module is imported only when a Docling document is converted. It checks
first that ``docling-core`` is installed, in a version that it works with: it can
also be installed without the extra, e.g. with an older Docling.
"""

from __future__ import annotations

import base64
import importlib.metadata
import io
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from PIL import Image as PILImage

from contextgem.internal.base.images import _Image
from contextgem.internal.base.paras_and_sents import _Paragraph
from contextgem.internal.loggers import logger
from contextgem.internal.registry import _publicize
from contextgem.internal.utils import _is_text_content_empty


# The oldest docling-core that the converter works with: the lower bound of
# docling-core in the `docling` extra of pyproject.toml, which a test compares, and
# with which `uv run fab docling-floor` runs the tests. It is the first that reads
# the image of a picture from a local file, given by a path or a file:// URI, only
# after an opt-in (`DoclingDocument.load_local_images()`): 2.102.0 reads one given
# by a path, from a document that could come from anyone.
_MIN_DOCLING_CORE_VERSION = "2.103.0"

# How to install or upgrade docling-core, with Docling, for the errors below
_INSTALL_DOCLING_EXTRA = (
    'uv add "contextgem[docling]", or pip install -U "contextgem[docling]"'
)


def _release_numbers(version: str) -> tuple[int, ...] | None:
    """
    Returns the release numbers of a version, e.g. (2, 87, 0) for "2.87.0rc1".

    :param version: The version.
    :type version: str
    :return: The numbers, or None for a version that does not start with them.
    :rtype: tuple[int, ...] | None
    """
    match = re.match(r"\d+(?:\.\d+)*", version)
    return tuple(int(number) for number in match.group().split(".")) if match else None


def _check_docling_core() -> None:
    """
    Checks that docling-core is installed, in a version that the converter works
    with.

    :return: None
    :rtype: None
    :raises ImportError: If docling-core is not installed, or is older than
        the converter needs.
    """
    try:
        installed = importlib.metadata.version("docling-core")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            "Converting a Docling document requires docling-core, the document "
            "model of Docling, which the `docling` extra of ContextGem installs "
            f"with Docling: {_INSTALL_DOCLING_EXTRA}"
        ) from None
    numbers = _release_numbers(installed)
    minimum = _release_numbers(_MIN_DOCLING_CORE_VERSION)
    if numbers is not None and minimum is not None and numbers < minimum:
        raise ImportError(
            f"Converting a Docling document requires docling-core "
            f"{_MIN_DOCLING_CORE_VERSION} or later, and docling-core {installed} is "
            f"installed. The `docling` extra of ContextGem upgrades it, with "
            f"Docling: {_INSTALL_DOCLING_EXTRA}"
        )


_check_docling_core()

from docling_core.types.doc import (  # noqa: E402 - imported once its version is checked
    CodeItem,
    ContentLayer,
    DocItemLabel,
    DoclingDocument,
    FieldItem,
    FormItem,
    GraphLinkLabel,
    GroupItem,
    ImageRef,
    InlineGroup,
    KeyValueItem,
    ListGroup,
    ListItem,
    NodeItem,
    PageItem,
    PictureItem,
    ProvenanceItem,
    RefItem,
    RichTableCell,
    SectionHeaderItem,
    TableCell,
    TableItem,
    TextItem,
    TrackSource,
)


# The Docling labels of body text, which the additional context of a paragraph
# does not name
_BODY_TEXT_LABELS = frozenset({DocItemLabel.TEXT, DocItemLabel.PARAGRAPH})

# What the additional context of a paragraph calls a text, by its Docling label.
# A text with another label is named after the label, e.g. "Footnote" or "Code".
_TEXT_KINDS = {
    DocItemLabel.CHECKBOX_SELECTED: "Checkbox, checked",
    DocItemLabel.CHECKBOX_UNSELECTED: "Checkbox, unchecked",
}

# The states of checkboxes, by their Docling label, as the text of a form field or
# of a table cell gives them
_CHECKBOX_VALUES = {
    DocItemLabel.CHECKBOX_SELECTED: "checked",
    DocItemLabel.CHECKBOX_UNSELECTED: "unchecked",
}

# Why the pictures that are not attached as images are not, by where their image
# is (see `_missing_image_source()`), for the log of the conversion
_MISSING_IMAGE_REASONS = {
    "none": (
        "as the Docling document has no image data for them. Docling keeps "
        "the images of the pictures of a PDF file with "
        "`generate_picture_images=True` in its PDF pipeline options, and reads "
        "those of an HTML or Markdown file with `fetch_images=True` in the options "
        "of its backend, with `enable_local_fetch=True` for images in local files "
        "or `enable_remote_fetch=True` for images at URLs."
    ),
    "file": (
        "as their images are in local files, which docling-core reads only for "
        "a Docling document from a trusted source: load it with "
        "`DoclingDocument.load_from_json(path, allow_local_images=True)`, or call "
        "`docling_document.load_local_images()` with `base_dir` set to the folder "
        "that the paths are relative to, e.g. that of the JSON file saved by "
        "Docling with `image_mode=ImageRefMode.REFERENCED`."
    ),
    "url": "as their images are at URLs, which are not fetched.",
    "data": "as their image data cannot be read, e.g. that of an SVG image.",
}

# The extensions of the image files that Docling converts (its "image" input
# format), by which a Docling document is known to be made of an image file:
# Docling gives such a document the MIME type of a PDF file, and keeps the name of
# the file
_IMAGE_FILE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
)

# Why the pages of an image file are not attached as images, for the log of the
# conversion
_MISSING_PAGE_IMAGES_REASON = (
    "as the Docling document has no images of its pages. Docling keeps them with "
    "`generate_page_images=True` in the pipeline options of its image format."
)

# The parts of an inline text are joined with a space, except before closing
# punctuation and after an opening bracket
_NO_SPACE_BEFORE = tuple(",.;:!?)]}")
_NO_SPACE_AFTER = tuple("([{")

# The name that Docling gives the group of a slide, e.g. "slide-0" for the first one
_SLIDE_GROUP_NAME = re.compile(r"slide-(\d+)")

# The length that the caption of a table is shortened to in the additional context
# of its rows, which repeats it for each row. The whole caption is a paragraph of
# its own before the rows.
_MAX_CAPTION_LENGTH_IN_ROWS = 100


def _clean(text: str) -> str:
    """
    Collapses the whitespace of a text, line breaks included, to single spaces.

    :param text: The text to clean.
    :type text: str
    :return: The text on a single line, without leading or trailing whitespace.
    :rtype: str
    """
    return " ".join(text.split())


def _shorten(text: str, max_length: int) -> str:
    """
    Shortens a text to a maximum length, at the end of a word.

    :param text: The text, on a single line.
    :type text: str
    :param max_length: The maximum length of the text, the ellipsis included.
    :type max_length: int
    :return: The text, or its beginning followed by "...".
    :rtype: str
    """
    if len(text) <= max_length:
        return text
    shortened = text[: max_length - 3].rsplit(" ", 1)[0] or text[: max_length - 3]
    return shortened.rstrip(",;:") + "..."


def _labelled(label: str, value: str) -> str:
    """
    Returns a value after its label, e.g. the value of a table cell after the header
    of its column, or the value of a form field after its key.

    :param label: The label, which may end with a colon, e.g. "Full name:".
    :type label: str
    :param value: The value.
    :type value: str
    :return: e.g. "Full name: Ada Lovelace", or the value or the label alone if the
        other one is empty, or the value alone if it is the label.
    :rtype: str
    """
    label = label.strip().rstrip(":").rstrip()
    if label and value and label != value:
        return f"{label}: {value}"
    return value or label


def _checkbox_text(label: DocItemLabel, text: str) -> str:
    """
    Returns the text of a checkbox with its state, or the text of another item as
    it is.

    :param label: The Docling label of the item, e.g. that of a checked checkbox.
    :type label: DocItemLabel
    :param text: The text of the item.
    :type text: str
    :return: e.g. "Tea (unchecked)" for a checkbox, or its state alone if it has no
        text, e.g. "checked".
    :rtype: str
    """
    state = _CHECKBOX_VALUES.get(label)
    if state is None:
        return text
    return f"{text} ({state})" if text else state


def _join_inline(parts: Iterable[str]) -> str:
    """
    Joins the parts of an inline text, e.g. the differently formatted runs of
    a sentence, into one text.

    :param parts: The parts of the text, in order.
    :type parts: Iterable[str]
    :return: The parts joined with a space, except before closing punctuation and
        after an opening bracket.
    :rtype: str
    """
    text = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if (
            text
            and not part.startswith(_NO_SPACE_BEFORE)
            and not text.endswith(_NO_SPACE_AFTER)
        ):
            text += " "
        text += part
    return text


def _label_of(item: NodeItem) -> str:
    """
    Returns the Docling label of an item.

    :param item: The Docling item.
    :type item: NodeItem
    :return: The label, e.g. "list_item" or "table", or an empty string for an item
        without one.
    :rtype: str
    """
    label = getattr(item, "label", None)
    return str(getattr(label, "value", label or ""))


def _provenance_entry(prov: ProvenanceItem) -> dict[str, Any]:
    """
    Returns the page and the bounding box of a Docling provenance item.

    :param prov: The provenance item.
    :type prov: ProvenanceItem
    :return: The page number and the bounding box, as JSON-serializable data.
    :rtype: dict[str, Any]
    """
    box = prov.bbox
    return {
        "page_no": prov.page_no,
        "bbox": {
            "l": box.l,
            "t": box.t,
            "r": box.r,
            "b": box.b,
            "coord_origin": str(box.coord_origin.value),
        },
    }


def _provenance(provs: Iterable[object]) -> list[dict[str, Any]]:
    """
    Returns the pages and the bounding boxes of Docling provenance items.

    :param provs: The provenance items. Other objects, e.g. ``None`` for a part
        without provenance, are skipped.
    :type provs: Iterable[object]
    :return: The page and the bounding box of each item, without repetitions.
    :rtype: list[dict[str, Any]]
    """
    entries: list[dict[str, Any]] = []
    for prov in provs:
        if isinstance(prov, ProvenanceItem):
            entry = _provenance_entry(prov)
            if entry not in entries:
                entries.append(entry)
    return entries


def _provenance_of_items(items: Iterable[NodeItem]) -> list[dict[str, Any]]:
    """
    Returns the pages and the bounding boxes of Docling items.

    :param items: The Docling items. Groups have no provenance of their own.
    :type items: Iterable[NodeItem]
    :return: The page and the bounding box of each part of the items.
    :rtype: list[dict[str, Any]]
    """
    return _provenance(prov for item in items for prov in getattr(item, "prov", []))


def _tracks_of_items(items: Iterable[NodeItem]) -> list[dict[str, Any]]:
    """
    Returns the tracks that Docling items are from: the cues of a subtitle file,
    e.g. WebVTT, or the segments of a transcript.

    :param items: The Docling items. Groups have no tracks of their own.
    :type items: Iterable[NodeItem]
    :return: The start and the end time of each track, in seconds, and its
        identifier and voice (the speaker) where it has them, as JSON-serializable
        data, without repetitions.
    :rtype: list[dict[str, Any]]
    """
    entries: list[dict[str, Any]] = []
    for item in items:
        for source in getattr(item, "source", None) or []:
            if isinstance(source, TrackSource):
                entry = source.model_dump(mode="json", exclude_none=True)
                if entry not in entries:
                    entries.append(entry)
    return entries


def _count_header_rows(grid: Sequence[Sequence[TableCell]]) -> int:
    """
    Counts the rows at the top of a table whose cells are column headers.

    :param grid: The cells of the table, by row and column.
    :type grid: Sequence[Sequence[TableCell]]
    :return: The number of header rows.
    :rtype: int
    """
    count = 0
    for row in grid:
        cells = [cell for cell in row if cell.text.strip()]
        if not cells or not all(cell.column_header for cell in cells):
            break
        count += 1
    return count


def _column_labels(header_rows: Sequence[Sequence[TableCell]]) -> list[str]:
    """
    Returns the label of each column of a table, from its header rows.

    :param header_rows: The header rows of the table.
    :type header_rows: Sequence[Sequence[TableCell]]
    :return: The headers of each column, from the top, joined with " > ", e.g.
        "Fees > 2026" under a header that spans several columns.
    :rtype: list[str]
    """
    labels = []
    for column in range(max((len(row) for row in header_rows), default=0)):
        headers: list[str] = []
        for row in header_rows:
            header = _clean(row[column].text) if column < len(row) else ""
            # A header that spans several rows is in each of them
            if header and header not in headers:
                headers.append(header)
        labels.append(" > ".join(headers))
    return labels


def _cell_label(cell: TableCell, labels: Sequence[str]) -> str:
    """
    Returns the label of a table cell, from the labels of the columns it spans.

    :param cell: The cell.
    :type cell: TableCell
    :param labels: The label of each column of the table.
    :type labels: Sequence[str]
    :return: The labels of the columns of the cell, e.g. "Monthly fee", or "Q1, Q2"
        for a cell that spans two columns. A cell that spans the row under different
        labels, as a section of a table does, has no label.
    :rtype: str
    """
    if cell.row_section:
        return ""
    columns = range(
        cell.start_col_offset_idx, min(cell.end_col_offset_idx, len(labels))
    )
    names = list(dict.fromkeys(labels[column] for column in columns if labels[column]))
    if len(names) > 1 and len(columns) >= len(labels):
        return ""
    return ", ".join(names)


def _row_text(row: Sequence[TableCell], labels: Sequence[str]) -> str:
    """
    Returns the text of a table row, with each value after the label of its column.

    :param row: The cells of the row.
    :type row: Sequence[TableCell]
    :param labels: The label of each column, if the table has headers.
    :type labels: Sequence[str]
    :return: The values of the row, e.g. "Service: Hosting; Monthly fee: 1,200 EUR".
        A value whose column has no label is given without one.
    :rtype: str
    """
    parts = []
    starts = set()
    for cell in row:
        # A cell that spans several columns is in each of them
        start = (cell.start_row_offset_idx, cell.start_col_offset_idx)
        if start in starts:
            continue
        starts.add(start)
        value = _clean(cell.text)
        if not value:
            continue
        parts.append(_labelled(_cell_label(cell, labels), value))
    return "; ".join(parts)


def _image_type(data: bytes) -> str | None:
    """
    Returns the MIME type of an image, by its first bytes, if it is of a type that
    a ContextGem document holds.

    :param data: The bytes of the image.
    :type data: bytes
    :return: "image/png", "image/jpeg" or "image/webp", or None for data of another
        type.
    :rtype: str | None
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_data(
    picture: PictureItem, document: DoclingDocument
) -> tuple[str, str] | None:
    """
    Returns the image data of a picture.

    An image embedded in the Docling document is taken as it is, with the type that
    its data has, if it is of a type that a ContextGem document holds, else
    converted to PNG. A picture without an embedded image is cropped from the image
    of its page, if the Docling document has the images of its pages. An image
    referenced by a remote URL is not fetched, and an image in a local file is read
    only if Docling allows it (see ``DoclingDocument.load_local_images()``). An
    image larger than Pillow decodes (``PIL.Image.MAX_IMAGE_PIXELS``) is not read.

    :param picture: The picture.
    :type picture: PictureItem
    :param document: The Docling document of the picture.
    :type document: DoclingDocument
    :return: The MIME type and the base64-encoded data of the image, or None if the
        Docling document has no image data for the picture that can be read.
    :rtype: tuple[str, str] | None
    """
    image_ref = picture.image
    if image_ref is not None and not isinstance(image_ref.uri, Path):
        header, _, data = str(image_ref.uri).partition(",")
        if header.startswith("data:") and not data:
            # Docling fails to read a data URI without data
            return None
    embedded = _embedded_image_data(image_ref)
    if embedded is not None:
        return embedded
    try:
        image = picture.get_image(document)
        return None if image is None else _png_data(image)
    except (OSError, ValueError, PILImage.DecompressionBombError) as e:
        logger.debug(f"Image of the picture {picture.self_ref} not read: {e}")
        return None


def _page_image_data(page: PageItem) -> tuple[str, str] | None:
    """
    Returns the image data of a page, which the Docling document of a file in
    a format with pages has if it was converted with ``generate_page_images=True``.

    An image embedded in the Docling document is taken as it is, with the type that
    its data has, if it is of a type that a ContextGem document holds, else
    converted to PNG, as the image of a picture is (see ``_image_data()``).

    :param page: The page.
    :type page: PageItem
    :return: The MIME type and the base64-encoded data of the image, or None if the
        page has no image data that can be read.
    :rtype: tuple[str, str] | None
    """
    if page.image is None:
        return None
    embedded = _embedded_image_data(page.image)
    if embedded is not None:
        return embedded
    try:
        image = page.image.pil_image
        return None if image is None else _png_data(image)
    except (OSError, ValueError, PILImage.DecompressionBombError) as e:
        logger.debug(f"Image of page {page.page_no} not read: {e}")
        return None


def _embedded_image_data(image_ref: ImageRef | None) -> tuple[str, str] | None:
    """
    Returns the data of an image embedded in a Docling document as a data URI, as
    it is, if it is of a type that a ContextGem document holds. The type that the
    data URI gives is not relied on, as a model provider can refuse an image whose
    data is of another type.

    :param image_ref: The reference of the image, if any.
    :type image_ref: ImageRef | None
    :return: The MIME type and the base64-encoded data of the image, or None if it
        is not embedded, its data is not valid base64, or it is of another type,
        which Pillow converts.
    :rtype: tuple[str, str] | None
    """
    if image_ref is None or isinstance(image_ref.uri, Path):
        return None
    header, _, data = str(image_ref.uri).partition(",")
    if not (header.startswith("data:") and header.endswith(";base64") and data):
        return None
    try:
        mime_type = _image_type(base64.b64decode(data, validate=True))
    except ValueError:
        return None
    return None if mime_type is None else (mime_type, data)


def _png_data(image: PILImage.Image) -> tuple[str, str]:
    """
    Encodes an image as PNG, in a mode that PNG holds.

    :param image: The image.
    :type image: PILImage.Image
    :return: "image/png" and the base64-encoded data of the image.
    :rtype: tuple[str, str]
    """
    if image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"):
        image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "image/png", base64.b64encode(buffer.getvalue()).decode("ascii")


def _is_image_file(document: DoclingDocument) -> bool:
    """
    Returns whether a Docling document is made of an image file, e.g. a scan, by
    the extension of the name of the file that it keeps.

    :param document: The Docling document.
    :type document: DoclingDocument
    :return: True for a document made of an image file.
    :rtype: bool
    """
    origin = document.origin
    return (
        origin is not None
        and Path(origin.filename).suffix.lower() in _IMAGE_FILE_SUFFIXES
    )


def _missing_image_source(picture: PictureItem) -> str:
    """
    Returns where the image of a picture is, whose image data cannot be read, for
    the log of the conversion.

    :param picture: The picture.
    :type picture: PictureItem
    :return: "none" for a picture without an image, "file" for an image in a local
        file, "url" for an image at another URL, or "data" for an image embedded in
        the Docling document.
    :rtype: str
    """
    image_ref = picture.image
    if image_ref is None:
        return "none"
    if isinstance(image_ref.uri, Path) or image_ref.uri.scheme == "file":
        return "file"
    if image_ref.uri.scheme == "data":
        return "data"
    return "url"


class _PendingParagraph(NamedTuple):
    """
    A paragraph that the converter has added, before the headings of its section
    are written in its additional context.
    """

    text: str
    # What the paragraph is, the speakers of its tracks, and its sheet or slide
    context: list[str]
    headings: list[tuple[int, str]]
    page_numbers: list[int]
    custom_data: dict[str, Any]


class _DoclingConverter:
    """
    Converts the body of a Docling document into ContextGem paragraphs and images,
    in reading order.
    """

    def __init__(self, document: DoclingDocument) -> None:
        """
        Prepares the conversion of a Docling document.

        :param document: The Docling document to convert.
        :type document: DoclingDocument
        """
        self._document = document
        self.paragraphs: list[_Paragraph] = []
        self.images: list[_Image] = []
        # The number of pictures that are not attached as images, by where their
        # image is (see `_missing_image_source()`)
        self.pictures_without_images: Counter[str] = Counter()
        # The number of pages of an image file whose image is not attached
        self.pages_without_images = 0
        # The headings of the current section, from the top, as (level, text) pairs,
        # a title at level 0
        self._headings: list[tuple[int, str]] = []
        # The paragraphs added, with the headings of their sections, until
        # the headings are written in their additional context once all are added
        # (see `convert()`), and the number of titles among them
        self._pending: list[_PendingParagraph] = []
        self._titles = 0
        # The sheet or the slide that the converted items are on, e.g. "Slide 2"
        self._containers: list[str] = []
        # The captions and the footnotes of tables, pictures, code blocks, key-value
        # regions and forms in the body, which are converted with the item that
        # they belong to, not where the body of the document has them. Those of
        # an item that is not converted, e.g. a table in a page header, and those
        # that are not texts of the body, e.g. a picture given as a caption, are
        # converted where the body has them.
        self._attached_refs = {
            ref.cref
            for item in (
                *document.tables,
                *document.pictures,
                *(text for text in document.texts if isinstance(text, CodeItem)),
                *document.key_value_items,
                *document.form_items,
            )
            if item.content_layer == ContentLayer.BODY
            for ref in (*item.captions, *item.footnotes)
            if isinstance(attached := ref.resolve(document), TextItem)
            and attached.content_layer == ContentLayer.BODY
        }
        # The numbers of the items of each numbered list that Docling gives no
        # numbers, by the reference of the list, worked out once for each list
        self._list_numbers: dict[str, dict[str, str]] = {}

    def convert(self) -> None:
        """
        Converts the body of the Docling document into paragraphs and images.

        The additional context of a paragraph ends with the headings of its section.
        The titles are among them in a document with several titles, e.g.
        the slides of a presentation, the chapters of an EPUB file, or the top-level
        headings of a Markdown file, of which Docling makes titles. The title of
        a document with one title is not: it would be in the additional context of
        all the other paragraphs.

        The images of the pages of an image file are attached first (see
        ``_add_page_images()``), and then those of the pictures in the body.

        :return: None
        :rtype: None
        """
        if _is_image_file(self._document):
            self._add_page_images()
        self._convert_children(self._document.body, list_level=0)
        titles_in_sections = self._titles > 1
        for pending in self._pending:
            context = list(pending.context)
            path = [
                text for level, text in pending.headings if level or titles_in_sections
            ]
            if path:
                context.append("Section: " + " > ".join(path))
            self.paragraphs.append(
                _publicize(
                    _Paragraph,
                    raw_text=pending.text,
                    additional_context=_clean("; ".join(context)) or None,
                    page_numbers=pending.page_numbers,
                    custom_data=pending.custom_data,
                )
            )

    def _add_page_images(self) -> None:
        """
        Adds the image of each page of a Docling document made of an image file,
        e.g. a scan: the image itself, whose text Docling reads with OCR into the
        paragraphs, so that a model can read the image too, e.g. for a concept of
        the "multimodal" modality. A page without image data is counted, for the
        log of the conversion, and a page whose image repeats an earlier one, e.g.
        a blank page, is attached once.

        :return: None
        :rtype: None
        """
        for page_no in sorted(self._document.pages):
            page = self._document.pages[page_no]
            image_data = _page_image_data(page)
            if image_data is None:
                self.pages_without_images += 1
                continue
            mime_type, base64_data = image_data
            if any(image.base64_data == base64_data for image in self.images):
                logger.debug(f"Page {page_no} repeats an earlier page")
                continue
            self.images.append(
                _publicize(
                    _Image,
                    mime_type=mime_type,
                    base64_data=base64_data,
                    custom_data={
                        "docling": {
                            "self_ref": f"#/pages/{page_no}",
                            "label": "page",
                            # The whole page
                            "prov": [
                                {
                                    "page_no": page_no,
                                    "bbox": {
                                        "l": 0.0,
                                        "t": page.size.height,
                                        "r": page.size.width,
                                        "b": 0.0,
                                        "coord_origin": "BOTTOMLEFT",
                                    },
                                }
                            ],
                        }
                    },
                )
            )

    def _children(self, item: NodeItem) -> list[NodeItem]:
        """
        Returns the children of an item that are converted where the body of
        the document has them.

        :param item: The parent item.
        :type item: NodeItem
        :return: The children in the body layer of the document, except the captions
            and the footnotes of tables and pictures.
        :rtype: list[NodeItem]
        """
        children = []
        for ref in item.children:
            child = ref.resolve(self._document)
            if (
                child.content_layer == ContentLayer.BODY
                and child.self_ref not in self._attached_refs
            ):
                children.append(child)
        return children

    def _convert_children(self, item: NodeItem, list_level: int) -> None:
        """
        Converts the children of an item.

        :param item: The parent item.
        :type item: NodeItem
        :param list_level: The nesting level of the list that the children are in,
            0 outside lists.
        :type list_level: int
        :return: None
        :rtype: None
        """
        for child in self._children(item):
            self._convert_item(child, list_level)

    def _convert_item(self, item: NodeItem, list_level: int) -> None:
        """
        Converts an item and the items under it.

        :param item: The item.
        :type item: NodeItem
        :param list_level: The nesting level of the list that the item is in,
            0 outside lists.
        :type list_level: int
        :return: None
        :rtype: None
        """
        if isinstance(item, TableItem):
            self._convert_table(item)
        elif isinstance(item, PictureItem):
            self._convert_picture(item)
        elif isinstance(item, KeyValueItem | FormItem):
            self._convert_graph(item)
        elif isinstance(item, FieldItem):
            self._convert_field(item)
        elif isinstance(item, InlineGroup):
            texts = self._inline_texts(item)
            self._add_paragraph(
                _join_inline(text.text for text in texts),
                None,
                item,
                _provenance_of_items(texts),
                _tracks_of_items(texts),
            )
            # The tables and the pictures in the text, e.g. an image in a line of
            # Markdown
            for child in self._floating_items_in(item):
                self._convert_item(child, list_level)
        elif isinstance(item, ListItem):
            self._convert_list_item(item, list_level)
        elif isinstance(item, SectionHeaderItem) or (
            isinstance(item, TextItem) and item.label == DocItemLabel.TITLE
        ):
            self._convert_heading(item, list_level)
        elif isinstance(item, CodeItem):
            self._convert_code(item, list_level)
        elif isinstance(item, TextItem):
            for child in self._add_text(item, self._text_kind(item)):
                self._convert_item(child, list_level)
        elif isinstance(item, ListGroup):
            self._convert_children(item, list_level + 1)
        elif isinstance(item, GroupItem):
            container = self._container_name(item)
            # The headings in a sheet or a slide, e.g. its title, end with it
            headings = list(self._headings)
            if container:
                self._containers.append(container)
            self._convert_children(item, list_level)
            if container:
                self._containers.pop()
                self._headings = headings
        else:
            # Other items, e.g. the regions of form fields, hold no text of their
            # own
            self._convert_children(item, list_level)

    def _add_paragraph(
        self,
        text: str,
        kind: str | None,
        source: NodeItem,
        provenance: list[dict[str, Any]],
        tracks: Sequence[dict[str, Any]] = (),
        **docling_data: Any,
    ) -> bool:
        """
        Adds a paragraph, unless its text is empty.

        :param text: The text of the paragraph.
        :type text: str
        :param kind: What the paragraph is, e.g. "List item", or None for body text.
        :type kind: str | None
        :param source: The Docling item that the paragraph is made from.
        :type source: NodeItem
        :param provenance: The pages and the bounding boxes of the paragraph.
        :type provenance: list[dict[str, Any]]
        :param tracks: The tracks that the paragraph is from, e.g. the cue of
            a subtitle file, whose speakers the additional context gives.
        :type tracks: Sequence[dict[str, Any]]
        :param docling_data: Other data to keep in the custom data of the paragraph,
            e.g. the index of a table row.
        :type docling_data: Any
        :return: Whether the paragraph is added.
        :rtype: bool
        """
        text = text.strip()
        if _is_text_content_empty(text):
            return False
        context = [kind] if kind else []
        speakers = list(
            dict.fromkeys(
                _clean(track.get("voice") or "")
                for track in tracks
                if _clean(track.get("voice") or "")
            )
        )
        if speakers:
            label = "Speaker" if len(speakers) == 1 else "Speakers"
            context.append(f"{label}: {', '.join(speakers)}")
        context.extend(self._containers)
        self._pending.append(
            _PendingParagraph(
                text=text,
                context=context,
                # The headings of its section, which `convert()` adds to the context
                headings=list(self._headings),
                page_numbers=[entry["page_no"] for entry in provenance],
                custom_data={
                    "docling": {
                        "self_ref": source.self_ref,
                        "label": _label_of(source),
                        "prov": provenance,
                        **({"source": list(tracks)} if tracks else {}),
                        **docling_data,
                    }
                },
            )
        )
        return True

    def _add_text(self, item: TextItem, kind: str | None) -> list[NodeItem]:
        """
        Adds a paragraph with the text of a text item.

        :param item: The text item.
        :type item: TextItem
        :param kind: What the paragraph is, or None for body text.
        :type kind: str | None
        :return: The items under the text item, to convert after it.
        :rtype: list[NodeItem]
        """
        text, sources, children = self._text_of(item)
        self._add_paragraph(
            text,
            kind,
            item,
            _provenance_of_items(sources),
            _tracks_of_items(sources),
        )
        return children

    def _text_of(self, item: TextItem) -> tuple[str, list[NodeItem], list[NodeItem]]:
        """
        Returns the text of a text item, with the items that it is made from and
        the items under it that are converted after it.

        A text item whose text is in an inline group or a text item under it, e.g.
        a heading or a list item with formatted text, takes the text of that item.
        The tables and the pictures in that inline group, e.g. an image in a line
        of Markdown, are converted after the text item.

        :param item: The text item.
        :type item: TextItem
        :return: The text, the items that it is made from, and the items to convert
            after it.
        :rtype: tuple[str, list[NodeItem], list[NodeItem]]
        """
        children = self._children(item)
        text = item.text.strip()
        sources: list[NodeItem] = [item]
        if not text and children:
            first = children[0]
            if isinstance(first, InlineGroup):
                texts = self._inline_texts(first)
                text = _join_inline(text_item.text for text_item in texts)
                sources.extend(texts)
                children = [*self._floating_items_in(first), *children[1:]]
            elif isinstance(first, TextItem) and first.label in _BODY_TEXT_LABELS:
                text = first.text.strip()
                sources.append(first)
                # The items under that text item are converted after the item
                children = [*self._children(first), *children[1:]]
        return text, sources, children

    def _add_attached_texts(self, refs: Iterable[RefItem], kind: str) -> None:
        """
        Adds a paragraph for each caption or footnote of an item, e.g. a table or
        a picture.

        :param refs: The references of the captions or footnotes.
        :type refs: Iterable[RefItem]
        :param kind: What the paragraphs are, e.g. "Table caption".
        :type kind: str
        :return: None
        :rtype: None
        """
        for ref in refs:
            item = ref.resolve(self._document)
            if isinstance(item, TextItem) and item.content_layer == ContentLayer.BODY:
                for child in self._add_text(item, kind):
                    self._convert_item(child, list_level=0)

    @staticmethod
    def _text_kind(item: TextItem) -> str | None:
        """
        Returns what the additional context of a paragraph calls a text item.

        :param item: The text item.
        :type item: TextItem
        :return: The kind of the text, e.g. "Footnote", or None for body text.
        :rtype: str | None
        """
        if item.label in _BODY_TEXT_LABELS:
            return None
        return _TEXT_KINDS.get(
            item.label, _label_of(item).replace("_", " ").capitalize()
        )

    @staticmethod
    def _container_name(group: GroupItem) -> str | None:
        """
        Returns the name of a sheet or a slide, for the additional context of its
        paragraphs.

        :param group: The Docling group.
        :type group: GroupItem
        :return: e.g. "Sheet: Q3" or "Slide 2", or None for a group that is not
            a sheet or a slide.
        :rtype: str | None
        """
        label = _label_of(group)
        if label == "sheet" and _clean(group.name):
            return f"Sheet: {_clean(group.name)}"
        slide = _SLIDE_GROUP_NAME.fullmatch(group.name)
        if label in ("chapter", "slide") and slide:
            return f"Slide {int(slide.group(1)) + 1}"
        return None

    def _convert_heading(self, item: TextItem, list_level: int) -> None:
        """
        Adds a paragraph for the title or a heading, and converts the items under
        it. A heading becomes the heading of the paragraphs that follow it, up to
        the next heading of its level or above. The title ends the sections before
        it.

        :param item: The title or the section header.
        :type item: TextItem
        :param list_level: The nesting level of the list that the item is in,
            0 outside lists.
        :type list_level: int
        :return: None
        :rtype: None
        """
        level = item.level if isinstance(item, SectionHeaderItem) else 0
        self._headings = [heading for heading in self._headings if heading[0] < level]
        text, sources, children = self._text_of(item)
        added = self._add_paragraph(
            text,
            f"Heading, level {level}" if level else "Title",
            item,
            _provenance_of_items(sources),
            _tracks_of_items(sources),
        )
        if added:
            self._headings.append((level, _clean(text)))
            if not level:
                self._titles += 1
        for child in children:
            self._convert_item(child, list_level)

    def _inline_texts(self, group: NodeItem) -> list[TextItem]:
        """
        Returns the text items of an inline group, in order.

        :param group: The inline group.
        :type group: NodeItem
        :return: The text items with text, including those of the groups in it.
        :rtype: list[TextItem]
        """
        texts: list[TextItem] = []
        for child in self._children(group):
            if isinstance(child, TextItem) and child.text.strip():
                texts.append(child)
            elif isinstance(child, GroupItem):
                texts.extend(self._inline_texts(child))
        return texts

    def _convert_list_item(self, item: ListItem, list_level: int) -> None:
        """
        Adds a paragraph for a list item, and converts the items under it, e.g.
        a nested list.

        :param item: The list item.
        :type item: ListItem
        :param list_level: The nesting level of the list, 1 for a list that is not
            in another list.
        :type list_level: int
        :return: None
        :rtype: None
        """
        text, sources, children = self._list_item_text(item)
        level = max(list_level, 1)
        self._add_paragraph(
            text,
            "List item" if level == 1 else f"List item, level {level}",
            item,
            _provenance_of_items(sources),
            _tracks_of_items(sources),
        )
        for child in children:
            self._convert_item(child, list_level)

    def _list_item_text(
        self, item: ListItem
    ) -> tuple[str, list[NodeItem], list[NodeItem]]:
        """
        Returns the text of a list item (see ``_text_of()``), which starts with its
        number if the item is numbered, or with its position in its list where
        Docling gives it no number, e.g. in an HTML or AsciiDoc file.

        :param item: The list item.
        :type item: ListItem
        :return: The text, the items that it is made from, and the items to convert
            after it.
        :rtype: tuple[str, list[NodeItem], list[NodeItem]]
        """
        text, sources, children = self._text_of(item)
        marker = item.marker.strip()
        if item.enumerated and not marker:
            marker = self._list_number(item)
        if text and item.enumerated and marker:
            text = f"{marker} {text}"
        return text, sources, children

    def _list_number(self, item: ListItem) -> str:
        """
        Returns the number of a numbered list item that Docling gives no number:
        its position among the items of its list, as Docling numbers it when it
        exports the document as Markdown, if the list starts with a numbered item.

        :param item: The list item.
        :type item: ListItem
        :return: e.g. "2.", or an empty string for an item that is not in a list
            that starts with a numbered item.
        :rtype: str
        """
        parent = item.parent.resolve(self._document) if item.parent else None
        if not isinstance(parent, ListGroup):
            return ""
        if parent.self_ref not in self._list_numbers:
            items = [
                child for child in self._children(parent) if isinstance(child, ListItem)
            ]
            numbered = bool(items) and items[0].enumerated
            self._list_numbers[parent.self_ref] = (
                {child.self_ref: f"{i + 1}." for i, child in enumerate(items)}
                if numbered
                else {}
            )
        return self._list_numbers[parent.self_ref].get(item.self_ref, "")

    def _convert_code(self, item: CodeItem, list_level: int) -> None:
        """
        Adds a paragraph for a code block, after the paragraphs of its captions and
        before those of its footnotes, and converts the items under it.

        :param item: The code block.
        :type item: CodeItem
        :param list_level: The nesting level of the list that the item is in,
            0 outside lists.
        :type list_level: int
        :return: None
        :rtype: None
        """
        self._add_attached_texts(item.captions, "Code caption")
        children = self._add_text(item, self._text_kind(item))
        self._add_attached_texts(item.footnotes, "Code footnote")
        for child in children:
            self._convert_item(child, list_level)

    def _convert_table(self, table: TableItem) -> None:
        """
        Adds a paragraph for the header and for each row of a table, after its
        captions and before its footnotes.

        The header paragraph lists the labels of the columns. A row is given with
        the label of each column, so that it can be read on its own, as a decision
        model reads it. The text of a rich cell is made from its items (see
        ``_rich_cell_text()``). The tables and the pictures in the cells of
        the table are converted after it: a cell has their captions, but not their
        content, in its text.

        :param table: The table.
        :type table: TableItem
        :return: None
        :rtype: None
        """
        caption = self._caption_text(table)
        self._add_attached_texts(table.captions, "Table caption")
        in_caption = (
            f"; Table caption: {_shorten(caption, _MAX_CAPTION_LENGTH_IN_ROWS)}"
            if caption
            else ""
        )
        provenance = _provenance_of_items([table])
        grid = self._grid(table)
        header_count = _count_header_rows(grid)
        labels = _column_labels(grid[:header_count])
        # A label that spans several columns is listed once
        header = [
            label
            for i, label in enumerate(labels)
            if label and (i == 0 or label != labels[i - 1])
        ]
        self._add_paragraph(
            "; ".join(header), "Table header" + in_caption, table, provenance
        )
        for index, row in list(enumerate(grid))[header_count:]:
            self._add_paragraph(
                _row_text(row, labels),
                "Table row" + in_caption,
                table,
                provenance,
                row_index=index,
            )
        self._add_attached_texts(table.footnotes, "Table footnote")
        for item in self._floating_items_in(table):
            self._convert_item(item, list_level=0)

    def _caption_text(self, item: TableItem | PictureItem) -> str:
        """
        Returns the text of the captions of a table or a picture that are texts of
        the body.

        :param item: The table or the picture.
        :type item: TableItem | PictureItem
        :return: The text of the captions, on a single line, or an empty string.
        :rtype: str
        """
        return _clean(
            " ".join(
                caption.text
                for ref in item.captions
                if isinstance(caption := ref.resolve(self._document), TextItem)
                and caption.content_layer == ContentLayer.BODY
            )
        )

    def _grid(self, table: TableItem) -> list[list[TableCell]]:
        """
        Returns the cells of a table, by row and column, each rich cell as a copy
        with the text made from its items. The Docling document is left as it is.

        :param table: The table.
        :type table: TableItem
        :return: The cells, a cell that spans several rows or columns at each of
            its places.
        :rtype: list[list[TableCell]]
        """
        rich_cells: dict[int, TableCell] = {}
        grid = []
        for row in table.data.grid:
            cells = []
            for cell in row:
                if isinstance(cell, RichTableCell):
                    if id(cell) not in rich_cells:
                        rich_cells[id(cell)] = cell.model_copy(
                            update={"text": self._rich_cell_text(cell)}
                        )
                    cell = rich_cells[id(cell)]
                cells.append(cell)
            grid.append(cells)
        return grid

    def _rich_cell_text(self, cell: RichTableCell) -> str:
        """
        Returns the text of a rich table cell, whose content Docling gives as items
        under the table, e.g. paragraphs, lists or a nested table, as in a DOCX,
        HTML or OpenDocument file. The text is made from those items as they are
        converted elsewhere: a numbered list item starts with its number, and
        a checkbox is followed by its state. A table or a picture in the cell,
        which is converted after the table, is given by its captions only. The text
        that Docling gives the cell is not used: it has the text of a nested table,
        but not the numbers of the list items or the states of the checkboxes.

        :param cell: The rich cell.
        :type cell: RichTableCell
        :return: The text of the cell, on a single line.
        :rtype: str
        """
        return _clean(" ".join(self._cell_parts(cell.ref.resolve(self._document))))

    def _cell_parts(self, item: NodeItem) -> list[str]:
        """
        Returns the texts of an item in a rich table cell and of the items under
        it, in order. An item outside the body of the document is left out, and
        the caption and the footnote of a code block are before and after its text,
        as they are elsewhere.

        :param item: The item, e.g. the group of the items of the cell.
        :type item: NodeItem
        :return: The texts, some of which can be empty.
        :rtype: list[str]
        """
        if item.content_layer != ContentLayer.BODY:
            return []
        if isinstance(item, TableItem | PictureItem):
            return [self._caption_text(item)]
        if isinstance(item, InlineGroup):
            texts = self._inline_texts(item)
            parts = [_join_inline(text.text for text in texts)]
            children = self._floating_items_in(item)
        elif isinstance(item, ListItem):
            text, _, children = self._list_item_text(item)
            parts = [text]
        elif isinstance(item, TextItem):
            text, _, children = self._text_of(item)
            parts = [_checkbox_text(item.label, text)]
            if isinstance(item, CodeItem):
                parts = [
                    *self._attached_cell_parts(item.captions),
                    *parts,
                    *self._attached_cell_parts(item.footnotes),
                ]
        else:
            parts = []
            children = self._children(item)
        for child in children:
            parts.extend(self._cell_parts(child))
        return parts

    def _attached_cell_parts(self, refs: Iterable[RefItem]) -> list[str]:
        """
        Returns the texts of the captions or the footnotes of an item in a rich
        table cell that are texts of the body, and of the items under them.

        :param refs: The references of the captions or the footnotes.
        :type refs: Iterable[RefItem]
        :return: The texts.
        :rtype: list[str]
        """
        parts = []
        for ref in refs:
            item = ref.resolve(self._document)
            if isinstance(item, TextItem):
                parts.extend(self._cell_parts(item))
        return parts

    def _floating_items_in(self, item: NodeItem) -> list[NodeItem]:
        """
        Returns the tables and the pictures under an item, e.g. in the cells of
        a table or in an inline group, without the items around them, whose text
        is converted with the item.

        :param item: The item, e.g. a table.
        :type item: NodeItem
        :return: The tables and the pictures, in order.
        :rtype: list[NodeItem]
        """
        items: list[NodeItem] = []
        for child in self._children(item):
            if isinstance(child, TableItem | PictureItem):
                items.append(child)
            else:
                items.extend(self._floating_items_in(child))
        return items

    def _convert_picture(self, picture: PictureItem) -> None:
        """
        Adds the image of a picture, after the paragraphs of its captions and before
        those of its footnotes. The text found inside the picture is not added.

        :param picture: The picture.
        :type picture: PictureItem
        :return: None
        :rtype: None
        """
        self._add_attached_texts(picture.captions, "Picture caption")
        image_data = _image_data(picture, self._document)
        if image_data is None:
            self.pictures_without_images[_missing_image_source(picture)] += 1
        elif any(image.base64_data == image_data[1] for image in self.images):
            # The images of a document are unique: a picture that repeats an
            # earlier one, e.g. a logo on each page, is attached once
            logger.debug(f"Picture {picture.self_ref} repeats an earlier picture")
        else:
            mime_type, base64_data = image_data
            # The time of a frame of a video
            tracks = _tracks_of_items([picture])
            self.images.append(
                _publicize(
                    _Image,
                    mime_type=mime_type,
                    base64_data=base64_data,
                    custom_data={
                        "docling": {
                            "self_ref": picture.self_ref,
                            "label": _label_of(picture),
                            "prov": _provenance_of_items([picture]),
                            **({"source": tracks} if tracks else {}),
                        }
                    },
                )
            )
        self._add_attached_texts(picture.footnotes, "Picture footnote")

    def _convert_field(self, field: FieldItem) -> None:
        """
        Adds a paragraph for a form field, with its key and its value, e.g.
        "Full name: Ada Lovelace" or "I agree to the terms: checked".

        :param field: The form field.
        :type field: FieldItem
        :return: None
        :rtype: None
        """
        keys = []
        values = []
        sources: list[NodeItem] = [field]
        for child in self._children(field):
            if not isinstance(child, TextItem) or child.label == DocItemLabel.MARKER:
                continue
            if child.label == DocItemLabel.FIELD_KEY:
                keys.append(_clean(child.text))
                sources.append(child)
            else:
                values.append(self._field_value(child, sources))
        self._add_paragraph(
            _labelled(
                " ".join(key for key in keys if key),
                " ".join(value for value in values if value),
            ),
            "Form field",
            field,
            _provenance_of_items(sources),
            _tracks_of_items(sources),
        )

    def _field_value(self, item: TextItem, sources: list[NodeItem]) -> str:
        """
        Returns the value of a form field, from its value item and the items under
        it, e.g. a checkbox.

        :param item: The value item, or an item under it.
        :type item: TextItem
        :param sources: The items that the paragraph of the field is made from,
            which the items of the value are added to.
        :type sources: list[NodeItem]
        :return: The value, e.g. "Ada Lovelace" or "checked".
        :rtype: str
        """
        sources.append(item)
        parts = [_checkbox_text(item.label, _clean(item.text))]
        for child in self._children(item):
            if isinstance(child, TextItem):
                parts.append(self._field_value(child, sources))
        return " ".join(part for part in parts if part)

    def _convert_graph(self, item: KeyValueItem | FormItem) -> None:
        """
        Adds a paragraph for each key of a key-value region or a form, with its
        values, and for each cell that is not linked to a key, after the paragraphs
        of its captions and before those of its footnotes. A key and its value are
        linked from the key to the value, from the value to the key, or both.
        A value that is also the key of other values has a paragraph of its own
        too, with them.

        :param item: The key-value region or the form.
        :type item: KeyValueItem | FormItem
        :return: None
        :rtype: None
        """
        name = "Form" if isinstance(item, FormItem) else "Key-value"
        self._add_attached_texts(item.captions, f"{name} caption")
        cells = {cell.cell_id: cell for cell in item.graph.cells}
        values_of_keys: dict[int, list[int]] = {}
        for link in item.graph.links:
            if link.label == GraphLinkLabel.TO_VALUE:
                key, value = link.source_cell_id, link.target_cell_id
            elif link.label == GraphLinkLabel.TO_KEY:
                key, value = link.target_cell_id, link.source_cell_id
            else:
                continue
            if key in cells and value in cells:
                values = values_of_keys.setdefault(key, [])
                # A pair linked both ways is one pair
                if value not in values:
                    values.append(value)
        linked_values = {
            value for values in values_of_keys.values() for value in values
        }
        kind = "Form field" if isinstance(item, FormItem) else "Key-value pair"
        for cell in item.graph.cells:
            if cell.cell_id in linked_values and cell.cell_id not in values_of_keys:
                continue
            parts = [cells[value] for value in values_of_keys.get(cell.cell_id, [])]
            value = "; ".join(_clean(part.text) for part in parts if _clean(part.text))
            self._add_paragraph(
                _labelled(_clean(cell.text), value),
                kind,
                item,
                # Where the cells have no provenance, e.g. from DocTags, the pages
                # of the region
                _provenance(part.prov for part in (cell, *parts))
                or _provenance_of_items([item]),
            )
        self._add_attached_texts(item.footnotes, f"{name} footnote")


def _convert_docling_document(
    document: DoclingDocument,
) -> tuple[list[_Paragraph], list[_Image]]:
    """
    Converts a Docling document into the paragraphs and the images of a ContextGem
    document.

    :param document: The Docling document.
    :type document: DoclingDocument
    :return: The paragraphs and the images, in the order of the Docling document.
    :rtype: tuple[list[_Paragraph], list[_Image]]
    :raises TypeError: If ``document`` is not a Docling document.
    :raises ValueError: If the body of the Docling document has no text or images.
    """
    if not isinstance(document, DoclingDocument):
        hint = (
            " A Docling conversion result holds it in `document`: "
            "Document.from_docling(result.document)."
            if isinstance(getattr(document, "document", None), DoclingDocument)
            else ""
        )
        raise TypeError(
            f"Expected a DoclingDocument of docling-core, "
            f"got {type(document).__name__}.{hint}"
        )
    converter = _DoclingConverter(document)
    converter.convert()
    for source, reason in _MISSING_IMAGE_REASONS.items():
        if converter.pictures_without_images[source]:
            logger.info(
                f"{converter.pictures_without_images[source]} picture(s) of "
                f"the Docling document are not attached to the document as images, "
                f"{reason}"
            )
    if converter.pages_without_images:
        logger.info(
            f"{converter.pages_without_images} page(s) of the image file of the "
            f"Docling document are not attached to the document as images, "
            f"{_MISSING_PAGE_IMAGES_REASON}"
        )
    if not converter.paragraphs and not converter.images:
        raise ValueError(
            "The body of the Docling document has no text or images to create "
            "a document from."
        )
    return converter.paragraphs, converter.images
