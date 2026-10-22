# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Module defining tests for documents and text segmentation.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import importlib.metadata
import io
import os
import re
import shutil
import socket
import subprocess  # nosec B404 - runs the Python of the tests, no shell
import sys
import tempfile
import threading
import time
import tomllib
import weakref
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
from docling_core.types.doc import (
    BoundingBox,
    CodeItem,
    ContentLayer,
    CoordOrigin,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    FormItem,
    GraphCell,
    GraphCellLabel,
    GraphData,
    GraphLink,
    GraphLinkLabel,
    GroupLabel,
    ImageRef,
    KeyValueItem,
    NodeItem,
    PictureItem,
    ProvenanceItem,
    RichTableCell,
    Size,
    TableCell,
    TableData,
    TableItem,
    TextItem,
    TrackSource,
)
from huggingface_hub import try_to_load_from_cache
from PIL import Image as PILImage

from contextgem.internal import utils as utils_module
from contextgem.internal.base import documents as documents_module
from contextgem.internal.converters import docling as converter_module
from contextgem.internal.items import _StringItem
from contextgem.internal.utils import (
    _SAT_BATCH_SIZE,
    _SAT_MODELS_IN_USE,
    _get_sat_model,
    _load_sat_model,
    _split_into_sentences,
)
from contextgem.public import (
    Aspect,
    BooleanConcept,
    Document,
    ExtractionPipeline,
    Paragraph,
    Sentence,
    StringConcept,
    release_sat_model,
)
from dev.docling_fixtures import (
    SOURCE_SUFFIXES,
    TEST_CONVERTED_SUFFIXES,
    contextgem_output,
    conversion_input,
    converter_key,
    format_options,
    layout_path,
    output_path,
    read_output,
    source_files,
    taxonomy_path,
)
from tests.base import BaseTest
from tests.utils import capture_logs, get_test_document_text


if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter


# A PDF file and the Docling document that Docling converted it into
_DOCLING_FILES_DIR = Path(__file__).parent / "other_files" / "docling"

# For the tests that run code in a new Python process
_SKIP_NEW_PROCESS_ON_WINDOWS_PY311 = pytest.mark.skipif(
    sys.platform == "win32" and sys.version_info < (3, 12),
    reason=(
        "On Windows, Python 3.11 launches `cmd /c ver` for `platform.system()`, "
        "which aiohttp calls on import, and the network guard that the new "
        "Python process inherits refuses to launch it"
    ),
)


def _docling_prov(page_no: int = 1, top: float = 700.0) -> ProvenanceItem:
    """
    Returns the provenance of a Docling item: a box 20 points high on a page.

    :param page_no: The number of the page.
    :type page_no: int
    :param top: The top of the box, from the bottom of the page.
    :type top: float
    :return: The provenance.
    :rtype: ProvenanceItem
    """
    return ProvenanceItem(
        page_no=page_no,
        bbox=BoundingBox(
            l=72.0, t=top, r=520.0, b=top - 20.0, coord_origin=CoordOrigin.BOTTOMLEFT
        ),
        charspan=(0, 0),
    )


def _docling_cell(
    row: int,
    column: int,
    text: str,
    *,
    row_span: int = 1,
    col_span: int = 1,
    header: bool = False,
    section: bool = False,
) -> TableCell:
    """
    Returns a cell of a Docling table.

    :param row: The row of the cell, from 0.
    :type row: int
    :param column: The column of the cell, from 0.
    :type column: int
    :param text: The text of the cell.
    :type text: str
    :param row_span: The number of rows that the cell spans.
    :type row_span: int
    :param col_span: The number of columns that the cell spans.
    :type col_span: int
    :param header: Whether the cell is a column header.
    :type header: bool
    :param section: Whether the cell is the title of a section of the table.
    :type section: bool
    :return: The cell.
    :rtype: TableCell
    """
    return TableCell(
        text=text,
        start_row_offset_idx=row,
        end_row_offset_idx=row + row_span,
        start_col_offset_idx=column,
        end_col_offset_idx=column + col_span,
        row_span=row_span,
        col_span=col_span,
        column_header=header,
        row_section=section,
    )


def _docling_table_data(cells: list[TableCell]) -> TableData:
    """
    Returns the data of a Docling table, as large as its cells need.

    :param cells: The cells of the table.
    :type cells: list[TableCell]
    :return: The data of the table.
    :rtype: TableData
    """
    return TableData(
        num_rows=max(cell.end_row_offset_idx for cell in cells),
        num_cols=max(cell.end_col_offset_idx for cell in cells),
        table_cells=cells,
    )


def _docling_image(color: str, size: int = 4) -> ImageRef:
    """
    Returns a Docling image reference that embeds a square image of one color.

    :param color: The color of the image, e.g. "red".
    :type color: str
    :param size: The width and the height of the image, in pixels.
    :type size: int
    :return: The image reference, with the image as a PNG data URI.
    :rtype: ImageRef
    """
    return ImageRef.from_pil(PILImage.new("RGB", (size, size), color), dpi=72)


# The converters of Docling for the tests, by what they depend on (`converter_key()`)
_DOCLING_CONVERTERS: dict[tuple[bool, Path | None, Path | None], DocumentConverter] = {}


def _docling_converter(path: Path) -> DocumentConverter:
    """
    Returns the converter of Docling for a fixture file, created once for the files
    that need the same options (`converter_key()`): the torch-free docling-slim of
    the dev group, which converts the formats that need no models, e.g. DOCX, XLSX,
    PPTX, HTML and Markdown files (`TEST_CONVERTED_SUFFIXES`), with the options of
    `uv run fab docling-fixtures`, e.g. the layout of the records of an EBCDIC file.

    :param path: The fixture file.
    :type path: Path
    :return: The converter.
    :rtype: DocumentConverter
    """
    # Imported here: the other tests of Document.from_docling() run without Docling,
    # with docling-core only, as `uv run fab docling-floor` runs them
    from docling.document_converter import DocumentConverter

    key = converter_key(path)
    if key not in _DOCLING_CONVERTERS:
        _DOCLING_CONVERTERS[key] = DocumentConverter(
            format_options=format_options(path)
        )
    return _DOCLING_CONVERTERS[key]


def _check_docling_text_kept(
    docling_document: DoclingDocument, document: Document
) -> None:
    """
    Checks that a document has each text, each table cell and each key-value cell
    of the body of the Docling document that it was created from, outside
    pictures, each in the paragraph made from it, which the Docling reference in
    the custom data of the paragraph gives: a text in its own paragraph, or in
    that of the item that it is part of, e.g. an inline group, a heading or
    a list item with formatted text, or a form field; a caption or a footnote in
    its own paragraph; a table cell in the paragraph of its row, or of the header
    for a header row, and a text in a rich cell, whose items are under the table,
    in a paragraph of the table; and a key-value cell in a paragraph of its
    region. The text that Docling gives a rich cell is not looked for, as it has
    that of a table in the cell, which is converted after the table. A text
    is not looked for in the other paragraphs, so that a paragraph left out is
    noticed also where its text is in another one, e.g. a repeated heading or
    a number.

    Also checks that only a table, a key-value region or a form is made into
    several paragraphs.

    :param docling_document: The Docling document.
    :type docling_document: DoclingDocument
    :param document: The document created from it.
    :type document: Document
    :return: None
    :rtype: None
    """

    def clean(text: str) -> str:
        """
        Collapses the whitespace of a text.
        """
        return " ".join(text.split())

    def ancestors(item: NodeItem) -> list[NodeItem]:
        """
        Returns an item and the items above it, from the item up.
        """
        chain = [item]
        while (parent := chain[-1].parent) is not None:
            chain.append(parent.resolve(docling_document))
        return chain

    # The row index, for the row of a table, and the text of each paragraph made
    # from a Docling item, by the reference of the item
    paragraphs_of: dict[str, list[tuple[int | None, str]]] = {}
    for paragraph in document.paragraphs:
        data = paragraph.custom_data["docling"]
        paragraphs_of.setdefault(data["self_ref"], []).append(
            (data.get("row_index"), clean(paragraph.raw_text))
        )

    def text_of(self_ref: str) -> str:
        """
        Returns the text of the paragraphs made from a Docling item.
        """
        return "\n".join(text for _, text in paragraphs_of.get(self_ref, []))

    several = {
        item.self_ref
        for item in (
            *docling_document.tables,
            *docling_document.key_value_items,
            *docling_document.form_items,
        )
    }
    for self_ref, paragraphs in paragraphs_of.items():
        assert len(paragraphs) == 1 or self_ref in several, self_ref

    for item, _ in docling_document.iterate_items():
        chain = ancestors(item)
        if any(isinstance(node, PictureItem) for node in chain[1:]):
            continue
        if isinstance(
            item, TableItem | PictureItem | CodeItem | KeyValueItem | FormItem
        ):
            # A caption or a footnote, wherever the Docling document has it, e.g.
            # under a picture
            for ref in (*item.captions, *item.footnotes):
                attached = ref.resolve(docling_document)
                if (
                    isinstance(attached, TextItem)
                    and attached.content_layer == ContentLayer.BODY
                    and clean(attached.text)
                ):
                    assert clean(attached.text) in text_of(attached.self_ref), (
                        attached.self_ref
                    )
        if isinstance(item, TextItem):
            text = clean(item.text)
            if not text or item.label == DocItemLabel.MARKER:
                continue
            # The text of an item in a rich cell is in a paragraph of its table, if
            # the table is converted
            tables = [node for node in chain[1:] if isinstance(node, TableItem)]
            if item.self_ref not in paragraphs_of and tables:
                if tables[0].self_ref in paragraphs_of:
                    assert text in text_of(tables[0].self_ref), item.self_ref
                continue
            owner = next(
                (node for node in chain if node.self_ref in paragraphs_of), None
            )
            assert owner is not None, f"{item.self_ref} not converted: {text}"
            assert text in text_of(owner.self_ref), item.self_ref
        elif isinstance(item, TableItem):
            rows = dict(paragraphs_of.get(item.self_ref, []))
            for index, row in enumerate(item.data.grid):
                # The items of a rich cell are checked above
                plain = [cell for cell in row if not isinstance(cell, RichTableCell)]
                cells = [clean(cell.text) for cell in plain if clean(cell.text)]
                if not cells:
                    continue
                if index not in rows:
                    # A header row is in the header paragraph, without a row index
                    assert all(
                        cell.column_header for cell in plain if clean(cell.text)
                    ), f"Row {index} of {item.self_ref} not converted"
                for cell in cells:
                    assert cell in rows.get(index, rows.get(None, "")), (
                        f"{cell} of row {index} of {item.self_ref}"
                    )
        elif isinstance(item, KeyValueItem | FormItem):
            for cell in item.graph.cells:
                # A key is given without the colon at its end
                key_or_value = clean(cell.text).rstrip(":").rstrip()
                assert key_or_value in text_of(item.self_ref), item.self_ref


# The fixture files that the tests convert with Docling, in the formats that the
# docling-slim of the dev group converts, and the others, e.g. PDF files, which
# they read the committed Docling documents of: the files written for ContextGem
# and the test documents of Docling, by their paths in `tests/other_files/docling/`
_DOCLING_CONVERTED_FILES = [
    path.relative_to(_DOCLING_FILES_DIR).as_posix()
    for path in source_files()
    if path.suffix in TEST_CONVERTED_SUFFIXES
]
_DOCLING_UNCONVERTED_FILES = [
    path.relative_to(_DOCLING_FILES_DIR).as_posix()
    for path in source_files()
    if path.suffix in SOURCE_SUFFIXES - TEST_CONVERTED_SUFFIXES
]

# The Docling labels of the items whose paragraphs are not running text, which
# `from_docling()` gives one sentence each, their whole text: headings, table
# headers and rows, key-value pairs and form fields, code, formulas, checkboxes,
# references and form headings. The SaT model segments the other paragraphs.
_DOCLING_ONE_SENTENCE_LABELS = {
    "title",
    "section_header",
    "table",
    "key_value_region",
    "form",
    "field_item",
    "code",
    "formula",
    "checkbox_selected",
    "checkbox_unselected",
    "reference",
    "field_heading",
}

# The paragraphs that each fixture file written for ContextGem must give, which show
# how its format is converted: (text, additional context, page numbers)
_DOCLING_FILE_PARAGRAPHS = {
    "service_agreement.docx": [
        # A sentence with a part in bold is one paragraph
        (
            "Invoices are payable within 30 days of the invoice date.",
            "Section: 1. Fees and Payment",
            [],
        ),
        # A merged cell has the headers of the columns that it spans
        (
            "Service: Support; Monthly fee, Billing: Included in hosting",
            "Table row; Section: 1. Fees and Payment",
            [],
        ),
        # A nested numbered list, under a heading of level 2
        (
            "2. if an invoice is 45 days overdue.",
            "List item, level 2; Section: 1. Fees and Payment > 1.1 Late Payment",
            [],
        ),
    ],
    # The page of a sheet is its position in the workbook
    "fees.xlsx": [
        (
            (
                "Party: Supplier; Contact: Ingrid Solberg; "
                "Email: ingrid.solberg@example.com"
            ),
            "Table row; Sheet: Contacts",
            [2],
        ),
    ],
    # The page of a slide is its number, and the content of a slide is in the
    # section of its title, as the presentation has several titles
    "quarterly_review.pptx": [
        (
            "Both were caused by a network provider.",
            "List item, level 2; Slide 2; Section: Results",
            [2],
        ),
        (
            "Service: Hosting; Monthly fee: 1,200 EUR",
            "Table row; Slide 3; Section: Fees",
            [3],
        ),
    ],
    "service_agreement.html": [
        (
            (
                "It renews automatically for 12 months, unless either party terminates "
                "it with 90 days' notice. See the general terms."
            ),
            "Section: 1. Term > 1.1 Renewal",
            [],
        ),
        (
            "Service: Hosting; Monthly fee: 1,200 EUR",
            "Table row; Table caption: Monthly fees; Section: 2. Fees",
            [],
        ),
    ],
    "meeting_notes.md": [
        (
            "1. The monitoring fee stays at 300 EUR per month.",
            "List item; Section: Decisions",
            [],
        ),
        (
            "Action: Review the invoices; Owner: Daniel Hughes; Due: 30 April 2026",
            "Table row; Section: Decisions",
            [],
        ),
        (
            "SLA-2026-04: critical incidents within 2 hours",
            "Code; Section: Decisions",
            [],
        ),
    ],
    # A JATS article: its sections and subsections, its table with the label and
    # the title of its caption, and its references as a list
    "field_study.nxml": [
        (
            "deficit irrigation, at 70% from flowering to harvest;",
            "List item; Section: Materials and methods > Treatments",
            [],
        ),
        (
            "Treatment: Deficit irrigation; Yield (t/ha): 79.1; Water use (mm): 331",
            (
                "Table row; Table caption: Table 1. Marketable yield and water use by "
                "treatment, mean of the two seasons.; Section: Results"
            ),
            [],
        ),
        (
            (
                "Brandt L. Deficit irrigation of field crops. Irrigation Science "
                "39:211–224 (2021)."
            ),
            "List item; Section: References",
            [],
        ),
    ],
    # A patent grant of the USPTO: its description and its claims, a claim with
    # its parts in one paragraph
    "patent_grant.xml": [
        (
            "Hole: Second; Angle: 20 degrees",
            "Table row; Section: DETAILED DESCRIPTION",
            [],
        ),
        (
            (
                "1. A laptop stand comprising: a base; a support plate; and a hinge "
                "between the base and the support plate, the hinge having a locking "
                "pin that engages one of a row of holes."
            ),
            "Section: CLAIMS",
            [],
        ),
    ],
    # An EPUB file: the documents of its spine in order, the heading of each
    # a title, which is the section of its paragraphs
    "employee_handbook.epub": [
        ("Leave", "Title", []),
        (
            "their manager has agreed to the start time;",
            "List item; Section: Working hours > Flexible hours",
            [],
        ),
        (
            "Type of leave: Unpaid leave; Days per year: Up to 20; Paid: No",
            (
                "Table row; Table caption: Days of leave per year, by type; "
                "Section: Leave"
            ),
            [],
        ),
    ],
}


def _paragraph_summary(
    document: Document,
) -> list[tuple[str, str | None, list[int]]]:
    """
    Returns the text, the additional context and the page numbers of each
    paragraph of a document.

    :param document: The document.
    :type document: Document
    :return: The text, the additional context and the page numbers of each
        paragraph, in order.
    :rtype: list[tuple[str, str | None, list[int]]]
    """
    return [
        (p.raw_text, p.additional_context, p.page_numbers) for p in document.paragraphs
    ]


def _image_pixels(document: Document) -> list[tuple[str, tuple[int, int], bytes]]:
    """
    Returns the mode, the size and the pixels of each image of a document.

    :param document: The document.
    :type document: Document
    :return: The mode, the size and the pixels of each image, in order.
    :rtype: list[tuple[str, tuple[int, int], bytes]]
    """
    pixels = []
    for image in document.images:
        with PILImage.open(io.BytesIO(base64.b64decode(image.base64_data))) as pil:
            pixels.append((pil.mode, pil.size, pil.tobytes()))
    return pixels


def _texts_and_contexts(document: Document) -> list[tuple[str, str | None]]:
    """
    Returns the text and the additional context of each paragraph of a document.

    :param document: The document.
    :type document: Document
    :return: The text and the additional context of each paragraph, in order.
    :rtype: list[tuple[str, str | None]]
    """
    return [(p.raw_text, p.additional_context) for p in document.paragraphs]


def _docling_formats_extensions() -> set[str]:
    """
    Returns the extensions that the table of the supported formats in the
    documentation of `Document.from_docling()` (`docs/source/documents/docling.rst`)
    lists.

    :return: The extensions, e.g. `.docx` and `.tar.gz`.
    :rtype: set[str]
    """
    page = (
        Path(__file__).parents[1] / "docs" / "source" / "documents" / "docling.rst"
    ).read_text(encoding="utf-8")
    section = page.split(".. _docling-formats-label:", 1)[1]
    table_lines = section.split(".. list-table::", 1)[1].splitlines()[1:]
    # The table ends at the first line that is not indented
    end = next(
        index
        for index, line in enumerate(table_lines)
        if line and not line.startswith(" ")
    )
    return set(re.findall(r"``(\.[a-z0-9][a-z0-9.]*)``", "\n".join(table_lines[:end])))


# The name that the README gives the format of an extension, where it is not the
# extension in capitals (`.docx` is DOCX)
_README_FORMAT_NAMES = {
    ".png": "Images",
    ".tiff": "Images",
    ".pages": "Pages",
    ".numbers": "Numbers",
    ".key": "Keynote",
    ".md": "Markdown",
    ".asciidoc": "AsciiDoc",
    ".tex": "LaTeX",
    ".vtt": "WebVTT",
    ".eml": "Email",
    ".msg": "Email",
    ".boxnote": "Box Notes",
    ".nxml": "JATS XML",
    ".xml": "USPTO XML",
    ".dclg": "DocLang",
    ".dclx": "DocLang",
    ".tar.gz": "METS GBS",
    ".ebc": "EBCDIC",
    ".mp3": "Audio",
    ".mp4": "Video",
    ".json": "Docling JSON",
}


class _StubSaTModel:
    """
    Stands in for a SaT model: splits a text into paragraphs at blank lines,
    and a paragraph into sentences after each full stop.
    """

    def split(
        self,
        text_or_texts: str | list[str],
        do_paragraph_segmentation: bool = False,
        **kwargs: Any,
    ) -> list[list[str]]:
        """
        Splits a text into paragraphs, each given as its sentences, or each of
        the texts of paragraphs into sentences.

        :param text_or_texts: The text, or the texts of the paragraphs.
        :type text_or_texts: str | list[str]
        :param do_paragraph_segmentation: Whether to split a text into paragraphs.
        :type do_paragraph_segmentation: bool
        :param kwargs: The other options of the segmentation, e.g. the batch
            size, which the stub does not use.
        :type kwargs: Any
        :return: The sentences of each paragraph.
        :rtype: list[list[str]]
        """
        if do_paragraph_segmentation:
            # Each paragraph is given as one sentence
            return [[paragraph] for paragraph in str(text_or_texts).split("\n\n")]
        return [re.split(r"(?<=\.) ", text) for text in text_or_texts]


@pytest.fixture
def release_kept_sat_model() -> Generator[None]:
    """
    Releases the SaT model that is kept loaded before and after the test, so that
    the test starts with no model kept, and that the tests that follow do not get
    a model that it kept, e.g. a stub.
    """
    release_sat_model()
    yield
    release_sat_model()


class TestDocuments(BaseTest):
    """
    Tests for documents, paragraphs, sentences and extraction pipelines,
    and for the segmentation of text with SaT models.
    """

    # Shared objects passed to the `parametrize` decorators of the tests below.
    # The decorators are evaluated in the class body, where the attributes inherited
    # from the base class are not in scope, so the names are bound here.
    document = BaseTest.document
    extraction_pipeline = BaseTest.extraction_pipeline

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_kept_until_released(self):
        """
        Tests that the SaT model used last is kept loaded, and is released by
        `release_sat_model()` as soon as nothing else holds it.
        """
        model = _get_sat_model("sat-3l-sm")
        model_ref = weakref.ref(model)
        del model
        # Kept, whatever form its ID is given in
        assert model_ref() is not None
        assert _get_sat_model("sat-3l-sm") is model_ref()
        assert _get_sat_model(Path("sat-3l-sm")) is model_ref()
        release_sat_model()
        # Released without a garbage collection, i.e. as the last reference to it
        # is dropped: the model is in no reference cycle
        assert model_ref() is None
        assert "sat-3l-sm" not in _SAT_MODELS_IN_USE

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_loaded_once_for_threads(self):
        """
        Tests that threads that need a SaT model at the same time load it once,
        and share it.
        """
        n_threads = 8
        loaded: list[_StubSaTModel] = []

        def load_slowly(model_id: str) -> _StubSaTModel:
            """
            Loads a model slowly, so that the threads ask for it while it is loaded.
            """
            time.sleep(0.1)
            loaded.append(_StubSaTModel())
            return loaded[-1]

        started = threading.Barrier(n_threads)
        done = threading.Barrier(n_threads)
        models: list[Any] = [None] * n_threads

        def get_model(position: int) -> None:
            """
            Gets the model, and holds it until each thread has got it.
            """
            started.wait()
            models[position] = _get_sat_model("stub-model")
            done.wait()

        with patch.object(utils_module, "_load_sat_model", new=load_slowly):
            threads = [
                threading.Thread(target=get_model, args=(i,)) for i in range(n_threads)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            assert len(loaded) == 1
            assert all(model is loaded[0] for model in models)

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_loading_does_not_block_other_models(self):
        """
        Tests that the threads that need other SaT models than the one that is
        being loaded, e.g. downloaded, do not wait for it.
        """
        loading_started = threading.Event()
        loading_released = threading.Event()

        def load(model_id: str) -> _StubSaTModel:
            """
            Loads a model, the slow one only when the test releases it.
            """
            if model_id == "slow-model":
                loading_started.set()
                loading_released.wait(timeout=30)
            return _StubSaTModel()

        slow_models: list[Any] = []
        other_models: list[Any] = []

        def get_other_models() -> None:
            """
            Gets the model in use, and another model that is loaded.
            """
            other_models.append(_get_sat_model("model-in-use"))
            other_models.append(_get_sat_model("other-model"))

        with patch.object(utils_module, "_load_sat_model", new=load):
            model_in_use = _get_sat_model("model-in-use")
            slow_loading = threading.Thread(
                target=lambda: slow_models.append(_get_sat_model("slow-model"))
            )
            slow_loading.start()
            try:
                assert loading_started.wait(timeout=30)
                other_loading = threading.Thread(target=get_other_models)
                other_loading.start()
                other_loading.join(timeout=10)
                assert not other_loading.is_alive(), "waited for the slow model"
            finally:
                loading_released.set()
                slow_loading.join()
            assert other_models[0] is model_in_use
            assert len(other_models) == 2
            assert len(slow_models) == 1

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_shared_by_paths_of_same_folder(self, tmp_path, monkeypatch):
        """
        Tests that the paths of a folder with a SaT model are those of one model,
        which is loaded once, in whatever form they are given, and that the same
        relative path is that of another model once the current directory has
        changed.
        """
        # The models loaded, recorded without holding them
        loaded: list[weakref.ref[_StubSaTModel]] = []

        def load(model_id: str) -> _StubSaTModel:
            """
            Loads a model.
            """
            model = _StubSaTModel()
            loaded.append(weakref.ref(model))
            return model

        (tmp_path / "models" / "sat").mkdir(parents=True)
        (tmp_path / "other" / "models" / "sat").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)
        absolute_path = str(tmp_path / "models" / "sat")
        paths = [
            "models/sat",
            "models/sat/",
            os.path.join(".", "models", "..", "models", "sat"),
            absolute_path,
            absolute_path + os.sep,
        ]
        if sys.platform == "win32":
            # Windows paths do not tell the case of the letters apart
            paths.append(absolute_path.upper())
        text = "First sentence. Second sentence."
        with patch.object(utils_module, "_load_sat_model", new=load):
            for sat_model_id in [*paths, Path("models") / "sat"]:
                Document(
                    raw_text=text,
                    sat_model_id=sat_model_id,
                    pre_segment_sentences=True,
                )
            assert len(loaded) == 1
            # The same relative path in another directory: another folder
            monkeypatch.chdir(tmp_path / "other")
            Document(
                raw_text=text,
                sat_model_id="models/sat",
                pre_segment_sentences=True,
            )
            assert len(loaded) == 2

    @pytest.mark.usefixtures("release_kept_sat_model")
    def test_sat_model_kept_for_documents_one_after_another(self):
        """
        Tests that documents processed one after another use the SaT model loaded
        for the first of them, that a model used later is kept in place of
        the one kept before, which is released, and that a document loads
        the model again after `release_sat_model()`.
        """
        # The models loaded, recorded without holding them
        loaded: list[weakref.ref[_StubSaTModel]] = []

        def load(model_id: str) -> _StubSaTModel:
            """
            Loads a model.
            """
            model = _StubSaTModel()
            loaded.append(weakref.ref(model))
            return model

        text = "First sentence. Second sentence.\n\nThird sentence."
        with patch.object(utils_module, "_load_sat_model", new=load):
            for _ in range(3):
                document = Document(raw_text=text, pre_segment_sentences=True)
                assert [len(p.sentences) for p in document.paragraphs] == [2, 1]
            # A paragraph segmentation uses the same model
            Document(raw_text=text, paragraph_segmentation_mode="sat")
            assert len(loaded) == 1
            # Another model is kept in place of the first one, which is released
            Document(
                raw_text=text,
                sat_model_id="sat-12l-sm",
                pre_segment_sentences=True,
            )
            assert len(loaded) == 2
            assert loaded[0]() is None
            assert loaded[1]() is not None
            Document(raw_text=text, pre_segment_sentences=True)
            assert len(loaded) == 3
            assert loaded[1]() is None
            # Released: the next document loads the model again
            release_sat_model()
            assert loaded[2]() is None
            Document(raw_text=text, pre_segment_sentences=True)
            assert len(loaded) == 4

    def test_sentences_segmented_by_tasks_at_the_same_time(self):
        """
        Tests that when two tasks segment the sentences of the same document at
        the same time, each paragraph keeps the sentences of the task that finishes
        first, rather than the sentences of the other task, or a mix of both.
        """
        calls: list[int] = []
        calls_lock = threading.Lock()
        sentences_assigned = threading.Event()

        class SplitterOfTwoTasks:
            """
            Replaces the SaT model: splits a text after each full stop for
            the first task, and into one sentence for the second task, which
            finishes first: the first one waits for its sentences to be assigned.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts into sentences.

                :param texts: The texts of the paragraphs.
                :type texts: list[str]
                :param kwargs: The options of the segmentation, which the stub
                    does not use.
                :type kwargs: Any
                :return: The sentences of each paragraph.
                :rtype: list[list[str]]
                """
                with calls_lock:
                    call = len(calls)
                    calls.append(call)
                if call == 0:
                    assert sentences_assigned.wait(timeout=10)
                    return [re.split(r"(?<=\.) ", text) for text in texts]
                return [[text] for text in texts]

        assign_segmented_sents = Document._assign_segmented_sents

        def assign_and_signal(
            document: Document, split_sents_for_paras: list[list[str]]
        ) -> None:
            """
            Assigns the sentences, and signals that they are assigned.
            """
            assign_segmented_sents(document, split_sents_for_paras)
            sentences_assigned.set()

        document = Document(
            raw_text="First sentence. Second sentence.\n\nThird sentence. Fourth."
        )

        async def segment_twice() -> None:
            """
            Segments the sentences of the document in two tasks at the same time.
            """
            await asyncio.gather(
                document._segment_sents_async(), document._segment_sents_async()
            )

        with (
            patch.object(
                documents_module,
                "_get_sat_model",
                new=lambda model_id: SplitterOfTwoTasks(),
            ),
            patch.object(Document, "_assign_segmented_sents", assign_and_signal),
        ):
            asyncio.run(segment_twice())
        assert calls == [0, 1]
        # The sentences of the second task, which finished first
        assert [[s.raw_text for s in p.sentences] for p in document.paragraphs] == [
            ["First sentence. Second sentence."],
            ["Third sentence. Fourth."],
        ]

    def test_split_into_sentences_in_batches_by_length(self):
        """
        Tests that the texts are split into sentences sorted by length, in batches
        of their own, and that the sentences are returned in the order of the texts.
        """
        calls: list[tuple[list[str], dict[str, Any]]] = []

        class RecordingModel:
            """
            Stands in for a SaT model: records each call, and splits a text into
            its words, so that each text gets sentences of its own.
            """

            def split(self, texts: list[str], **kwargs: Any) -> list[list[str]]:
                """
                Splits each of the texts.

                :param texts: The texts.
                :type texts: list[str]
                :param kwargs: The options of the segmentation.
                :type kwargs: Any
                :return: The words of each text.
                :rtype: list[list[str]]
                """
                calls.append((texts, kwargs))
                return [text.split() for text in texts]

        texts = ["a b c d", "a", "a b c d e f", "a b", "a b c"]
        model = RecordingModel()
        assert _split_into_sentences(model, texts) == [t.split() for t in texts]  # ty: ignore[invalid-argument-type]
        assert calls == [
            (
                sorted(texts, key=len),
                {
                    "batch_size": _SAT_BATCH_SIZE,
                    "outer_batch_size": _SAT_BATCH_SIZE,
                },
            )
        ]
        assert _split_into_sentences(model, []) == []  # ty: ignore[invalid-argument-type]

    def test_split_into_sentences_as_in_text_order(self):
        """
        Tests that a SaT model splits the texts into the same sentences when they
        are passed to it sorted by length, in batches of their own, as when they
        are passed to it in their order, for short texts and for a text longer than
        the block size of the model, which is split into chunks.
        """
        texts = [
            p.raw_text for p in Document(raw_text=get_test_document_text()).paragraphs
        ]
        # Longer than the 512 tokens of a block of the model
        texts.insert(len(texts) // 2, " ".join(texts))
        model = _get_sat_model("sat-3l-sm")
        assert _split_into_sentences(model, texts) == list(model.split(texts))

    def test_local_sat_model(self):
        """
        Tests the loading of a local SAT model.
        """

        # Test nonexistent path
        non_existent_path = "/nonexistent/path/to/model"
        with pytest.raises(ValueError) as exc_info:
            _load_sat_model(non_existent_path)
        assert "does not exist or is not a directory" in str(exc_info.value)
        # Document creation should also fail
        with pytest.raises(ValueError):
            Document(
                raw_text="Sample text",
                paragraph_segmentation_mode="sat",
                sat_model_id=non_existent_path,
            )

        # Test file path (not a directory)
        with tempfile.NamedTemporaryFile() as temp_file:
            with pytest.raises(ValueError) as exc_info:
                _load_sat_model(temp_file.name)
            assert "does not exist or is not a directory" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(ValueError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=temp_file.name,
                )

        # Test valid path but invalid model
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(RuntimeError) as exc_info:
                _load_sat_model(temp_dir)
            assert "does not contain a valid SaT model" in str(exc_info.value)
            # Document creation should also fail
            with pytest.raises(RuntimeError):
                Document(
                    raw_text="Sample text",
                    paragraph_segmentation_mode="sat",
                    sat_model_id=temp_dir,
                )

    @_SKIP_NEW_PROCESS_ON_WINDOWS_PY311
    def test_local_sat_model_without_network_access(self, tmp_path):
        """
        Tests that a SaT model in a local directory that also has the tokenizer
        of the model is loaded without network access, and without the cache of
        Hugging Face.

        The model is loaded in a new Python process, which has an empty cache,
        and which the network guard of the test also applies to: in this process,
        wtpsplit-lite keeps the tokenizer that it loaded before, so the tokenizer
        would not be downloaded again here even if it were not in the directory.
        """
        model_dir = tmp_path / "sat-3l-sm"
        model_dir.mkdir()
        # The files are in the cache, where the loading of the model in the base
        # test class put them
        for repo_id, filename in (
            ("segment-any-text/sat-3l-sm", "config.json"),
            ("segment-any-text/sat-3l-sm", "model_optimized.onnx"),
            ("facebookAI/xlm-roberta-base", "tokenizer.json"),
        ):
            cached_path = try_to_load_from_cache(repo_id, filename)
            assert isinstance(cached_path, str), f"{repo_id}/{filename} not cached"
            # Where the system has symbolic links, the file in the cache is a
            # relative link to the file itself, and os.link on Linux links the
            # link, not the file (before Python 3.14): the copy would point nowhere
            cached_path = os.path.realpath(cached_path)
            try:
                # The model file is large, so it is linked rather than copied
                os.link(cached_path, model_dir / filename)
            except OSError:
                shutil.copyfile(cached_path, model_dir / filename)

        script = (
            "import sys\n"
            "from contextgem import Document\n"
            "document = Document(\n"
            "    raw_text='First sentence. Second sentence.\\n\\nNew paragraph.',\n"
            "    sat_model_id=sys.argv[1],\n"
            "    pre_segment_sentences=True,\n"
            ")\n"
            "print([len(p.sentences) for p in document.paragraphs])\n"
        )
        empty_cache_dir = tmp_path / "empty_hf_home"
        env = {
            **os.environ,
            "HF_HOME": str(empty_cache_dir),
            "HF_HUB_CACHE": str(empty_cache_dir / "hub"),
            # litellm reads its bundled model data instead of fetching it on import
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
        }
        result = subprocess.run(  # nosec B603 - the Python of the tests, no shell
            [sys.executable, "-c", script, str(model_dir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        # Each of the two paragraphs has its sentences
        sentence_counts = ast.literal_eval(result.stdout.strip().splitlines()[-1])
        assert len(sentence_counts) == 2
        assert all(count > 0 for count in sentence_counts)
        # Nothing was taken into the cache
        assert not any(path.is_file() for path in empty_cache_dir.rglob("*"))

    def test_requires_sentence_segmentation(self):
        """
        Tests the `_requires_sentence_segmentation` method of the Document class.
        """
        # === Direct aspect/concept assignment ===

        document = Document(raw_text="This is a sentence.")

        # Aspects
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    reference_depth="paragraphs",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()
        document.remove_all_aspects()

        # Concepts
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_concepts()
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=False,
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()
        document.remove_all_concepts()
        document.add_concepts(
            [
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()

        # Sub-aspects
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    aspects=[
                        Aspect(
                            name="Sub-aspect 1",
                            description="Sub-aspect 1",
                            reference_depth="sentences",
                        ),
                    ],
                ),
            ]
        )
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()
        document.add_aspects(
            [
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    aspects=[
                        Aspect(
                            name="Sub-aspect 1",
                            description="Sub-aspect 1",
                            reference_depth="paragraphs",
                        ),
                    ],
                ),
            ]
        )
        assert not document._requires_sentence_segmentation()

        # === Assignment of aspects/concepts via ExtractionPipeline ===

        document = Document(raw_text="This is a sentence.")

        # Aspects
        extraction_pipeline = ExtractionPipeline(
            aspects=[
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert document._requires_sentence_segmentation()
        document.remove_all_aspects()

        # Concepts
        extraction_pipeline = ExtractionPipeline(
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert document._requires_sentence_segmentation()

    @pytest.mark.vcr
    def test_sat_model_deferred_segmentation(self):
        """
        Tests for the SaT model deferred segmentation.
        """

        # Trigger based on param
        document = Document(
            raw_text=get_test_document_text(),
            pre_segment_sentences=True,
        )
        assert document.paragraphs
        assert document.sentences

        # Trigger manually
        document = Document(raw_text=get_test_document_text())
        assert document.paragraphs
        # By default, sentences are not segmented.
        assert not document.sentences
        document._segment_sents()  # called when sentence-level refs are needed for aspects/concepts
        assert document.sentences

        # Selective segmentation of paragraphs
        document = Document(
            paragraphs=[
                Paragraph(
                    raw_text="This is sentence 1. This is sentence 2.",
                    sentences=[
                        Sentence(raw_text="This is sentence 1."),
                        Sentence(raw_text="This is sentence 2."),
                    ],
                ),
                Paragraph(
                    raw_text="This is a short sentence. And this is a bit longer sentence."
                ),  # this paragraph will be segmented
            ],
            pre_segment_sentences=True,
        )
        assert len(document.paragraphs[1].sentences) == 2
        assert len(document.sentences) == 4

        # === Segmentation is triggered when LLM extraction method is called for aspects/concepts
        # that require sentence segmentation ===

        # Some aspects require sentence segmentation (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            aspects=[
                Aspect(
                    name="Aspect 1", description="Aspect 1", reference_depth="sentences"
                ),
            ],
        )
        assert not document.sentences

        # Some concepts require sentence segmentation (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        assert not document.sentences

        # Some sub-aspects require sentence segmentation (assignment during Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.aspects = [
            Aspect(
                name="Liability",
                description="Liability",
                aspects=[
                    Aspect(
                        name="Liability cap",
                        description="Total liability cap",
                        reference_depth="sentences",
                    ),
                ],
            ),
        ]
        assert not document.sentences

        # Mixed - some aspects and concepts require sentence segmentation, some do not
        # (assignment during Document initialization)
        document = Document(
            raw_text=get_test_document_text(),
            aspects=[
                Aspect(
                    name="Aspect 1",
                    description="Aspect 1",
                    reference_depth="paragraphs",
                ),
                Aspect(
                    name="Aspect 2", description="Aspect 2", reference_depth="sentences"
                ),
            ],
            concepts=[
                StringConcept(
                    name="Concept 1",
                    description="Concept 1",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
                StringConcept(
                    name="Concept 2",
                    description="Concept 2",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        assert not document.sentences

        # Some aspects require sentence segmentation (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.aspects = [
            Aspect(
                name="Liability", description="Liability", reference_depth="sentences"
            ),
        ]
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_aspects_from_document(document)
        assert document.sentences

        # Some concepts require sentence segmentation (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.concepts = [
            StringConcept(
                name="Liability cap",
                description="Liability cap",
                add_references=True,
                reference_depth="sentences",
            ),
        ]
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_concepts_from_document(document)

        # Some sub-aspects require sentence segmentation (assignment during Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Liability",
                    aspects=[
                        Aspect(
                            name="Liability cap",
                            description="Total liability cap",
                            reference_depth="sentences",
                        ),
                    ],
                ),
            ]
        )
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_aspects_from_document(document)
        assert document.sentences

        # Mixed - some aspects and concepts require sentence segmentation, some do not
        # (assignment after Document initialization)
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Confidentiality",
                    description="Confidentiality",
                    reference_depth="paragraphs",
                ),
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ]
        )
        document.add_concepts(
            [
                StringConcept(
                    name="Confidential information",
                    description="Confidential information",
                    add_references=True,
                    reference_depth="paragraphs",
                ),
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document.sentences
        # Segmentation is triggered when LLM extraction method is called
        self.llm_extractor.extract_all(document)
        assert document.sentences

        # Test "from_aspects" and "from_concepts" methods
        document = Document(raw_text=get_test_document_text())
        document.add_aspects(
            [
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ]
        )
        assert not document.sentences
        self.llm_extractor.extract_aspects_from_document(
            document, from_aspects=[document.aspects[0]]
        )
        assert document.sentences
        document = Document(raw_text=get_test_document_text())
        document.add_concepts(
            [
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                )
            ]
        )
        assert not document.sentences
        self.llm_extractor.extract_concepts_from_document(
            document, from_concepts=[document.concepts[0]]
        )
        assert document.sentences

        # === Assignment via ExtractionPipeline ===

        document = Document(raw_text=get_test_document_text())
        extraction_pipeline = ExtractionPipeline(
            aspects=[
                Aspect(
                    name="Liability",
                    description="Liability",
                    reference_depth="sentences",
                ),
            ],
            concepts=[
                StringConcept(
                    name="Liability cap",
                    description="Liability cap",
                    add_references=True,
                    reference_depth="sentences",
                ),
            ],
        )
        document.assign_pipeline(extraction_pipeline)
        assert not document.sentences
        self.llm_extractor.extract_all(document)
        assert document.sentences

    def test_sentence_segmentation_of_paragraphs_without_sentences(self):
        """
        Tests that the sentences of the paragraphs that have none are segmented
        when an aspect requires sentences, while other paragraphs have their
        sentences, as the table rows of a document converted from Docling do:
        these keep them, and are not passed to the SaT model.
        """
        row_text = "Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance"
        row = Paragraph(raw_text=row_text, sentences=[Sentence(raw_text=row_text)])
        prose = Paragraph(
            raw_text=(
                "2.2 Invoices are payable within 30 days of the invoice date. "
                "Late payments bear interest of 8% per year."
            )
        )
        document = Document(paragraphs=[row, prose])
        assert not document._requires_sentence_segmentation()
        document.aspects = [
            Aspect(name="Fees", description="Fees", reference_depth="sentences")
        ]
        assert document._requires_sentence_segmentation()

        texts_segmented: list[str] = []
        split_into_sentences = documents_module._split_into_sentences

        def record_split(model: Any, texts: list[str]) -> list[list[str]]:
            """
            Records the texts that the SaT model segments into sentences.
            """
            texts_segmented.extend(texts)
            return split_into_sentences(model, texts)

        with patch.object(documents_module, "_split_into_sentences", new=record_split):
            document._segment_sents()
        assert texts_segmented == [prose.raw_text]
        assert [sentence.raw_text for sentence in row.sentences] == [row_text]
        assert [sentence.raw_text for sentence in prose.sentences] == [
            "2.2 Invoices are payable within 30 days of the invoice date.",
            "Late payments bear interest of 8% per year.",
        ]
        assert not document._requires_sentence_segmentation()

    def test_init_paragraph(self):
        """
        Tests for constructing a Paragraph instance and attaching it to a Document instance.
        """
        paragraph = Paragraph(raw_text="Test paragraph")
        assert paragraph.raw_text
        document = Document(paragraphs=[paragraph])
        assert document.raw_text  # to be populated from paragraphs
        with pytest.raises(ValueError):
            Paragraph()  # ty: ignore[missing-argument]
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="Test paragraph",
                additional_context=1,  # type: ignore
            )
        sentence = Sentence(raw_text="Test sentence")
        with pytest.raises(ValueError):
            Paragraph(
                raw_text="XYZ",
                sentences=[sentence],  # not matched in paragraph text
            )
        with pytest.raises(ValueError):
            Paragraph(
                raw_text=sentence.raw_text,
                sentences=[sentence, sentence],  # sentences with the same ID
            )
        # Warning is logged if linebreaks occur in additional context
        Paragraph(
            raw_text="Test paragraph",
            additional_context="""
            Test
            linebreaks
            """,
        )
        # Test with non-empty text but containing only control chars
        with pytest.raises(ValueError, match="control characters"):
            Paragraph(raw_text=" \u200c ")  # zero-width non-joiner

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(paragraph)

    def test_segmented_sentences_inherit_paragraph_data(self):
        """
        Tests that the sentences segmented from a paragraph inherit its additional
        context, its page numbers and its custom data, each sentence with a copy of
        the custom data of its own: a change to the nested data of one sentence does
        not change the paragraph or the other sentences.
        """
        paragraph = Paragraph(
            raw_text="Hosting is 1,200 EUR. Support is included.",
            additional_context="Table row; Section: 2. Fees",
            page_numbers=[2],
            custom_data={"docling": {"self_ref": "#/tables/0", "row_index": 1}},
        )
        with patch.object(
            documents_module, "_get_sat_model", return_value=_StubSaTModel()
        ):
            document = Document(paragraphs=[paragraph], pre_segment_sentences=True)
        paragraph = document.paragraphs[0]
        first_sentence, second_sentence = paragraph.sentences
        for sentence in (first_sentence, second_sentence):
            assert sentence.additional_context == "Table row; Section: 2. Fees"
            assert sentence.page_numbers == [2]
            assert sentence.custom_data == paragraph.custom_data

        first_sentence.custom_data["docling"]["note"] = "Checked"
        assert "note" not in paragraph.custom_data["docling"]
        assert "note" not in second_sentence.custom_data["docling"]

    def test_page_numbers(self):
        """
        Tests the page numbers of paragraphs and sentences: their validation, their
        inheritance by the sentences segmented from a paragraph, and the reference
        page numbers of aspects and extracted items, which are made from them.
        """
        # In ascending order, each once
        paragraph = Paragraph(
            raw_text="Fees are payable monthly. Interest applies.",
            page_numbers=[5, 4, 5],
        )
        assert paragraph.page_numbers == [4, 5]
        assert Paragraph(raw_text="No pages.").page_numbers == []
        for invalid in ([0], [-1], [True], ["3"], [2.0]):
            with pytest.raises(ValueError):
                Paragraph(raw_text="Text.", page_numbers=invalid)
        self.check_instance_serialization_and_cloning(paragraph)

        # Sentences segmented from a paragraph inherit its page numbers
        with patch.object(
            documents_module, "_get_sat_model", return_value=_StubSaTModel()
        ):
            document = Document(
                paragraphs=[
                    paragraph,
                    Paragraph(raw_text="The term is 24 months.", page_numbers=[6]),
                ],
                pre_segment_sentences=True,
            )
        first_sentence, second_sentence = document.paragraphs[0].sentences
        assert first_sentence.page_numbers == second_sentence.page_numbers == [4, 5]
        restored = Document.from_json(document.to_json())
        assert restored.paragraphs[0].sentences[1].page_numbers == [4, 5]

        # The reference page numbers of an aspect and of an extracted item
        aspect = Aspect(name="Payment", description="Payment terms")
        assert aspect.reference_page_numbers == []
        aspect.reference_paragraphs = [document.paragraphs[1], document.paragraphs[0]]
        assert aspect.reference_page_numbers == [4, 5, 6]
        item = _StringItem(value="Monthly")
        item.reference_sentences = [second_sentence]
        assert item.reference_page_numbers == [4, 5]
        # References without page numbers have none
        item.reference_paragraphs = [Paragraph(raw_text="No pages.")]
        item.reference_sentences = []
        assert item.reference_page_numbers == []

    def test_init_sentence(self):
        """
        Tests for constructing a Sentence instance and attaching it to a Paragraph instance.
        """
        sentence = Sentence(raw_text="Test sentence")
        assert sentence.raw_text
        Paragraph(raw_text=sentence.raw_text, sentences=[sentence])
        # Warning is logged if linebreaks occur in additional context
        with pytest.raises(ValueError):
            Sentence()  # ty: ignore[missing-argument]
        Sentence(
            raw_text="Test sentence",
            additional_context="""
            Test
            linebreaks
            """,
        )
        # Test with non-empty text but containing only control chars
        with pytest.raises(ValueError, match="control characters"):
            Sentence(raw_text=" \u200c ")  # zero-width non-joiner

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(sentence)

    def test_get_paragraph_and_sentence_index(self):
        """
        Tests for looking up paragraph/sentence positions in a document by object,
        covering duplicate texts, unsupported in-place list mutations, lazy
        sentence population, empty documents, and serialization round-trips.
        """
        # Paragraph lookup, with duplicate paragraph texts (adjacent and
        # separated) resolving to their specific occurrences
        para_texts = [
            "Duplicate clause.",
            "Duplicate clause.",
            "Intro.",
            "Duplicate clause.",
            "End.",
        ]
        document = Document(paragraphs=[Paragraph(raw_text=i) for i in para_texts])
        for idx, paragraph in enumerate(document.paragraphs):
            assert document.get_paragraph_index(paragraph) == idx
            # Repeated lookups are consistent
            assert document.get_paragraph_index(paragraph) == idx
        # Sorting references by position works with duplicate texts
        refs = [
            document.paragraphs[3],
            document.paragraphs[0],
            document.paragraphs[1],
        ]
        assert sorted(refs, key=document.get_paragraph_index) == [
            document.paragraphs[0],
            document.paragraphs[1],
            document.paragraphs[3],
        ]
        # Single-paragraph document
        single_para_document = Document(paragraphs=[Paragraph(raw_text="Only one.")])
        assert (
            single_para_document.get_paragraph_index(single_para_document.paragraphs[0])
            == 0
        )
        # Foreign paragraph is not found, even with identical text
        with pytest.raises(ValueError, match="not found"):
            document.get_paragraph_index(Paragraph(raw_text="Duplicate clause."))
        # Paragraphs of a same-text twin document are not found either
        # (lookup is keyed by unique ID, not text equality)
        twin_document = Document(paragraphs=[Paragraph(raw_text=i) for i in para_texts])
        with pytest.raises(ValueError, match="not found"):
            document.get_paragraph_index(twin_document.paragraphs[0])
        # Document without paragraphs (image-only)
        image_document = Document(images=[self.test_img_png_invoice])
        with pytest.raises(ValueError, match="not found"):
            image_document.get_paragraph_index(document.paragraphs[0])
        with pytest.raises(ValueError, match="not found"):
            image_document.get_sentence_index(Sentence(raw_text="Sent A."))

        # Unsupported in-place mutations of the paragraphs list must not
        # yield stale positions (lookups always reflect the live list state)
        mutated_document = Document(
            paragraphs=[Paragraph(raw_text=i) for i in ["P0.", "P1.", "P2."]]
        )
        first_para = mutated_document.paragraphs[0]
        last_para = mutated_document.paragraphs[-1]
        assert mutated_document.get_paragraph_index(last_para) == 2
        # Insertion shifts positions
        inserted_para = Paragraph(raw_text="Inserted.")
        mutated_document.paragraphs.insert(0, inserted_para)
        assert mutated_document.get_paragraph_index(first_para) == 1
        assert mutated_document.get_paragraph_index(last_para) == 3
        assert mutated_document.get_paragraph_index(inserted_para) == 0
        # Removed paragraph no longer resolves
        del mutated_document.paragraphs[-1]
        with pytest.raises(ValueError, match="not found"):
            mutated_document.get_paragraph_index(last_para)
        assert mutated_document.get_paragraph_index(first_para) == 1
        # Removal from the start shifts positions
        del mutated_document.paragraphs[0]
        assert mutated_document.get_paragraph_index(first_para) == 0

        # Sentence lookup, with duplicate sentence texts within and
        # across paragraphs resolving to their specific occurrences
        document_with_sents = Document(
            paragraphs=[
                Paragraph(
                    raw_text="Sent A. Sent A. Sent B.",
                    sentences=[
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent B."),
                    ],
                ),
                Paragraph(
                    raw_text="Sent A. Sent C.",
                    sentences=[
                        Sentence(raw_text="Sent A."),
                        Sentence(raw_text="Sent C."),
                    ],
                ),
            ]
        )
        for para_idx, paragraph in enumerate(document_with_sents.paragraphs):
            for sent_idx, sentence in enumerate(paragraph.sentences):
                assert document_with_sents.get_sentence_index(sentence) == (
                    para_idx,
                    sent_idx,
                )
                # Repeated lookups are consistent
                assert document_with_sents.get_sentence_index(sentence) == (
                    para_idx,
                    sent_idx,
                )
        # Positions sort in document order across paragraphs, matching the
        # flattened `document.sentences` order
        all_sents = document_with_sents.sentences
        assert (
            sorted(all_sents, key=document_with_sents.get_sentence_index) == all_sents
        )
        # Foreign sentence is not found, even with identical text
        with pytest.raises(ValueError, match="not found"):
            document_with_sents.get_sentence_index(Sentence(raw_text="Sent A."))

        # Lookup error hints that sentences are not segmented yet
        with pytest.raises(ValueError, match="no sentences segmented"):
            document.get_sentence_index(Sentence(raw_text="Intro."))

        # Sentences populated after earlier lookups resolve correctly,
        # including partial (per-paragraph) population
        document.paragraphs[2].sentences = [Sentence(raw_text="Intro.")]
        assert document.get_sentence_index(document.paragraphs[2].sentences[0]) == (
            2,
            0,
        )
        document.paragraphs[4].sentences = [Sentence(raw_text="End.")]
        assert document.get_sentence_index(document.paragraphs[4].sentences[0]) == (
            4,
            0,
        )
        # Previously populated sentences still resolve
        assert document.get_sentence_index(document.paragraphs[2].sentences[0]) == (
            2,
            0,
        )
        # Once some sentences exist, the not-segmented hint no longer applies
        with pytest.raises(ValueError, match="not found") as exc_info:
            document.get_sentence_index(Sentence(raw_text="Intro."))
        assert "no sentences segmented" not in str(exc_info.value)

        # Unsupported in-place mutations of a paragraph's sentences list must not
        # yield stale positions (lookups always reflect the live list state)
        mutable_sents_para = document_with_sents.paragraphs[1]
        tracked_sent = mutable_sents_para.sentences[-1]  # "Sent C." at (1, 1)
        assert document_with_sents.get_sentence_index(tracked_sent) == (1, 1)
        # Insertion shifts positions within the paragraph
        inserted_sent = Sentence(raw_text="Sent A.")
        mutable_sents_para.sentences.insert(0, inserted_sent)
        assert document_with_sents.get_sentence_index(tracked_sent) == (1, 2)
        assert document_with_sents.get_sentence_index(inserted_sent) == (1, 0)
        # Removed sentence no longer resolves
        del mutable_sents_para.sentences[-1]  # removes tracked_sent
        with pytest.raises(ValueError, match="not found"):
            document_with_sents.get_sentence_index(tracked_sent)
        # Other paragraphs' sentences are unaffected
        assert document_with_sents.get_sentence_index(
            document_with_sents.paragraphs[0].sentences[0]
        ) == (0, 0)

        # Unique IDs survive serialization round-trips, so positions resolve
        # across original/deserialized document boundaries, in both directions
        deserialized_document = Document.from_dict(document.to_dict())
        for idx in (0, 1, 3):
            assert (
                deserialized_document.get_paragraph_index(document.paragraphs[idx])
                == idx
            )
            assert (
                document.get_paragraph_index(deserialized_document.paragraphs[idx])
                == idx
            )
        assert deserialized_document.get_sentence_index(
            document.paragraphs[4].sentences[0]
        ) == (4, 0)
        assert document.get_sentence_index(
            deserialized_document.paragraphs[2].sentences[0]
        ) == (2, 0)
        # Deserialized document's own objects resolve within itself
        for idx, paragraph in enumerate(deserialized_document.paragraphs):
            assert deserialized_document.get_paragraph_index(paragraph) == idx

        # Performing lookups (which build internal position caches) must not
        # affect document equality, serialization, or cloning
        assert Document.from_dict(document.to_dict()) == document
        assert Document.from_dict(document_with_sents.to_dict()) == document_with_sents
        self.check_instance_serialization_and_cloning(document)

    @pytest.mark.parametrize("context", [document, extraction_pipeline])
    def test_init_document_and_extraction_pipeline(
        self, context: Document | ExtractionPipeline
    ):
        """
        Tests different initialization scenarios and validations associated with
        the `Document` and `ExtractionPipeline` classes.
        """
        context = context.clone()  # clone for method-scoped state modification

        # Test instance serialization and cloning
        self.check_instance_serialization_and_cloning(context)

        # Document initialization
        if isinstance(context, Document):
            for sat_model_id in [
                "sat-3l-sm",
                # "sat-6l-sm",
                # "sat-12l-sm",
            ]:
                for lang in ["en", "ua", "zh"]:  # EN and other langs
                    document = Document(
                        raw_text=get_test_document_text(lang=lang),
                        sat_model_id=sat_model_id,
                        paragraph_segmentation_mode="sat",
                        pre_segment_sentences=True,
                    )
                    assert document.paragraphs  # to be segmented from text
                    assert all(
                        i.sentences for i in document.paragraphs
                    )  # segmented from paragraphs since `pre_segment_sentences` is True
            Document(
                raw_text="Random text",
                paragraphs=[
                    Paragraph(raw_text="Random text"),
                ],
            )
            document = Document(
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                    Paragraph(raw_text="Random text 3"),
                ],
            )
            assert document.raw_text  # to be populated from paragraphs
            assert not any(
                i.sentences for i in document.paragraphs
            )  # sentences not segmented yet since `pre_segment_sentences` is False (default)
            with pytest.raises(ValueError):
                document.raw_text = "Random text 1"  # cannot be set once populated
            with pytest.raises(ValueError):
                document.paragraphs = [
                    Paragraph(raw_text="Random text 1"),
                ]  # cannot be set once populated
            Document(
                images=[
                    self.test_img_png_invoice,
                ]
            )
            with pytest.raises(ValueError):
                Document()
            with pytest.raises(ValueError):
                # Paragraphs do not match the text
                Document(
                    raw_text="XYZ",
                    paragraphs=[
                        Paragraph(raw_text="Random text 1"),
                        Paragraph(raw_text="Random text 2"),
                        Paragraph(raw_text="Random text 3"),
                    ],
                )
            with pytest.raises(ValueError):
                # Paragraphs are not ordered according to their appearance in the text
                Document(
                    raw_text="AAA BBB CCC",
                    paragraphs=[
                        Paragraph(raw_text="CCC"),
                        Paragraph(raw_text="BBB"),
                        Paragraph(raw_text="AAA"),
                    ],
                )
            # Paragraphs are ordered according to their appearance in the text,
            # but some paragraphs are duplicated
            Document(
                raw_text="AAA BBB CCC AAA DDD EEE BBB FFF",
                paragraphs=[
                    Paragraph(raw_text="AAA"),
                    Paragraph(raw_text="BBB"),
                    Paragraph(raw_text="CCC"),
                    Paragraph(raw_text="AAA"),
                    Paragraph(raw_text="DDD"),
                    Paragraph(raw_text="EEE"),
                    Paragraph(raw_text="BBB"),
                    Paragraph(raw_text="FFF"),
                ],
            )
            with pytest.raises(ValueError):
                # Paragraph duplicates exist but not all duplicates are matched
                Document(
                    raw_text="AAA BBB CCC",
                    paragraphs=[
                        Paragraph(raw_text="AAA"),
                        Paragraph(raw_text="BBB"),
                        Paragraph(raw_text="CCC"),
                        Paragraph(raw_text="BBB"),
                    ],
                )
            with pytest.raises(ValueError):
                Document(
                    raw_text="Random text",
                    extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
                )
            Document(
                raw_text="Random text 1\n\nRandom text 2",
                paragraphs=[
                    Paragraph(raw_text="Random text 1"),
                    Paragraph(raw_text="Random text 2"),
                ],
                images=[
                    self.test_img_png_invoice,
                ],
            )
            # List field items' unique IDs
            para = Paragraph(raw_text="Random text")
            with pytest.raises(ValueError):
                Document(paragraphs=[para, para, para])
            with pytest.raises(ValueError):
                Document(
                    images=[
                        self.test_img_png_invoice,
                        self.test_img_png_invoice,
                    ]
                )
            # Test with non-empty text but containing only control chars
            with pytest.raises(ValueError, match="control characters"):
                Document(raw_text=" \u200c ")  # zero-width non-joiner

        # Extraction pipeline initialization
        elif isinstance(context, ExtractionPipeline):
            ExtractionPipeline()  # works as we can interactive add aspects and concepts after initialization
            # Pipeline assignment
            document = self.document.clone()
            document.assign_pipeline(context)
            with pytest.raises(ValueError, match="already has aspects"):
                document.assign_pipeline(
                    context
                )  # document already has aspects and concepts
            document.aspects = []
            document.concepts = []
            document.assign_pipeline(context)
            assert context.aspects is not document.aspects
            assert context.concepts is not document.concepts
            # Pipeline params
            with pytest.raises(ValueError):
                ExtractionPipeline(
                    aspects=[
                        Aspect(
                            name="Liability",
                            description="Clauses describing liability of the parties",
                        )
                    ],
                    extra=True,  # ty: ignore[unknown-argument]  # extra fields not permitted
                )
            ExtractionPipeline(
                aspects=[
                    Aspect(
                        name="Liability",
                        description="Clauses describing liability of the parties",
                        add_justifications=True,
                    )
                ]
            )
            ExtractionPipeline(
                concepts=[
                    StringConcept(
                        name="Business Information",
                        description="Categories of Business Information",
                        add_justifications=True,
                    ),
                ]
            )

        # Document and extraction pipeline initialization
        # Document aspects
        document_aspects = [
            Aspect(
                name="Categories of confidential information",
                description="Clauses describing confidential information covered by the NDA",
            ),
            Aspect(
                name="Liability",
                description="Clauses describing liability of the parties",
                add_justifications=True,
            ),
        ]
        context.aspects = document_aspects
        assert context.aspects is not document_aspects

        # Validate assignment
        with pytest.raises(ValueError, match="non-empty list"):
            context.add_aspects([])  # empty list
        with pytest.raises(ValueError, match="non-empty list"):
            context.add_concepts([])  # empty list
        with pytest.raises(ValueError):
            context.add_aspects(
                [  # ty: ignore[invalid-argument-type]
                    "Random string",
                ]  # invalid aspect type
            )
        with pytest.raises(ValueError):
            context.add_aspects(
                [
                    Aspect(
                        name="liability",
                        description="clauses describing liability of the parties",
                        add_justifications=True,
                    )
                ]  # duplicate aspect (case-insensitive)
            )
        # Invalid sequence types
        with pytest.raises(ValueError, match="list"):
            context.add_concepts(
                (
                    StringConcept(
                        name="Random",
                        description="Random",
                    ),
                )  # tuple instead of list  # type: ignore
            )
        with pytest.raises(ValueError, match="list"):
            context.concepts = (
                StringConcept(
                    name="Random",
                    description="Random",
                ),
            )  # tuple instead of list
        with pytest.raises(ValueError, match="list"):
            context.concepts = {1, 2, 3}  # set instead of list  # type: ignore
        with pytest.raises(ValueError, match="list"):
            context.concepts = range(10)  # range instead of list  # type: ignore

        assert context.aspects == document_aspects
        assert context.aspects is not document_aspects
        context.get_aspect_by_name("Liability")
        with pytest.raises(ValueError):
            context.get_aspect_by_name("Non-existent")
        # Document concepts
        document_concepts = [
            StringConcept(
                name="Business Information",
                description="Categories of Business Information",
                add_justifications=True,
            ),
            BooleanConcept(
                name="Contract type check",
                description="Contract is an NDA",
                modality="image",
                add_justifications=True,
            ),
        ]
        context.concepts = document_concepts
        assert context.concepts is not document_concepts
        # Validate assignment
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [  # ty: ignore[invalid-assignment]
                123
            ]  # invalid concept type
        with pytest.raises(ValueError):
            context.concepts = document_concepts + [
                StringConcept(
                    name="business information",
                    description="categories of business information",
                ),
            ]  # duplicate concept (case-insensitive)
        assert context.concepts == document_concepts
        assert context.concepts is not document_concepts
        context.get_concept_by_name("Business Information")
        with pytest.raises(ValueError):
            context.get_concept_by_name("Non-existent")

        # Adding instances
        context.add_aspects(
            [
                Aspect(
                    name="Penalties",
                    description="Clauses describing contractual penalties",
                    add_justifications=True,
                    model_role="reasoner",
                ),
                Aspect(
                    name="Term and duration",
                    description="Clauses describing term and duration of the contract",
                ),
            ]
        )
        assert context.get_aspects_by_names(["Penalties", "Term and duration"])
        context.remove_aspects_by_names(["Penalties", "Term and duration"])
        context.add_concepts(
            [
                BooleanConcept(
                    name="NDA type check",
                    description="Is NDA one-way or mutual?",
                )
            ]
        )
        assert context.get_concepts_by_names(["NDA type check"])
        context.remove_concepts_by_names(["NDA type check"])

        # Removal of instances
        # Aspects
        context.remove_aspect_by_name("Liability")
        assert len(context.aspects) == 1
        assert len(document_aspects) == 2
        with pytest.raises(ValueError):
            context.get_aspect_by_name("Liability")  # not assigned aspect
        context.remove_all_aspects()
        assert len(context.aspects) == 0
        assert len(document_aspects) == 2
        # Concepts
        context.remove_concept_by_name("Contract type check")
        assert len(context.concepts) == 1
        assert len(document_concepts) == 2
        with pytest.raises(ValueError):
            context.get_concept_by_name("Contract type check")  # not assigned concept
        context.remove_all_concepts()
        assert len(context.concepts) == 0
        assert len(document_concepts) == 2
        # All instances
        context.aspects = document_aspects
        context.concepts = document_concepts
        context.remove_all_instances()
        assert len(context.aspects) == 0
        assert len(document_aspects) == 2
        assert len(context.concepts) == 0
        assert len(document_concepts) == 2

        # List field items' unique IDs
        context.remove_all_instances()
        assert not context.aspects and not context.concepts
        aspect = Aspect(
            name="Test",
            description="Test",
        )
        with pytest.raises(ValueError):
            context.add_aspects([aspect, aspect])
        concept = StringConcept(
            name="Test",
            description="Test",
        )
        with pytest.raises(ValueError):
            context.add_concepts([concept, concept])

    def test_from_docling_text_structure(self):
        """
        Tests that a document created from a Docling document has a paragraph for
        each text of the body of the Docling document, in reading order, with what
        the text is and the headings of its section in its additional context.
        """
        docling_document = DoclingDocument(name="agreement")
        docling_document.add_title("Service Agreement")
        docling_document.add_text(
            DocItemLabel.TEXT, "This Agreement is made on 1 March 2026."
        )
        docling_document.add_heading("1. Services", level=1)
        docling_document.add_text(
            DocItemLabel.PARAGRAPH, "The Supplier provides hosting."
        )
        docling_document.add_heading("1.1 Support", level=2)
        # A sentence with formatted parts, e.g. a bold one, is an inline group
        inline = docling_document.add_inline_group()
        for part in (
            "Support is available",
            "on business days",
            ", from 08:00",
            "(CET)",
            ".",
        ):
            docling_document.add_text(DocItemLabel.TEXT, part, parent=inline)
        # A heading of a higher level ends the sections below it
        docling_document.add_heading("2. Fees", level=1)
        bullets = docling_document.add_list_group()
        docling_document.add_list_item("Hosting", parent=bullets)
        docling_document.add_list_item("Monitoring", parent=bullets)
        numbered = docling_document.add_list_group()
        first_item = docling_document.add_list_item(
            "Invoices are payable within 30 days.",
            enumerated=True,
            marker="2.1",
            parent=numbered,
        )
        nested = docling_document.add_list_group(parent=first_item)
        docling_document.add_list_item(
            "Late payments bear interest.",
            enumerated=True,
            marker="(a)",
            parent=nested,
        )
        # A list item with formatted text has it in an inline group
        formatted_item = docling_document.add_list_item(
            "", enumerated=True, marker="2.2", parent=numbered
        )
        formatted_text = docling_document.add_inline_group(parent=formatted_item)
        docling_document.add_text(DocItemLabel.TEXT, "Fees", parent=formatted_text)
        docling_document.add_text(
            DocItemLabel.TEXT, "exclude VAT.", parent=formatted_text
        )
        docling_document.add_code("total = sum(fees)")
        docling_document.add_formula("f = 1.08 x")
        docling_document.add_text(DocItemLabel.FOOTNOTE, "1 Prices of 2026.")
        # A heading with formatted text has it in an inline group, e.g. from
        # Markdown, and is the heading of the paragraphs that follow it
        formatted_heading = docling_document.add_heading("", level=1)
        heading_text = docling_document.add_inline_group(parent=formatted_heading)
        docling_document.add_text(DocItemLabel.TEXT, "3. Term", parent=heading_text)
        docling_document.add_text(
            DocItemLabel.TEXT, "and termination", parent=heading_text
        )
        docling_document.add_text(DocItemLabel.TEXT, "Either party may terminate.")
        # Not converted: the furniture of the pages, the notes, and a text with
        # nothing but control characters
        docling_document.add_text(
            DocItemLabel.PAGE_HEADER,
            "Confidential",
            content_layer=ContentLayer.FURNITURE,
        )
        docling_document.add_text(
            DocItemLabel.PAGE_FOOTER,
            "Page 1 of 1",
            content_layer=ContentLayer.FURNITURE,
        )
        docling_document.add_text(
            DocItemLabel.TEXT,
            "Reviewer: please check.",
            content_layer=ContentLayer.NOTES,
        )
        docling_document.add_text(DocItemLabel.TEXT, chr(7) + chr(0))

        document = Document.from_docling(docling_document)
        assert _texts_and_contexts(document) == [
            ("Service Agreement", "Title"),
            ("This Agreement is made on 1 March 2026.", None),
            ("1. Services", "Heading, level 1"),
            ("The Supplier provides hosting.", "Section: 1. Services"),
            ("1.1 Support", "Heading, level 2; Section: 1. Services"),
            (
                "Support is available on business days, from 08:00 (CET).",
                "Section: 1. Services > 1.1 Support",
            ),
            ("2. Fees", "Heading, level 1"),
            ("Hosting", "List item; Section: 2. Fees"),
            ("Monitoring", "List item; Section: 2. Fees"),
            ("2.1 Invoices are payable within 30 days.", "List item; Section: 2. Fees"),
            (
                "(a) Late payments bear interest.",
                "List item, level 2; Section: 2. Fees",
            ),
            ("2.2 Fees exclude VAT.", "List item; Section: 2. Fees"),
            ("total = sum(fees)", "Code; Section: 2. Fees"),
            ("f = 1.08 x", "Formula; Section: 2. Fees"),
            ("1 Prices of 2026.", "Footnote; Section: 2. Fees"),
            ("3. Term and termination", "Heading, level 1"),
            ("Either party may terminate.", "Section: 3. Term and termination"),
        ]
        # The text of the document is the text of its paragraphs
        assert document.raw_text == "\n\n".join(
            paragraph.raw_text for paragraph in document.paragraphs
        )
        # The custom data of a paragraph has the reference and the label of the
        # Docling item that it is made from
        assert document.paragraphs[0].custom_data == {
            "docling": {"self_ref": "#/texts/0", "label": "title", "prov": []}
        }
        assert document.paragraphs[5].custom_data["docling"]["self_ref"] == (
            inline.self_ref
        )
        assert document.paragraphs[5].custom_data["docling"]["label"] == "inline"
        assert document.paragraphs[11].custom_data["docling"]["self_ref"] == (
            formatted_item.self_ref
        )
        # Without provenance, e.g. in a DOCX file, the pages are not known
        assert all(not paragraph.page_numbers for paragraph in document.paragraphs)

        # A title ends the sections before it, and the title of a document with one
        # title is in the section of no paragraph
        docling_document = DoclingDocument(name="two documents")
        docling_document.add_heading("Annex", level=1)
        docling_document.add_title("Second document")
        docling_document.add_text(DocItemLabel.TEXT, "Its first paragraph.")
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Annex", "Heading, level 1"),
            ("Second document", "Title"),
            ("Its first paragraph.", None),
        ]

        # In a document with several titles, e.g. the top-level headings of
        # a Markdown file or the chapters of an EPUB file, of which Docling makes
        # titles, the titles are the top level of the sections
        docling_document = DoclingDocument(name="readme")
        docling_document.add_title("Install")
        docling_document.add_text(DocItemLabel.TEXT, "Run the installer.")
        docling_document.add_title("Configure")
        docling_document.add_heading("Logging", level=1)
        docling_document.add_text(DocItemLabel.TEXT, "Set the log level.")
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Install", "Title"),
            ("Run the installer.", "Section: Install"),
            ("Configure", "Title"),
            ("Logging", "Heading, level 1; Section: Configure"),
            ("Set the log level.", "Section: Configure > Logging"),
        ]

        # The numbered items of a list that Docling gives no numbers, e.g. of HTML,
        # are numbered by their position in the list, as Docling's Markdown export
        # numbers them
        docling_document = DoclingDocument(name="steps")
        steps = docling_document.add_list_group()
        docling_document.add_list_item(
            "Clone the repository.", enumerated=True, parent=steps
        )
        second_step = docling_document.add_list_item(
            "Run the install script.", enumerated=True, parent=steps
        )
        nested = docling_document.add_list_group(parent=second_step)
        docling_document.add_list_item("Check the log.", enumerated=True, parent=nested)
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("1. Clone the repository.", "List item"),
            ("2. Run the install script.", "List item"),
            ("1. Check the log.", "List item, level 2"),
        ]
        # A list that starts with a bullet is not numbered, and the number that
        # Docling gives an item is kept as it is
        docling_document = DoclingDocument(name="restore")
        steps = docling_document.add_list_group()
        docling_document.add_list_item("Back up the data first.", parent=steps)
        docling_document.add_list_item(
            "Stop the service.", enumerated=True, parent=steps
        )
        docling_document.add_list_item(
            "Restore the backup.", enumerated=True, marker=" iii. ", parent=steps
        )
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Back up the data first.", "List item"),
            ("Stop the service.", "List item"),
            ("iii. Restore the backup.", "List item"),
        ]

        # The items under a text, e.g. a list under the sentence that introduces
        # it, follow the text
        docling_document = DoclingDocument(name="payment")
        introduction = docling_document.add_text(
            DocItemLabel.TEXT, "Payment is accepted by:"
        )
        methods = docling_document.add_list_group(parent=introduction)
        docling_document.add_list_item("bank transfer;", parent=methods)
        docling_document.add_list_item("card.", parent=methods)
        docling_document.add_text(DocItemLabel.TEXT, "Invoices are sent monthly.")
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Payment is accepted by:", None),
            ("bank transfer;", "List item"),
            ("card.", "List item"),
            ("Invoices are sent monthly.", None),
        ]

    def test_from_docling_provenance(self):
        """
        Tests that the custom data of a paragraph has the page and the box of each
        part of the Docling items that it is made from, once each: e.g. the two
        boxes of a text that goes on in the next column of its page, and the boxes
        of the parts of a heading with formatted text, on two pages.
        """

        def box(page_no: int, top: float) -> dict[str, Any]:
            """
            Returns the provenance entry of `_docling_prov()` in the custom data
            of a paragraph.
            """
            return {
                "page_no": page_no,
                "bbox": {
                    "l": 72.0,
                    "t": top,
                    "r": 520.0,
                    "b": top - 20.0,
                    "coord_origin": "BOTTOMLEFT",
                },
            }

        docling_document = DoclingDocument(name="report")
        columns = docling_document.add_text(
            DocItemLabel.TEXT,
            "The service was available all year.",
            prov=_docling_prov(1, 100.0),
        )
        columns.prov.append(_docling_prov(1, 700.0))
        # The same box twice is given once
        columns.prov.append(_docling_prov(1, 100.0))
        heading = docling_document.add_heading("", level=1, prov=_docling_prov(1, 60.0))
        heading_text = docling_document.add_inline_group(parent=heading)
        docling_document.add_text(
            DocItemLabel.TEXT,
            "2. Fees",
            parent=heading_text,
            prov=_docling_prov(1, 60.0),
        )
        docling_document.add_text(
            DocItemLabel.TEXT,
            "and Payment",
            parent=heading_text,
            prov=_docling_prov(2, 750.0),
        )
        # A sentence with formatted parts, whose inline group has no provenance of
        # its own
        sentence = docling_document.add_inline_group()
        docling_document.add_text(
            DocItemLabel.TEXT, "Invoices are", parent=sentence, prov=_docling_prov(2)
        )
        docling_document.add_text(
            DocItemLabel.TEXT, "payable", parent=sentence, prov=_docling_prov(2, 680.0)
        )

        document = Document.from_docling(docling_document)
        assert [
            (p.raw_text, p.page_numbers, p.custom_data["docling"])
            for p in document.paragraphs
        ] == [
            (
                "The service was available all year.",
                [1],
                {
                    "self_ref": columns.self_ref,
                    "label": "text",
                    "prov": [box(1, 100.0), box(1, 700.0)],
                },
            ),
            (
                "2. Fees and Payment",
                [1, 2],
                {
                    "self_ref": heading.self_ref,
                    "label": "section_header",
                    "prov": [box(1, 60.0), box(2, 750.0)],
                },
            ),
            (
                "Invoices are payable",
                [2],
                {
                    "self_ref": sentence.self_ref,
                    "label": "inline",
                    "prov": [box(2, 700.0), box(2, 680.0)],
                },
            ),
        ]

    def test_from_docling_tables(self):
        """
        Tests that a table of a Docling document is converted into a paragraph
        with the headers of its columns and a paragraph for each row, with each
        value after the header of its column, between the paragraphs of its
        captions and its footnotes.
        """
        docling_document = DoclingDocument(name="fees")
        docling_document.add_heading("2. Fees", level=1)
        caption = docling_document.add_text(
            DocItemLabel.CAPTION, "Table 1: Monthly fees"
        )
        fees = docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Service", header=True),
                    _docling_cell(0, 1, "Monthly fee", header=True),
                    # A colon at the end of a header is not repeated in the rows
                    _docling_cell(0, 2, "Billing:", header=True),
                    _docling_cell(1, 0, "Hosting"),
                    _docling_cell(1, 1, "1,200 EUR"),
                    _docling_cell(1, 2, "In advance"),
                    # The title of a section of the table has no label
                    _docling_cell(2, 0, "Optional services", col_span=3, section=True),
                    # An empty cell is left out
                    _docling_cell(3, 0, "Monitoring"),
                    _docling_cell(3, 1, "300 EUR"),
                ]
            ),
            caption=caption,
            prov=_docling_prov(page_no=2),
        )
        footnote = docling_document.add_text(
            DocItemLabel.FOOTNOTE, "Fees exclude VAT.", parent=fees
        )
        fees.footnotes.append(footnote.get_ref())
        # A header that spans two columns, above a header for each of them
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Service", row_span=2, header=True),
                    _docling_cell(0, 1, "Fee", col_span=2, header=True),
                    _docling_cell(1, 1, "2025", header=True),
                    _docling_cell(1, 2, "2026", header=True),
                    _docling_cell(2, 0, "Hosting"),
                    _docling_cell(2, 1, "1,100 EUR"),
                    _docling_cell(2, 2, "1,200 EUR"),
                    _docling_cell(3, 0, "Support"),
                    _docling_cell(3, 1, "Included", col_span=2),
                ]
            )
        )
        # A table without headers
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Supplier"),
                    _docling_cell(0, 1, "Northwind Analytics AS"),
                    _docling_cell(1, 0, "Customer"),
                    _docling_cell(1, 1, "Riverbank Logistics Ltd"),
                ]
            )
        )
        docling_document.add_text(DocItemLabel.TEXT, "2.2 Invoices are payable.")

        document = Document.from_docling(docling_document)
        fees_context = "Table caption: Table 1: Monthly fees; Section: 2. Fees"
        assert _texts_and_contexts(document) == [
            ("2. Fees", "Heading, level 1"),
            ("Table 1: Monthly fees", "Table caption; Section: 2. Fees"),
            ("Service; Monthly fee; Billing:", f"Table header; {fees_context}"),
            (
                "Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance",
                f"Table row; {fees_context}",
            ),
            ("Optional services", f"Table row; {fees_context}"),
            ("Service: Monitoring; Monthly fee: 300 EUR", f"Table row; {fees_context}"),
            ("Fees exclude VAT.", "Table footnote; Section: 2. Fees"),
            ("Service; Fee > 2025; Fee > 2026", "Table header; Section: 2. Fees"),
            (
                "Service: Hosting; Fee > 2025: 1,100 EUR; Fee > 2026: 1,200 EUR",
                "Table row; Section: 2. Fees",
            ),
            (
                "Service: Support; Fee > 2025, Fee > 2026: Included",
                "Table row; Section: 2. Fees",
            ),
            ("Supplier; Northwind Analytics AS", "Table row; Section: 2. Fees"),
            ("Customer; Riverbank Logistics Ltd", "Table row; Section: 2. Fees"),
            ("2.2 Invoices are payable.", "Section: 2. Fees"),
        ]
        # The custom data of a row has the index of the row in the table, and the
        # provenance of the table
        table_prov = [
            {
                "page_no": 2,
                "bbox": {
                    "l": 72.0,
                    "t": 700.0,
                    "r": 520.0,
                    "b": 680.0,
                    "coord_origin": "BOTTOMLEFT",
                },
            }
        ]
        assert document.paragraphs[2].custom_data == {
            "docling": {"self_ref": fees.self_ref, "label": "table", "prov": table_prov}
        }
        assert [
            p.custom_data["docling"]["row_index"] for p in document.paragraphs[3:6]
        ] == [
            1,
            2,
            3,
        ]
        assert document.paragraphs[3].custom_data["docling"]["prov"] == table_prov
        # The index of a row counts the header rows, e.g. two above the rows
        assert [
            p.custom_data["docling"].get("row_index") for p in document.paragraphs[7:10]
        ] == [None, 2, 3]
        # The page numbers of a paragraph are those of its provenance
        assert document.paragraphs[3].page_numbers == [2]
        assert document.paragraphs[0].page_numbers == []

        # A header that spans two columns is listed once in the header. A cell that
        # spans the row under different headers has no label, as the title of
        # a section of the table has none, also where it spans one column.
        docling_document = DoclingDocument(name="fees by year")
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Service", header=True),
                    _docling_cell(0, 1, "Fee", col_span=2, header=True),
                    _docling_cell(1, 0, "Hosting"),
                    _docling_cell(1, 1, "1,100 EUR"),
                    _docling_cell(1, 2, "1,200 EUR"),
                    # A merged cell, e.g. of a DOCX or HTML file, which Docling does
                    # not mark as a section
                    _docling_cell(2, 0, "Waived for the first year", col_span=3),
                    _docling_cell(3, 0, "Optional services", section=True),
                    _docling_cell(4, 0, "Monitoring"),
                    _docling_cell(4, 1, "300 EUR"),
                ]
            )
        )
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Service; Fee", "Table header"),
            ("Service: Hosting; Fee: 1,100 EUR; Fee: 1,200 EUR", "Table row"),
            ("Waived for the first year", "Table row"),
            ("Optional services", "Table row"),
            ("Service: Monitoring; Fee: 300 EUR", "Table row"),
        ]

        # A first row of which only some cells are column headers, e.g. a key in
        # bold before its value, is not a header row, and an empty cell, e.g.
        # the corner of a table, does not make a row a header row or not
        docling_document = DoclingDocument(name="parties and fees")
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Supplier", header=True),
                    _docling_cell(0, 1, "Northwind Analytics AS"),
                    _docling_cell(1, 0, "Customer", header=True),
                    _docling_cell(1, 1, "Riverbank Logistics Ltd"),
                ]
            )
        )
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 1, "2026", header=True),
                    _docling_cell(1, 0, "Hosting"),
                    _docling_cell(1, 1, "1,200 EUR"),
                ]
            )
        )
        document = Document.from_docling(docling_document)
        assert [
            (
                p.raw_text,
                p.additional_context,
                p.custom_data["docling"].get("row_index"),
            )
            for p in document.paragraphs
        ] == [
            ("Supplier; Northwind Analytics AS", "Table row", 0),
            ("Customer; Riverbank Logistics Ltd", "Table row", 1),
            ("2026", "Table header", None),
            ("Hosting; 2026: 1,200 EUR", "Table row", 1),
        ]

        # A long caption is shortened in the additional context of the rows, which
        # repeats it, and is whole in its own paragraph
        docling_document = DoclingDocument(name="service levels")
        long_caption = (
            "Table 4: Service levels. The response times apply on business days and "
            "exclude the time spent waiting for information from the Customer."
        )
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Priority", header=True),
                    _docling_cell(1, 0, "Critical"),
                ]
            ),
            caption=docling_document.add_text(DocItemLabel.CAPTION, long_caption),
        )
        caption_paragraph, _, row = Document.from_docling(docling_document).paragraphs
        assert caption_paragraph.raw_text == long_caption
        assert row.additional_context is not None
        shortened = row.additional_context.removeprefix("Table row; Table caption: ")
        assert shortened.endswith("...")
        assert len(shortened) <= 100
        assert long_caption.startswith(shortened.removesuffix("..."))
        # A comma, a semicolon or a colon before the cut is left out
        docling_document = DoclingDocument(name="response times")
        docling_document.add_table(
            data=_docling_table_data([_docling_cell(0, 0, "Critical")]),
            caption=docling_document.add_text(
                DocItemLabel.CAPTION,
                (
                    "Table 5: Response times. The times apply on business days and on "
                    "public holidays in Norway, Sweden and Denmark."
                ),
            ),
        )
        row = Document.from_docling(docling_document).paragraphs[-1]
        assert row.additional_context == (
            "Table row; Table caption: Table 5: Response times. The times apply on "
            "business days and on public holidays in Norway..."
        )

        # A caption in the furniture of the page, or that is not a text, is not
        # a paragraph
        docling_document = DoclingDocument(name="draft")
        table = docling_document.add_table(
            data=_docling_table_data([_docling_cell(0, 0, "Northwind Analytics AS")]),
            caption=docling_document.add_text(
                DocItemLabel.CAPTION, "Draft", content_layer=ContentLayer.FURNITURE
            ),
        )
        table.captions.append(docling_document.add_group(name="not a text").get_ref())
        assert [
            p.raw_text for p in Document.from_docling(docling_document).paragraphs
        ] == ["Northwind Analytics AS"]

        # The caption in the body of a table that is not converted, e.g. a table in
        # a page header, is a paragraph where the body has it
        docling_document = DoclingDocument(name="letterhead")
        docling_document.add_text(DocItemLabel.TEXT, "Dear Customer,")
        docling_document.add_table(
            data=_docling_table_data([_docling_cell(0, 0, "Northwind Analytics AS")]),
            caption=docling_document.add_text(DocItemLabel.CAPTION, "Our offices"),
            content_layer=ContentLayer.FURNITURE,
        )
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Dear Customer,", None),
            ("Our offices", "Caption"),
        ]

    def test_from_docling_rich_table_cells(self):
        """
        Tests that the text of a rich table cell, whose items are under the table,
        e.g. in a DOCX or HTML file, is made from its items, not taken from the text
        that Docling gives the cell: with the state of each checkbox and the number
        of each numbered list item, and with the caption of a table or a picture in
        the cell, which is converted after the table.
        """
        docling_document = DoclingDocument(name="order form")
        table = docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Service", header=True),
                    _docling_cell(0, 1, "Options", header=True),
                    _docling_cell(1, 0, "Breakfast"),
                    _docling_cell(2, 0, "Delivery"),
                ]
            )
        )
        breakfast = docling_document.add_group(name="rich_cell_group", parent=table)
        docling_document.add_text(DocItemLabel.TEXT, "Choose any:", parent=breakfast)
        docling_document.add_text(
            DocItemLabel.CHECKBOX_SELECTED, "Tea", parent=breakfast
        )
        docling_document.add_text(
            DocItemLabel.CHECKBOX_UNSELECTED, "Coffee", parent=breakfast
        )
        delivery = docling_document.add_group(name="rich_cell_group", parent=table)
        methods = docling_document.add_list_group(parent=delivery)
        docling_document.add_list_item("Standard", enumerated=True, parent=methods)
        docling_document.add_list_item("Express", enumerated=True, parent=methods)
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Zone", header=True),
                    _docling_cell(0, 1, "Days", header=True),
                    _docling_cell(1, 0, "Oslo"),
                    _docling_cell(1, 1, "1"),
                ]
            ),
            caption=docling_document.add_text(
                DocItemLabel.CAPTION, "Table 2: Delivery times", parent=delivery
            ),
            parent=delivery,
        )
        docling_document.add_picture(
            image=_docling_image("red"),
            caption=docling_document.add_text(
                DocItemLabel.CAPTION, "Courier map", parent=delivery
            ),
            parent=delivery,
        )
        # The text that Docling gives each cell: that of all its items, the nested
        # table included
        for row, group, text in (
            (1, breakfast, "Choose any:\nTea\nCoffee"),
            (
                2,
                delivery,
                (
                    "Standard\nExpress\nTable 2: Delivery times\nZone Days\nOslo 1\n"
                    "Courier map"
                ),
            ),
        ):
            docling_document.add_table_cell(
                table,
                RichTableCell(
                    text=text,
                    ref=group.get_ref(),
                    start_row_offset_idx=row,
                    end_row_offset_idx=row + 1,
                    start_col_offset_idx=1,
                    end_col_offset_idx=2,
                ),
            )

        document = Document.from_docling(docling_document)
        nested_context = "Table caption: Table 2: Delivery times"
        assert _texts_and_contexts(document) == [
            ("Service; Options", "Table header"),
            (
                (
                    "Service: Breakfast; Options: Choose any: Tea (checked) "
                    "Coffee (unchecked)"
                ),
                "Table row",
            ),
            (
                (
                    "Service: Delivery; Options: 1. Standard 2. Express "
                    "Table 2: Delivery times Courier map"
                ),
                "Table row",
            ),
            # The table and the picture in the cell, after the table
            ("Table 2: Delivery times", "Table caption"),
            ("Zone; Days", f"Table header; {nested_context}"),
            ("Zone: Oslo; Days: 1", f"Table row; {nested_context}"),
            ("Courier map", "Picture caption"),
        ]
        assert [image.custom_data["docling"]["label"] for image in document.images] == [
            "picture"
        ]

    def test_from_docling_pictures(self):
        """
        Tests that the pictures of a Docling document that it has image data for
        are attached to the document as images, once each, and that their captions
        are paragraphs, but not the text found inside them.
        """

        def page_pixels(left: int, top: int, right: int, bottom: int) -> bytes:
            """
            Returns the RGB pixels of a region of the image of the page, from its
            top left corner.
            """
            return bytes(
                channel
                for y in range(top, bottom)
                for x in range(left, right)
                for channel in (2 * x, 2 * y, 255 - x - y)
            )

        docling_document = DoclingDocument(name="report")
        caption = docling_document.add_text(
            DocItemLabel.CAPTION, "Figure 1: Uptime in 2026"
        )
        chart = docling_document.add_picture(
            image=_docling_image("red"), caption=caption, prov=_docling_prov(3)
        )
        docling_document.add_text(DocItemLabel.TEXT, "99.9%", parent=chart)
        # A picture that repeats an earlier one, e.g. a logo on each page
        docling_document.add_picture(image=_docling_image("red"))
        # A picture without image data, e.g. of a PDF file converted without
        # picture images
        docling_document.add_picture(prov=_docling_prov(3))
        # A picture without an embedded image, cropped from the image of its page,
        # whose pixels each have a colour of their own
        docling_document.add_page(
            page_no=4,
            size=Size(width=100, height=100),
            image=ImageRef.from_pil(
                PILImage.frombytes("RGB", (100, 100), page_pixels(0, 0, 100, 100)),
                dpi=72,
            ),
        )
        docling_document.add_picture(
            prov=ProvenanceItem(
                page_no=4,
                bbox=BoundingBox(
                    l=10, t=90, r=30, b=70, coord_origin=CoordOrigin.BOTTOMLEFT
                ),
                charspan=(0, 0),
            )
        )
        docling_document.add_text(DocItemLabel.TEXT, "Uptime was above the target.")
        # A picture in a line of text, e.g. an image in Markdown, is in its inline
        # group: it follows the text, with its caption
        line = docling_document.add_inline_group()
        docling_document.add_text(DocItemLabel.TEXT, "The trend", parent=line)
        trend_caption = docling_document.add_text(DocItemLabel.CAPTION, "Trend")
        docling_document.add_picture(
            image=_docling_image("green"), caption=trend_caption, parent=line
        )
        docling_document.add_text(DocItemLabel.TEXT, "is stable.", parent=line)
        # Also in a list item with formatted text
        bullets = docling_document.add_list_group()
        logo_item = docling_document.add_list_item("", parent=bullets)
        logo_text = docling_document.add_inline_group(parent=logo_item)
        docling_document.add_text(DocItemLabel.TEXT, "Our", parent=logo_text)
        docling_document.add_text(DocItemLabel.TEXT, "logo", parent=logo_text)
        docling_document.add_picture(image=_docling_image("yellow"), parent=logo_text)

        with capture_logs() as logs:
            document = Document.from_docling(docling_document)
        assert _texts_and_contexts(document) == [
            ("Figure 1: Uptime in 2026", "Picture caption"),
            ("Uptime was above the target.", None),
            ("The trend is stable.", None),
            ("Trend", "Picture caption"),
            ("Our logo", "List item"),
        ]
        assert len(document.images) == 4
        chart_image, cropped_image, trend_image, logo_image = document.images
        assert trend_image.custom_data["docling"]["self_ref"] == "#/pictures/4"
        assert logo_image.custom_data["docling"]["self_ref"] == "#/pictures/5"
        assert chart_image.mime_type == "image/png"
        assert chart.image is not None
        assert chart_image.base64_data == str(chart.image.uri).split(",", 1)[1]
        assert chart_image.custom_data == {
            "docling": {
                "self_ref": chart.self_ref,
                "label": "picture",
                "prov": [
                    {
                        "page_no": 3,
                        "bbox": {
                            "l": 72.0,
                            "t": 700.0,
                            "r": 520.0,
                            "b": 680.0,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            }
        }
        assert cropped_image.mime_type == "image/png"
        # The box of the picture is 10 to 30 points from the left and from the top
        # of the page
        assert _image_pixels(document)[1] == (
            "RGB",
            (20, 20),
            page_pixels(10, 10, 30, 30),
        )
        assert any(
            "1 picture(s) of the Docling document are not attached" in message
            and "generate_picture_images=True" in message
            for message in logs
        )

    def test_from_docling_image_file_pages(self):
        """
        Tests that the image of each page of a Docling document made of an image
        file, e.g. a scan, is attached to the document before its pictures, with
        the reference and the box of its page, once for pages that repeat each
        other; that the pages without images are logged with what keeps them; and
        that the pages of a PDF file are not attached, though its Docling document
        has their images, which crop its pictures.
        """

        def scan(file_name: str, page_images: bool = True) -> DoclingDocument:
            """
            Returns a Docling document of three pages, as Docling converts a file:
            with the MIME type of a PDF file for an image file too, and the name of
            the file. The third page repeats the first.
            """
            docling_document = DoclingDocument(
                name=Path(file_name).stem,
                origin=DocumentOrigin(
                    filename=file_name, mimetype="application/pdf", binary_hash=1
                ),
            )
            for page_no, color in ((1, "red"), (2, "green"), (3, "red")):
                docling_document.add_page(
                    page_no=page_no,
                    size=Size(width=595, height=842),
                    image=_docling_image(color) if page_images else None,
                )
                docling_document.add_text(
                    DocItemLabel.TEXT, f"Clause {page_no}", prov=_docling_prov(page_no)
                )
            docling_document.add_picture(
                image=_docling_image("blue"), prov=_docling_prov(2)
            )
            return docling_document

        # The extension is not case sensitive
        document = Document.from_docling(scan("contract.TIFF"))
        assert [
            image.custom_data["docling"]["self_ref"] for image in document.images
        ] == [
            "#/pages/1",
            "#/pages/2",
            "#/pictures/0",
        ]
        assert document.images[0].custom_data == {
            "docling": {
                "self_ref": "#/pages/1",
                "label": "page",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 0.0,
                            "t": 842.0,
                            "r": 595.0,
                            "b": 0.0,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
            }
        }
        assert [paragraph.page_numbers for paragraph in document.paragraphs] == [
            [1],
            [2],
            [3],
        ]

        with capture_logs() as logs:
            document = Document.from_docling(scan("contract.png", page_images=False))
        assert [image.custom_data["docling"]["label"] for image in document.images] == [
            "picture"
        ]
        assert any(
            "3 page(s) of the image file of the Docling document are not attached"
            in message
            and "generate_page_images=True" in message
            for message in logs
        )

        # The images of the pages of a PDF file are not attached
        document = Document.from_docling(scan("contract.pdf"))
        assert [image.custom_data["docling"]["label"] for image in document.images] == [
            "picture"
        ]

    def test_from_docling_picture_image_sources(self, tmp_path: Path):
        """
        Tests that a picture whose image is not in the Docling document, or cannot
        be read, is left out without stopping the conversion, with a message that
        says why: an image at a remote URL is not fetched, an image in a local file
        is not read, and a malformed or oversized image is skipped. An image of
        another format or mode is converted to PNG, and an embedded image is given
        the type of its data, whatever type its data URI gives.
        """

        def data_uri(image: PILImage.Image, image_format: str, mime_type: str) -> str:
            """
            Returns a data URI with an image in a format.
            """
            buffer = io.BytesIO()
            image.save(buffer, format=image_format)
            data = base64.b64encode(buffer.getvalue()).decode("ascii")
            return f"data:{mime_type};base64,{data}"

        local_image = tmp_path / "logo.png"
        PILImage.new("RGB", (4, 4), "red").save(local_image)
        # A GIF whose header gives it more pixels than Pillow decodes
        oversized = bytearray(
            base64.b64decode(
                data_uri(PILImage.new("P", (1, 1)), "GIF", "image/gif").split(",")[1]
            )
        )
        oversized[6:10] = (20000).to_bytes(2, "little") * 2
        uris = [
            # An address of the documentation range, which no host has: a fetch
            # would not need a name lookup, and fails the test below
            "https://192.0.2.1/logo.png",
            local_image,
            local_image.as_uri(),
            "data:image/gif;base64",
            "data:image/gif;base64,@@@",
            "data:image/gif;base64," + base64.b64encode(oversized).decode("ascii"),
            data_uri(PILImage.new("P", (4, 4)), "GIF", "image/gif"),
            data_uri(
                PILImage.new("CMYK", (4, 4), (0, 255, 0, 0)), "TIFF", "image/tiff"
            ),
            # A JPEG image that the data URI calls a PNG image, which a model
            # provider can refuse as such
            data_uri(PILImage.new("RGB", (4, 4), "blue"), "JPEG", "image/png"),
            # Not base64, and an SVG image, which Pillow does not read
            "data:image/png;base64,@@@",
            "data:image/svg+xml;base64,"
            + base64.b64encode(b'<svg xmlns="http://www.w3.org/2000/svg"/>').decode(
                "ascii"
            ),
        ]
        docling_document = DoclingDocument(name="pictures")
        docling_document.add_text(DocItemLabel.TEXT, "Text.")
        for uri in uris:
            docling_document.add_picture(
                image=ImageRef.model_validate(
                    {
                        "mimetype": "image/png",
                        "dpi": 72,
                        "size": {"width": 4, "height": 4},
                        "uri": uri,
                    }
                )
            )

        # Each attempt to reach the network is recorded, as the error that it raises
        # could be caught on the way, e.g. as an image that cannot be read
        attempts = []

        def no_network(*args: Any, **kwargs: Any) -> None:
            """
            Records an attempt to reach the network, and fails it.
            """
            attempts.append(args)
            raise AssertionError(f"Network access: {args}")

        with (
            patch.object(socket, "getaddrinfo", no_network),
            patch.object(socket, "create_connection", no_network),
            patch.object(socket.socket, "connect", no_network),
            patch.object(socket.socket, "connect_ex", no_network),
            capture_logs() as logs,
        ):
            document = Document.from_docling(docling_document)
        assert not attempts
        assert [
            (image.custom_data["docling"]["self_ref"], image.mime_type)
            for image in document.images
        ] == [
            ("#/pictures/6", "image/png"),
            ("#/pictures/7", "image/png"),
            ("#/pictures/8", "image/jpeg"),
        ]
        assert docling_document.pictures[8].image is not None
        assert (
            document.images[2].base64_data
            == str(docling_document.pictures[8].image.uri).split(",", 1)[1]
        )
        for image, image_format in zip(
            document.images, ["PNG", "PNG", "JPEG"], strict=True
        ):
            with PILImage.open(io.BytesIO(base64.b64decode(image.base64_data))) as png:
                assert png.format == image_format
                assert png.size == (4, 4)
        # The pictures that are not attached are counted by where their image is
        not_attached = [message for message in logs if "are not attached" in message]
        assert len(not_attached) == 3
        assert not_attached[0].startswith("2 picture(s)")
        assert "load_local_images()" in not_attached[0]
        assert not_attached[1].startswith("1 picture(s)")
        assert "at URLs, which are not fetched" in not_attached[1]
        assert not_attached[2].startswith("5 picture(s)")
        assert "image data cannot be read" in not_attached[2]

    def test_from_docling_forms(self):
        """
        Tests that each form field and each key-value pair of a Docling document is
        a paragraph with its key and its value.
        """
        docling_document = DoclingDocument(name="form")
        region = docling_document.add_field_region()
        name_field = docling_document.add_field_item(parent=region)
        # The number of the field is left out
        docling_document.add_text(DocItemLabel.MARKER, "1", parent=name_field)
        docling_document.add_field_key("Full name:", parent=name_field)
        docling_document.add_field_value("Ada Lovelace", parent=name_field)
        for key, checkbox_label in (
            ("I agree to the terms", DocItemLabel.CHECKBOX_SELECTED),
            ("Subscribe to the newsletter", DocItemLabel.CHECKBOX_UNSELECTED),
        ):
            field = docling_document.add_field_item(parent=region)
            docling_document.add_field_key(key, parent=field)
            value = docling_document.add_field_value("", parent=field)
            docling_document.add_text(checkbox_label, "", parent=value)
        docling_document.add_key_values(
            graph=GraphData(
                cells=[
                    GraphCell(
                        label=GraphCellLabel.KEY,
                        cell_id=0,
                        text="Invoice no.",
                        orig="Invoice no.",
                    ),
                    GraphCell(
                        label=GraphCellLabel.VALUE,
                        cell_id=1,
                        text="INV-2026-042",
                        orig="INV-2026-042",
                    ),
                    GraphCell(
                        label=GraphCellLabel.VALUE, cell_id=2, text="Paid", orig="Paid"
                    ),
                ],
                links=[
                    GraphLink(
                        label=GraphLinkLabel.TO_VALUE,
                        source_cell_id=0,
                        target_cell_id=1,
                    )
                ],
            )
        )
        docling_document.add_text(DocItemLabel.CHECKBOX_SELECTED, "Express delivery")

        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Full name: Ada Lovelace", "Form field"),
            ("I agree to the terms: checked", "Form field"),
            ("Subscribe to the newsletter: unchecked", "Form field"),
            ("Invoice no.: INV-2026-042", "Key-value pair"),
            ("Paid", "Key-value pair"),
            ("Express delivery", "Checkbox, checked"),
        ]

        # Where the cells have no provenance, only their region, e.g. from DocTags,
        # a pair is on the pages of the region
        docling_document = DoclingDocument(name="invoice")
        docling_document.add_key_values(
            graph=GraphData(
                cells=[
                    GraphCell(
                        label=GraphCellLabel.KEY, cell_id=0, text="Total", orig="Total"
                    ),
                    GraphCell(
                        label=GraphCellLabel.VALUE,
                        cell_id=1,
                        text="100 EUR",
                        orig="100 EUR",
                    ),
                ],
                links=[
                    GraphLink(
                        label=GraphLinkLabel.TO_VALUE,
                        source_cell_id=0,
                        target_cell_id=1,
                    )
                ],
            ),
            prov=_docling_prov(2),
        )
        assert [
            (paragraph.raw_text, paragraph.page_numbers)
            for paragraph in Document.from_docling(docling_document).paragraphs
        ] == [("Total: 100 EUR", [2])]

        # A value can be linked to its key, and a pair both ways
        to_key = GraphLink(
            label=GraphLinkLabel.TO_KEY, source_cell_id=1, target_cell_id=0
        )
        to_value = GraphLink(
            label=GraphLinkLabel.TO_VALUE, source_cell_id=0, target_cell_id=1
        )
        for links in ([to_key], [to_value, to_key]):
            docling_document = DoclingDocument(name="invoice")
            docling_document.add_key_values(
                graph=GraphData(
                    cells=[
                        GraphCell(
                            label=GraphCellLabel.KEY,
                            cell_id=0,
                            text="Total",
                            orig="Total",
                        ),
                        GraphCell(
                            label=GraphCellLabel.VALUE,
                            cell_id=1,
                            text="100 EUR",
                            orig="100 EUR",
                        ),
                    ],
                    links=links,
                )
            )
            assert _texts_and_contexts(Document.from_docling(docling_document)) == [
                ("Total: 100 EUR", "Key-value pair")
            ]

        # A value that is also the key of other values has a paragraph of its own
        # with them, as have two cells that are each the value of the other
        for texts, links, expected in (
            (
                ["Shipping address", "Street", "Main Street 1"],
                [(0, 1), (1, 2)],
                ["Shipping address: Street", "Street: Main Street 1"],
            ),
            (
                ["Total", "100 EUR"],
                [(0, 1), (1, 0)],
                ["Total: 100 EUR", "100 EUR: Total"],
            ),
        ):
            docling_document = DoclingDocument(name="delivery")
            docling_document.add_key_values(
                graph=GraphData(
                    cells=[
                        GraphCell(
                            label=GraphCellLabel.KEY, cell_id=i, text=text, orig=text
                        )
                        for i, text in enumerate(texts)
                    ],
                    links=[
                        GraphLink(
                            label=GraphLinkLabel.TO_VALUE,
                            source_cell_id=source,
                            target_cell_id=target,
                        )
                        for source, target in links
                    ],
                )
            )
            assert [
                paragraph.raw_text
                for paragraph in Document.from_docling(docling_document).paragraphs
            ] == expected

        # Links of other kinds, e.g. to a child or to a parent, do not pair a key
        # with a value, and links to cells that the graph does not have are left out
        docling_document = DoclingDocument(name="invoice")
        key_values = docling_document.add_key_values(
            graph=GraphData(
                cells=[
                    GraphCell(
                        label=GraphCellLabel.KEY,
                        cell_id=0,
                        text="Invoice no.",
                        orig="Invoice no.",
                    ),
                    GraphCell(
                        label=GraphCellLabel.VALUE,
                        cell_id=1,
                        text="INV-2026-042",
                        orig="INV-2026-042",
                    ),
                ],
                links=[
                    GraphLink(
                        label=GraphLinkLabel.TO_CHILD,
                        source_cell_id=0,
                        target_cell_id=1,
                    ),
                    GraphLink(
                        label=GraphLinkLabel.TO_PARENT,
                        source_cell_id=1,
                        target_cell_id=0,
                    ),
                ],
            )
        )
        # Docling checks the cells of the links when the graph is created, not
        # after
        key_values.graph.links += [
            GraphLink(
                label=GraphLinkLabel.TO_VALUE, source_cell_id=0, target_cell_id=7
            ),
            GraphLink(label=GraphLinkLabel.TO_KEY, source_cell_id=1, target_cell_id=9),
        ]
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Invoice no.", "Key-value pair"),
            ("INV-2026-042", "Key-value pair"),
        ]

    def test_from_docling_code_blocks(self):
        """
        Tests that the caption of a code block is a paragraph before it, and its
        footnote one after it, also where the body of the Docling document has the
        caption after the code block, e.g. from AsciiDoc or a PDF file, and that
        the items under each follow it.
        """
        docling_document = DoclingDocument(name="manual")
        docling_document.add_heading("Configuration", level=1)
        code = docling_document.add_code('{ "retries": 3 }')
        docling_document.add_text(DocItemLabel.TEXT, "The default is 3.", parent=code)
        caption = docling_document.add_text(DocItemLabel.CAPTION, "Listing 1: Retries")
        code.captions.append(caption.get_ref())
        footnote = docling_document.add_text(DocItemLabel.FOOTNOTE, "In seconds:")
        delays = docling_document.add_list_group(parent=footnote)
        docling_document.add_list_item("2 after a timeout;", parent=delays)
        docling_document.add_list_item("10 after a server error.", parent=delays)
        code.footnotes.append(footnote.get_ref())
        docling_document.add_heading("Logging", level=1)

        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Configuration", "Heading, level 1"),
            ("Listing 1: Retries", "Code caption; Section: Configuration"),
            ('{ "retries": 3 }', "Code; Section: Configuration"),
            ("In seconds:", "Code footnote; Section: Configuration"),
            ("2 after a timeout;", "List item; Section: Configuration"),
            ("10 after a server error.", "List item; Section: Configuration"),
            ("The default is 3.", "Section: Configuration"),
            ("Logging", "Heading, level 1"),
        ]

    def test_from_docling_transcripts(self):
        """
        Tests that a paragraph from a track, e.g. a cue of a WebVTT subtitle file,
        has the speaker of the track in its additional context, and the track in its
        custom data, with its times.
        """
        docling_document = DoclingDocument(name="meeting")
        docling_document.add_text(
            DocItemLabel.TEXT,
            "Shall we start with the budget?",
            source=TrackSource(
                start_time=4.9, end_time=8.5, identifier="cue-1", voice="Ann Berg"
            ),
        )
        # A cue with formatted text is an inline group, each part of it from
        # the track
        inline = docling_document.add_inline_group()
        for part in ("The budget is", "approved", "."):
            docling_document.add_text(
                DocItemLabel.TEXT,
                part,
                parent=inline,
                source=TrackSource(start_time=8.5, end_time=11.0, voice="Bo Lind"),
            )
        # A cue without a voice
        docling_document.add_text(
            DocItemLabel.TEXT,
            "Applause.",
            source=TrackSource(start_time=11.0, end_time=12.0),
        )

        document = Document.from_docling(docling_document)
        assert _texts_and_contexts(document) == [
            ("Shall we start with the budget?", "Speaker: Ann Berg"),
            ("The budget is approved.", "Speaker: Bo Lind"),
            ("Applause.", None),
        ]
        assert document.paragraphs[0].custom_data["docling"]["source"] == [
            {
                "kind": "track",
                "start_time": 4.9,
                "end_time": 8.5,
                "identifier": "cue-1",
                "voice": "Ann Berg",
            }
        ]
        # The track of the parts of the inline group is given once
        assert document.paragraphs[1].custom_data["docling"]["source"] == [
            {"kind": "track", "start_time": 8.5, "end_time": 11.0, "voice": "Bo Lind"}
        ]
        # The times are not pages
        assert all(not paragraph.page_numbers for paragraph in document.paragraphs)
        # A paragraph from no track has no track in its custom data
        docling_document = DoclingDocument(name="notes")
        docling_document.add_text(DocItemLabel.TEXT, "No speaker here.")
        paragraph = Document.from_docling(docling_document).paragraphs[0]
        assert "source" not in paragraph.custom_data["docling"]
        self.check_instance_serialization_and_cloning(document)

    def test_from_docling_sheets_and_slides(self):
        """
        Tests that the additional context of a paragraph made from a spreadsheet or
        a presentation names the sheet or the slide of the paragraph.
        """
        docling_document = DoclingDocument(name="workbook")
        sheet = docling_document.add_group(label=GroupLabel.SHEET, name="Q3 figures")
        docling_document.add_table(
            data=_docling_table_data(
                [
                    _docling_cell(0, 0, "Region", header=True),
                    _docling_cell(0, 1, "Revenue", header=True),
                    _docling_cell(1, 0, "North"),
                    _docling_cell(1, 1, "1.2M"),
                ]
            ),
            parent=sheet,
        )
        # Docling names the group of each slide after its index, from 0
        slide = docling_document.add_group(label=GroupLabel.CHAPTER, name="slide-1")
        docling_document.add_title("Results", parent=slide)
        docling_document.add_list_item(
            "Revenue grew by 12%.",
            parent=docling_document.add_list_group(parent=slide),
        )
        docling_document.add_text(DocItemLabel.TEXT, "After the slides.")

        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Region; Revenue", "Table header; Sheet: Q3 figures"),
            ("Region: North; Revenue: 1.2M", "Table row; Sheet: Q3 figures"),
            ("Results", "Title; Slide 2"),
            ("Revenue grew by 12%.", "List item; Slide 2"),
            ("After the slides.", None),
        ]

        # In a presentation with several titles, the content of a slide is in
        # the section of its title, which ends with the slide
        docling_document = DoclingDocument(name="presentation")
        for index, title in enumerate(("Agenda", "Results", None)):
            slide = docling_document.add_group(
                label=GroupLabel.SLIDE, name=f"slide-{index}"
            )
            if title:
                docling_document.add_title(title, parent=slide)
            docling_document.add_text(
                DocItemLabel.TEXT, f"Text of slide {index + 1}.", parent=slide
            )
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("Agenda", "Title; Slide 1"),
            ("Text of slide 1.", "Slide 1; Section: Agenda"),
            ("Results", "Title; Slide 2"),
            ("Text of slide 2.", "Slide 2; Section: Results"),
            ("Text of slide 3.", "Slide 3"),
        ]

        # A sheet without a name, and a group whose name only starts as that of
        # a slide, are not named
        docling_document = DoclingDocument(name="unnamed")
        for label, name in (
            (GroupLabel.SHEET, " "),
            (GroupLabel.SLIDE, "slide-3-notes"),
        ):
            group = docling_document.add_group(label=label, name=name)
            docling_document.add_text(
                DocItemLabel.TEXT, f"In the {label.value}.", parent=group
            )
        assert _texts_and_contexts(Document.from_docling(docling_document)) == [
            ("In the sheet.", None),
            ("In the slide.", None),
        ]

    def test_from_docling_errors_and_document_attributes(self):
        """
        Tests the errors of creating a document from a Docling document, and that
        the other attributes of the document are passed to it.
        """
        docling_document = DoclingDocument(name="footer only")
        docling_document.add_text(
            DocItemLabel.PAGE_FOOTER, "Page 1", content_layer=ContentLayer.FURNITURE
        )
        with pytest.raises(ValueError, match="no text or images"):
            Document.from_docling(docling_document)
        docling_document.add_text(DocItemLabel.TEXT, "The only paragraph.")

        with pytest.raises(TypeError, match="Expected a DoclingDocument"):
            Document.from_docling("The only paragraph.")  # ty: ignore[invalid-argument-type]

        class _ConversionResult:
            """
            Stands in for the result of a Docling conversion, which holds the
            Docling document.
            """

            document = docling_document

        with pytest.raises(TypeError, match=r"from_docling\(result\.document\)"):
            Document.from_docling(_ConversionResult())  # ty: ignore[invalid-argument-type]

        for name, value in (
            ("raw_text", "Text."),
            ("paragraphs", [Paragraph(raw_text="Text.")]),
            ("images", []),
        ):
            with pytest.raises(TypeError, match=f"`{name}` cannot be passed"):
                Document.from_docling(docling_document, **{name: value})

        document = Document.from_docling(
            docling_document,
            aspects=[Aspect(name="Fees", description="Fees and payment terms")],
            sat_model_id="sat-1l-sm",
        )
        assert [aspect.name for aspect in document.aspects] == ["Fees"]
        assert document.sat_model_id == "sat-1l-sm"
        assert document.raw_text == "The only paragraph."

    def test_from_docling_without_docling_core(self):
        """
        Tests that creating a document from a Docling document without docling-core,
        or with a docling-core older than the converter needs, raises an error that
        names the extra that installs it.

        Also tests that the oldest docling-core that the converter accepts is
        the lower bound of docling-core in the `docling` extra and in the dev group.
        """
        installed_version = importlib.metadata.version
        module_name = "contextgem.internal.converters.docling"
        # Each import of the converter module below makes it the attribute of its
        # package, which `patch.dict(sys.modules)` does not restore: the package
        # would keep a module of its own, which a later test could patch in vain
        converters_package = sys.modules["contextgem.internal.converters"]
        patch_package = patch.object(converters_package, "docling", converter_module)

        def fake_version(name: str, docling_core_version: str | None) -> str:
            """
            Returns the installed version of a package, or the given one for
            docling-core, which is not installed for None.
            """
            if name != "docling-core":
                return installed_version(name)
            if docling_core_version is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return docling_core_version

        for docling_core_version, message in (
            (None, r"requires docling-core, the document model of Docling"),
            (
                "2.102.0",
                (
                    r"requires docling-core 2\.103\.0 or later, "
                    r"and docling-core 2\.102\.0 is installed"
                ),
            ),
            ("2.48.0", r"docling-core 2\.48\.0 is installed"),
        ):
            with (
                patch.dict(sys.modules),
                patch_package,
                patch.object(
                    importlib.metadata,
                    "version",
                    lambda name, v=docling_core_version: fake_version(name, v),
                ),
            ):
                # The converter module is imported again, and checks docling-core
                sys.modules.pop(module_name, None)
                with pytest.raises(ImportError, match=message) as error:
                    Document.from_docling(DoclingDocument(name="any"))
                assert 'uv add "contextgem[docling]"' in str(error.value)
        # A newer docling-core, or a pre-release of the oldest, is accepted
        for docling_core_version in ("2.103.0rc1", "2.110.0", "3.0.0"):
            with (
                patch.dict(sys.modules),
                patch_package,
                patch.object(
                    importlib.metadata,
                    "version",
                    lambda name, v=docling_core_version: fake_version(name, v),
                ),
            ):
                sys.modules.pop(module_name, None)
                document = Document.from_docling(
                    DoclingDocument.load_from_json(
                        _DOCLING_FILES_DIR / "meeting_notes.md.json"
                    )
                )
                assert document.paragraphs
                # The module imported again is the attribute of its package
                assert converters_package.docling is not converter_module
        # The converter module of the other tests is back in both places
        assert sys.modules[module_name] is converter_module
        assert converters_package.docling is converter_module

        # The oldest docling-core that the converter accepts is the lower bound of
        # the `docling` extra, and of the dev group, which tests the extra
        project = tomllib.loads(
            (Path(__file__).parent.parent / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        requirements = [
            *project["project"]["optional-dependencies"]["docling"],
            *project["dependency-groups"]["dev"],
        ]
        lower_bounds = set()
        for requirement in requirements:
            if requirement.startswith("docling-core "):
                lower_bound = re.search(r">=\s*([\d.]+)", requirement)
                assert lower_bound, requirement
                lower_bounds.add(lower_bound.group(1))
        assert lower_bounds == {converter_module._MIN_DOCLING_CORE_VERSION}

    @_SKIP_NEW_PROCESS_ON_WINDOWS_PY311
    def test_import_without_docling_core(self):
        """
        Tests that ContextGem is imported without importing docling-core, which
        only the `docling` extra installs.

        The import runs in a new Python process, in which nothing has imported
        docling-core before.
        """
        script = "import sys\nimport contextgem\nprint('docling_core' in sys.modules)\n"
        result = subprocess.run(  # nosec B603 - the Python of the tests, no shell
            [sys.executable, "-c", script],
            env={**os.environ, "LITELLM_LOCAL_MODEL_COST_MAP": "True"},
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip().splitlines()[-1] == "False"

    def test_from_docling_converted_pdf(self):
        """
        Tests a document created from a PDF file converted with Docling.

        A PDF file needs the models of Docling, which run on PyTorch, and is not
        converted in the tests: `service_agreement.pdf.json` in
        `tests/other_files/docling/` is the Docling document that full Docling
        converted `service_agreement.pdf` into, without OCR. It is written and
        checked against the Docling of `uv.lock` with `uv run fab docling-fixtures`.
        """
        docling_document = DoclingDocument.load_from_json(
            _DOCLING_FILES_DIR / "service_agreement.pdf.json"
        )
        document = Document.from_docling(docling_document)
        _check_docling_text_kept(docling_document, document)
        texts_and_contexts = _texts_and_contexts(document)
        assert len(texts_and_contexts) == 26
        assert texts_and_contexts[:3] == [
            ("SERVICE AGREEMENT", "Heading, level 1"),
            (
                (
                    'This Service Agreement (the "Agreement") is made on 1 March 2026 '
                    'between Northwind Analytics AS, Oslo, Norway (the "Supplier"), '
                    "and Riverbank Logistics Ltd, Leeds, United Kingdom (the "
                    '"Customer").'
                ),
                "Section: SERVICE AGREEMENT",
            ),
            ("1. Services", "Heading, level 1"),
        ]
        # The rows of the table, each with the headers of the columns
        fees_context = (
            "Table caption: Table 1: Monthly fees; Section: 2. Fees and Payment"
        )
        assert texts_and_contexts[7:12] == [
            ("Table 1: Monthly fees", "Table caption; Section: 2. Fees and Payment"),
            ("Service; Monthly fee; Billing", f"Table header; {fees_context}"),
            (
                "Service: Hosting; Monthly fee: 1,200 EUR; Billing: In advance",
                f"Table row; {fees_context}",
            ),
            (
                "Service: Monitoring; Monthly fee: 300 EUR; Billing: In advance",
                f"Table row; {fees_context}",
            ),
            (
                "Service: Support; Monthly fee: 450 EUR; Billing: In arrears",
                f"Table row; {fees_context}",
            ),
        ]
        assert texts_and_contexts[19:22] == [
            ("indirect or consequential losses;", "List item; Section: 4. Liability"),
            ("loss of profit or revenue;", "List item; Section: 4. Liability"),
            (
                (
                    "loss of data, except where caused by a breach of the backup "
                    "obligations of the Supplier."
                ),
                "List item; Section: 4. Liability",
            ),
        ]
        # The page footers are not converted
        assert not any("page 1 of 2" in text for text, _ in texts_and_contexts)
        # The page of each paragraph, e.g. to cite it, with its box on the page
        pages = {
            paragraph.raw_text[:3]: paragraph.page_numbers
            for paragraph in document.paragraphs
        }
        assert pages["1.1"] == [1]
        assert pages["3.1"] == [2]
        assert all(paragraph.page_numbers for paragraph in document.paragraphs)
        prov = document.paragraphs[3].custom_data["docling"]["prov"]
        assert len(prov) == 1
        assert prov[0]["page_no"] == 1
        assert set(prov[0]["bbox"]) == {"l", "t", "r", "b", "coord_origin"}
        # The additional context and the custom data are kept in serialization
        self.check_instance_serialization_and_cloning(document)

    # Docling skips the broken pictures of its test presentations, and reads the
    # rows of its test CSV file that are longer than the header, with a warning
    @pytest.mark.filterwarnings("ignore:Skipping malformed picture shape:UserWarning")
    @pytest.mark.filterwarnings("ignore:Inconsistent column lengths:UserWarning")
    @pytest.mark.parametrize("file_name", _DOCLING_CONVERTED_FILES)
    def test_from_docling_converted_files(self, file_name: str):
        """
        Tests documents created from files in the formats that Docling converts
        without models (`TEST_CONVERTED_SUFFIXES`): DOCX, XLSX, PPTX, HTML,
        Markdown, CSV, AsciiDoc, LaTeX, WebVTT, EPUB, JATS and USPTO XML, those
        written for ContextGem and test documents of Docling (`docling_tests/`).

        Each file of `tests/other_files/docling/` is converted in the test with the
        torch-free docling-slim of the dev group, i.e. with the Docling of
        `uv.lock`. ContextGem must make the same paragraphs and images from it as
        from the committed Docling document, which full Docling converted the file
        into. When they differ after an upgrade of Docling, check the new output,
        and write the Docling documents again with
        `uv run fab docling-fixtures --write`.
        """
        path = _DOCLING_FILES_DIR / file_name
        with conversion_input(path) as source:
            converted = _docling_converter(path).convert(source).document
        document = Document.from_docling(converted)
        committed = Document.from_docling(
            DoclingDocument.load_from_json(path.with_name(f"{path.name}.json"))
        )
        assert _paragraph_summary(document) == _paragraph_summary(committed)
        # The same pictures: their bytes differ with the version of Pillow, which
        # Docling encodes them with
        assert _image_pixels(document) == _image_pixels(committed)
        _check_docling_text_kept(converted, document)
        assert all(
            "\n" not in (paragraph.additional_context or "")
            for paragraph in document.paragraphs
        )
        if file_name in _DOCLING_FILE_PARAGRAPHS:
            for paragraph in _DOCLING_FILE_PARAGRAPHS[file_name]:
                assert paragraph in _paragraph_summary(document)
            # The picture of the DOCX file, but not its page header and footer
            assert len(document.images) == (1 if file_name.endswith(".docx") else 0)
            assert not any("Confidential" in p.raw_text for p in document.paragraphs)

    # docling-core moves the boxes of the cells of the test spreadsheet of Docling
    # onto its page, with a warning
    @pytest.mark.filterwarnings("ignore:Provenance bbox coordinate:UserWarning")
    @pytest.mark.parametrize("file_name", _DOCLING_UNCONVERTED_FILES)
    def test_from_docling_unconverted_files(self, file_name: str):
        """
        Tests documents created from the Docling documents of the files of
        `tests/other_files/docling/` that the tests do not convert, which full
        Docling converted them into, those written for ContextGem and test
        documents of Docling (`docling_tests/`): a PDF file or a Google Books
        package (METS GBS) needs the models of Docling, and a scanned PDF file its
        OCR too; an email (EML, MSG), OpenDocument file (ODT, ODS, ODP) or XBRL
        report packages that the dev group does not have; and a legacy Office file
        (DOC, XLS, PPT) or an RTF file LibreOffice, which Docling converts it with.
        """
        path = _DOCLING_FILES_DIR / file_name
        docling_document = DoclingDocument.load_from_json(
            path.with_name(f"{path.name}.json")
        )
        document = Document.from_docling(docling_document)
        assert document.paragraphs
        _check_docling_text_kept(docling_document, document)
        for paragraph in document.paragraphs:
            assert "\n" not in (paragraph.additional_context or "")
            # Each paragraph of a file with pages read with OCR, or of a PDF file,
            # is on a page
            if path.suffix in {".pdf", ".png", ".tiff", ".gz"}:
                assert paragraph.page_numbers
            # Each segment of a transcript has its times
            if path.suffix in {".mp3", ".mp4"}:
                assert paragraph.page_numbers == []
                assert paragraph.custom_data["docling"]["source"]
        # The image of each page of an image file is attached, before the pictures
        if path.suffix in {".png", ".tiff"}:
            labels = [
                image.custom_data["docling"]["label"] for image in document.images
            ]
            assert labels[: len(docling_document.pages)] == ["page"] * len(
                docling_document.pages
            )
        # The frames of a video are attached, each with its time
        if path.suffix == ".mp4":
            assert document.images
            for image in document.images:
                assert image.custom_data["docling"]["source"][0]["start_time"] >= 0

    @pytest.mark.filterwarnings("ignore:Provenance bbox coordinate:UserWarning")
    @pytest.mark.parametrize(
        "file_name", [*_DOCLING_CONVERTED_FILES, *_DOCLING_UNCONVERTED_FILES]
    )
    def test_from_docling_fixture_outputs(self, file_name: str):
        """
        Tests that ContextGem makes of the committed Docling document of each file of
        `tests/other_files/docling/` what its committed output holds
        (`<file name>.contextgem.jsonl`): each paragraph, with its text, additional
        context and page numbers, and each image. The outputs show in review how
        a change of the converter, or of the Docling documents after an upgrade of
        Docling, changes what ContextGem makes of these files.
        """
        path = _DOCLING_FILES_DIR / file_name
        output = contextgem_output(
            DoclingDocument.load_from_json(path.with_name(f"{path.name}.json"))
        )
        assert output == read_output(output_path(path)), (
            "If the new output is right, write the outputs with "
            "`uv run fab docling-fixtures --outputs`"
        )

    @pytest.mark.filterwarnings("ignore:Provenance bbox coordinate:UserWarning")
    @pytest.mark.parametrize(
        "file_name", [*_DOCLING_CONVERTED_FILES, *_DOCLING_UNCONVERTED_FILES]
    )
    def test_from_docling_fixture_sentences(self, file_name: str):
        """
        Tests the sentences of the paragraphs that ContextGem makes of the committed
        Docling document of each file of `tests/other_files/docling/`.

        A paragraph that is not running text, e.g. a table row, a heading or code,
        is one sentence from the start, its whole text: the SaT model, which is
        made for running text, splits e.g. a table row within the text of a cell.
        The SaT model segments each other paragraph into sentences that are in
        its text, in order, and leave out nothing of it but whitespace. Each
        sentence has the additional context and the pages of its paragraph, and
        a copy of its custom data.
        """
        path = _DOCLING_FILES_DIR / file_name
        docling_document = DoclingDocument.load_from_json(
            path.with_name(f"{path.name}.json")
        )
        document = Document.from_docling(docling_document)
        for paragraph in document.paragraphs:
            whole = (
                paragraph.custom_data["docling"]["label"]
                in _DOCLING_ONE_SENTENCE_LABELS
            )
            assert [sentence.raw_text for sentence in paragraph.sentences] == (
                [paragraph.raw_text] if whole else []
            ), paragraph.raw_text

        document = Document.from_docling(docling_document, pre_segment_sentences=True)
        for paragraph in document.paragraphs:
            sentences = [sentence.raw_text for sentence in paragraph.sentences]
            assert sentences, paragraph.raw_text
            if paragraph.custom_data["docling"]["label"] in (
                _DOCLING_ONE_SENTENCE_LABELS
            ):
                assert sentences == [paragraph.raw_text]
            start = 0
            for sentence in sentences:
                found = paragraph.raw_text.find(sentence, start)
                assert found >= 0, (paragraph.raw_text, sentence)
                start = found + len(sentence)
            assert re.sub(r"\s", "", "".join(sentences)) == re.sub(
                r"\s", "", paragraph.raw_text
            )
            for sentence in paragraph.sentences:
                assert sentence.additional_context == paragraph.additional_context
                assert sentence.page_numbers == paragraph.page_numbers
                assert sentence.custom_data == paragraph.custom_data
                assert sentence.custom_data is not paragraph.custom_data

    def test_from_docling_fixture_files(self):
        """
        Tests that each file of `tests/other_files/docling/` is a fixture file in
        a format of `SOURCE_SUFFIXES`, which the tests above and
        `uv run fab docling-fixtures` take, its Docling document
        (`<file name>.json`), its output (`<file name>.contextgem.jsonl`), what
        Docling reads it with (the layout of an EBCDIC file, the taxonomy of an XBRL
        report), or a known other file, and that each fixture file has its Docling
        document, its output and what it is read with. A file in another format
        would be left out of the tests without notice.
        """
        # The licenses of the test documents of Docling and of the EBCDIC sample
        # that Docling took from ebcdic-parser
        other_files = {"docling_tests/LICENSE", "docling_tests/LICENSE-ebcdic-parser"}
        files = {
            path.relative_to(_DOCLING_FILES_DIR).as_posix()
            for path in _DOCLING_FILES_DIR.rglob("*")
            if path.is_file()
        }
        sources = {*_DOCLING_CONVERTED_FILES, *_DOCLING_UNCONVERTED_FILES}
        companions = set()
        for name in sources:
            path = _DOCLING_FILES_DIR / name
            companions |= {path.with_name(f"{path.name}.json"), output_path(path)}
            if path.suffix == ".ebc":
                companions.add(layout_path(path))
            if path.suffix == ".xbrl":
                assert any(taxonomy_path(path).glob("*.xsd")), name
                companions |= set(taxonomy_path(path).iterdir())
        companions = {
            path.relative_to(_DOCLING_FILES_DIR).as_posix() for path in companions
        }
        assert not companions - files, "Missing companions of fixture files"
        assert files - sources - companions == other_files, "Unknown files"

    def test_from_docling_supported_formats(self):
        """
        Tests that the table of the supported formats in the documentation of
        `Document.from_docling()` (`docs/source/documents/docling.rst`) lists the
        extensions of the fixture files (`SOURCE_SUFFIXES`, by their last suffix,
        e.g. `.gz` for `.tar.gz`), and Docling JSON (`.json`), which the tests load
        the Docling document of each fixture file from, and no other: a format is
        supported when the tests cover it, and each format that they cover is
        listed.
        """
        extensions = {
            "." + extension.rsplit(".", 1)[-1]
            for extension in _docling_formats_extensions()
        }
        assert extensions == {*SOURCE_SUFFIXES, ".json"}

    def test_from_docling_readme_formats(self):
        """
        Tests that the table of formats under the Docling quick start of the README
        (`dev/readme.template.md`) names each format of the table of the supported
        formats in the documentation, by its name in `_README_FORMAT_NAMES` or by
        its extension in capitals, and no other, and that the icon of each of its
        cells exists: a format that becomes supported is shown in the README too.
        """
        root = Path(__file__).parents[1]
        template = (root / "dev" / "readme.template.md").read_text(encoding="utf-8")
        section = template.split("### Start from a file with Docling", 1)[1]
        table = section.split("<table", 1)[1].split("</table>", 1)[0]
        cells = re.findall(
            r'<img src="(assets/icons/format-[a-z-]+\.svg)"[^>]*><br><sub>(.+?)</sub>',
            table,
        )
        assert len(cells) == table.count("<td")
        for icon, _ in cells:
            assert (root / icon).is_file(), icon
        names = {
            name.strip()
            for _, label in cells
            for name in re.split(r",|<br>", label)
            if name.strip()
        }
        assert names == {
            _README_FORMAT_NAMES.get(extension, extension[1:].upper())
            for extension in _docling_formats_extensions()
        }
