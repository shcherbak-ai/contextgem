# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Writes the Docling document of each evaluation case, from which the harness can
create the document of the case with ``Document.from_docling()`` (``--input
docling`` of ``run_eval`` and ``decision_probabilities``), to compare extraction
from the structure that Docling gives a file with extraction from the text.

The text of a case has one paragraph per line, and its structure in its lines:
the section headings, which ``cases/<case>.neutral.json`` lists, and in
``complex_mcu``, tables, a line per row that starts with "|", most of them under
a title such as "Table 2. ...". This script renders the text as HTML, as
a document of these sections and tables would be published: each heading at the
level of its section, the title of a table as its caption, and the first row of
a table as its header row. It converts the HTML with Docling (the docling-slim of
the dev group, which needs no models for HTML), and writes the Docling document
as ``cases/<case>.docling.json``, and the HTML that it converted as
``cases/<case>.docling.html``, to be reviewed and committed.

The HTML backend of Docling replaces some characters with ASCII ones, e.g. "—"
in a table cell with "-", which a PDF file of the same text would keep. Where
a cell of a table differs from the case only by this replacement, its text is
set back to that of the case, so that the Docling document differs from the text
only in its structure. Each document is then checked: the paragraphs that
``Document.from_docling()`` makes of it are those of the reference answers of
the case, position by position, as the scores refer to them by their positions.

Sends no request to a model. Run it from the root of the repo:

```bash
uv run python -m dev.evals.docling_cases
```
"""

from __future__ import annotations

import html
import io
import json
import re
import sys

from docling.backend.html_backend import HTMLDocumentBackend
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DoclingDocument

from dev.evals.harness import CASE_NAMES, CASES_DIR, load_case, table_cells


def _heading_level(case: str, position: int, heading: str) -> int:
    """
    Returns the HTML level of a section heading of a case: 1 for the title of
    the document, 2 for a top-level section, and 3 for a section within one.

    :param case: The name of the case.
    :type case: str
    :param position: The position of the heading among the paragraphs.
    :type position: int
    :param heading: The text of the heading.
    :type heading: str
    :return: The level, 1 to 3.
    :rtype: int
    """
    if position == 0:
        return 1
    if case == "complex_dpa":
        # The clauses of the agreement, its annexes, its two amendments and its
        # records are top-level sections; the clauses and the annex of
        # an amendment are sections of the amendment
        top_level = re.match(
            r"(RECITALS$|\d+\. |ANNEX \d|AMENDMENT NO\. |OPERATIONAL RECORDS)", heading
        )
        return 2 if top_level else 3
    if case == "complex_mcu":
        # The parts of the package, and the numbered sections and the records
        # within them
        return 2 if heading.startswith("PART ") else 3
    return 2


def _table_html(rows: list[list[str]], caption: str | None) -> list[str]:
    """
    Renders a table as HTML, with its first row as its header row.

    :param rows: The values of the cells of each row.
    :type rows: list[list[str]]
    :param caption: The title of the table, or None.
    :type caption: str | None
    :return: The lines of HTML.
    :rtype: list[str]
    """
    lines = ["<table>"]
    if caption is not None:
        lines.append(f"<caption>{html.escape(caption)}</caption>")
    header = "".join(f"<th>{html.escape(value)}</th>" for value in rows[0])
    lines += [f"<thead><tr>{header}</tr></thead>", "<tbody>"]
    for row in rows[1:]:
        cells = "".join(f"<td>{html.escape(value)}</td>" for value in row)
        lines.append(f"<tr>{cells}</tr>")
    lines.append("</tbody></table>")
    return lines


def render_html(case: str) -> tuple[str, list[list[list[str]]]]:
    """
    Renders the text of a case as HTML.

    :param case: The name of the case.
    :type case: str
    :return: The HTML, and the values of the cells of each of its tables, row by
        row, in the order of the tables.
    :rtype: tuple[str, list[list[list[str]]]]
    """
    lines = (CASES_DIR / f"{case}.txt").read_text(encoding="utf-8").split("\n")
    headings = set(
        json.loads((CASES_DIR / f"{case}.neutral.json").read_text(encoding="utf-8"))[
            "headings"
        ]
    )
    out = ['<!DOCTYPE html>\n<html><head><meta charset="utf-8"></head><body>']
    tables: list[list[list[str]]] = []
    position = 0
    while position < len(lines):
        line = lines[position]
        # A table, under its title if the line before it is one
        caption = None
        if (
            position in headings
            and line.startswith("Table ")
            and position + 1 < len(lines)
            and lines[position + 1].startswith("|")
        ):
            caption = line
            position += 1
        if lines[position].startswith("|"):
            rows = []
            while position < len(lines) and lines[position].startswith("|"):
                rows.append(table_cells(lines[position]))
                position += 1
            out += _table_html(rows, caption)
            tables.append(rows)
            continue
        if position in headings:
            level = _heading_level(case, position, line)
            out.append(f"<h{level}>{html.escape(line)}</h{level}>")
        else:
            out.append(f"<p>{html.escape(line)}</p>")
        position += 1
    out.append("</body></html>")
    return "\n".join(out) + "\n", tables


def _restore_table_characters(
    document: DoclingDocument, tables: list[list[list[str]]]
) -> int:
    """
    Sets back the text of each table cell that the HTML backend of Docling changed
    only by its replacements of characters, e.g. "—" with "-", to that of the case.

    :param document: The Docling document, changed in place.
    :type document: DoclingDocument
    :param tables: The values of the cells of each table of the case, row by row.
    :type tables: list[list[list[str]]]
    :return: The number of cells set back.
    :rtype: int
    :raises ValueError: If the document does not have the tables of the case.
    """
    if len(document.tables) != len(tables):
        raise ValueError(
            f"The Docling document has {len(document.tables)} tables, and the case "
            f"has {len(tables)}."
        )
    restored = 0
    for table, rows in zip(document.tables, tables, strict=True):
        for cell in table.data.table_cells:
            original = rows[cell.start_row_offset_idx][cell.start_col_offset_idx]
            # The replacements of the backend (a private method of Docling, which
            # the script uses as it is, to set back exactly what they changed)
            if cell.text != original and (
                HTMLDocumentBackend._clean_unicode(original) == cell.text
            ):
                cell.text = original
                restored += 1
    return restored


def main() -> int:
    """
    Writes the Docling document of each case, and checks it.

    :return: 0.
    :rtype: int
    """
    converter = DocumentConverter()
    for case in CASE_NAMES:
        source, tables = render_html(case)
        result = converter.convert(
            DocumentStream(name=f"{case}.html", stream=io.BytesIO(source.encode()))
        )
        document = result.document
        restored = _restore_table_characters(document, tables)
        html_path = CASES_DIR / f"{case}.docling.html"
        html_path.write_text(source, encoding="utf-8", newline="\n")
        json_path = CASES_DIR / f"{case}.docling.json"
        document.save_as_json(json_path)
        # Saved in text mode, i.e. with CRLF line ends on Windows
        text = json_path.read_bytes().replace(b"\r\n", b"\n")
        json_path.write_bytes(text.rstrip(b"\n") + b"\n")
        # Raises if a paragraph is not that of the reference answers
        load_case(case, case_input="docling")
        print(
            f"{case}: wrote {json_path.name} and {html_path.name} "
            f"({len(tables)} tables, {restored} cells set back)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
