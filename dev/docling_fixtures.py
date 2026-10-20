# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Writes and checks the Docling documents of the test fixtures of the docling extra.

``tests/other_files/docling/`` holds files in the formats that Docling converts,
each with the Docling document that full Docling converts it into
(``<file name>.json``): files written for ContextGem, and in ``docling_tests/``,
test documents of Docling. The tests create ContextGem documents from these Docling
documents. They also convert the DOCX, XLSX, PPTX, HTML and Markdown files with
the torch-free docling-slim of the dev group, and check that the result is the
committed Docling document. A PDF file needs the layout and table models of
Docling, which run on PyTorch, and is not converted in the tests.

This script converts each file with full Docling, the version in ``uv.lock`` by
default, in a throwaway environment with the CPU-only build of PyTorch:

- By default, it checks that ContextGem makes the same paragraphs and images from
  the new Docling documents as from the committed ones: their text, additional
  context and page numbers. The bounding boxes are not compared, as they differ
  slightly between releases of the models. It prints each difference, and exits
  with 1 if there is one.
- With ``--write``, it writes the Docling documents, to be reviewed and committed.

The models of Docling are downloaded from Hugging Face on the first run, about
0.5 GB, and cached there. Sends no LLM request. On Windows, a known crash of
docling-parse stops the conversions now and then: they run again, up to
``WINDOWS_ATTEMPTS`` times. The workflow
``.github/workflows/docling-fixtures.yml`` runs the check when ``uv.lock`` or the
converter changes, and weekly with the latest Docling.

Example:
    uv run fab docling-fixtures
    uv run fab docling-fixtures --write
    uv run fab docling-fixtures --docling-version latest
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess  # nosec B404 - runs uv and Python with arguments of its own, no shell
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "other_files" / "docling"
# The formats of the fixture files
SOURCE_SUFFIXES = {".pdf", ".docx", ".xlsx", ".pptx", ".html", ".md"}
# How many times the conversions run on Windows, where the page renderer of
# docling-parse crashes them now and then, as its threads corrupt the heap when
# they exit (https://github.com/docling-project/docling-parse/issues/357). With
# docling-parse 7.22.3, 14 of 27 runs crashed, also with one parser thread, each
# on the first PDF it converted, some 8 seconds in; the others converted every
# file. Linux and macOS are not affected: there, a crash fails the check at once.
WINDOWS_ATTEMPTS = 10


def _locked_docling_version() -> str:
    """
    Read the version of Docling in ``uv.lock``.

    :returns: The version, e.g. ``2.140.0``.
    :rtype: str
    """
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    for package in lock["package"]:
        if package["name"] == "docling":
            return package["version"]
    sys.exit("Docling is not in uv.lock: run `uv lock`.")


def _source_files() -> list[Path]:
    """
    List the fixture files that Docling converts, also those of the test documents
    of Docling in ``docling_tests/``.

    :returns: The files, by path.
    :rtype: list[Path]
    """
    return sorted(
        path for path in FIXTURES_DIR.rglob("*") if path.suffix in SOURCE_SUFFIXES
    )


def _setup(docling_version: str, scratch: Path) -> Path:
    """
    Create a throwaway environment with ContextGem and full Docling.

    :param docling_version: "locked", "latest" or a version of Docling.
    :type docling_version: str
    :param scratch: A folder to create the environment in, outside the checkout.
    :type scratch: Path
    :returns: The Python of the environment.
    :rtype: Path
    """
    uv = shutil.which("uv")
    if uv is None:
        sys.exit("uv was not found.")
    if docling_version == "locked":
        requirement = f"docling=={_locked_docling_version()}"
    elif docling_version == "latest":
        requirement = "docling"
    else:
        requirement = f"docling=={docling_version}"
    venv = scratch / "venv"
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    subprocess.run(  # nosec B603 - uv with arguments of its own, no shell
        [uv, "venv", "--quiet", "--python", python_version, str(venv)], check=True
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print(
        f"Installing ContextGem and {requirement}, with the CPU-only PyTorch...",
        flush=True,
    )
    subprocess.run(  # nosec B603 - uv with arguments of its own, no shell
        [
            *(uv, "pip", "install", "--quiet", "--python", str(python)),
            # The CPU-only build: on Linux, the default one comes with CUDA
            *("--torch-backend", "cpu"),
            *("--editable", str(ROOT), requirement),
        ],
        check=True,
    )
    return python


def _peak_memory() -> int | None:
    """
    Measure the peak memory of this process, i.e. of the conversions.

    :returns: The peak resident memory in bytes, or None on Windows, where it is
        not measured.
    :rtype: int | None
    """
    if os.name == "nt":
        return None
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # In bytes on macOS, in kilobytes on Linux
    return peak if sys.platform == "darwin" else peak * 1024


def _paragraphs(docling_document: DoclingDocument) -> list[tuple[str, str, str]]:
    """
    Describe what ContextGem makes from a Docling document.

    :param docling_document: The Docling document.
    :type docling_document: DoclingDocument
    :returns: The text, the additional context and the page numbers of each
        paragraph, and a line for each image.
    :rtype: list[tuple[str, str, str]]
    """
    from contextgem import Document

    document = Document.from_docling(docling_document)
    parts = [
        (p.raw_text, p.additional_context or "", f"pages {p.page_numbers}")
        for p in document.paragraphs
    ]
    parts += [(f"image {image.mime_type}", "", "") for image in document.images]
    return parts


def _summary(parts: list[tuple[str, str, str]]) -> list[str]:
    """
    Write the paragraphs of a document as lines, for a diff.

    :param parts: The text, the additional context and the page numbers of each
        paragraph, and a line for each image.
    :type parts: list[tuple[str, str, str]]
    :returns: A line for each paragraph and image.
    :rtype: list[str]
    """
    return [" | ".join(part for part in row if part) for row in parts]


def _convert(write: bool) -> int:
    """
    Convert the fixture files with Docling, and write or check their Docling
    documents. Runs in the throwaway environment.

    :param write: Whether to write the Docling documents, instead of checking
        the committed ones.
    :type write: bool
    :returns: 0 when nothing differs, else 1.
    :rtype: int
    """
    import importlib.metadata

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import DoclingDocument

    print(f"Docling {importlib.metadata.version('docling')}")
    # The PDF fixtures have a text layer: no OCR, whose models they do not need
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(do_ocr=False)
            )
        }
    )
    differences = 0
    for source in _source_files():
        converted = converter.convert(source).document
        committed_path = source.with_name(f"{source.name}.json")
        if write:
            converted.save_as_json(committed_path)
            # Saved in text mode, i.e. with CRLF line ends on Windows
            text = committed_path.read_bytes().replace(b"\r\n", b"\n")
            committed_path.write_bytes(text.rstrip(b"\n") + b"\n")
            print(f"wrote {committed_path.relative_to(ROOT)}")
            continue
        if not committed_path.exists():
            print(
                f"! {source.relative_to(FIXTURES_DIR).as_posix()}: no committed Docling document"
            )
            differences += 1
            continue
        committed = _paragraphs(DoclingDocument.load_from_json(committed_path))
        current = _paragraphs(converted)
        if current == committed:
            print(f"ok {source.relative_to(FIXTURES_DIR).as_posix()}")
            continue
        diff = list(
            difflib.unified_diff(
                _summary(committed),
                _summary(current),
                fromfile=f"{source.relative_to(FIXTURES_DIR).as_posix()} (committed)",
                tofile=f"{source.relative_to(FIXTURES_DIR).as_posix()} (converted now)",
                lineterm="",
            )
        )
        differences += 1
        print("\n".join(diff))
    peak = _peak_memory()
    if peak is not None:
        # E.g. against the memory of a CI runner, 16 GB on GitHub for a public
        # repository
        print(f"Peak memory of the conversions: {peak / 1024**3:.1f} GB")
    if differences:
        print(
            f"{differences} file(s) differ. If the new output is right, write the "
            f"Docling documents with `uv run fab docling-fixtures --write` and "
            f"commit them; otherwise, adapt the converter."
        )
    return 1 if differences else 0


def main() -> int:
    """
    Write or check the Docling documents of the fixtures with full Docling.

    :returns: 0 when nothing differs, else 1.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Convert the Docling fixtures of the tests with full Docling, "
        "and check the committed Docling documents, or write them. Sends no LLM "
        "request."
    )
    parser.add_argument(
        "--write", action="store_true", help="write the Docling documents"
    )
    parser.add_argument(
        "--docling-version",
        default="locked",
        help="the version of Docling: 'locked' (uv.lock, default), 'latest' or X.Y.Z",
    )
    parser.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inner:
        return _convert(args.write)
    attempts = WINDOWS_ATTEMPTS if os.name == "nt" else 1
    # Outside the checkout, so that no configuration of the project applies
    with tempfile.TemporaryDirectory(prefix="contextgem-docling-") as scratch:
        python = _setup(args.docling_version, Path(scratch))
        for attempt in range(1, attempts + 1):
            result = subprocess.run(  # nosec B603 - the Python of the environment, no shell
                [
                    str(python),
                    # Unbuffered, so that what it printed is shown also if it
                    # crashes, and with the traceback of a crash in native code
                    "-u",
                    *("-X", "faulthandler"),
                    str(Path(__file__).resolve()),
                    "--inner",
                    *(["--write"] if args.write else []),
                ],
                env={**os.environ, "CONTEXTGEM_LOGGER_LEVEL": "ERROR"},
                check=False,
            )
            if result.returncode in (0, 1):
                return result.returncode
            print(
                f"The conversions stopped with exit code {result.returncode}.",
                flush=True,
            )
            if attempt < attempts:
                print(
                    "On Windows, this is likely a known crash of docling-parse "
                    "(https://github.com/docling-project/docling-parse/issues/357): "
                    f"running them again ({attempt + 1} of {attempts}).",
                    flush=True,
                )
    return 1


if __name__ == "__main__":
    sys.exit(main())
