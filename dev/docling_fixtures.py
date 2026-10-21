# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Writes and checks the Docling documents of the test fixtures of the docling extra,
and the outputs of ContextGem from them.

``tests/other_files/docling/`` holds files in the formats that Docling converts,
each with the Docling document that full Docling converts it into
(``<file name>.json``), and with what ContextGem makes of that Docling document
(``<file name>.contextgem.jsonl``, see ``contextgem_output()``): files written for
ContextGem, and in ``docling_tests/``, test documents of Docling. The tests create
ContextGem documents from these Docling documents, and check that they give the
committed outputs, which show in review how a change of the converter or of the
Docling documents changes what ContextGem makes of these files. They also convert
the files in the formats that the torch-free docling-slim of the dev group
converts (``TEST_CONVERTED_SUFFIXES``), and check that the result is the committed
Docling document. A PDF file or a Google Books package (METS GBS) needs the layout
and table models of Docling, which run on PyTorch, and a scanned PDF file
(``OCR_FILES``) its OCR models too; an email, OpenDocument file or XBRL report
packages that the dev group does not have; and a legacy Office file (DOC, XLS,
PPT) or an RTF file LibreOffice, which Docling converts it with into a DOCX, XLSX
or PPTX file first: they are not converted in the tests. An EBCDIC file is
decoded with the layout of its records next to it (``layout_path()``), and an
XBRL report with the folder of its taxonomy (``taxonomy_path()``).

This script converts each file with full Docling, the version in ``uv.lock`` by
default, and the support of OpenDocument, Apple iWork and XBRL of docling-slim,
which full Docling does not install, in a throwaway environment with the CPU-only
build of PyTorch. It needs LibreOffice, for the legacy Office and RTF files: on
the ``PATH``, in its default folder on Windows or macOS, or named by
``DOCLING_LIBREOFFICE_CMD``, as Docling finds it:

- By default, it checks that ContextGem makes the same paragraphs and images from
  the new Docling documents as from the committed ones: the text, additional
  context and page numbers of each paragraph, and each image. The bounding boxes
  are not compared, as they differ slightly between releases of the models. It
  prints each difference, and exits with 1 if there is one.
- With ``--write``, it writes the Docling documents, and then the outputs of
  ContextGem from them, to be reviewed and committed.

With ``--outputs``, it only writes the outputs of ContextGem from the committed
Docling documents, in the environment of the project, without Docling: after
a change of the converter.

The models of Docling are downloaded from Hugging Face on the first run, about
0.5 GB, and cached there. Sends no LLM request. On Windows, a known crash of
docling-parse stops the conversions now and then: they run again, up to
``WINDOWS_ATTEMPTS`` times. The workflow
``.github/workflows/docling-fixtures.yml`` runs the check when ``uv.lock`` or the
converter changes, and weekly with the latest Docling.

Example:
    uv run fab docling-fixtures
    uv run fab docling-fixtures --write
    uv run fab docling-fixtures --outputs
    uv run fab docling-fixtures --docling-version latest
"""

from __future__ import annotations

import argparse
import contextlib
import difflib
import json
import os
import shutil
import subprocess  # nosec B404 - runs uv and Python with arguments of its own, no shell
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Generator

    from docling_core.types.doc import DoclingDocument


ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = ROOT / "tests" / "other_files" / "docling"
# The formats of the fixture files that the tests convert with the docling-slim of
# the dev group, which needs neither models nor the packages of other formats
TEST_CONVERTED_SUFFIXES = {
    *(".docx", ".xlsx", ".pptx", ".html", ".md", ".csv", ".asciidoc", ".tex"),
    # WebVTT subtitles, EPUB, JATS articles and USPTO patents
    *(".vtt", ".epub", ".nxml", ".xml"),
    # Web archives, Box Notes, DocLang (XML and archive), Apple iWork (Keynote,
    # Numbers, Pages), mainframe records (EBCDIC) and print streams (AFP)
    *(".mhtml", ".boxnote", ".dclg", ".dclx", ".key", ".numbers", ".pages"),
    *(".ebc", ".afp"),
}
# The formats of all the fixture files: also PDF, email (EML and Outlook),
# OpenDocument (text, spreadsheet and presentation), legacy Office (Word, Excel
# and PowerPoint 97-2003) and RTF, XBRL reports, and Google Books packages (METS
# GBS, .tar.gz), images, read with OCR, and audio and video, transcribed with
# Whisper. The documentation lists them as the supported formats
# (docs/source/documents/docling.rst), which a test checks
SOURCE_SUFFIXES = {
    *TEST_CONVERTED_SUFFIXES,
    *(".pdf", ".eml", ".msg", ".odt", ".ods", ".odp", ".doc", ".xls", ".ppt"),
    *(".rtf", ".xbrl", ".gz"),
    # Images, audio and video
    *(".png", ".tiff", ".mp3", ".mp4"),
}
# The scanned PDF files, which Docling reads with OCR: the other PDF files have
# a text layer, and are converted without OCR, whose models they do not need
OCR_FILES = {"service_agreement_scan.pdf", "docling_tests/ocr_test.pdf"}
# Where LibreOffice is installed by default on Windows, where Docling finds it only
# on the PATH, which its installer does not change
WINDOWS_LIBREOFFICE = Path(r"C:\Program Files\LibreOffice\program\soffice.exe")
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


def _libreoffice() -> str | None:
    """
    Find LibreOffice as Docling finds it, and also in its default folder on
    Windows, where Docling looks only on the PATH.

    :returns: The command of LibreOffice, or None if it is not found.
    :rtype: str | None
    """
    mac_app = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")
    return (
        os.environ.get("DOCLING_LIBREOFFICE_CMD")
        or shutil.which("libreoffice")
        or shutil.which("soffice")
        or next(
            (str(path) for path in (mac_app, WINDOWS_LIBREOFFICE) if path.is_file()),
            None,
        )
    )


def layout_path(source: Path) -> Path:
    """
    Return the path of the layout of the records of an EBCDIC fixture file, which
    Docling decodes the file with, next to the file.

    :param source: The fixture file, e.g. ``gas_disposition.ebc``.
    :type source: Path
    :returns: The path of its layout, e.g. ``gas_disposition.ebc.layout.json``.
    :rtype: Path
    """
    return source.with_name(f"{source.name}.layout.json")


def taxonomy_path(source: Path) -> Path:
    """
    Return the path of the folder of the taxonomy of an XBRL fixture file, which
    Docling reads the report with, next to the file.

    :param source: The fixture file, e.g. ``quarterly_fees.xbrl``.
    :type source: Path
    :returns: The path of the folder, e.g. ``quarterly_fees.xbrl.taxonomy``.
    :rtype: Path
    """
    return source.with_name(f"{source.name}.taxonomy")


def converter_key(source: Path) -> tuple[bool, Path | None, Path | None]:
    """
    Return what the converter of a fixture file depends on, so that the files that
    need the same options share a converter, and its models.

    :param source: The fixture file.
    :type source: Path
    :returns: Whether it is read with OCR, and the layout or the taxonomy that it
        needs, if any.
    :rtype: tuple[bool, Path | None, Path | None]
    """
    return (
        source.relative_to(FIXTURES_DIR).as_posix() in OCR_FILES,
        layout_path(source) if source.suffix == ".ebc" else None,
        taxonomy_path(source) if source.suffix == ".xbrl" else None,
    )


def format_options(source: Path) -> dict:
    """
    Return the options of the converter of Docling for a fixture file: OCR for
    a scanned PDF file, the layout of the records of an EBCDIC file, the taxonomy
    of an XBRL report, read from its folder, for an MHTML file the images that it
    holds, for an image file its image, and the small model of Whisper for the
    transcripts of audio and video files.

    :param source: The fixture file.
    :type source: Path
    :returns: The format options, by input format.
    :rtype: dict
    """
    from docling.datamodel import asr_model_specs
    from docling.datamodel.backend_options import (
        EbcdicBackendOptions,
        HTMLBackendOptions,
        XBRLBackendOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AsrPipelineOptions,
        PdfPipelineOptions,
        VideoPipelineOptions,
    )
    from docling.document_converter import (
        AudioFormatOption,
        EbcdicFormatOption,
        HTMLFormatOption,
        ImageFormatOption,
        PdfFormatOption,
        VideoFormatOption,
        XBRLFormatOption,
    )

    ocr, layout, taxonomy = converter_key(source)
    options: dict = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(do_ocr=ocr)
        ),
        # The images of a web archive are in the file: no fetching beyond it
        InputFormat.MHTML: HTMLFormatOption(
            backend_options=HTMLBackendOptions(fetch_images=True)
        ),
        # An image file is read with OCR, and its image is kept, which ContextGem
        # attaches to the document
        InputFormat.IMAGE: ImageFormatOption(
            pipeline_options=PdfPipelineOptions(generate_page_images=True)
        ),
        # The small model of Whisper, as the tiny one, the default of Docling,
        # misheard the synthesized speech of the fixtures, which a change of
        # platform can then turn either way
        InputFormat.AUDIO: AudioFormatOption(
            pipeline_options=AsrPipelineOptions(
                asr_options=asr_model_specs.WHISPER_SMALL
            )
        ),
        InputFormat.VIDEO: VideoFormatOption(
            pipeline_options=VideoPipelineOptions(
                asr_options=asr_model_specs.WHISPER_SMALL
            )
        ),
    }
    if layout is not None:
        options[InputFormat.EBCDIC] = EbcdicFormatOption(
            backend_options=EbcdicBackendOptions(layout_file=layout)
        )
    if taxonomy is not None:
        # Docling reads a taxonomy from files only when it is allowed to
        options[InputFormat.XML_XBRL] = XBRLFormatOption(
            backend_options=XBRLBackendOptions(
                taxonomy=taxonomy, enable_local_fetch=True
            )
        )
    return options


@contextlib.contextmanager
def conversion_input(source: Path) -> Generator[Path]:
    """
    Give the path to convert a fixture file from: the file itself, but for
    a DocLang archive (DCLX) a copy of it in a temporary folder, as docling-core
    extracts an archive into a folder next to it (``<name>_artifacts``), which the
    folder of the fixtures must not get. The pictures of the archive are read into
    the Docling document, so it needs the copy no more once it is converted.

    :param source: The fixture file.
    :type source: Path
    :returns: The path to convert it from, while the context lasts.
    :rtype: Generator[Path]
    """
    if source.suffix != ".dclx":
        yield source
        return
    with tempfile.TemporaryDirectory(
        prefix="contextgem-dclx-", ignore_cleanup_errors=True
    ) as scratch:
        yield Path(shutil.copy2(source, scratch))


def source_files() -> list[Path]:
    """
    List the fixture files that Docling converts, also those of the test documents
    of Docling in ``docling_tests/``, but not the files of the taxonomy of an XBRL
    report (``taxonomy_path()``), which Docling reads with the report.

    :returns: The files, by path.
    :rtype: list[Path]
    """
    return sorted(
        path
        for path in FIXTURES_DIR.rglob("*")
        if path.is_file()
        and path.suffix in SOURCE_SUFFIXES
        and not any(folder.suffix == ".taxonomy" for folder in path.parents)
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
        version = f"=={_locked_docling_version()}"
    elif docling_version == "latest":
        version = ""
    else:
        version = f"=={docling_version}"
    # With speech recognition (Whisper), for the audio and video files
    requirement = f"docling[asr]{version}"
    # The support of OpenDocument, Apple iWork and XBRL, which full Docling does not
    # install, of the docling-slim release that Docling is made of
    other_formats = (
        f"docling-slim[format-opendocument,format-iwork,format-xml-xbrl]{version}"
    )
    venv = scratch / "venv"
    python_version = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    subprocess.run(  # nosec B603 - uv with arguments of its own, no shell
        [uv, "venv", "--quiet", "--python", python_version, str(venv)], check=True
    )
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    print(
        f"Installing ContextGem, {requirement} and {other_formats}, with the CPU-only "
        f"PyTorch...",
        flush=True,
    )
    subprocess.run(  # nosec B603 - uv with arguments of its own, no shell
        [
            *(uv, "pip", "install", "--quiet", "--python", str(python)),
            # The CPU-only build: on Linux, the default one comes with CUDA
            *("--torch-backend", "cpu"),
            *("--editable", str(ROOT), requirement, other_formats),
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


def contextgem_output(docling_document: DoclingDocument) -> list[str]:
    """
    Describe what ContextGem makes of a Docling document, as the output files of
    the fixtures hold it.

    :param docling_document: The Docling document.
    :type docling_document: DoclingDocument
    :returns: A line for each paragraph, a JSON object with its text, its additional
        context and its page numbers, and then for each image, with its MIME type and
        the Docling reference of its picture.
    :rtype: list[str]
    """
    from contextgem import Document

    document = Document.from_docling(docling_document)
    records: list[dict[str, object]] = [
        {
            "text": paragraph.raw_text,
            "context": paragraph.additional_context,
            "pages": paragraph.page_numbers,
        }
        for paragraph in document.paragraphs
    ]
    records += [
        {"image": image.mime_type, "ref": image.custom_data["docling"]["self_ref"]}
        for image in document.images
    ]
    return [json.dumps(record, ensure_ascii=False) for record in records]


def output_path(source: Path) -> Path:
    """
    Return the path of the output of ContextGem from the Docling document of
    a fixture file, next to the file.

    :param source: The fixture file, e.g. ``fees.xlsx``.
    :type source: Path
    :returns: The path of the output, e.g. ``fees.xlsx.contextgem.jsonl``.
    :rtype: Path
    """
    return source.with_name(f"{source.name}.contextgem.jsonl")


def read_output(path: Path) -> list[str]:
    """
    Read the output of ContextGem from the Docling document of a fixture file.

    :param path: The output file.
    :type path: Path
    :returns: Its lines, split at line feeds only: a text can hold the line
        separators of Unicode, which JSON leaves as they are.
    :rtype: list[str]
    """
    text = path.read_text(encoding="utf-8")
    return text.removesuffix("\n").split("\n") if text else []


def _write_outputs() -> int:
    """
    Write the output of ContextGem from the committed Docling document of each
    fixture file. Runs in the environment of the project, i.e. with its
    docling-core, with which the tests read the outputs.

    :returns: 0, or 1 if a fixture file has no committed Docling document.
    :rtype: int
    """
    from docling_core.types.doc import DoclingDocument

    missing = 0
    for source in source_files():
        committed_path = source.with_name(f"{source.name}.json")
        if not committed_path.exists():
            print(
                f"! {source.relative_to(FIXTURES_DIR).as_posix()}: no committed Docling document"
            )
            missing += 1
            continue
        lines = contextgem_output(DoclingDocument.load_from_json(committed_path))
        path = output_path(source)
        path.write_text(
            "".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n"
        )
        print(f"wrote {path.relative_to(ROOT).as_posix()}")
    return 1 if missing else 0


def _convert(write: bool) -> int:
    """
    Convert the fixture files with Docling, and write or check their Docling
    documents. Runs in the throwaway environment.

    :param write: Whether to write the Docling documents, instead of checking
        the committed ones.
    :type write: bool
    :returns: 0 when every file is converted and nothing differs, else 1.
    :rtype: int
    """
    import importlib.metadata

    from docling.datamodel.base_models import ConversionStatus
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import DoclingDocument

    print(f"Docling {importlib.metadata.version('docling')}")
    converters: dict[tuple[bool, Path | None, Path | None], DocumentConverter] = {}
    differences = 0
    failures = 0
    for source in source_files():
        key = converter_key(source)
        if key not in converters:
            converters[key] = DocumentConverter(format_options=format_options(source))
        with conversion_input(source) as path:
            result = converters[key].convert(path)
        # A file that Docling converts in part, e.g. without a page that it failed
        # on, is neither written nor compared as if it were whole
        if result.status != ConversionStatus.SUCCESS:
            errors = "; ".join(error.error_message for error in result.errors)
            print(
                f"! {source.relative_to(FIXTURES_DIR).as_posix()}: "
                f"{result.status.value}: {errors}"
            )
            failures += 1
            continue
        converted = result.document
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
        committed = contextgem_output(DoclingDocument.load_from_json(committed_path))
        current = contextgem_output(converted)
        if current == committed:
            print(f"ok {source.relative_to(FIXTURES_DIR).as_posix()}")
            continue
        diff = list(
            difflib.unified_diff(
                committed,
                current,
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
            f"Docling documents and the outputs of ContextGem with "
            f"`uv run fab docling-fixtures --write` and commit them; otherwise, adapt "
            f"the converter."
        )
    if failures:
        print(f"{failures} file(s) not converted in full.")
    return 1 if differences or failures else 0


def main() -> int:
    """
    Write or check the Docling documents of the fixtures with full Docling.

    :returns: 0 when nothing differs, else 1.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Convert the Docling fixtures of the tests with full Docling, "
        "and check the committed Docling documents, or write them with the outputs "
        "of ContextGem from them. Sends no LLM request."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write",
        action="store_true",
        help="write the Docling documents, and the outputs of ContextGem from them",
    )
    mode.add_argument(
        "--outputs",
        action="store_true",
        help="only write the outputs of ContextGem from the committed Docling "
        "documents, without Docling",
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
    # The outputs are written here, in the environment of the project
    os.environ.setdefault("CONTEXTGEM_LOGGER_LEVEL", "ERROR")
    if args.outputs:
        return _write_outputs()
    # Before the environment is created, which takes minutes
    libreoffice = _libreoffice()
    if libreoffice is None:
        sys.exit(
            "LibreOffice was not found, with which Docling converts the legacy Office "
            "and RTF fixtures (DOC, XLS, PPT, RTF). Install it "
            "(https://www.libreoffice.org), and "
            "put it on the PATH or name its soffice in DOCLING_LIBREOFFICE_CMD."
        )
    if shutil.which("ffmpeg") is None:
        sys.exit(
            "ffmpeg was not found, which Whisper reads the audio and video fixtures "
            "with, and Docling the frames of a video. Install it "
            "(https://ffmpeg.org/download.html) and put it on the PATH."
        )
    attempts = WINDOWS_ATTEMPTS if os.name == "nt" else 1
    # Outside the checkout, so that no configuration of the project applies. A file
    # of the environment can still be open on Windows when the folder is removed.
    with tempfile.TemporaryDirectory(
        prefix="contextgem-docling-", ignore_cleanup_errors=True
    ) as scratch:
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
                env={
                    **os.environ,
                    "CONTEXTGEM_LOGGER_LEVEL": "ERROR",
                    "DOCLING_LIBREOFFICE_CMD": libreoffice,
                },
                check=False,
            )
            if result.returncode == 0 and args.write:
                return _write_outputs()
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
