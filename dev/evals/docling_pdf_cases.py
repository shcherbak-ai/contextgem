# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Prints the evaluation cases with tables as PDF files and converts them with full
Docling, for extraction from the Docling document of a real PDF file (``--input
pdf`` of ``run_eval`` and ``decision_probabilities``), with what the layout and
table models of Docling make of it: the headings without their levels, the tables
as the table model reads them, the paragraphs split at the ends of pages, and the
headers and footers of the pages left out.

For each case of ``PDF_CASES``, it prints the HTML rendering of the case
(``cases/<case>.docling.html``, written by ``docling_cases.py``) with headless
Chrome, as a report is printed: on A4 pages with the title of the case in their
header and their number in their footer, and the tables with borders. It writes
the PDF file as ``cases/<case>.pdf``, converts it with full Docling, the version in
``uv.lock``, without OCR, as the file has a text layer, in a throwaway environment
with the CPU-only PyTorch (as ``uv run fab docling-fixtures`` does), and writes
the Docling document as ``cases/<case>.pdf.docling.json``, both to be reviewed and
committed. It then prints how the paragraphs of the Docling document map to those
of the reference answers (see ``dev.evals.harness.map_paragraphs``).

Needs Chrome, Chromium or Edge (``--browser`` names it where it is not found).
Downloads the models of Docling on the first run, about 0.5 GB. Sends no request to
a model. Run it from the root of the repo:

```bash
uv run python -m dev.evals.docling_pdf_cases
```
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404 - runs a browser and Python with arguments of its own, no shell
import sys
import tempfile
from pathlib import Path


CASES_DIR = Path(__file__).resolve().parent / "cases"
REPO_DIR = CASES_DIR.parents[2]

# The cases that are printed as PDF files: those with tables
PDF_CASES = ("complex_mcu", "complex_stability")

# Where a browser is looked for, after the commands on the PATH
_BROWSER_COMMANDS = ("google-chrome", "chromium", "chromium-browser", "chrome")
_BROWSER_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)

# The style of the printed report, with the title of the case in the header of
# each page (filled in) and the number of the page in its footer
_PRINT_STYLE = """<style>
@page {
  size: A4;
  margin: 22mm 20mm 20mm 20mm;
  @top-center { content: "%s"; font: 8pt Arial, sans-serif; color: #555; }
  @bottom-center {
    content: "Page " counter(page) " of " counter(pages);
    font: 8pt Arial, sans-serif;
    color: #555;
  }
}
body { font: 10pt/1.35 "Times New Roman", Times, serif; }
h1 { font-size: 16pt; }
h2 { font-size: 13pt; margin-top: 14pt; }
h3 { font-size: 11pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 8.5pt; }
caption { caption-side: top; text-align: left; font-weight: bold; padding-bottom: 3pt; }
th, td { border: 0.5pt solid #444; padding: 2pt 4pt; vertical-align: top; }
th { background: #eee; }
</style>"""


def _find_browser(name: str | None) -> str:
    """
    Finds the browser that prints the PDF files.

    :param name: The browser given on the command line, or None.
    :type name: str | None
    :return: The path of the browser.
    :rtype: str
    """
    if name:
        return name
    for command in _BROWSER_COMMANDS:
        found = shutil.which(command)
        if found:
            return found
    for path in _BROWSER_PATHS:
        if Path(path).exists():
            return path
    sys.exit("No Chrome, Chromium or Edge was found: name one with --browser.")


def _print_pdf(case: str, browser: str, scratch: Path) -> Path:
    """
    Prints the HTML rendering of a case as a PDF file.

    :param case: The name of the case.
    :type case: str
    :param browser: The path of the browser.
    :type browser: str
    :param scratch: A folder for the HTML to print.
    :type scratch: Path
    :return: The PDF file, ``cases/<case>.pdf``.
    :rtype: Path
    """
    source = (CASES_DIR / f"{case}.docling.html").read_text(encoding="utf-8")
    title = (CASES_DIR / f"{case}.txt").read_text(encoding="utf-8").split("\n", 1)[0]
    # The title in a CSS string, which escapes its quotes and backslashes (the text
    # of a style element is not HTML-escaped)
    css_title = title.replace("\\", "\\\\").replace('"', '\\"')
    page = source.replace("<head>", "<head>" + _PRINT_STYLE % css_title, 1)
    page_path = scratch / f"{case}.print.html"
    page_path.write_text(page, encoding="utf-8", newline="\n")
    pdf_path = CASES_DIR / f"{case}.pdf"
    subprocess.run(  # nosec B603 - the browser with arguments of its own, no shell
        [
            browser,
            "--headless=new",
            "--disable-gpu",
            # Without the date, the URL and the title that a browser prints
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            page_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    print(f"printed {pdf_path.relative_to(REPO_DIR).as_posix()}", flush=True)
    return pdf_path


def _convert(pdf_paths: list[Path]) -> int:
    """
    Converts the PDF files with Docling, and writes their Docling documents. Runs
    in the throwaway environment.

    :param pdf_paths: The PDF files.
    :type pdf_paths: list[Path]
    :return: 0.
    :rtype: int
    """
    import importlib.metadata

    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    print(f"Docling {importlib.metadata.version('docling')}", flush=True)
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(do_ocr=False)
            )
        }
    )
    for pdf_path in pdf_paths:
        document = converter.convert(pdf_path).document
        json_path = pdf_path.with_name(
            f"{pdf_path.name.removesuffix('.pdf')}.pdf.docling.json"
        )
        document.save_as_json(json_path)
        # Saved in text mode, i.e. with CRLF line ends on Windows
        text = json_path.read_bytes().replace(b"\r\n", b"\n")
        json_path.write_bytes(text.rstrip(b"\n") + b"\n")
        print(f"wrote {json_path.name}", flush=True)
    return 0


def _report(case: str) -> None:
    """
    Prints how the paragraphs of the Docling document of the PDF file of a case map
    to those of its reference answers.

    :param case: The name of the case.
    :type case: str
    :return: None
    :rtype: None
    """
    from dev.evals.harness import load_case

    document, gold = load_case(case, case_input="pdf")
    paragraph_map = gold["paragraph_map"]
    n_gold = len(gold["paragraphs"])
    mapped = {position for positions in paragraph_map for position in positions}
    unmatched = [
        index for index, positions in enumerate(paragraph_map) if not positions
    ]
    merged = [positions for positions in paragraph_map if len(positions) > 1]
    split = (
        n_gold
        - len(mapped)
        + sum(
            1
            for position in mapped
            if sum(position in positions for positions in paragraph_map) > 1
        )
    )
    print(
        f"{case}: {len(document.paragraphs)} paragraphs for {n_gold} of the reference "
        f"answers; {len(mapped)} of these found, {len(merged)} paragraphs with "
        f"several, {split} split or not found, {len(unmatched)} paragraphs with "
        f"none",
        flush=True,
    )
    for position in sorted(set(range(n_gold)) - mapped):
        print(
            f"  not found: [{position}] {gold['paragraphs'][position]['raw_text'][:80]!r}"
        )
    for index in unmatched:
        print(
            f"  no paragraph of the reference answers: {document.paragraphs[index].raw_text[:80]!r}"
        )


def main() -> int:
    """
    Prints the PDF files of the cases, converts them and reports the mapping.

    :return: 0 when the conversions ran, else 1.
    :rtype: int
    """
    parser = argparse.ArgumentParser(
        description="Prints the evaluation cases with tables as PDF files, converts "
        "them with full Docling, and reports how their paragraphs map to those of "
        "the reference answers. Sends no request to a model."
    )
    parser.add_argument("--browser", default=None, help="Chrome, Chromium or Edge")
    parser.add_argument(
        "--report",
        action="store_true",
        help="only report the mapping of the committed Docling documents",
    )
    parser.add_argument("--inner", nargs="*", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.inner is not None:
        return _convert([Path(path) for path in args.inner])
    if not args.report:
        # The setup of the environment of full Docling, and the retries of the
        # known crash of its PDF parser on Windows, are those of the fixtures
        from dev.docling_fixtures import WINDOWS_ATTEMPTS, _setup

        browser = _find_browser(args.browser)
        with tempfile.TemporaryDirectory(prefix="contextgem-pdf-cases-") as scratch:
            pdf_paths = [_print_pdf(case, browser, Path(scratch)) for case in PDF_CASES]
            python = _setup("locked", Path(scratch))
            attempts = WINDOWS_ATTEMPTS if os.name == "nt" else 1
            for attempt in range(1, attempts + 1):
                result = subprocess.run(  # nosec B603 - the Python of the environment, no shell
                    [
                        str(python),
                        *("-u", "-X", "faulthandler"),
                        str(Path(__file__).resolve()),
                        "--inner",
                        *(str(path) for path in pdf_paths),
                    ],
                    env={**os.environ, "CONTEXTGEM_LOGGER_LEVEL": "ERROR"},
                    check=False,
                )
                if result.returncode == 0:
                    break
                print(f"The conversions stopped with exit code {result.returncode}.")
                if attempt == attempts:
                    return 1
    os.environ.setdefault("CONTEXTGEM_LOGGER_LEVEL", "ERROR")
    for case in PDF_CASES:
        _report(case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
