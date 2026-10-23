# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Extracts an evaluation case with a model, one or more times, and scores each
result against the reference answers of the case.

The requests are live and are paid for. The API keys are read from the environment
or from the `.env` file of the repo, and are never printed.

Run it from the root of the repo:

```bash
uv run python -m dev.evals.run_eval --case complex_nda --model openai/gpt-6-luna \
    --runs 3 --label my_experiment
```

Each run is written to ``results/<label>/runs/`` as a JSON file, and one line is
printed for it, which starts with the name of the file. A run whose file exists is
skipped. The library prints its logs as usual, at the level set with
``CONTEXTGEM_LOGGER_LEVEL``.

``--input docling`` creates the document of the case from its Docling document,
with ``Document.from_docling()``, instead of from its text (see
``dev/evals/docling_cases.py``). Record such runs under a label of their own.
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, get_args

from dev.evals.harness import (
    CASE_NAMES,
    RESULTS_DIR,
    CaseInput,
    ModelKind,
    create_model,
    get_model_slug,
    get_provenance,
    get_usage_record,
    load_case,
    scrub,
    write_json,
)
from dev.evals.scoring import score_document


# Name of the logger of the library
_LIBRARY_LOGGER_NAME = "contextgem"


class _LogCapture(logging.Handler):
    """
    Keeps the messages that the library logs during an extraction.

    :ivar messages: The messages logged.
    :vartype messages: list[str]
    """

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []
        # The messages logged at the WARNING level: the library logs a call that
        # fails without an exception, and the answers that a decision model
        # refused to give for an aspect, at that level only
        self.warnings: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        Keeps the message of a log record.

        :param record: The log record.
        :type record: logging.LogRecord
        :return: None
        :rtype: None
        """
        message = record.getMessage()
        self.messages.append(message)
        if record.levelno == logging.WARNING:
            self.warnings.append(message)


@contextmanager
def _capture_library_logs() -> Generator[_LogCapture]:
    """
    Captures the messages that the library logs at the INFO level and above,
    which is the level that it logs the retries for invalid data at, whatever
    the level of its logger is. What the library prints is left as it is.

    :return: Context manager that gives the handler with the messages captured.
    :rtype: Generator[_LogCapture]
    """
    library_logger = logging.getLogger(_LIBRARY_LOGGER_NAME)
    logger_level = library_logger.level
    handler_levels = [(handler, handler.level) for handler in library_logger.handlers]
    if not library_logger.isEnabledFor(logging.INFO):
        # The handlers of the library keep printing at the level of the logger
        for handler, handler_level in handler_levels:
            handler.setLevel(max(handler_level, library_logger.getEffectiveLevel()))
        library_logger.setLevel(logging.INFO)
    log_capture = _LogCapture()
    library_logger.addHandler(log_capture)
    try:
        yield log_capture
    finally:
        library_logger.removeHandler(log_capture)
        library_logger.setLevel(logger_level)
        for handler, handler_level in handler_levels:
            handler.setLevel(handler_level)


def get_mode(kind: ModelKind, use_concurrency: bool) -> str:
    """
    Returns the name of the mode of a run, which is a part of the name of its file.

    :param kind: The kind of the model: "llm" or "decision".
    :type kind: Literal["llm", "decision"]
    :param use_concurrency: Whether the extraction is run with concurrency.
    :type use_concurrency: bool
    :return: "default" or "concurrent", with the "decision_" prefix for a decision
        model.
    :rtype: str
    """
    mode = "concurrent" if use_concurrency else "default"
    return f"decision_{mode}" if kind == "decision" else mode


def run_once(
    *,
    case: str,
    model: str,
    kind: ModelKind,
    api_base: str | None,
    use_concurrency: bool,
    label: str,
    run: int,
    case_input: CaseInput = "text",
) -> dict[str, Any]:
    """
    Extracts everything of a case with a model, once, and scores the result.

    An error of the extraction does not stop the run: the error is recorded, and
    what was extracted before it is scored.

    :param case: The name of the case.
    :type case: str
    :param case_input: What the document of the case is created from (see
        :func:`~dev.evals.harness.load_case`). Defaults to "text".
    :type case_input: CaseInput
    :param model: The model identifier.
    :type model: str
    :param kind: The kind of the model: "llm" or "decision".
    :type kind: Literal["llm", "decision"]
    :param api_base: The address of a local server, or None for a hosted model.
    :type api_base: str | None
    :param use_concurrency: Whether to run the extraction with concurrency.
    :type use_concurrency: bool
    :param label: The label of the experiment that the run belongs to.
    :type label: str
    :param run: The number of the run.
    :type run: int
    :return: The record of the run: what was run, the error if any, the time,
        the usage, the retries for invalid data, the warnings, and the scores.
    :rtype: dict[str, Any]
    """
    document, gold = load_case(
        case, for_decision_model=kind == "decision", case_input=case_input
    )
    extractor, private_texts = create_model(model, kind, api_base)

    error = None
    started = time.time()
    # The retries for invalid data, the failed calls and the refused answers are
    # told by the logs of the library. Its Python warnings, on the configuration,
    # are not part of the record.
    with (
        _capture_library_logs() as log_capture,
        warnings.catch_warnings(record=True),
    ):
        warnings.simplefilter("always")
        try:
            extractor.extract_all(
                document,
                use_concurrency=use_concurrency,
                raise_exception_on_extraction_error=False,
            )
        except Exception as e:  # noqa: BLE001 - any error is recorded for the run
            error = scrub(f"{type(e).__name__}: {str(e)[:500]}", private_texts)
    seconds = time.time() - started

    warning_messages = log_capture.warnings
    return {
        "case": case,
        "model": model,
        "kind": kind,
        "mode": get_mode(kind, use_concurrency),
        "input": case_input,
        "label": label,
        "run": run,
        "error": error,
        "seconds": round(seconds, 1),
        "usage": get_usage_record(extractor),
        "retries_for_invalid_data": sum(
            message.startswith("Retrying ") for message in log_capture.messages
        ),
        "responses_not_valid_json": sum(
            "did not return valid JSON" in message for message in log_capture.messages
        ),
        # A failed call that is not raised as an exception: for invalid data,
        # a refusal or a response cut off, with the hint on raising, or for
        # a failed request
        "extraction_warnings": [
            scrub(message[:300], private_texts)
            for message in warning_messages
            if "If you want to raise an exception instead" in message
            or "Exception occurred" in message
        ],
        # One for each aspect that a decision model refused to answer about on
        # some of the paragraphs or sentences
        "refusal_warnings": [
            message[:600]
            for message in warning_messages
            if "not included in the aspect" in message
        ],
        "provenance": get_provenance(),
        "scores": score_document(document, gold),
    }


def main() -> None:
    """
    Runs the extraction of a case with a model the given number of times, and
    writes and prints the result of each run.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Extracts an evaluation case with a model and scores the result."
    )
    parser.add_argument("--case", required=True, choices=CASE_NAMES)
    parser.add_argument("--model", required=True, help="e.g. openai/gpt-6-luna")
    parser.add_argument("--kind", default="llm", choices=get_args(ModelKind))
    parser.add_argument(
        "--api-base", default=None, help="address of a local server, for its models"
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--first-run", type=int, default=1)
    parser.add_argument(
        "--label", required=True, help="name of the experiment, e.g. my_experiment"
    )
    parser.add_argument(
        "--concurrency", action="store_true", help="extract with use_concurrency=True"
    )
    parser.add_argument(
        "--input",
        default="text",
        choices=get_args(CaseInput),
        help="what the document is created from: the text of the case (default), "
        "its Docling document, with or without the additional context, or the "
        "Docling document of the case printed as a PDF file",
    )
    args = parser.parse_args()

    runs_dir = RESULTS_DIR / args.label / "runs"
    file_name_start = (
        f"{args.case}__{get_model_slug(args.model)}__"
        f"{get_mode(args.kind, args.concurrency)}"
    )
    for run in range(args.first_run, args.first_run + args.runs):
        path = runs_dir / f"{file_name_start}__run{run}.json"
        if path.exists():
            print(f"exists, skipped: {path.name}", flush=True)
            continue
        record = run_once(
            case=args.case,
            model=args.model,
            kind=args.kind,
            api_base=args.api_base,
            use_concurrency=args.concurrency,
            label=args.label,
            run=run,
            case_input=args.input,
        )
        write_json(path, record)
        summary = record["scores"]["summary"]
        usage = record["usage"]
        # None if the reference answers of the concepts extracted have no references
        reference_f1 = summary["reference_f1"]
        print(
            f"{path.name}: {record['seconds']} s, {usage['requests']} requests, "
            f"{usage['input_tokens']} input / {usage['output_tokens']} output / "
            f"{usage['cache_read_tokens']} cached tokens, "
            f"aspect F1 {summary['aspect_f1']:.3f}, "
            f"concepts {summary['n_concepts_correct']}/{summary['n_concepts_scored']}, "
            f"reference F1 "
            f"{'n/a' if reference_f1 is None else format(reference_f1, '.3f')}, "
            f"retries {record['retries_for_invalid_data']}, "
            f"aspects with refused answers {len(record['refusal_warnings'])}, "
            f"error {record['error']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
