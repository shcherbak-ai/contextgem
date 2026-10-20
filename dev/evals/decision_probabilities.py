# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Records the probability that a decision model gives for each paragraph of a case
belonging to each aspect, in one pass of the extraction of the aspects.

The probabilities do not depend on the decision threshold, so one pass shows what
each threshold would extract (see ``threshold_sweep``). They are read from
the answers that the library records in the usage of the model. They are written
for the paragraphs of the reference answers: for a document whose paragraphs are
not those, e.g. created from the Docling document of a PDF file, a paragraph of
the reference answers has the highest probability of the paragraphs of
the document that hold it (see :func:`project_probabilities`), and the
probabilities of the paragraphs of the document are kept too.

The requests are live and are paid for. The API keys are read from the environment
or from the `.env` file of the repo, and are never printed.

Run it from the root of the repo:

```bash
uv run python -m dev.evals.decision_probabilities --case complex_nda \
    --model typesafe/jev-latest --label my_experiment
```

Writes ``results/<label>/decision_probabilities/<case>__<model>.json``, with
the usage and the cost of the pass. A pass whose file exists is skipped.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from typing import get_args

from contextgem import Document, DocumentDecisionModel
from dev.evals.harness import (
    CASE_NAMES,
    RESULTS_DIR,
    CaseInput,
    create_model,
    get_model_slug,
    get_provenance,
    get_usage_record,
    load_case,
    write_json,
)
from dev.evals.scoring import score_document


def record_probabilities(
    document: Document, model: DocumentDecisionModel
) -> list[list[float | None]]:
    """
    Extracts the aspects of a document with a decision model, and returns
    the probability of each paragraph belonging to each aspect.

    One request is sent for each paragraph, with one question for each aspect.
    The requests are sent one after another, i.e. they are recorded in the usage
    of the model in the order of the paragraphs.

    :param document: The document, with its aspects assigned and not extracted.
        The aspects must have paragraphs as their extracted items and no
        sub-aspects, so that each request is for one paragraph and all the aspects.
    :type document: Document
    :param model: The decision model, with no requests recorded in its usage.
    :type model: DocumentDecisionModel
    :return: For each paragraph, in the order of the paragraphs, the probability
        for each aspect, in the order of the aspects, or None for an aspect that
        the model refused to answer about on that paragraph.
    :rtype: list[list[float | None]]
    :raises ValueError: If the aspects cannot be recorded in this way, or if
        the requests recorded are not one for each paragraph with a question for
        each aspect.
    """
    aspects = document.aspects
    if any(
        aspect.reference_depth != "paragraphs" or aspect.aspects for aspect in aspects
    ):
        raise ValueError(
            "The probabilities are recorded only for aspects with paragraphs as "
            "their extracted items and without sub-aspects."
        )
    model.extract_aspects_from_document(document)
    calls = model.get_usage()[0].usage.calls
    if len(calls) != len(document.paragraphs):
        raise ValueError(
            f"{len(calls)} requests are recorded for {len(document.paragraphs)} "
            f"paragraphs."
        )
    probabilities: list[list[float | None]] = []
    for call in calls:
        questions = json.loads(call.prompt)["questions"]
        answers = json.loads(call.response or "[]")
        # The questions of a request are in the order of the aspects
        if len(questions) != len(aspects) or len(answers) != len(aspects):
            raise ValueError(
                f"A request has {len(questions)} questions and {len(answers)} "
                f"answers for {len(aspects)} aspects."
            )
        for aspect, question in zip(aspects, questions, strict=True):
            if f"Aspect name: {aspect.name}\n" not in question["instructions"]:
                raise ValueError(
                    f"A question of a request is not for the aspect `{aspect.name}`."
                )
        probabilities.append(
            [
                float(answer["probability"]) if answer["type"] == "predicate" else None
                for answer in answers
            ]
        )
    return probabilities


def project_probabilities(
    probabilities: list[list[float | None]],
    paragraph_map: list[list[int]],
    n_gold_paragraphs: int,
) -> list[list[float | None]]:
    """
    Gives the paragraphs of the reference answers the probabilities of
    the paragraphs of a document that hold them, for a document whose paragraphs
    are not those of the reference answers (see
    :func:`dev.evals.harness.map_paragraphs`).

    A paragraph of the reference answers has, for each aspect, the highest
    probability of the paragraphs that hold it: a threshold extracts it if it
    extracts one of them, as the scores of the extraction count it.

    :param probabilities: For each paragraph of the document, the probability for
        each aspect, or None for a refused answer.
    :type probabilities: list[list[float | None]]
    :param paragraph_map: For each paragraph of the document, the positions of
        the paragraphs of the reference answers that it holds.
    :type paragraph_map: list[list[int]]
    :param n_gold_paragraphs: The number of paragraphs of the reference answers.
    :type n_gold_paragraphs: int
    :return: For each paragraph of the reference answers, the probability for each
        aspect, or None if no paragraph that holds it has one.
    :rtype: list[list[float | None]]
    """
    n_aspects = len(probabilities[0]) if probabilities else 0
    projected: list[list[float | None]] = [
        [None] * n_aspects for _ in range(n_gold_paragraphs)
    ]
    for row, positions in zip(probabilities, paragraph_map, strict=True):
        for position in positions:
            for aspect_index, probability in enumerate(row):
                current = projected[position][aspect_index]
                if probability is not None and (
                    current is None or probability > current
                ):
                    projected[position][aspect_index] = probability
    return projected


def main() -> None:
    """
    Records the probabilities of a decision model on a case, and writes them.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Records the probabilities of a decision model for the aspects "
        "of a case."
    )
    parser.add_argument("--case", required=True, choices=CASE_NAMES)
    parser.add_argument("--model", required=True, help="e.g. typesafe/jev-latest")
    parser.add_argument(
        "--api-base", default=None, help="address of a local server, for its models"
    )
    parser.add_argument(
        "--label", required=True, help="name of the experiment, e.g. my_experiment"
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

    path = (
        RESULTS_DIR
        / args.label
        / "decision_probabilities"
        / f"{args.case}__{get_model_slug(args.model)}.json"
    )
    if path.exists():
        print(f"exists, skipped: {path.name}", flush=True)
        return
    document, gold = load_case(
        args.case, for_decision_model=True, case_input=args.input
    )
    model, _ = create_model(args.model, "decision", args.api_base)
    if not isinstance(model, DocumentDecisionModel):
        raise TypeError("A decision model is expected.")
    started = time.time()
    with warnings.catch_warnings():
        # The warnings on refused answers: the refused answers are recorded
        warnings.simplefilter("ignore")
        probabilities = record_probabilities(document, model)
    seconds = round(time.time() - started, 1)
    # The score is printed only: the probabilities do not depend on the reference
    # answers, which a score recorded with them would
    aspect_f1 = score_document(document, gold)["summary"]["aspect_f1"]
    usage = get_usage_record(model)
    paragraph_probabilities: dict[str, list[list[float | None]]] = {}
    if "paragraph_map" in gold:
        # Those of the paragraphs of the document, which are not those of
        # the reference answers
        paragraph_probabilities["document_probabilities"] = probabilities
        probabilities = project_probabilities(
            probabilities, gold["paragraph_map"], len(gold["paragraphs"])
        )
    write_json(
        path,
        {
            "case": args.case,
            "model": args.model,
            "input": args.input,
            "aspects": [aspect.name for aspect in document.aspects],
            "probabilities": probabilities,
            **paragraph_probabilities,
            "seconds": seconds,
            "usage": usage,
            "provenance": get_provenance(),
        },
    )
    document_probabilities = paragraph_probabilities.get(
        "document_probabilities", probabilities
    )
    n_refused = sum(
        probability is None for row in document_probabilities for probability in row
    )
    print(
        f"{path.name}: {len(document_probabilities)} paragraphs x {len(document.aspects)} "
        f"aspects, {n_refused} refused, {seconds} s, {usage['input_tokens']} input "
        f"tokens, USD {usage['cost_total']}, aspect F1 at the default threshold "
        f"{aspect_f1:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
