# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Checks the scorer of the evaluation harness on each case, with no request sent:
a document with nothing extracted must score nothing, and a document filled with
the reference answers of the case must score everything. Each case is checked
with its document created from its text, and from its Docling document, whose
paragraphs must be those of the reference answers (see
``dev/evals/docling_cases.py``, which writes it again after a change in a case).

Run it from the root of the repo, after a change in the scorer, in a case or in
its reference answers:

```bash
uv run python -m dev.evals.check_scorer
```
"""

from __future__ import annotations

from datetime import date
from typing import Any

from contextgem import Document
from dev.evals.harness import CASE_NAMES, CaseInput, load_case
from dev.evals.scoring import score_document


def _fill_concept(
    concept: Any, gold_concept: dict[str, Any], document: Document
) -> None:
    """
    Gives a concept the extracted items that its reference answers describe.

    :param concept: The concept.
    :type concept: Any
    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :param document: The document of the concept, whose paragraphs the items
        reference.
    :type document: Document
    :return: None
    :rtype: None
    """
    items = []
    for gold_item in gold_concept["items"]:
        value = gold_item["value"]
        if gold_concept["type"] == "DateConcept":
            value = date.fromisoformat(value)
        # The class of the items that the concept accepts
        item = concept._item_class(value=value)
        item.reference_paragraphs = [
            document.paragraphs[position]
            for position in gold_item["reference_paragraphs"]
        ]
        items.append(item)
    concept.extracted_items = items


def check_case(case: str, case_input: CaseInput = "text") -> None:
    """
    Checks the scores of the document of a case with nothing extracted, and of
    the same document filled with the reference answers.

    :param case: The name of the case.
    :type case: str
    :param case_input: What the document of the case is created from (see
        :func:`~dev.evals.harness.load_case`). Defaults to "text".
    :type case_input: CaseInput
    :return: None
    :rtype: None
    :raises AssertionError: If a score is not the expected one.
    """
    document, gold = load_case(case, case_input=case_input)
    scores = score_document(document, gold)
    summary = scores["summary"]
    assert summary["aspect_f1"] == 0, summary
    # A concept without items in the reference answers is correct when nothing is extracted
    assert summary["n_concepts_correct"] == sum(
        concept["n_gold_items"] == 0
        for concept in scores["concepts"]
        if concept["scoring"] != "exact_or_accepted"
    ), summary
    assert summary["string_exact"] == 0, summary

    for concept, gold_concept in zip(document.concepts, gold["concepts"], strict=True):
        _fill_concept(concept, gold_concept, document)
    for aspect, gold_aspect in zip(document.aspects, gold["aspects"], strict=True):
        aspect.reference_paragraphs = [
            document.paragraphs[position]
            for position in gold_aspect["reference_paragraphs"]
        ]
        for concept, gold_concept in zip(
            aspect.concepts, gold_aspect["concepts"], strict=True
        ):
            _fill_concept(concept, gold_concept, document)
    summary = score_document(document, gold)["summary"]
    assert summary["aspect_f1"] == 1, summary
    assert summary["aspect_precision"] == 1 and summary["aspect_recall"] == 1, summary
    assert summary["n_concepts_correct"] == summary["n_concepts_scored"] > 0, summary
    assert summary["string_exact"] == 1, summary
    assert summary["reference_f1"] == 1, summary
    assert summary["n_concepts_without_items"] == 0, summary

    # The neutral paragraphs of an aspect are not scored: the scores are the same
    # with all of them extracted, and with none of them
    n_neutral = 0
    for extract_neutral in (True, False):
        for aspect, gold_aspect in zip(document.aspects, gold["aspects"], strict=True):
            neutral = set(gold_aspect["neutral_paragraphs"])
            positions = set(gold_aspect["reference_paragraphs"]) - neutral
            if extract_neutral:
                n_neutral += len(neutral)
                positions |= neutral
            aspect.reference_paragraphs = [
                document.paragraphs[position] for position in sorted(positions)
            ]
        scores = score_document(document, gold)
        assert scores["summary"]["aspect_f1"] == 1, scores["summary"]
        assert all(
            aspect_scores["n_predicted"] == aspect_scores["n_gold"]
            for aspect_scores in scores["aspects"]
        ), scores["aspects"]
    print(
        f"{case} ({case_input}): ok ({len(document.aspects)} aspects with "
        f"{n_neutral} neutral "
        f"paragraphs, {summary['n_concepts_scored']} concepts scored without "
        f"a judge, {summary['n_strings']} string concepts)"
    )


def main() -> None:
    """
    Checks the scorer on each case.

    :return: None
    :rtype: None
    """
    for case in CASE_NAMES:
        for case_input in ("text", "docling"):
            check_case(case, case_input)


if __name__ == "__main__":
    main()
