# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Scoring of what is extracted from the document of an evaluation case against
the reference answers of the case. Nothing here sends a request.

What is scored, and how:

- Aspects: the paragraphs extracted for an aspect against those of the reference
  answers, as precision, recall and F1 of the positions of the paragraphs.
  The paragraphs that are neutral for the aspect (the headings of its sections,
  and the paragraphs that careful readers disagree on) are not scored: extracting
  one, or leaving it out, changes nothing.
- Date, numerical, Boolean, label and rating concepts: correct if the values of
  the extracted items equal the reference values, whatever their order.
- JSON object concepts: correct if the names of the extracted objects equal
  the reference ones, or the values of another field that identifies the objects,
  if the reference answers name one (``object_key`` in the ``custom_data`` of
  the concept). The share of the reference fields that match is given too.
- String concepts: correct only if each value equals the reference value or one
  of its accepted values. These are counted apart from the other concepts, as
  a narrative answer can be right in other words, which only a judge can tell.
- References of concepts: the paragraphs referenced by the items of a concept
  against those of the reference answers, as precision, recall and F1, where
  the reference answers have references.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

from contextgem import Document


def _normalize_text(value: str) -> str:
    """
    Normalizes a text for a comparison: whitespace and letter case do not count.

    :param value: The text.
    :type value: str
    :return: The text with each run of whitespace as one space, without leading
        and trailing whitespace, case-folded.
    :rtype: str
    """
    return re.sub(r"\s+", " ", value).strip().casefold()


def _normalize_value(value: Any) -> Any:
    """
    Brings the value of an extracted item, or a reference value, to a form in
    which the two can be compared: a date as its ISO string, a number as a float,
    a text normalized, a list sorted, and a dictionary with its keys sorted.

    :param value: The value.
    :type value: Any
    :return: The value in the form to compare.
    :rtype: Any
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, list | tuple):
        return sorted(_normalize_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in sorted(value.items())}
    return value


def _get_sort_key(value: Any) -> str:
    """
    Returns the key that normalized values of any type are sorted by.

    :param value: A normalized value.
    :type value: Any
    :return: The value as a JSON string with sorted keys.
    :rtype: str
    """
    return json.dumps(value, sort_keys=True)


def get_set_scores(predicted: set[int], expected: set[int]) -> dict[str, float]:
    """
    Scores a set of predicted positions against the expected ones.

    :param predicted: The predicted positions, e.g. of the paragraphs extracted.
    :type predicted: set[int]
    :param expected: The expected positions, e.g. of the paragraphs of the reference answers.
    :type expected: set[int]
    :return: ``precision``, ``recall`` and ``f1``. With nothing predicted,
        the precision is 1 if nothing is expected, else 0, and the same holds
        for the recall with nothing expected.
    :rtype: dict[str, float]
    """
    true_positives = len(predicted & expected)
    precision = true_positives / len(predicted) if predicted else float(not expected)
    recall = true_positives / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_concept_values(
    name: str,
    predicted_values: list[Any],
    predicted_references: set[int],
    gold_concept: dict[str, Any],
) -> dict[str, Any]:
    """
    Scores the values and the references extracted for a concept against its
    reference answers.

    :param name: The name of the concept.
    :type name: str
    :param predicted_values: The values of the extracted items, normalized (see
        :func:`_normalize_value`).
    :type predicted_values: list[Any]
    :param predicted_references: The positions of the paragraphs that
        the extracted items reference.
    :type predicted_references: set[int]
    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :return: The name and the type of the concept, the extracted and the reference
        values, how the concept is scored (``scoring``), whether it is correct
        (``correct``), the positions of the paragraphs that its items reference
        (``predicted_references``), and the scores of its references
        (``references``) if the reference answers have them.
    :rtype: dict[str, Any]
    """
    concept_type = gold_concept["type"]
    gold_items = gold_concept["items"]
    gold_values = [_normalize_value(item["value"]) for item in gold_items]
    result: dict[str, Any] = {
        "name": name,
        "type": concept_type,
        "n_items": len(predicted_values),
        "n_gold_items": len(gold_items),
        # Kept with the references, so that a run can be scored again after
        # the reference answers change
        "predicted": [json.loads(json.dumps(v, default=str)) for v in predicted_values],
        "gold": gold_values,
    }
    if concept_type == "StringConcept":
        # A narrative answer can be right in other words, which only a judge can
        # tell: scored as an exact match with the value or with an accepted one
        accepted = [
            {
                _normalize_text(value)
                for value in [item["value"], *item.get("accepted_values", [])]
            }
            for item in gold_items
        ]
        result["scoring"] = "exact_or_accepted"
        result["correct"] = len(predicted_values) == len(accepted) and all(
            any(value in options for options in accepted) for value in predicted_values
        )
    elif concept_type == "JsonObjectConcept":
        # The objects are matched by their names, or by the field that the reference
        # answers name: their other fields can be texts
        object_key = gold_concept.get("custom_data", {}).get("object_key", "name")
        predicted_keys = sorted(value.get(object_key, "") for value in predicted_values)
        gold_keys = sorted(value.get(object_key, "") for value in gold_values)
        result["scoring"] = (
            "names_of_objects" if object_key == "name" else "keys_of_objects"
        )
        result["correct"] = predicted_keys == gold_keys
        predicted_by_key = {
            value.get(object_key, ""): value for value in predicted_values
        }
        fields = [
            predicted_by_key.get(gold_value.get(object_key, ""), {}).get(key) == value
            for gold_value in gold_values
            for key, value in gold_value.items()
        ]
        result["field_match_rate"] = sum(fields) / len(fields) if fields else None
    else:
        result["scoring"] = "exact"
        result["correct"] = sorted(predicted_values, key=_get_sort_key) == sorted(
            gold_values, key=_get_sort_key
        )
    # The references of the items, as the positions of paragraphs in the document
    gold_references = {
        position for item in gold_items for position in item["reference_paragraphs"]
    }
    result["predicted_references"] = sorted(predicted_references)
    if gold_references:
        result["references"] = get_set_scores(predicted_references, gold_references)
    return result


def score_concept(
    concept: Any, gold_concept: dict[str, Any], document: Document
) -> dict[str, Any]:
    """
    Scores the extracted items of a concept against its reference answers.

    :param concept: The concept, after the extraction.
    :type concept: Any
    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :param document: The document that the concept is extracted from.
    :type document: Document
    :return: The scores of the concept (see :func:`score_concept_values`).
    :rtype: dict[str, Any]
    """
    items = list(concept.extracted_items)
    result = score_concept_values(
        concept.name,
        [_normalize_value(item.value) for item in items],
        {
            document.get_paragraph_index(paragraph)
            for item in items
            for paragraph in item.reference_paragraphs
        },
        gold_concept,
    )
    # Kept for the judge (see dev.evals.judge): the values as extracted, not
    # normalized, and the justification of each item
    result["values"] = [
        json.loads(json.dumps(item.value, default=str)) for item in items
    ]
    result["justifications"] = [item.justification for item in items]
    return result


def score_aspect_paragraphs(
    name: str, predicted: set[int], gold_aspect: dict[str, Any]
) -> dict[str, Any]:
    """
    Scores the paragraphs extracted for an aspect against its reference answers.

    The neutral paragraphs of the aspect (``neutral_paragraphs`` of its reference
    answers) are not scored: they are left out of the extracted paragraphs and
    of the reference ones.

    :param name: The name of the aspect.
    :type name: str
    :param predicted: The positions of the paragraphs extracted for the aspect.
    :type predicted: set[int]
    :param gold_aspect: The reference answers of the aspect.
    :type gold_aspect: dict[str, Any]
    :return: The name of the aspect, the number of the scored paragraphs
        extracted and of those of the reference answers, the number of
        the neutral paragraphs extracted, the positions of all the paragraphs
        extracted (``predicted_paragraphs``), and the precision, recall and F1.
    :rtype: dict[str, Any]
    """
    neutral = set(gold_aspect.get("neutral_paragraphs", []))
    expected = set(gold_aspect["reference_paragraphs"]) - neutral
    return {
        "name": name,
        "n_predicted": len(predicted - neutral),
        "n_gold": len(expected),
        "n_neutral_predicted": len(predicted & neutral),
        # Kept, so that a run can be scored again after the reference answers
        # change
        "predicted_paragraphs": sorted(predicted),
        **get_set_scores(predicted - neutral, expected),
    }


def _get_mean(values: list[float]) -> float | None:
    """
    Returns the mean of the values.

    :param values: The values.
    :type values: list[float]
    :return: The mean, or None if there are no values.
    :rtype: float | None
    """
    return sum(values) / len(values) if values else None


def _get_scores(
    aspects: list[dict[str, Any]], concepts: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Puts the scores of the aspects and of the concepts of a document together
    with their summary.

    :param aspects: The scores of each aspect.
    :type aspects: list[dict[str, Any]]
    :param concepts: The scores of each concept, of the document and of the aspects.
    :type concepts: list[dict[str, Any]]
    :return: The scores of the aspects (``aspects``) and of the concepts
        (``concepts``), and their summary (``summary``): the precision, recall
        and F1 averaged over the aspects, the number of the concepts scored
        without a judge and of the correct ones, the share of the string concepts
        with an exact or accepted value, and the F1 of the references averaged
        over the concepts that have them.
    :rtype: dict[str, Any]
    """
    deterministic = [c for c in concepts if c["scoring"] != "exact_or_accepted"]
    strings = [c for c in concepts if c["scoring"] == "exact_or_accepted"]
    return {
        "summary": {
            "aspect_f1": _get_mean([a["f1"] for a in aspects]),
            "aspect_precision": _get_mean([a["precision"] for a in aspects]),
            "aspect_recall": _get_mean([a["recall"] for a in aspects]),
            "concept_accuracy": _get_mean([float(c["correct"]) for c in deterministic]),
            "n_concepts_scored": len(deterministic),
            "n_concepts_correct": sum(c["correct"] for c in deterministic),
            "string_exact": _get_mean([float(c["correct"]) for c in strings]),
            "n_strings": len(strings),
            "reference_f1": _get_mean(
                [c["references"]["f1"] for c in concepts if "references" in c]
            ),
            "n_concepts_without_items": sum(
                c["n_items"] == 0 and c["n_gold_items"] > 0 for c in concepts
            ),
        },
        "aspects": aspects,
        "concepts": concepts,
    }


def score_document(document: Document, gold: dict[str, Any]) -> dict[str, Any]:
    """
    Scores what is extracted from the document of a case against the reference
    answers of the case (see :func:`dev.evals.harness.load_case`, which gives both).

    :param document: The document, after the extraction.
    :type document: Document
    :param gold: The reference answers of the document, for the same aspects and
        concepts, in the same order.
    :type gold: dict[str, Any]
    :return: The scores of each aspect, the scores of each concept of the document
        and of the aspects, and their summary (see :func:`_get_scores`).
    :rtype: dict[str, Any]
    """
    aspects: list[dict[str, Any]] = []
    concepts: list[dict[str, Any]] = []
    for concept, gold_concept in zip(document.concepts, gold["concepts"], strict=True):
        concepts.append(
            {"context": "document", **score_concept(concept, gold_concept, document)}
        )
    for aspect, gold_aspect in zip(document.aspects, gold["aspects"], strict=True):
        aspects.append(
            score_aspect_paragraphs(
                aspect.name,
                {
                    document.get_paragraph_index(paragraph)
                    for paragraph in aspect.reference_paragraphs
                },
                gold_aspect,
            )
        )
        for concept, gold_concept in zip(
            aspect.concepts, gold_aspect["concepts"], strict=True
        ):
            concepts.append(
                {
                    "context": f"aspect: {aspect.name}",
                    **score_concept(concept, gold_concept, document),
                }
            )
    return _get_scores(aspects, concepts)


def rescore(scores: dict[str, Any], gold: dict[str, Any]) -> dict[str, Any]:
    """
    Scores a recorded run again, against the reference answers as they are now,
    from what the run keeps of its extraction: the paragraphs extracted for each
    aspect, and the values and the references extracted for each concept.

    :param scores: The scores recorded for the run (see :func:`score_document`).
    :type scores: dict[str, Any]
    :param gold: The reference answers of the document of the case, for the same
        aspects and concepts as when the run was recorded, in the same order.
    :type gold: dict[str, Any]
    :return: The scores of the run against these reference answers.
    :rtype: dict[str, Any]
    :raises ValueError: If the aspects or the concepts of the run are not those
        of the reference answers, or if the run does not keep its extraction.
    """
    gold_concepts = [("document", concept) for concept in gold["concepts"]] + [
        (f"aspect: {aspect['name']}", concept)
        for aspect in gold["aspects"]
        for concept in aspect["concepts"]
    ]
    if [aspect["name"] for aspect in scores["aspects"]] != [
        aspect["name"] for aspect in gold["aspects"]
    ] or [(concept["context"], concept["name"]) for concept in scores["concepts"]] != [
        (context, concept["name"]) for context, concept in gold_concepts
    ]:
        raise ValueError(
            "The aspects or the concepts of the run are not those of the reference "
            "answers."
        )
    if any("predicted_paragraphs" not in aspect for aspect in scores["aspects"]) or any(
        "predicted_references" not in concept for concept in scores["concepts"]
    ):
        raise ValueError(
            "The run does not keep the paragraphs and references that it extracted: "
            "record it again."
        )
    aspects = [
        score_aspect_paragraphs(
            aspect["name"], set(aspect["predicted_paragraphs"]), gold_aspect
        )
        for aspect, gold_aspect in zip(scores["aspects"], gold["aspects"], strict=True)
    ]
    concepts = [
        {
            "context": context,
            **score_concept_values(
                concept["name"],
                [_normalize_value(value) for value in concept["predicted"]],
                set(concept["predicted_references"]),
                gold_concept,
            ),
            # What the run keeps for the judge, which the scores do not depend on
            **{
                key: concept[key]
                for key in ("values", "justifications")
                if key in concept
            },
        }
        for concept, (context, gold_concept) in zip(
            scores["concepts"], gold_concepts, strict=True
        )
    ]
    return _get_scores(aspects, concepts)
