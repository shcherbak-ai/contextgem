# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Judges what the scorer cannot compare exactly, with a decision model: the values
of string concepts, the justifications of the extracted items, and the text fields
of the objects of JSON object concepts. Dates, numbers, labels, Boolean values and
ratings are compared exactly by the scorer (see ``dev.evals.scoring``) and are not
judged.

The judge is asked one yes/no question at a time through the Decisions API of
``litellm``, with the questions of this module, not with the prompts of
the library: a change of the library does not change the judge. Its answer is
the probability of "yes"; a verdict is "correct" at a probability of
``JUDGE_THRESHOLD`` or more. A value that equals the reference value or one of its
accepted values is correct without a request, and a concept without items is not
correct where the reference answers have items.

The requests are live and are paid for. Run it from the root of the repo:

```bash
# Check the judge on the checks of known verdict (cases/judge_checks.json)
uv run python -m dev.evals.judge --checks --label my_experiment

# Judge the runs of an experiment
uv run python -m dev.evals.judge --label my_experiment
```

The verdicts of a run are written to ``results/<label>/judgments/`` under the name
of the file of the run, and those of the checks to ``results/<label>/judge_checks.json``.
A run whose verdicts were given by the same judge, with the same questions and
against the same reference answers, is skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any, Literal

import litellm

from dev.evals.harness import (
    CASES_DIR,
    RESULTS_DIR,
    create_model,
    load_case,
    scrub,
    write_json,
)
from dev.evals.scoring import _normalize_value


# The default judge
JUDGE_MODEL = "typesafe/jev-latest"

# The probability of "yes" from which a verdict is "correct"
JUDGE_THRESHOLD = 0.5

# Requests sent to the judge at the same time
_MAX_CONCURRENT_REQUESTS = 8

# Retries of a request that fails with a transient error, and the errors retried
_MAX_RETRIES = 4
_TRANSIENT_ERRORS = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.Timeout,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
)

# What is judged
JudgedKind = Literal["value", "justification", "field"]

# The question to the judge on each kind of what is judged. The text that it is
# asked on is built by the functions of this module.
JUDGE_INSTRUCTIONS: dict[str, str] = {
    "value": (
        "Decide whether the candidate answer is correct. It is correct if it gives "
        "the same answer to the question as the reference answer, in any words and "
        "in any order: it has every fact of the reference answer that answers "
        "the question, and no fact that is wrong or that changes the answer. "
        "Details that the question does not ask for do not count."
    ),
    "justification": (
        "Decide whether the candidate reasoning is correct. It is correct if "
        "the candidate value is the reference answer, or one of the values of "
        "the reference answer, and every fact and every step of the candidate "
        "reasoning agrees with the reference answer and the reference reasoning. "
        "It may be shorter than the reference reasoning, use other words or cite "
        "the clauses otherwise, as long as nothing in it is wrong."
    ),
    "field": (
        "Decide whether the candidate value of the field is correct. It is correct "
        "if it states the same as the reference value of the field, in any words; "
        "a shorter value is correct if it leaves out nothing that matters."
    ),
}

# The version of the texts that the questions are asked on, which the functions
# `build_*_text` build: raise it when they change, so that the verdicts on
# the texts of another version are told apart
_TEXT_FORMAT_VERSION = 2


def get_judge_fingerprint(model: str) -> dict[str, str]:
    """
    Returns what tells the verdicts of one judge apart from those of another:
    the model, and a short hash of the questions of this module and of the version
    of the texts that they are asked on.

    :param model: The model identifier of the judge.
    :type model: str
    :return: The model, the first 12 characters of the SHA-256 hash of
        the questions and the version of their texts, and the version of
        ``litellm``.
    :rtype: dict[str, str]
    """
    questions = json.dumps(
        [JUDGE_INSTRUCTIONS, _TEXT_FORMAT_VERSION], sort_keys=True
    ).encode("utf-8")
    return {
        "model": model,
        "questions": hashlib.sha256(questions).hexdigest()[:12],
        "litellm": metadata.version("litellm"),
    }


def _get_reference_fingerprint(case: str) -> str:
    """
    Returns a short hash of the reference answers of a case, which the verdicts
    on its runs depend on.

    :param case: The name of the case.
    :type case: str
    :return: The first 12 characters of the SHA-256 hash of the gold file.
    :rtype: str
    """
    data = (CASES_DIR / f"{case}.gold.json").read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:12]


@dataclass
class _Question:
    """
    One question to the judge.

    :ivar kind: What is judged: "value", "justification" or "field".
    :vartype kind: Literal["value", "justification", "field"]
    :ivar text: The text that the question is asked on.
    :vartype text: str
    :ivar where: Where the verdict is written, e.g. the position of the item.
    :vartype where: dict[str, Any]
    """

    kind: JudgedKind
    text: str
    where: dict[str, Any] = field(default_factory=dict)


def _format_value(value: Any) -> str:
    """
    Formats a value for the text that the judge is asked on.

    :param value: The value of an item, as in the reference answers or in a run.
    :type value: Any
    :return: The value as text: a list joined with commas, an object as JSON.
    :rtype: str
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ", ".join(_format_value(element) for element in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _format_values(values: list[Any]) -> str:
    """
    Formats the values of the items of a concept.

    :param values: The values of the items.
    :type values: list[Any]
    :return: The value, or a line for each value if there are several, or "none".
    :rtype: str
    """
    if not values:
        return "none"
    if len(values) == 1:
        return _format_value(values[0])
    return "".join(f"\n- {_format_value(value)}" for value in values)


def _get_question_line(gold_concept: dict[str, Any]) -> str:
    """
    Returns the question that a concept answers, from its name and description.

    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :return: The line of the question.
    :rtype: str
    """
    return f"Question: {gold_concept['name']}. {gold_concept['description']}"


def build_value_text(gold_concept: dict[str, Any], values: list[Any]) -> str:
    """
    Builds the text of the question on the values of a string concept.

    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :param values: The values of the candidate items.
    :type values: list[Any]
    :return: The question, the reference answer with the other wordings that
        the reference answers accept, and the candidate answer.
    :rtype: str
    """
    gold_values = [item["value"] for item in gold_concept["items"]]
    # An accepted wording can say what the reference value does not, e.g. the
    # names of the parties that the value refers to by their roles
    accepted_lines = "".join(
        f"Also a correct wording of the reference answer: {wording}\n"
        for item in gold_concept["items"]
        for wording in item.get("accepted_values", [])
    )
    return (
        f"{_get_question_line(gold_concept)}\n"
        f"Reference answer: {_format_values(gold_values)}\n"
        f"{accepted_lines}"
        f"Candidate answer: {_format_values(values)}"
    )


def build_justification_text(
    gold_concept: dict[str, Any], value: Any, justification: str
) -> str:
    """
    Builds the text of the question on the justification of an item.

    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :param value: The value of the candidate item.
    :type value: Any
    :param justification: The justification of the candidate item.
    :type justification: str
    :return: The question, the reference answer with its reasoning, and
        the candidate value with its reasoning.
    :rtype: str
    """
    gold_items = gold_concept["items"]
    if len(gold_items) == 1:
        reference = (
            f"Reference answer: {_format_value(gold_items[0]['value'])}\n"
            f"Reference reasoning: {gold_items[0]['justification']}"
        )
    else:
        reference = "Reference answer, each value with its reasoning:" + "".join(
            f"\n- {_format_value(item['value'])}: {item['justification']}"
            for item in gold_items
        )
    return (
        f"{_get_question_line(gold_concept)}\n{reference}\n"
        f"Candidate value: {_format_value(value)}\n"
        f"Candidate reasoning: {justification}"
    )


def build_field_text(
    gold_concept: dict[str, Any],
    object_key: str,
    key_value: Any,
    field_name: str,
    gold_value: Any,
    value: Any,
) -> str:
    """
    Builds the text of the question on a field of an object of a JSON object
    concept.

    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :param object_key: The field that identifies the objects.
    :type object_key: str
    :param key_value: The value of that field of the object.
    :type key_value: Any
    :param field_name: The field that is judged.
    :type field_name: str
    :param gold_value: The reference value of the field.
    :type gold_value: Any
    :param value: The candidate value of the field.
    :type value: Any
    :return: The question, the object, the field, and the reference and
        the candidate values.
    :rtype: str
    """
    return (
        f"{_get_question_line(gold_concept)}\n"
        f"Object: the one whose {object_key} is {_format_value(key_value)}\n"
        f"Field: {field_name}\n"
        f"Reference value: {_format_value(gold_value)}\n"
        f"Candidate value: {_format_value(value)}"
    )


def _is_same(value: Any, gold_values: list[Any]) -> bool:
    """
    Tells whether a value equals one of the given reference values, as the scorer
    compares values: letter case and whitespace do not count.

    :param value: The value.
    :type value: Any
    :param gold_values: The reference values to compare with.
    :type gold_values: list[Any]
    :return: True if the value equals one of them.
    :rtype: bool
    """
    return any(
        _normalize_value(value) == _normalize_value(gold) for gold in gold_values
    )


def _plan_concept(
    concept: dict[str, Any], gold_concept: dict[str, Any]
) -> tuple[dict[str, Any], list[_Question]]:
    """
    Plans the judgment of the extracted items of a concept: the verdicts that
    need no request, and the questions to the judge for the others.

    :param concept: The scores of the concept in a run, with the values and
        the justifications of its items.
    :type concept: dict[str, Any]
    :param gold_concept: The reference answers of the concept.
    :type gold_concept: dict[str, Any]
    :return: The judgment of the concept with the verdicts that need no request,
        and the questions for the others, each with where its verdict goes.
    :rtype: tuple[dict[str, Any], list[_Question]]
    """
    values: list[Any] = concept["values"]
    justifications: list[str | None] = concept["justifications"]
    gold_items = gold_concept["items"]
    judgment: dict[str, Any] = {
        "context": concept["context"],
        "name": concept["name"],
        "type": concept["type"],
    }
    questions: list[_Question] = []

    if concept["type"] == "StringConcept" and gold_items:
        if concept["correct"]:
            judgment["value"] = {"verdict": True, "decided_by": "exact"}
        elif not values:
            judgment["value"] = {"verdict": False, "decided_by": "no items"}
        else:
            judgment["value"] = {}
            questions.append(
                _Question(
                    kind="value",
                    text=build_value_text(gold_concept, values),
                    where={"part": "value"},
                )
            )

    if gold_concept["add_justifications"] and gold_items:
        judgment["justifications"] = []
        for position, (value, justification) in enumerate(
            zip(values, justifications, strict=True)
        ):
            entry: dict[str, Any] = {"item": position}
            judgment["justifications"].append(entry)
            if not justification:
                entry.update(verdict=False, decided_by="no justification")
                continue
            questions.append(
                _Question(
                    kind="justification",
                    text=build_justification_text(gold_concept, value, justification),
                    where={"part": "justifications", "position": position},
                )
            )

    if concept["type"] == "JsonObjectConcept":
        object_key = gold_concept.get("custom_data", {}).get("object_key", "name")
        candidates = {
            _normalize_value(value.get(object_key, "")): value for value in values
        }
        judgment["fields"] = []
        for gold_item in gold_items:
            gold_object = gold_item["value"]
            candidate = candidates.get(
                _normalize_value(gold_object.get(object_key, ""))
            )
            if candidate is None:
                # The objects themselves are scored by the scorer
                continue
            for field_name, gold_value in gold_object.items():
                if field_name == object_key:
                    continue
                entry = {"object": gold_object[object_key], "field": field_name}
                judgment["fields"].append(entry)
                value = candidate.get(field_name)
                if value is None:
                    entry.update(verdict=False, decided_by="missing")
                elif _is_same(value, [gold_value]):
                    entry.update(verdict=True, decided_by="exact")
                elif not isinstance(gold_value, str):
                    # Only text is judged: a number, a list or a Boolean value
                    # is compared exactly
                    entry.update(verdict=False, decided_by="exact")
                else:
                    questions.append(
                        _Question(
                            kind="field",
                            text=build_field_text(
                                gold_concept,
                                object_key,
                                gold_object[object_key],
                                field_name,
                                gold_value,
                                value,
                            ),
                            where={
                                "part": "fields",
                                "position": len(judgment["fields"]) - 1,
                            },
                        )
                    )
    return judgment, questions


@dataclass
class _Usage:
    """
    The requests sent to the judge and their tokens.

    :ivar requests: The number of requests.
    :vartype requests: int
    :ivar input_tokens: The input tokens of the requests.
    :vartype input_tokens: int
    :ivar output_tokens: The output tokens of the requests.
    :vartype output_tokens: int
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class _Judge:
    """
    Sends the questions to the judge.

    :ivar model: The model identifier of the judge.
    :vartype model: str
    """

    def __init__(self, model: str) -> None:
        decision_model, self._private_texts = create_model(model, "decision")
        self.model = model
        self._api_key = decision_model.api_key
        self._api_base = decision_model.api_base
        pricing = decision_model.pricing_details
        self._input_price = (
            float(pricing.input_per_1m_tokens) if pricing is not None else 0.0
        )
        self._semaphore: asyncio.Semaphore | None = None

    def get_cost(self, usage: _Usage) -> float:
        """
        Returns the cost of the requests, in USD (output tokens are free).

        :param usage: The requests and their tokens.
        :type usage: _Usage
        :return: The cost.
        :rtype: float
        """
        return usage.input_tokens * self._input_price / 1_000_000

    async def ask(self, question: _Question, usage: _Usage) -> dict[str, Any]:
        """
        Asks the judge one question, retrying a request that fails with
        a transient error.

        :param question: The question.
        :type question: _Question
        :param usage: The usage, which the request is added to.
        :type usage: _Usage
        :return: The verdict: the probability of "yes", and whether it is
            "correct"; or, if the judge refused to answer, the refusal.
        :rtype: dict[str, Any]
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(_MAX_CONCURRENT_REQUESTS)
        request_question = {
            "type": "predicate",
            "name": "J",
            "instructions": JUDGE_INSTRUCTIONS[question.kind],
        }
        async with self._semaphore:
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    response = await litellm.adecisions(
                        model=self.model,
                        input=question.text,
                        questions=[request_question],
                        api_key=self._api_key,
                        api_base=self._api_base,
                        timeout=120,
                    )
                    break
                except _TRANSIENT_ERRORS:
                    if attempt == _MAX_RETRIES:
                        raise
                    await asyncio.sleep(2**attempt + random.random())  # nosec B311 - not a security context
        usage.requests += 1
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens
        answer = response.answers[0]
        if getattr(answer, "type", None) != "predicate":
            return {"verdict": None, "decided_by": "refused"}
        probability = float(answer.probability)
        return {
            "probability": round(probability, 4),
            "verdict": probability >= JUDGE_THRESHOLD,
            "decided_by": "judge",
        }

    async def ask_all(
        self, questions: list[_Question], usage: _Usage
    ) -> list[dict[str, Any]]:
        """
        Asks the judge the questions, several at a time.

        :param questions: The questions.
        :type questions: list[_Question]
        :param usage: The usage, which the requests are added to.
        :type usage: _Usage
        :return: The verdict on each question, in the order of the questions.
        :rtype: list[dict[str, Any]]
        """
        return list(
            await asyncio.gather(*(self.ask(question, usage) for question in questions))
        )

    def scrub(self, text: str) -> str:
        """
        Replaces the texts that are not to be written or printed, e.g. the host of
        a Microsoft Foundry resource.

        :param text: The text.
        :type text: str
        :return: The text without them.
        :rtype: str
        """
        return scrub(text, self._private_texts)


def _summarize_judgments(concepts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Counts the verdicts of the judgment of a run.

    :param concepts: The judgment of each concept of the run.
    :type concepts: list[dict[str, Any]]
    :return: For the string values, the justifications and the fields of objects:
        the number of verdicts (refusals left out) and of "correct" ones.
    :rtype: dict[str, Any]
    """
    summary: dict[str, Any] = {}
    for part, entries in (
        ("strings", [c["value"] for c in concepts if "value" in c]),
        ("justifications", [e for c in concepts for e in c.get("justifications", [])]),
        ("fields", [e for c in concepts for e in c.get("fields", [])]),
    ):
        verdicts = [
            entry["verdict"] for entry in entries if entry["verdict"] is not None
        ]
        summary[f"n_{part}"] = len(verdicts)
        summary[f"n_{part}_correct"] = sum(verdicts)
    return summary


async def judge_run(judge: _Judge, run: dict[str, Any]) -> dict[str, Any]:
    """
    Judges what the scorer cannot compare exactly in a run.

    :param judge: The judge.
    :type judge: _Judge
    :param run: The record of the run (see ``dev.evals.run_eval``).
    :type run: dict[str, Any]
    :return: The judgment of the run: the judge, the reference answers, the verdicts
        on each concept, their counts, and the requests sent to the judge.
    :rtype: dict[str, Any]
    :raises ValueError: If the run does not keep the values and the justifications
        of its items, which runs recorded before the judge was added do not.
    """
    _, gold = load_case(run["case"], for_decision_model=run["kind"] == "decision")
    gold_concepts = {("document", c["name"]): c for c in gold["concepts"]} | {
        (f"aspect: {a['name']}", c["name"]): c
        for a in gold["aspects"]
        for c in a["concepts"]
    }
    concepts: list[dict[str, Any]] = []
    planned: list[tuple[dict[str, Any], _Question]] = []
    for concept in run["scores"]["concepts"]:
        if "values" not in concept:
            raise ValueError(
                "The run does not keep the values and the justifications of its "
                "items: record it again."
            )
        judgment, questions = _plan_concept(
            concept, gold_concepts[(concept["context"], concept["name"])]
        )
        concepts.append(judgment)
        planned += [(judgment, question) for question in questions]

    usage = _Usage()
    verdicts = await judge.ask_all([question for _, question in planned], usage)
    for (judgment, question), verdict in zip(planned, verdicts, strict=True):
        part = question.where["part"]
        if part == "value":
            judgment["value"] = verdict
        else:
            judgment[part][question.where["position"]].update(verdict)
    return {
        "judge": get_judge_fingerprint(judge.model),
        "threshold": JUDGE_THRESHOLD,
        "reference": _get_reference_fingerprint(run["case"]),
        "summary": _summarize_judgments(concepts),
        "usage": {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_total": f"{judge.get_cost(usage):.5f}",
        },
        "concepts": concepts,
    }


def _get_gold_concept(check: dict[str, Any]) -> dict[str, Any]:
    """
    Returns the reference answers of the concept that a check is about.

    :param check: The check.
    :type check: dict[str, Any]
    :return: The reference answers of the concept.
    :rtype: dict[str, Any]
    :raises ValueError: If the case has no such concept.
    """
    _, gold = load_case(check["case"])
    if check["context"] == "document":
        concepts = gold["concepts"]
    else:
        aspect_name = check["context"].removeprefix("aspect: ")
        concepts = next(
            a["concepts"] for a in gold["aspects"] if a["name"] == aspect_name
        )
    for concept in concepts:
        if concept["name"] == check["concept"]:
            return concept
    raise ValueError(f"The check `{check['id']}` is about an unknown concept.")


def build_check_question(check: dict[str, Any]) -> _Question:
    """
    Builds the question to the judge of a check of known verdict.

    :param check: The check (see ``cases/judge_checks.json``).
    :type check: dict[str, Any]
    :return: The question, as the judge is asked it on a run.
    :rtype: _Question
    """
    gold_concept = _get_gold_concept(check)
    candidate = check["candidate"]
    if check["kind"] == "value":
        text = build_value_text(gold_concept, candidate["values"])
    elif check["kind"] == "justification":
        text = build_justification_text(
            gold_concept, candidate["value"], candidate["justification"]
        )
    else:
        object_key = gold_concept.get("custom_data", {}).get("object_key", "name")
        gold_object = next(
            item["value"]
            for item in gold_concept["items"]
            if item["value"][object_key] == candidate["object"]
        )
        text = build_field_text(
            gold_concept,
            object_key,
            candidate["object"],
            candidate["field"],
            gold_object[candidate["field"]],
            candidate["value"],
        )
    return _Question(kind=check["kind"], text=text)


async def run_checks(judge: _Judge) -> dict[str, Any]:
    """
    Asks the judge the checks of known verdict, and counts the verdicts that
    agree with the expected ones.

    :param judge: The judge.
    :type judge: _Judge
    :return: The judge, the result of each check, the agreement for each kind of
        check, and the requests sent.
    :rtype: dict[str, Any]
    """
    checks = json.loads((CASES_DIR / "judge_checks.json").read_text(encoding="utf-8"))
    usage = _Usage()
    verdicts = await judge.ask_all([build_check_question(c) for c in checks], usage)
    results = [
        {
            "id": check["id"],
            "kind": check["kind"],
            "expected": check["expected"],
            **verdict,
            "agrees": verdict["verdict"] == check["expected"],
        }
        for check, verdict in zip(checks, verdicts, strict=True)
    ]
    agreement: dict[str, Any] = {}
    for kind in ("value", "justification", "field", "all"):
        of_kind = [r for r in results if kind in ("all", r["kind"])]
        agreement[kind] = {
            "checks": len(of_kind),
            "agree": sum(r["agrees"] for r in of_kind),
            "correct_judged_wrong": sum(
                r["expected"] and r["verdict"] is False for r in of_kind
            ),
            "wrong_judged_correct": sum(
                not r["expected"] and r["verdict"] is True for r in of_kind
            ),
            "refused": sum(r["verdict"] is None for r in of_kind),
        }
    return {
        "judge": get_judge_fingerprint(judge.model),
        "threshold": JUDGE_THRESHOLD,
        "agreement": agreement,
        "usage": {
            "requests": usage.requests,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cost_total": f"{judge.get_cost(usage):.5f}",
        },
        "checks": results,
    }


async def judge_label(judge: _Judge, label: str) -> None:
    """
    Judges the runs of an experiment that have something to judge and are not
    judged yet by the same judge against the same reference answers, and writes
    and prints the verdicts of each.

    :param judge: The judge.
    :type judge: _Judge
    :param label: The label of the experiment.
    :type label: str
    :return: None
    :rtype: None
    :raises ValueError: If the experiment has no runs.
    """
    paths = sorted((RESULTS_DIR / label / "runs").glob("*.json"))
    if not paths:
        raise ValueError(f"No runs are recorded for the label `{label}`.")
    fingerprint = get_judge_fingerprint(judge.model)
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["kind"] == "decision":
            # A decision model extracts no string concepts, no JSON object concepts
            # and no justifications
            continue
        judgment_path = RESULTS_DIR / label / "judgments" / path.name
        if judgment_path.exists():
            recorded = json.loads(judgment_path.read_text(encoding="utf-8"))
            if recorded["judge"] == fingerprint and recorded[
                "reference"
            ] == _get_reference_fingerprint(run["case"]):
                print(f"judged, skipped: {path.name}", flush=True)
                continue
        judgment = await judge_run(judge, run)
        write_json(judgment_path, judgment)
        summary = judgment["summary"]
        print(
            f"{path.name}: strings {summary['n_strings_correct']}/"
            f"{summary['n_strings']}, justifications "
            f"{summary['n_justifications_correct']}/{summary['n_justifications']}, "
            f"fields {summary['n_fields_correct']}/{summary['n_fields']}, "
            f"{judgment['usage']['requests']} requests, "
            f"USD {judgment['usage']['cost_total']}",
            flush=True,
        )


def main() -> None:
    """
    Judges the runs of an experiment, or checks the judge.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Judges what the scorer cannot compare exactly, with a decision "
        "model."
    )
    parser.add_argument("--label", required=True, help="name of the experiment")
    parser.add_argument("--model", default=JUDGE_MODEL, help="the judge")
    parser.add_argument(
        "--checks",
        action="store_true",
        help="ask the checks of known verdict instead of judging the runs",
    )
    args = parser.parse_args()

    judge = _Judge(args.model)
    try:
        if args.checks:
            result = asyncio.run(run_checks(judge))
            write_json(RESULTS_DIR / args.label / "judge_checks.json", result)
            for kind, counts in result["agreement"].items():
                print(
                    f"{kind}: {counts['agree']}/{counts['checks']} agree, "
                    f"{counts['correct_judged_wrong']} correct judged wrong, "
                    f"{counts['wrong_judged_correct']} wrong judged correct, "
                    f"{counts['refused']} refused",
                    flush=True,
                )
            print(f"USD {result['usage']['cost_total']}", flush=True)
        else:
            asyncio.run(judge_label(judge, args.label))
    except Exception as e:  # noqa: BLE001 - any error is reported, scrubbed
        raise SystemExit(judge.scrub(f"{type(e).__name__}: {e}")) from None


if __name__ == "__main__":
    main()
