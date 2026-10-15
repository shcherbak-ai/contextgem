# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Summarizes the runs of an experiment, recorded by ``run_eval``: the runs of
the same case, model and mode are put together, with the mean and the range of
each score over them. Reads the files of the runs only, i.e. sends no request.

Run it from the root of the repo:

```bash
uv run python -m dev.evals.summarize --label my_experiment
```

Writes ``results/<label>/summary.md`` and ``results/<label>/summary.json``, and
prints the summary. The verdicts of the judge on the runs (``results/<label>/judgments/``,
written by ``dev.evals.judge``) are summarized too, where they exist, and what
the requests of the experiment cost (see ``dev.evals.costs``).
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any

from dev.evals.costs import PART_NAMES, get_costs, get_total
from dev.evals.harness import RESULTS_DIR, write_json


# The concepts that LLMs and decision models both extract, i.e. the closed
# questions: on these, the two kinds of model are compared on the same answers
_CLOSED_CONCEPT_TYPES = ("BooleanConcept", "LabelConcept", "RatingConcept")


def _format_spread(values: list[float | None], digits: int = 3) -> str:
    """
    Formats the scores of the runs of a configuration.

    :param values: The score of each run. A run without the score is left out.
    :type values: list[float | None]
    :param digits: The number of decimal places. Defaults to 3.
    :type digits: int
    :return: The mean, followed by the lowest and the highest value in brackets
        if the values differ, or "n/a" if no run has the score.
    :rtype: str
    """
    scores = [value for value in values if value is not None]
    if not scores:
        return "n/a"
    mean = statistics.mean(scores)
    if min(scores) == max(scores):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ({min(scores):.{digits}f}-{max(scores):.{digits}f})"


def _format_count(values: list[float]) -> str:
    """
    Formats the counts of the runs of a configuration, e.g. of tokens.

    :param values: The count of each run.
    :type values: list[float]
    :return: The mean, followed by the lowest and the highest value in brackets
        if the values differ.
    :rtype: str
    """
    mean = statistics.mean(values)
    if min(values) == max(values):
        return f"{mean:,.0f}"
    return f"{mean:,.0f} ({min(values):,.0f}-{max(values):,.0f})"


def _format_correct(correct: list[int], total: int) -> str:
    """
    Formats the numbers of correct concepts of the runs of a configuration.

    :param correct: The number of correct concepts of each run.
    :type correct: list[int]
    :param total: The number of concepts scored in a run.
    :type total: int
    :return: E.g. "22/22", or the mean with the lowest and the highest number in
        brackets if the runs differ.
    :rtype: str
    """
    if min(correct) == max(correct):
        return f"{correct[0]}/{total}"
    return f"{statistics.mean(correct):.1f}/{total} ({min(correct)}-{max(correct)})"


def _count_closed_concepts(run: dict[str, Any]) -> tuple[int, int]:
    """
    Counts the closed concepts of a run (see ``_CLOSED_CONCEPT_TYPES``) and
    the correct ones among them.

    :param run: The record of the run.
    :type run: dict[str, Any]
    :return: The number of the correct closed concepts, and of the closed concepts.
    :rtype: tuple[int, int]
    """
    closed = [
        concept
        for concept in run["scores"]["concepts"]
        if concept["type"] in _CLOSED_CONCEPT_TYPES
    ]
    return sum(concept["correct"] for concept in closed), len(closed)


def _format_share(correct: int, total: int) -> str:
    """
    Formats a number of correct verdicts out of a number of verdicts.

    :param correct: The number of correct verdicts.
    :type correct: int
    :param total: The number of verdicts.
    :type total: int
    :return: E.g. "45/50 (0.90)", or "n/a" without verdicts.
    :rtype: str
    """
    if not total:
        return "n/a"
    return f"{correct}/{total} ({correct / total:.2f})"


def _get_judged_lines(
    groups: dict[tuple[str, str, str], list[dict[str, Any]]],
    judgments: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[str]:
    """
    Returns the section of the summary on the verdicts of the judge: for each
    case, model and mode, the string values, the justifications and the fields
    of objects judged correct, over the runs judged, and the cost of the judge.

    :param groups: The runs by case, model and mode.
    :type groups: dict[tuple[str, str, str], list[dict[str, Any]]]
    :param judgments: The judgments of the runs by case, model and mode.
    :type judgments: dict[tuple[str, str, str], list[dict[str, Any]]]
    :return: The lines of the section.
    :rtype: list[str]
    """
    judges = sorted(
        {j["judge"]["model"] for group in judgments.values() for j in group}
    )
    lines = [
        "",
        f"## Judged by {', '.join(judges)}",
        "",
        (
            "String values, justifications and text fields of objects, over all the "
            "runs judged; a value equal to the reference value or to an accepted one "
            "is correct without the judge."
        ),
        "",
        (
            "| Case | Model | Mode | Runs judged | Strings correct | "
            "Justifications correct | Object fields correct | Judge requests | "
            "Judge cost (USD) |"
        ),
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(groups):
        group = judgments.get(key, [])
        if not group:
            continue
        counts = {
            name: sum(j["summary"][name] for j in group) for name in group[0]["summary"]
        }
        case, model, mode = key
        lines.append(
            f"| {case} | {model} | {mode} | {len(group)} | "
            f"{_format_share(counts['n_strings_correct'], counts['n_strings'])} | "
            f"{_format_share(counts['n_justifications_correct'], counts['n_justifications'])} | "
            f"{_format_share(counts['n_fields_correct'], counts['n_fields'])} | "
            f"{sum(j['usage']['requests'] for j in group)} | "
            f"{sum(float(j['usage']['cost_total']) for j in group):.4f} |"
        )
    return lines


def _get_cost_lines(label: str) -> list[str]:
    """
    Returns the section of the summary on what the requests of the experiment
    cost: the runs, the judge and the passes of ``decision_probabilities``, and
    the total.

    :param label: The label of the experiment.
    :type label: str
    :return: The lines of the section.
    :rtype: list[str]
    """
    costs = get_costs(label)
    lines = [
        "",
        "## Cost",
        "",
        "| Part | Files | Requests | Cost (USD) |",
        "|---|---|---|---|",
    ]
    for part, name in PART_NAMES.items():
        cost = costs[part]
        if not cost.files:
            continue
        recorded = cost.files - cost.files_without_cost
        lines.append(
            f"| {name} | {cost.files} | "
            f"{f'{cost.requests:,}' if recorded else 'n/a'} | "
            f"{f'{cost.cost:.4f}' if recorded else 'not recorded'} |"
        )
    lines.append(f"| Total | | | {get_total(costs):.4f} |")
    n_without_cost = sum(cost.files_without_cost for cost in costs.values())
    if n_without_cost:
        lines += [
            "",
            (
                f"{n_without_cost} files record no cost (the passes of "
                "`decision_probabilities` recorded before 2026-10-15): the total "
                "leaves them out."
            ),
        ]
    return lines


def summarize(label: str) -> tuple[str, list[dict[str, Any]]]:
    """
    Summarizes the runs of an experiment.

    :param label: The label of the experiment.
    :type label: str
    :return: The summary as Markdown, and the rows of its table as data: for each
        case, model and mode, the scores and the usage of each run.
    :rtype: tuple[str, list[dict[str, Any]]]
    :raises ValueError: If the experiment has no runs.
    """
    paths = sorted((RESULTS_DIR / label / "runs").glob("*.json"))
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError(f"No runs are recorded for the label `{label}`.")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["case"], run["model"], run["mode"])].append(run)
    # The verdicts of the judge on each run that has them
    judgment_paths = [RESULTS_DIR / label / "judgments" / path.name for path in paths]
    judgments: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run, judgment_path in zip(runs, judgment_paths, strict=True):
        if judgment_path.exists():
            judgments[(run["case"], run["model"], run["mode"])].append(
                json.loads(judgment_path.read_text(encoding="utf-8"))
            )

    lines = [
        f"# Extraction evaluation: {label}",
        "",
        (
            "Decision models extract only the closed concepts (Boolean, label and rating "
            "concepts), without justifications and references: compare them with LLMs "
            "on the aspects and on the closed concepts."
        ),
        "",
        (
            "| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | "
            "Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | "
            "Output tokens | Seconds | Retries |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows: list[dict[str, Any]] = []
    for (case, model, mode), group in sorted(groups.items()):
        summaries = [run["scores"]["summary"] for run in group]
        usages = [run["usage"] for run in group]
        n_concepts_scored = summaries[0]["n_concepts_scored"]
        closed_counts = [_count_closed_concepts(run) for run in group]
        row: dict[str, Any] = {
            "case": case,
            "model": model,
            "mode": mode,
            "runs": len(group),
            "aspect_f1": [summary["aspect_f1"] for summary in summaries],
            "aspect_precision": [summary["aspect_precision"] for summary in summaries],
            "aspect_recall": [summary["aspect_recall"] for summary in summaries],
            "concepts_correct": [
                summary["n_concepts_correct"] for summary in summaries
            ],
            "n_concepts_scored": n_concepts_scored,
            "closed_concepts_correct": [correct for correct, _ in closed_counts],
            "n_closed_concepts": closed_counts[0][1],
            "reference_f1": [summary["reference_f1"] for summary in summaries],
            "requests": [usage["requests"] for usage in usages],
            "input_tokens": [usage["input_tokens"] for usage in usages],
            "cache_read_tokens": [usage["cache_read_tokens"] for usage in usages],
            "output_tokens": [usage["output_tokens"] for usage in usages],
            "cost_total": [usage["cost_total"] for usage in usages],
            "seconds": [run["seconds"] for run in group],
            "retries": [run["retries_for_invalid_data"] for run in group],
            "errors": [run["error"] for run in group if run["error"]],
            "warnings": sum(len(run["extraction_warnings"]) for run in group),
        }
        if (case, model, mode) in judgments:
            row["judged"] = [j["summary"] for j in judgments[(case, model, mode)]]
        rows.append(row)
        lines.append(
            f"| {case} | {model} | {mode} | {len(group)} | "
            f"{_format_spread(row['aspect_f1'])} | "
            f"{_format_correct(row['concepts_correct'], n_concepts_scored)} | "
            f"{_format_correct(row['closed_concepts_correct'], row['n_closed_concepts'])} | "
            f"{_format_spread(row['reference_f1'])} | "
            f"{_format_count(row['requests'])} | "
            f"{_format_count(row['input_tokens'])} | "
            f"{_format_count(row['cache_read_tokens'])} | "
            f"{_format_count(row['output_tokens'])} | "
            f"{_format_count(row['seconds'])} | {sum(row['retries'])} |"
        )

    if judgments:
        lines += _get_judged_lines(groups, judgments)

    lines += _get_cost_lines(label)

    lines += ["", "## Concepts missed", ""]
    for (case, model, mode), group in sorted(groups.items()):
        # The string concepts are left out: they are not scored without a judge
        missed: dict[str, int] = defaultdict(int)
        for run in group:
            for concept in run["scores"]["concepts"]:
                if not concept["correct"] and concept["scoring"] != "exact_or_accepted":
                    missed[f"{concept['name']} ({concept['type']})"] += 1
        if missed:
            lines.append(
                f"- **{case}, {model}, {mode}** ({len(group)} runs): "
                + "; ".join(
                    f"{name}: {n}x"
                    for name, n in sorted(missed.items(), key=lambda item: -item[1])
                )
            )

    lines += ["", "## Aspects (mean F1, precision and recall of each aspect)", ""]
    for (case, model, mode), group in sorted(groups.items()):
        scores_by_aspect: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for run in group:
            for aspect in run["scores"]["aspects"]:
                scores_by_aspect[aspect["name"]].append(aspect)
        lines.append(
            f"- **{case}, {model}, {mode}**: "
            + "; ".join(
                f"{name}: F1 {statistics.mean(s['f1'] for s in scores):.2f} "
                f"(P {statistics.mean(s['precision'] for s in scores):.2f}, "
                f"R {statistics.mean(s['recall'] for s in scores):.2f})"
                for name, scores in scores_by_aspect.items()
            )
        )

    problems = [
        f"- {run['case']}, {run['model']}, {run['mode']}, run {run['run']}: "
        f"error {run['error']}, {len(run['extraction_warnings'])} warning(s)"
        for run in runs
        if run["error"] or run["extraction_warnings"]
    ]
    lines += ["", "## Errors and warnings", "", *(problems or ["- none"])]
    return "\n".join(lines) + "\n", rows


def main() -> None:
    """
    Writes and prints the summary of an experiment.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Summarizes the runs of an experiment recorded by run_eval."
    )
    parser.add_argument("--label", required=True, help="name of the experiment")
    args = parser.parse_args()

    summary, rows = summarize(args.label)
    (RESULTS_DIR / args.label / "summary.md").write_text(
        summary, encoding="utf-8", newline="\n"
    )
    write_json(RESULTS_DIR / args.label / "summary.json", rows)
    print(summary)


if __name__ == "__main__":
    main()
