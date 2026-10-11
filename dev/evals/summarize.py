#
# ContextGem
#
# Copyright 2026 Shcherbak AI AS. All rights reserved. Developed by Sergii Shcherbak.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Summarizes the runs of an experiment, recorded by ``run_eval``: the runs of
the same case, model and mode are put together, with the mean and the range of
each score over them. Reads the files of the runs only, i.e. sends no request.

Run it from the root of the repo:

```bash
uv run python -m dev.evals.summarize --label my_experiment
```

Writes ``results/<label>/summary.md`` and ``results/<label>/summary.json``, and
prints the summary.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any

from dev.evals.harness import RESULTS_DIR, write_json


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
    runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((RESULTS_DIR / label / "runs").glob("*.json"))
    ]
    if not runs:
        raise ValueError(f"No runs are recorded for the label `{label}`.")
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[(run["case"], run["model"], run["mode"])].append(run)

    lines = [
        f"# Extraction evaluation: {label}",
        "",
        "| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Reference F1 | "
        "Requests | Input tokens | Cached | Output tokens | Seconds | Retries |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows: list[dict[str, Any]] = []
    for (case, model, mode), group in sorted(groups.items()):
        summaries = [run["scores"]["summary"] for run in group]
        usages = [run["usage"] for run in group]
        n_concepts_scored = summaries[0]["n_concepts_scored"]
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
        rows.append(row)
        correct = row["concepts_correct"]
        correct_text = (
            f"{correct[0]}/{n_concepts_scored}"
            if min(correct) == max(correct)
            else f"{statistics.mean(correct):.1f}/{n_concepts_scored} "
            f"({min(correct)}-{max(correct)})"
        )
        lines.append(
            f"| {case} | {model} | {mode} | {len(group)} | "
            f"{_format_spread(row['aspect_f1'])} | {correct_text} | "
            f"{_format_spread(row['reference_f1'])} | "
            f"{_format_count(row['requests'])} | "
            f"{_format_count(row['input_tokens'])} | "
            f"{_format_count(row['cache_read_tokens'])} | "
            f"{_format_count(row['output_tokens'])} | "
            f"{_format_count(row['seconds'])} | {sum(row['retries'])} |"
        )

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
