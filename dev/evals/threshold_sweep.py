# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Sweeps the decision threshold over the probabilities recorded by
``decision_probabilities``, and shows how the probabilities of each model relate
to the reference answers. Reads the recorded files only, i.e. sends no request.

A paragraph is extracted for an aspect if its probability is the threshold or
more, as in the library, and a refused answer leaves the paragraph out. The scores
are those of ``run_eval``: the precision, recall and F1 of each aspect, averaged
over the aspects of a case, and then over the cases. The paragraphs that are
neutral for an aspect are not scored and not counted.

Run it from the root of the repo:

```bash
uv run python -m dev.evals.threshold_sweep --label my_experiment
```

Writes ``results/<label>/thresholds.md`` and ``results/<label>/thresholds.json``,
and prints the report.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any

from dev.evals.harness import RESULTS_DIR, load_case, write_json
from dev.evals.scoring import get_set_scores


# The thresholds swept, and those of them shown in the table
_THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]
_THRESHOLDS_SHOWN = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# The default decision threshold of a decision model
_DEFAULT_THRESHOLD = 0.5
# How close to the best F1 a threshold must be to count as about as good
_NEAR_BEST_MARGIN = 0.01
# The ranges of probability that the reference answers are counted in: from
# the lower bound, up to the upper bound exclusive
_PROBABILITY_RANGES = [
    (0.0, 0.1),
    (0.1, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.0001),
]

# Precision, recall and F1
_Scores = tuple[float, float, float]


def _get_mean_scores(scores: list[_Scores]) -> _Scores:
    """
    Averages the precision, recall and F1 of several aspects, or of several cases.

    :param scores: The scores to average.
    :type scores: list[tuple[float, float, float]]
    :return: The mean precision, recall and F1.
    :rtype: tuple[float, float, float]
    """
    n = len(scores)
    return (
        sum(score[0] for score in scores) / n,
        sum(score[1] for score in scores) / n,
        sum(score[2] for score in scores) / n,
    )


def get_case_scores(
    probabilities: list[list[float | None]],
    gold_positions: list[set[int]],
    neutral_positions: list[set[int]],
    threshold: float,
) -> _Scores:
    """
    Scores what a threshold extracts for the aspects of a case.

    :param probabilities: For each paragraph, the probability for each aspect, or
        None for a refused answer.
    :type probabilities: list[list[float | None]]
    :param gold_positions: For each aspect, the positions of its paragraphs in
        the reference answers, without the neutral ones.
    :type gold_positions: list[set[int]]
    :param neutral_positions: For each aspect, the positions of the paragraphs
        that are not scored for it.
    :type neutral_positions: list[set[int]]
    :param threshold: The decision threshold.
    :type threshold: float
    :return: The precision, recall and F1, averaged over the aspects.
    :rtype: tuple[float, float, float]
    """
    scores: list[_Scores] = []
    for aspect_position, expected in enumerate(gold_positions):
        predicted = {
            position
            for position, row in enumerate(probabilities)
            if position not in neutral_positions[aspect_position]
            and (probability := row[aspect_position]) is not None
            and probability >= threshold
        }
        aspect_scores = get_set_scores(predicted, expected)
        scores.append(
            (aspect_scores["precision"], aspect_scores["recall"], aspect_scores["f1"])
        )
    return _get_mean_scores(scores)


def _load_recorded(label: str) -> dict[str, dict[str, dict[str, Any]]]:
    """
    Loads the probabilities recorded for an experiment, each with the paragraphs
    of the aspects of its case in the reference answers.

    :param label: The label of the experiment.
    :type label: str
    :return: By the model and by the case: the recorded data, with the positions
        of the paragraphs of each aspect in the reference answers, without
        the neutral ones, added as ``gold_positions``, and the positions of
        the paragraphs that are not scored for each aspect as ``neutral_positions``.
    :rtype: dict[str, dict[str, dict[str, Any]]]
    :raises ValueError: If nothing is recorded, or if the aspects recorded are not
        those of the case.
    """
    paths = sorted((RESULTS_DIR / label / "decision_probabilities").glob("*.json"))
    if not paths:
        raise ValueError(f"No probabilities are recorded for the label `{label}`.")
    gold_of_cases: dict[str, tuple[list[str], list[set[int]], list[set[int]]]] = {}
    recorded: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        case = data["case"]
        if case not in gold_of_cases:
            document, gold = load_case(case, for_decision_model=True)
            neutral_positions = [
                set(aspect["neutral_paragraphs"]) for aspect in gold["aspects"]
            ]
            gold_of_cases[case] = (
                [aspect.name for aspect in document.aspects],
                [
                    set(aspect["reference_paragraphs"]) - neutral
                    for aspect, neutral in zip(
                        gold["aspects"], neutral_positions, strict=True
                    )
                ],
                neutral_positions,
            )
        aspect_names, data["gold_positions"], data["neutral_positions"] = gold_of_cases[
            case
        ]
        if data["aspects"] != aspect_names:
            raise ValueError(
                f"The aspects recorded in `{path.name}` are not those of the case."
            )
        recorded[data["model"]][case] = data
    return recorded


def sweep(label: str) -> tuple[str, dict[str, Any]]:
    """
    Sweeps the decision threshold over the probabilities recorded for
    an experiment.

    :param label: The label of the experiment.
    :type label: str
    :return: The report as Markdown, and its data: the scores of each model at
        each threshold, over the cases and for each case, and how the probabilities
        of each model relate to the reference answers.
    :rtype: tuple[str, dict[str, Any]]
    """
    recorded = _load_recorded(label)
    cases = sorted({case for model_cases in recorded.values() for case in model_cases})
    report: dict[str, Any] = {"models": {}, "calibration": {}}
    lines = [
        f"# Decision threshold sweep: {label}",
        "",
        "## Aspect F1 by threshold",
        "",
        "Mean of the cases. The second line of a model has the precision and "
        "the recall. Only the models recorded on each case are shown.",
        "",
        "| Model | "
        + " | ".join(f"{threshold:.1f}" for threshold in _THRESHOLDS_SHOWN)
        + " | Best |",
        "|---|" + "---|" * (len(_THRESHOLDS_SHOWN) + 1),
    ]
    f1_of_models: dict[float, list[float]] = defaultdict(list)
    for model, model_cases in recorded.items():
        if set(model_cases) != set(cases):
            continue
        scores_by_case = {
            case: {
                threshold: get_case_scores(
                    data["probabilities"],
                    data["gold_positions"],
                    data["neutral_positions"],
                    threshold,
                )
                for threshold in _THRESHOLDS
            }
            for case, data in model_cases.items()
        }
        scores = {
            threshold: _get_mean_scores(
                [scores_by_case[case][threshold] for case in cases]
            )
            for threshold in _THRESHOLDS
        }
        f1_by_threshold = {threshold: scores[threshold][2] for threshold in _THRESHOLDS}
        for threshold in _THRESHOLDS:
            f1_of_models[threshold].append(f1_by_threshold[threshold])
        best = max(_THRESHOLDS, key=f1_by_threshold.__getitem__)
        near_best = [
            threshold
            for threshold in _THRESHOLDS
            if scores[best][2] - scores[threshold][2] <= _NEAR_BEST_MARGIN
        ]
        lines.append(
            f"| {model} | "
            + " | ".join(
                f"{scores[threshold][2]:.3f}" for threshold in _THRESHOLDS_SHOWN
            )
            + f" | {scores[best][2]:.3f} at {best:.2f} (within "
            f"{_NEAR_BEST_MARGIN}: {min(near_best):.2f}-{max(near_best):.2f}) |"
        )
        lines.append(
            "| | "
            + " | ".join(
                f"{scores[threshold][0]:.2f} / {scores[threshold][1]:.2f}"
                for threshold in _THRESHOLDS_SHOWN
            )
            + " | |"
        )
        report["models"][model] = {
            "best_threshold": best,
            "f1_at_default": scores[_DEFAULT_THRESHOLD][2],
            "f1_at_best": scores[best][2],
            "scores": {str(threshold): scores[threshold] for threshold in _THRESHOLDS},
            "scores_by_case": {
                case: {
                    str(threshold): scores_by_case[case][threshold]
                    for threshold in _THRESHOLDS
                }
                for case in cases
            },
        }
    if f1_of_models:
        mean_f1 = {
            threshold: sum(values) / len(values)
            for threshold, values in f1_of_models.items()
        }
        best = max(_THRESHOLDS, key=mean_f1.__getitem__)
        lines.append(
            "| **Mean of the models** | "
            + " | ".join(f"{mean_f1[threshold]:.3f}" for threshold in _THRESHOLDS_SHOWN)
            + f" | {mean_f1[best]:.3f} at {best:.2f} |"
        )
        report["mean_of_models"] = {
            str(threshold): mean_f1[threshold] for threshold in _THRESHOLDS
        }
        report["best_common_threshold"] = best

    lines += [
        "",
        "## Best threshold of each case",
        "",
        f"With the F1 at it, and the F1 at {_DEFAULT_THRESHOLD}, in brackets.",
        "",
    ]
    for model, model_cases in recorded.items():
        parts = []
        for case, data in sorted(model_cases.items()):
            f1_by_threshold = {
                threshold: get_case_scores(
                    data["probabilities"],
                    data["gold_positions"],
                    data["neutral_positions"],
                    threshold,
                )[2]
                for threshold in _THRESHOLDS
            }
            best = max(_THRESHOLDS, key=f1_by_threshold.__getitem__)
            parts.append(
                f"{case}: {best:.2f} ({f1_by_threshold[best]:.3f}, "
                f"{f1_by_threshold[_DEFAULT_THRESHOLD]:.3f})"
            )
        lines.append(f"- **{model}**: " + "; ".join(parts))

    lines += [
        "",
        "## How the probabilities relate to the reference answers",
        "",
        "The share of the (paragraph, aspect) pairs in each range of probability "
        "that are in the reference answers, with the number of the pairs in "
        "brackets. All the cases recorded for a model together, without the pairs "
        "that are not scored. For the refused answers, the number of the pairs "
        "that are in the reference answers.",
        "",
        "| Model | "
        + " | ".join(
            f"{low:.1f}-{min(high, 1.0):.1f}" for low, high in _PROBABILITY_RANGES
        )
        + " | Refused |",
        "|---|" + "---|" * (len(_PROBABILITY_RANGES) + 1),
    ]
    for model, model_cases in recorded.items():
        # The number of the pairs, and of those in the reference answers
        counts = [[0, 0] for _ in _PROBABILITY_RANGES]
        refused = [0, 0]
        for data in model_cases.values():
            for aspect_position, expected in enumerate(data["gold_positions"]):
                neutral = data["neutral_positions"][aspect_position]
                for position, row in enumerate(data["probabilities"]):
                    if position in neutral:
                        continue
                    probability = row[aspect_position]
                    if probability is None:
                        count = refused
                    else:
                        count = next(
                            counts[i]
                            for i, (low, high) in enumerate(_PROBABILITY_RANGES)
                            if low <= probability < high
                        )
                    count[0] += 1
                    count[1] += position in expected
        lines.append(
            f"| {model} | "
            + " | ".join(
                f"{n_gold / n:.0%} ({n})" if n else "n/a" for n, n_gold in counts
            )
            + f" | {refused[1]} of {refused[0]} |"
        )
        report["calibration"][model] = {"ranges": counts, "refused": refused}
    return "\n".join(lines) + "\n", report


def main() -> None:
    """
    Writes and prints the report on the thresholds of an experiment.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Sweeps the decision threshold over the recorded probabilities."
    )
    parser.add_argument("--label", required=True, help="name of the experiment")
    args = parser.parse_args()

    text, report = sweep(args.label)
    (RESULTS_DIR / args.label / "thresholds.md").write_text(
        text, encoding="utf-8", newline="\n"
    )
    write_json(RESULTS_DIR / args.label / "thresholds.json", report)
    print(text)


if __name__ == "__main__":
    main()
