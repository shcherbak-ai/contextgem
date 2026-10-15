# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Adds up what the requests of the experiments cost, from the costs that their
results record: the runs (``run_eval``), the judgments and the checks of the judge
(``judge``), and the passes of ``decision_probabilities``. Reads the results only,
i.e. sends no request.

Run it from the root of the repo:

```bash
# Every experiment
uv run python -m dev.evals.costs

# One experiment
uv run python -m dev.evals.costs --label my_experiment
```

The passes of ``decision_probabilities`` recorded before 2026-10-15 have no cost:
they are counted, and the totals leave them out. The summary of an experiment
has its costs too (``summarize``).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from decimal import Decimal

from dev.evals.harness import REPO_DIR, RESULTS_DIR


# The parts of an experiment that send requests, by what the costs name them
PART_NAMES = {
    "runs": "Runs",
    "judge": "Judge",
    "probabilities": "Decision probabilities",
}


@dataclass
class PartCost:
    """
    What the requests of a part of an experiment cost.

    :ivar files: The number of result files of the part.
    :vartype files: int
    :ivar files_without_cost: The number of those files that record no cost,
        which the cost leaves out.
    :vartype files_without_cost: int
    :ivar requests: The number of requests, in the files that record a cost.
    :vartype requests: int
    :ivar cost: The cost in USD.
    :vartype cost: Decimal
    """

    files: int = 0
    files_without_cost: int = 0
    requests: int = 0
    cost: Decimal = field(default_factory=Decimal)


def get_costs(label: str) -> dict[str, PartCost]:
    """
    Adds up what the requests of an experiment cost, for each part of it.

    :param label: The label of the experiment.
    :type label: str
    :return: The cost of each part (see ``PART_NAMES``), including the parts
        without files.
    :rtype: dict[str, PartCost]
    """
    label_dir = RESULTS_DIR / label
    judge_checks = label_dir / "judge_checks.json"
    paths = {
        "runs": sorted((label_dir / "runs").glob("*.json")),
        "judge": [
            *sorted((label_dir / "judgments").glob("*.json")),
            *([judge_checks] if judge_checks.exists() else []),
        ],
        "probabilities": sorted((label_dir / "decision_probabilities").glob("*.json")),
    }
    costs = {part: PartCost() for part in PART_NAMES}
    for part, part_paths in paths.items():
        cost = costs[part]
        for path in part_paths:
            usage = json.loads(path.read_text(encoding="utf-8")).get("usage") or {}
            cost.files += 1
            if "cost_total" not in usage:
                cost.files_without_cost += 1
                continue
            cost.requests += usage["requests"]
            cost.cost += Decimal(usage["cost_total"])
    return costs


def get_total(costs: dict[str, PartCost]) -> Decimal:
    """
    Returns the cost of all the parts of an experiment.

    :param costs: The cost of each part.
    :type costs: dict[str, PartCost]
    :return: The cost in USD, without the files that record no cost.
    :rtype: Decimal
    """
    return sum((cost.cost for cost in costs.values()), Decimal(0))


def _format_cost(cost: PartCost) -> str:
    """
    Formats the cost of a part of an experiment for the table of the experiments.

    :param cost: The cost of the part.
    :type cost: PartCost
    :return: The cost, "-" for a part without files, or "not recorded" for
        a part whose files record no cost, with the number of such files where
        only some do.
    :rtype: str
    """
    if not cost.files:
        return "-"
    if cost.files_without_cost == cost.files:
        return "not recorded"
    if cost.files_without_cost:
        return f"{cost.cost:.4f} (+{cost.files_without_cost} not recorded)"
    return f"{cost.cost:.4f}"


def main() -> None:
    """
    Prints what the requests of the experiments cost.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Adds up the costs recorded in the results of the experiments. "
        "Sends no request."
    )
    parser.add_argument(
        "--label", default=None, help="name of the experiment (default: all of them)"
    )
    args = parser.parse_args()
    if args.label and not (RESULTS_DIR / args.label).is_dir():
        parser.error(f"no results are recorded for the label `{args.label}`")

    labels = (
        [args.label]
        if args.label
        else sorted(path.name for path in RESULTS_DIR.iterdir() if path.is_dir())
    )
    rows = [["Experiment", *PART_NAMES.values(), "Total"]]
    all_total = Decimal(0)
    has_not_recorded = False
    for label in labels:
        costs = get_costs(label)
        total = get_total(costs)
        all_total += total
        has_not_recorded |= any(cost.files_without_cost for cost in costs.values())
        rows.append(
            [label, *(_format_cost(costs[part]) for part in PART_NAMES), f"{total:.4f}"]
        )
    if len(labels) > 1:
        rows.append(["All", *("" for _ in PART_NAMES), f"{all_total:.4f}"])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]

    results_dir = RESULTS_DIR.relative_to(REPO_DIR).as_posix()
    print(f"Costs recorded in the results of the experiments ({results_dir}/), USD:")
    print()
    for row in rows:
        print(
            "  ".join(
                cell.ljust(width) if i == 0 else cell.rjust(width)
                for i, (cell, width) in enumerate(zip(row, widths, strict=True))
            ).rstrip()
        )
    if has_not_recorded:
        print()
        print(
            "Not recorded: result files without a cost, such as the passes of "
            "decision_probabilities recorded before 2026-10-15. The totals leave "
            "them out."
        )


if __name__ == "__main__":
    main()
