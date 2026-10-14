# SPDX-FileCopyrightText: Shcherbak AI AS
# SPDX-License-Identifier: Apache-2.0

"""
Scores the recorded runs of an experiment again, against the reference answers as
they are now. For use after a change in the reference answers of a case, or in
the paragraphs that are not scored: no request is sent.

A run is scored from what it keeps of its extraction: the paragraphs extracted
for each aspect, and the values and the references extracted for each concept.
Only the scores of a run are rewritten.

Run it from the root of the repo, and summarize the experiment afterwards:

```bash
uv run python -m dev.evals.rescore --label my_experiment
uv run python -m dev.evals.summarize --label my_experiment
```
"""

from __future__ import annotations

import argparse
import json

from dev.evals.harness import RESULTS_DIR, load_case, write_json
from dev.evals.scoring import rescore


def rescore_runs(label: str) -> tuple[int, int]:
    """
    Scores the runs of an experiment again, and rewrites the scores of those
    whose scores change.

    :param label: The label of the experiment.
    :type label: str
    :return: The number of the runs of the experiment, and of those whose scores
        changed.
    :rtype: tuple[int, int]
    :raises ValueError: If the experiment has no runs, or if a run cannot be
        scored again (see :func:`dev.evals.scoring.rescore`).
    """
    paths = sorted((RESULTS_DIR / label / "runs").glob("*.json"))
    if not paths:
        raise ValueError(f"No runs are recorded for the label `{label}`.")
    n_changed = 0
    for path in paths:
        run = json.loads(path.read_text(encoding="utf-8"))
        _, gold = load_case(run["case"], for_decision_model=run["kind"] == "decision")
        scores = rescore(run["scores"], gold)
        if scores != run["scores"]:
            before = run["scores"]["summary"]
            after = scores["summary"]
            print(
                f"{path.name}: aspect F1 {before['aspect_f1']:.3f} -> "
                f"{after['aspect_f1']:.3f}, concepts "
                f"{before['n_concepts_correct']} -> {after['n_concepts_correct']} "
                f"of {after['n_concepts_scored']}",
                flush=True,
            )
            run["scores"] = scores
            write_json(path, run)
            n_changed += 1
    return len(paths), n_changed


def main() -> None:
    """
    Scores the runs of an experiment again, and prints what changed.

    :return: None
    :rtype: None
    """
    parser = argparse.ArgumentParser(
        description="Scores the recorded runs of an experiment again."
    )
    parser.add_argument("--label", required=True, help="name of the experiment")
    args = parser.parse_args()

    n_runs, n_changed = rescore_runs(args.label)
    print(f"{n_runs} runs scored again, the scores of {n_changed} changed")


if __name__ == "__main__":
    main()
