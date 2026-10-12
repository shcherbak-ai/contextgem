# dev/evals - Extraction Evaluation Harness

Measures the quality and the cost of extraction on evaluation cases with reference answers, so that a change (a prompt, a model, a setting) can be compared with a recorded baseline. It is a development tool: it is not a part of the package, and the tests do not run it.

## Layout

| Path | Role |
|------|------|
| `cases/<case>.txt` | Text of the document of a case |
| `cases/<case>.py` | Extraction pipeline of the case (`build_<case>_pipeline()`) |
| `cases/<case>.gold.json` | Reference answers of the case: paragraphs of each aspect, items of each concept |
| `cases/<case>.neutral.json` | Paragraphs that are not scored for an aspect: section headings, and borderline and linked paragraphs with the reasons |
| `cases/reference_review.md` | Record of the review of the reference answers: method, rulings and limits |
| `harness.py` | Loads a case, creates the model, writes results |
| `scoring.py` | Scores an extracted document against the reference answers |
| `run_eval.py` | Extracts a case with a model and scores it (**live requests**) |
| `summarize.py` | Summary of the runs of a label (offline) |
| `rescore.py` | Scores the recorded runs of a label again, after a change in the reference answers (offline) |
| `decision_probabilities.py` | Probabilities of a decision model for the aspects of a case (**live requests**) |
| `threshold_sweep.py` | Decision threshold sweep over the recorded probabilities (offline) |
| `check_scorer.py` | Self-check of the scorer on each case (offline) |
| `results/<label>/` | Results of one experiment: `runs/`, `decision_probabilities/`, and the generated `summary.*` and `thresholds.*` |

## Rules

- **`run_eval` and `decision_probabilities` send live, paid requests.** Run them only when the maintainer asks for it. The other scripts send nothing.
- **Keys come from the environment or the `.env` file of the repo** (`CONTEXTGEM_OPENAI_API_KEY`, `CONTEXTGEM_TYPESAFE_API_KEY`, `CONTEXTGEM_AZURE_AI_API_KEY` with `CONTEXTGEM_AZURE_AI_API_BASE`). Never print them. The host of a Microsoft Foundry resource is replaced in what is written.
- **A label is one experiment.** A run whose file exists is skipped, so results are never overwritten: record a change under a new label, and leave `baseline` as it is.
- **Results are tracked artifacts.** Do not edit or delete the files under `results/`. `summary.*` and `thresholds.*` are generated: regenerate them, do not edit them. The scores of a run are rewritten only by `rescore`.
- **The reference answers change only by a recorded review.** Do not change a `.gold.json` or `.neutral.json` file to make a result pass. Record a change with its reason in `cases/reference_review.md`, and run `check_scorer` after a change in a case, in its reference answers or in the scorer. After a change in the reference answers, score the recorded runs again with `rescore`, and regenerate the summaries.

## The Reference Answers

The `.gold.json` files of both cases were produced by an AI agent, and reviewed on 2026-10-11 by AI agents: a second, blind annotation of each case, and a ruling on every disagreement against the text (see `cases/reference_review.md`). No person has verified them, whatever the `reviewed` flags in the files say. So a score is the agreement with these answers, not a proof of correctness:

- **Use the scores to compare**: one prompt, model or setting with another, on the same reference.
- **Do not read a score as accuracy.** An extraction that differs from the reference is not necessarily wrong, and one that agrees with it is not necessarily right.
- **Before concluding that a change is better or worse, read the disagreements** that make the difference, and judge them against the text of the case. Each run keeps the paragraphs extracted for each aspect and the values of each concept.
- **Some paragraphs are not scored for an aspect** (`cases/<case>.neutral.json`): the headings of the sections that the aspect has paragraphs in, the paragraphs that careful readers disagree on, and linked context, i.e. a paragraph that the aspect depends on or that refers to it without being a part of it. Extracting one, or leaving it out, changes no score.
- **A decision model judges a paragraph on its own**, while the reference lists what belongs in the context of the document. Part of the difference between the two is this.

## Commands

Run from the root of the repo:

```bash
# Extract a case with an LLM, 3 times, and score each run (live)
uv run python -m dev.evals.run_eval --case complex_nda --model openai/gpt-6-luna --runs 3 --label my_experiment

# The same with a decision model, and with a model of a local server (live)
uv run python -m dev.evals.run_eval --case complex_nda --kind decision --model typesafe/jev-latest --label my_experiment
uv run python -m dev.evals.run_eval --case complex_nda --kind decision --model strands_decider/nimble --api-base http://localhost:11434 --label my_experiment

# Summary of the runs of a label (offline)
uv run python -m dev.evals.summarize --label my_experiment

# Score the recorded runs again after a change in the reference answers (offline)
uv run python -m dev.evals.rescore --label my_experiment

# Probabilities of a decision model (live), then the threshold sweep (offline)
uv run python -m dev.evals.decision_probabilities --case complex_nda --model typesafe/jev-latest --label my_experiment
uv run python -m dev.evals.threshold_sweep --label my_experiment

# Self-check of the scorer (offline)
uv run python -m dev.evals.check_scorer
```

`--concurrency` runs the extraction with `use_concurrency=True`. One line is printed per run, which starts with the name of its file. The library prints its logs as usual: set `CONTEXTGEM_LOGGER_LEVEL=WARNING` for a quiet run.

## What Is Measured

- **Aspects**: precision, recall and F1 of the extracted paragraphs against those of the reference answers, averaged over the aspects. The neutral paragraphs of an aspect are left out of both.
- **Concepts**: date, numerical, Boolean, label and rating concepts by exact match of the values; JSON object concepts by the names of the objects. String concepts are counted apart, as an exact or accepted value only: a narrative answer needs a judge.
- **References of concepts**: F1 of the referenced paragraphs.
- **Cost**: requests, input, cached and output tokens, seconds, retries for invalid data.
- **Decision models** extract all the aspects, and of the concepts only the Boolean, label and rating ones, without justifications and references. Their concept counts are not comparable with those of the LLMs.

Each run records a short hash of every prompt template of the library, including the template of the default system message (`provenance`), which tells the runs of a changed prompt apart. The runs of the baseline were recorded before the system message template was hashed, so they have no hash of it.

## Comparing a Change with the Baseline

1. Make the change in the working tree, e.g. in a prompt template.
2. Run the same cases, models and modes as the baseline under a new label, at least 3 runs each for an LLM.
3. Summarize the label, and compare its `summary.md` with `results/baseline/summary.md`.

Runs of one configuration differ: in the baseline, the aspect F1 of a hosted LLM varies by up to 0.05 between runs. Compare the means of several runs, and treat a smaller difference as noise.

The hosted LLMs of the baseline are at or near the top of the scale on both cases, for aspects and for concepts. So the cases show whether a change makes extraction worse, what it costs, and how weaker models and decision models differ; they have little room left to show that a change makes a capable model better.

## The Baseline

Recorded on 2026-10-11 with `litellm` 1.105.0, after the first part of the review of the reference answers, and scored again after its second part:

- **LLMs** (`default` mode unless stated): `openai/gpt-6-luna` (3 runs, and 3 with concurrency), `openai/gpt-5.6-luna` (3), `openai/gpt-6.1-sol` (1), and on Ollama with a context length of 16k, `ollama_chat/mistral-small:24b` (2) and `ollama_chat/llama3.1:8b` (2, which failed on invalid JSON).
- **Decision models**: `typesafe/jev-latest` (2), `openai/gpt-6-luna` (2), `azure_ai/ms-decision-model`, a deployment of Microsoft-Decision-1 in Microsoft Foundry (2), `strands_decider/nimble` on Ollama (1) and `strands_decider/kev-4b` on LM Studio (1).
- **Probabilities of the decision models** (`decision_probabilities/`): recorded earlier the same day, one pass per model and case. They do not depend on the reference answers, so they were kept; they have no `provenance`. Those of `openai/gpt-6-luna` were recorded with `litellm` 1.104.2.

## Tried and Rejected

Experiments that changed nothing in the library, recorded so that they are not run again without a new reason.

### Wording of the aspect question of the decision models (2026-10-12)

Three wordings of `decision_question_aspect.j2`, each with one change, were compared with the current one on the five decision models of the baseline and both cases, in one pass of `decision_probabilities` each:

- **Exclusion**: the sentence on "no" ends with "including when it is about another subject and only mentions the aspect."
- **Narrower criterion**: "belongs to the aspect" and "is a part of what the aspect covers, as its name and description define it", in place of "is relevant to" and "addresses or clearly relates to".
- **Question**: "Does the paragraph address the aspect above or clearly relate to it?" at the end, in place of the instruction to decide and the criteria for "yes" and "no".

Aspect F1 at the threshold of 0.5, mean of the two cases. The current wording was run again for this, at the same time as the others:

| Model | Current | Exclusion | Narrower criterion | Question |
|-------|---------|-----------|--------------------|----------|
| `openai/gpt-6-luna` | 0.849 | 0.849 | 0.850 | 0.733 |
| `typesafe/jev-latest` | 0.815 | 0.808 | 0.777 | 0.773 |
| `azure_ai/ms-decision-model` | 0.772 | 0.797 | 0.726 | 0.832 |
| `strands_decider/kev-4b` | 0.747 | 0.744 | 0.769 | 0.699 |
| `strands_decider/nimble` | 0.695 | 0.682 | 0.675 | 0.688 |
| Mean | 0.776 | 0.776 | 0.759 | 0.745 |
| Mean, at the best threshold of each model | 0.820 | 0.814 | 0.810 | 0.780 |

No wording is better across the models:

- **A wording moves the probabilities of each model in its own direction**, e.g. the question form gains 0.06 for the model in Microsoft Foundry and loses 0.12 for `openai/gpt-6-luna`. The decision threshold of a model makes such a shift for that model only.
- **No wording tells the paragraphs of the reference answers from the others better**: the mean at the best threshold of each model is not higher for any of the three.
- **The paragraphs that four or five models extract beyond the reference answers stay**: of the 35 answers behind these seven paragraph-aspect pairs, 31 are "yes" with the current wording, and 30, 31 and 26 with the three others.

The differences are not the variation between passes: with the current wording, three of the models give the same answers as in the baseline, and the other two differ in about 10 of 1,206 answers, i.e. by 0.003 or less in F1.

Not tested: the questions of concepts (each case has six concepts that a decision model extracts, too few to compare wordings), and aspects extracted with sentences or with additional context. The recorded probabilities were not kept: the numbers above are the record.

## Adding a Case

Add the `.txt`, `.py` and `.gold.json` files of the case to `cases/`, and a `.neutral.json` file if the case has paragraphs that are not scored, register its pipeline builder in `_PIPELINE_BUILDERS` of `harness.py`, and run `check_scorer`. The paragraphs of the text, split by line, must be those of the reference answers.
