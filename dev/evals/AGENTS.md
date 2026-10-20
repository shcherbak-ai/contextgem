# dev/evals - Extraction Evaluation Harness

Measures the quality and the cost of extraction on evaluation cases with reference answers, so that a change (a prompt, a model, a setting) can be compared with the version without it, and models with one another, with recorded baselines as snapshots. It is a development tool: it is not a part of the package, and the tests do not run it.

## Layout

| Path | Role |
|------|------|
| `cases/<case>.txt` | Text of the document of a case |
| `cases/<case>.py` | Extraction pipeline of the case (`build_<case>_pipeline()`) |
| `cases/<case>.gold.json` | Reference answers of the case: paragraphs of each aspect, items of each concept |
| `cases/<case>.neutral.json` | Paragraphs that are not scored for an aspect: section headings, and borderline and linked paragraphs with the reasons |
| `cases/reference_review.md` | Record of the review of the reference answers: method, rulings and limits |
| `cases/judge_checks.json` | Checks of the judge: candidate answers whose verdict is known |
| `cases/<case>.docling.json` | Docling document of the case, from which `--input docling` creates its document with `Document.from_docling()` |
| `cases/<case>.docling.html` | HTML rendering of the text of the case that Docling converted into it |
| `docling_cases.py` | Writes the Docling documents of the cases and checks them (offline, no request) |
| `cases/<case>.pdf` | The HTML rendering of a case with tables printed as a PDF file, as a report is printed (`complex_mcu`, `complex_stability`) |
| `cases/<case>.pdf.docling.json` | The Docling document that full Docling converts the PDF file into, from which `--input pdf` creates the document of the case |
| `docling_pdf_cases.py` | Prints the PDF files, converts them with full Docling in a throwaway environment, and reports how their paragraphs map to those of the reference answers (no request to a model; needs Chrome, Chromium or Edge) |
| `harness.py` | Loads a case, creates the model, writes results |
| `scoring.py` | Scores an extracted document against the reference answers |
| `run_eval.py` | Extracts a case with a model and scores it (**live requests**) |
| `summarize.py` | Summary of the runs of a label (offline) |
| `costs.py` | What the requests of each label cost, from the costs that its results record (offline) |
| `rescore.py` | Scores the recorded runs of a label again, after a change in the reference answers (offline) |
| `decision_probabilities.py` | Probabilities of a decision model for the aspects of a case (**live requests**) |
| `threshold_sweep.py` | Decision threshold sweep over the recorded probabilities (offline) |
| `judge.py` | Judges the string values, justifications and text fields of objects of the runs of a label, and checks the judge (**live requests**) |
| `check_scorer.py` | Self-check of the scorer on each case (offline) |
| `results/<label>/` | Results of one experiment: `runs/`, `decision_probabilities/`, `judgments/` and `judge_checks.json`, and the generated `summary.*` and `thresholds.*` |

## The Cases

| Case | Document | Paragraphs | Aspects | Concepts |
|------|----------|------------|---------|----------|
| `complex_nda` | A non-disclosure agreement with annexes and a disclosure register | 72 | 4 | 21 |
| `complex_dpa` | A data processing agreement with two amendments and operational records | 153 | 6 | 24 |
| `complex_mcu` | The technical documentation of a microcontroller: datasheet, errata sheet, change notification with an amendment, application note, and the design review and purchasing records of a customer | 268 | 7 | 27 |
| `complex_stability` | The stability study report of a tablet: three batches in three container closure systems, a release and a shelf-life specification, and the results of each batch at each storage condition | 125 | 4 | 18 |

`complex_mcu` is a technical document with dense terminology (voltage-scaling ranges, PLL dividers, wait states, thermal resistances, errata conditions). It has many values in tables, one row per paragraph, and later documents that change values of earlier ones. Many of its answers are computed: junction temperatures from currents and thermal resistances, clock frequencies from divider settings, calendar dates and ISO weeks. Its document concepts ask for references, as its aspect concepts do.

`complex_stability` has most of its answers in tables that look alike: two specifications with the same tests, and a table of results for each batch and condition with the same tests and, for the long-term and accelerated conditions, the same number of columns. A row tells its batch, its condition and its time points only through the caption and the header of its table, and the concepts of its aspects are extracted from the paragraphs of the aspect only, where a row can come without them. It was made to measure the structure of a document (see Extraction from Docling Documents), and its narrative contradicts its tables, which record the results: two of its five results outside the specification are only in the tables.

## Rules

- **`run_eval`, `decision_probabilities` and `judge` send live, paid requests.** Run them only when the maintainer asks for it. The other scripts send nothing.
- **Keys come from the environment or the `.env` file of the repo** (`CONTEXTGEM_OPENAI_API_KEY`, `CONTEXTGEM_ANTHROPIC_API_KEY`, `CONTEXTGEM_TYPESAFE_API_KEY`, `CONTEXTGEM_AZURE_AI_API_KEY` with `CONTEXTGEM_AZURE_AI_API_BASE`). Never print them. The host of a Microsoft Foundry resource is replaced in what is written.
- **A label is one experiment.** A run whose file exists is skipped, so results are never overwritten: record a change under a new label, and leave the baselines (`baseline_2026-10-11`, `baseline_2026-10-15`) as they are.
- **Results are tracked artifacts.** Do not edit or delete the files under `results/`. `summary.*` and `thresholds.*` are generated: regenerate them, do not edit them. The scores of a run are rewritten only by `rescore`, and its judgment (`judgments/`) only by `judge`, when the judge, its questions or the reference answers of the case changed.
- **The reference answers change only by a recorded review.** Do not change a `.gold.json` or `.neutral.json` file to make a result pass. Record a change with its reason in `cases/reference_review.md`, and run `check_scorer` after a change in a case, in its reference answers or in the scorer. After a change in the reference answers, score the recorded runs again with `rescore`, and regenerate the summaries.

## The Reference Answers

The `.gold.json` files of `complex_nda` and `complex_dpa` were produced by an AI agent, and reviewed on 2026-10-11 by AI agents: a second, blind annotation of each case, and a ruling on every disagreement against the text. `complex_mcu` was written with its reference answers by an AI agent on 2026-10-15, with the computed answers recomputed by a script, and reviewed in the same way (see `cases/reference_review.md`). No person has verified them, whatever the `reviewed` flags in the files say. So a score is the agreement with these answers, not a proof of correctness:

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

# What the requests of every label, or of one, cost (offline)
uv run python -m dev.evals.costs
uv run python -m dev.evals.costs --label my_experiment

# Score the recorded runs again after a change in the reference answers (offline)
uv run python -m dev.evals.rescore --label my_experiment

# Probabilities of a decision model (live), then the threshold sweep (offline)
uv run python -m dev.evals.decision_probabilities --case complex_nda --model typesafe/jev-latest --label my_experiment
uv run python -m dev.evals.threshold_sweep --label my_experiment

# Check the judge on the checks of known verdict, then judge the runs of a label (live)
uv run python -m dev.evals.judge --checks --label my_experiment
uv run python -m dev.evals.judge --label my_experiment

# Self-check of the scorer (offline)
uv run python -m dev.evals.check_scorer

# The same extraction from the Docling document of the case (live), and
# the Docling documents written again after a change in a case (offline)
uv run python -m dev.evals.run_eval --case complex_mcu --model openai/gpt-6-luna --runs 3 --label my_experiment --input docling
uv run python -m dev.evals.docling_cases

# The same from the Docling document of the case printed as a PDF file (live), and
# the PDF files printed and converted again (offline), or only their mapping reported
uv run python -m dev.evals.run_eval --case complex_mcu --model openai/gpt-6-luna --runs 3 --label my_experiment --input pdf
uv run python -m dev.evals.docling_pdf_cases
uv run python -m dev.evals.docling_pdf_cases --report
```

`--concurrency` runs the extraction with `use_concurrency=True`. One line is printed per run, which starts with the name of its file. The library prints its logs as usual: set `CONTEXTGEM_LOGGER_LEVEL=WARNING` for a quiet run.

## What Is Measured

- **Aspects**: precision, recall and F1 of the extracted paragraphs against those of the reference answers, averaged over the aspects. The neutral paragraphs of an aspect are left out of both.
- **Concepts**: date, numerical, Boolean, label and rating concepts by exact match of the values; JSON object concepts by the names of the objects, or by the field that identifies them where the reference answers name one (`object_key` in the `custom_data` of the concept, e.g. `id` in `complex_mcu`). String concepts are counted apart from these, and judged (below).
- **What exact matching cannot compare is judged** by a decision model, `typesafe/jev-latest` unless `--model` names another (`judge.py`):
  - the values of string concepts, against the reference value (a value equal to the reference value or to an accepted one is correct without a request);
  - the justification of each extracted item, against the reference answer and its reasoning: it is correct only for a value of the reference answer, with no wrong fact or step;
  - the text fields of the objects of JSON object concepts, against the reference object with the same name or ID (the other fields are compared exactly).

  The judge is asked one yes/no question at a time, with the questions of `judge.py`, not with the prompts of the library, so that a change of the library does not change the judge. A verdict is "correct" at a probability of 0.5 or more; the probability is kept. The judgment of a run records the judge, a hash of its questions and of the reference answers it judged against, and its cost. Decision models give no strings, objects or justifications, so their runs are not judged.
- **The judge is checked before it is used**: `cases/judge_checks.json` has candidate answers of the three cases whose verdict is known, right ones in other words and wrong ones with the mistakes a capable model makes. `judge --checks` records how often the judge agrees, for each kind, in `results/<label>/judge_checks.json`. Read a judged score with that agreement in mind.
- **References of concepts**: F1 of the referenced paragraphs.
- **Cost**: requests, input, cached and output tokens, seconds, retries for invalid data, and the cost in USD, which each run, judgment, check of the judge and pass of `decision_probabilities` records (the passes recorded before 2026-10-15 have none). The summary of a label adds them up (Cost), and `costs` does so for every label; `uv run fab costs` prints it with what the requests recorded for the tests cost.
- **Decision models** extract all the aspects, and of the concepts only the Boolean, label and rating ones (the closed concepts), without justifications and references. Their concept counts are not comparable with those of the LLMs: compare the two kinds of model on the aspects and on the column "Closed concepts correct" of the summary, which counts the closed concepts only, for every model. The judged columns are of LLMs only, and no score adds them up with the others, so a decision model loses nothing for what it does not extract.
- **The aspects of the cases favour models that read the whole document**: most are defined by what is needed to answer a question about a board, an incident or a claim, which a decision model, judging each paragraph on its own, cannot see. On `complex_mcu`, the decision models score an aspect F1 of 0.40 to 0.66 on the four aspects of the boards, and 0.66 to 0.84 on the other three; the hosted LLMs 0.85 to 0.95, and 0.95 to 1.00. A difference between the two kinds of model on the aspects is partly this.

Each run records a short hash of every prompt template of the library, including the template of the default system message (`provenance`), which tells the runs of a changed prompt apart. The runs of `baseline_2026-10-11` were recorded before the system message template was hashed, so they have no hash of it.

## The Models

Every experiment and baseline runs the same models, each for a purpose, so that labels compare with one another:

| Model | Purpose | Runs of each case | Recorded in |
|-------|---------|-------------------|-------------|
| `openai/gpt-6-luna` | The LLM of the tests, the documentation and the examples | 3 | Every label |
| `anthropic/claude-haiku-5-5` | A second LLM, of another provider at the same price | 3 | Every label |
| `typesafe/jev-latest`, `openai/gpt-6-luna` and `azure_ai/ms-decision-model` | The hosted decision models | 1 (2 in a baseline), and one pass of `decision_probabilities` | Every label |
| `typesafe/jev-latest` | The judge (`judge.py`) | | Every label |
| `openai/gpt-6.1-sol` | A top LLM | 1 | Baselines |
| `openai/gpt-6-luna` with `--concurrency` | The extraction with concurrency | 3 | Baselines |

- **The second LLM is of the same tier as the first**, not a stronger one: it tells whether the effect of a change holds for another model, and a change of a prompt shows at this tier. `openai/gpt-6.1-sol` already extracts nearly all the concepts of the cases (all 22 of `complex_dpa` and all 26 of `complex_mcu` in `baseline_2026-10-15`).
- **The top LLM tells the limit of a model from a problem of a case** or of its reference answers. A run costs about 20 times one of `openai/gpt-6-luna`, so it is in the baselines only.
- **Each model runs with its defaults**: the harness sets no temperature, reasoning effort or output limit.
- **Local models are run when a question concerns them**, under a label of their own: `baseline_2026-10-11` is what is known of them.
- **The second LLM was `openai/gpt-5.6-luna` until 2026-10-20**, in every label recorded until then: an older model of the same provider, which got fewer concepts than `openai/gpt-6-luna` at about twice the cost of a run. `anthropic/claude-haiku-5-5` has no baseline yet: compare its runs with the control recorded at the same time. It returns invalid JSON for the aspects of `complex_mcu` often enough to fail whole runs (see From a PDF File): count its failed runs, and compare the valid ones. The tests keep `openai/gpt-5.6-luna` as a second, distinct model (fallbacks, a model with a role), whose extraction they do not measure.

## Comparing a Change with the Baseline

1. **Pick what the change concerns**, among the models above. For a prompt of the LLMs: the two LLMs, 3 runs of each case each. For a question of the decision models: one pass of `decision_probabilities` of each hosted decision model, and the threshold sweep; these models give nearly the same answers in each pass.
2. **Make the change.** It can be run from a copy of the code outside the repo (`git archive HEAD contextgem dev/__init__.py dev/evals`, with the changed template, and a launcher that puts the copy first on `sys.path`), so that the working tree stays as it is meanwhile.
3. **Record the version with the change and the version without it at the same time**, each under a label of its own, and judge both. A hosted model can change under the same name, and the version of `litellm` moves: a baseline of another day is not a control.
4. **Summarize both labels and compare their `summary.md` files**, then read the disagreements behind a difference (see The Reference Answers).
5. **Record the outcome.** A change that is kept goes into the library, and a new dated baseline is recorded once the library is in its new state; a change that is not kept goes into Tried and Rejected below.

For a first look, the version with the change alone can be compared with the current baseline, with the same version of `litellm`: a large difference may settle it, a small one needs the control.

Runs of one configuration differ: the aspect F1 of a hosted LLM varies by up to 0.05 between runs. Compare the means of several runs, and treat a smaller difference as noise.

A baseline is a snapshot: what the models score and cost on the cases at one state of the library. It is the comparison when the models of the tests change (is the new model as good and as cheap on these cases?), and it shows whether a change of the library makes extraction worse while the models stay. Record a new one, under a dated label, when the models or the library change; the older ones stay as they are.

## The Baseline

**`baseline_2026-10-15`**, the current one, on the three cases, recorded on 2026-10-15 with `litellm` 1.105.1, from commit `1ecd68c` of the library (the reading of model output made strict), with the hosted models only:

- **LLMs**, each case: `openai/gpt-6-luna` (3 runs, and 3 with concurrency), `openai/gpt-5.6-luna` (3), `openai/gpt-6.1-sol` (1).
- **Decision models**, each case: `typesafe/jev-latest`, `openai/gpt-6-luna` and `azure_ai/ms-decision-model`, a deployment of Microsoft-Decision-1 in Microsoft Foundry (2 runs each), and one pass of `decision_probabilities` each.
- **The judge** (`judgments/`, `judge_checks.json`): `typesafe/jev-latest`, which agreed with all 62 checks. On the runs, every verdict "not correct" was read against the reference answers: of the 24 justifications so judged in the first pass, 19 were wrong; 4 were judged against a reference justification that called the 24-hour limit of `complex_dpa` "effective on 1 November", which is now worded as in force from 1 October (no answer changed), and 1 was correct but shorter than the reference. Of the 13 justifications of values that are wrong by exact match, the judge found 12 wrong. The 6 breach summaries of `openai/gpt-6-luna` in `complex_dpa` that it judged not correct leave out the duties to document breaches and to cooperate, which the reference answer has.
- **No retries** for invalid data in any run, with the stricter reading of model output of that commit.
- **Cost**: USD 1.33 for the runs, USD 0.02 for the judge, about USD 0.10 for the probabilities.

`complex_mcu` separates the models where the contract cases no longer do: its concepts give `openai/gpt-5.6-luna` 23 of 26 in each run, against 26 for `openai/gpt-6.1-sol`, and the decision models score an aspect F1 of 0.57 to 0.65 on it, against 0.77 to 0.89 on the contracts.

Local models are not in it: their results of 2026-10-11 (`baseline_2026-10-11`) are what is known of them, and they are recorded under a label of their own when a question needs them. A local LLM needs a context of more than 16k tokens for `complex_mcu`.

**`baseline_2026-10-11`**, the earlier one, named `baseline` until 2026-10-15 (its runs keep that label), recorded on 2026-10-11 with `litellm` 1.105.0, on `complex_nda` and `complex_dpa`, after the first part of the review of the reference answers, and scored again after its second part. Its runs keep neither the values as extracted nor the justifications, so they cannot be judged:

- **LLMs** (`default` mode unless stated): `openai/gpt-6-luna` (3 runs, and 3 with concurrency), `openai/gpt-5.6-luna` (3), `openai/gpt-6.1-sol` (1), and on Ollama with a context length of 16k, `ollama_chat/mistral-small:24b` (2) and `ollama_chat/llama3.1:8b` (2, which failed on invalid JSON).
- **Decision models**: `typesafe/jev-latest` (2), `openai/gpt-6-luna` (2), `azure_ai/ms-decision-model` (2), `strands_decider/nimble` on Ollama (1) and `strands_decider/kev-4b` on LM Studio (1).
- **Probabilities of the decision models** (`decision_probabilities/`): recorded earlier the same day, one pass per model and case. They do not depend on the reference answers, so they were kept; they have no `provenance`. Those of `openai/gpt-6-luna` were recorded with `litellm` 1.104.2.

## Tried and Rejected

Experiments that changed nothing in the library, recorded so that they are not run again without a new reason.

### Wording of the aspect question of the decision models (2026-10-12)

Three wordings of `decision_question_aspect.j2`, each with one change, were compared with the current one on the five decision models of `baseline_2026-10-11` and both cases, in one pass of `decision_probabilities` each:

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

The differences are not the variation between passes: with the current wording, three of the models give the same answers as in `baseline_2026-10-11`, and the other two differ in about 10 of 1,206 answers, i.e. by 0.003 or less in F1.

Not tested: the questions of concepts (each case has six concepts that a decision model extracts, too few to compare wordings), and aspects extracted with sentences or with additional context. The recorded probabilities were not kept: the numbers above are the record.

## Extraction from Docling Documents

`--input` of `run_eval` and `decision_probabilities` says what the document of a case is created from: `text` (the default, as in every baseline), `docling`, `docling_without_context`, or `pdf` (below). The two Docling inputs create the document with `Document.from_docling()` from `cases/<case>.docling.json`, i.e. with the structure that Docling gives a file: the additional context of each paragraph (what it is, e.g. `Table row`, the headings of its section, the caption of its table) and each table row with the header of each column before its value. `docling_without_context` leaves out the additional context, to tell its effect from that of the table rows. A run and a pass of the probabilities record their input (`input`); those recorded before 2026-10-20 have none, and are of the text.

`docling_cases.py` writes the Docling documents: it renders the text of each case as HTML (each heading of `.neutral.json` at the level of its section, the title of a table as its caption, the first row of a table as its header row), converts it with the docling-slim of the dev group, and writes the HTML and the Docling document. The HTML backend of Docling replaces some characters with ASCII ones, e.g. "—" with "-", which a PDF file of the same text keeps: the script sets a table cell that differs from the case only by them back to the text of the case, so that the Docling document differs from the text only in its structure. The paragraphs of a Docling document are those of the reference answers, position by position (a table row with the values of the cells of its line, in order): `load_case` checks it each time, and `check_scorer` checks each case with both inputs. After a change in the text of a case, write the Docling documents again.

This measures what ContextGem makes of the structure of a document, not how well Docling reads a file: the HTML comes from the text, so Docling finds exactly its sections and tables.

### From a PDF File

`--input pdf` creates the document from the Docling document of the case printed as a PDF file, i.e. with what the layout and table models of Docling read in a real file. `docling_pdf_cases.py` prints the HTML rendering of each case with tables (`PDF_CASES`: `complex_mcu` and `complex_stability`) with headless Chrome, on A4 pages with the title of the case in their header and their number in their footer, and the tables with borders, and converts the file with full Docling, without OCR, as it has a text layer. Docling then gives every heading at level 1, leaves the headers and footers of the pages out, splits a paragraph or a table at the end of a page (the part of a table on the next page is a table of its own, without the caption, under the header row that the browser repeats), reads a part of a page out of order now and then (e.g. a table after the paragraphs below it), and spaces out underscores ("RCC _ PLLCFGR").

The paragraphs of such a document are not those of the reference answers, so they are mapped to them by their text (`map_paragraphs` in `harness.py`): the words of all the paragraphs are aligned in order, which tells apart the rows of tables that look alike by their places, and the words left unaligned are aligned again among themselves, for the parts that Docling read out of order. A paragraph holds a paragraph of the reference answers when half of the words of one of the two are aligned with words of the other (a table row compared by the values of its cells). The scores then count what is extracted as the paragraphs of the reference answers that it holds, so the three inputs are scored on the same paragraphs; the probabilities of a decision model are written for the paragraphs of the reference answers, each with the highest probability of the paragraphs that hold it (and those of the paragraphs of the document are kept, `document_probabilities`). `docling_pdf_cases.py --report` prints the mapping, and `check_scorer` checks that the paragraphs that hold the reference ones find all of them. On the files of 2026-10-21, every paragraph of the reference answers is found, no paragraph holds two of them, one is split between two pages, and the paragraphs that hold none are the header rows that the browser repeats.

#### Measured on 2026-10-21

Three labels recorded at the same time: `pdf_2026-10-21_text` (the control), `pdf_2026-10-21_docling` (the HTML rendering) and `pdf_2026-10-21_pdf`, each with the models of every label (`openai/gpt-6-luna` and `anthropic/claude-haiku-5-5`, 3 runs each; the three hosted decision models, 1 run and one pass of `decision_probabilities` each; the judge, which agreed with 75 of the 76 checks for each label), on `complex_mcu` and `complex_stability`. Cost: USD 1.28 for the three labels.

| Case | Model | Measure | Text | Docling (HTML) | PDF |
|------|-------|---------|------|----------------|-----|
| `complex_stability` | `openai/gpt-6-luna` | Aspect F1 | 0.961 | 0.962 | 0.958 |
| `complex_stability` | `openai/gpt-6-luna` | Concepts correct (of 16) | 15.7 | 15.3 | 14.7 |
| `complex_stability` | `openai/gpt-6-luna` | Reference F1 | 0.855 | 0.865 | 0.860 |
| `complex_stability` | `anthropic/claude-haiku-5-5` | Aspect F1 | 0.903 | 0.870 | 0.917 |
| `complex_stability` | `anthropic/claude-haiku-5-5` | Reference F1 | 0.755 | 0.714 | 0.769 |
| `complex_mcu` | `openai/gpt-6-luna` | Aspect F1 | 0.936 | 0.956 | 0.950 |
| `complex_mcu` | `openai/gpt-6-luna` | Concepts correct (of 26) | 25.7 | 23.3 | 25.7 |
| `complex_mcu` | `openai/gpt-6-luna` | Reference F1 | 0.733 | 0.673 | 0.740 |
| `complex_mcu` | `anthropic/claude-haiku-5-5` | Aspect F1 (valid runs) | none of 3 | 0.757 (3) | 0.714 (2) |
| `complex_stability` | Decision models, each | Aspect F1 | 0.34 to 0.41 | 0.64 to 0.77 | 0.62 to 0.76 |
| `complex_mcu` | Decision models, each | Aspect F1 | 0.56 to 0.65 | 0.69 to 0.75 | 0.65 to 0.73 |
| Both | Decision models, each | Aspect F1 at its best threshold (mean of the cases) | 0.51 to 0.55 | 0.73 to 0.78 | 0.74 to 0.76 |

- **The PDF file keeps most of what the structure gives the decision models**: the mean aspect F1 of the three rises from the text by 0.33 on `complex_stability` and 0.09 on `complex_mcu` with the PDF file, against 0.34 and 0.10 with the HTML rendering, although the parts of tables after a page break come without their captions and every heading is at level 1. `typesafe/jev-latest` does better with the PDF file than with the HTML rendering on `complex_mcu` (0.725 against 0.690).
- **The LLMs show no difference beyond the variation between runs**, as with the HTML rendering on 2026-10-20. With `openai/gpt-6-luna` on `complex_mcu`, the HTML rendering gave fewer concepts and a lower reference F1 than both the text and the PDF file, as on 2026-10-20 (24.0 against 25.7 then); its section paths are the longest of the three inputs, but three runs do not tell that apart from the variation.
- **`anthropic/claude-haiku-5-5` returns invalid JSON for the aspects of `complex_mcu`**, e.g. `{"aspect_id":"A4":"A4","paragraph_ids":[]}`, and leaves the last aspects empty: all three runs from the text, and one of the three from the PDF file, failed after three retries, and most of its other runs on the case needed retries. On `complex_stability` it failed none. The length of the answer is the cause: extracting only the aspects of `complex_mcu` from the text, 2 runs each, 3 of its 5 answers with all the aspects in one request were invalid (both runs succeeded on a retry), against none of its 14 answers with one aspect in each request (`max_items_per_call=1`), whose aspect F1 was lower, though (0.54 and 0.64, against 0.75 and 0.79), at about three times the cost. It gets no cached tokens, as ContextGem does not ask Anthropic to cache, so a run costs about three times one of `openai/gpt-6-luna`.
- **The input tokens**: the PDF file adds about as many as the HTML rendering: 32% to 39% for `openai/gpt-6-luna`, 10% for `anthropic/claude-haiku-5-5` on `complex_stability`.

The limits: two cases, printed from their text by one browser, i.e. a born-digital PDF file with a text layer, not a scanned one; one run of each decision model.

The environment of the project was upgraded by another piece of work during the recording, so the runs record `litellm` 1.105.1, 1.106.1 or 1.105.2 (`provenance`). The three inputs share one version on `complex_stability` for both LLMs and for `typesafe/jev-latest` and `openai/gpt-6-luna` as decision models (1.105.1), and on `complex_mcu` for `openai/gpt-6-luna` and `azure_ai/ms-decision-model` as decision models (1.105.2), and these show what is said above; the other comparisons mix versions. Record an experiment in an environment that nothing else changes meanwhile, e.g. a worktree.

### Measured on 2026-10-20

Three labels recorded at the same time, with `litellm` 1.105.1, before `Document.from_docling()` was released: `docling_2026-10-20_text` (the control), `docling_2026-10-20_docling`, and `docling_2026-10-20_without_context` (`complex_mcu` only, as the other cases have no tables). Each: `openai/gpt-6-luna` and `openai/gpt-5.6-luna`, 3 runs each; the three hosted decision models, 1 run and one pass of `decision_probabilities` each; the judge, which agreed with all 62 checks for each label. Cost: USD 1.35 for the three labels.

**Decision models gain on every case.** Aspect F1 at the default threshold, one run each (the decision models give nearly the same answers in each pass):

| Case | Model | Text | Docling | Without context |
|------|-------|------|---------|-----------------|
| `complex_nda` | `typesafe/jev-latest` | 0.822 | 0.894 | |
| `complex_nda` | `openai/gpt-6-luna` | 0.892 | 0.932 | |
| `complex_nda` | `azure_ai/ms-decision-model` | 0.779 | 0.830 | |
| `complex_dpa` | `typesafe/jev-latest` | 0.793 | 0.845 | |
| `complex_dpa` | `openai/gpt-6-luna` | 0.823 | 0.908 | |
| `complex_dpa` | `azure_ai/ms-decision-model` | 0.763 | 0.776 | |
| `complex_mcu` | `typesafe/jev-latest` | 0.623 | 0.685 | 0.683 |
| `complex_mcu` | `openai/gpt-6-luna` | 0.633 | 0.749 | 0.659 |
| `complex_mcu` | `azure_ai/ms-decision-model` | 0.555 | 0.706 | 0.624 |

- The mean of the nine is 0.743 with the text and 0.814 with the Docling document. At the best threshold of each model (`thresholds.md`, mean of the cases), 0.794 to 0.848 (`azure_ai/ms-decision-model`), 0.787 to 0.873 (`openai/gpt-6-luna`) and 0.757 to 0.820 (`typesafe/jev-latest`): the probabilities tell the reference paragraphs from the others better, not only on the other side of the threshold.
- The closed concepts: 55 of 69 correct, against 50 with the text.
- On `complex_mcu`, the table rows with the headers of their columns account for the gain of `typesafe/jev-latest`; the additional context adds 0.08 to 0.09 for the two other models.
- The requests have 17% to 32% more input tokens.

**LLMs show no difference beyond the variation between runs**, at a cost in tokens:

- Aspect F1 within 0.025 of the text in each case, and the reference F1 within 0.035, in both directions. The judged strings and justifications differ by one or two runs' worth.
- On `complex_mcu`, `openai/gpt-6-luna` got 24.0 of 26 concepts (22 to 26) against 25.7 with the text: two runs gave no item for the computed junction temperature of DR-114 and the HDR-7 check that depends on it, while all the paragraphs of their aspect were extracted. Without the additional context it got 25.7, and both LLMs had their highest reference F1 there (0.743 and 0.657, against 0.716 and 0.616 with the text).
- The input tokens: 5% to 12% more on the contracts, 40% more on `complex_mcu`, whose table rows repeat the caption of their table and the headings of their section in their additional context.

So the additional context is worth its tokens for decision models, which judge each paragraph on its own; LLMs, which read the paragraphs of the whole document in order, do as well without it on these cases. The prompts of LLMs keep the additional context of table rows (decided by the maintainer on 2026-10-20): it also tells which table a row is from where its caption and header are not in the same prompt, i.e. for the concepts of an aspect, which are extracted from the paragraphs of the aspect only, and when `max_paragraphs_to_analyze_per_call` splits a table between requests. `complex_stability` was added the same day to test the first of these (below); none of the cases tests the second. The limits: the structure is that of the text, so it is clean, where a converted PDF has the errors of Docling; three cases, one of them with tables; one run of each decision model.

### `complex_stability`, measured on 2026-10-20

The same configurations on `complex_stability`, under `stability_2026-10-20_text`, `stability_2026-10-20_docling` and `stability_2026-10-20_without_context`, recorded at the same time with `litellm` 1.105.1 (USD 0.29 for the three labels). Mean of 3 runs for the LLMs, 1 run for the decision models:

| Model | Measure | Text | Docling | Without context |
|-------|---------|------|---------|-----------------|
| `openai/gpt-6-luna` | Aspect F1 | 0.936 | 0.978 | 0.977 |
| `openai/gpt-6-luna` | Concepts correct (of 16) | 14.7 | 15.7 | 14.7 |
| `openai/gpt-6-luna` | Reference F1 | 0.802 | 0.871 | 0.801 |
| `openai/gpt-6-luna` | Justifications correct (of 51, judged) | 47 | 50 | 47 |
| `openai/gpt-5.6-luna` | Aspect F1 | 0.967 | 0.983 | 0.980 |
| `openai/gpt-5.6-luna` | Concepts correct (of 16) | 13.7 | 13.7 | 14.3 |
| `openai/gpt-5.6-luna` | Reference F1 | 0.728 | 0.864 | 0.829 |
| Decision models, each | Aspect F1 | 0.36 to 0.41 | 0.65 to 0.76 | 0.39 to 0.45 |
| Decision models, each | Aspect F1 at its best threshold | 0.45 to 0.48 | 0.83 to 0.85 | 0.47 to 0.54 |

- **The concepts of an aspect gain where the case puts the table rows without their caption**: the number of results outside the specification is right in 0 of 3 runs of `openai/gpt-6-luna` with the text and without the additional context, and in 2 of 3 with it (the wrong answers, 3 and 4, are the count of the narrative, or leave out the result of V20-2607, whose batch only the caption tells); whether every such result is investigated, in 2, 3 and 2 of 3. `openai/gpt-5.6-luna` mostly gives the count of the narrative with every input, and gave the proposed 24 months for the Alu/Alu blister in 2 of its 3 runs with the Docling document (a concept of the document as a whole, 0 of 3 with the text).
- **The references gain for both LLMs** (0.07 and 0.14), and the aspects for `openai/gpt-6-luna`, with the table rows with the headers of their columns as much as with the additional context.
- **The decision models gain from the additional context only**: without it, their aspect F1 is that of the text. The table rows of this case differ only by their caption, which a decision model, judging each paragraph on its own, sees only in the additional context.
- **The input tokens**: 38% more for the LLMs, 20% to 40% for the decision models.

So on a document whose table rows look alike, the additional context of table rows pays for both kinds of model, as the maintainer expected; on the three other cases, it pays for the decision models.

## Adding a Case

1. **Write the files** in `cases/`: the `.txt` text, one paragraph per line, which must be the paragraphs of the reference answers; the `.py` pipeline; the `.gold.json` reference answers, in the format of the other cases (the paragraphs of each aspect and the items of each concept by the positions of the paragraphs, with a justification and evidence for each item); and a `.neutral.json` file if some paragraphs are not scored. A JSON object concept whose objects have no `name` field names the field that identifies them as `object_key` in its `custom_data`.
2. **Register the pipeline builder** in `_PIPELINE_BUILDERS` of `harness.py`, and run `check_scorer`.
3. **Review the reference answers** as those of the other cases were (see `cases/reference_review.md`): separate agents that see only the numbered text and the definitions derive every concept answer, with its evidence and the errors they find in the document, and select the paragraphs of every aspect. Rule on every difference against the text, and record the rulings.
4. **Add checks of the judge** for the string concepts, the justifications and the text fields of objects of the case to `cases/judge_checks.json`, right answers in other words and wrong ones with the mistakes a capable model makes, and run `judge --checks`.
5. **After the first runs of the hosted models, review the aspects again**: a paragraph that at least 4 of 7 runs of the LLMs, or all the hosted decision models, extract beyond the reference answers or leave out of them is a candidate for the linked or borderline paragraphs, as the text supports; score the runs again with `rescore`.
