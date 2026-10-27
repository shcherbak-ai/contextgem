# Extraction evaluation: language_2026-10-27_pt-PT

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.934 (0.908-0.952) | 7.7/10 (7-8) | 3/3 | 0.791 (0.750-0.848) | 6 (6-7) | 71,963 (65,776-83,666) | 0 | 14,012 (11,426-15,417) | 64 (57-69) | 1 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.977 | 10/10 | 3/3 | 0.916 | 6 | 67,438 | 0 | 7,507 | 50 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 17/18 (0.94) | 45/52 (0.87) | 17/18 (0.94) | 259 | 0.0050 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 6/6 (1.00) | 17/18 (0.94) | 5/6 (0.83) | 99 | 0.0019 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | pt-PT | Portuguese (Portugal) [pt-PT] | 18/18 (1.00) | 51/52 (0.98) | 110/113 (0.97) | 179/183 (0.98) | 2 / 0 / 0 / 181 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | pt-PT | Portuguese (Portugal) [pt-PT] | 6/6 (1.00) | 17/18 (0.94) | 48/48 (1.00) | 71/72 (0.99) | 8 / 0 / 0 / 64 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 25 | 0.2526 |
| Judge | 5 | 470 | 0.0092 |
| Total | | | 0.2618 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x; Net value of the initial stocking order in EUR (NumericalConcept): 2x; Payment due date of the initial stocking order invoice (DateConcept): 2x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.86 (P 0.92, R 0.81); Warranty and product performance: F1 0.88 (P 1.00, R 0.79)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 0.94 (P 0.89, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.97 (P 0.93, R 1.00); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
