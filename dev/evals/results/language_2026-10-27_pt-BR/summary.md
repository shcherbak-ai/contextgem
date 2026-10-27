# Extraction evaluation: language_2026-10-27_pt-BR

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.951 (0.932-0.961) | 7.7/10 (7-8) | 3/3 | 0.799 (0.775-0.816) | 7 (6-8) | 75,102 (66,216-92,666) | 0 | 13,631 (11,952-16,931) | 60 (53-72) | 2 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.991 | 10/10 | 3/3 | 0.926 | 6 | 66,762 | 0 | 7,414 | 49 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 17/18 (0.94) | 45/53 (0.85) | 18/18 (1.00) | 268 | 0.0052 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 5/6 (0.83) | 18/18 (1.00) | 6/6 (1.00) | 95 | 0.0018 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | pt-BR | Portuguese (Brazil) [pt-BR] | 18/18 (1.00) | 53/53 (1.00) | 117/117 (1.00) | 188/188 (1.00) | 10 / 0 / 0 / 178 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | pt-BR | Portuguese (Brazil) [pt-BR] | 6/6 (1.00) | 18/18 (1.00) | 44/44 (1.00) | 68/68 (1.00) | 0 / 0 / 0 / 68 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 26 | 0.2506 |
| Judge | 5 | 475 | 0.0093 |
| Total | | | 0.2599 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Payment due date of the initial stocking order invoice (DateConcept): 3x; Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x; Net value of the initial stocking order in EUR (NumericalConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.93 (P 0.93, R 0.93); Warranty and product performance: F1 0.88 (P 1.00, R 0.79)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.97 (P 0.93, R 1.00); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
