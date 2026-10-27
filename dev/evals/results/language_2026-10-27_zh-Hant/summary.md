# Extraction evaluation: language_2026-10-27_zh-Hant

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.942 (0.932-0.952) | 7.3/10 (7-8) | 3/3 | 0.783 (0.749-0.841) | 6 | 66,404 (66,283-66,576) | 0 | 11,355 (10,873-11,683) | 56 (52-58) | 0 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.982 | 10/10 | 3/3 | 0.916 | 6 | 66,739 | 0 | 6,371 | 49 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 18/18 (1.00) | 45/52 (0.87) | 18/18 (1.00) | 266 | 0.0052 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 6/6 (1.00) | 17/18 (0.94) | 5/6 (0.83) | 95 | 0.0018 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | zh-Hant | Chinese (Traditional) [zh-Hant] | 18/18 (1.00) | 52/52 (1.00) | 117/117 (1.00) | 187/187 (1.00) | 187 / 0 / 0 / 0 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | zh-Hant | Chinese (Traditional) [zh-Hant] | 6/6 (1.00) | 18/18 (1.00) | 44/44 (1.00) | 68/68 (1.00) | 68 / 0 / 0 / 0 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 24 | 0.2342 |
| Judge | 5 | 473 | 0.0093 |
| Total | | | 0.2435 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Payment due date of the initial stocking order invoice (DateConcept): 3x; Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x; Net value of the initial stocking order in EUR (NumericalConcept): 2x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.89 (P 0.92, R 0.86); Warranty and product performance: F1 0.88 (P 1.00, R 0.79)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.93 (P 0.93, R 0.93); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
