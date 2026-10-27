# Extraction evaluation: language_2026-10-27_zh-Hans

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.934 (0.928-0.942) | 7/10 | 3/3 | 0.753 (0.702-0.782) | 6 | 66,253 (66,097-66,354) | 0 | 9,640 (8,552-10,393) | 50 (47-53) | 0 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.977 | 10/10 | 3/3 | 0.895 | 6 | 67,308 | 0 | 6,894 | 50 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 17/18 (0.94) | 44/50 (0.88) | 18/18 (1.00) | 258 | 0.0050 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 6/6 (1.00) | 18/18 (1.00) | 6/6 (1.00) | 98 | 0.0019 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | zh-Hans | Chinese (Simplified) [zh-Hans] | 18/18 (1.00) | 49/50 (0.98) | 113/113 (1.00) | 180/181 (0.99) | 181 / 0 / 0 / 0 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | zh-Hans | Chinese (Simplified) [zh-Hans] | 6/6 (1.00) | 18/18 (1.00) | 47/47 (1.00) | 71/71 (1.00) | 71 / 0 / 0 / 0 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 24 | 0.2379 |
| Judge | 5 | 468 | 0.0092 |
| Total | | | 0.2471 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Net value of the initial stocking order in EUR (NumericalConcept): 3x; Payment due date of the initial stocking order invoice (DateConcept): 3x; Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.86 (P 0.95, R 0.79); Warranty and product performance: F1 0.88 (P 1.00, R 0.79)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 0.94 (P 0.89, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.97 (P 0.93, R 1.00); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
