# Extraction evaluation: language_2026-10-27_adapt

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.928 (0.920-0.932) | 7/10 | 3/3 | 0.784 (0.776-0.790) | 6 | 66,357 (66,205-66,455) | 0 | 13,124 (12,597-13,395) | 56 (54-58) | 0 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.991 | 10/10 | 3/3 | 0.945 | 6 | 66,725 | 0 | 8,769 | 55 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 18/18 (1.00) | 44/51 (0.86) | 18/18 (1.00) | 250 | 0.0047 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 5/6 (0.83) | 18/18 (1.00) | 6/6 (1.00) | 91 | 0.0017 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | adapt | German [de] | 15/18 (0.83) | 51/51 (1.00) | 112/112 (1.00) | 178/181 (0.98) | n/a |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | adapt | German [de] | 1/6 (0.17) | 5/18 (0.28) | 43/43 (1.00) | 49/67 (0.73) | n/a |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 24 | 0.2607 |
| Judge | 5 | 453 | 0.0088 |
| Total | | | 0.2695 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Net value of the initial stocking order in EUR (NumericalConcept): 3x; Payment due date of the initial stocking order invoice (DateConcept): 3x; Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.83 (P 0.91, R 0.76); Warranty and product performance: F1 0.88 (P 1.00, R 0.79)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.97 (P 0.93, R 1.00); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
