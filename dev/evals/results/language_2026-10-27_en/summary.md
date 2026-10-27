# Extraction evaluation: language_2026-10-27_en

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.901 (0.879-0.920) | 7.7/10 (7-8) | 3/3 | 0.788 (0.776-0.800) | 6 | 64,366 (64,069-64,815) | 0 | 8,760 (8,470-9,107) | 38 (38-39) | 0 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.982 | 10/10 | 3/3 | 0.924 | 6 | 65,286 | 0 | 6,900 | 44 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 18/18 (1.00) | 47/51 (0.92) | 18/18 (1.00) | 241 | 0.0044 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 6/6 (1.00) | 18/18 (1.00) | 6/6 (1.00) | 91 | 0.0017 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | en | English [en] | 18/18 (1.00) | 51/51 (1.00) | 103/103 (1.00) | 172/172 (1.00) | n/a |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | en | English [en] | 6/6 (1.00) | 18/18 (1.00) | 43/43 (1.00) | 67/67 (1.00) | n/a |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 24 | 0.2320 |
| Judge | 5 | 444 | 0.0084 |
| Total | | | 0.2404 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Net value of the initial stocking order in EUR (NumericalConcept): 3x; Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x; Payment due date of the initial stocking order invoice (DateConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 0.98 (P 1.00, R 0.96); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.78 (P 0.94, R 0.67); Warranty and product performance: F1 0.85 (P 1.00, R 0.74)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.93 (P 0.93, R 0.93); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
