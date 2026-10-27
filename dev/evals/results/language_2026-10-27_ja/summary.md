# Extraction evaluation: language_2026-10-27_ja

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 0.923 (0.904-0.934) | 7.3/10 (7-8) | 3/3 | 0.739 (0.701-0.762) | 6 (6-7) | 71,693 (66,046-82,660) | 0 | 11,954 (9,820-13,839) | 63 (50-75) | 1 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 0.991 | 10/10 | 3/3 | 0.914 | 6 | 66,681 | 0 | 6,961 | 50 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | 3 | 16/18 (0.89) | 46/50 (0.92) | 18/18 (1.00) | 249 | 0.0049 |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | 1 | 6/6 (1.00) | 18/18 (1.00) | 6/6 (1.00) | 96 | 0.0019 |

## Output language

The texts that the LLMs wrote in the required language, judged one by one: the output language of the run, or for `adapt` the language of the document. The judge tells languages apart, but not the scripts of Chinese or the variants of Portuguese: the last column counts the texts by the markers of their variant (`dev.evals.language_markers`), those of the required variant / of the other one / of both / of none.

| Case | Model | Mode | Output language | Required | Strings | Justifications | Aspect justifications | All | Variant markers |
|---|---|---|---|---|---|---|---|---|---|
| complex_multilingual | anthropic/claude-haiku-5-5 | default | ja | Japanese [ja] | 18/18 (1.00) | 50/50 (1.00) | 109/110 (0.99) | 177/178 (0.99) | n/a |
| complex_multilingual | anthropic/claude-sonnet-5-5 | default | ja | Japanese [ja] | 6/6 (1.00) | 16/18 (0.89) | 45/45 (1.00) | 67/69 (0.97) | n/a |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 4 | 25 | 0.2424 |
| Judge | 5 | 457 | 0.0091 |
| Total | | | 0.2515 |

## Concepts missed

- **complex_multilingual, anthropic/claude-haiku-5-5, default** (3 runs): Net value of the initial stocking order in EUR (NumericalConcept): 3x; Guaranteed minimum efficiency of HP-500 at BEP in % (NumericalConcept): 3x; Payment due date of the initial stocking order invoice (DateConcept): 2x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_multilingual, anthropic/claude-haiku-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.83 (P 0.92, R 0.76); Warranty and product performance: F1 0.86 (P 1.00, R 0.76)
- **complex_multilingual, anthropic/claude-sonnet-5-5, default**: Territory and minimum purchases: F1 1.00 (P 1.00, R 1.00); Delivery and transport risk: F1 1.00 (P 1.00, R 1.00); Prices and payment: F1 0.97 (P 0.93, R 1.00); Warranty and product performance: F1 1.00 (P 1.00, R 1.00)

## Errors and warnings

- none
