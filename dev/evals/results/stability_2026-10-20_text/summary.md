# Extraction evaluation: stability_2026-10-20_text

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_stability | azure_ai/ms-decision-model | decision_default | 1 | 0.362 | 2/5 | 2/5 | n/a | 128 | 64,078 | 0 | 511 | 40 | 0 |
| complex_stability | openai/gpt-5.6-luna | default | 3 | 0.967 (0.940-0.990) | 13.7/16 (13-14) | 4/5 | 0.728 (0.656-0.777) | 7 | 55,415 (55,199-55,540) | 37,446 (9,569-51,385) | 3,673 (3,429-3,856) | 39 (36-41) | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 1 | 0.414 | 3/5 | 3/5 | n/a | 128 | 127,173 | 0 | 0 | 31 | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.936 (0.896-0.979) | 14.7/16 (14-15) | 4.7/5 (4-5) | 0.802 (0.755-0.831) | 7 | 56,129 (55,555-56,771) | 33,636 (9,569-51,765) | 4,801 (4,259-5,522) | 47 (42-54) | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 1 | 0.375 | 3/5 | 3/5 | n/a | 128 | 92,806 | 0 | 9,241 | 30 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_stability | openai/gpt-5.6-luna | default | 3 | 6/6 (1.00) | 44/51 (0.86) | 18/18 (1.00) | 54 | 0.0012 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 47/51 (0.92) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 9 | 426 | 0.0649 |
| Judge | 7 | 184 | 0.0040 |
| Decision probabilities | 3 | 375 | 0.0180 |
| Total | | | 0.0868 |

## Concepts missed

- **complex_stability, azure_ai/ms-decision-model, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-5.6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x; Date on which V20-2604 was placed at the intermediate condition (DateConcept): 1x
- **complex_stability, openai/gpt-6-luna, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, typesafe/jev-latest, decision_default** (1 runs): The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.18 (P 0.13, R 0.29); Stability results of V20-2604: F1 0.38 (P 0.36, R 0.40); Results outside the shelf-life specification: F1 0.39 (P 0.33, R 0.46); Shelf-life rules and proposal: F1 0.50 (P 0.35, R 0.88)
- **complex_stability, openai/gpt-5.6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.87 (P 0.97, R 0.79); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.18 (P 0.25, R 0.14); Stability results of V20-2604: F1 0.33 (P 1.00, R 0.20); Results outside the shelf-life specification: F1 0.56 (P 1.00, R 0.38); Shelf-life rules and proposal: F1 0.58 (P 0.44, R 0.88)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.99 (P 0.98, R 1.00); Results outside the shelf-life specification: F1 0.86 (P 0.83, R 0.92); Shelf-life rules and proposal: F1 0.89 (P 0.82, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.13 (P 0.12, R 0.14); Stability results of V20-2604: F1 0.26 (P 1.00, R 0.15); Results outside the shelf-life specification: F1 0.47 (P 1.00, R 0.31); Shelf-life rules and proposal: F1 0.64 (P 0.50, R 0.88)

## Errors and warnings

- none
