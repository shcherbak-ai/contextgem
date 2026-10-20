# Extraction evaluation: stability_2026-10-20_without_context

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_stability | azure_ai/ms-decision-model | decision_default | 1 | 0.446 | 1/5 | 1/5 | n/a | 128 | 66,664 | 0 | 511 | 39 | 0 |
| complex_stability | openai/gpt-5.6-luna | default | 3 | 0.980 (0.972-0.990) | 14.3/16 (14-15) | 4/5 | 0.829 (0.799-0.852) | 7 | 58,203 (58,121-58,245) | 35,214 (9,569-50,732) | 3,767 (3,686-3,864) | 40 (38-43) | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 1 | 0.437 | 5/5 | 5/5 | n/a | 128 | 128,932 | 0 | 0 | 30 | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.977 (0.951-0.990) | 14.7/16 (14-15) | 4.7/5 (4-5) | 0.801 (0.669-0.872) | 7 | 58,581 (58,442-58,806) | 37,818 (9,569-53,470) | 4,959 (4,573-5,291) | 49 (45-51) | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 1 | 0.392 | 1/5 | 1/5 | n/a | 128 | 94,674 | 0 | 9,241 | 28 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_stability | openai/gpt-5.6-luna | default | 3 | 6/6 (1.00) | 45/51 (0.88) | 18/18 (1.00) | 54 | 0.0012 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 47/51 (0.92) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 9 | 426 | 0.0690 |
| Judge | 7 | 184 | 0.0040 |
| Decision probabilities | 3 | 375 | 0.0181 |
| Total | | | 0.0910 |

## Concepts missed

- **complex_stability, azure_ai/ms-decision-model, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-5.6-luna, default** (3 runs): Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x; Number of results outside the shelf-life specification (NumericalConcept): 2x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, typesafe/jev-latest, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.40 (P 0.26, R 0.86); Stability results of V20-2604: F1 0.54 (P 0.37, R 1.00); Results outside the shelf-life specification: F1 0.44 (P 0.31, R 0.77); Shelf-life rules and proposal: F1 0.40 (P 0.26, R 0.88)
- **complex_stability, openai/gpt-5.6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.92 (P 0.95, R 0.90); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.29 (P 0.29, R 0.29); Stability results of V20-2604: F1 0.33 (P 1.00, R 0.20); Results outside the shelf-life specification: F1 0.55 (P 0.67, R 0.46); Shelf-life rules and proposal: F1 0.58 (P 0.44, R 0.88)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.96 (P 1.00, R 0.92); Shelf-life rules and proposal: F1 0.95 (P 0.91, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.12 (P 0.11, R 0.14); Stability results of V20-2604: F1 0.43 (P 0.47, R 0.40); Results outside the shelf-life specification: F1 0.40 (P 0.57, R 0.31); Shelf-life rules and proposal: F1 0.61 (P 0.47, R 0.88)

## Errors and warnings

- none
