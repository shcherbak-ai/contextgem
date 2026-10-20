# Extraction evaluation: stability_2026-10-20_docling

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_stability | azure_ai/ms-decision-model | decision_default | 1 | 0.648 | 1/5 | 1/5 | n/a | 128 | 89,439 | 0 | 511 | 42 | 0 |
| complex_stability | openai/gpt-5.6-luna | default | 3 | 0.983 (0.979-0.990) | 13.7/16 (13-14) | 4.3/5 (4-5) | 0.864 (0.831-0.894) | 7 | 76,635 (76,183-76,913) | 47,717 (9,569-72,146) | 3,739 (3,566-4,006) | 39 (36-44) | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 1 | 0.742 | 5/5 | 5/5 | n/a | 128 | 152,560 | 4,120 | 0 | 29 | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.978 (0.972-0.991) | 15.7/16 (15-16) | 5/5 | 0.871 (0.844-0.921) | 7 | 77,909 (77,460-78,145) | 48,688 (9,569-68,248) | 5,258 (4,566-5,702) | 52 (46-57) | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 1 | 0.760 | 2/5 | 2/5 | n/a | 128 | 117,345 | 0 | 9,241 | 27 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_stability | openai/gpt-5.6-luna | default | 3 | 6/6 (1.00) | 44/51 (0.86) | 18/18 (1.00) | 54 | 0.0012 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 50/51 (0.98) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 9 | 426 | 0.0822 |
| Judge | 7 | 184 | 0.0040 |
| Decision probabilities | 3 | 375 | 0.0223 |
| Total | | | 0.1085 |

## Concepts missed

- **complex_stability, azure_ai/ms-decision-model, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-5.6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Shelf life supported for the Alu/Alu blister by the rules of section 6 in months (NumericalConcept): 2x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 2x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 1x
- **complex_stability, typesafe/jev-latest, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.61 (P 0.44, R 1.00); Stability results of V20-2604: F1 0.98 (P 0.95, R 1.00); Results outside the shelf-life specification: F1 0.42 (P 0.28, R 0.85); Shelf-life rules and proposal: F1 0.59 (P 0.42, R 1.00)
- **complex_stability, openai/gpt-5.6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.93 (P 1.00, R 0.87); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.82 (P 0.70, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.59 (P 0.57, R 0.62); Shelf-life rules and proposal: F1 0.55 (P 0.38, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.95 (P 0.93, R 0.97); Shelf-life rules and proposal: F1 0.96 (P 0.93, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.64 (P 0.47, R 1.00); Stability results of V20-2604: F1 0.97 (P 1.00, R 0.95); Results outside the shelf-life specification: F1 0.67 (P 0.73, R 0.62); Shelf-life rules and proposal: F1 0.76 (P 0.62, R 1.00)

## Errors and warnings

- none
