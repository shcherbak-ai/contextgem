# Extraction evaluation: pdf_2026-10-21_docling

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0.757 (0.702-0.787) | 21.7/26 (20-23) | 8.3/11 (8-9) | 0.670 (0.642-0.686) | 11 (10-12) | 280,429 (239,690-305,413) | 0 | 18,596 (17,979-19,814) | 87 (84-91) | 3 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 1 | 0.703 | 8/11 | 8/11 | n/a | 275 | 363,191 | 0 | 1,903 | 84 | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 1 | 0.749 | 10/11 | 10/11 | n/a | 275 | 600,881 | 0 | 0 | 90 | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.956 (0.941-0.965) | 23.3/26 (22-25) | 10.0/11 (9-11) | 0.673 (0.664-0.678) | 10 | 142,086 (140,480-144,603) | 69,071 (13,670-103,091) | 13,320 (13,173-13,531) | 136 (131-142) | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 1 | 0.690 | 9/11 | 9/11 | n/a | 275 | 421,574 | 0 | 33,516 | 64 | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 0.870 (0.869-0.871) | 14/16 | 4/5 | 0.714 (0.673-0.770) | 7 | 124,918 (124,352-126,051) | 0 | 5,684 (5,496-5,951) | 27 (25-30) | 0 |
| complex_stability | azure_ai/ms-decision-model | decision_default | 1 | 0.641 | 1/5 | 1/5 | n/a | 128 | 89,604 | 0 | 511 | 39 | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 1 | 0.742 | 5/5 | 5/5 | n/a | 128 | 152,560 | 0 | 0 | 27 | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.962 (0.945-0.990) | 15.3/16 (15-16) | 5/5 | 0.865 (0.832-0.922) | 7 | 77,890 (77,325-78,439) | 45,917 (9,569-66,518) | 4,597 (4,161-5,032) | 50 (44-54) | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 1 | 0.770 | 1/5 | 1/5 | n/a | 128 | 117,251 | 0 | 9,241 | 30 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 3/3 (1.00) | 66/70 (0.94) | 36/36 (1.00) | 70 | 0.0015 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 68/74 (0.92) | 36/36 (1.00) | 77 | 0.0016 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 6/6 (1.00) | 45/51 (0.88) | 18/18 (1.00) | 53 | 0.0012 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 48/51 (0.94) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 18 | 1,314 | 0.3447 |
| Judge | 13 | 330 | 0.0071 |
| Decision probabilities | 6 | 1,179 | 0.1103 |
| Total | | | 0.4621 |

## Concepts missed

- **complex_mcu, anthropic/claude-haiku-5-5, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 3x; Errata applicable to the DR-114 production boards (LabelConcept): 3x; Errata applicable to the DR-115 boards (LabelConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 2x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; DR-114 maximum ADC INL in LSB (NumericalConcept): 1x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (1 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (1 runs): Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 2x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; Last-shipment date of revision B (DateConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x
- **complex_stability, azure_ai/ms-decision-model, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 2x
- **complex_stability, typesafe/jev-latest, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.80 (P 0.68, R 1.00); Flash memory specifications: F1 0.94 (P 0.88, R 1.00); DR-114 clock configuration: F1 0.82 (P 0.85, R 0.79); DR-114 thermal assessment: F1 0.71 (P 0.60, R 0.86); DR-114 errata and firmware: F1 0.72 (P 0.85, R 0.67); DR-115 errata and PLL1 limits: F1 0.51 (P 0.77, R 0.38); Revision B end of life: F1 0.81 (P 0.68, R 1.00)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.75 (P 0.60, R 1.00); Flash memory specifications: F1 0.88 (P 0.78, R 1.00); DR-114 clock configuration: F1 0.78 (P 0.72, R 0.85); DR-114 thermal assessment: F1 0.62 (P 0.50, R 0.81); DR-114 errata and firmware: F1 0.56 (P 0.71, R 0.45); DR-115 errata and PLL1 limits: F1 0.55 (P 0.80, R 0.41); Revision B end of life: F1 0.80 (P 0.75, R 0.86)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.83 (P 0.83, R 0.83); Flash memory specifications: F1 0.90 (P 0.82, R 1.00); DR-114 clock configuration: F1 0.76 (P 0.68, R 0.85); DR-114 thermal assessment: F1 0.48 (P 0.44, R 0.52); DR-114 errata and firmware: F1 0.88 (P 0.90, R 0.86); DR-115 errata and PLL1 limits: F1 0.53 (P 0.75, R 0.41); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 1.00 (P 1.00, R 1.00); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.96 (P 0.99, R 0.94); DR-114 thermal assessment: F1 0.92 (P 0.92, R 0.92); DR-114 errata and firmware: F1 0.93 (P 0.98, R 0.88); DR-115 errata and PLL1 limits: F1 0.89 (P 0.83, R 0.95); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.92 (P 0.92, R 0.92); Flash memory specifications: F1 0.97 (P 0.93, R 1.00); DR-114 clock configuration: F1 0.81 (P 0.96, R 0.70); DR-114 thermal assessment: F1 0.40 (P 0.42, R 0.38); DR-114 errata and firmware: F1 0.58 (P 1.00, R 0.41); DR-115 errata and PLL1 limits: F1 0.39 (P 1.00, R 0.24); Revision B end of life: F1 0.77 (P 0.83, R 0.71)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 0.96 (P 0.92, R 1.00); Stability results of V20-2604: F1 0.93 (P 0.97, R 0.90); Results outside the shelf-life specification: F1 0.59 (P 0.87, R 0.46); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.61 (P 0.44, R 1.00); Stability results of V20-2604: F1 0.95 (P 0.91, R 1.00); Results outside the shelf-life specification: F1 0.39 (P 0.25, R 0.85); Shelf-life rules and proposal: F1 0.62 (P 0.44, R 1.00)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.82 (P 0.70, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.59 (P 0.57, R 0.62); Shelf-life rules and proposal: F1 0.55 (P 0.38, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.99 (P 0.98, R 1.00); Results outside the shelf-life specification: F1 0.96 (P 0.98, R 0.95); Shelf-life rules and proposal: F1 0.89 (P 0.82, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.64 (P 0.47, R 1.00); Stability results of V20-2604: F1 0.95 (P 1.00, R 0.90); Results outside the shelf-life specification: F1 0.70 (P 0.80, R 0.62); Shelf-life rules and proposal: F1 0.80 (P 0.67, R 1.00)

## Errors and warnings

- none
