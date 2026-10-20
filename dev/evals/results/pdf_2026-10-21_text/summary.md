# Extraction evaluation: pdf_2026-10-21_text

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0.000 | 0/26 | 0/11 | 0.000 | 4 | 151,780 | 0 | 2,038 (1,287-2,587) | 11 (7-12) | 9 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 1 | 0.560 | 10/11 | 10/11 | n/a | 275 | 282,621 | 0 | 1,903 | 83 | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 1 | 0.633 | 8/11 | 8/11 | n/a | 275 | 517,005 | 0 | 0 | 92 | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.936 (0.909-0.981) | 25.7/26 (25-26) | 10.7/11 (10-11) | 0.733 (0.724-0.739) | 10 | 102,258 (100,153-103,418) | 45,537 (13,670-61,470) | 15,053 (14,759-15,560) | 127 (120-133) | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 1 | 0.646 | 9/11 | 9/11 | n/a | 275 | 339,607 | 0 | 33,516 | 63 | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 0.903 (0.812-0.972) | 14.3/16 (14-15) | 4.3/5 (4-5) | 0.755 (0.725-0.775) | 8 (8-9) | 114,105 (110,718-120,022) | 0 | 9,593 (7,689-11,046) | 43 (36-49) | 4 |
| complex_stability | azure_ai/ms-decision-model | decision_default | 1 | 0.344 | 1/5 | 1/5 | n/a | 128 | 64,073 | 0 | 511 | 38 | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 1 | 0.414 | 3/5 | 3/5 | n/a | 128 | 127,173 | 0 | 0 | 29 | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.961 (0.951-0.982) | 15.7/16 (15-16) | 5/5 | 0.855 (0.840-0.880) | 7 | 56,073 (55,933-56,156) | 34,557 (8,202-47,734) | 5,184 (4,623-5,690) | 56 (49-62) | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 1 | 0.367 | 2/5 | 2/5 | n/a | 128 | 92,850 | 0 | 9,241 | 30 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0/3 (0.00) | n/a | n/a | 0 | 0.0000 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 77/78 (0.99) | 36/36 (1.00) | 80 | 0.0017 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 6/6 (1.00) | 45/51 (0.88) | 18/18 (1.00) | 54 | 0.0012 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 50/51 (0.98) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 18 | 1,297 | 0.2564 |
| Judge | 13 | 264 | 0.0057 |
| Decision probabilities | 6 | 1,179 | 0.0915 |
| Total | | | 0.3536 |

## Concepts missed

- **complex_mcu, anthropic/claude-haiku-5-5, default** (3 runs): Lowest HBM rating of a QFN-48 pin in V (NumericalConcept): 3x; Revision C BOR level 2 minimum falling threshold in V (NumericalConcept): 3x; VOS0 permitted at a VDD of 1.8 V (BooleanConcept): 3x; Marking code of revision C (LabelConcept): 3x; DR-115 junction temperature in °C (NumericalConcept): 3x; Start of the first revision C date code (DateConcept): 3x; Limitations present in revision C (JsonObjectConcept): 3x; Impact class of ES-08 (RatingConcept): 3x; ES-09 present in revision C (BooleanConcept): 3x; HCLK maximum for 2 wait states in VOS1 in MHz (NumericalConcept): 3x; Revision C retention after 10 000 cycles at 85 °C (NumericalConcept): 3x; Revision C retention after 1 000 cycles at 105 °C (NumericalConcept): 3x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 3x; DR-114 PLL1 configuration within specification (BooleanConcept): 3x; DR-114 required flash wait states (NumericalConcept): 3x; DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 3x; Errata applicable to the DR-114 production boards (LabelConcept): 3x; DR-114 maximum ADC INL in LSB (NumericalConcept): 3x; FW 2.3.0 meets HDR-9 for the DR-114 production boards (BooleanConcept): 3x; Errata applicable to the DR-115 boards (LabelConcept): 3x; DR-115 maximum PLL1 VCO frequency in MHz (NumericalConcept): 3x; DR-115 PLL1 configuration within its limits (BooleanConcept): 3x; Halyard last-time-buy date for revision B (DateConcept): 3x; Last-shipment date of revision B (DateConcept): 3x; PO-5521 valid last-time-buy order (BooleanConcept): 3x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (1 runs): Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x; PO-5521 valid last-time-buy order (BooleanConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 2x
- **complex_stability, azure_ai/ms-decision-model, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-6-luna, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 1x
- **complex_stability, typesafe/jev-latest, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.00 (P 0.00, R 0.00); Flash memory specifications: F1 0.00 (P 0.00, R 0.00); DR-114 clock configuration: F1 0.00 (P 0.00, R 0.00); DR-114 thermal assessment: F1 0.00 (P 0.00, R 0.00); DR-114 errata and firmware: F1 0.00 (P 0.00, R 0.00); DR-115 errata and PLL1 limits: F1 0.00 (P 0.00, R 0.00); Revision B end of life: F1 0.00 (P 0.00, R 0.00)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.38 (P 0.24, R 1.00); Flash memory specifications: F1 0.56 (P 0.50, R 0.64); DR-114 clock configuration: F1 0.64 (P 0.57, R 0.73); DR-114 thermal assessment: F1 0.48 (P 0.44, R 0.52); DR-114 errata and firmware: F1 0.56 (P 0.65, R 0.50); DR-115 errata and PLL1 limits: F1 0.43 (P 0.59, R 0.34); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.87 (P 0.91, R 0.83); Flash memory specifications: F1 0.64 (P 0.64, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.70, R 0.64); DR-114 thermal assessment: F1 0.37 (P 0.55, R 0.29); DR-114 errata and firmware: F1 0.58 (P 0.69, R 0.50); DR-115 errata and PLL1 limits: F1 0.37 (P 0.57, R 0.28); Revision B end of life: F1 0.92 (P 1.00, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.85 (P 0.76, R 1.00); Flash memory specifications: F1 0.98 (P 0.96, R 1.00); DR-114 clock configuration: F1 0.93 (P 0.97, R 0.89); DR-114 thermal assessment: F1 0.89 (P 0.81, R 1.00); DR-114 errata and firmware: F1 0.99 (P 1.00, R 0.98); DR-115 errata and PLL1 limits: F1 0.92 (P 0.93, R 0.92); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.92 (P 0.92, R 0.92); Flash memory specifications: F1 0.72 (P 0.82, R 0.64); DR-114 clock configuration: F1 0.65 (P 0.82, R 0.55); DR-114 thermal assessment: F1 0.44 (P 0.64, R 0.33); DR-114 errata and firmware: F1 0.67 (P 1.00, R 0.50); DR-115 errata and PLL1 limits: F1 0.41 (P 0.80, R 0.28); Revision B end of life: F1 0.71 (P 0.71, R 0.71)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 0.91 (P 0.86, R 1.00); Stability results of V20-2604: F1 0.96 (P 0.95, R 0.97); Results outside the shelf-life specification: F1 0.74 (P 0.95, R 0.64); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.10 (P 0.08, R 0.14); Stability results of V20-2604: F1 0.38 (P 0.36, R 0.40); Results outside the shelf-life specification: F1 0.37 (P 0.32, R 0.46); Shelf-life rules and proposal: F1 0.52 (P 0.37, R 0.88)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.18 (P 0.25, R 0.14); Stability results of V20-2604: F1 0.33 (P 1.00, R 0.20); Results outside the shelf-life specification: F1 0.56 (P 1.00, R 0.38); Shelf-life rules and proposal: F1 0.58 (P 0.44, R 0.88)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.95 (P 0.93, R 0.97); Shelf-life rules and proposal: F1 0.89 (P 0.82, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.12 (P 0.11, R 0.14); Stability results of V20-2604: F1 0.26 (P 1.00, R 0.15); Results outside the shelf-life specification: F1 0.44 (P 0.80, R 0.31); Shelf-life rules and proposal: F1 0.64 (P 0.50, R 0.88)

## Errors and warnings

- complex_mcu, anthropic/claude-haiku-5-5, default, run 1: error ValueError: Aspect `Device limitations of revision C` is not yet processed. Use `extract_aspects_from_document` first.`, 0 warning(s)
- complex_mcu, anthropic/claude-haiku-5-5, default, run 2: error ValueError: Aspect `Device limitations of revision C` is not yet processed. Use `extract_aspects_from_document` first.`, 0 warning(s)
- complex_mcu, anthropic/claude-haiku-5-5, default, run 3: error ValueError: Aspect `Device limitations of revision C` is not yet processed. Use `extract_aspects_from_document` first.`, 0 warning(s)
