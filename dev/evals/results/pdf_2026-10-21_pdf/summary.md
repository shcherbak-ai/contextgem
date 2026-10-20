# Extraction evaluation: pdf_2026-10-21_pdf

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0.476 (0.000-0.767) | 12.0/26 (0-21) | 4.7/11 (0-8) | 0.369 (0.000-0.625) | 10 (4-13) | 320,570 (223,048-401,161) | 0 | 12,206 (2,554-17,876) | 60 (13-86) | 8 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 1 | 0.645 | 8/11 | 8/11 | n/a | 278 | 365,420 | 0 | 1,924 | 83 | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 1 | 0.730 | 9/11 | 9/11 | n/a | 278 | 606,507 | 0 | 0 | 100 | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.950 (0.944-0.955) | 25.7/26 (25-26) | 10.7/11 (10-11) | 0.740 (0.717-0.763) | 10 | 134,825 (134,324-135,615) | 60,478 (13,670-84,524) | 13,871 (12,982-14,433) | 146 (138-155) | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 1 | 0.725 | 9/11 | 9/11 | n/a | 278 | 423,583 | 0 | 33,885 | 68 | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 0.917 (0.901-0.940) | 14/16 | 4/5 | 0.769 (0.735-0.808) | 7 | 123,848 (123,716-123,932) | 0 | 6,361 (5,430-8,033) | 31 (26-38) | 0 |
| complex_stability | azure_ai/ms-decision-model | decision_default | 1 | 0.619 | 1/5 | 1/5 | n/a | 130 | 90,924 | 0 | 519 | 39 | 0 |
| complex_stability | openai/gpt-6-luna | decision_default | 1 | 0.726 | 2/5 | 2/5 | n/a | 130 | 155,163 | 0 | 0 | 43 | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.958 (0.928-0.975) | 14.7/16 (14-15) | 4.7/5 (4-5) | 0.860 (0.818-0.890) | 7 | 77,606 (77,466-77,844) | 44,249 (9,569-61,589) | 5,040 (4,722-5,638) | 57 (51-61) | 0 |
| complex_stability | typesafe/jev-latest | decision_default | 1 | 0.760 | 2/5 | 2/5 | n/a | 130 | 118,949 | 0 | 9,385 | 29 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 2/3 (0.67) | 38/40 (0.95) | 24/24 (1.00) | 41 | 0.0009 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 76/78 (0.97) | 36/36 (1.00) | 80 | 0.0017 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 6/6 (1.00) | 44/51 (0.86) | 18/18 (1.00) | 51 | 0.0011 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 47/51 (0.92) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 18 | 1,325 | 0.3512 |
| Judge | 13 | 302 | 0.0065 |
| Decision probabilities | 6 | 1,194 | 0.1111 |
| Total | | | 0.4689 |

## Concepts missed

- **complex_mcu, anthropic/claude-haiku-5-5, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 3x; Errata applicable to the DR-114 production boards (LabelConcept): 3x; Errata applicable to the DR-115 boards (LabelConcept): 3x; Revision C retention after 1 000 cycles at 105 °C (NumericalConcept): 2x; Lowest HBM rating of a QFN-48 pin in V (NumericalConcept): 2x; Revision C BOR level 2 minimum falling threshold in V (NumericalConcept): 2x; VOS0 permitted at a VDD of 1.8 V (BooleanConcept): 2x; Marking code of revision C (LabelConcept): 2x; DR-115 junction temperature in °C (NumericalConcept): 2x; Start of the first revision C date code (DateConcept): 2x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 2x; Limitations present in revision C (JsonObjectConcept): 1x; Impact class of ES-08 (RatingConcept): 1x; ES-09 present in revision C (BooleanConcept): 1x; HCLK maximum for 2 wait states in VOS1 in MHz (NumericalConcept): 1x; Revision C retention after 10 000 cycles at 85 °C (NumericalConcept): 1x; DR-114 PLL1 configuration within specification (BooleanConcept): 1x; DR-114 required flash wait states (NumericalConcept): 1x; DR-114 maximum ADC INL in LSB (NumericalConcept): 1x; FW 2.3.0 meets HDR-9 for the DR-114 production boards (BooleanConcept): 1x; DR-115 maximum PLL1 VCO frequency in MHz (NumericalConcept): 1x; DR-115 PLL1 configuration within its limits (BooleanConcept): 1x; Halyard last-time-buy date for revision B (DateConcept): 1x; Last-shipment date of revision B (DateConcept): 1x; PO-5521 valid last-time-buy order (BooleanConcept): 1x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (1 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x
- **complex_stability, azure_ai/ms-decision-model, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, openai/gpt-6-luna, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Tests with a result outside the shelf-life specification (LabelConcept): 1x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x
- **complex_stability, typesafe/jev-latest, decision_default** (1 runs): Batches with a significant change at the accelerated condition (LabelConcept): 1x; The conclusion on V20-2607 at the accelerated condition agrees with its results (BooleanConcept): 1x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.51 (P 0.42, R 0.67); Flash memory specifications: F1 0.64 (P 0.62, R 0.67); DR-114 clock configuration: F1 0.49 (P 0.46, R 0.54); DR-114 thermal assessment: F1 0.46 (P 0.36, R 0.63); DR-114 errata and firmware: F1 0.45 (P 0.49, R 0.42); DR-115 errata and PLL1 limits: F1 0.32 (P 0.41, R 0.26); Revision B end of life: F1 0.46 (P 0.36, R 0.67)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.43 (P 0.27, R 1.00); Flash memory specifications: F1 0.85 (P 0.74, R 1.00); DR-114 clock configuration: F1 0.75 (P 0.69, R 0.82); DR-114 thermal assessment: F1 0.58 (P 0.47, R 0.76); DR-114 errata and firmware: F1 0.57 (P 0.77, R 0.45); DR-115 errata and PLL1 limits: F1 0.51 (P 0.79, R 0.38); Revision B end of life: F1 0.82 (P 0.70, R 1.00)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.80 (P 0.77, R 0.83); Flash memory specifications: F1 0.93 (P 0.88, R 1.00); DR-114 clock configuration: F1 0.77 (P 0.70, R 0.85); DR-114 thermal assessment: F1 0.47 (P 0.40, R 0.57); DR-114 errata and firmware: F1 0.78 (P 0.75, R 0.82); DR-115 errata and PLL1 limits: F1 0.48 (P 0.65, R 0.38); Revision B end of life: F1 0.88 (P 0.78, R 1.00)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.97 (P 0.95, R 1.00); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.95 (P 0.98, R 0.93); DR-114 thermal assessment: F1 0.93 (P 0.89, R 0.98); DR-114 errata and firmware: F1 0.95 (P 0.97, R 0.94); DR-115 errata and PLL1 limits: F1 0.84 (P 0.74, R 0.99); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.96 (P 0.92, R 1.00); Flash memory specifications: F1 0.93 (P 0.88, R 1.00); DR-114 clock configuration: F1 0.85 (P 0.93, R 0.79); DR-114 thermal assessment: F1 0.59 (P 0.48, R 0.76); DR-114 errata and firmware: F1 0.53 (P 1.00, R 0.36); DR-115 errata and PLL1 limits: F1 0.43 (P 1.00, R 0.28); Revision B end of life: F1 0.77 (P 0.83, R 0.71)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.99 (P 1.00, R 0.98); Results outside the shelf-life specification: F1 0.68 (P 1.00, R 0.51); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, azure_ai/ms-decision-model, decision_default**: Shelf-life acceptance criteria: F1 0.56 (P 0.39, R 1.00); Stability results of V20-2604: F1 0.98 (P 0.95, R 1.00); Results outside the shelf-life specification: F1 0.41 (P 0.27, R 0.85); Shelf-life rules and proposal: F1 0.53 (P 0.36, R 1.00)
- **complex_stability, openai/gpt-6-luna, decision_default**: Shelf-life acceptance criteria: F1 0.88 (P 0.78, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.57 (P 0.53, R 0.62); Shelf-life rules and proposal: F1 0.46 (P 0.30, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.95 (P 0.91, R 1.00); Results outside the shelf-life specification: F1 0.92 (P 0.97, R 0.87); Shelf-life rules and proposal: F1 0.96 (P 0.93, R 1.00)
- **complex_stability, typesafe/jev-latest, decision_default**: Shelf-life acceptance criteria: F1 0.64 (P 0.47, R 1.00); Stability results of V20-2604: F1 0.97 (P 1.00, R 0.95); Results outside the shelf-life specification: F1 0.67 (P 0.88, R 0.54); Shelf-life rules and proposal: F1 0.76 (P 0.62, R 1.00)

## Errors and warnings

- complex_mcu, anthropic/claude-haiku-5-5, default, run 2: error ValueError: Aspect `Device limitations of revision C` is not yet processed. Use `extract_aspects_from_document` first.`, 0 warning(s)
