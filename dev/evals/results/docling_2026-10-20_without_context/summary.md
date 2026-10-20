# Extraction evaluation: docling_2026-10-20_without_context

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_mcu | azure_ai/ms-decision-model | decision_default | 1 | 0.624 | 9/11 | 9/11 | n/a | 275 | 289,995 | 0 | 1,903 | 94 | 0 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 0.939 (0.927-0.961) | 24/26 | 10.3/11 (10-11) | 0.657 (0.623-0.684) | 10 | 108,381 (108,061-108,771) | 46,880 (13,670-65,286) | 9,523 (8,989-10,381) | 98 (92-102) | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 1 | 0.659 | 9/11 | 9/11 | n/a | 275 | 519,668 | 12,207 | 0 | 82 | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.959 (0.944-0.967) | 25.7/26 (25-26) | 10.7/11 (10-11) | 0.743 (0.728-0.752) | 10 | 107,664 (107,077-108,528) | 48,081 (13,670-65,286) | 14,219 (13,542-14,770) | 145 (136-150) | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 1 | 0.683 | 9/11 | 9/11 | n/a | 275 | 346,466 | 0 | 33,516 | 62 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 3/3 (1.00) | 66/76 (0.87) | 36/36 (1.00) | 76 | 0.0016 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 75/78 (0.96) | 36/36 (1.00) | 78 | 0.0017 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 9 | 885 | 0.2070 |
| Judge | 7 | 216 | 0.0046 |
| Decision probabilities | 3 | 804 | 0.0736 |
| Total | | | 0.2852 |

## Concepts missed

- **complex_mcu, azure_ai/ms-decision-model, decision_default** (1 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-5.6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 2x; DR-114 maximum ADC INL in LSB (NumericalConcept): 2x; Start of the first revision C date code (DateConcept): 1x; Last-shipment date of revision B (DateConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (1 runs): Errata applicable to the DR-115 boards (LabelConcept): 1x; DR-115 PLL1 configuration within its limits (BooleanConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.33 (P 0.20, R 1.00); Flash memory specifications: F1 0.74 (P 0.58, R 1.00); DR-114 clock configuration: F1 0.71 (P 0.59, R 0.91); DR-114 thermal assessment: F1 0.55 (P 0.43, R 0.76); DR-114 errata and firmware: F1 0.70 (P 0.71, R 0.68); DR-115 errata and PLL1 limits: F1 0.48 (P 0.65, R 0.38); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-5.6-luna, default**: Device limitations of revision C: F1 0.91 (P 0.84, R 1.00); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.92 (P 0.97, R 0.88); DR-114 thermal assessment: F1 0.85 (P 0.80, R 0.92); DR-114 errata and firmware: F1 0.96 (P 0.94, R 0.97); DR-115 errata and PLL1 limits: F1 0.93 (P 0.90, R 0.97); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.82 (P 0.90, R 0.75); Flash memory specifications: F1 0.64 (P 0.64, R 0.64); DR-114 clock configuration: F1 0.71 (P 0.68, R 0.76); DR-114 thermal assessment: F1 0.42 (P 0.47, R 0.38); DR-114 errata and firmware: F1 0.70 (P 0.87, R 0.59); DR-115 errata and PLL1 limits: F1 0.39 (P 0.67, R 0.28); Revision B end of life: F1 0.92 (P 1.00, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.94 (P 0.89, R 1.00); Flash memory specifications: F1 0.99 (P 0.98, R 1.00); DR-114 clock configuration: F1 0.91 (P 0.97, R 0.87); DR-114 thermal assessment: F1 0.95 (P 0.91, R 0.98); DR-114 errata and firmware: F1 0.98 (P 0.97, R 0.98); DR-115 errata and PLL1 limits: F1 0.95 (P 0.90, R 1.00); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.88 (P 0.85, R 0.92); Flash memory specifications: F1 0.90 (P 0.87, R 0.93); DR-114 clock configuration: F1 0.76 (P 0.74, R 0.79); DR-114 thermal assessment: F1 0.41 (P 0.39, R 0.43); DR-114 errata and firmware: F1 0.67 (P 1.00, R 0.50); DR-115 errata and PLL1 limits: F1 0.45 (P 0.82, R 0.31); Revision B end of life: F1 0.71 (P 0.71, R 0.71)

## Errors and warnings

- none
