# Extraction evaluation: docling_2026-10-20_text

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_dpa | azure_ai/ms-decision-model | decision_default | 1 | 0.763 | 5/6 | 5/6 | n/a | 157 | 109,264 | 0 | 934 | 50 | 0 |
| complex_dpa | openai/gpt-5.6-luna | default | 3 | 0.959 (0.955-0.966) | 21.3/22 (21-22) | 6/6 | 0.815 (0.795-0.849) | 7 | 57,662 (57,279-58,298) | 29,214 (9,569-39,036) | 5,802 (5,115-6,414) | 57 (50-65) | 0 |
| complex_dpa | openai/gpt-6-luna | decision_default | 1 | 0.823 | 4/6 | 4/6 | n/a | 157 | 222,536 | 0 | 0 | 47 | 0 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 0.977 (0.973-0.981) | 21.7/22 (21-22) | 6/6 | 0.837 (0.793-0.861) | 8 (7-8) | 64,245 (57,253-67,908) | 39,168 (20,121-53,968) | 8,558 (7,015-9,470) | 86 (70-94) | 2 |
| complex_dpa | typesafe/jev-latest | decision_default | 1 | 0.793 | 5/6 | 5/6 | n/a | 157 | 144,022 | 0 | 16,545 | 35 | 0 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 1 | 0.555 | 9/11 | 9/11 | n/a | 275 | 282,452 | 0 | 1,903 | 92 | 0 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 0.917 (0.900-0.934) | 22.7/26 (22-23) | 9.7/11 (9-10) | 0.616 (0.591-0.646) | 10 | 102,248 (101,319-103,578) | 44,336 (13,670-61,470) | 9,349 (8,627-10,768) | 100 (92-114) | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 1 | 0.633 | 8/11 | 8/11 | n/a | 275 | 517,005 | 0 | 0 | 84 | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.953 (0.944-0.958) | 25.7/26 (25-26) | 11/11 | 0.716 (0.674-0.751) | 10 | 101,213 (99,145-103,525) | 45,537 (17,272-61,470) | 14,878 (14,181-15,506) | 156 (143-164) | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 1 | 0.623 | 8/11 | 8/11 | n/a | 275 | 339,697 | 0 | 33,516 | 61 | 0 |
| complex_nda | azure_ai/ms-decision-model | decision_default | 1 | 0.779 | 4/6 | 4/6 | n/a | 76 | 50,742 | 0 | 312 | 26 | 0 |
| complex_nda | openai/gpt-5.6-luna | default | 3 | 1.000 | 17.7/18 (17-18) | 6/6 | 0.836 (0.830-0.839) | 6 | 48,264 (48,248-48,297) | 33,223 (8,202-48,230) | 5,162 (4,997-5,390) | 51 (46-54) | 0 |
| complex_nda | openai/gpt-6-luna | decision_default | 1 | 0.892 | 4/6 | 4/6 | n/a | 76 | 86,263 | 0 | 0 | 23 | 0 |
| complex_nda | openai/gpt-6-luna | default | 3 | 0.992 (0.985-1.000) | 18/18 | 6/6 | 0.861 (0.846-0.876) | 6 (6-7) | 50,628 (48,029-55,412) | 28,348 (8,202-44,500) | 6,056 (5,538-6,906) | 61 (56-68) | 1 |
| complex_nda | typesafe/jev-latest | decision_default | 1 | 0.822 | 3/6 | 3/6 | n/a | 76 | 65,315 | 0 | 5,652 | 17 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_dpa | openai/gpt-5.6-luna | default | 3 | 5/6 (0.83) | 64/66 (0.97) | 24/24 (1.00) | 77 | 0.0016 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 3/6 (0.50) | 61/66 (0.92) | 24/24 (1.00) | 69 | 0.0014 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 3/3 (1.00) | 64/72 (0.89) | 36/36 (1.00) | 72 | 0.0015 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 76/78 (0.97) | 36/36 (1.00) | 78 | 0.0017 |
| complex_nda | openai/gpt-5.6-luna | default | 3 | 8/9 (0.89) | 61/63 (0.97) | 12/12 (1.00) | 69 | 0.0015 |
| complex_nda | openai/gpt-6-luna | default | 3 | 9/9 (1.00) | 63/63 (1.00) | 12/12 (1.00) | 70 | 0.0015 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 27 | 1,665 | 0.3660 |
| Judge | 19 | 497 | 0.0105 |
| Decision probabilities | 9 | 1,479 | 0.1158 |
| Total | | | 0.4924 |

## Concepts missed

- **complex_dpa, azure_ai/ms-decision-model, decision_default** (1 runs): Applicable initial notification regime (LabelConcept): 1x
- **complex_dpa, openai/gpt-5.6-luna, default** (3 runs): Distinct data locations count as of 1 February 2027 (NumericalConcept): 2x
- **complex_dpa, openai/gpt-6-luna, decision_default** (1 runs): Meridian Mail approved on 1 February 2027 (BooleanConcept): 1x; Permitted uses of held ORD records on 1 September 2027 (LabelConcept): 1x
- **complex_dpa, openai/gpt-6-luna, default** (3 runs): Distinct data locations count as of 1 February 2027 (NumericalConcept): 1x
- **complex_dpa, typesafe/jev-latest, decision_default** (1 runs): Authorised data locations on 15 May 2027 (LabelConcept): 1x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (1 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-5.6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 3x; DR-114 maximum ADC INL in LSB (NumericalConcept): 3x; DR-114 junction temperature in °C (NumericalConcept): 2x; DR-115 junction temperature in °C (NumericalConcept): 1x; DR-114 meets HDR-7 (BooleanConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x; PO-5521 valid last-time-buy order (BooleanConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x; DR-115 PLL1 configuration within its limits (BooleanConcept): 1x
- **complex_nda, azure_ai/ms-decision-model, decision_default** (1 runs): Protected assets at the April snapshot (LabelConcept): 1x; People permitted LUMEN access at the September snapshot (LabelConcept): 1x
- **complex_nda, openai/gpt-5.6-luna, default** (3 runs): WORK-1 deletion deadline (DateConcept): 1x
- **complex_nda, openai/gpt-6-luna, decision_default** (1 runs): Protected assets at the April snapshot (LabelConcept): 1x; Artifacts permitted retention at the December snapshot (LabelConcept): 1x
- **complex_nda, typesafe/jev-latest, decision_default** (1 runs): Protected assets at the April snapshot (LabelConcept): 1x; People permitted LUMEN access at the September snapshot (LabelConcept): 1x; Artifacts permitted retention at the December snapshot (LabelConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_dpa, azure_ai/ms-decision-model, decision_default**: Amendment effectiveness and precedence: F1 0.77 (P 0.62, R 1.00); Personal data breach notification: F1 0.76 (P 0.62, R 1.00); Incident IR-27: F1 0.59 (P 0.43, R 0.93); Retention TR-27: F1 0.65 (P 0.48, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 1.00, R 0.95); Liability LC-27: F1 0.83 (P 0.71, R 1.00)
- **complex_dpa, openai/gpt-5.6-luna, default**: Amendment effectiveness and precedence: F1 0.83 (P 0.87, R 0.80); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 0.98 (P 0.96, R 1.00); Retention TR-27: F1 0.99 (P 1.00, R 0.97); Supplier approvals in May 2027: F1 0.98 (P 0.97, R 0.98); Liability LC-27: F1 0.98 (P 0.97, R 1.00)
- **complex_dpa, openai/gpt-6-luna, decision_default**: Amendment effectiveness and precedence: F1 0.80 (P 0.80, R 0.80); Personal data breach notification: F1 0.89 (P 0.80, R 1.00); Incident IR-27: F1 0.74 (P 0.77, R 0.71); Retention TR-27: F1 0.82 (P 0.90, R 0.75); Supplier approvals in May 2027: F1 0.74 (P 0.93, R 0.62); Liability LC-27: F1 0.95 (P 1.00, R 0.90)
- **complex_dpa, openai/gpt-6-luna, default**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 1.00 (P 1.00, R 1.00); Retention TR-27: F1 1.00 (P 1.00, R 1.00); Supplier approvals in May 2027: F1 0.99 (P 0.98, R 1.00); Liability LC-27: F1 0.98 (P 1.00, R 0.97)
- **complex_dpa, typesafe/jev-latest, decision_default**: Amendment effectiveness and precedence: F1 0.67 (P 0.50, R 1.00); Personal data breach notification: F1 0.80 (P 0.67, R 1.00); Incident IR-27: F1 0.74 (P 0.62, R 0.93); Retention TR-27: F1 0.78 (P 0.82, R 0.75); Supplier approvals in May 2027: F1 0.76 (P 1.00, R 0.62); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.39 (P 0.24, R 1.00); Flash memory specifications: F1 0.58 (P 0.53, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.60, R 0.76); DR-114 thermal assessment: F1 0.45 (P 0.43, R 0.48); DR-114 errata and firmware: F1 0.55 (P 0.61, R 0.50); DR-115 errata and PLL1 limits: F1 0.39 (P 0.53, R 0.31); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-5.6-luna, default**: Device limitations of revision C: F1 0.91 (P 0.84, R 1.00); Flash memory specifications: F1 0.99 (P 0.98, R 1.00); DR-114 clock configuration: F1 0.91 (P 0.95, R 0.87); DR-114 thermal assessment: F1 0.79 (P 0.70, R 0.90); DR-114 errata and firmware: F1 0.95 (P 0.97, R 0.94); DR-115 errata and PLL1 limits: F1 0.89 (P 0.86, R 0.94); Revision B end of life: F1 0.98 (P 0.96, R 1.00)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.87 (P 0.91, R 0.83); Flash memory specifications: F1 0.64 (P 0.64, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.70, R 0.64); DR-114 thermal assessment: F1 0.37 (P 0.55, R 0.29); DR-114 errata and firmware: F1 0.58 (P 0.69, R 0.50); DR-115 errata and PLL1 limits: F1 0.37 (P 0.57, R 0.28); Revision B end of life: F1 0.92 (P 1.00, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.90 (P 0.83, R 1.00); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.93 (P 0.98, R 0.88); DR-114 thermal assessment: F1 0.95 (P 0.93, R 0.98); DR-114 errata and firmware: F1 0.96 (P 0.97, R 0.95); DR-115 errata and PLL1 limits: F1 0.93 (P 0.92, R 0.95); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.83 (P 0.83, R 0.83); Flash memory specifications: F1 0.72 (P 0.82, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.86, R 0.55); DR-114 thermal assessment: F1 0.44 (P 0.64, R 0.33); DR-114 errata and firmware: F1 0.58 (P 1.00, R 0.41); DR-115 errata and PLL1 limits: F1 0.41 (P 0.80, R 0.28); Revision B end of life: F1 0.71 (P 0.71, R 0.71)
- **complex_nda, azure_ai/ms-decision-model, decision_default**: NDA information protection: F1 0.64 (P 0.47, R 1.00); NDA recipient permissions: F1 0.91 (P 0.83, R 1.00); NDA compulsory disclosure Q-27: F1 0.75 (P 0.60, R 1.00); NDA exit obligations X-27: F1 0.81 (P 0.72, R 0.93)
- **complex_nda, openai/gpt-5.6-luna, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 1.00 (P 1.00, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, openai/gpt-6-luna, decision_default**: NDA information protection: F1 0.82 (P 0.88, R 0.78); NDA recipient permissions: F1 0.97 (P 1.00, R 0.93); NDA compulsory disclosure Q-27: F1 0.90 (P 0.82, R 1.00); NDA exit obligations X-27: F1 0.88 (P 1.00, R 0.79)
- **complex_nda, openai/gpt-6-luna, default**: NDA information protection: F1 0.99 (P 0.98, R 1.00); NDA recipient permissions: F1 0.98 (P 0.96, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, typesafe/jev-latest, decision_default**: NDA information protection: F1 0.68 (P 0.61, R 0.78); NDA recipient permissions: F1 0.93 (P 1.00, R 0.87); NDA compulsory disclosure Q-27: F1 0.84 (P 0.80, R 0.89); NDA exit obligations X-27: F1 0.83 (P 1.00, R 0.71)

## Errors and warnings

- none
