# Extraction evaluation: docling_2026-10-20_docling

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_dpa | azure_ai/ms-decision-model | decision_default | 1 | 0.776 | 5/6 | 5/6 | n/a | 157 | 144,788 | 0 | 934 | 50 | 0 |
| complex_dpa | openai/gpt-5.6-luna | default | 3 | 0.953 (0.910-0.981) | 21.7/22 (21-22) | 6/6 | 0.834 (0.821-0.858) | 7 | 64,687 (64,085-65,631) | 32,029 (9,569-43,259) | 6,285 (5,811-6,547) | 62 (60-64) | 0 |
| complex_dpa | openai/gpt-6-luna | decision_default | 1 | 0.908 | 4/6 | 4/6 | n/a | 157 | 263,087 | 0 | 0 | 46 | 0 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 0.954 (0.930-0.981) | 22/22 | 6/6 | 0.834 (0.785-0.870) | 7 (7-8) | 67,683 (64,019-74,325) | 35,546 (9,569-53,811) | 7,591 (7,084-8,160) | 79 (71-85) | 1 |
| complex_dpa | typesafe/jev-latest | decision_default | 1 | 0.845 | 6/6 | 6/6 | n/a | 157 | 179,567 | 0 | 16,545 | 35 | 0 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 1 | 0.706 | 8/11 | 8/11 | n/a | 275 | 362,945 | 0 | 1,903 | 92 | 0 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 0.933 (0.915-0.958) | 22.3/26 (21-24) | 9.7/11 (9-10) | 0.640 (0.610-0.666) | 10 | 142,559 (140,793-145,612) | 65,009 (13,670-90,678) | 8,397 (8,105-8,738) | 88 (86-91) | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 1 | 0.749 | 10/11 | 10/11 | n/a | 275 | 600,881 | 0 | 0 | 84 | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.955 (0.932-0.973) | 24.0/26 (22-26) | 10.0/11 (9-11) | 0.682 (0.638-0.734) | 10 | 142,349 (141,124-143,264) | 63,572 (13,670-90,574) | 13,633 (12,522-14,824) | 146 (140-150) | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 1 | 0.685 | 9/11 | 9/11 | n/a | 275 | 421,530 | 0 | 33,516 | 61 | 0 |
| complex_nda | azure_ai/ms-decision-model | decision_default | 1 | 0.830 | 4/6 | 4/6 | n/a | 76 | 61,711 | 0 | 312 | 25 | 0 |
| complex_nda | openai/gpt-5.6-luna | default | 3 | 1.000 | 17.7/18 (17-18) | 5.7/6 (5-6) | 0.802 (0.779-0.830) | 6 | 53,476 (53,432-53,564) | 32,153 (6,835-47,836) | 4,656 (4,484-4,860) | 46 (45-47) | 0 |
| complex_nda | openai/gpt-6-luna | decision_default | 1 | 0.932 | 5/6 | 5/6 | n/a | 76 | 98,963 | 0 | 0 | 22 | 0 |
| complex_nda | openai/gpt-6-luna | default | 3 | 0.992 (0.991-0.992) | 17.3/18 (17-18) | 5.7/6 (5-6) | 0.876 (0.848-0.894) | 6 | 53,682 (53,428-53,940) | 28,706 (8,202-41,707) | 5,854 (5,469-6,185) | 60 (56-62) | 0 |
| complex_nda | typesafe/jev-latest | decision_default | 1 | 0.894 | 4/6 | 4/6 | n/a | 76 | 77,042 | 0 | 5,652 | 17 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_dpa | openai/gpt-5.6-luna | default | 3 | 4/6 (0.67) | 65/66 (0.98) | 24/24 (1.00) | 77 | 0.0016 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 4/6 (0.67) | 64/66 (0.97) | 24/24 (1.00) | 69 | 0.0014 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 3/3 (1.00) | 63/72 (0.88) | 36/36 (1.00) | 73 | 0.0016 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 71/73 (0.97) | 36/36 (1.00) | 75 | 0.0016 |
| complex_nda | openai/gpt-5.6-luna | default | 3 | 9/9 (1.00) | 62/63 (0.98) | 12/12 (1.00) | 67 | 0.0014 |
| complex_nda | openai/gpt-6-luna | default | 3 | 9/9 (1.00) | 61/63 (0.97) | 12/12 (1.00) | 67 | 0.0014 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 27 | 1,663 | 0.4201 |
| Judge | 19 | 490 | 0.0104 |
| Decision probabilities | 9 | 1,479 | 0.1393 |
| Total | | | 0.5698 |

## Concepts missed

- **complex_dpa, azure_ai/ms-decision-model, decision_default** (1 runs): Applicable initial notification regime (LabelConcept): 1x
- **complex_dpa, openai/gpt-5.6-luna, default** (3 runs): Distinct data locations count as of 1 February 2027 (NumericalConcept): 1x
- **complex_dpa, openai/gpt-6-luna, decision_default** (1 runs): Meridian Mail approved on 1 February 2027 (BooleanConcept): 1x; Permitted uses of held ORD records on 1 September 2027 (LabelConcept): 1x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (1 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-5.6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 2x; DR-114 maximum ADC INL in LSB (NumericalConcept): 2x; DR-114 junction temperature in °C (NumericalConcept): 2x; DR-114 meets HDR-7 (BooleanConcept): 2x; DR-115 junction temperature in °C (NumericalConcept): 1x; Halyard last-time-buy date for revision B (DateConcept): 1x; Last-shipment date of revision B (DateConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (1 runs): Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 2x; DR-114 meets HDR-7 (BooleanConcept): 2x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; DR-114 maximum ADC INL in LSB (NumericalConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (1 runs): Errata applicable to the DR-114 production boards (LabelConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_nda, azure_ai/ms-decision-model, decision_default** (1 runs): Protected assets at the April snapshot (LabelConcept): 1x; People permitted LUMEN access at the September snapshot (LabelConcept): 1x
- **complex_nda, openai/gpt-5.6-luna, default** (3 runs): Protected assets at the April snapshot (LabelConcept): 1x
- **complex_nda, openai/gpt-6-luna, decision_default** (1 runs): Protected assets at the April snapshot (LabelConcept): 1x
- **complex_nda, openai/gpt-6-luna, default** (3 runs): Protected assets at the April snapshot (LabelConcept): 1x; WORK-1 deletion deadline (DateConcept): 1x
- **complex_nda, typesafe/jev-latest, decision_default** (1 runs): Protected assets at the April snapshot (LabelConcept): 1x; People permitted LUMEN access at the September snapshot (LabelConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_dpa, azure_ai/ms-decision-model, decision_default**: Amendment effectiveness and precedence: F1 0.77 (P 0.62, R 1.00); Personal data breach notification: F1 0.80 (P 0.67, R 1.00); Incident IR-27: F1 0.60 (P 0.42, R 1.00); Retention TR-27: F1 0.63 (P 0.48, R 0.92); Supplier approvals in May 2027: F1 0.95 (P 0.95, R 0.95); Liability LC-27: F1 0.91 (P 0.83, R 1.00)
- **complex_dpa, openai/gpt-5.6-luna, default**: Amendment effectiveness and precedence: F1 0.79 (P 0.79, R 0.80); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 0.98 (P 0.98, R 0.98); Retention TR-27: F1 0.99 (P 0.97, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 0.97, R 1.00); Liability LC-27: F1 0.98 (P 0.97, R 1.00)
- **complex_dpa, openai/gpt-6-luna, decision_default**: Amendment effectiveness and precedence: F1 0.91 (P 0.83, R 1.00); Personal data breach notification: F1 0.94 (P 0.89, R 1.00); Incident IR-27: F1 0.93 (P 0.93, R 0.93); Retention TR-27: F1 0.92 (P 0.92, R 0.92); Supplier approvals in May 2027: F1 0.86 (P 1.00, R 0.76); Liability LC-27: F1 0.89 (P 1.00, R 0.80)
- **complex_dpa, openai/gpt-6-luna, default**: Amendment effectiveness and precedence: F1 0.86 (P 0.93, R 0.80); Personal data breach notification: F1 0.92 (P 1.00, R 0.88); Incident IR-27: F1 0.97 (P 0.96, R 0.98); Retention TR-27: F1 0.99 (P 1.00, R 0.97); Supplier approvals in May 2027: F1 0.99 (P 0.98, R 1.00); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_dpa, typesafe/jev-latest, decision_default**: Amendment effectiveness and precedence: F1 0.77 (P 0.62, R 1.00); Personal data breach notification: F1 0.89 (P 0.80, R 1.00); Incident IR-27: F1 0.78 (P 0.64, R 1.00); Retention TR-27: F1 0.83 (P 0.83, R 0.83); Supplier approvals in May 2027: F1 0.80 (P 1.00, R 0.67); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.77 (P 0.63, R 1.00); Flash memory specifications: F1 0.88 (P 0.78, R 1.00); DR-114 clock configuration: F1 0.79 (P 0.74, R 0.85); DR-114 thermal assessment: F1 0.59 (P 0.48, R 0.76); DR-114 errata and firmware: F1 0.56 (P 0.71, R 0.45); DR-115 errata and PLL1 limits: F1 0.50 (P 0.73, R 0.38); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-5.6-luna, default**: Device limitations of revision C: F1 0.89 (P 0.81, R 1.00); Flash memory specifications: F1 0.99 (P 0.98, R 1.00); DR-114 clock configuration: F1 0.92 (P 0.99, R 0.86); DR-114 thermal assessment: F1 0.84 (P 0.88, R 0.83); DR-114 errata and firmware: F1 0.95 (P 0.97, R 0.92); DR-115 errata and PLL1 limits: F1 0.95 (P 0.98, R 0.92); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.83 (P 0.83, R 0.83); Flash memory specifications: F1 0.90 (P 0.82, R 1.00); DR-114 clock configuration: F1 0.76 (P 0.68, R 0.85); DR-114 thermal assessment: F1 0.48 (P 0.44, R 0.52); DR-114 errata and firmware: F1 0.88 (P 0.90, R 0.86); DR-115 errata and PLL1 limits: F1 0.53 (P 0.75, R 0.41); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.97 (P 0.95, R 1.00); Flash memory specifications: F1 0.99 (P 0.98, R 1.00); DR-114 clock configuration: F1 0.95 (P 0.96, R 0.95); DR-114 thermal assessment: F1 0.91 (P 0.89, R 0.95); DR-114 errata and firmware: F1 0.91 (P 0.98, R 0.86); DR-115 errata and PLL1 limits: F1 0.96 (P 1.00, R 0.93); Revision B end of life: F1 0.98 (P 0.96, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.92 (P 0.92, R 0.92); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.83 (P 0.96, R 0.73); DR-114 thermal assessment: F1 0.36 (P 0.39, R 0.33); DR-114 errata and firmware: F1 0.53 (P 1.00, R 0.36); DR-115 errata and PLL1 limits: F1 0.39 (P 1.00, R 0.24); Revision B end of life: F1 0.77 (P 0.83, R 0.71)
- **complex_nda, azure_ai/ms-decision-model, decision_default**: NDA information protection: F1 0.67 (P 0.50, R 1.00); NDA recipient permissions: F1 0.97 (P 0.94, R 1.00); NDA compulsory disclosure Q-27: F1 0.78 (P 0.64, R 1.00); NDA exit obligations X-27: F1 0.90 (P 0.82, R 1.00)
- **complex_nda, openai/gpt-5.6-luna, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 1.00 (P 1.00, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, openai/gpt-6-luna, decision_default**: NDA information protection: F1 0.95 (P 0.90, R 1.00); NDA recipient permissions: F1 1.00 (P 1.00, R 1.00); NDA compulsory disclosure Q-27: F1 0.78 (P 0.64, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, openai/gpt-6-luna, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 0.98 (P 0.96, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 0.99 (P 1.00, R 0.98)
- **complex_nda, typesafe/jev-latest, decision_default**: NDA information protection: F1 0.80 (P 0.73, R 0.89); NDA recipient permissions: F1 0.97 (P 1.00, R 0.93); NDA compulsory disclosure Q-27: F1 0.89 (P 0.89, R 0.89); NDA exit obligations X-27: F1 0.92 (P 1.00, R 0.86)

## Errors and warnings

- none
