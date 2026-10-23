# Extraction evaluation: markup_2026-10-23_lean

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_dpa | anthropic/claude-haiku-5-5 | default | 6 | 0.889 (0.824-0.935) | 20.7/22 (20-22) | 5.3/6 (5-6) | 0.831 (0.793-0.874) | 7 | 83,142 (82,358-84,430) | 0 | 12,475 (10,974-14,446) | 56 (49-64) | 0 |
| complex_dpa | openai/gpt-6-luna | default | 6 | 0.981 (0.974-0.996) | 21.5/22 (20-22) | 5.8/6 (5-6) | 0.859 (0.812-0.920) | 8 (7-10) | 56,996 (51,563-83,365) | 40,806 (39,070-43,145) | 8,289 (6,590-13,632) | 98 (66-132) | 3 |
| complex_mcu | anthropic/claude-haiku-5-5 | default | 6 | 0.715 (0.672-0.753) | 20.2/26 (19-21) | 8.0/11 (7-9) | 0.625 (0.601-0.648) | 10 (10-11) | 129,538 (125,715-139,658) | 0 | 17,219 (16,032-17,995) | 81 (75-84) | 1 |
| complex_mcu | openai/gpt-6-luna | default | 6 | 0.928 (0.902-0.951) | 24.8/26 (23-26) | 10.3/11 (9-11) | 0.720 (0.665-0.759) | 10 | 82,933 (82,141-83,605) | 41,696 (13,670-48,289) | 13,923 (12,742-15,284) | 181 (164-213) | 0 |
| complex_nda | anthropic/claude-haiku-5-5 | default | 6 | 0.945 (0.906-0.975) | 17.8/18 (17-18) | 6/6 | 0.802 (0.740-0.834) | 7 (6-7) | 80,011 (69,021-87,261) | 0 | 10,596 (9,025-11,758) | 49 (44-53) | 4 |
| complex_nda | openai/gpt-6-luna | default | 6 | 0.991 (0.963-1.000) | 18/18 | 6/6 | 0.873 (0.851-0.890) | 6 | 42,836 (42,661-43,310) | 31,260 (6,835-42,643) | 5,621 (5,204-5,924) | 72 (53-94) | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 6 | 0.960 (0.938-0.990) | 14/16 | 4/5 | 0.735 (0.665-0.829) | 7 (7-8) | 73,861 (70,062-84,445) | 0 | 7,222 (6,023-8,237) | 34 (28-39) | 2 |
| complex_stability | openai/gpt-6-luna | default | 6 | 0.972 (0.941-0.991) | 14.3/16 (13-16) | 4.2/5 (3-5) | 0.814 (0.732-0.863) | 7 | 44,859 (44,744-45,043) | 31,448 (9,569-37,437) | 5,047 (4,195-5,677) | 78 (52-100) | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_dpa | anthropic/claude-haiku-5-5 | default | 6 | 6/12 (0.50) | 116/129 (0.90) | 48/48 (1.00) | 147 | 0.0031 |
| complex_dpa | openai/gpt-6-luna | default | 6 | 9/12 (0.75) | 126/132 (0.95) | 48/48 (1.00) | 142 | 0.0029 |
| complex_mcu | anthropic/claude-haiku-5-5 | default | 6 | 6/6 (1.00) | 113/136 (0.83) | 72/72 (1.00) | 136 | 0.0029 |
| complex_mcu | openai/gpt-6-luna | default | 6 | 6/6 (1.00) | 142/152 (0.93) | 72/72 (1.00) | 158 | 0.0034 |
| complex_nda | anthropic/claude-haiku-5-5 | default | 6 | 14/18 (0.78) | 124/126 (0.98) | 24/24 (1.00) | 138 | 0.0030 |
| complex_nda | openai/gpt-6-luna | default | 6 | 17/18 (0.94) | 125/126 (0.99) | 24/24 (1.00) | 137 | 0.0029 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 6 | 12/12 (1.00) | 89/102 (0.87) | 36/36 (1.00) | 106 | 0.0023 |
| complex_stability | openai/gpt-6-luna | default | 6 | 12/12 (1.00) | 92/102 (0.90) | 36/36 (1.00) | 108 | 0.0023 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 48 | 370 | 0.5316 |
| Judge | 49 | 1,148 | 0.0246 |
| Total | | | 0.5562 |

## Concepts missed

- **complex_dpa, anthropic/claude-haiku-5-5, default** (6 runs): Held ORD records deletion deadline (DateConcept): 4x; Permitted uses of held ORD records on 1 September 2027 (LabelConcept): 3x; Authorised data locations on 15 May 2027 (LabelConcept): 1x
- **complex_dpa, openai/gpt-6-luna, default** (6 runs): Completed follow-up report deadline (DateConcept): 1x; Liability cap covers confidentiality breaches (BooleanConcept): 1x; Distinct data locations count as of 1 February 2027 (NumericalConcept): 1x
- **complex_mcu, anthropic/claude-haiku-5-5, default** (6 runs): DR-114 junction temperature in °C (NumericalConcept): 6x; DR-114 meets HDR-7 (BooleanConcept): 6x; Errata applicable to the DR-115 boards (LabelConcept): 6x; Errata applicable to the DR-114 production boards (LabelConcept): 5x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 4x; HCLK maximum for 2 wait states in VOS1 in MHz (NumericalConcept): 3x; DR-114 required flash wait states (NumericalConcept): 2x; DR-114 maximum ADC INL in LSB (NumericalConcept): 1x; Revision C retention after 1 000 cycles at 105 °C (NumericalConcept): 1x; FW 2.3.0 meets HDR-9 for the DR-114 production boards (BooleanConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (6 runs): DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x
- **complex_nda, anthropic/claude-haiku-5-5, default** (6 runs): WORK-1 deletion deadline (DateConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (6 runs): Number of results outside the shelf-life specification (NumericalConcept): 6x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 6x
- **complex_stability, openai/gpt-6-luna, default** (6 runs): Number of results outside the shelf-life specification (NumericalConcept): 4x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 4x; Date on which V20-2604 was placed at the intermediate condition (DateConcept): 1x; Batches with a significant change at the accelerated condition (LabelConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_dpa, anthropic/claude-haiku-5-5, default**: Amendment effectiveness and precedence: F1 0.83 (P 0.88, R 0.80); Personal data breach notification: F1 0.98 (P 1.00, R 0.96); Incident IR-27: F1 0.93 (P 0.99, R 0.88); Retention TR-27: F1 0.85 (P 0.96, R 0.78); Supplier approvals in May 2027: F1 0.90 (P 0.99, R 0.83); Liability LC-27: F1 0.84 (P 1.00, R 0.73)
- **complex_dpa, openai/gpt-6-luna, default**: Amendment effectiveness and precedence: F1 0.91 (P 1.00, R 0.83); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 1.00 (P 1.00, R 1.00); Retention TR-27: F1 1.00 (P 1.00, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 0.96, R 1.00); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.69 (P 0.53, R 0.97); Flash memory specifications: F1 0.95 (P 0.94, R 0.96); DR-114 clock configuration: F1 0.67 (P 0.68, R 0.67); DR-114 thermal assessment: F1 0.71 (P 0.61, R 0.86); DR-114 errata and firmware: F1 0.69 (P 0.75, R 0.64); DR-115 errata and PLL1 limits: F1 0.52 (P 0.79, R 0.41); Revision B end of life: F1 0.77 (P 0.71, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.89 (P 0.80, R 1.00); Flash memory specifications: F1 0.99 (P 0.99, R 1.00); DR-114 clock configuration: F1 0.90 (P 0.92, R 0.89); DR-114 thermal assessment: F1 0.83 (P 0.74, R 0.96); DR-114 errata and firmware: F1 0.97 (P 0.95, R 0.99); DR-115 errata and PLL1 limits: F1 0.91 (P 0.88, R 0.97); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, anthropic/claude-haiku-5-5, default**: NDA information protection: F1 0.91 (P 0.84, R 0.99); NDA recipient permissions: F1 0.97 (P 0.95, R 1.00); NDA compulsory disclosure Q-27: F1 0.95 (P 0.93, R 0.98); NDA exit obligations X-27: F1 0.95 (P 0.95, R 0.96)
- **complex_nda, openai/gpt-6-luna, default**: NDA information protection: F1 1.00 (P 0.99, R 1.00); NDA recipient permissions: F1 0.98 (P 0.97, R 1.00); NDA compulsory disclosure Q-27: F1 0.99 (P 0.98, R 1.00); NDA exit obligations X-27: F1 0.99 (P 0.99, R 1.00)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.84 (P 0.97, R 0.74); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 0.99, R 1.00); Results outside the shelf-life specification: F1 0.94 (P 0.93, R 0.96); Shelf-life rules and proposal: F1 0.95 (P 0.91, R 1.00)

## Errors and warnings

- none
