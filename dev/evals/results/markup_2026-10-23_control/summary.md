# Extraction evaluation: markup_2026-10-23_control

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_dpa | anthropic/claude-haiku-5-5 | default | 6 | 0.913 (0.881-0.959) | 21.2/22 (19-22) | 5.8/6 (5-6) | 0.846 (0.820-0.866) | 7 | 93,878 (93,206-95,526) | 0 | 12,717 (12,077-13,540) | 55 (52-58) | 0 |
| complex_dpa | openai/gpt-6-luna | default | 6 | 0.972 (0.941-0.981) | 20.5/22 (13-22) | 5.5/6 (3-6) | 0.859 (0.849-0.872) | 8 (7-10) | 67,936 (57,205-89,234) | 50,479 (27,947-79,517) | 9,088 (6,487-12,541) | 110 (70-167) | 6 |
| complex_mcu | anthropic/claude-haiku-5-5 | default | 6 | 0.602 (0.000-0.748) | 18.7/26 (6-23) | 7.3/11 (2-9) | 0.575 (0.196-0.719) | 10 (5-14) | 216,463 (163,040-325,135) | 0 | 17,866 (6,243-29,586) | 84 (28-137) | 11 |
| complex_mcu | openai/gpt-6-luna | default | 6 | 0.954 (0.900-0.985) | 25.0/26 (23-26) | 10.5/11 (10-11) | 0.717 (0.662-0.803) | 10 | 101,399 (99,566-103,146) | 53,734 (13,670-65,627) | 14,479 (13,432-15,829) | 191 (168-232) | 0 |
| complex_nda | anthropic/claude-haiku-5-5 | default | 6 | 0.924 (0.891-0.966) | 17.8/18 (17-18) | 6/6 | 0.822 (0.767-0.840) | 6 (6-8) | 88,460 (80,328-112,777) | 0 | 10,466 (9,148-12,603) | 48 (42-60) | 3 |
| complex_nda | openai/gpt-6-luna | default | 6 | 0.987 (0.971-1.000) | 17.8/18 (17-18) | 6/6 | 0.854 (0.822-0.887) | 6 | 48,352 (48,029-48,668) | 33,032 (8,202-43,133) | 5,735 (5,339-6,117) | 74 (54-102) | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 6 | 0.878 (0.818-0.955) | 14/16 | 4/5 | 0.781 (0.736-0.831) | 7 (7-8) | 94,970 (90,878-112,236) | 0 | 6,768 (5,551-8,201) | 33 (28-38) | 1 |
| complex_stability | openai/gpt-6-luna | default | 6 | 0.970 (0.933-0.990) | 14.7/16 (14-15) | 4.7/5 (4-5) | 0.824 (0.777-0.862) | 7 | 55,773 (55,594-55,972) | 40,666 (9,569-51,796) | 4,872 (4,697-5,108) | 78 (59-100) | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_dpa | anthropic/claude-haiku-5-5 | default | 6 | 6/12 (0.50) | 121/132 (0.92) | 48/48 (1.00) | 162 | 0.0033 |
| complex_dpa | openai/gpt-6-luna | default | 6 | 6/12 (0.50) | 119/122 (0.98) | 48/48 (1.00) | 127 | 0.0027 |
| complex_mcu | anthropic/claude-haiku-5-5 | default | 6 | 5/6 (0.83) | 106/122 (0.87) | 60/60 (1.00) | 123 | 0.0027 |
| complex_mcu | openai/gpt-6-luna | default | 6 | 6/6 (1.00) | 142/152 (0.93) | 72/72 (1.00) | 158 | 0.0034 |
| complex_nda | anthropic/claude-haiku-5-5 | default | 6 | 15/18 (0.83) | 123/126 (0.98) | 24/24 (1.00) | 138 | 0.0030 |
| complex_nda | openai/gpt-6-luna | default | 6 | 14/18 (0.78) | 125/126 (0.99) | 24/24 (1.00) | 140 | 0.0030 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 6 | 11/12 (0.92) | 84/101 (0.83) | 36/36 (1.00) | 103 | 0.0022 |
| complex_stability | openai/gpt-6-luna | default | 6 | 12/12 (1.00) | 93/102 (0.91) | 36/36 (1.00) | 108 | 0.0023 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 48 | 373 | 0.6246 |
| Judge | 49 | 1,135 | 0.0243 |
| Total | | | 0.6489 |

## Concepts missed

- **complex_dpa, anthropic/claude-haiku-5-5, default** (6 runs): Held ORD records deletion deadline (DateConcept): 3x; BK-0630 deletion deadline (DateConcept): 1x; Authorised data locations on 15 May 2027 (LabelConcept): 1x
- **complex_dpa, openai/gpt-6-luna, default** (6 runs): Breach notification deadline in force on 15 September 2026 (NumericalConcept): 1x; Breach notification deadline in force on 1 November 2026 (NumericalConcept): 1x; Audit notice period in force on 1 November 2026 (NumericalConcept): 1x; Liability cap in force on 1 November 2026 (NumericalConcept): 1x; Liability cap covers confidentiality breaches (BooleanConcept): 1x; Meridian Mail approved on 1 November 2026 (BooleanConcept): 1x; Meridian Mail approved on 1 February 2027 (BooleanConcept): 1x; Distinct data locations count as of 1 February 2027 (NumericalConcept): 1x; Deletion deadline for year-end termination (DateConcept): 1x
- **complex_mcu, anthropic/claude-haiku-5-5, default** (6 runs): Errata applicable to the DR-115 boards (LabelConcept): 6x; DR-114 junction temperature in °C (NumericalConcept): 5x; DR-114 meets HDR-7 (BooleanConcept): 5x; HCLK maximum for 2 wait states in VOS1 in MHz (NumericalConcept): 4x; Errata applicable to the DR-114 production boards (LabelConcept): 4x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 4x; DR-114 maximum ADC INL in LSB (NumericalConcept): 2x; FW 2.3.0 meets HDR-9 for the DR-114 production boards (BooleanConcept): 2x; Limitations present in revision C (JsonObjectConcept): 1x; Impact class of ES-08 (RatingConcept): 1x; ES-09 present in revision C (BooleanConcept): 1x; Revision C retention after 10 000 cycles at 85 °C (NumericalConcept): 1x; Revision C retention after 1 000 cycles at 105 °C (NumericalConcept): 1x; DR-114 PLL1 configuration within specification (BooleanConcept): 1x; DR-114 required flash wait states (NumericalConcept): 1x; DR-115 maximum PLL1 VCO frequency in MHz (NumericalConcept): 1x; DR-115 PLL1 configuration within its limits (BooleanConcept): 1x; Halyard last-time-buy date for revision B (DateConcept): 1x; Last-shipment date of revision B (DateConcept): 1x; PO-5521 valid last-time-buy order (BooleanConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (6 runs): DR-114 junction temperature in °C (NumericalConcept): 2x; DR-114 meets HDR-7 (BooleanConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 1x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x
- **complex_nda, anthropic/claude-haiku-5-5, default** (6 runs): WORK-1 deletion deadline (DateConcept): 1x
- **complex_nda, openai/gpt-6-luna, default** (6 runs): WORK-1 deletion deadline (DateConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (6 runs): Number of results outside the shelf-life specification (NumericalConcept): 6x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 6x
- **complex_stability, openai/gpt-6-luna, default** (6 runs): Number of results outside the shelf-life specification (NumericalConcept): 6x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 1x; Rule that gives the proposed shelf life of the PVC/Alu blister (LabelConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_dpa, anthropic/claude-haiku-5-5, default**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 0.96 (P 1.00, R 0.94); Incident IR-27: F1 0.94 (P 0.98, R 0.90); Retention TR-27: F1 0.91 (P 1.00, R 0.85); Supplier approvals in May 2027: F1 0.92 (P 0.99, R 0.87); Liability LC-27: F1 0.86 (P 1.00, R 0.75)
- **complex_dpa, openai/gpt-6-luna, default**: Amendment effectiveness and precedence: F1 0.87 (P 0.97, R 0.80); Personal data breach notification: F1 0.98 (P 0.97, R 1.00); Incident IR-27: F1 0.99 (P 0.98, R 1.00); Retention TR-27: F1 0.99 (P 0.99, R 1.00); Supplier approvals in May 2027: F1 0.99 (P 0.99, R 1.00); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.59 (P 0.47, R 0.81); Flash memory specifications: F1 0.80 (P 0.80, R 0.80); DR-114 clock configuration: F1 0.65 (P 0.65, R 0.68); DR-114 thermal assessment: F1 0.57 (P 0.52, R 0.65); DR-114 errata and firmware: F1 0.52 (P 0.57, R 0.48); DR-115 errata and PLL1 limits: F1 0.42 (P 0.59, R 0.33); Revision B end of life: F1 0.66 (P 0.60, R 0.76)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.97 (P 0.95, R 1.00); Flash memory specifications: F1 0.99 (P 0.99, R 1.00); DR-114 clock configuration: F1 0.92 (P 0.97, R 0.87); DR-114 thermal assessment: F1 0.93 (P 0.91, R 0.94); DR-114 errata and firmware: F1 0.97 (P 0.98, R 0.96); DR-115 errata and PLL1 limits: F1 0.90 (P 0.86, R 0.97); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, anthropic/claude-haiku-5-5, default**: NDA information protection: F1 0.87 (P 0.78, R 1.00); NDA recipient permissions: F1 0.97 (P 0.94, R 1.00); NDA compulsory disclosure Q-27: F1 0.91 (P 0.85, R 0.98); NDA exit obligations X-27: F1 0.95 (P 0.91, R 0.99)
- **complex_nda, openai/gpt-6-luna, default**: NDA information protection: F1 0.99 (P 0.98, R 1.00); NDA recipient permissions: F1 0.98 (P 0.96, R 1.00); NDA compulsory disclosure Q-27: F1 0.99 (P 0.98, R 1.00); NDA exit obligations X-27: F1 0.99 (P 0.99, R 0.99)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 0.83 (P 0.73, R 1.00); Stability results of V20-2604: F1 0.97 (P 0.98, R 0.97); Results outside the shelf-life specification: F1 0.74 (P 0.89, R 0.65); Shelf-life rules and proposal: F1 0.96 (P 0.93, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 0.99, R 1.00); Results outside the shelf-life specification: F1 0.94 (P 0.95, R 0.92); Shelf-life rules and proposal: F1 0.95 (P 0.91, R 1.00)

## Errors and warnings

- none
