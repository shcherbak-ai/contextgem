# Extraction evaluation: markup_2026-10-23_docling_control

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0.758 (0.752-0.764) | 21.7/26 (21-23) | 8.3/11 (8-9) | 0.655 (0.620-0.678) | 12 (12-13) | 341,580 (257,358-415,835) | 0 | 18,947 (16,992-20,735) | 89 (80-95) | 7 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.958 (0.942-0.986) | 25.0/26 (24-26) | 10.7/11 (10-11) | 0.720 (0.716-0.728) | 10 | 140,060 (138,918-142,289) | 67,259 (13,670-101,634) | 14,015 (13,080-14,630) | 189 (171-201) | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 0.916 (0.888-0.938) | 14/16 | 4/5 | 0.722 (0.665-0.764) | 8 (7-8) | 138,490 (126,072-153,416) | 0 | 6,955 (5,957-7,760) | 33 (28-36) | 2 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.977 (0.951-0.990) | 15/16 | 5/5 | 0.831 (0.810-0.871) | 7 | 77,540 (77,289-77,907) | 46,997 (9,569-68,741) | 4,835 (4,094-5,347) | 100 (94-105) | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 3/3 (1.00) | 60/70 (0.86) | 36/36 (1.00) | 70 | 0.0015 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 72/78 (0.92) | 36/36 (1.00) | 80 | 0.0017 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 6/6 (1.00) | 45/51 (0.88) | 18/18 (1.00) | 52 | 0.0011 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 48/51 (0.94) | 18/18 (1.00) | 54 | 0.0011 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 12 | 111 | 0.2534 |
| Judge | 13 | 332 | 0.0072 |
| Total | | | 0.2605 |

## Concepts missed

- **complex_mcu, anthropic/claude-haiku-5-5, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 3x; Errata applicable to the DR-115 boards (LabelConcept): 3x; Errata applicable to the DR-114 production boards (LabelConcept): 2x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; Revision C retention after 1 000 cycles at 105 °C (NumericalConcept): 1x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; Errata applicable to the DR-114 production boards (LabelConcept): 1x; DR-115 junction temperature in °C (NumericalConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.85 (P 0.74, R 1.00); Flash memory specifications: F1 0.98 (P 0.96, R 1.00); DR-114 clock configuration: F1 0.80 (P 0.88, R 0.74); DR-114 thermal assessment: F1 0.63 (P 0.61, R 0.67); DR-114 errata and firmware: F1 0.66 (P 0.89, R 0.55); DR-115 errata and PLL1 limits: F1 0.52 (P 0.87, R 0.37); Revision B end of life: F1 0.88 (P 0.78, R 1.00)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 1.00 (P 1.00, R 1.00); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.92 (P 0.99, R 0.87); DR-114 thermal assessment: F1 0.98 (P 1.00, R 0.97); DR-114 errata and firmware: F1 0.95 (P 0.97, R 0.94); DR-115 errata and PLL1 limits: F1 0.85 (P 0.78, R 0.95); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.97 (P 1.00, R 0.95); Results outside the shelf-life specification: F1 0.71 (P 0.88, R 0.62); Shelf-life rules and proposal: F1 0.98 (P 0.96, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.96 (P 1.00, R 0.92); Shelf-life rules and proposal: F1 0.95 (P 0.91, R 1.00)

## Errors and warnings

- none
