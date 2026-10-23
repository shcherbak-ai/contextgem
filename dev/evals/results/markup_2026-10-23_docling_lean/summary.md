# Extraction evaluation: markup_2026-10-23_docling_lean

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 0.723 (0.693-0.768) | 20.7/26 (20-21) | 7.7/11 (7-8) | 0.657 (0.632-0.677) | 12 (10-13) | 275,449 (190,582-342,126) | 0 | 18,133 (15,480-20,487) | 85 (72-95) | 5 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.969 (0.951-0.979) | 26/26 | 11/11 | 0.730 (0.702-0.754) | 10 | 118,797 (116,633-121,069) | 53,501 (13,670-73,416) | 13,615 (13,193-14,063) | 180 (169-189) | 0 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 0.945 (0.940-0.955) | 14/16 | 4/5 | 0.873 (0.846-0.891) | 7 | 100,572 (100,383-100,826) | 0 | 7,122 (6,939-7,266) | 33 (32-33) | 0 |
| complex_stability | openai/gpt-6-luna | default | 3 | 0.940 (0.920-0.961) | 15.7/16 (15-16) | 5/5 | 0.885 (0.855-0.921) | 7 | 64,074 (63,768-64,325) | 36,487 (9,569-50,292) | 4,633 (4,390-5,006) | 100 (95-106) | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_mcu | anthropic/claude-haiku-5-5 | default | 3 | 3/3 (1.00) | 58/68 (0.85) | 36/36 (1.00) | 68 | 0.0015 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 74/78 (0.95) | 36/36 (1.00) | 81 | 0.0018 |
| complex_stability | anthropic/claude-haiku-5-5 | default | 3 | 6/6 (1.00) | 45/51 (0.88) | 18/18 (1.00) | 51 | 0.0011 |
| complex_stability | openai/gpt-6-luna | default | 3 | 6/6 (1.00) | 50/51 (0.98) | 18/18 (1.00) | 54 | 0.0012 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 12 | 107 | 0.2156 |
| Judge | 13 | 330 | 0.0071 |
| Total | | | 0.2227 |

## Concepts missed

- **complex_mcu, anthropic/claude-haiku-5-5, default** (3 runs): DR-114 junction temperature in °C (NumericalConcept): 3x; DR-114 meets HDR-7 (BooleanConcept): 3x; Errata applicable to the DR-114 production boards (LabelConcept): 3x; Errata applicable to the DR-115 boards (LabelConcept): 3x; DR-114 maximum ADC INL in LSB (NumericalConcept): 2x; DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; FW 2.3.0 meets HDR-9 for the DR-114 production boards (BooleanConcept): 1x
- **complex_stability, anthropic/claude-haiku-5-5, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 3x; Every result outside the shelf-life specification is investigated in section 5 (BooleanConcept): 3x
- **complex_stability, openai/gpt-6-luna, default** (3 runs): Number of results outside the shelf-life specification (NumericalConcept): 1x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_mcu, anthropic/claude-haiku-5-5, default**: Device limitations of revision C: F1 0.62 (P 0.46, R 0.97); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.79 (P 0.78, R 0.80); DR-114 thermal assessment: F1 0.66 (P 0.54, R 0.86); DR-114 errata and firmware: F1 0.66 (P 0.77, R 0.59); DR-115 errata and PLL1 limits: F1 0.62 (P 0.87, R 0.52); Revision B end of life: F1 0.71 (P 0.59, R 0.90)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.95 (P 0.92, R 1.00); Flash memory specifications: F1 0.99 (P 0.98, R 1.00); DR-114 clock configuration: F1 0.94 (P 0.95, R 0.94); DR-114 thermal assessment: F1 0.98 (P 0.96, R 1.00); DR-114 errata and firmware: F1 0.99 (P 0.97, R 1.00); DR-115 errata and PLL1 limits: F1 0.94 (P 0.90, R 0.99); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, anthropic/claude-haiku-5-5, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 1.00 (P 1.00, R 1.00); Results outside the shelf-life specification: F1 0.78 (P 1.00, R 0.64); Shelf-life rules and proposal: F1 1.00 (P 1.00, R 1.00)
- **complex_stability, openai/gpt-6-luna, default**: Shelf-life acceptance criteria: F1 1.00 (P 1.00, R 1.00); Stability results of V20-2604: F1 0.97 (P 0.94, R 1.00); Results outside the shelf-life specification: F1 0.95 (P 0.96, R 0.95); Shelf-life rules and proposal: F1 0.84 (P 0.73, R 1.00)

## Errors and warnings

- none
