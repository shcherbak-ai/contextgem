# Extraction evaluation: baseline_2026-10-15

Decision models extract only the closed concepts (Boolean, label and rating concepts), without justifications and references: compare them with LLMs on the aspects and on the closed concepts.

| Case | Model | Mode | Runs | Aspect F1 | Concepts correct | Closed concepts correct | Reference F1 | Requests | Input tokens | Cached | Output tokens | Seconds | Retries |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| complex_dpa | azure_ai/ms-decision-model | decision_default | 2 | 0.772 (0.768-0.776) | 5/6 | 5/6 | n/a | 157 | 109,239 (109,229-109,249) | 0 | 934 | 132 (76-189) | 0 |
| complex_dpa | openai/gpt-5.6-luna | default | 3 | 0.952 (0.917-0.974) | 20.7/22 (20-21) | 6/6 | 0.826 (0.802-0.843) | 7 | 57,403 (57,097-57,932) | 30,666 (9,569-43,394) | 6,126 (5,503-6,646) | 60 (57-66) | 0 |
| complex_dpa | openai/gpt-6-luna | concurrent | 3 | 0.954 (0.937-0.981) | 22/22 | 6/6 | 0.900 (0.884-0.920) | 30 | 259,171 (258,503-259,723) | 154,871 (48,836-214,404) | 11,964 (11,004-12,819) | 35 (34-36) | 0 |
| complex_dpa | openai/gpt-6-luna | decision_default | 2 | 0.823 | 4/6 | 4/6 | n/a | 157 | 222,536 | 0 | 0 | 44 (44-45) | 0 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 0.980 (0.978-0.981) | 21.3/22 (21-22) | 6/6 | 0.855 (0.831-0.869) | 7 | 57,288 (57,087-57,630) | 32,112 (9,569-47,732) | 7,497 (7,072-7,812) | 78 (77-80) | 0 |
| complex_dpa | openai/gpt-6.1-sol | default | 1 | 0.946 | 22/22 | 6/6 | 0.937 | 7 | 58,664 | 9,569 | 4,617 | 89 | 0 |
| complex_dpa | typesafe/jev-latest | decision_default | 2 | 0.790 (0.787-0.793) | 5/6 | 5/6 | n/a | 157 | 144,042 (144,034-144,050) | 0 | 16,545 | 34 (34-35) | 0 |
| complex_mcu | azure_ai/ms-decision-model | decision_default | 2 | 0.567 (0.567-0.568) | 9/11 | 9/11 | n/a | 275 | 282,474 (282,448-282,499) | 0 | 1,903 | 154 (118-191) | 0 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 0.948 (0.942-0.957) | 23/26 | 10/11 | 0.636 (0.589-0.685) | 10 | 101,001 (100,469-101,364) | 44,377 (13,670-61,593) | 9,633 (9,216-9,882) | 96 (87-106) | 0 |
| complex_mcu | openai/gpt-6-luna | concurrent | 3 | 0.920 (0.897-0.940) | 25.3/26 (24-26) | 10.7/11 (10-11) | 0.767 (0.756-0.782) | 34 | 434,962 (428,758-439,823) | 234,914 (46,478-334,166) | 28,859 (27,986-29,896) | 66 (62-70) | 0 |
| complex_mcu | openai/gpt-6-luna | decision_default | 2 | 0.633 | 8/11 | 8/11 | n/a | 275 | 517,005 | 0 | 0 | 84 (83-84) | 0 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 0.940 (0.924-0.962) | 25.7/26 (25-26) | 11/11 | 0.711 (0.685-0.745) | 10 | 103,428 (101,783-104,314) | 46,466 (12,303-65,626) | 14,238 (13,402-14,774) | 147 (144-149) | 0 |
| complex_mcu | openai/gpt-6.1-sol | default | 1 | 0.954 | 26/26 | 11/11 | 0.740 | 10 | 102,631 | 13,670 | 6,643 | 132 | 0 |
| complex_mcu | typesafe/jev-latest | decision_default | 2 | 0.645 (0.640-0.649) | 8/11 | 8/11 | n/a | 275 | 339,680 (339,640-339,720) | 0 | 33,516 | 60 (60-61) | 0 |
| complex_nda | azure_ai/ms-decision-model | decision_default | 2 | 0.772 (0.764-0.781) | 4/6 | 4/6 | n/a | 76 | 50,908 (50,742-51,074) | 0 | 312 | 48 (31-65) | 0 |
| complex_nda | openai/gpt-5.6-luna | default | 3 | 1.000 | 17.3/18 (17-18) | 5.7/6 (5-6) | 0.829 (0.818-0.837) | 6 | 48,119 (48,028-48,190) | 27,400 (6,835-43,023) | 4,605 (4,499-4,679) | 46 (42-49) | 0 |
| complex_nda | openai/gpt-6-luna | concurrent | 3 | 0.977 (0.969-0.985) | 18/18 | 6/6 | 0.857 (0.843-0.874) | 25 | 197,177 (196,592-198,103) | 108,742 (34,175-152,767) | 9,067 (8,395-9,485) | 32 (25-36) | 0 |
| complex_nda | openai/gpt-6-luna | decision_default | 2 | 0.892 | 4/6 | 4/6 | n/a | 76 | 86,263 | 0 | 0 | 21 (19-22) | 0 |
| complex_nda | openai/gpt-6-luna | default | 3 | 0.995 (0.992-1.000) | 17.7/18 (17-18) | 5.7/6 (5-6) | 0.857 (0.848-0.870) | 6 | 48,282 (48,138-48,460) | 29,557 (8,202-43,133) | 5,909 (5,661-6,266) | 59 (55-61) | 0 |
| complex_nda | openai/gpt-6.1-sol | default | 1 | 1.000 | 18/18 | 6/6 | 0.892 | 6 | 48,358 | 6,835 | 3,525 | 64 | 0 |
| complex_nda | typesafe/jev-latest | decision_default | 2 | 0.840 (0.839-0.840) | 3/6 | 3/6 | n/a | 76 | 65,410 (65,268-65,553) | 0 | 5,652 | 16 | 0 |

## Judged by typesafe/jev-latest

String values, justifications and text fields of objects, over all the runs judged; a value equal to the reference value or to an accepted one is correct without the judge.

| Case | Model | Mode | Runs judged | Strings correct | Justifications correct | Object fields correct | Judge requests | Judge cost (USD) |
|---|---|---|---|---|---|---|---|---|
| complex_dpa | openai/gpt-5.6-luna | default | 3 | 6/6 (1.00) | 60/66 (0.91) | 24/24 (1.00) | 81 | 0.0016 |
| complex_dpa | openai/gpt-6-luna | concurrent | 3 | 3/6 (0.50) | 66/66 (1.00) | 24/24 (1.00) | 69 | 0.0015 |
| complex_dpa | openai/gpt-6-luna | default | 3 | 3/6 (0.50) | 63/66 (0.95) | 24/24 (1.00) | 69 | 0.0014 |
| complex_dpa | openai/gpt-6.1-sol | default | 1 | 2/2 (1.00) | 22/22 (1.00) | 8/8 (1.00) | 27 | 0.0006 |
| complex_mcu | openai/gpt-5.6-luna | default | 3 | 3/3 (1.00) | 66/74 (0.89) | 36/36 (1.00) | 74 | 0.0016 |
| complex_mcu | openai/gpt-6-luna | concurrent | 3 | 3/3 (1.00) | 77/78 (0.99) | 36/36 (1.00) | 79 | 0.0017 |
| complex_mcu | openai/gpt-6-luna | default | 3 | 3/3 (1.00) | 77/77 (1.00) | 36/36 (1.00) | 78 | 0.0017 |
| complex_mcu | openai/gpt-6.1-sol | default | 1 | 1/1 (1.00) | 26/26 (1.00) | 12/12 (1.00) | 26 | 0.0006 |
| complex_nda | openai/gpt-5.6-luna | default | 3 | 9/9 (1.00) | 60/63 (0.95) | 12/12 (1.00) | 66 | 0.0014 |
| complex_nda | openai/gpt-6-luna | concurrent | 3 | 9/9 (1.00) | 62/63 (0.98) | 12/12 (1.00) | 67 | 0.0015 |
| complex_nda | openai/gpt-6-luna | default | 3 | 9/9 (1.00) | 62/63 (0.98) | 12/12 (1.00) | 70 | 0.0015 |
| complex_nda | openai/gpt-6.1-sol | default | 1 | 3/3 (1.00) | 21/21 (1.00) | 4/4 (1.00) | 22 | 0.0005 |

## Cost

| Part | Files | Requests | Cost (USD) |
|---|---|---|---|
| Runs | 48 | 3,476 | 1.3253 |
| Judge | 31 | 790 | 0.0169 |
| Decision probabilities | 9 | n/a | not recorded |
| Total | | | 1.3422 |

9 files record no cost (the passes of `decision_probabilities` recorded before 2026-10-15): the total leaves them out.

## Concepts missed

- **complex_dpa, azure_ai/ms-decision-model, decision_default** (2 runs): Applicable initial notification regime (LabelConcept): 2x
- **complex_dpa, openai/gpt-5.6-luna, default** (3 runs): Distinct data locations count as of 1 February 2027 (NumericalConcept): 2x; Audit notice period in force on 1 November 2026 (NumericalConcept): 1x; Completed follow-up report deadline (DateConcept): 1x
- **complex_dpa, openai/gpt-6-luna, decision_default** (2 runs): Meridian Mail approved on 1 February 2027 (BooleanConcept): 2x; Permitted uses of held ORD records on 1 September 2027 (LabelConcept): 2x
- **complex_dpa, openai/gpt-6-luna, default** (3 runs): Distinct data locations count as of 1 February 2027 (NumericalConcept): 2x
- **complex_dpa, typesafe/jev-latest, decision_default** (2 runs): Authorised data locations on 15 May 2027 (LabelConcept): 2x
- **complex_mcu, azure_ai/ms-decision-model, decision_default** (2 runs): DR-114 PLL1 configuration within specification (BooleanConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 2x
- **complex_mcu, openai/gpt-5.6-luna, default** (3 runs): Errata applicable to the DR-114 production boards (LabelConcept): 3x; DR-114 maximum ADC INL in LSB (NumericalConcept): 3x; DR-114 junction temperature in °C (NumericalConcept): 2x; DR-115 junction temperature in °C (NumericalConcept): 1x
- **complex_mcu, openai/gpt-6-luna, concurrent** (3 runs): DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x; Errata applicable to the DR-115 boards (LabelConcept): 1x
- **complex_mcu, openai/gpt-6-luna, decision_default** (2 runs): Errata applicable to the DR-114 production boards (LabelConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 2x; PO-5521 valid last-time-buy order (BooleanConcept): 2x
- **complex_mcu, openai/gpt-6-luna, default** (3 runs): DR-114 maximum permitted SYSCLK in MHz (NumericalConcept): 1x
- **complex_mcu, typesafe/jev-latest, decision_default** (2 runs): Errata applicable to the DR-114 production boards (LabelConcept): 2x; Errata applicable to the DR-115 boards (LabelConcept): 2x; DR-115 PLL1 configuration within its limits (BooleanConcept): 2x
- **complex_nda, azure_ai/ms-decision-model, decision_default** (2 runs): Protected assets at the April snapshot (LabelConcept): 2x; People permitted LUMEN access at the September snapshot (LabelConcept): 2x
- **complex_nda, openai/gpt-5.6-luna, default** (3 runs): WORK-1 deletion deadline (DateConcept): 1x; Protected assets at the April snapshot (LabelConcept): 1x
- **complex_nda, openai/gpt-6-luna, decision_default** (2 runs): Protected assets at the April snapshot (LabelConcept): 2x; Artifacts permitted retention at the December snapshot (LabelConcept): 2x
- **complex_nda, openai/gpt-6-luna, default** (3 runs): Protected assets at the April snapshot (LabelConcept): 1x
- **complex_nda, typesafe/jev-latest, decision_default** (2 runs): Protected assets at the April snapshot (LabelConcept): 2x; People permitted LUMEN access at the September snapshot (LabelConcept): 2x; Artifacts permitted retention at the December snapshot (LabelConcept): 2x

## Aspects (mean F1, precision and recall of each aspect)

- **complex_dpa, azure_ai/ms-decision-model, decision_default**: Amendment effectiveness and precedence: F1 0.83 (P 0.71, R 1.00); Personal data breach notification: F1 0.76 (P 0.62, R 1.00); Incident IR-27: F1 0.60 (P 0.44, R 0.93); Retention TR-27: F1 0.65 (P 0.48, R 1.00); Supplier approvals in May 2027: F1 0.98 (P 1.00, R 0.95); Liability LC-27: F1 0.82 (P 0.69, R 1.00)
- **complex_dpa, openai/gpt-5.6-luna, default**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 0.92 (P 1.00, R 0.88); Incident IR-27: F1 0.98 (P 0.96, R 1.00); Retention TR-27: F1 0.96 (P 1.00, R 0.92); Supplier approvals in May 2027: F1 0.98 (P 0.97, R 1.00); Liability LC-27: F1 0.98 (P 1.00, R 0.97)
- **complex_dpa, openai/gpt-6-luna, concurrent**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 0.88 (P 0.80, R 1.00); Incident IR-27: F1 0.96 (P 0.92, R 1.00); Retention TR-27: F1 1.00 (P 1.00, R 1.00); Supplier approvals in May 2027: F1 1.00 (P 1.00, R 1.00); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_dpa, openai/gpt-6-luna, decision_default**: Amendment effectiveness and precedence: F1 0.80 (P 0.80, R 0.80); Personal data breach notification: F1 0.89 (P 0.80, R 1.00); Incident IR-27: F1 0.74 (P 0.77, R 0.71); Retention TR-27: F1 0.82 (P 0.90, R 0.75); Supplier approvals in May 2027: F1 0.74 (P 0.93, R 0.62); Liability LC-27: F1 0.95 (P 1.00, R 0.90)
- **complex_dpa, openai/gpt-6-luna, default**: Amendment effectiveness and precedence: F1 0.89 (P 1.00, R 0.80); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 1.00 (P 1.00, R 1.00); Retention TR-27: F1 1.00 (P 1.00, R 1.00); Supplier approvals in May 2027: F1 0.99 (P 0.98, R 1.00); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_dpa, openai/gpt-6.1-sol, default**: Amendment effectiveness and precedence: F1 1.00 (P 1.00, R 1.00); Personal data breach notification: F1 1.00 (P 1.00, R 1.00); Incident IR-27: F1 0.93 (P 0.88, R 1.00); Retention TR-27: F1 0.92 (P 0.86, R 1.00); Supplier approvals in May 2027: F1 0.91 (P 0.84, R 1.00); Liability LC-27: F1 0.91 (P 0.83, R 1.00)
- **complex_dpa, typesafe/jev-latest, decision_default**: Amendment effectiveness and precedence: F1 0.67 (P 0.50, R 1.00); Personal data breach notification: F1 0.80 (P 0.67, R 1.00); Incident IR-27: F1 0.74 (P 0.62, R 0.93); Retention TR-27: F1 0.77 (P 0.78, R 0.75); Supplier approvals in May 2027: F1 0.76 (P 1.00, R 0.62); Liability LC-27: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, azure_ai/ms-decision-model, decision_default**: Device limitations of revision C: F1 0.38 (P 0.23, R 1.00); Flash memory specifications: F1 0.61 (P 0.58, R 0.64); DR-114 clock configuration: F1 0.66 (P 0.60, R 0.74); DR-114 thermal assessment: F1 0.47 (P 0.45, R 0.50); DR-114 errata and firmware: F1 0.56 (P 0.65, R 0.50); DR-115 errata and PLL1 limits: F1 0.43 (P 0.57, R 0.34); Revision B end of life: F1 0.86 (P 0.86, R 0.86)
- **complex_mcu, openai/gpt-5.6-luna, default**: Device limitations of revision C: F1 0.97 (P 0.95, R 1.00); Flash memory specifications: F1 1.00 (P 1.00, R 1.00); DR-114 clock configuration: F1 0.94 (P 0.99, R 0.89); DR-114 thermal assessment: F1 0.84 (P 0.80, R 0.89); DR-114 errata and firmware: F1 0.96 (P 0.98, R 0.94); DR-115 errata and PLL1 limits: F1 0.93 (P 0.92, R 0.93); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6-luna, concurrent**: Device limitations of revision C: F1 1.00 (P 1.00, R 1.00); Flash memory specifications: F1 0.97 (P 0.94, R 1.00); DR-114 clock configuration: F1 0.85 (P 0.78, R 0.93); DR-114 thermal assessment: F1 0.96 (P 0.92, R 1.00); DR-114 errata and firmware: F1 0.91 (P 0.86, R 0.95); DR-115 errata and PLL1 limits: F1 0.76 (P 0.85, R 0.77); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6-luna, decision_default**: Device limitations of revision C: F1 0.87 (P 0.91, R 0.83); Flash memory specifications: F1 0.64 (P 0.64, R 0.64); DR-114 clock configuration: F1 0.67 (P 0.70, R 0.64); DR-114 thermal assessment: F1 0.37 (P 0.55, R 0.29); DR-114 errata and firmware: F1 0.58 (P 0.69, R 0.50); DR-115 errata and PLL1 limits: F1 0.37 (P 0.57, R 0.28); Revision B end of life: F1 0.92 (P 1.00, R 0.86)
- **complex_mcu, openai/gpt-6-luna, default**: Device limitations of revision C: F1 0.86 (P 0.77, R 1.00); Flash memory specifications: F1 0.99 (P 0.98, R 1.00); DR-114 clock configuration: F1 0.93 (P 0.91, R 0.96); DR-114 thermal assessment: F1 0.95 (P 0.91, R 1.00); DR-114 errata and firmware: F1 0.99 (P 0.99, R 1.00); DR-115 errata and PLL1 limits: F1 0.86 (P 0.76, R 0.99); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, openai/gpt-6.1-sol, default**: Device limitations of revision C: F1 1.00 (P 1.00, R 1.00); Flash memory specifications: F1 0.97 (P 0.93, R 1.00); DR-114 clock configuration: F1 0.97 (P 0.94, R 1.00); DR-114 thermal assessment: F1 0.98 (P 0.95, R 1.00); DR-114 errata and firmware: F1 0.90 (P 0.81, R 1.00); DR-115 errata and PLL1 limits: F1 0.87 (P 0.76, R 1.00); Revision B end of life: F1 1.00 (P 1.00, R 1.00)
- **complex_mcu, typesafe/jev-latest, decision_default**: Device limitations of revision C: F1 0.92 (P 0.92, R 0.92); Flash memory specifications: F1 0.72 (P 0.82, R 0.64); DR-114 clock configuration: F1 0.66 (P 0.84, R 0.55); DR-114 thermal assessment: F1 0.44 (P 0.64, R 0.33); DR-114 errata and firmware: F1 0.62 (P 1.00, R 0.45); DR-115 errata and PLL1 limits: F1 0.41 (P 0.80, R 0.28); Revision B end of life: F1 0.74 (P 0.77, R 0.71)
- **complex_nda, azure_ai/ms-decision-model, decision_default**: NDA information protection: F1 0.64 (P 0.47, R 1.00); NDA recipient permissions: F1 0.91 (P 0.83, R 1.00); NDA compulsory disclosure Q-27: F1 0.77 (P 0.62, R 1.00); NDA exit obligations X-27: F1 0.78 (P 0.67, R 0.93)
- **complex_nda, openai/gpt-5.6-luna, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 1.00 (P 1.00, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, openai/gpt-6-luna, concurrent**: NDA information protection: F1 0.97 (P 0.95, R 1.00); NDA recipient permissions: F1 0.95 (P 0.90, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 0.99 (P 0.98, R 1.00)
- **complex_nda, openai/gpt-6-luna, decision_default**: NDA information protection: F1 0.82 (P 0.88, R 0.78); NDA recipient permissions: F1 0.97 (P 1.00, R 0.93); NDA compulsory disclosure Q-27: F1 0.90 (P 0.82, R 1.00); NDA exit obligations X-27: F1 0.88 (P 1.00, R 0.79)
- **complex_nda, openai/gpt-6-luna, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 0.98 (P 0.96, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, openai/gpt-6.1-sol, default**: NDA information protection: F1 1.00 (P 1.00, R 1.00); NDA recipient permissions: F1 1.00 (P 1.00, R 1.00); NDA compulsory disclosure Q-27: F1 1.00 (P 1.00, R 1.00); NDA exit obligations X-27: F1 1.00 (P 1.00, R 1.00)
- **complex_nda, typesafe/jev-latest, decision_default**: NDA information protection: F1 0.74 (P 0.65, R 0.86); NDA recipient permissions: F1 0.95 (P 1.00, R 0.90); NDA compulsory disclosure Q-27: F1 0.87 (P 0.84, R 0.89); NDA exit obligations X-27: F1 0.81 (P 1.00, R 0.68)

## Errors and warnings

- none
